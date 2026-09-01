from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from click import get_app_dir
from larex_actions import ActionContext, ParameterChoice
from larex_actions.fastapi import create_larex_action_app
from lxml import etree
from PIL import Image
from platformdirs import user_data_dir


def configure_sdk_transport_logging() -> None:
    enabled = os.getenv("LAREX_SDK_TRANSPORT_LOGGING", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return
    sdk_logger = logging.getLogger("larex_actions.transport")
    sdk_logger.setLevel(logging.DEBUG)
    if not sdk_logger.hasHandlers():
        sdk_logger.addHandler(logging.StreamHandler())
        sdk_logger.propagate = False


configure_sdk_transport_logging()


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.")
    return value


PROCESSOR_ID = os.getenv("LAREX_PROCESSOR_ID", "kraken-segmentation")
DISPATCH_SECRET_ENV = "LAREX_DISPATCH_HMAC_SECRET"
DEVICE = os.getenv("KRAKEN_DEVICE", "cpu")
PRECISION = os.getenv("KRAKEN_PRECISION", "32-true")
TEXT_DIRECTION = os.getenv("KRAKEN_TEXT_DIRECTION", "horizontal-lr")
SEGMENTATION_MODEL = os.getenv("KRAKEN_SEGMENTATION_MODEL", "").strip()
MODEL_DIRECTORY = os.getenv("KRAKEN_MODEL_DIRECTORY", "").strip()
MAX_PROCESS_SECONDS = positive_int_env("KRAKEN_MAX_PROCESS_SECONDS", 900)
MAX_CONCURRENT_RUNS = positive_int_env("KRAKEN_MAX_CONCURRENT_RUNS", 1)
PROGRESS_COMPLETE_BEFORE_UPLOAD = 95
RUN_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_RUNS)

IMAGE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tif",
    "image/tif": ".tif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}

MODEL_EXTENSIONS = {".mlmodel", ".safetensors"}


def default_model_directories() -> tuple[Path, ...]:
    return (
        Path(user_data_dir("htrmopo")),
        Path(get_app_dir("kraken")),
    )


def is_kraken_segmentation_model(path: Path) -> bool:
    try:
        from kraken.tasks import SegmentationTaskModel

        SegmentationTaskModel.load_model(path)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def discover_kraken_segmentation_models(
    model_directory: str | Path | None = MODEL_DIRECTORY,
    search_directories: Sequence[Path] | None = None,
    validator: Callable[[Path], bool] = is_kraken_segmentation_model,
) -> list[ParameterChoice]:
    default_label = (
        f"Processor default ({SEGMENTATION_MODEL})" if SEGMENTATION_MODEL else "Kraken built-in default (blla.mlmodel)"
    )[:256]
    choices = [ParameterChoice(value="", label=default_label)]
    roots = list(search_directories if search_directories is not None else default_model_directories())
    if model_directory:
        roots.append(Path(model_directory).expanduser())

    discovered: dict[Path, str] = {}
    for configured_root in roots:
        try:
            root = configured_root.resolve(strict=True)
        except OSError:
            continue
        if not root.is_dir():
            continue
        for candidate in root.rglob("*"):
            if candidate.suffix.lower() not in MODEL_EXTENSIONS:
                continue
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_file() or not resolved.is_relative_to(root):
                continue
            if len(str(resolved)) > 1024:
                continue
            if resolved in discovered or not validator(resolved):
                continue
            relative = resolved.relative_to(root).as_posix()
            discovered[resolved] = f"{relative} ({root.name})"[:256]

    for path, label in sorted(discovered.items(), key=lambda item: (item[1].casefold(), str(item[0])))[:999]:
        choices.append(ParameterChoice(value=str(path), label=label))
    return choices


def segmentation_model_from_input(action_input) -> str:
    parameters = getattr(action_input, "parameters", {}) or {}
    value = parameters.get("segmentationModel", "")
    if not isinstance(value, str):
        raise TypeError("segmentationModel must be a string.")
    return value.strip() or SEGMENTATION_MODEL


async def process_run(ctx: ActionContext) -> None:
    async with RUN_SEMAPHORE:
        await _process_run(ctx)


async def _process_run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.complete(message="Kraken segmentation received no pages.")
        return

    total = len(action_input.pages)
    xml_count = 0

    with tempfile.TemporaryDirectory(prefix="larex-kraken-") as temp_dir:
        work_dir = Path(temp_dir)
        for index, page in enumerate(action_input.pages, start=1):
            await ctx.check_cancelled()
            await ctx.heartbeat(
                page_progress(index - 1, total),
                f"Segmenting page {index}/{total}: {page.name}",
                raise_on_cancel=True,
            )

            async with ctx.step(f"Kraken segmentation for {page.name}"):
                output_path = await segment_page(ctx, action_input, page, work_dir)
                results = ctx.result_builder()
                results.add_xml_path(
                    page_id=page.id,
                    path=output_path,
                    file_name=f"{safe_stem(page.name or page.id)}.xml",
                )
                await ctx.submit_page_results(
                    page.id,
                    results,
                    f"Finished segmentation for page {index}/{total}",
                )
                xml_count += 1

            await ctx.heartbeat(
                page_progress(index, total),
                f"Finished segmentation for page {index}/{total}",
                raise_on_cancel=True,
            )

    await ctx.complete(message=result_message(xml_count))


async def segment_page(ctx: ActionContext, action_input, page, work_dir: Path) -> Path:
    if not page.images:
        raise ValueError(f"Page {page.id} does not expose an image input.")

    image = page.images[0]
    image_path = work_dir / safe_file_name(image.file_name, page.name, image.mime_type)
    output_path = work_dir / f"{safe_stem(page.name or page.id)}.xml"
    await ctx.download_to_path(image, image_path)

    if is_region_target(action_input):
        return await segment_selected_regions(ctx, action_input, page, image_path, work_dir)

    await run_kraken(ctx, image_path, output_path, segmentation_model_from_input(action_input))
    return output_path


async def segment_selected_regions(
    ctx: ActionContext,
    action_input,
    page,
    image_path: Path,
    work_dir: Path,
) -> Path:
    if not page.xml:
        raise ValueError(f"Page {page.id} does not expose PAGE XML for scoped region import.")

    original_xml_path = work_dir / f"{safe_stem(page.id)}-original.xml"
    output_xml_path = work_dir / f"{safe_stem(page.name or page.id)}.xml"
    await ctx.download_to_path(page.xml[0], original_xml_path)
    original_root = parse_xml(original_xml_path.read_bytes())
    page_image_size = page_xml_image_size_from_root(original_root)
    segmentation_model = segmentation_model_from_input(action_input)
    for region_id in selected_region_ids(action_input, page.id):
        await ctx.check_cancelled()
        region_points = page_xml_region_points_from_root(original_root, region_id)
        crop_path = work_dir / f"{safe_stem(region_id)}.png"
        crop = crop_target_image_to_path(
            image_path,
            region_points,
            crop_path,
            source_size=page_image_size,
        )
        crop_output_path = work_dir / f"{safe_stem(region_id)}.xml"
        await run_kraken(ctx, crop_path, crop_output_path, segmentation_model)
        merge_region_layout_root(
            original_root,
            crop_output_path.read_bytes(),
            region_id,
            crop.offset_x,
            crop.offset_y,
            crop.scale_x,
            crop.scale_y,
        )
    output_xml_path.write_bytes(etree.tostring(original_root, encoding="utf-8", xml_declaration=True))
    return output_xml_path


def is_region_target(action_input) -> bool:
    target_selection = action_input.target_selection
    return target_selection is not None and target_selection.type == "REGION"


def selected_region_ids(action_input, page_id: str) -> list[str]:
    target_selection = action_input.target_selection
    if target_selection is None:
        return []

    region_ids: list[str] = []
    for target_page in target_selection.pages:
        if target_page.page_id != page_id:
            continue
        page_region_ids = list(target_page.region_ids)
        if not page_region_ids:
            raise ValueError(f"Region-targeted run for page {page_id} does not contain region ids.")
        region_ids.extend(page_region_ids)
    return region_ids


def page_progress(completed_pages: int, total_pages: int) -> int:
    return int((completed_pages / total_pages) * PROGRESS_COMPLETE_BEFORE_UPLOAD)


async def run_kraken(
    ctx: ActionContext,
    image_path: Path,
    output_path: Path,
    segmentation_model: str = SEGMENTATION_MODEL,
) -> None:
    command = [
        "kraken",
        "-x",
        "-i",
        str(image_path),
        str(output_path),
        "--device",
        DEVICE,
        "--precision",
        PRECISION,
        "segment",
        "-bl",
        "-d",
        TEXT_DIRECTION,
    ]
    if segmentation_model:
        command.extend(["-i", segmentation_model])

    result = await ctx.run_subprocess(
        command,
        timeout=MAX_PROCESS_SECONDS,
        terminate_grace_seconds=5.0,
        capture_output=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        output = (stderr or stdout).strip()
        raise RuntimeError(f"Kraken segmentation failed with exit code {result.returncode}: {output}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Kraken segmentation did not produce PAGE XML output.")


@dataclass(frozen=True)
class TargetCrop:
    content: bytes
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float


@dataclass(frozen=True)
class TargetCropGeometry:
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float


def crop_target_image(
    image_bytes: bytes,
    points: list[tuple[float, float]],
    *,
    source_size: tuple[int, int] | None = None,
) -> TargetCrop:
    if not points:
        raise ValueError("Target region does not contain polygon coordinates.")

    with Image.open(BytesIO(image_bytes)) as image:
        crop, source_left, source_top, scale_x, scale_y = crop_image(image, points, source_size)
        output = BytesIO()
        crop.save(output, format="PNG")
        return TargetCrop(
            content=output.getvalue(),
            offset_x=source_left,
            offset_y=source_top,
            scale_x=scale_x,
            scale_y=scale_y,
        )


def crop_target_image_to_path(
    image_path: Path,
    points: list[tuple[float, float]],
    output_path: Path,
    *,
    source_size: tuple[int, int] | None = None,
) -> TargetCropGeometry:
    if not points:
        raise ValueError("Target region does not contain polygon coordinates.")
    with Image.open(image_path) as image:
        crop, offset_x, offset_y, scale_x, scale_y = crop_image(image, points, source_size)
        crop.save(output_path, format="PNG")
    return TargetCropGeometry(offset_x, offset_y, scale_x, scale_y)


def crop_image(
    image: Image.Image,
    points: list[tuple[float, float]],
    source_size: tuple[int, int] | None,
) -> tuple[Image.Image, float, float, float, float]:
    source_width, source_height = source_size or image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("PAGE XML image dimensions are invalid.")
    scale_x = image.width / source_width
    scale_y = image.height / source_height
    source_left = max(0.0, min(point[0] for point in points))
    source_top = max(0.0, min(point[1] for point in points))
    source_right = min(float(source_width), max(point[0] for point in points))
    source_bottom = min(float(source_height), max(point[1] for point in points))
    left = max(0, min(image.width, int(source_left * scale_x)))
    top = max(0, min(image.height, int(source_top * scale_y)))
    right = max(0, min(image.width, int(source_right * scale_x + 0.999)))
    bottom = max(0, min(image.height, int(source_bottom * scale_y + 0.999)))
    if right <= left or bottom <= top:
        raise ValueError(
            "Target region produces an empty crop "
            f"(image={image.width}x{image.height}, source={source_width}x{source_height}, "
            f"bbox={source_left},{source_top},{source_right},{source_bottom})."
        )
    return image.crop((left, top, right, bottom)), source_left, source_top, scale_x, scale_y


def page_xml_image_size(xml_bytes: bytes) -> tuple[int, int] | None:
    return page_xml_image_size_from_root(parse_xml(xml_bytes))


def page_xml_image_size_from_root(root: etree._Element) -> tuple[int, int] | None:
    page = next((element for element in root.iter() if local_name(element.tag) == "Page"), None)
    if page is None:
        return None
    width = page.get("imageWidth")
    height = page.get("imageHeight")
    if not width or not height:
        return None
    return int(width), int(height)


def page_xml_region_points(xml_bytes: bytes, region_id: str) -> list[tuple[float, float]]:
    return page_xml_region_points_from_root(parse_xml(xml_bytes), region_id)


def page_xml_region_points_from_root(
    root: etree._Element,
    region_id: str,
) -> list[tuple[float, float]]:
    region = find_by_local_name_and_id(root, "TextRegion", region_id)
    if region is None:
        raise ValueError(f"Selected region {region_id} is missing from PAGE XML.")
    coords = next((child for child in region if local_name(child.tag) == "Coords"), None)
    points = coords.get("points") if coords is not None else None
    if not points:
        raise ValueError(f"Selected region {region_id} has no PAGE XML coordinates.")
    return [parse_point_pair(pair) for pair in points.split() if pair.strip()]


def parse_point_pair(value: str) -> tuple[float, float]:
    x_value, y_value = value.split(",", 1)
    return float(x_value), float(y_value)


def merge_region_layout_xml(
    original_xml: bytes,
    layout_xml: bytes,
    region_id: str,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
) -> bytes:
    original_root = parse_xml(original_xml)
    merge_region_layout_root(
        original_root,
        layout_xml,
        region_id,
        offset_x,
        offset_y,
        scale_x,
        scale_y,
    )
    return etree.tostring(original_root, encoding="utf-8", xml_declaration=True)


def merge_region_layout_root(
    original_root: etree._Element,
    layout_xml: bytes,
    region_id: str,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
) -> None:
    layout_root = parse_xml(layout_xml)
    target_namespace = namespace_uri(original_root.tag)

    target_region = find_by_local_name_and_id(original_root, "TextRegion", region_id)
    if target_region is None:
        raise ValueError(f"Selected region {region_id} is missing from original PAGE XML.")

    existing_text_lines = [child for child in list(target_region) if local_name(child.tag) == "TextLine"]
    for text_line in existing_text_lines:
        target_region.remove(text_line)

    layout_text_lines = [element for element in layout_root.iter() if local_name(element.tag) == "TextLine"]
    insert_at = text_line_insert_index(target_region)
    for index, text_line in enumerate(layout_text_lines, start=1):
        copied = copy.deepcopy(text_line)
        normalize_namespace(copied, target_namespace)
        copied.set("id", f"{region_id}-kraken-line-{index}")
        translate_page_xml_geometry(copied, offset_x, offset_y, scale_x, scale_y)
        target_region.insert(insert_at, copied)
        insert_at += 1

def text_line_insert_index(region: etree._Element) -> int:
    for index, child in enumerate(region):
        if local_name(child.tag) in {"TextEquiv", "TextStyle"}:
            return index
    return len(region)


def normalize_namespace(element: etree._Element, namespace: str | None) -> None:
    if namespace:
        for child in element.iter():
            child.tag = f"{{{namespace}}}{local_name(child.tag)}"


def translate_page_xml_geometry(
    element: etree._Element,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
) -> None:
    for child in element.iter():
        if local_name(child.tag) not in {"Coords", "Baseline"}:
            continue
        points = child.get("points")
        if not points:
            continue
        child.set(
            "points",
            " ".join(translate_point_pair(pair, offset_x, offset_y, scale_x, scale_y) for pair in points.split()),
        )


def translate_point_pair(
    value: str,
    offset_x: float,
    offset_y: float,
    scale_x: float,
    scale_y: float,
) -> str:
    x_value, y_value = value.split(",", 1)
    return f"{int(round(float(x_value) / scale_x + offset_x))},{int(round(float(y_value) / scale_y + offset_y))}"


def find_by_local_name_and_id(root: etree._Element, name: str, element_id: str) -> etree._Element | None:
    for element in root.iter():
        if local_name(element.tag) == name and element.get("id") == element_id:
            return element
    return None


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def namespace_uri(tag: str) -> str | None:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else None


def parse_xml(xml_bytes: bytes) -> etree._Element:
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    return etree.fromstring(xml_bytes, parser=parser)


def safe_file_name(file_name: str | None, page_name: str | None, mime_type: str | None) -> str:
    source = file_name or page_name or "page"
    name = Path(source).name
    stem = safe_stem(Path(name).stem or "page")
    suffix = Path(name).suffix.lower()
    if not suffix:
        suffix = IMAGE_EXTENSIONS.get((mime_type or "").lower(), ".png")
    return f"{stem}{suffix}"


def safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")
    return stem[:96] or "page"


def result_message(xml_count: int) -> str:
    return f"Kraken segmentation produced {xml_count} PAGE XML file(s)."


app = create_larex_action_app(
    processor_id=PROCESSOR_ID,
    dispatch_secret_env=DISPATCH_SECRET_ENV,
    handler=process_run,
    parameter_value_providers={
        "krakenSegmentationModels": discover_kraken_segmentation_models,
    },
)


@app.get("/ready")
async def readiness():
    if RUN_SEMAPHORE.locked():
        from fastapi.responses import JSONResponse

        return JSONResponse({"status": "busy"}, status_code=503)
    return {"status": "ready", "capacity": MAX_CONCURRENT_RUNS}
