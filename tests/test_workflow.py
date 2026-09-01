from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from larex_actions import ResultBuilder

from larex_action_kraken import main


class FakeContext:
    def __init__(self, pages):
        self.input = SimpleNamespace(pages=pages)
        self.events: list[tuple[str, object]] = []

    async def pull_input(self):
        return self.input

    async def check_cancelled(self):
        self.events.append(("cancel", None))

    async def heartbeat(self, progress, message, *, raise_on_cancel=False):
        self.events.append(("heartbeat", progress))

    @asynccontextmanager
    async def step(self, name):
        self.events.append(("step", name))
        yield

    def result_builder(self):
        return ResultBuilder()

    async def submit_page_results(self, page_id, results, message=None):
        self.events.append(("submit", (page_id, len(results.files))))

    async def complete(self, results=None, message=None):
        self.events.append(("complete", message))


def pages(count: int):
    return [SimpleNamespace(id=f"page-{index}", name=f"Page {index}") for index in range(1, count + 1)]


def test_discovers_nested_models_and_sorts_them(tmp_path):
    nested = tmp_path / "trained" / "run-2"
    nested.mkdir(parents=True)
    second = nested / "z-model.mlmodel"
    second.write_bytes(b"model")
    first = tmp_path / "a-model.safetensors"
    first.write_bytes(b"model")
    ignored = tmp_path / "recognition.mlmodel"
    ignored.write_bytes(b"recognition")

    choices = main.discover_kraken_segmentation_models(
        tmp_path,
        search_directories=(),
        validator=lambda path: path != ignored,
    )

    assert choices[0].value == ""
    assert [choice.value for choice in choices[1:]] == [str(first), str(second)]


def test_model_discovery_rejects_symlink_escape(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside.mlmodel"
    outside.write_bytes(b"model")
    (root / "escaped.mlmodel").symlink_to(outside)

    choices = main.discover_kraken_segmentation_models(
        root,
        search_directories=(),
        validator=lambda _path: True,
    )

    assert [choice.value for choice in choices] == [""]


def test_resolves_selected_model_and_processor_default(monkeypatch):
    monkeypatch.setattr(main, "SEGMENTATION_MODEL", "configured.mlmodel")

    assert (
        main.segmentation_model_from_input(SimpleNamespace(parameters={"segmentationModel": "/models/trained.mlmodel"}))
        == "/models/trained.mlmodel"
    )
    assert main.segmentation_model_from_input(SimpleNamespace(parameters={})) == "configured.mlmodel"


def test_rejects_non_string_model_parameter():
    with pytest.raises(TypeError, match="must be a string"):
        main.segmentation_model_from_input(SimpleNamespace(parameters={"segmentationModel": 42}))


@pytest.mark.asyncio
async def test_selected_model_is_passed_to_kraken(tmp_path):
    output_path = tmp_path / "output.xml"

    class CommandContext:
        command = None

        async def run_subprocess(self, command, **_kwargs):
            self.command = command
            output_path.write_bytes(b"<PcGts/>")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    context = CommandContext()
    await main.run_kraken(
        context,
        tmp_path / "input.png",
        output_path,
        "/models/trained.mlmodel",
    )

    assert context.command[-2:] == ["-i", "/models/trained.mlmodel"]


@pytest.mark.asyncio
async def test_process_run_submits_each_page_before_processing_next(monkeypatch):
    context = FakeContext(pages(2))

    async def segment_page(_ctx, _action_input, page, _work_dir):
        context.events.append(("segment", page.id))
        output = _work_dir / f"{page.id}.xml"
        output.write_bytes(b"<PcGts/>")
        return output

    monkeypatch.setattr(main, "segment_page", segment_page)

    await main.process_run(context)

    significant = [event for event in context.events if event[0] in {"segment", "submit", "complete"}]
    assert significant == [
        ("segment", "page-1"),
        ("submit", ("page-1", 1)),
        ("segment", "page-2"),
        ("submit", ("page-2", 1)),
        ("complete", "Kraken segmentation produced 2 PAGE XML file(s)."),
    ]


@pytest.mark.asyncio
async def test_process_run_stops_without_completion_when_page_fails(monkeypatch):
    context = FakeContext(pages(2))

    async def segment_page(_ctx, _action_input, page, _work_dir):
        if page.id == "page-2":
            raise RuntimeError("boom")
        output = _work_dir / f"{page.id}.xml"
        output.write_bytes(b"<PcGts/>")
        return output

    monkeypatch.setattr(main, "segment_page", segment_page)

    with pytest.raises(RuntimeError, match="boom"):
        await main.process_run(context)

    assert ("submit", ("page-1", 1)) in context.events
    assert not any(event[0] == "complete" for event in context.events)
