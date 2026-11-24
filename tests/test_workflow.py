from types import SimpleNamespace

import pytest
from larex_actions import ResultBuilder

from larex_action_kraken import main


class FakeContext:
    def __init__(self, pages):
        self.input = SimpleNamespace(pages=pages, parameters={}, capabilities=SimpleNamespace(run_reports=True))
        self.events: list[tuple[str, object]] = []

    async def pull_input(self):
        return self.input

    async def check_cancelled(self):
        self.events.append(("cancel", None))

    async def heartbeat(self, progress_percent=None, status_message=None, *, raise_on_cancel=False):
        self.events.append(("heartbeat", status_message))

    def result_builder(self):
        return ResultBuilder()

    async def submit_page_results(
        self, page_id, results=None, message=None, *, page_status=None, reason=None, warnings=None
    ):
        self.events.append(("submit", (page_id, len(results.files) if results else 0)))
        if page_status == "FAILED":
            assert reason
            self.events.append(("failed", page_id))

    async def complete(self, results=None, message=None):
        self.events.append(("complete", message))


@pytest.fixture(autouse=True)
def validate_runtime(monkeypatch):
    monkeypatch.setattr(main, "validate_segmentation_runtime", lambda _model: None)


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
    assert [choice.default for choice in choices] == [True, False, False]
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


@pytest.mark.asyncio
async def test_recoverable_page_failure_continues_and_is_acknowledged(monkeypatch):
    context = FakeContext(pages(3))

    async def segment_page(_ctx, _input, page, work_dir):
        if page.id == "page-2":
            raise main.PageProcessingError("Unreadable page")
        output = work_dir / f"{page.id}.xml"
        output.write_bytes(b"<PcGts/>")
        return output

    monkeypatch.setattr(main, "segment_page", segment_page)
    await main.process_run(context)
    assert ("failed", "page-2") in context.events
    assert ("submit", ("page-3", 1)) in context.events
    assert context.events[-1] == (
        "complete", "Kraken segmentation produced 2 PAGE XML file(s). 1 page(s) failed."
    )


@pytest.mark.asyncio
async def test_server_page_rejection_does_not_stop_following_pages(monkeypatch):
    import httpx
    from larex_actions import PageResultRejected

    context = FakeContext(pages(2))
    original_submit = context.submit_page_results

    async def submit(
        page_id, results=None, message=None, *, page_status=None, reason=None, warnings=None
    ):
        if page_id == "page-1":
            request = httpx.Request("POST", "https://larex.example/results")
            raise PageResultRejected("Invalid output", request=request, response=httpx.Response(422))
        await original_submit(
            page_id, results, message, page_status=page_status, reason=reason, warnings=warnings
        )

    async def segment_page(_ctx, _input, page, work_dir):
        output = work_dir / f"{page.id}.xml"
        output.write_bytes(b"<PcGts/>")
        return output

    monkeypatch.setattr(context, "submit_page_results", submit)
    monkeypatch.setattr(main, "segment_page", segment_page)
    await main.process_run(context)
    assert ("submit", ("page-2", 1)) in context.events
    assert "1 page(s) failed" in context.events[-1][1]


@pytest.mark.asyncio
async def test_shared_model_failure_stops_before_any_page(monkeypatch):
    context = FakeContext(pages(2))

    def invalid_model(_name):
        raise RuntimeError("Model missing")

    monkeypatch.setattr(main, "validate_segmentation_runtime", invalid_model)
    with pytest.raises(RuntimeError, match="Model missing"):
        await main.process_run(context)
    assert not any(event[0] in {"submit", "complete"} for event in context.events)


@pytest.mark.asyncio
async def test_old_servers_are_rejected_before_any_page():
    from larex_actions import RunReportsUnsupported

    context = FakeContext(pages(1))
    context.input.capabilities.run_reports = False
    with pytest.raises(RunReportsUnsupported):
        await main.process_run(context)
    assert context.events == []
