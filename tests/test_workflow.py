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


@pytest.mark.asyncio
async def test_process_run_submits_each_page_before_processing_next(monkeypatch):
    context = FakeContext(pages(2))

    async def segment_page(_ctx, _action_input, page, _work_dir):
        context.events.append(("segment", page.id))
        return b"<PcGts/>"

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
        return b"<PcGts/>"

    monkeypatch.setattr(main, "segment_page", segment_page)

    with pytest.raises(RuntimeError, match="boom"):
        await main.process_run(context)

    assert ("submit", ("page-1", 1)) in context.events
    assert not any(event[0] == "complete" for event in context.events)
