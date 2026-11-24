import json
import os
from email.parser import BytesParser
from email.policy import HTTP

import httpx
import pytest
from larex_actions import ActionCancelled, ActionClient, ActionContext, ActionDispatchPayload

os.environ.setdefault("LAREX_DISPATCH_HMAC_SECRET", "test-dispatch-secret")

from larex_action_kraken import main

BASE_URL = "https://larex.example/runs/run-1"
CAPABILITIES = {"incrementalPageResults": True, "runReports": True}


def manifest_from(request: httpx.Request) -> dict:
    header = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    message = BytesParser(policy=HTTP).parsebytes(header + request.content)
    for part in message.iter_parts():
        if part.get_param("name", header="content-disposition") == "manifest":
            return json.loads(part.get_payload(decode=True))
    raise AssertionError("Result request does not contain a manifest part.")


class LarexServer:
    def __init__(self, *, cancel_after_pages: int | None = None):
        self.heartbeats: list[dict] = []
        self.manifests: list[dict] = []
        self.cancel_after_pages = cancel_after_pages

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/input"):
            return httpx.Response(200, json=self.action_input())
        if path.endswith("/heartbeat"):
            self.heartbeats.append(json.loads(request.content))
            cancelled = self.cancel_after_pages is not None and len(self.manifests) >= self.cancel_after_pages
            return httpx.Response(200, json={"cancelRequested": cancelled})
        if path.endswith("/results"):
            self.manifests.append(manifest_from(request))
            return httpx.Response(200, json={"status": "RUNNING"})
        if "/files/" in path:
            return httpx.Response(200, content=b"image")
        return httpx.Response(404)

    @staticmethod
    def action_input() -> dict:
        return {
            "protocolVersion": 1,
            "runId": "run-1",
            "processorKey": "kraken-segmentation",
            "projectId": "project-1",
            "capabilities": CAPABILITIES,
            "pages": [
                {
                    "id": f"page-{index}",
                    "name": f"Page {index}",
                    "images": [
                        {
                            "id": f"image-{index}",
                            "fileName": f"page-{index}.png",
                            "mimeType": "image/png",
                            "imageType": "COLOR",
                            "downloadUrl": f"{BASE_URL}/files/image-{index}",
                        }
                    ],
                }
                for index in (1, 2)
            ],
        }


async def run_processor(server: LarexServer) -> None:
    payload = ActionDispatchPayload.model_validate(
        {
            "protocolVersion": 1,
            "runId": "run-1",
            "processorId": "kraken-segmentation",
            "workspaceId": "workspace-1",
            "projectId": "project-1",
            "pageIds": ["page-1", "page-2"],
            "secret": "run-secret",
            "pullUrl": f"{BASE_URL}/input",
            "heartbeatUrl": f"{BASE_URL}/heartbeat",
            "resultUrl": f"{BASE_URL}/results",
            "heartbeatTimeoutSeconds": 1800,
            "capabilities": CAPABILITIES,
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as http_client:
        async with ActionClient.from_dispatch(payload, client=http_client) as client:
            await main.process_run(ActionContext(payload=payload, client=client))


@pytest.fixture(autouse=True)
def fake_kraken(monkeypatch):
    async def run_kraken(_ctx, image_path, output_path, _model):
        if image_path.name == "page-2.png":
            raise main.PageProcessingError("Unreadable page")
        output_path.write_bytes(b"<PcGts/>")

    monkeypatch.setattr(main, "validate_segmentation_runtime", lambda _model: None)
    monkeypatch.setattr(main, "run_kraken", run_kraken)


@pytest.mark.asyncio
async def test_run_reports_every_page_through_the_sdk():
    server = LarexServer()

    await run_processor(server)

    first, second, final = server.manifests
    assert first["pageId"] == "page-1"
    assert [file["type"] for file in first["files"]] == ["xml"]
    assert second["pageOutcome"]["status"] == "FAILED"
    assert second["pageOutcome"]["message"] == "Unreadable page"
    assert final["status"] == "completed"
    assert final["files"] == []
    assert [beat["statusMessage"] for beat in server.heartbeats] == [
        "Segmenting page 1/2: Page 1",
        "Segmenting page 2/2: Page 2",
    ]
    assert not any("progressPercent" in beat for beat in server.heartbeats)


@pytest.mark.asyncio
async def test_cancellation_between_pages_stops_the_run():
    server = LarexServer(cancel_after_pages=1)

    with pytest.raises(ActionCancelled):
        await run_processor(server)

    assert [manifest["pageId"] for manifest in server.manifests] == ["page-1"]
    assert server.heartbeats[-1]["status"] == "cancelled"
