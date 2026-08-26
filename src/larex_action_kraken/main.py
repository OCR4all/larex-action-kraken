from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from multiprocessing.queues import Queue
from pathlib import Path

from fastapi.responses import JSONResponse
from larex_actions import ActionContext
from larex_actions.fastapi import create_larex_action_app

from .worker import kraken_worker


def getintenv(key: str, default: int) -> int:
    raw = os.getenv(key, '')
    if not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f'{key} must be an integer, got {raw!r}.') from exc


def getboolenv(key: str, default: bool) -> bool:
    raw = os.getenv(key, default='')
    if not raw.strip():
        return default
    return raw.lower() in ['true', '1', 't', 'y', 'yes']


def progress(index: int, total: int) -> int:
    return int((index / total) * 100)


PROCESSOR_ID = os.getenv('LAREX_PROCESSOR_ID', 'kraken-segmentation')
DISPATCH_HMAC_SECRET = os.getenv('LAREX_DISPATCH_HMAC_SECRET', None)

MAX_CONCURRENT_RUNS = getintenv('KRAKEN_MAX_CONCURRENT_RUNS', 1)
MODELS_BASE_PATH = Path(os.getenv('KRAKEN_MODELS_BASE_PATH', '/models'))

RUN_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_RUNS)


async def run(ctx: ActionContext) -> None:
    async with RUN_SEMAPHORE:
        await _run(ctx)


async def _run(ctx: ActionContext) -> None:
    action_input = await ctx.pull_input()
    if not action_input.pages:
        await ctx.fail('Kraken segmentation received no pages')
        return
    
    device = action_input.parameters.get('device', 'cpu')
    model = action_input.parameters.get('model', '')
    mode = action_input.parameters.get('mode', 'all')
    line_fallback_height = action_input.parameters.get('fallback', None)
    sort = action_input.parameters.get('sort', False)
    
    count = len(action_input.pages)
    mp_context = mp.get_context('spawn')
    in_queue: Queue[bytes | None] = mp_context.Queue()
    out_queue: Queue[tuple[bytes | None, str | None]] = mp_context.Queue()
    process = mp_context.Process(
        target=kraken_worker,
        args=(
            in_queue, 
            out_queue, 
            mode, 
            None if not model else MODELS_BASE_PATH / f'{model}.mlmodel', 
            device, 
            line_fallback_height, 
            sort
        )
    )
    process.start()
    
    try:
        for i, page in enumerate(action_input.pages, start=1):
            await ctx.check_cancelled()
            await ctx.heartbeat(
                progress(i-1, count),
                f'Segmenting page {i}/{count}: {page.name}',
                raise_on_cancel=True
            )
            print(f'Segmentation for {page.name}')
            async with ctx.step(f'Kraken segmentation for {page.name}'):
                img_bytes: bytes = await ctx.download_bytes(page.images[0])
                in_queue.put(img_bytes)
                xml_bytes, error = await asyncio.to_thread(out_queue.get)
                results = ctx.result_builder()
                if error:
                    await ctx.submit_page_results(
                        page_id=page.id, 
                        results=results, 
                        message=f'Segmentation failed for page {i}/{count}: {error}'
                    )
                else:
                    if xml_bytes is None:
                        await ctx.submit_page_results(
                            page_id=page.id, 
                            results=results, 
                            message=f'Segmentation returned no XML for page {i}/{count}'
                        )
                    else:
                        results.add_xml_bytes(
                            page_id=page.id, 
                            content=xml_bytes, 
                            file_name=f'{page.images[0].file_name or page.name or "page"}.xml'
                        )
                        await ctx.submit_page_results(
                            page_id=page.id, 
                            results=results, 
                            message=f'Finished segmentation for page {i}/{count}'
                        )
            await ctx.heartbeat(
                progress(i, count),
                f'Finished segmentation for page {i}/{count}',
                raise_on_cancel=True,
            )
        in_queue.put(None)
        await asyncio.to_thread(process.join)
        if process.exitcode != 0:
            await ctx.fail(f'Kraken worker exited with code {process.exitcode}')
            return
    except Exception as exc:
        await ctx.submit_page_results(
            page_id=page.id, 
            results=results, 
            message=f'Segmentation failed for page {i}/{count}: {exc}'
        )
    finally:
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join)
        in_queue.close()
        out_queue.close()

        in_queue.join_thread()
        out_queue.join_thread()
    await ctx.complete(message=f'Kraken segmentation produced {count} PAGE XML file(s).')


if DISPATCH_HMAC_SECRET is None:
    raise ValueError('No DISPATCH_HMAC_SECRET environment variable set')

app = create_larex_action_app(
    processor_id=PROCESSOR_ID,
    dispatch_secret=DISPATCH_HMAC_SECRET,
    handler=run
)

@app.get("/ready")
async def readiness():
    if RUN_SEMAPHORE.locked():
        return JSONResponse({'status': 'busy'}, status_code=503)
    return {'status': 'ready', 'capacity': MAX_CONCURRENT_RUNS}

