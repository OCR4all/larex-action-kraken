from __future__ import annotations

from io import BytesIO
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Literal


def kraken_worker(
    in_queue: Queue[bytes | None],  # image bytes or stop signal (None)
    out_queue: Queue[tuple[bytes | None, str | None]],  # xml bytes, error message
    mode: Literal['all', 'regions', 'lines'] = 'all',
    model: Path | None = None,
    device: str = 'cpu',
    line_fallback_height: int | None = None,
    sort: bool = False
) -> None:
    from lxml import etree
    from octopy.predict import Segmenter
    from PIL import Image
    
    segmenter = Segmenter(
        model=model,
        device=device,
        line_fallback_height=line_fallback_height
    )
    
    while True:
        in_item = in_queue.get()
        if in_item is None:
            break
        
        try:
            image = Image.open(BytesIO(in_item))
            xml = segmenter.predict(image, mode=mode, sort=sort)
            xml_bytes = etree.tostring(xml._to_etree(), pretty_print=True)
            out_queue.put((xml_bytes, None))
        except Exception as exc:  # noqa: BLE001
            out_queue.put((None, f'{type(exc).__name__}: {exc}'))
    