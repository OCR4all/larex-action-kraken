import os
from io import BytesIO

from PIL import Image

os.environ.setdefault("LAREX_DISPATCH_HMAC_SECRET", "test-dispatch-secret")

from larex_action_kraken.main import (
    crop_target_image,
    merge_region_layout_xml,
    page_xml_image_size,
    page_xml_region_points,
    safe_file_name,
    safe_stem,
)


def test_page_xml_image_size_and_region_points():
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
      <Page imageWidth="100" imageHeight="80">
        <TextRegion id="region-1">
          <Coords points="10,20 50,20 50,60 10,60"/>
        </TextRegion>
      </Page>
    </PcGts>
    """

    assert page_xml_image_size(xml) == (100, 80)
    assert page_xml_region_points(xml, "region-1") == [
        (10.0, 20.0),
        (50.0, 20.0),
        (50.0, 60.0),
        (10.0, 60.0),
    ]


def test_crop_target_image_uses_page_xml_source_scale():
    image = Image.new("RGB", (200, 100), "white")
    output = BytesIO()
    image.save(output, format="PNG")

    crop = crop_target_image(
        output.getvalue(),
        [(10, 20), (50, 20), (50, 60), (10, 60)],
        source_size=(100, 100),
    )

    with Image.open(BytesIO(crop.content)) as cropped:
        assert cropped.size == (80, 40)
    assert crop.offset_x == 10
    assert crop.offset_y == 20
    assert crop.scale_x == 2
    assert crop.scale_y == 1


def test_merge_region_layout_xml_translates_lines_into_target_region():
    original = b"""<?xml version="1.0" encoding="UTF-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
      <Page imageWidth="100" imageHeight="100">
        <TextRegion id="region-1">
          <Coords points="10,20 50,20 50,60 10,60"/>
          <TextLine id="old-line"><Coords points="1,1 2,2"/></TextLine>
        </TextRegion>
      </Page>
    </PcGts>
    """
    layout = b"""<?xml version="1.0" encoding="UTF-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
      <Page imageWidth="40" imageHeight="40">
        <TextRegion id="crop-region">
          <Coords points="0,0 40,0 40,40 0,40"/>
          <TextLine id="line-a">
            <Coords points="4,5 24,5 24,9 4,9"/>
            <Baseline points="4,8 24,8"/>
          </TextLine>
        </TextRegion>
      </Page>
    </PcGts>
    """

    merged = merge_region_layout_xml(
        original,
        layout,
        "region-1",
        offset_x=10,
        offset_y=20,
        scale_x=2,
        scale_y=1,
    )

    assert b"old-line" not in merged
    assert b"region-1-kraken-line-1" in merged
    assert b'points="12,25 22,25 22,29 12,29"' in merged
    assert b'points="12,28 22,28"' in merged


def test_safe_names_are_stable():
    assert safe_stem("Page 1!!") == "Page-1"
    assert safe_file_name(None, "Page 1", "image/png") == "Page-1.png"
