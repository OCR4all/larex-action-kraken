import os
from io import BytesIO
from pathlib import Path

from lxml import etree
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


PAGE_2024 = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2024-07-15"
KRAKEN_LAYOUT_2019 = b"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Metadata><Creator>kraken</Creator><Created>2026-11-16T10:00:00</Created><LastChange>2026-11-16T10:00:00</LastChange></Metadata>
  <Page imageFilename="crop.png" imageWidth="40" imageHeight="40">
    <TextRegion id="crop-region">
      <Coords points="0,0 40,0 40,40 0,40"/>
      <TextLine id="line-a">
        <Coords points="4,5 24,5 24,9 4,9"/>
        <Baseline points="4,8 24,8"/>
      </TextLine>
      <TextLine id="line-b">
        <Coords points="4,15 24,15 24,19 4,19"/>
        <Baseline points="4,18 24,18"/>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
"""


def page_2024(relation_target: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="{PAGE_2024}">
  <Metadata><Creator>LAREX</Creator><Created>2026-11-16T10:00:00</Created><LastChange>2026-11-16T10:00:00</LastChange></Metadata>
  <Page imageFilename="page.png" imageWidth="100" imageHeight="100">
    <ReadingOrder>
      <OrderedGroup id="ro">
        <RegionRefIndexed index="0" regionRef="form-1"/>
      </OrderedGroup>
    </ReadingOrder>
    <Relations>
      <Relation id="rel-1" type="link">
        <SourceRegionRef regionRef="{relation_target}"/>
        <TargetRegionRef regionRef="kept-line"/>
      </Relation>
    </Relations>
    <FormRegion id="form-1">
      <Coords points="5,5 95,5 95,95 5,95"/>
      <TextRegion id="region-1" type="paragraph">
        <Coords points="10,20 50,20 50,60 10,60"/>
        <TextLine id="old-line" mirrored="horizontally">
          <Coords points="11,21 49,21 49,29 11,29"/>
          <AscentLine points="11,22 49,22"/>
          <Baseline points="11,28 49,28"/>
          <Word id="old-word">
            <Coords points="11,21 30,21 30,29 11,29"/>
            <TextEquiv><Unicode>old</Unicode></TextEquiv>
          </Word>
          <TextEquiv><Unicode>old</Unicode></TextEquiv>
        </TextLine>
        <TextEquiv><Unicode>old</Unicode></TextEquiv>
      </TextRegion>
    </FormRegion>
    <TextRegion id="region-2">
      <Coords points="60,60 90,60 90,90 60,90"/>
      <TextLine id="kept-line">
        <Coords points="61,61 89,61 89,69 61,69"/>
      </TextLine>
    </TextRegion>
  </Page>
</PcGts>
""".encode()


def page_2024_schema():
    return etree.XMLSchema(etree.parse(str(Path(__file__).parent / "fixtures" / "pagecontent-2024-07-15.xsd")))


def merge_into_region_1(original: bytes) -> bytes:
    return merge_region_layout_xml(
        original,
        KRAKEN_LAYOUT_2019,
        "region-1",
        offset_x=10,
        offset_y=20,
        scale_x=1,
        scale_y=1,
    )


def test_page_2024_fixture_is_valid():
    schema = page_2024_schema()
    assert schema.validate(etree.fromstring(page_2024("region-1"))), schema.error_log


def test_merge_into_page_2024_keeps_2024_and_stays_valid():
    merged = merge_into_region_1(page_2024("region-1"))
    root = etree.fromstring(merged)

    schema = page_2024_schema()
    assert schema.validate(root), schema.error_log
    assert {element.tag.split("}")[0][1:] for element in root.iter() if isinstance(element.tag, str)} == {PAGE_2024}
    region = root.find(f".//{{{PAGE_2024}}}TextRegion[@id='region-1']")
    assert [line.get("id") for line in region.findall(f"{{{PAGE_2024}}}TextLine")] == [
        "region-1-kraken-line-1",
        "region-1-kraken-line-2",
    ]
    assert region[-1].tag == f"{{{PAGE_2024}}}TextEquiv"
    assert root.find(f".//{{{PAGE_2024}}}TextLine[@id='kept-line']") is not None


def test_merge_into_page_2024_drops_relations_to_replaced_lines():
    for removed_id in ("old-line", "old-word"):
        merged = merge_into_region_1(page_2024(removed_id))
        root = etree.fromstring(merged)

        schema = page_2024_schema()
        assert schema.validate(root), schema.error_log
        assert root.find(f".//{{{PAGE_2024}}}Relation") is None
        assert removed_id.encode() not in merged
