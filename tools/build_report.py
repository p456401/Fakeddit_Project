"""Build DOCX and PDF deliverables from the maintained Markdown report."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "Project_Report.md"
FIGURES = ROOT / "docs" / "figures"
DOCX_OUTPUT = ROOT / "docs" / "Project_Report.docx"
PDF_OUTPUT = ROOT / "docs" / "Project_Report.pdf"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("w", W)
ET.register_namespace("r", R)
ET.register_namespace("wp", WP)
ET.register_namespace("a", A)
ET.register_namespace("pic", PIC)


def parse_report(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            blocks.append({"kind": "image", "alt": image.group(1), "path": image.group(2)})
        elif line.startswith("|"):
            rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not row or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in row):
                    rows.append(row)
                index += 1
            if rows:
                blocks.append({"kind": "table", "rows": rows})
            continue
        elif line.startswith("### "):
            blocks.append({"kind": "heading", "level": 3, "text": line[4:]})
        elif line.startswith("## "):
            blocks.append({"kind": "heading", "level": 2, "text": line[3:]})
        elif line.startswith("# "):
            blocks.append({"kind": "heading", "level": 1, "text": line[2:]})
        elif line.startswith("- ") or line.startswith("* "):
            blocks.append({"kind": "bullet", "text": line[2:]})
        else:
            blocks.append({"kind": "paragraph", "text": line})
        index += 1
    return blocks


def plain_text(value: str) -> str:
    value = re.sub(r"\*\*(.+?)\*\*", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", value)
    return value


def append_run(paragraph: ET.Element, text: str, *, bold: bool = False) -> None:
    run = ET.SubElement(paragraph, f"{{{W}}}r")
    if bold:
        ET.SubElement(ET.SubElement(run, f"{{{W}}}rPr"), f"{{{W}}}b")
    node = ET.SubElement(run, f"{{{W}}}t")
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    node.text = text


def append_paragraph(
    body: ET.Element,
    text: str,
    *,
    style: str | None = None,
    bullet: bool = False,
) -> None:
    paragraph = ET.SubElement(body, f"{{{W}}}p")
    properties = ET.SubElement(paragraph, f"{{{W}}}pPr")
    if style:
        ET.SubElement(properties, f"{{{W}}}pStyle", {f"{{{W}}}val": style})
    if bullet:
        append_run(paragraph, "\u2022  ")
    tokens = re.split(r"(\*\*.*?\*\*|`[^`]+`)", text)
    for token in tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            append_run(paragraph, token[2:-2], bold=True)
        else:
            append_run(paragraph, plain_text(token))


def append_table(body: ET.Element, rows: list[list[str]]) -> None:
    table = ET.SubElement(body, f"{{{W}}}tbl")
    properties = ET.SubElement(table, f"{{{W}}}tblPr")
    ET.SubElement(
        properties,
        f"{{{W}}}tblLayout",
        {f"{{{W}}}type": "fixed"},
    )
    ET.SubElement(
        properties,
        f"{{{W}}}tblW",
        {f"{{{W}}}w": "10440", f"{{{W}}}type": "dxa"},
    )
    borders = ET.SubElement(properties, f"{{{W}}}tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(
            borders,
            f"{{{W}}}{edge}",
            {f"{{{W}}}val": "single", f"{{{W}}}sz": "4", f"{{{W}}}color": "A0A0A0"},
        )
    grid = ET.SubElement(table, f"{{{W}}}tblGrid")
    column_count = max(len(row) for row in rows)
    column_width = str(10440 // column_count)
    for _ in range(column_count):
        ET.SubElement(grid, f"{{{W}}}gridCol", {f"{{{W}}}w": column_width})
    for row_index, row in enumerate(rows):
        row_node = ET.SubElement(table, f"{{{W}}}tr")
        if row_index == 0:
            header_properties = ET.SubElement(row_node, f"{{{W}}}trPr")
            ET.SubElement(header_properties, f"{{{W}}}tblHeader", {f"{{{W}}}val": "true"})
        else:
            row_properties = ET.SubElement(row_node, f"{{{W}}}trPr")
            ET.SubElement(row_properties, f"{{{W}}}cantSplit")
        for column_index in range(column_count):
            cell = ET.SubElement(row_node, f"{{{W}}}tc")
            cell_properties = ET.SubElement(cell, f"{{{W}}}tcPr")
            ET.SubElement(
                cell_properties,
                f"{{{W}}}tcW",
                {f"{{{W}}}w": column_width, f"{{{W}}}type": "dxa"},
            )
            margins = ET.SubElement(cell_properties, f"{{{W}}}tcMar")
            for side in ("top", "start", "bottom", "end"):
                ET.SubElement(
                    margins,
                    f"{{{W}}}{side}",
                    {f"{{{W}}}w": "55", f"{{{W}}}type": "dxa"},
                )
            if row_index == 0:
                shading = ET.SubElement(cell_properties, f"{{{W}}}shd")
                shading.set(f"{{{W}}}fill", "D9EAF7")
            paragraph = ET.SubElement(cell, f"{{{W}}}p")
            text = row[column_index] if column_index < len(row) else ""
            append_run(paragraph, plain_text(text), bold=row_index == 0)


def append_picture(
    body: ET.Element, image_path: Path, alt_text: str, relationship_id: str
) -> None:
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    max_width = 6_000_000
    max_height = 5_200_000
    scale = min(max_width / (width_px * 9525), max_height / (height_px * 9525), 1.0)
    width = int(width_px * 9525 * scale)
    height = int(height_px * 9525 * scale)
    paragraph = ET.SubElement(body, f"{{{W}}}p")
    run = ET.SubElement(paragraph, f"{{{W}}}r")
    drawing = ET.SubElement(run, f"{{{W}}}drawing")
    inline = ET.SubElement(
        drawing,
        f"{{{WP}}}inline",
        {"distT": "0", "distB": "0", "distL": "0", "distR": "0"},
    )
    ET.SubElement(inline, f"{{{WP}}}extent", {"cx": str(width), "cy": str(height)})
    image_id = relationship_id.removeprefix("rId")
    ET.SubElement(inline, f"{{{WP}}}docPr", {"id": image_id, "name": alt_text[:200]})
    graphic = ET.SubElement(inline, f"{{{A}}}graphic")
    graphic_data = ET.SubElement(
        graphic,
        f"{{{A}}}graphicData",
        {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"},
    )
    picture = ET.SubElement(graphic_data, f"{{{PIC}}}pic")
    non_visual = ET.SubElement(picture, f"{{{PIC}}}nvPicPr")
    ET.SubElement(non_visual, f"{{{PIC}}}cNvPr", {"id": "0", "name": alt_text[:200]})
    ET.SubElement(non_visual, f"{{{PIC}}}cNvPicPr")
    fill = ET.SubElement(picture, f"{{{PIC}}}blipFill")
    ET.SubElement(fill, f"{{{A}}}blip", {f"{{{R}}}embed": relationship_id})
    stretch = ET.SubElement(fill, f"{{{A}}}stretch")
    ET.SubElement(stretch, f"{{{A}}}fillRect")
    shape = ET.SubElement(picture, f"{{{PIC}}}spPr")
    transform = ET.SubElement(shape, f"{{{A}}}xfrm")
    ET.SubElement(transform, f"{{{A}}}off", {"x": "0", "y": "0"})
    ET.SubElement(transform, f"{{{A}}}ext", {"cx": str(width), "cy": str(height)})
    geometry = ET.SubElement(shape, f"{{{A}}}prstGeom", {"prst": "rect"})
    ET.SubElement(geometry, f"{{{A}}}avLst")


def write_docx(blocks: list[dict[str, Any]]) -> None:
    document = ET.Element(f"{{{W}}}document")
    body = ET.SubElement(document, f"{{{W}}}body")
    media: list[Path] = []
    for block in blocks:
        kind = block["kind"]
        if kind == "heading":
            style = "Title" if block["level"] == 1 else f"Heading{block['level']}"
            append_paragraph(body, block["text"], style=style)
        elif kind == "paragraph":
            append_paragraph(body, block["text"])
        elif kind == "bullet":
            append_paragraph(body, block["text"], bullet=True)
        elif kind == "table":
            append_table(body, block["rows"])
        elif kind == "image":
            path = (REPORT.parent / block["path"]).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Report image does not exist: {path}")
            media.append(path)
            append_picture(body, path, block["alt"], f"rId{len(media) + 1}")

    section = ET.SubElement(body, f"{{{W}}}sectPr")
    page_size = ET.SubElement(section, f"{{{W}}}pgSz")
    page_size.set(f"{{{W}}}w", "12240")
    page_size.set(f"{{{W}}}h", "15840")
    margins = ET.SubElement(section, f"{{{W}}}pgMar")
    for side, value in (("top", "900"), ("right", "900"), ("bottom", "900"), ("left", "900")):
        margins.set(f"{{{W}}}{side}", value)

    styles = ET.fromstring(
        f'''<w:styles xmlns:w="{W}">
        <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults>
        <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
        <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="34"/></w:rPr></w:style>
        <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="28"/></w:rPr></w:style>
        <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="365F91"/><w:sz w:val="22"/></w:rPr></w:style>
        </w:styles>'''
    )
    relationships = ET.Element(f"{{{PKG_REL}}}Relationships")
    ET.SubElement(
        relationships,
        f"{{{PKG_REL}}}Relationship",
        {
            "Id": "rId1",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles",
            "Target": "styles.xml",
        },
    )
    for index, path in enumerate(media, start=2):
        ET.SubElement(
            relationships,
            f"{{{PKG_REL}}}Relationship",
            {
                "Id": f"rId{index}",
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": f"media/image{index - 1}.png",
            },
        )
    content_types = ET.Element(f"{{{CT}}}Types")
    ET.SubElement(content_types, f"{{{CT}}}Default", {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(content_types, f"{{{CT}}}Default", {"Extension": "xml", "ContentType": "application/xml"})
    ET.SubElement(content_types, f"{{{CT}}}Default", {"Extension": "png", "ContentType": "image/png"})
    ET.SubElement(content_types, f"{{{CT}}}Override", {"PartName": "/word/document.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"})
    ET.SubElement(content_types, f"{{{CT}}}Override", {"PartName": "/word/styles.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"})
    package_rels = ET.Element(f"{{{PKG_REL}}}Relationships")
    ET.SubElement(
        package_rels,
        f"{{{PKG_REL}}}Relationship",
        {
            "Id": "rId1",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
            "Target": "word/document.xml",
        },
    )
    with zipfile.ZipFile(DOCX_OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", ET.tostring(content_types, encoding="utf-8", xml_declaration=True))
        archive.writestr("_rels/.rels", ET.tostring(package_rels, encoding="utf-8", xml_declaration=True))
        archive.writestr("word/document.xml", ET.tostring(document, encoding="utf-8", xml_declaration=True))
        archive.writestr("word/styles.xml", ET.tostring(styles, encoding="utf-8", xml_declaration=True))
        archive.writestr("word/_rels/document.xml.rels", ET.tostring(relationships, encoding="utf-8", xml_declaration=True))
        for index, path in enumerate(media, start=1):
            archive.write(path, f"word/media/image{index}.png")


def table_lines(rows: list[list[str]], width: int) -> list[list[str]]:
    columns = max(len(row) for row in rows)
    cell_width = max(8, width // columns)
    normalized = []
    for row in rows:
        values = []
        for column in range(columns):
            cell = plain_text(row[column]) if column < len(row) else ""
            values.append("\n".join(textwrap.wrap(cell, cell_width, break_long_words=True)))
        normalized.append(values)
    return normalized


def render_pdf(blocks: list[dict[str, Any]]) -> None:
    page_width, page_height = 8.5, 11.0
    left, right, top, bottom = 48.0, 48.0, 52.0, 44.0
    content_width = page_width * 72 - left - right
    page_number = 0
    pdf = PdfPages(PDF_OUTPUT)
    figure = None
    y = 0.0
    has_content = False

    def start_page(landscape: bool = False) -> None:
        nonlocal figure, y, has_content, page_width, page_height, content_width
        page_width, page_height = (11.69, 8.27) if landscape else (8.5, 11.0)
        content_width = page_width * 72 - left - right
        figure = plt.figure(figsize=(page_width, page_height))
        y = page_height * 72 - top
        has_content = False

    def finish_page() -> None:
        nonlocal page_number, figure, has_content
        if figure is None or not has_content:
            return
        page_number += 1
        figure.text(
            0.5,
            0.025,
            f"Context-Aware Multimodal Misinformation Verification  |  {page_number}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#666666",
        )
        pdf.savefig(figure, bbox_inches=None)
        plt.close(figure)
        figure = None
        has_content = False

    def ensure_space(height: float) -> None:
        nonlocal figure, has_content
        if y - height < bottom:
            finish_page()
            start_page()

    def draw_text(
        text: str,
        *,
        size: float = 9.2,
        weight: str = "normal",
        color: str = "#222222",
        indent: float = 0.0,
        wrap_chars: int = 112,
        leading: float | None = None,
    ) -> None:
        nonlocal y, has_content
        line_height = leading or size * 1.32
        lines = textwrap.wrap(text, width=wrap_chars, break_long_words=True) or [""]
        for line in lines:
            ensure_space(line_height + 1)
            figure.text(
                (left + indent) / (page_width * 72),
                y / (page_height * 72),
                line,
                ha="left",
                va="top",
                fontsize=size,
                fontweight=weight,
                color=color,
            )
            y -= line_height
            has_content = True
        y -= 3.0

    def draw_table(rows: list[list[str]]) -> None:
        nonlocal y, has_content
        column_count = max(map(len, rows))
        weights = [
            max(
                1.0,
                min(
                    2.5,
                    max(
                        len(plain_text(row[column]))
                        for row in rows
                        if column < len(row)
                    )
                    / 18,
                ),
            )
            for column in range(column_count)
        ]
        widths = [content_width * value / sum(weights) for value in weights]
        font_size = 8.3 if column_count >= 6 else 8.4
        leading = font_size * 1.22
        prepared: list[tuple[list[str], float]] = []
        for row in rows:
            cells: list[str] = []
            max_lines = 1
            for column in range(column_count):
                raw = plain_text(row[column]) if column < len(row) else ""
                char_limit = max(7, int((widths[column] - 10) / (font_size * 0.51)))
                wrapped = textwrap.wrap(raw, width=char_limit, break_long_words=True) or [""]
                max_lines = max(max_lines, len(wrapped))
                cells.append("\n".join(wrapped))
            prepared.append((cells, max_lines * leading + 9))

        required_height = sum(height for _, height in prepared)
        if y - required_height < bottom:
            finish_page()
            start_page()

        for row_index, (cells, row_height) in enumerate(prepared):
            if y - row_height < bottom:
                finish_page()
                start_page()
            cursor = left
            for column, cell in enumerate(cells):
                width = widths[column]
                rectangle = plt.Rectangle(
                    (cursor / 72, (y - row_height) / 72),
                    width / 72,
                    row_height / 72,
                    transform=figure.transFigure,
                    facecolor="#D9EAF7" if row_index == 0 else "#FFFFFF",
                    edgecolor="#9099A3",
                    linewidth=0.55,
                )
                figure.add_artist(rectangle)
                figure.text(
                    (cursor + 5) / (page_width * 72),
                    (y - 4) / (page_height * 72),
                    cell,
                    ha="left",
                    va="top",
                    fontsize=font_size,
                    fontweight="bold" if row_index == 0 else "normal",
                    color="#17212B",
                    linespacing=1.22,
                )
                cursor += width
            y -= row_height
            has_content = True
        y -= 6

    def draw_image(block: dict[str, Any]) -> None:
        nonlocal figure, has_content
        finish_page()
        start_page(landscape=True)
        image_path = (REPORT.parent / block["path"]).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Report figure does not exist: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
            aspect = width / height
        area_width = page_width * 72 - 44
        area_height = page_height * 72 - 74
        display_width = min(area_width, area_height * aspect)
        display_height = display_width / aspect
        image = plt.imread(image_path)
        figure.add_axes(
            [
                0.5 - display_width / (page_width * 72) / 2,
                0.5 - display_height / (page_height * 72) / 2 - 0.02,
                display_width / (page_width * 72),
                display_height / (page_height * 72),
            ]
        ).imshow(image, aspect="auto")
        figure.axes[-1].set_axis_off()
        has_content = True
        finish_page()
        start_page()

    start_page()
    for block in blocks:
        kind = block["kind"]
        if kind == "image":
            draw_image(block)
            continue
        if kind == "table":
            draw_table(block["rows"])
            continue
        if kind == "heading":
            level = block["level"]
            size = {1: 20.0, 2: 14.5, 3: 11.0}[level]
            wrap_chars = {1: 52, 2: 74, 3: 94}[level]
            if y - size * 3.0 < bottom:
                finish_page()
                start_page()
            y -= 5 if level > 1 else 0
            draw_text(
                plain_text(block["text"]),
                size=size,
                weight="bold",
                color="#17365D",
                wrap_chars=wrap_chars,
                leading=size * 1.3,
            )
            continue

        text = plain_text(block["text"])
        if kind == "bullet":
            text = "\u2022  " + text
        elif re.match(r"^\d+\.\s+", text):
            text = textwrap.dedent(text)
        draw_text(
            text,
            size=8.9,
            indent=12 if kind == "bullet" else 0,
            wrap_chars=116,
            leading=11.7,
        )
    finish_page()
    pdf.close()
    plt.close("all")


def validate_outputs(blocks: list[dict[str, Any]]) -> None:
    with zipfile.ZipFile(DOCX_OUTPUT) as archive:
        if archive.testzip() is not None:
            raise ValueError("Generated DOCX has an invalid ZIP entry")
        document = ET.fromstring(archive.read("word/document.xml"))
        relationships = ET.fromstring(
            archive.read("word/_rels/document.xml.rels")
        )
        image_count = len(document.findall(f".//{{{A}}}blip"))
        expected_images = sum(block["kind"] == "image" for block in blocks)
        if image_count != expected_images:
            raise ValueError(
                f"DOCX embeds {image_count} report figures; expected {expected_images}"
            )
        relation_ids = {
            relationship.get("Id")
            for relationship in relationships.findall(f"{{{PKG_REL}}}Relationship")
        }
        for picture in document.findall(f".//{{{A}}}blip"):
            if picture.get(f"{{{R}}}embed") not in relation_ids:
                raise ValueError("DOCX contains an image without a relationship")

    pdf_bytes = PDF_OUTPUT.read_bytes()
    if not pdf_bytes.startswith(b"%PDF-") or not pdf_bytes.rstrip().endswith(b"%%EOF"):
        raise ValueError("Generated PDF has an invalid header or EOF marker")
    page_objects = re.findall(
        rb"(?ms)^\d+\s+\d+\s+obj\b(.*?)\bendobj\b", pdf_bytes
    )
    pages = [
        page
        for page in page_objects
        if re.search(rb"/Type\s*/Page\b", page)
    ]
    figure_count = sum(block["kind"] == "image" for block in blocks)
    if len(pages) < figure_count + 1:
        raise ValueError("PDF is missing report pages")
    for page_number, page in enumerate(pages, start=1):
        if not re.search(rb"/Contents\s+\d+\s+\d+\s+R", page):
            raise ValueError(f"PDF page {page_number} has no content stream")
    landscape_pages = sum(
        1
        for page in pages
        if (match := re.search(rb"/MediaBox\s*\[([^]]+)", page))
        and float(match.group(1).split()[-2])
        > float(match.group(1).split()[-1])
    )
    if landscape_pages != figure_count:
        raise ValueError(
            f"PDF contains {landscape_pages} landscape pages for "
            f"{figure_count} full-page figures"
        )


def main() -> None:
    if not REPORT.is_file():
        raise FileNotFoundError(f"Report source is missing: {REPORT}")
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_figures.py")],
        check=True,
    )
    blocks = parse_report(REPORT.read_text(encoding="utf-8"))
    write_docx(blocks)
    render_pdf(blocks)
    validate_outputs(blocks)
    print(f"Created {DOCX_OUTPUT} and {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
