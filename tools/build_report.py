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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table as PlotTable
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
        for column_index in range(column_count):
            cell = ET.SubElement(row_node, f"{{{W}}}tc")
            cell_properties = ET.SubElement(cell, f"{{{W}}}tcPr")
            ET.SubElement(
                cell_properties,
                f"{{{W}}}tcW",
                {f"{{{W}}}w": column_width, f"{{{W}}}type": "dxa"},
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
    page_width, page_height = 8.27, 11.69
    margin = 0.62
    line_height = 0.18
    pdf = PdfPages(PDF_OUTPUT)
    figure, axis = plt.subplots(figsize=(page_width, page_height))
    axis.set_axis_off()
    y = 1.0 - margin / page_height

    def new_page() -> None:
        nonlocal figure, axis, y
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(page_width, page_height))
        axis.set_axis_off()
        y = 1.0 - margin / page_height

    for block in blocks:
        if block["kind"] == "image":
            new_page()
            image_path = REPORT.parent / block["path"]
            image = mpimg.imread(image_path)
            image_height = min(0.68, 0.82 * image.shape[0] / image.shape[1])
            axis.imshow(image, extent=(0.08, 0.92, 0.48 - image_height / 2, 0.48 + image_height / 2), aspect="auto")
            axis.text(0.08, 0.9, plain_text(block["alt"]), transform=axis.transAxes, fontsize=12, weight="bold")
            new_page()
            continue
        if block["kind"] == "table":
            new_page()
            rows = table_lines(block["rows"], 100)
            table = axis.table(
                cellText=rows[1:],
                colLabels=rows[0],
                loc="center",
                cellLoc="left",
                bbox=(0.04, 0.22, 0.92, 0.56),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(6.8 if len(rows[0]) > 4 else 8)
            for (row, _), cell in table.get_celld().items():
                cell.set_edgecolor("#888888")
                cell.get_text().set_wrap(True)
                if row == 0:
                    cell.set_facecolor("#D9EAF7")
                    cell.get_text().set_weight("bold")
            new_page()
            continue
        if block["kind"] == "heading":
            level = block["level"]
            size = {1: 20, 2: 15, 3: 11}[level]
            text = plain_text(block["text"])
            height = {1: 0.055, 2: 0.04, 3: 0.032}[level]
            if y - height < margin / page_height:
                new_page()
            axis.text(0.07, y, text, transform=axis.transAxes, fontsize=size, weight="bold", color="#17365D", va="top")
            y -= height
            continue

        text = plain_text(block["text"])
        if block["kind"] == "bullet":
            text = "\u2022  " + text
        wrapped = textwrap.wrap(text, width=110, break_long_words=True) or [""]
        height = len(wrapped) * line_height / page_height + 0.01
        if y - height < margin / page_height:
            new_page()
        axis.text(
            0.07,
            y,
            "\n".join(wrapped),
            transform=axis.transAxes,
            fontsize=9,
            color="#222222",
            va="top",
            linespacing=1.25,
        )
        y -= height
    new_page()
    pdf.close()


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
    with zipfile.ZipFile(DOCX_OUTPUT) as archive:
        if archive.testzip() is not None:
            raise ValueError("Generated DOCX has an invalid ZIP entry")
        ET.fromstring(archive.read("word/document.xml"))
    with PDF_OUTPUT.open("rb") as pdf_file:
        if not pdf_file.read(5) == b"%PDF-":
            raise ValueError("Generated PDF does not have a PDF header")
    print(f"Created {DOCX_OUTPUT} and {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
