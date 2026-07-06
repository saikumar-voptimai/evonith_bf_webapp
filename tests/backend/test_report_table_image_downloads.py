from datetime import datetime, timezone
from io import BytesIO

from PIL import Image

from reports.rendering import ReportDocument, ReportSection
from ui.components import (
    ReportTableImageRenderer,
    _extract_report_tables,
    _report_tables_png,
)

REPORT_MARKDOWN = """Header text

| Parameter | UOM | Value |
|---|---|---|
| Production rate | t/hr | 100.00 |
| **Consumption** | | |
| Coke | tons | 20.00 |

| Parameter | UOM | Value | Std.Dev |
|---|---|---|---|
| Hot blast volume | Nm3/hr | 1200.00 | 1.25 |
"""


def test_extract_report_tables_keeps_pre_and_table_blocks() -> None:
    tables, pre, post = _extract_report_tables(REPORT_MARKDOWN)

    assert pre == ["Header text"]
    assert post == []
    assert len(tables) == 2


def test_report_tables_png_exports_all_markdown_tables_in_one_image() -> None:
    image = _report_tables_png(REPORT_MARKDOWN, "Live Report (2026-05-18_SHIFT_A)")

    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = Image.open(BytesIO(image)).size
    assert width > height


def test_report_table_image_footer_uses_document_generated_time() -> None:
    document = ReportDocument(
        title="Live Report (2026-05-18_SHIFT_A)",
        sections=(
            ReportSection.from_rows(
                "Parameters",
                ["Parameter", "Value"],
                [["Hot blast volume", "1200"]],
                placement="left",
            ),
        ),
        generated_at_ist=datetime(2026, 5, 29, 7, 5, 35, tzinfo=timezone.utc),
    )

    renderer = ReportTableImageRenderer(document)

    assert renderer._generated_at_text() == "Generated: 2026-05-29 12:35:35 IST"
