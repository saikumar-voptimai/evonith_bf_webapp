"""Reusable Streamlit UI components for the FurnaceMind dashboard."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
import re

import streamlit as st

from reports.rendering import (
    ReportDocument,
    ReportNote,
    ReportSection,
    clean_inline,
    document_from_markdown,
    markdown_table,
)
from utils.shift_windows import LOCAL_TIMEZONE, LOCAL_TIMEZONE_NAME

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SHIFT_REPORT_TITLE_RE = re.compile(
    r"\((\d{4}-\d{2}-\d{2})_SHIFT_([A-Za-z0-9]+)\)"
)
_DAY_REPORT_TITLE_RE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
_HEADER_UOM_RE = re.compile(r"^(.+?)\s+(\([^)]{1,24}\))$")
_REPORT_TABLE_IMAGE_STYLE_VERSION = 17


class ReportTableImageRenderer:
    """Render a structured report document into one PNG."""

    image_width_px = 3840
    image_height_px = 2160
    dpi = 150
    title_size = 34
    section_title_size = 24
    body_size = 20
    dense_body_size = 18
    footer_size = 13
    cell_padding = 0.075
    right_cell_padding = 0.12
    top_cell_padding = 0.1
    stack_hspace = 0.18
    content_top = 0.935
    section_title_pad = 4
    titled_table_height = 0.96

    text = "#172033"
    header = "#17324d"
    line = "#d8dee8"
    section = "#eef4fb"
    stripe = "#f8fafc"

    def __init__(
        self,
        document_or_title: ReportDocument | str,
        summary_text: str | None = None,
    ) -> None:
        self.document = (
            document_or_title
            if isinstance(document_or_title, ReportDocument)
            else document_from_markdown(str(document_or_title), summary_text or "")
        )

    @staticmethod
    def _image_title(title: str) -> str:
        match = _SHIFT_REPORT_TITLE_RE.search(title)
        if match:
            return f"{match.group(1)} {match.group(2)} SHIFT REPORT"
        day_match = _DAY_REPORT_TITLE_RE.search(title)
        if day_match and "day" in title.lower():
            return f"{day_match.group(1)} DAY REPORT"
        return title

    def render(self) -> bytes:
        if not self.document.sections:
            return b""

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(
            figsize=(
                self.image_width_px / self.dpi,
                self.image_height_px / self.dpi,
            ),
            dpi=self.dpi,
            facecolor="white",
        )
        FigureCanvasAgg(fig)
        fig.suptitle(
            self._image_title(self.document.title),
            x=0.02,
            y=0.995,
            ha="left",
            fontsize=self.title_size,
            fontweight="bold",
            color=self.text,
        )

        left_sections = [
            section for section in self.document.sections if section.placement == "left"
        ]
        right_sections = [
            section for section in self.document.sections if section.placement != "left"
        ]
        image_notes = [
            note for note in self.document.notes if note.kind == "material_usage"
        ]

        if left_sections and right_sections:
            grid = fig.add_gridspec(
                1,
                2,
                width_ratios=[1.05, 1],
                wspace=0.08,
                left=0.035,
                right=0.985,
                top=self.content_top,
                bottom=0.055,
            )
            self._draw_stack(
                fig,
                grid[0, 0],
                left_sections,
                first_padding=self.top_cell_padding,
            )
            self._draw_stack(
                fig,
                grid[0, 1],
                right_sections,
                image_notes,
                section_padding=self.right_cell_padding,
            )
        else:
            grid = fig.add_gridspec(
                1,
                1,
                left=0.035,
                right=0.985,
                top=self.content_top,
                bottom=0.055,
            )
            self._draw_stack(
                fig,
                grid[0, 0],
                left_sections or right_sections,
                image_notes,
                first_padding=self.top_cell_padding,
            )

        self._draw_generated_at(fig)
        output = BytesIO()
        fig.savefig(output, format="png", dpi=self.dpi)
        return output.getvalue()

    def _draw_generated_at(self, fig) -> None:
        fig.text(
            0.985,
            0.014,
            self._generated_at_text(),
            ha="right",
            va="bottom",
            fontsize=self.footer_size,
            color="#64748b",
        )

    def _generated_at_text(self) -> str:
        generated_at = self.document.generated_at_ist or datetime.now(LOCAL_TIMEZONE)
        if generated_at.tzinfo is None:
            generated_at = LOCAL_TIMEZONE.localize(generated_at)
        else:
            generated_at = generated_at.astimezone(LOCAL_TIMEZONE)
        timezone_label = (
            "IST" if LOCAL_TIMEZONE_NAME == "Asia/Kolkata" else LOCAL_TIMEZONE_NAME
        )
        return f"Generated: {generated_at:%Y-%m-%d %H:%M:%S} {timezone_label}"

    def _draw_stack(
        self,
        fig,
        grid_spec,
        sections: list[ReportSection],
        notes: list[ReportNote] | None = None,
        section_padding: float | None = None,
        first_padding: float | None = None,
    ) -> None:
        notes = notes or []
        count = len(sections) + len(notes)
        stack = grid_spec.subgridspec(
            count,
            1,
            hspace=self.stack_hspace,
            height_ratios=[
                *[self._section_weight(section) for section in sections],
                *[self._note_weight(note) for note in notes],
            ],
        )
        for index, section in enumerate(sections):
            padding = first_padding if index == 0 and first_padding is not None else section_padding
            self._draw_table(fig.add_subplot(stack[index, 0]), section, padding=padding)

        for offset, note in enumerate(notes, start=len(sections)):
            self._draw_note(fig.add_subplot(stack[offset, 0]), note.text)

    @staticmethod
    def _section_weight(section: ReportSection) -> int:
        return max(4, len(section.rows) + 2)

    @staticmethod
    def _note_weight(note: ReportNote) -> float:
        return max(2.5, len(note.text.splitlines()) * 0.9)

    def _draw_table(
        self,
        ax,
        section: ReportSection,
        *,
        padding: float | None = None,
    ) -> None:
        cell_padding = self.cell_padding if padding is None else padding
        ax.set_axis_off()
        ax.set_title(
            section.title,
            loc="left",
            pad=self.section_title_pad,
            fontsize=self.section_title_size,
            fontweight="bold",
            color=self.text,
        )
        artist = ax.table(
            cellText=section.rows or [[""] * len(section.headers)],
            colLabels=[self._image_header(header) for header in section.headers],
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0, 1, self.titled_table_height],
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(
            self.dense_body_size if len(section.headers) > 4 else self.body_size
        )
        artist.auto_set_column_width(col=list(range(len(section.headers))))

        for (row_index, _), cell in artist.get_celld().items():
            cell.set_edgecolor(self.line)
            cell.PAD = cell_padding
            text = cell.get_text()
            text.set_ha("left")
            text.set_va("center")
            text.set_wrap(True)
            if row_index == 0:
                cell.set_facecolor(self.header)
                text.set_color("white")
                text.set_fontweight("bold")
                continue

            row = section.rows[row_index - 1] if section.rows else ()
            if self._is_section_row(row):
                cell.set_facecolor(self.section)
                text.set_color(self.text)
                text.set_fontweight("bold")
            else:
                cell.set_facecolor("white" if row_index % 2 else self.stripe)
                text.set_color(self.text)

    def _draw_note(self, ax, text: str) -> None:
        ax.set_axis_off()
        ax.text(
            0,
            0.95,
            clean_inline(text),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=self.dense_body_size,
            color=self.text,
            wrap=True,
        )

    @staticmethod
    def _image_header(header: str) -> str:
        match = _HEADER_UOM_RE.fullmatch(header)
        if match:
            return f"{match.group(1)}\n{match.group(2)}"
        return header

    @staticmethod
    def _is_section_row(row: tuple[str, ...]) -> bool:
        return bool(row) and bool(row[0]) and all(not cell for cell in row[1:])


class ReportView:
    """Streamlit view for structured reports."""

    def __init__(self, document: ReportDocument) -> None:
        self.document = document

    def render(self) -> None:
        st.subheader(self.document.title)
        pre_blocks = _visible_pre_blocks(self.document.pre_blocks)
        if pre_blocks:
            st.markdown("\n\n".join(pre_blocks))

        left_sections = [
            section for section in self.document.sections if section.placement == "left"
        ]
        right_sections = [
            section for section in self.document.sections if section.placement != "left"
        ]
        right_notes = [
            note for note in self.document.notes if note.placement != "left"
        ]

        if left_sections and right_sections:
            left, right = st.columns([52, 48])
            with left:
                self._render_sections(left_sections)
            with right:
                self._render_sections(right_sections)
                self._render_notes(right_notes)
        else:
            self._render_sections(left_sections or right_sections)
            self._render_notes(self.document.notes)

        self._download_button()

    @staticmethod
    def _render_sections(sections: list[ReportSection]) -> None:
        for section in sections:
            st.markdown(f"**{section.title}**")
            st.markdown(markdown_table(section.headers, section.rows))

    @staticmethod
    def _render_notes(notes: list[ReportNote] | tuple[ReportNote, ...]) -> None:
        for note in notes:
            text = note.text.replace("\n", "  \n") if note.kind == "material_usage" else note.text
            st.markdown(text)

    def _download_button(self) -> None:
        image = _report_document_png(self.document, _REPORT_TABLE_IMAGE_STYLE_VERSION)
        if image:
            st.download_button(
                "Download Report",
                data=image,
                file_name=f"{_safe_filename(self.document.title)}_tables.png",
                mime="image/png",
                width="stretch",
                key=f"download_report_tables_png_{_safe_filename(self.document.title)}",
            )


def show_shift_summary(title: str, text: str) -> None:
    st.subheader(title)
    st.text_area(label="", value=text, height=250)


def _visible_pre_blocks(blocks: tuple[str, ...]) -> list[str]:
    hidden = ("**Status:**", "**Flags:**")
    return [
        "\n".join(
            line for line in block.splitlines() if not line.startswith(hidden)
        ).strip()
        for block in blocks
        if block.strip()
    ]


def show_report_document(document: ReportDocument) -> None:
    ReportView(document).render()


def show_report(title: str, summary_text: str) -> None:
    show_report_document(document_from_markdown(title, summary_text))


def _safe_filename(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", value.strip()).strip("_").lower() or "report"


def _extract_report_tables(summary_text: str) -> tuple[list[str], list[str], list[str]]:
    document = document_from_markdown("", summary_text)
    tables = [markdown_table(section.headers, section.rows) for section in document.sections]
    return tables, list(document.pre_blocks), [note.text for note in document.notes]


@st.cache_data(show_spinner=False)
def _report_document_png(
    document: ReportDocument,
    style_version: int = _REPORT_TABLE_IMAGE_STYLE_VERSION,
) -> bytes:
    del style_version
    return ReportTableImageRenderer(document).render()


@st.cache_data(show_spinner=False)
def _report_tables_png(
    summary_text: str,
    report_title: str,
    style_version: int = _REPORT_TABLE_IMAGE_STYLE_VERSION,
) -> bytes:
    del style_version
    return ReportTableImageRenderer(report_title, summary_text).render()


__all__ = [
    "ReportTableImageRenderer",
    "ReportView",
    "show_shift_summary",
    "show_report_document",
    "show_report",
    "_extract_report_tables",
    "_report_document_png",
    "_report_tables_png",
]
