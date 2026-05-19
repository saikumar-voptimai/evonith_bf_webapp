"""Reusable Streamlit UI components for the FurnaceMind dashboard."""

from __future__ import annotations

from io import BytesIO
import re

import streamlit as st

_BLOCK_RE = re.compile(r"\n{2,}")
_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TABLE_SEP = re.compile(r"^\|[-:| ]+\|", re.MULTILINE)
_SHIFT_REPORT_TITLE_RE = re.compile(
    r"\((\d{4}-\d{2}-\d{2})_SHIFT_([A-Za-z0-9]+)\)"
)
_TABLE_TITLES = (
    "",
    "Process Parameters",
    "Temperature Profile",
    "Hearth Pad Temperature",
    "Tapping Summary",
)
_REPORT_TABLE_IMAGE_STYLE_VERSION = 7


class ReportTableImageRenderer:
    """Render shift-report markdown tables into one two-column PNG."""

    image_width_px = 3840
    image_height_px = 2160
    dpi = 150
    title_size = 34
    section_title_size = 24
    body_size = 20
    dense_body_size = 18
    cell_padding = 0.075
    right_cell_padding = 0.12
    right_table_hspace = 0.18
    content_top = 0.935
    section_title_pad = 4
    titled_table_height = 0.96

    text = "#172033"
    header = "#17324d"
    line = "#d8dee8"
    section = "#eef4fb"
    stripe = "#f8fafc"

    def __init__(self, title: str, summary_text: str) -> None:
        self.title = self._image_title(title)
        self.tables = self._tables(summary_text)

    @staticmethod
    def _image_title(title: str) -> str:
        match = _SHIFT_REPORT_TITLE_RE.search(title)
        if match:
            return f"{match.group(1)} {match.group(2)} SHIFT REPORT"
        return title

    @staticmethod
    def split_blocks(summary_text: str) -> tuple[list[str], list[str], list[str]]:
        tables, pre, post, seen_table = [], [], [], False
        for block in filter(
            None, (b.strip() for b in _BLOCK_RE.split(summary_text.strip()))
        ):
            if _TABLE_SEP.search(block):
                tables.append(block)
                seen_table = True
            elif seen_table:
                post.append(block)
            else:
                pre.append(block)
        return tables, pre, post

    def render(self) -> bytes:
        if not self.tables:
            return b""

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        two_col = len(self.tables) > 1
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
            self.title,
            x=0.02,
            y=0.995,
            ha="left",
            fontsize=self.title_size,
            fontweight="bold",
            color=self.text,
        )

        if two_col:
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
            self._draw_table(fig.add_subplot(grid[0, 0]), self.tables[0])
            right = grid[0, 1].subgridspec(
                len(self.tables) - 1,
                1,
                hspace=self.right_table_hspace,
                height_ratios=[self._table_weight(table) for table in self.tables[1:]],
            )
            for index, table in enumerate(self.tables[1:]):
                self._draw_table(
                    fig.add_subplot(right[index, 0]),
                    table,
                    cell_padding=self.right_cell_padding,
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
            self._draw_table(fig.add_subplot(grid[0, 0]), self.tables[0])

        output = BytesIO()
        fig.savefig(output, format="png", dpi=self.dpi)
        return output.getvalue()

    def _tables(
        self, summary_text: str
    ) -> list[tuple[str, list[str], list[list[str]]]]:
        tables = []
        for index, markdown in enumerate(self.split_blocks(summary_text)[0]):
            headers, rows = self._parse_table(markdown)
            if headers:
                title = (
                    _TABLE_TITLES[index]
                    if index < len(_TABLE_TITLES)
                    else f"Report Table {index + 1}"
                )
                tables.append((title, headers, rows))
        return tables

    @staticmethod
    def _table_weight(table: tuple[str, list[str], list[list[str]]]) -> int:
        return max(4, len(table[2]) + 2)

    def _draw_table(
        self,
        ax,
        table: tuple[str, list[str], list[list[str]]],
        *,
        cell_padding: float | None = None,
    ) -> None:
        title, headers, rows = table
        padding = self.cell_padding if cell_padding is None else cell_padding
        ax.set_axis_off()
        if title:
            ax.set_title(
                title,
                loc="left",
                pad=self.section_title_pad,
                fontsize=self.section_title_size,
                fontweight="bold",
                color=self.text,
            )
        artist = ax.table(
            cellText=rows or [[""] * len(headers)],
            colLabels=headers,
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0, 1, self.titled_table_height if title else 1],
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(
            self.dense_body_size if len(headers) > 4 else self.body_size
        )
        artist.auto_set_column_width(col=list(range(len(headers))))

        for (row_index, _), cell in artist.get_celld().items():
            cell.set_edgecolor(self.line)
            cell.PAD = padding
            text = cell.get_text()
            text.set_ha("left")
            text.set_va("center")
            text.set_wrap(True)
            if row_index == 0:
                cell.set_facecolor(self.header)
                text.set_color("white")
                text.set_fontweight("bold")
                continue

            source = rows[row_index - 1] if rows else []
            if self._is_section(source):
                cell.set_facecolor(self.section)
                text.set_fontweight("bold")
            else:
                cell.set_facecolor("white" if row_index % 2 else self.stripe)
                text.set_color(self.text)

    @staticmethod
    def _parse_table(markdown: str) -> tuple[list[str], list[list[str]]]:
        lines = [line for line in markdown.splitlines() if line.strip().startswith("|")]
        sep_idx = next(
            (
                idx
                for idx, line in enumerate(lines)
                if ReportTableImageRenderer._is_separator(line)
            ),
            -1,
        )
        if sep_idx <= 0:
            return [], []
        headers = ReportTableImageRenderer._cells(lines[sep_idx - 1])
        width = len(headers)
        rows = [
            (ReportTableImageRenderer._cells(line) + [""] * width)[:width]
            for line in lines[sep_idx + 1 :]
            if not ReportTableImageRenderer._is_separator(line)
        ]
        return headers, rows

    @staticmethod
    def _cells(row: str) -> list[str]:
        return [
            _BOLD_RE.sub(
                r"\1", _BR_RE.sub("\n", cell.strip()).replace("&nbsp;", " ")
            ).strip()
            for cell in row.strip().strip("|").split("|")
        ]

    @staticmethod
    def _is_separator(row: str) -> bool:
        cells = ReportTableImageRenderer._cells(row)
        return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)

    @staticmethod
    def _is_section(row: list[str]) -> bool:
        return bool(row) and bool(row[0]) and all(not cell for cell in row[1:])


class ReportView:
    """Streamlit view for shift-report markdown."""

    def __init__(self, title: str, summary_text: str) -> None:
        self.title = title
        self.summary_text = summary_text
        self.tables, self.pre, self.post = ReportTableImageRenderer.split_blocks(
            summary_text
        )

    def render(self) -> None:
        st.subheader(self.title)
        if self.pre:
            st.markdown("\n\n".join(self.pre))
        if len(self.tables) >= 2:
            left, right = st.columns([52, 48])
            with left:
                st.markdown(self.tables[0])
            with right:
                st.markdown("\n\n".join(self.tables[1:]))
        elif self.tables:
            st.markdown("\n\n".join(self.tables))
        if self.post:
            st.markdown("\n\n".join(self.post))
        self._download_button()

    def _download_button(self) -> None:
        image = _report_tables_png(
            self.summary_text,
            self.title,
            _REPORT_TABLE_IMAGE_STYLE_VERSION,
        )
        if image:
            st.download_button(
                "Download Report",
                data=image,
                file_name=f"{_safe_filename(self.title)}_tables.png",
                mime="image/png",
                use_container_width=True,
                key=f"download_report_tables_png_{_safe_filename(self.title)}",
            )


def show_shift_summary(title: str, text: str) -> None:
    st.subheader(title)
    st.text_area(label="", value=text, height=250)


def _safe_filename(value: str) -> str:
    return _SAFE_NAME_RE.sub("_", value.strip()).strip("_").lower() or "report"


def _extract_report_tables(summary_text: str) -> tuple[list[str], list[str], list[str]]:
    return ReportTableImageRenderer.split_blocks(summary_text)


@st.cache_data(show_spinner=False)
def _report_tables_png(
    summary_text: str,
    report_title: str,
    style_version: int = _REPORT_TABLE_IMAGE_STYLE_VERSION,
) -> bytes:
    del style_version
    return ReportTableImageRenderer(report_title, summary_text).render()


def show_report(title: str, summary_text: str) -> None:
    ReportView(title, summary_text).render()
