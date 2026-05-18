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
_TABLE_TITLES = (
    "Shift Report",
    "Process Parameters",
    "Temperature Profile",
    "Hearth Pad Temperature",
    "Tapping Summary",
)


class ReportTableImageRenderer:
    """Render shift-report markdown tables into one two-column PNG."""

    text = "#172033"
    header = "#17324d"
    line = "#d8dee8"
    section = "#eef4fb"
    stripe = "#f8fafc"

    def __init__(self, title: str, summary_text: str) -> None:
        self.title = title
        self.tables = self._tables(summary_text)

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
            figsize=(15 if two_col else 8, self._figure_height(two_col)),
            dpi=150,
            facecolor="white",
        )
        FigureCanvasAgg(fig)
        fig.suptitle(
            self.title,
            x=0.02,
            y=0.995,
            ha="left",
            fontsize=15,
            fontweight="bold",
            color=self.text,
        )

        if two_col:
            grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.12, top=0.92)
            self._draw_table(fig.add_subplot(grid[0, 0]), self.tables[0])
            right = grid[0, 1].subgridspec(
                len(self.tables) - 1,
                1,
                hspace=0.34,
                height_ratios=[self._table_weight(table) for table in self.tables[1:]],
            )
            for index, table in enumerate(self.tables[1:]):
                self._draw_table(fig.add_subplot(right[index, 0]), table)
        else:
            self._draw_table(fig.add_subplot(111), self.tables[0])
            fig.subplots_adjust(top=0.9)

        output = BytesIO()
        fig.savefig(output, format="png", bbox_inches="tight", pad_inches=0.25)
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

    def _figure_height(self, two_col: bool) -> float:
        if not two_col:
            return max(3.0, self._table_weight(self.tables[0]) * 0.32)
        left = self._table_weight(self.tables[0])
        right = sum(self._table_weight(table) for table in self.tables[1:]) + len(
            self.tables
        )
        return max(4.5, max(left, right) * 0.32)

    @staticmethod
    def _table_weight(table: tuple[str, list[str], list[list[str]]]) -> int:
        return max(4, len(table[2]) + 2)

    def _draw_table(self, ax, table: tuple[str, list[str], list[list[str]]]) -> None:
        title, headers, rows = table
        ax.set_axis_off()
        ax.set_title(
            title, loc="left", pad=8, fontsize=10, fontweight="bold", color=self.text
        )
        artist = ax.table(
            cellText=rows or [[""] * len(headers)],
            colLabels=headers,
            cellLoc="left",
            colLoc="left",
            bbox=[0, 0, 1, 0.92],
        )
        artist.auto_set_font_size(False)
        artist.set_fontsize(7.5 if len(headers) > 4 else 8.5)
        artist.auto_set_column_width(col=list(range(len(headers))))

        for (row_index, _), cell in artist.get_celld().items():
            cell.set_edgecolor(self.line)
            cell.PAD = 0.035
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
        image = _report_tables_png(self.summary_text, self.title)
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
def _report_tables_png(summary_text: str, report_title: str) -> bytes:
    return ReportTableImageRenderer(report_title, summary_text).render()


def show_report(title: str, summary_text: str) -> None:
    ReportView(title, summary_text).render()
