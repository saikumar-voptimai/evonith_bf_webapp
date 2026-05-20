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
    "Fuel Rate",
    "Parameters",
    "Temperature Profile",
    "Hearth Pad Temperature",
    "Tapping Summary",
    "Consumption",
)
_REPORT_TABLE_IMAGE_STYLE_VERSION = 15


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
    top_table_hspace = 0.12
    top_cell_padding = 0.1
    content_top = 0.935
    section_title_pad = 4
    titled_table_height = 0.96

    text = "#172033"
    header = "#17324d"
    line = "#d8dee8"
    section = "#eef4fb"
    stripe = "#f8fafc"
    image_header_breaks = {
        "Fuel rate (kg/thm)": "Fuel rate\n(kg/thm)",
        "Coke rate (kg/thm)": "Coke rate\n(kg/thm)",
        "Nut Coke rate (kg/thm)": "Nut Coke rate\n(kg/thm)",
        "PCI rate (kg/thm)": "PCI rate\n(kg/thm)",
        "Total Taps (no's)": "Total Taps\n(no's)",
        "Tap Duration (mins)": "Tap Duration\n(mins)",
        "Slag Duration (mins)": "Slag Duration\n(mins)",
        "Slag Ratio (%)": "Slag Ratio\n(%)",
        "Casting Rate (T/min)": "Casting Rate\n(T/min)",
        "Coke (tons)": "Coke\n(tons)",
        "Nut coke (tons)": "Nut coke\n(tons)",
        "Ore (tons)": "Ore\n(tons)",
        "Flux (tons)": "Flux\n(tons)",
        "Sinter (tons)": "Sinter\n(tons)",
        "Pellet (tons)": "Pellet\n(tons)",
    }

    def __init__(self, title: str, summary_text: str) -> None:
        self.title = self._image_title(title)
        self.post = self.split_blocks(summary_text)[2]
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
        material_note = self._material_note_text()

        if len(self.tables) >= 3:
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
            left = grid[0, 0].subgridspec(
                2,
                1,
                height_ratios=[0.72, 4.28],
                hspace=self.top_table_hspace,
            )
            self._draw_table(
                fig.add_subplot(left[0, 0]),
                self.tables[0],
                cell_padding=self.top_cell_padding,
            )
            self._draw_table(fig.add_subplot(left[1, 0]), self.tables[1])
            right_tables = self.tables[2:]
            right_count = len(right_tables) + int(bool(material_note))
            right = grid[0, 1].subgridspec(
                right_count,
                1,
                hspace=self.right_table_hspace,
                height_ratios=[
                    *[self._table_weight(table) for table in right_tables],
                    *([0.9] if material_note else []),
                ],
            )
            for index, table in enumerate(right_tables):
                self._draw_table(
                    fig.add_subplot(right[index, 0]),
                    table,
                    cell_padding=self.right_cell_padding,
                )
            if material_note:
                self._draw_note(
                    fig.add_subplot(right[len(right_tables), 0]), material_note
                )
        elif two_col:
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
            right_tables = self.tables[1:]
            right_count = len(right_tables) + int(bool(material_note))
            right = grid[0, 1].subgridspec(
                right_count,
                1,
                hspace=self.right_table_hspace,
                height_ratios=[
                    *[self._table_weight(table) for table in right_tables],
                    *([0.9] if material_note else []),
                ],
            )
            for index, table in enumerate(right_tables):
                self._draw_table(
                    fig.add_subplot(right[index, 0]),
                    table,
                    cell_padding=self.right_cell_padding,
                )
            if material_note:
                self._draw_note(
                    fig.add_subplot(right[len(right_tables), 0]), material_note
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
        for index, markdown in enumerate(self.table_blocks(summary_text)):
            headers, rows = self._parse_table(markdown)
            if headers:
                title = (
                    _TABLE_TITLES[index]
                    if index < len(_TABLE_TITLES)
                    else f"Report Table {index + 1}"
                )
                tables.append((title, headers, rows))
        return tables

    @classmethod
    def table_blocks(cls, summary_text: str) -> list[str]:
        return cls._normalize_table_blocks(cls.split_blocks(summary_text)[0])

    @classmethod
    def _normalize_table_blocks(cls, tables: list[str]) -> list[str]:
        tables = [cls._normalize_tapping_table(table) for table in tables]
        if len(tables) < 2:
            return tables

        first_headers, first_rows = cls._parse_table(tables[0])
        previous_split = cls._split_previous_consumption_table(first_headers, first_rows)
        if previous_split is not None:
            fuel_rate_table, consumption_table = previous_split
            return [fuel_rate_table, *tables[1:], consumption_table]

        second_headers, second_rows = cls._parse_table(tables[1])
        if first_headers != ["Parameter", "UOM", "Value"]:
            return tables
        if second_headers != ["Parameter", "UOM", "Value", "Std.Dev"]:
            return tables

        section_names = {row[0] for row in first_rows if cls._is_section(row)}
        if not {"Consumption", "Quality"}.issubset(section_names):
            return tables

        sections = cls._legacy_shift_sections(first_rows)
        consumption_lookup = {
            row[0].lower(): row[2]
            for row in sections.get("Consumption", [])
            if len(row) >= 3
        }
        fuel_rate_table = cls._fuel_rate_table(consumption_lookup)
        consumption_table = cls._consumption_table(consumption_lookup)

        parameter_rows = [["**Production**", "", "", ""]]
        parameter_rows.extend(
            [row[0], row[1], row[2], "-"]
            for row in sections.get("Production", [])
            if len(row) >= 3
        )
        parameter_rows.append(["**Quality**", "", "", ""])
        parameter_rows.extend(
            [row[0], row[1], row[2], "-"]
            for row in sections.get("Quality", [])
            if len(row) >= 3
        )
        parameter_rows.append(["**Process Parameters**", "", "", ""])
        parameter_rows.extend(second_rows)
        params_table = cls._markdown_table(
            ["Parameter", "UOM", "Value", "Std.Dev"],
            parameter_rows,
        )
        return [fuel_rate_table, params_table, *tables[2:], consumption_table]

    @classmethod
    def _split_previous_consumption_table(
        cls,
        headers: list[str],
        rows: list[list[str]],
    ) -> tuple[str, str] | None:
        previous_headers = [
            "Coke (tons)",
            "Nut coke (tons)",
            "Ore (tons)",
            "Flux (tons)",
            "Sinter (tons)",
            "Pellet (tons)",
            "Fuel rate (kg/thm)",
            "Coke rate (kg/thm)",
            "Nut Coke rate (kg/thm)",
            "PCI rate (kg/thm)",
        ]
        if headers != previous_headers or not rows:
            return None

        values = {header.lower(): value for header, value in zip(headers, rows[0])}
        return cls._fuel_rate_table(values), cls._consumption_table(values)

    @classmethod
    def _fuel_rate_table(cls, values: dict[str, str]) -> str:
        headers = [
            "Fuel rate (kg/thm)",
            "Coke rate (kg/thm)",
            "Nut Coke rate (kg/thm)",
            "PCI rate (kg/thm)",
        ]
        keys = ["fuel rate", "coke rate", "nut coke rate", "pci rate"]
        return cls._markdown_table(
            headers,
            [
                [
                    values.get(header.lower(), values.get(key, "-"))
                    for header, key in zip(headers, keys)
                ]
            ],
        )

    @classmethod
    def _consumption_table(cls, values: dict[str, str]) -> str:
        headers = [
            "Coke (tons)",
            "Nut coke (tons)",
            "Ore (tons)",
            "Flux (tons)",
            "Sinter (tons)",
            "Pellet (tons)",
        ]
        material_keys = [
            "coke",
            "nut coke",
            "ore",
            "flux",
            "sinter",
            "pellet",
        ]
        return cls._markdown_table(
            headers,
            [
                [
                    values.get(header.lower(), values.get(key, "-"))
                    for header, key in zip(headers, material_keys)
                ]
            ],
        )

    @classmethod
    def _normalize_tapping_table(cls, table: str) -> str:
        headers, rows = cls._parse_table(table)
        if headers != ["Parameter", "UOM", "Value"]:
            return table

        values = {row[0].lower(): row[2] for row in rows if len(row) >= 3}
        if "total taps" not in values:
            return table

        return cls._markdown_table(
            [
                "Total Taps (no's)",
                "Tap Duration (mins)",
                "Slag Duration (mins)",
                "Slag Ratio (%)",
                "Casting Rate (T/min)",
            ],
            [
                [
                    values.get("total taps", "-"),
                    values.get("tap duration", "-"),
                    values.get("slag duration", "-"),
                    values.get("slag ratio", "-"),
                    values.get("casting rate", "-"),
                ]
            ],
        )

    @classmethod
    def _legacy_shift_sections(cls, rows: list[list[str]]) -> dict[str, list[list[str]]]:
        sections: dict[str, list[list[str]]] = {
            "Production": [],
            "Consumption": [],
            "Quality": [],
        }
        current = "Production"
        for row in rows:
            if cls._is_section(row):
                current = row[0]
                sections.setdefault(current, [])
                continue
            sections.setdefault(current, []).append(row)
        return sections

    @staticmethod
    def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
        width = len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * width) + "|",
        ]
        for row in rows:
            cells = (row + [""] * width)[:width]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

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
            colLabels=[self._image_header(header) for header in headers],
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
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=self.body_size,
            color=self.text,
            wrap=True,
        )

    def _material_note_text(self) -> str:
        lines: list[str] = []
        for block in self.post:
            for line in block.splitlines():
                clean = _BOLD_RE.sub(r"\1", _BR_RE.sub(" ", line)).strip()
                if clean.startswith(("Flux :", "Ore :")):
                    lines.append(clean)
        return "\n".join(lines)

    @staticmethod
    def _image_header(header: str) -> str:
        return ReportTableImageRenderer.image_header_breaks.get(header, header)

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
                r"\1", _BR_RE.sub(" ", cell.strip()).replace("&nbsp;", " ")
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
        tables, self.pre, self.post = ReportTableImageRenderer.split_blocks(
            summary_text
        )
        self.tables = ReportTableImageRenderer._normalize_table_blocks(tables)

    def render(self) -> None:
        st.subheader(self.title)
        if self.pre:
            st.markdown("\n\n".join(self.pre))
        post_rendered = False
        if len(self.tables) >= 3:
            left, right = st.columns([52, 48])
            with left:
                self._render_table(0, self.tables[0])
                self._render_table(1, self.tables[1])
            with right:
                for index, table in enumerate(self.tables[2:], start=2):
                    self._render_table(index, table)
                if self.post:
                    st.markdown("\n\n".join(self.post))
                    post_rendered = True
        elif len(self.tables) >= 2:
            left, right = st.columns([52, 48])
            with left:
                self._render_table(0, self.tables[0])
            with right:
                for index, table in enumerate(self.tables[1:], start=1):
                    self._render_table(index, table)
                if self.post:
                    st.markdown("\n\n".join(self.post))
                    post_rendered = True
        elif self.tables:
            self._render_table(0, self.tables[0])
        if self.post and not post_rendered:
            st.markdown("\n\n".join(self.post))
        self._download_button()

    @staticmethod
    def _render_table(index: int, markdown: str) -> None:
        title = (
            _TABLE_TITLES[index]
            if index < len(_TABLE_TITLES)
            else f"Report Table {index + 1}"
        )
        if title:
            st.markdown(f"**{title}**")
        st.markdown(_BR_RE.sub(" ", markdown))

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
    tables, pre, post = ReportTableImageRenderer.split_blocks(summary_text)
    return ReportTableImageRenderer._normalize_table_blocks(tables), pre, post


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
