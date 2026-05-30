"""Structured report rendering primitives shared by UI and report builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Literal, Sequence

Placement = Literal["left", "right"]

_BLOCK_RE = re.compile(r"\n{2,}")
_BOLD_RE = re.compile(r"\*\*(.*?)\*\*")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TABLE_SEP_RE = re.compile(r"^\|[-:| ]+\|", re.MULTILINE)


@dataclass(frozen=True)
class ReportSection:
    """One tabular section in a report."""

    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    placement: Placement = "right"

    @classmethod
    def from_rows(
        cls,
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[object]],
        *,
        placement: Placement = "right",
    ) -> "ReportSection":
        width = len(headers)
        return cls(
            title=title,
            headers=tuple(str(header) for header in headers),
            rows=tuple(_coerce_row(row, width) for row in rows),
            placement=placement,
        )


@dataclass(frozen=True)
class ReportNote:
    """Non-tabular report text, such as raw material usage notes."""

    text: str
    placement: Placement = "right"
    kind: str = "note"


@dataclass(frozen=True)
class ReportDocument:
    """Renderable report composed of preamble text, tables, and notes."""

    title: str
    pre_blocks: tuple[str, ...] = field(default_factory=tuple)
    sections: tuple[ReportSection, ...] = field(default_factory=tuple)
    notes: tuple[ReportNote, ...] = field(default_factory=tuple)
    generated_at_ist: datetime | None = None


def clean_inline(value: str) -> str:
    return _BOLD_RE.sub(r"\1", _BR_RE.sub(" ", str(value)).replace("&nbsp;", " ")).strip()


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    width = len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * width) + "|",
    ]
    for row in rows:
        cells = list(_coerce_row(row, width))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _coerce_row(row: Sequence[object], width: int) -> tuple[str, ...]:
    cells = [str(cell) for cell in row[:width]]
    return tuple([*cells, *[""] * max(0, width - len(cells))])


def document_to_markdown(document: ReportDocument) -> str:
    parts: list[str] = [block for block in document.pre_blocks if block.strip()]
    parts.extend(markdown_table(section.headers, section.rows) for section in document.sections)
    parts.extend(note.text for note in document.notes if note.text.strip())
    return "\n\n".join(parts)


def split_markdown_blocks(summary_text: str) -> tuple[list[str], list[str], list[str]]:
    tables, pre, post, seen_table = [], [], [], False
    for block in filter(None, (b.strip() for b in _BLOCK_RE.split(summary_text.strip()))):
        if _TABLE_SEP_RE.search(block):
            tables.append(block)
            seen_table = True
        elif seen_table:
            post.append(block)
        else:
            pre.append(block)
    return tables, pre, post


def parse_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    lines = [line for line in markdown.splitlines() if line.strip().startswith("|")]
    sep_idx = next((idx for idx, line in enumerate(lines) if is_table_separator(line)), -1)
    if sep_idx <= 0:
        return [], []

    headers = table_cells(lines[sep_idx - 1])
    width = len(headers)
    rows = [
        (table_cells(line) + [""] * width)[:width]
        for line in lines[sep_idx + 1 :]
        if not is_table_separator(line)
    ]
    return headers, rows


def table_cells(row: str) -> list[str]:
    return [clean_inline(cell) for cell in row.strip().strip("|").split("|")]


def is_table_separator(row: str) -> bool:
    cells = table_cells(row)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def is_section_row(row: Sequence[str]) -> bool:
    return bool(row) and bool(row[0]) and all(not cell for cell in row[1:])


def document_from_markdown(
    title: str,
    summary_text: str,
    *,
    table_titles: Sequence[str] = (),
    default_left_count: int = 1,
) -> ReportDocument:
    tables, pre, post = split_markdown_blocks(summary_text)
    sections: list[ReportSection] = []
    for index, markdown in enumerate(tables):
        headers, rows = parse_markdown_table(markdown)
        if not headers:
            continue
        section_title = (
            table_titles[index] if index < len(table_titles) else f"Report Table {index + 1}"
        )
        placement: Placement = "left" if index < default_left_count else "right"
        sections.append(
            ReportSection.from_rows(section_title, headers, rows, placement=placement)
        )

    return ReportDocument(
        title=title,
        pre_blocks=tuple(pre),
        sections=tuple(sections),
        notes=tuple(ReportNote(block, placement="right") for block in post),
    )
