"""Structured presentation for furnace reports."""

from __future__ import annotations

from typing import Optional, Sequence

from reports.rendering import (
    ReportDocument,
    ReportNote,
    ReportSection,
    document_from_markdown as generic_document_from_markdown,
    document_to_markdown,
    is_section_row,
    markdown_table,
    parse_markdown_table,
)
from reports.furnace_report.data import ParamStats, ShiftReportData, TempRow

SHIFT_SECTION_TITLES = (
    "Fuel Rate",
    "Parameters",
    "Temperature Profile",
    "Hearth Pad Temperature",
    "Tapping Summary",
    "Consumption",
)


def _v(val: Optional[float]) -> str:
    return f"{val:.2f}" if val is not None else "-"


def _v1(val: Optional[float]) -> str:
    return f"{val:.1f}" if val is not None else "-"


def _r(val: Optional[float]) -> str:
    return f"{val:.0f}" if val is not None else "-"


def _vi(val: Optional[int]) -> str:
    return str(val) if val is not None else "-"


def _ps(stats: ParamStats) -> tuple[str, str]:
    return _v(stats.mean), _v(stats.std)


def _psr(stats: ParamStats) -> tuple[str, str]:
    return _r(stats.mean), _r(stats.std)


def _temp_cells(row: TempRow) -> list[str]:
    return [_r(row.q1), _r(row.q2), _r(row.q3), _r(row.q4), _r(row.spread_std)]


def as_document(
    report: ShiftReportData,
    analysis: str = "",
    *,
    title: str = "",
) -> ReportDocument:
    """Render computed shift data as a structured report document."""
    r = report
    heading = (
        "SHIFT HANDOVER REPORT - BF2 EVONITH STEEL"
        if r.report_type == "Shift"
        else "DAY REPORT - BF2 EVONITH STEEL"
    )
    id_label = "Shift ID" if r.report_type == "Shift" else "Report ID"
    id_value = r.shift_label if r.report_type == "Shift" else "Day"
    preamble = (
        f"**{heading}**\n"
        f"**{id_label}:** {id_value} &nbsp;"
        f" **Period:** {r.shift_start_ist.strftime('%Y-%m-%d %H:%M')}"
        f" -> {r.shift_end_ist.strftime('%Y-%m-%d %H:%M')} (IST)"
    )

    sections = (
        ReportSection.from_rows(
            "Fuel Rate",
            [
                "Fuel rate (kg/thm)",
                "Coke rate (kg/thm)",
                "Nut Coke rate (kg/thm)",
                "PCI rate (kg/thm)",
            ],
            [[_v(r.fuel_rate), _v(r.coke_rate), _v(r.nut_coke_rate), _v(r.pci_rate)]],
            placement="left",
        ),
        ReportSection.from_rows(
            "Parameters",
            ["Parameter", "UOM", "Value", "Std.Dev"],
            _parameter_rows(r),
            placement="left",
        ),
        ReportSection.from_rows(
            "Temperature Profile",
            ["Parameter", "Q1", "Q2", "Q3", "Q4", "Std.Dev"],
            [
                ["Uptake Temp (degC)", *_temp_cells(r.uptake)],
                ["Skin flow", "-", "-", "-", "-", "-"],
                ["Lower stack (degC)", *_temp_cells(r.lower_stack)],
                ["Belly (degC)", *_temp_cells(r.belly)],
                ["Bosh (degC)", *_temp_cells(r.bosh)],
            ],
        ),
        ReportSection.from_rows(
            "Hearth Pad Temperature",
            ["Parameter", "4.3mtr A", "5.4mtr C", "5.7mtr C", "6.1mtr B"],
            [
                [
                    "Hearth Pad Temp (degC)",
                    _r(r.hearth_4_3_a),
                    _r(r.hearth_5_4_c),
                    _r(r.hearth_5_7_c),
                    _r(r.hearth_6_1_b),
                ]
            ],
        ),
        ReportSection.from_rows(
            "Tapping Summary",
            [
                "Total Taps (no's)",
                "Tap Duration (mins)",
                "Slag Duration (mins)",
                "Slag Ratio (%)",
                "Casting Rate (T/min)",
            ],
            [[_vi(r.total_taps), "-", "-", "-", "-"]],
        ),
        ReportSection.from_rows(
            "Consumption",
            [
                "Coke (tons)",
                "Nut coke (tons)",
                "Ore (tons)",
                "Flux (tons)",
                "Sinter (tons)",
                "Pellet (tons)",
            ],
            [
                [
                    _v(r.coke_t),
                    _v(r.nut_coke_t),
                    _v(r.ore_t),
                    _v(r.flux_t),
                    _v(r.sinter_t),
                    _v(r.pellet_t),
                ]
            ],
        ),
    )

    notes = []
    if r.used_materials:
        notes.append(
            ReportNote(
                "\n\n".join(
                    f"{category} :\n{materials}"
                    for category, materials in r.used_materials.items()
                ),
                kind="material_usage",
            )
        )
    if analysis.strip():
        notes.append(ReportNote(f"**Report Analysis**\n\n{analysis.strip()}", kind="analysis"))

    return ReportDocument(
        title=title,
        pre_blocks=(preamble,),
        sections=sections,
        notes=tuple(notes),
    )


def as_markdown(report: ShiftReportData, analysis: str = "") -> str:
    """Render a full shift handover report as markdown."""
    return document_to_markdown(as_document(report, analysis))


def document_from_markdown(title: str, summary_text: str) -> ReportDocument:
    """Build a structured report document from saved markdown."""
    tables, pre, post = _normalize_shift_markdown_tables(summary_text)
    document = generic_document_from_markdown(
        title,
        "\n\n".join([*pre, *tables, *post]),
        table_titles=SHIFT_SECTION_TITLES,
        default_left_count=2,
    )
    return ReportDocument(
        title=document.title,
        pre_blocks=document.pre_blocks,
        sections=document.sections,
        notes=tuple(_classify_shift_note(note) for note in document.notes),
    )


def _classify_shift_note(note: ReportNote) -> ReportNote:
    first_line = note.text.strip().splitlines()[0] if note.text.strip() else ""
    if first_line.startswith(("Flux :", "Ore :")):
        return ReportNote(note.text, placement=note.placement, kind="material_usage")
    if first_line in {"**Shift Analysis**", "**Report Analysis**"}:
        return ReportNote(note.text, placement=note.placement, kind="analysis")
    return note


def _parameter_rows(r: ShiftReportData) -> list[list[str]]:
    bv, bs = _psr(r.blast_volume)
    bt, bts = _psr(r.blast_temp)
    bp, bps = _ps(r.blast_pressure)
    top_dp, top_dps = _ps(r.furnace_top_dp)
    bottom_dp, bottom_dps = _ps(r.furnace_bottom_dp)
    total_dp, total_dps = _ps(r.furnace_total_dp)
    pe, pes = _psr(r.permeability)
    ec, ecs = _ps(r.etaco)
    ra, ras = _psr(r.raft)
    of, ofs = _psr(r.o2_flow)
    oe, oes = _ps(r.o2_enrichment)

    process_rows = [
        ["Hot blast volume", "Nm3/hr", bv, bs],
        ["Hot blast temperature", "degC", bt, bts],
        ["Hot blast pressure", "bar", bp, bps],
        ["Furnace Top DP", "bar", top_dp, top_dps],
        ["Furnace Bottom DP", "bar", bottom_dp, bottom_dps],
        ["Furnace Total DP", "bar", total_dp, total_dps],
        ["Oxygen Flow", "Nm3/hr", of, ofs],
    ]
    if r.report_type == "Day":
        process_rows.append(["Day Total Oxygen Flow", "Nm3", _r(r.total_o2_flow), "-"])

    process_rows.extend(
        [
            ["Oxygen enrichment", "%", oe, oes],
            ["Permeability", "-", pe, pes],
            ["ETA CO", "%", ec, ecs],
            ["RAFT", "degC", ra, ras],
            ["Burden Moisture Input", "kg/thm", _r(r.burden_moisture_input), "-"],
            ["Fines Input", "kg/thm", _r(r.fines_input), "-"],
            ["IBRM", "-", _v1(r.ibrm), "-"],
        ]
    )

    return [
        ["**Production**", "", "", ""],
        ["Production rate", "t/hr", _v(r.production_rate), "-"],
        ["Theoretical Production", "tons", _v(r.theoretical_production), "-"],
        ["Total Charges", "no's", _vi(r.total_charges), "-"],
        ["Burden Ratio", "Sinter:Ore:Pellet", r.burden_ratio or "-", "-"],
        ["**Process Parameters**", "", "", ""],
        *process_rows,
        ["**Quality**", "", "", ""],
        ["HM Si", "%", _v(r.hm_si), "-"],
        ["HM S", "%", _v(r.hm_s), "-"],
        ["HM Temperature", "degC", _r(r.hm_temp), "-"],
        ["Slag Basicity", "-", _v(r.slag_basicity), "-"],
    ]


def _normalize_shift_markdown_tables(
    summary_text: str,
) -> tuple[list[str], list[str], list[str]]:
    from reports.rendering import split_markdown_blocks

    tables, pre, post = split_markdown_blocks(summary_text)
    normalized = [_normalize_tapping_table(table) for table in tables]
    if len(normalized) < 2:
        return normalized, pre, post

    first_headers, first_rows = parse_markdown_table(normalized[0])
    previous_split = _split_previous_consumption_table(first_headers, first_rows)
    if previous_split is not None:
        fuel_rate_table, consumption_table = previous_split
        return [fuel_rate_table, *normalized[1:], consumption_table], pre, post

    second_headers, second_rows = parse_markdown_table(normalized[1])
    if first_headers != ["Parameter", "UOM", "Value"]:
        return normalized, pre, post
    if second_headers != ["Parameter", "UOM", "Value", "Std.Dev"]:
        return normalized, pre, post

    section_names = {row[0] for row in first_rows if is_section_row(row)}
    if not {"Consumption", "Quality"}.issubset(section_names):
        return normalized, pre, post

    sections = _legacy_shift_sections(first_rows)
    consumption_lookup = {
        row[0].lower(): row[2]
        for row in sections.get("Consumption", [])
        if len(row) >= 3
    }
    params_table = _legacy_parameter_table(sections, second_rows)
    return [
        _fuel_rate_table(consumption_lookup),
        params_table,
        *normalized[2:],
        _consumption_table(consumption_lookup),
    ], pre, post


def _legacy_parameter_table(
    sections: dict[str, list[list[str]]],
    second_rows: Sequence[Sequence[str]],
) -> str:
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
    parameter_rows.extend([list(row) for row in second_rows])
    return markdown_table(["Parameter", "UOM", "Value", "Std.Dev"], parameter_rows)


def _split_previous_consumption_table(
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
    return _fuel_rate_table(values), _consumption_table(values)


def _fuel_rate_table(values: dict[str, str]) -> str:
    headers = [
        "Fuel rate (kg/thm)",
        "Coke rate (kg/thm)",
        "Nut Coke rate (kg/thm)",
        "PCI rate (kg/thm)",
    ]
    keys = ["fuel rate", "coke rate", "nut coke rate", "pci rate"]
    return markdown_table(
        headers,
        [[values.get(header.lower(), values.get(key, "-")) for header, key in zip(headers, keys)]],
    )


def _consumption_table(values: dict[str, str]) -> str:
    headers = [
        "Coke (tons)",
        "Nut coke (tons)",
        "Ore (tons)",
        "Flux (tons)",
        "Sinter (tons)",
        "Pellet (tons)",
    ]
    keys = ["coke", "nut coke", "ore", "flux", "sinter", "pellet"]
    return markdown_table(
        headers,
        [[values.get(header.lower(), values.get(key, "-")) for header, key in zip(headers, keys)]],
    )


def _normalize_tapping_table(table: str) -> str:
    headers, rows = parse_markdown_table(table)
    if headers != ["Parameter", "UOM", "Value"]:
        return table

    values = {row[0].lower(): row[2] for row in rows if len(row) >= 3}
    if "total taps" not in values:
        return table

    return markdown_table(
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


def _legacy_shift_sections(rows: list[list[str]]) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {
        "Production": [],
        "Consumption": [],
        "Quality": [],
    }
    current = "Production"
    for row in rows:
        if is_section_row(row):
            current = row[0]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(row)
    return sections
