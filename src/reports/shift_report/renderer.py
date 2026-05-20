"""Convert ShiftReportData to markdown tables for the Streamlit report view."""

from __future__ import annotations

from typing import Optional

from reports.shift_report.data import ParamStats, ShiftReportData, TempRow


def _v(val: Optional[float]) -> str:
    return f"{val:.2f}" if val is not None else "-"


def _vi(val: Optional[int]) -> str:
    return str(val) if val is not None else "-"


def _ps(stats: ParamStats) -> tuple[str, str]:
    return _v(stats.mean), _v(stats.std)


def _tr(row: TempRow) -> str:
    return (
        f"| {_v(row.q1)} | {_v(row.q2)} | {_v(row.q3)} | "
        f"{_v(row.q4)} | {_v(row.spread_std)} |"
    )


def as_markdown(report: ShiftReportData, analysis: str = "") -> str:
    """Render a full shift handover report as markdown."""
    r = report
    flags_text = "; ".join(r.status_flags) if r.status_flags else "None"

    header = (
        "**SHIFT HANDOVER REPORT - BF2 EVONITH STEEL**\n"
        f"**Shift ID:** {r.shift_label} &nbsp;"
        f" **Period:** {r.shift_start_ist.strftime('%Y-%m-%d %H:%M')}"
        f" -> {r.shift_end_ist.strftime('%Y-%m-%d %H:%M')} (IST)\n\n"
        f"**Status:** {r.status}\n\n"
        f"**Flags:** {flags_text}"
    )

    fuel_rate_table = (
        "| Fuel rate (kg/thm) | Coke rate (kg/thm) | "
        "Nut Coke rate (kg/thm) | PCI rate (kg/thm) |\n"
        "|---|---|---|---|\n"
        f"| {_v(r.fuel_rate)} | {_v(r.coke_rate)} | "
        f"{_v(r.nut_coke_rate)} | {_v(r.pci_rate)} |"
    )

    bv, bs = _ps(r.blast_volume)
    bt, bts = _ps(r.blast_temp)
    bp, bps = _ps(r.blast_pressure)
    top_dp, top_dps = _ps(r.furnace_top_dp)
    bottom_dp, bottom_dps = _ps(r.furnace_bottom_dp)
    total_dp, total_dps = _ps(r.furnace_total_dp)
    pe, pes = _ps(r.permeability)
    ec, ecs = _ps(r.etaco)
    ra, ras = _ps(r.raft)
    of, ofs = _ps(r.o2_flow)
    oe, oes = _ps(r.o2_enrichment)

    params_table = (
        "| Parameter | UOM | Value | Std.Dev |\n"
        "|---|---|---|---|\n"
        "| **Production** | | | |\n"
        f"| Production rate | t/hr | {_v(r.production_rate)} | - |\n"
        f"| Theoretical Production | tons | {_v(r.theoretical_production)} | - |\n"
        f"| Total Charges | no's | {_vi(r.total_charges)} | - |\n"
        "| **Process Parameters** | | | |\n"
        f"| Hot blast volume | Nm3/hr | {bv} | {bs} |\n"
        f"| Hot blast temperature | degC | {bt} | {bts} |\n"
        f"| Hot blast pressure | bar | {bp} | {bps} |\n"
        f"| Furnace Top DP | bar | {top_dp} | {top_dps} |\n"
        f"| Furnace Bottom DP | bar | {bottom_dp} | {bottom_dps} |\n"
        f"| Furnace Total DP | bar | {total_dp} | {total_dps} |\n"
        f"| Oxygen Flow | Nm3/hr | {of} | {ofs} |\n"
        f"| Oxygen enrichment | % | {oe} | {oes} |\n"
        f"| Permeability | - | {pe} | {pes} |\n"
        f"| ETA CO | % | {ec} | {ecs} |\n"
        f"| RAFT | degC | {ra} | {ras} |\n"
        f"| Burden Moisture Input | kg/thm | {_v(r.burden_moisture_input)} | - |\n"
        f"| Fines Input | kg/thm | {_v(r.fines_input)} | - |\n"
        "| **Quality** | | | |\n"
        f"| HM Si | % | {_v(r.hm_si)} | - |\n"
        f"| HM S | % | {_v(r.hm_s)} | - |\n"
        f"| HM Temperature | degC | {_v(r.hm_temp)} | - |\n"
        f"| Slag Basicity | - | {_v(r.slag_basicity)} | - |\n"
    )

    temp_table = (
        "| Parameter | Q1 | Q2 | Q3 | Q4 | Std.Dev |\n"
        "|---|---|---|---|---|---|\n"
        f"| Uptake Temp (degC) {_tr(r.uptake)}\n"
        "| Skin flow | - | - | - | - | - |\n"
        f"| Lower stack (degC) {_tr(r.lower_stack)}\n"
        f"| Belly (degC) {_tr(r.belly)}\n"
        f"| Bosh (degC) {_tr(r.bosh)}"
    )

    hearth_table = (
        "| Parameter | 4.3mtr A | 5.4mtr C | 5.7mtr C | 6.1mtr B |\n"
        "|---|---|---|---|---|\n"
        f"| Hearth Pad Temp (degC)"
        f" | {_v(r.hearth_4_3_a)}"
        f" | {_v(r.hearth_5_4_c)}"
        f" | {_v(r.hearth_5_7_c)}"
        f" | {_v(r.hearth_6_1_b)} |"
    )

    tapping_table = (
        "| Total Taps (no's) | Tap Duration (mins) | "
        "Slag Duration (mins) | Slag Ratio (%) | Casting Rate (T/min) |\n"
        "|---|---|---|---|---|\n"
        f"| {_vi(r.total_taps)} | - | - | - | - |"
    )

    consumption_table = (
        "| Coke (tons) | Nut coke (tons) | Ore (tons) | Flux (tons) | "
        "Sinter (tons) | Pellet (tons) |\n"
        "|---|---|---|---|---|---|\n"
        f"| {_v(r.coke_t)} | {_v(r.nut_coke_t)} | {_v(r.ore_t)} | "
        f"{_v(r.flux_t)} | {_v(r.sinter_t)} | {_v(r.pellet_t)} |"
    )

    material_usage_note = ""
    if r.used_materials:
        material_usage_note = "\n\n".join(
            f"{category} : {materials}"
            for category, materials in r.used_materials.items()
        )

    parts = [
        header,
        fuel_rate_table,
        params_table,
        temp_table,
        hearth_table,
        tapping_table,
        consumption_table,
    ]
    if material_usage_note:
        parts.append(material_usage_note)

    if analysis.strip():
        parts.append(f"**Shift Analysis**\n\n{analysis.strip()}")

    return "\n\n".join(parts)
