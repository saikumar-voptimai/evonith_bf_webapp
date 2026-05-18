"""Convert ShiftReportData → markdown string.

No Streamlit imports — the output is plain text that ui/components.show_report()
renders.  This keeps the renderer fully unit-testable.
"""

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
    return f"| {_v(row.q1)} | {_v(row.q2)} | {_v(row.q3)} | {_v(row.q4)} | {_v(row.spread_std)} |"


def as_markdown(report: ShiftReportData, analysis: str = "") -> str:
    """Render a full shift handover report as a markdown string."""
    r = report
    flags_text = "; ".join(r.status_flags) if r.status_flags else "None"

    header = (
        f"**SHIFT HANDOVER REPORT — BF2 EVONITH STEEL**\n"
        f"**Shift ID:** {r.shift_label} &nbsp;"
        f" **Period:** {r.shift_start_ist.strftime('%Y-%m-%d %H:%M')}"
        f" → {r.shift_end_ist.strftime('%Y-%m-%d %H:%M')} (IST)\n\n"
        f"**Status:** {r.status}\n\n"
        f"**Flags:** {flags_text}"
    )

    shift_table = (
        "| Parameter | UOM | Value |\n"
        "|---|---|---|\n"
        f"| Production rate | t/hr | {_v(r.production_rate)} |\n"
        f"| Theoretical Production | tons | {_v(r.theoretical_production)} |\n"
        f"| Total Charges | no's | {_vi(r.total_charges)} |\n"
        "| **Consumption** | | |\n"
        f"| Coke | tons | {_v(r.coke_t)} |\n"
        f"| Nut coke | tons | {_v(r.nut_coke_t)} |\n"
        f"| Sinter | tons | {_v(r.sinter_t)} |\n"
        f"| Ore | tons | {_v(r.ore_t)} |\n"
        f"| Pellet | tons | {_v(r.pellet_t)} |\n"
        f"| Flux | tons | {_v(r.flux_t)} |\n"
        f"| Fuel rate | kg/thm | {_v(r.fuel_rate)} |\n"
        f"| Coke rate | kg/thm | {_v(r.coke_rate)} |\n"
        f"| Nut coke rate | kg/thm | {_v(r.nut_coke_rate)} |\n"
        f"| PCI rate | kg/thm | {_v(r.pci_rate)} |\n"
        "| **Quality** | | |\n"
        f"| HM Si | % | {_v(r.hm_si)} |\n"
        f"| HM S | % | {_v(r.hm_s)} |\n"
        f"| HM Temperature | degC | {_v(r.hm_temp)} |\n"
        f"| Slag Basicity | — | {_v(r.slag_basicity)} |"
    )

    bv, bs = _ps(r.blast_volume)
    bt, bts = _ps(r.blast_temp)
    bp, bps = _ps(r.blast_pressure)
    pe, pes = _ps(r.permeability)
    ec, ecs = _ps(r.etaco)
    ra, ras = _ps(r.raft)
    of, ofs = _ps(r.o2_flow)
    oe, oes = _ps(r.o2_enrichment)

    params_table = (
        "| Parameter | UOM | Value | Std.Dev |\n"
        "|---|---|---|---|\n"
        f"| Hot blast volume | Nm3/hr | {bv} | {bs} |\n"
        f"| Hot blast temperature | degC | {bt} | {bts} |\n"
        f"| Hot blast pressure | bar | {bp} | {bps} |\n"
        f"| Oxygen Flow | Nm3/hr | {of} | {ofs} |\n"
        f"| Oxygen enrichment | % | {oe} | {oes} |\n"
        f"| Permeability | — | {pe} | {pes} |\n"
        f"| ETA CO | % | {ec} | {ecs} |\n"
        f"| RAFT | degC | {ra} | {ras} |\n"
        f"| Burden Moisture Input | kg/thm | {_v(r.burden_moisture_input)} | - |\n"
        f"| Fines Input | kg/thm | {_v(r.fines_input)} | - |"
    )

    temp_table = (
        "| Parameter | Q1 | Q2 | Q3 | Q4 | Std.Dev |\n"
        "|---|---|---|---|---|---|\n"
        f"| Uptake Temp (degC) {_tr(r.uptake)}\n"
        "| Skin flow | - | - | - | - | - |\n"
        f"| Lower stack (degC) {_tr(r.lower_stack)}\n"
        f"| Belly (degC) {_tr(r.belly)}\n"
        f"| Bosh delta-T {_tr(r.bosh)}"
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
        "| Parameter | UOM | Value |\n"
        "|---|---|---|\n"
        f"| Total Taps | no's | {_vi(r.total_taps)} |\n"
        "| Tap Duration | mins | - |\n"
        "| Slag duration | mins | - |\n"
        "| Slag ratio | % | - |\n"
        "| Casting rate | T/min | - |"
    )

    parts = [header, shift_table, params_table, temp_table, hearth_table, tapping_table]

    if analysis.strip():
        parts.append(f"**Shift Analysis**\n\n{analysis.strip()}")

    return "\n\n".join(parts)
