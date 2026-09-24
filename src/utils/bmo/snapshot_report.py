"""A shift-report style Word document, generated from a snapshot.

NOT A SCREENSHOT. The page hides most of its content behind tabs and expanders,
and a picture of it would show whichever tab happened to be open. This builds the
report from the snapshot's DATA instead, so every section is written out in full
and in a fixed order a reader can rely on from one report to the next.
Sections with nothing to report are left out and the rest renumbered:

    1  Summary              the numbers a shift log would carry
    2  Recommended blend    shares, tonnes and a pie of the burden
    3  Cost                 ore, fuel, flux, total
    4  Fuel and coke        coke, nut coke, PCI, correction, silicon
    5  Slag                 rate, quantity, basicity ratios, alumina, magnesia
    6  LP against DE        side by side, when both ran
    7  Current blend        what the plant was charging, when recorded
    8  Process and path     when this build records them
    9  Furnace commentary   when one was generated
    10 Inputs               targets, limits and every input table used
    11 Provenance           branch, commit, and how to reproduce it

The pie is drawn with matplotlib, not taken from the page's Plotly chart:
exporting Plotly needs ``kaleido``, which is not installed, and matplotlib is.
"""

from __future__ import annotations

import io
import math
from typing import Any, Mapping

from utils.bmo.snapshot import decode, ore_names, recommended_result

TABLE_STYLE = "Light Grid Accent 1"

# Input suffix -> (label, format). Anything not listed still appears, with a
# label derived from its key, so a new page input is never dropped from a report.
INPUT_LABELS: dict[str, tuple[str, str]] = {
    "target_production_mt": ("Target hot metal (MT)", "{:,.1f}"),
    "target_slag_rate_kg_per_thm": ("Max slag rate (kg/THM)", "{:,.1f}"),
    "target_slag_basicity_min": ("B2 min (CaO/SiO2)", "{:.3f}"),
    "target_slag_basicity_max": ("B2 max (CaO/SiO2)", "{:.3f}"),
    "target_slag_t_basicity_min": ("T-basicity min", "{:.3f}"),
    "target_slag_t_basicity_max": ("T-basicity max", "{:.3f}"),
    "target_slag_al2o3_max_pct": ("Slag Al2O3 max (%)", "{:.2f}"),
    "target_slag_mgo_min_pct": ("Slag MgO min (%)", "{:.2f}"),
    "target_slag_mgo_al2o3_ratio_min": ("MgO/Al2O3 min", "{:.3f}"),
    "max_charges_per_hour": ("Max charges per hour", "{:.2f}"),
    "charge_mass_mt": ("Charge mass (MT)", "{:.2f}"),
    "burden_capacity_enabled": ("Enforce charging capacity", "{}"),
    "chemistry_mode": ("Chemistry mode", "{}"),
    "chemistry_window_days": ("Chemistry window (days)", "{}"),
    "pci_override_on": ("PCI override", "{}"),
    "pci_override_kg": ("PCI override (kg/THM)", "{:,.1f}"),
    "transition_move_pct": ("Transition step (%/rung)", "{:.1f}"),
}

TABLE_INPUTS = (
    ("applied_ore_editor_df", "Ores",
     ("ore_name", "min_share_pct", "max_share_pct", "stock_mt", "price_rs_per_mt",
      "moisture_pct", "fe_t_pct", "sio2_pct", "al2o3_pct", "cao_pct", "mgo_pct")),
    ("applied_flux_editor_df", "Flux",
     ("display_name", "enabled", "wet_qty_mt", "cao_pct", "mgo_pct", "sio2_pct",
      "al2o3_pct", "loi_pct", "price_rs_per_mt")),
    ("applied_fuel_ash_editor_df", "Fuel ash",
     ("display_name", "enabled", "rate_kg_per_thm", "ash_pct", "sio2_pct",
      "al2o3_pct", "moisture_pct", "price_rs_per_mt")),
    ("applied_dust_editor_df", "Dust",
     ("display_name", "enabled", "quantity_kg_per_charge", "wet_qty_mt", "fe_pct",
      "sio2_pct", "al2o3_pct", "cao_pct")),
)


# --- small helpers ---------------------------------------------------------------


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt(value: Any, spec: str = "{:,.2f}") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    number = _num(value)
    if number is not None and "{:" in spec and spec != "{}":
        try:
            return spec.format(number)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _label(suffix: str) -> str:
    return suffix.replace("_", " ").strip().capitalize()


def _table(doc, header: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(header))
    try:
        table.style = TABLE_STYLE
    except (KeyError, ValueError):
        table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = str(text)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    doc.add_paragraph()


def _kv(doc, pairs: list[tuple[str, str]]) -> None:
    _table(doc, ["Item", "Value"], [[k, v] for k, v in pairs])


def _pie_png(shares: Mapping[str, float], total_mt: float | None) -> bytes | None:
    items = [(k, float(v)) for k, v in shares.items() if _num(v) and float(v) > 0.05]
    if not items:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items.sort(key=lambda kv: -kv[1])
    labels, values = zip(*items)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=150)
    wedges, _texts, autotexts = ax.pie(
        values, labels=None, autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        startangle=90, counterclock=False, pctdistance=0.78,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(8)
    ax.legend(wedges, [f"{l} — {v:.1f}%" for l, v in items], loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    if total_mt:
        ax.text(0, 0, f"{total_mt:,.0f}\nMT burden", ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_title("Recommended blend — share of burden", fontsize=10)
    ax.axis("equal")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _result_metrics(fields: Mapping[str, Any], si: Any) -> list[tuple[str, Any, str]]:
    diag = fields.get("diagnostics") or {}
    rates = diag.get("fuel_rate_estimate") or {}
    flux = _num(diag.get("flux_cost_per_thm_rs")) or 0.0
    total = _num(diag.get("adjusted_objective_rs_per_thm"))
    if total is None:
        total = _num(fields.get("objective_rs_per_thm"))
    fuel = _num(diag.get("adjusted_fuel_cost_per_thm_rs"))
    if fuel is None:
        fuel = _num(fields.get("fuel_cost_per_thm_rs"))
    return [
        ("Total cost (Rs/THM)", (total + flux) if total is not None else None, "{:,.0f}"),
        ("Ore cost (Rs/THM)", fields.get("ore_cost_per_thm_rs"), "{:,.0f}"),
        ("Fuel cost (Rs/THM)", fuel, "{:,.0f}"),
        ("Flux cost (Rs/THM)", flux, "{:,.1f}"),
        ("Coke rate (kg/THM)", rates.get("coke_rate_kg_thm"), "{:,.1f}"),
        ("Nut coke (kg/THM)", rates.get("nut_coke_rate_kg_thm"), "{:,.1f}"),
        ("PCI (kg/THM)", rates.get("pci_rate_kg_thm"), "{:,.1f}"),
        ("Total fuel (kg/THM)", rates.get("total_fuel_rate_kg_thm"), "{:,.1f}"),
        ("Coke correction (kg/THM)", diag.get("coke_correction_delta_kg_thm"), "{:+,.1f}"),
        ("Hot metal Si (%)", si, "{:.3f}"),
        ("Burden Fe (%)", fields.get("fe_t_pct"), "{:.2f}"),
        ("Slag rate (kg/THM)", fields.get("slag_rate_kg_per_thm"), "{:,.1f}"),
        ("Slag (MT)", fields.get("slag_mt"), "{:,.1f}"),
        ("B2 CaO/SiO2", fields.get("slag_basicity"), "{:.3f}"),
        ("T-basicity (CaO+MgO)/SiO2", fields.get("slag_t_basicity"), "{:.3f}"),
        ("IB4 (CaO+MgO)/(SiO2+Al2O3)", fields.get("slag_ib4"), "{:.3f}"),
        ("Slag Al2O3 (%)", fields.get("slag_al2o3_pct"), "{:.2f}"),
        ("Slag MgO (%)", fields.get("slag_mgo_pct"), "{:.2f}"),
        ("MgO/Al2O3", fields.get("slag_mgo_al2o3_ratio"), "{:.3f}"),
        ("Feasible", fields.get("feasible"), "{}"),
    ]


def _fields(encoded: Any) -> dict[str, Any]:
    if isinstance(encoded, dict) and encoded.get("__type__") == "dataclass":
        return encoded.get("data") or {}
    return {}


# --- the document ----------------------------------------------------------------


def build_docx(snapshot: Mapping[str, Any]) -> bytes:
    """Render a snapshot as a Word document and return the file's bytes."""

    from docx import Document
    from docx.shared import Inches, Pt

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)

    summary = snapshot.get("summary") or {}
    results = snapshot.get("results") or {}
    context = snapshot.get("context") or {}
    inputs = snapshot.get("inputs") or {}
    names = ore_names(snapshot)
    basis, rec = recommended_result(snapshot)
    rec_si = results.get("de_si") if basis.startswith("DE") else results.get("lp_si")

    # Sections are numbered as they are written: some appear only when there is
    # something to put in them, and the numbering must not skip.
    counter = iter(range(1, 100))

    def section() -> str:
        return f"{next(counter)}. "

    doc.add_heading("Blend Mix Optimiser — Snapshot Report", level=0)
    _kv(doc, [
        ("Snapshot", str(snapshot.get("id") or "(unsaved)")),
        ("Taken at", str(snapshot.get("created_at") or "—")),
        ("Taken on page", str(snapshot.get("source") or "—")),
        ("Label", str(snapshot.get("label") or "—")),
        ("Recommendation basis", basis or "No optimiser result was recorded"),
    ])

    # 1. Summary
    doc.add_heading(f"{section()}Summary", level=1)
    blend_text = ", ".join(f"{k} {v:.1f}%" for k, v in (summary.get("blend_pct") or {}).items())
    _kv(doc, [
        ("Production (MT)", _fmt(summary.get("production_mt"), "{:,.1f}")),
        ("Blend", blend_text or "—"),
        ("Total cost (Rs/THM)", _fmt(summary.get("total_cost_rs_thm"), "{:,.0f}")),
        ("Fuel rate (kg/THM)", _fmt(summary.get("fuel_rate_kg_thm"), "{:,.1f}")),
        ("Coke rate (kg/THM)", _fmt(summary.get("coke_rate_kg_thm"), "{:,.1f}")),
        ("Slag rate (kg/THM)", _fmt(summary.get("slag_rate_kg_thm"), "{:,.1f}")),
        ("Basicity B2 / T-basicity",
         f"{_fmt(summary.get('basicity_b2'), '{:.3f}')} / "
         f"{_fmt(summary.get('t_basicity'), '{:.3f}')}"),
        ("Feasible", _fmt(summary.get("feasible"))),
    ])

    if not rec:
        doc.add_paragraph(
            "No optimiser result was recorded in this snapshot — it holds inputs "
            "only. Sections 2 to 6 are therefore empty; the inputs are in section 10."
        )
    else:
        # 2. Recommended blend
        doc.add_heading(f"{section()}Recommended blend ({basis})", level=1)
        shares = rec.get("shares_pct") or {}
        qty = rec.get("quantities_mt") or {}
        rows = [
            [names.get(str(k), str(k)), _fmt(v, "{:.2f}"), _fmt(qty.get(k), "{:,.1f}")]
            for k, v in sorted(shares.items(), key=lambda kv: -(_num(kv[1]) or 0))
            if (_num(v) or 0) > 0.05
        ]
        _table(doc, ["Ore", "Share (%)", "Wet quantity (MT)"], rows)
        png = _pie_png({names.get(str(k), str(k)): v for k, v in shares.items()},
                       _num(rec.get("total_qty_mt")))
        if png:
            doc.add_picture(io.BytesIO(png), width=Inches(5.8))

        metrics = _result_metrics(rec, rec_si)
        by_label = {m[0]: m for m in metrics}

        def rows_for(labels: list[str]) -> list[tuple[str, str]]:
            return [(l, _fmt(by_label[l][1], by_label[l][2])) for l in labels if l in by_label]

        # 3. Cost
        doc.add_heading(f"{section()}Cost", level=1)
        _kv(doc, rows_for(["Total cost (Rs/THM)", "Ore cost (Rs/THM)",
                           "Fuel cost (Rs/THM)", "Flux cost (Rs/THM)"]))

        # 4. Fuel and coke
        doc.add_heading(f"{section()}Fuel and coke", level=1)
        _kv(doc, rows_for(["Coke rate (kg/THM)", "Nut coke (kg/THM)", "PCI (kg/THM)",
                           "Total fuel (kg/THM)", "Coke correction (kg/THM)",
                           "Hot metal Si (%)"]))

        # 5. Slag
        doc.add_heading(f"{section()}Slag", level=1)
        _kv(doc, rows_for(["Burden Fe (%)", "Slag rate (kg/THM)", "Slag (MT)",
                           "B2 CaO/SiO2", "T-basicity (CaO+MgO)/SiO2",
                           "IB4 (CaO+MgO)/(SiO2+Al2O3)", "Slag Al2O3 (%)",
                           "Slag MgO (%)", "MgO/Al2O3", "Feasible"]))
        violations = rec.get("violations") or []
        if violations:
            doc.add_paragraph("Constraint violations:")
            for v in violations:
                doc.add_paragraph(str(v), style="List Bullet")

    # 6. LP against DE
    lp, de = _fields(results.get("lp_result")), _fields(results.get("de_result"))
    if lp and de:
        doc.add_heading(f"{section()}LP baseline against DE total cost", level=1)
        lp_m = _result_metrics(lp, results.get("lp_si"))
        de_m = {m[0]: m for m in _result_metrics(de, results.get("de_si"))}
        _table(doc, ["Metric", "LP baseline", "DE total cost"],
               [[l, _fmt(v, s), _fmt(de_m[l][1], de_m[l][2])] for l, v, s in lp_m])
        if (de.get("diagnostics") or {}).get("de_fell_back_to_lp"):
            doc.add_paragraph("DE did not improve on the LP; the LP blend is reported.")

    # 7. Current blend
    manual = decode(results.get("manual_quantities_mt")) or {}
    if isinstance(manual, dict) and manual:
        total = sum((_num(v) or 0.0) for v in manual.values())
        if total > 0:
            doc.add_heading(f"{section()}Current blend (last shift, as compared)", level=1)
            _table(doc, ["Ore", "Share (%)", "Quantity (MT)"], [
                [names.get(str(k), str(k)), _fmt((_num(v) or 0) / total * 100, "{:.2f}"),
                 _fmt(v, "{:,.1f}")]
                for k, v in sorted(manual.items(), key=lambda kv: -(_num(kv[1]) or 0))
                if (_num(v) or 0) > 0
            ])

    # 8. Process and path
    doc.add_heading(f"{section()}Process parameters and transition path", level=1)
    process = decode(results.get("process_recommendation"))
    ladder = decode(results.get("transition_ladder"))
    if process or ladder:
        for title, block in (("Process parameters", process), ("Transition path", ladder)):
            if isinstance(block, list) and block and isinstance(block[0], dict):
                doc.add_paragraph(title)
                header = list(block[0].keys())
                _table(doc, header, [[_fmt(r.get(h)) for h in header] for r in block])
    else:
        doc.add_paragraph(
            "Not recorded. On this build the process recommendation and the "
            "transition path are computed while the page is displayed and are not "
            "kept in the page state, so a snapshot cannot carry them."
        )

    # 9. Commentary
    commentary = decode(results.get("commentary"))
    if commentary:
        doc.add_heading(f"{section()}Furnace commentary", level=1)
        for block in str(commentary).split("\n\n"):
            if block.strip():
                doc.add_paragraph(block.replace("**", "").strip())

    # 10. Inputs
    doc.add_heading(f"{section()}Inputs", level=1)
    scalar_rows = []
    for suffix, encoded in sorted(inputs.items()):
        value = decode(encoded)
        if isinstance(value, (dict, list)) or hasattr(value, "columns"):
            continue
        label, spec = INPUT_LABELS.get(suffix, (_label(suffix), "{}"))
        scalar_rows.append((label, _fmt(value, spec)))
    if scalar_rows:
        doc.add_paragraph("Targets, limits and settings")
        _kv(doc, scalar_rows)

    hm = decode(context.get("hm_chem_values"))
    if isinstance(hm, dict) and hm:
        doc.add_paragraph("Hot metal chemistry assumed")
        _kv(doc, [(_label(k), _fmt(v, "{:.3f}")) for k, v in hm.items()])

    for suffix, title, preferred in TABLE_INPUTS:
        frame = decode(inputs.get(suffix))
        if frame is None or not hasattr(frame, "columns") or frame.empty:
            continue
        if "selected" in frame.columns:
            frame = frame[frame["selected"].astype(bool)]
        if "enabled" in frame.columns and title != "Ores":
            frame = frame[frame["enabled"].astype(bool)]
        cols = [c for c in preferred if c in frame.columns] or list(frame.columns[:8])
        doc.add_paragraph(f"{title} ({len(frame)} rows)")
        _table(doc, [_label(c) for c in cols],
               [[_fmt(row[c]) for c in cols] for _, row in frame.iterrows()])

    errors = [e for key in ("lp_errors", "de_errors") for e in (decode(results.get(key)) or [])]
    if errors:
        doc.add_heading("Optimiser messages", level=2)
        for e in errors:
            doc.add_paragraph(str(e), style="List Bullet")

    # 11. Provenance
    doc.add_heading(f"{section()}Provenance", level=1)
    prov = snapshot.get("provenance") or {}
    _kv(doc, [
        ("Branch", prov.get("branch", "—")),
        ("Commit", prov.get("commit", "—")),
        ("Inputs recorded", str(len(inputs))),
        ("Not restorable (buttons / editors)", ", ".join(snapshot.get("not_restorable") or []) or "—"),
    ])
    doc.add_paragraph(
        "To reproduce: open TestBMO, load this snapshot, and run the optimiser. "
        "TestBMO re-runs against today's code and model files, so results can "
        "differ from those recorded here if either has changed since."
    )

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
