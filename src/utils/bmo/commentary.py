"""Context assembly for the LLM's blend commentary.

WHAT THIS IS FOR.

A blend recommendation arrives as forty numbers with no argument attached. The
operator has to decide whether to act on it, and that decision needs the case
stated: why this blend rather than the current one, what it costs, and what
could go wrong. This module assembles everything the model needs to make that
case, and the next module renders what it says.

THE DESIGN RULE: NO LLM IN HERE.

Context assembly is pure and testable - dictionaries and strings in, one string
out. The model call lives in ``ui/bmo/commentary.py``. That separation is what
lets the interesting question ("was the model given the right facts?") be
answered by a unit test rather than by reading generated prose and hoping.

WHY THE CAVEATS ARE PART OF THE CONTEXT.

The numbers on this page carry known, measured defects, and a model that does
not know about them will write confident nonsense:

  - the coke rate is an energy balance plus a fitted bias offset, currently
    about +24 kg/THM. That offset measures how much the balance is still
    missing. It is not a fact about the furnace.
  - the operator's coke SETPOINT and the coke actually CHARGED differ by about
    4%, and agree within 10 kg/THM on only 42% of days.
  - the silicon model takes lagged silicon as an input, so its accuracy is
    partly yesterday's cast rather than burden chemistry.
  - the ML fuel-cost model is nearly blend-blind. All of the blend-to-fuel
    sensitivity comes from the physics correction, not from the model.

These are stated to the model as plainly as they are stated in the docs, so its
commentary can carry the same qualifications an honest engineer would.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)

# Kept small on purpose. A model handed a hundred series will average over them;
# handed the eight that drive fuel and stability, it can actually reason.
_TREND_FIELDS: dict[str, str] = {
    "fuel_rate": "Fuel rate (kg/THM)",
    "coke_rate": "Coke rate setpoint (kg/THM)",
    "coal_rate_actual_value": "PCI rate (kg/THM)",
    "production_per_hour": "Production (t/hr)",
    "body_etaco": "ETA CO (%)",
    "body_raft": "RAFT (C)",
    "body_perm": "Permeability index",
    "body_dp_total": "Total dP (bar)",
    "hot_blast_temp": "Hot blast temperature (C)",
    "hot_blast_vol_nm3h": "Hot blast volume (Nm3/hr)",
}

_LIVE_FIELDS: dict[str, str] = {
    "hot_blast_vol_nm3h": "Hot blast volume (Nm3/hr)",
    "hot_blast_temp": "Hot blast temperature (C)",
    "hot_blast_press": "Hot blast pressure (bar)",
    "oxygen_enrichment_pct": "Oxygen enrichment (%)",
    "steam_injection": "Steam (kg/hr)",
    "top_press_avg": "Top pressure (bar)",
    "top_temp_avg": "Top gas temperature (C)",
    "co_pct": "Top gas CO (%)",
    "co2_pct": "Top gas CO2 (%)",
    "h2_pct": "Top gas H2 (%)",
    "coal_rate_actual_value": "PCI, live tag (kg/THM)",
    "production_per_hour": "Production (t/hr)",
    "shell_loss_gj_per_hr": "Shell heat loss (GJ/hr)",
}

SYSTEM_PROMPT = """\
You are a senior blast furnace advisor writing a short commentary for the \
operating team at Evonith Steel BF-2. They are practising metallurgists. Do not \
explain what a blast furnace is, what coke does, or what basicity means.

WRITE IN INDIAN PROFESSIONAL ENGLISH - the register used in Indian plant \
engineering. Point-wise where it helps. Precise, not florid.

RULES YOU MUST FOLLOW:

1. Use ONLY the figures supplied below. Do not invent a number, a trend or a \
   material. If something needed for a judgement is missing, say plainly that \
   it is not available rather than estimating it.
2. Every claim gets ACTION, REASON and MAGNITUDE. "Raise blast temperature" is \
   useless; "raise blast temperature by 15 C to recover the 4 kg/THM of coke \
   the lower burden Fe is costing" is usable.
3. Respect the stated caveats. Where a number carries a known defect, say so in \
   the same breath as the number. Do not present the coke rate as measured fact.
4. If the recommended blend has constraint violations, lead with them. A cost \
   saving on an infeasible blend is not a saving.
5. Be willing to say the recommendation is not worth acting on. A small saving \
   inside the model's own error band is not a reason to change the burden, and \
   saying so is more useful than manufacturing a case.

STRUCTURE - use these four headings exactly:

**Furnace at present** - two or three lines on how the furnace is running now, \
from the live and 3-day figures. Note anything drifting.

**What the optimizer is proposing** - the blend change, the cost effect, and \
the fuel effect. State the delta against the current blend, not just levels.

**Why this makes sense** - the metallurgical argument. Tie the blend change to \
the slag chemistry, the burden Fe, and the fuel consequence. If the argument is \
weak, say so.

**What to watch out for** - the specific risks. Which constraint is closest to \
its bound, what could move against them in the next few shifts, and what to \
check first if the furnace does not respond as expected.

Keep the whole thing under 450 words."""


@dataclass
class CommentaryContext:
    """Assembled context, plus a record of what could not be gathered."""

    text: str
    missing: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        """A commentary with no blend and no furnace state is not worth asking for."""

        return bool(self.text.strip()) and "RECOMMENDED BLEND" in self.text


def _fmt(value: Any, spec: str = ",.2f") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number:
        return "n/a"
    return format(number, spec)


def _section(title: str, lines: Sequence[str]) -> str:
    if not lines:
        return ""
    body = "\n".join(f"  {line}" for line in lines)
    return f"\n{title}\n{body}\n"


def describe_live_conditions(snapshot: Mapping[str, Any] | None) -> str:
    """Current operating point, one field per line."""

    if not snapshot:
        return ""
    lines = [
        f"{label}: {_fmt(snapshot[key])}"
        for key, label in _LIVE_FIELDS.items()
        if key in snapshot and snapshot[key] is not None
    ]
    return _section("LIVE FURNACE CONDITION (1-hour average)", lines)


def summarise_recent_days(frame: Any, days: int = 3) -> str:
    """Mean, range and direction of travel for the fuel and stability drivers.

    DIRECTION MATTERS MORE THAN LEVEL. An operator already knows today's fuel
    rate. What they cannot see at a glance is whether it has been climbing for
    two days, so each line reports the change from the first quarter of the
    window to the last.
    """

    if frame is None or getattr(frame, "empty", True):
        return ""

    import pandas as pd

    lines: list[str] = []
    for key, label in _TREND_FIELDS.items():
        if key not in frame.columns:
            continue
        series = pd.to_numeric(frame[key], errors="coerce").dropna()
        if series.empty:
            continue
        quarter = max(1, len(series) // 4)
        drift = float(series.iloc[-quarter:].mean() - series.iloc[:quarter].mean())
        lines.append(
            f"{label}: mean {_fmt(series.mean())}, "
            f"range {_fmt(series.min())} to {_fmt(series.max())}, "
            f"trend {drift:+,.2f} over the window"
        )
    return _section(f"LAST {days} DAYS (hourly averages)", lines)


def describe_stock(ores: Sequence[Any] | None) -> str:
    """What is actually available to blend, and at what price."""

    if not ores:
        return ""
    lines = []
    for ore in ores:
        chemistry = getattr(ore, "chemistry", None)
        fe = getattr(chemistry, "fe_t_pct", None) if chemistry else None
        lines.append(
            f"{getattr(ore, 'display_name', getattr(ore, 'ore_id', '?'))}: "
            f"stock {_fmt(getattr(ore, 'stock_mt', None), ',.0f')} MT, "
            f"price Rs {_fmt(getattr(ore, 'price_rs_per_mt', None), ',.0f')}/MT, "
            f"Fe {_fmt(fe)}%, "
            f"share bounds {_fmt(getattr(ore, 'min_share_pct', None), ',.0f')}"
            f"-{_fmt(getattr(ore, 'max_share_pct', None), ',.0f')}%"
        )
    return _section("MATERIALS AVAILABLE", lines)


def describe_blend(blend: Any, label: str, ores: Sequence[Any] | None) -> str:
    """One solved blend: shares, cost, fuel, slag chemistry and feasibility."""

    if blend is None:
        return ""

    name_by_id = {
        getattr(o, "ore_id", ""): getattr(o, "display_name", "") for o in (ores or [])
    }
    shares = [
        f"{name_by_id.get(ore_id, ore_id)} {_fmt(pct, ',.1f')}%"
        for ore_id, pct in sorted(
            (getattr(blend, "shares_pct", {}) or {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        if float(pct or 0.0) > 0.0
    ]
    diagnostics = getattr(blend, "diagnostics", {}) or {}
    rates = diagnostics.get("fuel_rate_estimate") or {}
    flux_cost = float(diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0)
    total = float(
        diagnostics.get("adjusted_objective_rs_per_thm")
        if diagnostics.get("adjusted_objective_rs_per_thm") is not None
        else getattr(blend, "objective_rs_per_thm", 0.0)
    ) + flux_cost

    lines = [
        f"Blend: {', '.join(shares) if shares else 'n/a'}",
        f"Total cost: Rs {_fmt(total, ',.0f')}/THM "
        f"(ore {_fmt(getattr(blend, 'ore_cost_per_thm_rs', None), ',.0f')}, "
        f"fuel {_fmt(diagnostics.get('adjusted_fuel_cost_per_thm_rs'), ',.0f')}, "
        f"flux {_fmt(flux_cost, ',.0f')})",
        f"Coke rate: {_fmt(rates.get('coke_rate_kg_thm'), ',.1f')} kg/THM "
        f"(physics correction {_fmt(diagnostics.get('coke_correction_delta_kg_thm'), '+,.1f')})",
        f"Nut coke {_fmt(rates.get('nut_coke_rate_kg_thm'), ',.1f')}, "
        f"PCI {_fmt(rates.get('pci_rate_kg_thm'), ',.1f')}, "
        f"total fuel {_fmt(rates.get('total_fuel_rate_kg_thm'), ',.1f')} kg/THM",
        f"Slag: {_fmt(getattr(blend, 'slag_rate_kg_per_thm', None), ',.0f')} kg/THM, "
        f"B2 {_fmt(getattr(blend, 'slag_basicity', None), ',.3f')}, "
        f"T-basicity {_fmt(getattr(blend, 'slag_t_basicity', None), ',.3f')} "
        "(plant band 1.216-1.403), "
        f"Al2O3 {_fmt(getattr(blend, 'slag_al2o3_pct', None))}%, "
        f"MgO {_fmt(getattr(blend, 'slag_mgo_pct', None))}%",
    ]
    violations = list(getattr(blend, "violations", []) or [])
    lines.append(
        "Constraint violations: " + ("; ".join(violations) if violations else "none")
    )
    return _section(label, lines)


def describe_known_limitations(calibration: Any | None, anchor: Any | None) -> str:
    """The measured defects in the numbers above.

    Supplied so the model qualifies its commentary the way the documentation
    does. Without these it will read the coke rate as a measurement.
    """

    lines = [
        "The coke rate is an ENERGY BALANCE plus a fitted bias offset, not a "
        "measurement. Backtested over 239 days the raw balance runs about "
        "+20 kg/THM high (MAPE 7.24%, R2 0.07); with the offset it is "
        "MAPE 3.37%, R2 0.74. The offset's size measures how much the balance "
        "is still missing.",
        "Two known causes of that bias are unresolved: the top-gas analyser "
        "appears to under-read CO+CO2 by about 3 percentage points, and the "
        "shell heat-loss basis is undecided (worth about 11% of the coke rate).",
        "The operator's coke SETPOINT and the coke actually CHARGED are "
        "different series - about 4% apart, agreeing within 10 kg/THM on only "
        "42% of days. Do not treat them as interchangeable.",
        "The ML fuel-cost model is nearly blend-blind. ALL of the blend-to-fuel "
        "sensitivity comes from the physics coke correction, which is zero at "
        "current conditions by construction.",
        "The silicon prediction takes lagged silicon as an input, so its "
        "accuracy is partly yesterday's cast rather than burden chemistry.",
        "Operator behaviour over 188 days: 244 coke setpoint changes, cuts "
        "outnumbering raises about 1.5 to 1, roughly 80% reactive to something "
        "already moving. About one change in five could not be explained from "
        "the available tags at all.",
    ]
    if calibration is not None:
        age = calibration.age_days()
        lines.append(
            f"Current bias offset: {_fmt(getattr(calibration, 'offset_kg_per_thm', None), '+,.1f')} "
            f"kg/THM, fitted on {getattr(calibration, 'sample_days', 0)} days"
            + (f", {age} days ago" if age is not None else "")
            + f", day-to-day scatter +/-{_fmt(getattr(calibration, 'residual_sd_kg_per_thm', None), ',.0f')} kg/THM."
        )
        if getattr(calibration, "is_stale", lambda: False)():
            lines.append(
                "THE CALIBRATION IS STALE. The bias drifts about 3.3 kg/THM per "
                "month, so treat the coke level as unreliable and say so."
            )
    if anchor is not None and not getattr(anchor, "usable", False):
        lines.append(
            "The energy-balance anchor did NOT solve for this run, so the coke "
            "rate fell back to the observed plant rate. It will not respond to "
            "a change in controls. Reason: "
            + "; ".join(getattr(anchor, "notes", []) or ["unknown"])
        )
    return _section("KNOWN LIMITATIONS OF THE NUMBERS ABOVE", lines)


def build_commentary_context(
    *,
    live_snapshot: Mapping[str, Any] | None = None,
    recent_frame: Any = None,
    ores: Sequence[Any] | None = None,
    lp_blend: Any = None,
    de_blend: Any = None,
    manual_blend: Any = None,
    recommended_label: str = "LP baseline",
    calibration: Any = None,
    energy_anchor: Any = None,
    production_target_mt: float | None = None,
    recent_days: int = 3,
) -> CommentaryContext:
    """Assemble everything the model needs, and note what was unavailable.

    Args:
         - live_snapshot: Mapping | None - Current blast and top-gas tags.
         - recent_frame: DataFrame | None - Hourly online data for the window.
         - ores: Sequence[OreInput] | None - Materials available to blend.
         - lp_blend \\ de_blend \\ manual_blend: Solved blends, any may be None.
         - recommended_label: str - Which of the two the page is recommending.
         - calibration: CokeCalibration | None - The bias offset in force.
         - energy_anchor: EnergyAnchor | None - Whether physics set the level.
         - production_target_mt: float | None - Hot metal basis.
         - recent_days: int - Length of the trend window, for the heading.

    Returns:
         - return CommentaryContext - Prompt text plus a list of gaps.
    """

    missing: list[str] = []
    if not live_snapshot:
        missing.append("live furnace tags")
    if recent_frame is None or getattr(recent_frame, "empty", True):
        missing.append(f"last {recent_days} days of history")
    if not ores:
        missing.append("material stock")
    if lp_blend is None and de_blend is None:
        missing.append("a solved blend")

    parts = [
        f"PRODUCTION BASIS: {_fmt(production_target_mt, ',.0f')} MT hot metal.",
        f"THE PAGE IS RECOMMENDING: {recommended_label}.",
        describe_live_conditions(live_snapshot),
        summarise_recent_days(recent_frame, days=recent_days),
        describe_stock(ores),
        describe_blend(manual_blend, "CURRENT BLEND (what the plant is running)", ores),
        describe_blend(lp_blend, "RECOMMENDED BLEND - LP baseline", ores),
        describe_blend(de_blend, "RECOMMENDED BLEND - DE total cost", ores),
        describe_known_limitations(calibration, energy_anchor),
    ]
    if missing:
        parts.append(
            _section(
                "NOT AVAILABLE FOR THIS RUN",
                [
                    "The following could not be gathered. Do not speculate about "
                    "them; say they were unavailable if they matter:",
                    ", ".join(missing),
                ],
            )
        )
    return CommentaryContext(
        text="\n".join(part for part in parts if part).strip(), missing=missing
    )
