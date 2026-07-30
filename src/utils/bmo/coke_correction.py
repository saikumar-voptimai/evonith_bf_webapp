"""Physics-based coke-rate correction for BMO blend evaluation.

WHY THIS EXISTS
---------------
The deployed fuel unit-cost model (``unitcost_fuel_model.json``, 256 features) is
functionally blind to the burden. Measured by direct sweep: doubling
``ORE_CALC_THM`` (0.515 -> 1.03) moves the prediction 1.6 Rs/THM out of 13,364
(0.012%), an implied coke change of 0.06 kg/THM; flux x8 gives -5 Rs/THM;
``TOTAL_CLO_THM`` x1.5 gives +1.2 Rs/THM. ``hm_si_model.json`` is likewise
blend-flat (0.371 -> 0.384 %Si at 2x sinter).

The plant history cannot supply the missing sensitivity either. Across 6,219
hourly rows and 477 DPR days the slag coefficient bounces from -35 to +0.3 kg
fuel per 100 kg slag depending on specification (R2 <= 0.155, r ~ 0.05-0.15), Si
comes out at -7.2 per 0.1% where physics says +5, and blast temperature has the
wrong sign. That is a lagged control loop, not physics: cross-correlating fuel
rate against thermal state gives -0.45 contemporaneously (the operator cutting
fuel when the furnace runs hot) but flips positive at lags 4-9 h, peaking at
6-7 h - exactly burden descent time. The direction is recoverable from plant
data; the magnitude is not, because the controller cancels the very excursion
one would need to measure.

So every coefficient here is derived from first principles and the plant data is
used only as a plausibility bound and a sign check. This module supplies the
*entire* blend -> fuel sensitivity in BMO; the ML model is demoted to setting the
anchor level for current conditions, which is the one thing it does well.

FORM
----
``coke_rate_corrected = coke_rate_anchor + clamp(sum of dCoke_i)`` where
``dCoke_i = k_i * (x_i_blend - x_i_reference)``. The anchor is the existing
model-derived coke rate and the reference is the recent *observed* operating
point, so the correction is exactly zero when the blend reproduces current
conditions. That is the non-regression guarantee.

TERMS
-----
1. Slag heat, 30 kg coke / 100 kg slag. Slag enthalpy at ~1500 C is roughly
   cp 1.25 kJ/kg.K x 1475 K + 0.21 MJ/kg fusion ~ 1.8 MJ/kg; marginal coke heat
   delivered to the lower zone ~ 6.0 MJ/kg; 1.8/6.0 = 0.30 kg coke per kg slag.
2. Flux calcination, 30 kg coke / 100 kg CO2, driven by flux LOI and *not* by
   flux mass. CaCO3 -> CaO + CO2 costs 1.78 MJ/kg CaCO3, but the dominant term is
   Boudouard solution loss, C + CO2 -> 2CO, which consumes 12 kg C per 44 kg CO2
   = 0.273 kg C/kg CO2, or ~0.31 kg coke at 87% C in coke. NO DOUBLE COUNTING
   WITH TERM 1: the flux CaO/MgO residue enters slag mass and is already charged
   by term 1, so this term charges only the CO2 that leaves as gas.
3. Burden oxygen, 0.26 kg coke / kg O. Fe2O3 carries 48/112 = 0.4286 kg O per kg
   Fe against FeO's 16/56 = 0.2857. Direct reduction consumes 1 mol C per mol O
   = 0.75 kg C/kg O; at a ~30% direct-reduction fraction and 87% C in coke that
   is 0.30 x 0.75 / 0.87 = 0.26.
4. Hot-metal Si, 5 kg coke / 0.1% Si. SiO2 + 2C -> Si + 2CO at 690 kJ/mol Si =
   24.6 MJ/kg Si gives 4.1 kg coke of heat, plus 2 mol C per mol Si = 0.857 kg
   C/kg Si = 0.98 kg coke of carbon. 0.1% Si is 1.0 kg Si/THM.
5. HM temperature (2.5 kg / 10 C) and blast temperature (-10 kg / 100 C) are
   schema-only: they do not move with the blend, so with reference == current
   they contribute exactly 0.0. Shipped disabled.

The math layer holds no Streamlit imports and no pandas in the hot path -
``compute_coke_correction`` runs once per DE candidate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.bmo.calculations import (
    FE_FROM_FEO_FACTOR,
    compute_dry_fraction,
    compute_dry_weight_mt,
)
from utils.bmo.fuel_rates import ASSUMED_FUEL_PRICES_RS_PER_KG
from utils.bmo.types import BlendEvaluation, FluxInput, OreInput

# Oxygen bound per unit Fe in each oxide form. Fe2O3: 3 x 16 / (2 x 55.845)
# = 0.4297; FeO: 16 / 55.845 = 0.2865. Iron entering as FeO therefore arrives
# with a third less oxygen to strip than iron entering as Fe2O3.
O_PER_FE_IN_FE2O3 = (3.0 * 15.999) / (2.0 * 55.845)
O_PER_FE_IN_FEO = 15.999 / 55.845

# Term identifiers, in the order they are shown to the operator.
TERM_SLAG_HEAT = "slag_heat"
TERM_FLUX_CALCINATION = "flux_calcination"
TERM_BURDEN_OXYGEN = "burden_oxygen"
TERM_HOT_METAL_SI = "hot_metal_si"
TERM_HM_TEMPERATURE = "hm_temperature"
TERM_BLAST_TEMPERATURE = "blast_temperature"

TERM_ORDER = (
    TERM_SLAG_HEAT,
    TERM_FLUX_CALCINATION,
    TERM_BURDEN_OXYGEN,
    TERM_HOT_METAL_SI,
    TERM_HM_TEMPERATURE,
    TERM_BLAST_TEMPERATURE,
)

TERM_LABELS = {
    TERM_SLAG_HEAT: "Slag heat",
    TERM_FLUX_CALCINATION: "Flux calcination",
    TERM_BURDEN_OXYGEN: "Burden oxygen",
    TERM_HOT_METAL_SI: "Hot-metal Si",
    TERM_HM_TEMPERATURE: "HM temperature",
    TERM_BLAST_TEMPERATURE: "Blast temperature",
}

TERM_DRIVER_UNITS = {
    TERM_SLAG_HEAT: "kg/THM slag",
    TERM_FLUX_CALCINATION: "kg/THM flux CO2",
    TERM_BURDEN_OXYGEN: "kg/THM bound O",
    TERM_HOT_METAL_SI: "% Si",
    TERM_HM_TEMPERATURE: "degC",
    TERM_BLAST_TEMPERATURE: "degC",
}

# Each entry maps the operator-facing config key to the multiplier that converts
# it into the internal k (kg coke per THM per one natural unit of the driver).
# Keeping the operator's units in the yml and normalising here means the config
# stays readable without the math carrying awkward per-100 factors.
_TERM_CONFIG_UNITS: dict[str, tuple[str, float, str]] = {
    TERM_SLAG_HEAT: ("kg_coke_per_100kg_slag", 1.0 / 100.0, "kg coke / 100 kg slag"),
    TERM_FLUX_CALCINATION: (
        "kg_coke_per_100kg_co2",
        1.0 / 100.0,
        "kg coke / 100 kg CO2",
    ),
    TERM_BURDEN_OXYGEN: ("kg_coke_per_kg_oxygen", 1.0, "kg coke / kg O"),
    TERM_HOT_METAL_SI: ("kg_coke_per_0p1pct_si", 10.0, "kg coke / 0.1% Si"),
    TERM_HM_TEMPERATURE: ("kg_coke_per_10c", 1.0 / 10.0, "kg coke / 10 degC"),
    TERM_BLAST_TEMPERATURE: ("kg_coke_per_100c", 1.0 / 100.0, "kg coke / 100 degC"),
}

_TERM_ENVELOPE_KEYS = (
    "envelope_halfwidth_kg_per_thm",
    "envelope_halfwidth_pct",
    "envelope_halfwidth_c",
    "envelope_halfwidth",
)

_DEFAULT_MAX_ABS_CORRECTION_KG_THM = 60.0
_DEFAULT_TAPER_START_FRACTION = 0.6

# The per-term cap is a backstop, not a shaping curve: the driver-space taper is
# what softens extrapolation. Starting its compression at 0.6 of the cap (as the
# total does) would bend a perfectly ordinary 100 kg/THM slag deviation, so it
# stays linear until it is nearly at the cap and only rounds off the corner.
_TERM_CLAMP_START_FRACTION = 0.9
_DEFAULT_COKE_RATE_BAND = (280.0, 420.0)
_DEFAULT_TOTAL_FUEL_RATE_BAND = (480.0, 680.0)

# Beyond this modelled-vs-observed slag gap the correction inherits an offset
# large enough that the operator should be told rather than left to wonder.
SLAG_BASIS_GAP_WARN_KG_PER_THM = 40.0


@dataclass(frozen=True)
class CokeCorrectionTermSettings:
    """Configuration for one physical correction term.

    Args:
         - enabled: bool - Whether this term contributes to the correction.
         - k: float - Internal coefficient, kg coke/THM per natural driver unit.
         - k_config_value: float - Coefficient exactly as written in the yml.
         - k_config_units: str - Operator-facing units of ``k_config_value``.
         - max_abs_kg_thm: float - Per-term clamp on the resulting delta.
         - envelope_halfwidth: float | None - Driver-space taper half-width.
         - reference_source: str - How this term's reference value is resolved.
         - reference_fixed_value: float | None - Literal reference fallback.

    Returns:
         - return CokeCorrectionTermSettings - One term's settings.
    """

    enabled: bool = False
    k: float = 0.0
    k_config_value: float = 0.0
    k_config_units: str = ""
    max_abs_kg_thm: float = 0.0
    envelope_halfwidth: float | None = None
    reference_source: str = "model_current"
    reference_fixed_value: float | None = None

    @property
    def k_display(self) -> str:
        return f"{self.k_config_value:g} {self.k_config_units}"


@dataclass(frozen=True)
class CokeCorrectionSettings:
    """Top-level coke-correction configuration.

    Args:
         - enabled: bool - Whether the correction is computed at all.
         - apply_to_objective: bool - Whether it enters LP and DE cost.
         - apply_to_manual_blend: bool - Whether it changes the manual blend cost.
         - max_abs_correction_kg_thm: float - Total clamp on the summed delta.
         - taper_start_fraction: float - Fraction of the total clamp where the
           smooth taper begins.
         - coke_rate_band_kg_thm: tuple[float, float] - Corrected coke rate band.
         - total_fuel_rate_band_kg_thm: tuple[float, float] - Total fuel band.
         - terms: dict[str, CokeCorrectionTermSettings] - Per-term settings.

    Returns:
         - return CokeCorrectionSettings - Parsed correction configuration.
    """

    enabled: bool = False
    apply_to_objective: bool = False
    apply_to_manual_blend: bool = False
    max_abs_correction_kg_thm: float = _DEFAULT_MAX_ABS_CORRECTION_KG_THM
    taper_start_fraction: float = _DEFAULT_TAPER_START_FRACTION
    coke_rate_band_kg_thm: tuple[float, float] = _DEFAULT_COKE_RATE_BAND
    total_fuel_rate_band_kg_thm: tuple[float, float] = _DEFAULT_TOTAL_FUEL_RATE_BAND
    terms: dict[str, CokeCorrectionTermSettings] = field(default_factory=dict)

    def term(self, term_id: str) -> CokeCorrectionTermSettings:
        return self.terms.get(term_id, CokeCorrectionTermSettings())

    def is_term_active(self, term_id: str) -> bool:
        return self.enabled and self.term(term_id).enabled


@dataclass(frozen=True)
class CokeCorrectionReference:
    """The recent observed operating point the correction is anchored to.

    Every field is a trailing-window value, never a single time step: the
    fuel/thermal control loop oscillates with a ~14-16 h period, so a one-hour
    anchor risks anchoring to a transient peak.

    Args:
         - slag_rate_kg_per_thm: float | None - Observed slag rate.
         - flux_co2_kg_per_thm: float | None - CO2 from currently charged flux.
         - burden_oxygen_kg_per_thm: float | None - Bound O in the current burden.
         - hot_metal_si_pct: float | None - Si model output for current burden.
         - hm_temperature_c: float | None - Recent hot-metal temperature.
         - blast_temperature_c: float | None - Recent hot blast temperature.
         - sources: dict[str, str] - Per-field provenance strings.
         - warnings: list[str] - Operator-facing notes about the reference.
         - slag_basis_gap_kg_per_thm: float | None - Modelled minus observed slag.

    Returns:
         - return CokeCorrectionReference - Reference operating point.
    """

    slag_rate_kg_per_thm: float | None = None
    flux_co2_kg_per_thm: float | None = None
    burden_oxygen_kg_per_thm: float | None = None
    hot_metal_si_pct: float | None = None
    hm_temperature_c: float | None = None
    blast_temperature_c: float | None = None
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    slag_basis_gap_kg_per_thm: float | None = None

    def value_for(self, term_id: str) -> float | None:
        return {
            TERM_SLAG_HEAT: self.slag_rate_kg_per_thm,
            TERM_FLUX_CALCINATION: self.flux_co2_kg_per_thm,
            TERM_BURDEN_OXYGEN: self.burden_oxygen_kg_per_thm,
            TERM_HOT_METAL_SI: self.hot_metal_si_pct,
            TERM_HM_TEMPERATURE: self.hm_temperature_c,
            TERM_BLAST_TEMPERATURE: self.blast_temperature_c,
        }.get(term_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slag_rate_kg_per_thm": self.slag_rate_kg_per_thm,
            "flux_co2_kg_per_thm": self.flux_co2_kg_per_thm,
            "burden_oxygen_kg_per_thm": self.burden_oxygen_kg_per_thm,
            "hot_metal_si_pct": self.hot_metal_si_pct,
            "hm_temperature_c": self.hm_temperature_c,
            "blast_temperature_c": self.blast_temperature_c,
            "sources": dict(self.sources),
            "warnings": list(self.warnings),
            "slag_basis_gap_kg_per_thm": self.slag_basis_gap_kg_per_thm,
        }


@dataclass(frozen=True)
class CokeCorrectionDrivers:
    """Blend-side driver values, on the same basis as the reference.

    Args:
         - slag_rate_kg_per_thm: float | None - Blend slag rate.
         - flux_co2_kg_per_thm: float | None - CO2 from the blend's flux.
         - burden_oxygen_kg_per_thm: float | None - Bound O in the blend.
         - hot_metal_si_pct: float | None - Predicted Si for the blend.
         - hm_temperature_c: float | None - Planned hot-metal temperature.
         - blast_temperature_c: float | None - Planned blast temperature.

    Returns:
         - return CokeCorrectionDrivers - Blend-side driver values.
    """

    slag_rate_kg_per_thm: float | None = None
    flux_co2_kg_per_thm: float | None = None
    burden_oxygen_kg_per_thm: float | None = None
    hot_metal_si_pct: float | None = None
    hm_temperature_c: float | None = None
    blast_temperature_c: float | None = None

    def value_for(self, term_id: str) -> float | None:
        return {
            TERM_SLAG_HEAT: self.slag_rate_kg_per_thm,
            TERM_FLUX_CALCINATION: self.flux_co2_kg_per_thm,
            TERM_BURDEN_OXYGEN: self.burden_oxygen_kg_per_thm,
            TERM_HOT_METAL_SI: self.hot_metal_si_pct,
            TERM_HM_TEMPERATURE: self.hm_temperature_c,
            TERM_BLAST_TEMPERATURE: self.blast_temperature_c,
        }.get(term_id)


@dataclass(frozen=True)
class CokeCorrectionTermResult:
    """One term's contribution, with everything needed to explain it.

    Args:
         - term_id: str - Stable term identifier.
         - label: str - Human-readable term name.
         - enabled: bool - Whether the term contributed.
         - disabled_reason: str | None - Why it did not, when it did not.
         - k: float - Internal coefficient used.
         - k_display: str - Operator-facing coefficient with units.
         - x_blend: float | None - Blend-side driver value.
         - x_reference: float | None - Reference driver value.
         - x_units: str - Driver units.
         - reference_source: str - Provenance of the reference value.
         - delta_raw_kg_thm: float - Delta before taper and clamp.
         - delta_kg_thm: float - Delta after taper and per-term clamp.
         - term_clamp_binding: bool - Whether the per-term clamp bound.
         - envelope_exceeded: bool - Whether the driver left its envelope.

    Returns:
         - return CokeCorrectionTermResult - One term's computed contribution.
    """

    term_id: str
    label: str
    enabled: bool
    disabled_reason: str | None
    k: float
    k_display: str
    x_blend: float | None
    x_reference: float | None
    x_units: str
    reference_source: str
    delta_raw_kg_thm: float
    delta_kg_thm: float
    term_clamp_binding: bool
    envelope_exceeded: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "label": self.label,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "k": self.k,
            "k_display": self.k_display,
            "x_blend": self.x_blend,
            "x_reference": self.x_reference,
            "x_units": self.x_units,
            "reference_source": self.reference_source,
            "delta_raw_kg_thm": self.delta_raw_kg_thm,
            "delta_kg_thm": self.delta_kg_thm,
            "term_clamp_binding": self.term_clamp_binding,
            "envelope_exceeded": self.envelope_exceeded,
        }


@dataclass(frozen=True)
class CokeCorrectionResult:
    """The complete correction outcome for one blend.

    Args:
         - enabled: bool - Whether the correction was computed.
         - applied_to_objective: bool - Whether it changed blend cost.
         - anchor_coke_rate_kg_thm: float - Model-derived coke rate before correction.
         - terms: list[CokeCorrectionTermResult] - Per-term breakdown.
         - sum_raw_kg_thm: float - Sum of untapered, unclamped term deltas.
         - sum_after_term_clamps_kg_thm: float - Sum after per-term treatment.
         - applied_delta_kg_thm: float - Delta actually applied to coke rate.
         - total_clamp_binding: bool - Whether the total clamp bound.
         - taper_active: bool - Whether any smooth taper engaged.
         - corrected_coke_rate_kg_thm: float - Coke rate after correction.
         - corrected_total_coke_rate_kg_thm: float - Coke plus nut coke.
         - corrected_total_fuel_rate_kg_thm: float - Coke plus nut coke plus PCI.
         - band_bindings: list[str] - Output bands that bound.
         - warnings: list[str] - Operator-facing warnings.
         - reference: CokeCorrectionReference - Reference point used.

    Returns:
         - return CokeCorrectionResult - Correction result and full audit trail.
    """

    enabled: bool
    applied_to_objective: bool
    anchor_coke_rate_kg_thm: float
    terms: list[CokeCorrectionTermResult]
    sum_raw_kg_thm: float
    sum_after_term_clamps_kg_thm: float
    applied_delta_kg_thm: float
    total_clamp_binding: bool
    taper_active: bool
    corrected_coke_rate_kg_thm: float
    corrected_total_coke_rate_kg_thm: float
    corrected_total_fuel_rate_kg_thm: float
    band_bindings: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reference: CokeCorrectionReference = field(
        default_factory=CokeCorrectionReference
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "applied_to_objective": self.applied_to_objective,
            "anchor_coke_rate_kg_thm": self.anchor_coke_rate_kg_thm,
            "terms": [term.to_dict() for term in self.terms],
            "sum_raw_kg_thm": self.sum_raw_kg_thm,
            "sum_after_term_clamps_kg_thm": self.sum_after_term_clamps_kg_thm,
            "applied_delta_kg_thm": self.applied_delta_kg_thm,
            "total_clamp_binding": self.total_clamp_binding,
            "taper_active": self.taper_active,
            "corrected_coke_rate_kg_thm": self.corrected_coke_rate_kg_thm,
            "corrected_total_coke_rate_kg_thm": self.corrected_total_coke_rate_kg_thm,
            "corrected_total_fuel_rate_kg_thm": self.corrected_total_fuel_rate_kg_thm,
            "band_bindings": list(self.band_bindings),
            "warnings": list(self.warnings),
            "reference": self.reference.to_dict(),
        }


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def load_coke_correction_settings(
    bmo_cfg: Mapping[str, Any] | None,
) -> CokeCorrectionSettings:
    """
    Parse ``bmo.coke_rate_correction`` into typed settings.

    A missing or malformed block returns fully-disabled settings so the whole
    feature is a strict no-op. That keeps every existing caller and test valid
    without them having to know the block exists.

    Args:
         - bmo_cfg: Mapping[str, Any] | None - The ``bmo`` config mapping.

    Returns:
         - return CokeCorrectionSettings - Parsed settings, disabled when absent.
    """

    block = (bmo_cfg or {}).get("coke_rate_correction")
    if not isinstance(block, Mapping):
        return CokeCorrectionSettings()

    guardrails = block.get("guardrails")
    guardrails = guardrails if isinstance(guardrails, Mapping) else {}

    raw_terms = block.get("terms")
    raw_terms = raw_terms if isinstance(raw_terms, Mapping) else {}

    terms: dict[str, CokeCorrectionTermSettings] = {}
    for term_id in TERM_ORDER:
        raw = raw_terms.get(term_id)
        terms[term_id] = _parse_term(term_id, raw if isinstance(raw, Mapping) else {})

    return CokeCorrectionSettings(
        enabled=bool(block.get("enabled", False)),
        apply_to_objective=bool(block.get("apply_to_objective", False)),
        apply_to_manual_blend=bool(block.get("apply_to_manual_blend", False)),
        max_abs_correction_kg_thm=_positive_float(
            guardrails.get("max_abs_correction_kg_thm"),
            _DEFAULT_MAX_ABS_CORRECTION_KG_THM,
        ),
        taper_start_fraction=min(
            0.99,
            max(
                0.0,
                _float_or(
                    guardrails.get("taper_start_fraction"),
                    _DEFAULT_TAPER_START_FRACTION,
                ),
            ),
        ),
        coke_rate_band_kg_thm=_parse_band(
            guardrails.get("coke_rate_band_kg_thm"), _DEFAULT_COKE_RATE_BAND
        ),
        total_fuel_rate_band_kg_thm=_parse_band(
            guardrails.get("total_fuel_rate_band_kg_thm"),
            _DEFAULT_TOTAL_FUEL_RATE_BAND,
        ),
        terms=terms,
    )


def _parse_term(
    term_id: str, raw: Mapping[str, Any]
) -> CokeCorrectionTermSettings:
    config_key, multiplier, units = _TERM_CONFIG_UNITS[term_id]
    config_value = _float_or(raw.get(config_key), 0.0)

    envelope: float | None = None
    for key in _TERM_ENVELOPE_KEYS:
        if key in raw:
            candidate = _float_or(raw.get(key), 0.0)
            if candidate > 0.0:
                envelope = candidate
            break

    fixed_reference: float | None = None
    for key in ("reference_fixed_kg_per_thm", "reference_fixed_value"):
        if raw.get(key) is not None:
            fixed_reference = _float_or(raw.get(key), 0.0)
            break

    return CokeCorrectionTermSettings(
        enabled=bool(raw.get("enabled", False)),
        k=config_value * multiplier,
        k_config_value=config_value,
        k_config_units=units,
        max_abs_kg_thm=_positive_float(raw.get("max_abs_kg_thm"), math.inf),
        envelope_halfwidth=envelope,
        reference_source=str(raw.get("reference_source", "model_current")),
        reference_fixed_value=fixed_reference,
    )


def _float_or(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _positive_float(value: Any, default: float) -> float:
    parsed = _float_or(value, default)
    return parsed if parsed > 0.0 else default


def _parse_band(
    value: Any, default: tuple[float, float]
) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if len(items) == 2:
            low = _float_or(items[0], default[0])
            high = _float_or(items[1], default[1])
            if high > low:
                return (low, high)
    return default


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def compute_flux_co2_kg_per_thm(
    *,
    flux_inputs: Sequence[FluxInput] | None,
    hot_metal_mt: float,
) -> float:
    """
    Calculate CO2 released by carbonate flux, per tonne of hot metal.

    LOI on a flux row is the carbonate that leaves as CO2 during calcination.
    This is the driver for the calcination term rather than total flux mass,
    because the CaO/MgO residue stays in the slag and is already charged by the
    slag-heat term.

    Args:
         - flux_inputs: Sequence[FluxInput] | None - Flux rows in the blend.
         - hot_metal_mt: float - Hot-metal basis in MT.

    Returns:
         - return float - Flux CO2 in kg per THM.
    """

    if hot_metal_mt is None or hot_metal_mt <= 0.0:
        return 0.0

    co2_mt = 0.0
    for flux in flux_inputs or []:
        if not bool(getattr(flux, "enabled", True)):
            continue
        dry_mt = compute_dry_weight_mt(
            getattr(flux, "wet_qty_mt", 0.0), getattr(flux, "moisture_pct", 0.0)
        )
        loi_pct = max(0.0, _float_or(getattr(flux, "loi_pct", 0.0), 0.0))
        co2_mt += dry_mt * (loi_pct / 100.0)

    return co2_mt * 1000.0 / float(hot_metal_mt)


def compute_burden_oxygen_kg_per_thm(
    *,
    ores: Sequence[OreInput],
    quantities_mt: Mapping[str, float],
    hot_metal_mt: float,
) -> float:
    """
    Calculate oxygen bound to iron in the burden, per tonne of hot metal.

    Iron arriving as Fe2O3 carries about half again as much oxygen as iron
    arriving as FeO, and stripping that oxygen costs carbon. Splitting each
    ore's total Fe into its FeO-borne and Fe2O3-borne parts is what lets the
    correction see a sinter-to-pellet swing at all.

    Args:
         - ores: Sequence[OreInput] - Ores in the blend.
         - quantities_mt: Mapping[str, float] - Wet quantities keyed by ore id.
         - hot_metal_mt: float - Hot-metal basis in MT.

    Returns:
         - return float - Bound oxygen in kg per THM.
    """

    if hot_metal_mt is None or hot_metal_mt <= 0.0:
        return 0.0

    oxygen_mt = 0.0
    for ore in ores:
        qty = _float_or(quantities_mt.get(ore.ore_id, 0.0), 0.0)
        if qty <= 0.0:
            continue
        chemistry = ore.chemistry
        dry_mt = compute_dry_weight_mt(qty, chemistry.moisture_pct)
        if dry_mt <= 0.0:
            continue

        fe_total_pct = max(0.0, _float_or(chemistry.fe_t_pct, 0.0))
        feo_pct = max(0.0, _float_or(chemistry.feo_pct, 0.0))
        # Fe locked in FeO, capped by the ore's total Fe so a stale FeO reading
        # cannot manufacture negative Fe2O3-borne iron.
        fe_in_feo_pct = min(fe_total_pct, feo_pct * FE_FROM_FEO_FACTOR)
        fe_in_fe2o3_pct = max(0.0, fe_total_pct - fe_in_feo_pct)

        oxygen_pct = (
            fe_in_fe2o3_pct * O_PER_FE_IN_FE2O3 + fe_in_feo_pct * O_PER_FE_IN_FEO
        )
        oxygen_mt += dry_mt * (oxygen_pct / 100.0)

    return oxygen_mt * 1000.0 / float(hot_metal_mt)


def build_drivers(
    *,
    blend: BlendEvaluation,
    ores: Sequence[OreInput],
    quantities_mt: Mapping[str, float],
    flux_inputs: Sequence[FluxInput] | None = None,
    hot_metal_si_pct: float | None = None,
    process_context: Mapping[str, Any] | None = None,
) -> CokeCorrectionDrivers:
    """
    Assemble blend-side driver values for the correction.

    Slag rate is read straight off the evaluated blend so the driver sits on the
    same basis the page displays, rather than being recomputed here and quietly
    disagreeing with the Slag Rate tile.

    Args:
         - blend: BlendEvaluation - The evaluated blend.
         - ores: Sequence[OreInput] - Ores in the blend.
         - quantities_mt: Mapping[str, float] - Wet quantities keyed by ore id.
         - flux_inputs: Sequence[FluxInput] | None - Flux rows in the blend.
         - hot_metal_si_pct: float | None - Predicted Si for this blend.
         - process_context: Mapping[str, Any] | None - Recent process values.

    Returns:
         - return CokeCorrectionDrivers - Blend-side driver values.
    """

    hot_metal_mt = _float_or(
        blend.diagnostics.get("hot_metal_target_mt"), 0.0
    )

    return CokeCorrectionDrivers(
        slag_rate_kg_per_thm=_float_or(blend.slag_rate_kg_per_thm, 0.0),
        flux_co2_kg_per_thm=compute_flux_co2_kg_per_thm(
            flux_inputs=flux_inputs, hot_metal_mt=hot_metal_mt
        ),
        burden_oxygen_kg_per_thm=compute_burden_oxygen_kg_per_thm(
            ores=ores, quantities_mt=quantities_mt, hot_metal_mt=hot_metal_mt
        ),
        hot_metal_si_pct=hot_metal_si_pct,
        # The blend does not move furnace setpoints, so these mirror the
        # reference and contribute exactly zero until a what-if input exists.
        hm_temperature_c=_context_float(process_context, _HM_TEMPERATURE_KEYS),
        blast_temperature_c=_context_float(
            process_context, _BLAST_TEMPERATURE_KEYS
        ),
    )


_HM_TEMPERATURE_KEYS = (
    "HMT_GT_1480C",
    "hm_temperature_c",
    "hmt_gt_1480c",
)

_BLAST_TEMPERATURE_KEYS = (
    "HOT BLAST TEMP.OC",
    "hot_blast_temp",
    "blast_temperature_c",
)


def _context_float(
    context: Mapping[str, Any] | None, keys: Sequence[str]
) -> float | None:
    for key in keys:
        if context and key in context:
            value = _float_or(context.get(key), math.nan)
            if math.isfinite(value):
                return value
    return None


def build_reference(
    *,
    settings: CokeCorrectionSettings,
    observed_slag_rate_kg_per_thm: float | None = None,
    current_drivers: CokeCorrectionDrivers | None = None,
    process_context: Mapping[str, Any] | None = None,
) -> CokeCorrectionReference:
    """
    Resolve the recent observed operating point the correction anchors to.

    Slag defaults to the observed DPR rate because that is plant truth and the
    page already shows modelled-versus-observed side by side. Flux CO2, burden
    oxygen, and Si use the current burden evaluated through the same helpers as
    the blend side, so those terms are exactly zero at the reference with no
    basis question at all. A term whose reference cannot be resolved is disabled
    with a reason rather than silently falling back to a training mean.

    Args:
         - settings: CokeCorrectionSettings - Parsed correction settings.
         - observed_slag_rate_kg_per_thm: float | None - Observed DPR slag rate.
         - current_drivers: CokeCorrectionDrivers | None - Drivers evaluated on
           the current burden, used by ``model_current`` reference sources.
         - process_context: Mapping[str, Any] | None - Recent process values.

    Returns:
         - return CokeCorrectionReference - Resolved reference point.
    """

    sources: dict[str, str] = {}
    warnings: list[str] = []

    modelled_slag = (
        current_drivers.slag_rate_kg_per_thm if current_drivers else None
    )
    observed_slag = (
        observed_slag_rate_kg_per_thm
        if observed_slag_rate_kg_per_thm is not None
        and math.isfinite(float(observed_slag_rate_kg_per_thm))
        and float(observed_slag_rate_kg_per_thm) > 0.0
        else None
    )

    slag_settings = settings.term(TERM_SLAG_HEAT)
    slag_reference: float | None = None
    if slag_settings.reference_source == "model_current":
        slag_reference = modelled_slag
        if slag_reference is not None:
            sources[TERM_SLAG_HEAT] = "model_current.blend_slag_rate"
    else:
        if observed_slag is not None:
            slag_reference = float(observed_slag)
            sources[TERM_SLAG_HEAT] = "observed_dpr.slag_generation_mt/hot_metal"
        elif modelled_slag is not None:
            slag_reference = modelled_slag
            sources[TERM_SLAG_HEAT] = "model_current.blend_slag_rate (observed unavailable)"
            warnings.append(
                "Observed DPR slag rate was unavailable, so the slag term is "
                "anchored to the modelled slag rate of the current burden."
            )
        elif slag_settings.reference_fixed_value is not None:
            slag_reference = float(slag_settings.reference_fixed_value)
            sources[TERM_SLAG_HEAT] = "config.reference_fixed_kg_per_thm"
            warnings.append(
                "Neither observed nor modelled slag rate was available; the slag "
                f"term is anchored to the configured fallback of "
                f"{slag_reference:,.0f} kg/THM."
            )

    slag_basis_gap: float | None = None
    if modelled_slag is not None and observed_slag is not None:
        slag_basis_gap = float(modelled_slag) - float(observed_slag)
        if abs(slag_basis_gap) > SLAG_BASIS_GAP_WARN_KG_PER_THM:
            warnings.append(
                f"Modelled slag rate for the current burden is "
                f"{slag_basis_gap:+,.0f} kg/THM against the observed DPR rate, and "
                "the coke correction inherits that offset. Calibrate "
                "bmo.slag_balance.slag_correction_factor, or set the slag term's "
                "reference_source to model_current."
            )

    flux_reference = current_drivers.flux_co2_kg_per_thm if current_drivers else None
    if flux_reference is not None:
        sources[TERM_FLUX_CALCINATION] = "model_current.charged_flux_loi"

    oxygen_reference = (
        current_drivers.burden_oxygen_kg_per_thm if current_drivers else None
    )
    if oxygen_reference is not None:
        sources[TERM_BURDEN_OXYGEN] = "model_current.burden_bound_oxygen"

    si_reference = current_drivers.hot_metal_si_pct if current_drivers else None
    if si_reference is not None:
        # Deliberately the Si model's own prediction for the current burden, not
        # the measured cast Si: using measured Si would turn the model's constant
        # offset into a blend-independent bias, which is pure noise.
        sources[TERM_HOT_METAL_SI] = "model_current.si_model_prediction"

    hm_temperature = _context_float(process_context, _HM_TEMPERATURE_KEYS)
    if hm_temperature is not None:
        sources[TERM_HM_TEMPERATURE] = "process_context.hot_metal_temperature"
    blast_temperature = _context_float(process_context, _BLAST_TEMPERATURE_KEYS)
    if blast_temperature is not None:
        sources[TERM_BLAST_TEMPERATURE] = "process_context.hot_blast_temperature"

    return CokeCorrectionReference(
        slag_rate_kg_per_thm=slag_reference,
        flux_co2_kg_per_thm=flux_reference,
        burden_oxygen_kg_per_thm=oxygen_reference,
        hot_metal_si_pct=si_reference,
        hm_temperature_c=hm_temperature,
        blast_temperature_c=blast_temperature,
        sources=sources,
        warnings=warnings,
        slag_basis_gap_kg_per_thm=slag_basis_gap,
    )


# --------------------------------------------------------------------------
# Taper, clamp, and the correction itself
# --------------------------------------------------------------------------


def soft_saturate(value: float, start: float, limit: float) -> float:
    """
    Compress ``value`` smoothly toward ``limit`` once it passes ``start``.

    Below ``start`` this is the identity; above it the excess is passed through
    ``tanh`` so the curve is continuous, C1, monotone, and asymptotically bounded
    by ``limit``. A hard clip would be simpler, but the resulting kink makes
    differential evolution behave badly right where the correction matters most.

    Args:
         - value: float - Raw value.
         - start: float - Magnitude where compression begins.
         - limit: float - Asymptotic bound on the magnitude.

    Returns:
         - return float - Saturated value with the sign preserved.
    """

    if not math.isfinite(value):
        return 0.0
    if not math.isfinite(limit):
        # No effective bound configured, so nothing to saturate toward.
        return float(value)
    magnitude = abs(value)
    if not math.isfinite(start) or magnitude <= start or limit <= start:
        return float(value) if magnitude <= limit else math.copysign(limit, value)

    span = limit - start
    saturated = start + span * math.tanh((magnitude - start) / span)
    return math.copysign(saturated, value)


def _taper_driver_delta(
    delta: float, halfwidth: float | None
) -> tuple[float, bool]:
    """Compress a driver deviation that runs past its historical envelope."""

    if halfwidth is None or halfwidth <= 0.0 or not math.isfinite(delta):
        return float(delta), False
    start = 1.5 * halfwidth
    limit = 3.0 * halfwidth
    if abs(delta) <= start:
        return float(delta), False
    return soft_saturate(delta, start, limit), True


def compute_coke_correction(
    *,
    anchor_coke_rate_kg_thm: float,
    anchor_nut_coke_rate_kg_thm: float,
    anchor_pci_rate_kg_thm: float,
    drivers: CokeCorrectionDrivers,
    reference: CokeCorrectionReference,
    settings: CokeCorrectionSettings,
) -> CokeCorrectionResult:
    """
    Compute the physics coke-rate correction for one blend.

    Returns a fully-populated, zero-delta result when the correction is disabled
    or the anchor is unusable, so callers never have to branch on ``None``.

    Args:
         - anchor_coke_rate_kg_thm: float - Model-derived coke rate.
         - anchor_nut_coke_rate_kg_thm: float - Nut coke rate, carried through.
         - anchor_pci_rate_kg_thm: float - PCI rate, carried through.
         - drivers: CokeCorrectionDrivers - Blend-side driver values.
         - reference: CokeCorrectionReference - Recent observed operating point.
         - settings: CokeCorrectionSettings - Parsed correction settings.

    Returns:
         - return CokeCorrectionResult - Correction result and audit trail.
    """

    anchor = _float_or(anchor_coke_rate_kg_thm, math.nan)
    nut = max(0.0, _float_or(anchor_nut_coke_rate_kg_thm, 0.0))
    pci = max(0.0, _float_or(anchor_pci_rate_kg_thm, 0.0))

    if not settings.enabled or not math.isfinite(anchor):
        return _inactive_result(
            anchor=anchor if math.isfinite(anchor) else 0.0,
            nut=nut,
            pci=pci,
            settings=settings,
            reference=reference,
            reason=(
                "coke-rate correction disabled in configuration"
                if not settings.enabled
                else "anchor coke rate unavailable"
            ),
        )

    term_results: list[CokeCorrectionTermResult] = []
    sum_raw = 0.0
    sum_clamped = 0.0
    taper_active = False
    warnings: list[str] = list(reference.warnings)

    for term_id in TERM_ORDER:
        term_settings = settings.term(term_id)
        x_blend = drivers.value_for(term_id)
        x_reference = reference.value_for(term_id)

        disabled_reason: str | None = None
        if not term_settings.enabled:
            disabled_reason = "term disabled in configuration"
        elif x_blend is None or not math.isfinite(float(x_blend)):
            disabled_reason = "blend driver value unavailable"
        elif x_reference is None or not math.isfinite(float(x_reference)):
            disabled_reason = "no current-burden reference available"

        if disabled_reason is not None:
            term_results.append(
                CokeCorrectionTermResult(
                    term_id=term_id,
                    label=TERM_LABELS[term_id],
                    enabled=False,
                    disabled_reason=disabled_reason,
                    k=term_settings.k,
                    k_display=term_settings.k_display,
                    x_blend=None if x_blend is None else float(x_blend),
                    x_reference=(
                        None if x_reference is None else float(x_reference)
                    ),
                    x_units=TERM_DRIVER_UNITS[term_id],
                    reference_source=reference.sources.get(term_id, ""),
                    delta_raw_kg_thm=0.0,
                    delta_kg_thm=0.0,
                    term_clamp_binding=False,
                    envelope_exceeded=False,
                )
            )
            continue

        driver_delta = float(x_blend) - float(x_reference)
        delta_raw = term_settings.k * driver_delta

        tapered_driver_delta, envelope_exceeded = _taper_driver_delta(
            driver_delta, term_settings.envelope_halfwidth
        )
        taper_active = taper_active or envelope_exceeded

        delta = term_settings.k * tapered_driver_delta
        clamped = soft_saturate(
            delta,
            term_settings.max_abs_kg_thm * _TERM_CLAMP_START_FRACTION,
            term_settings.max_abs_kg_thm,
        )
        term_clamp_binding = abs(clamped) < abs(delta) - 1e-9

        sum_raw += delta_raw
        sum_clamped += clamped

        term_results.append(
            CokeCorrectionTermResult(
                term_id=term_id,
                label=TERM_LABELS[term_id],
                enabled=True,
                disabled_reason=None,
                k=term_settings.k,
                k_display=term_settings.k_display,
                x_blend=float(x_blend),
                x_reference=float(x_reference),
                x_units=TERM_DRIVER_UNITS[term_id],
                reference_source=reference.sources.get(term_id, ""),
                delta_raw_kg_thm=float(delta_raw),
                delta_kg_thm=float(clamped),
                term_clamp_binding=bool(term_clamp_binding),
                envelope_exceeded=bool(envelope_exceeded),
            )
        )

    applied_delta = soft_saturate(
        sum_clamped,
        settings.max_abs_correction_kg_thm * settings.taper_start_fraction,
        settings.max_abs_correction_kg_thm,
    )
    total_clamp_binding = abs(applied_delta) < abs(sum_clamped) - 1e-9
    if total_clamp_binding:
        warnings.append(
            f"The physics terms summed to {sum_clamped:+,.1f} kg/THM of coke but "
            f"the total correction is capped at "
            f"{settings.max_abs_correction_kg_thm:,.0f} kg/THM, so "
            f"{applied_delta:+,.1f} kg/THM was applied."
        )

    corrected_coke = anchor + applied_delta
    band_bindings: list[str] = []

    low, high = settings.coke_rate_band_kg_thm
    if corrected_coke < low or corrected_coke > high:
        wanted = corrected_coke
        corrected_coke = min(max(corrected_coke, low), high)
        band_bindings.append("coke_rate")
        warnings.append(
            f"Corrected coke rate hit the {corrected_coke:,.0f} kg/THM band edge; "
            f"the physics delta wanted {wanted:,.0f} kg/THM."
        )
        applied_delta = corrected_coke - anchor

    corrected_total_fuel = corrected_coke + nut + pci
    fuel_low, fuel_high = settings.total_fuel_rate_band_kg_thm
    if corrected_total_fuel < fuel_low or corrected_total_fuel > fuel_high:
        wanted_fuel = corrected_total_fuel
        corrected_total_fuel = min(max(corrected_total_fuel, fuel_low), fuel_high)
        band_bindings.append("total_fuel_rate")
        warnings.append(
            f"Corrected total fuel rate hit the {corrected_total_fuel:,.0f} kg/THM "
            f"band edge; the physics delta wanted {wanted_fuel:,.0f} kg/THM."
        )
        corrected_coke = max(0.0, corrected_total_fuel - nut - pci)
        applied_delta = corrected_coke - anchor

    return CokeCorrectionResult(
        enabled=True,
        applied_to_objective=bool(settings.apply_to_objective),
        anchor_coke_rate_kg_thm=float(anchor),
        terms=term_results,
        sum_raw_kg_thm=float(sum_raw),
        sum_after_term_clamps_kg_thm=float(sum_clamped),
        applied_delta_kg_thm=float(applied_delta),
        total_clamp_binding=bool(total_clamp_binding),
        taper_active=bool(taper_active),
        corrected_coke_rate_kg_thm=float(corrected_coke),
        corrected_total_coke_rate_kg_thm=float(corrected_coke + nut),
        corrected_total_fuel_rate_kg_thm=float(corrected_coke + nut + pci),
        band_bindings=band_bindings,
        warnings=warnings,
        reference=reference,
    )


def _inactive_result(
    *,
    anchor: float,
    nut: float,
    pci: float,
    settings: CokeCorrectionSettings,
    reference: CokeCorrectionReference,
    reason: str,
) -> CokeCorrectionResult:
    return CokeCorrectionResult(
        enabled=False,
        applied_to_objective=False,
        anchor_coke_rate_kg_thm=float(anchor),
        terms=[
            CokeCorrectionTermResult(
                term_id=term_id,
                label=TERM_LABELS[term_id],
                enabled=False,
                disabled_reason=reason,
                k=settings.term(term_id).k,
                k_display=settings.term(term_id).k_display,
                x_blend=None,
                x_reference=None,
                x_units=TERM_DRIVER_UNITS[term_id],
                reference_source="",
                delta_raw_kg_thm=0.0,
                delta_kg_thm=0.0,
                term_clamp_binding=False,
                envelope_exceeded=False,
            )
            for term_id in TERM_ORDER
        ],
        sum_raw_kg_thm=0.0,
        sum_after_term_clamps_kg_thm=0.0,
        applied_delta_kg_thm=0.0,
        total_clamp_binding=False,
        taper_active=False,
        corrected_coke_rate_kg_thm=float(anchor),
        corrected_total_coke_rate_kg_thm=float(anchor + nut),
        corrected_total_fuel_rate_kg_thm=float(anchor + nut + pci),
        band_bindings=[],
        warnings=[],
        reference=reference,
    )


# --------------------------------------------------------------------------
# LP linearisation
# --------------------------------------------------------------------------


def build_linear_coke_correction_cost_coeffs(
    *,
    ores: Sequence[OreInput],
    variable_fluxes: Sequence[FluxInput] | None,
    settings: CokeCorrectionSettings,
    slag_coeff: np.ndarray,
    hot_metal_target_mt: float,
    coke_price_rs_per_kg: float = ASSUMED_FUEL_PRICES_RS_PER_KG["coke"],
) -> np.ndarray:
    """
    Build the LP objective addition that prices the physics correction.

    The LP minimises total Rs while the correction is defined per THM, so each
    term's per-THM gradient is multiplied back by the hot-metal basis. Every
    driver here is itself a per-THM quantity, so that basis cancels exactly in
    all three terms and each coefficient depends only on what one wet MT of the
    column contributes. ``hot_metal_target_mt`` therefore only guards validity.

    ``slag_coeff`` is *consumed*, not recomputed: it comes from
    ``lp_solver._build_linear_slag_and_basicity_terms`` so the LP's slag
    constraint and the LP's slag pricing can never drift apart.

    The baseline coke price is used rather than the operator's current price
    because the nonlinear path adds the same delta at the baseline price; using
    a different price here would make LP and DE optimise different objectives.

    Constant offsets are dropped throughout since they do not change ``argmin``.

    Args:
         - ores: Sequence[OreInput] - Ores, in LP column order.
         - variable_fluxes: Sequence[FluxInput] | None - Optimisable flux columns.
         - settings: CokeCorrectionSettings - Parsed correction settings.
         - slag_coeff: np.ndarray - Marginal slag MT per wet MT, per column.
         - hot_metal_target_mt: float - Hot-metal basis in MT.
         - coke_price_rs_per_kg: float - Coke price used to convert kg to Rs.

    Returns:
         - return np.ndarray - Cost addition per decision-variable column.
    """

    fluxes = list(variable_fluxes or [])
    n_columns = len(ores) + len(fluxes)
    coeffs = np.zeros(n_columns, dtype=float)

    if not settings.enabled or not settings.apply_to_objective or n_columns == 0:
        return coeffs
    if hot_metal_target_mt is None or hot_metal_target_mt <= 0.0:
        return coeffs

    price = float(coke_price_rs_per_kg)

    # Slag heat. d(slag rate)/dx_i = 1000 * slag_coeff_i / HM, and multiplying
    # the per-THM cost back by HM cancels the basis: 1000 * k * price * coeff.
    slag_settings = settings.term(TERM_SLAG_HEAT)
    if slag_settings.enabled and slag_coeff is not None:
        coeff_array = np.asarray(slag_coeff, dtype=float)
        if coeff_array.shape[0] == n_columns:
            coeffs += 1000.0 * slag_settings.k * price * coeff_array

    # Flux calcination. Only flux columns carry LOI; ore chemistry has no LOI
    # field, so ore columns contribute nothing here. See the module docstring
    # for why that is a known gap rather than an assumption.
    flux_settings = settings.term(TERM_FLUX_CALCINATION)
    if flux_settings.enabled and fluxes:
        for offset, flux in enumerate(fluxes):
            dry_fraction = compute_dry_fraction(getattr(flux, "moisture_pct", 0.0))
            loi_pct = max(0.0, _float_or(getattr(flux, "loi_pct", 0.0), 0.0))
            co2_kg_per_wet_mt = 1000.0 * dry_fraction * (loi_pct / 100.0)
            coeffs[len(ores) + offset] += (
                flux_settings.k * price * co2_kg_per_wet_mt
            )

    # Burden oxygen. Ore columns only; flux carries no iron worth reducing.
    oxygen_settings = settings.term(TERM_BURDEN_OXYGEN)
    if oxygen_settings.enabled:
        for index, ore in enumerate(ores):
            chemistry = ore.chemistry
            dry_fraction = compute_dry_fraction(chemistry.moisture_pct)
            fe_total_pct = max(0.0, _float_or(chemistry.fe_t_pct, 0.0))
            feo_pct = max(0.0, _float_or(chemistry.feo_pct, 0.0))
            fe_in_feo_pct = min(fe_total_pct, feo_pct * FE_FROM_FEO_FACTOR)
            fe_in_fe2o3_pct = max(0.0, fe_total_pct - fe_in_feo_pct)
            oxygen_pct = (
                fe_in_fe2o3_pct * O_PER_FE_IN_FE2O3
                + fe_in_feo_pct * O_PER_FE_IN_FEO
            )
            oxygen_kg_per_wet_mt = 1000.0 * dry_fraction * (oxygen_pct / 100.0)
            coeffs[index] += oxygen_settings.k * price * oxygen_kg_per_wet_mt

    return coeffs
