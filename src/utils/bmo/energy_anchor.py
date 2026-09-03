"""The coke rate the fuel cost is built on, from physics rather than history.

WHAT CHANGED AND WHY.

The BMO fuel cost used to start from one of two places, and neither was good:

    model_cost   back-solve coke from the ML model's predicted unit cost. The
                 model's output is very nearly a constant (~13,364 Rs/THM) and
                 it is blend-blind, so this pinned the reported fuel rate to
                 roughly 487 + 0.357 x PCI no matter what the plant was doing.
    observed     use the coke rate off the Fuel Ash table. Honest, but it is
                 yesterday's answer - it cannot respond to a change in controls,
                 and it carries whatever the operator last typed.

This module supplies a third: solve the closed energy balance at the CURRENT
controls and the CURRENT burden, then subtract the rolling bias offset. That is
the methodology backtested in ``coke_calibration.py`` - MAPE 3.4%, R2 0.74
forward - and it is the level the whole fuel cost then sits on.

WHAT THIS DOES NOT DO.

It does not make the cost blend-sensitive. The anchor is one number for the
whole run, identical for every candidate blend, exactly like the observed
anchor it replaces. Blend sensitivity comes from the physics coke correction
(``coke_correction.py``), which is zero at the reference point and moves the
coke rate as the burden's oxygen, slag and flux CO2 demands move. Anchor sets
the LEVEL; correction sets the SHAPE. Keeping them separate is what lets the
correction stay anchored at zero for current conditions.

Being one number per run also matters for the optimizer: a per-candidate energy
balance inside the DE loop would be tens of thousands of solves, and would make
the objective a function whose level moves with its own argument.

WHEN IT DECLINES.

Live blast tags missing, no calibration on file, or a solve that lands outside
150-600 kg/THM - it returns a result marked not usable, and the caller falls
back to the observed anchor. A fuel cost is always produced; it just says which
basis produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from utils.bmo.coke_calibration import CokeCalibration, load_calibration

log = logging.getLogger(__name__)

# Outside this band the balance has not solved, it has diverged. The plant runs
# 280-360 kg/THM; the widest excursion in 281 days of history was under 450.
PLAUSIBLE_COKE_RATE_KG_THM = (150.0, 600.0)


@dataclass(frozen=True)
class EnergyAnchor:
    """The energy balance's answer for right now, and whether to trust it."""

    coke_rate_kg_thm: float
    raw_coke_rate_kg_thm: float
    calibration: CokeCalibration
    usable: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def offset_kg_per_thm(self) -> float:
        return float(self.calibration.offset_kg_per_thm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coke_rate_kg_thm": self.coke_rate_kg_thm,
            "raw_coke_rate_kg_thm": self.raw_coke_rate_kg_thm,
            "offset_kg_per_thm": self.offset_kg_per_thm,
            "calibration_days": self.calibration.sample_days,
            "calibration_fitted_on": self.calibration.fitted_on,
            "usable": self.usable,
            "notes": list(self.notes),
        }


class _CurrentBurden:
    """The minimum ``blend_to_energy_inputs`` reads off a blend.

    The adapter wants a solved BlendEvaluation, but for the anchor there is no
    solved blend - the operating point is what the plant is charging now. Rather
    than build a full evaluation just to read two attributes off it, this states
    plainly which two they are.
    """

    def __init__(self, quantities_mt: Mapping[str, float], slag_mt: float) -> None:
        self.quantities_mt = dict(quantities_mt)
        self.slag_mt = float(slag_mt)


def solve_energy_anchor(
    *,
    quantities_mt: Mapping[str, float],
    ores: list[Any],
    slag_mt: float,
    hot_metal_mt: float,
    fuel_rates_kg_per_thm: Mapping[str, float],
    hm_chemistry: Mapping[str, float],
    process_snapshot: Mapping[str, Any],
    flux_mt: float = 0.0,
    fuel_vm_pct: Mapping[str, float] | None = None,
    shell_loss_gj_per_hr: float | None = None,
    calibration: CokeCalibration | None = None,
) -> EnergyAnchor:
    """Calibrated energy-balance coke rate for the current operating point.

    Args:
         - quantities_mt: Mapping[str, float] - Burden as currently charged.
         - ores: list[OreInput] - Ores backing those quantities.
         - slag_mt: float - Slag for the same basis.
         - hot_metal_mt: float - Hot-metal basis for the day.
         - fuel_rates_kg_per_thm: Mapping - coke / nut_coke / pci rates. Coke is
           what gets solved for, so its incoming value only seeds the burden
           mass; nut coke and PCI are held, being operator-set run inputs.
         - hm_chemistry: Mapping - carbon / silicon / iron / manganese pct and
           slag FeO.
         - process_snapshot: Mapping - Live blast and top-gas tags.
         - flux_mt: float - Flux charged.
         - fuel_vm_pct: Mapping | None - Volatile matter per fuel.
         - shell_loss_gj_per_hr: float | None - Measured stave heat load.
         - calibration: CokeCalibration | None - Override for tests; the stored
           calibration is loaded when omitted.

    Returns:
         - return EnergyAnchor - Always returned; check ``usable`` before use.
    """

    calib = calibration if calibration is not None else load_calibration()

    if not float(process_snapshot.get("hot_blast_vol_nm3h", 0.0) or 0.0):
        return EnergyAnchor(
            coke_rate_kg_thm=0.0, raw_coke_rate_kg_thm=0.0, calibration=calib,
            notes=["live blast tags unavailable - energy balance cannot be solved"],
        )
    if hot_metal_mt <= 0.0:
        return EnergyAnchor(
            coke_rate_kg_thm=0.0, raw_coke_rate_kg_thm=0.0, calibration=calib,
            notes=["hot metal basis is zero"],
        )

    # Imported here, not at module import: this pulls in scipy through the
    # recommendation layer, and the module is loaded on every page render
    # whether or not the anchor is switched on.
    from utils.bmo.process_recommendation import blend_to_energy_inputs
    from utils.energy_balance.assumptions import apply_overrides
    from utils.energy_balance.constants import load_config
    from utils.energy_balance.solve import solve_coke_rate_kg_per_thm

    try:
        inputs = blend_to_energy_inputs(
            _CurrentBurden(quantities_mt, slag_mt),
            hot_metal_mt=float(hot_metal_mt),
            ores=list(ores),
            fuel_rates_kg_per_thm=dict(fuel_rates_kg_per_thm),
            hm_chemistry=dict(hm_chemistry),
            process_snapshot=dict(process_snapshot),
            flux_mt=float(flux_mt),
            fuel_vm_pct=dict(fuel_vm_pct or {}),
            shell_loss_gj_per_hr=shell_loss_gj_per_hr,
        )
        # Operator overrides for the unmeasured constants are applied at the app
        # boundary, the same way the process-recommendation panel does it, so the
        # anchor and the panel cannot disagree about the assumptions.
        raw = float(solve_coke_rate_kg_per_thm(inputs, apply_overrides(load_config())))
    except Exception as exc:  # noqa: BLE001 - never break a blend evaluation
        log.warning("Energy anchor failed to solve: %s", exc)
        return EnergyAnchor(
            coke_rate_kg_thm=0.0, raw_coke_rate_kg_thm=0.0, calibration=calib,
            notes=[f"energy balance did not solve: {str(exc)[:120]}"],
        )

    low, high = PLAUSIBLE_COKE_RATE_KG_THM
    if not (low <= raw <= high):
        return EnergyAnchor(
            coke_rate_kg_thm=0.0, raw_coke_rate_kg_thm=raw, calibration=calib,
            notes=[
                f"energy balance solved to {raw:,.0f} kg/THM, outside the "
                f"plausible {low:,.0f}-{high:,.0f} band - inputs are wrong "
                "somewhere, so the observed rate is used instead"
            ],
        )

    notes: list[str] = []
    if not calib.is_usable:
        # Without an offset the raw balance runs ~20 kg/THM high, which is a
        # 550 Rs/THM error in the fuel cost. Better to say so and fall back.
        return EnergyAnchor(
            coke_rate_kg_thm=raw, raw_coke_rate_kg_thm=raw, calibration=calib,
            notes=[*calib.notes,
                   "no usable calibration - the raw balance runs about 20 kg/THM "
                   "high, so it is not used as the cost anchor"],
        )
    if calib.is_stale():
        notes.append(
            f"calibration is {calib.age_days()} days old; the bias drifts about "
            "3.3 kg/THM per month, so refit it"
        )

    return EnergyAnchor(
        coke_rate_kg_thm=calib.apply(raw),
        raw_coke_rate_kg_thm=raw,
        calibration=calib,
        usable=True,
        notes=notes,
    )
