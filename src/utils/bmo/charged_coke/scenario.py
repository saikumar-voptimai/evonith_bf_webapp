"""What-if coke for a proposed PCI, nut coke or blend, on top of the live forecast.

Port of the research package's ``scenario.estimate``. The live forecast is the
reference and stays fixed (same thermal state and history); expected slag is
recomputed for the proposal with the pinned physics, and the selected response
is applied once:

    coke = reference + b_pci * dPCI + b_nut * dNut + b_slag * dSlag

Two response modes, never mixed or relabelled:

* ``fitted``: the frozen model's own coefficients (PCI -0.452, nut -0.219,
  slag +0.0205 kg coke per kg). The slag term is weakly identified.
* ``engineering``: plant replacement assumptions (PCI -0.53, nut -1.0 i.e.
  1:1, slag +0.22). An assumption until intervention data support it.

A scenario carries no error range: the live forecast's range was calibrated on
historical forecast errors, not on operator interventions. Whether the proposed
control state lies inside historical support is checked separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.bmo.charged_coke.physics import expected_slag
from utils.bmo.charged_coke.policy import PolicyConfig

ENGINEERING_BETA = np.array([-0.53, -1.0, 0.22])
RESPONSE_MODES = ("fitted", "engineering")


def response_coefficients(
    mode: str,
    fitted_beta: np.ndarray,
    *,
    slag_coefficient: float | None = None,
) -> np.ndarray:
    """PCI, nut-coke and slag coefficients (kg coke per kg) for a scenario.

    The optional slag coefficient keeps the fitted PCI and nut-coke responses
    while the BMO tests one explicit candidate-blend response.
    """

    if mode == "fitted":
        beta = np.asarray(fitted_beta, dtype=float).copy()
    elif mode == "engineering":
        beta = ENGINEERING_BETA.copy()
    else:
        raise ValueError(f"Unknown response mode {mode!r}; use one of {RESPONSE_MODES}")
    if slag_coefficient is not None:
        beta[2] = float(slag_coefficient)
    return beta


@dataclass(frozen=True)
class ScenarioResult:
    """Conditional coke for one proposal. All rates in kg/THM.

    Attributes:
        mode: Response mode used.
        coefficients: The three coefficients applied.
        reference_coke: Live forecast the scenario is displaced from.
        coke: Scenario coke.
        pci, nut, slag: Scenario controls (slag recomputed).
        effects: Coke change from each of PCI, nut coke and slag.
        total_fuel: coke + PCI + nut coke (mass, not carbon-equivalent).
        support: Distance of the proposed state from historical support, and
            whether it is inside the policy's stop limit.
        flags: Plain-language cautions.
    """

    mode: str
    coefficients: tuple[float, float, float]
    reference_coke: float
    coke: float
    pci: float
    nut: float
    slag: float
    slag_change: float
    effects: dict[str, float]
    total_fuel: float
    pci_source: str
    nut_source: str
    support: dict[str, Any] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def estimate(
    reference_coke: float,
    causal_row: Mapping[str, Any],
    physics_config: Mapping[str, Any],
    *,
    mode: str,
    fitted_beta: np.ndarray,
    slag_coefficient: float | None = None,
    pci_override: float | None = None,
    nut_override: float | None = None,
    blend_scale: Mapping[str, float] | None = None,
) -> ScenarioResult:
    """Coke for a proposed PCI / nut coke / blend at the live state.

    Args:
        reference_coke: The live forecast (kg/THM) at this data row.
        causal_row: The data row's causal inputs (from ``build_features``).
        physics_config: Frozen physics configuration.
        mode: ``"fitted"`` or ``"engineering"``.
        fitted_beta: The bundle's fitted coefficients.
        pci_override: Proposed PCI, kg/THM; ``None`` keeps live. Zero is a
            real proposal, not missing.
        nut_override: Proposed nut coke, kg/THM.
        blend_scale: Multipliers on ``ORE``/``SINTER``/``PELLET``/``FLUX`` mass.

    Returns:
        The scenario, with the effect of each change shown separately.
    """

    beta = response_coefficients(
        mode, fitted_beta, slag_coefficient=slag_coefficient
    )
    pci = float(causal_row["PCI_KG/THM"])
    nut = float(causal_row["nut"])
    new_pci = pci if pci_override is None else float(pci_override)
    new_nut = nut if nut_override is None else float(nut_override)
    if not np.isfinite([reference_coke, pci, nut, new_pci, new_nut, *beta]).all():
        raise ValueError("Non-finite scenario inputs")
    if not 0 <= new_pci <= 350 or not 0 <= new_nut <= 150:
        raise ValueError("PCI or nut coke outside broad physical screen")
    if np.any(beta[:2] > 0) or beta[2] < 0:
        raise ValueError("Scenario coefficients violate signed contract")
    old = expected_slag(causal_row, physics_config)
    new = expected_slag(causal_row, physics_config, pci=new_pci, nut=new_nut, blend_scale=blend_scale)
    delta = np.array([new_pci - pci, new_nut - nut, new["slag"] - old["slag"]])
    effects = beta * delta
    coke = float(reference_coke + effects.sum())
    if not 150 <= coke <= 700:
        raise ValueError("Scenario coke outside screening envelope")
    flags = []
    if not 150 <= new_pci <= 250:
        flags.append("PCI outside the cruise range 150-250 kg/THM")
    if blend_scale:
        flags.append("Blend change: check hot-metal and burden constraints before use")
    if mode == "engineering":
        flags.append("Engineering assumption, not validated by intervention data")
    return ScenarioResult(
        mode=mode,
        coefficients=tuple(float(b) for b in beta),
        reference_coke=float(reference_coke),
        coke=coke,
        pci=new_pci,
        nut=new_nut,
        slag=float(new["slag"]),
        slag_change=float(delta[2]),
        effects={"pci": float(effects[0]), "nut": float(effects[1]), "slag": float(effects[2])},
        total_fuel=coke + new_pci + new_nut,
        pci_source="live" if pci_override is None else "override",
        nut_source="live" if nut_override is None else "override",
        flags=tuple(flags),
    )


def scenario_support(
    live_gate_row: Mapping[str, Any],
    policy: PolicyConfig,
    *,
    pci: float,
    nut: float,
    slag: float,
) -> dict[str, Any]:
    """Is the proposed control state inside historical support?

    The live hour's process levels and variability are kept; PCI, nut coke and
    expected slag are replaced by the proposal, and the nearest-neighbour
    distance is compared with the policy's warn and stop limits.
    """

    from utils.bmo.charged_coke.policy import novelty

    proposed = dict(live_gate_row)
    proposed.update({"pci": pci, "nut": nut, "slag": slag})
    distance = float(novelty(pd.DataFrame([proposed]), policy)[0])
    p = policy.parameters
    supported = bool(np.isfinite(distance) and distance <= p["novelty_stop"])
    return {
        "distance": distance,
        "warn_limit": float(p["novelty_warn"]),
        "stop_limit": float(p["novelty_stop"]),
        "supported": supported,
        "unusual": bool(np.isfinite(distance) and distance > p["novelty_warn"]),
    }


__all__ = ["ENGINEERING_BETA", "RESPONSE_MODES", "ScenarioResult", "estimate", "response_coefficients", "scenario_support"]
