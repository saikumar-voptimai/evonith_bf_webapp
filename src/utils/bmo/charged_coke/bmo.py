"""The charged-coke engine as the BMO's Data-Driven coke candidate.

The BMO needs one coke level for the run and one blend-to-blend response:

Level
    The issued live forecast for the latest complete hour (fitted model), with
    its window, state, range and provenance. If the operator overrides PCI or
    nut coke, the selected response mode moves it once, at the same live state
    (``scenario.estimate``, slag recomputed for the fuel-ash change). When the
    forecast is paused there is no level: the page blocks the run and offers
    Physics-Driven as a separate, separately labelled choice. Nothing is
    substituted under the Data-Driven name.

Blend response
    ``response_settings`` turns the page's physics coke correction into one
    linear slag term, coefficient from the selected mode, measured from the
    current burden's own modelled slag, all other terms off. Each blend's coke
    is then ``level + b_slag x (slag_blend - slag_current)``: one correction,
    applied once, on the same BMO slag basis for both sides.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.bmo.charged_coke.contract import ForecastWindow
from utils.bmo.charged_coke.features import build_features
from utils.bmo.charged_coke.ledger import ForecastLedger, PlannedChangeRegister
from utils.bmo.charged_coke.scenario import estimate, response_coefficients
from utils.bmo.charged_coke.service import ChargedCokeEngine, issue_latest
from utils.bmo.coke_correction import (
    TERM_SLAG_HEAT,
    CokeCorrectionSettings,
    CokeCorrectionTermSettings,
)
ENGINE_ID = "charged_4h"
_UNBOUNDED = 1e9


@dataclass(frozen=True)
class ChargedCokePrediction:
    """Guarded charged-coke value passed from the forecast into BMO."""

    value_kg_per_thm: float | None
    usable: bool
    origin_utc: str = ""
    aggregation: str = "charged_4h_block"
    target_window_hours: int | None = 4
    lookback_hours: float = 1.0
    hourly_prediction_count: int = 1
    deployment_id: str = "bundled"
    model_path: str = ""
    latest_input_diagnostics: dict[str, Any] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def response_settings(
    base: CokeCorrectionSettings,
    *,
    coefficient: float | None = None,
    mode: str = "fitted",
    fitted_beta: np.ndarray | None = None,
) -> CokeCorrectionSettings:
    """The candidate-blend response as a single linear slag term.

    Args:
        base: The page's configured correction settings (bands, objective use).
        coefficient: Explicit BMO trial value. When omitted, use ``mode``.
        mode: Backwards-compatible fitted/engineering response selector.
        fitted_beta: Bundle coefficients, required when ``coefficient`` is omitted.

    Returns:
        Settings whose only active term is slag, linear, referenced to the
        current burden's modelled slag.
    """

    if coefficient is None:
        if fitted_beta is None:
            raise ValueError("fitted_beta is required when coefficient is omitted")
        coefficient = float(response_coefficients(mode, fitted_beta)[2])
        source = mode
    else:
        coefficient = float(coefficient)
        source = "configured"
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("blend-response coefficient must be finite and non-negative")
    slag = CokeCorrectionTermSettings(
        enabled=True,
        k=coefficient,
        k_config_value=coefficient,
        k_config_units=f"kg coke per kg slag ({source})",
        max_abs_kg_thm=_UNBOUNDED,
        envelope_halfwidth=None,
        reference_source="model_current",
    )
    return replace(base, enabled=True, max_abs_correction_kg_thm=_UNBOUNDED, taper_start_fraction=1.0,
                   terms={TERM_SLAG_HEAT: slag})


def _built_at(meta_path: Path) -> pd.Timestamp | None:
    import json

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stamp = meta.get("source_last_modified") or ""
    try:
        return pd.Timestamp(stamp) if stamp else None
    except ValueError:
        return None


def charged_coke_prediction(
    source: pd.DataFrame,
    engine: ChargedCokeEngine,
    *,
    now: pd.Timestamp,
    storage_dir: str | Path,
    mode: str = "fitted",
    slag_response_coefficient: float | None = None,
    built_at: pd.Timestamp | None = None,
    pci_override: float | None = None,
    nut_override: float | None = None,
    force_predict: bool = False,
) -> tuple[ChargedCokePrediction, dict[str, Any]]:
    """Issue (once per hour) and return the Data-Driven level for the BMO.

    Returns:
        The page's prediction type, and the issued record with any operator
        override applied, for display.
    """

    record = issue_latest(
        source, engine, ledger=ForecastLedger(storage_dir), plans=PlannedChangeRegister(storage_dir),
        now=now, built_at=built_at, model_version=str(engine.model.manifest.get("version", "")),
    )
    detail: dict[str, Any] = {
        "record": record,
        "mode": mode,
        "override": None,
        "forced": False,
    }
    common = dict(
        origin_utc=str(record.get("data_row") or ""),
        aggregation="charged_4h_block",
        target_window_hours=4,
        lookback_hours=1.0,
        hourly_prediction_count=1,
        deployment_id=f"{ENGINE_ID}/{engine.model.manifest.get('version', '')}",
        model_path=str(engine.model.manifest.get("name", "")),
        latest_input_diagnostics={"state": record.get("state"), "coke_window": str(record.get("coke_window")),
                                  "issued_at": str(record.get("issued_at"))},
    )
    forced_value = record.get("withheld_value_kg_thm")
    can_force = bool(
        force_predict
        and forced_value is not None
        and np.isfinite(float(forced_value))
    )
    if not record.get("shown") and not can_force:
        return ChargedCokePrediction(
            value_kg_per_thm=None,
            usable=False,
            reasons=tuple(record.get("reasons") or ()),
            **common,
        ), detail

    if can_force:
        # Keep the append-only issued record untouched.  The page receives a
        # display copy with the model's already-computed, withheld values
        # exposed.  No range is invented outside the validated policy.
        record = dict(record)
        record["prediction_kg_thm"] = float(forced_value)
        record["path"] = [
            {
                **point,
                "prediction_kg_thm": point.get("withheld_value_kg_thm"),
            }
            for point in (record.get("path") or [])
        ]
        record["forced"] = True
        detail["record"] = record
        detail["forced"] = True

    level = float(record["prediction_kg_thm"])
    if pci_override is not None or nut_override is not None:
        row = pd.Timestamp(record["data_row"])
        features = build_features(source, engine.physics_config, start=row - pd.Timedelta(hours=4), end=row)
        scenario = estimate(
            level,
            features.causal.loc[row],
            engine.physics_config,
            mode=mode,
            fitted_beta=engine.model.beta,
            slag_coefficient=slag_response_coefficient,
            pci_override=pci_override,
            nut_override=nut_override,
        )
        detail["override"] = scenario.to_dict()
        level = scenario.coke
    return ChargedCokePrediction(
        value_kg_per_thm=level,
        usable=True,
        reasons=tuple(record.get("reasons") or ()) if can_force else (),
        **common,
    ), detail


def window_text(record: dict[str, Any]) -> str:
    """Plain description of an issued record's block."""

    if not record.get("data_row"):
        return ""
    window = ForecastWindow.for_row(pd.Timestamp(record["data_row"]), pd.Timestamp(record["issued_at"]))
    return window.describe()


__all__ = [
    "ENGINE_ID",
    "ChargedCokePrediction",
    "charged_coke_prediction",
    "response_settings",
    "window_text",
]
