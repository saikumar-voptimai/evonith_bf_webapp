from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from furnace_data.vsense.bounds import default_control_profile
from furnace_data.vsense.catalog import (
    control_parameter_by_feature,
    load_vsense_catalog,
    optimization_by_id,
    parameter_by_id,
    target_for_optimization,
)
from furnace_data.vsense.context import build_context_snapshot
from furnace_data.vsense.optimizer import run_legacy_optimization


def _history() -> pd.DataFrame:
    rows = []
    for idx in range(2):
        row = {}
        for feature, definition in control_parameter_by_feature().items():
            row[feature] = float(definition["default_value"]) + (idx * 0.01)
        for optimization_type_id in ("eta_co", "production_rate", "unit_cost"):
            target = target_for_optimization(optimization_type_id)
            row[target["feature_name"]] = 0.0 if optimization_type_id == "eta_co" else 100.0
        rows.append(row)
    return pd.DataFrame(
        rows,
        index=pd.date_range("2026-07-23T04:00:00Z", periods=2, freq="15min"),
    )


def test_vsense_catalog_ids_and_references_are_valid():
    catalog = load_vsense_catalog()
    params = parameter_by_id(catalog)
    optimizations = optimization_by_id(catalog)

    assert set(optimizations) == {"eta_co", "production_rate", "unit_cost"}
    assert params["hot_blast_pressure_bar"]["unit"] == "bar"
    for optimization in optimizations.values():
        assert optimization["target"]["direction"] in {"maximize", "minimize"}
        assert all(parameter_id in params for parameter_id in optimization["control_parameter_ids"])
        assert all(parameter_id in params for parameter_id in optimization["impact_target_ids"])


def test_legacy_v1_all_fixed_result_is_finite_and_advisory():
    context = build_context_snapshot(
        optimization_type_id="eta_co",
        data_mode="live",
        now=datetime(2026, 7, 23, 4, 30, tzinfo=timezone.utc),
        history_df=_history(),
    )
    fixed_plan = [
        {
            **item,
            "mode": "fixed",
            "fixed_value": next(
                control["value"]
                for control in context["controls"]
                if control["parameter_id"] == item["parameter_id"]
            ),
        }
        for item in default_control_profile("eta_co")
    ]

    result = run_legacy_optimization(
        context=context,
        control_plan=fixed_plan,
        input_overrides=[],
        lambda_reg=0.05,
    )

    assert result["advisory_only"] is True
    assert result["requires_operator_review"] is True
    assert result["diagnostics"]["algorithm_version"] == "legacy_v1"
    assert result["diagnostics"]["optimizer"]["all_controls_fixed"] is True
    assert result["target"]["delta_pct"] is None
    assert result["feasibility"]["feasible"] is True
