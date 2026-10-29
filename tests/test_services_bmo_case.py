"""services.bmo_case: page 9's run inputs and LP plan without Streamlit (SV-3.5, #302).

The parity tests compose each golden day of page 9 (``tests/bmo_golden.py``: frozen provider calls, the day's hourly
dataset, live tags and clock, the snapshot's applied tables and widget values) through ``compose_case`` and
``run_lp`` and compare the LP plan, the coke anchor, the derived targets and the Si the run used with the record page
9 wrote for that day (relative 1e-6). The other tests pin the derived quantities on small inputs.
"""

from __future__ import annotations

import ast
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from services import bmo, bmo_case, bmo_views
from tests import bmo_golden
from utils.bmo.types import OreChemistry, OreInput

CASES = {case.day: case for case in bmo_golden.CASES}
PLANT_TZ = "Asia/Kolkata"


# ---------------------------------------------------------------------------- the golden days through the service
def golden_inputs(case: bmo_golden.Case, plant: dict[str, Any], frozen: dict[str, Any]) -> bmo_case.CaseInputs:
    """The operator's values of a golden day: the snapshot's widget values and applied tables, the case's overrides."""

    from utils.bmo.snapshot import restorable_state

    values = {k.removeprefix("bmo_"): v for k, v in restorable_state(plant, prefix="bmo_").items()}
    values.update(case.inputs)
    pci = values.get("pci_override_kg") if values.get("pci_override_on") else None
    nut = values.get("nut_coke_override_kg") if values.get("nut_coke_override_on") else None
    scalar = {
        name: values.get(name)
        for name in (
            "target_production_mt",
            "target_slag_rate_kg_per_thm",
            "target_slag_basicity_min",
            "target_slag_basicity_max",
            "target_slag_t_basicity_min",
            "target_slag_t_basicity_max",
            "target_slag_al2o3_max_pct",
            "target_slag_mgo_min_pct",
            "target_slag_mgo_al2o3_ratio_min",
            "max_charges_per_hour",
            "charge_mass_mt",
            "burden_capacity_enabled",
            "chemistry_mode",
            "chemistry_window_days",
            "blend_response_kg_coke_per_kg_slag",
            "manual_blend_lookback_hours",
        )
    }
    return bmo_case.CaseInputs(
        **scalar,
        coke_model=bmo_case.COKE_MODELS[case.coke_model],
        pci_override_kg_thm=pci,
        nut_coke_override_kg_thm=nut,
        pellet_inputs_confirmed=True,  # as the golden runs: the operator has reviewed the pellet inputs
        hm_chemistry={name: values[f"hm_{name}"] for name in bmo_case.HM_CHEMISTRY_FIELDS if f"hm_{name}" in values},
        fuel_prices_rs_per_mt={
            fuel: values[f"fuel_price_{fuel}_rs_per_mt"]
            for fuel in ("coke", "nut_coke", "pci")
            if f"fuel_price_{fuel}_rs_per_mt" in values
        },
        ore_table=values.get("applied_ore_editor_df"),
        flux_table=values.get("applied_flux_editor_df"),
        fuel_ash_table=values.get("applied_fuel_ash_editor_df"),
        dust_table=values.get("applied_dust_editor_df"),
        previous_manual_quantities_mt=(frozen.get("session_at_run") or {}).get("manual_quantities_mt"),
    )


@contextmanager
def golden_day(day: str) -> Iterator[tuple[bmo_case.CaseSources, bmo_case.CaseInputs, dict[str, Any]]]:
    """The sources page 9 reads on a golden day (``bmo_golden.exec_page``'s substitutions) and the day's record."""

    from utils.bmo.charged_coke.bmo import _built_at
    from utils.bmo.replay import ReplayProvider, frozen_config

    case = CASES[day]
    plant = bmo_golden.load_plant(case.plant)
    with tempfile.TemporaryDirectory(prefix="bmo_case_") as tmp_name:
        tmp = Path(tmp_name)
        dataset = bmo_golden._day_dataset(case, tmp / "dataset")
        frozen = bmo_golden._decode_frozen(plant)
        frozen["page"]["bmo_cfg"] = bmo_golden._case_config(tmp, dataset)
        cfg = frozen_config(frozen["page"])
        page = frozen["page"]
        calls = {
            k: v for k, v in frozen["provider_calls"].items() if not k.startswith("get_recent_manual_blend_snapshot(")
        }
        provider = ReplayProvider(bmo_golden._DayProvider(dataset), calls, [])
        now_utc = pd.Timestamp(case.now).tz_localize(PLANT_TZ).tz_convert("UTC")
        live_tags = bmo_golden._live_tags(case, dataset)
        paths = bmo.forecast_service.charged_paths(cfg)

        def si_record() -> dict[str, Any] | None:
            view = bmo.si_forecast_for_page(
                cfg, frozen=False, frozen_record=None, now=now_utc.tz_convert(PLANT_TZ), source_cache_version=0
            )
            return bmo.issue_si_for_page(view.issue["origin_at"], cfg) if view.issue is not None else view.record

        def charged(**overrides: Any) -> Any:
            return bmo.charged_prediction(
                cfg=cfg,
                source=bmo.forecast_service.read_charged_source(paths.dataset_path),
                engine=bmo.forecast_service.charged_engine(
                    str(bmo.forecast_service.active_charged_bundle(paths.storage_dir, paths.default_bundle))
                ),
                storage=paths.storage_dir,
                built_at=_built_at(paths.dataset_path.parent / "cache_meta.json"),
                now=now_utc.tz_convert(PLANT_TZ).tz_localize(None),
                **overrides,
            )

        sources = bmo_case.CaseSources(
            cfg=cfg,
            provider=provider,
            preferences=page["operator_preferences"],
            now=now_utc,
            fuel_rates_static=lambda: dict(page["recent_fuel_rates"] or {}),
            fuel_rates_live=dict,
            model_input_defaults_static=lambda: dict(page["model_input_defaults"] or {}),
            dust_analysis=lambda days: bmo_case.read_dust_analysis(dataset, days),
            live_tags=lambda: dict(live_tags),
            si_record=si_record,
            charged=charged,
            model_service=lambda: bmo_case.fuel_model_service(cfg),
        )
        with bmo_golden._patched(case, tmp):
            yield sources, golden_inputs(case, plant, frozen), bmo_golden.load_golden(case)


@pytest.mark.parametrize("day", sorted(CASES))
def test_lp_plan_equals_page9(day: str) -> None:
    with golden_day(day) as (sources, inputs, golden):
        composed = bmo_case.compose_case(sources, inputs)
        run = bmo_case.run_lp(sources, composed, inputs)
    assert run.refusal is None and run.lp_result is not None
    actual = bmo_golden.normalise(
        {
            "lp_result": bmo_golden._plan(run.lp_result),
            "lp_errors": run.balanced.lp_errors,
            "coke_anchor_basis": run.anchor.basis,
            "coke_anchor_rate_kg_thm": run.anchor.rate_kg_thm,
        }
    )
    expected = {key: golden["plan"][key] for key in actual}
    diffs = bmo_golden.compare(expected, actual)
    assert not diffs, f"{len(diffs)} differences from page 9 on {day}:\n" + "\n".join(diffs[:40])
    context = bmo_golden.normalise(
        {
            "target_fe_mt": composed.case.target_fe_mt,
            "target_slag_qty_mt": composed.case.target_slag_qty_mt,
            "max_burden_qty_mt": composed.case.max_burden_qty_mt,
            "observed_slag_rate": composed.case.observed_slag_rate_kg_per_thm,
            "fuel_rate_anchor_basis": composed.case.fuel_rate_anchor_basis,
            "live_si_pct": composed.case.live_si_pct,
        }
    )
    expected_context = {**{k: golden["context"][k] for k in context if k != "live_si_pct"}}
    expected_context["live_si_pct"] = golden["si"]["live_si_pct"]
    assert not bmo_golden.compare(expected_context, context)

    # The twin-api's result of the run (``bmo.run``: services.bmo_views.run_result) carries page 9's plan.
    view = bmo_views.run_result(run, computed_at=pd.Timestamp("2026-10-29", tz="UTC"), seconds=0.0)
    view = bmo_views.BmoRunResult.model_validate_json(view.model_dump_json())  # through JSON, as the job result
    plan, lp = view.plan, golden["plan"]["lp_result"]
    rates = lp["fuel_rate_estimate"] or {}
    total = (lp["adjusted_objective_rs_per_thm"] or lp["objective_rs_per_thm"]) + (lp["flux_cost_per_thm_rs"] or 0.0)
    expected_view = {
        "status": "ok" if lp["feasible"] else "infeasible",
        "shares_pct": lp["shares_pct"],
        "quantities_mt": lp["quantities_mt"],
        "total_qty_t_d": lp["total_qty_mt"],
        "ore_cost_rs_thm": lp["ore_cost_per_thm_rs"],
        "total_cost_rs_thm": total,
        "fe_t_pct": lp["fe_t_pct"],
        "slag_t_d": lp["slag_mt"],
        "slag_rate_kg_thm": lp["slag_rate_kg_per_thm"],
        "slag_basicity": lp["slag_basicity"],
        "coke_rate_kg_thm": rates.get("coke_rate_kg_thm"),
        "fuel_rate_kg_thm": rates.get("total_fuel_rate_kg_thm"),
        "violations": lp["violations"],
        "anchor_kg_thm": golden["plan"]["coke_anchor_rate_kg_thm"],
        "fe_t_d": golden["context"]["target_fe_mt"],
        "slag_cap_t_d": golden["context"]["target_slag_qty_mt"],
        "max_burden_t_d": golden["context"]["max_burden_qty_mt"],
    }
    actual_view = bmo_golden.normalise(
        {
            "status": view.status,
            "shares_pct": {r.ore_id: r.share_pct for r in plan.ores if r.ore_id in lp["shares_pct"]},
            "quantities_mt": {r.ore_id: r.quantity_t_d for r in plan.ores if r.ore_id in lp["quantities_mt"]},
            "total_qty_t_d": plan.total_qty_t_d,
            "ore_cost_rs_thm": plan.ore_cost_rs_thm,
            "total_cost_rs_thm": plan.total_cost_rs_thm,
            "fe_t_pct": plan.fe_t_pct,
            "slag_t_d": plan.slag_t_d,
            "slag_rate_kg_thm": plan.slag_rate_kg_thm,
            "slag_basicity": plan.slag_basicity,
            "coke_rate_kg_thm": plan.coke_rate_kg_thm,
            "fuel_rate_kg_thm": plan.fuel_rate_kg_thm,
            "violations": plan.violations,
            "anchor_kg_thm": view.anchor.rate_kg_thm,
            "fe_t_d": view.targets.fe_t_d,
            "slag_cap_t_d": view.targets.slag_cap_t_d,
            "max_burden_t_d": view.targets.max_burden_t_d,
        }
    )
    diffs = bmo_golden.compare(bmo_golden.normalise(expected_view), actual_view)
    assert not diffs, f"the API result differs from page 9 on {day}:\n" + "\n".join(diffs[:40])


def test_a_bmo_job_runs_page_9s_case_from_its_params() -> None:
    """``bmo.run``'s function on a golden day's sources: the request's values reach the case, the result is JSON."""

    import json

    with golden_day("2026-10-01") as (sources, inputs, _golden):
        params = {"target_production_mt": 2200.0, "coke_model": "data_driven", "pellet_inputs_confirmed": True}
        if inputs.pci_override_kg_thm is not None:
            params["pci_override_kg_thm"] = inputs.pci_override_kg_thm
        out = bmo_views.run_case(params, sources=sources)
    result = bmo_views.BmoRunResult.model_validate(json.loads(json.dumps(out, allow_nan=False)))
    assert result.targets.production_t_d == 2200.0 and result.coke_model == "data_driven"
    assert result.status in ("ok", "infeasible") and result.plan is not None and result.seconds > 0
    assert result.targets.fe_t_d == pytest.approx(2200.0 * result.targets.hm_fe_pct / 100.0, rel=1e-12)


# ---------------------------------------------------------------------------- derived quantities
def _ore(ore_id: str, name: str, *, fe: float = 60.0) -> OreInput:
    return OreInput(
        ore_id=ore_id,
        display_name=name,
        stock_mt=1000.0,
        price_rs_per_mt=100.0,
        min_share_pct=0.0,
        max_share_pct=100.0,
        chemistry=OreChemistry(fe_t_pct=fe, moisture_pct=5.0),
    )


def test_overrides_reach_the_fuel_rates_at_source() -> None:
    static = {"pci_rate_kg_thm": 150.0, "coke_rate_kg_thm": 400.0, "nut_coke_rate_kg_thm": 92.0}
    live = {"pci_rate_kg_thm": 171.0, "pci_source": "influx_1h_avg.coal_rate_actual_value"}
    rates = bmo_case.effective_fuel_rates(static, live, pci_override_kg_thm=None, nut_coke_override_kg_thm=None)
    assert rates["pci_rate_kg_thm"] == 171.0 and rates["coke_rate_kg_thm"] == 400.0
    assert rates["nut_coke_rate_kg_thm"] == 70.0 and rates["nut_coke_source"] == "fixed plant set point"
    over = bmo_case.effective_fuel_rates(static, live, pci_override_kg_thm=-5.0, nut_coke_override_kg_thm=55.0)
    assert over["pci_rate_kg_thm"] == 0.0 and over["pci_source"] == "operator override"
    assert over["nut_coke_rate_kg_thm"] == 55.0 and over["nut_coke_source"] == "operator override"


def test_slag_window_switches_zero_limits_off_and_converts_the_cap() -> None:
    window = bmo_case.slag_window(
        basicity_min=1.05,
        basicity_max=1.20,
        t_basicity_min=0.0,
        t_basicity_max=0.0,
        al2o3_max_pct=0.0,
        mgo_min_pct=7.0,
        mgo_al2o3_ratio_min=0.36,
        slag_rate_kg_per_thm=300.0,
        target_production_mt=2000.0,
        model_to_plant_factor=0.8,
    )
    assert window.t_basicity_min is None and window.t_basicity_max is None and window.al2o3_max_pct is None
    assert window.mgo_min_pct == 7.0 and window.bounds_valid
    assert window.slag_qty_mt == pytest.approx(300.0 * 2000.0 / 1000.0 / 0.8)  # t/day on the model basis
    inverted = bmo_case.slag_window(
        basicity_min=1.3,
        basicity_max=1.2,
        t_basicity_min=0.0,
        t_basicity_max=10.0,
        al2o3_max_pct=20.0,
        mgo_min_pct=0.0,
        mgo_al2o3_ratio_min=0.0,
        slag_rate_kg_per_thm=290.0,
        target_production_mt=2270.0,
        model_to_plant_factor=1.0,
    )
    assert not inverted.bounds_valid


def test_target_fe_and_charging_ceiling() -> None:
    fe, pct, warning = bmo_case.target_fe(2300.0, {"hm_fe_pct_for_target": 94.0}, {})
    assert (fe, pct, warning) == (pytest.approx(2162.0), 94.0, None)
    fe, pct, warning = bmo_case.target_fe(2000.0, {"hm_fe_pct_for_target": -1.0}, {})
    assert pct == 94.5 and fe == pytest.approx(1890.0) and "fallback" in warning
    nut = bmo_case.NutCokeCharge(70.0, 5.0)
    off, note = bmo_case.charging_ceiling(
        enabled=False, max_charges_per_hour=6.35, charge_mass_mt=26.4, target_production_mt=2270.0, nut_coke=nut
    )
    assert off is None and note is None
    on, note = bmo_case.charging_ceiling(
        enabled=True, max_charges_per_hour=6.35, charge_mass_mt=26.4, target_production_mt=2270.0, nut_coke=nut
    )
    nut_t = 70.0 * 2270.0 / 1000.0 * 1.05  # t/day of wet nut coke in the charges
    assert on == pytest.approx(6.35 * 26.4 * 24.0 - nut_t, rel=1e-6) and note is None
    full, note = bmo_case.charging_ceiling(
        enabled=True, max_charges_per_hour=0.1, charge_mass_mt=1.0, target_production_mt=2270.0, nut_coke=nut
    )
    assert full is None and "nut coke alone" in note


def test_nut_coke_charge_adds_moisture_only_to_a_base_rate() -> None:
    cfg = {"fuel_ash_inputs": [{"fuel_id": "nut_coke", "rate_kg_per_thm": 65.0, "moisture_pct": 4.0}]}
    from_rates = bmo_case.nut_coke_charge(cfg, {"nut_coke_rate_kg_thm": 70.0}, {"nut_coke": {"moisture_pct": 6.5}})
    assert from_rates == bmo_case.NutCokeCharge(70.0, 6.5)
    from_cfg = bmo_case.nut_coke_charge(cfg, {}, {})
    assert from_cfg == bmo_case.NutCokeCharge(65.0, 0.0)  # a configured rate is already wet


def test_default_selection_and_applied_tables() -> None:
    ores = [_ore("sinter", "Sinter"), _ore("pellet_a", "Pellet A"), _ore("lump", "Lump")]
    ids = bmo_case.default_ore_selection(ores, {"default_selected_ores": ["Sinter", "Lump"]}, ["pellet_a"])
    assert ids == ["lump", "pellet_a", "sinter"]
    assert bmo_case.default_ore_selection(ores, {"default_selected_ores": ["Lump"]}, None) == ["lump"]
    base = bmo_case.ore_editor_table(ores, ids, {})
    applied = base.assign(price_rs_per_mt=[1.0, 2.0, 3.0])
    assert bmo_case.applied_or_base("ore", applied, base) is applied
    assert bmo_case.applied_or_base("ore", applied.iloc[:2], base) is base  # another ore list: the base
    assert bmo_case.applied_or_base("ore", None, base) is base
    fuel = pd.DataFrame({"fuel_id": ["coke"], "rate_kg_per_thm": [400.0]})
    assert bmo_case.applied_or_base("fuel_ash", fuel, fuel.assign(vm_pct=1.0)) is not fuel  # no VM column: the base
    dust = pd.DataFrame({"dust_id": ["old_dust"]})
    assert bmo_case.applied_or_base("dust", dust, pd.DataFrame({"dust_id": ["gcp_dust"]})) is dust
    chosen = bmo_case.with_selection(base, ["sinter"])
    selected = bmo_case.selected_ores_from_editor(chosen, ores)
    assert [o.ore_id for o in selected] == ["sinter"] and selected[0].price_rs_per_mt == 100.0


def test_fuel_table_pins_rates_and_takes_the_operator_prices() -> None:
    table = pd.DataFrame(
        {
            "fuel_id": ["coke", "nut_coke", "pci"],
            "rate_kg_per_thm": [400.0, 92.0, 150.0],
            "price_rs_per_mt": [30000.0, 28000.0, 12000.0],
        }
    )
    rates = {"nut_coke_rate_kg_thm": 70.0, "pci_rate_kg_thm": 165.0}
    out = bmo_case.fuel_ash_with_rates(table, rates, {"pci": 13500.0})
    assert out["rate_kg_per_thm"].tolist() == [400.0, 70.0, 165.0]
    assert out["price_rs_per_mt"].tolist() == [30000.0, 28000.0, 13500.0]


def test_ore_overrides_and_unknown_ids() -> None:
    ores = [_ore("sinter", "Sinter"), _ore("lump", "Lump")]
    base = bmo_case.ore_editor_table(ores, ["sinter", "lump"], {})
    edited, unknown = bmo_case.with_ore_overrides(
        base, {"lump": {"price_rs_per_mt": 4500.0, "max_share_pct": 20.0, "stock_mt": None}, "gomti": {"stock_mt": 1.0}}
    )
    assert unknown == ["gomti"]
    lump = edited.set_index("ore_id").loc["lump"]
    assert (lump["price_rs_per_mt"], lump["max_share_pct"], lump["stock_mt"]) == (4500.0, 20.0, 1000.0)
    assert base.set_index("ore_id").loc["lump", "price_rs_per_mt"] == 100.0  # the table itself is not changed
    selected = bmo_case.selected_ores_from_editor(edited, ores)
    assert [(o.ore_id, o.price_rs_per_mt, o.max_share_pct) for o in selected][1] == ("lump", 4500.0, 20.0)
    assert bmo_case.with_ore_overrides(base, None) == (base, [])


def test_compose_and_run_read_no_clock() -> None:
    """Only ``live_sources`` reads the clock: a composition at a given ``CaseSources.now`` is reproducible."""

    tree = ast.parse(Path(bmo_case.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name != "live_sources":
            calls = {
                ast.unparse(n.func)
                for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            }
            assert not {c for c in calls if c.endswith((".now", ".today", ".utcnow"))}, node.name
