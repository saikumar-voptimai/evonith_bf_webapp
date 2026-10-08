"""Production Si model parity, clocks, deployment and BMO isolation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import ast
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from data.bmo.si_forecast_context import (
    LiveSiliconForecastSource,
    SiliconForecastSources,
    _offline_frame,
)
from furnace_data.influx.query import query_builder
from utils.bmo.si_forecast import deployments
from utils.bmo.si_forecast.features import BLAST_FLOW_COLUMN, build_features
from utils.bmo.si_forecast.ledger import SiliconForecastLedger
from utils.bmo.si_forecast.model import SiliconExtraTreesPredictor
from utils.bmo.si_forecast.service import SiliconForecastService, forecast_origin
from utils.bmo.si_forecast.training import TrainingSettings, train, write_bundle
from utils.bmo.snapshot import encode, results_state
from ui.bmo.si_forecast import path_figure, trust_figure, trust_summary

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "src/assets/models/bmo_si_forecast/20260924_initial"
FIXTURE = ROOT / "tests/fixtures/bmo_si_forecast"


@pytest.fixture(scope="module")
def raw_sources() -> SiliconForecastSources:
    kwargs = {"float_precision": "round_trip"}
    return SiliconForecastSources(
        online=pd.read_csv(FIXTURE / "online_10min.csv.gz", **kwargs),
        labs=pd.read_csv(FIXTURE / "raw_hm_slag.csv", **kwargs),
        charge=pd.read_csv(FIXTURE / "raw_charge.csv.gz", **kwargs),
        hourly_gate=pd.read_csv(FIXTURE / "hourly_gate_source.csv", **kwargs),
        fetched_at=pd.Timestamp("2026-10-07T00:00:00Z"),
    )


@pytest.fixture(scope="module")
def expected_predictions() -> pd.DataFrame:
    return pd.read_csv(FIXTURE / "recent_predictions.csv", float_precision="round_trip")


@pytest.fixture(scope="module")
def replay(raw_sources, expected_predictions):
    return build_features(
        raw_sources.online,
        raw_sources.labs,
        raw_sources.charge,
        raw_sources.hourly_gate,
        pd.to_datetime(expected_predictions["origin"], utc=True),
        BUNDLE,
    )


def test_raw_149_cast_replay_matches_published_features_and_predictions(
    replay, expected_predictions
) -> None:
    expected = pd.read_csv(
        FIXTURE / "recent_features.csv", float_precision="round_trip"
    )
    assert np.array_equal(
        replay.features.isna().to_numpy(), expected.isna().to_numpy()
    )
    assert np.nanmax(
        np.abs(replay.features.to_numpy() - expected.to_numpy())
    ) < 2e-12
    actual = SiliconExtraTreesPredictor(BUNDLE).predict_frame(replay.features)
    assert np.max(np.abs(actual - expected_predictions["prediction"])) < 1e-12
    assert replay.quality["eligible"].all()


def test_read_only_history_replay_populates_trend_without_writing_live_ledger(
    raw_sources, expected_predictions, tmp_path
) -> None:
    origins = pd.to_datetime(expected_predictions["origin"], utc=True)
    service = SiliconForecastService(
        bundle_dir=BUNDLE,
        storage_dir=tmp_path / "empty-live-ledger",
    )
    result = service.replay_history(
        raw_sources,
        origin_start=origins.min(),
        origin_end=origins.max(),
        available_by=(
            pd.to_datetime(raw_sources.labs["created_at"], utc=True).max()
            + pd.Timedelta(hours=1)
        ),
    )

    history = result.forecasts.sort_values("origin_at").reset_index(drop=True)
    assert len(history) > len(expected_predictions)
    assert history["history_kind"].eq("causal_replay").all()
    assert history["horizon_minutes"].eq(120).all()
    expected = expected_predictions.assign(origin_at=origins)[
        ["origin_at", "prediction"]
    ]
    comparable = history.merge(
        expected,
        on="origin_at",
        how="inner",
        suffixes=("_replay", "_published"),
    )
    assert len(comparable) == len(expected_predictions)
    assert np.max(
        np.abs(
            comparable["prediction_replay"].to_numpy()
            - comparable["prediction_published"].to_numpy()
        )
    ) < 1e-12
    assert not result.actuals.empty
    assert result.actuals["sample_at"].is_monotonic_increasing
    assert not (tmp_path / "empty-live-ledger" / "issued_forecasts.jsonl").exists()


def test_live_frame_ending_at_origin_minus_10_minutes_keeps_latest_bin(
    raw_sources, expected_predictions
) -> None:
    """The empty origin grid row must expose, not erase, the delayed last bin."""

    origin = pd.Timestamp(expected_predictions.iloc[-1]["origin"])
    cutoff = origin.tz_convert("Asia/Kolkata") - pd.Timedelta(minutes=10)
    online_clock = pd.to_datetime(raw_sources.online["time (IST)"], utc=True).dt.tz_convert(
        "Asia/Kolkata"
    )
    live_online = raw_sources.online.loc[online_clock.le(cutoff)].copy()
    built = build_features(
        live_online,
        raw_sources.labs,
        raw_sources.charge,
        raw_sources.hourly_gate,
        [origin],
        BUNDLE,
    )
    assert bool(built.quality.iloc[0]["latest_flow_present"])
    assert built.quality.iloc[0]["online_6h_fraction"] >= 0.8
    expected = pd.read_csv(
        FIXTURE / "recent_features.csv", float_precision="round_trip"
    ).iloc[-1]
    assert np.nanmax(np.abs(built.features.iloc[0].to_numpy() - expected.to_numpy())) < 2e-12


def test_inference_module_has_no_training_runtime_imports() -> None:
    source = (ROOT / "src/utils/bmo/si_forecast/model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    for forbidden in ("sklearn", "joblib", "torch", "xgboost"):
        assert forbidden not in imports


def test_origin_is_floored_in_ist_before_storage_in_utc() -> None:
    # 07:44 UTC is 13:14 IST. Correct five-minute origin is 13:10 = 07:40 UTC.
    assert forecast_origin("2026-10-08T07:44:00Z") == pd.Timestamp(
        "2026-10-08T07:40:00Z"
    )


def test_five_minute_clock_uses_only_si_available_at_issue(
    raw_sources, expected_predictions
) -> None:
    origin = pd.Timestamp(expected_predictions.iloc[40]["origin"]) + pd.Timedelta(
        minutes=5
    )
    template = raw_sources.labs.iloc[-1].copy()

    def sample(identifier: str, sample_at: pd.Timestamp, created_at: pd.Timestamp, value: float):
        row = template.copy()
        row["id"] = identifier
        row["lab_sample_id"] = identifier
        row["time (IST)"] = sample_at.isoformat()
        row["created_at"] = created_at.isoformat()
        row["chem_pct_si"] = value
        return row

    known = sample("known-1250", origin - pd.Timedelta(minutes=5), origin, 0.91)
    late = sample(
        "late-1448",
        origin - pd.Timedelta(minutes=2),
        origin + pd.Timedelta(minutes=5),
        1.73,
    )
    labs = pd.concat(
        [raw_sources.labs, known.to_frame().T, late.to_frame().T], ignore_index=True
    )
    features = build_features(
        raw_sources.online,
        labs,
        raw_sources.charge,
        raw_sources.hourly_gate,
        [origin],
        BUNDLE,
        clock_version=2,
    ).features.iloc[0]
    assert features["si_last"] == pytest.approx(0.91)

    late["created_at"] = origin.isoformat()
    labs = pd.concat(
        [raw_sources.labs, known.to_frame().T, late.to_frame().T], ignore_index=True
    )
    refreshed = build_features(
        raw_sources.online,
        labs,
        raw_sources.charge,
        raw_sources.hourly_gate,
        [origin],
        BUNDLE,
        clock_version=2,
    ).features.iloc[0]
    assert refreshed["si_last"] == pytest.approx(1.73)


def test_future_source_changes_cannot_change_a_fixed_origin(
    raw_sources, expected_predictions
) -> None:
    origin = pd.Timestamp(expected_predictions.iloc[40]["origin"])
    baseline = build_features(
        raw_sources.online,
        raw_sources.labs,
        raw_sources.charge,
        raw_sources.hourly_gate,
        [origin],
        BUNDLE,
    ).features.iloc[0]
    online = raw_sources.online.copy()
    future_online = pd.to_datetime(online["time (IST)"], utc=True).gt(origin)
    online.loc[future_online, online.columns != "time (IST)"] = 999999.0
    labs = raw_sources.labs.copy()
    future_labs = pd.to_datetime(labs["time (IST)"], utc=True).ge(origin)
    labs.loc[future_labs, "chem_pct_si"] = 2.99
    charge = raw_sources.charge.copy()
    future_charge = pd.to_datetime(charge["time (IST)"], utc=True).ge(origin)
    mass_columns = [column for column in charge if column.endswith("_mt")]
    charge.loc[future_charge, mass_columns] = 9999.0
    changed = build_features(
        online, labs, charge, raw_sources.hourly_gate, [origin], BUNDLE
    ).features.iloc[0]
    pd.testing.assert_series_equal(baseline, changed)


def test_charge_window_includes_t_minus_3h_and_excludes_t_minus_1h(
    raw_sources, expected_predictions
) -> None:
    origin = pd.Timestamp(expected_predictions.iloc[40]["origin"])
    empty = raw_sources.charge.iloc[:0].copy()

    def event(identifier: str, when: pd.Timestamp, huge: bool = False) -> dict:
        row = {column: np.nan for column in empty.columns}
        row.update(
            id=identifier,
            lab_sample_id=np.nan,
            created_at=(origin - pd.Timedelta(hours=4)).isoformat(),
            charge_no=identifier,
            **{"time (IST)": when.isoformat()},
        )
        row.update(
            coke_1_mt=1000.0 if huge else 10.0,
            nut_coke_1_mt=1000.0 if huge else 2.0,
            sinter_1_mt=1000.0 if huge else 24.0,
            pellet_1_mt=1000.0 if huge else 12.0,
        )
        return row

    charge = pd.DataFrame(
        [
            event("included", origin - pd.Timedelta(hours=3)),
            event("excluded", origin - pd.Timedelta(hours=1), huge=True),
        ],
        columns=empty.columns,
    )
    features = build_features(
        raw_sources.online,
        raw_sources.labs,
        charge,
        raw_sources.hourly_gate,
        [origin],
        BUNDLE,
    ).features.iloc[0]
    assert features["charge_1_3h_ore_coke_ratio"] == pytest.approx(3.0)
    assert features["charge_1_3h_pellet_mtph"] == pytest.approx(6.0)


def test_created_after_origin_lab_is_not_used(raw_sources, expected_predictions) -> None:
    origin = pd.Timestamp(expected_predictions.iloc[40]["origin"])
    base = build_features(
        raw_sources.online,
        raw_sources.labs,
        raw_sources.charge,
        raw_sources.hourly_gate,
        [origin],
        BUNDLE,
    ).features.iloc[0]["si_last"]
    delayed = raw_sources.labs.iloc[-1].copy()
    delayed["id"] = "delayed-test"
    delayed["lab_sample_id"] = "delayed-test"
    delayed["time (IST)"] = (origin - pd.Timedelta(minutes=30)).isoformat()
    delayed["created_at"] = (origin + pd.Timedelta(minutes=5)).isoformat()
    delayed["chem_pct_si"] = 2.5
    labs = pd.concat([raw_sources.labs, delayed.to_frame().T], ignore_index=True)
    value = build_features(
        raw_sources.online,
        labs,
        raw_sources.charge,
        raw_sources.hourly_gate,
        [origin],
        BUNDLE,
    ).features.iloc[0]["si_last"]
    assert value == base


def test_service_reports_exact_out_of_domain_violation(
    raw_sources, expected_predictions, tmp_path
) -> None:
    origin = pd.Timestamp(expected_predictions.iloc[40]["origin"])
    gate = raw_sources.hourly_gate.copy()
    expected_gate = origin.tz_convert("Asia/Kolkata") - pd.Timedelta(hours=1)
    gate_clock = pd.to_datetime(gate["time"], utc=True).dt.tz_convert("Asia/Kolkata")
    gate.loc[gate_clock.eq(expected_gate), "ACT. FUEL RATEKG/THM."] = 650.0
    service = SiliconForecastService(
        bundle_dir=BUNDLE, storage_dir=tmp_path / "ledger"
    )
    record = service.issue(
        replace(raw_sources, hourly_gate=gate),
        origin=origin,
        issued_at=origin,
        mode="forecast",
    )
    assert record["status"] == "warning"
    assert record["si_pct"] is not None
    assert "fuel_rate_outside" in record["reason_codes"]
    assert "650.0 kg/THM" in " ".join(record["reasons"])


def test_missing_latest_blast_flow_warns_but_keeps_prediction(
    raw_sources, expected_predictions, tmp_path
) -> None:
    origin = pd.Timestamp(expected_predictions.iloc[40]["origin"])
    online = raw_sources.online.copy()
    clock = pd.to_datetime(online["time (IST)"], utc=True)
    latest = origin - pd.Timedelta(minutes=10)
    online.loc[clock.eq(latest), BLAST_FLOW_COLUMN] = np.nan
    service = SiliconForecastService(
        bundle_dir=BUNDLE, storage_dir=tmp_path / "missing-flow-ledger"
    )
    record = service.issue(
        replace(raw_sources, online=online),
        origin=origin,
        issued_at=origin,
        mode="forecast",
    )
    assert record["status"] == "warning"
    assert record["si_pct"] is not None
    assert "latest_blast_flow_missing" in record["reason_codes"]
    assert "latest blast-volume reading" in " ".join(record["reasons"])


def test_model_absence_is_nonblocking(raw_sources, expected_predictions, tmp_path) -> None:
    origin = pd.Timestamp(expected_predictions.iloc[0]["origin"])
    service = SiliconForecastService(
        bundle_dir=tmp_path / "missing", storage_dir=tmp_path / "ledger"
    )
    record = service.issue(raw_sources, origin=origin, issued_at=origin)
    assert record["status"] == "model_unavailable"
    assert record["si_pct"] is None


def test_first_issued_forecast_is_immutable(tmp_path) -> None:
    ledger = SiliconForecastLedger(tmp_path)
    first = ledger.record_issue(
        {"origin_at": "2026-10-08T07:30:00Z", "status": "ok", "si_pct": 0.41}
    )
    second = ledger.record_issue(
        {"origin_at": "2026-10-08T07:30:00Z", "status": "ok", "si_pct": 0.99}
    )
    assert first == second
    assert second["si_pct"] == 0.41


def test_ledger_scores_each_horizon_without_rewriting_issue(tmp_path) -> None:
    ledger = SiliconForecastLedger(tmp_path)
    origin = pd.Timestamp("2026-10-08T07:30:00Z")
    issue = ledger.record_issue(
        {
            "origin_at": origin.isoformat(),
            "issued_at": origin.isoformat(),
            "status": "ok",
            "last_sample_si_pct": 0.40,
            "horizons": [
                {
                    "horizon_minutes": horizon,
                    "target_start": (origin + pd.Timedelta(minutes=horizon)).isoformat(),
                    "target_end": (
                        origin + pd.Timedelta(minutes=horizon + 5)
                    ).isoformat(),
                    "si_pct": 0.42 + horizon / 10000,
                    "lower_pct": 0.30,
                    "upper_pct": 0.60,
                }
                for horizon in (0, 60)
            ],
        }
    )
    for horizon in (0, 60):
        ledger.record_outcome(
            origin_at=origin,
            horizon_minutes=horizon,
            sample_id=f"sample-{horizon}",
            lab_sample_id=f"lab-{horizon}",
            sample_at=origin + pd.Timedelta(minutes=horizon + 2),
            available_at=origin + pd.Timedelta(minutes=horizon + 20),
            actual_si_pct=0.45,
            recorded_at=origin + pd.Timedelta(hours=2),
        )
    scored = ledger.scored_frame()
    assert set(scored["horizon_minutes"]) == {0, 60}
    assert scored["inside_range"].all()
    assert ledger.get(origin) == issue


def test_trend_keeps_irregular_samples_and_equal_forecast_spacing() -> None:
    origin = pd.Timestamp("2026-10-08T09:20:00Z")
    record = {
        "origin_at": origin.isoformat(),
        "issued_at": origin.isoformat(),
        "horizons": [
            {
                "horizon_minutes": horizon,
                "target_start": (origin + pd.Timedelta(minutes=horizon)).isoformat(),
                "target_end": (
                    origin + pd.Timedelta(minutes=horizon + 5)
                ).isoformat(),
                "si_pct": 0.40 + horizon / 10000,
                "lower_pct": 0.30,
                "upper_pct": 0.55,
            }
            for horizon in (0, 60, 120, 180)
        ],
    }
    actuals = pd.DataFrame(
        {
            "sample_at": [
                pd.Timestamp("2026-10-08T07:20:00Z"),
                pd.Timestamp("2026-10-08T09:18:00Z"),
            ],
            "actual": [0.39, 0.44],
        }
    )
    figure = path_figure(record, pd.DataFrame(), actuals)
    measured = next(trace for trace in figure.data if trace.name == "Measured Si")
    measured_clock = list(pd.to_datetime(measured.x))
    assert measured_clock == [
        pd.Timestamp("2026-10-08 12:50:00"),
        pd.Timestamp("2026-10-08 14:48:00"),
    ]
    assert all(stamp.tzinfo is None for stamp in measured_clock)
    forecast = next(trace for trace in figure.data if trace.name == "Latest forecast")
    forecast_clock = pd.Series(pd.to_datetime(forecast.x))
    assert forecast_clock.iloc[0] == pd.Timestamp("2026-10-08 14:50:00")
    assert forecast_clock.iloc[0].tzinfo is None
    spacing = forecast_clock.diff().dropna()
    assert spacing.eq(pd.Timedelta(hours=1)).all()

    scored = pd.DataFrame(
        {
            "horizon_minutes": [0],
            "actual": [0.44],
            "prediction": [0.42],
            "persistence": [0.39],
            "inside_range": [True],
        }
    )
    summary = trust_summary(scored, 0)
    assert summary.iloc[0]["Matched samples"] == 1
    assert summary.iloc[0]["Inside range"] == "100%"
    trust = trust_figure(
        pd.DataFrame(
            {
                "horizon_minutes": [0],
                "sample_at": [pd.Timestamp("2026-10-08T07:20:00Z")],
                "origin_at": [pd.Timestamp("2026-10-08T07:15:00Z")],
                "actual": [0.44],
                "prediction": [0.42],
                "status": ["ok"],
            }
        ),
        0,
    )
    assert pd.Timestamp(trust.data[0].x[0]) == pd.Timestamp("2026-10-08 12:50:00")
    assert pd.Timestamp(trust.data[0].x[0]).tzinfo is None


def test_trend_renders_causal_replay_when_live_ledger_is_empty() -> None:
    origin = pd.Timestamp("2026-10-08T09:20:00Z")
    record = {
        "origin_at": origin.isoformat(),
        "issued_at": origin.isoformat(),
        "horizons": [
            {
                "horizon_minutes": horizon,
                "target_start": (origin + pd.Timedelta(minutes=horizon)).isoformat(),
                "target_end": (
                    origin + pd.Timedelta(minutes=horizon + 5)
                ).isoformat(),
                "si_pct": 0.40 + horizon / 10000,
                "lower_pct": 0.30,
                "upper_pct": 0.55,
            }
            for horizon in (0, 60, 120, 180)
        ],
    }
    replay_rows = pd.DataFrame(
        {
            "origin_at": pd.to_datetime(
                ["2026-10-08T07:20:00Z", "2026-10-08T08:20:00Z"], utc=True
            ),
            "target_start": pd.to_datetime(
                ["2026-10-08T07:20:00Z", "2026-10-08T08:20:00Z"], utc=True
            ),
            "horizon_minutes": [0, 0],
            "prediction": [0.38, 0.41],
        }
    )
    actuals = pd.DataFrame(
        {
            "sample_at": pd.to_datetime(
                ["2026-10-08T07:31:00Z", "2026-10-08T09:07:00Z"], utc=True
            ),
            "actual": [0.39, 0.44],
        }
    )

    figure = path_figure(
        record,
        pd.DataFrame(),
        actuals,
        replay_forecasts=replay_rows,
        replay_horizon_minutes=0,
    )
    replay_trace = next(
        trace for trace in figure.data if trace.name == "Model replay (Now)"
    )
    assert list(replay_trace.y) == [0.38, 0.41]
    assert any(trace.name == "Measured Si" for trace in figure.data)
    assert any(trace.name == "Latest forecast" for trace in figure.data)


def test_offline_adapter_preserves_naive_ist_and_converts_aware_utc() -> None:
    naive = pd.DataFrame(
        {"si": [0.42], "created_at": [pd.Timestamp("2026-10-08 12:55")]},
        index=pd.Index([pd.Timestamp("2026-10-08 12:50")], name="time"),
    )
    naive_out = _offline_frame(naive)
    assert naive_out.loc[0, "time (IST)"] == pd.Timestamp(
        "2026-10-08 12:50", tz="Asia/Kolkata"
    )
    assert naive_out.loc[0, "created_at"] == pd.Timestamp(
        "2026-10-08 12:55", tz="Asia/Kolkata"
    )

    aware = pd.DataFrame(
        {"si": [0.42], "created_at": [pd.Timestamp("2026-10-08T07:25:00Z")]},
        index=pd.Index([pd.Timestamp("2026-10-08T07:20:00Z")], name="time"),
    )
    aware_out = _offline_frame(aware)
    assert aware_out.loc[0, "time (IST)"] == pd.Timestamp(
        "2026-10-08 12:50", tz="Asia/Kolkata"
    )
    assert aware_out.loc[0, "created_at"] == pd.Timestamp(
        "2026-10-08 12:55", tz="Asia/Kolkata"
    )


def test_retrain_exports_a_portable_candidate(raw_sources, expected_predictions, tmp_path) -> None:
    end = pd.to_datetime(expected_predictions["origin"], utc=True).max()
    result = train(
        raw_sources,
        base_bundle=BUNDLE,
        data_end_origin=end,
        settings=TrainingSettings(
            n_estimators=8,
            min_samples_leaf=2,
            min_train_samples=20,
            min_holdout_samples=20,
        ),
    )
    candidate = write_bundle(result, tmp_path / result.version, base_bundle=BUNDLE)
    predictor = SiliconExtraTreesPredictor(candidate)
    assert predictor.horizons == (0, 60, 120, 180)
    assert predictor.format_version == 2
    for horizon in predictor.horizons:
        rows = result.holdout[result.holdout["horizon_minutes"].eq(horizon)]
        portable = predictor.predict_frame(rows, horizon_minutes=horizon)
        assert np.max(np.abs(portable - rows["prediction"])) < 1e-12
        assert len(rows) >= 20
        assert (
            rows["sample_time"].ge(rows["target_start"])
            & rows["sample_time"].lt(rows["target_end"])
        ).all()
        assert (
            rows["target_end"] - rows["target_start"]
        ).eq(pd.Timedelta(minutes=5)).all()

    issue_origin = result.holdout.loc[
        result.holdout["horizon_minutes"].eq(0), "origin"
    ].iloc[0]
    service = SiliconForecastService(
        bundle_dir=candidate, storage_dir=tmp_path / "v2-ledger"
    )
    record = service.issue(
        raw_sources, origin=issue_origin, issued_at=issue_origin, mode="forecast"
    )
    assert [row["horizon_minutes"] for row in record["horizons"]] == [0, 60, 120, 180]
    assert all(row["si_pct"] is not None for row in record["horizons"])
    starts = pd.to_datetime([row["target_start"] for row in record["horizons"]], utc=True)
    assert list((starts - starts[0]).total_seconds() / 60) == [0.0, 60.0, 120.0, 180.0]


def test_activation_and_rollback_are_explicit(tmp_path) -> None:
    storage = tmp_path / "storage"
    candidate = storage / "models/candidate"
    shutil.copytree(BUNDLE, candidate)
    deployments.activate(
        storage,
        "candidate",
        default_bundle=BUNDLE,
        accepted_by="supervisor",
        now=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )
    assert deployments.active_bundle(storage, BUNDLE) == candidate
    deployments.activate(
        storage,
        BUNDLE.name,
        default_bundle=BUNDLE,
        accepted_by="supervisor",
        now=datetime(2026, 10, 8, 1, tzinfo=timezone.utc),
        note="roll back",
    )
    assert deployments.active_bundle(storage, BUNDLE) == BUNDLE


def test_snapshot_restores_new_result_without_affecting_old_schema() -> None:
    record = {
        "status": "ok",
        "si_pct": 0.43,
        "origin_at": "2026-10-08T07:30:00Z",
        "horizons": [
            {
                "horizon_minutes": 0,
                "target_start": "2026-10-08T07:30:00Z",
                "target_end": "2026-10-08T07:35:00Z",
                "si_pct": 0.43,
                "lower_pct": 0.31,
                "upper_pct": 0.55,
            }
        ],
    }
    snapshot = {
        "schema": "bmo-snapshot/2",
        "inputs": {},
        "results": {"si_furnace_forecast": encode(record)},
        "frozen": {"provider_calls": {"dummy()": encode({})}},
    }
    restored = results_state(snapshot, prefix="testbmo_")
    assert restored["testbmo_si_furnace_forecast"] == record
    old = {**snapshot, "results": {}}
    assert results_state(old, prefix="testbmo_") == {}


def test_live_adapter_requests_only_the_51_versioned_influx_fields() -> None:
    seen: dict = {}

    def online_fetch(**kwargs):
        seen.update(kwargs)
        fields = [field for values in kwargs["fields_by_measurement"].values() for field in values]
        index = pd.date_range("2026-10-08 12:00", periods=7, freq="10min", tz="Asia/Kolkata")
        return pd.DataFrame({field: 1.0 for field in fields}, index=index)

    source = LiveSiliconForecastSource(
        bundle_dir=BUNDLE,
        static_dataset_path=ROOT / "src/assets/data/furnace_dataset.csv",
        online_fetch=online_fetch,
    )
    source._online(
        pd.Timestamp("2026-10-08 13:00", tz="Asia/Kolkata"),
        pd.Timestamp("2026-10-08 13:00", tz="Asia/Kolkata"),
    )
    selected = [
        field for values in seen["fields_by_measurement"].values() for field in values
    ]
    assert len(selected) == 51
    assert len(seen["selected_measurements"]) == 5


def test_influx_query_field_filter_is_backward_compatible() -> None:
    start = datetime(2026, 10, 8, tzinfo=timezone.utc)
    field = "hot_blast_vol_nm3h"
    selected = query_builder(
        "process_params", start, start + pd.Timedelta(hours=1),
        type="windowed-average", window_by="10m", fields=[field]
    )
    assert f"MEAN({field}) AS {field}" in selected
    assert "top_temp_1" not in selected
    legacy = query_builder(
        "process_params", start, start + pd.Timedelta(hours=1), type="ts"
    )
    assert "SELECT * FROM process_params" in legacy


def test_new_forecast_replaces_legacy_blend_si_in_bmo() -> None:
    service = (ROOT / "src/utils/bmo/si_forecast/service.py").read_text(encoding="utf-8")
    assert "coke_correction" not in service
    assert "charged_coke" not in service
    page = (ROOT / "src/custom_pages/9_Blend_Optimizer.py").read_text(encoding="utf-8")
    assert "SiPredictionService" not in page
    assert "_predict_blend_si" not in page
    assert 'st.session_state["bmo_lp_si"]' not in page
    assert 'st.session_state["bmo_de_si"]' not in page
    assert 'st.session_state["bmo_manual_si"]' not in page
    assert 'st.session_state["bmo_si_furnace_forecast"] = record' in page
    assert "hot_metal_si_pct=live_si_pct" in page

    accuracy = (ROOT / "src/ui/bmo/model_accuracy.py").read_text(encoding="utf-8")
    assert "Legacy proposed-blend Si model" not in accuracy
    assert "render_si_accuracy" not in accuracy

    settings = (ROOT / "src/config/setting_bmo.yml").read_text(encoding="utf-8")
    assert "si_model_bundle:" not in settings
    assert "hot_metal_si:\n        enabled: false" in settings


def test_seven_day_trend_plots_continuous_model_and_irregular_actuals() -> None:
    forecasts = pd.DataFrame(
        {
            "horizon_minutes": [120, 120, 120],
            "target_start": pd.to_datetime(
                [
                    "2026-10-01T06:30:00Z",
                    "2026-10-01T07:30:00Z",
                    "2026-10-01T08:30:00Z",
                ],
                utc=True,
            ),
            "prediction": [0.35, 0.37, 0.36],
            "status": ["ok", "ok", "warning"],
        }
    )
    actuals = pd.DataFrame(
        {
            "sample_at": pd.to_datetime(
                ["2026-10-01T06:49:00Z", "2026-10-01T08:18:00Z"], utc=True
            ),
            "actual": [0.42, 0.39],
        }
    )
    scored = pd.DataFrame(
        columns=["horizon_minutes", "sample_at", "actual", "prediction", "status"]
    )

    figure = trust_figure(
        scored,
        120,
        forecasts=forecasts,
        actuals=actuals,
    )

    measured = next(trace for trace in figure.data if trace.name == "Measured Si")
    model = next(trace for trace in figure.data if trace.name == "Model")
    assert list(model.y) == [0.35, 0.37, 0.36]
    assert list(measured.y) == [0.42, 0.39]
    assert pd.Timestamp(model.x[0]) == pd.Timestamp("2026-10-01 12:00:00")
    assert pd.Timestamp(measured.x[0]) == pd.Timestamp("2026-10-01 12:19:00")

    page = (ROOT / "src/custom_pages/9_Blend_Optimizer.py").read_text(encoding="utf-8")
    assert "start = end - pd.Timedelta(days=7)" in page
