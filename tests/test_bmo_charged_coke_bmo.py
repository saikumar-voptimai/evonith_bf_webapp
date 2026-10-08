"""The charged engine as the BMO Data-Driven candidate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo.charged_coke.bmo import charged_coke_prediction, response_settings
from utils.bmo.charged_coke.ledger import PlannedChangeRegister
from utils.bmo.charged_coke.service import ChargedCokeEngine
from utils.bmo.coke_correction import (
    TERM_SLAG_HEAT,
    CokeCorrectionDrivers,
    CokeCorrectionReference,
    compute_coke_correction,
    load_coke_correction_settings,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "charged_coke"


@pytest.fixture(scope="module")
def engine() -> ChargedCokeEngine:
    return ChargedCokeEngine.load()


@pytest.fixture(scope="module")
def source() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "source_window.csv.gz", parse_dates=["time"]).set_index("time")


@pytest.fixture(scope="module")
def base_settings():
    import yaml

    cfg = yaml.safe_load((ROOT / "src/config/setting_bmo.yml").read_text(encoding="utf-8"))["bmo"]
    return load_coke_correction_settings(cfg)


@pytest.mark.parametrize(("mode", "coefficient"), [("fitted", 0.020452), ("engineering", 0.22)])
def test_blend_response_is_one_linear_slag_term_from_the_current_burden(mode, coefficient, engine, base_settings) -> None:
    settings = response_settings(base_settings, mode=mode, fitted_beta=engine.model.beta)
    reference = CokeCorrectionReference(slag_rate_kg_per_thm=320.0, flux_co2_kg_per_thm=5.0, hot_metal_si_pct=0.6)

    for blend_slag in (320.0, 360.0, 260.0):
        drivers = CokeCorrectionDrivers(slag_rate_kg_per_thm=blend_slag, flux_co2_kg_per_thm=40.0, hot_metal_si_pct=1.2)
        result = compute_coke_correction(
            anchor_coke_rate_kg_thm=318.0, anchor_nut_coke_rate_kg_thm=70.0, anchor_pci_rate_kg_thm=190.0,
            drivers=drivers, reference=reference, settings=settings,
        )
        # No flux or Si term, no saturation: exactly level + b x dSlag.
        assert result.corrected_coke_rate_kg_thm == pytest.approx(318.0 + coefficient * (blend_slag - 320.0), abs=1e-4)
    assert set(settings.terms) == {TERM_SLAG_HEAT}
    assert settings.term(TERM_SLAG_HEAT).reference_source == "model_current"
    assert mode in settings.term(TERM_SLAG_HEAT).k_config_units


def test_configured_blend_response_overrides_only_the_slag_term(
    engine, base_settings
) -> None:
    settings = response_settings(base_settings, coefficient=0.10)
    reference = CokeCorrectionReference(
        slag_rate_kg_per_thm=320.0,
        flux_co2_kg_per_thm=5.0,
        hot_metal_si_pct=0.6,
    )
    result = compute_coke_correction(
        anchor_coke_rate_kg_thm=318.0,
        anchor_nut_coke_rate_kg_thm=70.0,
        anchor_pci_rate_kg_thm=190.0,
        drivers=CokeCorrectionDrivers(
            slag_rate_kg_per_thm=360.0,
            flux_co2_kg_per_thm=40.0,
            hot_metal_si_pct=1.2,
        ),
        reference=reference,
        settings=settings,
    )

    assert result.corrected_coke_rate_kg_thm == pytest.approx(322.0)
    assert settings.term(TERM_SLAG_HEAT).k_config_units.endswith("(configured)")


def test_a_paused_forecast_gives_no_data_driven_level(tmp_path, source, engine) -> None:
    row = pd.Timestamp("2026-09-28 13:00")  # large deviation in the replay
    published = source.loc[: row + pd.Timedelta(hours=1)]
    built = row + pd.Timedelta(hours=1, minutes=6)

    prediction, detail = charged_coke_prediction(published, engine, now=built, storage_dir=tmp_path,
                                                 mode="fitted", built_at=built)

    assert detail["record"]["state"] == "large_hold"
    assert not prediction.usable
    assert prediction.value_kg_per_thm is None
    assert prediction.reasons


def test_force_predict_exposes_withheld_value_and_keeps_exact_violations(
    tmp_path, source, engine
) -> None:
    row = pd.Timestamp("2026-09-28 13:00")  # large deviation in the replay
    published = source.loc[: row + pd.Timedelta(hours=1)]
    built = row + pd.Timedelta(hours=1, minutes=6)

    held, held_detail = charged_coke_prediction(
        published,
        engine,
        now=built,
        storage_dir=tmp_path,
        mode="fitted",
        built_at=built,
    )
    forced, forced_detail = charged_coke_prediction(
        published,
        engine,
        now=built,
        storage_dir=tmp_path,
        mode="fitted",
        built_at=built,
        force_predict=True,
    )

    assert not held.usable
    assert forced.usable
    assert forced_detail["forced"]
    assert forced.value_kg_per_thm == pytest.approx(
        held_detail["record"]["withheld_value_kg_thm"]
    )
    assert forced.reasons == tuple(held_detail["record"]["reasons"])
    assert forced_detail["record"]["state"] == "large_hold"
    assert forced_detail["record"]["lower_kg_thm"] is None
    assert all(
        point["prediction_kg_thm"] == point["withheld_value_kg_thm"]
        for point in forced_detail["record"]["path"]
    )
    # Force-predict is a display/use override, not a second issued forecast.
    assert len((tmp_path / "issued_forecasts.jsonl").read_text().strip().splitlines()) == 1


def test_operator_override_is_applied_once_on_the_issued_level(tmp_path, source, engine) -> None:
    row = pd.Timestamp("2026-09-21 09:00")
    published = source.loc[: row + pd.Timedelta(hours=1)]
    built = row + pd.Timedelta(hours=1, minutes=6)

    live, detail = charged_coke_prediction(published, engine, now=built, storage_dir=tmp_path, mode="fitted", built_at=built)
    pci = detail["record"]["controls"]["pci"]
    more, more_detail = charged_coke_prediction(published, engine, now=built, storage_dir=tmp_path, mode="fitted",
                                                built_at=built, pci_override=pci + 10)

    assert live.usable and more.usable
    assert live.value_kg_per_thm == pytest.approx(318.1162, abs=1e-3)
    effects = more_detail["override"]["effects"]
    assert more.value_kg_per_thm == pytest.approx(live.value_kg_per_thm + sum(effects.values()), abs=1e-9)
    assert more.value_kg_per_thm == pytest.approx(live.value_kg_per_thm - 4.50, abs=0.01)
    # The same row was not issued twice.
    assert len((tmp_path / "issued_forecasts.jsonl").read_text().strip().splitlines()) == 1


def test_a_planned_change_pauses_the_bmo_level(tmp_path, source, engine) -> None:
    row = pd.Timestamp("2026-09-21 09:00")
    published = source.loc[: row + pd.Timedelta(hours=1)]
    built = row + pd.Timedelta(hours=1, minutes=6)
    PlannedChangeRegister(tmp_path).declare(reason="Planned blast reduction", valid_from=built - pd.Timedelta(minutes=10),
                                            valid_until=built + pd.Timedelta(hours=4), created_by="shift", now=built)

    prediction, detail = charged_coke_prediction(published, engine, now=built, storage_dir=tmp_path, mode="fitted", built_at=built)

    assert detail["record"]["state"] == "planned_change_hold"
    assert not prediction.usable


def test_page_uses_only_the_charged_engine() -> None:
    import yaml

    cfg = yaml.safe_load((ROOT / "src/config/setting_bmo.yml").read_text(encoding="utf-8"))["bmo"]["data_driven_coke"]
    page = (ROOT / "src/custom_pages/9_Blend_Optimizer.py").read_text(encoding="utf-8")

    assert "engine" not in cfg
    assert cfg["charged_4h"]["blend_response_kg_coke_per_kg_slag"] == pytest.approx(
        0.10
    )
    assert "Robust 24-h model" not in page
    assert "DirectCokeModelService" not in page
    assert "direct_coke_model" not in page
    # One switch point: the charged response replaces the physics correction.
    assert page.count("charged_response_settings(") == 1
