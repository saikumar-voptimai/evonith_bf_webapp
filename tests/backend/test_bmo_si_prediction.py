"""Tests for the standalone hot-metal Si prediction service.

``SiPredictionService`` assembles the Si model's feature vector from a BMO blend
payload plus context (previous-cast Si, calendar) and runs the bundled XGBoost
model. These tests stub the model bundle so the feature-assembly contract is
verified without loading real artifacts or hitting the database.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from utils.bmo.si_prediction import SiPredictionService, _base_feature_name
from utils.bmo.types import OreChemistry, OreInput


class TestBaseFeatureName:
    @pytest.mark.parametrize(
        "feature,expected",
        [
            ("HOT BLAST VOLUMENM3/HR.__lag2h", "HOT BLAST VOLUMENM3/HR."),
            ("CHEM_PCT_SI__lag3h", "CHEM_PCT_SI"),
            ("COKE_VM%", "COKE_VM%"),
            ("SINTER_CALC_THM_lag4", "SINTER_CALC_THM"),
            ("ORE_1_PCT_lag1_(GasImpact)", "ORE_1_PCT"),
        ],
    )
    def test_strips_lag_suffix(self, feature, expected):
        assert _base_feature_name(feature) == expected


class _FakeScaler:
    """Identity scaler so the model sees the raw assembled feature values."""

    def transform(self, frame):
        return frame.to_numpy()


class _FakeModel:
    def __init__(self) -> None:
        self.seen: pd.DataFrame | None = None

    def predict(self, frame):
        self.seen = frame
        return np.asarray([0.42])


class _FakeBundle:
    def __init__(self, features: list[str]) -> None:
        self.expected_features = features
        self.scaler = _FakeScaler()
        self.model = _FakeModel()


_FEATURES = [
    "COKE_VM%",
    "ORE_FE(T)%",
    "CHEM_PCT_SI__lag3h",
    "CHEM_PCT_SI__lag4h",
    "HOT BLAST VOLUMENM3/HR.__lag2h",
    "month",
    "week_of_year",
    "day_of_year",
    "UNRESOLVED_FEATURE",
]


def _service_with_stub() -> tuple[SiPredictionService, _FakeBundle]:
    svc = SiPredictionService(bundle_cfg={})
    bundle = _FakeBundle(_FEATURES)
    # Pre-populate the lazy-load state so _ensure_loaded() short-circuits.
    svc._bundle = bundle
    svc._features = list(_FEATURES)
    svc._means = {"UNRESOLVED_FEATURE": 1.23}
    return svc, bundle


def _ores() -> list[OreInput]:
    return [
        OreInput(
            ore_id="ore_a",
            display_name="Ore A",
            stock_mt=1000.0,
            price_rs_per_mt=100.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=62.0, sio2_pct=4.0),
        ),
        OreInput(
            ore_id="ore_b",
            display_name="Ore B",
            stock_mt=1000.0,
            price_rs_per_mt=120.0,
            min_share_pct=0.0,
            max_share_pct=100.0,
            chemistry=OreChemistry(fe_t_pct=58.0, sio2_pct=6.0),
        ),
    ]


class TestPredictBlendSi:
    def test_returns_model_value(self):
        svc, _ = _service_with_stub()
        value = svc.predict_blend_si(
            ores=_ores(),
            quantities_mt={"ore_a": 100.0, "ore_b": 50.0},
            process_context={"COKE_VM%": 6.5},
            prev_si=0.45,
            now=datetime(2026, 6, 15),
        )
        assert value == pytest.approx(0.42)

    def test_feature_vector_resolution(self):
        svc, bundle = _service_with_stub()
        svc.predict_blend_si(
            ores=_ores(),
            quantities_mt={"ore_a": 100.0, "ore_b": 50.0},
            process_context={"COKE_VM%": 6.5},
            prev_si=0.45,
            now=datetime(2026, 6, 15),
        )
        seen = bundle.model.seen
        assert list(seen.columns) == _FEATURES
        row = seen.iloc[0]
        # Previous-cast Si feeds both CHEM_PCT_SI lag features.
        assert row["CHEM_PCT_SI__lag3h"] == pytest.approx(0.45)
        assert row["CHEM_PCT_SI__lag4h"] == pytest.approx(0.45)
        # Calendar features come from `now`.
        assert row["month"] == 6
        assert row["day_of_year"] == datetime(2026, 6, 15).timetuple().tm_yday
        # Process-context value flows through.
        assert row["COKE_VM%"] == pytest.approx(6.5)
        # Unresolved feature falls back to the scaler mean default.
        assert row["UNRESOLVED_FEATURE"] == pytest.approx(1.23)

    def test_prev_si_none_uses_default(self):
        svc, bundle = _service_with_stub()
        svc._means = {"CHEM_PCT_SI__lag3h": 0.5, "CHEM_PCT_SI__lag4h": 0.5}
        svc.predict_blend_si(
            ores=_ores(),
            quantities_mt={"ore_a": 100.0, "ore_b": 50.0},
            process_context=None,
            prev_si=None,
            now=datetime(2026, 6, 15),
        )
        row = bundle.model.seen.iloc[0]
        assert row["CHEM_PCT_SI__lag3h"] == pytest.approx(0.5)

    def test_missing_model_returns_none(self):
        svc = SiPredictionService(bundle_cfg={})
        svc._bundle = _FakeBundle(_FEATURES)
        svc._bundle.model = None  # model artifact unavailable
        svc._features = list(_FEATURES)
        result = svc.predict_blend_si(
            ores=_ores(),
            quantities_mt={"ore_a": 100.0},
            process_context=None,
        )
        assert result is None

    def test_get_status(self):
        svc, _ = _service_with_stub()
        status = svc.get_status()
        assert status["model_loaded"] is True
        assert status["scaler_loaded"] is True
        assert status["feature_count"] == len(_FEATURES)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
