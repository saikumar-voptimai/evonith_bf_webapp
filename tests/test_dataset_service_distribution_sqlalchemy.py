"""Regression tests for DatasetService burden-distribution repository path."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

import furnace_data.dataset.service as service_module
from furnace_data.dataset.service import DatasetService


class FakeEngine:
    disposed = False

    def dispose(self):
        self.disposed = True


class FakeRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def fetch_distribution_frame(self, *, start_date, end_date):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-02T00:00:00Z"),
            ],
            name="time",
        )
        return pd.DataFrame(
            {
                "coke_p01_rings": [4.0, 4.0],
                "coke_p01_angles": [30.0, 35.0],
                "burden_changing_purpose": [None, "stability"],
            },
            index=idx,
        )


def test_fetch_distribution_data_returns_expected_windowed_rows(monkeypatch) -> None:
    """DatasetService should expand burden rows from the repository."""
    engine = FakeEngine()
    monkeypatch.setattr(DatasetService, "_get_engine", lambda self: engine)
    monkeypatch.setattr(
        service_module,
        "build_relational_session_factory",
        lambda engine: object(),
    )
    monkeypatch.setattr(service_module, "BurdenHistoryRepository", FakeRepository)

    output = DatasetService().fetch_distribution_data(
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    assert engine.disposed is True
    assert not output.empty
    assert "coke_p01_rings" in output.columns
    assert "coke_p01_angles" in output.columns
    assert "burden_changing_purpose" in output.columns
    assert "total_coke_portions" in output.columns
    assert "weighted_coke_angle" in output.columns

    day = output.loc[datetime(2026, 1, 2)]
    assert float(day["coke_p01_rings"]) == 4.0
    assert float(day["coke_p01_angles"]) == 35.0
    assert str(day["burden_changing_purpose"]) == "stability"
    assert float(day["total_coke_portions"]) == 4.0
    assert float(day["weighted_coke_angle"]) == 35.0


def test_fetch_distribution_data_propagates_engine_errors(monkeypatch) -> None:
    """No silent empty-frame fallback when PostgreSQL configuration is invalid."""
    monkeypatch.setattr(
        DatasetService,
        "_get_engine",
        lambda self: (_ for _ in ()).throw(ValueError("Shared relational persistence requires PostgreSQL.")),
    )

    try:
        DatasetService().fetch_distribution_data(
            start_date=date(2026, 3, 15),
            end_date=date(2026, 3, 15),
        )
    except ValueError as exc:
        assert "PostgreSQL" in str(exc)
    else:
        raise AssertionError("Expected invalid PostgreSQL configuration to propagate")
