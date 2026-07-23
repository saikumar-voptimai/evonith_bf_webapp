"""Focused unit tests for the canonical static dataset service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from apps.backend_api.app.api.v1.schemas.datasets import (
    RegressionRequest,
    ScatterAnalysisRequest,
)
from apps.backend_api.app.core.errors import ApiError


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fuel_rate": [500.0, 510.0, 520.0],
            "production_per_hour": [100.0, 110.0, 120.0],
            "eta_co": [40.0, 41.0, 42.0],
            "coke_rate": [390.0, 391.0, 392.0],
            "actual_kg_thm": [85.0, 86.0, 87.0],
        },
        index=pd.DatetimeIndex(
            ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T02:00:00Z"],
            name="timestamp",
        ),
    )


def _context(frame: pd.DataFrame, version: str = "canonical-v1"):
    from apps.backend_api.app.services import dataset_service

    canonical, labels = dataset_service._canonicalise_dataframe(frame)
    return canonical, labels, None, version


def test_metadata_does_not_run_validation_on_page_open(monkeypatch):
    from apps.backend_api.app.services import dataset_service

    frame = _frame()
    monkeypatch.setattr(dataset_service, "_dataset_context", lambda: _context(frame))
    monkeypatch.setattr(dataset_service, "_make_manager", lambda: SimpleNamespace(get_meta=lambda: None))
    monkeypatch.setattr(dataset_service, "_stored_validation_status", lambda _version: "not_run")
    monkeypatch.setattr(
        dataset_service,
        "_validation_for_frame",
        lambda *_args: (_ for _ in ()).throw(AssertionError("validation must be explicit")),
    )

    metadata = dataset_service.get_static_metadata()

    assert metadata.validation_status == "not_run"
    assert "unit_cost_lakhs_per_thm" in {column.id for column in metadata.columns}


def test_scatter_rejects_fewer_than_two_distinct_numeric_rows(monkeypatch):
    from apps.backend_api.app.services import dataset_service

    frame = _frame().iloc[:1]
    monkeypatch.setattr(dataset_service, "_dataset_context", lambda: _context(frame))

    request = ScatterAnalysisRequest(
        dataset_version="canonical-v1",
        x_field="fuel_rate",
        y_field="production_per_hour",
        regression=RegressionRequest(enabled=False),
    )

    with pytest.raises(ApiError) as exc_info:
        dataset_service.get_scatter_analysis(request)

    assert exc_info.value.code == "INSUFFICIENT_REGRESSION_DATA"
    assert exc_info.value.status_code == 422


def test_current_download_serializes_canonical_public_columns(monkeypatch, tmp_path):
    from apps.backend_api.app.services import dataset_service

    source = tmp_path / "raw.csv"
    source.write_text("raw source", encoding="utf-8")
    raw = pd.DataFrame(
        {
            "body_etaco": [40.0, 41.0],
            "COKE RATE KG/THM": [390.0, 391.0],
            "PCI_KG/THM": [85.0, 86.0],
        },
        index=pd.DatetimeIndex(["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], name="timestamp"),
    )
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(dataset_service, "_current_dataset_path", lambda: Path(source))
    monkeypatch.setattr(dataset_service, "load_static_dataset_dataframe", lambda: raw.copy())

    path, version = dataset_service.current_dataset_download()
    downloaded = pd.read_csv(path)

    assert path.exists()
    assert version == dataset_service._version_from_path(source, dataset_service._canonicalise_dataframe(raw)[0])
    assert "eta_co" in downloaded.columns
    assert "unit_cost_lakhs_per_thm" in downloaded.columns
    assert "body_etaco" not in downloaded.columns

def test_dataset_version_tracks_canonical_values_not_raw_file_bytes(tmp_path):
    from apps.backend_api.app.services import dataset_service

    source = tmp_path / "raw.csv"
    canonical = _frame()
    source.write_text("first raw representation", encoding="utf-8")
    first = dataset_service._version_from_path(source, canonical)
    source.write_text("second raw representation", encoding="utf-8")
    second = dataset_service._version_from_path(source, canonical)
    changed = canonical.copy()
    changed.loc[changed.index[0], "fuel_rate"] = 999.0

    assert second == first
    assert dataset_service._version_from_path(source, changed) != first