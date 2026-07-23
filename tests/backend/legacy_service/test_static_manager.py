"""
Tests for StaticDatasetManager â€” the lag-aware cache logic.

No InfluxDB or PostgreSQL calls. MlDatasetFetcher is fully mocked.
"""
import json
import re
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import furnace_data.dataset.static as static_module
import pytest

from furnace_data.dataset.static import (
    CacheMeta,
    StaticDatasetManager,
    _load_meta,
    _rotate_csvs,
    _save_meta,
    _versioned_filename,
)


# ---------------------------------------------------------------------------
# CacheMeta
# ---------------------------------------------------------------------------

class TestCacheMeta:
    def test_date_properties_parsed(self):
        meta = CacheMeta(
            confirmed_end="2026-03-28",
            raw_end="2026-03-31",
            data_start="2023-01-01",
        )
        assert meta.confirmed_end_date == date(2026, 3, 28)
        assert meta.raw_end_date == date(2026, 3, 31)
        assert meta.data_start_date == date(2023, 1, 1)

    def test_empty_strings_return_none(self):
        meta = CacheMeta()
        assert meta.confirmed_end_date is None
        assert meta.raw_end_date is None

    def test_round_trips_json(self, tmp_path):
        meta = CacheMeta(
            rm_choice="charge",
            data_start="2023-01-01",
            confirmed_end="2026-03-28",
            raw_end="2026-03-31",
            rows=1000,
            columns=50,
            csv_file="furnace_dataset_20260401.csv",
        )
        _save_meta(tmp_path, meta)
        loaded = _load_meta(tmp_path)
        assert loaded.confirmed_end == meta.confirmed_end
        assert loaded.rows == meta.rows
        assert loaded.csv_file == meta.csv_file

    def test_load_meta_missing_file_returns_none(self, tmp_path):
        assert _load_meta(tmp_path) is None

    def test_load_meta_corrupt_file_returns_none(self, tmp_path):
        (tmp_path / "cache_meta.json").write_text("not json")
        assert _load_meta(tmp_path) is None


# ---------------------------------------------------------------------------
# CSV rotation
# ---------------------------------------------------------------------------

class TestCsvRotation:
    def test_rotates_to_max_versions(self, tmp_path):
        # Create 5 CSV files with different mtimes
        for i in range(5):
            f = tmp_path / f"furnace_dataset_202601{i:02d}_000000.csv"
            f.write_text("a,b\n1,2")
            import time; time.sleep(0.01)   # ensure distinct mtimes

        _rotate_csvs(tmp_path, keep=3)
        remaining = list(tmp_path.glob("furnace_dataset_*.csv"))
        assert len(remaining) == 3

    def test_no_deletion_when_under_limit(self, tmp_path):
        for i in range(2):
            (tmp_path / f"furnace_dataset_2026010{i}_000000.csv").write_text("a")
        _rotate_csvs(tmp_path, keep=3)
        assert len(list(tmp_path.glob("furnace_dataset_*.csv"))) == 2

    def test_versioned_filename_format(self):
        fn = _versioned_filename()
        assert re.fullmatch(r"furnace_dataset_\d{8}_\d{6}_\d{6}\.csv", fn)


# ---------------------------------------------------------------------------
# _compute_fetch_window
# ---------------------------------------------------------------------------

class TestComputeFetchWindow:
    LAG = 3

    def _manager(self, tmp_path):
        mgr = StaticDatasetManager.__new__(StaticDatasetManager)
        mgr.static_dir = tmp_path
        mgr.offline_lag_days = self.LAG
        mgr.max_versions = 3
        mgr.legacy_csv_path = None
        return mgr

    def test_empty_df_returns_default_start(self, tmp_path):
        mgr = self._manager(tmp_path)
        fetch_start, fetch_end, frozen = mgr._compute_fetch_window(
            pd.DataFrame(), None, date.today()
        )
        assert fetch_start == StaticDatasetManager._DEFAULT_START
        assert fetch_end == date.today()
        assert frozen.empty

    def test_normal_incremental_returns_confirmed_end(self, tmp_path):
        mgr = self._manager(tmp_path)
        today = date(2026, 4, 1)
        raw_end = date(2026, 3, 31)
        confirmed_end = raw_end - timedelta(days=self.LAG)  # Mar 28

        meta = CacheMeta(raw_end=raw_end.isoformat(), confirmed_end=confirmed_end.isoformat())
        idx = pd.date_range("2026-01-01", periods=5, freq="7D")
        existing = pd.DataFrame({"col": range(5)}, index=idx)

        fetch_start, fetch_end, frozen = mgr._compute_fetch_window(existing, meta, today)

        assert fetch_start == confirmed_end
        assert fetch_end == today
        # Frozen rows are those up to and including confirmed_end
        assert len(frozen) <= len(existing)

    def test_up_to_date_returns_none(self, tmp_path):
        mgr = self._manager(tmp_path)
        today = date(2026, 4, 1)
        # raw_end = today, so confirmed_end = today - lag
        raw_end = today
        confirmed_end = raw_end - timedelta(days=self.LAG)

        meta = CacheMeta(raw_end=raw_end.isoformat(), confirmed_end=confirmed_end.isoformat())
        existing = pd.DataFrame({"col": [1]}, index=pd.date_range(today, periods=1))

        fetch_start, _, _ = mgr._compute_fetch_window(existing, meta, today)
        assert fetch_start is None

    def test_no_meta_falls_back_to_df_max_date(self, tmp_path):
        mgr = self._manager(tmp_path)
        today = date(2026, 4, 1)
        idx = pd.date_range("2026-03-25", periods=3, freq="D")
        existing = pd.DataFrame({"col": range(3)}, index=idx)

        fetch_start, fetch_end, frozen = mgr._compute_fetch_window(existing, None, today)
        # raw_end = max date in df = Mar 27, confirmed_end = Mar 27 - 3 = Mar 24
        expected_confirmed = date(2026, 3, 27) - timedelta(days=self.LAG)
        assert fetch_start == expected_confirmed
        assert fetch_end == today


# ---------------------------------------------------------------------------
# save() â€” CSV written and meta updated
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_creates_versioned_csv_and_meta(self, tmp_path, sample_ml_df):
        mgr = StaticDatasetManager.__new__(StaticDatasetManager)
        mgr.static_dir = tmp_path
        mgr.offline_lag_days = 3
        mgr.max_versions = 3

        path = mgr.save(sample_ml_df, "RM Charge")

        assert path.exists()
        assert path.name.startswith("furnace_dataset_")
        meta = _load_meta(tmp_path)
        assert meta is not None
        assert meta.rows == len(sample_ml_df)
        assert meta.csv_file == path.name
        assert meta.rm_choice == "charge"
        assert meta.data_start == sample_ml_df.index.min().date().isoformat()
        assert meta.raw_end == sample_ml_df.index.max().date().isoformat()
        assert meta.confirmed_end == (
            sample_ml_df.index.max().date() - timedelta(days=mgr.offline_lag_days)
        ).isoformat()
        assert "+00:00" in meta.last_updated

    def test_save_preserves_old_metadata_pointer_when_csv_promotion_fails(
        self, monkeypatch, tmp_path, sample_ml_df
    ):
        mgr = StaticDatasetManager.__new__(StaticDatasetManager)
        mgr.static_dir = tmp_path
        mgr.offline_lag_days = 3
        mgr.max_versions = 3
        old_path = mgr.save(sample_ml_df, "RM Charge")
        old_meta = _load_meta(tmp_path)
        assert old_meta is not None

        original_replace = static_module.os.replace

        def fail_csv_promotion(source, destination):
            if Path(destination).suffix == ".csv":
                raise OSError("simulated CSV promotion failure")
            return original_replace(source, destination)

        monkeypatch.setattr(static_module.os, "replace", fail_csv_promotion)
        with pytest.raises(OSError, match="promotion failure"):
            mgr.save(sample_ml_df.iloc[:10], "RM Charge")

        current_meta = _load_meta(tmp_path)
        assert current_meta is not None
        assert current_meta.csv_file == old_meta.csv_file
        assert old_path.exists()

    def test_save_preserves_old_pointer_when_metadata_publish_fails(
        self, monkeypatch, tmp_path, sample_ml_df
    ):
        mgr = StaticDatasetManager.__new__(StaticDatasetManager)
        mgr.static_dir = tmp_path
        mgr.offline_lag_days = 3
        mgr.max_versions = 1
        old_path = mgr.save(sample_ml_df, "RM Charge")
        old_meta = _load_meta(tmp_path)
        assert old_meta is not None

        monkeypatch.setattr(
            static_module,
            "_save_meta",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("metadata publish failed")),
        )
        with pytest.raises(OSError, match="metadata publish failed"):
            mgr.save(sample_ml_df.iloc[:10], "RM Charge")

        current_meta = _load_meta(tmp_path)
        assert current_meta is not None
        assert current_meta.csv_file == old_meta.csv_file
        assert mgr.current_csv_path() == old_path
        assert old_path.exists()
        # Rotation follows pointer publication, so it cannot remove the active file.
        assert len(list(tmp_path.glob("furnace_dataset_*.csv"))) == 2

    def test_save_rotates_old_files(self, tmp_path, sample_ml_df):
        mgr = StaticDatasetManager.__new__(StaticDatasetManager)
        mgr.static_dir = tmp_path
        mgr.offline_lag_days = 3
        mgr.max_versions = 3

        # Pre-seed 3 old CSVs
        import time
        for i in range(3):
            (tmp_path / f"furnace_dataset_20260{i+1}01_000000.csv").write_text("old")
            time.sleep(0.01)

        mgr.save(sample_ml_df, "RM Charge")

        remaining = list(tmp_path.glob("furnace_dataset_*.csv"))
        assert len(remaining) == 3  # old ones rotated out, new one in


# ---------------------------------------------------------------------------
# update_static() â€” end-to-end with mocked fetcher
# ---------------------------------------------------------------------------

class TestUpdateStatic:
    def _make_manager(self, tmp_path, legacy=None):
        with patch("furnace_data.dataset.static.DatasetFetcher") as MockFetcher:
            mgr = StaticDatasetManager(
                static_dir=tmp_path,
                offline_lag_days=3,
                max_versions=3,
                legacy_csv_path=legacy,
            )
            mgr.fetcher = MockFetcher.return_value
            return mgr

    def test_empty_cache_fetches_from_default_start(self, tmp_path, sample_ml_df):
        mgr = self._make_manager(tmp_path)
        mgr.fetcher.get_dataset.return_value = sample_ml_df

        result = mgr.update_static(rm_choice="RM Charge", apply_cleaning=False)

        mgr.fetcher.get_dataset.assert_called_once()
        call_kwargs = mgr.fetcher.get_dataset.call_args
        assert call_kwargs.kwargs["start_date"] == StaticDatasetManager._DEFAULT_START
        assert not result.empty

    def test_reprocess_from_truncates_existing(self, tmp_path, sample_ml_df):
        # Write a CSV with data going up to Nov 1
        idx = pd.date_range("2025-09-01", periods=60, freq="1D")
        existing_df = pd.DataFrame({"col": range(60)}, index=idx)
        existing_df.to_csv(tmp_path / "furnace_dataset_20260101_000000.csv")

        meta = CacheMeta(
            raw_end="2025-10-31",
            confirmed_end="2025-10-28",
            csv_file="furnace_dataset_20260101_000000.csv",
            rm_choice="charge",
        )
        _save_meta(tmp_path, meta)

        mgr = self._make_manager(tmp_path)
        mgr.fetcher.get_dataset.return_value = sample_ml_df

        result = mgr.update_static(
            rm_choice="RM Charge",
            reprocess_from=date(2025, 10, 1),
            apply_cleaning=False,
        )

        # Frozen part: rows before Oct 1 (Sep rows only)
        call_kwargs = mgr.fetcher.get_dataset.call_args
        assert call_kwargs.kwargs["start_date"] == date(2025, 10, 1)
        assert not result.empty

    def test_returns_existing_when_nothing_to_fetch(self, tmp_path, sample_ml_df):
        # raw_end = today â†’ nothing to fetch
        today = date.today()
        confirmed_end = today - timedelta(days=3)
        sample_ml_df.to_csv(tmp_path / "furnace_dataset_20260101_000000.csv")
        meta = CacheMeta(
            raw_end=today.isoformat(),
            confirmed_end=confirmed_end.isoformat(),
            csv_file="furnace_dataset_20260101_000000.csv",
        )
        _save_meta(tmp_path, meta)

        mgr = self._make_manager(tmp_path)
        result = mgr.update_static(rm_choice="RM Charge", apply_cleaning=False)

        mgr.fetcher.get_dataset.assert_not_called()
        assert not result.empty

    def test_legacy_bootstrap(self, tmp_path, sample_ml_df):
        legacy_path = tmp_path / "legacy.csv"
        sample_ml_df.to_csv(legacy_path)

        mgr = self._make_manager(tmp_path, legacy=legacy_path)
        mgr.fetcher.get_dataset.return_value = pd.DataFrame()  # nothing new

        # Should not crash; legacy data loaded as base
        result = mgr.update_static(rm_choice="RM Charge", apply_cleaning=False)
        # get_dataset was called (to fetch from confirmed_end onward)
        mgr.fetcher.get_dataset.assert_called_once()
