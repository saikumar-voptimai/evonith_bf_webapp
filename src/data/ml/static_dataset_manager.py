"""Rotating local cache manager for the static ML dataset."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.ml.static_csv import (
    fetch_static_dataset_from_database,
    get_static_dataset_path,
    load_static_dataset,
)
from furnace_data.dataset.cleaning import DataCleaner, build_default_config

log = logging.getLogger(__name__)

_CACHE_META_NAME = "cache_meta.json"


@dataclass
class CacheMeta:
    """JSON-serialisable metadata for the local static dataset cache."""

    version: int = 1
    rm_choice: str = "full"
    data_start: str = ""
    confirmed_end: str = ""
    raw_end: str = ""
    last_updated: str = ""
    offline_lag_days: int = 0
    rows: int = 0
    columns: int = 0
    csv_file: str = ""

    @property
    def confirmed_end_date(self) -> date | None:
        return date.fromisoformat(self.confirmed_end) if self.confirmed_end else None

    @property
    def raw_end_date(self) -> date | None:
        return date.fromisoformat(self.raw_end) if self.raw_end else None


class StaticDatasetManager:
    """Maintain a stable CSV plus timestamped rotating copies of the full dataset."""

    _MAX_VERSIONED_FILES: int = 3

    def __init__(self, static_path: str | Path | None = None) -> None:
        self.static_path = (
            Path(static_path) if static_path is not None else get_static_dataset_path()
        )
        self.static_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.static_path.parent / _CACHE_META_NAME

    def update_static(
        self,
        rm_choice: str = "Full",
        start_date: date | None = None,
    ) -> pd.DataFrame:
        """Fetch, clean, and extend the static ML dataset with a local delta.

        Loads the cleaned base from ``historical_static_ml_dataset`` (Neon),
        then appends a post-cutoff delta (Steps 2-5) from the base end date
        to today without writing back to the database.  Falls back to the
        base alone if the delta fetch fails.
        """
        _ = start_date  # legacy compat

        df_base = fetch_static_dataset_from_database()
        df_base = self._clean_dataset(df_base)
        if df_base.empty:
            return df_base

        base_end = df_base.index.max().date()
        today = date.today()

        if base_end >= today - timedelta(days=1):
            return df_base

        delta_start = base_end + timedelta(days=1)
        rm_mode = "dpr" if str(rm_choice).lower() in ("rm dpr", "dpr") else "charge"
        df_delta = self._fetch_and_clean_delta(delta_start, today, rm_mode)

        if df_delta.empty:
            log.warning(
                "Delta fetch returned no rows; returning base only (base ends %s, today %s).",
                base_end, today,
            )
            return df_base

        combined = pd.concat([df_base, df_delta]).sort_index()

        # After concat, rows from each side may have NaN for columns that only
        # the other side carries (e.g. temperature averages in base but not delta,
        # or new sensors in delta but not base).  Fill those NaN cells with the
        # column median so that ML inference rows are not silently dropped by
        # process_dataframe()'s dropna().
        delta_start = df_delta.index.min()
        delta_mask = combined.index >= delta_start
        base_mask  = ~delta_mask

        # 1. Columns added by the delta that are NaN in base rows
        for col in combined.columns:
            if base_mask.any() and combined.loc[base_mask, col].isna().all():
                combined[col] = combined[col].fillna(combined[col].median())

        # 2. Columns carried only by the base that are NaN in delta rows
        for col in combined.columns:
            if delta_mask.any() and combined.loc[delta_mask, col].isna().all():
                combined[col] = combined[col].fillna(combined[col].median())

        log.info(
            "Combined base (%d rows, ends %s) + delta (%d rows, ends %s) = %d rows total.",
            len(df_base), base_end,
            len(df_delta), df_delta.index.max().date(),
            len(combined),
        )
        return combined

    def _fetch_and_clean_delta(
        self,
        start: date,
        end: date,
        rm_mode: str,
    ) -> pd.DataFrame:
        """Fetch Steps 2-5 for [start, end], apply full cleaning. Local only.

        Before cleaning, resamples to hourly and forward-fills so that RM
        chemistry lab values (valid between charges) propagate to all hourly
        slots.  Uses a lower sparse-row threshold (0.2) because the joined
        post-cutoff data is naturally sparser than the pre-merged historical
        dataset.
        """
        try:
            from dataclasses import replace as dc_replace
            from furnace_data.dataset.fetcher import DatasetFetcher

            fetcher = DatasetFetcher()
            df_raw = fetcher.build_local_delta(start, end, rm_mode)
            if df_raw.empty:
                return df_raw

            # Drop genuinely non-numeric columns (charge pattern labels, burden purpose)
            # but coerce numeric-looking object columns (e.g. STEAMKGS/HR. from InfluxDB
            # with mixed types) rather than silently dropping them.
            object_cols = df_raw.select_dtypes(include="object").columns.tolist()
            coerced, dropped = [], []
            for col in object_cols:
                numeric = pd.to_numeric(df_raw[col], errors="coerce")
                if numeric.notna().any():
                    df_raw[col] = numeric
                    coerced.append(col)
                else:
                    dropped.append(col)
            if dropped:
                df_raw = df_raw.drop(columns=dropped)

            # Resample outer-joined multi-granularity data to a regular hourly cadence
            # and forward-fill.  RM chemistry lab values are valid until the next sample,
            # so ffill(limit=24) propagates them across up to 24 hourly slots.
            df_raw = df_raw.resample("1h").mean().ffill(limit=24)

            # Derive PCI_CALC_MT from online process params when the charge-system
            # column is absent or all-NaN (PCI is injected via lances, not charged
            # through hoppers so it never appears in charge_data).
            # Formula: PCI rate (kg/tHM) × production (t/hr) ÷ 1000 = PCI mass (MT/hr).
            pci_col = "PCI_CALC_MT"
            if (
                pci_col not in df_raw.columns or df_raw[pci_col].isna().all()
            ) and "PCI_KG/THM" in df_raw.columns and "PRODUCTIONTONNESPERHR" in df_raw.columns:
                df_raw[pci_col] = (
                    df_raw["PCI_KG/THM"] * df_raw["PRODUCTIONTONNESPERHR"] / 1000
                )

            # Use a lower sparse-row threshold (0.2 vs default 0.5): post-cutoff joined
            # data is naturally sparser than the pre-merged historical dataset.
            # Disable tonnage_caps: these caps were designed for hourly sums in the
            # historical InfluxDB data; the delta uses per-charge averages (3–6 MT coke
            # vs a cap of 55), so the caps are a no-op for valid data but cause false
            # drops when PCI outlier rules reset zero-filled values back to NaN.
            delta_config = dc_replace(
                build_default_config(),
                row_min_non_na_fraction=0.2,
                tonnage_caps={},
            )
            cleaned = DataCleaner(delta_config).clean(df_raw)

            if cleaned.empty:
                log.warning("Delta cleaning returned no rows even with lower threshold.")
                return pd.DataFrame()
            return cleaned
        except Exception:
            log.warning("Post-cutoff delta fetch/clean failed.", exc_info=True)
            return pd.DataFrame()

    def save(self, df: pd.DataFrame) -> Path:
        """Save a cleaned full dataset snapshot and rotate old versioned files."""
        if df.empty:
            raise ValueError("Cannot save an empty static ML dataset.")

        saved_path = self.static_path.parent / self._versioned_filename()
        sequence = 1
        while saved_path.exists():
            saved_path = self.static_path.parent / self._versioned_filename(sequence)
            sequence += 1
        df.to_csv(saved_path, index=True)
        if saved_path.resolve() != self.static_path.resolve():
            shutil.copyfile(saved_path, self.static_path)

        self._save_meta(self._build_meta(df, saved_path))
        self._rotate_versioned_files()

        try:
            load_static_dataset.clear()
        except Exception:
            pass

        return saved_path

    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        cleaner = DataCleaner(build_default_config())
        cleaned = cleaner.clean(df)
        if cleaned.empty:
            raise ValueError("Static ML dataset cleaning returned no rows.")
        return cleaned

    def get_db_end_date(self) -> date | None:
        """Return the latest ``date_time`` in ``historical_static_ml_dataset``."""
        try:
            from furnace_data.neon_db.offline import get_offline_table_bounds
            _, end, _ = get_offline_table_bounds("offline_feed.historical_static_ml_dataset")
            return end.date() if end else None
        except Exception:
            return None

    def get_meta(self) -> CacheMeta | None:
        """Return local cache metadata without touching the database."""
        if not self._meta_path.exists():
            return None
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            fields = CacheMeta.__dataclass_fields__
            return CacheMeta(**{key: value for key, value in data.items() if key in fields})
        except Exception:
            return None

    def current_csv_path(self) -> Path:
        """Return the active local CSV path."""
        meta = self.get_meta()
        if meta and meta.csv_file:
            candidate = self.static_path.parent / meta.csv_file
            if candidate.exists():
                return candidate
        latest = self._latest_versioned_file()
        if latest is not None:
            return latest
        return self.static_path

    def _versioned_filename(self, sequence: int | None = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"_{sequence}" if sequence is not None else ""
        return f"{self.static_path.stem}_{timestamp}{suffix}{self.static_path.suffix}"

    def _latest_versioned_file(self) -> Path | None:
        files = sorted(
            self.static_path.parent.glob(
                f"{self.static_path.stem}_*{self.static_path.suffix}"
            ),
            key=lambda path: path.stat().st_mtime,
        )
        return files[-1] if files else None

    def _rotate_versioned_files(self) -> None:
        files = sorted(
            self.static_path.parent.glob(
                f"{self.static_path.stem}_*{self.static_path.suffix}"
            ),
            key=lambda path: path.stat().st_mtime,
        )
        for old_file in files[: -self._MAX_VERSIONED_FILES]:
            old_file.unlink(missing_ok=True)

    def _build_meta(self, df: pd.DataFrame, saved_path: Path) -> CacheMeta:
        first = df.index.min()
        last = df.index.max()
        return CacheMeta(
            data_start=str(first.date()) if hasattr(first, "date") else "",
            confirmed_end=str(last.date()) if hasattr(last, "date") else "",
            raw_end=str(last.date()) if hasattr(last, "date") else "",
            last_updated=datetime.now().isoformat(timespec="seconds"),
            rows=len(df),
            columns=len(df.columns),
            csv_file=saved_path.name,
        )

    def _save_meta(self, meta: CacheMeta) -> None:
        self._meta_path.write_text(
            json.dumps(asdict(meta), indent=2),
            encoding="utf-8",
        )


__all__ = ["CacheMeta", "StaticDatasetManager"]
