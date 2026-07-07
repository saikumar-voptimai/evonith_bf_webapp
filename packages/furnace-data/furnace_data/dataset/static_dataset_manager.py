"""Rotating local cache manager for the static ML dataset."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, is_dataclass, replace as dc_replace
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from furnace_data.config import load_config
from furnace_data.dataset.static_csv import (
    fetch_static_dataset_from_database,
    _fallback_static_dataset_path,
    get_static_dataset_path,
    load_static_dataset,
)
from furnace_data.dataset.cleaning import DataCleaner, build_default_config

log = logging.getLogger(__name__)

_CACHE_META_NAME = "cache_meta.json"
_DEFAULT_LOAD_CONFIG = load_config


def _load_dataset_config() -> dict:
    if load_config is not _DEFAULT_LOAD_CONFIG:
        return load_config("setting_ds_dv.yml")
    try:
        from furnace_data.dataset import static_csv as static_csv_module

        return static_csv_module.load_config("setting_ds_dv.yml")
    except Exception:
        return load_config("setting_ds_dv.yml")


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

        Loads the cleaned base from ``historical_static_ml_dataset`` in the offline DB,
        then appends a post-cutoff delta (Steps 2-5) from the base end date
        to today without writing back to the database.
        """
        _ = start_date  # legacy compat

        df_base = fetch_static_dataset_from_database()
        df_base = self._clean_dataset(df_base)
        df_base = self._clip_to_current_hour(df_base)
        cutoff_filter_removed_all = False
        cutoff_value = _load_dataset_config().get("ml_dataset", {}).get("cutoff_date")
        if cutoff_value:
            cutoff = date.fromisoformat(str(cutoff_value))
            filtered_base = df_base.loc[df_base.index.date <= cutoff]
            if filtered_base.empty:
                cutoff_filter_removed_all = True
                log.warning(
                    "Configured static ML cutoff %s would remove all canonical rows; "
                    "keeping the fetched base dataset.",
                    cutoff,
                )
            else:
                df_base = filtered_base
        else:
            return df_base
        if df_base.empty:
            return df_base

        base_end = df_base.index.max().date()
        today = self._current_local_hour().date()

        if cutoff_filter_removed_all and self._uses_default_delta_fetcher():
            return df_base

        if base_end >= today - timedelta(days=1):
            return df_base

        delta_start = base_end + timedelta(days=1)
        rm_mode = "dpr" if str(rm_choice).lower() in ("rm dpr", "dpr") else "charge"
        df_delta = self._fetch_and_clean_delta(delta_start, today, rm_mode)

        if df_delta.empty:
            raise RuntimeError(
                "Delta fetch returned no rows "
                f"(base ends {base_end}, today {today})."
            )

        combined = self._clip_to_current_hour(
            pd.concat([df_base, df_delta]).sort_index()
        )

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
            from furnace_data.dataset.fetcher import DatasetFetcher

            fetcher = DatasetFetcher()
            df_raw = fetcher.build_local_delta(start, end, rm_mode, raise_on_error=True)
            if df_raw.empty:
                return df_raw

            # Drop genuinely non-numeric columns (charge pattern labels, burden purpose)
            # but coerce numeric-looking object columns (e.g. STEAMKGS/HR. from InfluxDB
            # with mixed types) rather than silently dropping them.
            object_cols = df_raw.select_dtypes(include="object").columns.tolist()
            dropped = []
            for col in object_cols:
                numeric = pd.to_numeric(df_raw[col], errors="coerce")
                if numeric.notna().any():
                    df_raw[col] = numeric
                else:
                    dropped.append(col)
            if dropped:
                df_raw = df_raw.drop(columns=dropped)

            # Resample outer-joined multi-granularity data to a regular hourly cadence.
            # Material quantities are hourly totals; lab/process context is averaged and
            # forward-filled because a lab sample remains valid until the next sample.
            df_raw = self._resample_local_delta_hourly(df_raw)

            # Derive PCI_CALC_MT from online process params when the charge-system
            # column is absent or all-NaN (PCI is injected via lances, not charged
            # through hoppers so it never appears in charge_data).
            # Formula: PCI rate (kg/tHM) x production (t/hr) / 1000 = PCI mass (MT/hr).
            pci_col = "PCI_CALC_MT"
            if "PCI_KG/THM" in df_raw.columns and "PRODUCTIONTONNESPERHR" in df_raw.columns:
                derived_pci_mt = (
                    df_raw["PCI_KG/THM"] * df_raw["PRODUCTIONTONNESPERHR"] / 1000
                )
                if pci_col in df_raw.columns:
                    df_raw[pci_col] = df_raw[pci_col].combine_first(derived_pci_mt)
                else:
                    df_raw[pci_col] = derived_pci_mt

            # Use a lower sparse-row threshold (0.2 vs default 0.5): post-cutoff joined
            # data is naturally sparser than the pre-merged historical dataset.
            # Disable tonnage_caps: these caps were designed for hourly sums in the
            # historical InfluxDB data; the delta uses per-charge averages (3-6 MT coke
            # vs a cap of 55), so the caps are a no-op for valid data but cause false
            # drops when PCI outlier rules reset zero-filled values back to NaN.
            base_config = build_default_config()
            if is_dataclass(base_config):
                delta_config = dc_replace(
                    base_config,
                    row_min_non_na_fraction=0.2,
                    tonnage_caps={},
                )
            else:
                delta_config = base_config
            cleaned = DataCleaner(delta_config).clean(df_raw)
            cleaned = self._repair_material_quantity_totals(cleaned)

            if cleaned.empty:
                raise RuntimeError("Delta cleaning returned no rows even with lower threshold.")
            return cleaned
        except Exception:
            log.warning("Post-cutoff delta fetch/clean failed.", exc_info=True)
            raise

    def save(self, df: pd.DataFrame) -> Path:
        """Save a cleaned full dataset snapshot and rotate old versioned files."""
        df = self._clip_to_current_hour(df)
        if df.empty:
            raise ValueError("Cannot save an empty static ML dataset.")

        saved_path = self.static_path.parent / self._versioned_filename()
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
        cleaned = self._repair_material_quantity_totals(cleaned)
        if cleaned.empty:
            raise ValueError("Static ML dataset cleaning returned no rows.")
        return cleaned

    @staticmethod
    def _current_local_hour() -> pd.Timestamp:
        local_tz = (
            _load_dataset_config()
            .get("ml_dataset", {})
            .get("local_tz", "Asia/Kolkata")
        )
        return pd.Timestamp.now(tz=ZoneInfo(local_tz)).floor("h").tz_localize(None)

    @classmethod
    def _clip_to_current_hour(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
        current_hour = cls._current_local_hour()
        if df.index.min() > current_hour:
            return df
        return df.loc[df.index <= current_hour]

    def _uses_default_delta_fetcher(self) -> bool:
        return (
            getattr(self._fetch_and_clean_delta, "__func__", None)
            is StaticDatasetManager._fetch_and_clean_delta
        )

    @staticmethod
    def _resample_local_delta_hourly(df: pd.DataFrame) -> pd.DataFrame:
        quantity_cols = [
            col for col in df.columns
            if StaticDatasetManager._is_hourly_quantity_column(col)
        ]
        context_cols = [col for col in df.columns if col not in quantity_cols]

        frames: list[pd.DataFrame] = []
        if context_cols:
            frames.append(df[context_cols].resample("1h").mean().ffill(limit=24))
        if quantity_cols:
            frames.append(df[quantity_cols].resample("1h").sum(min_count=1))

        if not frames:
            return df.resample("1h").mean()
        return pd.concat(frames, axis=1).sort_index()

    @staticmethod
    def _is_hourly_quantity_column(column: object) -> bool:
        return isinstance(column, str) and column.endswith("_CALC_MT")

    @staticmethod
    def _repair_material_quantity_totals(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        aggregate_specs = {
            "ORE_CALC_MT": [f"ORE_{i}_CALC_MT" for i in range(1, 13)],
            "FLUX_CALC_MT": [f"FLUX_{i}_CALC_MT" for i in range(1, 4)],
        }
        for aggregate, columns in aggregate_specs.items():
            present = [column for column in columns if column in out.columns]
            if present and aggregate in out.columns:
                values = out[present].apply(pd.to_numeric, errors="coerce")
                has_slot_data = values.notna().any(axis=1)
                out.loc[has_slot_data, aggregate] = (
                    values.loc[has_slot_data].fillna(0.0).sum(axis=1)
                )
        return out

    def get_db_end_date(self) -> date | None:
        """Return the latest ``date_time`` in ``historical_static_ml_dataset``."""
        try:
            from furnace_data.offline import get_offline_table_bounds
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
        fallback = _fallback_static_dataset_path(self.static_path)
        if fallback and fallback.exists():
            return fallback
        return self.static_path

    def _versioned_filename(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{self.static_path.stem}_{timestamp}{self.static_path.suffix}"

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
