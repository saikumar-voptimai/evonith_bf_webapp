"""Rotating local cache manager for the static ML dataset."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from config.config_loader import load_config
from data.ml.static_csv import (
    fetch_static_dataset_from_database,
    fetch_static_dataset_from_url,
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
    source_url: str = ""

    @property
    def confirmed_end_date(self) -> date | None:
        return date.fromisoformat(self.confirmed_end) if self.confirmed_end else None

    @property
    def raw_end_date(self) -> date | None:
        return date.fromisoformat(self.raw_end) if self.raw_end else None


class StaticDatasetManager:
    """Maintain a stable CSV plus timestamped rotating fallback copies.

    When ``remote_url`` is set, the published CSV is authoritative and the
    database rebuild/delta path is bypassed completely.
    """

    _MAX_VERSIONED_FILES: int = 3

    # Hour-ending convention: a row stamped HH:00 summarises [HH-1:00, HH:00), so
    # [23:00, 00:00) is stamped 00:00 of the next day.  It covers RM lab chemistry,
    # online InfluxDB columns, charge ``*_CALC_MT`` quantities and PCI_CALC_MT;
    # HM/slag and burden distribution keep their own labels.
    # PCI_CALC_MT is a lance quantity, not a charge one, so it is named explicitly.
    _PCI_QUANTITY_COLUMN: str = "PCI_CALC_MT"

    # LEGACY ALIGNMENT ONLY - not part of normal resampling.  Rows of the
    # historical DB table from this IST timestamp onward carry hour-BEGINNING
    # labels for the chemistry/online/PCI columns, so they are moved one hour
    # forward by ``_align_legacy_hour_beginning_labels``.
    _LEGACY_HOUR_ENDING_START: pd.Timestamp = pd.Timestamp("2025-06-01 00:00:00")

    def __init__(
        self,
        static_path: str | Path | None = None,
        *,
        remote_url: str | None = None,
        remote_timeout_seconds: float = 60.0,
    ) -> None:
        self.static_path = (
            Path(static_path) if static_path is not None else get_static_dataset_path()
        )
        self.remote_url = str(remote_url or "").strip()
        self.remote_timeout_seconds = max(1.0, float(remote_timeout_seconds))
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

        The base is first relabelled from hour-beginning to hour-ending (see
        ``_align_legacy_hour_beginning_labels``); the delta is built hour-ending.
        """
        _ = start_date  # legacy compat

        if self.remote_url:
            return self._clip_to_current_hour(
                fetch_static_dataset_from_url(
                    self.remote_url,
                    timeout_seconds=self.remote_timeout_seconds,
                )
            )

        df_base = fetch_static_dataset_from_database()
        # Must precede cleaning: the cleaner filters and imputes rows, so shifting
        # afterwards would leave NaN holes next to every dropped row.  The delta
        # below is already hour-ending and must NOT go through this correction.
        df_base = self._align_legacy_hour_beginning_labels(df_base)
        df_base = self._clean_dataset(df_base)
        df_base = self._clip_to_current_hour(df_base)
        cutoff_value = (
            load_config("setting_ds_dv.yml").get("ml_dataset", {}).get("cutoff_date")
        )
        if cutoff_value:
            cutoff = date.fromisoformat(str(cutoff_value))
            df_base = df_base.loc[df_base.index.date <= cutoff]
        if df_base.empty:
            return df_base

        base_end = df_base.index.max().date()
        today = self._current_local_hour().date()

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
            len(df_base),
            base_end,
            len(df_delta),
            df_delta.index.max().date(),
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

            # Hour-ending labels: the first requested row (00:00) summarises
            # [23:00, 24:00) of the previous day, so fetch one extra day and trim
            # back to ``start`` after resampling.  Without it that row would be
            # empty and dropped as sparse, leaving a gap at the base/delta seam.
            fetcher = DatasetFetcher()
            df_raw = fetcher.build_local_delta(
                start - timedelta(days=1), end, rm_mode, raise_on_error=True
            )
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
            # Material quantities, RM chemistry and online columns are labelled by
            # the END of their hour; remaining context (HM/slag, burden distribution)
            # keeps its hour-beginning label.  Context is averaged and forward-filled
            # because a lab sample remains valid until the next sample.
            df_raw = self._resample_local_delta_hourly(df_raw)
            # Right-labelling turns the in-progress hour into a future-stamped
            # bucket, and the lookback day precedes ``start``; drop both here so
            # neither reaches the cleaner.
            df_raw = self._clip_to_current_hour(
                df_raw.loc[df_raw.index >= pd.Timestamp(start)]
            )

            # Derive PCI_CALC_MT from online process params when the charge-system
            # column is absent or all-NaN (PCI is injected via lances, not charged
            # through hoppers so it never appears in charge_data).
            # Formula: PCI rate (kg/tHM) x production (t/hr) / 1000 = PCI mass (MT/hr).
            # Both inputs were hour-ending labelled by the resample above, so the
            # derived value already sits on the right row and must not be shifted.
            pci_col = self._PCI_QUANTITY_COLUMN
            if (
                "PCI_KG/THM" in df_raw.columns
                and "PRODUCTIONTONNESPERHR" in df_raw.columns
            ):
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
            delta_config = dc_replace(
                build_default_config(),
                row_min_non_na_fraction=0.2,
                tonnage_caps={},
            )
            cleaned = DataCleaner(delta_config).clean(df_raw)
            cleaned = self._repair_material_quantity_totals(cleaned)

            if cleaned.empty:
                raise RuntimeError(
                    "Delta cleaning returned no rows even with lower threshold."
                )
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
            load_config("setting_ds_dv.yml")
            .get("ml_dataset", {})
            .get("local_tz", "Asia/Kolkata")
        )
        return pd.Timestamp.now(tz=ZoneInfo(local_tz)).floor("h").tz_localize(None)

    @classmethod
    def _clip_to_current_hour(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
        return df.loc[df.index <= cls._current_local_hour()]

    @classmethod
    def _resample_local_delta_hourly(cls, df: pd.DataFrame) -> pd.DataFrame:
        hour_ending = cls._hour_ending_context_columns()
        quantity_cols = [
            col for col in df.columns if cls._is_hourly_quantity_column(col)
        ]
        hour_ending_context_cols = [
            col
            for col in df.columns
            if not cls._is_hourly_quantity_column(col)
            and str(col).upper() in hour_ending
        ]
        # HM/slag, burden distribution, lab ash/dust analyses: unchanged behaviour.
        other_context_cols = [
            col
            for col in df.columns
            if not cls._is_hourly_quantity_column(col)
            and str(col).upper() not in hour_ending
        ]

        # Every group below that is hour-ending uses closed="left", label="right":
        # values from [11:00, 12:00) belong to 12:00, and [23:00, 00:00) to 00:00.
        # Native resampling keeps this vectorized and handles day boundaries.
        # Forward-fill only after labelling so it cannot pull data back an hour.
        frames: list[pd.DataFrame] = []
        if other_context_cols:
            frames.append(
                df[other_context_cols].resample("1h").mean().ffill(limit=24)
            )
        if hour_ending_context_cols:
            frames.append(
                df[hour_ending_context_cols]
                .resample("1h", closed="left", label="right")
                .mean()
                .ffill(limit=24)
            )
        if quantity_cols:
            # Charge materials and PCI_CALC_MT share one convention: summed per
            # hour and labelled by the hour end.
            frames.append(
                df[quantity_cols]
                .resample("1h", closed="left", label="right")
                .sum(min_count=1)
            )

        if not frames:
            return df.resample("1h").mean()
        return pd.concat(frames, axis=1).sort_index()

    @staticmethod
    def _is_hourly_quantity_column(column: object) -> bool:
        return isinstance(column, str) and column.endswith("_CALC_MT")

    @staticmethod
    def _hour_ending_context_columns() -> frozenset[str]:
        """Upper-cased ML names of the non-quantity columns labelled by hour end.

        Derived from ``setting_ds_dv.yml`` (through ``rename_dict``) so a newly
        configured source is picked up without editing a column list here:

        * online InfluxDB columns: process params, temperature profile, total
          heat load and miscellaneous params;
        * RM lab columns of ``cleaning.column_groups.rm_params`` (weighted
          ORE/pellet/SINTER/COKE/NUTCOKE/FLUX/PCI chemistry and strength), minus
          the ``*_mt`` quantities and keys the cleaner discards.
        """
        cfg = load_config("setting_ds_dv.yml")
        rename = cfg.get("rename_dict") or {}
        ml = cfg.get("ml_dataset") or {}
        cleaning = cfg.get("cleaning") or {}
        groups = cleaning.get("column_groups") or {}

        def ml_names(keys) -> set[str]:
            return {str(rename.get(key, key)).upper() for key in keys}

        online = (
            ml_names((ml.get("online_params") or {}).values())
            | ml_names((ml.get("temperature_params") or {}).values())
            | ml_names((ml.get("misc_params") or {}).values())
            | ml_names(groups.get("temp_params") or [])
        )
        rm_lab = (
            ml_names(
                key
                for key in groups.get("rm_params") or []
                if not str(key).endswith("_mt")
            )
            - ml_names(cleaning.get("unnecessary_alias_keys") or [])
        )
        return frozenset(online | rm_lab)

    @classmethod
    def _align_legacy_hour_beginning_labels(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Relabel the historical DB columns from hour-beginning to hour-ending.

        LEGACY CORRECTION, not resampling.  From ``_LEGACY_HOUR_ENDING_START``
        the historical table stamped RM chemistry, online columns and
        ``PCI_CALC_MT`` with the START of their hour.  Each such value moves to
        the row one hour later, e.g. 2025-06-01 23:00 -> 2025-06-02 00:00, and
        2025-05-31 23:00 feeds 2025-06-01 00:00.  Rows before the start, and
        all other columns (HM/slag, burden distribution, charge ``*_CALC_MT``,
        which are already hour-ending), are left untouched; the index itself is
        never shifted.

        Run on the base only: the post-cutoff delta is hour-ending from the
        start, so applying this to it would shift it twice.
        """
        hour_ending = cls._hour_ending_context_columns()
        cols = [
            col
            for col in df.columns
            if str(col).upper() in hour_ending
            or str(col).upper() == cls._PCI_QUANTITY_COLUMN
        ]
        if df.empty or not cols or not isinstance(df.index, pd.DatetimeIndex):
            return df

        # The raw table has duplicate and sub-hourly rows (the cleaner only
        # floors and averages them afterwards), so build one value per source
        # hour first.  Mean matches the cleaner's duplicate-timestamp strategy.
        row_hour = df.index.floor("h")
        values = df[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        # Source hour h (hour-beginning) -> destination hour h + 1h (hour-ending);
        # a destination whose source hour has no data stays NaN for the cleaner.
        shifted = values.groupby(row_hour).mean().shift(freq="1h")

        corrected = row_hour >= cls._LEGACY_HOUR_ENDING_START
        values.loc[corrected] = shifted.reindex(row_hour[corrected]).to_numpy()

        out = df.copy()
        # Positional assignment: the raw index may hold duplicate labels.
        out[cols] = values.to_numpy()
        return out

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

            _, end, _ = get_offline_table_bounds(
                "offline_feed.historical_static_ml_dataset"
            )
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
            return CacheMeta(
                **{key: value for key, value in data.items() if key in fields}
            )
        except Exception:
            return None

    def current_csv_path(self) -> Path:
        """Return the active local CSV path."""
        if self.remote_url:
            # The published URL owns the dataset. Always expose the stable local
            # fallback that is atomically replaced by each hourly fetch, rather
            # than a database-generated versioned snapshot from an older run.
            return self.static_path
        meta = self.get_meta()
        if meta and meta.csv_file:
            candidate = self.static_path.parent / meta.csv_file
            if candidate.exists():
                return candidate
        latest = self._latest_versioned_file()
        if latest is not None:
            return latest
        return self.static_path

    def get_csv_end_timestamp(self) -> pd.Timestamp | None:
        """Read the time value from the CSV's last data row.

        Only the tail of the file is read, avoiding a full multi-megabyte CSV
        parse just to render dataset status in the UI.
        """
        path = self.current_csv_path()
        if not path.exists():
            return None

        try:
            with path.open("rb") as handle:
                handle.seek(0, 2)
                position = handle.tell()
                buffer = b""
                while position > 0 and len(buffer.splitlines()) < 2:
                    chunk_size = min(4096, position)
                    position -= chunk_size
                    handle.seek(position)
                    buffer = handle.read(chunk_size) + buffer

            lines = [line for line in buffer.splitlines() if line.strip()]
            if len(lines) < 2:
                return None
            row = next(csv.reader([lines[-1].decode("utf-8-sig")]))
            if not row:
                return None
            value = pd.to_datetime(row[0], errors="coerce")
            return None if pd.isna(value) else pd.Timestamp(value)
        except (OSError, UnicodeError, csv.Error, ValueError):
            log.warning(
                "Could not read final CSV timestamp from %s", path, exc_info=True
            )
            return None

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
            source_url=self.remote_url,
        )

    def _save_meta(self, meta: CacheMeta) -> None:
        self._meta_path.write_text(
            json.dumps(asdict(meta), indent=2),
            encoding="utf-8",
        )


__all__ = ["CacheMeta", "StaticDatasetManager"]
