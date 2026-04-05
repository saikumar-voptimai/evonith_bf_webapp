# data_cleaning.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# IterativeImputer is experimental in many sklearn versions.
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
except Exception:
    pass

from sklearn.impute import IterativeImputer, SimpleImputer


# -----------------------------
# Config models
# -----------------------------
@dataclass(frozen=True)
class RangeFilter:
    """Row filter based on a numeric range in a column."""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_inclusive: bool = False  # False => '>' ; True => '>='
    max_inclusive: bool = False  # False => '<' ; True => '<='


@dataclass(frozen=True)
class OutlierRule:
    """Values outside [min_value, max_value] are set to NaN."""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    inclusive: bool = True  # If False, min/max are treated as strict bounds.


@dataclass(frozen=True)
class ImputationPlan:
    """Controls selective imputation before the final global imputation."""
    # Columns to impute with IterativeImputer (MICE). Temperature columns are added dynamically.
    iterative_base_columns: Tuple[str, ...] = ()
    include_temperature_columns: bool = True
    iterative_random_state: int = 0
    iterative_max_iter: int = 10

    # Columns to impute with SimpleImputer and their per-column strategies
    simple_column_strategies: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ColumnGroups:
    rm_params: Tuple[str, ...]
    hm_slag_params: Tuple[str, ...]
    bd_params: Tuple[str, ...]
    temp_params: Tuple[str, ...]
    op_params: Tuple[str, ...]
    prcs_params: Tuple[str, ...]
    proxy_params: Tuple[str, ...]

    @property
    def final_columns(self) -> List[str]:
        # Preserve the original ordering across groups.
        return list(
            self.rm_params
            + self.hm_slag_params
            + self.bd_params
            + self.temp_params
            + self.op_params
            + self.prcs_params
            + self.proxy_params
        )

    @property
    def keep_columns(self) -> List[str]:
        """
        Columns to keep early in the pipeline (includes intermediates if needed).

        We keep SP_01/SP_02 calc columns (if present) so we can combine them into SINTER_CALC_MT.
        """
        keep = self.final_columns.copy()
        for extra in ("SINTER_SP_01_CALC_MT", "SINTER_SP_02_CALC_MT"):
            if extra not in keep:
                keep.append(extra)
        return keep


@dataclass(frozen=True)
class CleaningConfig:
    """All parameters required for the cleaning pipeline."""
    columns: ColumnGroups

    # Index normalization
    time_column: Optional[str] = None  # If provided and exists, set as index.
    time_index_name: str = "time"
    floor_freq: str = "H"
    sort_index: bool = True
    duplicate_timestamp_strategy: str = "mean"  # {"mean", "first", "last", "error"}

    # Column normalization
    uppercase_columns: bool = True
    rename_mt_to_calc_mt: bool = True

    # Schema control
    keep_only_configured_columns: bool = True
    strict_schema: bool = False  # If True, missing required columns raise.

    # NaN handling / sparsity
    row_min_non_na_fraction: float = 0.5
    col_max_nan_fraction: float = 0.30  # Drop columns with > 30% NaN

    # Fill defaults (zeros)
    zero_fill_columns: Tuple[str, ...] = (
        "SINTER_SP_01_CALC_MT",
        "SINTER_SP_02_CALC_MT",
        "STEAMKGS/HR.",
        "FLUX_CALC_MT",
    )
    zero_fill_name_contains: Tuple[str, ...] = ("LLOYDS", "PELLET")

    # Derived columns
    combine_sinter_sources: bool = True
    sinter_sp01_col: str = "SINTER_SP_01_CALC_MT"
    sinter_sp02_col: str = "SINTER_SP_02_CALC_MT"
    sinter_combined_col: str = "SINTER_CALC_MT"

    # Drop columns
    drop_unnamed_columns: bool = True
    unnecessary_columns: Tuple[str, ...] = (
        "ACTUALTON/HR.",
        "BF GAS NETWORK PRESSUREMMWC",
        "DAILYPRODUCTION",
        "NON_COKE_CHARGE_PATTERN",
        "COKE_CHARGE_PATTERN",
        "PCI_2_TM%",
        "BELLY - 15162_TEMP. OC.4AVG.DEG. OC",
        "LOWER STACK - 18660_TEMP. OC.4AVG.DEG. OC",
        "BOSH - 12975_TEMP. OC.4AVG.DEG. OC",
        "BELLY - 15162_TEMP. OC.1BT-12DEG. OC",
    )

    # Feature engineering
    add_unit_cost_feature: bool = True
    unit_cost_col: str = "UNITCOST LAKHS/THM"
    unit_cost_coke_rate_col: str = "COKE RATE KG/THM"
    unit_cost_pci_rate_col: str = "ACTUALKG/THM."
    unit_cost_pci_multiplier: float = 0.53
    unit_cost_unit_multiplier: float = 0.25

    # Operational filters ("non-cruising")
    cruising_filters: Tuple[RangeFilter, ...] = ()

    # Outlier handling (NaN then impute)
    outlier_rules: Tuple[OutlierRule, ...] = ()

    # Selective imputation + final imputation
    imputation_plan: ImputationPlan = ImputationPlan()
    final_numeric_strategy: str = "median"
    final_non_numeric_strategy: str = "most_frequent"

    # Production/tonnage sanity filters (keep rows under these caps)
    tonnage_caps: Dict[str, float] = field(default_factory=dict)


# -----------------------------
# Cleaner implementation
# -----------------------------
class DataCleaner:
    """Configurable, production-friendly cleaning pipeline."""

    def __init__(self, config: CleaningConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline on a raw dataframe."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        df = df.copy(deep=True)

        df = self._normalize_time_index(df)
        df = self._normalize_columns(df)
        df = self._apply_schema(df)

        df = self._rename_mt_columns(df)
        df = self._drop_unnamed(df)
        df = self._fill_default_zeros(df)

        df = self._combine_sinter(df)

        df = self._drop_sparse_rows(df)
        df = self._drop_unnecessary_columns(df)

        df = self._add_features(df)
        df = self._drop_high_nan_columns(df)

        df = self._apply_row_filters(df)
        df = self._apply_outlier_rules(df)

        df = self._selective_imputation(df)
        df = self._apply_tonnage_caps(df)

        df = self._final_imputation(df)

        # df = self._select_and_order_final_columns(df)
        return df

    # -----------------------------
    # Steps
    # -----------------------------
    def _normalize_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        if cfg.time_column and cfg.time_column in df.columns:
            df[cfg.time_column] = pd.to_datetime(df[cfg.time_column], errors="coerce")
            df = df.set_index(cfg.time_column)
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df.index.name = cfg.time_index_name

        if cfg.floor_freq:
            df.index = df.index.floor(cfg.floor_freq)

        if df.index.has_duplicates:
            df = self._handle_duplicate_timestamps(df)

        if cfg.sort_index:
            df = df.sort_index()

        return df

    def _handle_duplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        strategy = (self.config.duplicate_timestamp_strategy or "mean").lower()
        if strategy == "error":
            raise ValueError("Duplicate timestamps found in index.")
        if strategy in {"first", "last"}:
            keep = "first" if strategy == "first" else "last"
            return df[~df.index.duplicated(keep=keep)]

        # Default: mean numeric, first non-numeric
        num_cols = df.select_dtypes(include=[np.number]).columns
        non_num_cols = [c for c in df.columns if c not in num_cols]
        agg: Dict[str, str] = {c: "mean" for c in num_cols}
        agg.update({c: "first" for c in non_num_cols})
        out = df.groupby(df.index).agg(agg)
        out.index.name = self.config.time_index_name
        return out

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.uppercase_columns:
            df.columns = df.columns.map(lambda c: c.upper() if isinstance(c, str) else c)
        return df

    def _apply_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if not cfg.keep_only_configured_columns:
            return df

        keep_cols = cfg.columns.keep_columns
        present = [c for c in keep_cols if c in df.columns]
        missing = [c for c in keep_cols if c not in df.columns]

        if missing:
            msg = (
                f"{len(missing)} configured columns are missing in the input: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
            if cfg.strict_schema:
                raise KeyError(msg)
            self.logger.warning(msg)

        return df[present]

    def _rename_mt_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.rename_mt_to_calc_mt:
            return df

        rename_map: Dict[str, str] = {}
        for col in list(df.columns):
            if not isinstance(col, str):
                continue
            if "_MT" in col and "_CALC_MT" not in col:
                new_col = col.replace("_MT", "_CALC_MT")
                if new_col in df.columns:
                    # Prefer existing _CALC_MT; backfill missing with legacy _MT.
                    df[new_col] = df[new_col].combine_first(df[col])
                    df = df.drop(columns=[col])
                else:
                    rename_map[col] = new_col

        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _drop_unnamed(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.drop_unnamed_columns:
            return df
        unnamed = [c for c in df.columns if isinstance(c, str) and "UNNAMED" in c.upper()]
        return df.drop(columns=unnamed) if unnamed else df

    def _fill_default_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        for col in cfg.zero_fill_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        if cfg.zero_fill_name_contains:
            for col in df.columns:
                if not isinstance(col, str):
                    continue
                if any(substr in col for substr in cfg.zero_fill_name_contains):
                    df[col] = df[col].fillna(0)

        return df

    def _combine_sinter(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if not cfg.combine_sinter_sources:
            return df

        sp01, sp02, combined = cfg.sinter_sp01_col, cfg.sinter_sp02_col, cfg.sinter_combined_col

        if sp01 not in df.columns and sp02 not in df.columns:
            return df

        a = df[sp01] if sp01 in df.columns else 0
        b = df[sp02] if sp02 in df.columns else 0

        if isinstance(a, pd.Series):
            a = pd.to_numeric(a, errors="coerce").fillna(0)
        if isinstance(b, pd.Series):
            b = pd.to_numeric(b, errors="coerce").fillna(0)

        df[combined] = a + b

        drop_cols = [c for c in (sp01, sp02) if c in df.columns]
        return df.drop(columns=drop_cols) if drop_cols else df

    def _drop_sparse_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        frac = self.config.row_min_non_na_fraction
        if not (0.0 <= frac <= 1.0):
            raise ValueError("row_min_non_na_fraction must be in [0, 1]")
        min_non_na = int(np.ceil(frac * df.shape[1]))
        return df.dropna(axis=0, thresh=min_non_na)

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.config.unnecessary_columns if c in df.columns]
        return df.drop(columns=cols) if cols else df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if not cfg.add_unit_cost_feature:
            return df
        if cfg.unit_cost_coke_rate_col not in df.columns or cfg.unit_cost_pci_rate_col not in df.columns:
            self.logger.warning(
                "Unit cost feature skipped (missing columns): %s, %s",
                cfg.unit_cost_coke_rate_col,
                cfg.unit_cost_pci_rate_col,
            )
            return df

        coke_rate = pd.to_numeric(df[cfg.unit_cost_coke_rate_col], errors="coerce")
        pci_rate = pd.to_numeric(df[cfg.unit_cost_pci_rate_col], errors="coerce")
        df[cfg.unit_cost_col] = (coke_rate + cfg.unit_cost_pci_multiplier * pci_rate) * cfg.unit_cost_unit_multiplier
        return df

    def _drop_high_nan_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        thresh = self.config.col_max_nan_fraction
        if not (0.0 <= thresh <= 1.0):
            raise ValueError("col_max_nan_fraction must be in [0, 1]")

        nan_frac = df.isna().mean()
        drop_cols = nan_frac[nan_frac > thresh].index.tolist()
        if drop_cols:
            self.logger.info("Dropping %d columns with NaN fraction > %.2f", len(drop_cols), thresh)
            df = df.drop(columns=drop_cols)
        return df

    def _apply_row_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        for rule in self.config.cruising_filters:
            if rule.column not in df.columns:
                self.logger.warning("Skipping filter; column missing: %s", rule.column)
                continue

            series = pd.to_numeric(df[rule.column], errors="coerce")

            if rule.min_value is not None:
                df = df[series >= rule.min_value] if rule.min_inclusive else df[series > rule.min_value]
                series = pd.to_numeric(df[rule.column], errors="coerce")  # recompute after filtering

            if rule.max_value is not None:
                df = df[series <= rule.max_value] if rule.max_inclusive else df[series < rule.max_value]

        return df

    def _apply_outlier_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        for rule in self.config.outlier_rules:
            if rule.column not in df.columns:
                self.logger.warning("Skipping outlier rule; column missing: %s", rule.column)
                continue

            s = pd.to_numeric(df[rule.column], errors="coerce")
            mask = pd.Series(False, index=df.index)

            if rule.min_value is not None:
                mask |= (s < rule.min_value) if rule.inclusive else (s <= rule.min_value)

            if rule.max_value is not None:
                mask |= (s > rule.max_value) if rule.inclusive else (s >= rule.max_value)

            if mask.any():
                df.loc[mask, rule.column] = np.nan

        return df

    def _selective_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        plan = self.config.imputation_plan

        iter_cols: List[str] = [c for c in plan.iterative_base_columns if c in df.columns]
        if plan.include_temperature_columns:
            iter_cols.extend([c for c in self.config.columns.temp_params if c in df.columns])

        # De-duplicate while preserving order
        seen = set()
        iter_cols = [c for c in iter_cols if not (c in seen or seen.add(c))]

        if iter_cols:
            df = self._iterative_impute(
                df,
                iter_cols,
                random_state=plan.iterative_random_state,
                max_iter=plan.iterative_max_iter,
            )

        for col, strategy in (plan.simple_column_strategies or {}).items():
            if col not in df.columns:
                self.logger.warning("Skipping simple imputation; column missing: %s", col)
                continue
            df[col] = self._simple_impute_series(df[col], strategy=strategy)

        return df

    def _iterative_impute(self, df: pd.DataFrame, cols: Sequence[str], random_state: int, max_iter: int) -> pd.DataFrame:
        X = df[list(cols)].apply(pd.to_numeric, errors="coerce")
        imputer = IterativeImputer(random_state=random_state, max_iter=max_iter)
        imputed = imputer.fit_transform(X)
        df.loc[:, list(cols)] = pd.DataFrame(imputed, columns=list(cols), index=df.index)
        return df

    def _simple_impute_series(self, s: pd.Series, strategy: str) -> pd.Series:
        imputer = SimpleImputer(strategy=strategy)
        out = imputer.fit_transform(s.to_frame()).ravel()
        return pd.Series(out, index=s.index, name=s.name)

    def _apply_tonnage_caps(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, cap in (self.config.tonnage_caps or {}).items():
            if col not in df.columns:
                self.logger.warning("Skipping tonnage cap; column missing: %s", col)
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            df = df[series < cap]
        return df

    def _final_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num_cols = [c for c in df.columns if c not in num_cols]

        if num_cols:
            imp_num = SimpleImputer(strategy=self.config.final_numeric_strategy)
            df[num_cols] = imp_num.fit_transform(df[num_cols])

        if non_num_cols:
            imp_non = SimpleImputer(strategy=self.config.final_non_numeric_strategy)
            df[non_num_cols] = imp_non.fit_transform(df[non_num_cols])

        return df

    def _select_and_order_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        final_cols = self.config.columns.final_columns
        present = [c for c in final_cols if c in df.columns]
        missing = [c for c in final_cols if c not in df.columns]

        if missing:
            msg = (
                f"{len(missing)} final columns are missing after cleaning: "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
            if self.config.strict_schema:
                raise KeyError(msg)
            self.logger.warning(msg)

        return df[present]


# -----------------------------
# Default config that matches your notebook
# -----------------------------
def build_default_config() -> CleaningConfig:
    # ---- Column groups copied from your notebook (Cell 1) ----
    rm_params = (
        'COKE_VM%', 'COKE_ASH%', 'COKE_IM%', 'COKE_FC%', 'COKE_MOIST%', 'COKE_CALC_MT',
        'NUTCOKE_MOIST%',  'NUTCOKE_VM%', 'NUTCOKE_IM%', 'NUTCOKE_FC%', 'NUTCOKE_ASH%',
        'NUTCOKE_CALC_MT',
        'PCI_2_FC%', 'PCI_2_ASH%', 'PCI_2_VM%', 'PCI_2_IM%', 'PCI_2_CALC_MT',
        'SINTER_SP_02_COLD_STRENGTH_AI', 'SINTER_SP_02_COLD_STRENGTH_TI', 'SINTER_SP_02_HOT_STRENGTH_RI', 'SINTER_SP_02_HOT_STRENGTH_RDI',
        'SINTER_SP_02_P%', 'SINTER_SP_02_SIO2%', 'SINTER_SP_02_MGO%', 'SINTER_SP_02_FEO%',
        'SINTER_SP_02_AL2O3%', 'SINTER_SP_02_NA2O%', 'SINTER_SP_02_TIO2%', 'SINTER_SP_02_FE(T)%',
        'SINTER_SP_02_CAO%', 'SINTER_SP_02_K2O%', 'SINTER_SP_02_BASICITY', 'SINTER_SP_02_MNO%',
        'SINTER_CALC_MT',
        'FLUX_TM%', 'FLUX_SIO2%', 'FLUX_FE2O3%', 'FLUX_AL2O3%', 'FLUX_LOI%', 'FLUX_MGO%', 'FLUX_CAO%', 'FLUX_CALC_MT',
        'GEOMIN TYPE', 'ORE_FE(T)%', 'ORE_LOI%', 'ORE_TM%', 'ORE_MGO%', 'ORE_NA2O%', 'ORE_P%', 'ORE_TIO2%', 'ORE_AL2O3%',
        'ORE_SIO2%', 'ORE_MNO%', 'ORE_CAO%', 'ORE_K2O%', 'ORE_CALC_MT',
        'LLOYDS_PELLET_PCT_SIO2', 'LLOYDS_PELLET_PCT_AL2O3', 'LLOYDS_PELLET_PCT_FE2O3',
        'LLOYDS_PELLET_PCT_MGO', 'LLOYDS_PELLET_PCT_TIO2', 'LLOYDS_PELLET_PCT_NA2O',
        'TOTAL_PELLET_CALC_MT','LLOYDS_PELLET_PCT_K2O', 'LLOYDS_PELLET_PCT_TM', 'LLOYDS_PELLET_PCT_LOI',
        'LLOYDS_PELLET_PCT_MNO','LLOYDS_PELLET_PCT_CAO','LLOYDS_PELLET_PCT_P',
        'DAILYPRODUCTION', 'CHARGES/HRS.', 'STOCKRODLEVEL'
    )

    hm_slag_params = (
        'CHEM_PCT_C', 'CHEM_PCT_CR', 'CHEM_PCT_FE', 'CHEM_PCT_MN', 'CHEM_PCT_P',
        'CHEM_PCT_S', 'CHEM_PCT_SI', 'CHEM_PCT_TI', 'SLAG_BASICITY', 'SLAG_PCT_AL2O3', 'SLAG_PCT_CAO',
        'SLAG_PCT_FEO', 'SLAG_PCT_K2O', 'SLAG_PCT_MGO', 'SLAG_PCT_MNO', 'SLAG_PCT_NA2O', 'SLAG_PCT_S',
        'SLAG_PCT_SIO2', 'SLAG_PCT_TIO2', 'SLAG_T_BASICITY', 'HMT_GT_1480C'
    )

    bd_params = (
        'COKE_DISCHARGE_TIME', 'WEIGHTED_COKE_ANGLE', 'TOTAL_COKE_PORTIONS',
        'NON_COKE_DISCHARGE_TIME', 'WEIGHTED_NON_COKE_ANGLE','TOTAL_NON_COKE_PORTIONS'
    )

    temp_params = (
        'TOTAL HEAT LOAD',
        'FURNACE TOP GAS_UPTAKE TEMP. OCAT-16DEG. OC',
        'FURNACE TOP GAS_UPTAKE TEMP. OC.1BT-12DEG. OC',
        'FURNACE TOP GAS_UPTAKE TEMP. OC.2CT-08DEG. OC',
        'FURNACE TOP GAS_UPTAKE TEMP. OC.3DT-04DEG. OC',
        'FURNACE TOP GAS_UPTAKE TEMP. OC.4AVG.DEG. OC',
        'HEARTH PAD CENTER_TEMP. OCA4.3 MTR.DEG. OC',
        'HEARTH PAD CENTER_TEMP. OC.1B5.4 MTR.DEG. OC',
        'HEARTH PAD CENTER_TEMP. OC.2C5.7 MTR.DEG. OC',
        'HEARTH PAD CENTER_TEMP. OC.3D6.1 MTR.DEG. OC',
        'HEARTH PAD CENTER_TEMP. OC.4AVG.DEG. OC',
        'BOSH - 12975_TEMP. OCAT-07DEG. OC',
        'BOSH - 12975_TEMP. OC.1BT-12DEG. OC',
        'BOSH - 12975_TEMP. OC.2CT-17DEG. OC',
        'BOSH - 12975_TEMP. OC.3DT-03DEG. OC',
        'BOSH - 12975_TEMP. OC.4AVG.DEG. OC',
        'BELLY - 15162_TEMP. OCAT-07DEG. OC',
        'BELLY - 15162_TEMP. OC.2CT-17DEG. OC',
        'BELLY - 15162_TEMP. OC.3DT-03DEG. OC',
        'BELLY - 15162_TEMP. OC.1BT-12DEG. OC',
        'BELLY - 15162_TEMP. OC.4AVG.DEG. OC',
        'LOWER STACK - 18660_TEMP. OCAT-07DEG. OC',
        'LOWER STACK - 18660_TEMP. OC.1BT-12DEG. OC',
        'LOWER STACK - 18660_TEMP. OC.2CT-17DEG. OC',
        'LOWER STACK - 18660_TEMP. OC.3DT-03DEG. OC',
        'LOWER STACK - 18660_TEMP. OC.4AVG.DEG. OC',
    )

    op_params = (
        'ACTUALKG/THM.', 'ACT. FUEL RATEKG/THM.',
        'FURNACETOPGASANALYSISCO2ETACO', 'COKE RATE KG/THM', 'PRODUCTIONTONNESPERHR'
    )

    prcs_params = (
        'HOT BLAST PRESSUREBAR', 'TOPPRESSUREBAR', 'DIFFERENTIAL PRESSURETOTALBAR',
        'HOT BLAST TEMP.OC', 'OXYGENFLOWNM3/HR.', 'STEAMKGS/HR.', 'RAFTOC',
        'PERMEABILITYKGS/HR.', 'TUYEREVELOCITYM/S', 'HOT BLAST VOLUMENM3/HR.', 'O2 ENRICHMENT %', 'FURNACE TOP GAS ANALYSISCO2%',
        'BOTTOMBAR', 'FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%', 'TOPBAR'
    )

    proxy_params = ('FURNACE TOP GAS ANALYSISH2%',)

    col_groups = ColumnGroups(
        rm_params=rm_params,
        hm_slag_params=hm_slag_params,
        bd_params=bd_params,
        temp_params=temp_params,
        op_params=op_params,
        prcs_params=prcs_params,
        proxy_params=proxy_params,
    )

    # Non-cruising filters (same as your notebook)
    cruising_filters = (
        RangeFilter("HOT BLAST VOLUMENM3/HR.", min_value=90000, min_inclusive=False),
        RangeFilter("ACTUALKG/THM.", min_value=70, min_inclusive=False),
        RangeFilter("FURNACETOPGASANALYSISCO2ETACO", min_value=38, max_value=47, min_inclusive=False, max_inclusive=False),
        RangeFilter("PRODUCTIONTONNESPERHR", min_value=60, min_inclusive=False),
        RangeFilter("ACT. FUEL RATEKG/THM.", min_value=100, max_value=670, min_inclusive=False, max_inclusive=False),
    )

    # Outlier rules (strictest bounds kept; values outside become NaN then imputed)
    outlier_rules = (
        OutlierRule("RAFTOC", min_value=300, max_value=2800),
        OutlierRule("PCI_2_CALC_MT", min_value=0.5, max_value=200),
        OutlierRule("SINTER_SP_02_HOT_STRENGTH_RI", min_value=0.5, max_value=100),
        OutlierRule("CHARGES/HRS.", min_value=3, max_value=10),
        OutlierRule("O2 ENRICHMENT %", min_value=0.5, max_value=10),
        OutlierRule("OXYGENFLOWNM3/HR.", min_value=1000, max_value=6000),
        OutlierRule("DIFFERENTIAL PRESSURETOTALBAR", min_value=0.5, max_value=2),
        OutlierRule("PRODUCTIONTONNESPERHR", min_value=60, max_value=100),
        OutlierRule("COKE_ASH%", min_value=5, max_value=20),
        OutlierRule("PCI_2_FC%", min_value=60, max_value=90),
        OutlierRule("NUTCOKE_FC%", min_value=60, max_value=90),
        OutlierRule("TOTAL_PELLET_CALC_MT", min_value=0, max_value=100),
    )

    # Imputation plan (your notebook logic)
    imputation_plan = ImputationPlan(
        iterative_base_columns=(
            "RAFTOC",
            "PCI_2_CALC_MT",
            "SINTER_SP_02_HOT_STRENGTH_RI",
            "DIFFERENTIAL PRESSURETOTALBAR",
            "O2 ENRICHMENT %",
            "OXYGENFLOWNM3/HR.",
            "PRODUCTIONTONNESPERHR",
        ),
        include_temperature_columns=True,
        iterative_random_state=0,
        iterative_max_iter=10,
        simple_column_strategies={"CHARGES/HRS.": "most_frequent"},
    )

    # Sanity caps (your notebook logic)
    tonnage_caps = {
        "COKE_CALC_MT": 55,
        "NUTCOKE_CALC_MT": 20,
        "PCI_2_CALC_MT": 30,
        "SINTER_CALC_MT": 200,
        "FLUX_CALC_MT": 10,
        "ORE_CALC_MT": 150,
        "TOTAL_PELLET_CALC_MT": 150,
    }

    return CleaningConfig(
        columns=col_groups,
        cruising_filters=cruising_filters,
        outlier_rules=outlier_rules,
        imputation_plan=imputation_plan,
        tonnage_caps=tonnage_caps,
        row_min_non_na_fraction=0.5,
        col_max_nan_fraction=0.30,
    )
