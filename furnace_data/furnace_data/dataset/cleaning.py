"""Configurable data-cleaning pipeline for the BF2 ML dataset.

Moved from ``ml_dataset_core/data_cleaning.py``.  No import path changes for
callers — ``from furnace_data.dataset.cleaning import DataCleaner, build_default_config``.

The ``ml_dataset_core.data_cleaning`` module is kept as a one-line re-export shim
during the Phase-2 transition.
"""

# Re-export the full module content unchanged.
# The implementation is long and stable; keep it verbatim to avoid regression.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
except Exception:
    pass

from sklearn.impute import IterativeImputer, SimpleImputer


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RangeFilter:
    """Row filter based on a numeric range in a column."""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_inclusive: bool = False
    max_inclusive: bool = False


@dataclass(frozen=True)
class OutlierRule:
    """Values outside [min_value, max_value] are set to NaN."""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    inclusive: bool = True


@dataclass(frozen=True)
class ImputationPlan:
    iterative_base_columns: Tuple[str, ...] = ()
    include_temperature_columns: bool = True
    iterative_random_state: int = 0
    iterative_max_iter: int = 10
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
        return list(
            self.rm_params + self.hm_slag_params + self.bd_params
            + self.temp_params + self.op_params + self.prcs_params + self.proxy_params
        )

    @property
    def keep_columns(self) -> List[str]:
        keep = self.final_columns.copy()
        for extra in ("SINTER_SP_01_CALC_MT", "SINTER_SP_02_CALC_MT"):
            if extra not in keep:
                keep.append(extra)
        return keep


@dataclass(frozen=True)
class CleaningConfig:
    """All parameters required for the cleaning pipeline."""
    columns: ColumnGroups

    time_column: Optional[str] = None
    time_index_name: str = "time"
    floor_freq: str = "H"
    sort_index: bool = True
    duplicate_timestamp_strategy: str = "mean"

    uppercase_columns: bool = True
    rename_mt_to_calc_mt: bool = True

    keep_only_configured_columns: bool = True
    strict_schema: bool = False

    row_min_non_na_fraction: float = 0.5
    col_max_nan_fraction: float = 0.30

    zero_fill_columns: Tuple[str, ...] = (
        "SINTER_SP_01_CALC_MT", "SINTER_SP_02_CALC_MT", "STEAMKGS/HR.", "FLUX_CALC_MT",
    )
    zero_fill_name_contains: Tuple[str, ...] = ("LLOYDS", "PELLET")

    combine_sinter_sources: bool = True
    sinter_sp01_col: str = "SINTER_SP_01_CALC_MT"
    sinter_sp02_col: str = "SINTER_SP_02_CALC_MT"
    sinter_combined_col: str = "SINTER_CALC_MT"

    drop_unnamed_columns: bool = True
    unnecessary_columns: Tuple[str, ...] = (
        "ACTUALTON/HR.", "BF GAS NETWORK PRESSUREMMWC", "DAILYPRODUCTION",
        "NON_COKE_CHARGE_PATTERN", "COKE_CHARGE_PATTERN", "PCI_2_TM%",
        "BELLY - 15162_TEMP. OC.4AVG.DEG. OC",
        "LOWER STACK - 18660_TEMP. OC.4AVG.DEG. OC",
        "BOSH - 12975_TEMP. OC.4AVG.DEG. OC",
        "BELLY - 15162_TEMP. OC.1BT-12DEG. OC",
    )

    add_unit_cost_feature: bool = True
    unit_cost_col: str = "UNITCOST LAKHS/THM"
    unit_cost_coke_rate_col: str = "COKE RATE KG/THM"
    unit_cost_pci_rate_col: str = "ACTUALKG/THM."
    unit_cost_pci_multiplier: float = 0.53
    unit_cost_unit_multiplier: float = 0.25

    cruising_filters: Tuple[RangeFilter, ...] = ()
    outlier_rules: Tuple[OutlierRule, ...] = ()
    imputation_plan: ImputationPlan = field(default_factory=ImputationPlan)
    final_numeric_strategy: str = "median"
    final_non_numeric_strategy: str = "most_frequent"
    tonnage_caps: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class DataCleaner:
    """Configurable, production-friendly cleaning pipeline."""

    def __init__(self, config: CleaningConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return df

    def _normalize_time_index(self, df):
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

    def _handle_duplicate_timestamps(self, df):
        strategy = (self.config.duplicate_timestamp_strategy or "mean").lower()
        if strategy == "error":
            raise ValueError("Duplicate timestamps found in index.")
        if strategy in {"first", "last"}:
            return df[~df.index.duplicated(keep=strategy)]
        num_cols = df.select_dtypes(include=[np.number]).columns
        non_num = [c for c in df.columns if c not in num_cols]
        agg: Dict[str, str] = {c: "mean" for c in num_cols}
        agg.update({c: "first" for c in non_num})
        out = df.groupby(df.index).agg(agg)
        out.index.name = self.config.time_index_name
        return out

    def _normalize_columns(self, df):
        if self.config.uppercase_columns:
            df.columns = df.columns.map(lambda c: c.upper() if isinstance(c, str) else c)
        return df

    def _apply_schema(self, df):
        cfg = self.config
        if not cfg.keep_only_configured_columns:
            return df
        keep_cols = cfg.columns.keep_columns
        present = [c for c in keep_cols if c in df.columns]
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            msg = (f"{len(missing)} configured columns missing: "
                   f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
            if cfg.strict_schema:
                raise KeyError(msg)
            self.logger.warning(msg)
        return df[present]

    def _rename_mt_columns(self, df):
        if not self.config.rename_mt_to_calc_mt:
            return df
        rename_map: Dict[str, str] = {}
        for col in list(df.columns):
            if isinstance(col, str) and "_MT" in col and "_CALC_MT" not in col:
                new_col = col.replace("_MT", "_CALC_MT")
                if new_col in df.columns:
                    df[new_col] = df[new_col].combine_first(df[col])
                    df = df.drop(columns=[col])
                else:
                    rename_map[col] = new_col
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _drop_unnamed(self, df):
        if not self.config.drop_unnamed_columns:
            return df
        unnamed = [c for c in df.columns if isinstance(c, str) and "UNNAMED" in c.upper()]
        return df.drop(columns=unnamed) if unnamed else df

    def _fill_default_zeros(self, df):
        cfg = self.config
        for col in cfg.zero_fill_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        if cfg.zero_fill_name_contains:
            for col in df.columns:
                if isinstance(col, str) and any(s in col for s in cfg.zero_fill_name_contains):
                    df[col] = df[col].fillna(0)
        return df

    def _combine_sinter(self, df):
        cfg = self.config
        if not cfg.combine_sinter_sources:
            return df
        sp01, sp02, combined = cfg.sinter_sp01_col, cfg.sinter_sp02_col, cfg.sinter_combined_col
        if sp01 not in df.columns and sp02 not in df.columns:
            return df
        a = pd.to_numeric(df[sp01], errors="coerce").fillna(0) if sp01 in df.columns else 0
        b = pd.to_numeric(df[sp02], errors="coerce").fillna(0) if sp02 in df.columns else 0
        df[combined] = a + b
        drop = [c for c in (sp01, sp02) if c in df.columns]
        return df.drop(columns=drop) if drop else df

    def _drop_sparse_rows(self, df):
        frac = self.config.row_min_non_na_fraction
        return df.dropna(axis=0, thresh=int(np.ceil(frac * df.shape[1])))

    def _drop_unnecessary_columns(self, df):
        cols = [c for c in self.config.unnecessary_columns if c in df.columns]
        return df.drop(columns=cols) if cols else df

    def _add_features(self, df):
        cfg = self.config
        if not cfg.add_unit_cost_feature:
            return df
        if cfg.unit_cost_coke_rate_col not in df.columns or cfg.unit_cost_pci_rate_col not in df.columns:
            self.logger.warning("Unit cost feature skipped (missing columns).")
            return df
        coke = pd.to_numeric(df[cfg.unit_cost_coke_rate_col], errors="coerce")
        pci  = pd.to_numeric(df[cfg.unit_cost_pci_rate_col], errors="coerce")
        df[cfg.unit_cost_col] = (coke + cfg.unit_cost_pci_multiplier * pci) * cfg.unit_cost_unit_multiplier
        return df

    def _drop_high_nan_columns(self, df):
        thresh = self.config.col_max_nan_fraction
        nan_frac = df.isna().mean()
        drop_cols = nan_frac[nan_frac > thresh].index.tolist()
        if drop_cols:
            self.logger.info("Dropping %d high-NaN columns.", len(drop_cols))
            df = df.drop(columns=drop_cols)
        return df

    def _apply_row_filters(self, df):
        for rule in self.config.cruising_filters:
            if rule.column not in df.columns:
                continue
            s = pd.to_numeric(df[rule.column], errors="coerce")
            if rule.min_value is not None:
                df = df[s >= rule.min_value] if rule.min_inclusive else df[s > rule.min_value]
                s = pd.to_numeric(df[rule.column], errors="coerce")
            if rule.max_value is not None:
                df = df[s <= rule.max_value] if rule.max_inclusive else df[s < rule.max_value]
        return df

    def _apply_outlier_rules(self, df):
        for rule in self.config.outlier_rules:
            if rule.column not in df.columns:
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

    def _selective_imputation(self, df):
        plan = self.config.imputation_plan
        iter_cols: List[str] = [c for c in plan.iterative_base_columns if c in df.columns]
        if plan.include_temperature_columns:
            iter_cols.extend([c for c in self.config.columns.temp_params if c in df.columns])
        seen: set = set()
        iter_cols = [c for c in iter_cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
        if iter_cols:
            X = df[iter_cols].apply(pd.to_numeric, errors="coerce")
            imp = IterativeImputer(
                random_state=plan.iterative_random_state,
                max_iter=plan.iterative_max_iter,
            )
            df.loc[:, iter_cols] = pd.DataFrame(
                imp.fit_transform(X), columns=iter_cols, index=df.index
            )
        for col, strategy in (plan.simple_column_strategies or {}).items():
            if col not in df.columns:
                continue
            imp2 = SimpleImputer(strategy=strategy)
            df[col] = imp2.fit_transform(df[[col]]).ravel()
        return df

    def _apply_tonnage_caps(self, df):
        for col, cap in (self.config.tonnage_caps or {}).items():
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            df = df[s < cap]
        return df

    def _final_imputation(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_num  = [c for c in df.columns if c not in num_cols]
        if num_cols:
            df[num_cols] = SimpleImputer(strategy=self.config.final_numeric_strategy).fit_transform(df[num_cols])
        if non_num:
            df[non_num] = SimpleImputer(strategy=self.config.final_non_numeric_strategy).fit_transform(df[non_num])
        return df


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def build_default_config() -> CleaningConfig:
    """Build the default :class:`CleaningConfig` for BF2 ML dataset cleaning."""
    rm_params = (
        'COKE_VM%', 'COKE_ASH%', 'COKE_IM%', 'COKE_FC%', 'COKE_MOIST%', 'COKE_CALC_MT',
        'NUTCOKE_MOIST%', 'NUTCOKE_VM%', 'NUTCOKE_IM%', 'NUTCOKE_FC%', 'NUTCOKE_ASH%',
        'NUTCOKE_CALC_MT',
        'PCI_2_FC%', 'PCI_2_ASH%', 'PCI_2_VM%', 'PCI_2_IM%', 'PCI_2_CALC_MT',
        'SINTER_SP_02_COLD_STRENGTH_AI', 'SINTER_SP_02_COLD_STRENGTH_TI',
        'SINTER_SP_02_HOT_STRENGTH_RI', 'SINTER_SP_02_HOT_STRENGTH_RDI',
        'SINTER_SP_02_P%', 'SINTER_SP_02_SIO2%', 'SINTER_SP_02_MGO%', 'SINTER_SP_02_FEO%',
        'SINTER_SP_02_AL2O3%', 'SINTER_SP_02_NA2O%', 'SINTER_SP_02_TIO2%', 'SINTER_SP_02_FE(T)%',
        'SINTER_SP_02_CAO%', 'SINTER_SP_02_K2O%', 'SINTER_SP_02_BASICITY', 'SINTER_SP_02_MNO%',
        'SINTER_CALC_MT',
        'FLUX_TM%', 'FLUX_SIO2%', 'FLUX_FE2O3%', 'FLUX_AL2O3%', 'FLUX_LOI%',
        'FLUX_MGO%', 'FLUX_CAO%', 'FLUX_CALC_MT',
        'GEOMIN TYPE', 'ORE_FE(T)%', 'ORE_LOI%', 'ORE_TM%', 'ORE_MGO%', 'ORE_NA2O%',
        'ORE_P%', 'ORE_TIO2%', 'ORE_AL2O3%', 'ORE_SIO2%', 'ORE_MNO%', 'ORE_CAO%',
        'ORE_K2O%', 'ORE_CALC_MT',
        'LLOYDS_PELLET_PCT_SIO2', 'LLOYDS_PELLET_PCT_AL2O3', 'LLOYDS_PELLET_PCT_FE2O3',
        'LLOYDS_PELLET_PCT_MGO', 'LLOYDS_PELLET_PCT_TIO2', 'LLOYDS_PELLET_PCT_NA2O',
        'TOTAL_PELLET_CALC_MT', 'LLOYDS_PELLET_PCT_K2O', 'LLOYDS_PELLET_PCT_TM',
        'LLOYDS_PELLET_PCT_LOI', 'LLOYDS_PELLET_PCT_MNO', 'LLOYDS_PELLET_PCT_CAO',
        'LLOYDS_PELLET_PCT_P',
        'DAILYPRODUCTION', 'CHARGES/HRS.', 'STOCKRODLEVEL',
    )
    hm_slag_params = (
        'CHEM_PCT_C', 'CHEM_PCT_CR', 'CHEM_PCT_FE', 'CHEM_PCT_MN', 'CHEM_PCT_P',
        'CHEM_PCT_S', 'CHEM_PCT_SI', 'CHEM_PCT_TI', 'SLAG_BASICITY', 'SLAG_PCT_AL2O3',
        'SLAG_PCT_CAO', 'SLAG_PCT_FEO', 'SLAG_PCT_K2O', 'SLAG_PCT_MGO', 'SLAG_PCT_MNO',
        'SLAG_PCT_NA2O', 'SLAG_PCT_S', 'SLAG_PCT_SIO2', 'SLAG_PCT_TIO2',
        'SLAG_T_BASICITY', 'HMT_GT_1480C',
    )
    bd_params = (
        'COKE_DISCHARGE_TIME', 'WEIGHTED_COKE_ANGLE', 'TOTAL_COKE_PORTIONS',
        'NON_COKE_DISCHARGE_TIME', 'WEIGHTED_NON_COKE_ANGLE', 'TOTAL_NON_COKE_PORTIONS',
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
        'FURNACETOPGASANALYSISCO2ETACO', 'COKE RATE KG/THM', 'PRODUCTIONTONNESPERHR',
    )
    prcs_params = (
        'HOT BLAST PRESSUREBAR', 'TOPPRESSUREBAR', 'DIFFERENTIAL PRESSURETOTALBAR',
        'HOT BLAST TEMP.OC', 'OXYGENFLOWNM3/HR.', 'STEAMKGS/HR.', 'RAFTOC',
        'PERMEABILITYKGS/HR.', 'TUYEREVELOCITYM/S', 'HOT BLAST VOLUMENM3/HR.',
        'O2 ENRICHMENT %', 'FURNACE TOP GAS ANALYSISCO2%', 'BOTTOMBAR',
        'FURNACE TOP GAS ANALYSISONLINE (ANALYZER)CO%', 'TOPBAR',
    )
    proxy_params = ('FURNACE TOP GAS ANALYSISH2%',)

    col_groups = ColumnGroups(
        rm_params=rm_params, hm_slag_params=hm_slag_params, bd_params=bd_params,
        temp_params=temp_params, op_params=op_params, prcs_params=prcs_params,
        proxy_params=proxy_params,
    )
    cruising_filters = (
        RangeFilter("HOT BLAST VOLUMENM3/HR.", min_value=90000, min_inclusive=False),
        RangeFilter("ACTUALKG/THM.", min_value=70, min_inclusive=False),
        RangeFilter("FURNACETOPGASANALYSISCO2ETACO", min_value=38, max_value=47, min_inclusive=False, max_inclusive=False),
        RangeFilter("PRODUCTIONTONNESPERHR", min_value=60, min_inclusive=False),
        RangeFilter("ACT. FUEL RATEKG/THM.", min_value=100, max_value=670, min_inclusive=False, max_inclusive=False),
    )
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
    imputation_plan = ImputationPlan(
        iterative_base_columns=(
            "RAFTOC", "PCI_2_CALC_MT", "SINTER_SP_02_HOT_STRENGTH_RI",
            "DIFFERENTIAL PRESSURETOTALBAR", "O2 ENRICHMENT %",
            "OXYGENFLOWNM3/HR.", "PRODUCTIONTONNESPERHR",
        ),
        include_temperature_columns=True,
        iterative_random_state=0,
        iterative_max_iter=10,
        simple_column_strategies={"CHARGES/HRS.": "most_frequent"},
    )
    tonnage_caps = {
        "COKE_CALC_MT": 55, "NUTCOKE_CALC_MT": 20, "PCI_2_CALC_MT": 30,
        "SINTER_CALC_MT": 200, "FLUX_CALC_MT": 10,
        "ORE_CALC_MT": 150, "TOTAL_PELLET_CALC_MT": 150,
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
