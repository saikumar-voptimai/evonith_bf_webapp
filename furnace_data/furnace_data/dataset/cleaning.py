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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
except Exception:
    pass

from sklearn.impute import IterativeImputer, SimpleImputer

from furnace_data.config import load_config


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
    extra_keep_columns: Tuple[str, ...] = ()

    @property
    def final_columns(self) -> List[str]:
        return list(
            self.rm_params + self.hm_slag_params + self.bd_params
            + self.temp_params + self.op_params + self.prcs_params + self.proxy_params
        )

    @property
    def keep_columns(self) -> List[str]:
        keep = self.final_columns.copy()
        for extra in self.extra_keep_columns:
            if extra not in keep:
                keep.append(extra)
        return keep


@dataclass(frozen=True)
class CleaningConfig:
    """All parameters required for the cleaning pipeline."""
    columns: ColumnGroups

    time_column: Optional[str] = None
    time_index_name: str = "time"
    floor_freq: str = "h"
    sort_index: bool = True
    duplicate_timestamp_strategy: str = "mean"

    uppercase_columns: bool = True
    rename_mt_to_calc_mt: bool = True

    keep_only_configured_columns: bool = True
    strict_schema: bool = False

    row_min_non_na_fraction: float = 0.5
    col_max_nan_fraction: float = 0.30
    
    zero_fill_columns: Tuple[str, ...] = ()
    zero_fill_name_contains: Tuple[str, ...] = ()

    combine_sinter_sources: bool = False
    sinter_sp01_col: str = ""
    sinter_sp02_col: str = ""
    sinter_combined_col: str = ""

    drop_unnamed_columns: bool = True
    unnecessary_columns: Tuple[str, ...] = ()

    add_unit_cost_feature: bool = True
    unit_cost_col: str = ""
    unit_cost_coke_rate_col: str = ""
    unit_cost_pci_rate_col: str = ""
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
        before = len(df)
        thresh = int(np.ceil(frac * df.shape[1]))
        df = df.dropna(axis=0, thresh=thresh)
        dropped = before - len(df)
        if dropped:
            self.logger.warning(
                "Dropped %d sparse rows (<%d%% non-NaN values required across %d cols).",
                dropped, int(frac * 100), df.shape[1],
            )
        return df

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
            self.logger.warning(
                "Dropping %d high-NaN columns (>%d%% missing): %s",
                len(drop_cols), int(thresh * 100), drop_cols,
            )
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
            observed_cols = [col for col in iter_cols if X[col].notna().any()]
            if observed_cols:
                imp = IterativeImputer(
                    random_state=plan.iterative_random_state,
                    max_iter=plan.iterative_max_iter,
                )
                df.loc[:, observed_cols] = pd.DataFrame(
                    imp.fit_transform(X[observed_cols]),
                    columns=observed_cols,
                    index=df.index,
                )
        for col, strategy in (plan.simple_column_strategies or {}).items():
            if col not in df.columns:
                continue
            imp2 = SimpleImputer(strategy=strategy, keep_empty_features=True)
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
            # Columns that are entirely NaN after all filters have no signal;
            # drop them outright so they don't cause a shape mismatch in sklearn
            # (older sklearn without keep_empty_features drops them silently,
            # returning fewer columns than the key list, which raises ValueError).
            all_nan_num = [c for c in num_cols if df[c].isna().all()]
            if all_nan_num:
                self.logger.warning(
                    "Dropping %d all-NaN numeric columns before final imputation: %s",
                    len(all_nan_num), all_nan_num,
                )
                df = df.drop(columns=all_nan_num)
                num_cols = [c for c in num_cols if c not in all_nan_num]

            if num_cols:
                imputed = SimpleImputer(
                    strategy=self.config.final_numeric_strategy,
                    keep_empty_features=True,
                ).fit_transform(df[num_cols])
                # Wrap in DataFrame so pandas assignment is column-name-safe
                # regardless of pandas version or duplicate-column edge cases.
                df[num_cols] = pd.DataFrame(imputed, index=df.index, columns=num_cols)

        if non_num:
            all_nan_non = [c for c in non_num if df[c].isna().all()]
            if all_nan_non:
                self.logger.warning(
                    "Dropping %d all-NaN non-numeric columns before final imputation: %s",
                    len(all_nan_non), all_nan_non,
                )
                df = df.drop(columns=all_nan_non)
                non_num = [c for c in non_num if c not in all_nan_non]

            if non_num:
                imputed_non = SimpleImputer(
                    strategy=self.config.final_non_numeric_strategy,
                    keep_empty_features=True,
                ).fit_transform(df[non_num])
                df[non_num] = pd.DataFrame(imputed_non, index=df.index, columns=non_num)

        return df


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def _dataset_config() -> dict[str, Any]:
    return load_config("setting_ds_dv.yml")


def _dataset_rename_dict(config: dict[str, Any]) -> dict[str, str]:
    rename_dict = config.get("rename_dict", {}) or {}
    if not isinstance(rename_dict, dict):
        raise TypeError("setting_ds_dv.yml rename_dict must be a mapping.")
    return {str(key): str(value) for key, value in rename_dict.items()}


def _cleaning_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("cleaning")
    if not isinstance(settings, dict):
        raise KeyError("Missing cleaning section in setting_ds_dv.yml.")
    return settings


def _alias_for(rename_dict: dict[str, str], mapping_name: str) -> str:
    try:
        return rename_dict[mapping_name]
    except KeyError as exc:
        raise KeyError(
            f"Missing cleaning mapping '{mapping_name}' in setting_ds_dv.yml rename_dict."
        ) from exc


def _calc_alias_for(rename_dict: dict[str, str], mapping_name: str) -> str:
    alias = _alias_for(rename_dict, mapping_name)
    return alias.replace("_MT", "_CALC_MT")


def _aliases_for(rename_dict: dict[str, str], mapping_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(_alias_for(rename_dict, str(name)) for name in mapping_names)


def _column_spec_aliases(rename_dict: dict[str, str], spec: dict[str, Any] | None) -> tuple[str, ...]:
    if not spec:
        return ()
    aliases = list(_aliases_for(rename_dict, spec.get("alias_keys", []) or []))
    aliases.extend(
        _calc_alias_for(rename_dict, str(name))
        for name in spec.get("calc_alias_keys", []) or []
    )
    return tuple(dict.fromkeys(aliases))


def _range_filters(rename_dict: dict[str, str], rows: Sequence[dict[str, Any]]) -> tuple[RangeFilter, ...]:
    filters = []
    for row in rows:
        filters.append(
            RangeFilter(
                column=_alias_for(rename_dict, str(row["alias_key"])),
                min_value=row.get("min_value"),
                max_value=row.get("max_value"),
                min_inclusive=bool(row.get("min_inclusive", False)),
                max_inclusive=bool(row.get("max_inclusive", False)),
            )
        )
    return tuple(filters)


def _outlier_rules(rename_dict: dict[str, str], rows: Sequence[dict[str, Any]]) -> tuple[OutlierRule, ...]:
    rules = []
    for row in rows:
        rules.append(
            OutlierRule(
                column=_alias_for(rename_dict, str(row["alias_key"])),
                min_value=row.get("min_value"),
                max_value=row.get("max_value"),
                inclusive=bool(row.get("inclusive", True)),
            )
        )
    return tuple(rules)


def _simple_strategies(
    rename_dict: dict[str, str],
    rows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    return {
        _alias_for(rename_dict, str(row["alias_key"])): str(row["strategy"])
        for row in rows
    }


def _tonnage_caps(rename_dict: dict[str, str], rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        _alias_for(rename_dict, str(row["alias_key"])): float(row["max_value"])
        for row in rows
    }


def _unit_cost_aliases(rename_dict: dict[str, str], settings: dict[str, Any]) -> dict[str, Any]:
    unit_cost = settings.get("unit_cost", {}) or {}
    return {
        "enabled": bool(unit_cost.get("enabled", True)),
        "output": _alias_for(rename_dict, str(unit_cost["output_alias_key"])),
        "coke_rate": _alias_for(rename_dict, str(unit_cost["coke_rate_alias_key"])),
        "pci_rate": _alias_for(rename_dict, str(unit_cost["pci_rate_alias_key"])),
        "pci_multiplier": float(unit_cost.get("pci_multiplier", 0.53)),
        "unit_multiplier": float(unit_cost.get("unit_multiplier", 0.25)),
    }


def build_default_config() -> CleaningConfig:
    """Build the default :class:`CleaningConfig` for BF2 ML dataset cleaning."""
    full_config = _dataset_config()
    rename_dict = _dataset_rename_dict(full_config)
    settings = _cleaning_settings(full_config)
    groups = settings.get("column_groups", {}) or {}

    col_groups = ColumnGroups(
        rm_params=_aliases_for(rename_dict, groups.get("rm_params", []) or []),
        hm_slag_params=_aliases_for(rename_dict, groups.get("hm_slag_params", []) or []),
        bd_params=_aliases_for(rename_dict, groups.get("bd_params", []) or []),
        temp_params=_aliases_for(rename_dict, groups.get("temp_params", []) or []),
        op_params=_aliases_for(rename_dict, groups.get("op_params", []) or []),
        prcs_params=_aliases_for(rename_dict, groups.get("prcs_params", []) or []),
        proxy_params=_aliases_for(rename_dict, groups.get("proxy_params", []) or []),
        extra_keep_columns=_column_spec_aliases(
            rename_dict,
            settings.get("extra_keep_columns", {}) or {},
        ),
    )

    zero_fill = settings.get("zero_fill", {}) or {}
    sinter = settings.get("sinter", {}) or {}
    unit_cost = _unit_cost_aliases(rename_dict, settings)
    imputation = settings.get("imputation", {}) or {}

    return CleaningConfig(
        columns=col_groups,
        cruising_filters=_range_filters(
            rename_dict,
            settings.get("cruising_filters", []) or [],
        ),
        outlier_rules=_outlier_rules(
            rename_dict,
            settings.get("outlier_rules", []) or [],
        ),
        imputation_plan=ImputationPlan(
            iterative_base_columns=_aliases_for(
                rename_dict,
                imputation.get("iterative_base_alias_keys", []) or [],
            ),
            include_temperature_columns=bool(
                imputation.get("include_temperature_columns", True)
            ),
            iterative_random_state=int(imputation.get("iterative_random_state", 0)),
            iterative_max_iter=int(imputation.get("iterative_max_iter", 10)),
            simple_column_strategies=_simple_strategies(
                rename_dict,
                imputation.get("simple_strategies", []) or [],
            ),
        ),
        tonnage_caps=_tonnage_caps(
            rename_dict,
            settings.get("tonnage_caps", []) or [],
        ),
        row_min_non_na_fraction=float(settings.get("row_min_non_na_fraction", 0.5)),
        col_max_nan_fraction=float(settings.get("col_max_nan_fraction", 0.30)),
        zero_fill_columns=_column_spec_aliases(rename_dict, zero_fill),
        zero_fill_name_contains=tuple(zero_fill.get("name_contains", []) or []),
        combine_sinter_sources=bool(sinter.get("combine_sources", False)),
        sinter_sp01_col=_calc_alias_for(rename_dict, str(sinter["sp01_calc_alias_key"])),
        sinter_sp02_col=_alias_for(rename_dict, str(sinter["sp02_alias_key"])),
        sinter_combined_col=_alias_for(rename_dict, str(sinter["combined_alias_key"])),
        unnecessary_columns=_aliases_for(
            rename_dict,
            settings.get("unnecessary_alias_keys", []) or [],
        ),
        add_unit_cost_feature=unit_cost["enabled"],
        unit_cost_col=unit_cost["output"],
        unit_cost_coke_rate_col=unit_cost["coke_rate"],
        unit_cost_pci_rate_col=unit_cost["pci_rate"],
        unit_cost_pci_multiplier=unit_cost["pci_multiplier"],
        unit_cost_unit_multiplier=unit_cost["unit_multiplier"],
    )
