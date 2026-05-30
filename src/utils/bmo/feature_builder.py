"""Feature construction for BMO fuel-cost inference.

This module translates candidate ore quantities, chemistry, process context,
and historical lag data into the exact feature layout expected by the BMO V4
fuel-cost model artifacts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from domain.optimization_runtime import (
    FeatureBuildResult,
)
from domain.optimization_runtime import (
    normalize_feature_name as _normalize_feature_name,
)
from utils.bmo.types import OreInput


def normalize_feature_name(name: str) -> str:
    """
    Normalize a raw feature name using the shared runtime convention.

    The BMO payload mixes training names, process names, and normalized aliases.
    Reusing the shared normalizer keeps lookups consistent with the runtime
    feature builder and avoids duplicate alias rules.

    Args:
         - name: str - Feature name from model metadata, payload, or history.

    Returns:
         - return str - Lowercase underscore-normalized feature key.
    """

    return _normalize_feature_name(name)


_BMO_LAG_PATTERN = re.compile(r"(.+)_lag(\d+)(?:_\(([^)]+)\))?$")


_TRAINING_ALIASES: dict[str, str] = {
    "ACTUALKG/THM.": "PCI_KG/THM",
    "TOPBAR": "TOPPRESSUREBAR",
    "PCI_2_ASH%": "PCI_ASH%",
    "PCI_2_FC%": "PCI_FC%",
    "PCI_2_IM%": "PCI_IM%",
    "PCI_2_VM%": "PCI_VM%",
    "PCI_2_CALC_MT": "PCI_CALC_MT",
    "SINTER_SP_02_AL2O3%": "SINTER_AL2O3%",
    "SINTER_SP_02_CAO%": "SINTER_CAO%",
    "SINTER_SP_02_COLD_STRENGTH_AI": "SINTER_COLD_STRENGTH_AI",
    "SINTER_SP_02_COLD_STRENGTH_TI": "SINTER_COLD_STRENGTH_TI",
    "SINTER_SP_02_FE(T)%": "SINTER_FE(T)%",
    "SINTER_SP_02_FEO%": "SINTER_FEO%",
    "SINTER_SP_02_HOT_STRENGTH_RDI": "SINTER_HOT_STRENGTH_RDI",
    "SINTER_SP_02_HOT_STRENGTH_RI": "SINTER_HOT_STRENGTH_RI",
    "SINTER_SP_02_K2O%": "SINTER_K2O%",
    "SINTER_SP_02_MGO%": "SINTER_MGO%",
    "SINTER_SP_02_MNO%": "SINTER_MNO%",
    "SINTER_SP_02_NA2O%": "SINTER_NA2O%",
    "SINTER_SP_02_P%": "SINTER_P%",
    "SINTER_SP_02_SIO2%": "SINTER_SIO2%",
    "SINTER_SP_02_BASICITY": "SINTER_BASICITY",
    "SINTER_SP_02_TIO2%": "SINTER_TIO2%",
    "LLOYDS_PELLET_PCT_CAO": "PELLET_PCT_CAO",
    "LLOYDS_PELLET_PCT_AL2O3": "PELLET_PCT_AL2O3",
    "LLOYDS_PELLET_PCT_FE2O3": "PELLET_PCT_FE2O3",
    "LLOYDS_PELLET_PCT_K2O": "PELLET_PCT_K2O",
    "LLOYDS_PELLET_PCT_LOI": "PELLET_PCT_LOI",
    "LLOYDS_PELLET_PCT_MGO": "PELLET_PCT_MGO",
    "LLOYDS_PELLET_PCT_MNO": "PELLET_PCT_MNO",
    "LLOYDS_PELLET_PCT_NA2O": "PELLET_PCT_NA2O",
    "LLOYDS_PELLET_PCT_P": "PELLET_PCT_P",
    "LLOYDS_PELLET_PCT_SIO2": "PELLET_PCT_SIO2",
    "LLOYDS_PELLET_PCT_TIO2": "PELLET_PCT_TIO2",
    "LLOYDS_PELLET_PCT_TM": "PELLET_PCT_TM",
    "BELLY - 15162_TEMP. OCAT-07DEG. OC": "BELLY_TEMP_A",
    "BELLY - 15162_TEMP. OC.1BT-12DEG. OC": "BELLY_TEMP_B",
    "BELLY - 15162_TEMP. OC.2CT-17DEG. OC": "BELLY_TEMP_C",
    "BELLY - 15162_TEMP. OC.3DT-03DEG. OC": "BELLY_TEMP_D",
    "BOSH - 12975_TEMP. OCAT-07DEG. OC": "BOSH_TEMP_A",
    "BOSH - 12975_TEMP. OC.1BT-12DEG. OC": "BOSH_TEMP_B",
    "BOSH - 12975_TEMP. OC.2CT-17DEG. OC": "BOSH_TEMP_C",
    "BOSH - 12975_TEMP. OC.3DT-03DEG. OC": "BOSH_TEMP_D",
    "HEARTH PAD CENTER_TEMP. OCA4.3 MTR.DEG. OC": "HEARTH_TEMP_A",
    "HEARTH PAD CENTER_TEMP. OC.1B5.4 MTR.DEG. OC": "HEARTH_TEMP_B",
    "HEARTH PAD CENTER_TEMP. OC.2C5.7 MTR.DEG. OC": "HEARTH_TEMP_C",
    "HEARTH PAD CENTER_TEMP. OC.3D6.1 MTR.DEG. OC": "HEARTH_TEMP_D",
    "LOWER STACK - 18660_TEMP. OCAT-07DEG. OC": "LOWER_STACK_TEMP_A",
    "LOWER STACK - 18660_TEMP. OC.1BT-12DEG. OC": "LOWER_STACK_TEMP_B",
    "LOWER STACK - 18660_TEMP. OC.2CT-17DEG. OC": "LOWER_STACK_TEMP_C",
    "LOWER STACK - 18660_TEMP. OC.3DT-03DEG. OC": "LOWER_STACK_TEMP_D",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.4AVG.DEG. OC": "FTG_UPTAKE_TEMP_AVG",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.1BT-12DEG. OC": "FTG_UPTAKE_TEMP_B",
    "FURNACE TOP GAS_UPTAKE TEMP. OCAT-16DEG. OC": "FTG_UPTAKE_TEMP_A",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.2CT-08DEG. OC": "FTG_UPTAKE_TEMP_C",
    "FURNACE TOP GAS_UPTAKE TEMP. OC.3DT-04DEG. OC": "FTG_UPTAKE_TEMP_D",
}


_CHEMISTRY_TO_TRAINING_SUFFIX: dict[str, str] = {
    "fe_t_pct": "FE(T)%",
    "moisture_pct": "TM%",
    "feo_pct": "FEO%",
    "sio2_pct": "SIO2%",
    "al2o3_pct": "AL2O3%",
    "cao_pct": "CAO%",
    "mgo_pct": "MGO%",
    "mno_pct": "MNO%",
    "tio2_pct": "TIO2%",
    "p_pct": "P%",
    "na2o_pct": "NA2O%",
    "k2o_pct": "K2O%",
}


@dataclass(frozen=True)
class ParsedLagFeature:
    """
    Represent a BMO lagged training feature.

    BMO V4 feature names encode both row lag and, for some columns, the physical
    impact channel. Keeping the parsed pieces in a small value object makes the
    feature resolver easier to reason about and test.

    Args:
         - base_name: str - Feature name before the lag suffix.
         - lag_steps: int - Number of lag steps encoded in the feature name.
         - impact_label: str | None - Optional physical impact label.

    Returns:
         - return ParsedLagFeature - Parsed lag metadata.
    """

    base_name: str
    lag_steps: int
    impact_label: str | None = None


def parse_bmo_lag_feature_name(feature_name: str) -> ParsedLagFeature | None:
    """
    Parse BMO lag names including dual-impact suffixes.

    Supports simple names such as ``STOCKRODLEVEL_lag1`` and BMO physical
    impact names such as ``ORE_3_PCT_lag4_(MeltImpact)``.

    Args:
         - feature_name: str - Feature name that may include lag metadata.

    Returns:
         - return ParsedLagFeature | None - Parsed lag feature or None.
    """

    match = _BMO_LAG_PATTERN.match(str(feature_name))
    if not match:
        return None
    return ParsedLagFeature(
        base_name=match.group(1),
        lag_steps=int(match.group(2)),
        impact_label=match.group(3),
    )


def _is_sinter(ore: OreInput) -> bool:
    """
    Identify whether an ore input represents sinter material.

    The model feature payload treats sinter separately from CLO and pellet.
    This helper centralizes the loose naming check used by both display names
    and material keys from the ore mapping file.

    Args:
         - ore: OreInput - Ore input to classify.

    Returns:
         - return bool - True when the ore material or display name is sinter.
    """

    material_key = str(ore.metadata.get("material_key", "")).lower()
    return "sinter" in material_key or "sinter" in ore.display_name.lower()


def _is_pellet(ore: OreInput) -> bool:
    """
    Identify whether an ore input represents pellet material.

    Pellet burden has its own model features and ratios. This helper keeps the
    classification logic consistent whether the source name comes from display
    text or mapping metadata.

    Args:
         - ore: OreInput - Ore input to classify.

    Returns:
         - return bool - True when the ore material or display name is pellet.
    """

    material_key = str(ore.metadata.get("material_key", "")).lower()
    return "pellet" in material_key or "pellet" in ore.display_name.lower()


def _ore_slot_number(ore: OreInput) -> int | None:
    """
    Extract the training ore slot number from an ore material key.

    The trained model expects CLO chemistry and percentages in numbered
    ``ORE_1`` through ``ORE_12`` slots. This parser maps configured material keys
    back to those slot numbers.

    Args:
         - ore: OreInput - Ore input containing material metadata.

    Returns:
         - return int | None - Ore slot number from 1 to 12, or None.
    """

    material_key = str(ore.metadata.get("material_key", "")).lower()
    match = re.fullmatch(r"ore_(\d+)", material_key)
    if not match:
        return None
    slot = int(match.group(1))
    return slot if 1 <= slot <= 12 else None


def _weighted_chemistry(
    ores: list[OreInput], quantities_mt: Mapping[str, float], attr: str
) -> float | None:
    """
    Compute quantity-weighted chemistry for selected ore inputs.

    Candidate blend chemistry is sent to the fuel-cost model in the training
    schema. This helper aggregates the selected ore chemistry using the same
    candidate quantities being evaluated by LP or DE.

    Args:
         - ores: list[OreInput] - Ores contributing to the weighted average.
         - quantities_mt: Mapping[str, float] - Ore quantities keyed by ore id.
         - attr: str - Chemistry attribute name to average.

    Returns:
         - return float | None - Weighted chemistry value, or None with no quantity.
    """

    total_qty = 0.0
    weighted = 0.0
    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0) or 0.0)
        if qty <= 0:
            continue
        value = getattr(ore.chemistry, attr, None)
        if value is None:
            continue
        total_qty += qty
        weighted += qty * float(value or 0.0)
    if total_qty <= 0:
        return None
    return weighted / total_qty


def _candidate_blend_features(
    quantities_mt: Mapping[str, float],
    ores: list[OreInput],
) -> dict[str, float]:
    """
    Build training-schema burden features from a candidate BMO blend.

    These features let the BMO V4 fuel model see the candidate burden mix in
    the same naming style used during training. Moisture is included through
    ``TM%`` fields when it is present in the ore chemistry record.

    Args:
         - quantities_mt: Mapping[str, float] - Candidate ore quantities keyed by ore id.
         - ores: list[OreInput] - Ores selected for optimization.

    Returns:
         - return dict[str, float] - Raw burden features matching BMO V4 training names.
    """

    payload: dict[str, float] = {}
    ore_quantities_by_slot = {slot: 0.0 for slot in range(1, 13)}
    sinter_qty = 0.0
    pellet_qty = 0.0
    slot_ores: list[OreInput] = []

    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0) or 0.0)
        if _is_sinter(ore):
            sinter_qty += qty
            continue
        if _is_pellet(ore):
            pellet_qty += qty
            continue

        slot = _ore_slot_number(ore)
        if slot is not None:
            ore_quantities_by_slot[slot] += qty
            slot_ores.append(ore)

    ore_qty = float(sum(ore_quantities_by_slot.values()))
    payload["ORE_CALC_MT"] = ore_qty
    payload["SINTER_CALC_MT"] = sinter_qty
    payload["TOTAL_PELLET_CALC_MT"] = pellet_qty

    for slot, qty in ore_quantities_by_slot.items():
        payload[f"ORE_{slot}_PCT"] = (qty / ore_qty * 100.0) if ore_qty > 0 else 0.0

    payload["SINTER_CLO_RATIO"] = (sinter_qty / ore_qty) if ore_qty > 0 else 0.0
    payload["PELLET_CLO_RATIO"] = (pellet_qty / ore_qty) if ore_qty > 0 else 0.0

    for attr, suffix in _CHEMISTRY_TO_TRAINING_SUFFIX.items():
        ore_value = _weighted_chemistry(slot_ores, quantities_mt, attr)
        if ore_value is not None:
            payload[f"ORE_{suffix}"] = ore_value

    for ore in ores:
        qty = float(quantities_mt.get(ore.ore_id, 0.0) or 0.0)
        if qty <= 0:
            continue
        if _is_sinter(ore):
            for attr, suffix in _CHEMISTRY_TO_TRAINING_SUFFIX.items():
                payload[f"SINTER_SP_02_{suffix}"] = float(
                    getattr(ore.chemistry, attr, 0.0) or 0.0
                )
        elif _is_pellet(ore):
            for attr, suffix in _CHEMISTRY_TO_TRAINING_SUFFIX.items():
                pellet_suffix = suffix.replace("%", "")
                payload[f"LLOYDS_PELLET_PCT_{pellet_suffix}"] = float(
                    getattr(ore.chemistry, attr, 0.0) or 0.0
                )

    return payload


def build_feature_payload(
    quantities_mt: Mapping[str, float],
    ore_display_name_by_id: Mapping[str, str],
    process_context: Mapping[str, Any] | None = None,
    ores: list[OreInput] | None = None,
) -> dict[str, float]:
    """
    Build a raw feature payload for BMO fuel-cost inference.

    The payload includes generic quantity/share features for the fallback
    formula and, when *ores* are supplied, the BMO V4 training-schema burden
    features used by the XGBoost model.

    Args:
         - quantities_mt: Mapping[str, float] - Candidate ore quantities keyed by ore id.
         - ore_display_name_by_id: Mapping[str, str] - Ore display names keyed by ore id.
         - process_context: Mapping[str, Any] | None - Latest process variables.
         - ores: list[OreInput] | None - Ore inputs used to build training features.

    Returns:
         - return dict[str, float] - Raw feature payload for model service inference.
    """

    payload: dict[str, float] = {}

    total_qty = float(sum(float(v) for v in quantities_mt.values()))
    sinter_qty = 0.0

    for ore_id, qty in quantities_mt.items():
        qty = float(qty)
        display = ore_display_name_by_id.get(ore_id, ore_id)
        if "sinter" in display.lower():
            sinter_qty += qty

        ore_slug = normalize_feature_name(display)
        payload[f"{ore_slug}_qty_mt"] = qty
        payload[f"{ore_slug}_share_pct"] = (
            (qty / total_qty * 100.0) if total_qty > 0 else 0.0
        )

    sinter_share_pct = (sinter_qty / total_qty * 100.0) if total_qty > 0 else 0.0
    payload["sinter_qty_mt"] = sinter_qty
    payload["ore_qty_mt"] = max(0.0, total_qty - sinter_qty)
    payload["sinter_share_pct"] = sinter_share_pct
    payload["ore_share_pct"] = max(0.0, 100.0 - sinter_share_pct)
    payload["total_burden_qty_mt"] = total_qty

    if ores:
        payload.update(_candidate_blend_features(quantities_mt, ores))

    if process_context:
        for key, value in process_context.items():
            if value is None:
                continue
            try:
                payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
            payload[normalize_feature_name(str(key))] = payload[str(key)]

    return payload


def _numeric_or_none(value: Any) -> float | None:
    """
    Convert a value into float when it is present and numeric.

    Feature resolution tries payload, history, and defaults in order. Returning
    ``None`` for invalid values lets the caller continue to the next source
    without raising on blanks or non-numeric labels.

    Args:
         - value: Any - Candidate value from payload or history.

    Returns:
         - return float | None - Numeric float value, or None when missing/invalid.
    """

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_lookup(
    payload: Mapping[str, Any], key: str, aliases: Mapping[str, str]
) -> tuple[float | None, str | None]:
    """
    Look up a feature value in the candidate payload using aliases and normalization.

    Candidate payload keys can be raw training names or normalized process names.
    This helper checks both forms and returns a source label for diagnostics
    when a value is resolved.

    Args:
         - payload: Mapping[str, Any] - Candidate blend and process feature payload.
         - key: str - Requested training feature name.
         - aliases: Mapping[str, str] - Training-name aliases to alternate names.

    Returns:
         - return tuple[float | None, str | None] - Resolved value and source label.
    """

    candidates = [key]
    if key in aliases:
        candidates.append(aliases[key])

    norm_payload = {
        normalize_feature_name(str(name)): value for name, value in payload.items()
    }
    for candidate in candidates:
        if candidate in payload:
            value = _numeric_or_none(payload[candidate])
            if value is not None:
                return value, f"payload:{candidate}"
        norm_name = normalize_feature_name(candidate)
        if norm_name in norm_payload:
            value = _numeric_or_none(norm_payload[norm_name])
            if value is not None:
                return value, f"payload_norm:{candidate}"
    return None, None


def _history_lookup(
    history_df: pd.DataFrame | None,
    key: str,
    aliases: Mapping[str, str],
    *,
    lag_steps: int = 0,
) -> tuple[float | None, str | None]:
    """
    Look up a current or lagged feature value from historical process data.

    The model uses the static dataset for lagged operating context. This helper
    finds the requested column by raw or normalized name, drops missing values,
    and returns the row matching the requested lag step.

    Args:
         - history_df: pd.DataFrame | None - Historical process data frame.
         - key: str - Requested training feature name.
         - aliases: Mapping[str, str] - Training-name aliases to alternate names.
         - lag_steps: int - Number of rows to lag from the latest non-null value.

    Returns:
         - return tuple[float | None, str | None] - Resolved value and source label.
    """

    if history_df is None or history_df.empty:
        return None, None

    candidates = [key]
    if key in aliases:
        candidates.append(aliases[key])

    columns_norm = {normalize_feature_name(str(c)): str(c) for c in history_df.columns}
    for candidate in candidates:
        column = candidate if candidate in history_df.columns else None
        if column is None:
            column = columns_norm.get(normalize_feature_name(candidate))
        if not column:
            continue

        series = pd.to_numeric(history_df[column], errors="coerce").dropna()
        if series.empty:
            continue
        idx = max(0, int(lag_steps)) + 1
        if len(series) < idx:
            continue
        return float(series.iloc[-idx]), f"history:{column}:lag{lag_steps}"
    return None, None


def _latest_history_timestamp(history_df: pd.DataFrame | None) -> pd.Timestamp | None:
    """
    Return the latest valid timestamp from the history frame index.

    Temporal model features depend on the history index rather than a normal
    numeric column. Invalid or missing indexes return ``None`` so temporal
    values can fall through to configured defaults.

    Args:
         - history_df: pd.DataFrame | None - Historical process data frame.

    Returns:
         - return pd.Timestamp | None - Latest valid timestamp, or None.
    """

    if history_df is None or history_df.empty:
        return None
    try:
        index = pd.to_datetime(history_df.index, errors="coerce")
        index = index[~pd.isna(index)]
        if len(index) == 0:
            return None
        return pd.Timestamp(index[-1])
    except Exception:
        return None


def _temporal_feature_value(
    feature: str, history_df: pd.DataFrame | None
) -> tuple[float | None, str | None]:
    """
    Resolve engineered temporal features from the history frame.

    The trained model includes calendar and trend fields that are not explicit
    process columns. This resolver derives those values from the history index
    when enough timestamp context is available.

    Args:
         - feature: str - Requested temporal feature name.
         - history_df: pd.DataFrame | None - Historical process data frame.

    Returns:
         - return tuple[float | None, str | None] - Temporal value and source label.
    """

    timestamp = _latest_history_timestamp(history_df)
    if feature == "trend_index" and history_df is not None and not history_df.empty:
        return float(max(len(history_df) - 1, 0)), "temporal:history_length"
    if timestamp is None:
        return None, None
    if feature == "month":
        return float(timestamp.month), "temporal:month"
    if feature == "week_of_year":
        return float(timestamp.isocalendar().week), "temporal:week_of_year"
    if feature == "day_of_year":
        return float(timestamp.dayofyear), "temporal:day_of_year"
    return None, None


def default_candidate_lag_bases(feature_payload: Mapping[str, Any]) -> set[str]:
    """
    Return feature bases whose physical-lag columns should use the candidate.

    These are the burden/blend-dependent model inputs. Process and fuel
    operating variables still resolve from history for their lagged values.

    Args:
         - feature_payload: Mapping[str, Any] - Candidate blend and process features.

    Returns:
         - return set[str] - Feature base names eligible for candidate lag override.
    """

    bases: set[str] = set()
    for key in feature_payload:
        text = str(key)
        if (
            re.fullmatch(r"ORE_\d+_PCT", text)
            or text in {"ORE_CALC_MT", "SINTER_CALC_MT", "TOTAL_PELLET_CALC_MT"}
            or text in {"SINTER_CLO_RATIO", "PELLET_CLO_RATIO"}
            or text.startswith("ORE_")
            or text.startswith("SINTER_SP_02_")
            or text.startswith("LLOYDS_PELLET_")
        ):
            bases.add(text)
    return bases


def build_bmo_v4_feature_frame(
    *,
    feature_payload: Mapping[str, Any],
    history_df: pd.DataFrame | None,
    expected_features: list[str],
    default_values: Mapping[str, float] | None = None,
    candidate_lag_bases: set[str] | None = None,
    aliases: Mapping[str, str] | None = None,
) -> FeatureBuildResult:
    """
    Build one raw BMO V4 feature row in scaler order.

    The V4 fuel-cost model requires a raw 104-column row before scaling. This
    function resolves each feature from the candidate blend payload, historical
    data, temporal context, or scaler-mean defaults. Candidate-sensitive burden
    features can intentionally override their lagged ``GasImpact`` and
    ``MeltImpact`` columns so the optimizer evaluates planned blend effects.

    Args:
         - feature_payload: Mapping[str, Any] - Candidate blend and process features.
         - history_df: pd.DataFrame | None - Historical process data for lag features.
         - expected_features: list[str] - Raw feature order expected by the scaler.
         - default_values: Mapping[str, float] | None - Defaults for unresolved features.
         - candidate_lag_bases: set[str] | None - Bases whose lag columns use candidate values.
         - aliases: Mapping[str, str] | None - Optional training-name alias overrides.

    Returns:
         - return FeatureBuildResult - Feature frame, imputation lists, and source map.
    """

    aliases = dict(_TRAINING_ALIASES if aliases is None else aliases)
    defaults = dict(default_values or {})
    lag_override_bases = (
        set(candidate_lag_bases)
        if candidate_lag_bases is not None
        else default_candidate_lag_bases(feature_payload)
    )

    values: list[float] = []
    missing_features: list[str] = []
    imputed_features: list[str] = []
    source_map: dict[str, str] = {}

    for feature in expected_features:
        value: float | None = None
        source: str | None = None

        temporal_value, temporal_source = _temporal_feature_value(feature, history_df)
        if temporal_value is not None:
            value, source = temporal_value, temporal_source

        parsed = parse_bmo_lag_feature_name(feature)
        if value is None and parsed is not None:
            if parsed.base_name in lag_override_bases:
                value, source = _payload_lookup(
                    feature_payload, parsed.base_name, aliases
                )
                if value is not None:
                    source = f"candidate_lag_override:{parsed.base_name}"
            if value is None:
                value, source = _history_lookup(
                    history_df,
                    parsed.base_name,
                    aliases,
                    lag_steps=parsed.lag_steps,
                )

        if value is None and parsed is None:
            value, source = _payload_lookup(feature_payload, feature, aliases)
            if value is None:
                value, source = _history_lookup(
                    history_df, feature, aliases, lag_steps=0
                )

        if value is None:
            missing_features.append(feature)
            value = float(defaults.get(feature, 0.0))
            source = "default"
            imputed_features.append(feature)

        values.append(float(value))
        source_map[feature] = str(source or "unknown")

    vector_df = pd.DataFrame([values], columns=expected_features, dtype=float)
    return FeatureBuildResult(
        vector_df=vector_df,
        missing_features=missing_features,
        imputed_features=imputed_features,
        source_map=source_map,
    )
