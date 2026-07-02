"""Streamlit page flow for the Blend Mix Optimizer.

This page wires BMO configuration, data context, fuel-cost model inference,
LP baseline optimization, nonlinear DE optimization, and result rendering into
one Streamlit workflow for ore blend planning.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

log = logging.getLogger(__name__)

from config.config_loader import load_config
from data.bmo import EvonithBmoContextProvider
from data.bmo.basicity_defaults import derive_basicity_bounds_from_static_dataset
from data.bmo.ore_editor_preferences import (
    apply_model_input_preferences,
    apply_ore_editor_preferences,
    load_ore_editor_preferences,
    save_model_input_preferences,
    save_ore_editor_preferences,
)
from data.ml.static_dataset_manager import StaticDatasetManager
from domain.optimization_runtime import build_runtime_config
from ui.streamlit_fragments import fragment, rerun_fragment
from ui.bmo import (
    apply_bmo_styles,
    build_dust_editor_df,
    build_flux_editor_df,
    build_fuel_ash_editor_df,
    build_ore_editor_df,
    render_blend_metrics,
    render_blend_table,
    render_diagnostics,
    render_dust_editor,
    render_flux_editor,
    render_fuel_ash_editor,
    render_header,
    render_hot_metal_chemistry,
    render_ore_editor,
    render_slag_balance_details,
    render_slag_balance_settings,
)
from utils.bmo import (
    DustInput,
    FluxInput,
    FuelAshInput,
    FuelUnitCostModelService,
    OreInput,
    SlagBalanceSettings,
    run_lp_baseline,
    run_nonlinear_optimizer,
    validate_selected_pellet_inputs,
)
from ui.bmo.editor_inputs import (
    dust_inputs_from_editor,
    float_from_row,
    flux_inputs_from_editor,
    fuel_ash_inputs_from_editor,
    slag_balance_settings_from_editor,
)
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.si_prediction import SiPredictionService
from utils.bmo.fuel_rates import get_recent_fuel_input_rates
from utils.session import is_logged_in

if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()


_resource_cache = st.cache_resource
_data_cache = st.cache_data


@_data_cache(show_spinner=False)
def _load_bmo_config_cached(mtime_ns: int) -> dict[str, Any]:
    cfg = load_config("setting_bmo.yml")
    return cfg.get("bmo", {})


def _get_bmo_config() -> dict[str, Any]:
    """
    Load the BMO settings block from Streamlit app configuration.

    The page keeps all optimizer, model, data-source, and UI defaults in the
    dedicated BMO settings file. Returning only the ``bmo`` block avoids leaking
    unrelated application settings into the optimizer flow. The parse is cached
    on the settings file's modification time so unchanged config is not re-read
    and re-parsed on every Streamlit rerun.

    Args:
         - None

    Returns:
         - return dict[str, Any] - BMO configuration dictionary.
    """

    config_path = Path(__file__).resolve().parents[1] / "config" / "setting_bmo.yml"
    try:
        mtime_ns = config_path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return _load_bmo_config_cached(int(mtime_ns))


def _repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _ore_preferences_path(bmo_cfg: dict[str, Any]) -> Path:
    ui_cfg = bmo_cfg.get("ui", {}) or {}
    return _repo_path(
        str(
            ui_cfg.get(
                "ore_editor_preferences_path", "src/config/bmo_operator_inputs.yml"
            )
        )
    )


@_resource_cache(show_spinner=False)
def _get_context_provider() -> EvonithBmoContextProvider:
    """
    Create or return the cached Evonith BMO context provider.

    The provider can perform network and file reads, so Streamlit caches the
    instance across reruns. This keeps the page responsive while preserving the
    same mapping and settings for one app session.

    Args:
         - None

    Returns:
         - return EvonithBmoContextProvider - Cached stock, chemistry, and history provider.
    """

    return EvonithBmoContextProvider(
        setting_path="src/config/setting_bmo.yml",
        mapping_path="src/config/bmo_ore_mapping.yml",
    )


def _static_dataset_manager(bmo_cfg: dict[str, Any]) -> StaticDatasetManager:
    data_sources = bmo_cfg.get("data_sources", {}) or {}
    static_path = data_sources.get(
        "static_dataset_path", "src/assets/data/furnace_dataset.csv"
    )
    return StaticDatasetManager(static_path)


def _parse_meta_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _static_dataset_status(bmo_cfg: dict[str, Any]) -> dict[str, Any]:
    manager = _static_dataset_manager(bmo_cfg)
    meta = manager.get_meta()
    csv_path = manager.current_csv_path()
    last_updated = _parse_meta_datetime(meta.last_updated if meta else None)
    max_age_minutes = int(
        (bmo_cfg.get("data_sources", {}) or {}).get(
            "static_refresh_max_age_minutes", 60
        )
        or 60
    )
    now = (
        datetime.now(last_updated.tzinfo)
        if last_updated and last_updated.tzinfo
        else datetime.now()
    )
    stale = last_updated is None or now - last_updated >= timedelta(
        minutes=max_age_minutes
    )
    return {
        "manager": manager,
        "meta": meta,
        "csv_path": csv_path,
        "exists": csv_path.exists(),
        "last_updated": last_updated,
        "latest_data_end": meta.raw_end if meta else "",
        "state": "stale" if stale else "fresh",
        "max_age_minutes": max_age_minutes,
    }


def _static_dataset_cache_token(bmo_cfg: dict[str, Any]) -> tuple[str, int]:
    manager = _static_dataset_manager(bmo_cfg)
    path = manager.current_csv_path()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return str(path), int(mtime_ns)


@_data_cache(show_spinner=False, ttl=600)
def _recent_fuel_rates_from_static_csv(
    static_path: str, mtime_ns: int
) -> dict[str, float | str]:
    path = Path(static_path)
    if not path.exists() or mtime_ns <= 0:
        return {}
    header = pd.read_csv(path, nrows=0)
    wanted = [
        "PCI_KG/THM",
        "ACTUALKG/THM.",
        "ACTUALKG/THM",
        "NUTCOKE_CALC_THM",
        "NUTCOKE_CALC_KG_THM",
        "NUTCOKE_CALC_MT",
        "nutcoke_prime_mt",
        "COKE RATE KG/THM",
        "coke_rate_kg_per_thm",
        "coke_rate_kg_thm",
        "coke_rate",
        "PRODUCTIONTONNESPERHR",
        "PRODUCTIONTONNESPERHR.",
        "production_tonnes_per_hr",
    ]
    columns = [column for column in wanted if column in set(header.columns)]
    if not columns:
        return {}
    df = pd.read_csv(path, usecols=columns)
    return get_recent_fuel_input_rates(process_context=None, history_df=df)


@_data_cache(show_spinner=False, ttl=600)
def _basicity_defaults_from_static_csv(
    static_path: str, mtime_ns: int
) -> dict[str, float]:
    if mtime_ns <= 0:
        return {}
    return derive_basicity_bounds_from_static_dataset(static_path, window_days=30)


# --- Cached offline-source reads -------------------------------------------
# The provider methods below each issue offline Neon/Postgres queries on every
# call. The page reruns top-to-bottom on every widget interaction, so without
# caching every checkbox toggle or "Save" click re-fires all of these DB round
# trips. The source data is manual-entry (hourly / per 8-hour shift), so it is
# fetched once per session and reused until the operator clicks "Refresh source
# data" (which bumps ``bmo_source_cache_version``).
#
# These results carry provider-built objects (OreInput, FluxInput, and
# DB-derived diagnostics) that ``st.cache_data`` tries to pickle and can reject
# (UnserializableReturnValueError). They are therefore memoized in per-session
# ``st.session_state`` instead, keyed by ``(call, mode, window_days)`` under the
# current ``cache_version``. Returns are deep-copied so the page's downstream
# mutation of ``ore_diagnostics["warnings"]`` cannot accumulate into the cache.


def _bmo_source_cache(cache_version: int) -> dict[Any, Any]:
    """Return the per-session source-data cache for ``cache_version``.

    Bumping ``cache_version`` (the Refresh button) discards the whole bucket so
    every offline-source read is fetched fresh on the next rerun.
    """

    bucket = st.session_state.get("_bmo_source_cache")
    if not isinstance(bucket, dict) or bucket.get("_version") != cache_version:
        bucket = {"_version": cache_version}
        st.session_state["_bmo_source_cache"] = bucket
    return bucket


def _session_cached_source(
    cache_version: int, key: tuple[Any, ...], builder: Any
) -> Any:
    bucket = _bmo_source_cache(cache_version)
    if key not in bucket:
        bucket[key] = builder()
    return copy.deepcopy(bucket[key])


def _cached_build_ore_inputs(
    provider: EvonithBmoContextProvider,
    mode: str,
    window_days: int,
    cache_version: int,
) -> tuple[list[OreInput], dict[str, Any]]:
    return _session_cached_source(
        cache_version,
        ("ore_inputs", mode, window_days),
        lambda: provider.build_ore_inputs(mode=mode, window_days=window_days),
    )


def _cached_hm_slag_snapshot(
    provider: EvonithBmoContextProvider,
    mode: str,
    window_days: int,
    cache_version: int,
) -> dict[str, Any]:
    return _session_cached_source(
        cache_version,
        ("hm_slag", mode, window_days),
        lambda: provider.get_hm_slag_snapshot(mode=mode, window_days=window_days),
    )


def _cached_recent_active_pellet_ids(
    provider: EvonithBmoContextProvider,
    cache_version: int,
) -> tuple[list[str], list[str]]:
    return _session_cached_source(
        cache_version,
        ("pellet_ids",),
        lambda: provider.get_recent_active_pellet_ids(),
    )


@_data_cache(show_spinner=False)
def _cached_operator_preferences(path: str, mtime_ns: int) -> dict[str, Any]:
    return load_ore_editor_preferences(path)


def _refresh_static_dataset_if_needed(
    bmo_cfg: dict[str, Any], *, force: bool = False
) -> dict[str, Any]:
    status = _static_dataset_status(bmo_cfg)
    if not force and status["exists"] and status["state"] == "fresh":
        return {"refreshed": False, "usable": True, "status": status}

    manager: StaticDatasetManager = status["manager"]
    try:
        df = manager.update_static("Full")
        saved_path = manager.save(df)
        return {
            "refreshed": True,
            "usable": True,
            "saved_path": saved_path,
            "status": _static_dataset_status(bmo_cfg),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "refreshed": False,
            "usable": bool(status["exists"]),
            "error": str(exc),
            "status": status,
        }


@fragment
def _render_static_dataset_bar(bmo_cfg: dict[str, Any]) -> bool:
    status = _static_dataset_status(bmo_cfg)
    state = status["state"] if status["exists"] else "missing"
    force = False
    with st.expander("Static dataset refresh", expanded=False):
        cols = st.columns([1.1, 1.2, 1.0, 1.0, 1.0])
        cols[0].metric("Static Dataset", state.title())
        cols[1].metric("Latest Data", status["latest_data_end"] or "Unknown")
        last_updated = status["last_updated"]
        cols[2].metric(
            "Last Refresh",
            last_updated.strftime("%Y-%m-%d %H:%M") if last_updated else "Never",
        )
        force = cols[3].checkbox(
            "Force refresh on run",
            value=False,
            key="bmo_force_static_refresh",
        )
        notice = st.session_state.pop("bmo_static_refresh_notice", None)
        if notice:
            st.success(str(notice))
        if cols[4].button("Refresh now"):
            with st.spinner("Refreshing static ML dataset..."):
                result = _refresh_static_dataset_if_needed(bmo_cfg, force=True)
            if result.get("error"):
                st.error(f"Static dataset refresh failed: {result['error']}")
            else:
                st.session_state["bmo_static_refresh_notice"] = (
                    "Static ML dataset refreshed."
                )
                rerun_fragment()
        if state == "stale":
            st.caption("Static dataset is older than the BMO one-hour refresh gate.")
        elif state == "missing":
            st.warning(
                "Static ML dataset is missing; optimizer run will refresh first."
            )
        st.divider()
        src_col1, src_col2 = st.columns([1.0, 2.0])
        if src_col1.button("Refresh source data", key="bmo_refresh_source_data"):
            st.session_state["bmo_source_cache_version"] = (
                int(st.session_state.get("bmo_source_cache_version", 0)) + 1
            )
            # Full app rerun (not fragment-only) so the top-level chemistry,
            # stock, HM/slag, flux and pellet calls re-read with the new version.
            st.rerun()
        src_col2.caption(
            "Offline chemistry, stock, HM/slag, flux and pellet-usage data are "
            "cached from page open for responsiveness. Click to pull the latest "
            "manual-entry records."
        )
    return force


def _context_group(field: str) -> str:
    name = field.upper()
    if any(
        token in name
        for token in (
            "WEIGHTED_COKE",
            "WEIGHTED_NON_COKE",
            "PORTIONS",
            "DISCHARGE_TIME",
        )
    ):
        return "burden_distribution"
    if name.startswith("CHEM_") or name.startswith("SLAG_") or name.startswith("HMT"):
        return "hm_slag"
    if "CALC_MT" in name or "CALC_THM" in name or name.endswith("_PCT"):
        return "charge_quantities"
    if any(
        token in name
        for token in (
            "SIO2",
            "AL2O3",
            "FE_TOTAL",
            "FE(T)",
            "MGO",
            "CAO",
            "TIO2",
            "MNO",
            "BASICITY",
        )
    ):
        return "rm_composition"
    return "process_params"


def _show_df(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No diagnostic rows captured.")
    else:
        st.dataframe(df, hide_index=True, width="stretch")


def _form_submit_button(container: Any, label: str, **kwargs: Any) -> bool:
    return bool(container.form_submit_button(label, **kwargs))


def _clear_bmo_results() -> None:
    for key in (
        "bmo_lp_result",
        "bmo_lp_errors",
        "bmo_de_result",
        "bmo_de_errors",
    ):
        st.session_state.pop(key, None)


def _fuel_ash_cfg_with_recent_rates(
    fuel_ash_cfg: list[dict[str, Any]],
    fuel_rates: dict[str, float | str],
) -> list[dict[str, Any]]:
    rate_keys = {
        "coke": "coke_rate_kg_thm",
        "nut_coke": "nut_coke_rate_kg_thm",
        "pci": "pci_rate_kg_thm",
    }
    rows: list[dict[str, Any]] = []
    for item in fuel_ash_cfg or []:
        row = dict(item)
        rate_key = rate_keys.get(str(row.get("fuel_id", "")).strip())
        rate = fuel_rates.get(rate_key or "")
        if rate is not None:
            row["rate_kg_per_thm"] = float(rate)
        rows.append(row)
    return rows


def _render_share_pie(blend: Any, selected_ores: list[OreInput], title: str) -> None:
    rows = [
        {
            "ore_name": ore.display_name,
            "share_pct": float(blend.shares_pct.get(ore.ore_id, 0.0)),
        }
        for ore in selected_ores
        if float(blend.shares_pct.get(ore.ore_id, 0.0)) > 0
    ]
    if not rows:
        return
    fig = px.pie(
        pd.DataFrame(rows),
        names="ore_name",
        values="share_pct",
        title=title,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, width="stretch")


def _target_quantities_from_shares(
    shares_pct: dict[str, float],
    ores: list[OreInput],
    target_fe_mt: float,
) -> tuple[dict[str, float], float, list[str]]:
    """
    Scale operator-edited burden shares into wet quantities for the target HM.

    The manual-vs-optimizer comparison lets the operator type any share split.
    Shares are normalized to 100%, the blend Fe per wet MT is derived from each
    ore's dry Fe, and the total wet quantity is sized so dry Fe equals the same
    target Fe used by the optimizer. This keeps the manual blend on the same Fe
    basis as the LP/DE result so the cost and slag comparison is apples-to-apples.

    Args:
         - shares_pct: dict[str, float] - Operator share percentages keyed by ore id.
         - ores: list[OreInput] - Candidate ores (chemistry supplies Fe and moisture).
         - target_fe_mt: float - Required dry Fe production in MT.

    Returns:
         - return tuple[dict[str, float], float, list[str]] - Quantities by ore id,
           the normalized share total before scaling, and any warnings.
    """

    total_share = sum(max(0.0, float(v)) for v in shares_pct.values())
    if total_share <= 0:
        return {}, 0.0, ["Manual shares sum to zero; enter at least one positive share."]
    shares = {oid: max(0.0, float(v)) / total_share for oid, v in shares_pct.items()}
    fe_per_blend_mt = 0.0
    for ore in ores:
        dry_fraction = max(0.0, 1.0 - float(ore.chemistry.moisture_pct) / 100.0)
        fe_fraction = max(0.0, float(ore.chemistry.fe_t_pct) / 100.0)
        fe_per_blend_mt += shares.get(ore.ore_id, 0.0) * dry_fraction * fe_fraction
    if fe_per_blend_mt <= 0:
        return {}, total_share, ["Manual blend Fe% is unavailable; cannot scale to target HM."]
    total_qty = float(target_fe_mt) / fe_per_blend_mt
    return (
        {oid: sh * total_qty for oid, sh in shares.items()},
        total_share,
        [],
    )


def _render_blend_comparison(
    provider: EvonithBmoContextProvider,
    *,
    optimizer_candidates: list[tuple[str, Any, float | None]],
    selected_ores: list[OreInput],
    target_fe_mt: float,
    target_production_mt: float,
    feo_in_slag_pct: float,
    fuel_ash_inputs: list[FuelAshInput],
    flux_inputs: list[FluxInput],
    dust_inputs: list[DustInput],
    slag_balance_settings: SlagBalanceSettings,
    manual_ores: list[OreInput] | None = None,
) -> None:
    """Operator-focused comparison of the manual blend vs the optimizer blends.

    Presents two tables on the same target-Fe basis so the operator can decide at
    a glance:

      * **Inputs** - the suggested blend mix (ore share %) for each option, so the
        LP and DE mixes are read side by side against the editable manual blend.
      * **Outputs** - the outcomes that drive the decision: total / ore / fuel
        cost, fuel rate, hot-metal Si, slag basicity / T-basicity, and slag rate.

    Args:
         - provider: EvonithBmoContextProvider - Source for the last-shift manual blend + fuel context.
         - optimizer_candidates: list[tuple[str, BlendEvaluation, float | None]] -
           (label, blend, predicted Si) per optimizer result (LP, DE), in display order.
         - selected_ores: list[OreInput] - Ores the optimizer chose between.
         - target_fe_mt: float - Fe target the manual blend is scaled to.
         - target_production_mt: float - HM basis for cost / slag / model fields.
         - feo_in_slag_pct: float - FeO assumed to report into slag.
         - fuel_ash_inputs / flux_inputs / dust_inputs / slag_balance_settings - Slag-balance inputs.
         - manual_ores: list[OreInput] | None - Materials available for the manual blend.

    Returns:
         - return None - Renders the comparison tables to Streamlit.
    """

    if not optimizer_candidates:
        return
    # Prefer the last candidate (DE over LP) to seed the manual editor.
    primary_blend = optimizer_candidates[-1][1]

    manual_ores_by_id = {ore.ore_id: ore for ore in (manual_ores or selected_ores)}
    manual_ores_by_id.update({ore.ore_id: ore for ore in selected_ores})
    snapshot = provider.get_recent_manual_blend_snapshot(list(manual_ores_by_id.values()))
    rows_by_ore = {str(row.get("ore_id")): row for row in snapshot.get("rows", [])}
    compare_ores = selected_ores

    st.markdown("##### Manual blend vs optimizer")
    st.caption(
        "Edit the manual Share (%) to try any burden split. Shares are normalised to "
        "100% and scaled to the same target Fe as the optimizer, so every option is "
        "compared on the same basis."
    )
    start_time, end_time = snapshot.get("start_time"), snapshot.get("end_time")
    if rows_by_ore and start_time and end_time:
        st.caption(f"Manual blend seeded from last shift ({start_time} to {end_time}).")
    elif not rows_by_ore:
        st.caption("No last-shift manual blend found; seeded from the optimizer shares.")

    seed_rows = []
    for ore in compare_ores:
        seed_share = float(rows_by_ore.get(ore.ore_id, {}).get("share_pct", 0.0) or 0.0)
        if seed_share <= 0:
            seed_share = float(primary_blend.shares_pct.get(ore.ore_id, 0.0))
        seed_rows.append(
            {"ore_id": ore.ore_id, "ore_name": ore.display_name, "manual_share_pct": seed_share}
        )
    # Seed the editor with shares already normalised to 100%. The blend is
    # normalised before evaluation, so the "Suggested blend mix" table shows
    # normalised shares; seeding raw last-shift shares (which sum to <100% when
    # some materials aren't selected) would show a different % for the same ore
    # in the editor vs the mix table.
    _seed_total = sum(row["manual_share_pct"] for row in seed_rows)
    if _seed_total > 0:
        for row in seed_rows:
            row["manual_share_pct"] = round(
                row["manual_share_pct"] / _seed_total * 100.0, 1
            )
    edited_share_df = st.data_editor(
        pd.DataFrame(seed_rows),
        hide_index=True,
        width="stretch",
        key="bmo_manual_share_editor",
        column_order=("ore_name", "manual_share_pct"),
        column_config={
            "ore_id": None,
            "ore_name": st.column_config.TextColumn("Ore", disabled=True),
            "manual_share_pct": st.column_config.NumberColumn(
                "Manual Share (%)", min_value=0.0, max_value=100.0, step=0.5
            ),
        },
    )
    manual_shares_pct = {
        str(row["ore_id"]): float(row["manual_share_pct"] or 0.0)
        for _, row in edited_share_df.iterrows()
    }
    manual_quantities, normalized_total, scale_warnings = _target_quantities_from_shares(
        manual_shares_pct, compare_ores, target_fe_mt
    )
    if normalized_total > 0:
        st.caption(
            f"Entered manual shares sum to {normalized_total:,.1f}% (normalised to 100%)."
        )
    for warning in scale_warnings:
        st.warning(warning)

    manual_blend = None
    manual_si: float | None = None
    if manual_quantities:
        try:
            (
                model_service,
                process_context,
                history_df,
                _bundle_status,
                fuel_warnings,
            ) = _load_fuel_prediction_context(provider)
            manual_blend = evaluate_blend_with_fuel_prediction(
                ores=compare_ores,
                quantities_mt=manual_quantities,
                feo_in_slag_pct=feo_in_slag_pct,
                model_service=model_service,
                process_context=process_context,
                history_df=history_df,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=target_production_mt,
            )
            manual_si = _predict_blend_si(
                ores=compare_ores,
                quantities_mt=manual_quantities,
                process_context=process_context,
                history_df=history_df,
                hot_metal_target_mt=target_production_mt,
            )
            for warning in fuel_warnings:
                st.warning(str(warning))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not evaluate manual blend cost/slag: {exc}")

    # Display order: Manual first (operator's reference), then LP / DE.
    options: list[tuple[str, Any, float | None]] = []
    if manual_blend is not None:
        options.append(("Manual", manual_blend, manual_si))
    options.extend(optimizer_candidates)
    if len(options) < 2:
        return
    labels = [label for label, _, _ in options]

    # ---- Inputs: suggested blend mix (Share %), LP and DE side by side ----
    st.markdown("###### Suggested blend mix (Share %)")
    mix_rows = []
    for ore in compare_ores:
        row: dict[str, Any] = {"Ore": ore.display_name}
        for label, blend, _ in options:
            row[label] = float(blend.shares_pct.get(ore.ore_id, 0.0))
        mix_rows.append(row)
    st.dataframe(
        pd.DataFrame(mix_rows),
        hide_index=True,
        width="stretch",
        column_config={
            label: st.column_config.NumberColumn(label, format="%.1f") for label in labels
        },
    )

    # ---- Outputs: the KPIs an operator decides on ----
    def _basicity(blend: Any, denom_key: str, attr: str) -> float | None:
        if float(blend.diagnostics.get(denom_key, 0.0) or 0.0) <= 0:
            return None
        return float(getattr(blend, attr, 0.0) or 0.0)

    def _fuel_rate_total(blend: Any) -> float | None:
        value = (blend.diagnostics.get("fuel_rate_estimate") or {}).get(
            "total_fuel_rate_kg_thm"
        )
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    # (row label, accessor(blend, si) -> value | None, format string)
    metric_specs = [
        ("Total Cost (Rs/THM)", lambda b, si: b.objective_rs_per_thm, "{:,.0f}"),
        ("Ore Cost (Rs/THM)", lambda b, si: b.ore_cost_per_thm_rs, "{:,.0f}"),
        # 1-decimal so small but real blend-to-blend differences aren't hidden by
        # rounding (the fuel model is only weakly blend-sensitive -- see the help
        # note on the outcomes table).
        ("Fuel Cost (Rs/THM)", lambda b, si: b.fuel_cost_per_thm_rs, "{:,.1f}"),
        ("Fuel Rate (kg/THM)", lambda b, si: _fuel_rate_total(b), "{:,.1f}"),
        ("Hot-Metal Si (%)", lambda b, si: si, "{:,.3f}"),
        (
            "Slag Basicity (CaO/SiO2)",
            lambda b, si: _basicity(b, "slag_basicity_denominator_mt", "slag_basicity"),
            "{:,.3f}",
        ),
        (
            "Slag T-Basicity",
            lambda b, si: _basicity(b, "slag_t_basicity_denominator_mt", "slag_t_basicity"),
            "{:,.3f}",
        ),
        ("Slag Rate (kg/THM)", lambda b, si: b.slag_rate_kg_per_thm, "{:,.0f}"),
    ]
    out_rows = []
    for row_label, accessor, fmt in metric_specs:
        row = {"Outcome": row_label}
        for label, blend, si in options:
            value = accessor(blend, si)
            row[label] = fmt.format(value) if isinstance(value, (int, float)) else "n/a"
        out_rows.append(row)
    st.markdown("###### Key outcomes")
    st.dataframe(
        pd.DataFrame(out_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Outcome": st.column_config.Column(
                "Outcome",
                help=(
                    "Fuel Cost and Fuel Rate are model estimates. The current fuel "
                    "model is largely insensitive to the ore blend, so those two "
                    "rows can look nearly identical across blends even when the mix "
                    "differs. Ore cost, Si, basicity and slag rate are the "
                    "blend-driven outcomes."
                ),
            ),
        },
    )

    # Headline: cheapest option by total cost + optimizer saving vs the manual blend.
    cheapest_label = min(options, key=lambda option: option[1].objective_rs_per_thm)[0]
    cols = st.columns(len(options))
    for col, (label, blend, _) in zip(cols, options):
        col.metric(
            f"{label} (Rs/THM)",
            f"{blend.objective_rs_per_thm:,.0f}",
            delta="cheapest" if label == cheapest_label else None,
            delta_color="normal" if label == cheapest_label else "off",
        )
    if manual_blend is not None:
        best_optimizer = min(
            (blend for _, blend, _ in optimizer_candidates),
            key=lambda blend: blend.objective_rs_per_thm,
        )
        saving = manual_blend.objective_rs_per_thm - best_optimizer.objective_rs_per_thm
        st.caption(
            f"Best optimizer blend is {saving:+,.0f} Rs/THM vs the manual blend "
            "(positive = optimizer is cheaper)."
        )




@fragment
def _render_data_diagnostics(
    provider: EvonithBmoContextProvider,
    *,
    ore_diagnostics: dict[str, Any],
    hm_snapshot: dict[str, Any],
    edited_ore_df: pd.DataFrame,
    expanded: bool,
) -> None:
    with st.expander("Data Diagnostics", expanded=expanded):
        if st.button("Load diagnostics", key="bmo_load_diagnostics"):
            st.session_state["bmo_diagnostics_loaded"] = True
        if st.button("Refresh diagnostics", key="bmo_refresh_diagnostics"):
            st.session_state["bmo_diagnostics_loaded"] = True
            rerun_fragment()
        if not st.session_state.get("bmo_diagnostics_loaded", False):
            st.caption(
                "Diagnostics are loaded on demand because they read the static "
                "dataset, online context, charge mix, and source traces."
            )
            warnings = [
                *ore_diagnostics.get("warnings", []),
                *hm_snapshot.get("warnings", []),
            ]
            if warnings:
                st.warning("- " + "\n- ".join(str(w) for w in warnings[:8]))
            return

        model_service = _get_model_service()
        history_df, history_warnings = provider.get_history_frame(
            online_lag_hours=model_service.get_max_lag_steps()
        )
        process_context, process_warnings = provider.get_process_context(
            history_df=history_df
        )
        provider.get_charge_mix_snapshot()
        diagnostics = provider.get_data_diagnostics()
        warnings = [
            *ore_diagnostics.get("warnings", []),
            *hm_snapshot.get("warnings", []),
            *history_warnings,
            *process_warnings,
        ]

        source_rows = []
        stock = diagnostics.get("stock", {})
        chemistry = diagnostics.get("chemistry", {})
        hm = diagnostics.get("hm_slag", {})
        dpr = diagnostics.get("dpr", {})
        history = diagnostics.get("history", {})
        process = diagnostics.get("process", {})
        flux = diagnostics.get("flux", {})
        charge_mix = diagnostics.get("charge_mix", {})

        source_rows.extend(
            [
                {
                    "area": "stock",
                    "source": stock.get("table", "offline_feed.raw_material_stock"),
                    "timestamp/window": stock.get("time_range", ""),
                    "rows": stock.get("returned_rows", 0),
                    "note": f"{stock.get('fallback_count', 0)} fallback material(s)",
                },
                {
                    "area": "rm_chemistry",
                    "source": "offline chemistry tables",
                    "timestamp/window": f"{chemistry.get('start_time', '')} -> {chemistry.get('end_time', '')}",
                    "rows": sum(
                        int(row.get("returned_rows", 0) or 0)
                        for row in chemistry.get("tables", [])
                    ),
                    "note": f"{chemistry.get('fallback_count', 0)} fallback material(s), mode={chemistry.get('mode', '')}",
                },
                {
                    "area": "hm_slag",
                    "source": hm.get("source", ""),
                    "timestamp/window": hm.get("sample_timestamp")
                    or f"{hm.get('start_time', '')} -> {hm.get('end_time', '')}",
                    "rows": hm.get("n_rows_used", 0),
                    "note": f"HM Fe basis={hm.get('hm_fe_pct_for_target', 0.0):.2f}%",
                },
                {
                    "area": "dpr_observed_slag",
                    "source": dpr.get("source", "offline_db"),
                    "timestamp/window": f"{dpr.get('start_time', '')} -> {dpr.get('end_time', '')}",
                    "rows": dpr.get("row_count", 0),
                    "note": f"slag={dpr.get('slag_generation_mt', 0.0):.1f} MT, HM={dpr.get('total_hot_metal_mt', 0.0):.1f} MT",
                },
                {
                    "area": "static_ml_dataset",
                    "source": history.get("path", ""),
                    "timestamp/window": history.get("latest_timestamp", ""),
                    "rows": history.get("rows", 0),
                    "note": f"{history.get('columns', 0)} columns",
                },
                {
                    "area": "process_charge_burden_context",
                    "source": process.get("source", "static_csv"),
                    "timestamp/window": process.get("latest_timestamp", ""),
                    "rows": process.get("field_count", 0),
                    "note": "latest cleaned static dataset row",
                },
                {
                    "area": "flux_inputs",
                    "source": flux.get(
                        "source", "offline_feed.charge_data+offline_feed.flux_chemistry"
                    ),
                    "timestamp/window": flux.get("latest_timestamp", ""),
                    "rows": len(flux.get("rows", [])),
                    "note": f"mode={flux.get('mode', '')}",
                },
                {
                    "area": "charge_mix",
                    "source": charge_mix.get("source", "offline_feed.charge_data"),
                    "timestamp/window": f"{charge_mix.get('start_time', '')} -> {charge_mix.get('end_time', '')}",
                    "rows": len(charge_mix.get("rows", [])),
                    "note": "sinter/ore/pellet/coke/nut coke/flux",
                },
            ]
        )

        tab_summary, tab_materials, tab_charge, tab_static = st.tabs(
            ["Source Summary", "Material Inputs", "Charge Mix", "Initial Feature Row"]
        )
        with tab_summary:
            _show_df(pd.DataFrame(source_rows))
            if warnings:
                st.warning("- " + "\n- ".join(str(w) for w in warnings[:12]))

        with tab_materials:
            st.markdown("##### Stock")
            _show_df(pd.DataFrame(stock.get("material_rows", [])))
            st.markdown("##### RM Chemistry")
            _show_df(pd.DataFrame(chemistry.get("material_rows", [])))
            st.markdown("##### Flux Inputs")
            _show_df(pd.DataFrame(flux.get("rows", [])))
            st.markdown("##### Current Ore Editor Inputs")
            _show_df(edited_ore_df.copy())

        with tab_charge:
            _show_df(pd.DataFrame(charge_mix.get("rows", [])))

        with tab_static:
            st.caption(
                "This is the latest numeric model context used before candidate "
                "blend overrides. Recent charge quantities come directly from "
                "offline_feed.charge_data when lagged model context is required."
            )
            rows = [
                {
                    "group": _context_group(str(field)),
                    "field": str(field),
                    "value": value,
                    "row_timestamp": process.get("latest_timestamp", ""),
                }
                for field, value in sorted(process_context.items())
            ]
            _show_df(pd.DataFrame(rows))


@_resource_cache(show_spinner=False)
def _get_model_service() -> FuelUnitCostModelService:
    """
    Create or return the cached BMO fuel-cost model service.

    Loading the XGBoost model, scaler, and selected feature list is relatively
    expensive. Caching the service keeps model artifacts warm while allowing
    each candidate blend to request fresh predictions.

    Args:
         - None

    Returns:
         - return FuelUnitCostModelService - Cached fuel-cost prediction service.
    """

    bmo_cfg = _get_bmo_config()
    runtime_cfg = build_runtime_config(bmo_cfg)
    bundle_cfg = dict(runtime_cfg.get("model_bundle", {}))
    bundle_cfg.setdefault(
        "missing_feature_policy",
        runtime_cfg.get("feature_policy", {}).get(
            "missing_feature_policy", "default_warn"
        ),
    )
    return FuelUnitCostModelService(
        bundle_cfg=bundle_cfg,
        fallback_cfg=bmo_cfg.get("fallback_fuel_model", {}),
    )


@_resource_cache(show_spinner=False)
def _get_si_service() -> SiPredictionService:
    """
    Create or return the cached hot-metal Si prediction service.

    Si is predicted for the baseline and DE blends as a display-only signal and
    is never used as an optimization objective or constraint. Caching keeps the
    Si model artifacts warm across reruns.

    Returns:
         - return SiPredictionService - Cached Si prediction service.
    """

    bmo_cfg = _get_bmo_config()
    return SiPredictionService(bundle_cfg=bmo_cfg.get("si_model_bundle", {}))


def _render_si_metric(si_value: float | None) -> None:
    """Render the display-only predicted hot-metal Si for a blend."""

    if si_value is None:
        return
    st.metric("Predicted Hot-Metal Si", f"{si_value:.3f} %")
    st.caption(
        "Advisory only - standalone Si model; not an optimization objective or "
        "constraint and does not affect the blend decision."
    )


def _latest_si_from_history(history_df: pd.DataFrame | None) -> float | None:
    """Return the most recent measured hot-metal Si from the history frame, if any."""

    if history_df is None or history_df.empty:
        return None
    for col in ("CHEM_PCT_SI", "chem_pct_si"):
        if col in history_df.columns:
            series = pd.to_numeric(history_df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[-1])
    return None


def _predict_blend_si(
    *,
    ores: list[OreInput],
    quantities_mt: Mapping[str, float],
    process_context: Mapping[str, Any] | None,
    history_df: pd.DataFrame | None,
    hot_metal_target_mt: float | None,
) -> float | None:
    """
    Predict display-only hot-metal Si for one solved blend.

    Failures never interrupt the optimizer flow; a None result simply hides the
    Si metric for that blend.
    """

    try:
        si_service = _get_si_service()
        return si_service.predict_blend_si(
            ores=ores,
            quantities_mt=quantities_mt,
            process_context=process_context,
            prev_si=_latest_si_from_history(history_df),
            hot_metal_target_mt=hot_metal_target_mt,
        )
    except Exception as exc:  # noqa: BLE001 - Si is advisory; never break the run
        log.warning("Si prediction failed: %s", exc)
        return None


def _load_fuel_prediction_context(
    provider: EvonithBmoContextProvider,
) -> tuple[
    FuelUnitCostModelService,
    dict[str, float],
    pd.DataFrame,
    dict[str, Any],
    list[str],
]:
    """
    Load process, history, and model inputs required for fuel inference.

    Fuel prediction needs the latest process context, recent history for lagged
    features, and the cached model service. Keeping this as a small helper lets
    LP post-solve inference and DE candidate optimization share the same loaded
    inputs during one Streamlit run.

    Args:
         - provider: EvonithBmoContextProvider - Source for process context and history.

    Returns:
         - return tuple[FuelUnitCostModelService, dict[str, float], pd.DataFrame, dict[str, Any], list[str]] - Model service, process context, history, bundle status, and warnings.
    """

    model_service = _get_model_service()
    max_lag_hours = model_service.get_max_lag_steps()
    history_df, history_warnings = provider.get_history_frame(
        online_lag_hours=max_lag_hours
    )
    process_context, process_warnings = provider.get_process_context(
        history_df=history_df
    )
    bundle_status = model_service.get_bundle_status()
    warnings = [*process_warnings, *history_warnings]
    return model_service, process_context, history_df, bundle_status, warnings


def _selected_ores_from_editor(
    editor_df: pd.DataFrame, base_ores: list[OreInput]
) -> list[OreInput]:
    """
    Convert edited Streamlit ore rows into typed selected ore inputs.

    The editor lets users change wet stock, price, share limits, and moisture.
    Moisture is written back into the nested chemistry object so the optimizer
    uses the same dry-weight Fe calculation that the result table displays.

    Args:
         - editor_df: pd.DataFrame - User-edited ore selection table.
         - base_ores: list[OreInput] - Original ore inputs from the context provider.

    Returns:
         - return list[OreInput] - Selected ores with edited stock, price, and bounds.
    """

    by_id = {ore.ore_id: ore for ore in base_ores}
    selected_ores: list[OreInput] = []

    for _, row in editor_df.iterrows():
        if not bool(row.get("selected", False)):
            continue
        ore_id = str(row["ore_id"])
        if ore_id not in by_id:
            continue
        base = by_id[ore_id]
        edited_chemistry = replace(
            base.chemistry,
            moisture_pct=float_from_row(
                row, "moisture_pct", base.chemistry.moisture_pct
            ),
            fe_t_pct=float_from_row(row, "fe_t_pct", base.chemistry.fe_t_pct),
            sio2_pct=float_from_row(row, "sio2_pct", base.chemistry.sio2_pct),
            al2o3_pct=float_from_row(row, "al2o3_pct", base.chemistry.al2o3_pct),
            cao_pct=float_from_row(row, "cao_pct", base.chemistry.cao_pct),
            mgo_pct=float_from_row(row, "mgo_pct", base.chemistry.mgo_pct),
            mno_pct=float_from_row(row, "mno_pct", base.chemistry.mno_pct),
            tio2_pct=float_from_row(row, "tio2_pct", base.chemistry.tio2_pct),
        )
        selected_ores.append(
            replace(
                base,
                stock_mt=float(row["stock_mt"]),
                price_rs_per_mt=float(row["price_rs_per_mt"]),
                min_share_pct=float(row["min_share_pct"]),
                max_share_pct=float(row["max_share_pct"]),
                chemistry=edited_chemistry,
            )
        )
    return selected_ores



apply_bmo_styles()
bmo_cfg = _get_bmo_config()
provider = _get_context_provider()
model_service = _get_model_service()
bundle_status = model_service.get_bundle_status()
st.session_state["bmo_bundle_status"] = bundle_status
render_header(bundle_status)
force_static_refresh = _render_static_dataset_bar(bmo_cfg)
pending_run_after_refresh = st.session_state.pop("bmo_pending_run_after_refresh", None)
# Bumped by the "Refresh source data" button; keys the cached offline-source
# reads so they are fetched once per session and reused until the operator asks
# for fresh data.
source_cache_version = int(st.session_state.get("bmo_source_cache_version", 0))

ui_cfg = bmo_cfg.get("ui", {})
target_cfg = bmo_cfg.get("target", {})
runtime_cfg = build_runtime_config(
    bmo_cfg, default_optimizer=bmo_cfg.get("optimization", {})
)
opt_cfg = runtime_cfg.get("optimizer", {})
operator_preferences_path = _ore_preferences_path(bmo_cfg)
try:
    _pref_mtime_ns = Path(operator_preferences_path).stat().st_mtime_ns
except OSError:
    _pref_mtime_ns = 0
operator_preferences = _cached_operator_preferences(
    str(operator_preferences_path), int(_pref_mtime_ns)
)
static_path, static_mtime_ns = _static_dataset_cache_token(bmo_cfg)
recent_fuel_rates = _recent_fuel_rates_from_static_csv(static_path, static_mtime_ns)
basicity_defaults = {
    "target_slag_basicity_min": float(target_cfg.get("target_slag_basicity_min", 0.0)),
    "target_slag_basicity_max": float(target_cfg.get("target_slag_basicity_max", 10.0)),
    "target_slag_t_basicity_min": float(
        target_cfg.get("target_slag_t_basicity_min", 0.0)
    ),
    "target_slag_t_basicity_max": float(
        target_cfg.get("target_slag_t_basicity_max", 10.0)
    ),
}
basicity_defaults.update(
    _basicity_defaults_from_static_csv(static_path, static_mtime_ns)
)
basicity_defaults = apply_model_input_preferences(
    basicity_defaults, operator_preferences
)

with st.form("bmo_model_input_form", clear_on_submit=False):
    st.markdown("### Model Inputs")
    layout_col1, layout_col2, layout_col3, layout_col4 = st.columns(4)
    with layout_col1:
        chemistry_mode = st.selectbox(
            "Chemistry mode",
            options=["latest", "avg"],
            index=0 if str(bmo_cfg.get("chemistry_mode", "latest")) == "latest" else 1,
            key="bmo_chemistry_mode",
        )
    chemistry_window_days = layout_col2.slider(
        "Chemistry window for avg (days)",
        min_value=1,
        max_value=180,
        value=int(bmo_cfg.get("chemistry_window_days", 30)),
        key="bmo_chemistry_window_days",
        help="Used only when Chemistry mode is avg. Latest mode uses the last charged instance.",
    )
    target_production_mt = layout_col3.number_input(
        "Target HM / Pig Iron (MT)",
        min_value=0.0,
        value=float(target_cfg.get("target_production_mt", 2350.0)),
        step=5.0,
        key="bmo_target_production_mt",
    )
    target_slag_qty_mt = layout_col4.number_input(
        "Max Slag (MT)",
        min_value=0.0,
        value=float(target_cfg.get("target_slag_qty_mt", 750.0)),
        step=5.0,
        key="bmo_target_slag_qty_mt",
    )
    basicity_col1, basicity_col2, basicity_col3, basicity_col4 = st.columns(4)
    target_slag_basicity_min = basicity_col1.number_input(
        "Min Basicity CaO/SiO2",
        min_value=0.0,
        value=float(basicity_defaults["target_slag_basicity_min"]),
        step=0.01,
        format="%.3f",
        key="bmo_target_slag_basicity_min",
    )
    target_slag_basicity_max = basicity_col2.number_input(
        "Max Basicity CaO/SiO2",
        min_value=0.0,
        value=float(basicity_defaults["target_slag_basicity_max"]),
        step=0.01,
        format="%.3f",
        key="bmo_target_slag_basicity_max",
    )
    target_slag_t_basicity_min = basicity_col3.number_input(
        "Min T Basicity",
        min_value=0.0,
        value=float(basicity_defaults["target_slag_t_basicity_min"]),
        step=0.01,
        format="%.3f",
        key="bmo_target_slag_t_basicity_min",
        help="(CaO + MgO) / SiO2",
    )
    target_slag_t_basicity_max = basicity_col4.number_input(
        "Max T Basicity",
        min_value=0.0,
        value=float(basicity_defaults["target_slag_t_basicity_max"]),
        step=0.01,
        format="%.3f",
        key="bmo_target_slag_t_basicity_max",
        help="(CaO + MgO) / SiO2",
    )
    model_apply_col, model_save_col = st.columns(2)
    model_inputs_applied = _form_submit_button(
        model_apply_col,
        "Apply Model Inputs",
        type="primary",
        width="stretch",
    )
    model_inputs_saved = _form_submit_button(
        model_save_col,
        "Save Basicity Inputs for Next Time",
        type="secondary",
        width="stretch",
    )
if model_inputs_applied or model_inputs_saved:
    st.session_state.pop("bmo_applied_ore_editor_df", None)
    _clear_bmo_results()
    if model_inputs_saved:
        try:
            saved_path = save_model_input_preferences(
                operator_preferences_path,
                {
                    "target_slag_basicity_min": target_slag_basicity_min,
                    "target_slag_basicity_max": target_slag_basicity_max,
                    "target_slag_t_basicity_min": target_slag_t_basicity_min,
                    "target_slag_t_basicity_max": target_slag_t_basicity_max,
                },
            )
            st.success(f"Basicity inputs saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save basicity inputs: {exc}")
    else:
        st.success("Model inputs applied.")
elif chemistry_mode == "latest":
    st.caption("Latest chemistry uses the last charged instance for each material.")
basicity_bounds_valid = (
    target_slag_basicity_min <= target_slag_basicity_max
    and target_slag_t_basicity_min <= target_slag_t_basicity_max
)
if target_slag_basicity_min > target_slag_basicity_max:
    st.error("Min Basicity CaO/SiO2 must be less than or equal to Max Basicity.")
if target_slag_t_basicity_min > target_slag_t_basicity_max:
    st.error("Min T Basicity must be less than or equal to Max T Basicity.")
feo_in_slag_pct = float(bmo_cfg.get("chemistry", {}).get("feo_in_slag_pct", 0.4))


ores, ore_diagnostics = _cached_build_ore_inputs(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"] = list(ore_diagnostics.get("warnings", []))

hm_snapshot = _cached_hm_slag_snapshot(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"].extend(hm_snapshot.get("warnings", []))
observed_slag_rate = float(hm_snapshot.get("observed_slag_rate_kg_per_thm", 0.0) or 0.0)
hm_fe_pct_for_target = float(
    hm_snapshot.get("hm_fe_pct_for_target")
    or target_cfg.get("hm_fe_pct_fallback", 94.5)
    or 94.5
)
if hm_fe_pct_for_target <= 0:
    hm_fe_pct_for_target = 94.5
    ore_diagnostics["warnings"].append(
        "HM Fe% unavailable; using 94.5% fallback to convert target HM into required Fe."
    )
target_fe_mt = float(target_production_mt) * hm_fe_pct_for_target / 100.0
st.caption(
    f"Optimiser Fe requirement: {target_fe_mt:,.1f} MT "
    f"from {target_production_mt:,.1f} MT HM at {hm_fe_pct_for_target:.2f}% Fe."
)

default_selected_names = set(ui_cfg.get("default_selected_ores", []))
default_selected_ids = [
    ore.ore_id for ore in ores if ore.display_name in default_selected_names
]
if bool(ui_cfg.get("auto_select_active_pellet", True)):
    active_pellet_ids, pellet_usage_warnings = _cached_recent_active_pellet_ids(
        provider, source_cache_version
    )
    ore_diagnostics["warnings"].extend(pellet_usage_warnings)
    default_selected_ids = sorted(set(default_selected_ids).union(active_pellet_ids))
editor_df = build_ore_editor_df(ores, default_selected_ids=default_selected_ids)
editor_df = apply_ore_editor_preferences(editor_df, operator_preferences)

st.markdown("### Ore Selection, Stock, Pricing, Chemistry, and Share Bounds")
stored_ore_df = st.session_state.get("bmo_applied_ore_editor_df")
if (
    isinstance(stored_ore_df, pd.DataFrame)
    and "ore_id" in stored_ore_df.columns
    and set(stored_ore_df["ore_id"].astype(str)) == set(editor_df["ore_id"].astype(str))
):
    ore_editor_source_df = stored_ore_df
else:
    ore_editor_source_df = editor_df

with st.form("bmo_ore_input_form", clear_on_submit=False):
    edited_ore_candidate_df = render_ore_editor(ore_editor_source_df)
    st.caption(
        "Save keeps ore selection, prices, and share bounds for the next "
        "session. Stock and chemistry continue to come from the latest source data."
    )
    ore_apply_col, ore_save_col = st.columns(2)
    ore_inputs_applied = _form_submit_button(
        ore_apply_col,
        "Apply Ore Inputs",
        type="primary",
        width="stretch",
    )
    ore_inputs_saved = _form_submit_button(
        ore_save_col,
        "Save Ore Inputs for Next Time",
        type="secondary",
        width="stretch",
    )
if ore_inputs_applied or ore_inputs_saved:
    edited_df = edited_ore_candidate_df.copy()
    st.session_state["bmo_applied_ore_editor_df"] = edited_df
    _clear_bmo_results()
    if ore_inputs_saved:
        try:
            saved_path = save_ore_editor_preferences(
                operator_preferences_path, edited_df
            )
            st.success(f"Ore inputs saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save ore inputs: {exc}")
    else:
        st.success("Ore inputs applied.")
else:
    edited_df = ore_editor_source_df

with st.expander("Hot Metal Chemistry Assumptions", expanded=False):
    with st.form("bmo_assumption_input_form", clear_on_submit=False):
        hm_chem_values = render_hot_metal_chemistry(
            hm_snapshot, bmo_cfg.get("slag_balance", {})
        )
        assumptions_applied = _form_submit_button(
            st,
            "Apply Assumptions",
            type="primary",
            width="stretch",
        )
if assumptions_applied:
    _clear_bmo_results()
    st.success("Assumptions applied.")

selected_ores = _selected_ores_from_editor(edited_df, ores)
pellet_input_issues = validate_selected_pellet_inputs(
    selected_ores,
    max_chemistry_age_days=int(
        bmo_cfg.get("data_sources", {}).get("pellet_chemistry_max_age_days", 30)
    ),
)
pellet_input_confirmed = True
if pellet_input_issues:
    st.warning(
        "Selected pellet inputs need operator review before optimisation:\n- "
        + "\n- ".join(pellet_input_issues)
    )
    pellet_input_confirmed = st.checkbox(
        "I have reviewed/edited the selected pellet stock and chemistry values.",
        value=False,
        key="bmo_confirm_pellet_inputs",
    )
visible_data_warnings = [
    str(w)
    for w in ore_diagnostics.get("warnings", [])
    if any(token in str(w).lower() for token in ("fallback", "failed", "unavailable"))
]
if visible_data_warnings:
    st.warning(
        "BMO is using fallback or incomplete source data:\n- "
        + "\n- ".join(visible_data_warnings[:5])
    )

_render_data_diagnostics(
    provider,
    ore_diagnostics=ore_diagnostics,
    hm_snapshot=hm_snapshot,
    edited_ore_df=edited_df,
    expanded=bool(visible_data_warnings),
)

fuel_ash_df = build_fuel_ash_editor_df(
    _fuel_ash_cfg_with_recent_rates(
        bmo_cfg.get("fuel_ash_inputs", []), recent_fuel_rates
    )
)
if not fuel_ash_df.empty:
    st.markdown("### Fuel Ash Inputs")
    edited_fuel_ash_df = render_fuel_ash_editor(fuel_ash_df)
else:
    edited_fuel_ash_df = fuel_ash_df
fuel_ash_inputs = fuel_ash_inputs_from_editor(edited_fuel_ash_df)

flux_df = build_flux_editor_df(bmo_cfg.get("flux_inputs", []))
if not flux_df.empty:
    st.markdown("### Flux Inputs")
    edited_flux_df = render_flux_editor(flux_df)
else:
    edited_flux_df = flux_df
flux_inputs = flux_inputs_from_editor(edited_flux_df)

with st.expander("Advanced Slag Balance Inputs", expanded=False):
    slag_settings_values = render_slag_balance_settings(bmo_cfg.get("slag_balance", {}))
    dust_df = build_dust_editor_df(bmo_cfg.get("dust_inputs", []))
    if not dust_df.empty:
        st.markdown("##### BF Gas Dust")
        edited_dust_df = render_dust_editor(dust_df)
    else:
        edited_dust_df = dust_df
dust_inputs = dust_inputs_from_editor(edited_dust_df)
slag_balance_settings = slag_balance_settings_from_editor(
    slag_settings_values, hm_chem_values, hm_snapshot
)

# Operator-visible warning: if dust is entered but the full slag balance
# is disabled, the dust rows are silently ignored downstream. Surface
# this so the operator knows their dust entry isn't being applied.
_dust_entered_mt = sum(float(d.wet_qty_mt or 0.0) for d in dust_inputs if d.enabled)
if _dust_entered_mt > 0 and not slag_balance_settings.enabled:
    st.warning(
        f"BF gas dust ({_dust_entered_mt:,.1f} MT) is entered but "
        "'Use full slag balance' is unchecked - dust will NOT be deducted "
        "from the slag balance."
    )

run_lp_clicked = False
run_total_clicked = False
with st.form("bmo_run_form", clear_on_submit=False):
    run_col1, run_col2 = st.columns(2)
    run_lp_clicked = _form_submit_button(
        run_col1,
        "Run LP Baseline",
        type="secondary",
        width="stretch",
    )
    run_total_clicked = _form_submit_button(
        run_col2,
        "Run Total Cost Optimizer",
        type="primary",
        width="stretch",
    )

requested_lp = bool(run_lp_clicked or pending_run_after_refresh in {"lp", "both"})
requested_total = bool(
    run_total_clicked or pending_run_after_refresh in {"total", "both"}
)

if run_lp_clicked or run_total_clicked:
    with st.spinner("Checking static ML dataset freshness..."):
        refresh_result = _refresh_static_dataset_if_needed(
            bmo_cfg, force=force_static_refresh
        )
    if refresh_result.get("error"):
        if refresh_result.get("usable"):
            st.warning(
                "Static dataset refresh failed; continuing with the last usable local dataset. "
                f"Error: {refresh_result['error']}"
            )
        else:
            st.error(
                "Static dataset refresh failed and no usable local dataset exists. "
                f"Error: {refresh_result['error']}"
            )
            st.stop()
    elif refresh_result.get("refreshed"):
        st.session_state["bmo_pending_run_after_refresh"] = (
            "both"
            if run_lp_clicked and run_total_clicked
            else "lp" if run_lp_clicked else "total"
        )
        st.success(
            "Static ML dataset refreshed. Re-running optimizer with the updated dataset."
        )
        st.rerun()

if requested_lp or requested_total:
    if not basicity_bounds_valid:
        st.error("Correct the slag basicity bounds before running BMO.")
    elif pellet_input_issues and not pellet_input_confirmed:
        st.error(
            "Confirm the selected pellet stock and chemistry values before running BMO."
        )
    elif len(selected_ores) < 2:
        st.error("Select at least two ores before running optimization.")
    else:
        fuel_context = None

        with st.spinner("Running LP baseline..."):
            lp_result, lp_errors = run_lp_baseline(
                selected_ores,
                target_production_mt=target_fe_mt,
                target_slag_qty_mt=target_slag_qty_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                target_slag_basicity_min=target_slag_basicity_min,
                target_slag_basicity_max=target_slag_basicity_max,
                target_slag_t_basicity_min=target_slag_t_basicity_min,
                target_slag_t_basicity_max=target_slag_t_basicity_max,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=target_production_mt,
            )
            if lp_result is not None:
                lp_physical_result = lp_result
                fuel_context = _load_fuel_prediction_context(provider)
                (
                    model_service,
                    process_context,
                    history_df,
                    bundle_status,
                    fuel_warnings,
                ) = fuel_context
                ore_diagnostics["warnings"] = [
                    *ore_diagnostics.get("warnings", []),
                    *fuel_warnings,
                ]
                st.session_state["bmo_bundle_status"] = bundle_status
                lp_result = evaluate_blend_with_fuel_prediction(
                    ores=selected_ores,
                    quantities_mt=lp_physical_result.quantities_mt,
                    feo_in_slag_pct=feo_in_slag_pct,
                    model_service=model_service,
                    process_context=process_context,
                    history_df=history_df,
                    fuel_ash_inputs=fuel_ash_inputs,
                    flux_inputs=flux_inputs,
                    dust_inputs=dust_inputs,
                    slag_balance_settings=slag_balance_settings,
                    hot_metal_target_mt=target_production_mt,
                )
                lp_result.feasible = lp_physical_result.feasible
                lp_result.violations = lp_physical_result.violations
                # Display-only Si prediction for the baseline blend (not optimized).
                st.session_state["bmo_lp_si"] = _predict_blend_si(
                    ores=selected_ores,
                    quantities_mt=lp_physical_result.quantities_mt,
                    process_context=process_context,
                    history_df=history_df,
                    hot_metal_target_mt=target_production_mt,
                )
        # LP always runs in this rerun and shares DE's fuel-prediction context, so
        # always persist it. This keeps the LP tab and the LP-vs-DE comparison on
        # the SAME live snapshot DE used, avoiding a stale LP (from an earlier,
        # separate click) being shown next to a fresh DE.
        st.session_state["bmo_lp_result"] = lp_result
        st.session_state["bmo_lp_errors"] = lp_errors

        if requested_total:
            de_status = st.status("Total Cost Optimizer (DE) running...", expanded=True)
            iteration_lines: list[str] = []

            def _de_progress(
                iteration: int,
                best_obj: float,
                best_feas: float | None,
                nfev: int,
                elapsed_s: float,
            ) -> bool:
                """
                Stream DE iteration progress to the Streamlit status panel + log.

                Args:
                     - iteration: int - 1-based DE generation index.
                     - best_obj: float - Best (penalized) objective value seen so far in Rs/THM.
                     - best_feas: float | None - Best feasible objective seen so far, if any.
                     - nfev: int - Cumulative function-evaluation count.
                     - elapsed_s: float - Seconds elapsed since DE started.

                Returns:
                     - return bool - False to keep running (no user-cancel wired yet).
                """

                feas_txt = (
                    f", best feasible {best_feas:,.1f}" if best_feas is not None else ""
                )
                line = (
                    f"Iter {iteration:>2}  best {best_obj:,.1f} Rs/THM{feas_txt}"
                    f"  (nfev={nfev}, {elapsed_s:.1f}s)"
                )
                iteration_lines.append(line)
                de_status.write(line)
                log.info("BMO DE %s", line)
                return False

            if fuel_context is None:
                fuel_context = _load_fuel_prediction_context(provider)
                (
                    model_service,
                    process_context,
                    history_df,
                    bundle_status,
                    fuel_warnings,
                ) = fuel_context
                ore_diagnostics["warnings"] = [
                    *ore_diagnostics.get("warnings", []),
                    *fuel_warnings,
                ]
                st.session_state["bmo_bundle_status"] = bundle_status
            else:
                (
                    model_service,
                    process_context,
                    history_df,
                    bundle_status,
                    _fuel_warnings,
                ) = fuel_context
            de_result, de_errors = run_nonlinear_optimizer(
                selected_ores,
                target_production_mt=target_fe_mt,
                target_slag_qty_mt=target_slag_qty_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                target_slag_basicity_min=target_slag_basicity_min,
                target_slag_basicity_max=target_slag_basicity_max,
                target_slag_t_basicity_min=target_slag_t_basicity_min,
                target_slag_t_basicity_max=target_slag_t_basicity_max,
                model_service=model_service,
                process_context=process_context,
                history_df=history_df,
                de_cfg=opt_cfg,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=target_production_mt,
                progress_callback=_de_progress,
            )
            de_status.update(
                label=(
                    f"DE finished - {len(iteration_lines)} iterations"
                    if iteration_lines
                    else "DE finished"
                ),
                state="complete",
            )
            # Guardrail: DE jointly minimises ore + fuel and is seeded from the LP
            # baseline, so its total cost can never legitimately exceed LP's. If it
            # does (DE hit its iteration/time budget, or returned infeasible), report
            # the cheaper LP solution as the DE result so the operator never sees a
            # "worse optimum" than the baseline. LP here is from the same rerun /
            # context as DE, so the comparison is apples-to-apples.
            if lp_result is not None and lp_result.feasible and (
                de_result is None
                or not de_result.feasible
                or de_result.objective_rs_per_thm
                > lp_result.objective_rs_per_thm + 1e-6
            ):
                de_result = copy.deepcopy(lp_result)
                de_result.diagnostics = dict(de_result.diagnostics)
                de_result.diagnostics["de_fell_back_to_lp"] = True
                de_si = st.session_state.get("bmo_lp_si")
            elif de_result is not None:
                # Display-only Si prediction for the DE blend (Si is not optimized).
                de_si = _predict_blend_si(
                    ores=selected_ores,
                    quantities_mt=de_result.quantities_mt,
                    process_context=process_context,
                    history_df=history_df,
                    hot_metal_target_mt=target_production_mt,
                )
            else:
                de_si = None
            st.session_state["bmo_de_result"] = de_result
            st.session_state["bmo_de_errors"] = de_errors
            st.session_state["bmo_de_si"] = de_si


lp_result = st.session_state.get("bmo_lp_result")
lp_errors = st.session_state.get("bmo_lp_errors", [])
de_result = st.session_state.get("bmo_de_result")
de_errors = st.session_state.get("bmo_de_errors", [])

if lp_errors:
    st.error("LP baseline errors:\n- " + "\n- ".join(lp_errors))
if de_errors:
    st.error("Total cost optimizer errors:\n- " + "\n- ".join(de_errors))

if lp_result is not None or de_result is not None:
    tab_lp, tab_de, tab_cmp = st.tabs(["LP Baseline", "Total Cost (DE)", "Comparison"])

    with tab_lp:
        if lp_result is not None:
            render_blend_metrics(
                "LP Baseline Result",
                lp_result,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
                is_lp_mode=True,
            )
            _render_si_metric(st.session_state.get("bmo_lp_si"))
            render_blend_table(lp_result, selected_ores)
            _render_share_pie(lp_result, selected_ores, "LP Share (%)")
            render_slag_balance_details(
                lp_result, selected_ores, fuel_ash_inputs, flux_inputs
            )
        else:
            st.info("Run LP baseline to see deterministic cost-minimized blend.")

    with tab_de:
        if de_result is not None:
            if de_result.diagnostics.get("de_fell_back_to_lp"):
                st.info(
                    "The total-cost optimizer did not improve on the LP baseline "
                    "(it can hit its iteration/time budget). Showing the LP "
                    "baseline blend as the best available solution."
                )
            render_blend_metrics(
                "DE Total-Cost Result",
                de_result,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
            )
            _render_si_metric(st.session_state.get("bmo_de_si"))
            render_blend_table(de_result, selected_ores)
            _render_share_pie(de_result, selected_ores, "DE Share (%)")
            render_slag_balance_details(
                de_result, selected_ores, fuel_ash_inputs, flux_inputs
            )
            if not bundle_status.get("model_loaded"):
                st.info(
                    "Model artifact is missing. Fuel term is currently coming from fallback "
                    "formula until model/scaler bundle is provided."
                )
        else:
            st.info("Run total-cost optimizer to see ore + fuel optimized blend.")

    with tab_cmp:
        # Operator-focused comparison: the manual blend against each optimizer
        # blend (LP + DE) shown side by side -- inputs (blend mix) and outputs
        # (cost, fuel rate, Si, basicity, slag rate) on the same target-Fe basis.
        optimizer_candidates: list[tuple[str, Any, float | None]] = []
        if lp_result is not None:
            optimizer_candidates.append(
                ("LP Baseline", lp_result, st.session_state.get("bmo_lp_si"))
            )
        if de_result is not None:
            optimizer_candidates.append(
                ("DE Total-Cost", de_result, st.session_state.get("bmo_de_si"))
            )
        if optimizer_candidates:
            _render_blend_comparison(
                provider,
                optimizer_candidates=optimizer_candidates,
                selected_ores=selected_ores,
                target_fe_mt=target_fe_mt,
                target_production_mt=target_production_mt,
                feo_in_slag_pct=feo_in_slag_pct,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                manual_ores=ores,
            )
        else:
            st.info(
                "Run LP or DE to compare the suggested blend with the last manual shift."
            )

render_diagnostics(de_result or lp_result, ore_diagnostics)
