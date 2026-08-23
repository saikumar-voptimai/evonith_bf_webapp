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
import plotly.graph_objects as go
import streamlit as st

log = logging.getLogger(__name__)

from config.config_loader import load_config
from data.bmo import EvonithBmoContextProvider
from data.bmo.basicity_defaults import derive_basicity_bounds_from_static_dataset
from data.bmo.ore_editor_preferences import (
    apply_dust_preferences,
    apply_fuel_ash_preferences,
    apply_flux_preferences,
    apply_hm_chemistry_preferences,
    apply_model_input_preferences,
    apply_ore_editor_preferences,
    load_ore_editor_preferences,
    save_dust_preferences,
    save_fuel_ash_preferences,
    save_flux_preferences,
    save_hm_chemistry_preferences,
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
    render_coke_correction_breakdown,
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
    CokeCorrectionDrivers,
    CokeCorrectionReference,
    CokeCorrectionSettings,
    DustInput,
    FluxInput,
    FuelAshInput,
    FuelUnitCostModelService,
    OreInput,
    SlagBalanceSettings,
    build_reference,
    compute_burden_oxygen_kg_per_thm,
    compute_charging_requirements,
    compute_flux_co2_kg_per_thm,
    load_coke_correction_settings,
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
from utils.bmo.constraints import (
    CHARGING_HOURS_PER_DAY,
    DEFAULT_NUT_COKE_RATE_KG_PER_THM,
    calculate_wet_nut_coke_mt,
    max_ibrm_flux_capacity_mt,
    check_blend_constraints,
)
from utils.bmo.fuel_prediction import evaluate_blend_with_fuel_prediction
from utils.bmo.calculations import scale_ore_quantities_to_hot_metal
from utils.bmo.types import oxide_pct_from_basis
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


_STATIC_DATASET_LINK_KEY = "bmo_static_dataset_use_link"
_STATIC_DATASET_SOURCE_CHANGED_KEY = "_bmo_static_dataset_source_changed"


def _configured_static_dataset_url(bmo_cfg: dict[str, Any]) -> str:
    """Return a BMO-specific URL override or the central DATA_URL."""
    data_sources = bmo_cfg.get("data_sources", {}) or {}
    override = str(data_sources.get("static_dataset_url", "") or "").strip()
    if override:
        return override
    return str(load_config("setting_ds_dv.yml").get("DATA_URL", "") or "").strip()


def _use_static_dataset_link(bmo_cfg: dict[str, Any]) -> bool:
    source_url = _configured_static_dataset_url(bmo_cfg)
    return bool(source_url) and bool(
        st.session_state.get(_STATIC_DATASET_LINK_KEY, True)
    )


def _mark_static_dataset_source_changed() -> None:
    """Request a full-page rerun after the fragment-scoped toggle rerun."""
    st.session_state[_STATIC_DATASET_SOURCE_CHANGED_KEY] = True


def _static_dataset_manager(bmo_cfg: dict[str, Any]) -> StaticDatasetManager:
    data_sources = bmo_cfg.get("data_sources", {}) or {}
    static_path = data_sources.get(
        "static_dataset_path", "src/assets/data/furnace_dataset.csv"
    )
    static_url = (
        _configured_static_dataset_url(bmo_cfg)
        if _use_static_dataset_link(bmo_cfg)
        else ""
    )
    return StaticDatasetManager(static_path, remote_url=static_url or None)


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
    csv_end_timestamp = manager.get_csv_end_timestamp()
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
    expected_source = manager.remote_url
    if meta is None or str(meta.source_url or "").strip() != expected_source:
        stale = True
    return {
        "manager": manager,
        "meta": meta,
        "csv_path": csv_path,
        "exists": csv_path.exists(),
        "last_updated": last_updated,
        "last_fetch": csv_end_timestamp,
        "latest_data_end": meta.raw_end if meta else "",
        "source_mode": "link" if manager.remote_url else "code",
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
def _live_process_snapshot() -> dict[str, float]:
    """Blast, top-gas and shell-cooling tags for the Layer 2 energy balance.

    Layer 2 needs the CURRENT operating point: what the blast is doing now, what
    the top gas analysis reads, and how much heat the staves are shedding. All
    of it is one-hour-averaged so a single bad scan cannot move a
    recommendation. Returns an empty dict on any failure, and the caller then
    tells the operator the layer is unavailable rather than guessing.
    """

    wanted = (
        "hot_blast_vol_nm3h", "hot_blast_temp", "hot_blast_press",
        "oxygen_enrichment_pct", "top_press_avg", "steam_injection",
        "co_pct", "co2_pct", "h2_pct", "top_temp_avg",
    )
    try:
        from furnace_data.influx.online import fetch_online_df

        df = fetch_online_df(
            selected_measurements=["process_params", "heatload_delta_t"],
            time_range="last 1 hour",
            window_by="15 minutes",
            column_naming="field",
        )
    except Exception as exc:  # noqa: BLE001 - network/auth failures fall back
        log.warning("Live process snapshot unavailable: %s", exc)
        return {}
    if df is None or df.empty:
        return {}

    out: dict[str, float] = {}
    for field in wanted:
        if field in df.columns:
            value = pd.to_numeric(df[field], errors="coerce").mean()
            if pd.notna(value):
                out[field] = float(value)

    # Stave heat load: the tags read in MW, so x3.6 gives GJ/hr. The x3600
    # "GW.hr" conversion is 1000x too large - see energy_balance.yml.
    quad = [f"heat_load_r{r}_q{q}" for r in range(6, 11) for q in range(1, 5)]
    have = [c for c in quad if c in df.columns]
    if have:
        total_mw = df[have].apply(pd.to_numeric, errors="coerce").sum(axis=1).mean()
        if pd.notna(total_mw) and 2.0 <= float(total_mw) <= 12.0:
            out["shell_loss_gj_per_hr"] = float(total_mw) * 3.6
    return out


@_data_cache(show_spinner=False, ttl=600)
def _recent_fuel_rates_live() -> dict[str, float | str]:
    """Live coke / nut coke / PCI rates from InfluxDB, averaged over the last hour.

    The plant's ``process_params`` tags are the authoritative current rates
    (the static dataset lags and has to derive nut coke from charged MT, which
    was measured ~22 kg/THM above the live ``nut_coke_rate`` tag). PCI swings
    widely between 15-minute windows, so a 1-hour average is used for all three.
    Returns an empty dict on any fetch failure so callers fall back to the
    static-CSV rates.
    """

    field_to_rate_key = {
        "coke_rate": "coke_rate_kg_thm",
        "nut_coke_rate": "nut_coke_rate_kg_thm",
        "coal_rate_actual_value": "pci_rate_kg_thm",
    }
    try:
        from furnace_data.influx.online import fetch_online_df

        df = fetch_online_df(
            selected_measurements=["process_params"],
            time_range="last 1 hour",
            window_by="15 minutes",
            column_naming="field",
        )
    except Exception as exc:  # noqa: BLE001 - network/auth issues fall back to static
        log.warning("BMO live fuel-rate fetch failed; using static CSV: %s", exc)
        return {}

    rates: dict[str, float | str] = {}
    for field, rate_key in field_to_rate_key.items():
        if field not in df.columns:
            continue
        values = pd.to_numeric(df[field], errors="coerce").dropna()
        values = values[values > 0]
        if values.empty:
            continue
        rates[rate_key] = float(values.mean())
        prefix = rate_key.removesuffix("_rate_kg_thm")
        rates[f"{prefix}_source"] = f"influx_1h_avg.{field}"
    return rates


@_data_cache(show_spinner=False, ttl=600)
def _model_input_defaults_from_static_csv(
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


def _cached_fuel_analysis_snapshot(
    provider: EvonithBmoContextProvider,
    mode: str,
    window_days: int,
    cache_version: int,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    return _session_cached_source(
        cache_version,
        ("fuel_analysis", mode, window_days),
        lambda: provider.get_fuel_analysis_snapshot(
            mode=mode, window_days=window_days
        ),
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
def _render_static_dataset_bar(
    bmo_cfg: dict[str, Any], refresh_result: dict[str, Any] | None = None
) -> None:
    if st.session_state.pop(_STATIC_DATASET_SOURCE_CHANGED_KEY, False):
        st.rerun()

    status = _static_dataset_status(bmo_cfg)
    state = status["state"] if status["exists"] else "missing"
    source_url = _configured_static_dataset_url(bmo_cfg)
    with st.expander("Data sources", expanded=False):
        use_link = st.toggle(
            "Fetch dataset through DATA_URL",
            value=_use_static_dataset_link(bmo_cfg),
            key=_STATIC_DATASET_LINK_KEY,
            disabled=not bool(source_url),
            help=(
                "On: download the published furnace CSV. "
                "Off: rebuild it through the normal database/code pipeline."
            ),
            on_change=_mark_static_dataset_source_changed,
        )
        st.caption(
            "Selected dataset source: "
            + ("Published link" if use_link else "Normal code pipeline")
        )
        cols = st.columns([1.1, 1.2, 1.0])
        cols[0].metric("Static Dataset", state.title())
        cols[1].metric("Latest Data", status["latest_data_end"] or "Unknown")
        last_fetch = status["last_fetch"]
        cols[2].metric(
            "Last Fetch",
            last_fetch.strftime("%Y-%m-%d %H:%M") if last_fetch else "Never",
        )
        if use_link:
            st.caption(
                "The published furnace CSV is fetched automatically at most once "
                "per hour; the local CSV is retained as the fallback."
            )
            st.caption(f"Published source: {source_url}")
        else:
            st.caption(
                "The normal pipeline builds the dataset from the offline database "
                "and configured online process sources."
            )
        if refresh_result and refresh_result.get("error"):
            st.warning(
                "Dataset refresh failed; using the last local copy. "
                f"Error: {refresh_result['error']}"
            )
        if state == "stale":
            st.caption(
                "The selected dataset source has not refreshed successfully "
                "in the last hour."
            )
        elif state == "missing":
            st.warning("No usable furnace CSV is available.")
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
        "bmo_de_candidates",
    ):
        st.session_state.pop(key, None)


def _fuel_ash_cfg_with_recent_rates(
    fuel_ash_cfg: list[dict[str, Any]],
    fuel_rates: dict[str, float | str],
    fuel_analysis: Mapping[str, Mapping[str, float]] | None = None,
) -> list[dict[str, Any]]:
    rate_keys = {
        "coke": "coke_rate_kg_thm",
        "nut_coke": "nut_coke_rate_kg_thm",
        "pci": "pci_rate_kg_thm",
    }
    rows: list[dict[str, Any]] = []
    for item in fuel_ash_cfg or []:
        row = dict(item)
        fuel_id = str(row.get("fuel_id", "")).strip()
        rate_key = rate_keys.get(fuel_id)
        rate = fuel_rates.get(rate_key or "")
        if rate is not None:
            row["rate_kg_per_thm"] = float(rate)
            if fuel_id == "nut_coke":
                # The live/default nut-coke tag is the base 70 kg/THM quantity;
                # its TM is added later to produce the wet charge quantity.
                row["add_moisture_to_rate"] = True
        analysis = (fuel_analysis or {}).get(fuel_id, {})
        for field in ("moisture_pct", "vm_pct"):
            value = analysis.get(field)
            if value is not None:
                row[field] = float(value)
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
    *,
    target_hot_metal_mt: float | None = None,
    fuel_ash_inputs: list[FuelAshInput] | None = None,
    flux_inputs: list[FluxInput] | None = None,
    dust_inputs: list[DustInput] | None = None,
    slag_balance_settings: SlagBalanceSettings | None = None,
    charge_mass_mt: float = 26.4,
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
        return (
            {},
            0.0,
            ["Manual shares sum to zero; enter at least one positive share."],
        )
    shares = {oid: max(0.0, float(v)) / total_share for oid, v in shares_pct.items()}
    fe_per_blend_mt = 0.0
    for ore in ores:
        dry_fraction = max(0.0, 1.0 - float(ore.chemistry.moisture_pct) / 100.0)
        fe_fraction = max(0.0, float(ore.chemistry.fe_t_pct) / 100.0)
        fe_per_blend_mt += shares.get(ore.ore_id, 0.0) * dry_fraction * fe_fraction
    if fe_per_blend_mt <= 0:
        return (
            {},
            total_share,
            ["Manual blend Fe% is unavailable; cannot scale to target HM."],
        )
    total_qty = float(target_fe_mt) / fe_per_blend_mt
    quantities = {oid: sh * total_qty for oid, sh in shares.items()}
    if target_hot_metal_mt and slag_balance_settings and slag_balance_settings.enabled:
        quantities = scale_ore_quantities_to_hot_metal(
            ores=ores,
            reference_quantities_mt=quantities,
            target_hot_metal_mt=target_hot_metal_mt,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            charge_mass_mt=charge_mass_mt,
        )
    return quantities, total_share, []


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
    charge_mass_mt: float,
    manual_ores: list[OreInput] | None = None,
) -> None:
    """Operator-focused comparison of the manual blend vs the optimizer blends.

    Presents two tables on the same all-source iron-closure basis so the operator can decide at
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
         - target_fe_mt: float - Initial Fe-only scale used to seed the full closure.
         - target_production_mt: float - HM basis for cost / slag / model fields.
         - feo_in_slag_pct: float - FeO assumed to report into slag.
         - fuel_ash_inputs / flux_inputs / dust_inputs / slag_balance_settings - Slag-balance inputs.
         - charge_mass_mt: float - Tonnes carried by one furnace charge. Charging
           runs 24 h, so that is a constant rather than an argument.
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
    snapshot = provider.get_recent_manual_blend_snapshot(
        list(manual_ores_by_id.values())
    )
    rows_by_ore = {str(row.get("ore_id")): row for row in snapshot.get("rows", [])}
    compare_ores = selected_ores

    st.markdown("##### Manual blend vs optimizer")
    st.caption(
        "Edit the manual Share (%) to try any burden split. Shares are normalised to "
        "100% and scaled through the same all-source Fe/material closure as the "
        "optimizer, so every option is compared on the same basis."
    )
    start_time, end_time = snapshot.get("start_time"), snapshot.get("end_time")
    if rows_by_ore and start_time and end_time:
        st.caption(f"Manual blend seeded from last shift ({start_time} to {end_time}).")
    elif not rows_by_ore:
        st.caption(
            "No last-shift manual blend found; seeded from the optimizer shares."
        )

    seed_rows = []
    for ore in compare_ores:
        seed_share = float(rows_by_ore.get(ore.ore_id, {}).get("share_pct", 0.0) or 0.0)
        if seed_share <= 0:
            seed_share = float(primary_blend.shares_pct.get(ore.ore_id, 0.0))
        seed_rows.append(
            {
                "ore_id": ore.ore_id,
                "ore_name": ore.display_name,
                "manual_share_pct": seed_share,
            }
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
    manual_quantities, normalized_total, scale_warnings = (
        _target_quantities_from_shares(
            manual_shares_pct,
            compare_ores,
            target_fe_mt,
            target_hot_metal_mt=target_production_mt,
            fuel_ash_inputs=fuel_ash_inputs,
            flux_inputs=flux_inputs,
            dust_inputs=dust_inputs,
            slag_balance_settings=slag_balance_settings,
            charge_mass_mt=charge_mass_mt,
        )
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
            manual_si = _predict_blend_si(
                ores=compare_ores,
                quantities_mt=manual_quantities,
                process_context=process_context,
                history_df=history_df,
                hot_metal_target_mt=target_production_mt,
            )
            # The manual blend describes current operation, so it is also the
            # best available "current burden" for the correction's reference.
            # Persisting it lets the next optimizer run anchor the oxygen and Si
            # terms to a real burden instead of switching them off.
            st.session_state["bmo_manual_quantities_mt"] = dict(manual_quantities)
            st.session_state["bmo_manual_si"] = manual_si
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
                # Manual blend = current operation: show the realised fuel cost
                # from the actual current rates (Fuel Ash table, seeded from the
                # latest static-dataset row) at current prices. Optimized blends
                # instead convert the ML-predicted cost (default "model_cost").
                fuel_rate_basis="inputs",
                # Computed and shown but not applied (apply_to_manual_blend is
                # false): the manual blend is the realised-cost reference, and a
                # large correction here is the clearest signal that the reference
                # operating point itself is mis-set.
                coke_correction_settings=coke_correction_settings,
                coke_correction_reference=_build_coke_correction_reference(
                    settings=coke_correction_settings,
                    observed_slag_rate_kg_per_thm=observed_slag_rate,
                    flux_inputs=flux_inputs,
                    hot_metal_target_mt=target_production_mt,
                    process_context=process_context,
                    current_quantities_mt=manual_quantities,
                    ores=compare_ores,
                    current_si_pct=manual_si,
                ),
                hot_metal_si_pct=manual_si,
                charge_mass_mt=charge_mass_mt,
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
    charging_by_blend_id = {
        id(blend): compute_charging_requirements(
            blend,
            charge_mass_mt=charge_mass_mt,
        )
        for _, blend, _ in options
    }

    def _charging_value(blend: Any, key: str) -> float | None:
        value = charging_by_blend_id[id(blend)].get(key)
        return float(value) if value is not None else None

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
            label: st.column_config.NumberColumn(label, format="%.1f")
            for label in labels
        },
    )

    # ---- Outputs: the KPIs an operator decides on ----
    def _basicity(blend: Any, denom_key: str, attr: str) -> float | None:
        if float(blend.diagnostics.get(denom_key, 0.0) or 0.0) <= 0:
            return None
        return float(getattr(blend, attr, 0.0) or 0.0)

    def _rate_field(blend: Any, estimate_key: str, field: str) -> float | None:
        value = (blend.diagnostics.get(estimate_key) or {}).get(field)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _fuel_rate_total(blend: Any) -> float | None:
        return _rate_field(blend, "fuel_rate_estimate", "total_fuel_rate_kg_thm")

    # Fuel + total costs are shown re-priced at the operator's current fuel
    # prices when available (display-only; the optimizer used baseline prices).
    def _display_fuel(blend: Any) -> float:
        adjusted = blend.diagnostics.get("adjusted_fuel_cost_per_thm_rs")
        return (
            float(adjusted)
            if adjusted is not None
            else float(blend.fuel_cost_per_thm_rs)
        )

    def _flux_cost(blend: Any) -> float:
        return float(blend.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0)

    def _display_total(blend: Any) -> float:
        # Total = ore + (re-priced) fuel + optimizer-added flux, so the cheapest
        # option and the saving vs manual account for the flux spend too.
        adjusted = blend.diagnostics.get("adjusted_objective_rs_per_thm")
        base = (
            float(adjusted)
            if adjusted is not None
            else float(blend.objective_rs_per_thm)
        )
        return base + _flux_cost(blend)

    # (row label, accessor(blend, si) -> value | None, format string)
    metric_specs = [
        (
            "Production",
            lambda b, si: float(b.diagnostics.get("hot_metal_target_mt", 0.0) or 0.0),
            "{:,.1f} MT",
        ),
        ("Total Cost (Rs/THM)", lambda b, si: _display_total(b), "{:,.0f}"),
        ("Ore Cost (Rs/THM)", lambda b, si: b.ore_cost_per_thm_rs, "{:,.0f}"),
        (
            "Flux Rate (kg/THM)",
            lambda b, si: _charging_value(b, "flux_rate_kg_per_thm"),
            "{:,.1f}",
        ),
        (
            "Coke in Charges (MT)",
            lambda b, si: _charging_value(b, "coke_total_mt"),
            "{:,.1f}",
        ),
        (
            "Nut Coke in Charges (MT)",
            lambda b, si: _charging_value(b, "nut_coke_total_mt"),
            "{:,.1f}",
        ),
        (
            "PCI in Charges (MT)",
            lambda b, si: _charging_value(b, "pci_total_mt"),
            "{:,.1f}",
        ),
        (
            "Required Charges (/hr)",
            lambda b, si: _charging_value(b, "required_charges_per_hour"),
            "{:,.2f}",
        ),
        (
            "Chemical Hotmetal per Charge (MT)",
            lambda b, si: _charging_value(b, "chemical_hot_metal_per_charge_mt"),
            "{:,.2f}",
        ),
        # 1-decimal so small but real blend-to-blend differences aren't hidden by
        # rounding (the fuel model is only weakly blend-sensitive -- see the help
        # note on the outcomes table).
        ("Fuel Cost (Rs/THM)", lambda b, si: _display_fuel(b), "{:,.1f}"),
        ("Flux Cost (Rs/THM)", lambda b, si: _flux_cost(b), "{:,.1f}"),
        ("Fuel Rate (kg/THM)", lambda b, si: _fuel_rate_total(b), "{:,.1f}"),
        # Uncorrected and corrected side by side: a correction is only worth
        # trusting if the number it replaced is still visible next to it.
        (
            "Coke Rate, uncorrected (kg/THM)",
            lambda b, si: _rate_field(
                b, "fuel_rate_estimate_anchor", "coke_rate_kg_thm"
            ),
            "{:,.1f}",
        ),
        (
            "Coke Rate, corrected (kg/THM)",
            lambda b, si: _rate_field(b, "fuel_rate_estimate", "coke_rate_kg_thm"),
            "{:,.1f}",
        ),
        (
            "Coke Correction (kg/THM)",
            lambda b, si: (
                float(b.diagnostics["coke_correction_delta_kg_thm"])
                if "coke_correction_delta_kg_thm" in b.diagnostics
                else None
            ),
            "{:+,.1f}",
        ),
        ("Hot-Metal Si (%)", lambda b, si: si, "{:,.3f}"),
        (
            "Slag Basicity (CaO/SiO2)",
            lambda b, si: _basicity(b, "slag_basicity_denominator_mt", "slag_basicity"),
            "{:,.3f}",
        ),
        (
            # Constrained. Plant runs ~1.31.
            "Slag T-Basicity (CaO+MgO)/SiO2",
            lambda b, si: _basicity(
                b, "slag_t_basicity_denominator_mt", "slag_t_basicity"
            ),
            "{:,.3f}",
        ),
        (
            # Display only. Reads ~0.84 at the plant, not ~1.31, because Al2O3
            # joins the denominator - the two are easily mistaken for each other.
            "IB4 (CaO+MgO)/(SiO2+Al2O3)",
            lambda b, si: _basicity(b, "slag_ib4_denominator_mt", "slag_ib4"),
            "{:,.3f}",
        ),
        ("Slag Rate (kg/THM)", lambda b, si: b.slag_rate_kg_per_thm, "{:,.0f}"),
        (
            "Slag Al2O3 (%)",
            lambda b, si: _basicity(
                b, "slag_chemistry_denominator_mt", "slag_al2o3_pct"
            ),
            "{:,.2f}",
        ),
        (
            "Slag MgO (%)",
            lambda b, si: _basicity(b, "slag_chemistry_denominator_mt", "slag_mgo_pct"),
            "{:,.2f}",
        ),
        (
            "Slag MgO/Al2O3",
            lambda b, si: _basicity(
                b, "slag_mgo_al2o3_denominator_mt", "slag_mgo_al2o3_ratio"
            ),
            "{:,.3f}",
        ),
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
                    "Manual Fuel Cost is the realised cost: actual current "
                    "coke/nut-coke/PCI rates at current prices. LP/DE Fuel Cost "
                    "converts the ML-predicted fuel cost for that blend to "
                    "current prices; the optimizer itself still minimises the "
                    "model's baseline-price objective. Fuel Rate is the physical "
                    "rate basis behind each display."
                ),
            ),
        },
    )

    # Headline: cheapest option by total cost + optimizer saving vs the manual
    # blend, both at the operator's current fuel prices (display-only).
    cheapest_label = min(options, key=lambda option: _display_total(option[1]))[0]
    cols = st.columns(len(options))
    for col, (label, blend, _) in zip(cols, options):
        col.metric(
            f"{label} (Rs/THM)",
            f"{_display_total(blend):,.0f}",
            delta="cheapest" if label == cheapest_label else None,
            delta_color="normal" if label == cheapest_label else "off",
        )
    if manual_blend is not None:
        best_optimizer = min(
            (blend for _, blend, _ in optimizer_candidates),
            key=_display_total,
        )
        saving = _display_total(manual_blend) - _display_total(best_optimizer)
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
        fuel_analysis = diagnostics.get("fuel_analysis", {})
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
                    "area": "fuel_analysis",
                    "source": fuel_analysis.get(
                        "source", "offline_feed.fuel_chemistry"
                    ),
                    "timestamp/window": f"{fuel_analysis.get('start_time', '')} -> {fuel_analysis.get('end_time', '')}",
                    "rows": fuel_analysis.get("returned_rows", 0),
                    "note": f"Moisture for fuel basis; VM for ash analysis, mode={fuel_analysis.get('mode', '')}",
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
            st.markdown("##### Fuel Analysis")
            _show_df(pd.DataFrame(fuel_analysis.get("rows", [])))
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


def _render_energy_assumptions() -> None:
    """Every energy-balance number the plant has not measured, as an input table.

    These are the figures currently carrying literature or assumed values. The
    balance's structure is an accounting identity and needs no calibration, but
    these scalars do, and each one silently scales a real term. Putting them in
    front of the operator is the only way a guess ever gets replaced by a
    measurement.

    Physics is deliberately absent from this table - iron oxide reduction
    enthalpy, calorific values, molar volume. Overriding those would be breaking
    the balance rather than calibrating it.
    """

    from utils.energy_balance.assumptions import (
        BY_KEY,
        current_values,
        load_overrides,
        save_overrides,
    )

    with st.expander("Plant assumptions — operator input", expanded=False):
        st.caption(
            "Values the plant has not measured. Anything you enter here is used "
            "by the energy balance and the process recommendation, and is "
            "remembered between sessions. Leave a row alone to keep the shipped "
            "default."
        )

        rows = current_values()
        assumed = sum(1 for r in rows if r["Source"] == "assumed")
        supplied = sum(1 for r in rows if r["Source"] == "operator")
        cols = st.columns(3)
        cols[0].metric("Parameters", len(rows))
        cols[1].metric("Still assumed", assumed)
        cols[2].metric("You have supplied", supplied)

        frame = pd.DataFrame(rows)
        edited = st.data_editor(
            frame[["Parameter", "Value", "Unit", "Source", "Default",
                   "Basis", "Why it matters"]],
            hide_index=True,
            use_container_width=True,
            key="energy_assumptions_editor",
            column_config={
                "Value": st.column_config.NumberColumn(
                    "Value", help="Your figure. Bounds are enforced on save.",
                    format="%.3f",
                ),
                "Source": st.column_config.TextColumn("Source", disabled=True),
                "Default": st.column_config.NumberColumn(
                    "Shipped", disabled=True, format="%.3f"
                ),
                "Unit": st.column_config.TextColumn("Unit", disabled=True),
                "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
                "Basis": st.column_config.TextColumn(
                    "Where the default came from", disabled=True, width="large"
                ),
                "Why it matters": st.column_config.TextColumn(
                    "Why it matters", disabled=True, width="large"
                ),
            },
        )

        left, right = st.columns([1, 1])
        if left.button("Save assumptions", key="save_energy_assumptions"):
            # Match rows by position: Parameter is disabled, so the editor
            # cannot reorder or rename them.
            updated = {
                rows[i]["key"]: float(value)
                for i, value in enumerate(edited["Value"].tolist())
            }
            out_of_range = [
                BY_KEY[key].label
                for key, value in updated.items()
                if not (BY_KEY[key].minimum <= value <= BY_KEY[key].maximum)
            ]
            path = save_overrides(updated)
            if out_of_range:
                st.warning(
                    "Clamped to their physical bounds: " + ", ".join(out_of_range)
                )
            st.success(f"Saved to {path.name}. Re-run the optimiser to apply.")
        if right.button("Reset to shipped defaults", key="reset_energy_assumptions"):
            save_overrides({})
            st.success("Reset. Re-run the optimiser to apply.")

        if load_overrides():
            st.caption(
                "⚠️ Operator values are in force. The worked example in "
                "`docs/energy_balance_calculation_procedure.md` was generated "
                "with the shipped defaults, so it will no longer match exactly."
            )
        st.caption(
            "Highest-impact unknowns are listed first. The two dust-carbon rows "
            "are the weakest numbers in the whole balance — one lab analysis of "
            "each dust stream would settle them."
        )


def _render_process_recommendation(
    blend: Any,
    *,
    label: str,
    hot_metal_mt: float,
    ores: list[Any],
    hm_chem_values: dict[str, float],
    hm_snapshot: dict[str, Any],
    flux_inputs: list[Any],
    fuel_ash_inputs: list[Any],
) -> None:
    """Layer 2: control settings for the blend Layer 1 has just chosen.

    The blend is an input. This section never reconsiders it - it asks only
    what blast settings supply the energy that blend demands, at least fuel
    cost.
    """

    from utils.bmo.process_recommendation import (
        ControlSettings,
        blend_to_energy_inputs,
        recommend_controls,
    )
    from utils.energy_balance.assumptions import apply_overrides as apply_energy_overrides
    from utils.energy_balance.constants import load_config as load_energy_config

    st.markdown("##### Recommended process parameters")
    snapshot = _live_process_snapshot()
    if not snapshot.get("hot_blast_vol_nm3h"):
        st.info(
            "Live blast and top-gas tags are unavailable, so control parameters "
            "cannot be recommended for this blend. The blend itself is unaffected."
        )
        return

    fuel_rates = blend.diagnostics.get("fuel_rate_estimate") or {}
    if not fuel_rates:
        st.info("Fuel-rate estimate unavailable; cannot run the energy balance.")
        return

    flux_mt = sum(
        max(0.0, float(flux.wet_qty_mt or 0.0)) for flux in flux_inputs if flux.enabled
    )
    vm_by_fuel = {
        str(fuel.fuel_id): float(getattr(fuel, "vm_pct", 0.0) or 0.0)
        for fuel in fuel_ash_inputs
    }
    try:
        energy_inputs = blend_to_energy_inputs(
            blend,
            hot_metal_mt=hot_metal_mt,
            ores=ores,
            fuel_rates_kg_per_thm=fuel_rates,
            hm_chemistry={
                "carbon_pct": hm_chem_values.get("carbon_pct", 4.3),
                "silicon_pct": hm_chem_values.get("silicon_pct", 0.5),
                "iron_pct": float(hm_snapshot.get("hm_fe_pct_for_target") or 94.5),
                "manganese_pct": float(hm_snapshot.get("chem_pct_mn") or 0.2),
                "slag_feo_pct": float(hm_snapshot.get("slag_pct_feo") or 0.4),
            },
            process_snapshot=snapshot,
            flux_mt=flux_mt,
            fuel_vm_pct=vm_by_fuel,
            shell_loss_gj_per_hr=snapshot.get("shell_loss_gj_per_hr"),
        )
        current = ControlSettings(
            blast_temperature_c=snapshot.get("hot_blast_temp", 0.0),
            oxygen_enrichment_pct=snapshot.get("oxygen_enrichment_pct", 0.0),
            blast_volume_nm3_per_hr=snapshot.get("hot_blast_vol_nm3h", 0.0),
            pci_kg_per_thm=float(fuel_rates.get("pci_rate_kg_thm", 0.0) or 0.0),
            hot_blast_pressure_bar=snapshot.get("hot_blast_press", 0.0),
            top_pressure_bar=snapshot.get("top_press_avg", 0.0),
            steam_kg_per_hr=snapshot.get("steam_injection", 0.0),
        )
        release_pci = st.checkbox(
            "Allow PCI to move",
            value=False,
            key=f"bmo_pr_pci_{label}",
            help="PCI is held by default. Release it only when you intend to change it.",
        )
        recommendation = recommend_controls(
            blend_inputs=energy_inputs,
            current=current,
            prices_rs_per_kg={
                str(fuel.fuel_id): float(fuel.price_rs_per_mt or 0.0) / 1000.0
                for fuel in fuel_ash_inputs
            },
            optimise_pci=release_pci,
            # Operator-supplied values for the unmeasured constants. Applied
            # here at the app boundary rather than inside the math layer, so
            # the balance stays pure and tests stay hermetic - a saved override
            # file on a developer's machine must not change test results.
            energy_cfg=apply_energy_overrides(load_energy_config()),
        )
    except Exception as exc:  # noqa: BLE001 - never break the blend result pane
        log.warning("Process recommendation failed: %s", exc)
        st.warning(f"Could not compute process parameters: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Coke Rate (kg/THM)",
        f"{recommendation.coke_rate_kg_per_thm:,.1f}",
        delta=f"{recommendation.coke_rate_kg_per_thm - recommendation.current_coke_rate_kg_per_thm:+,.1f}",
        delta_color="inverse",
        help=(
            "ENERGY BALANCE figure, not a forecast of the plant's coke rate. "
            "It is what the balance says this burden needs at these controls. "
            "Trust the CHANGE, not the level - the level carries a shell-loss "
            "uncertainty that largely cancels between the two settings. See the "
            "coke rate comparison below."
        ),
    )
    c2.metric(
        "Fuel Cost (Rs/THM)",
        f"{recommendation.fuel_cost_rs_per_thm:,.0f}",
        delta=f"{-recommendation.fuel_cost_saving_rs_per_thm:+,.0f}",
        delta_color="inverse",
    )
    raft_delta = recommendation.raft_delta_c
    c3.metric(
        "RAFT (C)",
        f"{recommendation.raft_c:,.0f}" if recommendation.raft_c else "n/a",
        delta=(f"{raft_delta:+,.0f}" if raft_delta is not None else None),
        delta_color="off",
        help=(
            "Computed directly from the recommended controls: blast temperature, "
            "oxygen enrichment, blast moisture + steam, and PCI as a "
            "concentration in the blast. Forward-validated against body_raft, "
            "MAE 17 C and R2 0.63 - so a change smaller than about 17 C is "
            "not meaningful."
        ),
    )
    if raft_delta is not None and recommendation.current_raft_c:
        direction = (
            "hotter" if raft_delta > 0 else "colder" if raft_delta < 0 else "unchanged"
        )
        strength = "within measurement noise" if abs(raft_delta) < 17.0 else "significant"
        st.caption(
            f"RAFT moves from **{recommendation.current_raft_c:,.0f} °C** to "
            f"**{recommendation.raft_c:,.0f} °C** — raceway runs **{direction}** "
            f"by {abs(raft_delta):,.0f} °C ({strength}, formula is good to ±17 °C)."
        )

    # Three different coke rates exist in this app and they do not agree. Naming
    # them side by side is the only way to stop the comparison being misread.
    plant_coke = float(fuel_rates.get("coke_rate_kg_thm", 0.0) or 0.0)
    with st.expander("Where does this coke rate come from?", expanded=False):
        comparison = [
            {
                "Figure": "Plant actual (recent charge reports)",
                "kg/THM": round(plant_coke, 1) if plant_coke else None,
                "What it is": "What the furnace is actually being charged.",
            },
            {
                "Figure": "Energy balance, at CURRENT controls",
                "kg/THM": round(recommendation.current_coke_rate_kg_per_thm, 1),
                "What it is": (
                    "What the balance says this burden needs right now. The gap "
                    "against plant actual is the balance's own bias."
                ),
            },
            {
                "Figure": "Energy balance, at RECOMMENDED controls",
                "kg/THM": round(recommendation.coke_rate_kg_per_thm, 1),
                "What it is": "The headline figure above.",
            },
        ]
        st.dataframe(pd.DataFrame(comparison), hide_index=True,
                     use_container_width=True)
        bias = (
            recommendation.current_coke_rate_kg_per_thm - plant_coke
            if plant_coke else None
        )
        st.markdown(
            "**Only the last two are comparable.** Both come from the same "
            "energy balance, so whatever bias it carries cancels between them — "
            "that is why the delta is the number to act on and the level is not."
            + (
                f"\n\nAgainst plant actual the balance currently reads "
                f"**{bias:+,.1f} kg/THM**. "
                + (
                    "That is within the model's expected accuracy."
                    if abs(bias) < 15
                    else "That is larger than expected — most likely the "
                    "shell-loss question in "
                    "`docs/energy_balance_findings_and_open_decisions.md` §5, "
                    "which moves this figure by up to 11%."
                )
                if bias is not None else ""
            )
            + "\n\nThe **ML fuel-cost model** is a third, separate estimate. It "
            "is trained on plant history and is blend-blind; the energy balance "
            "is physics and responds to the blend. They will not agree, and are "
            "not meant to."
        )

    optimised = set(recommendation.diagnostics["optimised_controls"])
    rows = []
    for key, delta in recommendation.deltas().items():
        rows.append(
            {
                "Control": key.replace("_", " ").title(),
                "Current": getattr(recommendation.current, key),
                "Recommended": getattr(recommendation.settings, key),
                "Change": delta,
                "Role": "optimised" if key in optimised else "pass-through",
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Current": st.column_config.NumberColumn("Current", format="%.2f"),
            "Recommended": st.column_config.NumberColumn("Recommended", format="%.2f"),
            "Change": st.column_config.NumberColumn("Change", format="%+.2f"),
            "Role": st.column_config.Column(
                "Role",
                help=(
                    "Hot blast pressure, top pressure and steam do not appear in "
                    "an energy balance - they act through permeability and gas "
                    "utilisation - so they are shown unchanged rather than given "
                    "a fabricated value."
                ),
            ),
        },
    )
    for warning in recommendation.warnings:
        st.warning(warning)


def _render_transition_ladder(
    *,
    provider: Any,
    ores: list[Any],
    lp_kwargs: dict[str, Any],
    slag_rate_cap_kg_per_thm: float | None,
) -> None:
    """The path from what the plant is charging today to the LP optimum.

    The LP says where to go. It does not say how to get there, and a
    20-percentage-point share change is not an instruction anyone can act on -
    burden descent takes 6-7 hours and ore supply does not turn overnight. Each
    rung here is a full LP solve under a per-ore move cap, so every step on the
    path independently satisfies all six slag limits.
    """

    from utils.bmo.transition import build_transition_ladder

    st.markdown("##### Path from the current blend")
    move_col, _ = st.columns([1, 3])
    move_pct = move_col.number_input(
        "Max share change per step (%)",
        min_value=1.0, max_value=50.0, value=5.0, step=1.0,
        key="bmo_transition_move_pct",
        help=(
            "Your step-change policy. Smaller steps mean more shifts to reach "
            "the optimum but a gentler move for the furnace."
        ),
    )

    snapshot = provider.get_recent_manual_blend_snapshot(ores)
    manual_shares = {
        str(row.get("ore_id")): float(row.get("share_pct", 0.0) or 0.0)
        for row in snapshot.get("rows", [])
    }
    if not any(manual_shares.values()):
        st.info(
            "No recent manual blend found, so there is nothing to transition "
            "from. The optimal blend above still stands."
        )
        return

    try:
        ladder = build_transition_ladder(
            ores, manual_shares,
            max_share_move_pct=float(move_pct),
            _slag_rate_cap=slag_rate_cap_kg_per_thm,
            **lp_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - never break the result pane
        log.warning("Transition ladder failed: %s", exc)
        st.warning(f"Could not build the transition path: {exc}")
        return

    if not ladder.start_is_admissible:
        recovery = ladder.diagnostics.get("recovery_move_pct")
        note = (
            "**The blend currently being charged is outside your limits.** "
            "Step 1 below brings it back inside — make that move first, then "
            "continue down the path.\n- " + "\n- ".join(ladder.start_violations)
        )
        if recovery:
            note += (
                f"\n\n**Step 1 needs a {recovery:.1f}% share move**, more than "
                f"the {ladder.max_share_move_pct:.1f}% step you have set — "
                "recovery is not possible within it."
            )
        st.warning(note)

    rows = []
    if ladder.start_blend is not None:
        rows.append(
            {
                "Step": "now",
                **{
                    ore.display_name: ladder.start_shares_pct.get(ore.ore_id, 0.0)
                    for ore in ores
                },
                "Ore Cost (Rs/THM)": ladder.start_blend.ore_cost_per_thm_rs,
                "Slag (kg/THM)": ladder.start_blend.slag_rate_kg_per_thm,
                "B2": ladder.start_blend.slag_basicity,
                "Binding": "",
            }
        )
    for rung in ladder.rungs:
        if not rung.feasible:
            rows.append({"Step": str(rung.index), "Binding": "infeasible"})
            continue
        rows.append(
            {
                "Step": str(rung.index),
                **{
                    ore.display_name: rung.shares_pct.get(ore.ore_id, 0.0)
                    for ore in ores
                },
                "Ore Cost (Rs/THM)": rung.blend.ore_cost_per_thm_rs,
                "Slag (kg/THM)": rung.blend.slag_rate_kg_per_thm,
                "B2": rung.blend.slag_basicity,
                "Binding": "; ".join(rung.binding_limits),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Binding": st.column_config.Column(
                "Binding",
                help=(
                    "What stops this step going further - a slag limit, or an "
                    "ore's own maximum share. Raising the named limit is what "
                    "would unlock more saving."
                ),
            ),
        },
    )

    saving = ladder.ore_cost_saving_rs_per_thm()
    steps = len([r for r in ladder.rungs if r.feasible])
    if saving:
        st.success(
            f"**{steps} step{'s' if steps != 1 else ''}** to reach the optimum, "
            f"worth **Rs {saving:,.0f}/THM** in ore cost."
        )
    if not ladder.converged:
        st.info(
            "The path had not converged within the step limit - the optimum is "
            "further away than the steps allowed. Raise the step size or accept "
            "a longer transition."
        )


def _render_si_metric(si_value: float | None) -> None:
    """Render the display-only predicted hot-metal Si for a blend."""

    if si_value is None:
        return
    st.metric("Predicted Hot-Metal Si", f"{si_value:.3f} %")
    st.caption(
        "Advisory only - standalone Si model; not an optimization objective or "
        "constraint and does not affect the blend decision."
    )


def _render_lp_flux_additions(blend: Any) -> None:
    """Show the flux quantities the optimizer added to hold slag basicity in bounds."""

    flux_qty = (getattr(blend, "diagnostics", None) or {}).get(
        "lp_flux_quantities_mt"
    ) or {}
    added = {str(k): float(v) for k, v in flux_qty.items() if float(v) > 1e-6}
    if not added:
        return
    st.markdown("###### Flux to add (basicity control)")
    cols = st.columns(len(added))
    for col, (flux_id, qty) in zip(cols, added.items()):
        col.metric(f"{flux_id.title()} (MT)", f"{qty:,.1f}")
    st.caption(
        "Chosen by the optimizer to keep slag basicity within the target bounds "
        "(dolomite raises basicity, quartz lowers it)."
    )


def _render_de_exploration(
    candidates: list[dict[str, Any]] | None,
    selected_ores: list[OreInput] | None = None,
) -> None:
    """Render the DE search cloud as a 2D basicity/slag scatter + top-100 table.

    Each dot is one blend the optimizer evaluated: X = slag basicity,
    Y = slag quantity (MT), colour = total cost (ore + fuel + flux), running
    green (cheap) to red (expensive) on a black background. The table below
    lists the 100 cheapest distinct candidates as alternative solutions,
    including the full blend combination (ore shares + flux additions).

    Args:
         - candidates: list[dict[str, Any]] | None - Per-evaluation records with
           ``total_cost_rs_per_thm``, ``slag_basicity``, ``slag_mt``, ``feasible``,
           and optionally ``shares_pct`` / ``flux_mt`` blend-combination dicts.
         - selected_ores: list[OreInput] | None - Ores of this run, used to map
           ore ids to display names for the table columns.

    Returns:
         - return None - Renders directly to the Streamlit page.
    """

    rows = [c for c in (candidates or []) if isinstance(c, dict)]
    if not rows:
        return

    df = pd.DataFrame(rows)
    required = ("total_cost_rs_per_thm", "slag_basicity", "slag_mt")
    if any(col not in df.columns for col in required):
        return
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    if df.empty:
        return

    st.markdown("###### Optimizer search space")
    st.caption(
        f"{len(df):,} candidate blends evaluated. Each dot is one blend the "
        "optimizer tried — colour is total cost (ore + fuel + flux): green is "
        "cheaper, red is costlier. X = slag basicity, Y = slag (MT)."
    )

    cost = df["total_cost_rs_per_thm"]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["slag_basicity"],
                y=df["slag_mt"],
                mode="markers",
                marker=dict(
                    size=7,
                    color=cost,
                    colorscale=[
                        [0.0, "#00e676"],
                        [0.5, "#ffd600"],
                        [1.0, "#ff1744"],
                    ],
                    opacity=0.85,
                    line=dict(width=0),
                    colorbar=dict(
                        title=dict(
                            text="Total Cost (Rs/THM)", font=dict(color="white")
                        ),
                        tickfont=dict(color="white"),
                        outlinecolor="#333333",
                    ),
                ),
                customdata=cost,
                hovertemplate=(
                    "Basicity %{x:.3f}<br>Slag %{y:,.0f} MT<br>"
                    "Cost %{customdata:,.0f} Rs/THM<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            title="Slag Basicity (CaO/SiO2)",
            color="white",
            gridcolor="#333333",
            zerolinecolor="#333333",
        ),
        yaxis=dict(
            title="Slag (MT)",
            color="white",
            gridcolor="#333333",
            zerolinecolor="#333333",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top 100 cheapest distinct candidates as alternative solutions.
    ranked = (
        df.assign(
            _b=df["slag_basicity"].round(3),
            _s=df["slag_mt"].round(1),
            _c=df["total_cost_rs_per_thm"].round(0),
        )
        .drop_duplicates(subset=["_b", "_s", "_c"])
        .sort_values("total_cost_rs_per_thm")
        .head(100)
        .reset_index(drop=True)
    )
    feasible_vals = (
        ranked["feasible"].astype(bool).tolist()
        if "feasible" in ranked.columns
        else [True] * len(ranked)
    )
    table = pd.DataFrame(
        {
            "Rank": list(range(1, len(ranked) + 1)),
            "Total Cost (Rs/THM)": ranked["total_cost_rs_per_thm"].round(0).tolist(),
            "Slag Basicity": ranked["slag_basicity"].round(3).tolist(),
            "Slag (MT)": ranked["slag_mt"].round(1).tolist(),
        }
    )

    # Blend combination columns: one share column per ore, one MT column per
    # flux the optimizer controls. Older session records may lack these dicts.
    ore_names = {ore.ore_id: ore.display_name for ore in (selected_ores or [])}
    if "shares_pct" in ranked.columns:
        share_dicts = [d if isinstance(d, dict) else {} for d in ranked["shares_pct"]]
        ore_ids: list[str] = []
        for d in share_dicts:
            for ore_id in d:
                if ore_id not in ore_ids:
                    ore_ids.append(ore_id)
        for ore_id in ore_ids:
            label = f"{ore_names.get(ore_id, ore_id.upper())} (%)"
            table[label] = [round(float(d.get(ore_id, 0.0)), 1) for d in share_dicts]
    if "flux_mt" in ranked.columns:
        flux_dicts = [d if isinstance(d, dict) else {} for d in ranked["flux_mt"]]
        flux_ids: list[str] = []
        for d in flux_dicts:
            for flux_id in d:
                if flux_id not in flux_ids:
                    flux_ids.append(flux_id)
        for flux_id in flux_ids:
            table[f"{flux_id.title()} (MT)"] = [
                round(float(d.get(flux_id, 0.0)), 1) for d in flux_dicts
            ]

    table["Feasible"] = feasible_vals
    st.markdown("###### Top 100 alternative solutions (cheapest first)")
    if "shares_pct" in ranked.columns:
        st.caption(
            "Each row is a full blend: ore shares (%) and optimizer-added flux "
            "(MT) alongside its cost, basicity, and slag outcome."
        )
    else:
        st.info(
            "Blend-combination columns are unavailable for this run — re-run "
            "the Total Cost optimizer to record ore shares and flux additions "
            "per candidate. (After a code update, restart the app so the "
            "optimizer module reloads.)"
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


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


def _build_coke_correction_reference(
    *,
    settings: CokeCorrectionSettings,
    observed_slag_rate_kg_per_thm: float | None,
    flux_inputs: list[FluxInput] | None,
    hot_metal_target_mt: float,
    process_context: Mapping[str, Any] | None = None,
    current_quantities_mt: Mapping[str, float] | None = None,
    ores: list[OreInput] | None = None,
    current_si_pct: float | None = None,
) -> CokeCorrectionReference:
    """
    Resolve the operating point the coke correction is anchored to.

    Built once per run and passed unchanged to the LP, to DE, and to every DE
    candidate, so the corrected objective is one fixed function rather than one
    that shifts as reference data appears and disappears.

    The two terms enabled by default need no solved blend: slag comes from the
    observed DPR rate and flux CO2 from the flux rows as the operator has them
    charged. That avoids a circular dependency where the reference would need
    the LP result the LP needs the reference to produce.

    Args:
         - settings: CokeCorrectionSettings - Parsed correction settings.
         - observed_slag_rate_kg_per_thm: float | None - Observed DPR slag rate.
         - flux_inputs: list[FluxInput] | None - Flux rows as currently charged.
         - hot_metal_target_mt: float - HM basis for the per-THM drivers.
         - process_context: Mapping[str, Any] | None - Recent process values.
         - current_quantities_mt: Mapping[str, float] | None - Current burden, when known.
         - ores: list[OreInput] | None - Ores backing ``current_quantities_mt``.
         - current_si_pct: float | None - Si model output for the current burden.

    Returns:
         - return CokeCorrectionReference - Reference operating point.
    """

    burden_oxygen = None
    if ores and current_quantities_mt:
        burden_oxygen = compute_burden_oxygen_kg_per_thm(
            ores=ores,
            quantities_mt=current_quantities_mt,
            hot_metal_mt=hot_metal_target_mt,
        )

    current_drivers = CokeCorrectionDrivers(
        slag_rate_kg_per_thm=None,
        flux_co2_kg_per_thm=compute_flux_co2_kg_per_thm(
            flux_inputs=flux_inputs, hot_metal_mt=hot_metal_target_mt
        ),
        burden_oxygen_kg_per_thm=burden_oxygen,
        hot_metal_si_pct=current_si_pct,
    )
    return build_reference(
        settings=settings,
        observed_slag_rate_kg_per_thm=observed_slag_rate_kg_per_thm,
        current_drivers=current_drivers,
        process_context=process_context,
    )


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
            mno_pct=oxide_pct_from_basis(
                float_from_row(row, "mno_pct", base.chemistry.mno_pct),
                str(row.get("mn_basis", "mno")),
                element="mn",
            ),
            tio2_pct=oxide_pct_from_basis(
                float_from_row(row, "tio2_pct", base.chemistry.tio2_pct),
                str(row.get("ti_basis", "tio2")),
                element="ti",
            ),
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
with st.spinner("Checking the hourly furnace dataset..."):
    static_refresh_result = _refresh_static_dataset_if_needed(bmo_cfg)
if static_refresh_result.get("error") and not static_refresh_result.get("usable"):
    st.error(
        "The hourly furnace CSV could not be fetched and no local fallback exists. "
        f"Error: {static_refresh_result['error']}"
    )
    st.stop()
_render_static_dataset_bar(bmo_cfg, static_refresh_result)
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
# Live 1-hour-average plant tags win over static-CSV values (which lag and
# derive nut coke from charged MT); static remains the offline fallback.
recent_fuel_rates = {
    **_recent_fuel_rates_from_static_csv(static_path, static_mtime_ns),
    **_recent_fuel_rates_live(),
}


def _optional_target(key: str, fallback: float | None) -> float | None:
    """Read a slag-window limit from config, treating an absent/null key as off."""

    if key not in target_cfg:
        return fallback
    raw = target_cfg.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


model_input_defaults = {
    "target_slag_basicity_min": float(target_cfg.get("target_slag_basicity_min", 0.0)),
    "target_slag_basicity_max": float(target_cfg.get("target_slag_basicity_max", 10.0)),
    "target_slag_t_basicity_min": float(
        _optional_target("target_slag_t_basicity_min", 0.0) or 0.0
    ),
    "target_slag_t_basicity_max": float(
        _optional_target("target_slag_t_basicity_max", 10.0) or 10.0
    ),
    "target_slag_rate_kg_per_thm": float(
        target_cfg.get("target_slag_rate_kg_per_thm", 290.0)
    ),
    "target_slag_al2o3_max_pct": float(
        _optional_target("target_slag_al2o3_max_pct", 20.0) or 0.0
    ),
    "target_slag_mgo_min_pct": float(
        _optional_target("target_slag_mgo_min_pct", 7.0) or 0.0
    ),
    "target_slag_mgo_al2o3_ratio_min": float(
        _optional_target("target_slag_mgo_al2o3_ratio_min", 0.36) or 0.0
    ),
}
# plant_slag = model_slag * factor, so a plant-basis target divides by it to get
# the model-basis cap the solvers actually enforce. See setting_bmo.yml.
model_to_plant_slag_factor = float(
    target_cfg.get("model_to_plant_slag_factor", 1.0) or 1.0
)
if model_to_plant_slag_factor <= 0.0:
    model_to_plant_slag_factor = 1.0
model_input_defaults.update(
    _model_input_defaults_from_static_csv(static_path, static_mtime_ns)
)
# Charging plant is operator-configurable too. Charges per hour and tonnes per
# charge are the only two numbers that set the throughput ceiling, and both move
# with skip-car condition and burden bulk density, so pinning them in yml made
# the ceiling stale. Charging hours are always 24 and nut coke follows from its
# rate, so neither is an input. Defaults still come from config.
burden_capacity_cfg = bmo_cfg.get("burden_capacity", {}) or {}
model_input_defaults.update(
    {
        "max_charges_per_hour": float(
            burden_capacity_cfg.get("max_charges_per_hour", 7.5) or 7.5
        ),
        "charge_mass_mt": float(
            burden_capacity_cfg.get("charge_mass_mt", 26.4) or 26.4
        ),
    }
)
model_input_defaults = apply_model_input_preferences(
    model_input_defaults, operator_preferences
)
burden_capacity_enabled_default = bool(burden_capacity_cfg.get("enabled", True))
# Nut coke is a held set point, not an optimizer variable. The live/default
# rate is a base quantity and the Fuel Ash TM is added to obtain wet tonnes.
# Take both from the same source that seeds the editor so the cap and result do
# not disagree.
nut_coke_cfg = next(
    (
        row
        for row in bmo_cfg.get("fuel_ash_inputs", [])
        if str(row.get("fuel_id", "")).strip() == "nut_coke"
    ),
    {},
)
recent_nut_coke_rate = recent_fuel_rates.get("nut_coke_rate_kg_thm")
nut_coke_rate_kg_per_thm = float(
    recent_nut_coke_rate
    if recent_nut_coke_rate is not None
    else nut_coke_cfg.get("rate_kg_per_thm", DEFAULT_NUT_COKE_RATE_KG_PER_THM)
)
nut_coke_add_moisture = recent_nut_coke_rate is not None or bool(
    nut_coke_cfg.get("add_moisture_to_rate", False)
)

with st.form("bmo_model_input_form", clear_on_submit=False):
    st.markdown("### Model Inputs")

    with st.expander("Production target and chemistry source", expanded=True):
        layout_col1, layout_col2, layout_col3, layout_col4 = st.columns(4)
        with layout_col1:
            chemistry_mode = st.selectbox(
                "Chemistry mode",
                options=["latest", "avg"],
                index=(
                    0 if str(bmo_cfg.get("chemistry_mode", "latest")) == "latest" else 1
                ),
                key="bmo_chemistry_mode",
            )
        chemistry_window_days = layout_col2.slider(
            "Chemistry window for avg (days)",
            min_value=1,
            max_value=180,
            value=int(bmo_cfg.get("chemistry_window_days", 30)),
            key="bmo_chemistry_window_days",
            help=(
                "Used only when Chemistry mode is avg. Latest mode uses the last "
                "charged instance."
            ),
        )
        target_production_mt = layout_col3.number_input(
            "Target HM / Pig Iron (MT)",
            min_value=0.0,
            value=float(target_cfg.get("target_production_mt", 2350.0)),
            step=5.0,
            key="bmo_target_production_mt",
        )
        target_slag_rate_kg_per_thm = layout_col4.number_input(
            "Max Slag Rate (kg/THM)",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_rate_kg_per_thm"]),
            step=5.0,
            key="bmo_target_slag_rate_kg_per_thm",
            help=(
                "Plant basis: the slag rate the plant would measure. The model's own "
                "calculated slag runs above that, so the cap is divided by "
                f"{model_to_plant_slag_factor:.3f} before the optimizer sees it."
            ),
        )

    with st.expander("Slag chemistry window", expanded=False):
        st.caption(
            "Set any limit to 0 to switch it off. Al2O3 and MgO are inert, so their "
            "masses are fixed by what is charged and their percentages move "
            "inversely with total slag: cutting the slag rate pushes Al2O3 up "
            "towards its cap. MgO/Al2O3 is a mass ratio, so it does not move with "
            "slag rate at all and constrains the burden alone."
        )
        basicity_col1, basicity_col2, basicity_col3, basicity_col4 = st.columns(4)
        target_slag_basicity_min = basicity_col1.number_input(
            "Min Basicity CaO/SiO2",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_basicity_min"]),
            step=0.01,
            format="%.3f",
            key="bmo_target_slag_basicity_min",
        )
        target_slag_basicity_max = basicity_col2.number_input(
            "Max Basicity CaO/SiO2",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_basicity_max"]),
            step=0.01,
            format="%.3f",
            key="bmo_target_slag_basicity_max",
        )
        target_slag_t_basicity_min = basicity_col3.number_input(
            "Min T Basicity (CaO+MgO)/SiO2",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_t_basicity_min"]),
            step=0.01,
            format="%.3f",
            key="bmo_target_slag_t_basicity_min",
        )
        target_slag_t_basicity_max = basicity_col4.number_input(
            "Max T Basicity (CaO+MgO)/SiO2",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_t_basicity_max"]),
            step=0.01,
            format="%.3f",
            key="bmo_target_slag_t_basicity_max",
        )
        # Four columns again (last one left empty) so this row lines up with the
        # basicity row above instead of rendering at a different width.
        quality_col1, quality_col2, quality_col3, _quality_spacer = st.columns(4)
        target_slag_al2o3_max_pct = quality_col1.number_input(
            "Max Al2O3 in slag (%)",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_al2o3_max_pct"]),
            step=0.5,
            format="%.2f",
            key="bmo_target_slag_al2o3_max_pct",
        )
        target_slag_mgo_min_pct = quality_col2.number_input(
            "Min MgO in slag (%)",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_mgo_min_pct"]),
            step=0.25,
            format="%.2f",
            key="bmo_target_slag_mgo_min_pct",
        )
        target_slag_mgo_al2o3_ratio_min = quality_col3.number_input(
            "Min MgO/Al2O3",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_mgo_al2o3_ratio_min"]),
            step=0.01,
            format="%.3f",
            key="bmo_target_slag_mgo_al2o3_ratio_min",
        )

    with st.expander("Charging capacity", expanded=False):
        st.caption(
            "Charges per hour and tonnes per charge are the only two numbers that "
            "set the throughput ceiling - charging runs 24 h, and nut coke is a "
            "held set point so its tonnage follows from the HM target and is "
            "deducted off the top. What is left is the daily room for IBRM plus "
            "flux. Without this cap the optimizer can answer a low-Fe burden by "
            "simply charging more of it, which the plant cannot do at capacity."
        )
        # Same 4-column grid as the slag window above so every input in the form
        # sits on one consistent line width.
        charge_col1, charge_col2, charge_col3, _charge_spacer = st.columns(4)
        max_charges_per_hour = charge_col1.number_input(
            "Max charges per hour",
            min_value=0.0,
            value=float(model_input_defaults["max_charges_per_hour"]),
            step=0.05,
            format="%.2f",
            key="bmo_max_charges_per_hour",
            help="Observed: median 6.38, p95 6.75, p99 7.02, max 7.25.",
        )
        charge_mass_mt = charge_col2.number_input(
            "Max qty per charge (MT)",
            min_value=0.0,
            value=float(model_input_defaults["charge_mass_mt"]),
            step=0.1,
            format="%.2f",
            key="bmo_charge_mass_mt",
            help=(
                "Total charge mass including nut coke. Plant charge reports show an "
                "actual mean around 30.1 MT (p95 31.2) against the 26.4 default; see "
                "docs/bmo_fuel_slag_si_findings.md section 6."
            ),
        )
        burden_capacity_enabled = charge_col3.checkbox(
            "Enforce charging capacity",
            value=burden_capacity_enabled_default,
            key="bmo_burden_capacity_enabled",
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
        "Save Settings for Next Time",
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
                    "target_slag_rate_kg_per_thm": target_slag_rate_kg_per_thm,
                    "target_slag_al2o3_max_pct": target_slag_al2o3_max_pct,
                    "target_slag_mgo_min_pct": target_slag_mgo_min_pct,
                    "target_slag_mgo_al2o3_ratio_min": target_slag_mgo_al2o3_ratio_min,
                    "max_charges_per_hour": max_charges_per_hour,
                    "charge_mass_mt": charge_mass_mt,
                },
            )
            st.success(f"Slag and charging settings saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save settings: {exc}")
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

# A limit of 0 means "not enforced"; the solvers take None for that.
target_slag_t_basicity_min = target_slag_t_basicity_min or None
target_slag_t_basicity_max = (
    target_slag_t_basicity_max if target_slag_t_basicity_max > 0.0 else None
)
target_slag_al2o3_max_pct = target_slag_al2o3_max_pct or None
target_slag_mgo_min_pct = target_slag_mgo_min_pct or None
target_slag_mgo_al2o3_ratio_min = target_slag_mgo_al2o3_ratio_min or None

# The operator's rate is on the PLANT's basis; the solvers work in the model's,
# which reads high. Convert once here so every downstream call sees one number.
target_slag_qty_mt = (
    float(target_slag_rate_kg_per_thm)
    * float(target_production_mt)
    / 1000.0
    / model_to_plant_slag_factor
)
if model_to_plant_slag_factor != 1.0:
    st.caption(
        f"Slag cap: {target_slag_rate_kg_per_thm:,.0f} kg/THM plant basis "
        f"= {target_slag_qty_mt:,.0f} MT model basis at "
        f"{target_production_mt:,.0f} MT HM "
        f"(model/plant factor {model_to_plant_slag_factor:.3f})."
    )
feo_in_slag_pct = float(bmo_cfg.get("chemistry", {}).get("feo_in_slag_pct", 0.4))
coke_correction_settings = load_coke_correction_settings(bmo_cfg)
fuel_rate_anchor_basis = str(bmo_cfg.get("fuel_rate_anchor_basis", "model_cost"))


ores, ore_diagnostics = _cached_build_ore_inputs(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"] = list(ore_diagnostics.get("warnings", []))

hm_snapshot = _cached_hm_slag_snapshot(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"].extend(hm_snapshot.get("warnings", []))
fuel_analysis, fuel_analysis_warnings = _cached_fuel_analysis_snapshot(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"].extend(fuel_analysis_warnings)
nut_coke_moisture_raw = (fuel_analysis.get("nut_coke") or {}).get("moisture_pct")
if nut_coke_moisture_raw is None:
    nut_coke_moisture_raw = nut_coke_cfg.get("moisture_pct", 0.0)
try:
    nut_coke_moisture_pct = min(
        100.0, max(0.0, float(nut_coke_moisture_raw or 0.0))
    )
except (TypeError, ValueError):
    nut_coke_moisture_pct = 0.0
nut_coke_charge_moisture_pct = (
    nut_coke_moisture_pct if nut_coke_add_moisture else 0.0
)
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

# Charging ceiling, rebuilt from the operator's own charge rate / charge mass
# rather than the yml snapshot. This is an absolute daily tonnage - the skips run
# 24 h whatever HM is targeted - so only the nut-coke deduction tracks the target.
max_burden_qty_mt: float | None = None
if burden_capacity_enabled:
    max_burden_qty_mt = max_ibrm_flux_capacity_mt(
        {
            "max_charges_per_hour": max_charges_per_hour,
            "charge_mass_mt": charge_mass_mt,
        },
        target_hot_metal_mt=target_production_mt,
        nut_coke_rate_kg_per_thm=nut_coke_rate_kg_per_thm,
        nut_coke_moisture_pct=nut_coke_charge_moisture_pct,
    )
    daily_charge_capacity_mt = (
        charge_mass_mt * max_charges_per_hour * CHARGING_HOURS_PER_DAY
    )
    nut_coke_mt = calculate_wet_nut_coke_mt(
        nut_coke_rate_kg_per_thm,
        target_production_mt,
        nut_coke_charge_moisture_pct,
    )
    nut_coke_wet_rate_kg_per_thm = nut_coke_rate_kg_per_thm * (
        1.0 + nut_coke_charge_moisture_pct / 100.0
    )
    if max_burden_qty_mt > 0.0:
        st.caption(
            f"Charging capacity: {max_charges_per_hour:,.2f} charges/hr x "
            f"{charge_mass_mt:,.2f} MT x 24 h = {daily_charge_capacity_mt:,.0f} MT/day, "
            f"less {nut_coke_mt:,.1f} MT wet nut coke "
            f"({nut_coke_rate_kg_per_thm:,.1f} kg/THM base + "
            f"{nut_coke_charge_moisture_pct:,.2f}% moisture = "
            f"{nut_coke_wet_rate_kg_per_thm:,.2f} kg/THM wet) "
            f"= IBRM + flux limited to {max_burden_qty_mt:,.0f} MT. "
            "Blends needing more tonnes than this are rejected."
        )
    else:
        max_burden_qty_mt = None
        st.warning(
            "Charging capacity works out to zero or less - nut coke alone fills "
            "every charge. Check charges per hour and tonnes per charge."
        )
else:
    st.caption(
        "Charging capacity is not being enforced: the optimizer may plan more "
        "burden tonnes than the charging system can deliver."
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

# Flux editor lives inside the ore-input form so its price/stock are applied and
# saved together with the ore inputs. Seed from config + saved preferences, or
# from the session-applied edits.
flux_base_df = apply_flux_preferences(
    build_flux_editor_df(bmo_cfg.get("flux_inputs", [])), operator_preferences
)
stored_flux_df = st.session_state.get("bmo_applied_flux_editor_df")
if (
    isinstance(stored_flux_df, pd.DataFrame)
    and not flux_base_df.empty
    and "flux_id" in stored_flux_df.columns
    and set(stored_flux_df["flux_id"].astype(str))
    == set(flux_base_df["flux_id"].astype(str))
):
    flux_editor_source_df = stored_flux_df
else:
    flux_editor_source_df = flux_base_df

with st.form("bmo_ore_input_form", clear_on_submit=False):
    edited_ore_candidate_df = render_ore_editor(ore_editor_source_df)
    st.caption(
        "Save keeps ore selection, prices, share bounds, and flux price/stock for "
        "the next session. Stock and chemistry continue to come from the latest "
        "source data."
    )
    if not flux_editor_source_df.empty:
        st.markdown("#### Flux Inputs")
        st.caption(
            "Optimisable fluxes (dolomite/quartz) are added by the optimizer to hold "
            "slag basicity within bounds; set their price and available stock here."
        )
        edited_flux_candidate_df = render_flux_editor(flux_editor_source_df)
    else:
        edited_flux_candidate_df = flux_editor_source_df
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
    edited_flux_df = (
        edited_flux_candidate_df.copy()
        if isinstance(edited_flux_candidate_df, pd.DataFrame)
        else flux_editor_source_df
    )
    st.session_state["bmo_applied_flux_editor_df"] = edited_flux_df
    _clear_bmo_results()
    if ore_inputs_saved:
        try:
            saved_path = save_ore_editor_preferences(
                operator_preferences_path, edited_df
            )
            save_flux_preferences(operator_preferences_path, edited_flux_df)
            st.success(f"Ore and flux inputs saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save ore inputs: {exc}")
    else:
        st.success("Ore inputs applied.")
else:
    edited_df = ore_editor_source_df
    edited_flux_df = flux_editor_source_df
flux_inputs = flux_inputs_from_editor(edited_flux_df)

with st.expander("Hot Metal Chemistry Assumptions", expanded=False):
    with st.form("bmo_assumption_input_form", clear_on_submit=False):
        hm_chem_values = render_hot_metal_chemistry(
            hm_snapshot,
            apply_hm_chemistry_preferences(
                bmo_cfg.get("slag_balance", {}) or {}, operator_preferences
            ),
        )
        hm_apply_col, hm_save_col = st.columns(2)
        assumptions_applied = _form_submit_button(
            hm_apply_col,
            "Apply Assumptions",
            type="primary",
            width="stretch",
        )
        assumptions_saved = _form_submit_button(
            hm_save_col,
            "Save for Later",
            type="secondary",
            width="stretch",
        )
if assumptions_applied or assumptions_saved:
    _clear_bmo_results()
    if assumptions_saved:
        try:
            saved_path = save_hm_chemistry_preferences(
                operator_preferences_path, hm_chem_values
            )
            st.success(f"Hot metal chemistry saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save hot metal chemistry: {exc}")
    else:
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

fuel_ash_base_df = apply_fuel_ash_preferences(
    build_fuel_ash_editor_df(
        _fuel_ash_cfg_with_recent_rates(
            bmo_cfg.get("fuel_ash_inputs", []),
            recent_fuel_rates,
            fuel_analysis,
        )
    ),
    operator_preferences,
)
stored_fuel_ash_df = st.session_state.get("bmo_applied_fuel_ash_editor_df")
if (
    isinstance(stored_fuel_ash_df, pd.DataFrame)
    and not fuel_ash_base_df.empty
    and "fuel_id" in stored_fuel_ash_df.columns
    and "vm_pct" in stored_fuel_ash_df.columns
    and set(stored_fuel_ash_df["fuel_id"].astype(str))
    == set(fuel_ash_base_df["fuel_id"].astype(str))
):
    fuel_ash_editor_source_df = stored_fuel_ash_df
else:
    fuel_ash_editor_source_df = fuel_ash_base_df

with st.form("bmo_fuel_ash_input_form", clear_on_submit=False):
    st.markdown("### Fuel Ash Inputs")
    st.caption(
        "**These rows exist to put fuel ash into the slag balance.** The rate and "
        "ash chemistry of each fuel decide how much ash it charges, and that ash "
        "is part of the slag the LP constrains. Set a fuel's rate to 0 to drop its "
        "ash from the slag entirely - nothing else changes."
    )
    st.caption(
        "Two columns are also read by the separate fuel-cost step, which runs "
        "AFTER the LP and never feeds back into slag: the **prices**, and the "
        "**nut coke and PCI rates**. Coke rate is not read there at all - it is "
        "back-solved from the model's predicted cost."
    )
    if not fuel_ash_editor_source_df.empty:
        edited_fuel_ash_candidate_df = render_fuel_ash_editor(fuel_ash_editor_source_df)
    else:
        edited_fuel_ash_candidate_df = fuel_ash_editor_source_df
    st.caption(
        "Fuel analysis uses Moisture from fuel_chemistry (TM for coke/nut coke; "
        "IM for PCI). Ash analysis uses VM from fuel_chemistry. Moisture is "
        "removed once from the wet fuel; VM is not deducted as moisture."
    )
    fuel_apply_col, fuel_save_col = st.columns(2)
    fuel_ash_inputs_applied = _form_submit_button(
        fuel_apply_col,
        "Apply Fuel Ash Inputs",
        type="primary",
        width="stretch",
    )
    fuel_ash_inputs_saved = _form_submit_button(
        fuel_save_col,
        "Save Fuel Ash Inputs for Next Time",
        type="secondary",
        width="stretch",
    )
if fuel_ash_inputs_applied or fuel_ash_inputs_saved:
    edited_fuel_ash_df = edited_fuel_ash_candidate_df.copy()
    st.session_state["bmo_applied_fuel_ash_editor_df"] = edited_fuel_ash_df
    _clear_bmo_results()
    if fuel_ash_inputs_saved:
        try:
            saved_path = save_fuel_ash_preferences(
                operator_preferences_path, edited_fuel_ash_df
            )
            st.success(f"Fuel Ash inputs saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save Fuel Ash inputs: {exc}")
    else:
        st.success("Fuel Ash inputs applied.")
else:
    edited_fuel_ash_df = fuel_ash_editor_source_df
fuel_ash_inputs = fuel_ash_inputs_from_editor(edited_fuel_ash_df)

dust_base_df = apply_dust_preferences(
    build_dust_editor_df(bmo_cfg.get("dust_inputs", [])), operator_preferences
)
stored_dust_df = st.session_state.get("bmo_applied_dust_editor_df")
if (
    isinstance(stored_dust_df, pd.DataFrame)
    and not dust_base_df.empty
    and "dust_id" in stored_dust_df.columns
    and set(stored_dust_df["dust_id"].astype(str))
    == set(dust_base_df["dust_id"].astype(str))
):
    dust_editor_source_df = stored_dust_df
else:
    dust_editor_source_df = dust_base_df

dust_inputs_applied = False
dust_inputs_saved = False
with st.expander("Advanced Slag Balance Inputs", expanded=False):
    slag_settings_values = render_slag_balance_settings(bmo_cfg.get("slag_balance", {}))
    if not dust_editor_source_df.empty:
        st.markdown("##### BF Gas Dust")
        with st.form("bmo_dust_input_form", clear_on_submit=False):
            edited_dust_candidate_df = render_dust_editor(dust_editor_source_df)
            st.caption(
                "Save keeps the BF Gas Dust quantity, moisture, enabled state, "
                "and chemistry values for the next session."
            )
            dust_apply_col, dust_save_col = st.columns(2)
            dust_inputs_applied = _form_submit_button(
                dust_apply_col,
                "Apply BF Gas Dust Inputs",
                type="primary",
                width="stretch",
            )
            dust_inputs_saved = _form_submit_button(
                dust_save_col,
                "Save BF Gas Dust Inputs for Next Time",
                type="secondary",
                width="stretch",
            )
    else:
        edited_dust_candidate_df = dust_editor_source_df
if dust_inputs_applied or dust_inputs_saved:
    edited_dust_df = edited_dust_candidate_df.copy()
    st.session_state["bmo_applied_dust_editor_df"] = edited_dust_df
    _clear_bmo_results()
    if dust_inputs_saved:
        try:
            saved_path = save_dust_preferences(
                operator_preferences_path, edited_dust_df
            )
            st.success(f"BF Gas Dust inputs saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save BF Gas Dust inputs: {exc}")
    else:
        st.success("BF Gas Dust inputs applied.")
else:
    edited_dust_df = dust_editor_source_df
dust_inputs = dust_inputs_from_editor(edited_dust_df)
slag_balance_settings = slag_balance_settings_from_editor(
    slag_settings_values, hm_chem_values, hm_snapshot
)

# Operator-visible warning: if dust is entered but the full slag balance
# is disabled, the dust rows are silently ignored downstream. Surface
# this so the operator knows their dust entry isn't being applied.
_dust_entered = any(
    d.enabled
    and (
        float(d.wet_qty_mt or 0.0) > 0.0
        or float(getattr(d, "quantity_kg_per_charge", 0.0) or 0.0) > 0.0
    )
    for d in dust_inputs
)
if _dust_entered and not slag_balance_settings.enabled:
    st.warning(
        "BF gas dust is entered but "
        "'Use full slag balance' is unchecked - dust will NOT be deducted "
        "from the slag balance."
    )

_DE_SEED_LABELS = {
    "lp_else_random": "LP seed, random fallback (recommended)",
    "lp": "LP seed only (skip DE if LP is infeasible)",
    "random": "Random start (ignore the LP)",
}

run_lp_clicked = False
run_total_clicked = False
with st.form("bmo_run_form", clear_on_submit=False):
    seed_options = list(_DE_SEED_LABELS)
    configured_seed = str(opt_cfg.get("initial_solution", "lp_else_random")).lower()
    de_seed_choice = st.selectbox(
        "Total-cost optimizer start point",
        options=seed_options,
        index=(
            seed_options.index(configured_seed)
            if configured_seed in seed_options
            else 0
        ),
        format_func=lambda key: _DE_SEED_LABELS[key],
        help=(
            "The optimizer normally starts from the LP baseline. When the LP is "
            "infeasible that used to stop it running at all, even though an "
            "infeasible LP only means the *linearised* slag and basicity model "
            "found no solution. A random start searches the whole share range "
            "instead and returns the best blend it can reach, with any "
            "constraint violations listed on the result."
        ),
    )
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

requested_lp = bool(run_lp_clicked)
requested_total = bool(run_total_clicked)

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

        # Resolved once for the whole run and reused by the LP, DE, and every DE
        # candidate, so both solvers optimise one identical objective.
        coke_correction_reference = _build_coke_correction_reference(
            settings=coke_correction_settings,
            observed_slag_rate_kg_per_thm=observed_slag_rate,
            flux_inputs=flux_inputs,
            hot_metal_target_mt=target_production_mt,
            current_quantities_mt=st.session_state.get("bmo_manual_quantities_mt"),
            ores=selected_ores,
            current_si_pct=st.session_state.get("bmo_manual_si"),
        )

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
                target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
                target_slag_mgo_min_pct=target_slag_mgo_min_pct,
                target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
                max_burden_qty_mt=max_burden_qty_mt,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=target_production_mt,
                coke_correction_settings=coke_correction_settings,
                coke_correction_reference=coke_correction_reference,
                charge_mass_mt=charge_mass_mt,
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
                # The LP may have added optimizable flux (e.g. dolomite) to hold
                # slag basicity within bounds. Carry those solved quantities into
                # this display re-evaluation -- otherwise it silently falls back
                # to the raw editor flux rows (wet_qty_mt=0 for LP-auto fluxes),
                # dropping the flux the LP actually added and understating the
                # displayed basicity/slag versus what the LP solved for.
                lp_solved_flux_mt = {
                    str(flux_id): float(qty)
                    for flux_id, qty in (
                        lp_physical_result.diagnostics.get("lp_flux_quantities_mt", {})
                        or {}
                    ).items()
                }
                flux_inputs_for_display = [
                    (
                        replace(flux, wet_qty_mt=lp_solved_flux_mt[flux.flux_id])
                        if flux.flux_id in lp_solved_flux_mt
                        else flux
                    )
                    for flux in flux_inputs
                ]
                # Si is predicted before the fuel re-evaluation, not after, so the
                # correction's Si term sees this blend's own Si instead of nothing.
                lp_si = _predict_blend_si(
                    ores=selected_ores,
                    quantities_mt=lp_physical_result.quantities_mt,
                    process_context=process_context,
                    history_df=history_df,
                    hot_metal_target_mt=target_production_mt,
                )
                lp_result = evaluate_blend_with_fuel_prediction(
                    ores=selected_ores,
                    quantities_mt=lp_physical_result.quantities_mt,
                    feo_in_slag_pct=feo_in_slag_pct,
                    model_service=model_service,
                    process_context=process_context,
                    history_df=history_df,
                    fuel_ash_inputs=fuel_ash_inputs,
                    flux_inputs=flux_inputs_for_display,
                    dust_inputs=dust_inputs,
                    slag_balance_settings=slag_balance_settings,
                    hot_metal_target_mt=target_production_mt,
                    coke_correction_settings=coke_correction_settings,
                    coke_correction_reference=coke_correction_reference,
                    hot_metal_si_pct=lp_si,
                    fuel_rate_anchor_basis=fuel_rate_anchor_basis,
                    charge_mass_mt=charge_mass_mt,
                )
                lp_result.diagnostics["lp_flux_quantities_mt"] = lp_solved_flux_mt
                lp_result.diagnostics["flux_cost_per_thm_rs"] = float(
                    lp_physical_result.diagnostics.get("flux_cost_per_thm_rs", 0.0)
                    or 0.0
                )
                # Re-check feasibility against the blend actually being displayed,
                # rather than copying the pre-recompute blend's flags -- so a
                # re-pricing/re-evaluation drift can never silently show an
                # out-of-bounds result as feasible with no warning.
                lp_result.violations = check_blend_constraints(
                    lp_result,
                    selected_ores,
                    target_production_mt=target_fe_mt,
                    target_slag_qty_mt=target_slag_qty_mt,
                    target_slag_basicity_min=target_slag_basicity_min,
                    target_slag_basicity_max=target_slag_basicity_max,
                    target_slag_t_basicity_min=target_slag_t_basicity_min,
                    target_slag_t_basicity_max=target_slag_t_basicity_max,
                    target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
                    target_slag_mgo_min_pct=target_slag_mgo_min_pct,
                    target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
                    max_burden_qty_mt=max_burden_qty_mt,
                )
                lp_result.feasible = len(lp_result.violations) == 0
                st.session_state["bmo_lp_si"] = lp_si
        # LP always runs in this rerun and shares DE's fuel-prediction context, so
        # always persist it. This keeps the LP tab and the LP-vs-DE comparison on
        # the SAME live snapshot DE used, avoiding a stale LP (from an earlier,
        # separate click) being shown next to a fresh DE.
        st.session_state["bmo_lp_result"] = lp_result
        st.session_state["bmo_lp_errors"] = lp_errors

        if requested_total:
            de_status = st.status("Total Cost Optimizer (DE) running…", expanded=True)
            # Live "thinking" line, refreshed every generation so the operator
            # watches the solver churn through thousands of candidate blends.
            de_thinking_ph = de_status.empty()
            # Milestone snapshot (every 5 generations + final) — a single
            # placeholder that is overwritten in place, not an appended log.
            de_milestone_ph = de_status.empty()
            iteration_lines: list[str] = []
            de_progress_state: dict[str, float] = {
                "iteration": 0,
                "nfev": 0,
                "best_obj": float("inf"),
                "elapsed_s": 0.0,
            }

            def _de_progress(
                iteration: int,
                best_obj: float,
                best_feas: float | None,
                nfev: int,
                elapsed_s: float,
            ) -> bool:
                """
                Stream DE progress to the Streamlit status panel + log.

                A live line updates every generation (emphasising the running
                function-evaluation count to convey how many blends are tried),
                while a fuller snapshot is written every 5 generations. The final
                generation is written after the run so the last one always shows.

                Args:
                     - iteration: int - 1-based DE generation index.
                     - best_obj: float - Best (penalized) objective value seen so far in Rs/THM.
                     - best_feas: float | None - Best feasible objective seen so far, if any.
                     - nfev: int - Cumulative function-evaluation count.
                     - elapsed_s: float - Seconds elapsed since DE started.

                Returns:
                     - return bool - False to keep running (no user-cancel wired yet).
                """

                de_progress_state.update(
                    iteration=iteration,
                    nfev=nfev,
                    best_obj=best_obj,
                    elapsed_s=elapsed_s,
                )
                de_thinking_ph.markdown(
                    f"🧠 **Exploring blends…** &nbsp; `{nfev:,}` candidate blends "
                    f"evaluated &nbsp;·&nbsp; best **{best_obj:,.0f}** Rs/THM "
                    f"&nbsp;·&nbsp; {elapsed_s:.1f}s"
                )
                if iteration % 5 == 0:
                    feas_txt = (
                        f" · feasible {best_feas:,.0f}" if best_feas is not None else ""
                    )
                    line = (
                        f"Gen {iteration:>3} · {nfev:,} evals · "
                        f"best {best_obj:,.0f} Rs/THM{feas_txt} · {elapsed_s:.1f}s"
                    )
                    iteration_lines.append(line)
                    de_milestone_ph.write(line)
                log.info("BMO DE gen=%s nfev=%s best=%.1f", iteration, nfev, best_obj)
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
                target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
                target_slag_mgo_min_pct=target_slag_mgo_min_pct,
                target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
                max_burden_qty_mt=max_burden_qty_mt,
                model_service=model_service,
                process_context=process_context,
                history_df=history_df,
                # The selectbox overrides the configured default for this run.
                de_cfg={**opt_cfg, "initial_solution": de_seed_choice},
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hot_metal_target_mt=target_production_mt,
                coke_correction_settings=coke_correction_settings,
                coke_correction_reference=coke_correction_reference,
                # Constant across candidates on purpose: the Si model is
                # blend-flat, so calling it per candidate would buy thousands of
                # inferences for a fraction of a kg/THM. A fixed offset applies
                # equally to every candidate and cannot distort the search.
                hot_metal_si_pct=st.session_state.get("bmo_lp_si"),
                fuel_rate_anchor_basis=fuel_rate_anchor_basis,
                progress_callback=_de_progress,
                charge_mass_mt=charge_mass_mt,
            )
            # Persist the full candidate cloud now, before the guardrail below may
            # replace de_result with an LP deepcopy (which has no de_candidates).
            if de_result is not None:
                st.session_state["bmo_de_candidates"] = list(
                    de_result.diagnostics.get("de_candidates", []) or []
                )
            # Always show the final generation (the "last one"), overwriting the
            # milestone placeholder so the panel keeps a single summary line.
            final_iter = int(de_progress_state["iteration"])
            final_nfev = int(de_progress_state["nfev"])
            if final_iter:
                de_thinking_ph.markdown(
                    f"✅ **Search complete** &nbsp; `{final_nfev:,}` candidate "
                    f"blends evaluated across {final_iter} generations"
                )
                de_milestone_ph.write(
                    f"Gen {final_iter:>3} · {final_nfev:,} evals · "
                    f"best {de_progress_state['best_obj']:,.0f} Rs/THM · "
                    f"{de_progress_state['elapsed_s']:.1f}s (final)"
                )
            de_status.update(
                label=(
                    f"DE finished — {final_iter} generations · "
                    f"{final_nfev:,} blend evaluations"
                    if final_iter
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
            # Compare ore + fuel + flux at baseline prices (the basis DE optimises
            # on) so a DE blend that saved ore cost by over-dosing expensive flux
            # cannot look cheaper than LP.
            def _baseline_total_with_flux(blend: Any) -> float:
                return float(blend.objective_rs_per_thm) + float(
                    blend.diagnostics.get("flux_cost_per_thm_rs", 0.0) or 0.0
                )

            if (
                lp_result is not None
                and lp_result.feasible
                and (
                    de_result is None
                    or not de_result.feasible
                    or _baseline_total_with_flux(de_result)
                    > _baseline_total_with_flux(lp_result) + 1e-6
                )
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
                charge_mass_mt=charge_mass_mt,
            )
            _render_si_metric(st.session_state.get("bmo_lp_si"))
            render_coke_correction_breakdown(lp_result)
            _render_lp_flux_additions(lp_result)
            render_blend_table(lp_result, selected_ores, charge_mass_mt=charge_mass_mt)
            _render_share_pie(lp_result, selected_ores, "LP Share (%)")
            render_slag_balance_details(
                lp_result, selected_ores, fuel_ash_inputs, flux_inputs
            )
            _render_energy_assumptions()
            _render_process_recommendation(
                lp_result,
                label="lp",
                hot_metal_mt=target_production_mt,
                ores=selected_ores,
                hm_chem_values=hm_chem_values,
                hm_snapshot=hm_snapshot,
                flux_inputs=flux_inputs,
                fuel_ash_inputs=fuel_ash_inputs,
            )
            _render_transition_ladder(
                provider=provider,
                ores=selected_ores,
                lp_kwargs=dict(
                    target_production_mt=target_fe_mt,
                    target_slag_qty_mt=target_slag_qty_mt,
                    feo_in_slag_pct=feo_in_slag_pct,
                    target_slag_basicity_min=target_slag_basicity_min,
                    target_slag_basicity_max=target_slag_basicity_max,
                    target_slag_t_basicity_min=target_slag_t_basicity_min,
                    target_slag_t_basicity_max=target_slag_t_basicity_max,
                    target_slag_al2o3_max_pct=target_slag_al2o3_max_pct,
                    target_slag_mgo_min_pct=target_slag_mgo_min_pct,
                    target_slag_mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
                    max_burden_qty_mt=max_burden_qty_mt,
                    fuel_ash_inputs=fuel_ash_inputs,
                    flux_inputs=flux_inputs,
                    dust_inputs=dust_inputs,
                    slag_balance_settings=slag_balance_settings,
                    hot_metal_target_mt=target_production_mt,
                    charge_mass_mt=charge_mass_mt,
                ),
                slag_rate_cap_kg_per_thm=(
                    target_slag_qty_mt / target_production_mt * 1000.0
                    if target_production_mt
                    else None
                ),
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
            _de_seed = de_result.diagnostics.get("de_seed") or {}
            if _de_seed.get("strategy_used") == "random":
                # Without this the operator cannot tell an unguided search from an
                # LP-seeded one, and the LP's reasons for failing would be lost.
                _lp_reasons = _de_seed.get("lp_seed_errors") or []
                st.warning(
                    "This result came from a **random start**, not the LP "
                    "baseline"
                    + (
                        " — the LP was infeasible."
                        if not _de_seed.get("lp_seed_available")
                        else "."
                    )
                    + " It is the best blend the search reached; check the "
                    "constraint violations below before acting on it."
                )
                if _lp_reasons:
                    with st.expander("Why the LP could not solve", expanded=False):
                        for _reason in _lp_reasons:
                            st.markdown(f"- {_reason}")
            render_blend_metrics(
                "DE Total-Cost Result",
                de_result,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
                charge_mass_mt=charge_mass_mt,
            )
            _render_si_metric(st.session_state.get("bmo_de_si"))
            render_coke_correction_breakdown(de_result)
            _render_lp_flux_additions(de_result)
            render_blend_table(de_result, selected_ores, charge_mass_mt=charge_mass_mt)
            _render_share_pie(de_result, selected_ores, "DE Share (%)")
            _render_de_exploration(
                st.session_state.get("bmo_de_candidates"), selected_ores
            )
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
                charge_mass_mt=charge_mass_mt,
                manual_ores=ores,
            )
        else:
            st.info(
                "Run LP or DE to compare the suggested blend with the last manual shift."
            )

render_diagnostics(de_result or lp_result, ore_diagnostics)
