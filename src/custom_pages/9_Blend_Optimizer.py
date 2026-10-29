"""Streamlit page flow for the Blend Mix Optimizer.

This page wires BMO configuration, data context, fuel-cost model inference,
LP baseline optimization, nonlinear DE optimization, and result rendering into
one Streamlit workflow for ore blend planning.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timedelta, timezone
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
from data.bmo.si_forecast_context import LiveSiliconForecastSource
from services import bmo as bmo_service
from services import bmo_case as case_service
from services import forecasts as forecast_service
from data.bmo.dust_inputs import dust_caption, dust_cfg_with_live
from data.bmo.ore_editor_preferences import saved_dust_ids

# Operator inputs: versioned in the twin store (Neon), the YAML file as seed and fallback (SV-3.1). The savers keep
# the YAML savers' names, which the Sandbox refuses.
from ui.bmo.operator_inputs import (
    clear_dust_preferences,
    inputs_token as _operator_inputs_token,
    page_inputs as _operator_inputs_for_page,
    save_dust_preferences,
    save_fuel_ash_preferences,
    save_fuel_price_preferences,
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
    render_blend_metrics,
    render_blend_table,
    render_coke_correction_breakdown,
    render_diagnostics,
    render_dust_editor,
    render_flux_editor,
    render_fuel_ash_editor,
    render_fuel_price_inputs,
    render_furnace_commentary,
    render_header,
    render_hot_metal_chemistry,
    render_model_accuracy_tab,
    render_ore_editor,
    render_slag_balance_details,
    render_slag_balance_settings,
)
from utils.bmo import (
    FuelUnitCostModelService,
    OreInput,
    validate_selected_pellet_inputs,
)
from ui.bmo.editor_inputs import (
    dust_inputs_from_editor,
    flux_inputs_from_editor,
    fuel_ash_inputs_from_editor,
    slag_balance_settings_from_editor,
)
from utils.bmo.constraints import (
    DEFAULT_NUT_COKE_RATE_KG_PER_THM,
)
from utils.bmo.charged_coke.bmo import (
    ChargedCokePrediction,
)
from utils.bmo.charged_coke.deployments import active_bundle as active_charged_bundle
from utils.bmo.charged_coke.service import ChargedCokeEngine
from ui.bmo.charged_coke import render_panel as render_charged_panel
from ui.bmo.si_forecast import render_forecast as render_si_furnace_forecast
from utils.bmo.si_forecast.deployments import active_bundle as active_si_bundle
from utils.bmo.si_forecast.service import (
    SiliconForecastService,
)
from utils.bmo.coke_calibration import load_calibration as load_coke_calibration
from utils.bmo.pci_anchoring import (
    ANCHOR_HIGH_PCI,
    MODEL_PCI_MAX,
    MODEL_PCI_MIN,
)
from utils.session import is_admin, is_logged_in, is_supervisor

if not is_logged_in():
    st.warning("Please log in to access this page.")
    st.stop()

# Live / Sandbox switch. In Sandbox mode this replays a saved snapshot, with the
# plant data its run used, in a separate copy of this page and stops here.
from ui.bmo.snapshot_panel import DE_SEARCH_ANCHOR, render_mode_gate  # noqa: E402
from utils.bmo.replay import record_provider  # noqa: E402

render_mode_gate(prefix="bmo_")


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
_STATIC_DATASET_FORCE_REFRESH_KEY = "_bmo_static_dataset_force_refresh"
_COKE_ANCHOR_LABELS = {
    "Physics-Driven": "energy_balance",
    "Data-Driven": "data_driven",
}


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
def _dust_analysis_from_static_csv(
    static_path: str, mtime_ns: int, window_days: int
) -> dict[str, dict[str, Any]]:
    """Dust-catcher and GCP dust analyses (DUST_1 / DUST_2) from the hourly dataset
    (``services.bmo_case.read_dust_analysis``)."""
    if mtime_ns <= 0:
        return {}
    return case_service.read_dust_analysis(static_path, window_days)


@_data_cache(show_spinner=False, ttl=600)
def _recent_fuel_rates_from_static_csv(
    static_path: str, mtime_ns: int
) -> dict[str, float | str]:
    """Latest coke / nut coke / PCI rates of the hourly dataset
    (``services.bmo_case.read_recent_fuel_rates_static``)."""
    if mtime_ns <= 0:
        return {}
    return case_service.read_recent_fuel_rates_static(static_path)


@_data_cache(show_spinner=False, ttl=600)
def _live_process_snapshot() -> dict[str, float]:
    """Blast, top-gas and shell-cooling tags for the Layer 2 energy balance, 1-h
    averages (``services.bmo_case.read_live_process_snapshot``); {} when InfluxDB
    is unavailable, and the caller then says the layer is unavailable."""

    return case_service.read_live_process_snapshot()


@_data_cache(show_spinner=False, ttl=600)
def _recent_fuel_rates_live() -> dict[str, float | str]:
    """Live coke / nut coke / PCI rates from InfluxDB, 1-h averages
    (``services.bmo_case.read_recent_fuel_rates_live``); {} on a fetch failure, so
    the static-CSV rates are used."""

    return case_service.read_recent_fuel_rates_live()


@_data_cache(show_spinner=False, ttl=600)
def _last_shift_production(static_path: str, static_mtime_ns: int) -> dict[str, Any]:
    """Daily production of the last complete plant shift, as a plain dict.

    Live ``process_params`` tags first (15-minute averages, so the window lines
    up with the shift boundaries); the furnace dataset is the fallback when
    InfluxDB is unreachable. The baseline every "change vs last shift" figure on
    this page is measured against. Empty dict when neither source has a usable
    shift.
    """

    from utils.bmo.shift_production import shift_production

    now = datetime.now(timezone.utc)
    try:
        from furnace_data.influx.online import fetch_online_df

        live = fetch_online_df(
            selected_measurements=["process_params"],
            time_range="last 1 week",
            request_type="windowed-average",
            window_by="15 minutes",
            start_time_override=now - timedelta(hours=34),
            end_time_override=now,
            column_naming="field",
        )
        result = shift_production(
            live, now=now, timestamps="stop", source="live plant tags"
        )
        if result.usable:
            return result.to_dict()
    except Exception as exc:  # noqa: BLE001 - fall back to the furnace dataset
        log.warning("Live shift production unavailable; trying the dataset: %s", exc)

    path = Path(static_path)
    if static_mtime_ns <= 0 or not path.exists():
        return {}
    try:
        frame = pd.read_csv(
            path, usecols=["time", "PRODUCTIONTONNESPERHR", "CHARGES/HRS."]
        )
    except (ValueError, OSError) as exc:
        log.warning("Furnace dataset has no charging columns: %s", exc)
        return {}
    stamps = pd.to_datetime(frame["time"], errors="coerce", format="mixed", dayfirst=True)
    frame = frame.assign(time=stamps).dropna(subset=["time"]).set_index("time")
    # Dataset rows are IST hour starts; the dataset lags, so its own newest hour
    # decides which shift is the last complete one.
    frame.index = frame.index.tz_localize("Asia/Kolkata")
    production = pd.to_numeric(frame["PRODUCTIONTONNESPERHR"], errors="coerce")
    charges = pd.to_numeric(frame["CHARGES/HRS."], errors="coerce")
    frame["hm_per_charge"] = production / charges.where(charges > 0)
    result = shift_production(
        frame,
        now=frame.index.max() + pd.Timedelta(hours=1),
        charges_col="CHARGES/HRS.",
        hm_col="hm_per_charge",
        timestamps="start",
        source="furnace dataset",
    )
    return result.to_dict() if result.usable else {}


@_data_cache(show_spinner=False, ttl=600)
def _model_input_defaults_from_static_csv(
    static_path: str, mtime_ns: int
) -> dict[str, float]:
    if mtime_ns <= 0:
        return {}
    return case_service.read_model_input_defaults(static_path)


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


def _cached_dpr_dust_rates(
    provider: EvonithBmoContextProvider,
    window_days: int,
    min_charges_per_day: int,
    cache_version: int,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    return _session_cached_source(
        cache_version,
        ("dpr_dust", window_days, min_charges_per_day),
        lambda: provider.get_dpr_dust_rates(
            window_days=window_days, min_charges_per_day=min_charges_per_day
        ),
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
        lambda: provider.get_fuel_analysis_snapshot(mode=mode, window_days=window_days),
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


def _cached_operator_preferences(path: str, token: str) -> dict[str, Any]:
    # Cached per version of the operator inputs (ui.bmo.operator_inputs), with a warning while the YAML file stands
    # in for a twin store that failed; a frozen Sandbox replay substitutes it.
    return _operator_inputs_for_page(path, token)


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


def _render_static_dataset_bar(
    bmo_cfg: dict[str, Any], refresh_result: dict[str, Any] | None = None
) -> None:
    status = _static_dataset_status(bmo_cfg)
    state = status["state"] if status["exists"] else "missing"
    source_url = _configured_static_dataset_url(bmo_cfg)
    with st.expander("Data and model sources", expanded=False):
        configured_anchor = str(bmo_cfg.get("fuel_rate_anchor_basis", "data_driven"))
        default_anchor_label = (
            "Data-Driven" if configured_anchor == "data_driven" else "Physics-Driven"
        )
        model_widget_key = "bmo_coke_rate_model"
        if str(st.session_state.get(model_widget_key, default_anchor_label)) not in (
            _COKE_ANCHOR_LABELS
        ):
            st.session_state.pop(model_widget_key, None)
        direct_cfg = dict(bmo_cfg.get("data_driven_coke", {}) or {})
        charged_cfg = dict(direct_cfg.get("charged_4h", {}) or {})
        setting_cols = st.columns([1.25, 1.0, 1.0, 0.9], gap="medium")
        with setting_cols[0]:
            selected_model = st.segmented_control(
                "Coke model",
                options=list(_COKE_ANCHOR_LABELS),
                default=default_anchor_label,
                key=model_widget_key,
                on_change=_clear_bmo_results,
                help=(
                    "Data-Driven uses the charged-coke forecast for the next five "
                    "hours. Physics-Driven uses the furnace heat balance."
                ),
            )
        with setting_cols[1]:
            st.number_input(
                "Blend response",
                min_value=0.0,
                max_value=1.0,
                value=float(
                    charged_cfg.get(
                        "blend_response_kg_coke_per_kg_slag", 0.10
                    )
                ),
                step=0.01,
                format="%.2f",
                key="bmo_blend_response_kg_coke_per_kg_slag",
                disabled=not is_admin(),
                on_change=_clear_bmo_results,
                help=(
                    "Extra kg coke/THM for each 1 kg/THM increase in candidate "
                    "slag versus the current burden. The forecast already sees "
                    "raw-material composition. Default 0.10 while plant testing "
                    "continues; only an admin can change it."
                ),
            )
        with setting_cols[2]:
            st.number_input(
                "Blend history (h)",
                min_value=1,
                max_value=72,
                value=int(direct_cfg.get("manual_blend_lookback_hours", 6)),
                step=1,
                key="bmo_manual_blend_lookback_hours",
                on_change=_clear_bmo_results,
                help=(
                    "Hours of recent charging used to build the manual-blend "
                    "comparison and transition path."
                ),
            )
        with setting_cols[3]:
            st.radio(
                "Force-predict",
                options=[False, True],
                index=0,
                format_func=lambda enabled: "On" if enabled else "Off",
                horizontal=True,
                key="bmo_force_predict",
                disabled=selected_model != "Data-Driven",
                on_change=_clear_bmo_results,
                help=(
                    "Data-Driven only. Shows the model's raw prediction even "
                    "when an operating parameter is outside its validated "
                    "range. The exact violations remain visible and no "
                    "validated error range is claimed."
                ),
            )
        st.divider()
        use_link = st.toggle(
            "Fetch dataset through DATA_URL",
            value=_use_static_dataset_link(bmo_cfg),
            key=_STATIC_DATASET_LINK_KEY,
            disabled=not bool(source_url),
            help=(
                "On: download the published furnace CSV. "
                "Off: rebuild it through the normal database/code pipeline."
            ),
            on_change=_clear_bmo_results,
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
            # Do not merely invalidate this browser session's derived values.
            # Ask the full rerun below to fetch the published CSV even when its
            # one-hour metadata TTL still says the local copy is fresh.
            st.session_state[_STATIC_DATASET_FORCE_REFRESH_KEY] = True
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


def _render_pinned_fuel_rates(
    fuel_ash_df: pd.DataFrame, fuel_rates: Mapping[str, Any]
) -> None:
    """State the effective read-only rates and their source."""

    if "fuel_id" not in fuel_ash_df.columns:
        return
    rates = {
        str(row["fuel_id"])
        .strip()
        .lower(): float(row.get("rate_kg_per_thm", 0.0) or 0.0)
        for _, row in fuel_ash_df.iterrows()
    }
    pci_source = str(fuel_rates.get("pci_source") or "fallback - LIVE TAG UNAVAILABLE")
    nut_source = str(fuel_rates.get("nut_coke_source") or "fixed plant set point")
    st.caption(
        f"**Fuel rates in force** - PCI **{rates.get('pci', 0.0):,.1f}** kg/THM "
        f"({pci_source}), nut coke **{rates.get('nut_coke', 0.0):,.1f}** kg/THM "
        f"({nut_source}). Coke comes from the selected coke-rate model, not here. "
        "The ash chemistry below feeds the slag balance."
    )


def _pci_override_kg_thm() -> float | None:
    """The operator's PCI override (kg/THM) when it is switched on, else None."""

    if st.session_state.get("bmo_pci_override_on"):
        override = st.session_state.get("bmo_pci_override_kg")
        if override is not None:
            return float(override)
    return None


def _nut_coke_override_kg_thm() -> float | None:
    """The operator's nut-coke override (kg/THM) when it is switched on, else None."""

    if st.session_state.get("bmo_nut_coke_override_on"):
        override = st.session_state.get("bmo_nut_coke_override_kg")
        if override is not None:
            return float(override)
    return None


def _effective_pci_kg_thm(snapshot: Mapping[str, Any]) -> float:
    """The PCI the WHOLE page must use: the operator's override, else the tag
    (``services.bmo.effective_pci_kg_thm``)."""

    return bmo_service.effective_pci_kg_thm(snapshot, _pci_override_kg_thm())


def _render_share_pie(blend: Any, selected_ores: list[OreInput], title: str) -> None:
    """Blend shares as a donut, largest first, tonnage in the middle.

    Ordered by share rather than by ore id so the eye lands on the material
    that dominates the blend; small slivers keep their label outside the ring
    instead of overprinting each other inside it.
    """

    rows = sorted(
        (
            {
                "ore_name": ore.display_name,
                "share_pct": float(blend.shares_pct.get(ore.ore_id, 0.0)),
                "qty_mt": float(blend.quantities_mt.get(ore.ore_id, 0.0) or 0.0),
            }
            for ore in selected_ores
            if float(blend.shares_pct.get(ore.ore_id, 0.0)) > 0
        ),
        key=lambda row: row["share_pct"],
        reverse=True,
    )
    if not rows:
        return

    frame = pd.DataFrame(rows)
    total_mt = float(frame["qty_mt"].sum())
    fig = go.Figure(
        go.Pie(
            labels=frame["ore_name"],
            values=frame["share_pct"],
            hole=0.62,
            sort=False,
            direction="clockwise",
            textposition="auto",
            texttemplate="%{percent:.1%}",
            insidetextorientation="horizontal",
            customdata=frame[["qty_mt"]],
            hovertemplate=(
                "<b>%{label}</b><br>%{percent:.1%} of blend"
                "<br>%{customdata[0]:,.1f} MT<extra></extra>"
            ),
            marker=dict(
                colors=px.colors.qualitative.Safe,
                line=dict(color="rgba(128,128,128,0.35)", width=1),
            ),
        )
    )
    fig.add_annotation(
        text=(
            f"<span style='font-size:19px'><b>{total_mt:,.0f}</b></span>"
            f"<br><span style='font-size:11px'>MT burden</span>"
        ),
        showarrow=False,
        font=dict(size=13),
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=8, r=8, t=36, b=8),
        height=330,
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.0,
            xanchor="left",
            y=0.5,
            yanchor="middle",
            font=dict(size=11),
        ),
    )
    st.plotly_chart(fig, width="stretch")


def _render_blend_comparison(
    provider: EvonithBmoContextProvider,
    *,
    case: bmo_service.BlendCase,
    optimizer_candidates: list[tuple[str, Any, float | None]],
    shift_baseline: dict[str, Any] | None = None,
) -> None:
    """Operator-focused comparison of the manual blend vs the optimizer blends.

    Presents two tables on the same all-source iron-closure basis so the operator can decide at
    a glance:

      * **Inputs** - the suggested blend mix (ore share %) for each option, so the
        LP and DE mixes are read side by side against the editable manual blend.
      * **Outputs** - the outcomes that drive the decision: total / ore / fuel
        cost, fuel rate, slag basicity / T-basicity, and slag rate.

    The seed, the manual blend and the tables come from ``services.bmo``
    (``manual_seed``, ``evaluate_manual_blend``, ``comparison_view``); this
    function renders them and keeps the editor's state.

    Args:
         - provider: EvonithBmoContextProvider - Source for the last-shift manual blend + fuel context.
         - case: BlendCase - The run's inputs (ores, targets, slag-balance inputs, Si).
         - optimizer_candidates: list[tuple[str, BlendEvaluation, float | None]] -
           Optimizer results in display order. The third value is the common
           furnace-state Si input retained by the fuel-evaluation contract.
         - shift_baseline: dict | None - The last complete shift's production.

    Returns:
         - return None - Renders the comparison tables to Streamlit.
    """

    if not optimizer_candidates:
        return
    # Prefer the last candidate (DE over LP) to seed the manual editor.
    primary_blend = optimizer_candidates[-1][1]
    compare_ores = case.ores
    lookback_hours = case.lookback_hours

    snapshot = provider.get_recent_manual_blend_snapshot(
        compare_ores, lookback_hours=lookback_hours
    )

    st.markdown("##### Manual blend vs optimizer")
    st.caption(
        "Edit a share to try any burden split. Shares are normalised to 100% and go "
        "through the same all-source Fe/material closure as the optimizer, so every "
        "option is compared on the same basis. Manual, Balanced Optimizer and "
        "Intensive Optimizer all use the currently applied ore prices from Ore Selection; "
        "fuel is shown at the currently applied fuel prices."
    )
    seed = bmo_service.manual_seed(
        snapshot, compare_ores, primary_blend, lookback_hours=lookback_hours
    )
    # Editor on the left, the cost verdict beside it; both are filled below.
    editor_col, result_col = st.columns([2, 3], gap="large")
    with editor_col:
        st.markdown("**Manual burden**")
        edited_share_df = st.data_editor(
            pd.DataFrame(seed.rows),
            hide_index=True,
            width="stretch",
            key="bmo_manual_share_editor",
            column_order=("ore_name", "manual_share_pct"),
            column_config={
                "ore_id": None,
                "ore_name": st.column_config.TextColumn(
                    "Ore", disabled=True, width="medium"
                ),
                "manual_share_pct": st.column_config.NumberColumn(
                    "Share (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    format="%.1f %%",
                    width="small",
                    help="Edit to try another split. Shares are normalised to 100%.",
                ),
            },
        )
    manual_shares_pct = {
        str(row["ore_id"]): float(row["manual_share_pct"] or 0.0)
        for _, row in edited_share_df.iterrows()
    }
    manual_quantities, normalized_total, scale_warnings = (
        bmo_service.target_quantities_from_shares(
            manual_shares_pct,
            compare_ores,
            case.target_fe_mt,
            target_hot_metal_mt=case.target_production_mt,
            fuel_ash_inputs=case.fuel_ash_inputs,
            flux_inputs=case.flux_inputs,
            dust_inputs=case.dust_inputs,
            slag_balance_settings=case.slag_balance_settings,
            charge_mass_mt=case.charge_mass_mt,
        )
    )
    with editor_col:
        if normalized_total > 0:
            st.caption(
                f"Shares entered total {normalized_total:,.1f}%, normalised to 100%. "
                + seed.note
            )
        else:
            st.caption(seed.note)
        for warning in scale_warnings:
            st.warning(warning)

    manual_blend = None
    manual_si = case.live_si_pct
    if manual_quantities:
        try:
            fuel_context = _load_fuel_prediction_context(provider)
            fuel_warnings = fuel_context[4]
            # The live furnace forecast is independent of candidate burden. If
            # the optional Si correction is ever enabled, using the same causal
            # furnace value on both sides gives the honest zero blend response.
            st.session_state["bmo_manual_quantities_mt"] = dict(manual_quantities)
            manual_blend = bmo_service.evaluate_manual_blend(
                case,
                compare_ores=compare_ores,
                quantities_mt=manual_quantities,
                fuel_context=fuel_context,
                manual_si=manual_si,
            )
            # Persisted so the furnace commentary below can state the optimizer's
            # proposal as a DELTA against what the plant is actually running.
            # Without it the model can only describe levels, which an operator
            # already knows.
            st.session_state["bmo_manual_blend"] = manual_blend
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
    view = bmo_service.comparison_view(
        options=options,
        optimizer_candidates=optimizer_candidates,
        manual_blend=manual_blend,
        compare_ores=compare_ores,
        charge_mass_mt=case.charge_mass_mt,
        shift_baseline=shift_baseline,
        rows_by_ore=seed.rows_by_ore,
        use_last_shift=seed.use_last_shift,
    )
    baseline = view.baseline
    baseline_mt = baseline.get("daily_production_mt")

    # Headline FIRST. The cheapest option and the saving are the decision; the
    # detail below is the justification, and reading order should match.
    with result_col:
        st.markdown("**Total cost per tonne of hot metal, and production**")
        with st.container(border=True):
            cols = st.columns(len(view.headlines))
            for col, headline in zip(cols, view.headlines):
                col.metric(
                    f"{headline.label} (Rs/THM)",
                    f"{headline.total_rs_per_thm:,.0f}",
                    help="Ore + fuel at the currently applied prices + any flux "
                    "the optimizer added.",
                )
                change = headline.production_change_pct
                if headline.is_manual and baseline_mt is not None:
                    col.caption(f"{headline.production_mt:,.0f} MT/day · last shift")
                elif change is not None:
                    col.caption(
                        f"{headline.production_mt:,.0f} MT/day · Δ Production "
                        f"{headline.production_delta_mt:+,.0f} MT/day ({change:+.1f}%)"
                    )
                else:
                    col.caption(f"{headline.production_mt:,.0f} MT/day")
            for headline in view.headlines:
                change = headline.production_change_pct
                if headline.is_manual or change is None:
                    continue
                st.badge(
                    f"{headline.label}: Δ Production {headline.production_delta_mt:+,.0f} MT/day "
                    f"({change:+.1f}%) vs last shift, {float(baseline_mt):,.0f} to "
                    f"{headline.production_mt:,.0f} MT/day",
                    icon=(
                        ":material/trending_up:"
                        if change >= 0
                        else ":material/trending_down:"
                    ),
                    color="green" if change >= 0 else "orange",
                    help=(
                        f"Last shift is {baseline.get('description', '')}: "
                        f"{float(baseline.get('charges_per_hour') or 0):,.2f} charges/hr "
                        f"x {float(baseline.get('hm_per_charge_mt') or 0):,.2f} t hot "
                        f"metal per charge x 24 ({baseline.get('source', '')})."
                    ),
                )
            if view.saving is not None:
                best_label, saving = view.saving
                if saving > 0:
                    st.badge(
                        f"{best_label} saves {saving:,.0f} Rs/THM against the manual blend",
                        icon=":material/savings:",
                        color="green",
                    )
                elif saving < 0:
                    st.badge(
                        f"Manual blend is {-saving:,.0f} Rs/THM cheaper than the "
                        f"best optimizer blend ({best_label})",
                        icon=":material/check_circle:",
                        color="blue",
                    )
                else:
                    st.badge(
                        "Manual blend matches the best optimizer cost",
                        icon=":material/drag_handle:",
                        color="gray",
                    )
        st.caption(
            "Details by topic below; the lowest cost in each cost row is highlighted."
        )

    st.markdown("###### Key outcomes")
    outcome_tabs = st.tabs([*view.groups, "🧱 Mix"])
    with outcome_tabs[-1]:
        mix_frame = view.mix
        st.dataframe(
            mix_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Ore": st.column_config.TextColumn("Ore", width="medium"),
                **{
                    column: st.column_config.NumberColumn(
                        column,
                        format="%.1f %%" if column.endswith("(%)") else "localized",
                    )
                    for column in mix_frame.columns
                    if column != "Ore"
                },
            },
        )
        st.caption(
            "Share of the burden and wet tonnes per day for each option. "
            + (
                "Manual tonnage is what was actually charged over the last "
                f"{lookback_hours:g} hours, as a daily rate; "
                if view.manual_actual_tonnage
                else ""
            )
            + "optimizer tonnage is each solution's charge per day at its target "
            "production."
        )
    for tab, (group_name, outcome) in zip(outcome_tabs, view.groups.items()):
        with tab:
            if group_name == "🚚 Charging" and baseline_mt is not None:
                st.caption(
                    f"Manual production is the last complete shift's rate "
                    f"({baseline.get('description', '')}, charges/hr x hot metal "
                    "per charge x 24). The other Manual rows are evaluated at the "
                    "target production, like the optimizer columns."
                )
            frame = outcome.frame()
            if len(frame.index) == 0:
                st.caption("Nothing to show for this group.")
                continue

            def _highlight_best(
                row: pd.Series, outcome: bmo_service.OutcomeRows = outcome
            ) -> list[str]:
                best = outcome.best(row.name)
                return [
                    "background-color: rgba(18,184,134,0.16); font-weight: 600"
                    if column in best
                    else ""
                    for column in row.index
                ]

            st.dataframe(
                frame.style.apply(_highlight_best, axis=1),
                hide_index=True,
                width="stretch",
                column_config={
                    "Outcome": st.column_config.TextColumn("Outcome", width="medium"),
                    **{
                        label: st.column_config.TextColumn(label, alignment="right")
                        for label in view.labels
                    },
                },
            )


@fragment
def _render_data_diagnostics(
    provider: EvonithBmoContextProvider,
    *,
    ore_diagnostics: dict[str, Any],
    hm_snapshot: dict[str, Any],
    edited_ore_df: pd.DataFrame,
) -> None:
    with st.expander("Data Diagnostics", expanded=False):
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
        dust = diagnostics.get("dust", {})
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
                    "area": "dpr_top_gas_dust",
                    "source": dust.get("source", "offline_feed.dpr_data"),
                    "timestamp/window": f"{dust.get('start_time', '')} -> {dust.get('end_time', '')}",
                    "rows": dust.get("dpr_rows", 0),
                    "note": ", ".join(
                        f"{r.get('dust_id')} {float(r.get('kg_per_charge', 0.0)):.0f} kg/charge "
                        f"over {r.get('days', 0)} d"
                        for r in dust.get("rows", [])
                    )
                    or dust.get("error", "no DPR dust; configured rows"),
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

    Loading the non-linear model, scaler, and selected feature list is relatively
    expensive. Caching the service keeps model artifacts warm while allowing
    each candidate blend to request fresh predictions.

    Args:
         - None

    Returns:
         - return FuelUnitCostModelService - Cached fuel-cost prediction service.
    """

    return case_service.fuel_model_service(_get_bmo_config())


def _data_driven_coke_config() -> dict[str, Any]:
    return dict(_get_bmo_config().get("data_driven_coke", {}) or {})


def _charged_dataset_path() -> Path:
    return _repo_path(
        str(
            _data_driven_coke_config().get(
                "dataset_path", "src/assets/data/furnace_dataset.csv"
            )
        )
    )


def _charged_cfg() -> dict[str, Any]:
    return dict(_data_driven_coke_config().get("charged_4h", {}) or {})


def _charged_storage() -> Path:
    return _repo_path(str(_charged_cfg().get("storage_dir", "src/storage/bmo_charged_coke")))


def _charged_default_bundle() -> Path:
    return _repo_path(str(_charged_cfg().get("bundle_dir", "src/assets/models/bmo_charged_coke/20261007_0400")))


@_resource_cache(show_spinner=False)
def _load_charged_engine(bundle_path: str) -> ChargedCokeEngine:
    """One model, policy and physics per bundle directory, loaded once."""

    return ChargedCokeEngine.load(bundle_path)


def _get_charged_engine() -> ChargedCokeEngine:
    """The accepted charged-coke model, or the one shipped with the app."""

    return _load_charged_engine(str(active_charged_bundle(_charged_storage(), _charged_default_bundle())))


@st.cache_data(ttl=900, show_spinner=False)
def _charged_recent(dataset_path: str, dataset_mtime_ns: int, bundle_path: str, end_row: str) -> pd.DataFrame:
    """The active model's last 7 days, replayed hour by hour (cached per data/model version)."""

    from utils.bmo.charged_coke.service import recent_forecasts

    source = _charged_source_frame(dataset_path, dataset_mtime_ns)
    return recent_forecasts(source, _load_charged_engine(bundle_path), end=pd.Timestamp(end_row), days=7)


@_resource_cache(show_spinner=False)
def _charged_source_frame(dataset_path: str, dataset_mtime_ns: int) -> pd.DataFrame:
    """The published hourly dataset, read once per file version."""

    del dataset_mtime_ns
    return pd.read_csv(dataset_path, parse_dates=["time"]).set_index("time")


def _charged_prediction_for_page(
    *,
    pci_override: float | None,
    nut_override: float | None,
    slag_response_coefficient: float,
    force_predict: bool,
) -> tuple[ChargedCokePrediction, dict[str, Any]]:
    """The charged forecast for the BMO level (``services.bmo.charged_prediction``): read
    from the ledger while the worker issues it (SV-1.5), else issued here once per
    completed hour (``bmo.forecast_issuer``)."""

    from utils.bmo.charged_coke.bmo import _built_at

    dataset_path = _charged_dataset_path()
    mtime = int(dataset_path.stat().st_mtime_ns) if dataset_path.is_file() else 0
    return bmo_service.charged_prediction(
        cfg=_get_bmo_config(),
        source=_charged_source_frame(str(dataset_path), mtime),
        engine=_get_charged_engine(),
        storage=_charged_storage(),
        built_at=_built_at(dataset_path.parent / "cache_meta.json"),
        now=pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None),
        pci_override=pci_override,
        nut_override=nut_override,
        slag_response_coefficient=slag_response_coefficient,
        force_predict=force_predict,
    )


def _si_furnace_paths(
    bmo_cfg: Mapping[str, Any],
) -> tuple[str, Path, Path, Path]:
    """Independent production-forecast mode and paths (never legacy Si paths)."""

    paths = forecast_service.si_paths(bmo_cfg)
    return paths.mode, paths.default_bundle, paths.storage_dir, paths.static_dataset


@_resource_cache(show_spinner=False)
def _get_si_furnace_service(
    bundle_path: str,
    storage_path: str,
    forest_mtime_ns: int,
) -> SiliconForecastService:
    del forest_mtime_ns
    return SiliconForecastService(
        bundle_dir=bundle_path,
        storage_dir=storage_path,
        furnace_id="BF2",
    )


@_data_cache(ttl=300, show_spinner=False)
def _issue_si_furnace_forecast(
    *,
    mode: str,
    bundle_path: str,
    storage_path: str,
    static_dataset_path: str,
    origin_at: str,
    source_cache_version: int,
    forest_mtime_ns: int,
) -> dict[str, Any]:
    """Fetch/issue once per origin, active version and explicit refresh token (when
    page 9 issues: ``bmo.forecast_issuer`` page, or auto without a worker)."""

    del source_cache_version, mode, bundle_path, storage_path, static_dataset_path, forest_mtime_ns
    return bmo_service.issue_si_for_page(origin_at, _get_bmo_config())


@_data_cache(ttl=600, show_spinner=False)
def _replay_si_furnace_history(
    *,
    bundle_path: str,
    storage_path: str,
    static_dataset_path: str,
    history_end_at: str,
    source_cache_version: int,
    forest_mtime_ns: int,
) -> dict[str, Any]:
    """Causally replay the active model for the trend; never write live issues."""

    del source_cache_version
    return bmo_service.replay_si_history(
        service=_get_si_furnace_service(bundle_path, storage_path, forest_mtime_ns),
        make_source=lambda: LiveSiliconForecastSource(
            bundle_dir=bundle_path,
            static_dataset_path=static_dataset_path,
        ),
        history_end_at=history_end_at,
    )


def _si_furnace_forecast_for_page(
    bmo_cfg: Mapping[str, Any], *, source_cache_version: int
) -> tuple[str, dict[str, Any] | None, str | None]:
    """Return the global production forecast without touching blend Si state.

    Read from the ledger while the worker issues it (SV-1.5), else issued here
    (``bmo.forecast_issuer``; ``services.bmo.si_forecast_for_page``). The third
    value is a note on the record shown when it is not the current origin's.
    """

    # TestBMO rewrites this key to testbmo_ui_frozen. A loaded sandbox must show
    # the archived result and must never query today's production feeds.
    frozen = st.session_state.get("bmo_ui_frozen") is not None
    view = bmo_service.si_forecast_for_page(
        bmo_cfg,
        frozen=frozen,
        frozen_record=st.session_state.get("bmo_si_furnace_forecast"),
        now=pd.Timestamp.now(tz="Asia/Kolkata"),
        source_cache_version=int(source_cache_version),
    )
    if view.mode == "legacy" or frozen:
        return view.mode, view.record, view.note
    record = (
        _issue_si_furnace_forecast(**view.issue)
        if view.issue is not None
        else view.record
    )
    st.session_state["bmo_si_furnace_forecast"] = record
    return view.mode, record, view.note


@st.fragment(run_every=300)
def _render_live_si_furnace_panel(
    bmo_cfg: Mapping[str, Any], *, source_cache_version: int
) -> None:
    """Refresh only the advisory Si panel every five minutes."""

    mode, record, note = _si_furnace_forecast_for_page(
        bmo_cfg, source_cache_version=source_cache_version
    )
    if mode == "legacy" or record is None:
        if note:
            st.caption(note)
        return
    if note:
        st.caption(note)
    _mode, default_bundle, storage_dir, static_dataset = _si_furnace_paths(bmo_cfg)
    history: dict[str, Any] = {
        "forecasts": pd.DataFrame(),
        "actuals": pd.DataFrame(),
        "scored": pd.DataFrame(),
        "error": None,
    }
    # A sandbox snapshot must never reach into today's production databases.
    if st.session_state.get("bmo_ui_frozen") is None:
        active = active_si_bundle(storage_dir, default_bundle)
        try:
            model_mtime = int((active / "manifest.json").stat().st_mtime_ns)
        except OSError:
            model_mtime = 0
        history_end = pd.Timestamp.now(tz="Asia/Kolkata").floor("10min")
        history = _replay_si_furnace_history(
            bundle_path=str(active),
            storage_path=str(storage_dir),
            static_dataset_path=str(static_dataset),
            history_end_at=history_end.isoformat(),
            source_cache_version=int(source_cache_version),
            forest_mtime_ns=model_mtime,
        )
    render_si_furnace_forecast(
        record,
        storage_dir=storage_dir,
        default_bundle=default_bundle,
        static_dataset_path=static_dataset,
        can_manage=bool(is_admin() or is_supervisor()),
        user=str(st.session_state.get("auth_user") or "unknown"),
        on_change=_clear_bmo_results,
        replay_forecasts=history["forecasts"],
        raw_actuals=history["actuals"],
        replay_scored=history["scored"],
        history_error=history["error"],
    )


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
            "by the Physics-Driven model and the process recommendation, and is "
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
            frame[
                [
                    "Parameter",
                    "Value",
                    "Unit",
                    "Source",
                    "Default",
                    "Basis",
                    "Why it matters",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            key="energy_assumptions_editor",
            column_config={
                "Value": st.column_config.NumberColumn(
                    "Value",
                    help="Your figure. Bounds are enforced on save.",
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

    st.markdown("##### Recommended process parameters")
    snapshot = _live_process_snapshot()
    # The live tags, the blend's fuel rates and the PCI the balance uses
    # (live tag or override, never the Fuel Ash box) come from services.bmo.
    basis = bmo_service.control_basis(
        blend, snapshot=snapshot, live_pci=_effective_pci_kg_thm(snapshot)
    )
    if basis.unavailable:
        st.info(basis.unavailable)
        return
    fuel_rates = basis.fuel_rates
    if basis.pci_note:
        st.caption(basis.pci_note)
    try:
        energy_inputs, current = bmo_service.control_inputs(
            blend,
            basis,
            hot_metal_mt=hot_metal_mt,
            ores=ores,
            hm_chem_values=hm_chem_values,
            hm_snapshot=hm_snapshot,
            flux_inputs=flux_inputs,
            fuel_ash_inputs=fuel_ash_inputs,
            snapshot=snapshot,
        )
        release_pci = st.checkbox(
            "Allow PCI to move",
            value=False,
            key=f"bmo_pr_pci_{label}",
            help="PCI is held by default. Release it only when you intend to change it.",
        )
        recommendation = bmo_service.recommend_process(
            energy_inputs,
            current,
            fuel_ash_inputs=fuel_ash_inputs,
            release_pci=release_pci,
        )
    except Exception as exc:  # noqa: BLE001 - never break the blend result pane
        log.warning("Process recommendation failed: %s", exc)
        st.warning(f"Could not compute process parameters: {exc}")
        return

    # The energy balance predicts coke with the right shape but the wrong level
    # - +19.7 kg/tHM over 239 days. A rolling offset fitted on recent plant
    # history takes MAPE from 7.2% to 3.4%. Applied AFTER the optimise, because
    # a constant shift cannot change which controls win, only the reported
    # level. See utils/bmo/coke_calibration.py.
    from utils.bmo.coke_calibration import load_calibration

    calib = load_calibration()
    coke_shown = calib.apply(recommendation.coke_rate_kg_per_thm)
    coke_now = calib.apply(recommendation.current_coke_rate_kg_per_thm)

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Coke Rate (kg/THM)",
        f"{coke_shown:,.1f}",
    )
    c2.metric(
        "Fuel Cost (Rs/THM)",
        f"{recommendation.fuel_cost_rs_per_thm:,.0f}",
    )
    raft_delta = recommendation.raft_delta_c
    c3.metric(
        "RAFT (C)",
        f"{recommendation.raft_c:,.0f}" if recommendation.raft_c else "n/a",
    )
    if raft_delta is not None and recommendation.current_raft_c:
        direction = (
            "hotter" if raft_delta > 0 else "colder" if raft_delta < 0 else "unchanged"
        )
        strength = (
            "within measurement noise" if abs(raft_delta) < 17.0 else "significant"
        )
        st.caption(
            f"RAFT moves from **{recommendation.current_raft_c:,.0f} °C** to "
            f"**{recommendation.raft_c:,.0f} °C** — raceway runs **{direction}** "
            f"by {abs(raft_delta):,.0f} °C ({strength}, formula is good to ±17 °C)."
        )

    # Three different coke rates exist in this app and they do not agree. Naming
    # them side by side is the only way to stop the comparison being misread.
    plant_coke = float(fuel_rates.get("coke_rate_kg_thm", 0.0) or 0.0)
    with st.expander("Where does this coke rate come from?", expanded=False):
        comparison = bmo_service.coke_source_rows(
            recommendation,
            calib,
            plant_coke=plant_coke,
            coke_now=coke_now,
            coke_shown=coke_shown,
        )
        st.dataframe(
            pd.DataFrame(comparison), hide_index=True, use_container_width=True
        )
        residual = coke_now - plant_coke if plant_coke else None
        st.markdown(
            "**Why an offset at all.** Backtested over 239 days, the raw energy "
            "balance runs **+19.7 kg/THM high** with MAPE 7.2% and R² 0.07 — the "
            "shape is right, the level is not. One rolling offset takes that to "
            "MAPE 3.4%, R² 0.74 on a forward test. It is shown rather than "
            "folded in, because its size measures how much the balance is still "
            "missing: when the top-gas analyser and shell-loss questions are "
            "settled, it should shrink on its own."
            + (
                f"\n\nAfter correction the balance sits **{residual:+,.1f} kg/THM** "
                "from plant actual"
                + (
                    " — within the model's expected scatter."
                    if abs(residual) < 15
                    else ". That is wider than the ±10 kg/THM the offset usually "
                    "leaves, so this blend or these controls are further from "
                    "recent operation than the calibration window covers."
                )
                if residual is not None
                else ""
            )
            + "\n\n**The delta is the more reliable half.** The offset cancels "
            "between the two settings, so the recommended change is unaffected "
            "by any of this."
            + "\n\nThe **ML fuel-cost model** is a third, separate estimate. It "
            "is trained on plant history and is blend-blind; the Physics-Driven "
            "energy balance responds to the blend. They will not agree, and are "
            "not meant to."
        )
        if calib.is_stale():
            st.warning(
                f"Calibration was fitted {calib.age_days()} days ago. The bias "
                "drifts about 2 kg/THM per quarter — run "
                "`scripts/refresh_coke_calibration.py`."
            )
        for note in calib.notes:
            st.caption(f"⚠️ {note}")

    st.dataframe(
        pd.DataFrame(bmo_service.control_rows(recommendation)),
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
    case: bmo_service.BlendCase,
) -> None:
    """The path from what the plant is charging today to the LP optimum.

    The LP says where to go. It does not say how to get there, and a
    20-percentage-point share change is not an instruction anyone can act on -
    burden descent takes 6-7 hours and ore supply does not turn overnight. Each
    rung here is a full LP solve under a per-ore move cap, so every step on the
    path independently satisfies all six slag limits.
    """

    ores = case.ores
    st.markdown("##### Path from the current blend")
    move_col, _ = st.columns([1, 3])
    move_pct = move_col.number_input(
        "Max share change per step (%)",
        min_value=1.0,
        max_value=50.0,
        value=2.0,
        step=1.0,
        key="bmo_transition_move_pct",
        help=(
            "Your step-change policy. Smaller steps mean more shifts to reach "
            "the optimum but a gentler move for the furnace."
        ),
    )

    snapshot = provider.get_recent_manual_blend_snapshot(
        ores, lookback_hours=case.lookback_hours
    )
    manual_shares = bmo_service.manual_shares(snapshot)
    if not any(manual_shares.values()):
        st.info(
            "No recent manual blend found, so there is nothing to transition "
            "from. The optimal blend above still stands."
        )
        return

    try:
        ladder = bmo_service.transition_ladder(
            case, shares_pct=manual_shares, max_share_move_pct=float(move_pct)
        )
    except Exception as exc:  # noqa: BLE001 - never break the result pane
        log.warning("Transition ladder failed: %s", exc)
        st.warning(f"Could not build the transition path: {exc}")
        return

    note = bmo_service.transition_note(ladder)
    if note:
        st.warning(note)

    st.dataframe(
        pd.DataFrame(bmo_service.transition_rows(ladder, ores)),
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


def _render_fuel_basis_note(blend: Any) -> None:
    """Say which of the three coke rates this blend's fuel cost was built on.

    The anchor is invisible in the numbers - a fuel cost of 15,406 Rs/THM looks
    the same whether it came from the energy balance or from yesterday's Fuel
    Ash row. When the physics anchor silently falls back, the operator has to be
    able to see that it did, and why.
    """

    source = str(blend.diagnostics.get("fuel_rate_estimate_source", ""))
    anchor = st.session_state.get("bmo_energy_anchor")

    if source == "data_driven_coke_anchor":
        return

    if source == "energy_balance_anchor" and anchor is not None:
        st.caption(
            f"Coke level from the **Physics-Driven energy balance** - solved at "
            "current controls "
            f"({anchor.raw_coke_rate_kg_thm:,.1f} kg/THM), less the "
            f"{anchor.offset_kg_per_thm:+,.1f} kg/THM bias offset fitted on "
            f"{anchor.calibration.sample_days} recent days. Blend-to-blend "
            "differences come from the physics correction above, not from this."
        )
        for note in anchor.notes:
            st.caption(f"⚠️ {note}")
        return

    if anchor is not None and not anchor.usable:
        # Configured for the physics anchor and did not get it. Say so plainly
        # rather than letting the observed figure pass for the balance's answer.
        st.warning(
            "**Fuel cost is anchored on the observed coke rate, not the energy "
            "balance.** " + " ".join(anchor.notes)
        )
        return

    if source in ("observed_fuel_ash_inputs", "fuel_ash_inputs"):
        st.caption(
            "Coke level from the **Fuel Ash table** — what the plant is charging "
            "now. It does not respond to a change in controls."
        )
    elif source == "model_cost_residual":
        st.caption(
            "Coke level back-solved from the **ML fuel-cost model**. That model's "
            "output is near-constant across blends, so treat the level as "
            "indicative and read the correction above for the blend effect."
        )
    elif anchor is None and str(fuel_rate_anchor_basis) == "energy_balance":
        st.caption(
            "The energy-balance anchor could not find a current burden to solve "
            "against — no charge data for the last shift. The coke level falls "
            "back to the observed rate."
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

    exploration = bmo_service.de_exploration(candidates, selected_ores)
    if exploration is None:
        return
    df = exploration.cloud

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

    table = exploration.table
    st.markdown("###### Top 100 alternative solutions (cheapest first)")
    if exploration.has_shares:
        st.caption(
            "Each row is a full blend: ore shares (%) and optimizer-added flux "
            "(MT) alongside its cost, basicity, and slag outcome."
        )
    else:
        st.info(
            "Blend-combination columns are unavailable for this run — re-run "
            "the Intensive Optimizer to record ore shares and flux additions "
            "per candidate. (After a code update, restart the app so the "
            "optimizer module reloads.)"
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


def _load_fuel_prediction_context(
    provider: EvonithBmoContextProvider,
) -> bmo_service.FuelContext:
    """Process, history, and model inputs required for fuel inference
    (``services.bmo.load_fuel_prediction_context`` on the cached model service)."""

    return bmo_service.load_fuel_prediction_context(provider, _get_model_service())


def _resolve_energy_anchor(**kwargs: Any) -> Any | None:
    """Solve the energy-balance coke anchor for the current operating point
    (``services.bmo.resolve_energy_anchor``) at the live process tags and the
    operator's PCI override."""

    return bmo_service.resolve_energy_anchor(
        **kwargs,
        process_snapshot=_live_process_snapshot,
        pci_override_kg_thm=_pci_override_kg_thm(),
    )


def _render_physics_coke_prediction(anchor: Any | None) -> None:
    """Compact current coke-rate prediction from the energy balance."""

    st.markdown("### Coke-rate prediction")
    st.markdown("**Physics-Driven energy balance**")
    if anchor is None:
        st.warning(
            "The Physics-Driven coke rate could not be calculated because a "
            "current burden was not available."
        )
        return
    if not anchor.usable:
        notes = [str(note) for note in (anchor.notes or []) if note]
        message = "The Physics-Driven coke rate is not available."
        if notes:
            message += "\n\n" + "\n".join(f"- {note}" for note in notes)
        st.warning(message)
        return

    value_col, raw_col, adjustment_col = st.columns(3)
    value_col.caption("Predicted coke rate")
    value_col.markdown(f"**{anchor.coke_rate_kg_thm:,.1f} kg/THM**")
    raw_col.caption("Raw energy balance")
    raw_col.markdown(f"**{anchor.raw_coke_rate_kg_thm:,.1f} kg/THM**")
    adjustment_col.caption("Calibration adjustment")
    adjustment_col.markdown(
        f"**{anchor.coke_rate_kg_thm - anchor.raw_coke_rate_kg_thm:+,.1f} kg/THM**"
    )
    st.caption(
        "Calculated for the current burden, blast and fuel settings. This is "
        "the coke-rate level used by the Physics-Driven optimiser."
    )
    if anchor.notes:
        with st.popover("Prediction details"):
            for note in anchor.notes:
                st.caption(f"- {note}")


apply_bmo_styles()
bmo_cfg = _get_bmo_config()
# Recorded so a snapshot can freeze the plant data each run used; the wrapper
# returns exactly what the provider returns.
provider = record_provider(_get_context_provider(), prefix="bmo_")
model_service = _get_model_service()
bundle_status = model_service.get_bundle_status()
st.session_state["bmo_bundle_status"] = bundle_status
render_header(bundle_status)
force_static_refresh = bool(
    st.session_state.pop(_STATIC_DATASET_FORCE_REFRESH_KEY, False)
)
with st.spinner("Checking the hourly furnace dataset..."):
    static_refresh_result = _refresh_static_dataset_if_needed(
        bmo_cfg, force=force_static_refresh
    )
if static_refresh_result.get("error") and not static_refresh_result.get("usable"):
    st.error(
        "The hourly furnace CSV could not be fetched and no local fallback exists. "
        f"Error: {static_refresh_result['error']}"
    )
    st.stop()
_render_static_dataset_bar(bmo_cfg, static_refresh_result)

_configured_anchor = str(bmo_cfg.get("fuel_rate_anchor_basis", "data_driven"))
_default_anchor_label = (
    "Data-Driven" if _configured_anchor == "data_driven" else "Physics-Driven"
)
fuel_rate_anchor_basis = _COKE_ANCHOR_LABELS.get(
    str(st.session_state.get("bmo_coke_rate_model", _default_anchor_label)),
    "data_driven",
)
_direct_cfg_top = dict(bmo_cfg.get("data_driven_coke", {}) or {})
_charged_cfg_top = dict(_direct_cfg_top.get("charged_4h", {}) or {})
blend_response_kg_coke_per_kg_slag = float(
    st.session_state.get(
        "bmo_blend_response_kg_coke_per_kg_slag",
        _charged_cfg_top.get("blend_response_kg_coke_per_kg_slag", 0.10),
    )
)
use_charged_engine = fuel_rate_anchor_basis == "data_driven"
force_data_driven_prediction = bool(
    use_charged_engine and st.session_state.get("bmo_force_predict", False)
)
manual_blend_lookback_hours = int(
    st.session_state.get(
        "bmo_manual_blend_lookback_hours",
        (bmo_cfg.get("data_driven_coke", {}) or {}).get(
            "manual_blend_lookback_hours", 6
        ),
    )
)
# Bumped by the "Refresh source data" button; keys the cached offline-source
# reads so they are fetched once per session and reused until the operator asks
# for fresh data.
source_cache_version = int(st.session_state.get("bmo_source_cache_version", 0))
_si_forecast_mode, si_furnace_forecast, _si_furnace_note = _si_furnace_forecast_for_page(
    bmo_cfg, source_cache_version=source_cache_version
)
live_si_result_tile = bmo_service.live_si_result_tile(si_furnace_forecast)
live_si_pct = (
    float(live_si_result_tile["si_pct"])
    if live_si_result_tile is not None
    else None
)

ui_cfg = bmo_cfg.get("ui", {})
target_cfg = bmo_cfg.get("target", {})
runtime_cfg = build_runtime_config(
    bmo_cfg, default_optimizer=bmo_cfg.get("optimization", {})
)
opt_cfg = runtime_cfg.get("optimizer", {})
operator_preferences_path = _ore_preferences_path(bmo_cfg)
operator_preferences = _cached_operator_preferences(
    str(operator_preferences_path), _operator_inputs_token(operator_preferences_path)
)
static_path, static_mtime_ns = _static_dataset_cache_token(bmo_cfg)
# Live 1-hour-average plant tags win over static-CSV values (which lag and
# derive nut coke from charged MT); static remains the offline fallback.
# THE OVERRIDES ARE APPLIED HERE, AT SOURCE (services.bmo_case.effective_fuel_rates):
# live 1-hour tags win over the static CSV (which lags and derives nut coke from
# charged MT), then the operator's PCI / nut-coke overrides. Everything downstream -
# the fuel rows, the slag balance's fuel ash, the re-priced fuel cost - reads
# recent_fuel_rates. The controls render further down, immediately above the run
# buttons; Streamlit restores widget state before the script runs, so their values
# are already available at this point.
recent_fuel_rates = case_service.effective_fuel_rates(
    _recent_fuel_rates_from_static_csv(static_path, static_mtime_ns),
    _recent_fuel_rates_live(),
    pci_override_kg_thm=_pci_override_kg_thm(),
    nut_coke_override_kg_thm=_nut_coke_override_kg_thm(),
)

# Config, the hourly dataset's slag window, the charging plant, then saved values
# (services.bmo_case.model_input_defaults). plant_slag = model_slag * factor, so a
# plant-basis target divides by it to get the model-basis cap the solvers enforce.
model_input_defaults = case_service.model_input_defaults(
    bmo_cfg,
    _model_input_defaults_from_static_csv(static_path, static_mtime_ns),
    operator_preferences,
)
model_to_plant_slag_factor = case_service.model_to_plant_slag_factor(bmo_cfg)
burden_capacity_cfg = bmo_cfg.get("burden_capacity", {}) or {}
burden_capacity_enabled_default = bool(burden_capacity_cfg.get("enabled", True))

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
            min_value=float(target_cfg.get("min_production_mt", 2100.0)),
            value=float(target_cfg.get("target_production_mt", 2270.0)),
            step=5.0,
            key="bmo_target_production_mt",
        )
        target_slag_rate_kg_per_thm = layout_col4.number_input(
            "Max Slag Rate (kg/THM)",
            min_value=0.0,
            value=float(model_input_defaults["target_slag_rate_kg_per_thm"]),
            step=5.0,
            key="bmo_target_slag_rate_kg_per_thm",
        )
    with st.expander("Slag chemistry window", expanded=False):
        st.caption("Set any limit to 0 to switch it off.")
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
        # Same 4-column grid as the slag window above so every input in the form
        # sits on one consistent line width.
        charge_col1, charge_col2, charge_col3, _charge_spacer = st.columns(4)
        max_charges_per_hour = charge_col1.number_input(
            "Max charges per hour",
            min_value=float(burden_capacity_cfg.get("min_charges_per_hour", 6.0)),
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
# A limit of 0 means "not enforced"; the solvers take None for that. The
# operator's slag rate is on the PLANT's basis; the solvers work in the model's,
# which reads high (services.bmo_case.slag_window).
slag_limits = case_service.slag_window(
    basicity_min=target_slag_basicity_min,
    basicity_max=target_slag_basicity_max,
    t_basicity_min=target_slag_t_basicity_min,
    t_basicity_max=target_slag_t_basicity_max,
    al2o3_max_pct=target_slag_al2o3_max_pct,
    mgo_min_pct=target_slag_mgo_min_pct,
    mgo_al2o3_ratio_min=target_slag_mgo_al2o3_ratio_min,
    slag_rate_kg_per_thm=target_slag_rate_kg_per_thm,
    target_production_mt=target_production_mt,
    model_to_plant_factor=model_to_plant_slag_factor,
)
basicity_bounds_valid = slag_limits.bounds_valid
if target_slag_basicity_min > target_slag_basicity_max:
    st.error("Min Basicity CaO/SiO2 must be less than or equal to Max Basicity.")
if target_slag_t_basicity_min > target_slag_t_basicity_max:
    st.error("Min T Basicity must be less than or equal to Max T Basicity.")
target_slag_t_basicity_min = slag_limits.t_basicity_min
target_slag_t_basicity_max = slag_limits.t_basicity_max
target_slag_al2o3_max_pct = slag_limits.al2o3_max_pct
target_slag_mgo_min_pct = slag_limits.mgo_min_pct
target_slag_mgo_al2o3_ratio_min = slag_limits.mgo_al2o3_ratio_min
target_slag_qty_mt = slag_limits.slag_qty_mt
if model_to_plant_slag_factor != 1.0:
    st.caption(
        f"Slag cap: {target_slag_rate_kg_per_thm:,.0f} kg/THM plant basis "
        f"= {target_slag_qty_mt:,.0f} MT model basis at "
        f"{target_production_mt:,.0f} MT HM "
        f"(model/plant factor {model_to_plant_slag_factor:.3f})."
    )
feo_in_slag_pct = float(bmo_cfg.get("chemistry", {}).get("feo_in_slag_pct", 0.4))
# The charged engine's blend response replaces the physics correction on
# Data-Driven: one linear slag term from the current burden, applied once.
coke_correction_settings = case_service.coke_correction(
    bmo_cfg, fuel_rate_anchor_basis, blend_response_kg_coke_per_kg_slag
)
# Resolved by the model switch directly below the page header.


ores, ore_diagnostics = _cached_build_ore_inputs(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"] = list(ore_diagnostics.get("warnings", []))

hm_snapshot = _cached_hm_slag_snapshot(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"].extend(hm_snapshot.get("warnings", []))
# The Si model's strongest input is the previous cast, so it always gets the
# single latest cast - even when the chemistry above is a window average.
_latest_cast_snapshot = (
    hm_snapshot
    if chemistry_mode == "latest"
    else _cached_hm_slag_snapshot(
        provider, "latest", chemistry_window_days, source_cache_version
    )
)
st.session_state["bmo_latest_cast"] = {
    "chem_pct_si": _latest_cast_snapshot.get("chem_pct_si"),
    "sample_timestamp": _latest_cast_snapshot.get("sample_timestamp"),
}
fuel_analysis, fuel_analysis_warnings = _cached_fuel_analysis_snapshot(
    provider, chemistry_mode, chemistry_window_days, source_cache_version
)
ore_diagnostics["warnings"].extend(fuel_analysis_warnings)
# Top-gas dust for the slag balance: DPR quantity and the plant's dust analyses.
_ds_cfg = bmo_cfg.get("data_sources", {})
dpr_dust_rates, _dpr_dust_warnings = _cached_dpr_dust_rates(
    provider,
    int(_ds_cfg.get("dust_window_days", 30)),
    int(_ds_cfg.get("dust_min_charges_per_day", 100)),
    source_cache_version,
)
ore_diagnostics["warnings"].extend(_dpr_dust_warnings)
# Dust chemistry is the configured values unless the config opts in to the
# dataset's DUST_1 / DUST_2 analyses (sparse: see setting_bmo.yml).
dust_analysis = (
    _dust_analysis_from_static_csv(
        static_path, static_mtime_ns, int(_ds_cfg.get("dust_analysis_window_days", 60))
    )
    if str(_ds_cfg.get("dust_chemistry_source", "configured")).lower() == "dataset"
    else {}
)
# Nut coke is a held set point, not an optimizer variable: the live/default rate
# is a base quantity and the Fuel Ash TM is added to obtain wet tonnes
# (services.bmo_case.nut_coke_charge).
nut_coke = case_service.nut_coke_charge(bmo_cfg, recent_fuel_rates, fuel_analysis)
nut_coke_rate_kg_per_thm = nut_coke.rate_kg_per_thm
nut_coke_charge_moisture_pct = nut_coke.charge_moisture_pct
observed_slag_rate = float(hm_snapshot.get("observed_slag_rate_kg_per_thm", 0.0) or 0.0)
target_fe_mt, hm_fe_pct_for_target, _fe_warning = case_service.target_fe(
    target_production_mt, hm_snapshot, bmo_cfg
)
if _fe_warning:
    ore_diagnostics["warnings"].append(_fe_warning)
st.caption(
    f"Optimiser Fe requirement: {target_fe_mt:,.1f} MT "
    f"from {target_production_mt:,.1f} MT HM at {hm_fe_pct_for_target:.2f}% Fe."
)

# Charging ceiling, rebuilt from the operator's own charge rate / charge mass
# rather than the yml snapshot. This is an absolute daily tonnage - the skips run
# 24 h whatever HM is targeted - so only the nut-coke deduction tracks the target.
max_burden_qty_mt, _capacity_warning = case_service.charging_ceiling(
    enabled=burden_capacity_enabled,
    max_charges_per_hour=max_charges_per_hour,
    charge_mass_mt=charge_mass_mt,
    target_production_mt=target_production_mt,
    nut_coke=nut_coke,
)
if _capacity_warning:
    st.warning(_capacity_warning)
if not burden_capacity_enabled:
    st.caption(
        "Charging capacity is not being enforced: the optimizer may plan more "
        "burden tonnes than the charging system can deliver."
    )

active_pellet_ids = None
if bool(ui_cfg.get("auto_select_active_pellet", True)):
    active_pellet_ids, pellet_usage_warnings = _cached_recent_active_pellet_ids(
        provider, source_cache_version
    )
    ore_diagnostics["warnings"].extend(pellet_usage_warnings)
editor_df = case_service.ore_editor_table(
    ores,
    case_service.default_ore_selection(ores, ui_cfg, active_pellet_ids),
    operator_preferences,
)

st.markdown("### Ore Selection, Stock, Pricing, Chemistry, and Share Bounds")
# The applied table wins while it holds the same ores (services.bmo_case.applied_or_base).
ore_editor_source_df = case_service.applied_or_base(
    "ore", st.session_state.get("bmo_applied_ore_editor_df"), editor_df
)

# Flux editor lives inside the ore-input form so its price/stock are applied and
# saved together with the ore inputs. Seed from config + saved preferences, or
# from the session-applied edits.
flux_base_df = case_service.flux_editor_table(bmo_cfg, operator_preferences)
flux_editor_source_df = case_service.applied_or_base(
    "flux", st.session_state.get("bmo_applied_flux_editor_df"), flux_base_df
)

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
                operator_preferences_path, edited_df, flux_df=edited_flux_df
            )
            st.success(f"Ore and flux inputs saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save ore inputs: {exc}")
    else:
        st.success("Ore inputs applied.")
else:
    edited_df = ore_editor_source_df
    edited_flux_df = flux_editor_source_df
flux_inputs = flux_inputs_from_editor(edited_flux_df)

_hm_chem_defaults = case_service.hm_chem_defaults(bmo_cfg, operator_preferences)
_hm_widget_keys = {
    "carbon_pct": "bmo_hm_carbon_pct",
    "silicon_pct": "bmo_hm_silicon_pct",
    "sulphur_pct": "bmo_hm_sulphur_pct",
    "other_pct": "bmo_hm_other_pct",
}
hm_chem_values = case_service.hm_chem_values(
    bmo_cfg,
    operator_preferences,
    {
        _hm_field: st.session_state[_hm_widget_key]
        for _hm_field, _hm_widget_key in _hm_widget_keys.items()
        if _hm_widget_key in st.session_state
    },
)

selected_ores = case_service.selected_ores_from_editor(edited_df, ores)
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


fuel_ash_base_df = case_service.fuel_ash_editor_table(
    bmo_cfg, recent_fuel_rates, fuel_analysis, operator_preferences
)
fuel_ash_editor_source_df = case_service.applied_or_base(
    "fuel_ash", st.session_state.get("bmo_applied_fuel_ash_editor_df"), fuel_ash_base_df
)

# The editable ash chemistry is rendered in the collapsed assumptions area at
# the end of the page. Calculations above it use the last applied/saved table.
# PCI and nut-coke rates remain authoritative in their dedicated controls, so
# they are pinned before this table reaches either slag or fuel-cost
# calculations, and the operator's fuel prices applied
# (services.bmo_case.fuel_ash_with_rates).
edited_fuel_ash_df = case_service.fuel_ash_with_rates(
    fuel_ash_editor_source_df,
    recent_fuel_rates,
    {
        fuel_id: st.session_state[f"bmo_fuel_price_{fuel_id}_rs_per_mt"]
        for fuel_id in ("coke", "nut_coke", "pci")
        if f"bmo_fuel_price_{fuel_id}_rs_per_mt" in st.session_state
    },
)
# Top-gas dust: quantity from the DPR (kg per charge over the window), chemistry
# from the plant's dust analyses in the hourly dataset, the configured rows as
# fallback; saved operator values override both until "Use DPR values". An
# applied (or snapshot-restored) table wins for the session, even when its rows
# differ from today's configuration, so an older snapshot replays as run.
dust_base_df = case_service.dust_editor_table(
    bmo_cfg, dpr_dust_rates, dust_analysis, operator_preferences
)
dust_editor_source_df = case_service.applied_or_base(
    "dust", st.session_state.get("bmo_applied_dust_editor_df"), dust_base_df
)

dust_inputs_applied = False
dust_inputs_saved = False
dust_inputs_reset = False
with st.expander("Advanced Slag Balance Inputs", expanded=False):
    slag_settings_values = render_slag_balance_settings(bmo_cfg.get("slag_balance", {}))
    if not dust_editor_source_df.empty:
        st.markdown("##### Top-gas dust (deducted from the slag)")
        with st.form("bmo_dust_input_form", clear_on_submit=False):
            edited_dust_candidate_df = render_dust_editor(dust_editor_source_df)
            # Filled after the buttons are handled, so it reflects a save or reset
            # made in this rerun.
            dust_status_slot = st.empty()
            st.caption(
                "Quantities come from the DPR (dust over the charges counted on "
                "the same days); chemistry is configured in setting_bmo.yml. "
                "Edit and Apply for this session; Save keeps your values for "
                "next time (they then override the DPR); Use DPR values drops "
                "saved values."
            )
            dust_apply_col, dust_save_col, dust_reset_col = st.columns(3)
            dust_inputs_applied = _form_submit_button(
                dust_apply_col,
                "Apply dust inputs",
                type="primary",
                width="stretch",
            )
            dust_inputs_saved = _form_submit_button(
                dust_save_col,
                "Save dust inputs for next time",
                type="secondary",
                width="stretch",
            )
            dust_inputs_reset = _form_submit_button(
                dust_reset_col,
                "Use DPR values",
                type="secondary",
                width="stretch",
            )
    else:
        edited_dust_candidate_df = dust_editor_source_df
        dust_status_slot = None
_dust_saved_ids = saved_dust_ids(operator_preferences)
if dust_inputs_reset:
    st.session_state.pop("bmo_applied_dust_editor_df", None)
    _clear_bmo_results()
    try:
        clear_dust_preferences(operator_preferences_path)
        _dust_saved_ids = []
        st.success("Saved dust inputs removed: using the DPR and the dust analyses.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not remove the saved dust inputs: {exc}")
    edited_dust_df = build_dust_editor_df(
        dust_cfg_with_live(bmo_cfg.get("dust_inputs", []), dpr_dust_rates, dust_analysis)
    )
elif dust_inputs_applied or dust_inputs_saved:
    edited_dust_df = edited_dust_candidate_df.copy()
    st.session_state["bmo_applied_dust_editor_df"] = edited_dust_df
    _clear_bmo_results()
    if dust_inputs_saved:
        try:
            saved_path = save_dust_preferences(
                operator_preferences_path, edited_dust_df
            )
            _dust_saved_ids = list(edited_dust_df["dust_id"].astype(str))
            st.success(f"Dust inputs saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save dust inputs: {exc}")
    else:
        st.success("Dust inputs applied.")
else:
    edited_dust_df = dust_editor_source_df
if dust_status_slot is not None:
    _dust_status = dust_caption(dpr_dust_rates, dust_analysis, _dust_saved_ids)
    if not dust_inputs_reset and isinstance(
        st.session_state.get("bmo_applied_dust_editor_df"), pd.DataFrame
    ):
        _dust_status += "  \n*The table holds values applied in this session.*"
    dust_status_slot.caption(_dust_status)
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
        "Top-gas dust is entered but "
        "'Use full slag balance' is unchecked - dust will NOT be deducted "
        "from the slag balance."
    )

_DE_SEED_LABELS = {
    "lp_else_random": "Balanced Optimizer seed, random fallback (recommended)",
    "lp": "Balanced Optimizer seed only",
    "random": "Random start",
}

# --- Fuel planning controls, immediately above the buttons that consume them --------
_live_pci_tag = float(_live_process_snapshot().get("coal_rate_actual_value") or 0.0)
if "bmo_nut_coke_override_kg" not in st.session_state:
    st.session_state["bmo_nut_coke_override_kg"] = DEFAULT_NUT_COKE_RATE_KG_PER_THM

fuel_prices_applied = False
fuel_prices_saved = False
with st.expander("Fuel rates and prices", expanded=False):
    _fuel_rate_col, _fuel_price_col = st.columns(
        [1.15, 0.85], gap="large", vertical_alignment="top"
    )

    with _fuel_rate_col:
        st.markdown("**Fuel rate overrides**")
        _pci_toggle_col, _pci_value_col = st.columns(
            [0.85, 1.4], vertical_alignment="center"
        )
        _pci_on = _pci_toggle_col.toggle(
            "PCI override",
            key="bmo_pci_override_on",
            help=(
                "Off: use the live plant tag. On: use the entered rate for the "
                "coke rate, slag balance and fuel cost."
            ),
        )
        if _pci_on:
            _pci_value_col.number_input(
                "PCI (kg/THM)",
                min_value=0.0,
                max_value=float(ANCHOR_HIGH_PCI),
                step=5.0,
                key="bmo_pci_override_kg",
                help=(
                    f"Outside {MODEL_PCI_MIN:g}-{MODEL_PCI_MAX:g} kg/THM, "
                    "the coke rate uses an operating-anchor interpolation."
                ),
            )
            _pci_now = float(st.session_state.get("bmo_pci_override_kg") or 0.0)
            if _live_pci_tag > 0.0:
                st.caption(
                    f"Live: {_live_pci_tag:,.1f}; delta: "
                    f"{_pci_now - _live_pci_tag:+,.1f} kg/THM."
                )
            if not MODEL_PCI_MIN <= _pci_now <= MODEL_PCI_MAX:
                st.warning(
                    f"Outside the modelled {MODEL_PCI_MIN:g}-"
                    f"{MODEL_PCI_MAX:g} kg/THM band."
                )
        else:
            _pci_value_col.metric("Live PCI", f"{_live_pci_tag:,.1f} kg/THM")

        st.divider()
        _nut_toggle_col, _nut_value_col = st.columns(
            [0.85, 1.4], vertical_alignment="center"
        )
        _nut_coke_on = _nut_toggle_col.toggle(
            "Nut-coke override",
            key="bmo_nut_coke_override_on",
            help=(
                "Off: use the fixed 70 kg/THM set point. On: use the entered "
                "rate for charging, slag, the selected coke-rate model and fuel "
                "cost."
            ),
        )
        if _nut_coke_on:
            _nut_value_col.number_input(
                "Nut coke (kg/THM)",
                min_value=0.0,
                step=5.0,
                key="bmo_nut_coke_override_kg",
            )
            _nut_coke_value = st.session_state.get("bmo_nut_coke_override_kg")
            _nut_coke_now = float(
                DEFAULT_NUT_COKE_RATE_KG_PER_THM
                if _nut_coke_value is None
                else _nut_coke_value
            )
            st.caption(
                f"Fixed: {DEFAULT_NUT_COKE_RATE_KG_PER_THM:,.1f}; delta: "
                f"{_nut_coke_now - DEFAULT_NUT_COKE_RATE_KG_PER_THM:+,.1f} "
                "kg/THM."
            )
        else:
            _nut_value_col.metric(
                "Fixed nut coke",
                f"{DEFAULT_NUT_COKE_RATE_KG_PER_THM:,.1f} kg/THM",
            )

    with _fuel_price_col:
        st.markdown("**Fuel prices**")
        with st.form("bmo_fuel_price_form", clear_on_submit=False):
            edited_fuel_price_candidate_df = render_fuel_price_inputs(
                edited_fuel_ash_df, key_prefix="bmo_"
            )
            _price_apply_col, _price_save_col = st.columns(2)
            fuel_prices_applied = _form_submit_button(
                _price_apply_col,
                "Apply",
                type="primary",
                width="stretch",
                key="bmo_ui_apply_fuel_prices",
                on_click=_clear_bmo_results,
            )
            fuel_prices_saved = _form_submit_button(
                _price_save_col,
                "Save",
                type="secondary",
                width="stretch",
                key="bmo_ui_save_fuel_prices",
                on_click=_clear_bmo_results,
                help="Save these prices as the defaults for future sessions.",
            )

edited_fuel_ash_df = edited_fuel_price_candidate_df.copy()
if fuel_prices_applied or fuel_prices_saved:
    if fuel_prices_saved:
        try:
            saved_path = save_fuel_price_preferences(
                operator_preferences_path, edited_fuel_ash_df
            )
            st.success(f"Fuel prices saved to {saved_path}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not save fuel prices: {exc}")
    else:
        st.success("Fuel prices applied.")

_fuel_price_state = tuple(
    (
        str(row.get("fuel_id", "")).strip().lower(),
        float(row.get("price_rs_per_mt", 0.0) or 0.0),
    )
    for _, row in edited_fuel_ash_df.iterrows()
    if (str(row.get("fuel_id", "")).strip().lower() in {"coke", "nut_coke", "pci"})
)
if st.session_state.get("bmo_ui_fuel_price_state_last") != _fuel_price_state:
    if st.session_state.get("bmo_ui_fuel_price_state_last") is not None:
        _clear_bmo_results()
    st.session_state["bmo_ui_fuel_price_state_last"] = _fuel_price_state

# Results computed against different rates are stale, so clear them on a change.
_fuel_rate_state = (
    bool(_pci_on),
    st.session_state.get("bmo_pci_override_kg"),
    bool(_nut_coke_on),
    st.session_state.get("bmo_nut_coke_override_kg"),
)
if st.session_state.get("bmo_pci_state_last") != _fuel_rate_state:
    if st.session_state.get("bmo_pci_state_last") is not None:
        _clear_bmo_results()
    st.session_state["bmo_pci_state_last"] = _fuel_rate_state

fuel_ash_inputs = fuel_ash_inputs_from_editor(edited_fuel_ash_df)
# One run's inputs, as the service layer takes them (services.bmo.BlendCase):
# the LP, DE, the comparison and the transition path all read this one object.
bmo_case = bmo_service.BlendCase(
    ores=selected_ores,
    target_fe_mt=target_fe_mt,
    target_production_mt=target_production_mt,
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
    charge_mass_mt=charge_mass_mt,
    coke_correction_settings=coke_correction_settings,
    observed_slag_rate_kg_per_thm=observed_slag_rate,
    live_si_pct=live_si_pct,
    fuel_rate_anchor_basis=fuel_rate_anchor_basis,
    lookback_hours=manual_blend_lookback_hours,
)
data_driven_prediction: ChargedCokePrediction | None = None
_charged_panel_payload: dict[str, Any] | None = None
physics_prediction: Any | None = None
if not use_charged_engine:
    try:
        with st.spinner("Calculating the Physics-Driven coke rate..."):
            physics_prediction = _resolve_energy_anchor(
                basis=fuel_rate_anchor_basis,
                provider=provider,
                ores=selected_ores,
                fuel_ash_inputs=fuel_ash_inputs,
                flux_inputs=flux_inputs,
                dust_inputs=dust_inputs,
                slag_balance_settings=slag_balance_settings,
                hm_chem_values=hm_chem_values,
                hm_snapshot=hm_snapshot,
                hot_metal_mt=target_production_mt,
                target_fe_mt=target_fe_mt,
                charge_mass_mt=charge_mass_mt,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
                lookback_hours=manual_blend_lookback_hours,
            )
    except Exception as exc:  # noqa: BLE001 - keep the optimiser page available
        log.exception("Physics-Driven coke prediction failed")
        st.error(f"The Physics-Driven coke rate could not be calculated: {exc}")
    st.session_state["bmo_energy_anchor"] = physics_prediction
# Always obtain the charged-coke forecast as an advisory production signal (read
# from the worker's ledger, or issued here when no worker issues it). The
# switch above still decides which coke level the optimiser uses; Physics-Driven
# therefore remains fully physics-anchored while operators can see and snapshot
# the independent Data-Driven forecast and its accuracy.
try:
    with st.spinner("Charged-coke forecast for the latest complete hour..."):
        data_driven_prediction, _charged_detail = _charged_prediction_for_page(
            pci_override=(
                float(recent_fuel_rates.get("pci_rate_kg_thm", 0.0) or 0.0)
                if st.session_state.get("bmo_pci_override_on")
                else None
            ),
            nut_override=(
                float(recent_fuel_rates.get("nut_coke_rate_kg_thm", 0.0) or 0.0)
                if st.session_state.get("bmo_nut_coke_override_on")
                else None
            ),
            slag_response_coefficient=blend_response_kg_coke_per_kg_slag,
            force_predict=force_data_driven_prediction,
        )
except Exception as exc:  # noqa: BLE001 - keep Physics-Driven available
    log.exception("Charged-coke forecast failed")
    st.error(f"The charged-coke forecast could not run: {exc}")
else:
    st.session_state["bmo_data_driven_coke_prediction"] = data_driven_prediction.to_dict()
    st.session_state["bmo_charged_coke_record"] = _charged_detail.get("record")
    _record = _charged_detail.get("record") or {}
    _dataset_path = _charged_dataset_path()
    _bundle = str(active_charged_bundle(_charged_storage(), _charged_default_bundle()))
    _recent = None
    if _record.get("data_row"):
        try:
            _recent = _charged_recent(
                str(_dataset_path),
                int(_dataset_path.stat().st_mtime_ns) if _dataset_path.is_file() else 0,
                _bundle,
                str(_record["data_row"]),
            )
        except Exception:  # noqa: BLE001 - the trust chart must not block the page
            log.exception("Could not replay the last 7 days")
    _charged_panel_payload = {
        "prediction": data_driven_prediction,
        "detail": _charged_detail,
        "recent": _recent,
        "retrain": {
            "storage_dir": _charged_storage(),
            "dataset_path": _dataset_path,
            "base_bundle": _charged_default_bundle(),
            "data_end": pd.Timestamp(_record["data_row"]) if _record.get("data_row") else None,
            "can_manage": bool(is_admin() or is_supervisor()),
            "user": str(st.session_state.get("auth_user") or "unknown"),
            "on_change": _clear_bmo_results,
        },
    }
run_lp_clicked = False
run_total_clicked = False
# The SPA's /bmo page links here for the Intensive Optimizer (DE search): /legacy/Blend_Optimizer#de-search.
st.subheader("Balanced and Intensive Optimizer", anchor=DE_SEARCH_ANCHOR)
with st.form("bmo_run_form", clear_on_submit=False):
    seed_options = list(_DE_SEED_LABELS)
    configured_seed = str(opt_cfg.get("initial_solution", "lp_else_random")).lower()
    de_seed_choice = st.selectbox(
        "Intensive Optimizer start point",
        options=seed_options,
        index=(
            seed_options.index(configured_seed)
            if configured_seed in seed_options
            else 0
        ),
        format_func=lambda key: _DE_SEED_LABELS[key],
    )
    run_col1, run_col2 = st.columns(2)
    run_lp_clicked = _form_submit_button(
        run_col1,
        "Run Balanced Optimizer",
        type="secondary",
        width="stretch",
    )
    run_total_clicked = _form_submit_button(
        run_col2,
        "Run Intensive Optimizer",
        type="primary",
        width="stretch",
    )

requested_lp = bool(run_lp_clicked)
requested_total = bool(run_total_clicked)

if requested_lp or requested_total:
    # Refusals, the coke anchor, the correction reference and the LP plan come
    # from services.bmo; the page keeps the spinners and the session state.
    # The LP step (refusal, correction reference, anchor, LP) is services.bmo_case.plan_lp,
    # shared with the twin-api's runs; the page keeps the spinners and the session state.
    # The reference and the anchor are resolved once and reused by DE, so both solvers
    # optimise one identical objective. The anchor is the LEVEL the whole fuel cost sits
    # on; blend sensitivity is the correction's job, not the anchor's.
    with st.spinner("Running Balanced Optimizer..."):
        lp_plan = case_service.plan_lp(
            bmo_case,
            provider=provider,
            data_driven_prediction=data_driven_prediction,
            energy_anchor=physics_prediction,
            basicity_bounds_valid=basicity_bounds_valid,
            pellet_blocked=bool(pellet_input_issues and not pellet_input_confirmed),
            previous_manual_quantities_mt=st.session_state.get("bmo_manual_quantities_mt"),
            load_fuel_context=lambda: _load_fuel_prediction_context(provider),
        )
    if lp_plan.refusal:
        st.error(lp_plan.refusal)
    else:
        fuel_context = None
        coke_correction_reference = lp_plan.reference
        energy_anchor = physics_prediction
        st.session_state["bmo_energy_anchor"] = energy_anchor
        coke_anchor = lp_plan.anchor
        anchor_coke_rate = coke_anchor.rate_kg_thm
        st.session_state["bmo_coke_anchor_basis"] = fuel_rate_anchor_basis
        st.session_state["bmo_coke_anchor_rate_kg_thm"] = anchor_coke_rate
        balanced_run = lp_plan.balanced
        lp_result, lp_errors = balanced_run.lp_result, balanced_run.lp_errors
        if balanced_run.fuel_context is not None:
            fuel_context = balanced_run.fuel_context
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
        # LP always runs in this rerun and shares DE's fuel-prediction context, so
        # always persist it. This keeps the LP tab and the LP-vs-DE comparison on
        # the SAME live snapshot DE used, avoiding a stale LP (from an earlier,
        # separate click) being shown next to a fresh DE.
        st.session_state["bmo_lp_result"] = lp_result
        st.session_state["bmo_lp_errors"] = lp_errors

        if requested_total:
            de_status = st.status("Intensive Optimizer running...", expanded=True)
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
            de_result, de_errors = bmo_service.run_intensive(
                bmo_case,
                fuel_context=fuel_context,
                reference=coke_correction_reference,
                anchor=coke_anchor,
                # The selectbox overrides the configured default for this run.
                de_cfg={**opt_cfg, "initial_solution": de_seed_choice},
                progress_callback=_de_progress,
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
                    f"Intensive Optimizer finished - {final_iter} generations, "
                    f"{final_nfev:,} blend evaluations"
                    if final_iter
                    else "Intensive Optimizer finished"
                ),
                state="complete",
            )

            # Guardrail: DE can never report a "worse optimum" than the LP
            # baseline of the same rerun (services.bmo.fall_back_to_lp).
            de_result = bmo_service.fall_back_to_lp(lp_result, de_result)
            st.session_state["bmo_de_result"] = de_result
            st.session_state["bmo_de_errors"] = de_errors


lp_result = st.session_state.get("bmo_lp_result")
lp_errors = st.session_state.get("bmo_lp_errors", [])
de_result = st.session_state.get("bmo_de_result")
de_errors = st.session_state.get("bmo_de_errors", [])

# THE RESULTS RENDER ON EVERY RERUN, THE OPTIMISER RUNS ON ALMOST NONE.
#
# history_df, process_context and bundle_status are bound inside the run block,
# which only executes when a Run button was clicked. But the result tabs below
# render from session state on ANY rerun - changing the transition step size,
# toggling the PCI override, opening a tab. On those reruns the names simply did
# not exist and the page died with "NameError: name 'history_df' is not
# defined", which is exactly what an operator hit when they moved the step size
# to 1%.
#
# Rebinding them here costs nothing: _load_fuel_prediction_context is cached, so
# on a rerun it returns the same objects the run block already built.
if lp_result is not None or de_result is not None:
    try:
        (
            model_service,
            process_context,
            history_df,
            bundle_status,
            _rerun_warnings,
        ) = _load_fuel_prediction_context(provider)
    except Exception as exc:  # noqa: BLE001 - never let this take the page down
        log.warning("Could not reload fuel context for result rendering: %s", exc)
        history_df = locals().get("history_df")
        process_context = locals().get("process_context")
        bundle_status = locals().get("bundle_status") or {}

if lp_errors:
    st.error("Balanced Optimizer errors:\n- " + "\n- ".join(lp_errors))
if de_errors:
    st.error("Intensive Optimizer errors:\n- " + "\n- ".join(de_errors))

if lp_result is not None or de_result is not None:
    # What the furnace produced over the last complete shift: the baseline for
    # every "change vs last shift" figure in the result cards and Comparison.
    try:
        shift_baseline = _last_shift_production(static_path, static_mtime_ns)
    except Exception:  # noqa: BLE001 - a missing baseline hides one tile, no more
        log.exception("Could not compute last-shift production")
        shift_baseline = {}
    tab_lp, tab_de, tab_cmp, tab_acc = st.tabs(
        [
            "Balanced Optimizer",
            "Intensive Optimizer",
            "Comparison",
            "Model accuracy",
        ]
    )

    with tab_lp:
        if lp_result is not None:
            # The headline stays above the sub-tabs: cost, slag rate and
            # feasibility are what the operator checks first, and burying them
            # one click deep would mean every tab needs its own copy.
            render_blend_metrics(
                "Balanced Optimizer Result",
                lp_result,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
                is_lp_mode=True,
                charge_mass_mt=charge_mass_mt,
                shift_baseline=shift_baseline,
                si_forecast=live_si_result_tile,
            )
            # Controls and Path are ONE tab, not two.
            #
            # They were split apart in 1d0ac2b and that was a mistake: a blend
            # recommendation is not actionable without the blast settings that
            # supply what it demands, and an operator should not have to click
            # between them to see both. The order below is the order the
            # decision is made in - what settings does this blend need, then
            # how do I get the burden there from where it is today.
            lp_blend_tab, lp_fuel_tab, lp_slag_tab, lp_path_tab = st.tabs(
                ["🧱 Blend", "🔥 Fuel & coke", "🌋 Slag", "🎛️ Path & controls"]
            )

            with lp_blend_tab:
                table_col, donut_col = st.columns([3, 2], gap="medium")
                with table_col:
                    render_blend_table(
                        lp_result, selected_ores, charge_mass_mt=charge_mass_mt
                    )
                with donut_col:
                    _render_share_pie(
                        lp_result, selected_ores, "Balanced Optimizer share of burden"
                    )
                _render_lp_flux_additions(lp_result)

            with lp_fuel_tab:
                render_coke_correction_breakdown(lp_result)
                _render_fuel_basis_note(lp_result)

            with lp_slag_tab:
                render_slag_balance_details(
                    lp_result, selected_ores, fuel_ash_inputs, flux_inputs
                )

            with lp_path_tab:
                # WHAT SETTINGS THIS BLEND NEEDS, first. recommend_controls
                # solves the blast temperature, oxygen and volume that supply
                # this blend's energy demand at least fuel cost - so this single
                # panel answers both halves of the question: what the blend
                # requires, and what the cheapest way of supplying it is.
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
                _render_energy_assumptions()
                st.divider()

                # THEN how to get the burden there from where it is today.
                st.markdown("##### Getting there from the current blend")
                _render_transition_ladder(provider=provider, case=bmo_case)
        else:
            st.info("Run the Balanced Optimizer to see its recommended blend.")

    with tab_de:
        if de_result is not None:
            if de_result.diagnostics.get("de_fell_back_to_lp"):
                st.info(
                    "The Intensive Optimizer did not improve on the Balanced "
                    "Optimizer within its iteration/time budget. Showing the "
                    "Balanced Optimizer blend as the best available solution."
                )
            _de_seed = de_result.diagnostics.get("de_seed") or {}
            if _de_seed.get("strategy_used") == "random":
                # Without this the operator cannot tell an unguided search from an
                # LP-seeded one, and the LP's reasons for failing would be lost.
                _lp_reasons = _de_seed.get("lp_seed_errors") or []
                st.warning(
                    "This result came from a **random start**, not the Balanced "
                    "Optimizer seed"
                    + (
                        " - the Balanced Optimizer was infeasible."
                        if not _de_seed.get("lp_seed_available")
                        else "."
                    )
                    + " It is the best blend the search reached; check the "
                    "constraint violations below before acting on it."
                )
                if _lp_reasons:
                    with st.expander(
                        "Why the Balanced Optimizer could not solve", expanded=False
                    ):
                        for _reason in _lp_reasons:
                            st.markdown(f"- {_reason}")
            render_blend_metrics(
                "Intensive Optimizer Result",
                de_result,
                observed_slag_rate_kg_per_thm=observed_slag_rate,
                charge_mass_mt=charge_mass_mt,
                shift_baseline=shift_baseline,
                si_forecast=live_si_result_tile,
            )
            de_blend_tab, de_fuel_tab, de_slag_tab, de_search_tab = st.tabs(
                ["🧱 Blend", "🔥 Fuel & coke", "🌋 Slag", "🔎 Search"]
            )

            with de_blend_tab:
                table_col, donut_col = st.columns([3, 2], gap="medium")
                with table_col:
                    render_blend_table(
                        de_result, selected_ores, charge_mass_mt=charge_mass_mt
                    )
                with donut_col:
                    _render_share_pie(
                        de_result,
                        selected_ores,
                        "Intensive Optimizer share of burden",
                    )
                _render_lp_flux_additions(de_result)

            with de_fuel_tab:
                render_coke_correction_breakdown(de_result)
                _render_fuel_basis_note(de_result)
                if not bundle_status.get("model_loaded"):
                    st.info(
                        "Model artifact is missing. Fuel term is currently coming "
                        "from fallback formula until model/scaler bundle is provided."
                    )

            with de_slag_tab:
                render_slag_balance_details(
                    de_result, selected_ores, fuel_ash_inputs, flux_inputs
                )

            with de_search_tab:
                _render_de_exploration(
                    st.session_state.get("bmo_de_candidates"), selected_ores
                )
        else:
            st.info("Run the Intensive Optimizer to see its recommended blend.")

    with tab_cmp:
        # Operator-focused comparison: the manual blend against each optimizer
        # blend (LP + DE) shown side by side -- inputs (blend mix) and outputs
        # (cost, fuel rate, basicity, slag rate) on the same target-Fe basis.
        optimizer_candidates: list[tuple[str, Any, float | None]] = []
        if lp_result is not None:
            optimizer_candidates.append(
                ("Balanced Optimizer", lp_result, live_si_pct)
            )
        if de_result is not None:
            optimizer_candidates.append(
                ("Intensive Optimizer", de_result, live_si_pct)
            )
        if optimizer_candidates:
            _render_blend_comparison(
                provider,
                case=bmo_case,
                optimizer_candidates=optimizer_candidates,
                shift_baseline=shift_baseline,
            )
        else:
            st.info(
                "Run either optimizer to compare its suggested blend with the "
                "current manual blend."
            )

    with tab_acc:
        render_model_accuracy_tab()

    if si_furnace_forecast is not None:
        st.divider()
        _render_live_si_furnace_panel(
            bmo_cfg, source_cache_version=source_cache_version
        )

    # The operating trend belongs after the recommendations it supports and
    # immediately before the commentary that interprets those recommendations.
    if not use_charged_engine:
        st.divider()
        _render_physics_coke_prediction(physics_prediction)
    if _charged_panel_payload is not None:
        st.divider()
        render_charged_panel(**_charged_panel_payload)

    st.divider()
    render_furnace_commentary(
        live_snapshot=_live_process_snapshot(),
        ores=selected_ores,
        lp_blend=lp_result,
        de_blend=de_result,
        manual_blend=st.session_state.get("bmo_manual_blend"),
        calibration=load_coke_calibration(),
        energy_anchor=st.session_state.get("bmo_energy_anchor"),
        production_target_mt=target_production_mt,
    )

elif si_furnace_forecast is not None:
    st.divider()
    _render_live_si_furnace_panel(
        bmo_cfg, source_cache_version=source_cache_version
    )
    if not use_charged_engine:
        st.divider()
        _render_physics_coke_prediction(physics_prediction)
    if _charged_panel_payload is not None:
        st.divider()
        render_charged_panel(**_charged_panel_payload)
elif _charged_panel_payload is not None or not use_charged_engine:
    # Before the first optimiser run there are no results or commentary yet;
    # keep both independent coke estimates available below the run controls.
    if not use_charged_engine:
        st.divider()
        _render_physics_coke_prediction(physics_prediction)
    if _charged_panel_payload is not None:
        st.divider()
        render_charged_panel(**_charged_panel_payload)

st.markdown("### Diagnostics and assumptions")
with st.expander("Hot Metal Chemistry Assumptions", expanded=False):
    with st.form("bmo_assumption_input_form", clear_on_submit=False):
        rendered_hm_chem_values = render_hot_metal_chemistry(
            hm_snapshot, _hm_chem_defaults
        )
        hm_apply_col, hm_save_col = st.columns(2)
        assumptions_applied = _form_submit_button(
            hm_apply_col,
            "Apply Assumptions",
            type="primary",
            width="stretch",
            key="bmo_ui_apply_hm_assumptions",
            on_click=_clear_bmo_results,
        )
        assumptions_saved = _form_submit_button(
            hm_save_col,
            "Save for Later",
            type="secondary",
            width="stretch",
            key="bmo_ui_save_hm_assumptions",
            on_click=_clear_bmo_results,
        )
if assumptions_applied or assumptions_saved:
    hm_chem_values = rendered_hm_chem_values
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

_fuel_ash_flash = st.session_state.pop("bmo_fuel_ash_flash", None)
if isinstance(_fuel_ash_flash, tuple) and len(_fuel_ash_flash) == 2:
    _flash_level, _flash_message = _fuel_ash_flash
    if _flash_level == "error":
        st.error(str(_flash_message))
    else:
        st.success(str(_flash_message))

fuel_ash_inputs_applied = False
fuel_ash_inputs_saved = False
with st.expander("Fuel Ash Inputs", expanded=False):
    st.caption(
        "Edit fuel moisture and ash chemistry used by the slag balance. Fuel "
        "prices remain in **Fuel rates and prices** above and are intentionally "
        "not repeated here."
    )
    _render_pinned_fuel_rates(edited_fuel_ash_df, recent_fuel_rates)
    st.caption(
        "Rates are shown for calculation context but are controlled by the "
        "dedicated PCI and nut-coke controls above."
    )
    with st.form("bmo_fuel_ash_input_form", clear_on_submit=False):
        if not edited_fuel_ash_df.empty:
            edited_fuel_ash_candidate_df = render_fuel_ash_editor(
                edited_fuel_ash_df,
                show_prices=False,
                rate_editable=False,
            )
        else:
            edited_fuel_ash_candidate_df = edited_fuel_ash_df
        st.caption(
            "Fuel analysis uses Moisture from fuel_chemistry (TM for coke/nut "
            "coke; IM for PCI). Ash analysis uses VM from fuel_chemistry. "
            "Moisture is removed once from wet fuel; VM is not moisture."
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
    st.session_state["bmo_applied_fuel_ash_editor_df"] = (
        edited_fuel_ash_candidate_df.copy()
    )
    _clear_bmo_results()
    _flash_level = "success"
    _flash_message = "Fuel Ash inputs applied."
    if fuel_ash_inputs_saved:
        try:
            saved_path = save_fuel_ash_preferences(
                operator_preferences_path, edited_fuel_ash_candidate_df
            )
            _flash_message = f"Fuel Ash inputs saved to {saved_path}."
        except Exception as exc:  # noqa: BLE001
            _flash_level = "error"
            _flash_message = f"Could not save Fuel Ash inputs: {exc}"
    st.session_state["bmo_fuel_ash_flash"] = (_flash_level, _flash_message)
    # The panel sits after all calculations. Re-run immediately so Apply/Save
    # recalculates the page from the new ash chemistry instead of leaving the
    # already-rendered results above it stale for one interaction.
    st.rerun()

_render_data_diagnostics(
    provider,
    ore_diagnostics=ore_diagnostics,
    hm_snapshot=hm_snapshot,
    edited_ore_df=edited_df,
)
render_diagnostics(de_result or lp_result, ore_diagnostics)

# Snapshots. Inside TestBMO this prefix arrives as "testbmo_" (utils/bmo/sandbox.py).
from ui.bmo.snapshot_panel import render_snapshot_panel  # noqa: E402

render_snapshot_panel(prefix="bmo_", page_vars=globals())

# SBFE is intentionally a separate product surface after commentary,
# diagnostics, and snapshots. Its controls and results do not mutate the main
# BMO run state.
from ui.bmo.production_frontier import render_production_frontier  # noqa: E402

render_production_frontier(
    # The whole yard, with the operator's edits to price, bounds and chemistry;
    # the main selection only decides which materials start ticked.
    ores=case_service.selected_ores_from_editor(edited_df.assign(selected=True), ores),
    selected_ore_ids={ore.ore_id for ore in selected_ores},
    slag_limits={
        "max_slag_rate_kg_per_thm": target_slag_rate_kg_per_thm,
        "basicity_min": target_slag_basicity_min,
        "basicity_max": target_slag_basicity_max,
        "t_basicity_min": target_slag_t_basicity_min,
        "t_basicity_max": target_slag_t_basicity_max,
        "al2o3_max_pct": target_slag_al2o3_max_pct,
        "mgo_min_pct": target_slag_mgo_min_pct,
        "mgo_al2o3_min": target_slag_mgo_al2o3_ratio_min,
    },
    fuel_ash_inputs=fuel_ash_inputs,
    flux_inputs=flux_inputs,
    dust_inputs=dust_inputs,
    slag_balance_settings=slag_balance_settings,
    hm_fe_pct=hm_fe_pct_for_target,
    feo_in_slag_pct=feo_in_slag_pct,
    model_to_plant_slag_factor=model_to_plant_slag_factor,
    default_max_charges_per_hour=max_charges_per_hour,
    default_charge_mass_mt=charge_mass_mt,
    default_nut_coke_rate_kg_per_thm=nut_coke_rate_kg_per_thm,
    default_nut_coke_moisture_pct=nut_coke_charge_moisture_pct,
)
