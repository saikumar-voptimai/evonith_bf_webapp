import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv

from config.config_loader import load_config
from furnace_data.influx.online import fetch_online_df
from furnace_data.neon_db.offline import (
    NEON_OFFLINE_REPORT_MAP as OFFLINE_DATABASE_REPORT_MAP,
    NEON_OFFLINE_TABLES as OFFLINE_DATABASE_TABLES,
    fetch_offline_data as fetch_table_data,
    fetch_offline_report as fetch_database_report,
)
from furnace_data.dataset.fetcher import DatasetFetcher as MlDatasetFetcher
from data.fetch_presets import OFFLINE_REPORT_LABEL_MAP, offline_table_label
from data.ml.static_csv import get_static_dataset_path, load_static_dataset
from data.ml.static_dataset_manager import StaticDatasetManager
from utils.dataset_refresher import maybe_refresh

config = load_config("setting_ds_dv.yml")  # Load the configuration file
config_vsense = load_config("setting_vsense.yml")

load_dotenv()

local_tz = pytz.timezone("Asia/Kolkata")  # or use your actual timezone
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

if maybe_refresh(config):
    st.sidebar.caption("Refreshing static dataset cache...")

fullpath = get_static_dataset_path(config["DATA"])
try:
    df = load_static_dataset(fullpath)
except Exception as exc:  # noqa: BLE001
    st.error(f"Static ML dataset could not be loaded: {exc}")
    st.stop()

if df.empty:
    st.warning("Static ML dataset returned no rows.")
    st.stop()


TIME_OPTIONS = {
    "last 1 minute": timedelta(minutes=1),
    "last 5 minutes": timedelta(minutes=5),
    "last 15 minutes": timedelta(minutes=10),
    "last 30 minutes": timedelta(minutes=30),
    "last 1 hour": timedelta(hours=1),
    "last 6 hours": timedelta(hours=6),
    "last 12 hours": timedelta(hours=12),
    "last 1 day": timedelta(days=1),
    "last 3 days": timedelta(days=3),
    "last 1 week": timedelta(weeks=1),
    "last 2 weeks": timedelta(weeks=2),
    "last 1 month": timedelta(days=30),
    "last 2 months": timedelta(days=60),
    "last 3 months": timedelta(days=90),
}

FREQUENCY_TO_TIMEDTA = {
    "None": None,
    "1 minute": "1min",
    "2 minute": "2min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "30 minutes": "30min",
    "1 hour": "1h",
    "6 hours": "6h",
    "12 hours": "12h",
    "1 day": "1d",
}

MEASUREMENT_LABELS = {
    "cooling_water": "Cooling Water",
    "delta_t": "Delta T",
    "heatload_delta_t": "Heatload Delta T",
    "miscellaneous": "Miscellaneous",
    "process_params": "Process Params",
    "temperature_profile": "Temperature Profile",
}

st.title("Visualisation tool")

# --------------------------------------------------------------------------
st.subheader("Distribution plots")

cols = st.columns([0.2, 0.2, 0.3, 0.3])
with cols[0]:
    from_date = st.date_input("From Date", value=pd.to_datetime(df.index[0]).date())

with cols[1]:
    to_date = st.date_input("To Date", value=pd.to_datetime(df.index[-1]).date())

with cols[2]:
    model_keys = list(config_vsense["Optimisation"].keys())
    control_params = config_vsense["Optimisation"][model_keys[1]]["control_params"]
    input_params = [
        item
        for sublist in config_vsense["Optimisation"][model_keys[1]][
            "input_params"
        ].values()
        for item in sublist
    ]
    all_params = control_params + input_params
    x_p = st.selectbox("Select X feature", all_params)

with cols[3]:
    output_params = [
        config_vsense["Optimisation"][key]["output_param"]
        for key in config_vsense["Optimisation"]
    ]
    y_p = st.selectbox(
        "Select Y feature", output_params + ["Unit Cost"] + control_params
    )

with st.sidebar:
    st.subheader("PCI/Coke cost")
    factor = st.number_input(
        "PCI/Coke Cost ratio", value=13250 / 25000, step=0.01, format="%.2f"
    )

df["UNITCOST LAKHS/THM"] = (
    df["COKE RATE KG/THM"] + factor * df["PCI_KG/THM"]
) * 0.25
df_filt = df[
    (pd.to_datetime(df.index, format="%d/%m/%Y %H:%M").date >= from_date)
    & (pd.to_datetime(df.index, format="%d/%m/%Y %H:%M").date <= to_date)
]

# UI layout
cols = st.columns([0.3, 0.2, 0.2, 0.2, 0.1])
with cols[0]:
    filter_feature = st.selectbox(
        "Filter by", df.columns, index=int(len(df.columns) - 3)
    )
with cols[1]:
    min_value = st.number_input(
        "Lower bound", value=float(df[filter_feature].quantile(0.25))
    )
with cols[2]:
    max_value = st.number_input(
        "Higher bound", value=float(df[filter_feature].quantile(0.75))
    )
with cols[3]:
    order = st.selectbox("Poly order", [1, 2, 3])
with cols[4]:
    inverse_filter = st.checkbox("Invert", value=True)

# Range filter
if inverse_filter:
    df_filt = df_filt[
        (df_filt[filter_feature] < min_value) | (df_filt[filter_feature] > max_value)
    ]
else:
    df_filt = df_filt[
        (df_filt[filter_feature] >= min_value) & (df_filt[filter_feature] <= max_value)
    ]
df_filt["Type"] = df_filt[filter_feature].apply(
    lambda x: "Low" if x < min_value else "High"
)

# Scatter and fit
if df_filt.empty:
    st.warning("No rows are available for the selected scatter plot.")
else:
    coeffs = np.polyfit(df_filt[x_p], df_filt[y_p], deg=order)
    eqn_str = None
    for i in range(len(coeffs)):
        if i == 0:
            eqn_str = f"{coeffs[len(coeffs) - 1]:.3g}"
        else:
            coeff_i = coeffs[len(coeffs) - (i + 1)]
            if i == 1:
                eqn_str += (
                    f" + {coeff_i:.3g}x"
                    if coeff_i >= 0
                    else f" - {abs(coeff_i):.3g}x"
                )
            else:
                eqn_str += (
                    f" + {coeff_i:.3g}x^{i}"
                    if coeff_i >= 0
                    else f" - {abs(coeff_i):.3g}x^{i}"
                )

    graph = sns.lmplot(
        x=x_p,
        y=y_p,
        markers="x",
        hue="Type",
        scatter_kws={"s": 8, "marker": "x"},
        data=df_filt,
        fit_reg=False,
    )

    ax = graph.axes[0, 0]
    plot = graph.axes[0, 0]
    sns.regplot(data=df_filt, x=x_p, y=y_p, order=order, scatter=False, ax=ax)

    if eqn_str:
        plt.text(
            0.05,
            0.95,
            f"y = {eqn_str}",
            transform=plot.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

    # Resize and render with Streamlit
    fig = plot.get_figure()
    fig.set_size_inches(10, 5)
    st.pyplot(fig, width="content")

# --------------------------------------------------------------------------
st.subheader("Timeseries plot")
st.sidebar.subheader("TS Plot - Select the date range.")
with st.sidebar:
    st.subheader("Timeseries Plot Options")
    cols = st.columns(2)
    with cols[0]:
        from_date = st.date_input(
            "From Date", value=pd.to_datetime(df.index[0]).date(), key="from_date2"
        )
    with cols[1]:
        to_date = st.date_input(
            "To Date", value=pd.to_datetime(df.index[-1]).date(), key="to_date2"
        )

_col_search = st.text_input(
    "Search columns",
    placeholder="Type to filter the column list…",
    key="de_ts_col_search",
)
_all_cols = list(df.columns)
_filtered_cols = (
    [c for c in _all_cols if _col_search.strip().lower() in c.lower()]
    if _col_search.strip()
    else _all_cols
)
features = st.multiselect(
    f"Select features ({len(_filtered_cols)} shown)",
    options=_filtered_cols,
    default=[_filtered_cols[0]] if _filtered_cols else [],
)

df_t = df[
    (pd.to_datetime(df.index, format="%d-%m-%Y %H:%M").date >= from_date)
    & (pd.to_datetime(df.index, format="%d-%m-%Y %H:%M").date <= to_date)
]
cols = st.columns([0.3, 0.15, 0.15, 0.1, 0.3])
with cols[0]:
    filter_feature = st.selectbox(
        "Filter by", _filtered_cols or _all_cols,
        index=min(max(len((_filtered_cols or _all_cols)) - 3, 0), len((_filtered_cols or _all_cols)) - 1),
        key="de_ts_filter_feature",
    )
with cols[1]:
    average_range = st.selectbox(
        "Avg Window:",
        list(FREQUENCY_TO_TIMEDTA.keys()),
        index=list(FREQUENCY_TO_TIMEDTA.keys()).index("None"),
    )
with cols[2]:
    min_value = st.number_input("Min", value=float(df_t[filter_feature].min()))
with cols[3]:
    max_value = st.number_input("Max", value=float(df_t[filter_feature].max()))
with cols[4]:
    inverse_filter = st.checkbox("Invert")

df_filtered = df_t[
    (df_t[filter_feature] >= min_value) & (df_t[filter_feature] <= max_value)
]

if FREQUENCY_TO_TIMEDTA[average_range] is not None:
    df_avg = df_filtered.resample(FREQUENCY_TO_TIMEDTA[average_range]).mean(numeric_only=True)
else:
    df_avg = df_filtered.copy()

fig = go.Figure()
colors = ["blue", "red", "green", "orange", "purple"]

for i, feature in enumerate(features):
    axis_name = f"y{i+1}" if i != 0 else "y"
    trace = go.Scatter(
        x=df_avg.index,
        y=df_avg[feature],
        name=feature,
        mode="lines+markers",
        yaxis=axis_name,
        line=dict(color=colors[i % len(colors)]),
    )
    fig.add_trace(trace)

    # Axis layout
    axis_layout = dict(
        title="",
        showgrid=False,
        showticklabels=False,
    )

    if i == 0:
        axis_layout["side"] = "left"
        fig.update_layout(yaxis=axis_layout)
    else:
        axis_layout.update(
            {
                "overlaying": "y",
                "side": "right" if i % 2 else "left",
                "anchor": "x",
                "position": 1.0 - (i * 0.05) if i % 2 else 0.05 + (i * 0.05),
            }
        )
        fig.update_layout({f"yaxis{i+1}": axis_layout})

# Final layout
fig.update_layout(
    xaxis=dict(title="Date"),
    height=600,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(yanchor="top", y=1.15, xanchor="left", x=0.01),
    margin=dict(l=40, r=40, t=60, b=40),
)

# Display the Plotly chart
st.plotly_chart(fig, key="data_explorer_multi_axis_plot")
# --------------------------------------------------------------------------------
# SHOW 6 PyGWalker
# --------------------------------------------------------------------------------
# st.header("DataWalker - Interactive Data Exploration")
# df_filt.reset_index(inplace=True)
# df_filt['datetime'] = pd.to_datetime(df_filt['datetime'], format="%d/%m/%Y %H:%M")
# pyg_app = StreamlitRenderer(df_filt)
# pyg_app.explorer()

# --------------------------------------------------------------------------------
# SHOW 7 MEASUREMENTS FROM INFLUXDB
# --------------------------------------------------------------------------------
measurements = list(MEASUREMENT_LABELS.keys())

# --- Streamlit UI ---
st.header("📊 Online Data Downloader")
st.subheader("Select Measurements")

# Session state defaults
if "selected_measurements" not in st.session_state:
    st.session_state.selected_measurements = set(measurements)
if "select_all" not in st.session_state:
    st.session_state.select_all = True
if "time_range" not in st.session_state:
    st.session_state.time_range = "last 1 hour"
if "window_by" not in st.session_state:
    st.session_state.window_by = "1 hour"
if "online_df" not in st.session_state:
    st.session_state.online_df = None

# Ensure per-measurement checkbox keys exist and follow select_all by default
for meas in measurements:
    st.session_state.setdefault(f"meas_{meas}", st.session_state.select_all)

# Select All toggle outside the form for immediate effect
select_all = st.toggle("Select All", key="select_all")
if select_all:
    st.session_state.selected_measurements = set(measurements)
    for meas in measurements:
        st.session_state[f"meas_{meas}"] = True

with st.form(key="measurement_form"):
    col1, col2 = st.columns(2)
    # Measurement checkboxes (manual selection)
    cols = st.columns(4)
    for i, meas in enumerate(measurements):
        col = cols[i % 4]
        label = MEASUREMENT_LABELS.get(meas, meas)
        with col:
            # Use session state-backed keys; avoid passing a conflicting value
            checked = st.checkbox(label, key=f"meas_{meas}")
            if checked:
                st.session_state.selected_measurements.add(meas)
            else:
                st.session_state.selected_measurements.discard(meas)

    # Options
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox(
            "Select Time Range:",
            list(TIME_OPTIONS.keys()),
            index=list(TIME_OPTIONS.keys()).index(st.session_state.time_range),
        )
        st.session_state.time_range = time_range
    with col2:
        window_by = st.selectbox(
            "Select Averaging Window:",
            list(FREQUENCY_TO_TIMEDTA.keys()),
            index=list(FREQUENCY_TO_TIMEDTA.keys()).index(st.session_state.window_by),
        )
        st.session_state.window_by = window_by

    # Fetch action
    fetch_clicked = st.form_submit_button("⬇️ Fetch")
    if fetch_clicked:
        sm = st.session_state

        invalid = (
            (wr := FREQUENCY_TO_TIMEDTA.get(sm.window_by))
            and (tr := TIME_OPTIONS.get(sm.time_range))
            and pd.to_timedelta(wr) > tr
        )

        if invalid:
            st.warning(f"⚠️ Avg window ({sm.window_by}) > time range ({sm.time_range})")
            st.stop()

        if not sm.selected_measurements:
            st.warning("Please select at least one measurement.")
            st.stop()

        sm.online_df = fetch_online_df(
            sorted(sm.selected_measurements),
            sm.time_range,
            request_type="windowed-average",
            window_by=sm.window_by,
        )

# Display and allow download after the form (survives reruns)
if (
    isinstance(st.session_state.online_df, pd.DataFrame)
    and not st.session_state.online_df.empty
):
    df_show = st.session_state.online_df.sort_index()
    st.dataframe(df_show.head(100))
    csv_bytes = df_show.reset_index().to_csv(index=False).encode("utf-8")
    ts_label = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%SZ")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"online_data_{ts_label}.csv",
        mime="text/csv",
    )
else:
    st.info("No data returned for the selected options.")


# 8 --- UI Section for Offline Data ---
UTC = pytz.UTC

st.header("📄 Offline Data Viewer")
st.caption("Source: Neon database.")

# ── Pre-compute options ──
TIME_OPTIONS_UI = list(TIME_OPTIONS.keys())[7:]
_REPORT_KEYS = list(OFFLINE_DATABASE_REPORT_MAP.keys())
_TABLE_KEYS = sorted(OFFLINE_DATABASE_TABLES.keys())

# Session-state defaults
if "db_fetch_type" not in st.session_state:
    st.session_state.db_fetch_type = "Logical report"
if "selected_db_report" not in st.session_state or st.session_state.selected_db_report not in _REPORT_KEYS:
    st.session_state.selected_db_report = _REPORT_KEYS[0]
if "selected_db_table" not in st.session_state or st.session_state.selected_db_table not in _TABLE_KEYS:
    st.session_state.selected_db_table = _TABLE_KEYS[0]
if "offline_time_range_choice" not in st.session_state:
    st.session_state.offline_time_range_choice = "Use Start/End Dates"

# ── Structural controls — outside the form so switching immediately re-renders ──
ctrl_left, ctrl_right = st.columns([1, 1])
with ctrl_left:
    db_mode = st.radio(
        "Fetch Type",
        ["Logical report", "Explicit table"],
        horizontal=True,
        key="db_fetch_type",
    )
with ctrl_right:
    time_range_choice = st.selectbox(
        "Select Time Range",
        ["Use Start/End Dates"] + TIME_OPTIONS_UI,
        key="offline_time_range_choice",
    )

# ── Form — data selection + dates + submit (no live refetch on widget change) ──
with st.form("offline_data_form"):
    fcol_sel, fcol_dates = st.columns([1, 1])

    with fcol_sel:
        if db_mode == "Logical report":
            selected_db_report = st.selectbox(
                "Select Offline Report",
                _REPORT_KEYS,
                format_func=lambda k: OFFLINE_REPORT_LABEL_MAP.get(k, k),
                key="selected_db_report",
            )
            selected_db_table = None
        else:
            selected_db_report = None
            selected_db_table = st.selectbox(
                "Select Table",
                _TABLE_KEYS,
                format_func=offline_table_label,
                key="selected_db_table",
            )

    with fcol_dates:
        if time_range_choice == "Use Start/End Dates":
            d1, d2 = st.columns(2)
            with d1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now().date() - timedelta(days=30),
                    key="offline_start_date",
                )
            with d2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now().date(),
                    key="offline_end_date",
                )
        else:
            start_date = end_date = None
            st.info(f"Fetching: **{time_range_choice}**")

    submitted = st.form_submit_button("Fetch Offline Data", use_container_width=False)

# ── Fetch on submit ──
if submitted:
    time_range_to_fetch = None
    if time_range_choice == "Use Start/End Dates":
        if start_date > end_date:
            st.error("❌ Start Date cannot be after End Date.")
        else:
            start_local = local_tz.localize(datetime.combine(start_date, time.min))
            end_local = local_tz.localize(datetime.combine(end_date, time.max))
            time_range_to_fetch = (start_local.astimezone(UTC), end_local.astimezone(UTC))
    else:
        time_range_to_fetch = time_range_choice

    if time_range_to_fetch is not None:
        with st.spinner("Fetching from database…"):
            if selected_db_table:
                df_offline = fetch_table_data(
                    table_name=selected_db_table,
                    time_range=time_range_to_fetch,
                )
                output_name = selected_db_table
            else:
                df_offline = fetch_database_report(
                    report_type=selected_db_report,
                    time_range=time_range_to_fetch,
                )
                output_name = selected_db_report.lower()

        if df_offline.empty:
            st.warning(f"No data found for **{output_name}**.")
            st.session_state.offline_viewer_result = None
        else:
            if isinstance(df_offline.index, pd.DatetimeIndex):
                if df_offline.index.tz is None:
                    df_offline.index = df_offline.index.tz_localize(UTC)
                df_offline.index = df_offline.index.tz_convert(local_tz)
                df_offline.index.name = "time (IST)"
            st.session_state.offline_viewer_result = {
                "df": df_offline,
                "name": output_name,
            }

# ── Show results (persisted in session state across reruns) ──
_result = st.session_state.get("offline_viewer_result")
if _result:
    _df = _result["df"]
    _name = _result["name"]
    _label = OFFLINE_REPORT_LABEL_MAP.get(_name.upper(), offline_table_label(_name))
    st.caption(f"Showing **{len(_df):,}** rows × **{len(_df.columns)}** cols — {_label}")
    st.dataframe(_df, use_container_width=True)
    st.download_button(
        label="Download as CSV",
        data=_df.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"{_name}.csv",
        mime="text/csv",
        key="offline_download_btn",
    )


# 9 --- ML Dataset Section ---


local_tz = ZoneInfo(config["ml_dataset"]["local_tz"])


# IMPORTANT: create once so cache survives reruns
@st.cache_resource
def get_fetcher():
    return MlDatasetFetcher()


fetcher = get_fetcher()

# ---------------- UI LAYOUT ----------------
left_col, right_col = st.columns(2)

# -------------- ML DATASET FETCHER --------------
with left_col:
    st.header("📄 ML Dataset")

    st.caption(
        "Source: Neon DB — joins charge quantities, weighted raw material chemistry, "
        "HM & Slag (hourly-interpolated), and burden distribution into one hourly dataset."
    )

    with st.form("ml_form"):
        rm_choice_raw = st.radio(
            "Select RM Dataset",
            ["RM Charge", "RM DPR"],
            horizontal=True,
        )

        _today_ml = datetime.now(local_tz).date()
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", _today_ml - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", _today_ml)

        cache_override = st.checkbox("Override Cache", value=True)

        st.caption(
            "**Partial**: per-source resampling only — RM chemistry 8 h ffill, "
            "HM & Slag 1 h interp, burden full ffill, charge masses hourly sum, "
            "online params 1 h average.  No DataCleaner.\n\n"
            "**ML Dataset**: partial + DataCleaner — drops rows with > 70% NaN, "
            "drops high-NaN columns (unless listed in `skip_imputation_alias_keys`), "
            "outlier rules, iterative + simple imputation."
        )
        btn_col_a, btn_col_b = st.columns(2)
        with btn_col_a:
            partial_submit = st.form_submit_button(
                "Fetch Partial (no imputation)",
                use_container_width=True,
            )
        with btn_col_b:
            cleaned_submit = st.form_submit_button(
                "Fetch ML Dataset (cleaned + imputed)",
                use_container_width=True,
            )

    if partial_submit or cleaned_submit:
        if start_date > end_date:
            st.error("Start Date cannot be after End Date.")
        else:
            mode_tag = "ML" if cleaned_submit else "PARTIAL"
            with st.spinner(f"Fetching {mode_tag} dataset..."):
                df_final = fetcher.get_dataset(
                    start_date=start_date,
                    end_date=end_date,
                    rm_choice=rm_choice_raw,
                    cache_override=cache_override,
                )
                if cleaned_submit and not df_final.empty:
                    # Apply DataCleaner with overrides tuned for ad-hoc viewing.
                    # The defaults are calibrated for the training dataset
                    # (must show cruising-only operation); on an arbitrary
                    # date range they wipe every off-cruise hour and report
                    # all columns as "all-NaN" at final imputation.
                    #
                    #   row_min_non_na_fraction = 0.3  → drop rows with >70% NaN
                    #   col_max_nan_fraction    = 0.95 → keep all but truly-empty cols
                    #                                    so SINTER strength etc. can be
                    #                                    imputed instead of dropped
                    #   cruising_filters        = ()   → don't restrict to cruising rows
                    #   tonnage_caps            = {}   → caps were sub-hour calibrated;
                    #                                    the per-source resample uses
                    #                                    hourly sums so they spuriously
                    #                                    drop rows
                    from dataclasses import replace as dc_replace
                    from furnace_data.dataset.cleaning import (
                        DataCleaner,
                        build_default_config,
                    )
                    cleaning_cfg = dc_replace(
                        build_default_config(),
                        row_min_non_na_fraction=0.3,
                        col_max_nan_fraction=0.95,
                        cruising_filters=(),
                        tonnage_caps={},
                    )
                    df_final = DataCleaner(cleaning_cfg).clean(df_final)

            if df_final.empty:
                st.warning("No data found.")
            else:
                st.success(f"Rows fetched ({mode_tag}): {len(df_final)}")
                st.dataframe(df_final, height=300)

                st.download_button(
                    "Download CSV",
                    df_final.to_csv(index=True).encode("utf-8"),
                    file_name=f"unified_ML_{mode_tag}_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                )

# -------------- GENERATE & REFRESH ML DATASET --------------
# Set to True to re-enable the extend / override buttons that write to Neon
# (offline_feed.historical_static_ml_dataset).  Kept False while the user
# wants Neon writes paused; flipping this back to True restores the original
# behaviour without any other change.
GENERATE_REFRESH_PANEL_ENABLED = False

with right_col:
    st.header("📄 Generate & Refresh ML Dataset")

    if not GENERATE_REFRESH_PANEL_ENABLED:
        st.info(
            "⏸️ **Temporarily disabled.** Write-to-Neon (Extend / Override) is "
            "paused.  Use the **ML Dataset** section on the left to fetch "
            "Partial or Cleaned data without persisting to the database.\n\n"
            "To re-enable: set `GENERATE_REFRESH_PANEL_ENABLED = True` near the "
            "top of this section in `2_Data_Explorer.py`."
        )

    st.caption("Extend Neon DB → rebuild cleaned local CSV → validate.")

    sm = StaticDatasetManager(fullpath)
    meta = sm.get_meta()
    db_end = sm.get_db_end_date()

    # Status row
    _today_gen = datetime.now(local_tz).date()
    _db_end_str  = str(db_end)  if db_end  else "unknown"
    _csv_upd_str = meta.last_updated[:16] if meta and meta.last_updated else "never"
    _csv_rows    = meta.rows    if meta else "?"
    _csv_cols    = meta.columns if meta else "?"
    st.caption(
        f"DB end: **{_db_end_str}** · "
        f"Local CSV: **{_csv_upd_str}** · "
        f"**{_csv_rows}** rows × **{_csv_cols}** cols"
    )

    # Skip the extend / override forms entirely when the panel is disabled —
    # avoids rendering buttons that would call extend_and_persist.
    if GENERATE_REFRESH_PANEL_ENABLED:
      with st.container(border=True):

        _tab_ext, _tab_ovr = st.tabs(["Extend to Date", "Override from Date"])

        # ── Extend tab ──────────────────────────────────────────────────────
        with _tab_ext:
            st.caption(
                "Fetches data from the DB end date up to the chosen date "
                "(Steps 2-5: charge/DPR, HM/Slag, burden, InfluxDB), "
                "writes new rows to Neon, then rebuilds the cleaned local CSV."
            )
            with st.form("extend_form"):
                _ext_rm = st.radio(
                    "RM Mode", ["RM Charge", "RM DPR"], horizontal=True, key="ext_rm"
                )
                _ext_end = st.date_input(
                    "Generate to date", value=_today_gen, key="ext_end_date"
                )
                _ext_submit = st.form_submit_button(
                    "Extend & Rebuild Dataset", use_container_width=True
                )

            if _ext_submit:
                if db_end and _ext_end <= db_end:
                    st.warning(
                        f"No new data: DB already has data up to {db_end}. "
                        f"Choose a date after {db_end} or use Override."
                    )
                else:
                    with st.spinner("Fetching new data and writing to Neon…"):
                        _raw = fetcher.extend_and_persist(
                            start_date=db_end or _today_gen,
                            end_date=_ext_end,
                            rm_choice=_ext_rm,
                            override_from_date=None,
                        )
                    if _raw.empty:
                        st.warning("No new data was returned.")
                    else:
                        st.info(f"Wrote {len(_raw)} raw rows to Neon DB.")
                        with st.spinner("Rebuilding cleaned static dataset…"):
                            _full_df = sm.update_static()
                            sm.save(_full_df)
                            from data.ml.static_csv import update_cutoff_date
                            update_cutoff_date(_ext_end)
                        st.session_state["static_ml_dataset_df"] = _full_df
                        st.success(
                            f"Dataset rebuilt: {len(_full_df)} rows, "
                            f"{_full_df.index.min().date()} → {_full_df.index.max().date()}"
                        )
                        with st.spinner("Running validation…"):
                            from furnace_data.dataset.validator import validate_dataset
                            st.session_state["ml_validation_report"] = validate_dataset(_full_df)

        # ── Override tab ─────────────────────────────────────────────────────
        with _tab_ovr:
            st.caption(
                "Deletes existing DB rows from the chosen date onwards, "
                "then re-generates and rewrites them.  Use this to fix bad data."
            )
            with st.form("override_form"):
                _ovr_rm = st.radio(
                    "RM Mode", ["RM Charge", "RM DPR"], horizontal=True, key="ovr_rm"
                )
                _ovr_from = st.date_input(
                    "Override from date",
                    value=(db_end - timedelta(days=30)) if db_end else _today_gen - timedelta(days=30),
                    key="ovr_from_date",
                )
                _ovr_end = st.date_input(
                    "Generate to date", value=_today_gen, key="ovr_end_date"
                )
                st.warning(
                    f"This will DELETE all DB rows from {_ovr_from} onwards "
                    "and re-generate them.",
                    icon="⚠️",
                )
                _ovr_submit = st.form_submit_button(
                    "Override & Rebuild", use_container_width=True
                )

            if _ovr_submit:
                with st.spinner(f"Deleting rows from {_ovr_from} and re-generating…"):
                    _raw = fetcher.extend_and_persist(
                        start_date=_ovr_from,
                        end_date=_ovr_end,
                        rm_choice=_ovr_rm,
                        override_from_date=_ovr_from,
                    )
                if _raw.empty:
                    st.warning("No data returned for the override range.")
                else:
                    st.info(f"Wrote {len(_raw)} raw rows to Neon DB.")
                    with st.spinner("Rebuilding cleaned static dataset…"):
                        _full_df = sm.update_static()
                        sm.save(_full_df)
                        from data.ml.static_csv import update_cutoff_date
                        update_cutoff_date(_ovr_end)
                    st.session_state["static_ml_dataset_df"] = _full_df
                    st.success(
                        f"Dataset rebuilt: {len(_full_df)} rows, "
                        f"{_full_df.index.min().date()} → {_full_df.index.max().date()}"
                    )
                    with st.spinner("Running validation…"):
                        from furnace_data.dataset.validator import validate_dataset
                        st.session_state["ml_validation_report"] = validate_dataset(_full_df)

    # Download
    _dl_df = st.session_state.get("static_ml_dataset_df") or df
    st.download_button(
        label="Download ML Dataset",
        data=_dl_df.to_csv(index=True).encode("utf-8"),
        file_name="furnace_static_ml_dataset.csv",
        mime="text/csv",
        disabled=_dl_df.empty,
    )

    # Validation report expander
    _vr = st.session_state.get("ml_validation_report")
    if _vr:
        with st.expander(
            ("✅ Validation passed" if _vr["passed"] else "❌ Validation has errors")
            + f"  —  {len(_vr['warnings'])} warnings, {len(_vr['errors'])} errors"
        ):
            _chk = _vr["checks"]

            # Shape + date range
            _sh = _chk.get("shape", {})
            st.markdown(
                f"**Rows:** {_sh.get('rows')} &nbsp;|&nbsp; "
                f"**Cols:** {_sh.get('cols')} &nbsp;|&nbsp; "
                f"**Range:** {_sh.get('start', '')[:10]} → {_sh.get('end', '')[:10]}"
            )

            # Monthly coverage
            _mc = _chk.get("monthly_coverage")
            if _mc is not None and not _mc.empty:
                st.markdown("**Monthly row coverage:**")
                _mc_df = _mc.reset_index()
                _mc_df.columns = ["Month", "Rows"]
                _mc_df = _mc_df.set_index("Month")
                st.dataframe(
                    _mc_df.style.highlight_between(
                        subset=["Rows"], left=0, right=20, color="#ffd0d0"
                    ),
                    use_container_width=True,
                    height=min(300, 35 * len(_mc_df) + 40),
                )

            # KPI stats
            _kpi = _chk.get("kpi_stats")
            if _kpi is not None and not _kpi.empty:
                st.markdown("**KPI statistics:**")
                st.dataframe(_kpi.round(2), use_container_width=True)

            # Warnings and errors
            if _vr["errors"]:
                for _e in _vr["errors"]:
                    st.error(_e)
            if _vr["warnings"]:
                for _w in _vr["warnings"]:
                    st.warning(_w)


# 10 --- HOT METAL AND SLAG DATA SECTION ---


service = fetcher.service

# ---------------- UI ----------------
st.header("📄 Hot Metal & Slag")
st.caption(
    "⚠️ **Interpolated data** — cast-level HM/Slag samples (typically spaced 1–3 hours "
    "apart) are time-interpolated onto a regular grid. Every row between actual casts is "
    "**synthetic** (linear interpolation). Use the Interval field to control grid spacing; "
    "default is 60 min (1 hour)."
)

with st.form("hotmetal_form_2"):
    _today_hm = datetime.now(local_tz).date()
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From Date", _today_hm - timedelta(days=7))
    with col2:
        to_date = st.date_input("To Date", _today_hm)

    interval_min = st.number_input(
        "Grid interval (minutes) — data interpolated to this resolution",
        min_value=1,
        max_value=600,
        value=60,
    )

    fetch_btn = st.form_submit_button("Fetch HM & Slag Data")

# ---------------- ACTION ----------------
if fetch_btn:

    if from_date > to_date:
        st.error("❌ From Date must be less than or equal to To Date.")
        st.stop()

    keep_cols = config.get("keep_cols", [])

    with st.spinner("Fetching Hot Metal & Slag data..."):
        df_final = service.fetch_hotmetal_hourly(
            start_date=from_date,
            end_date=to_date,
            keep_columns=keep_cols,
            interval_minutes=interval_min,
        )

    if df_final.empty:
        st.warning("No data found.")
        st.stop()

    # ---- DROP UNWANTED COLUMNS ----
    drop_cols = ["cast_no_ladle_spec", "lab_sample_id"]
    df_final = df_final.drop(columns=[c for c in drop_cols if c in df_final.columns])

    st.success("Data processed successfully!")
    st.dataframe(df_final)

    # ---- CSV DOWNLOAD ----
    st.download_button(
        "Download CSV",
        df_final.to_csv(index=True).encode("utf-8"),
        file_name=f"hotmetal_{from_date}_to_{to_date}_{interval_min}min.csv",
        mime="text/csv",
    )
