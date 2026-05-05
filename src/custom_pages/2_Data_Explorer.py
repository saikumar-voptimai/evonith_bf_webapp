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
from furnace_data.influx.offline import (
    clean_rm_data,
    fetch_offline_data as fetch_influx_offline_data,
)
from furnace_data.neon_db.offline import (
    NEON_OFFLINE_REPORT_MAP,
    NEON_OFFLINE_TABLES,
    fetch_offline_data as fetch_neon_table_data,
    fetch_offline_report as fetch_neon_offline_report,
)
from furnace_data.dataset.fetcher import DatasetFetcher as MlDatasetFetcher
from data.fetch_presets import OFFLINE_REPORT_LABEL_MAP
from data.ml.static_csv import get_static_dataset_path, load_static_dataset
from data.ml.static_dataset_manager import StaticDatasetManager
from utils.dataset_refresher import maybe_refresh

config = load_config("setting_ds_dv.yml")  # Load the configuration file
config_vsense = load_config("setting_vsense.yml")
offline_measurements = config.get("offline_measurements", {})
influx_cfg = config.get("influx_offline", {})

load_dotenv()
INFLUX_OFFLINE_TOKEN = os.getenv("INFLUX_OFFLINE_TOKEN", "")

local_tz = pytz.timezone("Asia/Kolkata")  # or use your actual timezone
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# ── dataset auto-refresh ───────────────────────────────────────────────────
if maybe_refresh(config):
    st.sidebar.caption("⏳ Refreshing dataset in background…")

fullpath = get_static_dataset_path(config["DATA"])
df = load_static_dataset(fullpath)


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
    df["COKE RATE KG/THM"] + factor * df["ACTUALKG/THM."]
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

# Fit
coeffs = np.polyfit(df_filt[x_p], df_filt[y_p], deg=order)
eqn_str = None
for i in range(len(coeffs)):
    if i == 0:
        eqn_str = f"{coeffs[len(coeffs) - 1]:.3g}"
    else:
        coeff_i = coeffs[len(coeffs) - (i + 1)]
        if i == 1:
            eqn_str += (
                f" + {coeff_i:.3g}x" if coeff_i >= 0 else f" - {abs(coeff_i):.3g}x"
            )
        else:
            eqn_str += (
                f" + {coeff_i:.3g}x^{i}"
                if coeff_i >= 0
                else f" - {abs(coeff_i):.3g}x^{i}"
            )

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
graph = sns.lmplot(
    x=x_p,
    y=y_p,
    markers="x",
    hue="Type",
    scatter_kws={"s": 8, "marker": "x"},
    data=df_filt,
    fit_reg=False,
)

# Create the regplot
ax = graph.axes[0, 0]
plot = sns.regplot(data=df_filt, x=x_p, y=y_p, order=order, scatter=False, ax=ax)

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

st.caption("Source switch applies to this raw offline fetch only.")

if "offline_source" not in st.session_state:
    st.session_state.offline_source = "Neon DB"
if "neon_fetch_type" not in st.session_state:
    st.session_state.neon_fetch_type = "Logical report"
if "selected_neon_report" not in st.session_state:
    st.session_state.selected_neon_report = list(NEON_OFFLINE_REPORT_MAP.keys())[0]
if "selected_neon_table" not in st.session_state:
    st.session_state.selected_neon_table = sorted(NEON_OFFLINE_TABLES.keys())[0]
if "selected_influx_measurement" not in st.session_state:
    st.session_state.selected_influx_measurement = list(offline_measurements.keys())[0]

TIME_OPTIONS_UI = list(TIME_OPTIONS.keys())[7:]


col_left, col_right = st.columns(2)

with col_left:
    offline_source = st.radio(
        "Offline Source",
        ["Neon DB", "InfluxDB rollback"],
        horizontal=True,
        key="offline_source",
    )

    if offline_source == "Neon DB":
        neon_mode = st.radio(
            "Neon Fetch Type",
            ["Logical report", "Explicit table"],
            horizontal=True,
            key="neon_fetch_type",
        )
        if neon_mode == "Logical report":
            neon_reports = list(NEON_OFFLINE_REPORT_MAP.keys())
            selected_neon_report = st.selectbox(
                "Select Offline Report",
                neon_reports,
                format_func=lambda key: OFFLINE_REPORT_LABEL_MAP.get(key, key),
                key="selected_neon_report",
            )
            selected_neon_table = None
        else:
            selected_neon_report = None
            selected_neon_table = st.selectbox(
                "Select Neon Table",
                sorted(NEON_OFFLINE_TABLES.keys()),
                key="selected_neon_table",
            )
        selected_influx_measurement = None
    else:
        selected_influx_measurement = st.selectbox(
            "Select Influx Offline Measurement",
            list(offline_measurements.keys()),
            key="selected_influx_measurement",
        )
        selected_neon_report = None
        selected_neon_table = None

with col_right:
    time_range_choice = st.selectbox(
        "Select Time Range (optional):",
        ["Use Start/End Dates"] + TIME_OPTIONS_UI,
        key="offline_time_range_choice",
    )

    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date(),
            key="offline_start_date",
        )
    with d2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date(),
            key="offline_end_date",
        )

submitted = st.button("Fetch Offline Data", key="offline_fetch_button")


if submitted:

    if start_date > end_date:
        st.error("❌ Invalid date range: Start Date cannot be after End Date.")
        st.stop()

    # ---- CORRECT TIME RANGE HANDLING ----
    if time_range_choice == "Use Start/End Dates":

        start_local = local_tz.localize(datetime.combine(start_date, time.min))
        end_local = local_tz.localize(datetime.combine(end_date, time.max))

        time_range_to_fetch = (
            start_local.astimezone(UTC),
            end_local.astimezone(UTC),
        )
    else:
        time_range_to_fetch = time_range_choice

    if offline_source == "Neon DB":
        if selected_neon_table:
            df_offline = fetch_neon_table_data(
                table_name=selected_neon_table,
                time_range=time_range_to_fetch,
            )
            output_name = selected_neon_table
        else:
            df_offline = fetch_neon_offline_report(
                report_type=selected_neon_report,
                time_range=time_range_to_fetch,
            )
            output_name = selected_neon_report.lower()
    else:
        database = influx_cfg.get("database", "bf2_evonith_offline_utc")
        df_offline = fetch_influx_offline_data(
            measurement=offline_measurements[selected_influx_measurement],
            time_range=time_range_to_fetch,
            database=database,
        )
        output_name = selected_influx_measurement

    if df_offline.empty:
        st.warning(f"No data found for {output_name}")
        st.stop()

    if offline_source != "Neon DB" and selected_influx_measurement == "Bunker Report":
        df_offline = clean_rm_data(df_offline)

    # Index is already UTC → convert once
    if isinstance(df_offline.index, pd.DatetimeIndex):
        if df_offline.index.tz is None:
            df_offline.index = df_offline.index.tz_localize(UTC)
        df_offline.index = df_offline.index.tz_convert(local_tz)
        df_offline.index.name = "time (IST)"

    st.dataframe(df_offline)
    df_download = df_offline.reset_index()
    st.download_button(
        label="Download as CSV",
        data=df_download.to_csv(index=False).encode("utf-8"),
        file_name=f"{output_name}.csv",
        mime="text/csv",
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
        "Source: Neon DB by default for RM Charge/RM DPR, RM-HM, HM & Slag, "
        "and burden distribution. InfluxDB remains available for rollback."
    )

    with st.form("ml_form"):
        ml_source_label = st.radio(
            "ML Source",
            ["Neon DB", "InfluxDB rollback"],
            horizontal=True,
        )
        rm_choice_raw = st.radio(
            "Select RM Dataset",
            ["RM Charge", "RM DPR"],
            horizontal=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now(local_tz).date())
        with col2:
            end_date = st.date_input("End Date", datetime.now(local_tz).date())

        cache_override = st.checkbox("Override Cache")
        submitted = st.form_submit_button("Fetch Dataset")

    if submitted:
        if start_date > end_date:
            st.error("Start Date cannot be after End Date.")
        else:
            with st.spinner("Fetching ML Dataset..."):
                df_final = fetcher.get_ml_dataset(
                    start_date=start_date,
                    end_date=end_date,
                    rm_choice=rm_choice_raw,
                    cache_override=cache_override,
                    source=(
                        "neon_db"
                        if ml_source_label == "Neon DB"
                        else "influx"
                    ),
                )

            if df_final.empty:
                st.warning("No data found.")
            else:
                st.success(f"Rows fetched: {len(df_final)}")
                st.dataframe(df_final, height=300)

                st.download_button(
                    "Download CSV",
                    df_final.to_csv(index=True).encode("utf-8"),
                    file_name=f"unified_ML_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                )

# -------------- STATIC FILTERED DATASET MANAGER --------------

with right_col:
    st.header("📄 Filtered ML Dataset")
    st.caption(
        "Source: static ML dataset updater. It uses the ML dataset pipeline, "
        "not the raw Offline Source switch or the interactive ML Source selector."
    )
    with st.container(border=True):

        sm = StaticDatasetManager(fullpath)

        # ---------------- RESET FLAG INIT ----------------
        if "reset_reprocess_date" not in st.session_state:
            st.session_state.reset_reprocess_date = False

        # ---------------- APPLY RESET BEFORE WIDGET ----------------
        if st.session_state.reset_reprocess_date:
            st.session_state.reprocess_date = None
            st.session_state.reset_reprocess_date = False

        # ---------------- INPUTS ----------------
        rm_choice = st.radio(
            "Select RM Dataset",
            ["RM Charge", "RM DPR"],
            horizontal=True,
            key="rm_static",
        )

        reprocess_date = st.date_input(
            "Reprocess data from date (optional)",
            value=None,
            key="reprocess_date",
        )

        # ---------------- ACTION BUTTON ---------------
        col1, col2 = st.columns(2)

        with col1:
            if st.button(" Fetch & Process"):
                with st.spinner("Fetching & updating static dataset..."):
                    df = sm.update_static(
                        rm_choice=rm_choice,
                        start_date=reprocess_date,
                    )

                    if df.empty:
                        st.warning("No data fetched.")
                    else:
                        sm.save(df)
                        st.success(f"Dataset ready ({len(df)} rows)")

                        # request reset for next rerun
                        st.session_state.reset_reprocess_date = True
                        st.rerun()

        with col2:
            with open(fullpath, "rb") as f:
                st.download_button(
                    label="Download ML Dataset",
                    data=f,
                    file_name="furnace_dataset.csv",
                    mime="text/csv",
                )


# 10 --- HOT METAL AND SLAG DATA SECTION ---


service = fetcher.service

# ---------------- UI ----------------
st.header("📄 HOT METAL AND SLAG")

with st.form("hotmetal_form_2"):
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From Date")
    with col2:
        to_date = st.date_input("To Date")

    interval_min = st.number_input(
        "Interval (minutes)",
        min_value=1,
        max_value=600,
        value=60,
    )

    fetch_btn = st.form_submit_button("Fetch HM & SLAG DATA")

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
