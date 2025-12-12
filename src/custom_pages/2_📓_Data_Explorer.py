import streamlit as st
from datetime import datetime, timedelta, date
from datetime import timezone
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import pytz
from pathlib import Path
from utils.helper_functions_explorer import data_retrieval as dr
from config.config_loader import load_config
from dotenv import load_dotenv
from domain.ml_dataset_service import MlDatasetService
from zoneinfo import ZoneInfo


config = load_config("setting_ds_dv.yml")  # Load the configuration file
config_vsense = load_config("setting_vsense.yml")
offline_measurements = config.get("offline_measurements", {})
influx_cfg = config.get("influx_offline", {})

load_dotenv() 
INFLUX_OFFLINE_TOKEN = os.getenv("INFLUX_OFFLINE_TOKEN", "")

local_tz = pytz.timezone("Asia/Kolkata")  # or use your actual timezone
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

fullpath = Path(__file__).resolve().parents[1] / config['DATA'].split('/')[1] /config['DATA'].split('/')[2]
df = pd.read_csv(fullpath, index_col=0, parse_dates=True)

FIELD_LABELS = {
    internal_key: human_label
    for mapping in config["data_mapping"].values()
    for human_label, internal_key in mapping.items()
}

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
    'last 2 weeks': timedelta(weeks=2),
    'last 1 month': timedelta(days=30),
    'last 2 months': timedelta(days=60),
    'last 3 months': timedelta(days=90),
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
    "temperature_profile": "Temperature Profile"
}

st.title("Visualisation tool")

#--------------------------------------------------------------------------
st.subheader("Distribution plots")

cols = st.columns([0.2, 0.2, 0.3, 0.3])
with cols[0]:
    from_date = st.date_input("From Date", value=pd.to_datetime(df.index[0]).date())

with cols[1]:
    to_date = st.date_input("To Date", value=pd.to_datetime(df.index[-1]).date())

with cols[2]:
    model_keys = list(config_vsense['Optimisation'].keys())
    control_params = config_vsense['Optimisation'][model_keys[1]]['control_params']
    input_params = [item for sublist in config_vsense['Optimisation'][model_keys[1]]['input_params'].values() for item in sublist]
    all_params = control_params + input_params
    x_p = st.selectbox("Select X feature", all_params)

with cols[3]:
    output_params = [config_vsense['Optimisation'][key]['output_param'] for key in config_vsense['Optimisation']]
    y_p = st.selectbox("Select Y feature", output_params+['Unit Cost']+control_params)

with st.sidebar:
    st.subheader("PCI/Coke cost")
    factor = st.number_input("PCI/Coke Cost ratio", value=13250/25000, step=0.01, format="%.2f")

df['Unit Cost'] = (df['Coke Rate Kg/Thm'] + factor * df['ActualKg/Thm.']) * 25100/1000
df_filt = df[(pd.to_datetime(df.index, format="%d-%m-%Y %H:%M").date >= from_date) & 
             (pd.to_datetime(df.index, format="%d-%m-%Y %H:%M").date <= to_date)]

# UI layout
cols = st.columns([0.3, 0.2, 0.2, 0.2, 0.1])
with cols[0]:
    filter_feature = st.selectbox('Filter by', df.columns, index=int(len(df.columns)-3))
with cols[1]:
    min_value = st.number_input('Lower bound', 
                                    value=float(df[filter_feature].quantile(0.25)))
with cols[2]:
    max_value = st.number_input('Higher bound', 
                                    value=float(df[filter_feature].quantile(0.75)))
with cols[3]:
    order = st.selectbox('Poly order', [1, 2, 3])
with cols[4]:
    inverse_filter = st.checkbox('Invert', value=True)

# Fit
coeffs = np.polyfit(df_filt[x_p], df_filt[y_p], deg=order)
eqn_str = None
for i in range(len(coeffs)):
    if i == 0:
        eqn_str = f"{coeffs[len(coeffs) - 1]:.3g}"
    else:
        coeff_i = coeffs[len(coeffs) - (i+1)]
        if i == 1:
            eqn_str += f" + {coeff_i:.3g}x" if coeff_i >= 0 else f" - {abs(coeff_i):.3g}x"
        else:
            eqn_str += f" + {coeff_i:.3g}x^{i}" if coeff_i >= 0 else f" - {abs(coeff_i):.3g}x^{i}"

# Range filter
if inverse_filter:
    df_filt = df_filt[(df_filt[filter_feature] < min_value) | (df_filt[filter_feature] > max_value)]
else:
    df_filt = df_filt[(df_filt[filter_feature] >= min_value) & (df_filt[filter_feature] <= max_value)]    
df_filt['Type'] = df_filt[filter_feature].apply(lambda x: 'Low' if x < min_value else 'High')

# Scatter and fit
graph = sns.lmplot(x=x_p, 
                   y=y_p,
                   markers='x', 
                   hue='Type',
                   scatter_kws={'s':8, 'marker': 'x'}, 
                   data=df_filt, 
                   fit_reg=False)

# Create the regplot
ax = graph.axes[0, 0]
plot = sns.regplot(data=df_filt, 
                   x=x_p, 
                   y=y_p,
                   order=order,
                   scatter=False,
                   ax=ax)

if eqn_str:
    plt.text(0.05, 0.95, f"y = {eqn_str}", transform=plot.transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

# Resize and render with Streamlit
fig = plot.get_figure()
fig.set_size_inches(10, 5)
st.pyplot(fig, use_container_width=False)

#--------------------------------------------------------------------------
st.subheader("Timeseries plot")
st.sidebar.subheader("TS Plot - Select the date range.")
with st.sidebar:
    st.subheader("Timeseries Plot Options")
    cols = st.columns(2)
    with cols[0]:
        from_date = st.date_input("From Date", value=pd.to_datetime(df.index[0]).date(), key="from_date2")
    with cols[1]:
        to_date = st.date_input("To Date", value=pd.to_datetime(df.index[-1]).date(), key="to_date2")

features = st.multiselect('Select features', df.columns, default=df.columns[0])
df_t = df[(pd.to_datetime(df.index, format="%d-%m-%Y %H:%M").date >= from_date) & 
             (pd.to_datetime(df.index, format="%d-%m-%Y %H:%M").date <= to_date)]
cols = st.columns([0.3, 0.15, 0.15, 0.1, 0.3])
with cols[0]:
    filter_feature = st.selectbox('Select feature to filter by', df.columns, index=int(len(df.columns)-3))
with cols[1]:
    average_range = st.selectbox(
        "Avg Window:", 
        list(FREQUENCY_TO_TIMEDTA.keys()), 
        index=list(FREQUENCY_TO_TIMEDTA.keys()).index("None")
    )
with cols[2]:
    min_value = st.number_input('Min', 
                                 value=float(df_t[filter_feature].min()))
with cols[3]:
    max_value = st.number_input('Max', 
                                 value=float(df_t[filter_feature].max()))
with cols[4]:
    inverse_filter = st.checkbox('Invert')

df_filtered = df_t[(df_t[filter_feature] >= min_value) & (df_t[filter_feature] <= max_value)]

if FREQUENCY_TO_TIMEDTA[average_range] is not None:
    df_avg = dr.average_data(df_filtered, FREQUENCY_TO_TIMEDTA[average_range])
else:
    df_avg = df_filtered.copy()

fig = go.Figure()
colors = ['blue', 'red', 'green', 'orange', 'purple']

for i, feature in enumerate(features):
    axis_name = f'y{i+1}' if i != 0 else 'y'
    trace = go.Scatter(
        x=df_avg.index,
        y=df_avg[feature],
        name=feature,
        mode='lines+markers',
        yaxis=axis_name,
        line=dict(color=colors[i % len(colors)])
    )
    fig.add_trace(trace)

    # Axis layout
    axis_layout = dict(
        title='',
        showgrid=False,
        showticklabels=False,
    )

    if i == 0:
        axis_layout['side'] = 'left'
        fig.update_layout(yaxis=axis_layout)
    else:
        axis_layout.update({
            "overlaying": "y",
            "side": "right" if i % 2 else "left",
            "anchor": "x",
            "position": 1.0 - (i * 0.05) if i % 2 else 0.05 + (i * 0.05),
        })
        fig.update_layout({f'yaxis{i+1}': axis_layout})

# Final layout
fig.update_layout(
    xaxis=dict(title='Date'),
    height=600,
    template="plotly_white",
    hovermode='x unified',
    legend=dict(
        yanchor="top",
        y=1.15,
        xanchor="left",
        x=0.01
    ),
    margin=dict(l=40, r=40, t=60, b=40),
)

# Display the Plotly chart
st.plotly_chart(fig)
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
if "average_range" not in st.session_state:
    st.session_state.average_range = "1 hour"
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
            index=list(TIME_OPTIONS.keys()).index(st.session_state.time_range)
        )
        st.session_state.time_range = time_range
    with col2:
        average_range = st.selectbox(
            "Select Averaging Window:",
            list(FREQUENCY_TO_TIMEDTA.keys()),
            index=list(FREQUENCY_TO_TIMEDTA.keys()).index(st.session_state.average_range)
        )
        st.session_state.average_range = average_range

    # Fetch action
    fetch_clicked = st.form_submit_button("⬇️ Fetch")
    if fetch_clicked:
        selected_measurements = st.session_state.selected_measurements
        if not selected_measurements:
            st.warning("Please select at least one measurement.")
        else:
            # Use keys exactly as defined (no lowercasing)
            tr = st.session_state.time_range
            ar = st.session_state.average_range
            combined_df = dr.fetch_online_df(sorted(list(selected_measurements)), 
                                             tr, 
                                             ar,
                                             FREQUENCY_TO_TIMEDTA,
                                             MEASUREMENT_LABELS,
                                             FIELD_LABELS)

            # Persist for display/download after rerun
            st.session_state.online_df = combined_df

# Display and allow download after the form (survives reruns)
if isinstance(st.session_state.online_df, pd.DataFrame) and not st.session_state.online_df.empty:
    df_show = st.session_state.online_df.sort_index()
    st.dataframe(df_show)
    csv_bytes = df_show.reset_index().to_csv(index=False).encode('utf-8')
    ts_label = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%SZ')
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"online_data_{ts_label}.csv",
        mime="text/csv",
    )
else:
    st.info("No data returned for the selected options.")


# 8 --- UI Section for Offline Data ---

st.header("📄 Offline Data Viewer")
if "time_range_off" not in st.session_state:
    st.session_state.time_range_off = "last 1 week"

if "selected_offline" not in st.session_state:
    st.session_state.selected_offline = list(offline_measurements.keys())[0]

# --- Time options WITHOUT Custom Range ---
TIME_OPTIONS_UI = list(TIME_OPTIONS.keys())[7:]

# --- FORM START ---
with st.form("offline_fetch_form"):

    # Columns: Left = measurement, Right = time range + dates
    col_left, col_right = st.columns([1, 1])

    # ---------------- LEFT COLUMN -------------------
    with col_left:
        selected_offline = st.selectbox(
            "Select Offline Measurement",
            list(offline_measurements.keys()),
            index=list(offline_measurements.keys()).index(st.session_state.selected_offline)
        )

    # ---------------- RIGHT COLUMN -------------------
    with col_right:

        time_range_choice = st.selectbox(
            "Select Time Range (optional):",
            ["Use Start/End Dates"] + TIME_OPTIONS_UI,
            index=0
        )

        # Start & End Date on the SAME ROW
        d1, d2 = st.columns([1, 1])
        with d1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date()
            )
        with d2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date()
            )

    # Submit button
    submitted = st.form_submit_button("Fetch Offline Data")

# --- FORM SUBMITTED ---
if submitted:

    # --- Validation ---
    if start_date > end_date:
        st.error("❌ Invalid date range: Start Date cannot be after End Date.")
        st.stop()

    database = influx_cfg.get("database", "bf2_evonith_offline_utc")

    # Decide fetch strategy
    if time_range_choice == "Use Start/End Dates":
        time_range_to_fetch = (
            f"{start_date}T00:00:00Z",
            f"{end_date}T23:59:59Z"
        )
    else:
        time_range_to_fetch = time_range_choice

    # Fetch data
    df_offline = dr.fetch_offline_data(
        measurement=offline_measurements[selected_offline],
        time_range=time_range_to_fetch,
        database=database,
    )

    if df_offline.empty:
        st.warning(f"No data found for {selected_offline}")
    else:
        if selected_offline == "rm_data":
            df_offline = dr.clean_rm_data(df_offline)

        df_offline.index = df_offline.index.tz_convert(local_tz)
        df_offline.index.name = 'time (IST)'

        st.dataframe(df_offline)

        csv = df_offline.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{selected_offline}.csv",
            mime="text/csv",
        )


# # -------------------- CONFIG --------------------
# local_tz = ZoneInfo("Asia/Kolkata")
# rename_dict = config.get("rename_dict", {})
# rename_values = list(rename_dict.values())
# keep_cols = config.get("keep_cols", [])

# service = MlDatasetService()

# # -------------------- UI --------------------
# st.header("📄 ML Dataset")

# with st.form("ml_form"):
#     st.subheader("Select Time Range")

#     rm_choice = st.radio(
#         "Select RM Dataset",
#         ["RM Charge", "RM DPR"],
#         horizontal=True,
#     )

#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input("Start Date", datetime.now(local_tz).date())
#     with col2:
#         end_date = st.date_input("End Date", datetime.now(local_tz).date())

#     submitted = st.form_submit_button("Fetch Dataset")

# # -------------------- PROCESSING --------------------
# if submitted:
#     if start_date > end_date:
#         st.error("❌ Start Date cannot be after End Date.")
#         st.stop()

#     CUTOFF = service.cutoff_date
#     mode = "charge" if rm_choice == "RM Charge" else "dpr"

#     df_step1 = pd.DataFrame()
#     df_step2 = pd.DataFrame()
#     df_hot = pd.DataFrame()

#     # ------------------------------------------------------
#     # CASE-1 → STEP-1 ONLY (ML)
#     # User selects 01-01-2024 to 05-12-2025 (end_date <= cutoff)
#     # ------------------------------------------------------
#     if end_date <= CUTOFF:
#         st.info("OLD DATA")

#         df_step1 = service.fetch_step1(start_date, end_date, allowed_columns=rename_dict)
#         df_step1 = df_step1.rename(columns=rename_dict)
#         df_step1 = df_step1[df_step1.columns.intersection(rename_values)]

#         df_final = df_step1

#     # ------------------------------------------------------
#     # CASE-2 → STEP-2 + STEP-3 (RM + HM)
#     # User selects 06-12-2025 to 09-12-2025 (start_date > cutoff)
#     # ------------------------------------------------------
#     elif start_date > CUTOFF:
#         st.info("New DATA")

#         # STEP-2: RM data
#         df_step2 = service.fetch_step2(start_date, end_date, mode, allowed_columns=rename_dict)
#         df_step2 = df_step2.rename(columns=rename_dict)
#         df_step2 = df_step2[df_step2.columns.intersection(rename_values)]

#         # STEP-3: Hot metal hourly (60 min)
#         df_hot = service.fetch_hotmetal_hourly(start_date, end_date, keep_columns=keep_cols, interval_minutes=60)
#         df_hot = df_hot.rename(columns=rename_dict)
#         df_hot = df_hot[df_hot.columns.intersection(rename_values)]

#         # STEP-4: Distribution (Neon DB)
#         df_dist = service.fetch_distribution_data(start_date, end_date)
#         df_dist = df_dist.rename(columns=rename_dict)
#         df_dist = df_dist[df_dist.columns.intersection(rename_values)]
#         # Align distribution data with hourly timestamps
#         df_dist = df_dist.reindex(df_step2.index.union(df_hot.index)).sort_index()
#         df_dist = df_dist.ffill()

#         # Merge RM + HM by time
#         # df_final = df_step2.join(df_hot, how="outer").sort_index()
#         df_final = df_step2.join([df_hot, df_dist], how="outer").sort_index()



#     # ------------------------------------------------------
#     # CASE-3 → MIXED RANGE (STEP-1 + STEP-2 + STEP-3)
#     # User selects 01-12-2025 to 10-12-2025
#     # ------------------------------------------------------
#     else:
#         st.info("Feching Old Data")

#         # STEP-1: ML part (start → cutoff)
#         df_step1 = service.fetch_step1(start_date, CUTOFF, allowed_columns=rename_dict)
#         df_step1 = df_step1.rename(columns=rename_dict)
#         df_step1 = df_step1[df_step1.columns.intersection(rename_values)]
#         st.info("Feching New ML Data")
#         # STEP-2: RM part (cutoff+1 → end)
#         df_step2 = service.fetch_step2(CUTOFF + timedelta(days=1), end_date, mode, allowed_columns=rename_dict)
#         df_step2 = df_step2.rename(columns=rename_dict)
#         df_step2 = df_step2[df_step2.columns.intersection(rename_values)]

#         st.info("Feching HM & SLAG Data")
#         # STEP-3: Hot metal for (cutoff+1 → end)
#         df_hot = service.fetch_hotmetal_hourly(
#             CUTOFF + timedelta(days=1),
#             end_date,
#             keep_columns=keep_cols,
#             interval_minutes=60,
#         )
#         df_hot = df_hot.rename(columns=rename_dict)
#         df_hot = df_hot[df_hot.columns.intersection(rename_values)]
#         # STEP-4: Distribution (Neon DB)
#         df_dist = service.fetch_distribution_data(CUTOFF + timedelta(days=1), end_date)
#         df_dist = df_dist.rename(columns=rename_dict)
#         df_dist = df_dist[df_dist.columns.intersection(rename_values)]
#         # Align distribution data with hourly timestamps
#         df_dist = df_dist.reindex(df_step2.index.union(df_hot.index)).sort_index()
#         df_dist = df_dist.ffill()
        
#         df_merged = pd.merge_asof(
#             df_step2.sort_index(),
#             df_hot.sort_index(),
#             left_index=True,
#             right_index=True,
#             direction="nearest",
#             tolerance=pd.Timedelta("1min")
#         )
#         df_merged = df_merged.join(df_dist, how="left")


        
#         # ML + RM merged
#         df_final = pd.concat([df_step1, df_merged]).sort_index()


#     # ------------------------------------------------------
#     # FINAL OUTPUT
#     # ------------------------------------------------------
#     if df_final.empty:
#         st.warning("No data found for the selected range and configuration.")
#     else:
#         df_final.index.name = "time"

#         st.subheader("📊 Final Dataset (Merged ML + RM + Hot Metal)")
#         st.dataframe(df_final)

#         st.download_button(
#             "Download CSV",
#             df_final.to_csv(index=True).encode("utf-8"),
#             file_name=f"unified_ML_RM_HM_{start_date}_to_{end_date}.csv",
#             mime="text/csv",
#         )



# -------------------- CONFIG --------------------
local_tz = ZoneInfo("Asia/Kolkata")
rename_dict = config.get("rename_dict", {})
rename_values = list(rename_dict.values())
rename_set = set(rename_values)   # faster filtering
keep_cols = config.get("keep_cols", [])

service = MlDatasetService()

# -------------------- HELPERS --------------------
def clean_df(df):
    """Rename + keep only allowed columns."""
    df = df.rename(columns=rename_dict)
    return df[[c for c in df.columns if c in rename_set]]

def align_dist(df_dist, df1, df2):
    """Align distribution with RM + HM timestamps."""
    idx = df1.index.union(df2.index)
    return df_dist.reindex(idx).sort_index().ffill()

def fetch_old(start, end):
    df = service.fetch_step1(start, end, allowed_columns=rename_dict)
    return clean_df(df)

def fetch_new_rm(start, end, mode):
    df = service.fetch_step2(start, end, mode, allowed_columns=rename_dict)
    return clean_df(df)

def fetch_hot(start, end):
    df = service.fetch_hotmetal_hourly(
        start, end, keep_columns=keep_cols, interval_minutes=60
    )
    return clean_df(df)

def fetch_dist(start, end):
    df = service.fetch_distribution_data(start, end)
    return clean_df(df)

# -------------------- UI --------------------
st.header("📄 ML Dataset")

with st.form("ml_form"):
    st.subheader("Select Time Range")

    rm_choice = st.radio(
        "Select RM Dataset",
        ["RM Charge", "RM DPR"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now(local_tz).date())
    with col2:
        end_date = st.date_input("End Date", datetime.now(local_tz).date())

    submitted = st.form_submit_button("Fetch Dataset")
status_box = st.empty()

# -------------------- PROCESSING --------------------
if submitted:

    if start_date > end_date:
        st.error("❌ Start Date cannot be after End Date.")
        st.stop()

    CUTOFF = service.cutoff_date
    mode = "charge" if rm_choice == "RM Charge" else "dpr"

    # ------------------------------------------------------
    # CASE 1 → OLD DATA ONLY
    # ------------------------------------------------------
    if end_date <= CUTOFF:
        status_box.info("Fetching OLD DATA (ML only)")
        df_final = fetch_old(start_date, end_date)

    # ------------------------------------------------------
    # CASE 2 → NEW DATA ONLY (RM + HOTMETAL + DIST)
    # ------------------------------------------------------
    elif start_date > CUTOFF:
        status_box.info("Fetching NEW DATA (ML+ HM + Distribution)")

        df_step2 = fetch_new_rm(start_date, end_date, mode)
        df_hot   = fetch_hot(start_date, end_date)
        df_dist  = fetch_dist(start_date, end_date)

        df_dist = align_dist(df_dist, df_step2, df_hot)

        df_final = df_step2.join([df_hot, df_dist], how="outer").sort_index()

    # ------------------------------------------------------
    # CASE 3 → MIXED RANGE (OLD ML + NEW RM/HM)
    # ------------------------------------------------------
    else:
        status_box.info("Fetching MIXED RANGE (OLD ML + NEW ML + HM + Distribution)")

        # OLD PART
        df_old = fetch_old(start_date, CUTOFF)

        # NEW PART
        new_start = CUTOFF + timedelta(days=1)

        df_step2 = fetch_new_rm(new_start, end_date, mode)
        df_hot   = fetch_hot(new_start, end_date)
        df_dist  = fetch_dist(new_start, end_date)

        df_dist = align_dist(df_dist, df_step2, df_hot)

        # Merge new RM + HM + Distribution
        df_new = df_step2.join([df_hot, df_dist], how="outer").sort_index()

        # Combine OLD + NEW
        df_final = pd.concat([df_old, df_new]).sort_index()

    # ------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------
    status_box.empty()
    if df_final.empty:
        st.warning("No data found for the selected range and configuration.")
    else:
        df_final.index.name = "time"

        st.subheader("📊 ML DATASET")
        st.dataframe(df_final)

        st.download_button(
            "Download CSV",
            df_final.to_csv(index=True).encode("utf-8"),
            file_name=f"unified_ML_RM_HM_{start_date}_to_{end_date}.csv",
            mime="text/csv",
        )

# ----- HOT METAL AND SLAG -----

st.header("📄 HOT METAL AND SLAG")

# ----- FORM -----
with st.form("hotmetal_form_2"):
    col1, col2 = st.columns([1, 1])
    with col1:
        from_date = st.date_input("From Date")
    with col2:
        to_date = st.date_input("To Date")

    interval_min = st.number_input(
        "Interval (minutes)",
        min_value=1,
        max_value=600,
        value=60
    )

    fetch_btn = st.form_submit_button("Fetch HM & SLAG DATA")

# ----- ACTION -----
if fetch_btn:
    if from_date > to_date:
        st.error("❌ From Date must be less than or equal to To Date.")
        st.stop()

    keep_cols = config.get("keep_cols", [])

    # ---- CALL DOMAIN SERVICE ----
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

    # --- CSV Download ---
    st.download_button(
        "Download CSV",
        df_final.to_csv().encode("utf-8"),
        file_name=f"hotmetal_{from_date}_to_{to_date}_{interval_min}min.csv",
        mime="text/csv",
    )




# # ----- HOT METAL AND SLAG -----

# st.header("📄 HOT METAL AND SLAG")

# # ----- FORM -----
# with st.form("hotmetal_form_2"):
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         from_date = st.date_input("From Date")
#     with col2:
#         to_date = st.date_input("To Date")

#     interval_min = st.number_input("Interval (minutes)", min_value=1, max_value=600, value=60)

#     fetch_btn = st.form_submit_button("Fetch HM & SLAG DATA")

# if fetch_btn:
#     if from_date > to_date:
#         st.error("❌ From Date must be less than or equal to To Date.")
#         st.stop()

#     keep_cols = config.get("keep_cols", [])

#     # --- Convert to timezone-aware timestamps ---
#     from_dt = pd.Timestamp(from_date).tz_localize("Asia/Kolkata")
#     to_dt   = pd.Timestamp(to_date).tz_localize("Asia/Kolkata") + pd.Timedelta(days=1)

#     # fetch one extra day before for smooth interpolation
#     fetch_start = from_dt - pd.Timedelta(days=1)
#     fetch_end   = to_dt

#     # --- Convert to UTC for InfluxDB ---
#     fetch_start_utc = fetch_start.tz_convert("UTC")
#     fetch_end_utc   = fetch_end.tz_convert("UTC")

#     df = dr.fetch_offline_data(
#         measurement="hotmetal_slag_updated_data",
#         time_range=(fetch_start_utc, fetch_end_utc),
#         database="bf2_evonith_offline_utc",
#     )

#     if df.empty:
#         st.warning("No data found.")
#         st.stop()

#     # --- Cleanup ---
#     df.index = df.index.tz_convert("Asia/Kolkata")
#     df = df.sort_index().loc[~df.index.duplicated(keep="last")]
#     df = df[[c for c in keep_cols if c in df.columns]]

#     numeric_cols = df.columns

#     # ------------ CREATE TARGET RANGE ------------
#     target_index = pd.date_range(
#         start=from_dt,
#         end=to_dt,
#         freq=f"{interval_min}min",
#         tz="Asia/Kolkata"
#     )

#     # ------------ MERGE RAW + TARGET ------------
#     combined_index = df.index.union(target_index)

#     df2 = df.reindex(combined_index)

#     # ------------ INTERPOLATE DATA ------------
#     df2[numeric_cols] = df2[numeric_cols].interpolate("time")

#     # ------------ SELECT EXACT TARGET POINTS ------------
#     df_final = df2.loc[target_index]

#     # ------------ Handle ToDate = Today ------------
#     today = pd.Timestamp.now(tz="Asia/Kolkata").date()
#     if to_date == today:
#         now = pd.Timestamp.now(tz="Asia/Kolkata")
#         cutoff = now.floor(f"{interval_min}min")
#         df_final = df_final.loc[from_dt:cutoff]
#     # ---- REMOVE TIMEZONE FROM INDEX ----
#     df_final.index = df_final.index.tz_localize(None)

#     st.success("Data processed successfully!")
#     st.dataframe(df_final)

#     # --- CSV Download ---
#     st.download_button(
#         "Download CSV",
#         df_final.to_csv().encode("utf-8"),
#         file_name=f"hotmetal_{from_date}_to_{to_date}_{interval_min}min.csv",
#         mime="text/csv",
#     )


