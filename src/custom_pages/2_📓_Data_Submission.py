import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import pytz
from pathlib import Path
from data_fetchers.base_data_fetcher import BaseDataFetcher
from utils.helper_functions_submission import data_retrieval as dr
from config.config_loader import load_config
from datetime import timedelta
from dotenv import load_dotenv

config = load_config("setting_ds_dv.yml")  # Load the configuration file
config_vsense = load_config("setting_vsense.yml")

load_dotenv() 

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
df_filt = df[(pd.to_datetime(df.index, format="%d/%m/%Y %H:%M").date >= from_date) & 
             (pd.to_datetime(df.index, format="%d/%m/%Y %H:%M").date <= to_date)]

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

st.write("Select the features to plot:")
features = st.multiselect('Select features', df.columns, default=df.columns[0])
df_t = df[(pd.to_datetime(df.index, format="%d/%m/%Y %H:%M").date >= from_date) & 
             (pd.to_datetime(df.index, format="%d/%m/%Y %H:%M").date <= to_date)]
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
offline_measurements = {"Bunker Report": "rm_updated_data", 
                        "DPR": "dpr_data", 
                        "HM & Slag": "hotmetal_slag_data"}
if "time_range_off" not in st.session_state:
    st.session_state.time_range_off = "last 1 week"
cols = st.columns(2)
with cols[0]:
    selected_offline = st.selectbox("Select Offline Measurement", list(offline_measurements.keys()))
with cols[1]:
    time_range_off = st.selectbox(
        "Select Time Range:",
        list(TIME_OPTIONS.keys())[7:],
        index=list(TIME_OPTIONS.keys())[7:].index(st.session_state.time_range_off)
    )
    st.session_state.time_range_off = time_range_off

if st.button("Fetch Offline Data"):
    df_offline = dr.fetch_offline_data(offline_measurements[selected_offline],
                                       time_range_off)

    if df_offline.empty:
        st.warning(f"No data found for {selected_offline}")
    else:
        if selected_offline == "rm_data":
            df_offline = dr.clean_rm_data(df_offline)

        st.dataframe(df_offline)
        csv = df_offline.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"{selected_offline}.csv",
            mime="text/csv",
        )