import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import pytz

from pathlib import Path
from data_fetchers.base_data_fetcher import BaseDataFetcher
from src.config.config_loader import load_config
from datetime import timedelta
from dotenv import load_dotenv

config = load_config("setting_ds_dv.yml")  # Load the configuration file

load_dotenv() 


local_tz = pytz.timezone("Asia/Kolkata")  # or use your actual timezone
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

fullpath = Path(__file__).resolve().parents[1] / config['DATA'].split('/')[1] /config['DATA'].split('/')[2]
df = pd.read_csv(fullpath, index_col=0, parse_dates=True)

st.title("Visualisation tool")

with st.expander("How to Use DataWalker", expanded=False):
    st.markdown("""
    **DataWalker** is a tool for interactive exploration of tabular data. You can:
    - Use the **global search** bar to find any specific term across the dataset.
    - Apply filters, sort columns, and navigate through data interactively.
    - Enable or disable columns for streamlined viewing.
    """)

#--------------------------------------------------------------------------
st.subheader("Distribution plots")
cols = st.columns(2)
with cols[0]:
    input_params = [item for sublist in config['Optimisation']['input_params'].values() for item in sublist]
    x = st.selectbox("Select X feature", input_params)

with cols[1]:
    output_params = config['Optimisation']['output_params']
    y = st.selectbox("Select Y feature", output_params)
plot = sns.regplot(data=df, 
                  x=x, 
                  y=y,     
                  ci=99, 
                  marker="x", 
                  color=".3", 
                  line_kws=dict(color="r"),
                  order=3)
fig = plot.get_figure()
fig.set_size_inches(10, 5)
st.pyplot(fig, use_container_width=False)


#--------------------------------------------------------------------------
st.subheader("Timeseries plot")
cols = st.columns(2)
y = []
cols_y = df.columns[1:]
with cols[0]:
    y.append(st.selectbox("Select feature 1", cols_y))
    
with cols[1]:
    remain_cols = cols_y.insert(0, 'None')
    y.append(st.selectbox("Select feature 2", remain_cols))

y = [i for i in y if i != 'None']
# Plot the data
fig = px.line(
    df,
    x=df.index,
    y=y,
    hover_data={df.columns[0]: "|%B %d, %Y"},
    markers=True
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------
# SHOW 6 MEASUREMENTS FROM INFLUXDB
# --------------------------------------------------------------------------------

# load once
_config = load_config()

FIELD_LABELS = {
    internal_key: human_label
    for mapping in _config["data_mapping"].values()
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
    'last 1 month': timedelta(days=30)
}

FREQUENCY_TO_TIMEDTA = {
    "None": None,
    "1 minute": "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "30 minutes": "30mins",
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

datafetchers = {}
for k, v in MEASUREMENT_LABELS.items():
    datafetchers[k] = BaseDataFetcher(k)

def average_data(df, freq):
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    df_avg = df.resample(freq).mean().dropna().reset_index()
    return df_avg

# --- Streamlit UI ---
st.title("📊 BF2 Data Downloader")
if "measurements" not in st.session_state:
    st.session_state.measurements = list(MEASUREMENT_LABELS.keys())
if "selected_measurements" not in st.session_state:
    st.session_state.selected_measurements = set(st.session_state.measurements)
if "avg_mode" not in st.session_state:
    st.session_state.avg_mode = True
if "select_all" not in st.session_state:
    st.session_state.select_all = True

st.subheader("Select Measurements")
col1, col2 = st.columns(2)

with col1:
    select_all = st.toggle("Select All", key="select_all", value=st.session_state.select_all)
    if select_all:
        st.session_state.selected_measurements = set(st.session_state.measurements)
    else:
        # Don't clear, just allow manual unchecking below
        pass

cols = st.columns(4)
for i, meas in enumerate(st.session_state.measurements):
    col = cols[i % 4]
    label = MEASUREMENT_LABELS.get(meas, meas)
    with col:
        checked = st.checkbox(label, value=meas in st.session_state.selected_measurements, key=meas)
        if checked:
            st.session_state.selected_measurements.add(meas)
        else:
            st.session_state.selected_measurements.discard(meas)

if st.session_state.selected_measurements:
    col1, col2 = st.columns(2)
    with col1:
        time_range = st.selectbox("Select Time Range:", list(TIME_OPTIONS.keys()), index=list(TIME_OPTIONS.keys()).index("last 1 hour")).lower()
    with col2:
        average_range = st.selectbox("Select Any Averaging Window:", list(FREQUENCY_TO_TIMEDTA.keys()), index=list(FREQUENCY_TO_TIMEDTA.keys()).index("1 hour")).lower()

    if st.button("⬇️ Show CSV File ", key="download_combined_csv"):
        combined_df = pd.DataFrame()
        for meas in st.session_state.selected_measurements:
            df = datafetchers[meas].fetch_averaged_data(average_by=time_range)
            if df.empty:
                continue
            # Always resample to 1 hour unless user changes it
            if FREQUENCY_TO_TIMEDTA[average_range] is not None:
                df = average_data(df, FREQUENCY_TO_TIMEDTA[average_range])
            else:
                df = average_data(df, "1h")
            df = df.rename(columns={
                col: f"{MEASUREMENT_LABELS.get(meas, meas)} - {FIELD_LABELS.get(col, col)}"
                for col in df.columns if col != "time"
            })
            combined_df = df if combined_df.empty else combined_df.merge(df, on="time", how="outer")
        prefix = "avg_" if (st.session_state.avg_mode and FREQUENCY_TO_TIMEDTA[average_range]) else "raw_"
        all_sel = st.session_state.selected_measurements
        sheet_part = (
            "combined"
            if len(all_sel) == len(st.session_state.measurements)
            else "_".join(sorted(all_sel)).replace(" ", "_")
        )
        file_name = f"{prefix}data_{sheet_part}_{time_range.replace(' ', '_')}.csv"
        if not combined_df.empty:
            combined_df = combined_df.sort_values("time").reset_index(drop=True)
            combined_df.index += 1
            st.dataframe(combined_df)
