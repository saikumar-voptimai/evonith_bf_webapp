import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import pytz
import streamlit as st
from datetime import datetime, timedelta, timezone
from data_fetchers.ts_heatload_data_fetcher import TimeSeriesHeatLoadDataFetcher
from data_fetchers.longitudinal_temperature_contour_data_fetcher import LongitudinalTemperatureDataFetcher
from data_fetchers.circular_temperature_contour_data_fetcher import CircumferentialTemperatureDataFetcher
from data_fetchers.average_heatload_data_fetcher import AverageHeatLoadDataFetcher
from plotters.circumferential_contour import CircumferentialPlotter
from plotters.longitudinal_temp_contour import LongitudinalTemperaturePlotter
from utils.helper_functions_visualisation.plotter_circum_heatloads import plotter_circum
from config.loader import load_config

TIMEZONE = pytz.timezone('Asia/Kolkata')  # GMT+5:30

# ----------------------------------------------------------------------------------------------------------
# 1. Longitudinal Contour Plotter
# Initialize the data fetcher
data_fetcher = LongitudinalTemperatureDataFetcher(debug=False, source="Historical", request_type="average")

# Streamlit UI
st.title("Furnace Temperature Data Visualization")
expander = st.expander("Set date and time")
with expander:
    # Date and time input
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        from_date = st.date_input("From Date:", value=datetime.now().date() - timedelta(days=1), key="left date for Longitudinal contour plot")
    with col2:
        from_time = st.time_input("From Time:", value=datetime.now().time(), key="left time for Longitudinal contour plot")
    with col3:
        to_date = st.date_input("To Date:", value=datetime.now().date(), key="right date for Longitudinal contour plot")
    with col4:
        to_time = st.time_input("To Time:", value=datetime.now().time(), key="right time for Longitudinal contour plot")

    # Combine date and time - ensure both are provided and convert to UTC
    start_time_local = datetime.combine(from_date, from_time) if from_date and from_time else None
    start_time_utc = start_time_local.astimezone(pytz.utc) if start_time_local else None
    end_time_local = datetime.combine(to_date, to_time) if to_date and to_time else None
    end_time_utc = end_time_local.astimezone(pytz.utc) if start_time_local else None
    
    if start_time_utc >= end_time_utc:
        st.error("Invalid time range: 'From' datetime must be earlier than 'To' datetime.")
        st.stop()

# Define dropdown options
time_options = [
    'Last 5 minutes', 'Last 15 minutes', 'Last 30 minutes', 'Last 1 hour', 'Last 6 hours',
    'Last 12 hours', 'Last 1 day', 'Last 3 days', 'Last 1 week', 'Last 2 weeks', 'Last 1 month', 'Over Selected Range'
]

# User selection
time_interval_1 = st.selectbox("Longitudinal - Select Averaging Interval:", time_options, key="time interval longitudinal contour plot")
try:
    temperature_dict = data_fetcher.fetch_averaged_data(time_interval_1, start_time_utc, end_time_utc)
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# Plotting function (assuming it's defined elsewhere)
plotter = LongitudinalTemperaturePlotter(mask_file="mask_longitudinal.pkl")
temperature_list = [[val[0] for val in temperature_dict[f"Q{1}"].values()],
                    [val[0] for val in temperature_dict[f"Q{2}"].values()],
                    [val[0] for val in temperature_dict[f"Q{3}"].values()],
                    [val[0] for val in temperature_dict[f"Q{4}"].values()]]

fig = plotter.plot(temperature_list)
if any(val == 0 for val in temperature_list[0]):
    st.warning("No valid temperature data available in the selected time range. Please try a different time range or check the data source.")

st.pyplot(fig, use_container_width=True)
st.markdown('-----------------------------------------------------------------------------------------')
#-------------------------------------------------------------------------------------------------------
# 2. Circumferential Contour Plotter - Heatload
# Circular Heat Load Plot
st.subheader("Heat Load Distribution")
st.markdown("Compares the average heat load distribution at a particular stave Row.")
heatload_fetcher = AverageHeatLoadDataFetcher(debug=False, source="historical")

# Corresponding elevations
rows = ["R6", "R7", "R8", "R9", "R10"]
heatload_row = st.selectbox("Select the stave row:", rows, key="row selection circular heatload plot")


expander = st.expander("Set date and time")
with expander:
    # Date and time input
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        from_date = st.date_input("From Date:", value=datetime.now().date() - timedelta(days=1), key="left date for circular heatload contour plot")
    with col2:
        from_time = st.time_input("From Time:", value=datetime.now().time(), key="left time for circular heatload contour plot")
    with col3:
        to_date = st.date_input("To Date:", value=datetime.now().date(), key="right date for circular heatload contour plot")
    with col4:
        to_time = st.time_input("To Time:", value=datetime.now().time(), key="right time for circular heatload contour plott")

    # Combine date and time
    start_time = datetime.combine(from_date, from_time) if from_date and from_time else None
    end_time = datetime.combine(to_date, to_time) if to_date and to_time else None

cols = st.columns(2)
with cols[0]:
    time_interval_2a = st.selectbox("Circular Heatload - Select Averaging Interval 1:", time_options, key="time interval left circular heatload plot")
with cols[1]:
    time_interval_2b = st.selectbox("Circular Heatload - Select Averaging Interval 2:", time_options, key="time interval right circular heatload plot")

cols = st.columns(2)
with cols[0]:
    try:
        heatloads_dict = heatload_fetcher.fetch_averaged_data(time_interval_2a, start_time, end_time, heatload_row)
        fig_circular, ax_circular = plt.subplots(dpi=120, subplot_kw=dict(projection='polar'))
        fig_circular = plotter_circum(list(heatloads_dict.values()), fig_circular, ax_circular, title=None)
        st.pyplot(fig_circular, use_container_width=False)
    except Exception as e:
        st.error(f"Failed to fetch or plot heat load data: {e}")
with cols[1]:
    try:
        heatloads_dict = heatload_fetcher.fetch_averaged_data(time_interval_2b, start_time, end_time, heatload_row)
        fig_circular, ax_circular = plt.subplots(dpi=120, subplot_kw=dict(projection='polar'))
        fig_circular = plotter_circum(list(heatloads_dict.values()), fig_circular, ax_circular, title=None)
        st.pyplot(fig_circular, use_container_width=False)
    except Exception as e:
        st.error(f"Failed to fetch or plot heat load data: {e}")
#------------------------------------------------------------------------------------------------------
# 3. Circular Temperature Plot
# Initialize the data fetcher

st.title("Circumferential Temperature Distribution")
circum_data_fetcher = CircumferentialTemperatureDataFetcher(debug=False, source="historical", request_type="average")

# Define dropdown options
time_options = [
    'Last 5 minutes', 'Last 15 minutes', 'Last 30 minutes', 'Last 1 hour', 'Last 6 hours',
    'Last 12 hours', 'Last 1 day', 'Last 1 week', 'Last 2 weeks', 'Last 1 month', 'Over Selected Range'
]

# Corresponding elevations
elevations = ["4.373m", "5.411m", "5.757m", "6.103m", "6.795m", "7.565m", "8.335m", "9.105m"]
preset_titles = ["12.975m - Bosh", "15.162m - Belly", "18.660m - Stack"]

cols = st.columns(2)

# User selection
with cols[0]:
    time_interval_3 = st.selectbox("Circumferential - Select Averaging Interval:", time_options, key="time interval circular temperature plot")
with cols[1]:
    # Dropdown for selecting the 4th elevation
    selected_elevation = st.selectbox(
        "Select Elevation for the 4th Contour Plot:",
        options=elevations,
        format_func=lambda x: x.replace("mm", "")
    )

# Combine titles
all_titles = preset_titles + [f"At {selected_elevation}"]
try:
    temperatures_dict = circum_data_fetcher.fetch_averaged_data(time_interval_3, start_time, end_time)
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# Combine preset temperatures with the selected temperature
temp_to_plot = [temperatures_dict[i][0] for i in preset_titles] + [temperatures_dict[selected_elevation][0]]

# User selection
plotter = CircumferentialPlotter(mask_file="mask_circular.pkl")
fig = plotter.plot(temp_to_plot, titles=all_titles)
if any(temp == 0 for temp in temp_to_plot[0]):
    st.warning("No valid temperature data available in the selected time range. Please try a different time range or check the data source.")

st.pyplot(fig, use_container_width=True)
st.markdown('-----------------------------------------------------------------------------------------')

# #-------------------------------------------------------------------------------------------------------

st.title("Heat Load Data - Timeseries")
# -------------------------------------------------------------------------------------------------------
# 4. Timeseries Heatload plot
# Initialize the data fetcher
data_fetcher = TimeSeriesHeatLoadDataFetcher(debug=False, source="historical", request_type="ts")
# Dropdown for selecting Row
row = st.selectbox("Select Row", ["R6", "R7", "R8", "R9", "R10"])

with st.expander("Set date and time"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        from_date = st.date_input("From Date:", value=datetime.now().date() - timedelta(days=1), key="left date input for heatload ts data")
    with col2:
        from_time = st.time_input("From Time:", value=datetime.now().time(), key="left time input for heatload ts data")
    with col3:
        to_date = st.date_input("To Date:", value=datetime.now().date(), key="right date input for heatload ts data")
    with col4:
        to_time = st.time_input("To Time:", value=datetime.now().time(), key="right time input for heatload ts data")

    # Combine date and time
    start_time = datetime.combine(from_date, from_time) if from_date and from_time else None
    end_time = datetime.combine(to_date, to_time) if to_date and to_time else None

# Define dropdown options
time_options = [
    'Last 15 minutes', 'Last 30 minutes', 'Last 1 hour', 'Last 6 hours',
    'Last 12 hours', 'Last 1 day', 'Last 1 week', 'Last 2 weeks', 'Last 1 month', 'Over Selected Range'
]
time_interval_4 = st.selectbox("TimeSeries - Select Interval:", time_options, key="time interval for ts plot")
# Fetch and process data for all quadrants
df = data_fetcher.fetch_data(time_interval=time_interval_4, start_time=start_time, end_time=end_time, row=row) # random_data=True

# Plot the data
fig = px.line(
    df,
    x=df.index,
    y=df.columns,
    title=f"Heat Load for {row} - Moving Average over 10 minutes",
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)
