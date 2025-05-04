import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import pytz
import streamlit as st
from datetime import datetime, timedelta
from data_fetchers.ts_heatload_data_fetcher import TimeSeriesHeatLoadDataFetcher
from src.data_fetchers.longitudinal_temperature_contour_data_fetcher import LongitudinalTemperatureDataFetcher
from src.data_fetchers.circular_temperature_contour_data_fetcher import CircumferentialTemperatureDataFetcher
from data_fetchers.average_heatload_data_fetcher import AverageHeatLoadDataFetcher
from plotters.circumferential_contour import CircumferentialPlotter
from plotters.longitudinal_temp_contour import LongitudinalTemperaturePlotter
from utils.helper_functions_visualisation.plotter_circum_heatloads import plotter_circum
from config.loader import load_config

TIMEZONE = pytz.timezone('Asia/Kolkata')  # GMT+5:30

st.title("Heat Load Data Visualisation")
# -------------------------------------------------------------------------------------------------------
# 1. Timeseries Heatload plot
# Initialize the data fetcher
data_fetcher = TimeSeriesHeatLoadDataFetcher(debug=True)
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

# Fetch and process data for all quadrants
data = data_fetcher.fetch_data(start_time=start_time, end_time=end_time, row=row) # random_data=True
df = pd.DataFrame.from_dict(data[list(data.keys())[0]])

# Plot the data
fig = px.line(
    df,
    x=df['timestamps'],
    y=df.columns,
    title=f"Heat Load for {row}",
    hover_data={"timestamps": "|%B %d, %Y"},
    labels={ "timestamps": "Time", "value": "Heatload", "variable":"Quadrants"},
    markers=True
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------------------------------------
# 2. Longitudinal Contour Plotter
# Initialize the data fetcher
data_fetcher = LongitudinalTemperatureDataFetcher(debug=True)

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

    # Combine date and time
    start_time = datetime.combine(from_date, from_time) if from_date and from_time else None
    end_time = datetime.combine(to_date, to_time) if to_date and to_time else None

    if start_time >= end_time:
        st.error("Invalid time range: 'From' datetime must be earlier than 'To' datetime.")
        st.stop()

# Define dropdown options
time_options = [
    'Live', 'Last 15 minutes', 'Last 1 hour', 'Last 6 hours',
    'Last 12 hours', 'Last 1 day', 'Last 1 week', 'Last 1 month', 'Over Selected Range'
]

# User selection
time_interval_1 = st.selectbox("Longitudinal - Select Averaging Interval:", time_options, key="time interval longitudinal contour plot")
try:
    temperature_dict = data_fetcher.fetch_averaged_data(time_interval_1, start_time, end_time)
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# Plotting function (assuming it's defined elsewhere)
plotter = LongitudinalTemperaturePlotter(mask_file="mask_longitudinal.pkl")
temperature_list = [temperature_dict["Q1"], temperature_dict["Q2"], temperature_dict["Q3"], temperature_dict["Q4"]]
fig = plotter.plot(temperature_list)
st.pyplot(fig, use_container_width=True)
st.markdown('-----------------------------------------------------------------------------------------')
#-------------------------------------------------------------------------------------------------------
# 3. Circumferential Contour Plotter - Heatload
# Circular Heat Load Plot
st.subheader("Heat Load Distribution")
st.markdown("Compares the average heat load distribution at a particular stave Row.")
heatload_fetcher = AverageHeatLoadDataFetcher(debug=True, source="Live")

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
# 4. Circular Temperature Plot
# Initialize the data fetcher

st.title("Circumferential Temperature Distribution")
circum_data_fetcher = CircumferentialTemperatureDataFetcher(debug=True, source="Live")

# Define dropdown options
time_options = [
    'Live', 'Last 1 minute', 'Last 15 minutes', 'Last 1 hour', 'Last 6 hours',
    'Last 12 hours', 'Last 1 day', 'Last 1 week', 'Last 1 month', 'Over Selected Range'
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
temp_to_plot = [temperatures_dict[i] for i in preset_titles] + [temperatures_dict[selected_elevation]]

# User selection
plotter = CircumferentialPlotter(mask_file="mask_circular.pkl")
fig = plotter.plot(temp_to_plot, titles=all_titles)
st.pyplot(fig, use_container_width=True)
st.markdown('-----------------------------------------------------------------------------------------')

# #-------------------------------------------------------------------------------------------------------
