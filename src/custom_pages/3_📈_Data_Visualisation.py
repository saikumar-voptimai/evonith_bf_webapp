import pytz
import streamlit as st
from datetime import datetime, timedelta
from data_fetchers.ts_heatload_data_fetcher import TimeSeriesHeatLoadDataFetcher
from data_fetchers.longitudinal_temperature_contour_data_fetcher import LongitudinalTemperatureDataFetcher
from data_fetchers.circular_temperature_contour_data_fetcher import CircumferentialTemperatureDataFetcher
from data_fetchers.average_heatload_data_fetcher import AverageHeatLoadDataFetcher
from plotters.circumferential_contour import CircumferentialPlotter
from plotters.longitudinal_temp_contour import LongitudinalTemperaturePlotter

import plotly.graph_objs as go

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
with st.sidebar:
    st.sidebar.markdown("### Contour - Options")
    time_interval = st.selectbox("Select Averaging/Display Interval:", time_options, key="time interval for plots")
    # Define dropdown options
    time_options = [
        'Last 15 minutes', 'Last 30 minutes', 'Last 1 hour', 'Last 6 hours',
        'Last 12 hours', 'Last 1 day', 'Last 1 week', 'Last 2 weeks', 'Last 1 month', 'Over Selected Range'
    ]
    st.sidebar.markdown("### Time Series - Options")
    time_interval_4 = st.selectbox("TimeSeries - Select Interval:", time_options, key="time interval for ts plot")

    st.sidebar.markdown("Or select a Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        from_date = st.date_input("From Date:", value=datetime.now().date() - timedelta(days=1), key="left date input for heatload ts data")
        from_time = st.time_input("From Time:", value=datetime.now().time(), key="left time input for heatload ts data")
    with col2:
        to_date = st.date_input("To Date:", value=datetime.now().date(), key="right date input for heatload ts data")
        to_time = st.time_input("To Time:", value=datetime.now().time(), key="right time input for heatload ts data")

    # Combine date and time
    start_time = datetime.combine(from_date, from_time) if from_date and from_time else None
    end_time = datetime.combine(to_date, to_time) if to_date and to_time else None
try:
    temperature_list = data_fetcher.fetch_averaged_data(time_interval, start_time_utc, end_time_utc)
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# Plotting function (assuming it's defined elsewhere)
plotter = LongitudinalTemperaturePlotter(mask_file="mask_longitudinal.pkl")
temperatures = [temperature_list[0][i] for i in range(4)]
temperatures_max = [temperature_list[1][i] for i in range(4)]
temperatures_min = [temperature_list[2][i] for i in range(4)]  # Extracting only the temperature values for Q1-Q4

fig = plotter.plot_plotly(temperatures, temperatures_max, temperatures_min)
st.plotly_chart(fig, use_container_width=True)

#-------------------------------------------------------------------------------------------------------
# 2. Circular Heatload distribution Plot
# Initialize the data fetcher

st.title("Circumferential HeatLoad Distribution")
circum_data_fetcher = AverageHeatLoadDataFetcher(debug=False, source="historical", request_type="average")
rows = ["R6", "R7", "R8", "R9", "R10"]
# Combine titles
try:
    heatloads_list = []
    for row in rows:
        heatloads_list.append(circum_data_fetcher.fetch_averaged_data(time_interval, start_time_utc, end_time_utc, row))
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# User selection
plotter = CircumferentialPlotter(mask_file="mask_circular.pkl")

fig = plotter.plot_circumferential_quadrants(heatloads_list, titles=rows, colorbar_title="Heatload (GJ)", unit="GJ")
st.plotly_chart(fig, use_container_width=True)

#------------------------------------------------------------------------------------------------------
# 3. Circular Temperature Plot
# Initialize the data fetcher

st.title("Circumferential Temperature Distribution")
circum_data_fetcher = CircumferentialTemperatureDataFetcher(debug=False, source="historical", request_type="average")

try:
    temperatures_dict = circum_data_fetcher.fetch_averaged_data(time_interval, start_time_utc, end_time_utc)
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()
# Combine titles
try:
    circum_temps_list = []
    circum_temps_list.append(circum_data_fetcher.fetch_averaged_data(time_interval, start_time_utc, end_time_utc))
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

# Combine preset temperatures with the selected temperature
temp_to_plot = [circum_temps_list[0]['9105'],
                circum_temps_list[0]['12975'],
                circum_temps_list[0]['15162'],
                circum_temps_list[0]['18660']]  # Convert elevation to string without 'm' and '.'

temp_to_plot2 = [circum_temps_list[0]['4373'],
                circum_temps_list[0]['5411'],
                circum_temps_list[0]['5757'],
                circum_temps_list[0]['6103']]  # Convert elevation to string without 'm' and '.'

temp_to_plot3 = [circum_temps_list[0]['6795'],
                circum_temps_list[0]['7565'],
                circum_temps_list[0]['8335']]  # Convert elevation to string without 'm' and '.'


# User selection
plotter = CircumferentialPlotter(mask_file="mask_circular.pkl")
# Corresponding elevations
elevations = ["4.373m", "5.411m", "5.757m", "6.103m", "6.795m", "7.565m", "8.335m", "9.105m"]
preset_titles = ["12.975m - Bosh", "15.162m - Belly", "18.660m - Stack"]

all_titles = [f"At {elevations[i]}" for i in range(len(elevations))] + preset_titles

fig = plotter.plot_circumferential_quadrants(temp_to_plot, titles=all_titles[-4:], colorbar_title="Temperature (°C)", unit="°C")
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
    <style>
    /* Shrink the vertical gap between ANY two elements */
    div[data-testid="stVerticalBlock"]{
        gap:0.rem;          /* default is 1rem */
    }

    /* Optional: make Plotly charts themselves have zero outer margin */
    .stPlotlyChart {
        margin-top:0 !important;
        margin-bottom:0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
fig = plotter.plot_circumferential_quadrants(temp_to_plot2, titles=all_titles[:5], colorbar_title="Temperature (°C)", unit="°C")
st.plotly_chart(fig, use_container_width=True)
fig = plotter.plot_circumferential_quadrants(temp_to_plot3, titles=all_titles[5:8], colorbar_title="Temperature (°C)", unit="°C")
st.plotly_chart(fig, use_container_width=True)
# -------------------------------------------------------------------------------------------------------
st.title("Heat Load Data - Timeseries")
# -------------------------------------------------------------------------------------------------------
# 4. Timeseries Heatload plot
# Initialize the data fetcher
data_fetcher = TimeSeriesHeatLoadDataFetcher(debug=False, source="historical", request_type="ts")
# Dropdown for selecting Row
row = st.selectbox("Select Row", ["R6", "R7", "R8", "R9", "R10"])

# Fetch and process data for all quadrants
df = data_fetcher.fetch_data(time_interval=time_interval_4, start_time=start_time, end_time=end_time, row=row) # random_data=True

layout = dict(
    hoversubplots="axis",
    title=dict(text="Time Series Heat Load Data", x=0.5, xanchor="center"),
    hovermode="x",
    grid=dict(rows=4, columns=1),
)

data = [
    go.Scatter(x=df.index, y=df[f"Heat load {row} Q1"], xaxis="x", yaxis="y", name=f"{row} Q1"),
    go.Scatter(x=df.index, y=df[f"Heat load {row} Q2"], xaxis="x", yaxis="y2", name=f"{row} Q2"),
    go.Scatter(x=df.index, y=df[f"Heat load {row} Q3"], xaxis="x", yaxis="y3", name=f"{row} Q3"),
    go.Scatter(x=df.index, y=df[f"Heat load {row} Q4"], xaxis="x", yaxis="y4", name=f"{row} Q4"),
]

fig = go.Figure(data=data, layout=layout)

# fig.show()
# # Plot the data
# fig = px.line(
#     df,
#     x=df.index,
#     y=df.columns,
#     title=f"Heat Load for {row} - Moving Average over 10 minutes",
# )

# Display the plot
st.plotly_chart(fig, use_container_width=True)