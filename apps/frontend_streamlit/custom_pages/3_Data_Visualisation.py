from datetime import datetime, timedelta

import plotly.graph_objs as go
import pytz
import streamlit as st

from furnace_data.domain.heatload import AverageHeatLoadDataFetcher
from apps.frontend_streamlit.data.fetchers.circular_temperature_contour_data_fetcher import (
    CircumferentialTemperatureDataFetcher,
)
from apps.frontend_streamlit.data.fetchers.longitudinal_temperature_contour_data_fetcher import (
    LongitudinalTemperatureDataFetcher,
)
from apps.frontend_streamlit.data.fetchers.ts_heatload_data_fetcher import TimeSeriesHeatLoadDataFetcher
from apps.frontend_streamlit.plotters.circumferential_contour import CircumferentialPlotter
from apps.frontend_streamlit.plotters.longitudinal_temp_contour import LongitudinalTemperaturePlotter

# ── Cached figure helpers — re-render only when data or params change ─────────

@st.cache_data(ttl=300, show_spinner=False)
def _circ_fig(field_values, titles, colorbar_title, resolution=36):
    plotter = CircumferentialPlotter(mask_file="mask_circular.pkl")
    return plotter.plot_circumferential_quadrants(
        field_values, titles=titles,
        colorbar_title=colorbar_title, unit="",
        resolution=resolution,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _long_fig(temperatures, temperatures_max, temperatures_min):
    plotter = LongitudinalTemperaturePlotter(mask_file="mask_longitudinal.pkl")
    return plotter.plot_plotly(temperatures, temperatures_max, temperatures_min)

TIMEZONE = pytz.timezone("Asia/Kolkata")  # GMT+5:30

st.markdown(
    """
<style>
/* Remove default vertical spacing between blocks */
div[data-testid="stVerticalBlock"] { gap: 0rem !important; }

/* Remove extra spacing around each element */
div[data-testid="element-container"] { margin: 0 !important; padding: 0 !important; }

/* Plotly chart wrapper */
div[data-testid="stPlotlyChart"] { margin: 0 !important; padding: 0 !important; }

/* Page padding */
.block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
.block-container h1:first-of-type {
    margin-bottom: 1rem !important;
}

</style>
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------------------------------------
# 1. Longitudinal Contour Plotter
# Initialize the data fetcher
data_fetcher = LongitudinalTemperatureDataFetcher(debug=False, source="Historical")

# Streamlit UI
st.title("Furnace Temperature Data Visualization")
expander = st.expander("Set date and time", key="vboard_datetime")
with expander:
    # Date and time input
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        from_date = st.date_input(
            "From Date:",
            value=datetime.now().date() - timedelta(days=1),
            key="left date for Longitudinal contour plot",
        )
    with col2:
        from_time = st.time_input(
            "From Time:",
            value=datetime.now().time(),
            key="left time for Longitudinal contour plot",
        )
    with col3:
        to_date = st.date_input(
            "To Date:",
            value=datetime.now().date(),
            key="right date for Longitudinal contour plot",
        )
    with col4:
        to_time = st.time_input(
            "To Time:",
            value=datetime.now().time(),
            key="right time for Longitudinal contour plot",
        )

    # Combine date and time - ensure both are provided and convert to UTC
    start_time_local = (
        datetime.combine(from_date, from_time) if from_date and from_time else None
    )
    start_time_utc = start_time_local.astimezone(pytz.utc) if start_time_local else None
    end_time_local = datetime.combine(to_date, to_time) if to_date and to_time else None
    end_time_utc = end_time_local.astimezone(pytz.utc) if start_time_local else None

    if start_time_utc >= end_time_utc:
        st.error(
            "Invalid time range: 'From' datetime must be earlier than 'To' datetime."
        )
        st.stop()

# Define dropdown options
time_options = [
    "Last 5 minutes",
    "Last 15 minutes",
    "Last 30 minutes",
    "Last 1 hour",
    "Last 6 hours",
    "Last 12 hours",
    "Last 1 day",
    "Last 3 days",
    "Last 1 week",
    "Last 2 weeks",
    "Last 1 month",
    "Over Selected Range",
]
with st.sidebar:
    st.sidebar.markdown("### Contour - Options")
    time_interval = st.selectbox(
        "Select Averaging/Display Interval:",
        time_options,
        key="time interval for plots",
    )
    # Define dropdown options
    time_options = [
        "Last 15 minutes",
        "Last 30 minutes",
        "Last 1 hour",
        "Last 6 hours",
        "Last 12 hours",
        "Last 1 day",
        "Last 1 week",
        "Last 2 weeks",
        "Last 1 month",
        "Over Selected Range",
    ]
    st.sidebar.markdown("### Time Series - Options")
    time_interval_4 = st.selectbox(
        "TimeSeries - Select Interval:", time_options, key="time interval for ts plot"
    )

    st.sidebar.markdown("Or select a Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        from_date = st.date_input(
            "From Date:",
            value=datetime.now().date() - timedelta(days=1),
            key="left date input for heatload ts data",
        )
        from_time = st.time_input(
            "From Time:",
            value=datetime.now().time(),
            key="left time input for heatload ts data",
        )
    with col2:
        to_date = st.date_input(
            "To Date:",
            value=datetime.now().date(),
            key="right date input for heatload ts data",
        )
        to_time = st.time_input(
            "To Time:",
            value=datetime.now().time(),
            key="right time input for heatload ts data",
        )

    # Combine date and time
    start_time = (
        datetime.combine(from_date, from_time) if from_date and from_time else None
    )
    end_time = datetime.combine(to_date, to_time) if to_date and to_time else None
try:
    temperature_list = data_fetcher.fetch_averaged_data(
        time_interval,
        start_time_utc,
        end_time_utc,
        request_type="avg-min-max",
        window_by=None,
    )
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

temperatures     = [temperature_list[0][i] for i in range(4)]
temperatures_max = [temperature_list[1][i] for i in range(4)]
temperatures_min = [temperature_list[2][i] for i in range(4)]

fig = _long_fig(tuple(temperatures), tuple(temperatures_max), tuple(temperatures_min))
st.plotly_chart(fig, width="stretch", key="data_vis_longitudinal_temp")

# -------------------------------------------------------------------------------------------------------
# 2. Circular Heatload distribution Plot
# Initialize the data fetcher

st.title("Circumferential HeatLoad")
circum_data_fetcher = AverageHeatLoadDataFetcher(debug=False, source="historical")
rows = ["R6", "R7", "R8", "R9", "R10"]
# Combine titles
try:
    heatloads_list = []
    for row in rows:
        heatloads_list.append(
            circum_data_fetcher.fetch_averaged_data(
                time_interval,
                start_time_utc,
                end_time_utc,
                row,
                request_type="avg-min-max",
                window_by=None,
            )
        )
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()

fig = _circ_fig(
    [tuple(map(tuple, r)) for r in heatloads_list],
    tuple(rows),
    "Heatload (GJ)",
)
st.plotly_chart(fig, width="stretch", key="data_vis_circ_heatload")

# ------------------------------------------------------------------------------------------------------
# 3. Circular Temperature Plot
# Initialize the data fetcher

st.title("Circumferential Temperature")
circum_data_fetcher = CircumferentialTemperatureDataFetcher(
    debug=False, source="historical"
)

try:
    circum_temps = circum_data_fetcher.fetch_averaged_data(
        time_interval,
        start_time_utc,
        end_time_utc,
        request_type="avg-min-max",
        window_by=None,
    )
except ValueError as e:
    st.error(f"Error: {e}")
    st.stop()
circum_temps_list = [circum_temps]

# Combine preset temperatures with the selected temperature
temp_to_plot = [
    circum_temps_list[0]["9105"],
    circum_temps_list[0]["12975"],
    circum_temps_list[0]["15162"],
    circum_temps_list[0]["18660"],
]  # Convert elevation to string without 'm' and '.'

temp_to_plot2 = [
    circum_temps_list[0]["4373"],
    circum_temps_list[0]["5411"],
    circum_temps_list[0]["5757"],
    circum_temps_list[0]["6103"],
]  # Convert elevation to string without 'm' and '.'

temp_to_plot3 = [
    circum_temps_list[0]["6795"],
    circum_temps_list[0]["7565"],
    circum_temps_list[0]["8335"],
]  # Convert elevation to string without 'm' and '.'


# Corresponding elevations
elevations = [
    "4.373m",
    "5.411m",
    "5.757m",
    "6.103m",
    "6.795m",
    "7.565m",
    "8.335m",
    "9.105m",
]
preset_titles = ["12.975m - Bosh", "15.162m - Belly", "18.660m - Stack"]

all_titles = [f"At {elevations[i]}" for i in range(len(elevations))] + preset_titles

_circ_title = "Temperature (°C)"
fig = _circ_fig(
    [tuple(map(tuple, r)) for r in temp_to_plot],
    tuple(all_titles[-4:]),
    _circ_title,
)
st.plotly_chart(fig, width="stretch", key="data_vis_circ_temp_stack")
fig = _circ_fig(
    [tuple(map(tuple, r)) for r in temp_to_plot2],
    tuple(all_titles[:5]),
    _circ_title,
)
st.plotly_chart(fig, width="stretch", key="data_vis_circ_temp_hearth")
fig = _circ_fig(
    [tuple(map(tuple, r)) for r in temp_to_plot3],
    tuple(all_titles[5:8]),
    _circ_title,
)
st.plotly_chart(fig, width="stretch", key="data_vis_circ_temp_tuyere")
# -------------------------------------------------------------------------------------------------------
st.title("Heat Load Data - Timeseries")
# -------------------------------------------------------------------------------------------------------
# 4. Timeseries Heatload plot
data_fetcher = TimeSeriesHeatLoadDataFetcher(debug=False, source="historical")
# Dropdown for selecting Row
row = st.selectbox("Select Row", ["R6", "R7", "R8", "R9", "R10"])

# Fetch and process data for all quadrants
df = data_fetcher.fetch_data(
    time_interval=time_interval_4,
    start_time=start_time,
    end_time=end_time,
    row=row,
    quadrant=None,
    request_type="ts",
    window_by=None,
)

df.sort_index(inplace=True)

data = [
    go.Scatter(
        x=df.index, y=df[f"Heat load {row} Q1"], xaxis="x", yaxis="y", name=f"{row} Q1"
    ),
    go.Scatter(
        x=df.index, y=df[f"Heat load {row} Q2"], xaxis="x", yaxis="y2", name=f"{row} Q2"
    ),
    go.Scatter(
        x=df.index, y=df[f"Heat load {row} Q3"], xaxis="x", yaxis="y3", name=f"{row} Q3"
    ),
    go.Scatter(
        x=df.index, y=df[f"Heat load {row} Q4"], xaxis="x", yaxis="y4", name=f"{row} Q4"
    ),
]

layout = go.Layout(
    title="Heat Load Over Time",
    xaxis=dict(title="Time"),
    yaxis=dict(title="", range=[0, 1]),
    yaxis2=dict(
        title="",
        range=[0, 1],  # Set the range for the second y-axis
        overlaying="y",  # Make it overlay the primary y-axis
        side="right",  # Place it on the right side of the plot
    ),
    yaxis3=dict(
        title="",
        range=[0, 1],  # Set the range for the third y-axis
        overlaying="y",  # Overlay the primary y-axis
        side="left",  # Place it on the left side of the plot
    ),
    yaxis4=dict(
        title="",
        range=[0, 1],  # Set the range for the fourth y-axis
        overlaying="y",  # Overlay the primary y-axis
        side="right",  # Place it on the right side of the plot
    ),
)

fig = go.Figure(data=data, layout=layout)

# Display the plot
st.plotly_chart(fig, width='stretch', key="data_vis_heatload_ts")
