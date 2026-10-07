"""Renderer for the V-Board furnace visualisations section."""

from __future__ import annotations

from datetime import datetime, timedelta

import plotly.graph_objs as go
import pytz
import streamlit as st

from data.fetchers.circular_temperature_contour_data_fetcher import (
    CircumferentialTemperatureDataFetcher,
)
from data.fetchers.longitudinal_temperature_contour_data_fetcher import (
    LongitudinalTemperatureDataFetcher,
)
from data.fetchers.ts_heatload_data_fetcher import TimeSeriesHeatLoadDataFetcher
from furnace_data.domain.heatload import AverageHeatLoadDataFetcher
from plotters.circumferential_contour import CircumferentialPlotter
from plotters.longitudinal_temp_contour import LongitudinalTemperaturePlotter


@st.cache_data(ttl=300, show_spinner=False)
def _circ_fig(field_values, titles, colorbar_title, resolution=36):
    plotter = CircumferentialPlotter(mask_file="mask_circular.pkl")
    return plotter.plot_circumferential_quadrants(
        field_values,
        titles=titles,
        colorbar_title=colorbar_title,
        unit="",
        resolution=resolution,
    )


@st.cache_data(ttl=300, show_spinner=False)
def _long_fig(temperatures, temperatures_max, temperatures_min):
    plotter = LongitudinalTemperaturePlotter(mask_file="mask_longitudinal.pkl")
    return plotter.plot_plotly(temperatures, temperatures_max, temperatures_min)


def render_visualisations() -> None:
    """Render the existing contour and heat-load visualisation experience."""
    st.markdown(
        """
<style>
/* V-Board visualisations spacing; injected only for this section. */
div[data-testid="stVerticalBlock"] { gap: 0rem !important; }
div[data-testid="element-container"] { margin: 0 !important; padding: 0 !important; }
div[data-testid="stPlotlyChart"] { margin: 0 !important; padding: 0 !important; }
.block-container { padding-top: 2rem !important; padding-bottom: 0rem !important; }
.block-container h1:first-of-type { margin-bottom: 1rem !important; }
</style>
""",
        unsafe_allow_html=True,
    )

    timezone_ist = pytz.timezone("Asia/Kolkata")
    now_ist = datetime.now(timezone_ist)

    st.title("Furnace Temperature Data Visualization")
    expander = st.expander("Set date and time", key="vboard_datetime")
    with expander:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            from_date = st.date_input(
                "From Date:",
                value=now_ist.date() - timedelta(days=1),
                key="left date for Longitudinal contour plot",
            )
        with col2:
            from_time = st.time_input(
                "From Time:",
                value=now_ist.time(),
                key="left time for Longitudinal contour plot",
            )
        with col3:
            to_date = st.date_input(
                "To Date:",
                value=now_ist.date(),
                key="right date for Longitudinal contour plot",
            )
        with col4:
            to_time = st.time_input(
                "To Time:",
                value=now_ist.time(),
                key="right time for Longitudinal contour plot",
            )

        start_time_local = (
            timezone_ist.localize(datetime.combine(from_date, from_time))
            if from_date and from_time
            else None
        )
        end_time_local = (
            timezone_ist.localize(datetime.combine(to_date, to_time))
            if to_date and to_time
            else None
        )
        start_time_utc = (
            start_time_local.astimezone(pytz.utc) if start_time_local else None
        )
        end_time_utc = end_time_local.astimezone(pytz.utc) if end_time_local else None

        if (
            start_time_utc is None
            or end_time_utc is None
            or start_time_utc >= end_time_utc
        ):
            st.error(
                "Invalid time range: 'From' datetime must be earlier than 'To' datetime."
            )
            st.stop()

    contour_options = [
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
    timeseries_options = [
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
    with st.sidebar:
        st.markdown("### Contour - Options")
        time_interval = st.selectbox(
            "Select Averaging/Display Interval:",
            contour_options,
            key="time interval for plots",
        )
        st.markdown("### Time Series - Options")
        time_interval_4 = st.selectbox(
            "TimeSeries - Select Interval:",
            timeseries_options,
            key="time interval for ts plot",
        )

        st.markdown("Or select a Range")
        col1, col2 = st.columns(2)
        with col1:
            ts_from_date = st.date_input(
                "From Date:",
                value=now_ist.date() - timedelta(days=1),
                key="left date input for heatload ts data",
            )
            ts_from_time = st.time_input(
                "From Time:",
                value=now_ist.time(),
                key="left time input for heatload ts data",
            )
        with col2:
            ts_to_date = st.date_input(
                "To Date:",
                value=now_ist.date(),
                key="right date input for heatload ts data",
            )
            ts_to_time = st.time_input(
                "To Time:",
                value=now_ist.time(),
                key="right time input for heatload ts data",
            )

        start_time = (
            datetime.combine(ts_from_date, ts_from_time)
            if ts_from_date and ts_from_time
            else None
        )
        end_time = (
            datetime.combine(ts_to_date, ts_to_time)
            if ts_to_date and ts_to_time
            else None
        )

    longitudinal_fetcher = LongitudinalTemperatureDataFetcher(
        debug=False, source="Historical"
    )
    try:
        temperature_list = longitudinal_fetcher.fetch_averaged_data(
            time_interval,
            start_time_utc,
            end_time_utc,
            request_type="avg-min-max",
            window_by=None,
        )
    except ValueError as error:
        st.error(f"Error: {error}")
        st.stop()

    temperatures = [temperature_list[0][i] for i in range(4)]
    temperatures_max = [temperature_list[1][i] for i in range(4)]
    temperatures_min = [temperature_list[2][i] for i in range(4)]
    figure = _long_fig(
        tuple(temperatures), tuple(temperatures_max), tuple(temperatures_min)
    )
    st.plotly_chart(figure, width="stretch", key="data_vis_longitudinal_temp")

    st.title("Circumferential HeatLoad")
    heatload_fetcher = AverageHeatLoadDataFetcher(debug=False, source="historical")
    rows = ["R6", "R7", "R8", "R9", "R10"]
    try:
        heatloads_list = [
            heatload_fetcher.fetch_averaged_data(
                time_interval,
                start_time_utc,
                end_time_utc,
                row,
                request_type="avg-min-max",
                window_by=None,
            )
            for row in rows
        ]
    except ValueError as error:
        st.error(f"Error: {error}")
        st.stop()

    figure = _circ_fig(
        [tuple(map(tuple, row_values)) for row_values in heatloads_list],
        tuple(rows),
        "Heatload (GJ)",
    )
    st.plotly_chart(figure, width="stretch", key="data_vis_circ_heatload")

    st.title("Circumferential Temperature")
    circum_fetcher = CircumferentialTemperatureDataFetcher(
        debug=False, source="historical"
    )
    try:
        circum_temps = circum_fetcher.fetch_averaged_data(
            time_interval,
            start_time_utc,
            end_time_utc,
            request_type="avg-min-max",
            window_by=None,
        )
    except ValueError as error:
        st.error(f"Error: {error}")
        st.stop()

    temp_to_plot = [
        circum_temps["9105"],
        circum_temps["12975"],
        circum_temps["15162"],
        circum_temps["18660"],
    ]
    temp_to_plot2 = [
        circum_temps["4373"],
        circum_temps["5411"],
        circum_temps["5757"],
        circum_temps["6103"],
    ]
    temp_to_plot3 = [
        circum_temps["6795"],
        circum_temps["7565"],
        circum_temps["8335"],
    ]
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
    all_titles = [f"At {elevation}" for elevation in elevations] + preset_titles
    circum_title = "Temperature (°C)"

    figure = _circ_fig(
        [tuple(map(tuple, values)) for values in temp_to_plot],
        tuple(all_titles[-4:]),
        circum_title,
    )
    st.plotly_chart(figure, width="stretch", key="data_vis_circ_temp_stack")
    figure = _circ_fig(
        [tuple(map(tuple, values)) for values in temp_to_plot2],
        tuple(all_titles[:5]),
        circum_title,
    )
    st.plotly_chart(figure, width="stretch", key="data_vis_circ_temp_hearth")
    figure = _circ_fig(
        [tuple(map(tuple, values)) for values in temp_to_plot3],
        tuple(all_titles[5:8]),
        circum_title,
    )
    st.plotly_chart(figure, width="stretch", key="data_vis_circ_temp_tuyere")

    st.title("Heat Load Data - Timeseries")
    timeseries_fetcher = TimeSeriesHeatLoadDataFetcher(debug=False, source="historical")
    row = st.selectbox("Select Row", rows)
    frame = timeseries_fetcher.fetch_data(
        time_interval=time_interval_4,
        start_time=start_time,
        end_time=end_time,
        row=row,
        quadrant=None,
        request_type="ts",
        window_by=None,
    )
    frame.sort_index(inplace=True)

    data = [
        go.Scatter(
            x=frame.index,
            y=frame[f"Heat load {row} Q{quadrant}"],
            xaxis="x",
            yaxis="y" if quadrant == 1 else f"y{quadrant}",
            name=f"{row} Q{quadrant}",
        )
        for quadrant in range(1, 5)
    ]
    layout = go.Layout(
        title="Heat Load Over Time",
        xaxis=dict(title="Time"),
        yaxis=dict(title="", range=[0, 1]),
        yaxis2=dict(title="", range=[0, 1], overlaying="y", side="right"),
        yaxis3=dict(title="", range=[0, 1], overlaying="y", side="left"),
        yaxis4=dict(title="", range=[0, 1], overlaying="y", side="right"),
    )
    st.plotly_chart(
        go.Figure(data=data, layout=layout),
        width="stretch",
        key="data_vis_heatload_ts",
    )
