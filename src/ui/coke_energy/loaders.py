"""Streamlit caching for the Coke Energy Model page.

The data assembly lives in ``services.coke_energy`` (#271), shared with the twin-api and the worker. This module only
adds ``st.cache_data`` around each loading step, so the page keeps one spinner per step and copy-on-read results, and
passes those cached steps to the service's composition (one code path for every UI).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import streamlit as st

from services import coke_energy as svc
from services.coke_energy import PLANT_TZ, SAMPLE_LOOKBACK, DataKeys, PageData, data_keys  # noqa: F401 (re-exported)
from utils.coke_energy.pipeline import PipelineSettings
from utils.coke_energy.rist_diagram import RistShift


@st.cache_data(show_spinner="Reading the hourly furnace dataset…", max_entries=2)
def hourly_dataset(path: str, mtime_ns: int) -> pd.DataFrame:
    return svc.load_hourly(path, mtime_ns)


@st.cache_data(
    show_spinner="Reading 10-min online data (the first run fetches about 15 months from InfluxDB, ~1 min)…",
    max_entries=2,
)
def online_data(
    start: str, clock: str, nonce: int, fields_json: str, cache_dir: str, refresh_minutes: float
) -> pd.DataFrame:
    return svc.load_online(start, clock, nonce, fields_json, cache_dir, refresh_minutes)


@st.cache_data(show_spinner="Reading hot metal / slag analyses and the DPR…", max_entries=2)
def neon_data(start: str, clock: str, nonce: int) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    return svc.load_neon(start, clock, nonce)


@st.cache_data(show_spinner="Reading lab-sample times for the freshness check…", max_entries=2)
def lab_samples(start: str, clock: str, nonce: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return svc.load_lab_samples(start, clock, nonce)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def names() -> dict:
    return svc.load_names()


_LOADERS = svc.Loaders(hourly=hourly_dataset, online=online_data, neon=neon_data, lab=lab_samples, names=names)


@st.cache_data(show_spinner="Balancing every shift and running both models…", max_entries=2)
def page_data(keys: DataKeys, settings: PipelineSettings) -> PageData:
    return svc.build_page_data(keys, settings, _LOADERS)


@st.cache_data(show_spinner="Drawing the Rist diagram for each shift…", max_entries=4)
def rist_frames(keys: DataKeys, settings: PipelineSettings, skeys: Sequence[str]) -> list[RistShift]:
    return svc.build_rist_frames(page_data(keys, settings), settings, skeys)


def clear_caches() -> None:
    """Forget everything loaded (the Refresh button also bumps the nonce, so the disk cache is re-checked)."""

    for fn in (hourly_dataset, online_data, neon_data, lab_samples, page_data, rist_frames):
        fn.clear()
