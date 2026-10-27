"""Operator inputs on page 9 (SV-3.1, #298): the Streamlit cache of ``services.operator_inputs`` and the page's saves.

Page 9 reads the inputs through ``page_inputs`` (cached per version token, so an edit made through the API or in
another session shows within ``services.operator_inputs.TOKEN_TTL_S`` and a save here on the next rerun; a warning
while the YAML file stands in for a twin store that failed) and saves
each section with the functions below. They keep the names of the YAML savers of ``data.bmo.ore_editor_preferences``
(same merges): the BMO Sandbox refuses ``save_*_preferences`` and ``clear_dust_preferences`` by name. A save is a new
version in the twin store with the signed-in user as author, or the YAML file without a configured store.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data.bmo import ore_editor_preferences as prefs
from data.twin.repositories import Actor
from services import operator_inputs


@st.cache_data(show_spinner=False, max_entries=8)
def cached_inputs(yaml_path: str, token: str) -> operator_inputs.OperatorInputs:
    """The operator inputs of this version token (twin store, else the YAML file) and where they came from; Streamlit
    returns a copy, so the page may change them. Inputs that do not fit the token (the store failed, or a version was
    saved, between the token and the read) raise ``TokenMismatch``, which Streamlit does not cache."""
    current = operator_inputs.load_current(yaml_path=yaml_path)
    return operator_inputs.require_fit(token, current.provenance(), current)


def inputs_token(yaml_path: str | Path) -> str:
    """Key of the version in force (one indexed query, trusted for 10 s)."""
    return operator_inputs.token(yaml_path)


def load(yaml_path: str | Path, token: str | None = None) -> operator_inputs.OperatorInputs:
    """The inputs in force: cached per version token, or read once, uncached, when they do not fit it."""
    key = inputs_token(yaml_path) if token is None else token
    try:
        return cached_inputs(str(yaml_path), key)
    except operator_inputs.TokenMismatch as mismatch:
        return mismatch.value


def current_inputs(yaml_path: str | Path) -> dict[str, Any]:
    """The inputs of the version in force (``load``)."""
    return load(yaml_path).inputs


def page_inputs(yaml_path: str | Path, token: str) -> dict[str, Any]:
    """What page 9 reads (``load``), with a warning while the YAML file stands in for a twin store that failed (not
    reachable, not migrated): the versions saved since the seed are then not in force."""
    current = load(yaml_path, token)
    message = operator_inputs.fallback_warning(current.provenance())
    if message:
        st.warning(message)
    return current.inputs


def session_actor() -> Actor:
    """The signed-in user (``utils.session``) as the author of a save."""
    name = str(st.session_state.get("auth_user") or "").strip() or "streamlit"
    role = st.session_state.get("role")
    try:
        client_ip = st.context.ip_address
    except Exception:  # noqa: BLE001 - outside a browser session (tests, scripts)
        client_ip = None
    return Actor(name=name, role=str(role) if role else None, client_ip=client_ip)


def _save(path: str | Path, change: Callable[[dict[str, Any]], dict[str, Any]], note: str) -> operator_inputs.Written:
    return operator_inputs.write(change, actor=session_actor(), origin="streamlit", note=note, yaml_path=path)


def save_ore_editor_preferences(
    path: str | Path, editor_df: pd.DataFrame, *, flux_df: pd.DataFrame | None = None
) -> operator_inputs.Written:
    """Ferrous selection, prices and share bounds; with ``flux_df`` the flux inputs of the same form (one version)."""

    def change(doc: dict[str, Any]) -> dict[str, Any]:
        doc = prefs.merge_ore_editor(doc, editor_df)
        return prefs.merge_flux(doc, flux_df) if flux_df is not None else doc

    return _save(path, change, "page 9: ore and flux inputs" if flux_df is not None else "page 9: ore inputs")


def save_model_input_preferences(path: str | Path, values: dict[str, Any]) -> operator_inputs.Written:
    """Slag targets and charging limits."""
    return _save(path, lambda doc: prefs.merge_model_inputs(doc, values), "page 9: slag and charging settings")


def save_fuel_price_preferences(path: str | Path, fuel_ash_df: pd.DataFrame) -> operator_inputs.Written:
    """Fuel prices only; saved rates and ash chemistry are kept."""
    return _save(path, lambda doc: prefs.merge_fuel_prices(doc, fuel_ash_df), "page 9: fuel prices")


def save_fuel_ash_preferences(path: str | Path, fuel_ash_df: pd.DataFrame) -> operator_inputs.Written:
    """Every editable Fuel Ash value."""
    return _save(path, lambda doc: prefs.merge_fuel_ash(doc, fuel_ash_df), "page 9: fuel ash inputs")


def save_hm_chemistry_preferences(path: str | Path, values: dict[str, Any]) -> operator_inputs.Written:
    """Hot-metal chemistry held for the slag balance."""
    return _save(path, lambda doc: prefs.merge_hm_chemistry(doc, values), "page 9: hot metal chemistry")


def save_dust_preferences(path: str | Path, dust_df: pd.DataFrame) -> operator_inputs.Written:
    """Every editable BF Gas Dust value."""
    return _save(path, lambda doc: prefs.merge_dust(doc, dust_df), "page 9: dust inputs")


def clear_dust_preferences(path: str | Path) -> operator_inputs.Written:
    """Drop the saved dust rows: the page goes back to the DPR and the dust analyses."""
    return _save(path, prefs.drop_dust, "page 9: saved dust inputs removed")
