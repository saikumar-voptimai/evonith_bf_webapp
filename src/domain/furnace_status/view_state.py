"""Safe parsing of namespaced Furnace Status query parameters."""

from __future__ import annotations

from collections.abc import Mapping

from data.furnace_status.catalogue import PARAMETERS_BY_KEY
from domain.furnace_status.types import ViewState

VIEW_STATUS = "status"
VIEW_TREND = "trend"
VIEW_QUERY_KEY = "vboard_fs_view"
PARAMETER_QUERY_KEY = "vboard_fs_parameter"


def _single_value(raw: object) -> str | None:
    if isinstance(raw, (list, tuple)):
        raw = raw[-1] if raw else None
    return raw if isinstance(raw, str) else None


def parse_view_state(params: Mapping[str, object]) -> ViewState:
    view = _single_value(params.get(VIEW_QUERY_KEY))
    key = _single_value(params.get(PARAMETER_QUERY_KEY))
    if view == VIEW_TREND:
        spec = PARAMETERS_BY_KEY.get(key) if key is not None else None
        if spec is not None:
            return ViewState(VIEW_TREND, spec)
        return ViewState(VIEW_STATUS, needs_reset=True)
    if view is None or view == VIEW_STATUS:
        return ViewState(VIEW_STATUS, needs_reset=key is not None)
    return ViewState(VIEW_STATUS, needs_reset=True)
