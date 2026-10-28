"""Working state of the FurnaceMind tools (SV-5.1, #312).

The tools pass data between calls through a mutable mapping: the fetched datasets (``fm_datasets``), the active
DataFrame (``fm_df``, ``fm_df_meta``), the latest figure (``fm_fig``), the knowledge-search queues and the stores the
search tools use (``shift_store``, ``knowledge_store``). Until SV-5.1 that mapping was ``st.session_state``.

``bind(state)`` makes ``state`` the mapping of the tools for the current thread or asyncio task (a context variable);
the agent service binds one mapping per conversation, so the tools run without Streamlit. Unbound calls inside a
Streamlit script run use the browser session's ``st.session_state`` (page 7 and its quick skills, unchanged); unbound
calls elsewhere use a process-wide dictionary (scripts, tests).

No Streamlit import here: ``streamlit`` is only looked up when the process has already loaded it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

#: Keys of the tools' working state read by the turn runner and the page.
CURRENT_TURN = "fm_current_artifact_turn_id"
FIGURE = "fm_fig"
FIGURE_TURN = "fm_fig_turn_id"
DATAFRAME = "fm_df"
DATAFRAME_META = "fm_df_meta"
DATAFRAME_TURN = "fm_df_turn_id"

_BOUND: ContextVar[MutableMapping[str, Any] | None] = ContextVar("furnacemind_tool_state", default=None)
_PROCESS_STATE: dict[str, Any] = {}


def _streamlit_session() -> MutableMapping[str, Any] | None:
    """``st.session_state`` when this thread runs a Streamlit script, else None (Streamlit is never imported here)."""
    st = sys.modules.get("streamlit")
    if st is None:
        return None
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx(suppress_warning=True) is None:
            return None
    except Exception:
        return None
    return st.session_state


def in_streamlit() -> bool:
    """True when no state is bound and this thread runs a Streamlit script (page 7)."""
    return _BOUND.get() is None and _streamlit_session() is not None


def current() -> MutableMapping[str, Any]:
    """The mapping the tools read and write: the bound state, else Streamlit's session state, else the process's."""
    bound = _BOUND.get()
    if bound is not None:
        return bound
    session = _streamlit_session()
    return session if session is not None else _PROCESS_STATE


@contextmanager
def bind(state: MutableMapping[str, Any]) -> Iterator[MutableMapping[str, Any]]:
    """Make ``state`` the tools' state within the block (this thread or task and the contexts copied from it)."""
    token = _BOUND.set(state)
    try:
        yield state
    finally:
        _BOUND.reset(token)


def owned_by_turn(state: MutableMapping[str, Any], turn_key: str, turn_id: str | None) -> bool:
    """True when the artifact whose owner is stored under ``turn_key`` belongs to ``turn_id`` (any turn when None)."""
    if turn_id is None:
        return True
    return str(state.get(turn_key) or "") == turn_id


__all__ = [
    "CURRENT_TURN",
    "DATAFRAME",
    "DATAFRAME_META",
    "DATAFRAME_TURN",
    "FIGURE",
    "FIGURE_TURN",
    "bind",
    "current",
    "in_streamlit",
    "owned_by_turn",
]
