"""Small helpers for Streamlit fragment-scoped refreshes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

F = TypeVar("F", bound=Callable[..., Any])


def fragment(func: F | None = None, *, run_every: Any = None) -> Callable[[F], F] | F:
    """Wrap ``st.fragment`` behind one import point for app code."""

    return st.fragment(func, run_every=run_every)


def rerun_fragment() -> None:
    """Rerun only the active fragment."""

    st.rerun(scope="fragment")
