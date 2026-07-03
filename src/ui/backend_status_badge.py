"""Small sidebar badge for backend API availability."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

try:
    from config.frontend_settings import load_frontend_settings
    from services.backend_status import BackendStatus, get_backend_status_summary
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from src.config.frontend_settings import load_frontend_settings
    from src.services.backend_status import BackendStatus, get_backend_status_summary


@st.cache_data(ttl=15, show_spinner=False)
def _cached_backend_status() -> dict:
    return asdict(get_backend_status_summary())


def render_backend_status_badge() -> None:
    """Render a low-noise backend status indicator in the sidebar."""
    settings = load_frontend_settings()
    if not settings.show_backend_status_badge:
        return

    try:
        status = BackendStatus(**_cached_backend_status())
    except Exception as exc:
        status = BackendStatus(
            is_available=False,
            is_ready=None,
            status="unavailable",
            message="Backend API unavailable",
            details={"error": str(exc)},
        )

    with st.sidebar:
        if status.is_ready is True:
            st.success("Backend API available", icon=":material/check_circle:")
        elif status.is_available:
            st.warning("Backend API not ready", icon=":material/error:")
        else:
            st.info("Backend API unavailable", icon=":material/cloud_off:")

        with st.expander("Backend API details"):
            st.caption(status.message)
            if status.latency_ms is not None:
                st.caption(f"Latency: {status.latency_ms:.0f} ms")
            if status.request_id:
                st.caption(f"Request ID: {status.request_id}")
