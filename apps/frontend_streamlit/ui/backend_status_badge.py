"""Small sidebar badge for backend API availability."""

from __future__ import annotations

from dataclasses import asdict

import streamlit as st

try:
    from apps.frontend_streamlit.config.frontend_settings import load_frontend_settings
    from apps.frontend_streamlit.services.backend_status import BackendStatus, get_backend_status_summary
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.config.frontend_settings import load_frontend_settings
    from apps.frontend_streamlit.services.backend_status import BackendStatus, get_backend_status_summary


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
            if settings.show_advanced_backend_status and settings.page_api_flags.get("ops"):
                try:
                    from apps.frontend_streamlit.services.status_api import get_status
                except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
                    from apps.frontend_streamlit.services.status_api import get_status

                token = str(st.session_state.get("auth_access_token") or "").strip() or None
                try:
                    advanced = get_status(access_token=token)
                    runtime = advanced.get("runtime", {})
                    st.caption(f"Runtime: {runtime.get('status', 'unknown')}")
                    dependencies = advanced.get("dependencies")
                    if dependencies:
                        st.caption(f"Dependencies: {dependencies.get('status', 'unknown')}")
                        profile = dependencies.get("profile") or {}
                        runtime_profile = profile.get("runtime_profile")
                        if runtime_profile:
                            st.caption(f"Profile: {runtime_profile}")
                except Exception as exc:
                    st.caption(f"Advanced status unavailable: {exc}")
