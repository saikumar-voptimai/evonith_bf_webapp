"""temporary Phase 12/cleanup compatibility shim for legacy Streamlit page."""

from apps.frontend_streamlit._legacy import run_canonical_page

run_canonical_page("custom_pages/1_Welcome.py")
