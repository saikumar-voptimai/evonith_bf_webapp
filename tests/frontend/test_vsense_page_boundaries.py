from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "apps" / "frontend_streamlit" / "custom_pages" / "4_Recommendations.py"


def test_vsense_page_has_no_reduced_api_ui_or_frontend_heavy_work():
    text = PAGE.read_text(encoding="utf-8")

    forbidden = [
        "_render_api_mode",
        "run_recommendations",
        "st.text_area",
        "joblib",
        "DataframesProcessor",
        "dataset_refresher",
        "run_optimiser",
        "call_llm",
        "yaml.safe_dump",
        "load_control_bounds",
        "save_control_bounds",
    ]
    assert [item for item in forbidden if item in text] == []
    assert "get_vsense_gateway" in text
    assert "V-Sense - Blast Parameter Optimisation" in text
