from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "apps" / "frontend_streamlit" / "custom_pages" / "6_Material_Balance.py"
API = REPO_ROOT / "apps" / "frontend_streamlit" / "services" / "material_balance_api.py"


def test_material_balance_page_has_one_gateway_rich_path():
    text = PAGE.read_text(encoding="utf-8")

    forbidden = [
        "_render_api_mode",
        "run_full_balance",
        "load_full_config",
        "save_ash_analyses",
        "save_dpr_mapping",
        "discover_dpr_fields",
        "get_csv_date_range",
        "clear_day_caches",
        "st.link_button",
        "material_balance.yml",
        "InfluxDB",
        "bosh_vol_from_formula",
    ]
    assert [item for item in forbidden if item in text] == []
    assert "get_material_balance_gateway" in text
    assert "st.form(\"material_balance_run_form\")" in text
    assert "Run Material Balance" in text
    assert "adapt_result_for_plotters" in text


def test_material_balance_api_adapter_has_no_domain_or_data_imports():
    text = API.read_text(encoding="utf-8")
    forbidden = ["pandas", "furnace_data", "StaticDatasetManager", "load_static_dataset", "fetch_offline", "yaml", "run_full_balance"]
    assert [item for item in forbidden if item in text] == []