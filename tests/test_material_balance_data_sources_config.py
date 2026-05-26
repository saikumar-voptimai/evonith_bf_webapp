from __future__ import annotations

import sys
import types
from pathlib import Path


def _install_streamlit_stub(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        types.SimpleNamespace(cache_data=lambda *args, **kwargs: (lambda fn: fn)),
    )


def test_material_balance_data_sources_use_mapping_config(monkeypatch) -> None:
    _install_streamlit_stub(monkeypatch)

    from utils.material_balance import data_sources

    assert not hasattr(data_sources, "CSV_TO_SCHEMA")
    assert not hasattr(data_sources, "MASS_CSV_COLS")
    assert data_sources._dataset_alias_to_mapping_name()["COKE_CALC_MT"] == "coke_mt"
    assert data_sources._dataset_alias_to_mapping_name()["HOT BLAST VOLUMENM3/HR."] == (
        "hot_blast_volume_nm3hr"
    )
    assert "hot_blast_volume_nm3hr" in data_sources._process_fields()
    assert "coke_mt" in data_sources._schema_mass_cols()


def test_material_balance_data_sources_do_not_reintroduce_alias_map(monkeypatch) -> None:
    _install_streamlit_stub(monkeypatch)

    from utils.material_balance import data_sources

    source = Path(data_sources.__file__).read_text(encoding="utf-8")
    assert "COKE_CALC_MT" not in source
    assert "HOT BLAST VOLUMENM3/HR." not in source
    assert "CSV_TO_SCHEMA" not in source


def test_gas_phase_uses_mapping_names() -> None:
    from utils.material_balance.gas_phase import compute_blast_elements

    elements, debug = compute_blast_elements(
        {
            "hot_blast_volume_nm3hr": 100_000,
            "oxygen_enrichment_pct": 2.0,
        }
    )

    assert elements["blast_O_t"] > 0
    assert debug["wind_nm3h"] == 100_000
