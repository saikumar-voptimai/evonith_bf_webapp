"""Top-gas dust for page 9 from the DPR and the plant's dust analyses (data.bmo.dust_inputs)."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from data.bmo import context_provider as context_module
from data.bmo.context_provider import EvonithBmoContextProvider
from data.bmo.dust_inputs import (
    MN_IN_MNO,
    ZN_IN_ZNO,
    dataset_dust_analysis,
    dpr_dust_rates,
    dust_caption,
    dust_cfg_with_live,
)
from data.bmo.ore_editor_preferences import (
    clear_dust_preferences,
    load_ore_editor_preferences,
    save_dust_preferences,
    saved_dust_ids,
)
from data.burden_engine.inputs import repo_path
from tests.test_bmo_context_provider import _write_bmo_files
from ui.bmo.components import build_dust_editor_df


def _dpr():
    # DPR rows are stamped 18:30 UTC: that is 00:00 IST of the day they report.
    idx = pd.to_datetime(["2026-10-11T18:30:00Z", "2026-10-12T18:30:00Z", "2026-10-13T18:30:00Z"], utc=True)
    return pd.DataFrame({"flue_dust_mt": [30.0, 33.0, 9.0], "gcp_dust_mt": [36.0, None, 10.0],
                         "total_hot_metal_mt": [2200.0, 2300.0, 600.0]}, index=idx)


def _charges():
    # 150 charges on 12 Oct IST, 150 on 13 Oct, 40 on 14 Oct (a stop day), spread over each IST day.
    days = {"2026-10-12": 150, "2026-10-13": 150, "2026-10-14": 40}
    out = []
    for day, n in days.items():
        start = pd.Timestamp(day, tz="Asia/Kolkata")
        out += list(pd.date_range(start, start + pd.Timedelta(hours=23, minutes=50), periods=n).tz_convert("UTC"))
    return pd.DatetimeIndex(out)


def test_dpr_rates_pool_dust_over_charges_on_the_same_ist_days():
    r = dpr_dust_rates(_dpr(), _charges(), min_charges=100)
    dc, gcp = r["dust_catcher"], r["gcp_dust"]
    # stop day (40 charges) left out; DPR row of 11 Oct 18:30 UTC is 12 Oct IST
    assert dc["days"] == 2 and (dc["first_day"], dc["last_day"]) == ("2026-10-12", "2026-10-13")
    assert dc["kg_per_charge"] == pytest.approx(1000 * (30 + 33) / 300)
    assert dc["kg_per_thm"] == pytest.approx(1000 * 63 / 4500)
    # a day without a GCP figure does not count for GCP
    assert gcp["days"] == 1 and gcp["kg_per_charge"] == pytest.approx(1000 * 36 / 150)


def test_dpr_rates_empty_without_dpr():
    assert dpr_dust_rates(pd.DataFrame(), _charges()) == {}


def test_dataset_analysis_converts_to_page_fields_and_dates_the_sample():
    t = pd.date_range("2026-08-01", periods=24 * 60, freq="h")
    n = len(t)
    df = pd.DataFrame({
        "time": t,
        "DUST_1_CHEMICAL_ANALYSIS_SIO2_PCT": [8.0] * (n - 100) + [9.0] * 100,
        "DUST_1_CHEMICAL_ANALYSIS_CAO_PCT": [2.0] * n,
        "DUST_1_CHEMICAL_ANALYSIS_MGO_PCT": [0.0] * (n // 2) + [0.5] * (n - n // 2),  # zeros are missing
        "DUST_1_CHEMICAL_ANALYSIS_MNO_PCT": [0.1] * n,
        "DUST_1_CHEMICAL_ANALYSIS_ZNO_PCT": [2.0] * n,
        "DUST_1_BASIC_ANALYSIS_FE_TOTAL_PCT": [36.0] * n,
        "DUST_2_CHEMICAL_ANALYSIS_SIO2_PCT": [7.0] * n,
        "DUST_2_CHEMICAL_ANALYSIS_FE2O3_PCT": [30.0] * n,
    })
    a = dataset_dust_analysis(df, window_days=60)
    dc, gcp = a["dust_catcher"], a["gcp_dust"]
    assert dc["cao_pct"] == pytest.approx(2.0) and dc["mgo_pct"] == pytest.approx(0.5)
    assert dc["mn_pct"] == pytest.approx(0.1 * MN_IN_MNO) and dc["zn_pct"] == pytest.approx(2.0 * ZN_IN_ZNO)
    assert dc["fe_pct"] == pytest.approx(36.0)
    assert dc["sample_time"] == str(t[n - 100].date())
    assert gcp["fe_pct"] == pytest.approx(30.0 * 0.6994, rel=1e-3)  # no basic Fe -> from Fe2O3


def test_live_values_overlay_the_configured_rows():
    cfg = yaml.safe_load(open(repo_path("src/config/setting_bmo.yml"), encoding="utf-8"))["bmo"]["dust_inputs"]
    rates = dpr_dust_rates(_dpr(), _charges())
    rows = {r["dust_id"]: r for r in dust_cfg_with_live(cfg, rates, {"gcp_dust": {"cao_pct": 1.5, "sample_time": "2026-10-08"}})}
    assert rows["dust_catcher"]["quantity_kg_per_charge"] == pytest.approx(210.0)
    assert rows["dust_catcher"]["rate_basis"] == "kg_per_charge"
    assert rows["dust_catcher"]["cao_pct"] == next(r for r in cfg if r["dust_id"] == "dust_catcher")["cao_pct"]
    assert rows["gcp_dust"]["cao_pct"] == 1.5 and "analysis 2026-10-08" in rows["gcp_dust"]["source"]
    assert "DPR 2026-10-12 to 2026-10-13" in rows["dust_catcher"]["source"]
    df = build_dust_editor_df(list(rows.values()))
    assert list(df["dust_id"]) == ["dust_catcher", "gcp_dust"]
    caption = dust_caption(rates, None, saved_ids=["gcp_dust"])
    assert "210 kg/charge" in caption and "saved values in use" in caption


def test_configured_dust_is_the_measured_dust_not_the_workbook_row():
    """The workbook row (CaO 5.63 %, MgO 3.57 %) read slag MgO and basicity low against the lab; the dust analyses
    carry CaO ~2 % and MgO < 1 %."""
    cfg = yaml.safe_load(open(repo_path("src/config/setting_bmo.yml"), encoding="utf-8"))["bmo"]
    rows = {r["dust_id"]: r for r in cfg["dust_inputs"]}
    assert set(rows) == {"dust_catcher", "gcp_dust"}
    for r in rows.values():
        assert r["rate_basis"] == "kg_per_charge" and 100 < r["quantity_kg_per_charge"] < 400
        assert r["cao_pct"] < 3.0 and r["mgo_pct"] < 1.5
    assert cfg["data_sources"]["dust_window_days"] > 0


def test_saved_dust_values_can_be_dropped(tmp_path):
    path = tmp_path / "prefs.yml"
    path.write_text(yaml.safe_dump({"model_inputs": {"charge_mass_mt": 26.4}}), encoding="utf-8")
    df = build_dust_editor_df([{"dust_id": "gcp_dust", "display_name": "GCP dust", "quantity_kg_per_charge": 250.0}])
    save_dust_preferences(path, df)
    prefs = load_ore_editor_preferences(path)
    assert saved_dust_ids(prefs) == ["gcp_dust"]
    clear_dust_preferences(path)
    prefs = load_ore_editor_preferences(path)
    assert saved_dust_ids(prefs) == [] and prefs["model_inputs"]["charge_mass_mt"] == 26.4


def test_provider_reads_dpr_dust_and_charges(tmp_path, monkeypatch):
    settings_path, mapping_path = _write_bmo_files(tmp_path)
    calls = []

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        if kwargs["table_name"] == "offline_feed.dpr_data":
            return _dpr()
        if kwargs["table_name"] == "offline_feed.charge_data":
            return pd.DataFrame(index=_charges().rename("time"))  # timestamps only, no columns
        raise AssertionError(kwargs["table_name"])

    monkeypatch.setattr(context_module, "_fetch_offline_data", fake_fetch)
    provider = EvonithBmoContextProvider(setting_path=str(settings_path), mapping_path=str(mapping_path))
    rates, warnings = provider.get_dpr_dust_rates(window_days=30)
    assert rates["dust_catcher"]["kg_per_charge"] == pytest.approx(210.0)
    assert warnings == []
    assert {c["table_name"] for c in calls} == {"offline_feed.dpr_data", "offline_feed.charge_data"}
    assert provider.get_data_diagnostics()["dust"]["rows"][0]["dust_id"] == "dust_catcher"


def test_provider_dust_failure_is_non_blocking(tmp_path, monkeypatch):
    settings_path, mapping_path = _write_bmo_files(tmp_path)

    def boom(**_kwargs):
        raise RuntimeError("no database")

    monkeypatch.setattr(context_module, "_fetch_offline_data", boom)
    provider = EvonithBmoContextProvider(setting_path=str(settings_path), mapping_path=str(mapping_path))
    rates, warnings = provider.get_dpr_dust_rates()
    assert rates == {} and "DPR dust query failed" in warnings[0]
