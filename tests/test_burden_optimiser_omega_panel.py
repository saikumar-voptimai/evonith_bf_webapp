"""Smoke test of the shaft-efficiency panel and auto-mode UI (Streamlit AppTest, no model call)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app():
    import streamlit as st
    from utils.burden_engine.engine import Engine, Limits, Settings
    from utils.burden_engine.scenarios import run_candidates
    from ui.burden_optimiser.omega_panel import render_auto, render_omega_result, render_omega_settings

    out = render_omega_settings("t_om")
    st.write(out)
    eng = Engine()
    r = eng.min_cost(prod_t_day=2270, s=Settings(omega_mode="blend"), L=Limits())
    today = eng.min_cost(prod_t_day=2270, s=Settings(omega_mode="blend", certainty="high"), L=Limits())
    render_omega_result(eng, r, today)
    render_omega_result(eng, eng.min_cost(prod_t_day=2270, s=Settings(), L=Limits()))
    auto = run_candidates(eng, 2270.0, Settings(), Limits(), today)
    render_auto(auto, {v.id: v.name for v in eng.vars}, mode_label="test", calibration=dict(source="fallback", omega=0.8))


def test_panel_renders():
    at = AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Shaft efficiency" in text and "Rule pick" in " ".join(s.value for s in at.success)
    assert any(b.label == "Review the pick" for b in at.button)
