"""The thermal inputs of the Burden & Fuel Optimiser page render (Streamlit AppTest).

Regression found by the /legacy check of #279: the line creating the five input columns was lost in ba8642b, and the
page stopped with ``NameError: name 'c1' is not defined``.
"""

import json

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app():
    import streamlit as st

    from ui.burden_optimiser.sections import render_thermal
    from utils.burden_engine import load_register

    thermal = {"HMT": 1480.0, "hm_Si": 0.42, "pci_kg_thm": 190.0, "nut_kg_thm": 70.0}
    moisture = {"value": 18.45, "day": "2026-10-15", "source": "Open-Meteo", "window": 18.45}
    cal = {"case": {"nut": {"moist_pct": 5.0}}}
    st.json(render_thermal(thermal, load_register(), moisture, "t_th", cal=cal))


def test_thermal_inputs_render():
    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert [n.label for n in at.number_input] == [
        "Hot metal temperature, °C",
        "Hot metal Si, %",
        "PCI, kg/tHM",
        "Nut coke (dry), kg/tHM",
        "Blast moisture, g/Nm³",
    ]
    assert [n.value for n in at.number_input] == [1480.0, 0.42, 190.0, 70.0, 18.4]
    # the nut coke is entered dry (the plant's rate) and handed to the engine as charged, at the calibration's moisture
    settings = json.loads(at.json[0].value)
    assert settings["nut_plant"] == pytest.approx(70.0 / 0.95) and settings["pci_plant"] == 190.0
    assert any("73.7 kg/tHM as charged (5.0 % moisture)" in c.value for c in at.caption)
