"""What-if coke: the research scenarios reproduced, modes kept apart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo.charged_coke.features import build_features
from utils.bmo.charged_coke.scenario import ENGINEERING_BETA, estimate, response_coefficients, scenario_support
from utils.bmo.charged_coke.service import ChargedCokeEngine, gate_inputs

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "charged_coke"
ORIGIN = pd.Timestamp("2026-09-21 09:00")


@pytest.fixture(scope="module")
def engine() -> ChargedCokeEngine:
    return ChargedCokeEngine.load()


@pytest.fixture(scope="module")
def features(engine):
    source = pd.read_csv(FIXTURES / "source_window.csv.gz", parse_dates=["time"]).set_index("time")
    return build_features(source, engine.physics_config, start=ORIGIN - pd.Timedelta(hours=4), end=ORIGIN)


@pytest.mark.parametrize("index", range(12))
def test_research_scenarios_are_reproduced_exactly(index, engine, features) -> None:
    example = json.loads((FIXTURES / "scenario_examples.json").read_text())[index]
    row = features.causal.loc[pd.Timestamp(example["origin"])]
    overrides = {}
    if example.get("pci_delta"):
        overrides["pci_override"] = float(row["PCI_KG/THM"]) + example["pci_delta"]
    if example.get("nut_delta"):
        overrides["nut_override"] = float(row["nut"]) + example["nut_delta"]
    if example.get("blend"):
        overrides["blend_scale"] = example["blend"]
    mode = "engineering" if example["mode"] == "fixed" else "fitted"

    result = estimate(example["reference_coke_kg_thm"], row, engine.physics_config,
                      mode=mode, fitted_beta=engine.model.beta, **overrides)

    assert result.coke == pytest.approx(example["coke_kg_thm"], abs=1e-9)
    assert result.slag == pytest.approx(example["slag_kg_thm"], abs=1e-9)
    assert result.total_fuel == pytest.approx(example["total_fuel_kg_thm"], abs=1e-9)
    for key in ("pci", "nut", "slag"):
        assert result.effects[key] == pytest.approx(example[f"coke_effect_{key}"], abs=1e-9)


def test_modes_are_separate_and_never_added(engine) -> None:
    fitted = response_coefficients("fitted", engine.model.beta)
    engineering = response_coefficients("engineering", engine.model.beta)

    assert fitted[2] == pytest.approx(0.02045, abs=1e-5)
    assert engineering[2] == pytest.approx(0.22)
    np.testing.assert_allclose(engineering, ENGINEERING_BETA)
    with pytest.raises(ValueError, match="Unknown response mode"):
        response_coefficients("fitted+engineering", engine.model.beta)


def test_both_modes_displace_the_same_live_reference(engine, features) -> None:
    row = features.causal.loc[ORIGIN]
    live = 318.1162

    fitted = estimate(live, row, engine.physics_config, mode="fitted", fitted_beta=engine.model.beta,
                      pci_override=float(row["PCI_KG/THM"]) + 10)
    engineering = estimate(live, row, engine.physics_config, mode="engineering", fitted_beta=engine.model.beta,
                           pci_override=float(row["PCI_KG/THM"]) + 10)

    assert fitted.reference_coke == engineering.reference_coke == live
    # +10 PCI adds ~0.86 kg/THM ash slag; the slag term is the only part that
    # differs in sign of magnitude between modes, each applied once.
    assert fitted.slag_change == pytest.approx(engineering.slag_change)
    assert fitted.coke == pytest.approx(live - 4.50, abs=0.01)
    assert engineering.coke == pytest.approx(live - 5.30 + 0.22 * engineering.slag_change, abs=1e-9)
    assert "Engineering assumption" in engineering.flags[-1]
    # A scenario has no error range of its own.
    assert not hasattr(fitted, "lower")


def test_explicit_zero_pci_is_a_proposal_not_missing(engine, features) -> None:
    row = features.causal.loc[ORIGIN]

    result = estimate(318.0, row, engine.physics_config, mode="fitted", fitted_beta=engine.model.beta, pci_override=0.0)

    assert result.pci == 0.0 and result.pci_source == "override"
    assert "PCI outside the cruise range 150-250 kg/THM" in result.flags


def test_scenario_support_is_checked_separately(engine, features) -> None:
    live = gate_inputs(features).loc[ORIGIN]

    near = scenario_support(live, engine.policy, pci=float(live["pci"]) + 5, nut=float(live["nut"]), slag=float(live["slag"]))
    far = scenario_support(live, engine.policy, pci=40.0, nut=float(live["nut"]), slag=float(live["slag"]))

    assert near["supported"]
    assert not far["supported"]
    assert far["distance"] > far["stop_limit"]
