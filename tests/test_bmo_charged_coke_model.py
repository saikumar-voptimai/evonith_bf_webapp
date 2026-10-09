from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.bmo.charged_coke import ChargedCokeModel

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "charged_coke"


@pytest.fixture(scope="module")
def model() -> ChargedCokeModel:
    return ChargedCokeModel()


def test_numpy_port_reproduces_the_original_model(model) -> None:
    # Outputs of the research package's own classes on fixed random inputs
    # (with gaps, to exercise median imputation), written by the export script.
    golden = np.load(FIXTURES / "golden_predictions.npz")

    out = model.predict(golden["sequences"], golden["anchor"], golden["deviation"])

    components = golden["components"]
    # The research bundle has one horizon: column 0.
    assert model.horizons == [5]
    np.testing.assert_allclose(out.linear[:, 0], components[:, 0], atol=1e-9)
    np.testing.assert_allclose(out.attention[:, 0], components[:, 1:].mean(axis=1), atol=1e-9)
    np.testing.assert_allclose(out.prediction[:, 0], golden["prediction"], atol=1e-9)
    np.testing.assert_allclose(out.spread[:, 0], golden["spread"], atol=1e-9)


def test_bundle_describes_a_single_four_hour_block(model) -> None:
    target = model.manifest["target"]

    assert target["rows_ahead"] == [2, 5]
    assert model.sequence_hours == 12
    assert len(model.features) == 20
    assert model.features[:2] == ["charge4", "charge24"]
    # The control layer the report recommends: PCI and nut coke reduce coke.
    assert model.control_order == ["pci", "nut", "slag"]
    assert model.beta[0] == pytest.approx(-0.4516, abs=1e-4)
    assert model.beta[1] < 0 < model.beta[2]


def test_control_layer_moves_the_forecast_by_beta_times_the_deviation(model) -> None:
    golden = np.load(FIXTURES / "golden_predictions.npz")
    x, anchor = golden["sequences"][:4], golden["anchor"][:4]
    base = model.predict(x, anchor, np.zeros((4, 3)))

    more_pci = model.predict(x, anchor, np.tile([10.0, 0.0, 0.0], (4, 1)))

    np.testing.assert_allclose(
        more_pci.prediction - base.prediction, 10.0 * model.beta[0], atol=1e-9
    )
    # The adjustment is common to every branch, so disagreement is unchanged.
    np.testing.assert_allclose(more_pci.spread, base.spread, atol=1e-9)


def test_control_deviation_is_against_a_trailing_reference_and_clipped(model) -> None:
    index = pd.date_range("2026-10-01", periods=30, freq="h")
    controls = pd.DataFrame({"pci": 180.0, "nut": 75.0, "slag": 330.0}, index=index)
    controls.loc[index[-1], ["pci", "nut", "slag"]] = [280.0, 70.0, np.nan]

    deviation = model.control_deviation(controls)

    last = deviation.iloc[-1]
    # 280 against a 24-h mean that includes it: (280 - 184.17) clipped at +40.
    assert last["pci"] == pytest.approx(40.0)
    assert last["nut"] == pytest.approx(70.0 - (75.0 * 23 + 70.0) / 24)
    assert last["slag"] == 0.0  # missing contributes nothing
    # Too little history for a reference: no deviation yet.
    assert (deviation.iloc[:11] == 0.0).all().all()


def test_wrong_shaped_input_is_refused(model) -> None:
    with pytest.raises(ValueError, match="sequences must be"):
        model.predict(np.zeros((2, 6, 20)), np.zeros(2), np.zeros((2, 3)))
