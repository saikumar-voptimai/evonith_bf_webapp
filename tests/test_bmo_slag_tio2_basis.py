"""TiO2 in the slag total (#258): the app counts it; frozen charged-coke bundles keep their training basis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.bmo.charged_coke import physics
from utils.bmo.types import SlagBalanceSettings

BUNDLES = sorted((Path(__file__).resolve().parents[1] / "src/assets/models/bmo_charged_coke").glob("*/physics_config.json"))


def test_setting_defaults_to_full_oxide_sum() -> None:
    assert SlagBalanceSettings().tio2_to_slag is True


@pytest.mark.skipif(not BUNDLES, reason="no charged-coke bundle in the repo")
def test_frozen_bundle_keeps_training_slag_basis(monkeypatch) -> None:
    seen = []
    real = physics.calculate_full_slag_balance

    def spy(**kw):
        seen.append(kw["settings"].tio2_to_slag)
        return real(**kw)

    monkeypatch.setattr(physics, "calculate_full_slag_balance", spy)
    cfg = json.loads(BUNDLES[-1].read_text(encoding="utf-8"))
    assert "tio2_to_slag" not in cfg["slag_balance"]          # trained before the change

    class Row(dict):                                             # every field 1.0: enough to reach the balance
        def get(self, key, default=None):
            return 1.0

    out = physics.expected_slag(Row(), cfg)
    assert seen == [False] and out["slag"] > 0
    cfg["slag_balance"]["tio2_to_slag"] = True                   # a retrained bundle records it
    physics.expected_slag(Row(), cfg)
    assert seen == [False, True]
