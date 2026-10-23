"""Fact-register groups MOD and VAL (utils.burden_engine.model_facts): Fixed-omega and STRATA-omega side by side."""

from __future__ import annotations

import pytest

from utils.burden_engine.facts import load_register
from utils.burden_engine.model_facts import model_facts, validation_facts


@pytest.fixture(scope="module")
def mod():
    return {r["id"]: r for r in model_facts()}


def test_model_facts_directions(mod):
    p = {k: v["params"] for k, v in mod.items()}
    assert set(p) == {f"MOD-{i:02d}" for i in range(1, 12)}
    assert p["MOD-01"]["strata_omega"] < p["MOD-01"]["fixed_omega"] - 2.0      # pellet for sinter: omega gain
    assert p["MOD-08"]["strata_omega"] < -2.0 and abs(p["MOD-08"]["fixed_omega"]) < 1.0
    assert p["MOD-02"]["strata_omega"] > p["MOD-02"]["fixed_omega"]            # lump ore: slower reduction
    assert p["MOD-09"]["fixed_omega"] < 0 < p["MOD-09"]["difference"]          # dilution; residence penalty
    assert p["MOD-10"]["fixed_omega"] > 0.5 and p["MOD-10"]["strata_omega"] < p["MOD-10"]["fixed_omega"]
    assert p["MOD-11"]["fixed_omega"] < 0 and p["MOD-11"]["strata_omega"] < p["MOD-11"]["fixed_omega"]   # nut coke (GAS-23)
    for v in p.values():
        assert v["difference"] == pytest.approx(v["strata_omega"] - v["fixed_omega"], abs=0.011)
        assert v["sigma_strata"] >= 0


def test_register_holds_generated_rows(mod):
    reg = load_register()
    for fid, row in mod.items():
        f = reg[fid]
        assert f.group == "MOD" and f.params["fixed_omega"] == pytest.approx(row["params"]["fixed_omega"], abs=0.05)
        assert f.params["strata_omega"] == pytest.approx(row["params"]["strata_omega"], abs=0.05)
    val = {r["id"]: r for r in validation_facts()}
    for fid, row in val.items():
        assert reg[fid].status == row["status"] and reg[fid].params["n"] == row["params"]["n"]
    assert {r["status"] for r in val.values()} <= set(reg.statuses)
    mt = reg.model_table()
    assert len(mt) == 23 and all(r["fixed_omega"] is not None and r["strata_omega"] is not None for r in mt)
