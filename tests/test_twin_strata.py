"""STRATA-omega of a window against its 14-day anchor (SV-1.1, #281).

``utils.burden_engine.strata_window`` (anchor, window state, omega_STRATA) and the K_STRATA step of
``utils.coke_energy.twin`` on the synthetic sources of ``tests/twin_data.py`` with four ore types. The STRATA-Flow
network runs in exact mode (about 1 s per solve; about ten solves in this file).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.twin_data import synthetic_frames
from utils.burden_engine import strata_network, strata_window
from utils.burden_engine.omega_model import OmegaModel
from utils.coke_energy.pipeline import PipelineSettings, _with_dates, run_pipeline
from utils.coke_energy.twin import Window, evaluate_window

SETTINGS = PipelineSettings(history_start="2026-04-01", predict_from="2026-04-01")
ORES = {"ORE_2_CALC_MT": 0.55, "ORE_3_CALC_MT": 0.30, "ORE_5_CALC_MT": 0.10, "ORE_6_CALC_MT": 0.05}
DAY = pd.Timestamp("2026-05-16")


@pytest.fixture(scope="module")
def world():
    fd, m10, hm, dpr = synthetic_frames(start="2026-05-01", days=18, ore_split=ORES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_pipeline(fd, m10, hm, dpr, SETTINGS)
    assert not result.errors
    return (fd, m10, hm, dpr), result


@pytest.fixture(scope="module")
def anchor(world):
    _, result = world
    R = _with_dates(result.R)
    keys = strata_window.anchor_keys(R, R.stable, DAY, SETTINGS.window_days)
    omega = float(np.nanmedian(result.Z.reindex(keys)["omega"]))
    return strata_window.build_anchor(result.S, result.R, result.Z, keys, omega, DAY)


def _anchor_row(world, anchor) -> pd.Series:
    _, result = world
    return result.S.loc[list(anchor.keys)].select_dtypes("number").mean()


# ---------------------------------------------------------------------------------------------- anchor
def test_the_anchor_is_the_pool_of_the_14_day_medians(world, anchor) -> None:
    _, result = world
    dates = pd.to_datetime(result.R.loc[list(anchor.keys), "sdate"])
    assert len(anchor.keys) >= SETTINGS.min_shifts
    assert dates.min() >= DAY - pd.Timedelta(days=14) and dates.max() < DAY
    # omega_A is the omega that K_FORGE uses for the date (pipeline omega_fixed)
    assert anchor.omega == pytest.approx(result.shifts.loc["2026-05-16_A", "omega_fixed"], abs=1e-12)
    m = anchor.model
    assert m.w(anchor.state, anchor.R) == pytest.approx(anchor.omega, abs=1e-9)  # kappa refitted on A
    assert 0.6 < anchor.R < 0.8 and m.net.J is None  # exact mode: no Jacobian
    assert anchor.version == "STRATA 1.5"
    # the blend by ore type, from the charge of each ore (ORE_k_CALC_MT) and the asset's plant map
    clo, geomin = anchor.state.shares["lloyds_clo"], anchor.state.shares["geomin"]
    assert clo / geomin == pytest.approx(0.55 / 0.30, rel=1e-9)
    assert set(anchor.state.shares) == {"sinter", "pellet", "lloyds_clo", "geomin", "nmdc_rom", "lloyds_oversize"}


def test_the_window_at_the_anchor_state_has_the_anchor_omega(world, anchor) -> None:
    sw = strata_window.evaluate_window(anchor, _anchor_row(world, anchor), humidity=anchor.state.humidity)
    assert sw.omega == pytest.approx(anchor.omega, abs=1e-12)
    assert sw.sigma == pytest.approx(0.0, abs=1e-12)
    assert sw.solves == 0  # blend, production and operation states are the anchor itself


# ---------------------------------------------------------------------------------------------- channels
def test_pellet_for_sinter_raises_omega_through_the_blend_channel(world, anchor) -> None:
    row = _anchor_row(world, anchor).copy()
    ferrous = row.SINTER_CALC_MT + row.ORE_CALC_MT + row.TOTAL_PELLET_CALC_MT
    row["SINTER_CALC_MT"] -= 0.10 * ferrous  # 10 wet pt of the ferrous charge from sinter to pellet
    row["TOTAL_PELLET_CALC_MT"] += 0.10 * ferrous
    sw = strata_window.evaluate_window(anchor, row, humidity=anchor.state.humidity)
    t, m = sw.terms, anchor.model
    b_pellet = m.post_mean[m.post_names.index("sh_pellet")]  # (lam_pellet - 1) g_pellet, per wet pt
    assert t.d_disc == pytest.approx(10 * b_pellet, rel=1e-9)
    assert t.d_rest == pytest.approx(0.0, abs=1e-12) and t.d_nut == pytest.approx(0.0, abs=1e-12)
    # STRATA 1.5: lam_pellet 1.42 x g_pellet 0.00096 per pt x 10 pt = +0.0136; network physics at A, not linearised
    assert 0.008 < sw.omega - anchor.omega < 0.020
    assert sw.solves == 1  # the blend state only: production and operation are the anchor's


def test_nut_coke_enters_by_its_data_term_and_the_coke_layers(world, anchor) -> None:
    row = _anchor_row(world, anchor).copy()
    row["NUTCOKE_CALC_MT"] += 10.0 * row.PRODUCTIONTONNESPERHR / 1000  # +10 kg/tHM as charged
    sw = strata_window.evaluate_window(anchor, row, humidity=anchor.state.humidity)
    t, m = sw.terms, anchor.model
    assert t.d_nut == pytest.approx(10 * m.a_nut, rel=1e-9)  # GAS-23: 0.000995 per kg/tHM
    assert t.d_blend == pytest.approx(0.0, abs=1e-12) and t.d_disc == pytest.approx(0.0, abs=1e-12)
    assert -0.004 < t.d_rest < 0.0  # network coke-layer mass, scaled by lam_other
    assert sw.sigma == pytest.approx(
        m.plan(replace(anchor.state, nut=anchor.state.nut + 10)).sigma(anchor.state.coke, anchor.R), rel=1e-9
    )
    assert sw.solves == 1  # the operation state only


# ---------------------------------------------------------------------------------------------- ore types
def test_ore_types_map_to_model_materials() -> None:
    m = OmegaModel.load()
    split, unmapped = strata_window.ore_split({"ORE_2_CALC_MT": 6.0, "ORE_10_CALC_MT": 2.0, "ORE_8_CALC_MT": 2.0}, m)
    assert split == pytest.approx({"lloyds_clo": 0.8, "gomti": 0.2})  # ore_8 is not in the plant map: Lloyds CLO
    assert unmapped == ["ore_8"]
    split, unmapped = strata_window.ore_split({"ORE_CALC_MT": 10.0}, m)  # no split: the asset's reference split
    assert split == pytest.approx(m.spec["reference"]["lump_split"]) and unmapped == []


# ---------------------------------------------------------------------------------------------- twin step
def test_k_strata_is_the_forge_balance_at_omega_strata(world) -> None:
    frames, result = world
    w = Window.ending("2026-05-16 15:00")
    t0 = time.perf_counter()
    ev = evaluate_window(*frames, w, result, SETTINGS)
    seconds = time.perf_counter() - t0
    assert seconds < 30  # acceptance on the AGX Orin; here with the anchor solve and the window's three
    assert ev.status == "complete" and not ev.notes
    assert ev.omega_cal == pytest.approx(result.shifts.loc["2026-05-16_A", "omega_fixed"], abs=1e-12)
    dk = ev.quantities["dk_domega_kg_thm"]
    assert -300 < dk < -200
    # linear check of the balance: K_STRATA - K_FORGE = dK/domega x (omega_STRATA - omega_cal), to second order
    assert ev.k_strata - ev.k_forge == pytest.approx(dk * (ev.omega_strata - ev.omega_cal), abs=0.3)
    assert ev.k_strata_sd == pytest.approx(abs(dk) * ev.omega_strata_sd, rel=1e-9)
    assert 0 < ev.omega_strata_sd < 0.02
    channels = ("strata_blend_channel", "strata_operation_channel", "strata_nut_coke_term")
    assert ev.omega_strata == pytest.approx(ev.omega_cal + sum(ev.quantities[c] for c in channels), abs=1e-12)
    strata = ev.flags["strata"]
    assert 3 <= strata["network_solves"] <= 4  # blend, blend + production, operation (+ the anchor on first use)
    assert strata["support"] in ("inside", "margin", "outside")
    assert strata["certainty"] in ("High", "Medium", "Low", "Investigate")
    assert ev.versions == {"strata": "STRATA 1.5", "strata_network": strata_network.NET_VERSION}

    again = evaluate_window(*frames, w, result, SETTINGS)  # every state solved before: no network solve
    assert again.flags["strata"]["network_solves"] == 0
    assert (again.k_strata, again.omega_strata) == (ev.k_strata, ev.omega_strata)


def test_without_a_14_day_anchor_there_is_no_k_strata(world) -> None:
    frames, result = world
    ev = evaluate_window(*frames, Window.of_shift("2026-05-02_A"), result, SETTINGS)
    assert ev.k_strata is None and ev.omega_strata is None and "strata" not in ev.flags
    off = evaluate_window(*frames, Window.ending("2026-05-16 15:00"), result, SETTINGS, strata=False)
    assert off.k_strata is None and off.status == "complete"


# ---------------------------------------------------------------------------------------------- network exact mode
def test_exact_mode_solves_each_state_once(anchor) -> None:
    net = strata_network.NetworkParts(anchor.model.net_ctx, cache_dir=None, jacobian=False)
    net.set_reference(anchor.state)
    other = replace(anchor.state, prod=anchor.state.prod + 2.0)
    n0 = net.n_solves
    f, o, t = net.parts(other)
    assert net.n_solves == n0 + 1 and net.distance(other) == 0.0
    net.parts(other)
    assert net.n_solves == n0 + 1
    expected = net._split(net.solve(net._fill(net.vec(other))))  # a fresh solve of the same state
    for a, b in zip((f, o, t), expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    with pytest.raises(RuntimeError):
        strata_network.NetworkParts(anchor.model.net_ctx, cache_dir=None).parts(other)  # Jacobian mode, no reference
