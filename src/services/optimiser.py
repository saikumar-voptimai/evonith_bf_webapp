"""Burden & Fuel Optimiser: yard, calibration and runs (moved from ``ui/burden_optimiser/loaders.py``, #271).

* Yard: the Blend Mix Optimiser's provider (Neon chemistry + stock) and the operator's saved inputs; if that fails
  (no database connection), the reference yard of 16 Oct 2026 shipped with the engine, clearly labelled.
* Calibration and regime: from the Coke Energy pipeline result (``services.coke_energy``), so the optimiser and the
  Coke Energy views calibrate on the same shifts; the shipped Jul-Oct 2026 calibration if the pipeline cannot run.
* Runs: keyed on every input, so repeating a request does not re-solve.

Uncached functions (``read_*``, ``compute_mode``) are what the Streamlit page wraps in ``st.cache_data``; the memoised
ones (``yard``, ``charged_blend``, ``run_mode``, ``engine``) serve the API and the worker. The worker runs one
optimiser job at a time, so the shared ``Engine`` is not used by two solves at once.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from config.config_loader import load_config
from data.burden_engine.calibration import CalibrationInfo, fallback_calibration, live_calibration
from data.burden_engine.inputs import load_charged_blend, load_yard
from services import coke_energy as ce
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH, Engine, Limits, Settings
from utils.burden_engine.scenarios import (
    run_candidates,
    run_cheapest_level,
    run_fixed,
    run_max,
    run_today,
    value_of_limits,
)
from utils.coke_energy.pipeline import PipelineSettings
from utils.memo import memo

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def config_mtime(name: str) -> int:
    """Modification time of a config file (0 when missing): the cache token of the config readers."""
    path = CONFIG_DIR / name
    return path.stat().st_mtime_ns if path.exists() else 0


def read_page_config(mtime_ns: int) -> dict:
    return load_config("burden_optimiser.yml").get("burden_optimiser", {})


def read_coke_energy_config(mtime_ns: int) -> dict:
    return load_config("coke_energy.yml").get("coke_energy", {})


page_config = memo(max_entries=2)(read_page_config)
coke_energy_config = memo(max_entries=2)(read_coke_energy_config)


def reference_yard(cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The 16 Oct 2026 UAT yard (case report), as yard rows, at the page's price basis (``prices`` in the page config,
    DATA-14) when one is configured."""
    with open(REFERENCE_INPUTS_PATH, encoding="utf-8") as fh:
        ref = json.load(fh)
    pb = dict((cfg or {}).get("prices") or {})
    mp, fp = dict(pb.get("materials") or {}), dict(pb.get("fluxes") or {})
    mats = [
        dict(
            m,
            selected=m.get("selected_in_uat", True),
            stock_mt=None,
            chem_source="reference 16 Oct 2026",
            chem_time=m.get("chem_period", ""),
            carried=True,
            reason="",
            price=float(mp.get(m["id"], m["price"])),
        )
        for m in ref["materials"]
    ]
    fl = [
        dict(f, stock_mt=None, max_qty_mt=f.get("max_t_day"), carried=True, price=float(fp.get(f["id"], f["price"])))
        for f in ref["fluxes"]
    ]
    fuel = {
        **ref["fuel"],
        **{k: float(v) for k, v in (pb.get("fuel") or {}).items()},
        "slag_price": float(pb.get("slag_price_rs_per_t", 0.0) or 0.0),
    }
    return dict(
        materials=mats,
        fluxes=fl,
        fuel=fuel,
        slag=ref.get("slag", {}),
        charging_saved=dict(
            charge_t=ref["charging"]["uat_charge_t"], charges_per_h=ref["charging"]["uat_charges_per_h"]
        ),
        warnings=[],
    )


def read_yard(cfg: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    """(yard, source, message). Source is 'live' or 'reference'."""
    try:
        y = load_yard(cfg)
        if not any(m["carried"] for m in y["materials"]):
            raise RuntimeError("no material has lab chemistry in the window")
        return y, "live", ""
    except Exception as exc:  # noqa: BLE001 - offline / no DB: fall back to the reference yard
        return reference_yard(cfg), "reference", f"Live yard unavailable ({type(exc).__name__}: {exc})."


@memo(ttl_seconds=1800)
def yard(cfg_json: str, nonce: int) -> tuple[dict[str, Any], str, str]:
    return read_yard(json.loads(cfg_json))


# ---------------------------------------------------------------------------------------------- calibration
_LIVE_MEMO: dict[tuple[str, int], CalibrationInfo] = {}
_LIVE_LOCK = threading.Lock()


def live_calibration_cached(cfg: Mapping[str, Any], data: Any) -> CalibrationInfo:
    """Live calibration of one pipeline result (one entry: the latest result object and calibration settings)."""
    key = (json.dumps(cfg.get("calibration", {}), sort_keys=True), id(data.result))
    with _LIVE_LOCK:
        if key in _LIVE_MEMO:
            return _LIVE_MEMO[key]
    info = live_calibration(data.result, cfg)
    with _LIVE_LOCK:
        _LIVE_MEMO.clear()
        _LIVE_MEMO[key] = info
    return info


def calibration(
    cfg: Mapping[str, Any],
    nonce: int,
    *,
    page_data_for: Callable[[ce.DataKeys, PipelineSettings], Any] | None = None,
    ce_config_for: Callable[[int], dict] | None = None,
    live: Callable[[Mapping[str, Any], Any], CalibrationInfo] | None = None,
) -> tuple[CalibrationInfo, Any]:
    """(CalibrationInfo, Coke Energy PageData or None). The Streamlit page passes its cached ``page_data`` so both
    of its pages share one pipeline run; the API and the worker use the service's memoised one."""
    page_data_for = page_data_for or ce.page_data
    ce_config_for = ce_config_for or coke_energy_config
    live = live or live_calibration_cached
    try:
        ce_cfg = ce_config_for(config_mtime("coke_energy.yml"))
        data = page_data_for(ce.data_keys(ce_cfg, nonce), PipelineSettings.from_config(ce_cfg))
        return live(cfg, data), data
    except Exception as exc:  # noqa: BLE001 - the optimiser must still open without the plant data
        info = CalibrationInfo(
            fallback_calibration(cfg.get("calibration", {}).get("fallback_path")),
            None,
            "fallback",
            ("", ""),
            0,
            f"Plant data unavailable ({type(exc).__name__}: {exc}): using the shipped "
            "Jul-Oct 2026 calibration and its operating point.",
        )
        return info, None


# ---------------------------------------------------------------------------------------------- runs
def input_hash(*parts: Any) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


_ENGINE: dict[str, Engine] = {}
_ENGINE_LOCK = threading.Lock()


def engine(inputs: Mapping[str, Any], cal: Mapping[str, Any]) -> Engine:
    """Engine for these inputs and calibration (one kept: the latest)."""
    key = input_hash(inputs, cal)
    with _ENGINE_LOCK:
        if key not in _ENGINE:
            _ENGINE.clear()
            _ENGINE[key] = Engine(dict(inputs), dict(cal))
        return _ENGINE[key]


def read_charged_blend(materials: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Ferrous blend charged over the last 24 h (Neon charge_data); empty with the reason when unavailable."""
    try:
        return load_charged_blend(materials)
    except Exception as exc:  # noqa: BLE001 - the plan still runs without the comparison
        return dict(shares={}, unmapped={}, charges=0, window=("", ""), error=f"{type(exc).__name__}: {exc}")


@memo(ttl_seconds=900)
def charged_blend(materials_json: str, nonce: int) -> dict[str, Any]:
    return read_charged_blend(json.loads(materials_json))


def compute_mode(
    mode: str,
    inputs_json: str,
    cal_json: str,
    settings: dict,
    limits: dict,
    prod_t_day: float,
    search_o2: bool,
    n_o2: int,
    P_lo: float,
    step: float,
    today_json: str = "{}",
    auto: bool = False,
) -> dict[str, Any]:
    """One optimiser run: 'fixed' (Step 2), 'cheapest' (Step 1) or 'max' (maximum production)."""
    eng = Engine(json.loads(inputs_json), json.loads(cal_json))
    s, L = Settings(**settings), Limits(**limits)
    if mode == "fixed":
        out = run_fixed(eng, prod_t_day, s, L)
        out["value_of_limits"] = value_of_limits(eng, prod_t_day, s, L, out["with_fuel"])
        shares = json.loads(today_json or "{}")
        out["today"] = (
            run_today(json.loads(inputs_json), json.loads(cal_json), shares, prod_t_day, s, L) if shares else None
        )
        if auto:
            out["auto"] = run_candidates(eng, prod_t_day, s, L, out["today"])
        return out
    if mode == "cheapest":
        out = run_cheapest_level(eng, s, L, search_o2=search_o2, n_o2=n_o2, P_lo=P_lo, step=step)
        plan = out.get("best")
    else:
        out = run_max(eng, s, L)
        plan = out.get("result")
    if auto and plan is not None:
        s_p = Settings(**plan["settings"])
        out["auto"] = run_candidates(eng, plan["P_tday"], s_p, Limits(**plan["limits"]), None)
    return out


run_mode = memo(max_entries=16)(compute_mode)


def as_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def clear() -> None:
    """Forget the yard, the charged blend, runs, the live calibration and the engine."""
    for fn in (yard, charged_blend, run_mode, page_config, coke_energy_config):
        fn.clear()
    with _LIVE_LOCK:
        _LIVE_MEMO.clear()
    with _ENGINE_LOCK:
        _ENGINE.clear()
