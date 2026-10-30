"""Burden & Fuel Optimiser: yard, calibration and runs (moved from ``ui/burden_optimiser/loaders.py``, #271).

* Yard: the Blend Mix Optimiser's provider (Neon chemistry + stock) and the operator's saved inputs (the version in
  force, ``services.operator_inputs``; a new version reads the yard again); if that fails (no database connection),
  the reference yard of 16 Oct 2026 shipped with the engine, clearly labelled. Since SV-7.3 the provider is the one on
  the shared live-input reads (``services.live_inputs``): a live case takes the live-inputs snapshot first, the yard is
  composed from the same reads and keyed on them, and the case carries the snapshot's id.
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
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.config_loader import load_config
from data.burden_engine.calibration import CalibrationInfo, fallback_calibration, live_calibration
from data.burden_engine.inputs import build_engine_inputs, load_charged_blend
from services import coke_energy as ce
from services import live_inputs, operator_inputs
from services.live_inputs import load_yard  # the yard on the shared live-input reads (SV-7.3)
from utils.burden_engine.engine import REFERENCE_INPUTS_PATH, Engine, Limits, Settings
from utils.burden_engine.facts import load_register
from utils.burden_engine.scenarios import (
    fuel_defaults,
    limits_from,
    run_candidates,
    run_cheapest_level,
    run_fixed,
    run_max,
    run_today,
    settings_from_regime,
    value_of_limits,
)
from utils.coke_energy.pipeline import PipelineSettings
from utils.coke_energy.weather import plan_blast_moisture
from utils.memo import memo

log = logging.getLogger(__name__)

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


def read_yard_for(cfg_json: str, inputs_token: str) -> tuple[dict[str, Any], str, str]:
    """``read_yard`` for a cache keyed on ``inputs_token``: ``operator_inputs.TokenMismatch`` (carrying the yard)
    when the operator inputs it was built with are not those the token names (the store failed, or a version was
    saved, between the token and the read)."""
    found = read_yard(json.loads(cfg_json))
    return operator_inputs.require_fit(inputs_token, found[0].get("operator_inputs"), found)


@memo(ttl_seconds=1800)
def cached_yard(
    cfg_json: str, nonce: int, inputs_token: str = "", live_key: str = ""
) -> tuple[dict[str, Any], str, str]:
    return read_yard_for(cfg_json, inputs_token)


def yard(cfg_json: str, nonce: int, inputs_token: str = "", live_key: str = "") -> tuple[dict[str, Any], str, str]:
    """``read_yard`` kept 30 min per ``inputs_token`` (``operator_inputs.token()``: a new version of the operator
    inputs is read at once) and ``live_key`` (``LiveInputsView.yard_key``: the stock, chemistry and operator-input
    reads of the live-inputs snapshot, so a yard is recomposed when a read refreshed, SV-7.3); a yard whose operator
    inputs do not fit the token is used once and not kept."""
    try:
        return cached_yard(cfg_json, nonce, inputs_token, live_key)
    except operator_inputs.TokenMismatch as mismatch:
        return mismatch.value


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
        out["critical_assumptions"] = critical_assumptions(eng, out.get("with_fuel"))
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
    out["critical_assumptions"] = critical_assumptions(eng, plan)
    return out


def critical_assumptions(eng: Engine, plan: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Coke effect of each critical assumption of a STRATA-ω plan (``Engine.critical_assumptions``, largest first,
    about 20 ms; page 11's table), stored with the run for the Live Twin's plan card (SV-1.9); empty for a Fixed-ω
    plan or when the sensitivity fails."""
    if plan is None or (plan.get("omega") or {}).get("mode") != "blend":
        return []
    try:
        s = Settings(**{k: v for k, v in dict(plan["settings"]).items() if k in Settings.__dataclass_fields__})
        return list(eng.critical_assumptions(plan, s))
    except Exception:  # noqa: BLE001 - the sensitivity never fails a run
        log.exception("critical assumptions of the plan failed")
        return []


run_mode = memo(max_entries=16)(compute_mode)


def solve_case(*args: Any, solvers: int | None = None) -> dict[str, Any]:
    """``compute_mode`` as the worker and page 11 run it (SV-1.8): with STRATA-ω, the network solves that do not
    depend on each other (the Jacobian of a new calibration, a plan's blend, production and operation states, the
    independent items of the searches) run in a pool of ``solvers`` processes (``strata_network.solver_count``:
    ``BF_STRATA_SOLVERS``, else min(8, cpus)) created for this run; the plan, the anchors and the solve counts equal
    the serial run's. Fixed-ω runs in this process."""
    from utils.burden_engine import strata_network

    settings = args[3] if len(args) > 3 else {}
    if dict(settings).get("omega_mode") != "blend":
        return compute_mode(*args)
    with strata_network.solver_processes(solvers):
        return compute_mode(*args)


def as_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def clear() -> None:
    """Forget the yard, the charged blend, runs, the live calibration and the engine."""
    for fn in (cached_yard, charged_blend, run_mode, page_config, coke_energy_config):
        fn.clear()
    with _LIVE_LOCK:
        _LIVE_MEMO.clear()
    with _ENGINE_LOCK:
        _ENGINE.clear()


# --------------------------------------------------------------------------------------- runs of the API (SV-1.6)
#: Mode names of the API and of ``twin.optimiser_run`` -> modes of ``compute_mode``.
MODES = {"step1": "cheapest", "step2": "fixed", "max_production": "max"}
#: Shaft-efficiency names of the API and the store -> ``Settings.omega_mode``.
OMEGA_MODES = {"strata": "blend", "fixed": "fixed"}
CERTAINTY_FILTERS = ("low", "any", "medium", "high")


@dataclass(frozen=True)
class Case:
    """One optimiser run as page 11 starts it without operator edits: the arguments of ``compute_mode`` and what
    they were built from."""

    mode: str  # step1 | step2 | max_production
    omega_mode: str  # strata | fixed
    args: tuple[Any, ...]
    yard_source: str  # live | reference
    yard_message: str
    calibration: dict[str, Any]
    names: dict[str, str]
    today_shares: dict[str, float]
    production_source: str = ""
    #: The operator inputs the yard was built with: source, version, valid_from, author (SV-3.1).
    operator_inputs: dict[str, Any] = field(default_factory=dict)
    #: Today's operating point (the regime window's medians), None on the shipped calibration.
    regime: dict[str, Any] | None = None
    #: The yard rows the engine inputs were built from: materials (carried or not), fluxes, fuel prices (SV-3.2).
    yard: dict[str, Any] = field(default_factory=dict)
    #: When today's charged blend was charged (IST), for Step 2.
    today_window: str = ""
    #: What the request set instead of the defaults, in the API's names (SV-3.2).
    overrides: dict[str, Any] = field(default_factory=dict)
    #: The live-inputs snapshot the live yard was built from (``services.live_inputs.LiveInputsRef`` as JSON, SV-7.3);
    #: None on the reference yard.
    live_inputs: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """The case as stored with the run (``optimiser_run.inputs``): settings, limits, target and sources."""
        _, _, _, settings, limits, target, search_o2, n_o2, p_lo, step, _, auto = self.args
        return dict(
            mode=self.mode,
            omega_mode=self.omega_mode,
            production_target_t_d=target if self.mode == "step2" else None,
            production_source=self.production_source if self.mode == "step2" else None,
            settings=settings,
            limits=limits,
            search_o2=search_o2,
            o2_levels=n_o2,
            frontier_lo_t_h=p_lo,
            frontier_step_t_h=step,
            auto=auto,
            yard_source=self.yard_source,
            yard_message=self.yard_message,
            operator_inputs=self.operator_inputs,
            calibration=self.calibration,
            today_shares=self.today_shares,
            today_window=self.today_window,
            materials=self.names,
            regime=self.regime,
            yard=self.yard,
            overrides=self.overrides,
            live_inputs=self.live_inputs,
        )


CALIBRATION_BASES = ("day", "live", "reference")
YARD_BASES = ("live", "reference")

#: Settings of a request (``services.jobs.OptimiserSettings``) -> ``Settings`` fields.
SETTINGS_FIELDS = {
    "hmt_c": "HMT",
    "hm_si_pct": "hm_Si",
    "pci_kg_thm": "pci_plant",
    "nut_coke_kg_thm": "nut_plant",
    "blast_moisture_g_nm3": "humidity_g",
    "hbt_c": "HBT",
    "o2_per_air": "O2_per_air",
    "hot_blast_pressure_bar_g": "Pb",
    "top_pressure_bar_g": "Pt",
}
#: Limits of a request (``services.jobs.OptimiserLimits``) -> ``Limits`` fields.
LIMITS_FIELDS = {
    "slag_max_kg_thm": "slag_max",
    "cao_sio2_min": "B2_min",
    "cao_sio2_max": "B2_max",
    "al2o3_max_pct": "Al2O3_max",
    "mgo_min_pct": "MgO_min",
    "mgo_al2o3_min": "MgO_Al2O3_min",
    "charge_t": "charge_t",
    "charges_per_h": "charges_per_h",
    "top_gas_min_c": "T_top_min",
    "hot_blast_pressure_max_bar_g": "Pb_max",
    "dp_max_bar": "dP_max",
    "top_pressure_min_bar_g": "Pt_min",
    "top_pressure_max_bar_g": "Pt_max",
    "blast_max_nm3h": "V_max",
    "o2_flow_max_nm3h": "O2_flow_max",
    "enrichment_max_pct": "enrich_max",
    "hbt_max_c": "HBT_max",
    "raft_min_c": "RAFT_min",
    "raft_max_c": "RAFT_max",
}


def engine_overrides(values: Mapping[str, Any] | None, fields: Mapping[str, str], what: str) -> dict[str, float]:
    """A request's settings or limits (API names, None = keep the default) as engine field values."""
    out: dict[str, float] = {}
    for key, value in dict(values or {}).items():
        if key not in fields:
            raise ValueError(f"unknown {what} {key!r}")
        if value is not None:
            out[fields[key]] = float(value)
    return out


def case_calibration(
    cfg: Mapping[str, Any], nonce: int = 0, basis: str = "day", day: Any = None
) -> tuple[CalibrationInfo, dict[str, Any]]:
    """(CalibrationInfo, how it was chosen). ``basis`` day: the calibration ``calibration.nightly`` made for ``day``
    (default today, IST; ``services.day_calibration``), the live one while there is none; live: page 11's; reference:
    the shipped Jul-Oct 2026 calibration (``calibration.fallback_path`` of the page config)."""
    from services import day_calibration

    if basis not in CALIBRATION_BASES:
        raise ValueError(f"calibration must be one of {', '.join(CALIBRATION_BASES)}")
    if basis == "reference":
        path = dict(cfg.get("calibration") or {}).get("fallback_path")
        message = "The shipped Jul-Oct 2026 calibration was requested."
        return CalibrationInfo(fallback_calibration(path), None, "fallback", ("", ""), 0, message), dict(
            basis="reference"
        )
    if basis == "day":
        d = day or day_calibration.today()
        doc = day_calibration.read(d)
        info = day_calibration.load(d) if doc is not None else None
        if info is not None:
            return info, dict(basis="day", day=d.isoformat(), made_at=doc.get("made_at"))
        info, _ = calibration(cfg, nonce)
        note = f"No calibration of {d.isoformat()} yet (calibration.nightly at 02:30): the live calibration was used."
        return info, dict(basis="live", requested="day", note=note)
    info, _ = calibration(cfg, nonce)
    return info, dict(basis="live")


def default_case(
    mode: str,
    *,
    omega_mode: str = "strata",
    production_t_d: float | None = None,
    production_source: str | None = None,
    certainty: str = "low",
    risk_k: float = 0.0,
    auto: bool = False,
    calibration_basis: str = "live",
    yard_basis: str = "live",
    settings: Mapping[str, Any] | None = None,
    limits: Mapping[str, Any] | None = None,
    nonce: int = 0,
    cfg: Mapping[str, Any] | None = None,
    plan_day: Any = None,
    calibration_info: tuple[CalibrationInfo, dict[str, Any]] | None = None,
) -> Case:
    """The run page 11 would start for ``mode`` with its defaults: live yard (else the reference yard) and
    calibration (else the shipped one), today's controls, PCI and nut coke (regime medians), today's Open-Meteo
    blast moisture, the register's limits with the saved slag limits and the plant charging; Step 2 at today's
    production (``production_t_d`` overrides it; ``production_source`` says where it came from) with the controls held
    as page 11 starts them, against today's charged blend; Step 1 and maximum production searching oxygen.
    ``omega_mode`` strata (default, certainty filter and risk aversion as given) or fixed. ``calibration_basis`` live
    (page 11's), day (the calibration of 02:30) or reference (the shipped one, ``case_calibration``); ``yard_basis``
    live (else the reference yard) or reference. ``settings`` and ``limits`` (``SETTINGS_FIELDS``,
    ``LIMITS_FIELDS``: the API's names) replace the defaults last, so a request's controls are held as given.
    ``plan_day`` (default today, IST) is the day whose Open-Meteo blast moisture the plan takes; ``calibration_info``
    (a ``case_calibration`` result) replaces the calibration of ``calibration_basis`` (the what-if days, SV-4.3)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    if omega_mode not in OMEGA_MODES:
        raise ValueError(f"omega_mode must be one of {', '.join(OMEGA_MODES)}")
    if certainty not in CERTAINTY_FILTERS:
        raise ValueError(f"certainty must be one of {', '.join(CERTAINTY_FILTERS)}")
    if yard_basis not in YARD_BASES:
        raise ValueError(f"yard must be one of {', '.join(YARD_BASES)}")
    s_over = engine_overrides(settings, SETTINGS_FIELDS, "setting")
    l_over = engine_overrides(limits, LIMITS_FIELDS, "limit")
    cfg = page_config(config_mtime("burden_optimiser.yml")) if cfg is None else cfg
    reg = load_register()
    live_ref = None
    if yard_basis == "reference":
        yard_rows, yard_source, yard_message = reference_yard(cfg), "reference", "The reference yard was requested."
    else:  # the live-inputs snapshot first (SV-7.3): the yard is composed from the same reads and keyed on them
        yard_cfg = dict(cfg.get("yard", {}))
        live_ref, live_key = live_inputs.snapshot_ref(
            chemistry_mode=yard_cfg.get("chemistry_mode"), chemistry_window_days=yard_cfg.get("chemistry_window_days")
        )
        yard_rows, yard_source, yard_message = yard(as_json(cfg), nonce, operator_inputs.token(), live_key)
    info, chosen = calibration_info or case_calibration(cfg, nonce, calibration_basis)
    regime = info.regime
    engine_mode = MODES[mode]
    process = engine_mode in ("cheapest", "max")
    prod_cfg = dict(cfg.get("production", {}))
    target = float(prod_cfg.get("default_target_t_day", 2270.0))
    source = "configured default"
    if engine_mode == "fixed":
        if production_t_d is not None:
            target, source = float(production_t_d), production_source or "requested"
        elif (regime or {}).get("prod_th"):
            target = float(round(float(regime["prod_th"]) * 24, -1))
            source = "median of the regime window (page 11's default)"
    humidity_g, _, _ = plan_blast_moisture(plan_day)
    thermal_cfg = cfg.get("thermal", {})
    th = {**thermal_cfg, **fuel_defaults(regime, thermal_cfg, info.cal)}
    thermal = dict(
        HMT=float(th.get("HMT", 1470.0)),
        hm_Si=float(th.get("hm_Si", 0.38)),
        pci_plant=float(th.get("pci_kg_thm", 190.0)),
        nut_plant=float(th.get("nut_kg_thm", 70.0)),
        humidity_g=round(float(humidity_g), 1),
    )
    limits_ = limits_from(reg, yard_rows.get("slag"), dict(cfg.get("charging", {})), **l_over)
    s = dict(settings_from_regime(regime, thermal_cfg, info.cal).__dict__, **thermal)
    if omega_mode == "fixed":
        s.update(omega_mode="fixed", certainty="any", risk_k=0.0)
    else:
        s.update(omega_mode="blend", certainty=certainty, risk_k=float(risk_k))
    if engine_mode == "fixed":  # the controls held for the plan start as page 11 shows them
        hy = Engine(cal=info.cal).hy
        s.update(HBT=float(round(s["HBT"])), O2_per_air=hy.o2_per_air_from_tag(float(hy.enrich_tag(s["O2_per_air"]))))
    s.update(s_over)
    materials = [m for m in yard_rows["materials"] if m.get("carried", True)]
    fu = dict(yard_rows["fuel"])
    fuel = dict(
        coke_price=float(fu["coke_price"]),
        nut_price=float(fu["nut_price"]),
        pci_price=float(fu["pci_price"]),
        slag_price=float(fu.get("slag_price", 0.0) or 0.0),
    )
    inputs = build_engine_inputs(
        materials,
        yard_rows["fluxes"],
        fuel,
        stock_horizon_days=float(cfg.get("yard", {}).get("stock_horizon_days", 1.0)),
    )
    names = {m["id"]: m.get("name", m["id"]) for m in inputs["materials"]} | {
        f["id"]: f.get("name", f["id"]) for f in inputs["fluxes"]
    }
    today = charged_blend(as_json(yard_rows["materials"]), nonce) if yard_source == "live" else {}
    shares = {k: float(v) for k, v in dict(today.get("shares") or {}).items()} if engine_mode == "fixed" else {}
    args = (
        engine_mode,
        as_json(inputs),
        as_json(info.cal),
        s,
        limits_.__dict__ | {"process": process},
        target,
        process,  # page 11 searches oxygen in the process modes by default
        int(prod_cfg.get("o2_levels", 9)),
        float(prod_cfg.get("frontier_lo_t_h", 84.0)),
        float(prod_cfg.get("frontier_step_t_h", 1.0)),
        as_json(shares),
        bool(auto),
    )
    cal_window = (
        f"{info.window[0]} to {info.window[1]}"
        if info.source == "live"
        else " to ".join(info.cal.get("period", ["", ""]))
    )
    return Case(
        mode=mode,
        omega_mode=omega_mode,
        args=args,
        yard_source=yard_source,
        yard_message=yard_message,
        calibration=dict(
            source=info.source,
            window=cal_window,
            omega=info.cal.get("omega"),
            Q_LZ_GJh=info.cal.get("Q_LZ_GJh"),
            Q_UZ_GJh=info.cal.get("Q_UZ_GJh"),
            message=info.message,
            **chosen,
        ),
        names=names,
        today_shares=shares,
        production_source=source,
        operator_inputs=dict(yard_rows.get("operator_inputs") or {}),
        regime=dict(regime) if regime else None,
        yard=dict(materials=yard_rows["materials"], fluxes=yard_rows["fluxes"], fuel=fuel),
        today_window=(
            "from {} to {} IST".format(*today["window"]) if shares and len(today.get("window") or ()) == 2 else ""
        ),
        overrides=dict(
            settings={k: v for k, v in dict(settings or {}).items() if v is not None},
            limits={k: v for k, v in dict(limits or {}).items() if v is not None},
        ),
        live_inputs=live_ref.model_dump(mode="json") if live_ref is not None and yard_source == "live" else None,
    )


PRODUCTION_RANGE_T_D = (1500.0, 3000.0)
CURRENT_MAX_AGE_H = 3.0


def current_production(
    store: Any, now: Any = None, *, max_age_h: float = CURRENT_MAX_AGE_H
) -> tuple[float | None, str]:
    """Step 2's current production: the production of the latest twin window (``twin.snapshot``, at most
    ``max_age_h`` old) in t/d rounded to 10, and where it came from; ``(None, reason)`` without a usable window (the
    run then takes page 11's default, the median of the regime window)."""
    import pandas as pd

    row = store.snapshots.latest() if store is not None else None
    if row is None:
        return None, "no twin window stored"
    now_utc = pd.Timestamp(now if now is not None else pd.Timestamp.now(tz="UTC"))
    now_utc = now_utc.tz_convert("UTC") if now_utc.tzinfo else now_utc.tz_localize("UTC")
    end = pd.Timestamp(row.window_end)
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    start = pd.Timestamp(row.window_start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    age_h = (now_utc - end) / pd.Timedelta(hours=1)
    if age_h > max_age_h:
        return None, f"the latest twin window ended {age_h:.1f} h ago (more than {max_age_h:g} h)"
    prod_th = _num((row.quantities or {}).get("production_t_h"))
    if prod_th is None:
        return None, "the latest twin window has no production"
    t_d = float(round(prod_th * 24, -1))
    lo, hi = PRODUCTION_RANGE_T_D
    if not lo <= t_d <= hi:
        return None, f"the latest twin window's production {t_d:.0f} t/d is outside {lo:.0f}-{hi:.0f} t/d"
    ist = "Asia/Kolkata"
    label = f"twin window {start.tz_convert(ist):%d %b %H:%M}-{end.tz_convert(ist):%H:%M} IST ({prod_th:.1f} t/h)"
    return t_d, label


def plan_of(res: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The plan a run reports: Step 2 the plan with fuel, Step 1 the cheapest level, else the maximum production."""
    mode = res.get("mode")
    if mode == "fixed":
        return res.get("with_fuel")
    if mode == "cheapest":
        return res.get("best")
    return res.get("result")


def jsonable(value: Any) -> Any:
    """A JSON document of a result: numpy and pandas values as numbers and lists, frames as records, non-finite
    numbers as null."""
    import math

    import numpy as np
    import pandas as pd

    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [jsonable(r) for r in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(value.__dict__)
    return value


def run_values(
    case: Case, res: Mapping[str, Any] | None, *, error: str | None = None, status: str | None = None
) -> dict[str, Any]:
    """The headline columns and documents of ``twin.optimiser_run`` for a case and its result (or failure): status
    ok, infeasible (no plan meets every limit, or the LP or a plausibility band refused the case) or failed;
    production t/d; cost, Rs/tHM (ore + flux + fuel - slag credit); saving against today's charged blend, Rs/tHM
    (Step 2); dry coke and its 1 sigma, kg/tHM; fuel rate, kg/tHM; certainty tier."""
    plan = plan_of(res) if res is not None else None
    if status is None:
        feasible = plan is not None and plan.get("feasible", True)
        status = "failed" if error is not None else ("ok" if feasible else "infeasible")
    values: dict[str, Any] = dict(
        mode=case.mode,
        omega_mode=case.omega_mode,
        inputs=jsonable(case.summary()),
        status=status,
        result=jsonable({k: v for k, v in dict(res or {}).items() if k != "z"}) | ({"error": error} if error else {}),
        versions=dict(
            calibration=case.calibration.get("source"),
            calibration_basis=case.calibration.get("basis"),
            yard=case.yard_source,
        ),
    )
    if plan is not None:
        out, cost, om = plan.get("out") or {}, plan.get("cost") or {}, plan.get("omega") or {}
        today = (res or {}).get("today")
        values.update(
            production_t_d=_num(plan.get("P_tday")),
            cost_rs_thm=_num(cost.get("total")),
            saving_rs_thm=(
                _num(today["cost"]["total"]) - _num(cost.get("total"))
                if today is not None and today.get("cost") and cost.get("total") is not None
                else None
            ),
            coke_kg_thm=_num(out.get("K")),
            coke_sd_kg_thm=_num(om.get("sigma_K")),
            fuel_rate_kg_thm=_num(out.get("fuel_rate")),
            certainty=om.get("certainty") if om.get("mode") == "blend" else None,
        )
    return values


def _num(value: Any) -> float | None:
    import math

    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None
