"""STRATA page (SV-1.12, #292): the quantities of ``/strata`` for the API and the worker.

omega panel     ``omega_panel(rows, config, start=...)``: omega_STRATA against the implied omega of every stored twin
                window (``twin.snapshot``) with the band omega_STRATA + mu +/- 2 sigma_e, sigma_e the 1 sigma of the
                AR(1) innovations of r = omega_imp - omega_STRATA, and the two-sided CUSUM of the drift alarm. The
                evaluation is the alarm's own (``services.drift.evaluate`` on the snapshots of ``history_days`` up to
                now), so the page shows the state the alarm acts on. Pure: rows in, values out.
model summary   ``model_summary()``: the STRATA version, its out-of-sample validation (asset ``validation``: 2026
                expanding test, strict 2023-24 test with frozen parameters), the per-family verdicts of the register's
                VAL group, the register's MOD facts at the reference point and the certainty tiers of the omega model
                (``OmegaModel``: sigma(coke) limits, the class envelope of Investigate, the support range). File reads
                only; no network solve.
current blend   ``current_blend(rows)``: the certainty tier of the charged blend in the latest window that has STRATA
                (``flags.strata`` of the twin evaluation: the omega model's ``certainty`` on the window's blend).
live point      ``compute_live_point(cal_json)``, run by the worker (``strata.live_point``, 03:15 IST after the nightly
                calibration; supervisors start it from the page): the MOD facts recomputed at the current calibration,
                ``model_facts.model_facts(model_facts.reference_engine(cal))`` (the 16 Oct 2026 reference yard and
                base blend on that calibration: about 30 network solves, plus 17 for a Jacobian that is not cached),
                and the ring view of the thermal reserve zone at the calibration state: the ring quantities f, o, t of
                the STRATA-Flow network solve there (``strata_network``, no surrogate) and the ring counter-current
                terms of ``OmegaModel.rings``. ``save_live_point`` writes the document to
                ``src/storage/burden_engine/strata/live_point.json``; the API reads it (``read_live_point``) and adds
                the register's reference values (``live_point_view``).

Units: omega and its parts dimensionless; coke kg/tHM dry; shares as fractions (0-1) unless the key says pct.
No Streamlit.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services import drift as drift_service
from services import twin_views
from utils.burden_engine.facts import FactRegister, load_register
from utils.burden_engine.omega_model import MODEL_PATH, OmegaModel

SRC = Path(__file__).resolve().parents[1]
STORE_DIR = SRC / "storage" / "burden_engine" / "strata"
LIVE_POINT_FILE = "live_point.json"
KIND = "strata.live_point"
BAND_SIGMAS = 2.0  # the omega band: omega_STRATA + mu +/- 2 sigma_e
RINGS = ("centre", "2", "3", "4", "5", "wall")
VAL_FAMILY = re.compile(r"^Out-of-sample burnt coke, (?P<family>.+?): realised")


def _get(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, Mapping) else getattr(row, key, default)


def _num(value: Any) -> float | None:
    return twin_views.number(value)


def _round(value: Any, digits: int = 6) -> float | None:
    v = _num(value)
    return None if v is None else round(v, digits)


# --------------------------------------------------------------------------------------------------- omega panel
def _state(report: Any) -> str:
    if report.ar1 is None:
        return "collecting"
    open_eps = report.open_episodes()
    if len(open_eps) == 2:
        return "both"
    if open_eps:
        return next(iter(open_eps))
    return "in_control"


def _episode(ep: Any) -> dict[str, Any]:
    return dict(
        direction=ep.direction,
        raised_at=twin_views.utc(ep.raised_at),
        last_signal=twin_views.utc(ep.last_signal),
        cleared_at=twin_views.utc(ep.cleared_at),
        signals=int(ep.signals),
        peak=_round(ep.peak, 3),
    )


def omega_panel(
    rows: Sequence[Any],
    config: drift_service.DriftConfig,
    *,
    start: datetime,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The omega panel of the windows ending at or after ``start``; ``rows`` are the snapshots of the drift history
    (``config.history`` up to ``now``, at least back to ``start``), oldest first."""
    now = twin_views.utc(now or datetime.now(UTC))
    start = twin_views.utc(start)
    report = drift_service.evaluate(rows, config, now)
    ar = report.ar1
    settings = report.settings
    mu = ar.mu if ar is not None else settings.mu
    sigma_e = ar.sigma_e if ar is not None else None
    points = report.points
    out = []
    for row in rows:
        end = twin_views.utc(_get(row, "window_end"))
        if end is None or end < start or end > now:
            continue
        strata = _num(_get(row, "omega_strata"))
        p = points.loc[pd.Timestamp(end)] if pd.Timestamp(end) in points.index else None
        band = strata is not None and sigma_e is not None
        out.append(
            dict(
                window_end=end,
                status=_get(row, "status"),
                omega_imp=_num(_get(row, "omega_imp")),
                omega_strata=strata,
                omega_strata_sd=_num(_get(row, "omega_strata_sd")),
                omega_cal=_num(_get(row, "omega_cal")),
                band_low=strata + mu - BAND_SIGMAS * sigma_e if band else None,
                band_high=strata + mu + BAND_SIGMAS * sigma_e if band else None,
                r=_num(p.r) if p is not None else None,
                z=_num(p.z) if p is not None else None,
                s_pos=_num(p.s_pos) if p is not None else None,
                s_neg=_num(p.s_neg) if p is not None else None,
                signal=(p.signal if p is not None and p.signal in ("high", "low") else None),
            )
        )
    latest = report.latest()
    episodes = [e for e in report.episodes if e.open or twin_views.utc(e.last_signal) >= start]
    cusum = dict(
        state=_state(report),
        k=settings.k,
        h=settings.h,
        mu=mu,
        lag_h=settings.lag / pd.Timedelta(hours=1),
        clear_after_h=settings.clear / pd.Timedelta(hours=1),
        history_days=config.history / pd.Timedelta(days=1),
        rho=ar.rho if ar is not None else None,
        sigma_e=sigma_e,
        mu_hat=_num(ar.mu_hat) if ar is not None else None,
        n_pairs=ar.n_pairs if ar is not None else None,
        source=ar.source if ar is not None else None,
        band_sigmas=BAND_SIGMAS,
        latest=(
            dict(
                window_end=twin_views.utc(latest["t"]),
                r=latest.get("r"),
                z=latest.get("z"),
                s_pos=latest.get("s_pos"),
                s_neg=latest.get("s_neg"),
            )
            if latest
            else None
        ),
        episodes=[_episode(e) for e in episodes],
        note=report.note,
    )
    return dict(start=start, end=now, count=len(out), rows=out, cusum=cusum)


# ------------------------------------------------------------------------------------------------- current blend
def current_blend(rows: Iterable[Any]) -> dict[str, Any] | None:
    """The certainty tier of the charged blend in the latest window evaluated with STRATA (None without one)."""
    for row in reversed(list(rows)):
        flags = dict(_get(row, "flags") or {})
        st = flags.get("strata")
        if not isinstance(st, Mapping) or not st.get("certainty"):
            continue
        return dict(
            window_end=twin_views.utc(_get(row, "window_end")),
            evaluated_at=twin_views.utc(_get(row, "evaluated_at")),
            certainty=str(st["certainty"]),
            support=st.get("support"),
            outside=[str(x) for x in st.get("outside") or []],
            investigate=[str(x) for x in st.get("investigate") or []],
            unmapped_ores=[str(x) for x in st.get("unmapped_ores") or []],
            k_strata=_num(_get(row, "k_strata")),
            k_strata_sd=_num(_get(row, "k_strata_sd")),
            omega_strata=_num(_get(row, "omega_strata")),
            omega_strata_sd=_num(_get(row, "omega_strata_sd")),
        )
    return None


# ------------------------------------------------------------------------------------------------- model summary
def read_spec(path: Path | str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or MODEL_PATH).read_text(encoding="utf-8"))


def _strata_key(d: Mapping[str, Any]) -> str | None:
    return next((k for k in d if str(k).startswith("STRATA")), None)


def _fixed_strata(d: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    d = dict(d or {})
    key = _strata_key(d)
    return _num(d.get("Fixed-omega")), _num(d.get(key)) if key else None


def validation(spec: Mapping[str, Any]) -> dict[str, Any]:
    """The out-of-sample tests of the asset: 2026 expanding window, the two time splits, strict 2023-24."""
    val = dict(spec.get("validation") or {})
    exp = dict(val.get("expanding") or {})
    strict = dict(val.get("strict_2023_24") or {})
    periods = []
    for name, d in dict(strict.get("rmse") or {}).items():
        fixed, strata = _fixed_strata(d)
        periods.append(dict(period=str(name), rmse_fixed=fixed, rmse_strata=strata))
    whole = next((p for p in periods if p["period"] == "All days"), periods[0] if periods else {})
    weekly_fixed, weekly_strata = _fixed_strata(strict.get("weekly_rmse"))
    _, skill_day = _fixed_strata(strict.get("skill_day"))
    _, skill_7d = _fixed_strata(strict.get("skill_7d"))
    _, skill_omega = _fixed_strata(strict.get("skill_omega"))
    return dict(
        method=str(val.get("method", "")),
        note=str(val.get("note", "")),
        verdict_rule=str(val.get("verdict_rule", "")),
        expanding=dict(
            rmse_strata=_num(exp.get("rmse_coke")),
            rmse_fixed=_num(exp.get("rmse_fixed")),
            skill_coke_day=_num(exp.get("coke_day")),
            skill_coke_7d=_num(exp.get("coke_7d")),
            skill_omega_day=_num(exp.get("omega_day")),
        ),
        time_splits=[
            dict(split=str(k), skill_coke_day=_num(d.get("coke_day")), skill_omega_day=_num(d.get("omega_day")))
            for k, d in dict(val.get("time_split") or {}).items()
        ],
        strict=dict(
            method=str(strict.get("method", "")),
            rmse_fixed=whole.get("rmse_fixed"),
            rmse_strata=whole.get("rmse_strata"),
            periods=periods,
            weekly_rmse_fixed=weekly_fixed,
            weekly_rmse_strata=weekly_strata,
            skill_coke_day=skill_day,
            skill_coke_7d=skill_7d,
            skill_omega_day=skill_omega,
        ),
    )


def family_rows(register: FactRegister) -> list[dict[str, Any]]:
    """The register's VAL group: out-of-sample burnt coke by blend family, kg/tHM dry, with the verdict."""
    out = []
    for f in register.facts(group="VAL", include_retired=False):
        p = dict(f.params or {})
        m = VAL_FAMILY.match(f.statement or "")
        out.append(
            dict(
                id=f.id,
                family=m.group("family") if m else f.statement,
                verdict=f.status,
                grade=f.grade,
                n=int(p["n"]) if p.get("n") is not None else None,
                realised=_num(p.get("realised")),
                fixed_omega=_num(p.get("fixed_omega")),
                strata_omega=_num(p.get("strata_omega")),
                rmse_fixed=_num(p.get("rmse_fixed")),
                rmse_strata=_num(p.get("rmse_strata")),
                skill=_num(p.get("skill")),
                bias=_num(p.get("bias")),
                se_bias=_num(p.get("se_bias")),
                composition={k: float(v) for k, v in dict(p.get("composition") or {}).items() if _num(v) is not None},
            )
        )
    return out


def reference_responses(register: FactRegister) -> dict[str, dict[str, Any]]:
    """The register's MOD facts (coke responses at the reference point, shipped calibration) by id."""
    out = {}
    for f in register.facts(group="MOD", include_retired=False):
        p = dict(f.params or {})
        out[f.id] = dict(
            id=f.id,
            statement=f.statement,
            unit=f.unit,
            fixed_omega=_num(p.get("fixed_omega")),
            strata_omega=_num(p.get("strata_omega")),
            difference=_num(p.get("difference")),
            sigma_strata=_num(p.get("sigma_strata", f.se)),
            d_omega=_num(p.get("d_omega")),
            version=f.version,
            date=f.date,
        )
    return out


def tiers(model: OmegaModel) -> dict[str, Any]:
    """The certainty tiers of a plan or window (``OmegaModel.certainty``, GAS-18), sigma(coke) on wet coke, kg/tHM."""
    caps = ", ".join(f"{c} > {v:g} %" for c, v in model.inv_max.items())
    floors = ", ".join(f"{c} < {v:g} %" for c, v in model.inv_min.items())
    envelope = "; ".join(x for x in (caps, floors) if x)
    return dict(
        levels=[
            dict(
                tier="High",
                sigma_max_kg_thm=model.sig_high,
                support="inside",
                rule=f"σ(coke) ≤ {model.sig_high:g} kg/tHM and blend and production inside the range BF-02 ran",
            ),
            dict(
                tier="Medium",
                sigma_max_kg_thm=model.sig_medium,
                support="inside or margin",
                rule=f"σ(coke) ≤ {model.sig_medium:g} kg/tHM and blend and production within the margin of that range",
            ),
            dict(tier="Low", sigma_max_kg_thm=model.inv_sigma, support="any", rule="Otherwise, unless Investigate"),
            dict(
                tier="Investigate",
                sigma_max_kg_thm=None,
                support="any",
                rule=f"{envelope} of ferrous, or σ(coke) > {model.inv_sigma:g} kg/tHM"
                if envelope
                else f"σ(coke) > {model.inv_sigma:g} kg/tHM",
            ),
        ],
        investigate_sigma_kg_thm=model.inv_sigma,
        investigate_classes_max_pct=dict(model.inv_max),
        investigate_classes_min_pct=dict(model.inv_min),
        investigate_note=model.inv_note,
        support_classes_pct={c: [float(lo), float(hi)] for c, (lo, hi) in model.support.items()},
        support_production_t_h=[float(x) for x in model.support_prod],
        margin_pt=model.margin_pt,
        margin_t_h=model.margin_prod,
    )


def model_summary(spec: Mapping[str, Any] | None = None, register: FactRegister | None = None) -> dict[str, Any]:
    """Version, validation, VAL families, MOD facts at the reference point and the certainty tiers."""
    spec = dict(spec) if spec is not None else read_spec()
    register = register or load_register()
    model = OmegaModel(spec)
    version = str(spec.get("version", ""))
    return dict(
        version=version.split(" (")[0],
        version_detail=version,
        register_version=getattr(register, "version", None),
        validation=validation(spec),
        families=family_rows(register),
        reference=list(reference_responses(register).values()),
        tiers=tiers(model),
    )


# ---------------------------------------------------------------------------------------------------- live point
def response_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A MOD row of ``model_facts`` as the page shows it: Fixed-omega, STRATA-omega, their difference, 1 sigma."""
    p = dict(row.get("params") or {})
    return dict(
        id=row["id"],
        statement=row["statement"],
        unit=row["unit"],
        fixed_omega=_num(p.get("fixed_omega")),
        strata_omega=_num(p.get("strata_omega")),
        difference=_num(p.get("difference")),
        sigma_strata=_num(p.get("sigma_strata")),
        d_omega=_num(p.get("d_omega")),
    )


def state_inputs(st: Any) -> dict[str, Any]:
    """The operating state of a ring view in plant units."""
    return dict(
        blend_wet_pct={m: round(100.0 * float(v), 4) for m, v in st.shares.items()},
        classes_wet_pct={c: round(float(v), 4) for c, v in st.cls.items()},
        coke_kg_thm=_num(st.coke),
        nut_coke_kg_thm=_num(st.nut),
        pci_kg_thm=_num(st.pci),
        production_t_h=_num(st.prod),
        blast_nm3_thm=_num(st.blast_t),
        o2_pct=_num(100.0 * float(st.o2)),
        sinter_rdi_pct=_num(st.rdi),
        ferrous_kg_thm=_num(st.fer),
        blast_moisture_g_nm3=_num(st.humidity),
    )


def ring_view(model: OmegaModel, state: Any = None, *, label: str = "calibration") -> dict[str, Any]:
    """Ring quantities of the STRATA-Flow network at ``state`` (default: the calibration state, the network solve of
    the calibration) and the ring counter-current terms at the model's capacity ratio."""
    if model.backend != "network" or model.net is None:
        raise RuntimeError("the ring view needs the network backend, calibrated (no surrogate)")
    st = model.ref if state is None else state
    f, o, t = model.parts(st)
    rg = model.rings(st, model.R_ref, parts=(f, o, t))
    rings = []
    for i, name in enumerate(RINGS):
        rings.append(
            dict(
                ring=i + 1,
                name=name,
                gas_share=float(f[i]),
                fe_share=float(o[i]),
                residence_h=float(t[i]),
                gas_capacity=float(rg["c_gas"][i]),
                ore_oxygen=float(rg["c_ore"][i]),
                ntu=float(rg["ntu"][i]),
                efficiency=float(rg["eff"][i]),
                omega_share=float(rg["share"][i]),
                limited_by="gas" if rg["c_gas"][i] < rg["c_ore"][i] else "ore",
            )
        )
    return dict(
        state=label,
        distance=float(model.net.distance(st)),
        omega=float(rg["w"]),
        capacity_ratio=float(model.R_ref),
        k_blend=float(rg["k"]),
        kappa=float(model.kappa),
        gas_in_excess=float(np.sum(f[(model.R_ref * f) >= o])),
        inputs=state_inputs(st),
        rings=rings,
    )


def compute_live_point(cal_json: str, solvers: int | None = None) -> dict[str, Any]:
    """Runs in a pool process: the MOD facts at the calibration ``cal_json`` and the ring view at its state.

    The network pool of the process is cleared first, so the facts are those of ``python -m
    utils.burden_engine.model_facts`` on this calibration in a fresh process (the Jacobian comes from the disk cache
    of the nightly calibration when present). Independent network solves run in ``solvers`` processes
    (``strata_network.solver_processes``: ``BF_STRATA_SOLVERS``, else min(8, cpus))."""
    from utils.burden_engine import model_facts, strata_network

    t0 = time.perf_counter()
    n0, s0 = strata_network.solve_stats()
    strata_network.clear_pool()
    eng = model_facts.reference_engine(json.loads(cal_json))
    with strata_network.solver_processes(solvers):
        rows = model_facts.model_facts(eng)
    model = eng.omega_model()
    rings = ring_view(model)
    n1, s1 = strata_network.solve_stats()
    return dict(
        version=model.version.split(" (")[0],
        responses=[response_row(r) for r in rows],
        reference_point=dict(
            production_t_d=round(model_facts.P_REF * 24, 1),
            blend_wet_pct=dict(model_facts.BASE_BLEND),
            yard="16 Oct 2026 UAT yard (reference inputs)",
        ),
        rings=rings,
        network=dict(model.network_stats(), solves_run=n1 - n0, solve_seconds=round(s1 - s0, 1)),
        seconds=round(time.perf_counter() - t0, 1),
    )


def calibration_view(info: Any, chosen: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The calibration a live point was computed on: source, window, shifts, omega, losses and how it was chosen."""
    from services import day_calibration

    out = day_calibration.summary(info)
    chosen = dict(chosen or {})
    out.update(basis=chosen.get("basis"), day=chosen.get("day"), note=chosen.get("note"))
    return out


def live_point_document(
    result: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    made_at: datetime,
    job_id: str | None = None,
) -> dict[str, Any]:
    return dict(kind=KIND, made_at=made_at.isoformat(), job_id=job_id, calibration=dict(calibration), **result)


def folder(directory: Path | str | None = None) -> Path:
    """``directory``, else ``STORE_DIR`` (read at the call: tests point it elsewhere)."""
    return Path(directory) if directory is not None else STORE_DIR


def save_live_point(doc: Mapping[str, Any], directory: Path | str | None = None) -> Path:
    """Write the latest live point (atomic replace)."""
    from services.optimiser import jsonable

    path = folder(directory) / LIVE_POINT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(dict(doc)), indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_live_point(directory: Path | str | None = None) -> dict[str, Any] | None:
    """The latest live point (None when the job has not run or the file does not parse)."""
    try:
        return json.loads((folder(directory) / LIVE_POINT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def live_point_view(doc: Mapping[str, Any] | None, register: FactRegister | None = None) -> dict[str, Any]:
    """The live point with the register's value at the reference point beside each response."""
    if doc is None:
        return dict(kind=KIND, made_at=None, responses=[], rings=None)
    ref = reference_responses(register or load_register())
    responses = []
    for r in doc.get("responses") or []:
        at_ref = ref.get(r.get("id"))
        responses.append(
            dict(
                r,
                reference_strata_omega=at_ref["strata_omega"] if at_ref else None,
                reference_sigma_strata=at_ref["sigma_strata"] if at_ref else None,
                reference_fixed_omega=at_ref["fixed_omega"] if at_ref else None,
            )
        )
    return dict(doc, responses=responses)


__all__ = [
    "BAND_SIGMAS",
    "KIND",
    "LIVE_POINT_FILE",
    "STORE_DIR",
    "calibration_view",
    "compute_live_point",
    "current_blend",
    "live_point_document",
    "live_point_view",
    "model_summary",
    "omega_panel",
    "read_live_point",
    "ring_view",
    "save_live_point",
    "tiers",
    "validation",
]
