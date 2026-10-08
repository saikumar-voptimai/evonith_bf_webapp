"""Retrain the charged-coke model for a 1-5 hour path, calibrate, report.

What is fitted, for each horizon h = 1..5 rows ahead of a complete row t:

    y_h = 4-h charged coke rate as it reads at row t+h
        = 1000 * sum(COKE_CALC_MT[t+h-3..t+h]) / sum(production[t+h-3..t+h])

The single-hour ratio is not used: whole-charge counting makes it swing
about 44 kg/THM hour to hour, against about 9 for the 4-h rate.

The structure is the research package's ``structured.py`` with five outputs:

1. Per horizon, a signed level model of PCI, nut coke and expected slag gives
   beta_h (PCI and nut coke may only lower coke, slag only raise it).
2. Residual target: y_h - beta_h . clip(control - trailing 24-h mean).
3. A ridge (alpha 1000) and three attention seeds (7, 17, 37) each predict the
   five residual deltas from 12 hours of the 20 learned channels. Forecast =
   anchor + 0.5 ridge + 0.5 mean(attention) + beta_h . deviation.

Honest evaluation. Four 7-day folds are predicted by models fitted only on
labels that had matured before each fold starts. Their out-of-fold errors set
the error range for every condition and horizon, and the novelty and branch-
disagreement limits. The model deployed is the last fold's: the most recent
seven days were not used to fit it, so the 7-day forecast-vs-actual chart is
out of sample at deployment, and stays so afterwards.

Nothing is deployed here: a candidate is written with its report and waits
for an operator to accept it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from utils.bmo.charged_coke.estimators import DeltaAttention, LinearDelta
from utils.bmo.charged_coke.features import FeatureSet, build_features
from utils.bmo.charged_coke.policy import DISPLAY_STATES, PolicyConfig, apply_policy, audit_hour
from utils.bmo.charged_coke.service import gate_inputs, sequences

AR = ["charge4", "charge24"]
PSN = ["pci", "nut", "slag"]
QUALITY = PSN + ["eta", "wind", "blast", "raft", "oxygen", "pressure", "top_temp", "hm_si", "hm_ti", "hm_temp",
                 "ore_fe", "sinter_fe", "coke_ash", "pci_ash", "csr", "cri", "sinter_share", "pellet_share"]
FEATURES = AR + [c for c in QUALITY if c not in PSN]
NOVELTY_FEATURES = ["wind", "pci", "eta", "blast", "prod", "raft", "dp", "wind_cv4", "prod_cv4", "pci_cv4",
                    "blast_range4", "eta_range4", "slag", "nut"]
CLIP = (np.array([-40.0, -20.0, -60.0]), np.array([40.0, 20.0, 60.0]))
DELTA_SCALE, LEVEL_SCALE = 15.0, 30.0


@dataclass(frozen=True)
class TrainingSettings:
    """Retraining choices; defaults reproduce the research study's."""

    horizons: tuple[int, ...] = (1, 2, 3, 4, 5)
    train_start: str = "2026-01-01"
    train_days: int = 400
    fold_days: int = 7
    folds: int = 4
    seeds: tuple[int, ...] = (7, 17, 37)
    ridge_alpha: float = 1000.0
    signed_alpha: float = 20.0
    attention_epochs: int = 75
    minimum_train_rows: int = 500
    display_width_cap: float = 35.0
    minimum_calibration_hours: int = 50
    minimum_calibration_days: int = 10


# ---- fitting ----------------------------------------------------------------


def _signed_controls(seq: np.ndarray, ids: list[int], y: np.ndarray, alpha: float) -> np.ndarray:
    """Research ``Model('signed', PSN, form='level')``: bounded PCI/nut/slag slopes."""

    a = seq[:, -1, :][:, ids].astype(float)
    center = float(np.median(y))
    yy = (y - center) / LEVEL_SCALE
    med = np.nanmedian(a, axis=0)
    a = np.where(np.isfinite(a), a, med)
    mean, sd = a.mean(0), a.std(0)
    sd[sd < 1e-6] = 1
    z = np.column_stack([np.ones(len(a)), (a - mean) / sd])
    pen = np.eye(z.shape[1]) * np.sqrt(alpha)
    pen[0, 0] = 0
    low, high = np.full(z.shape[1], -np.inf), np.full(z.shape[1], np.inf)
    low[1:4] = np.array([-1.5, -1.5, 0]) * sd[:3] / LEVEL_SCALE
    high[1:4] = np.array([0, 0, 0.5]) * sd[:3] / LEVEL_SCALE
    coef = lsq_linear(np.vstack([z, pen]), np.r_[yy, np.zeros(z.shape[1])], bounds=(low, high)).x
    return coef[1:] * LEVEL_SCALE / sd


@dataclass
class FittedModel:
    """One structured fit: beta per horizon, ridge and attention residuals."""

    horizons: tuple[int, ...]
    beta: np.ndarray  # (horizons, 3)
    ridge: Any
    attention: list[Any]
    train_rows: pd.DatetimeIndex
    trained_until: pd.Timestamp

    def predict(self, x: np.ndarray, base: np.ndarray, deviation: np.ndarray) -> dict[str, np.ndarray]:
        branches = [self.ridge.predict(x)] + [net.predict(x) for net in self.attention]
        components = np.stack([b * DELTA_SCALE + base[:, None] for b in branches], axis=2)
        components = components + (deviation @ self.beta.T)[:, :, None]
        attention = components[:, :, 1:].mean(axis=2)
        return {"prediction": 0.5 * components[:, :, 0] + 0.5 * attention, "spread": components.std(axis=2),
                "linear": components[:, :, 0], "attention": attention}


def fit_structured(seq, labels: np.ndarray, base: np.ndarray, deviation: np.ndarray, mask: np.ndarray,
                   index: pd.DatetimeIndex, settings: TrainingSettings, *, trained_until: pd.Timestamp,
                   progress: Callable[[str], None] | None = None) -> FittedModel:
    """Fit beta per horizon, then the ridge and attention residual branches."""

    ids = list(range(len(FEATURES)))
    psn = seq["psn"]
    x = seq["features"][mask]
    beta = np.vstack([
        _signed_controls(psn[mask], [0, 1, 2], labels[mask, i], settings.signed_alpha)
        for i in range(labels.shape[1])
    ])
    shifted = labels[mask] - deviation[mask] @ beta.T
    target = (shifted - base[mask, None]) / DELTA_SCALE
    if progress:
        progress("ridge")
    ridge = LinearDelta(alpha=settings.ridge_alpha).fit(x[:, :, ids], target)
    attention = []
    for seed in settings.seeds:
        if progress:
            progress(f"attention seed {seed}")
        attention.append(DeltaAttention(seed=seed, epochs=settings.attention_epochs, dim=10, relative=True,
                                        skip=True, decay=0.03).fit(x[:, :, ids], target))
    return FittedModel(tuple(settings.horizons), beta, ridge, attention, index[mask], trained_until)


# ---- data -------------------------------------------------------------------


@dataclass
class TrainingData:
    fs: FeatureSet
    hours: pd.DataFrame
    seq: dict[str, np.ndarray]
    labels: np.ndarray
    base: np.ndarray
    deviation: np.ndarray


def prepare(source: pd.DataFrame, physics_config: dict, *, end: pd.Timestamp, settings: TrainingSettings,
            policy: PolicyConfig) -> TrainingData:
    """Features, gate inputs, sequences and the five labels on one clock."""

    start = max(pd.Timestamp(settings.train_start), end - pd.Timedelta(days=settings.train_days))
    fs = build_features(source, physics_config, start=start, end=end)
    hours = gate_inputs(fs)
    audits = [audit_hour(row, policy) for _, row in hours.iterrows()]
    for key in ("severity", "data_ready", "reasons", "codes"):
        hours[key] = [a[key] for a in audits]
    ch = fs.channels
    seq = {"features": sequences(ch, FEATURES, 12), "psn": sequences(ch, PSN, 12)}
    current = fs.labels.current4
    labels = np.column_stack([current.shift(-h).to_numpy() for h in settings.horizons])
    controls = ch[PSN].astype(float)
    reference = controls.rolling(24, min_periods=12).mean()
    deviation = (controls - reference).fillna(0.0).clip(lower=CLIP[0], upper=CLIP[1], axis=1).to_numpy()
    return TrainingData(fs, hours, seq, labels, current.to_numpy(), deviation)


def _train_mask(data: TrainingData, cutoff: pd.Timestamp, settings: TrainingSettings) -> np.ndarray:
    g = data.fs.gates
    longest = max(settings.horizons)
    return (
        g.usable.to_numpy() & g.core.to_numpy() & np.isfinite(data.labels).all(axis=1) & np.isfinite(data.base)
        & np.asarray(g.index < cutoff - pd.Timedelta(hours=longest))
    )


# ---- calibration ------------------------------------------------------------


def _robust_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0)
    scale = np.nanpercentile(x, 90, axis=0) - np.nanpercentile(x, 10, axis=0)
    scale[scale == 0] = 1.0
    return center, scale


def _leave_day_distance(query: np.ndarray, query_days: np.ndarray, rows: np.ndarray, row_days: np.ndarray) -> np.ndarray:
    """Distance to the nearest support row from another calendar day."""

    out = np.empty(len(query))
    row_sq = (rows**2).sum(axis=1)
    for start in range(0, len(query), 512):
        block = query[start:start + 512]
        d2 = (block**2).sum(axis=1)[:, None] - 2 * block @ rows.T + row_sq[None, :]
        d2[query_days[start:start + 512, None] == row_days[None, :]] = np.inf
        out[start:start + 512] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def _p90(values: np.ndarray) -> float:
    a = np.sort(np.asarray(values, dtype=float))
    return float(a[min(len(a) - 1, int(np.ceil((len(a) + 1) * 0.9)) - 1)])


def _bins(frame: pd.DataFrame, horizons, settings: TrainingSettings) -> tuple[dict, dict]:
    bins: dict[str, Any] = {}
    floor = {h: 0.0 for h in horizons}
    for severity in ("cruise", "mild", "moderate"):
        rows = frame[frame["severity"] == severity]
        radius = {}
        for h in horizons:
            err = (rows[f"prediction_h{h}"] - rows[f"actual_h{h}"]).abs().dropna()
            floor[h] = max(floor[h], _p90(err) if len(err) else floor[h])
            radius[str(h)] = floor[h]
        longest = str(max(horizons))
        scored = rows.dropna(subset=[f"actual_h{max(horizons)}"])
        bins[severity] = {
            "radius": radius[longest], "radius_by_horizon": radius, "n": int(len(scored)),
            "days": int(pd.DatetimeIndex(scored.index).floor("D").nunique()),
        }
    unusual_rows = frame[frame["unusual"]]
    unusual = {}
    for h in horizons:
        err = (unusual_rows[f"prediction_h{h}"] - unusual_rows[f"actual_h{h}"]).abs().dropna()
        if len(err) >= 30:
            unusual[str(h)] = max(bins["moderate"]["radius_by_horizon"][str(h)], _p90(err))
        else:
            all_err = (frame[f"prediction_h{h}"] - frame[f"actual_h{h}"]).abs().dropna()
            unusual[str(h)] = max(bins["moderate"]["radius_by_horizon"][str(h)], float(np.quantile(all_err, 0.95)))
    return bins, unusual


# ---- the run ----------------------------------------------------------------


@dataclass
class TrainingResult:
    version: str
    model: FittedModel
    parameters: dict[str, Any]
    support: dict[str, np.ndarray]
    report: dict[str, Any]
    holdout: pd.DataFrame
    settings: TrainingSettings = field(default_factory=TrainingSettings)


def train(source: pd.DataFrame, physics_config: dict, base_policy: PolicyConfig, *, end: pd.Timestamp,
          settings: TrainingSettings = TrainingSettings(),
          progress: Callable[[str], None] | None = None) -> TrainingResult:
    """Fit, calibrate on out-of-fold predictions and report one candidate.

    Args:
        source: Hourly furnace dataset (interval-start labels, IST).
        physics_config: Frozen physics configuration for expected slag.
        base_policy: Display thresholds (cruise, extended and variability
            limits) to keep; error ranges and support limits are recalibrated.
        end: Latest complete data row to use.
        settings: Retraining choices.
        progress: Optional callback for status text.
    """

    def say(text: str) -> None:
        if progress:
            progress(text)

    say("building features")
    data = prepare(source, physics_config, end=end, settings=settings, policy=base_policy)
    index = data.hours.index
    horizons = list(settings.horizons)
    fold_starts = [end - pd.Timedelta(days=settings.fold_days * k) for k in range(settings.folds, 0, -1)]
    oof = []
    model = None
    for number, fold_start in enumerate(fold_starts, start=1):
        mask = _train_mask(data, fold_start, settings)
        if mask.sum() < settings.minimum_train_rows:
            raise ValueError(f"Only {int(mask.sum())} training hours before {fold_start}; need {settings.minimum_train_rows}.")
        say(f"fold {number}/{len(fold_starts)}: fitting on {int(mask.sum())} hours before {fold_start:%d %b %H:%M}")
        model = fit_structured(data.seq, data.labels, data.base, data.deviation, mask, index, settings,
                               trained_until=fold_start, progress=lambda t, n=number: say(f"fold {n}: {t}"))
        fold_end = fold_start + pd.Timedelta(days=settings.fold_days)
        # Every clock hour of the fold, as the live service decides them: an hour
        # without a forecast still resets the recovery count.
        in_fold = np.asarray((index >= fold_start) & (index < fold_end))
        computable = data.hours["shadow_possible"].to_numpy()[in_fold]
        out = model.predict(data.seq["features"][in_fold][computable], data.base[in_fold][computable],
                            data.deviation[in_fold][computable])
        frame = data.hours.loc[in_fold].copy()
        frame["fold"] = number
        for i, h in enumerate(horizons):
            for name in ("prediction", "spread", "linear", "attention"):
                column = np.full(len(frame), np.nan)
                column[computable] = out[name][:, i]
                frame[f"{name}_h{h}"] = column
        frame["spread"] = frame[[f"spread_h{h}" for h in horizons]].max(axis=1, skipna=False)
        oof.append(frame)
    oof_frame = pd.concat(oof)

    say("calibrating ranges and support")
    support_mask = np.isin(index, model.train_rows) & data.hours["severity"].ne("large").to_numpy()
    support_x = data.hours.loc[support_mask, NOVELTY_FEATURES].astype(float)
    support_x = support_x[np.isfinite(support_x.to_numpy()).all(axis=1)]
    center, scale = _robust_scale(support_x.to_numpy())
    support_rows = (support_x.to_numpy() - center) / scale
    support_days = support_x.index.floor("D").to_numpy()
    leave_day = _leave_day_distance(support_rows, support_days, support_rows, support_days)
    novelty_warn, novelty_stop = float(np.quantile(leave_day, 0.95)), float(np.quantile(leave_day, 0.99))
    q = oof_frame[NOVELTY_FEATURES].astype(float)
    complete = np.isfinite(q.to_numpy()).all(axis=1)
    # Hours inside the deployed model's support are measured leave-one-day-out
    # (else they would find themselves); every other hour exactly as the live
    # service measures it, against the whole support.
    in_support = np.isin(q.index, support_x.index)
    query_days = np.where(in_support, q.index.floor("D").to_numpy(), np.datetime64("NaT"))
    oof_frame["novelty"] = np.nan
    oof_frame.loc[complete, "novelty"] = _leave_day_distance(
        (q.to_numpy()[complete] - center) / scale, query_days[complete], support_rows, support_days)
    scored = oof_frame[oof_frame["data_ready"] & oof_frame["severity"].ne("large")]
    spread_warn = float(scored["spread"].quantile(0.95))
    spread_stop = float(scored["spread"].quantile(0.995))
    oof_frame["unusual"] = oof_frame["novelty"].gt(novelty_warn) | oof_frame["spread"].gt(spread_warn)
    calibration = (oof_frame["data_ready"] & oof_frame["severity"].ne("large")
                   & oof_frame["novelty"].le(novelty_stop) & oof_frame["spread"].le(spread_stop))
    bins, unusual_radius = _bins(oof_frame[calibration], horizons, settings)
    parameters = {
        **{k: v for k, v in base_policy.parameters.items() if k not in ("bins", "unusual_radius", "unusual_n")},
        "features": NOVELTY_FEATURES, "novelty_warn": novelty_warn, "novelty_stop": novelty_stop,
        "spread_warn": spread_warn, "spread_stop": spread_stop, "bins": bins,
        "unusual_radius": unusual_radius[str(max(horizons))], "unusual_radius_by_horizon": unusual_radius,
        "display_width_cap": settings.display_width_cap,
        "minimum_calibration_hours": settings.minimum_calibration_hours,
        "minimum_calibration_days": settings.minimum_calibration_days, "recovery_updates": 2,
        "interpretation": "Empirical 90% error ranges from out-of-fold predictions; not a guaranteed coverage.",
    }

    say("scoring the 7-day holdout")
    holdout = oof_frame[oof_frame["fold"] == len(fold_starts)].copy()
    honest_bins, honest_unusual = _bins(oof_frame[calibration & oof_frame["fold"].lt(len(fold_starts))], horizons, settings)
    holdout = _decide_holdout(holdout, base_policy, parameters, honest_bins, honest_unusual, horizons)
    report = _report(holdout, model, horizons, settings, end, fold_starts, data)
    version = f"{end:%Y%m%d_%H%M}"
    return TrainingResult(version, model, parameters, {"center": center, "scale": scale, "rows": support_rows},
                          report, holdout, settings)


def _decide_holdout(holdout, base_policy, parameters, honest_bins, honest_unusual, horizons) -> pd.DataFrame:
    """States from the deployed parameters; ranges from the earlier folds only."""

    from utils.bmo.charged_coke.service import horizon_radius

    policy = PolicyConfig(parameters=parameters, core=base_policy.core, extended=base_policy.extended,
                          variability=base_policy.variability, minimum_history=base_policy.minimum_history,
                          minimum_reference=base_policy.minimum_reference, large_charge_cv=base_policy.large_charge_cv,
                          mild_charge_cv=base_policy.mild_charge_cv, support_center=np.zeros(1),
                          support_scale=np.ones(1), support_rows=np.zeros((1, 1)))
    longest = max(horizons)
    frame = holdout.assign(prediction=holdout[f"prediction_h{longest}"])
    decided = apply_policy(frame, policy)
    honest = {**parameters, "bins": honest_bins, "unusual_radius_by_horizon": honest_unusual,
              "unusual_radius": honest_unusual[str(longest)]}
    shown = decided["state"].isin(DISPLAY_STATES)
    for h in horizons:
        radius = [horizon_radius(honest, s, u, h) if ok else np.nan
                  for s, u, ok in zip(decided["severity"], decided["unusual"], shown)]
        decided[f"lower_h{h}"] = decided[f"prediction_h{h}"].where(shown) - radius
        decided[f"upper_h{h}"] = decided[f"prediction_h{h}"].where(shown) + radius
    return decided


def _report(holdout, model, horizons, settings, end, fold_starts, data) -> dict[str, Any]:
    shown = holdout[holdout["state"].isin(DISPLAY_STATES)]
    per_horizon = []
    for h in horizons:
        rows = shown.dropna(subset=[f"actual_h{h}", f"prediction_h{h}", "charge4"])
        err = rows[f"prediction_h{h}"] - rows[f"actual_h{h}"]
        naive = rows["charge4"] - rows[f"actual_h{h}"]
        inside = rows[f"actual_h{h}"].between(rows[f"lower_h{h}"], rows[f"upper_h{h}"])
        per_horizon.append({
            "horizon": h, "n": int(len(rows)), "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err**2).mean())),
            "bias": float(err.mean()), "persistence_mae": float(naive.abs().mean()),
            "skill_vs_persistence": float(1 - err.abs().mean() / naive.abs().mean()) if naive.abs().mean() else None,
            "range_coverage": float(inside.mean()) if len(rows) else None,
        })
    by_state = []
    for state, rows in shown.groupby("state"):
        rows = rows.dropna(subset=[f"actual_h{max(horizons)}"])
        err = rows[f"prediction_h{max(horizons)}"] - rows[f"actual_h{max(horizons)}"]
        naive = rows["charge4"] - rows[f"actual_h{max(horizons)}"]
        by_state.append({"state": state, "n": int(len(rows)), "mae": float(err.abs().mean()) if len(rows) else None,
                         "persistence_mae": float(naive.abs().mean()) if len(rows) else None})
    beta = [{"horizon": h, "pci": float(b[0]), "nut": float(b[1]), "slag": float(b[2])}
            for h, b in zip(horizons, model.beta)]
    longest = per_horizon[-1]
    checks = {
        "Beats persistence at every horizon": all((r["skill_vs_persistence"] or -1) > 0 for r in per_horizon),
        "PCI and nut coke lower coke, slag raises it (every horizon)": all(
            b["pci"] <= 0 and b["nut"] <= 0 and b["slag"] >= 0 for b in beta),
        "Ranges cover 80-98% of the holdout": all(
            r["range_coverage"] is not None and 0.80 <= r["range_coverage"] <= 0.98 for r in per_horizon),
        "At least 60% of holdout hours shown": float(holdout["state"].isin(DISPLAY_STATES).mean()) >= 0.60,
        f"+{max(horizons)} h error at most 10 kg/THM": (longest["mae"] or 99) <= 10.0,
    }
    return {
        "trained_until": str(model.trained_until), "data_end": str(end),
        "holdout": [str(fold_starts[-1]), str(end)], "train_hours": int(len(model.train_rows)),
        "train_start": str(model.train_rows.min()), "per_horizon": per_horizon, "by_state": by_state,
        "availability": {"hours": int(len(holdout)), "shown": int(holdout["state"].isin(DISPLAY_STATES).sum()),
                         "by_state": holdout["state"].value_counts().to_dict()},
        "beta": beta, "checks": checks, "settings": asdict(settings),
    }


# ---- bundle -----------------------------------------------------------------


def write_bundle(result: TrainingResult, directory: Path, *, base_bundle: Path) -> Path:
    """Write a candidate in the app's array format (no pickles)."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    m = result.model
    ridge = m.ridge
    arrays = {
        "beta": m.beta.astype(float),
        "ridge_trans_med": np.asarray(ridge.trans.med, float), "ridge_trans_mean": np.asarray(ridge.trans.mean, float),
        "ridge_trans_std": np.asarray(ridge.trans.std, float),
        "ridge_scaler_mean": np.asarray(ridge.scale.mean_, float), "ridge_scaler_scale": np.asarray(ridge.scale.scale_, float),
        "ridge_coef": np.asarray(ridge.model.coef_, float), "ridge_intercept": np.asarray(ridge.model.intercept_, float),
    }
    meta = []
    for k, net in enumerate(m.attention):
        arrays[f"att{k}_trans_med"] = np.asarray(net.trans.med, float)
        arrays[f"att{k}_trans_mean"] = np.asarray(net.trans.mean, float)
        arrays[f"att{k}_trans_std"] = np.asarray(net.trans.std, float)
        for name in ("Wg", "bg", "Wa", "Wr", "be", "Wq", "bt", "wo", "bo", "Ws"):
            arrays[f"att{k}_{name}"] = np.asarray(net.p[name], float)
        meta.append({"seed": int(net.seed), "epochs": int(net.epochs_), "dim": int(net.dim),
                     "relative": bool(net.relative), "skip": bool(net.skip)})
    np.savez(directory / "model.npz", **arrays)
    manifest = {
        "version": result.version, "name": "Structured_fitted_mix_hourly_path",
        "trained_until": str(m.trained_until), "horizons": list(m.horizons),
        "target": {"definition": "4-h charged coke rate at row t+h: 1000*sum(COKE_CALC_MT[t+h-3..t+h])/sum(production[t+h-3..t+h])",
                   "rows_ahead": [min(m.horizons), max(m.horizons)], "unit": "kg/THM"},
        "features": FEATURES, "sequence_hours": 12, "delta_scale": DELTA_SCALE, "mix": {"linear": 0.5, "attention": 0.5},
        "attention": meta,
        "controls": {"order": PSN, "reference_hours": 24, "reference_min_periods": 12,
                     "clip_lower": CLIP[0].tolist(), "clip_upper": CLIP[1].tolist(),
                     "coefficients_kg_coke_per_kg": m.beta.tolist()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    thresholds = json.loads((Path(base_bundle) / "policy.json").read_text(encoding="utf-8"))["thresholds"]
    (directory / "policy.json").write_text(json.dumps({"parameters": result.parameters, "thresholds": thresholds},
                                                      indent=2, default=float), encoding="utf-8")
    np.savez(directory / "support.npz", **result.support)
    shutil.copy(Path(base_bundle) / "physics_config.json", directory / "physics_config.json")
    (directory / "report.json").write_text(json.dumps(result.report, indent=2, default=str), encoding="utf-8")
    keep = ["state", "severity", "charge4", "observed_charge4"] + [
        f"{p}_h{h}" for h in m.horizons for p in ("prediction", "actual", "lower", "upper")]
    result.holdout[keep].to_csv(directory / "holdout.csv.gz", compression="gzip")
    return directory


__all__ = ["FEATURES", "TrainingResult", "TrainingSettings", "train", "write_bundle"]
