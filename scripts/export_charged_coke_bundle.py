"""Export the frozen charged-coke model to version-free files the webapp loads.

The ChatGPT research package ships the September 15 model as pickles written
with scikit-learn 1.8 / NumPy 2. The webapp runs older versions and should not
unpickle at runtime anyway, so this script reads them once, through loaders
that admit only the classes they are known to contain, and writes:

    src/assets/models/bmo_charged_coke/<version>/
        model.npz      ridge and three attention branches, control coefficients
        manifest.json  features, target definition, mixing and clipping rules
        physics_config.json  fuel ash, dust and slag settings behind expected slag
        policy.json    display policy: error ranges, limits, support thresholds
        support.npz    robust-scaled historical support set for the novelty check

It also writes parity fixtures for the tests: the original classes' outputs on
fixed random inputs, and the frozen-period rows of the operator replay.

Usage:
    python scripts/export_charged_coke_bundle.py <extracted package dir> [version]

The package dir is the folder holding ``charged_final/`` and ``all_conditions/``.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSION = "20260915"

_MODEL_CLASSES = {
    ("models", "Model"),
    ("estimators", "LinearDelta"),
    ("estimators", "SequenceTransform"),
    ("estimators", "DeltaAttention"),
    ("sklearn.preprocessing._data", "StandardScaler"),
    ("sklearn.linear_model._ridge", "Ridge"),
}
_SUPPORT_CLASSES = {
    ("sklearn.preprocessing._data", "RobustScaler"),
    ("sklearn.neighbors._unsupervised", "NearestNeighbors"),
    ("sklearn.neighbors._kd_tree", "KDTree"),
    ("sklearn.neighbors._kd_tree", "newObj"),
    ("sklearn.metrics._dist_metrics", "EuclideanDistance64"),
    ("sklearn.metrics._dist_metrics", "newObj"),
}
_NUMPY_CLASSES = {
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}
# NumPy 2 pickles name numpy._core; NumPy 1.x only has numpy.core.
_NUMPY_RENAMES = {"numpy._core.multiarray": "numpy.core.multiarray"}

# Columns of the replay ledger the policy tests need.
_REPLAY_COLUMNS = [
    "wind", "blast", "prod", "pci", "eta", "raft", "dp", "gas_sum", "charge4",
    "slag", "nut", "wind_cv4", "prod_cv4", "pci_cv4", "blast_range4",
    "eta_range4", "charge_cv4", "history_count", "mass_count", "reference_count",
    "stable_original", "source_present", "sensor_frozen", "identity_bad",
    "operating_readings_valid", "shadow_possible", "data_ready", "severity",
    "outside_core_count", "major_level", "major_dynamic", "prediction", "spread",
    "novelty", "unusual", "state", "radius90", "observed_charge4", "actual",
    "linear", "attention", "display_forecast", "lower", "upper", "last_forecast",
    "last_issue", "last_window_end",
]
FREEZE = pd.Timestamp("2026-09-15")
REPLAY_END = pd.Timestamp("2026-10-04 23:00")
# Eight days of feature warm-up plus 48 h of decision context before the freeze.
SOURCE_WINDOW_START = FREEZE - pd.Timedelta(days=8) - pd.Timedelta(hours=48)


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, file, allowed: set[tuple[str, str]]) -> None:
        super().__init__(file)
        self._allowed = allowed | _NUMPY_CLASSES

    def find_class(self, module: str, name: str):  # noqa: D401 - pickle hook
        if (module, name) not in self._allowed:
            raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
        module = _NUMPY_RENAMES.get(module, module) if not _has_module(module) else module
        return super().find_class(module, name)


def _has_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _load(path: Path, allowed: set[tuple[str, str]]):
    with open(path, "rb") as handle:
        return _RestrictedUnpickler(handle, allowed).load()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transform_arrays(prefix: str, trans) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_med": np.asarray(trans.med, dtype=float),
        f"{prefix}_mean": np.asarray(trans.mean, dtype=float),
        f"{prefix}_std": np.asarray(trans.std, dtype=float),
    }


def export(package_dir: Path, version: str) -> Path:
    model_pkl = package_dir / "charged_final/out/models/4_Structured_fitted_mix.pkl"
    out_dir = package_dir / "all_conditions/out"
    sys.path.insert(0, str(package_dir / "charged_final"))
    bundle = _load(model_pkl, _MODEL_CLASSES)
    support = _load(out_dir / "policy_support.pkl", _SUPPORT_CLASSES)

    branches = bundle["models"]
    ridge, attention = branches[0], branches[1:]
    assert ridge.kind == "ridge" and all(m.kind == "attention" for m in attention)
    assert all(m.ids == ridge.ids for m in branches)
    assert all(m.form == "delta" and m.scale == ridge.scale for m in branches)

    arrays: dict[str, np.ndarray] = {
        "beta": np.asarray(bundle["coefficients"], dtype=float),
        **_transform_arrays("ridge_trans", ridge.model.trans),
        "ridge_scaler_mean": np.asarray(ridge.model.scale.mean_, dtype=float),
        "ridge_scaler_scale": np.asarray(ridge.model.scale.scale_, dtype=float),
        "ridge_coef": np.asarray(ridge.model.model.coef_, dtype=float).reshape(-1),
        "ridge_intercept": np.asarray(ridge.model.model.intercept_, dtype=float).reshape(-1),
    }
    attention_meta = []
    for k, branch in enumerate(attention):
        net = branch.model
        arrays.update(_transform_arrays(f"att{k}_trans", net.trans))
        for name in ("Wg", "bg", "Wa", "Wr", "be", "Wq", "bt", "wo", "bo", "Ws"):
            arrays[f"att{k}_{name}"] = np.asarray(net.p[name], dtype=float)
        attention_meta.append(
            {"seed": int(net.seed), "epochs": int(net.epochs_), "dim": int(net.dim),
             "relative": bool(net.relative), "skip": bool(net.skip)}
        )

    target_dir = ROOT / "src/assets/models/bmo_charged_coke" / version
    target_dir.mkdir(parents=True, exist_ok=True)
    np.savez(target_dir / "model.npz", **arrays)

    features = list(bundle["features"])
    manifest = {
        "version": version,
        "name": "Structured_fitted_mix",
        "frozen_at": "2026-09-15T00:00:00",
        "last_training_target": "2026-09-14T23:00:00",
        "target": {
            "definition": "1000 * sum(COKE_CALC_MT[t+2..t+5]) / sum(PRODUCTIONTONNESPERHR[t+2..t+5])",
            "rows_ahead": [2, 5],
            "plain": "Charged coke over the 4-hour block starting 1 h and ending 5 h after the latest complete hour.",
            "unit": "kg/THM",
        },
        "anchor": "1000 * sum(COKE_CALC_MT[t-3..t]) / sum(PRODUCTIONTONNESPERHR[t-3..t])",
        "features": features,
        "sequence_hours": int(attention[0].model.p["bt"].shape[1]),
        "delta_scale": float(ridge.scale),
        "mix": {"linear": 0.5, "attention": 0.5},
        "attention": attention_meta,
        "controls": {
            "order": ["pci", "nut", "slag"],
            "reference_hours": int(bundle["reference_hours"]),
            "reference_min_periods": 12,
            "clip_lower": [-40.0, -20.0, -60.0],
            "clip_upper": [40.0, 20.0, 60.0],
            "coefficients_kg_coke_per_kg": [float(v) for v in bundle["coefficients"]],
        },
        "source": {
            "model_pickle_sha256": _sha256(model_pkl),
            "policy_support_sha256": _sha256(out_dir / "policy_support.pkl"),
            "package": "BF02_All_Conditions_Research_Package.zip",
        },
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # The physics the model's expected-slag input was computed with, frozen so
    # later edits to setting_bmo.yml cannot change a model input.
    import yaml

    snapshot_cfg = package_dir / "charged_final/repo_snapshot/src/config/setting_bmo.yml"
    bmo = yaml.safe_load(snapshot_cfg.read_text(encoding="utf-8"))["bmo"]
    (target_dir / "physics_config.json").write_text(
        json.dumps(
            {
                "fuel_ash_inputs": bmo["fuel_ash_inputs"],
                "dust_inputs": bmo["dust_inputs"],
                "slag_balance": bmo["slag_balance"],
                "reference_coke_kg_per_thm": 300.0,
                "source": {"file": "repo_snapshot/src/config/setting_bmo.yml", "sha256": _sha256(snapshot_cfg)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    thresholds = json.loads((out_dir / "thresholds.json").read_text(encoding="utf-8"))
    parameters = json.loads((out_dir / "policy_parameters.json").read_text(encoding="utf-8"))
    (target_dir / "policy.json").write_text(
        json.dumps({"parameters": parameters, "thresholds": thresholds}, indent=2),
        encoding="utf-8",
    )
    scaler, neighbours = support["scaler"], support["neighbors"]
    assert list(support["parameters"]["features"]) == list(parameters["features"])
    np.savez(
        target_dir / "support.npz",
        center=np.asarray(scaler.center_, dtype=float),
        scale=np.asarray(scaler.scale_, dtype=float),
        rows=np.asarray(neighbours._fit_X, dtype=float),
    )

    _write_fixtures(branches, bundle, features, out_dir, package_dir)
    return target_dir


def _write_fixtures(branches, bundle, features, out_dir: Path, package_dir: Path) -> None:
    """Original-class outputs on fixed inputs, and the frozen replay rows."""

    fixtures = ROOT / "tests/fixtures/charged_coke"
    fixtures.mkdir(parents=True, exist_ok=True)
    ridge = branches[0]
    rng = np.random.default_rng(20260915)
    n, hours = 48, int(branches[1].model.p["bt"].shape[1])
    mean, std = ridge.model.trans.mean, ridge.model.trans.std
    x = mean + std * rng.normal(size=(n, hours, len(features)))
    x[rng.random(x.shape) < 0.03] = np.nan  # exercise median imputation
    width = max(ridge.ids) + 1
    seq = np.full((n, hours, width), np.nan)
    seq[:, :, ridge.ids] = x
    anchor = rng.normal(315.0, 12.0, size=n)
    deviation = rng.uniform([-40, -20, -60], [40, 20, 60], size=(n, 3))
    components = np.column_stack([b.predict(seq, anchor) for b in branches])
    components = components + (deviation @ np.asarray(bundle["coefficients"]))[:, None]
    np.savez(
        fixtures / "golden_predictions.npz",
        sequences=x, anchor=anchor, deviation=deviation, components=components,
        prediction=0.5 * components[:, 0] + 0.5 * components[:, 1:].mean(axis=1),
        spread=components.std(axis=1),
    )

    replay = pd.read_csv(out_dir / "operator_replay.csv.gz", parse_dates=["time"])
    audit = pd.read_csv(out_dir / "all_hour_audit.csv.gz", parse_dates=["time"])
    window = replay["time"].between(pd.Timestamp("2026-09-13"), pd.Timestamp("2026-10-04 23:00"))
    frame = replay.loc[window, ["time", *_REPLAY_COLUMNS]].copy()
    frame["audit_reasons_json"] = audit.set_index("time").loc[frame["time"], "reasons_json"].to_numpy()
    frame["audit_codes_json"] = audit.set_index("time").loc[frame["time"], "codes_json"].to_numpy()
    frame["state_reasons_json"] = replay.loc[window, "reasons_json"].to_numpy()
    frame.to_csv(fixtures / "frozen_replay.csv.gz", index=False, compression="gzip")

    # The exact bundled export over the window the frozen replay needs, so the
    # end-to-end test runs from source rows, not from intermediates.
    source = pd.read_csv(package_dir / "upload/furnace_dataset-1.csv", parse_dates=["time"])
    source = source[source["time"].between(SOURCE_WINDOW_START, REPLAY_END)]
    source.to_csv(fixtures / "source_window.csv.gz", index=False, compression="gzip")

    # The research scenario examples (21 Sep 09:00): exact what-if references.
    (fixtures / "scenario_examples.json").write_text(
        (package_dir / "charged_final/out/scenario_examples.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Recovery count and last shown forecast entering the freeze, from the
    # replay's pre-freeze rows (earlier rolling models, not this bundle).
    pre = replay.set_index("time").loc[: FREEZE - pd.Timedelta(hours=1)]
    streak = 0
    for state in pre["state"]:
        streak = streak + 1 if state in {"cruise", "unsettled", "off_cruise", "recovering"} else 0
    last_issue = pd.Timestamp(pre.iloc[-1]["last_issue"])
    (fixtures / "freeze_seed.json").write_text(
        json.dumps(
            {"initial_streak": streak, "last_issue": str(last_issue),
             "last_value": float(pre.loc[last_issue, "display_forecast"])},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    written = export(Path(sys.argv[1]).resolve(), sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VERSION)
    print(f"Exported to {written}")
