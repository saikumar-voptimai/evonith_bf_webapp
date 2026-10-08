"""NumPy inference for the charged-coke model (structured linear + attention).

What it predicts, for each horizon h (rows ahead of the latest complete row t):
the 4-hour charged coke rate as it will read at row t+h,
1000 x sum(COKE_CALC_MT[t+h-3..t+h]) / sum(production[t+h-3..t+h]). A trained
bundle lists its horizons (1..5 for the hourly-path model; the research's
frozen bundle has the single horizon 5, its rows t+2..t+5 block).

How: the current 4-hour charged ratio (the anchor), plus a residual that is the
50/50 mix of a ridge branch and the mean of three attention branches, plus an
explicit control layer per horizon: beta_h . clip(control - its trailing 24-h
mean) for PCI, nut coke and expected slag.

The arithmetic is a line-for-line port of the research package's
``estimators.py`` and ``models.py`` prediction paths. Weights come from
``model.npz``, written once by ``scripts/export_charged_coke_bundle.py``, so
no pickle is loaded and no particular scikit-learn version is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BUNDLE_DIR = (
    Path(__file__).resolve().parents[3] / "assets/models/bmo_charged_coke/20260915"
)


def _expit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = np.exp(x - x.max(axis=axis, keepdims=True))
    return shifted / shifted.sum(axis=axis, keepdims=True)


@dataclass(frozen=True)
class _Transform:
    """Median-impute, standardise and clip to +/-6, per feature."""

    med: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(x), x, self.med)
        return np.clip((filled - self.mean) / self.std, -6.0, 6.0)


def _summarize(x: np.ndarray) -> np.ndarray:
    """The ridge branch's sequence summary; channel blocks stay ordered."""

    return np.concatenate(
        [
            x[:, -1],
            x[:, -1] - x[:, -2],
            x[:, -1] - x[:, -5],
            x[:, -1] - x[:, 0],
            np.mean(x[:, -4:], axis=1),
            np.mean(x, axis=1),
            np.std(x[:, -4:], axis=1),
        ],
        axis=1,
    )


@dataclass(frozen=True)
class _Ridge:
    trans: _Transform
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray

    def __call__(self, x: np.ndarray) -> np.ndarray:
        summary = (_summarize(self.trans(x)) - self.scaler_mean) / self.scaler_scale
        return summary @ self.coef.T + self.intercept


@dataclass(frozen=True)
class _Attention:
    """Forward pass of ``DeltaAttention`` (prediction head only)."""

    trans: _Transform
    p: dict[str, np.ndarray]
    dim: int
    relative: bool
    skip: bool

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        p, x = self.p, self.trans(raw)
        gate = _expit(x[:, -1] @ p["Wg"] + p["bg"])
        z = x * gate[:, None, :]
        rel = z - z[:, -1:]
        h = np.tanh(z @ p["Wa"] + (rel @ p["Wr"] if self.relative else 0.0) + p["be"])
        query = np.einsum("nd,odk->nok", h[:, -1], p["Wq"])
        score = np.einsum("nld,nod->nol", h, query) / np.sqrt(self.dim) + p["bt"]
        context = np.einsum("nol,nld->nod", _softmax(score, axis=2), h)
        outputs = p["bo"].shape[0]
        u = np.concatenate(
            [context, np.repeat(h[:, -1, None, :], outputs, axis=1)], axis=2
        )
        summary = np.concatenate(
            [x[:, -1], x[:, -1] - x[:, -5], x[:, -1] - x.mean(axis=1)], axis=1
        )
        pred = np.einsum("nod,od->no", u, p["wo"]) + p["bo"]
        if self.skip:
            pred = pred + summary @ p["Ws"]
        return pred


@dataclass(frozen=True)
class ChargedCokePrediction:
    """Model outputs for a batch of issue hours, ``(n, horizons)``, kg/THM.

    Attributes:
        prediction: Mixed forecast for each horizon.
        spread: Standard deviation across the four branches (disagreement).
        linear: Ridge branch, control layer included.
        attention: Mean of the attention branches, control layer included.
        control_adjustment: beta_h . clipped control deviation.
    """

    prediction: np.ndarray
    spread: np.ndarray
    linear: np.ndarray
    attention: np.ndarray
    control_adjustment: np.ndarray


class ChargedCokeModel:
    """The frozen model, loaded from its exported bundle directory."""

    def __init__(self, bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> None:
        bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        weights = np.load(bundle_dir / "model.npz")
        self.features: list[str] = list(self.manifest["features"])
        self.sequence_hours = int(self.manifest["sequence_hours"])
        self.delta_scale = float(self.manifest["delta_scale"])
        controls = self.manifest["controls"]
        self.control_order: list[str] = list(controls["order"])
        self.reference_hours = int(controls["reference_hours"])
        self.reference_min_periods = int(controls["reference_min_periods"])
        self.clip_lower = np.asarray(controls["clip_lower"], dtype=float)
        self.clip_upper = np.asarray(controls["clip_upper"], dtype=float)
        self.horizons: list[int] = [int(h) for h in self.manifest.get(
            "horizons", [self.manifest["target"]["rows_ahead"][1]]
        )]
        outputs = len(self.horizons)
        # Rows: horizons; columns: PCI, nut coke, slag (kg coke per kg).
        self.beta_by_horizon = np.asarray(weights["beta"], dtype=float).reshape(outputs, 3)
        # Coefficients of the longest horizon, which feeds the BMO level.
        self.beta = self.beta_by_horizon[-1]

        def transform(prefix: str) -> _Transform:
            return _Transform(
                weights[f"{prefix}_med"], weights[f"{prefix}_mean"], weights[f"{prefix}_std"]
            )

        self._ridge = _Ridge(
            transform("ridge_trans"),
            weights["ridge_scaler_mean"],
            weights["ridge_scaler_scale"],
            np.asarray(weights["ridge_coef"], dtype=float).reshape(outputs, -1),
            np.asarray(weights["ridge_intercept"], dtype=float).reshape(outputs),
        )
        self._attention = [
            _Attention(
                transform(f"att{k}_trans"),
                {
                    name: weights[f"att{k}_{name}"]
                    for name in ("Wg", "bg", "Wa", "Wr", "be", "Wq", "bt", "wo", "bo", "Ws")
                },
                dim=int(meta["dim"]),
                relative=bool(meta["relative"]),
                skip=bool(meta["skip"]),
            )
            for k, meta in enumerate(self.manifest["attention"])
        ]

    def control_deviation(self, controls: pd.DataFrame) -> pd.DataFrame:
        """Each control minus its trailing reference, clipped as in training.

        Args:
            controls: Hourly frame with the ``control_order`` columns
                (PCI, nut coke and expected slag, kg/THM), on a complete clock.

        Returns:
            The clipped deviations; a missing value contributes nothing.
        """

        frame = controls[self.control_order].astype(float)
        reference = frame.rolling(
            self.reference_hours, min_periods=self.reference_min_periods
        ).mean()
        return (frame - reference).fillna(0.0).clip(
            lower=self.clip_lower, upper=self.clip_upper, axis=1
        )

    def predict(
        self,
        sequences: np.ndarray,
        anchor: np.ndarray,
        deviation: np.ndarray,
    ) -> ChargedCokePrediction:
        """Forecast the 4-hour block for each issue hour.

        Args:
            sequences: ``(n, sequence_hours, len(features))`` inputs in
                ``features`` order, oldest hour first, ending at the issue row.
            anchor: ``(n,)`` trailing 4-hour charged ratio at each issue row.
            deviation: ``(n, 3)`` clipped control deviations at each issue row.

        Returns:
            The mixed forecast and its parts, one column per horizon.
        """

        x = np.asarray(sequences, dtype=float)
        if x.ndim != 3 or x.shape[1:] != (self.sequence_hours, len(self.features)):
            raise ValueError(
                f"sequences must be (n, {self.sequence_hours}, {len(self.features)}), "
                f"got {x.shape}"
            )
        base = np.asarray(anchor, dtype=float)[:, None]
        adjustment = np.asarray(deviation, dtype=float) @ self.beta_by_horizon.T
        branches = [self._ridge(x)] + [branch(x) for branch in self._attention]
        # (n, horizons, branches): every branch is a delta on the anchor.
        components = np.stack(
            [branch * self.delta_scale + base for branch in branches], axis=2
        ) + adjustment[:, :, None]
        attention = components[:, :, 1:].mean(axis=2)
        mix = self.manifest["mix"]
        return ChargedCokePrediction(
            prediction=mix["linear"] * components[:, :, 0] + mix["attention"] * attention,
            spread=components.std(axis=2),
            linear=components[:, :, 0],
            attention=attention,
            control_adjustment=adjustment,
        )


__all__ = ["ChargedCokeModel", "ChargedCokePrediction", "DEFAULT_BUNDLE_DIR"]
