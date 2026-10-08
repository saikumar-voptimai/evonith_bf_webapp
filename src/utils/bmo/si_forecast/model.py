"""NumPy-only inference for versioned BF2 ExtraTrees Si model bundles.

Format 1 is the shipped hourly 2-3 h model. Format 2 contains one independent
forest for each five-minute-aligned horizon (Now, +1 h, +2 h and +3 h).
Training candidates are always exported to numeric arrays, so page inference
does not deserialize sklearn estimators or import research code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


_REQUIRED_ARRAYS = {
    "left",
    "right",
    "feature",
    "threshold",
    "value",
    "roots",
    "median",
    "clip_low",
    "clip_high",
    "scale_mean",
    "scale_scale",
}


class _Forest:
    def __init__(self, source: Path, expected_sha: str, feature_count: int) -> None:
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if not expected_sha or expected_sha != actual:
            raise ValueError("Silicon model artifact checksum mismatch")
        with np.load(source, allow_pickle=False) as arrays:
            self.arrays = {name: arrays[name] for name in arrays.files}
        missing = sorted(_REQUIRED_ARRAYS - set(self.arrays))
        if missing:
            raise ValueError(f"Silicon model arrays missing: {missing}")
        a = self.arrays
        nodes = len(a["feature"])
        if any(
            len(a[name]) != nodes
            for name in ("left", "right", "threshold", "value")
        ):
            raise ValueError("Inconsistent silicon tree arrays")
        if any(
            len(a[name]) != feature_count
            for name in ("median", "clip_low", "clip_high", "scale_mean", "scale_scale")
        ):
            raise ValueError("Silicon preprocessing/schema mismatch")

    def diagnostics(self, values: np.ndarray, features: tuple[str, ...]) -> dict[str, Any]:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(features):
            raise ValueError("Invalid silicon feature matrix shape")
        if not x.size:
            return {"missing_feature_fraction": 0.0, "clipped_feature_fraction": 0.0}
        finite = np.isfinite(x)
        clipped = finite & (
            (x < self.arrays["clip_low"]) | (x > self.arrays["clip_high"])
        )
        return {
            "missing_feature_fraction": float(1.0 - finite.mean()),
            "clipped_feature_fraction": float(clipped.mean()),
            "missing_features": (
                [
                    name
                    for name, present in zip(features, finite[0], strict=True)
                    if not present
                ]
                if len(x) == 1
                else []
            ),
        }

    def predict(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if len(x) == 0:
            return np.empty(0, dtype=np.float64)
        a = self.arrays
        x = np.where(np.isfinite(x), x, a["median"])
        x = np.clip(x, a["clip_low"], a["clip_high"])
        x = ((x - a["scale_mean"]) / a["scale_scale"]).astype(np.float32)
        result = np.zeros(len(x), dtype=np.float64)
        for root in a["roots"]:
            nodes = np.full(len(x), root, dtype=np.int64)
            active = np.flatnonzero(a["left"][nodes] >= 0)
            while len(active):
                current = nodes[active]
                go_left = (
                    x[active, a["feature"][current]] <= a["threshold"][current]
                )
                nodes[active] = np.where(
                    go_left, a["left"][current], a["right"][current]
                )
                active = active[a["left"][nodes[active]] >= 0]
            result += a["value"][nodes]
        return result / len(a["roots"])


class SiliconExtraTreesPredictor:
    """Load and run either a legacy or multi-horizon silicon bundle."""

    def __init__(self, bundle_dir: str | Path) -> None:
        self.root = Path(bundle_dir)
        self.metadata = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.format_version = int(self.metadata.get("format_version", -1))
        if self.format_version not in {1, 2}:
            raise ValueError("Unsupported silicon bundle format")
        self.features = tuple(str(value) for value in self.metadata["feature_order"])
        if self.format_version == 1:
            specifications: list[dict[str, Any]] = [
                {
                    "minutes": 120,
                    "target_window_minutes": 60,
                    "artifact": "forest.npz",
                    "sha256": self.metadata.get("forest_sha256"),
                    "interval_half_width": None,
                    "legacy": True,
                }
            ]
        else:
            specifications = [dict(row) for row in self.metadata.get("horizons", [])]
            if not specifications:
                raise ValueError("Multi-horizon silicon bundle has no horizons")

        self.horizon_specs: dict[int, dict[str, Any]] = {}
        self._forests: dict[int, _Forest] = {}
        for specification in specifications:
            minutes = int(specification["minutes"])
            if minutes in self._forests:
                raise ValueError(f"Duplicate silicon horizon: {minutes}")
            artifact = self.root / str(specification["artifact"])
            self._forests[minutes] = _Forest(
                artifact,
                str(specification.get("sha256") or ""),
                len(self.features),
            )
            self.horizon_specs[minutes] = specification

    @property
    def model_id(self) -> str:
        return str(self.metadata.get("model_id") or self.root.name)

    @property
    def version(self) -> str:
        return str(self.metadata.get("version") or self.root.name)

    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self._forests))

    @property
    def cadence_minutes(self) -> int:
        return 5 if self.format_version >= 2 else 60

    @property
    def is_multi_horizon(self) -> bool:
        return self.format_version >= 2

    def target_window_minutes(self, horizon_minutes: int) -> int:
        return int(self.horizon_specs[int(horizon_minutes)].get("target_window_minutes", 5))

    def interval_half_width(self, horizon_minutes: int) -> float | None:
        value = self.horizon_specs[int(horizon_minutes)].get("interval_half_width")
        return float(value) if value is not None else None

    def input_diagnostics(
        self, values: np.ndarray, horizon_minutes: int | None = None
    ) -> dict[str, Any]:
        horizon = int(horizon_minutes if horizon_minutes is not None else self.horizons[0])
        return self._forests[horizon].diagnostics(values, self.features)

    def predict_array(
        self,
        values: np.ndarray,
        feature_names: Iterable[str],
        *,
        horizon_minutes: int | None = None,
    ) -> np.ndarray:
        if tuple(feature_names) != self.features:
            raise ValueError("Silicon feature names/order do not match manifest")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError("Invalid silicon feature matrix shape")
        horizon = int(horizon_minutes if horizon_minutes is not None else self.horizons[0])
        if horizon not in self._forests:
            raise ValueError(f"Silicon bundle has no {horizon}-minute horizon")
        return self._forests[horizon].predict(x)

    def _ordered_values(self, frame: Any) -> np.ndarray:
        if frame.columns.has_duplicates:
            raise ValueError("Duplicate silicon feature column names")
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing silicon feature columns: {missing}")
        return frame.loc[:, self.features].to_numpy(dtype=float)

    def predict_frame(
        self, frame: Any, *, horizon_minutes: int | None = None
    ) -> np.ndarray:
        horizon = int(horizon_minutes if horizon_minutes is not None else self.horizons[0])
        return self.predict_array(
            self._ordered_values(frame), self.features, horizon_minutes=horizon
        )

    def predict_horizons(self, frame: Any) -> Mapping[int, np.ndarray]:
        values = self._ordered_values(frame)
        return {
            horizon: self._forests[horizon].predict(values)
            for horizon in self.horizons
        }


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = ["SiliconExtraTreesPredictor", "sha256_file"]
