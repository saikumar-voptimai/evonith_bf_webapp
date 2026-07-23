"""Trusted V-Sense model-bundle readiness metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from furnace_data.assets import resolve_model_asset_path
from furnace_data.config import load_config
from furnace_data.vsense.catalog import _OPTIMIZATION_ID_BY_LABEL  # noqa: PLC2701


def model_readiness() -> list[dict[str, Any]]:
    """Return model readiness without leaking artifact paths."""

    cfg = load_config("setting_vsense.yml")
    rows: list[dict[str, Any]] = []
    for label, raw in (cfg.get("Optimisation") or {}).items():
        optimization_id = _OPTIMIZATION_ID_BY_LABEL.get(str(label))
        if not optimization_id:
            continue
        model_path = resolve_model_asset_path(raw.get("model"))
        scaler_path = resolve_model_asset_path(raw.get("scaling"))
        model_ok = _trusted_artifact_exists(model_path)
        scaler_ok = _trusted_artifact_exists(scaler_path)
        rows.append(
            {
                "optimization_type_id": optimization_id,
                "bundle_version": _bundle_version(model_path, scaler_path),
                "status": "ready" if model_ok and scaler_ok else "unavailable",
            }
        )
    return rows


def bundle_version_for(optimization_type_id: str) -> str:
    """Return a stable public bundle version for one optimization type."""

    for row in model_readiness():
        if row["optimization_type_id"] == optimization_type_id:
            return str(row["bundle_version"])
    return "model-bundle-unavailable"


def _trusted_artifact_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _bundle_version(model_path: Path, scaler_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (model_path, scaler_path):
        digest.update(path.name.encode("utf-8", errors="ignore"))
        try:
            stat = path.stat()
        except OSError:
            digest.update(b":missing")
        else:
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    return f"model-bundle-{digest.hexdigest()[:16]}"
