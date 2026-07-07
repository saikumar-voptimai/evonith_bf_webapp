"""Canonical packaged assets for the shared furnace_data package."""

from __future__ import annotations

import os
from pathlib import Path

from furnace_data.runtime_paths import get_repo_root


_ASSET_DIR = Path(__file__).resolve().parent


def package_asset_path(*parts: str) -> Path:
    """Return a path below the packaged asset directory."""
    return _ASSET_DIR.joinpath(*parts).resolve()


def package_model_dir() -> Path:
    """Return the canonical packaged model directory."""
    return package_asset_path("models")


def package_copilot_analysis_dir() -> Path:
    """Return the canonical packaged Copilot analysis directory."""
    return package_asset_path("copilot_analysis")


def package_furnacemind_assets_dir() -> Path:
    """Return the canonical packaged FurnaceMind source asset directory."""
    return package_asset_path("furnacemind")


def model_dir_from_config(configured: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the active model directory from config or EVONITH_MODEL_DIR."""
    raw = str(configured or "").strip()
    if not raw:
        raw = os.getenv("EVONITH_MODEL_DIR", "").strip()
    if not raw:
        return package_model_dir()

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (get_repo_root() / path).resolve()


def configured_model_dir() -> Path:
    """Return the active model directory from EVONITH_MODEL_DIR or package assets."""
    return model_dir_from_config(os.getenv("EVONITH_MODEL_DIR", "").strip())


def resolve_model_asset_path(path_value: str | os.PathLike[str] | None) -> Path:
    """Resolve a model artifact path from absolute or repository-relative config."""
    path = Path(str(path_value or "")).expanduser()
    if path.is_absolute():
        return path

    repo_root = get_repo_root()
    candidates = [
        repo_root / path,
        package_model_dir() / path.name,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()