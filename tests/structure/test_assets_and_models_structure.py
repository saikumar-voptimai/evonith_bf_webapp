"""Asset and model-location safeguards for post-Phase 13 cleanup."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys

import joblib

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.services.model_registry_service import ModelRegistryService
from furnace_data.assets import (
    package_copilot_analysis_dir,
    package_furnacemind_assets_dir,
    package_model_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ASSETS = REPO_ROOT / "apps" / "frontend_streamlit" / "assets"
CANONICAL_MODELS = REPO_ROOT / "packages" / "furnace-data" / "furnace_data" / "assets" / "models"
RUNTIME_FILE_NAMES = {"cache_meta.json", "control_bounds.json"}
MODEL_SUFFIXES = {".joblib", ".onnx", ".pkl", ".pth"}


class _TinyModel:
    def predict(self, frame):
        return [1.0 for _ in range(len(frame))]


def test_no_model_archive_folders_remain_in_frontend_app_assets():
    archive_dirs = [
        path
        for path in FRONTEND_ASSETS.rglob("*")
        if path.is_dir() and path.name.startswith("old_")
    ]

    assert archive_dirs == []
    assert not (FRONTEND_ASSETS / "models").exists()


def test_model_registry_lists_only_active_allowed_model_ids(monkeypatch):
    monkeypatch.delenv("EVONITH_MODEL_DIR", raising=False)

    registry = ModelRegistryService()
    model_names = {entry["name"] for entry in registry.list_models()}

    assert model_names
    assert "unitcost_fuel_model" in model_names
    assert "bmo_fuel.feature_manifest" in model_names
    assert all("old_" not in name for name in model_names)
    assert all("archive" not in name.lower() for name in model_names)


def test_model_registry_does_not_recursively_expose_old_dirs(tmp_path):
    joblib.dump(_TinyModel(), tmp_path / "active.joblib")
    old_dir = tmp_path / "old_26_14"
    old_dir.mkdir()
    joblib.dump(_TinyModel(), old_dir / "stale.joblib")
    nested_dir = tmp_path / "bmo_fuel" / "nested"
    nested_dir.mkdir(parents=True)
    joblib.dump(_TinyModel(), nested_dir / "hidden.joblib")

    settings = BackendSettings(model_dir=str(tmp_path))
    registry = ModelRegistryService(settings)
    model_names = {entry["name"] for entry in registry.list_models()}

    assert "active" in model_names
    assert "old_26_14.stale" not in model_names
    assert "bmo_fuel.nested.hidden" not in model_names


def test_model_loading_remains_lazy(monkeypatch):
    monkeypatch.delenv("EVONITH_MODEL_DIR", raising=False)

    registry = ModelRegistryService()

    assert registry._cache == {}
    assert registry.list_models()
    assert registry._cache == {}


def test_backend_app_import_does_not_load_model_files(monkeypatch):
    monkeypatch.delenv("EVONITH_MODEL_DIR", raising=False)

    from apps.backend_api.app.main import app

    registry = app.state.model_registry_service

    assert registry._cache == {}


def test_frontend_assets_do_not_contain_backend_model_files():
    model_files = [
        path
        for path in FRONTEND_ASSETS.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    ]

    assert model_files == []
    assert not (FRONTEND_ASSETS / "models").exists()


def test_furnacemind_assets_are_loaded_from_canonical_source_runtime_location():
    source_dir = package_furnacemind_assets_dir()

    assert source_dir == REPO_ROOT / "packages" / "furnace-data" / "furnace_data" / "assets" / "furnacemind"
    assert (source_dir / "TOOLS1.md").exists()
    assert (source_dir / "SKILLS_BESTSHIFT.md").exists()

    context_module = "apps.frontend_streamlit.agents.furnacemind.context"
    context = sys.modules.get(context_module)
    if context is not None and not hasattr(context, "_FURNACEMIND_SOURCE_DIR"):
        sys.modules.pop(context_module, None)
    context = importlib.import_module(context_module)

    assert context._FURNACEMIND_SOURCE_DIR == source_dir
    assert context._source_context_path("SKILLS_BESTSHIFT.md") == source_dir / "SKILLS_BESTSHIFT.md"

    skills_module = "apps.frontend_streamlit.agents.furnacemind.skills"
    sys.modules.pop(skills_module, None)
    skills = importlib.import_module(skills_module)

    assert skills._PARAMS_PATH == source_dir / "skill_params.yml"
    assert skills._PARAMS_PATH.exists()
    assert not hasattr(skills, "_LEGACY_PARAMS_PATH")
    assert skills._PARAMS["tier1"]


def test_copilot_and_model_assets_use_canonical_source_locations():
    assert package_model_dir() == CANONICAL_MODELS
    assert (package_model_dir() / "unitcost_fuel_model.json").exists()
    assert (package_model_dir() / "unitcost_opt_dec.pkl").exists()
    assert (package_copilot_analysis_dir() / "BURDEN_UNITCOST.md").exists()


def test_no_generated_runtime_files_exist_under_apps_or_packages():
    roots = [REPO_ROOT / "apps", REPO_ROOT / "packages"]
    generated: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            if path.name in RUNTIME_FILE_NAMES or path.name.startswith("furnace_dataset_"):
                generated.append(path.relative_to(REPO_ROOT))
            elif path.name == "furnace_dataset.csv":
                generated.append(path.relative_to(REPO_ROOT))
            elif path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".log"}:
                generated.append(path.relative_to(REPO_ROOT))

    assert generated == []
