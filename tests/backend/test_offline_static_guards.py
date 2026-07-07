from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_DIRS = [
    ROOT / "packages" / "furnace-data" / "furnace_data",
    ROOT / "apps" / "backend_api" / "app",
    ROOT / "apps" / "frontend_streamlit",
]


def _python_files():
    for base in APP_DIRS:
        for path in base.rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_no_runtime_path_patching_in_app_code() -> None:
    offenders = [
        path
        for path in _python_files()
        if "sys.path.insert" in _read(path) and path.name != "app.py"
    ]
    assert offenders == []


def test_no_hardcoded_material_fallback_map() -> None:
    offenders = [path for path in _python_files() if "_fallback_material_code" in _read(path)]
    assert offenders == []


def test_offline_app_paths_do_not_import_influx_offline_fetcher() -> None:
    offenders = []
    for path in _python_files():
        if "packages/furnace-data/furnace_data/influx" in path.as_posix():
            continue
        text = _read(path)
        if "from furnace_data.influx.offline" in text:
            offenders.append(path)
    assert offenders == []


def test_old_influx_offline_module_is_removed() -> None:
    assert not (ROOT / "packages" / "furnace-data" / "furnace_data" / "influx" / "offline.py").exists()


def test_bmo_and_static_dataset_paths_use_neutral_offline_api() -> None:
    checked_roots = [
        ROOT / "packages" / "furnace-data" / "furnace_data" / "bmo" / "data",
        ROOT / "apps" / "frontend_streamlit" / "data" / "ml",
        ROOT / "apps" / "frontend_streamlit" / "custom_pages" / "9_Blend_Optimizer.py",
        ROOT / "packages" / "furnace-data" / "furnace_data" / "dataset",
    ]
    offenders = []
    for root in checked_roots:
        paths = [root] if root.is_file() else list(root.rglob("*.py"))
        for path in paths:
            text = _read(path)
            if "furnace_data.influx.offline" in text:
                offenders.append(path)
    assert offenders == []


def test_app_code_does_not_import_old_neon_package() -> None:
    offenders = [
        path
        for path in _python_files()
        if "furnace_data.neon_db" in _read(path)
    ]
    assert offenders == []


def test_no_offline_influx_rollback_copy_remains() -> None:
    needles = [
        "InfluxDB rollback",
        "source='influx'",
        'source="influx"',
        '"source": "influx"',
    ]
    offenders = [
        path
        for path in _python_files()
        if any(needle in _read(path) for needle in needles)
    ]
    assert offenders == []


def test_sqlite_usage_is_limited_to_ticketing_and_runtime_audit() -> None:
    allowed = {
        "apps/backend_api/app/repositories/audit_repository.py",
        "apps/backend_api/app/repositories/feedback_repository.py",
        "apps/backend_api/app/repositories/furnacemind_conversation_repository.py",
        "apps/backend_api/app/repositories/furnacemind_document_repository.py",
        "apps/backend_api/app/repositories/furnacemind_run_repository.py",
        "apps/backend_api/app/services/feedback_migration_service.py",
    }
    offenders = []
    for path in _python_files():
        text = _read(path).lower()
        if "sqlite://" not in text and "sqlite3" not in text:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("apps/frontend_streamlit/data/tickets/") or relative in allowed:
            continue
        offenders.append(path)
    assert offenders == []