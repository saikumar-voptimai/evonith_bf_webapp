"""Tests for Phase 11 dependency and boundary scripts."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_check_import_boundaries_script_passes_current_tree():
    result = _run("scripts/check_import_boundaries.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_check_import_boundaries_script_detects_fake_violations(tmp_path):
    backend = tmp_path / "apps" / "backend_api" / "app"
    frontend = tmp_path / "apps" / "frontend_streamlit" / "services"
    backend.mkdir(parents=True)
    frontend.mkdir(parents=True)
    (backend / "bad.py").write_text("import streamlit\n", encoding="utf-8")
    (frontend / "bad_api.py").write_text("from " + "app.main import app\n", encoding="utf-8")
    shim = tmp_path / "src" / "app.py"
    shim.parent.mkdir(parents=True)
    shim.write_text("from apps.frontend_streamlit.app import *\n", encoding="utf-8")

    result = _run("scripts/check_import_boundaries.py", "--root", str(tmp_path))

    assert result.returncode == 1
    assert "Backend imports frontend/UI module" in result.stdout
    assert "Frontend API adapter imports backend/heavy module" in result.stdout


def test_check_dependency_profiles_script_passes_current_metadata():
    result = _run("scripts/check_dependency_profiles.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "backend-base" in result.stdout
    assert "PASS" in result.stdout


def test_check_dependency_profiles_script_detects_missing_group(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n[dependency-groups]\nfrontend=['streamlit']\n",
        encoding="utf-8",
    )

    result = _run(
        "scripts/check_dependency_profiles.py",
        "--pyproject",
        str(pyproject),
        "--requirements-dir",
        str(tmp_path / "requirements"),
    )

    assert result.returncode == 1
    assert "Missing dependency groups" in result.stdout


def test_backend_minimal_startup_and_frontend_api_import_scripts_pass():
    backend = _run("scripts/check_backend_minimal_startup.py")
    frontend = _run("scripts/check_frontend_api_imports.py")

    assert backend.returncode == 0, backend.stdout + backend.stderr
    assert "PASS backend-minimal startup check" in backend.stdout
    assert frontend.returncode == 0, frontend.stdout + frontend.stderr
    assert "PASS frontend API import checks" in frontend.stdout


def test_backend_minimal_startup_detects_simulated_streamlit_import(monkeypatch):
    env = dict(**os_environ_without_pytest_noise())
    env["EVONITH_CHECK_SIMULATE_IMPORTED_MODULE"] = "streamlit"

    result = _run("scripts/check_backend_minimal_startup.py", env=env)

    assert result.returncode == 1
    assert "Forbidden modules loaded" in result.stdout


def os_environ_without_pytest_noise() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    return env


def test_frontend_api_import_script_detects_fake_backend_import(tmp_path):
    services = tmp_path / "apps" / "frontend_streamlit" / "services"
    services.mkdir(parents=True)
    (services / "auth_api.py").write_text("from " + "app.main import app\n", encoding="utf-8")

    result = _run("scripts/check_frontend_api_imports.py", "--root", str(tmp_path))

    assert result.returncode == 1
    assert "imports forbidden module app" in result.stdout


def test_edge_start_scripts_are_safe_and_configurable():
    for script_name in ("edge_start_backend.sh", "edge_start_frontend.sh"):
        text = (REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "OMP_NUM_THREADS" in text
        assert "OPENBLAS_NUM_THREADS" in text
        assert "EVONITH_RUNTIME_DIR" in text
        assert "EVONITH_AUTH_SECRET_KEY=" not in text
        assert "OPENAI_API_KEY=" not in text
        assert "QDRANT_API_KEY=" not in text

    backend = (REPO_ROOT / "scripts" / "edge_start_backend.sh").read_text(encoding="utf-8")
    frontend = (REPO_ROOT / "scripts" / "edge_start_frontend.sh").read_text(encoding="utf-8")
    assert "EVONITH_UVICORN_PORT" in backend
    assert "EVONITH_FRONTEND_PORT" in frontend
