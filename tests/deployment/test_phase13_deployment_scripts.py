from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def run_script(script: str, *args: str | Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bootstrap_runtime_create_and_dry_run(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    created = run_script("bootstrap_runtime.py", "--create", "--runtime-dir", runtime, "--json")

    assert created.returncode == 0, created.stdout + created.stderr
    for relative in ("cache", "jobs", "uploads/feedback", "datasets/results", "logs", "qdrant", "temp"):
        assert (runtime / relative).is_dir()

    dry_run_runtime = tmp_path / "dry-run-runtime"
    dry_run = run_script("bootstrap_runtime.py", "--dry-run", "--runtime-dir", dry_run_runtime, "--json")
    payload = json.loads(dry_run.stdout)

    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    assert payload["status"] == "warn"
    assert not dry_run_runtime.exists()


def test_bootstrap_runtime_rejects_repository_root() -> None:
    result = run_script("bootstrap_runtime.py", "--check", "--runtime-dir", REPO_ROOT)

    assert result.returncode != 0
    assert "repository root" in result.stdout


def test_validate_deployment_local_offline_and_production_secret_policy(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    local = run_script("validate_deployment.py", "--profile", "local", "--offline", "--runtime-dir", runtime)

    assert local.returncode == 0, local.stdout + local.stderr
    assert "PASS secret_scan" in local.stdout

    runtime.mkdir()
    production = run_script(
        "validate_deployment.py",
        "--profile",
        "production",
        "--offline",
        "--strict",
        "--runtime-dir",
        runtime,
        env={"EVONITH_AUTH_SECRET_KEY": "<set-a-strong-random-secret>"},
    )

    assert production.returncode != 0
    assert "EVONITH_AUTH_SECRET_KEY" in production.stdout


def test_validate_api_cutover_partial_mode_is_non_blocking() -> None:
    partial = run_script("validate_api_cutover.py", "--allow-partial", "--json")
    payload = json.loads(partial.stdout)

    assert partial.returncode == 0, partial.stdout + partial.stderr
    assert payload["status"] in {"pass", "warn"}
    assert any(check["name"] == "openapi_paths" and check["status"] == "pass" for check in payload["checks"])

    strict = run_script("validate_api_cutover.py", "--strict")
    assert strict.returncode != 0
    assert "cutover_flags" in strict.stdout


class SmokeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.endswith("/does-not-exist-for-smoke"):
            self._send_json(404, {"detail": "not found"})
            return
        if self.path.endswith("/openapi.json"):
            self._send_json(200, {"openapi": "3.1.0", "paths": {}})
            return
        self._send_json(200, {"ok": True, "path": self.path})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        request_id = self.headers.get("X-Request-ID", "phase13-smoke-test")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", request_id)
        self.end_headers()
        self.wfile.write(body)


def test_smoke_test_deployment_against_fake_backend() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), SmokeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = run_script(
            "smoke_test_deployment.py",
            "--backend-url",
            f"http://127.0.0.1:{server.server_port}/api/v1",
            "--skip-auth",
            "--timeout",
            "2",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS /health" in result.stdout
    assert "PASS structured_error" in result.stdout


def test_backup_and_restore_runtime_are_non_destructive(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    feedback_file = runtime / "uploads" / "feedback" / "ticket.txt"
    feedback_file.parent.mkdir(parents=True)
    feedback_file.write_text("hello", encoding="utf-8")
    (runtime / "audit").mkdir()
    (runtime / "audit" / "event.log").write_text("audit", encoding="utf-8")
    archive = tmp_path / "backup" / "runtime.tar.gz"

    backup = run_script("backup_runtime.py", "--runtime-dir", runtime, "--output", archive)
    assert backup.returncode == 0, backup.stdout + backup.stderr
    assert archive.is_file()

    target = tmp_path / "restored"
    dry_run = run_script("restore_runtime.py", "--backup", archive, "--target-runtime-dir", target)
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    assert not target.exists()

    applied = run_script("restore_runtime.py", "--apply", "--backup", archive, "--target-runtime-dir", target)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    assert (target / "uploads" / "feedback" / "ticket.txt").read_text(encoding="utf-8") == "hello"


def test_backup_runtime_skips_symlink_that_escapes_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    uploads = runtime / "uploads"
    uploads.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret-ish", encoding="utf-8")
    link = uploads / "outside-link.txt"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable on this platform: {exc}")

    result = run_script("backup_runtime.py", "--runtime-dir", runtime, "--dry-run")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "skipped symlink outside runtime" in result.stdout


def test_restore_runtime_rejects_path_traversal_archive(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    manifest = json.dumps({"file_count": 1}).encode("utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        manifest_info = tarfile.TarInfo("manifest.json")
        manifest_info.size = len(manifest)
        tar.addfile(manifest_info, BytesIO(manifest))
        unsafe = b"bad"
        unsafe_info = tarfile.TarInfo("../evil.txt")
        unsafe_info.size = len(unsafe)
        tar.addfile(unsafe_info, BytesIO(unsafe))

    result = run_script("restore_runtime.py", "--backup", archive, "--target-runtime-dir", tmp_path / "runtime")

    assert result.returncode != 0
    assert "unsafe archive entry" in result.stdout


def test_release_readiness_skip_tests_gate_passes_with_dirty_tree_allowed(tmp_path: Path) -> None:
    result = run_script(
        "verify_release_readiness.py",
        "--allow-dirty",
        "--skip-tests",
        env={"EVONITH_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS docs" in result.stdout
    assert "PASS validate_deployment.py" in result.stdout


def test_infra_templates_use_placeholders_and_safe_defaults() -> None:
    backend_env = (REPO_ROOT / "infra" / "env" / "backend.env.example").read_text(encoding="utf-8")
    edge_env = (REPO_ROOT / "infra" / "env" / "edge.env.example").read_text(encoding="utf-8")
    backend_script = (REPO_ROOT / "scripts" / "edge_start_backend.sh").read_text(encoding="utf-8")
    frontend_script = (REPO_ROOT / "scripts" / "edge_start_frontend.sh").read_text(encoding="utf-8")
    backend_unit = (REPO_ROOT / "infra" / "systemd" / "evonith-backend.service.example").read_text(encoding="utf-8")
    frontend_unit = (REPO_ROOT / "infra" / "systemd" / "evonith-frontend.service.example").read_text(encoding="utf-8")
    nginx = (REPO_ROOT / "infra" / "nginx" / "evonith.conf.example").read_text(encoding="utf-8")
    caddy = (REPO_ROOT / "infra" / "caddy" / "Caddyfile.example").read_text(encoding="utf-8")

    assert "EVONITH_AUTH_SECRET_KEY=<set-a-strong-random-secret>" in backend_env
    assert "EVONITH_AUTH_SECRET_KEY=<set-a-strong-random-secret>" in edge_env
    assert "DRY_RUN" in backend_script
    assert "DRY_RUN" in frontend_script
    assert "apps.backend_api.app.main:app" in backend_script
    assert "apps/frontend_streamlit/app.py" in frontend_script
    assert "PrivateTmp=true" in backend_unit
    assert "PrivateTmp=true" in frontend_unit
    assert "server_name <your-domain-or-ip>;" in nginx
    assert "<your-domain-or-ip>" in caddy
