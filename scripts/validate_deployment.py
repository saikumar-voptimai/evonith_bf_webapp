#!/usr/bin/env python
"""Validate Phase 13 deployment configuration before startup or cutover."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

from deployment_common import (
    CheckResult,
    bool_env,
    disk_free_mb,
    env_path,
    exit_code,
    is_placeholder_secret,
    is_production_like,
    min_free_mb,
    print_results,
    repo_root,
    scan_files_for_secrets,
    unsafe_runtime_path,
    validate_writable,
)


CUTOVER_FLAGS = [
    "USE_BACKEND_API",
    "USE_BACKEND_API_AUTH",
    "USE_BACKEND_API_ADMIN",
    "USE_BACKEND_API_DATA_EXPLORER",
    "USE_BACKEND_API_DATASETS",
    "USE_BACKEND_API_FEEDBACK",
    "USE_BACKEND_API_MATERIAL_BALANCE",
    "USE_BACKEND_API_RECOMMENDATIONS",
    "USE_BACKEND_API_BLEND_OPTIMIZER",
    "USE_BACKEND_API_COPILOT",
    "USE_BACKEND_API_FURNACEMIND",
    "USE_BACKEND_API_OPS",
]


def _url_ok(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def validate(args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []
    root = repo_root()
    runtime = (args.runtime_dir or env_path("EVONITH_RUNTIME_DIR", "runtime")).resolve()
    unsafe = unsafe_runtime_path(runtime)
    if unsafe:
        results.append(CheckResult("runtime_path", "fail", unsafe, {"path": str(runtime)}))
    elif runtime.exists() and validate_writable(runtime):
        results.append(CheckResult("runtime_writable", "pass", "runtime directory exists and is writable", {"path": str(runtime)}))
    elif args.offline:
        results.append(CheckResult("runtime_writable", "warn", "runtime directory missing or not writable yet", {"path": str(runtime)}))
    else:
        results.append(CheckResult("runtime_writable", "fail", "runtime directory missing or not writable", {"path": str(runtime)}))

    try:
        free = disk_free_mb(runtime)
        status = "pass" if free >= min_free_mb() else "fail"
        results.append(CheckResult("disk_free", status, f"{free} MB free", {"free_mb": free}))
    except OSError as exc:
        results.append(CheckResult("disk_free", "fail", f"could not check disk space: {exc}"))

    auth_secret = os.getenv("EVONITH_AUTH_SECRET_KEY", "")
    if is_production_like(args.profile) and is_placeholder_secret(auth_secret):
        results.append(CheckResult("auth_secret", "fail", "production-like profile requires non-placeholder EVONITH_AUTH_SECRET_KEY"))
    else:
        results.append(CheckResult("auth_secret", "pass", "auth secret policy satisfied"))

    backend_base = args.backend_url or os.getenv("BACKEND_API_BASE_URL") or os.getenv("EVONITH_BACKEND_API_URL", "")
    if backend_base:
        results.append(CheckResult("backend_api_base_url", "pass", "backend API URL configured"))
    else:
        results.append(CheckResult("backend_api_base_url", "warn", "backend API URL not configured"))

    if (root / "apps" / "frontend_streamlit" / "app.py").exists():
        results.append(CheckResult("frontend_path", "pass", "canonical frontend app exists"))
    else:
        results.append(CheckResult("frontend_path", "fail", "canonical frontend app missing"))

    try:
        importlib.import_module("apps.backend_api.app.main")
        results.append(CheckResult("backend_import", "pass", "canonical backend app imports"))
    except Exception as exc:
        status = "fail" if args.strict and not args.offline else "warn"
        results.append(CheckResult("backend_import", status, f"backend import unavailable in this environment: {exc.__class__.__name__}"))

    for script, label in (
        ("check_repository_structure.py", "repository_structure"),
        ("check_import_boundaries.py", "import_boundaries"),
        ("check_dependency_profiles.py", "dependency_profiles"),
    ):
        path = root / "scripts" / script
        if not path.exists():
            results.append(CheckResult(label, "fail", f"{script} missing"))
        else:
            results.append(CheckResult(label, "pass", f"{script} present"))

    if bool_env("EVONITH_ENABLE_OPTIONAL_AI", False) and not os.getenv("EVONITH_COPILOT_PROVIDER"):
        results.append(CheckResult("optional_ai", "warn", "AI enabled but provider config is incomplete"))
    else:
        results.append(CheckResult("optional_ai", "pass", "AI optional config is consistent"))
    if bool_env("EVONITH_ENABLE_OPTIONAL_VECTOR", False) and not os.getenv("EVONITH_FURNACEMIND_QDRANT_URL"):
        results.append(CheckResult("optional_vector", "warn", "vector enabled but Qdrant URL is missing"))
    else:
        results.append(CheckResult("optional_vector", "pass", "vector optional config is consistent"))

    invalid_flags = [flag for flag in CUTOVER_FLAGS if os.getenv(flag, "").strip().lower() not in {"", "1", "0", "true", "false", "yes", "no", "on", "off"}]
    results.append(CheckResult("cutover_flags", "fail" if invalid_flags else "pass", "cutover flags valid" if not invalid_flags else "invalid cutover flags", {"invalid": invalid_flags}))

    if args.profile == "edge":
        missing_threads = [
            name
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            if os.getenv(name, "1") != "1"
        ]
        results.append(CheckResult("edge_threads", "fail" if missing_threads else "pass", "edge thread defaults set" if not missing_threads else "edge thread defaults missing", {"missing": missing_threads}))

    findings = scan_files_for_secrets([root / "infra", root / "scripts"])
    results.append(CheckResult("secret_scan", "fail" if findings else "pass", "no obvious secrets in infra/scripts" if not findings else "possible secrets found", {"findings": findings}))

    if not args.offline and backend_base:
        health = backend_base.rstrip("/") + "/health"
        results.append(CheckResult("backend_health", "pass" if _url_ok(health, 3) else "warn", "backend health reachable" if _url_ok(health, 3) else "backend health not reachable", {"url": health}))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["local", "staging", "edge", "production"], default=os.getenv("EVONITH_DEPLOYMENT_PROFILE", "local"))
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--frontend-url", default="")
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args(argv)
    if args.no_strict:
        args.strict = False
    results = validate(args)
    print_results(results, json_output=args.json)
    return exit_code(results, strict_warnings=args.strict)


if __name__ == "__main__":
    sys.exit(main())

