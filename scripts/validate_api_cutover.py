#!/usr/bin/env python
"""Validate frontend API-mode cutover flags and endpoint availability."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import urllib.error
import urllib.request

from deployment_common import CheckResult, bool_env, exit_code, print_results, repo_root


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

REQUIRED_PATHS = {
    "/api/v1/health",
    "/api/v1/readiness",
    "/api/v1/status",
    "/api/v1/data/sources",
    "/api/v1/datasets",
    "/api/v1/feedback/config",
    "/api/v1/material-balance/config",
    "/api/v1/recommendations/config",
    "/api/v1/blend-optimizer/context",
    "/api/v1/copilot/config",
    "/api/v1/furnacemind/config",
}

ADAPTERS = [
    "auth_api",
    "admin_api",
    "data_api",
    "dataset_api",
    "feedback_api",
    "material_balance_api",
    "recommendations_api",
    "blend_optimizer_api",
    "copilot_api",
    "furnacemind_api",
    "status_api",
    "ops_api",
]


def _openapi_paths(backend_url: str) -> set[str] | None:
    try:
        with urllib.request.urlopen(backend_url.rstrip("/") + "/openapi.json", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return set(payload.get("paths", {}))
    except Exception:
        local = repo_root() / "docs" / "api" / "openapi-v1.json"
        if local.exists():
            payload = json.loads(local.read_text(encoding="utf-8"))
            return set(payload.get("paths", {}))
    return None


def validate(args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []
    missing_flags = [flag for flag in CUTOVER_FLAGS if not bool_env(flag, False)]
    if missing_flags and args.strict and not args.allow_partial:
        results.append(CheckResult("cutover_flags", "fail", "required API cutover flags are not all true", {"missing": missing_flags}))
    elif missing_flags:
        results.append(CheckResult("cutover_flags", "warn", "partial API cutover configuration", {"missing": missing_flags}))
    else:
        results.append(CheckResult("cutover_flags", "pass", "all API cutover flags enabled"))

    if bool_env("EVONITH_ALLOW_DIRECT_MODE_FALLBACK", True):
        results.append(CheckResult("direct_mode_fallback", "pass", "direct-mode rollback flag is available"))
    else:
        results.append(CheckResult("direct_mode_fallback", "warn", "direct-mode rollback flag is disabled"))

    for adapter in ADAPTERS:
        try:
            importlib.import_module(f"apps.frontend_streamlit.services.{adapter}")
        except Exception as exc:
            results.append(CheckResult(f"adapter:{adapter}", "fail", f"import failed: {exc.__class__.__name__}"))
        else:
            results.append(CheckResult(f"adapter:{adapter}", "pass", "imported"))

    backend_url = args.backend_url or os.getenv("BACKEND_API_BASE_URL", "http://localhost:8080/api/v1")
    paths = _openapi_paths(backend_url)
    if paths is None:
        results.append(CheckResult("openapi", "warn" if args.allow_partial else "fail", "OpenAPI unavailable"))
    else:
        missing_paths = sorted(REQUIRED_PATHS - paths)
        status = "fail" if missing_paths and args.strict and not args.allow_partial else "warn" if missing_paths else "pass"
        results.append(CheckResult("openapi_paths", status, "required endpoint paths present" if not missing_paths else "required endpoint paths missing", {"missing": missing_paths}))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = validate(args)
    print_results(results, json_output=args.json)
    return exit_code(results, strict_warnings=args.strict and not args.allow_partial)


if __name__ == "__main__":
    sys.exit(main())

