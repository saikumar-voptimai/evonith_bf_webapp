#!/usr/bin/env python
"""Run a high-level Phase 13 release readiness gate."""

from __future__ import annotations

import argparse
import subprocess
import sys

from deployment_common import CheckResult, exit_code, print_results, repo_root, required_docs, run_python_script, scan_files_for_secrets


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo_root(), text=True, capture_output=True, check=False)


def verify(args: argparse.Namespace) -> list[CheckResult]:
    results: list[CheckResult] = []
    branch = _git(["branch", "--show-current"])
    current_branch = branch.stdout.strip()
    allowed = args.allowed_branch or "migration/backend-frontend-split"
    results.append(CheckResult("branch", "pass" if current_branch == allowed else "warn", f"current branch is {current_branch or 'unknown'}", {"allowed": allowed}))
    status = _git(["status", "--porcelain"])
    if status.stdout.strip() and not args.allow_dirty:
        results.append(CheckResult("working_tree", "fail", "working tree has changes"))
    else:
        results.append(CheckResult("working_tree", "pass" if not status.stdout.strip() else "warn", "working tree clean" if not status.stdout.strip() else "working tree dirty allowed"))

    docs_missing = [str(path.relative_to(repo_root())) for path in required_docs() if not path.exists()]
    results.append(CheckResult("docs", "fail" if docs_missing else "pass", "required docs exist" if not docs_missing else "required docs missing", {"missing": docs_missing}))
    secret_findings = scan_files_for_secrets([repo_root() / "infra", repo_root() / "scripts", repo_root() / "docs" / "deployment"])
    results.append(CheckResult("secret_scan", "fail" if secret_findings else "pass", "no obvious secrets in release assets" if not secret_findings else "possible secrets found", {"findings": secret_findings}))

    env_text = (repo_root() / ".env.example").read_text(encoding="utf-8", errors="ignore") if (repo_root() / ".env.example").exists() else ""
    required_envs = ["EVONITH_DEPLOYMENT_PROFILE", "EVONITH_BACKUP_DIR", "EVONITH_ALLOW_DIRECT_MODE_FALLBACK"]
    missing_envs = [name for name in required_envs if name not in env_text]
    results.append(CheckResult("env_example", "fail" if missing_envs else "pass", "Phase 13 env variables documented" if not missing_envs else "Phase 13 env variables missing", {"missing": missing_envs}))

    if "runtime/*" in (repo_root() / ".gitignore").read_text(encoding="utf-8", errors="ignore"):
        results.append(CheckResult("runtime_gitignore", "pass", "runtime is ignored"))
    else:
        results.append(CheckResult("runtime_gitignore", "fail", "runtime ignore rule missing"))

    commands = [
        ("check_repository_structure.py", []),
        ("check_import_boundaries.py", []),
        ("check_dependency_profiles.py", []),
        ("validate_deployment.py", ["--profile", "local", "--offline"]),
    ]
    if not args.skip_tests:
        commands.extend(
            [
                ("export_backend_openapi.py", []),
                ("check_backend_minimal_startup.py", []),
                ("check_frontend_api_imports.py", []),
            ]
        )
    for script, script_args in commands:
        completed = run_python_script(script, *script_args)
        status = "pass" if completed.returncode == 0 else "fail"
        results.append(CheckResult(script, status, "passed" if status == "pass" else "failed", {"stdout": completed.stdout[-500:]}))

    if args.backend_url:
        smoke = run_python_script("smoke_test_deployment.py", "--backend-url", args.backend_url, "--skip-auth")
        results.append(CheckResult("smoke_test", "pass" if smoke.returncode == 0 else "fail", "smoke test passed" if smoke.returncode == 0 else "smoke test failed"))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--allowed-branch", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    results = verify(args)
    print_results(results, json_output=args.json)
    return exit_code(results, strict_warnings=args.strict)


if __name__ == "__main__":
    sys.exit(main())
