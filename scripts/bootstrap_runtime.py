#!/usr/bin/env python
"""Prepare and validate the Phase 13 runtime directory layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from deployment_common import (
    CheckResult,
    STANDARD_RUNTIME_SUBDIRS,
    disk_free_mb,
    env_path,
    exit_code,
    min_free_mb,
    print_results,
    repo_root,
    unsafe_runtime_path,
    validate_writable,
    warn_free_mb,
)


def build_results(
    runtime: Path,
    *,
    create: bool,
    check: bool,
    allow_repo_root: bool,
    probe_writable: bool = True,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    unsafe = unsafe_runtime_path(runtime, allow_repo_root=allow_repo_root)
    if unsafe:
        return [CheckResult("runtime_path", "fail", unsafe, {"path": str(runtime)})]

    missing = [name for name in STANDARD_RUNTIME_SUBDIRS if not (runtime / name).exists()]
    if create:
        for name in STANDARD_RUNTIME_SUBDIRS:
            (runtime / name).mkdir(parents=True, exist_ok=True)
        missing = []
        results.append(CheckResult("create_directories", "pass", "runtime directories created", {"count": len(STANDARD_RUNTIME_SUBDIRS)}))
    elif check and missing:
        results.append(CheckResult("runtime_directories", "fail", "required runtime directories are missing", {"missing": missing}))
    else:
        status = "warn" if missing else "pass"
        message = "directories would be created" if missing else "directories already exist"
        results.append(CheckResult("runtime_directories", status, message, {"missing": missing}))

    if not probe_writable:
        results.append(CheckResult("writable", "pass", "not checked in dry run"))
    elif runtime.exists():
        writable = validate_writable(runtime)
        results.append(CheckResult("writable", "pass" if writable else "fail", "runtime is writable" if writable else "runtime is not writable"))
    elif check:
        results.append(CheckResult("writable", "fail", "runtime directory does not exist"))
    else:
        results.append(CheckResult("writable", "warn", "runtime directory does not exist yet"))

    try:
        free = disk_free_mb(runtime)
        min_free = min_free_mb()
        warn_free = warn_free_mb()
        if free < min_free:
            results.append(CheckResult("disk_free", "fail", f"free disk below minimum: {free} MB", {"free_mb": free, "min_mb": min_free}))
        elif free < warn_free:
            results.append(CheckResult("disk_free", "warn", f"free disk below warning threshold: {free} MB", {"free_mb": free, "warn_mb": warn_free}))
        else:
            results.append(CheckResult("disk_free", "pass", f"free disk ok: {free} MB", {"free_mb": free}))
    except OSError as exc:
        results.append(CheckResult("disk_free", "fail", f"could not check disk space: {exc}"))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-repo-root", action="store_true")
    args = parser.parse_args(argv)

    runtime = (args.runtime_dir or env_path("EVONITH_RUNTIME_DIR", "runtime")).resolve()
    local_dev = str(runtime) == str(repo_root() / "runtime")
    allow_repo_root = args.allow_repo_root and local_dev
    create = args.create and not args.dry_run
    check = args.check

    results = build_results(runtime, create=create, check=check, allow_repo_root=allow_repo_root, probe_writable=not args.dry_run)
    if args.dry_run:
        results.insert(0, CheckResult("dry_run", "pass", "no filesystem changes made", {"runtime_dir": str(runtime)}))
    print_results(results, json_output=args.json)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
