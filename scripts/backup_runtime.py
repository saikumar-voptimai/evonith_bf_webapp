#!/usr/bin/env python
"""Create a safe archive backup of EVONITH_RUNTIME_DIR."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tarfile

from deployment_common import (
    CheckResult,
    backup_dir,
    bool_env,
    env_path,
    exit_code,
    is_within,
    print_results,
    runtime_dir,
    unsafe_runtime_path,
)


ARTIFACT_DIRS = {"jobs", "datasets", "compute", "copilot", "furnacemind"}
DEFAULT_DIRS = {"feedback", "uploads", "audit"}


def _selected_dirs(args: argparse.Namespace) -> tuple[set[str], set[str]]:
    include = set(DEFAULT_DIRS)
    exclude = {"temp", "cache", "backups"}
    if args.include_artifacts or bool_env("EVONITH_BACKUP_INCLUDE_ARTIFACTS", False):
        include.update(ARTIFACT_DIRS)
    if args.include_logs or bool_env("EVONITH_BACKUP_INCLUDE_LOGS", False):
        include.add("logs")
    else:
        exclude.add("logs")
    if not (args.include_uploads or bool_env("EVONITH_BACKUP_INCLUDE_UPLOADS", True)):
        include.discard("uploads")
    if not (args.include_audit or bool_env("EVONITH_BACKUP_INCLUDE_AUDIT", True)):
        include.discard("audit")
    if args.exclude_temp:
        exclude.add("temp")
    return include, exclude


def _iter_files(runtime: Path, include: set[str], exclude: set[str]) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    warnings: list[str] = []
    for top in sorted(include):
        root = runtime / top
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if any(part in exclude for part in path.relative_to(runtime).parts):
                continue
            if path.is_symlink():
                target = path.resolve()
                if not is_within(runtime, target):
                    warnings.append(f"skipped symlink outside runtime: {path.relative_to(runtime)}")
                continue
            if path.is_file():
                files.append(path)
                if path.name.endswith(".env") or path.suffix == ".key":
                    warnings.append(f"possible secret file included: {path.relative_to(runtime)}")
    return files, warnings


def _apply_retention(directory: Path) -> None:
    raw = os.getenv("EVONITH_BACKUP_MAX_ARCHIVES", "10")
    try:
        keep = max(0, int(raw))
    except ValueError:
        keep = 10
    if keep <= 0 or not directory.exists():
        return
    archives = sorted(directory.glob("evonith-runtime-*.tar.gz"), key=lambda path: path.stat().st_mtime, reverse=True)
    for old in archives[keep:]:
        old.unlink(missing_ok=True)


def backup(args: argparse.Namespace) -> list[CheckResult]:
    runtime = (args.runtime_dir or runtime_dir()).resolve()
    unsafe = unsafe_runtime_path(runtime)
    if unsafe:
        return [CheckResult("runtime_path", "fail", unsafe)]
    include, exclude = _selected_dirs(args)
    files, warnings = _iter_files(runtime, include, exclude)
    total_bytes = sum(path.stat().st_size for path in files)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_runtime_dir": str(runtime),
        "included_directories": sorted(include),
        "excluded_directories": sorted(exclude),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "release_version": os.getenv("EVONITH_RELEASE_VERSION", ""),
    }
    out = Path(args.output).expanduser().resolve() if args.output else backup_dir(runtime) / f"evonith-runtime-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
    results = [
        CheckResult("plan", "pass", "backup plan created", {"output": str(out), "file_count": len(files), "total_bytes": total_bytes}),
    ]
    results.extend(CheckResult("warning", "warn", warning) for warning in warnings)
    if args.dry_run:
        results.insert(0, CheckResult("dry_run", "pass", "no archive written"))
        return results

    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as archive:
        info = tarfile.TarInfo("manifest.json")
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        info.size = len(manifest_bytes)
        archive.addfile(info, fileobj=__import__("io").BytesIO(manifest_bytes))
        for path in files:
            archive.add(path, arcname=str(Path("runtime") / path.relative_to(runtime)), recursive=False)
    _apply_retention(out.parent)
    results.append(CheckResult("archive", "pass", "backup archive written", {"output": str(out)}))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--include-uploads", action="store_true")
    parser.add_argument("--include-artifacts", action="store_true")
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--include-audit", action="store_true")
    parser.add_argument("--exclude-temp", action="store_true", default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    results = backup(args)
    print_results(results, json_output=args.json)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())

