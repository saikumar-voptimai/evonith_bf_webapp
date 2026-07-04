#!/usr/bin/env python
"""Safely restore a Phase 13 runtime backup archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

from deployment_common import CheckResult, exit_code, is_within, print_results, runtime_dir, unsafe_runtime_path


ARTIFACT_TOPS = {"jobs", "datasets", "compute", "copilot", "furnacemind"}


def _safe_member(member: tarfile.TarInfo) -> bool:
    name = Path(member.name)
    return not member.name.startswith("/") and ".." not in name.parts


def _allowed(member: tarfile.TarInfo, args: argparse.Namespace) -> bool:
    if member.name == "manifest.json":
        return False
    parts = Path(member.name).parts
    if not parts or parts[0] != "runtime":
        return False
    if len(parts) < 2:
        return False
    top = parts[1]
    filters = [args.restore_uploads, args.restore_artifacts, args.restore_audit]
    if not any(filters):
        return True
    if args.restore_uploads and top == "uploads":
        return True
    if args.restore_audit and top == "audit":
        return True
    if args.restore_artifacts and top in ARTIFACT_TOPS:
        return True
    return False


def restore(args: argparse.Namespace) -> list[CheckResult]:
    archive_path = Path(args.backup).expanduser().resolve()
    target = (args.target_runtime_dir or runtime_dir()).resolve()
    unsafe = unsafe_runtime_path(target)
    if unsafe:
        return [CheckResult("target_path", "fail", unsafe)]
    if not archive_path.exists():
        return [CheckResult("backup", "fail", "backup archive does not exist", {"path": str(archive_path)})]
    results: list[CheckResult] = []
    restored = 0
    skipped = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
        if "manifest.json" not in names:
            return [CheckResult("manifest", "fail", "backup manifest missing")]
        manifest_file = archive.extractfile("manifest.json")
        manifest = json.loads(manifest_file.read().decode("utf-8")) if manifest_file else {}
        results.append(CheckResult("manifest", "pass", "backup manifest valid", {"file_count": manifest.get("file_count")}))
        for member in archive.getmembers():
            if not _safe_member(member):
                return [CheckResult("archive_safety", "fail", f"unsafe archive entry: {member.name}")]
            if not _allowed(member, args):
                skipped += 1
                continue
            if not member.isfile():
                skipped += 1
                continue
            relative = Path(member.name).relative_to("runtime")
            destination = target / relative
            if not is_within(target, destination):
                return [CheckResult("restore_path", "fail", f"restore entry escapes target: {member.name}")]
            if destination.exists() and not args.overwrite:
                skipped += 1
                continue
            if not args.dry_run and args.apply:
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    skipped += 1
                    continue
                destination.write_bytes(source.read())
            restored += 1
    mode = "dry-run" if args.dry_run or not args.apply else "apply"
    results.append(CheckResult("restore", "pass", f"restore {mode} completed", {"files_restored": restored, "files_skipped": skipped}))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true", help="Perform restore writes; otherwise dry-run is used.")
    parser.add_argument("--backup", required=True)
    parser.add_argument("--target-runtime-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--restore-uploads", action="store_true")
    parser.add_argument("--restore-artifacts", action="store_true")
    parser.add_argument("--restore-audit", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.apply:
        args.dry_run = False
    results = restore(args)
    print_results(results, json_output=args.json)
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())

