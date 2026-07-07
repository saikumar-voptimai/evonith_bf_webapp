"""Copy supported legacy runtime artifacts into EVONITH_RUNTIME_DIR.

Source-folder runtime artifacts were removed during repository cleanup. This
script remains non-destructive and can host future non-source migration mappings.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "furnace-data"
for path in (str(PACKAGE_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@dataclass(frozen=True)
class CopyMapping:
    source: Path
    target: Path
    kind: str


def _mappings() -> list[CopyMapping]:
    return []


def _iter_files(mapping: CopyMapping) -> list[tuple[Path, Path]]:
    if mapping.kind == "file":
        return [(mapping.source, mapping.target)]
    if not mapping.source.exists():
        return []
    files: list[tuple[Path, Path]] = []
    for source_file in mapping.source.rglob("*"):
        if source_file.is_file():
            files.append(
                (source_file, mapping.target / source_file.relative_to(mapping.source))
            )
    return files


def copy_runtime_files(*, dry_run: bool, overwrite: bool) -> int:
    """Copy configured legacy files and return the number copied."""
    copied = 0
    for mapping in _mappings():
        if not mapping.source.exists():
            print(f"MISS {mapping.source.relative_to(REPO_ROOT)}")
            continue
        for source, target in _iter_files(mapping):
            source_label = source.relative_to(REPO_ROOT)
            if target.exists() and not overwrite:
                print(f"SKIP exists {source_label} -> {target}")
                continue
            action = "WOULD COPY" if dry_run else "COPY"
            print(f"{action} {source_label} -> {target}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print planned copies without writing files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runtime files.")
    args = parser.parse_args()
    copied = copy_runtime_files(dry_run=args.dry_run, overwrite=args.overwrite)
    verb = "would copy" if args.dry_run else "copied"
    print(f"{verb} {copied} file(s)")


if __name__ == "__main__":
    main()