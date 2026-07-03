"""Copy legacy runtime artifacts into EVONITH_RUNTIME_DIR.

The script is intentionally non-destructive: it never deletes source files and
skips existing targets unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "furnace_data"))

from furnace_data.runtime_paths import (  # noqa: E402
    get_cache_dir,
    get_dataset_results_dir,
    get_dataset_static_dir,
    get_feedback_db_path,
    get_feedback_upload_dir,
    get_logs_dir,
    runtime_path,
)


@dataclass(frozen=True)
class CopyMapping:
    source: Path
    target: Path
    kind: str


def _file(source: str, target: Path) -> CopyMapping:
    return CopyMapping(REPO_ROOT / source, target, "file")


def _dir(source: str, target: Path) -> CopyMapping:
    return CopyMapping(REPO_ROOT / source, target, "dir")


def _mappings() -> list[CopyMapping]:
    cache_dir = get_cache_dir()
    return [
        _file("src/storage/feedback/tickets.db", get_feedback_db_path()),
        _dir("src/storage/feedback/images", get_feedback_upload_dir()),
        _dir("furnace-data-service/data/results", get_dataset_results_dir()),
        _dir("furnace-data-service/data/static", get_dataset_static_dir()),
        _file("src/storage/shift_summaries.json", cache_dir / "shift_summaries.json"),
        _file("src/storage/daily_summaries.json", cache_dir / "daily_summaries.json"),
        _file("src/storage/weekly_summaries.json", cache_dir / "weekly_summaries.json"),
        _file(
            "src/storage/biweekly_summaries.json",
            cache_dir / "biweekly_summaries.json",
        ),
        _file("src/assets/data/control_bounds.json", cache_dir / "control_bounds.json"),
        _file("src/config/bmo_operator_inputs.yml", cache_dir / "bmo_operator_inputs.yml"),
        _dir(
            "src/storage/furnacemind/mrag_images",
            runtime_path("uploads", "furnacemind", "mrag_images"),
        ),
        _file("src/agents/tool_errors.md", get_logs_dir() / "tool_errors.md"),
    ]


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
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace targets that already exist.",
    )
    args = parser.parse_args()

    count = copy_runtime_files(dry_run=args.dry_run, overwrite=args.overwrite)
    verb = "would copy" if args.dry_run else "copied"
    print(f"{count} file(s) {verb}.")


if __name__ == "__main__":
    main()
