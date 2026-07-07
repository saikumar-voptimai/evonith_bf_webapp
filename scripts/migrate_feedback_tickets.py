"""Copy legacy Feedback/Tickets data into Phase 6 backend feedback tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "packages" / "furnace-data"

for source_path in (REPO_ROOT, PACKAGE_ROOT):
    path = str(source_path)
    if path not in sys.path:
        sys.path.insert(0, path)

from apps.backend_api.app.services.feedback_migration_service import FeedbackMigrationService
from furnace_data.runtime_paths import get_runtime_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Safely copy legacy direct-mode feedback tickets and attachments "
            "into backend-owned Phase 6 feedback tables."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files or database rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing backend ticket/comment/attachment rows and files.",
    )
    parser.add_argument(
        "--source-db",
        type=Path,
        default=None,
        help="Optional explicit legacy tickets.db path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Runtime directory: {get_runtime_dir()}")
    service = FeedbackMigrationService()
    result = service.migrate(
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        source_db=args.source_db,
    )

    for message in result.messages:
        print(message)

    print(
        "Summary: "
        f"tickets copied={result.copied_tickets}, "
        f"tickets skipped={result.skipped_tickets}, "
        f"comments copied={result.copied_comments}, "
        f"attachments copied={result.copied_attachments}, "
        f"attachments skipped={result.skipped_attachments}, "
        f"dry_run={result.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


