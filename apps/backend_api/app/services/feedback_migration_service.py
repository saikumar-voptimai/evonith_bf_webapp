"""Copy legacy direct-mode feedback tickets into backend feedback tables."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import mimetypes
from pathlib import Path
import shutil
import sqlite3
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.repositories.feedback_repository import FeedbackRepository, _sqlite_url_to_path
from apps.backend_api.app.services.feedback_attachment_service import FeedbackAttachmentService
from furnace_data.app_catalog import canonical_page_id, page_label
from furnace_data.runtime_paths import get_feedback_db_path, get_feedback_upload_dir, get_repo_root


@dataclass
class FeedbackMigrationResult:
    """Counters and messages emitted by a feedback migration run."""

    dry_run: bool
    copied_tickets: int = 0
    skipped_tickets: int = 0
    copied_comments: int = 0
    copied_attachments: int = 0
    skipped_attachments: int = 0
    messages: list[str] = field(default_factory=list)


def _legacy_db_candidates() -> list[Path]:
    root = get_repo_root()
    candidates = [
        get_feedback_db_path(),
        root / "storage" / "feedback" / "tickets.db",
    ]
    seen: set[Path] = set()
    output: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            output.append(candidate)
    return output


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _normalize_enum(value: Any, default: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return default
    return raw.lower()


def _legacy_status(value: Any) -> str:
    return _normalize_enum(value, "open")


def _legacy_priority(value: Any) -> str:
    return _normalize_enum(value, "medium")


def _legacy_timestamp(value: Any) -> str:
    return str(value or "").strip()


class FeedbackMigrationService:
    """Non-destructive migration from legacy ticket tables to backend tables."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        repository: FeedbackRepository | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = self.settings.feedback_database_url.strip() or None
        self.repository = repository or FeedbackRepository(
            database_url=database_url,
            ticket_prefix=self.settings.feedback_ticket_id_prefix,
        )
        self.attachment_service = FeedbackAttachmentService(
            repository=self.repository,
            settings=self.settings,
        )

    def migrate(
        self,
        *,
        dry_run: bool = False,
        overwrite: bool = False,
        source_db: Path | None = None,
    ) -> FeedbackMigrationResult:
        """Copy legacy direct-mode records into backend tables."""
        result = FeedbackMigrationResult(dry_run=dry_run)
        if not dry_run:
            self.repository.ensure_schema()
        sources = [source_db] if source_db else _legacy_db_candidates()
        for candidate in sources:
            if candidate is None:
                continue
            path = Path(candidate)
            if not path.exists():
                result.messages.append(f"skip missing source db: {path}")
                continue
            self._migrate_source(path=path, result=result, dry_run=dry_run, overwrite=overwrite)
        return result

    def _migrate_source(
        self,
        *,
        path: Path,
        result: FeedbackMigrationResult,
        dry_run: bool,
        overwrite: bool,
    ) -> None:
        with sqlite3.connect(str(path)) as source:
            source.row_factory = sqlite3.Row
            tables = {
                row["name"]
                for row in source.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "tickets" not in tables:
                result.messages.append(f"skip source without legacy tickets table: {path}")
                return

            tickets = source.execute("SELECT * FROM tickets ORDER BY id ASC").fetchall()
            result.messages.append(f"source {path}: found {len(tickets)} legacy ticket(s)")
            for ticket_row in tickets:
                legacy_ticket = _row_dict(ticket_row)
                ticket_id = self._copy_ticket(
                    legacy_ticket=legacy_ticket,
                    result=result,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    source_path=path,
                )
                if ticket_id is None:
                    continue
                if "ticket_events" in tables:
                    self._copy_comments(
                        source=source,
                        legacy_ticket=legacy_ticket,
                        target_ticket_id=ticket_id,
                        result=result,
                        dry_run=dry_run,
                        overwrite=overwrite,
                    )
                if "ticket_images" in tables:
                    self._copy_attachments(
                        source=source,
                        legacy_ticket=legacy_ticket,
                        target_ticket_id=ticket_id,
                        result=result,
                        dry_run=dry_run,
                        overwrite=overwrite,
                    )

    def _target_connection(self) -> sqlite3.Connection:
        database_url = self.repository.database_url
        db_path = _sqlite_url_to_path(database_url)
        if db_path is None:
            raise RuntimeError("Feedback migration requires a file-backed SQLite database.")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _target_database_path(self) -> Path:
        db_path = _sqlite_url_to_path(self.repository.database_url)
        if db_path is None:
            raise RuntimeError("Feedback migration requires a file-backed SQLite database.")
        return db_path

    def _target_ticket_exists(self, *values: str) -> bool:
        db_path = self._target_database_path()
        if not db_path.exists():
            return False
        with sqlite3.connect(str(db_path)) as target:
            table = target.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name = 'feedback_tickets'
                """
            ).fetchone()
            if table is None:
                return False
            return (
                target.execute(
                    """
                    SELECT 1 FROM feedback_tickets
                    WHERE id IN (?, ?) OR ticket_number IN (?, ?)
                    """,
                    (values[0], values[1], values[0], values[1]),
                ).fetchone()
                is not None
            )

    def _copy_ticket(
        self,
        *,
        legacy_ticket: dict[str, Any],
        result: FeedbackMigrationResult,
        dry_run: bool,
        overwrite: bool,
        source_path: Path,
    ) -> str | None:
        legacy_id = int(legacy_ticket["id"])
        target_id = f"fb_legacy_{legacy_id}"
        ticket_number = str(legacy_ticket.get("ticket_code") or f"TKT-{legacy_id:06d}")
        if dry_run:
            existing = self._target_ticket_exists(ticket_number, target_id)
        else:
            existing = self.repository.get_ticket(ticket_number) or self.repository.get_ticket(target_id)
        if existing and not overwrite:
            result.skipped_tickets += 1
            result.messages.append(f"skip existing ticket {ticket_number}")
            return None

        page_name = str(legacy_ticket.get("page_name") or "Feedback").strip() or "Feedback"
        page_id = canonical_page_id(page_name) or "feedback"
        page_name = page_label(page_id) or page_name
        reported_by = str(legacy_ticket.get("reported_by") or "").strip()
        ideal_closure = str(legacy_ticket.get("ideal_closure_text") or "").strip()
        metadata = {
            "legacy_ticket_id": legacy_id,
            "legacy_ticket_code": ticket_number,
            "legacy_reported_by": reported_by,
            "ideal_closure_text": ideal_closure,
            "legacy_updated_by": legacy_ticket.get("updated_by"),
            "migrated_from": source_path.as_posix(),
        }
        title = f"{page_name} feedback"
        created_at = _legacy_timestamp(legacy_ticket.get("created_at"))
        updated_at = _legacy_timestamp(legacy_ticket.get("updated_at")) or created_at
        status = _legacy_status(legacy_ticket.get("status"))
        resolved_at = updated_at if status in {"resolved", "rejected"} else None
        closed_at = updated_at if status == "closed" else None

        result.messages.append(f"{'would copy' if dry_run else 'copy'} ticket {ticket_number}")
        if dry_run:
            result.copied_tickets += 1
            return target_id

        self.repository.ensure_schema()
        with self._target_connection() as target:
            target.execute(
                """
                INSERT OR REPLACE INTO feedback_tickets (
                    id, ticket_number, version, title, description, ideal_closure,
                    category, priority, status, page_id, page, tags_json,
                    metadata_json, created_by, created_by_username, updated_by,
                    updated_by_username, assigned_to, resolution_notes, created_at,
                    updated_at, last_activity_at, resolved_at, closed_at, deleted_at
                )
                VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    target_id,
                    ticket_number,
                    title,
                    str(legacy_ticket.get("description") or ""),
                    ideal_closure or None,
                    page_name,
                    _legacy_priority(legacy_ticket.get("criticality")),
                    status,
                    page_id,
                    page_name,
                    json.dumps(["legacy-direct-mode"], sort_keys=True),
                    json.dumps(metadata, sort_keys=True),
                    str(legacy_ticket.get("created_by") or reported_by or "") or None,
                    reported_by or str(legacy_ticket.get("created_by") or "") or None,
                    str(legacy_ticket.get("updated_by") or "") or None,
                    str(legacy_ticket.get("updated_by") or "") or None,
                    None,
                    None,
                    created_at,
                    updated_at,
                    updated_at,
                    resolved_at,
                    closed_at,
                ),
            )
        result.copied_tickets += 1
        return target_id

    def _copy_comments(
        self,
        *,
        source: sqlite3.Connection,
        legacy_ticket: dict[str, Any],
        target_ticket_id: str,
        result: FeedbackMigrationResult,
        dry_run: bool,
        overwrite: bool,
    ) -> None:
        events = source.execute(
            "SELECT * FROM ticket_events WHERE ticket_id = ? ORDER BY id ASC",
            (legacy_ticket["id"],),
        ).fetchall()
        if not events:
            return
        if dry_run:
            for event_row in events:
                event = _row_dict(event_row)
                event_id = f"fbe_legacy_{event['id']}"
                result.messages.append(f"would copy event {event_id}")
                result.copied_comments += 1
            return

        with self._target_connection() as target:
            for event_row in events:
                event = _row_dict(event_row)
                event_id = f"fbe_legacy_{event['id']}"
                exists = target.execute(
                    "SELECT 1 FROM feedback_events WHERE id = ?",
                    (event_id,),
                ).fetchone()
                if exists and not overwrite:
                    continue
                sequence = int(
                    target.execute(
                        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM feedback_events WHERE ticket_id = ?",
                        (target_ticket_id,),
                    ).fetchone()[0]
                )
                event_type = str(event.get("event_type") or "legacy_event").strip().lower().replace("-", "_")
                if event_type == "created":
                    event_type = "ticket_created"
                note = str(event.get("comment") or "").strip() or None
                target.execute(
                    """
                    INSERT OR REPLACE INTO feedback_events (
                        id, ticket_id, event_type, actor_user_id, actor_username,
                        old_status, new_status, note, payload_json, sequence, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        target_ticket_id,
                        event_type,
                        str(event.get("actor") or "") or None,
                        str(event.get("actor") or "") or None,
                        _legacy_status(event.get("old_status")) if event.get("old_status") else None,
                        _legacy_status(event.get("new_status")) if event.get("new_status") else None,
                        note,
                        json.dumps({"legacy_event_id": event.get("id")}, sort_keys=True),
                        sequence,
                        _legacy_timestamp(event.get("created_at")),
                    ),
                )
                result.copied_comments += 1
    def _copy_attachments(
        self,
        *,
        source: sqlite3.Connection,
        legacy_ticket: dict[str, Any],
        target_ticket_id: str,
        result: FeedbackMigrationResult,
        dry_run: bool,
        overwrite: bool,
    ) -> None:
        images = source.execute(
            "SELECT * FROM ticket_images WHERE ticket_id = ? ORDER BY id ASC",
            (legacy_ticket["id"],),
        ).fetchall()
        if not images:
            return

        upload_dir = get_feedback_upload_dir()
        if not dry_run:
            upload_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            for image_row in images:
                image = _row_dict(image_row)
                attachment_id = f"fba_legacy_{image['id']}"
                original_filename = Path(str(image.get("original_filename") or "upload")).name
                safe_name = self.attachment_service.sanitize_filename(original_filename)
                stored_filename = f"{target_ticket_id}_{attachment_id}_{safe_name}"
                source_file = self._resolve_legacy_attachment_path(str(image.get("image_path") or ""))
                target_file = upload_dir / stored_filename

                if not source_file or not source_file.exists():
                    result.skipped_attachments += 1
                    result.messages.append(f"skip missing attachment {image.get('image_path')}")
                    continue
                if target_file.exists() and not overwrite:
                    result.skipped_attachments += 1
                    result.messages.append(f"skip existing attachment file {target_file}")
                    continue

                result.messages.append(
                    f"would copy attachment {source_file} -> {target_file}"
                )
                result.copied_attachments += 1
            return

        with self._target_connection() as target:
            for image_row in images:
                image = _row_dict(image_row)
                attachment_id = f"fba_legacy_{image['id']}"
                exists = target.execute(
                    "SELECT 1 FROM feedback_attachments WHERE id = ?",
                    (attachment_id,),
                ).fetchone()
                if exists and not overwrite:
                    result.skipped_attachments += 1
                    continue

                original_filename = Path(str(image.get("original_filename") or "upload")).name
                safe_name = self.attachment_service.sanitize_filename(original_filename)
                stored_filename = f"{target_ticket_id}_{attachment_id}_{safe_name}"
                source_file = self._resolve_legacy_attachment_path(str(image.get("image_path") or ""))
                target_file = upload_dir / stored_filename

                if not source_file or not source_file.exists():
                    result.skipped_attachments += 1
                    result.messages.append(f"skip missing attachment {image.get('image_path')}")
                    continue
                if target_file.exists() and not overwrite:
                    result.skipped_attachments += 1
                    result.messages.append(f"skip existing attachment file {target_file}")
                    continue

                result.messages.append(f"copy attachment {source_file} -> {target_file}")

                shutil.copy2(source_file, target_file)
                checksum_sha256 = hashlib.sha256(target_file.read_bytes()).hexdigest()
                content_type = mimetypes.guess_type(original_filename)[0] or "application/octet-stream"
                target.execute(
                    """
                    INSERT OR REPLACE INTO feedback_attachments (
                        id, ticket_id, filename, original_filename, stored_filename,
                        content_type, size_bytes, checksum_sha256, storage_status,
                        created_by, created_by_username, created_at, deleted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        attachment_id,
                        target_ticket_id,
                        safe_name,
                        original_filename,
                        stored_filename,
                        content_type,
                        target_file.stat().st_size,
                        checksum_sha256,
                        "stored",
                        str(image.get("uploaded_by") or "") or None,
                        str(image.get("uploaded_by") or "") or None,
                        _legacy_timestamp(image.get("created_at")),
                    ),
                )
                result.copied_attachments += 1

    @staticmethod
    def _resolve_legacy_attachment_path(value: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        root = get_repo_root()
        for candidate in (root / path, Path.cwd() / path):
            if candidate.exists():
                return candidate
        return root / path









