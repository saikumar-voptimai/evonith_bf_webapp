"""Repository classes for PostgreSQL-backed relational persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import MetaData, Table, and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased, sessionmaker

from .models import (
    BURDEN_VALUE_COLUMNS,
    HOPPER_COLUMNS,
    BurdenHistory,
    Conversation,
    ConversationMessage,
    FeedbackItem,
    Hopper,
    HopperRawMaterialHistory,
    Material,
    MemoryDocument,
    MemoryFact,
    MemorySummary,
    ScheduledJob,
    ScheduledJobCommand,
    ScheduledJobCommandAction,
    ScheduledJobCommandStatus,
    ScheduledJobOutput,
    ScheduledJobRevision,
    ScheduledJobRun,
    ScheduledJobRunLog,
    ScheduledJobRunStatus,
    ScheduledJobStatus,
    User,
    UserRole,
    UserRoleAssignment,
    utc_now,
)


def _new_id(prefix: str) -> str:
    """Return a compact application id with a stable prefix."""
    return f"{prefix}_{uuid4().hex}"


_UNSET: Any = object()
_MAX_SCHEDULED_JOB_ERROR_LENGTH = 4096
_SCHEDULED_JOB_CLEANUP_PENDING_MESSAGE = "Timer cleanup is pending."


class ScheduledJobRunLeaseBusyError(RuntimeError):
    """Raised when resume or recovery meets a still-owned execution lease."""

    def __init__(self, lease_expires_at: datetime) -> None:
        """Store the database lease deadline for bounded retry decisions."""

        self.lease_expires_at = _as_aware_utc(lease_expires_at)
        super().__init__(
            "A scheduled-job execution is still in flight until "
            f"{self.lease_expires_at.isoformat()}."
        )


@dataclass(frozen=True, slots=True)
class ScheduledJobResumeSettlement:
    """Atomic resume result and any expired run cancelled before activation."""

    job: ScheduledJob
    cancelled_run: ScheduledJobRun | None


def _bounded_scheduled_job_error(error: str) -> str:
    """Return a non-empty scheduler error bounded for safe persistence."""

    normalized = error.strip()
    if not normalized:
        normalized = "External scheduler operation failed without an error message."
    return normalized[:_MAX_SCHEDULED_JOB_ERROR_LENGTH]


def _scheduled_timer_unit_name(job_id: str) -> str:
    """Return the only timer unit name allowed for a scheduled-job row."""

    try:
        canonical_job_id = str(UUID(job_id))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("job_id must be a valid UUID for timer activation.") from exc
    return f"furnacemind-job-{canonical_job_id}.timer"


def _json_string_values(value: Any) -> list[str]:
    """Flatten JSON-compatible metadata into searchable string values."""
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_json_string_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_json_string_values(item))
        return values
    return [str(value)]


def _metadata_document_ids(metadata: dict[str, Any]) -> set[str]:
    """Return document ids carried by memory metadata, if present."""
    ids: set[str] = set()
    for key in (
        "document_id",
        "document_ids",
        "knowledge_document_id",
        "knowledge_document_ids",
        "mrag_document_id",
        "source_document_id",
    ):
        value = metadata.get(key)
        if isinstance(value, (list, tuple, set)):
            ids.update(str(item).strip() for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            ids.add(str(value).strip())
    return ids


def _memory_fact_matches_document(
    *,
    fact_text: str,
    metadata: dict[str, Any],
    sql_document_id: str,
    mrag_document_id: str,
    filename: str,
) -> bool:
    """Return True when a memory fact carries direct document provenance."""
    fact_text = str(fact_text or "")
    metadata_text = " ".join(_json_string_values(metadata))
    combined = f"{fact_text} {metadata_text}".lower()
    metadata_ids = _metadata_document_ids(metadata)

    direct_ids = {sql_document_id, mrag_document_id} - {""}
    if direct_ids & metadata_ids:
        return True

    filename_stem = filename.rsplit(".", 1)[0] if filename else ""
    for identifier in (sql_document_id, mrag_document_id, filename, filename_stem):
        normalized = str(identifier or "").strip().lower()
        if len(normalized) >= 3 and normalized in combined:
            return True
    return False


def _as_aware_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime for comparisons and inserts."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _database_utc_now(session: Session) -> datetime:
    """Return the authoritative database wall clock normalized to UTC.

    PostgreSQL ``CURRENT_TIMESTAMP`` is fixed at transaction start, which is
    unsafe for leases after a row-lock wait.  ``clock_timestamp()`` returns the
    actual wall clock.  Other dialects retain the portable SQL timestamp used
    by the SQLite-backed repository tests.
    """

    bind = session.get_bind()
    clock = (
        func.clock_timestamp()
        if bind.dialect.name == "postgresql"
        else func.current_timestamp()
    )
    value = session.execute(select(clock)).scalar_one()
    if not isinstance(value, datetime):
        raise ValueError("Database clock did not return a timestamp.")
    return _as_aware_utc(value)


def _positive_seconds(value: int, *, field_name: str) -> int:
    """Validate a positive integer duration used by an execution lease."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _is_one_time_job(job: ScheduledJob) -> bool:
    """Return whether a stored job definition has a one-time trigger."""

    definition = job.definition_json
    if not isinstance(definition, dict):
        return False
    schedule = definition.get("schedule")
    if not isinstance(schedule, dict):
        return False
    trigger = schedule.get("trigger")
    return isinstance(trigger, dict) and trigger.get("type") == "once"


def _monotonic_transition_time(previous: datetime, candidate: datetime) -> datetime:
    """Return a UTC row-version timestamp strictly newer than ``previous``."""

    normalized_previous = _as_aware_utc(previous)
    normalized_candidate = _as_aware_utc(candidate)
    if normalized_candidate <= normalized_previous:
        return normalized_previous + timedelta(microseconds=1)
    return normalized_candidate


class UserRepository:
    """User/auth repository operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def seed_admin_user(self, *, password_hash: str) -> None:
        """Seed default admin user if missing."""
        with self._session_factory() as session:
            exists_stmt = select(User).where(User.username == "admin").limit(1)
            if session.execute(exists_stmt).scalar_one_or_none():
                return

            user = User(
                username="admin",
                password_hash=password_hash,
                role=UserRole.ADMIN.value,
            )
            session.add(user)
            session.flush()
            session.add(UserRoleAssignment(user_id=user.id, role=UserRole.ADMIN.value))
            session.commit()

    def add_user(self, username: str, password_hash: str, role: str) -> None:
        """Create a user row."""
        role = UserRole(role).value
        with self._session_factory() as session:
            user = User(username=username, password_hash=password_hash, role=role)
            session.add(user)
            try:
                session.flush()
                session.add(UserRoleAssignment(user_id=user.id, role=role))
                session.commit()
            except IntegrityError:
                session.rollback()
                raise

    def validate_user(
        self, username: str, password_hash: str
    ) -> tuple[str, str] | None:
        """Return ``(username, role)`` when credentials are valid."""
        with self._session_factory() as session:
            stmt = select(User.username, User.role).where(
                User.username == username,
                User.password_hash == password_hash,
                User.is_active.is_(True),
            )
            row = session.execute(stmt).first()
            return (row[0], row[1]) if row else None

    def get_user_id(self, username: str | None) -> UUID | None:
        """Return the identity UUID for *username*, if present."""
        if not username:
            return None
        with self._session_factory() as session:
            stmt = select(User.id).where(
                User.username == username, User.is_active.is_(True)
            )
            row = session.execute(stmt).first()
            return row[0] if row else None


class PlantMasterRepository:
    """Read-only plant master lookups used by app repositories and UI."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def list_active_hoppers(self) -> list[dict[str, Any]]:
        """Return active hopper display metadata ordered for UI use."""
        with self._session_factory() as session:
            stmt = (
                select(Hopper)
                .where(Hopper.is_active.is_(True))
                .order_by(Hopper.sort_order.asc(), Hopper.hopper_code.asc())
            )
            return [
                {
                    "hopper_code": row.hopper_code,
                    "display_name": row.display_name or row.hopper_code,
                    "sort_order": row.sort_order,
                }
                for row in session.execute(stmt).scalars().all()
            ]

    def list_active_materials(self) -> list[dict[str, Any]]:
        """Return active raw-material metadata ordered by category and code."""
        with self._session_factory() as session:
            stmt = (
                select(Material)
                .where(Material.is_active.is_(True))
                .order_by(Material.category_code.asc(), Material.material_code.asc())
            )
            return [
                {
                    "material_code": row.material_code,
                    "material_name": row.material_name,
                    "category_code": row.category_code,
                    "unit_code": row.unit_code,
                }
                for row in session.execute(stmt).scalars().all()
            ]

    def material_code_by_name(self) -> dict[str, str]:
        """Return active material codes keyed by material name."""
        return {
            row["material_name"]: row["material_code"]
            for row in self.list_active_materials()
        }

    def material_name_by_code(self) -> dict[str, str]:
        """Return active material names keyed by material code."""
        return {
            row["material_code"]: row["material_name"]
            for row in self.list_active_materials()
        }


class HopperHistoryRepository:
    """Repository for wide hopper-material snapshot history."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    @staticmethod
    def _row_to_codes(row: HopperRawMaterialHistory | None) -> dict[str, str | None]:
        """Convert a hopper history row into a hopper-code mapping."""
        if row is None:
            return {column: None for column in HOPPER_COLUMNS}
        return {column: getattr(row, column) for column in HOPPER_COLUMNS}

    def _latest_row(
        self, session: Session, ts: datetime | None = None
    ) -> HopperRawMaterialHistory | None:
        """Return the latest hopper snapshot at or before the timestamp."""
        stmt = select(HopperRawMaterialHistory)
        if ts is not None:
            stmt = stmt.where(HopperRawMaterialHistory.date_time <= _as_aware_utc(ts))
        stmt = stmt.order_by(
            HopperRawMaterialHistory.date_time.desc(),
            HopperRawMaterialHistory.id.desc(),
        ).limit(1)
        return session.execute(stmt).scalar_one_or_none()

    def update_hopper_snapshot(
        self,
        *,
        hopper_material_codes: dict[str, str | None],
        from_time: datetime,
        user_id: UUID | None,
        ip_address: str | None,
        source_type: str = "webapp",
    ) -> None:
        """Insert one full hopper-material snapshot."""
        from_time = _as_aware_utc(from_time)
        with self._session_factory() as session:
            snapshot = self._row_to_codes(self._latest_row(session, from_time))
            for hopper, material_code in hopper_material_codes.items():
                if hopper not in HOPPER_COLUMNS:
                    raise ValueError(f"Invalid hopper: {hopper}")
                snapshot[hopper] = material_code
            session.add(
                HopperRawMaterialHistory(
                    date_time=from_time,
                    ip_address=ip_address,
                    user_modified=user_id,
                    source_type=source_type,
                    **snapshot,
                )
            )
            session.commit()

    def get_current_hopper_material_codes(self) -> dict[str, str | None]:
        """Return current hopper to material-code map."""
        with self._session_factory() as session:
            return self._row_to_codes(self._latest_row(session))

    def get_hopper_material_code_at(self, hopper: str, ts: datetime) -> str | None:
        """Return assigned material code for hopper at timestamp."""
        if hopper not in HOPPER_COLUMNS:
            raise ValueError(f"Invalid hopper: {hopper}")
        with self._session_factory() as session:
            row = self._latest_row(session, ts)
            return getattr(row, hopper) if row else None

    def get_hopper_material_history(self) -> list[dict[str, Any]]:
        """Return complete hopper snapshot history rows."""
        with self._session_factory() as session:
            stmt = select(HopperRawMaterialHistory).order_by(
                HopperRawMaterialHistory.date_time.desc(),
                HopperRawMaterialHistory.id.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            out = []
            for row in rows:
                payload = {
                    "id": row.id,
                    "date_time": row.date_time,
                    "source_type": row.source_type,
                    "ip_address": row.ip_address,
                    "user_modified": (
                        str(row.user_modified) if row.user_modified else None
                    ),
                }
                payload.update(
                    {column: getattr(row, column) for column in HOPPER_COLUMNS}
                )
                out.append(payload)
            return out

    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """Delete hopper snapshot rows by IDs."""
        if not record_ids:
            return
        with self._session_factory() as session:
            session.execute(
                delete(HopperRawMaterialHistory).where(
                    HopperRawMaterialHistory.id.in_(record_ids)
                )
            )
            session.commit()


class BurdenHistoryRepository:
    """Repository for wide burden-distribution snapshot history."""

    TEXT_FIELDS = frozenset(
        {
            "coke_charge_pattern",
            "non_coke_charge_pattern",
            "burden_changing_purpose",
        }
    )

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    @staticmethod
    def burden_fields() -> list[str]:
        """Return the wide burden snapshot fields tracked by the app."""
        return list(BURDEN_VALUE_COLUMNS)

    def _latest_row(
        self, session: Session, ts: datetime | None = None
    ) -> BurdenHistory | None:
        """Return the latest burden snapshot at or before the timestamp."""
        stmt = select(BurdenHistory)
        if ts is not None:
            stmt = stmt.where(BurdenHistory.date_time <= _as_aware_utc(ts))
        stmt = stmt.order_by(
            BurdenHistory.date_time.desc(), BurdenHistory.id.desc()
        ).limit(1)
        return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def _row_to_values(row: BurdenHistory | None) -> dict[str, Any]:
        """Convert a burden history row into a burden-value mapping."""
        if row is None:
            return {column: None for column in BURDEN_VALUE_COLUMNS}
        return {column: getattr(row, column) for column in BURDEN_VALUE_COLUMNS}

    def update_burden_field(
        self,
        *,
        field_name: str,
        value: Any,
        valid_from: datetime,
        user_id: UUID | None = None,
        ip: str = "",
    ) -> None:
        """Insert one snapshot with a single changed burden value."""
        self.update_burden_row(
            row_values={field_name: value},
            timestamp=valid_from,
            user_id=user_id,
            ip=ip,
        )

    def update_burden_row(
        self,
        *,
        row_values: dict[str, Any],
        timestamp: datetime,
        user_id: UUID | None = None,
        ip: str = "",
        source_type: str = "webapp",
    ) -> None:
        """Insert one full burden snapshot copied from latest prior row plus edits."""
        timestamp = _as_aware_utc(timestamp)
        unknown = sorted(set(row_values) - set(BURDEN_VALUE_COLUMNS))
        if unknown:
            raise ValueError(f"Invalid burden field(s): {unknown}")

        with self._session_factory() as session:
            snapshot = self._row_to_values(self._latest_row(session, timestamp))
            for field, value in row_values.items():
                if value == "":
                    snapshot[field] = None
                elif field in self.TEXT_FIELDS or value is None:
                    snapshot[field] = value
                else:
                    snapshot[field] = float(value)
            session.add(
                BurdenHistory(
                    date_time=timestamp,
                    source_type=source_type,
                    ip_address=ip,
                    user_modified=user_id,
                    **snapshot,
                )
            )
            session.commit()

    def get_burden_history(self) -> list[dict[str, Any]]:
        """Return full burden snapshot history."""
        with self._session_factory() as session:
            stmt = select(BurdenHistory).order_by(
                BurdenHistory.date_time.desc(),
                BurdenHistory.id.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            out = []
            for row in rows:
                payload = {
                    "id": row.id,
                    "date_time": row.date_time,
                    "source_type": row.source_type,
                    "ip_address": row.ip_address,
                    "user_modified": (
                        str(row.user_modified) if row.user_modified else None
                    ),
                }
                payload.update(self._row_to_values(row))
                out.append(payload)
            return out

    def get_all_current_burden_values(self, ts: datetime) -> dict[str, Any]:
        """Return active burden values at timestamp."""
        with self._session_factory() as session:
            return self._row_to_values(self._latest_row(session, ts))

    def delete_burden_history(self, record_ids: list[int]) -> None:
        """Delete burden snapshot rows by IDs."""
        if not record_ids:
            return
        with self._session_factory() as session:
            session.execute(
                delete(BurdenHistory).where(BurdenHistory.id.in_(record_ids))
            )
            session.commit()

    def fetch_distribution_frame(
        self, *, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Return latest burden snapshots overlapping the date window."""
        window_start = datetime.combine(start_date, time.min).replace(
            tzinfo=timezone.utc
        )
        window_end = datetime.combine(end_date, time.max).replace(tzinfo=timezone.utc)
        with self._session_factory() as session:
            prior = self._latest_row(session, window_start)
            stmt = (
                select(BurdenHistory)
                .where(BurdenHistory.date_time >= window_start)
                .where(BurdenHistory.date_time <= window_end)
                .order_by(BurdenHistory.date_time.asc(), BurdenHistory.id.asc())
            )
            rows = session.execute(stmt).scalars().all()
            if prior and (not rows or prior.id != rows[0].id):
                rows.insert(0, prior)

        if not rows:
            return pd.DataFrame()
        records = []
        for row in rows:
            payload = {"time": pd.to_datetime(row.date_time)}
            payload.update(self._row_to_values(row))
            records.append(payload)
        return pd.DataFrame(records).set_index("time").sort_index()


class ConversationRepository:
    """Repository for FurnaceMind conversation rows."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def create_conversation(
        self,
        *,
        user_id: str,
        model_mode: str | None = None,
        metadata: dict | None = None,
    ) -> Conversation:
        """
        Create and return a FurnaceMind conversation.

        Args:
             - user_id: str - User that owns the conversation.
             - model_mode: str | None - Selected reasoning or model mode.
             - metadata: dict | None - Optional JSON metadata for the conversation.

        Returns:
             - return: Conversation - Created conversation ORM row.
        """
        now = utc_now()
        conversation = Conversation(
            conversation_id=_new_id("conv"),
            user_id=user_id,
            model_mode=model_mode,
            metadata_json=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._session_factory() as session:
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            session.expunge(conversation)
            return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """
        Return one conversation by id when it exists.

        Args:
             - conversation_id: str - Conversation id to fetch.

        Returns:
             - return: Conversation | None - Conversation row when found, otherwise None.
        """
        with self._session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is not None:
                session.expunge(conversation)
            return conversation

    def list_conversations(
        self, *, user_id: str, limit: int = 30
    ) -> list[Conversation]:
        """
        List recent conversations for one user.

        Args:
             - user_id: str - User whose conversations should be listed.
             - limit: int - Maximum number of conversations to return.

        Returns:
             - return: list[Conversation] - Recent conversation rows.
        """
        with self._session_factory() as session:
            stmt = (
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.updated_at.desc())
                .limit(limit)
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

    def touch_conversation(
        self,
        *,
        conversation_id: str,
        model_mode: str | None = None,
    ) -> None:
        """
        Update conversation mode or timestamp.

        Args:
             - conversation_id: str - Conversation id to update.
             - model_mode: str | None - Optional replacement model mode.

        Returns:
             - return: None - This function does not return a value.
        """
        values: dict[str, Any] = {"updated_at": utc_now()}
        if model_mode is not None:
            values["model_mode"] = model_mode
        with self._session_factory() as session:
            session.execute(
                update(Conversation)
                .where(Conversation.conversation_id == conversation_id)
                .values(**values)
            )
            session.commit()


class ConversationMessageRepository:
    """Repository for FurnaceMind chat messages."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def add_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        token_count: int | None = None,
        model: str | None = None,
        tool_calls: dict | list | None = None,
        metadata: dict | None = None,
    ) -> ConversationMessage:
        """
        Append one message to a conversation.

        Args:
             - conversation_id: str - Conversation that owns the message.
             - user_id: str - User that owns the conversation.
             - role: str - Message role, such as user or assistant.
             - content: str - Message text content.
             - token_count: int | None - Optional estimated token count.
             - model: str | None - Optional model name for assistant messages.
             - tool_calls: dict | list | None - Optional tool call payload.
             - metadata: dict | None - Optional JSON metadata for the message.

        Returns:
             - return: ConversationMessage - Created message ORM row.
        """
        with self._session_factory() as session:
            next_sequence = (
                session.execute(
                    select(
                        func.coalesce(func.max(ConversationMessage.sequence_num), 0)
                    ).where(ConversationMessage.conversation_id == conversation_id)
                ).scalar_one()
                + 1
            )
            message = ConversationMessage(
                message_id=_new_id("msg"),
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=content,
                sequence_num=next_sequence,
                token_count=token_count,
                model=model,
                tool_calls=tool_calls,
                metadata_json=metadata or {},
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            session.expunge(message)
            return message

    def get_message(self, message_id: str) -> ConversationMessage | None:
        """
        Return one message by id when it exists.

        Args:
             - message_id: str - Message id to fetch.

        Returns:
             - return: ConversationMessage | None - Message row when found, otherwise None.
        """
        with self._session_factory() as session:
            message = session.get(ConversationMessage, message_id)
            if message is not None:
                session.expunge(message)
            return message

    def list_recent_messages(
        self,
        *,
        conversation_id: str,
        limit: int = 100,
    ) -> list[ConversationMessage]:
        """
        List recent messages in chronological order.

        Args:
             - conversation_id: str - Conversation id to fetch messages for.
             - limit: int - Maximum number of recent messages to return.

        Returns:
             - return: list[ConversationMessage] - Recent message rows in chronological order.
        """
        with self._session_factory() as session:
            stmt = (
                select(ConversationMessage)
                .where(ConversationMessage.conversation_id == conversation_id)
                .order_by(ConversationMessage.sequence_num.desc())
                .limit(limit)
            )
            rows = list(session.execute(stmt).scalars().all())
            rows.reverse()
            for row in rows:
                session.expunge(row)
            return rows

    def list_messages_after(
        self,
        *,
        conversation_id: str,
        sequence_num: int,
    ) -> list[ConversationMessage]:
        """
        List messages after a sequence number.

        Args:
             - conversation_id: str - Conversation id to fetch messages for.
             - sequence_num: int - Sequence number after which messages should be returned.

        Returns:
             - return: list[ConversationMessage] - Message rows after the sequence number.
        """
        with self._session_factory() as session:
            stmt = (
                select(ConversationMessage)
                .where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.sequence_num > sequence_num,
                )
                .order_by(ConversationMessage.sequence_num.asc())
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows


class MemoryDocumentRepository:
    """Repository for uploaded FurnaceMind knowledge documents."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def create_document(
        self,
        *,
        user_id: str,
        filename: str,
        file_type: str | None = None,
        qdrant_collection: str | None = None,
        qdrant_point_ids: list | None = None,
        metadata: dict | None = None,
    ) -> MemoryDocument:
        """
        Create and return uploaded document metadata.

        Args:
             - user_id: str - User that owns the document.
             - filename: str - Original uploaded file name.
             - file_type: str | None - Optional file type or extension.
             - qdrant_collection: str | None - Optional Qdrant collection name.
             - qdrant_point_ids: list | None - Optional Qdrant point ids for chunks.
             - metadata: dict | None - Optional JSON metadata for the document.

        Returns:
             - return: MemoryDocument - Created document ORM row.
        """
        now = utc_now()
        metadata_payload = {**(metadata or {})}
        metadata_payload["qdrant_point_ids"] = qdrant_point_ids or []
        document = MemoryDocument(
            document_id=_new_id("doc"),
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            qdrant_collection=qdrant_collection,
            metadata_json=metadata_payload,
            created_at=now,
            updated_at=now,
        )
        with self._session_factory() as session:
            session.add(document)
            session.commit()
            session.refresh(document)
            session.expunge(document)
            return document

    def store_document_file(
        self,
        *,
        document_id: str,
        user_id: str,
        filename: str,
        file_type: str | None,
        content_type: str | None,
        file_bytes: bytes,
    ) -> None:
        """Store the original uploaded document on its metadata row.

        Args:
             - document_id: str - SQL document row id from ``memory_documents``.
             - user_id: str - Owner of the document; retained for interface clarity.
             - filename: str - Original uploaded file name.
             - file_type: str | None - File extension/type used by ingestion.
             - content_type: str | None - Browser-provided MIME type when known.
             - file_bytes: bytes - Exact uploaded file bytes.

        Returns:
             - return: None - The existing document row is updated in PostgreSQL.
        """
        if not document_id or not user_id or file_bytes is None:
            return
        payload = {
            "document_id": str(document_id),
            "filename": str(filename or "upload"),
            "file_type": file_type,
            "content_type": content_type,
            "file_size": len(file_bytes),
            "sha256": hashlib.sha256(file_bytes).hexdigest(),
            "file_bytes": bytes(file_bytes),
        }
        with self._session_factory() as session:
            session.execute(
                update(MemoryDocument)
                .where(MemoryDocument.document_id == payload["document_id"])
                .values(
                    filename=payload["filename"],
                    file_type=payload["file_type"],
                    content_type=payload["content_type"],
                    file_size=payload["file_size"],
                    sha256=payload["sha256"],
                    file_bytes=payload["file_bytes"],
                    updated_at=utc_now(),
                )
            )
            session.commit()

    def get_document_file(self, *, document_id: str) -> SimpleNamespace | None:
        """Return original upload bytes stored on ``memory_documents``.

        Args:
             - document_id: str - SQL document row id.

        Returns:
             - return: SimpleNamespace | None - File metadata and bytes, or
               ``None`` when the row/columns are unavailable.
        """
        if not document_id:
            return None
        try:
            with self._session_factory() as session:
                row = (
                    session.execute(
                        select(
                            MemoryDocument.document_id,
                            MemoryDocument.user_id,
                            MemoryDocument.filename,
                            MemoryDocument.file_type,
                            MemoryDocument.content_type,
                            MemoryDocument.file_size,
                            MemoryDocument.sha256,
                            MemoryDocument.file_bytes,
                        ).where(MemoryDocument.document_id == str(document_id))
                    )
                    .mappings()
                    .first()
                )
        except Exception:
            return None
        if row is None:
            return None
        values = dict(row)
        raw_bytes = values.get("file_bytes")
        if isinstance(raw_bytes, memoryview):
            values["file_bytes"] = raw_bytes.tobytes()
        elif raw_bytes is not None:
            values["file_bytes"] = bytes(raw_bytes)
        return SimpleNamespace(**values)

    def get_document_file_by_mrag_id(
        self,
        *,
        user_id: str | None = None,
        mrag_document_id: str,
    ) -> SimpleNamespace | None:
        """Return stored upload bytes for the active SQL row backing an MRAG id.

        Qdrant payloads carry the stable content-hash MRAG ``document_id``. The
        SQL row has its own ``document_id`` primary key, so this method bridges
        the two through ``memory_documents.metadata['document_id']`` before
        loading bytes from the same row. ``user_id`` is optional because uploaded
        knowledge is a shared FurnaceMind library; when provided, it narrows the
        lookup to documents uploaded by that user.
        """
        if not mrag_document_id:
            return None
        for document in self.list_documents(user_id=user_id, active_only=True):
            metadata = getattr(document, "metadata_json", None)
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("document_id") or "") != str(mrag_document_id):
                continue
            return self.get_document_file(document_id=str(document.document_id))
        return None

    def delete_document_file(self, *, document_id: str) -> None:
        """Clear original upload bytes from a deactivated document row."""
        if not document_id:
            return
        try:
            with self._session_factory() as session:
                session.execute(
                    update(MemoryDocument)
                    .where(MemoryDocument.document_id == str(document_id))
                    .values(
                        content_type=None,
                        file_size=None,
                        sha256=None,
                        file_bytes=None,
                        updated_at=utc_now(),
                    )
                )
                session.commit()
        except Exception:
            return

    def list_documents(
        self,
        *,
        user_id: str | None = None,
        active_only: bool = True,
    ) -> list[MemoryDocument]:
        """List uploaded knowledge documents from SQL.

        Uploaded knowledge is shared across FurnaceMind users, so callers can
        omit ``user_id`` to read the global active library. Passing ``user_id``
        keeps the older uploader-scoped behavior for audit or user-specific
        maintenance screens.

        Args:
             - user_id: Optional uploader id used to narrow the result set.
             - active_only: Whether to return only active documents.

        Returns:
             - return: Uploaded document rows detached from the session.
        """
        with self._session_factory() as session:
            stmt = select(MemoryDocument)
            if user_id:
                stmt = stmt.where(MemoryDocument.user_id == user_id)
            if active_only:
                stmt = stmt.where(MemoryDocument.is_active.is_(True))
            stmt = stmt.order_by(MemoryDocument.created_at.desc())
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

    def deactivate_document(self, document_id: str) -> None:
        """
        Mark an uploaded document inactive.

        Args:
             - document_id: str - Document id to deactivate.

        Returns:
             - return: None - This function does not return a value.
        """
        with self._session_factory() as session:
            session.execute(
                update(MemoryDocument)
                .where(MemoryDocument.document_id == document_id)
                .values(is_active=False, updated_at=utc_now())
            )
            session.commit()
        self.delete_document_file(document_id=document_id)


class _ReflectedTableRepository:
    """Small helper for optional live tables that are not mapped as ORM models."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        table_name: str,
        schema: str = "furnace_mind",
    ) -> None:
        """Store reflection settings for one optional relational table."""

        self._session_factory = session_factory
        self._table_name = table_name
        self._schema = schema
        self._table: Table | None = None

    def _reflect_table(self, session: Session) -> Table:
        """Reflect the target table, retrying without schema for test databases."""
        if self._table is not None:
            return self._table

        bind = session.get_bind()
        metadata = MetaData()
        try:
            self._table = Table(
                self._table_name,
                metadata,
                schema=self._schema,
                autoload_with=bind,
            )
        except Exception:
            metadata = MetaData()
            self._table = Table(
                self._table_name,
                metadata,
                autoload_with=bind,
            )
        return self._table

    @staticmethod
    def _project_row(table: Table, values: dict[str, Any]) -> dict[str, Any]:
        """Return only non-null values accepted by the reflected table."""
        columns = set(table.c.keys())
        return {
            key: value
            for key, value in values.items()
            if key in columns and value is not None
        }


class MemoryChunkRepository(_ReflectedTableRepository):
    """Best-effort repository for the optional ``memory_chunks`` table."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Bind the repository to the optional ``memory_chunks`` table."""

        super().__init__(session_factory, table_name="memory_chunks")

    def create_chunks(
        self,
        *,
        document: Any,
        parts: list[Any],
        qdrant_collection: str | None,
    ) -> int:
        """
        Persist searchable MRAG chunk metadata when the live table is available.

        The live database owns the exact table shape, so this method reflects the
        table and inserts only matching columns. Existing rows for the SQL
        document are skipped to avoid duplicate chunk inserts on Streamlit reruns.
        """
        sql_document_id = str(getattr(document, "document_id", "") or "").strip()
        if not sql_document_id or not parts:
            return 0

        with self._session_factory() as session:
            table = self._reflect_table(session)
            columns = set(table.c.keys())
            if "document_id" in columns:
                existing_count = session.execute(
                    select(func.count())
                    .select_from(table)
                    .where(table.c.document_id == sql_document_id)
                ).scalar_one()
                if existing_count:
                    return 0

            now = utc_now()
            rows: list[dict[str, Any]] = []
            for part in parts:
                metadata = {
                    **(getattr(part, "metadata", None) or {}),
                    "mrag_document_id": getattr(part, "document_id", None),
                    "source": getattr(part, "source", None),
                    "file_type": getattr(part, "file_type", None),
                    "logical_chunk_id": getattr(part, "chunk_id", None),
                    "image_path": getattr(part, "image_path", None),
                }
                row = self._project_row(
                    table,
                    {
                        "chunk_id": getattr(part, "point_id", None),
                        "document_id": sql_document_id,
                        "user_id": getattr(part, "user_id", None),
                        "qdrant_point_id": getattr(part, "point_id", None),
                        "qdrant_collection": qdrant_collection,
                        "collection_name": qdrant_collection,
                        "source": getattr(part, "source", None),
                        "filename": getattr(part, "source", None),
                        "file_type": getattr(part, "file_type", None),
                        "modality": getattr(part, "modality", None),
                        "chunk_index": getattr(part, "chunk_index", None),
                        "content": getattr(part, "content", None),
                        "chunk_text": getattr(part, "content", None),
                        "text": getattr(part, "content", None),
                        "image_path": getattr(part, "image_path", None),
                        "page_number": getattr(part, "page_number", None),
                        "slide_number": getattr(part, "slide_number", None),
                        "sheet_name": getattr(part, "sheet_name", None),
                        "metadata": metadata,
                        "metadata_json": metadata,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
                if row:
                    rows.append(row)

            if not rows:
                return 0

            session.execute(table.insert(), rows)
            session.commit()
            return len(rows)


class RetrievalTraceRepository(_ReflectedTableRepository):
    """Best-effort repository for the optional ``retrieval_traces`` table."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Bind the repository to the optional ``retrieval_traces`` table."""

        super().__init__(session_factory, table_name="retrieval_traces")

    def create_trace(
        self,
        *,
        user_id: str | None,
        conversation_id: str | None,
        query: str,
        qdrant_collection: str | None,
        results: list[dict[str, Any]],
        active_document_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Persist one MRAG retrieval trace when the reflected table supports it."""
        trace_id = _new_id("trace")
        now = utc_now()
        result_summaries = []
        for result in results:
            payload = result.get("payload") or {}
            result_summaries.append(
                {
                    "score": result.get("score"),
                    "rerank_score": result.get("rerank_score"),
                    "document_id": payload.get("document_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "source": payload.get("source"),
                    "modality": payload.get("modality"),
                    "page_number": payload.get("page_number"),
                    "slide_number": payload.get("slide_number"),
                    "sheet_name": payload.get("sheet_name"),
                    "content_preview": str(payload.get("content") or "")[:1000],
                }
            )

        metadata_payload = {
            "source": "furnacemind_knowledge",
            "active_document_ids": active_document_ids or [],
            "results": result_summaries,
            **(metadata or {}),
        }

        with self._session_factory() as session:
            table = self._reflect_table(session)
            row = self._project_row(
                table,
                {
                    "trace_id": trace_id,
                    "retrieval_trace_id": trace_id,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "query": query,
                    "query_text": query,
                    "tool_name": "search_knowledge_docs",
                    "qdrant_collection": qdrant_collection,
                    "collection_name": qdrant_collection,
                    "top_k": len(results),
                    "result_count": len(results),
                    "results": result_summaries,
                    "retrieved_results": result_summaries,
                    "metadata": metadata_payload,
                    "metadata_json": metadata_payload,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            if not row:
                return False
            session.execute(table.insert().values(**row))
            session.commit()
            return True


class MemorySummaryRepository:
    """Repository for compressed conversation summaries."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def create_summary(
        self,
        *,
        conversation_id: str,
        user_id: str,
        summary_text: str,
        source_message_id_start: str | None = None,
        source_message_id_end: str | None = None,
        token_count: int | None = None,
        metadata: dict | None = None,
    ) -> MemorySummary:
        """
        Create and return one compressed summary.

        Args:
             - conversation_id: str - Conversation that owns the summary.
             - user_id: str - User that owns the summary.
             - summary_text: str - Compressed summary text.
             - source_message_id_start: str | None - First compressed message id.
             - source_message_id_end: str | None - Last compressed message id.
             - token_count: int | None - Optional estimated token count.
             - metadata: dict | None - Optional JSON metadata for the summary.

        Returns:
             - return: MemorySummary - Created summary ORM row.
        """
        summary = MemorySummary(
            summary_id=_new_id("sum"),
            conversation_id=conversation_id,
            user_id=user_id,
            summary_text=summary_text,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
            token_count=token_count,
            metadata_json=metadata or {},
        )
        with self._session_factory() as session:
            session.add(summary)
            session.commit()
            session.refresh(summary)
            session.expunge(summary)
            return summary

    def list_summaries(
        self,
        *,
        conversation_id: str,
        limit: int = 5,
    ) -> list[MemorySummary]:
        """
        List recent summaries for one conversation.

        Args:
             - conversation_id: str - Conversation id to fetch summaries for.
             - limit: int - Maximum number of summaries to return.

        Returns:
             - return: list[MemorySummary] - Recent summary rows.
        """
        with self._session_factory() as session:
            stmt = (
                select(MemorySummary)
                .where(MemorySummary.conversation_id == conversation_id)
                .order_by(MemorySummary.created_at.desc())
                .limit(limit)
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows


class MemoryFactRepository:
    """Repository for the existing furnace_mind.memory_facts table."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def create_fact(
        self,
        *,
        user_id: str,
        fact_text: str,
        source_conversation_id: str | None = None,
        qdrant_collection: str | None = None,
        qdrant_point_id: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryFact:
        """
        Create and return one long-term memory fact.

        Args:
             - user_id: str - User that owns the fact.
             - fact_text: str - Durable memory fact text.
             - source_conversation_id: str | None - Source conversation id.
             - qdrant_collection: str | None - Optional Qdrant collection name.
             - qdrant_point_id: str | None - Optional Qdrant point id.
             - metadata: dict | None - Optional JSON metadata for audit/recovery.

        Returns:
             - return: MemoryFact - Created fact ORM row.
        """
        now = utc_now()
        fact = MemoryFact(
            fact_id=_new_id("fact"),
            user_id=user_id,
            fact_text=fact_text,
            source_conversation_id=source_conversation_id,
            qdrant_collection=qdrant_collection,
            qdrant_point_id=qdrant_point_id,
            metadata_json=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._session_factory() as session:
            session.add(fact)
            session.commit()
            session.refresh(fact)
            session.expunge(fact)
            return fact

    def fact_exists(self, *, user_id: str, fact_text: str) -> bool:
        """
        Check whether a fact with the same text already exists for the user.

        Args:
             - user_id: str - User that owns the fact.
             - fact_text: str - Durable memory fact text to de-duplicate.

        Returns:
             - return: bool - True when a matching fact already exists.
        """
        normalized_text = " ".join(str(fact_text or "").split())
        if not normalized_text:
            return False

        with self._session_factory() as session:
            normalized_fact_text = func.btrim(
                func.regexp_replace(
                    func.lower(MemoryFact.fact_text),
                    r"\s+",
                    " ",
                    "g",
                )
            )
            stmt = (
                select(MemoryFact.fact_id)
                .where(
                    MemoryFact.user_id == user_id,
                    normalized_fact_text == normalized_text.lower(),
                )
                .limit(1)
            )
            return session.execute(stmt).first() is not None

    def mark_fact_indexed(
        self,
        *,
        fact_id: str,
        qdrant_collection: str,
        qdrant_point_id: str,
    ) -> None:
        """
        Store the Qdrant index location for a memory fact.

        Args:
             - fact_id: str - PostgreSQL fact id to update.
             - qdrant_collection: str - Qdrant collection that holds the vector.
             - qdrant_point_id: str - Qdrant point id used for this fact.

        Returns:
             - return: None - The database row is updated in place.
        """
        with self._session_factory() as session:
            stmt = (
                update(MemoryFact)
                .where(MemoryFact.fact_id == fact_id)
                .values(
                    qdrant_collection=qdrant_collection,
                    qdrant_point_id=qdrant_point_id,
                    updated_at=utc_now(),
                )
            )
            session.execute(stmt)
            session.commit()

    def list_unindexed_facts(self, *, limit: int = 100) -> list[MemoryFact]:
        """
        List SQL-saved facts that have not yet reached Qdrant.

        Args:
             - limit: int - Maximum number of rows to return.

        Returns:
             - return: list[MemoryFact] - Facts available for recovery indexing.
        """
        with self._session_factory() as session:
            stmt = (
                select(MemoryFact)
                .where(
                    MemoryFact.qdrant_point_id.is_(None),
                )
                .order_by(MemoryFact.created_at.asc())
                .limit(limit)
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

    def list_document_related_facts(
        self,
        *,
        user_id: str,
        sql_document_id: str,
        mrag_document_id: str = "",
        filename: str = "",
    ) -> list[MemoryFact]:
        """
        Return long-term memory facts with direct document provenance.

        The match uses explicit ids in ``memory_facts.metadata`` first, then
        direct mentions of the SQL document id, MRAG document id, filename, or
        filename stem. Cleanup intentionally avoids text-similarity matching so unrelated memories are not deleted.
        """
        if not user_id or not sql_document_id:
            return []

        with self._session_factory() as session:
            stmt = select(MemoryFact).where(MemoryFact.user_id == user_id)
            rows = list(session.execute(stmt).scalars().all())
            matches: list[MemoryFact] = []
            for row in rows:
                metadata = (
                    row.metadata_json if isinstance(row.metadata_json, dict) else {}
                )
                if not _memory_fact_matches_document(
                    fact_text=row.fact_text,
                    metadata=metadata,
                    sql_document_id=sql_document_id,
                    mrag_document_id=mrag_document_id,
                    filename=filename,
                ):
                    continue
                session.expunge(row)
                matches.append(row)
            return matches

    def delete_facts(self, fact_ids: list[str]) -> int:
        """Delete memory facts by id after their Qdrant points are removed."""
        ids = [str(fact_id) for fact_id in fact_ids if str(fact_id).strip()]
        if not ids:
            return 0
        with self._session_factory() as session:
            result = session.execute(
                delete(MemoryFact).where(MemoryFact.fact_id.in_(ids))
            )
            session.commit()
            return int(result.rowcount or 0)


class SkillRepository(_ReflectedTableRepository):
    """Repository for built-in and uploaded FurnaceMind skills.

    The production ``furnace_mind.skills`` table can be older than the ORM model
    during feature rollout. This repository reflects the live table and reads or
    writes only the columns that actually exist, matching the deployed skills
    table during rollout.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        super().__init__(session_factory, table_name="skills")

    @staticmethod
    def _metadata_column_name(table: Table) -> str | None:
        """Return the JSON metadata column name used by the live skills table."""
        columns = set(table.c.keys())
        if "metadata" in columns:
            return "metadata"
        if "metadata_json" in columns:
            return "metadata_json"
        return None

    @staticmethod
    def _metadata_value(value: Any) -> dict[str, Any]:
        """Normalize a JSON/JSONB metadata value from PostgreSQL."""
        if isinstance(value, dict):
            return value
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @classmethod
    def _skill_from_mapping(cls, values: dict[str, Any]) -> Any:
        """Return an attribute-style skill object with defaults for old schemas."""
        metadata = cls._metadata_value(
            values.get("metadata_json")
            if "metadata_json" in values
            else values.get("metadata")
        )
        is_active = values.get("is_active")
        return SimpleNamespace(
            skill_id=str(values.get("skill_id") or ""),
            name=str(values.get("name") or "Skill"),
            symbol=str(values.get("symbol") or metadata.get("symbol") or ""),
            description=values.get("description"),
            instruction=str(values.get("instruction") or ""),
            source_type=str(values.get("source_type") or "uploaded"),
            qdrant_collection=values.get("qdrant_collection"),
            is_active=True if is_active is None else bool(is_active),
            created_by=values.get("created_by"),
            metadata_json=metadata,
            created_at=values.get("created_at"),
            updated_at=values.get("updated_at"),
        )

    @classmethod
    def _skill_from_row(cls, row: Any) -> Any:
        """Convert a SQLAlchemy row mapping into the runtime skill shape."""
        return cls._skill_from_mapping(dict(row))

    def _get_skill_in_session(
        self, session: Session, table: Table, skill_id: str
    ) -> Any | None:
        """Return one skill from a reflected table within an existing session."""
        if "skill_id" not in table.c:
            return None
        row = (
            session.execute(select(table).where(table.c.skill_id == skill_id))
            .mappings()
            .first()
        )
        return self._skill_from_row(row) if row is not None else None

    @staticmethod
    def _set_if_present(
        payload: dict[str, Any], columns: set[str], key: str, value: Any
    ) -> None:
        """Add a write value only when the reflected table accepts the column."""
        if key in columns and value is not _UNSET:
            payload[key] = value

    @staticmethod
    def _ordered_skill_statement(table: Table) -> Any:
        """Build a deterministic SELECT for whichever ordering columns exist."""
        stmt = select(table)
        order_columns = [
            table.c[column]
            for column in ("created_at", "name", "skill_id")
            if column in table.c
        ]
        if order_columns:
            stmt = stmt.order_by(*order_columns)
        return stmt

    def create_skill(
        self,
        *,
        name: str,
        instruction: str,
        symbol: str | None = None,
        description: str | None = None,
        source_type: str = "uploaded",
        qdrant_collection: str | None = None,
        is_active: bool = True,
        created_by: str | None = None,
        metadata: dict | None = None,
    ) -> Any:
        """
        Create and return a skill definition.

        Only columns present in the live ``skills`` table are written, using the
        exported schema: identity, text fields, source metadata, active state, and
        timestamps.

        Args:
             - name: str - Skill display name.
             - instruction: str - Prompt instruction used when the skill is selected.
             - symbol: str | None - Optional unique short symbol for the skill.
             - description: str | None - Optional skill description.
             - source_type: str - Skill source, such as built_in or uploaded.
             - qdrant_collection: str | None - Optional Qdrant collection name.
             - is_active: bool - Whether the skill is active.
             - created_by: str | None - User that created the skill.
             - metadata: dict | None - Optional JSON metadata for the skill.

        Returns:
             - return: Any - Created skill row exposed as attribute-style data.
        """
        skill_id = _new_id("skill")
        with self._session_factory() as session:
            table = self._reflect_table(session)
            columns = set(table.c.keys())
            if "skill_id" not in columns:
                raise RuntimeError("furnace_mind.skills must include a skill_id column")

            now = utc_now()
            payload: dict[str, Any] = {}
            self._set_if_present(payload, columns, "skill_id", skill_id)
            self._set_if_present(payload, columns, "name", name)
            self._set_if_present(payload, columns, "symbol", symbol)
            self._set_if_present(payload, columns, "description", description)
            self._set_if_present(payload, columns, "instruction", instruction)
            self._set_if_present(payload, columns, "source_type", source_type)
            self._set_if_present(
                payload, columns, "qdrant_collection", qdrant_collection
            )
            self._set_if_present(payload, columns, "is_active", is_active)
            self._set_if_present(payload, columns, "created_by", created_by)
            metadata_column = self._metadata_column_name(table)
            if metadata_column:
                payload[metadata_column] = metadata or {}
            self._set_if_present(payload, columns, "created_at", now)
            self._set_if_present(payload, columns, "updated_at", now)

            session.execute(table.insert().values(**payload))
            session.commit()
            return self._get_skill_in_session(
                session, table, skill_id
            ) or self._skill_from_mapping(payload)

    def list_skills(self, *, active_only: bool = False) -> list[Any]:
        """
        List skills, optionally filtering to active skills.

        Args:
             - active_only: bool - Whether to return only active skills.

        Returns:
             - return: list[Any] - Skill rows exposed as attribute-style data.
        """
        with self._session_factory() as session:
            table = self._reflect_table(session)
            stmt = self._ordered_skill_statement(table)
            if active_only and "is_active" in table.c:
                stmt = stmt.where(table.c.is_active.is_(True))
            rows = session.execute(stmt).mappings().all()
            return [self._skill_from_row(row) for row in rows]

    def update_skill(
        self,
        *,
        skill_id: str,
        name: str = _UNSET,
        symbol: str | None = _UNSET,
        description: str | None = _UNSET,
        instruction: str = _UNSET,
        qdrant_collection: str | None = _UNSET,
        is_active: bool = _UNSET,
        metadata: dict | None = _UNSET,
    ) -> Any | None:
        """
        Update a skill definition and return the row.

        Missing optional columns are skipped so UI edits remain compatible with
        older database schemas.

        Args:
             - skill_id: str - Skill id to update.
             - name: str - New skill display name.
             - symbol: str | None - New unique short symbol.
             - description: str | None - New skill description.
             - instruction: str - New prompt instruction.
             - qdrant_collection: str | None - Qdrant collection holding skill vectors.
             - is_active: bool - New active state.
             - metadata: dict | None - New optional JSON metadata.

        Returns:
             - return: Any | None - Updated skill row when found, otherwise None.
        """
        with self._session_factory() as session:
            table = self._reflect_table(session)
            columns = set(table.c.keys())
            if "skill_id" not in columns:
                return None

            payload: dict[str, Any] = {}
            self._set_if_present(payload, columns, "name", name)
            self._set_if_present(payload, columns, "symbol", symbol)
            self._set_if_present(payload, columns, "description", description)
            self._set_if_present(payload, columns, "instruction", instruction)
            self._set_if_present(
                payload, columns, "qdrant_collection", qdrant_collection
            )
            self._set_if_present(payload, columns, "is_active", is_active)
            metadata_column = self._metadata_column_name(table)
            if metadata_column and metadata is not _UNSET:
                payload[metadata_column] = metadata or {}
            if "updated_at" in columns:
                payload["updated_at"] = utc_now()

            if not payload:
                return self._get_skill_in_session(session, table, skill_id)

            result = session.execute(
                update(table).where(table.c.skill_id == skill_id).values(**payload)
            )
            if int(result.rowcount or 0) <= 0:
                session.rollback()
                return None
            session.commit()
            return self._get_skill_in_session(session, table, skill_id)


class FeedbackItemRepository:
    """Repository for response feedback and extracted lessons."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def add_feedback(
        self,
        *,
        user_id: str,
        message_id: str,
        conversation_id: str,
        source: str,
        polarity: str,
        feedback_text: str | None = None,
        raw_user_message: str | None = None,
        assistant_response: str | None = None,
        metadata: dict | None = None,
    ) -> FeedbackItem:
        """
        Create and return one feedback item.

        Args:
             - user_id: str - User that submitted the feedback.
             - message_id: str - Assistant message receiving feedback.
             - conversation_id: str - Conversation where feedback was submitted.
             - source: str - Feedback source, such as form or chat.
             - polarity: str - Feedback polarity, such as positive or negative.
             - feedback_text: str | None - Optional feedback comment.
             - raw_user_message: str | None - Original user message before the response.
             - assistant_response: str | None - Assistant response being reviewed.
             - metadata: dict | None - Optional JSON metadata for the feedback.

        Returns:
             - return: FeedbackItem - Created feedback ORM row.
        """
        feedback = FeedbackItem(
            feedback_id=_new_id("fb"),
            user_id=user_id,
            message_id=message_id,
            conversation_id=conversation_id,
            source=source,
            polarity=polarity,
            feedback_text=feedback_text,
            raw_user_message=raw_user_message,
            assistant_response=assistant_response,
            metadata_json=metadata or {},
        )
        with self._session_factory() as session:
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
            session.expunge(feedback)
            return feedback

    def get_feedback(self, *, message_id: str, user_id: str) -> FeedbackItem | None:
        """
        Return feedback for a message and user when it exists.

        Args:
             - message_id: str - Message id to fetch feedback for.
             - user_id: str - User that owns the feedback.

        Returns:
             - return: FeedbackItem | None - Feedback row when found, otherwise None.
        """
        with self._session_factory() as session:
            stmt = (
                select(FeedbackItem)
                .where(
                    FeedbackItem.message_id == message_id,
                    FeedbackItem.user_id == user_id,
                )
                .order_by(FeedbackItem.created_at.desc())
                .limit(1)
            )
            feedback = session.execute(stmt).scalar_one_or_none()
            if feedback is not None:
                session.expunge(feedback)
            return feedback

    def list_pending_lessons(self, *, limit: int = 20) -> list[FeedbackItem]:
        """
        List feedback rows that still need lesson extraction.

        Args:
             - limit: int - Maximum number of feedback rows to return.

        Returns:
             - return: list[FeedbackItem] - Feedback rows pending lesson extraction.
        """
        with self._session_factory() as session:
            stmt = (
                select(FeedbackItem)
                .where(FeedbackItem.lesson_extracted.is_(False))
                .order_by(FeedbackItem.created_at.asc())
                .limit(limit)
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

    def mark_lesson_extracted(
        self,
        *,
        feedback_id: str,
        lesson: str,
        qdrant_collection: str | None = None,
        qdrant_point_id: str | None = None,
    ) -> None:
        """
        Persist an extracted feedback lesson on a feedback row.

        Args:
             - feedback_id: str - Feedback id to update.
             - lesson: str - Extracted lesson text.
             - qdrant_collection: str | None - Optional Qdrant collection name.
             - qdrant_point_id: str | None - Optional Qdrant point id.

        Returns:
             - return: None - This function does not return a value.
        """
        with self._session_factory() as session:
            session.execute(
                update(FeedbackItem)
                .where(FeedbackItem.feedback_id == feedback_id)
                .values(
                    lesson_extracted=True,
                    extracted_lesson=lesson,
                    qdrant_collection=qdrant_collection,
                    qdrant_point_id=qdrant_point_id,
                )
            )
            session.commit()


class ScheduledJobRepository:
    """Persistence operations for validated scheduled-job definitions."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""

        self._session_factory = session_factory

    def create_job(
        self,
        *,
        job_name: str,
        schema_version: str,
        definition: dict[str, Any],
        created_by_user_id: UUID,
        created_by_username: str,
        target_device_id: str,
    ) -> ScheduledJob:
        """Persist a definition in the pending-provisioning state."""

        job_type = definition.get("job_type")
        if not isinstance(job_type, str):
            raise ValueError("Validated definition is missing its job type.")

        job = ScheduledJob(
            job_name=job_name,
            job_type=job_type,
            is_active=False,
            schema_version=schema_version,
            definition_json=definition,
            status=ScheduledJobStatus.PENDING_PROVISIONING.value,
            created_by_user_id=created_by_user_id,
            created_by_username=created_by_username,
            target_device_id=target_device_id,
        )
        with self._session_factory() as session:
            session.add(job)
            session.flush()
            session.add(
                ScheduledJobRevision(
                    job_id=job.job_id,
                    revision_number=1,
                    schema_version=schema_version,
                    definition_json=definition,
                    changed_by_user_id=created_by_user_id,
                    changed_by_username=created_by_username,
                    change_kind="created",
                )
            )
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def get_job(self, job_id: str) -> ScheduledJob | None:
        """Return a scheduled job by its UUID-formatted text identifier."""

        with self._session_factory() as session:
            job = session.get(ScheduledJob, job_id)
            if job is not None:
                session.expunge(job)
            return job

    def list_jobs_for_owner(
        self,
        *,
        owner_user_id: UUID,
        limit: int = 50,
    ) -> list[tuple[ScheduledJob, ScheduledJobRun | None]]:
        """Return recent owner jobs with each latest run in one bounded query."""

        if not 1 <= limit <= 100:
            raise ValueError("Scheduled-job list limit must be between 1 and 100.")
        latest_run = aliased(ScheduledJobRun)
        latest_run_id = (
            select(latest_run.run_id)
            .where(latest_run.job_id == ScheduledJob.job_id)
            .order_by(latest_run.created_at.desc(), latest_run.run_id.desc())
            .limit(1)
            .correlate(ScheduledJob)
            .scalar_subquery()
        )
        statement = (
            select(ScheduledJob, ScheduledJobRun)
            .outerjoin(ScheduledJobRun, ScheduledJobRun.run_id == latest_run_id)
            .where(ScheduledJob.created_by_user_id == owner_user_id)
            .order_by(ScheduledJob.created_at.desc(), ScheduledJob.job_id.desc())
            .limit(limit)
        )
        with self._session_factory() as session:
            rows = session.execute(statement).all()
            results: list[tuple[ScheduledJob, ScheduledJobRun | None]] = []
            for job, run in rows:
                session.expunge(job)
                if run is not None:
                    session.expunge(run)
                results.append((job, run))
            return results

    def get_job_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: UUID,
    ) -> ScheduledJob | None:
        """Return one job only when it belongs to the authenticated owner."""

        with self._session_factory() as session:
            job = session.execute(
                select(ScheduledJob).where(
                    ScheduledJob.job_id == job_id,
                    ScheduledJob.created_by_user_id == owner_user_id,
                )
            ).scalar_one_or_none()
            if job is not None:
                session.expunge(job)
            return job

    def list_run_history_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: UUID,
        limit: int = 25,
    ) -> list[tuple[ScheduledJobRun, ScheduledJobOutput | None]]:
        """Return recent run/output pairs for one owner job without N+1 reads."""

        if not 1 <= limit <= 100:
            raise ValueError("Scheduled-job history limit must be between 1 and 100.")
        statement = (
            select(ScheduledJobRun, ScheduledJobOutput)
            .join(ScheduledJob, ScheduledJob.job_id == ScheduledJobRun.job_id)
            .outerjoin(
                ScheduledJobOutput,
                ScheduledJobOutput.run_id == ScheduledJobRun.run_id,
            )
            .where(
                ScheduledJobRun.job_id == job_id,
                ScheduledJob.created_by_user_id == owner_user_id,
            )
            .order_by(
                ScheduledJobRun.created_at.desc(),
                ScheduledJobRun.run_id.desc(),
            )
            .limit(limit)
        )
        with self._session_factory() as session:
            rows = session.execute(statement).all()
            results: list[tuple[ScheduledJobRun, ScheduledJobOutput | None]] = []
            for run, output in rows:
                session.expunge(run)
                if output is not None:
                    session.expunge(output)
                results.append((run, output))
            return results

    def reconciliation_cutoff(self) -> datetime:
        """Return the database clock used to freeze one reconciliation scan."""

        with self._session_factory() as session:
            value = session.execute(select(func.current_timestamp())).scalar_one()
            if not isinstance(value, datetime):
                raise ValueError("Database clock did not return a timestamp.")
            if session.bind is not None and session.bind.dialect.name == "sqlite":
                # SQLite truncates CURRENT_TIMESTAMP to whole seconds while
                # ORM defaults retain microseconds. Include that current second
                # so freshly committed test/development rows are not skipped.
                value += timedelta(seconds=1)
            return _as_aware_utc(value)

    def list_reconciliation_job_ids(
        self,
        *,
        target_device_id: str,
        created_through: datetime,
        after_created_at: datetime | None = None,
        after_job_id: str | None = None,
        limit: int = 100,
    ) -> list[tuple[str, datetime]]:
        """List one immutable keyset page of jobs for explicit reconciliation."""

        target = target_device_id.strip()
        if not target:
            raise ValueError("target_device_id must not be empty.")
        if not 1 <= limit <= 500:
            raise ValueError("Reconciliation page limit must be between 1 and 500.")
        if (after_created_at is None) != (after_job_id is None):
            raise ValueError("Both reconciliation cursor fields are required together.")

        cutoff = _as_aware_utc(created_through)
        statement = select(ScheduledJob.job_id, ScheduledJob.created_at).where(
            ScheduledJob.target_device_id == target,
            ScheduledJob.created_at <= cutoff,
            or_(
                ScheduledJob.status.not_in(
                    {
                        ScheduledJobStatus.COMPLETED.value,
                        ScheduledJobStatus.EXECUTION_FAILED.value,
                        ScheduledJobStatus.DELETED.value,
                    }
                ),
                ScheduledJob.provisioning_error.is_not(None),
            ),
        )
        if after_created_at is not None and after_job_id is not None:
            cursor_time = _as_aware_utc(after_created_at)
            statement = statement.where(
                or_(
                    ScheduledJob.created_at > cursor_time,
                    and_(
                        ScheduledJob.created_at == cursor_time,
                        ScheduledJob.job_id > after_job_id,
                    ),
                )
            )
        statement = statement.order_by(
            ScheduledJob.created_at.asc(),
            ScheduledJob.job_id.asc(),
        ).limit(limit)
        with self._session_factory() as session:
            rows = session.execute(statement).all()
        return [(str(row.job_id), _as_aware_utc(row.created_at)) for row in rows]

    def activate_job(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
        timer_unit_name: str,
        activated_at: datetime | None = None,
    ) -> ScheduledJob | None:
        """Activate a provisioned job when its expected lifecycle version matches.

        ``None`` means the job was not found, the source state was invalid, or
        another writer changed its status or ``updated_at`` timestamp first. A
        running occurrence also blocks this generic transition; callers
        resuming a paused job must use ``settle_and_activate_job`` so an expired
        lease is handled atomically.
        """

        allowed_statuses = {
            ScheduledJobStatus.PENDING_PROVISIONING.value,
            ScheduledJobStatus.PROVISIONING_FAILED.value,
            ScheduledJobStatus.PAUSED.value,
        }
        if expected_status not in allowed_statuses:
            return None
        if timer_unit_name != _scheduled_timer_unit_name(job_id):
            raise ValueError(
                "timer_unit_name must be derived from the scheduled job UUID."
            )

        with self._session_factory() as session:
            job = session.execute(
                select(ScheduledJob)
                .where(
                    ScheduledJob.job_id == job_id,
                    ScheduledJob.status == expected_status,
                    ScheduledJob.updated_at == expected_updated_at,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                session.rollback()
                return None
            running_run_id = session.execute(
                select(ScheduledJobRun.run_id)
                .where(
                    ScheduledJobRun.job_id == job_id,
                    ScheduledJobRun.status == ScheduledJobRunStatus.RUNNING.value,
                )
                .limit(1)
                .with_for_update()
            ).scalar_one_or_none()
            if running_run_id is not None:
                session.rollback()
                return None

            database_now = _database_utc_now(session)
            job.status = ScheduledJobStatus.ACTIVE.value
            job.is_active = True
            job.activated_at = _as_aware_utc(activated_at or database_now)
            job.activation_generation += 1
            job.timer_unit_name = timer_unit_name
            job.provisioning_error = None
            job.updated_at = _monotonic_transition_time(job.updated_at, database_now)
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def settle_and_activate_job(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
        timer_unit_name: str,
        activated_at: datetime | None = None,
    ) -> ScheduledJobResumeSettlement | None:
        """Atomically settle an expired in-flight run and resume a paused job.

        A live lease raises ``ScheduledJobRunLeaseBusyError``. An expired run is
        cancelled and fenced in the same short transaction that rotates the
        activation generation. ``None`` reports a lifecycle CAS conflict.
        """

        if expected_status != ScheduledJobStatus.PAUSED.value:
            return None
        if timer_unit_name != _scheduled_timer_unit_name(job_id):
            raise ValueError(
                "timer_unit_name must be derived from the scheduled job UUID."
            )

        with self._session_factory() as session:
            job = session.execute(
                select(ScheduledJob)
                .where(
                    ScheduledJob.job_id == job_id,
                    ScheduledJob.status == expected_status,
                    ScheduledJob.updated_at == expected_updated_at,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                session.rollback()
                return None

            run = session.execute(
                select(ScheduledJobRun)
                .where(
                    ScheduledJobRun.job_id == job_id,
                    ScheduledJobRun.status == ScheduledJobRunStatus.RUNNING.value,
                )
                .order_by(
                    ScheduledJobRun.created_at.asc(), ScheduledJobRun.run_id.asc()
                )
                .limit(1)
                .with_for_update()
            ).scalar_one_or_none()
            database_now = _database_utc_now(session)
            if run is not None:
                if (
                    run.lease_expires_at is not None
                    and _as_aware_utc(run.lease_expires_at) > database_now
                ):
                    raise ScheduledJobRunLeaseBusyError(run.lease_expires_at)
                run.status = ScheduledJobRunStatus.CANCELLED.value
                run.completed_at = database_now
                run.lease_token = None
                run.lease_expires_at = None
                run.attempt_deadline_at = None
                run.retry_not_before_at = None
                run.error_message = "Expired execution cancelled before job resume."
                session.add(
                    ScheduledJobRunLog(
                        run_id=run.run_id,
                        log_level="warning",
                        message="Expired execution lease cancelled before job resume.",
                        metadata_json={"attempt_number": run.attempt_number},
                    )
                )

            job.status = ScheduledJobStatus.ACTIVE.value
            job.is_active = True
            job.activated_at = _as_aware_utc(activated_at or database_now)
            job.activation_generation += 1
            job.timer_unit_name = timer_unit_name
            job.provisioning_error = None
            job.updated_at = _monotonic_transition_time(job.updated_at, database_now)
            session.commit()
            session.refresh(job)
            session.expunge(job)
            if run is not None:
                session.refresh(run)
                session.expunge(run)
            return ScheduledJobResumeSettlement(job=job, cancelled_run=run)

    def repair_timer_unit_name(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
        timer_unit_name: str,
    ) -> ScheduledJob | None:
        """Repair an active job's UUID-derived timer receipt with CAS semantics."""

        if expected_status != ScheduledJobStatus.ACTIVE.value:
            return None
        if timer_unit_name != _scheduled_timer_unit_name(job_id):
            raise ValueError(
                "timer_unit_name must be derived from the scheduled job UUID."
            )
        return self._compare_and_set_job(
            job_id=job_id,
            expected_status=expected_status,
            expected_updated_at=expected_updated_at,
            transitioned_at=utc_now(),
            values={"timer_unit_name": timer_unit_name},
        )

    def mark_provisioning_failed(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
        error: str,
    ) -> ScheduledJob | None:
        """Record a provisioning failure when the expected job version matches.

        Active jobs are accepted for compensation after a timer-start failure.
        ``None`` means the job was not found, the source state was invalid, or
        another writer changed its status or ``updated_at`` timestamp first.
        """

        allowed_statuses = {
            ScheduledJobStatus.PENDING_PROVISIONING.value,
            ScheduledJobStatus.ACTIVE.value,
            ScheduledJobStatus.PROVISIONING_FAILED.value,
        }
        if expected_status not in allowed_statuses:
            return None
        return self._compare_and_set_job(
            job_id=job_id,
            expected_status=expected_status,
            expected_updated_at=expected_updated_at,
            transitioned_at=utc_now(),
            values={
                "status": ScheduledJobStatus.PROVISIONING_FAILED.value,
                "is_active": False,
                "activated_at": None,
                "provisioning_error": _bounded_scheduled_job_error(error),
            },
        )

    def pause_job(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
    ) -> ScheduledJob | None:
        """Pause an active job when its expected lifecycle version matches.

        ``None`` means the job was not found, the source state was invalid, or
        another writer changed its status or ``updated_at`` timestamp first.
        """

        if expected_status != ScheduledJobStatus.ACTIVE.value:
            return None
        return self._compare_and_set_job(
            job_id=job_id,
            expected_status=expected_status,
            expected_updated_at=expected_updated_at,
            transitioned_at=utc_now(),
            values={
                "status": ScheduledJobStatus.PAUSED.value,
                "is_active": False,
                "activated_at": None,
                "provisioning_error": None,
            },
        )

    def mark_deleted(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
    ) -> ScheduledJob | None:
        """Archive a non-deleted job when its expected lifecycle version matches.

        ``None`` means the job was not found, was already deleted, or another
        writer changed its status or ``updated_at`` timestamp first.
        """

        allowed_statuses = {
            ScheduledJobStatus.PENDING_PROVISIONING.value,
            ScheduledJobStatus.ACTIVE.value,
            ScheduledJobStatus.PAUSED.value,
            ScheduledJobStatus.PROVISIONING_FAILED.value,
            ScheduledJobStatus.COMPLETED.value,
            ScheduledJobStatus.EXECUTION_FAILED.value,
        }
        if expected_status not in allowed_statuses:
            return None
        return self._compare_and_set_job(
            job_id=job_id,
            expected_status=expected_status,
            expected_updated_at=expected_updated_at,
            transitioned_at=utc_now(),
            values={
                "status": ScheduledJobStatus.DELETED.value,
                "is_active": False,
                "activated_at": None,
                "timer_unit_name": None,
                "provisioning_error": _SCHEDULED_JOB_CLEANUP_PENDING_MESSAGE,
            },
        )

    def record_external_error(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
        error: str,
    ) -> ScheduledJob | None:
        """Store an external-operation error without changing lifecycle state.

        Deleted jobs are supported so failed post-archive cleanup remains
        visible without reopening the terminal state. ``None`` means the job
        was not found, the source state was invalid, or its version changed.
        """

        if expected_status not in {status.value for status in ScheduledJobStatus}:
            return None
        return self._compare_and_set_job(
            job_id=job_id,
            expected_status=expected_status,
            expected_updated_at=expected_updated_at,
            transitioned_at=utc_now(),
            values={
                "provisioning_error": _bounded_scheduled_job_error(error),
            },
        )

    def clear_external_error(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
    ) -> ScheduledJob | None:
        """Clear an external-operation error without changing lifecycle state.

        Deleted jobs are supported for successful cleanup reconciliation.
        ``None`` means the job was not found, the source state was invalid, or
        another writer changed its status or ``updated_at`` timestamp first.
        """

        if expected_status not in {status.value for status in ScheduledJobStatus}:
            return None
        return self._compare_and_set_job(
            job_id=job_id,
            expected_status=expected_status,
            expected_updated_at=expected_updated_at,
            transitioned_at=utc_now(),
            values={"provisioning_error": None},
        )

    def _compare_and_set_job(
        self,
        *,
        job_id: str,
        expected_status: str,
        expected_updated_at: datetime,
        transitioned_at: datetime,
        values: dict[str, Any],
    ) -> ScheduledJob | None:
        """Apply one short atomic update when status and timestamp still match."""

        normalized_expected = _as_aware_utc(expected_updated_at)
        normalized_transition = _as_aware_utc(transitioned_at)
        if normalized_transition <= normalized_expected:
            normalized_transition = normalized_expected + timedelta(microseconds=1)
        with self._session_factory() as session:
            job = session.execute(
                update(ScheduledJob)
                .where(
                    ScheduledJob.job_id == job_id,
                    ScheduledJob.status == expected_status,
                    ScheduledJob.updated_at == expected_updated_at,
                )
                .values(updated_at=normalized_transition, **values)
                .returning(ScheduledJob)
            ).scalar_one_or_none()
            if job is None:
                session.rollback()
                return None
            session.commit()
            session.expunge(job)
            return job


class ScheduledJobCommandRepository:
    """Persist owner commands, processing leases, and definition revisions.

    Web requests and Jetson processing deliberately use short transactions.
    The external systemd operation occurs only after a command lease has been
    committed, so no database row lock is held while Linux is being changed.
    """

    _ALLOWED_SOURCE_STATUSES = {
        ScheduledJobCommandAction.PROVISION.value: {
            ScheduledJobStatus.PENDING_PROVISIONING.value,
            ScheduledJobStatus.PROVISIONING_FAILED.value,
        },
        ScheduledJobCommandAction.PAUSE.value: {
            ScheduledJobStatus.ACTIVE.value,
        },
        ScheduledJobCommandAction.RESUME.value: {
            ScheduledJobStatus.PAUSED.value,
        },
        ScheduledJobCommandAction.UPDATE.value: {
            ScheduledJobStatus.PENDING_PROVISIONING.value,
            ScheduledJobStatus.PROVISIONING_FAILED.value,
            ScheduledJobStatus.PAUSED.value,
            ScheduledJobStatus.ACTIVE.value,
        },
        ScheduledJobCommandAction.ARCHIVE.value: {
            ScheduledJobStatus.PENDING_PROVISIONING.value,
            ScheduledJobStatus.PROVISIONING_FAILED.value,
            ScheduledJobStatus.PAUSED.value,
            ScheduledJobStatus.ACTIVE.value,
            ScheduledJobStatus.COMPLETED.value,
            ScheduledJobStatus.EXECUTION_FAILED.value,
        },
    }

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""

        self._session_factory = session_factory

    def enqueue_for_owner(
        self,
        *,
        job_id: str,
        action: str,
        owner_user_id: UUID,
        requested_by_username: str,
        expected_job_updated_at: datetime,
        definition: dict[str, Any] | None = None,
        maximum_attempts: int = 3,
    ) -> ScheduledJobCommand:
        """Atomically validate ownership/state and enqueue one command.

        If the same job already has a pending or processing command, that
        command is returned. This makes button retries harmless and preserves
        the single-writer ordering required for lifecycle operations.
        """

        allowed_statuses = self._ALLOWED_SOURCE_STATUSES.get(action)
        if allowed_statuses is None:
            raise ValueError("Unsupported scheduled-job command action.")
        if (action == ScheduledJobCommandAction.UPDATE.value) != (
            definition is not None
        ):
            raise ValueError("Only update commands may contain a definition.")
        if not 1 <= maximum_attempts <= 10:
            raise ValueError("maximum_attempts must be between 1 and 10.")

        with self._session_factory() as session:
            job = session.execute(
                select(ScheduledJob)
                .where(
                    ScheduledJob.job_id == job_id,
                    ScheduledJob.created_by_user_id == owner_user_id,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                raise ValueError("Scheduled job was not found for this owner.")

            open_command = session.execute(
                select(ScheduledJobCommand)
                .where(
                    ScheduledJobCommand.job_id == job_id,
                    ScheduledJobCommand.status.in_(
                        {
                            ScheduledJobCommandStatus.PENDING.value,
                            ScheduledJobCommandStatus.PROCESSING.value,
                        }
                    ),
                )
                .order_by(
                    ScheduledJobCommand.created_at.asc(),
                    ScheduledJobCommand.command_id.asc(),
                )
                .limit(1)
            ).scalar_one_or_none()
            if open_command is not None:
                session.expunge(open_command)
                return open_command

            if job.status not in allowed_statuses:
                raise ValueError(
                    f"Cannot {action} a scheduled job with status {job.status}."
                )
            if definition is not None:
                target = definition.get("target_device")
                requested_target = (
                    target.get("device_id") if isinstance(target, dict) else None
                )
                if requested_target != job.target_device_id:
                    raise ValueError(
                        "A scheduled job cannot be moved to another device by "
                        "editing it."
                    )
            if _as_aware_utc(job.updated_at) != _as_aware_utc(expected_job_updated_at):
                raise ValueError(
                    "Scheduled job changed after the page was loaded; refresh it."
                )

            database_now = _database_utc_now(session)
            command = ScheduledJobCommand(
                job_id=job.job_id,
                action=action,
                requested_job_status=job.status,
                status=ScheduledJobCommandStatus.PENDING.value,
                target_device_id=job.target_device_id,
                requested_by_user_id=owner_user_id,
                requested_by_username=requested_by_username,
                expected_job_updated_at=_as_aware_utc(expected_job_updated_at),
                definition_json=definition,
                maximum_attempts=maximum_attempts,
                available_at=database_now,
                created_at=database_now,
            )
            session.add(command)
            session.commit()
            session.refresh(command)
            session.expunge(command)
            return command

    def latest_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: UUID,
    ) -> ScheduledJobCommand | None:
        """Return the newest command for one job only when ownership matches."""

        statement = (
            select(ScheduledJobCommand)
            .join(ScheduledJob, ScheduledJob.job_id == ScheduledJobCommand.job_id)
            .where(
                ScheduledJobCommand.job_id == job_id,
                ScheduledJob.created_by_user_id == owner_user_id,
            )
            .order_by(
                ScheduledJobCommand.created_at.desc(),
                ScheduledJobCommand.command_id.desc(),
            )
            .limit(1)
        )
        with self._session_factory() as session:
            command = session.execute(statement).scalar_one_or_none()
            if command is not None:
                session.expunge(command)
            return command

    def claim_next(
        self,
        *,
        target_device_id: str,
        worker_id: str,
        lease_seconds: int,
    ) -> ScheduledJobCommand | None:
        """Claim the oldest due command, reclaiming only expired leases."""

        if not target_device_id.strip() or len(target_device_id) > 128:
            raise ValueError("target_device_id must be 1 to 128 characters.")
        if not worker_id.strip() or len(worker_id) > 128:
            raise ValueError("worker_id must be 1 to 128 characters.")
        if not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 5 and 3600.")

        with self._session_factory() as session:
            database_now = _database_utc_now(session)
            session.execute(
                update(ScheduledJobCommand)
                .where(
                    ScheduledJobCommand.target_device_id == target_device_id,
                    ScheduledJobCommand.status
                    == ScheduledJobCommandStatus.PROCESSING.value,
                    ScheduledJobCommand.lease_expires_at <= database_now,
                    ScheduledJobCommand.attempt_count
                    >= ScheduledJobCommand.maximum_attempts,
                )
                .values(
                    status=ScheduledJobCommandStatus.FAILED.value,
                    lease_token=None,
                    lease_expires_at=None,
                    completed_at=database_now,
                    error_message=(
                        "Command worker lease expired after its final attempt."
                    ),
                )
            )
            command = session.execute(
                select(ScheduledJobCommand)
                .where(
                    ScheduledJobCommand.target_device_id == target_device_id,
                    ScheduledJobCommand.attempt_count
                    < ScheduledJobCommand.maximum_attempts,
                    or_(
                        and_(
                            ScheduledJobCommand.status
                            == ScheduledJobCommandStatus.PENDING.value,
                            ScheduledJobCommand.available_at <= database_now,
                        ),
                        and_(
                            ScheduledJobCommand.status
                            == ScheduledJobCommandStatus.PROCESSING.value,
                            ScheduledJobCommand.lease_expires_at <= database_now,
                        ),
                    ),
                )
                .order_by(
                    ScheduledJobCommand.created_at.asc(),
                    ScheduledJobCommand.command_id.asc(),
                )
                .limit(1)
                .with_for_update(skip_locked=True)
            ).scalar_one_or_none()
            if command is None:
                session.commit()
                return None

            command.status = ScheduledJobCommandStatus.PROCESSING.value
            command.attempt_count += 1
            command.lease_token = uuid4()
            command.lease_expires_at = database_now + timedelta(seconds=lease_seconds)
            command.worker_id = worker_id
            command.started_at = database_now
            command.completed_at = None
            session.commit()
            session.refresh(command)
            session.expunge(command)
            return command

    def mark_succeeded(
        self,
        *,
        command_id: UUID,
        lease_token: UUID,
        resulting_job_status: str,
    ) -> ScheduledJobCommand | None:
        """Complete a command only when the caller still owns its lease."""

        with self._session_factory() as session:
            database_now = _database_utc_now(session)
            command = session.execute(
                update(ScheduledJobCommand)
                .where(
                    ScheduledJobCommand.command_id == command_id,
                    ScheduledJobCommand.status
                    == ScheduledJobCommandStatus.PROCESSING.value,
                    ScheduledJobCommand.lease_token == lease_token,
                )
                .values(
                    status=ScheduledJobCommandStatus.SUCCEEDED.value,
                    resulting_job_status=resulting_job_status,
                    error_message=None,
                    lease_token=None,
                    lease_expires_at=None,
                    completed_at=database_now,
                )
                .returning(ScheduledJobCommand)
            ).scalar_one_or_none()
            if command is None:
                session.rollback()
                return None
            session.commit()
            session.expunge(command)
            return command

    def mark_failed_or_retry(
        self,
        *,
        command_id: UUID,
        lease_token: UUID,
        error: str,
        retry_delay_seconds: int,
        retry: bool = True,
    ) -> ScheduledJobCommand | None:
        """Release a failed attempt for retry or make it terminal when exhausted."""

        if not 1 <= retry_delay_seconds <= 3600:
            raise ValueError("retry_delay_seconds must be between 1 and 3600.")
        with self._session_factory() as session:
            command = session.execute(
                select(ScheduledJobCommand)
                .where(
                    ScheduledJobCommand.command_id == command_id,
                    ScheduledJobCommand.status
                    == ScheduledJobCommandStatus.PROCESSING.value,
                    ScheduledJobCommand.lease_token == lease_token,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if command is None:
                session.rollback()
                return None

            database_now = _database_utc_now(session)
            command.error_message = _bounded_scheduled_job_error(error)
            command.lease_token = None
            command.lease_expires_at = None
            if retry and command.attempt_count < command.maximum_attempts:
                command.status = ScheduledJobCommandStatus.PENDING.value
                command.available_at = database_now + timedelta(
                    seconds=retry_delay_seconds
                )
                command.completed_at = None
            else:
                command.status = ScheduledJobCommandStatus.FAILED.value
                command.completed_at = database_now
            session.commit()
            session.refresh(command)
            session.expunge(command)
            return command

    def apply_definition_revision(
        self,
        *,
        job_id: str,
        expected_updated_at: datetime,
        definition: dict[str, Any],
        changed_by_user_id: UUID,
        changed_by_username: str,
        change_kind: str = "edited",
    ) -> ScheduledJob | None:
        """CAS-apply a validated definition and append its immutable revision."""

        if change_kind not in {"edited", "rollback"}:
            raise ValueError("change_kind must be edited or rollback.")
        trusted_fields = (
            definition.get("job_name"),
            definition.get("job_type"),
            definition.get("schema_version"),
        )
        if not all(isinstance(value, str) for value in trusted_fields):
            raise ValueError("Validated definition is missing trusted job fields.")

        with self._session_factory() as session:
            job = session.execute(
                select(ScheduledJob)
                .where(ScheduledJob.job_id == job_id)
                .with_for_update()
            ).scalar_one_or_none()
            if job is None or _as_aware_utc(job.updated_at) != _as_aware_utc(
                expected_updated_at
            ):
                session.rollback()
                return None
            if job.status not in {
                ScheduledJobStatus.PENDING_PROVISIONING.value,
                ScheduledJobStatus.PROVISIONING_FAILED.value,
                ScheduledJobStatus.PAUSED.value,
            }:
                session.rollback()
                return None

            latest_revision = session.execute(
                select(func.max(ScheduledJobRevision.revision_number)).where(
                    ScheduledJobRevision.job_id == job_id
                )
            ).scalar_one()
            revision_number = int(latest_revision or 0) + 1
            database_now = _database_utc_now(session)
            job.job_name = str(definition["job_name"])
            job.job_type = str(definition["job_type"])
            job.schema_version = str(definition["schema_version"])
            job.definition_json = definition
            job.provisioning_error = None
            job.updated_at = _monotonic_transition_time(job.updated_at, database_now)
            session.add(
                ScheduledJobRevision(
                    job_id=job.job_id,
                    revision_number=revision_number,
                    schema_version=job.schema_version,
                    definition_json=definition,
                    changed_by_user_id=changed_by_user_id,
                    changed_by_username=changed_by_username,
                    change_kind=change_kind,
                    created_at=database_now,
                )
            )
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def list_revisions_for_owner(
        self,
        *,
        job_id: str,
        owner_user_id: UUID,
        limit: int = 25,
    ) -> list[ScheduledJobRevision]:
        """Return a bounded newest-first revision history for one owner job."""

        if not 1 <= limit <= 100:
            raise ValueError("Revision history limit must be between 1 and 100.")
        statement = (
            select(ScheduledJobRevision)
            .join(ScheduledJob, ScheduledJob.job_id == ScheduledJobRevision.job_id)
            .where(
                ScheduledJobRevision.job_id == job_id,
                ScheduledJob.created_by_user_id == owner_user_id,
            )
            .order_by(ScheduledJobRevision.revision_number.desc())
            .limit(limit)
        )
        with self._session_factory() as session:
            revisions = list(session.execute(statement).scalars())
            for revision in revisions:
                session.expunge(revision)
            return revisions


class ScheduledJobRunRepository:
    """Persist fenced claims, retries, terminal states, logs, and outputs."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""

        self._session_factory = session_factory

    @staticmethod
    def _execution_policy(
        job: ScheduledJob,
        *,
        timeout_seconds: int | None,
        lease_seconds: int,
        retry_interval_seconds: int | None = None,
    ) -> tuple[int, int, int]:
        """Return validated hard-timeout, lease, and backoff durations."""

        retry = job.definition_json.get("retry")
        retry = retry if isinstance(retry, dict) else {}
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else retry.get("timeout_seconds", 600)
        )
        retry_interval = (
            retry_interval_seconds
            if retry_interval_seconds is not None
            else retry.get("retry_interval_seconds", 60)
        )
        return (
            _positive_seconds(timeout, field_name="timeout_seconds"),
            _positive_seconds(lease_seconds, field_name="lease_seconds"),
            _positive_seconds(
                retry_interval,
                field_name="retry_interval_seconds",
            ),
        )

    @staticmethod
    def _clear_execution_lease(run: ScheduledJobRun) -> None:
        """Clear ownership fields when a run leaves the running state."""

        run.lease_token = None
        run.lease_expires_at = None
        run.attempt_deadline_at = None
        run.retry_not_before_at = None

    @staticmethod
    def _terminalize_one_time_job(
        job: ScheduledJob,
        run: ScheduledJobRun,
        *,
        succeeded: bool,
        completed_at: datetime,
    ) -> None:
        """Retire the matching activation of a one-time job atomically."""

        if (
            not _is_one_time_job(job)
            or job.status
            not in {
                ScheduledJobStatus.ACTIVE.value,
                ScheduledJobStatus.PAUSED.value,
            }
            or job.activation_generation != run.activation_generation
        ):
            return
        job.status = (
            ScheduledJobStatus.COMPLETED.value
            if succeeded
            else ScheduledJobStatus.EXECUTION_FAILED.value
        )
        job.is_active = False
        job.activated_at = None
        job.timer_unit_name = None
        job.provisioning_error = _SCHEDULED_JOB_CLEANUP_PENDING_MESSAGE
        job.updated_at = _monotonic_transition_time(job.updated_at, completed_at)

    @staticmethod
    def _cancel_locked_run(
        session: Session,
        run: ScheduledJobRun,
        *,
        completed_at: datetime,
        error_message: str,
        log_message: str,
    ) -> None:
        """Cancel and fence a locked running row with one audit message."""

        run.status = ScheduledJobRunStatus.CANCELLED.value
        run.completed_at = completed_at
        run.error_message = error_message
        ScheduledJobRunRepository._clear_execution_lease(run)
        session.add(
            ScheduledJobRunLog(
                run_id=run.run_id,
                log_level="warning",
                message=log_message,
                metadata_json={"attempt_number": run.attempt_number},
            )
        )

    @staticmethod
    def _require_fenced_running_run(
        run: ScheduledJobRun,
        *,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        database_now: datetime,
    ) -> None:
        """Reject terminal rows and workers with superseded or expired leases."""

        if run.status != ScheduledJobRunStatus.RUNNING.value:
            raise ValueError("Only a running scheduled-job run can transition.")
        if (
            run.lease_token != expected_lease_token
            or run.attempt_number != expected_attempt_number
        ):
            raise ValueError("Scheduled-job execution lease is no longer owned.")
        if (
            run.lease_expires_at is None
            or _as_aware_utc(run.lease_expires_at) <= database_now
        ):
            raise ValueError("Scheduled-job execution lease has expired.")

    def _locked_job_and_run(
        self,
        session: Session,
        *,
        run_id: UUID,
    ) -> tuple[ScheduledJob, ScheduledJobRun] | None:
        """Lock a run's job first and its run second to prevent deadlocks."""

        candidate_job_id = session.execute(
            select(ScheduledJobRun.job_id).where(ScheduledJobRun.run_id == run_id)
        ).scalar_one_or_none()
        if candidate_job_id is None:
            return None
        job = session.execute(
            select(ScheduledJob)
            .where(ScheduledJob.job_id == candidate_job_id)
            .with_for_update()
        ).scalar_one_or_none()
        if job is None:
            return None
        run = session.execute(
            select(ScheduledJobRun)
            .where(ScheduledJobRun.run_id == run_id)
            .with_for_update()
        ).scalar_one_or_none()
        if run is None:
            return None
        return job, run

    def create_run(
        self,
        *,
        job_id: str,
        scheduled_for: datetime,
        triggered_by: str,
        timeout_seconds: int = 600,
        lease_seconds: int = 90,
    ) -> ScheduledJobRun:
        """Claim one occurrence with a database-clock fenced execution lease."""

        with self._session_factory() as session:
            job = session.execute(
                select(ScheduledJob)
                .where(
                    ScheduledJob.job_id == job_id,
                    ScheduledJob.status == ScheduledJobStatus.ACTIVE.value,
                    ScheduledJob.is_active.is_(True),
                    ScheduledJob.activated_at.is_not(None),
                    ScheduledJob.activated_at <= scheduled_for,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                raise ValueError("Scheduled job is not active for this occurrence.")
            timeout, lease, _ = self._execution_policy(
                job,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
            )
            database_now = _database_utc_now(session)
            attempt_deadline = database_now + timedelta(seconds=timeout)
            run = ScheduledJobRun(
                job_id=job_id,
                scheduled_for=scheduled_for,
                status=ScheduledJobRunStatus.RUNNING.value,
                attempt_number=1,
                activation_generation=job.activation_generation,
                lease_token=uuid4(),
                lease_expires_at=min(
                    database_now + timedelta(seconds=lease),
                    attempt_deadline,
                ),
                attempt_deadline_at=attempt_deadline,
                retry_not_before_at=None,
                started_at=database_now,
                triggered_by=triggered_by,
            )
            session.add(run)
            session.flush()
            session.add(
                ScheduledJobRunLog(
                    run_id=run.run_id,
                    log_level="info",
                    message="Scheduled occurrence claimed for execution.",
                    metadata_json={"attempt_number": 1},
                )
            )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def get_run_for_occurrence(
        self,
        *,
        job_id: str,
        scheduled_for: datetime,
    ) -> ScheduledJobRun | None:
        """Return the run that already owns a logical scheduled occurrence."""

        with self._session_factory() as session:
            run = session.execute(
                select(ScheduledJobRun).where(
                    ScheduledJobRun.job_id == job_id,
                    ScheduledJobRun.scheduled_for == scheduled_for,
                )
            ).scalar_one_or_none()
            if run is not None:
                session.expunge(run)
            return run

    def get_running_run_for_job(self, *, job_id: str) -> ScheduledJobRun | None:
        """Return the sole running occurrence for one scheduled job."""

        with self._session_factory() as session:
            run = session.execute(
                select(ScheduledJobRun)
                .where(
                    ScheduledJobRun.job_id == job_id,
                    ScheduledJobRun.status == ScheduledJobRunStatus.RUNNING.value,
                )
                .limit(1)
            ).scalar_one_or_none()
            if run is not None:
                session.expunge(run)
            return run

    def renew_lease(
        self,
        *,
        run_id: UUID,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        lease_seconds: int = 90,
    ) -> ScheduledJobRun | None:
        """Renew a currently owned lease without reviving a superseded worker."""

        lease = _positive_seconds(lease_seconds, field_name="lease_seconds")
        with self._session_factory() as session:
            run = session.execute(
                select(ScheduledJobRun)
                .where(ScheduledJobRun.run_id == run_id)
                .with_for_update()
            ).scalar_one_or_none()
            if run is None:
                return None
            database_now = _database_utc_now(session)
            deadline = (
                _as_aware_utc(run.attempt_deadline_at)
                if run.attempt_deadline_at is not None
                else None
            )
            if deadline is not None and database_now >= deadline:
                raise ValueError("Scheduled-job attempt deadline has elapsed.")
            self._require_fenced_running_run(
                run,
                expected_lease_token=expected_lease_token,
                expected_attempt_number=expected_attempt_number,
                database_now=database_now,
            )
            candidate_expiry = database_now + timedelta(seconds=lease)
            if deadline is not None:
                candidate_expiry = min(candidate_expiry, deadline)
            run.lease_expires_at = candidate_expiry
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def schedule_retry(
        self,
        *,
        run_id: UUID,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        previous_error: str,
        retry_interval_seconds: int,
        lease_seconds: int = 90,
    ) -> ScheduledJobRun | None:
        """Durably schedule the next attempt before sleeping outside the DB."""

        retry_interval = _positive_seconds(
            retry_interval_seconds,
            field_name="retry_interval_seconds",
        )
        lease = _positive_seconds(lease_seconds, field_name="lease_seconds")
        with self._session_factory() as session:
            locked = self._locked_job_and_run(session, run_id=run_id)
            if locked is None:
                return None
            _, run = locked
            database_now = _database_utc_now(session)
            self._require_fenced_running_run(
                run,
                expected_lease_token=expected_lease_token,
                expected_attempt_number=expected_attempt_number,
                database_now=database_now,
            )
            run.attempt_number += 1
            run.retry_not_before_at = database_now + timedelta(seconds=retry_interval)
            run.attempt_deadline_at = None
            run.lease_expires_at = database_now + timedelta(seconds=lease)
            run.error_message = previous_error
            session.add(
                ScheduledJobRunLog(
                    run_id=run.run_id,
                    log_level="warning",
                    message="Execution attempt failed; retry scheduled.",
                    metadata_json={
                        "attempt_number": run.attempt_number,
                        "previous_error": previous_error,
                        "retry_not_before_at": run.retry_not_before_at.isoformat(),
                    },
                )
            )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def begin_retry_attempt(
        self,
        *,
        run_id: UUID,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        timeout_seconds: int,
        lease_seconds: int = 90,
    ) -> ScheduledJobRun | None:
        """Move an owned retry from durable backoff into execution."""

        with self._session_factory() as session:
            locked = self._locked_job_and_run(session, run_id=run_id)
            if locked is None:
                return None
            job, run = locked
            database_now = _database_utc_now(session)
            self._require_fenced_running_run(
                run,
                expected_lease_token=expected_lease_token,
                expected_attempt_number=expected_attempt_number,
                database_now=database_now,
            )
            timeout, lease, _ = self._execution_policy(
                job,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
            )
            if run.retry_not_before_at is None:
                raise ValueError("Scheduled-job run is not waiting for a retry.")
            retry_at = _as_aware_utc(run.retry_not_before_at)
            if database_now < retry_at:
                raise ScheduledJobRunLeaseBusyError(retry_at)
            deadline = database_now + timedelta(seconds=timeout)
            run.started_at = database_now
            run.attempt_deadline_at = deadline
            run.retry_not_before_at = None
            run.lease_expires_at = min(
                database_now + timedelta(seconds=lease),
                deadline,
            )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def recover_or_get_running_run(
        self,
        *,
        job_id: str,
        stale_before: datetime | None = None,
        triggered_by: str,
        maximum_attempts: int,
        timeout_seconds: int | None = None,
        lease_seconds: int = 90,
        retry_interval_seconds: int | None = None,
    ) -> tuple[ScheduledJobRun, bool] | None:
        """Return a live lease or atomically reclaim one using database time.

        ``stale_before`` remains accepted for source compatibility but is not an
        authority: the persisted lease and database clock decide ownership.
        """

        del stale_before
        with self._session_factory() as session:
            job = session.execute(
                select(ScheduledJob)
                .where(
                    ScheduledJob.job_id == job_id,
                    ScheduledJob.status == ScheduledJobStatus.ACTIVE.value,
                    ScheduledJob.is_active.is_(True),
                    ScheduledJob.activated_at.is_not(None),
                )
                .with_for_update()
            ).scalar_one_or_none()
            if job is None:
                raise ValueError("Scheduled job is no longer active.")
            run = session.execute(
                select(ScheduledJobRun)
                .where(
                    ScheduledJobRun.job_id == job_id,
                    ScheduledJobRun.status == ScheduledJobRunStatus.RUNNING.value,
                )
                .limit(1)
                .with_for_update()
            ).scalar_one_or_none()
            if run is None:
                return None
            database_now = _database_utc_now(session)
            if run.activation_generation != job.activation_generation:
                self._cancel_locked_run(
                    session,
                    run,
                    completed_at=database_now,
                    error_message="Occurrence predates the job's current activation.",
                    log_message="Pre-activation occurrence cancelled during recovery.",
                )
                session.commit()
                return None
            if (
                run.lease_expires_at is not None
                and _as_aware_utc(run.lease_expires_at) > database_now
            ):
                session.expunge(run)
                return run, False

            timeout, lease, retry_interval = self._execution_policy(
                job,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
                retry_interval_seconds=retry_interval_seconds,
            )
            self._recover_locked_run(
                session,
                job=job,
                run=run,
                database_now=database_now,
                triggered_by=triggered_by,
                maximum_attempts=maximum_attempts,
                timeout_seconds=timeout,
                lease_seconds=lease,
                retry_interval_seconds=retry_interval,
            )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run, True

    def _recover_locked_run(
        self,
        session: Session,
        *,
        job: ScheduledJob,
        run: ScheduledJobRun,
        database_now: datetime,
        triggered_by: str,
        maximum_attempts: int,
        timeout_seconds: int,
        lease_seconds: int,
        retry_interval_seconds: int,
    ) -> None:
        """Fence and recover one expired run while job then run are locked."""

        if run.retry_not_before_at is None and run.attempt_number >= maximum_attempts:
            run.status = ScheduledJobRunStatus.TIMED_OUT.value
            run.completed_at = database_now
            run.error_message = (
                "Execution lease expired after the final permitted attempt."
            )
            self._clear_execution_lease(run)
            self._terminalize_one_time_job(
                job,
                run,
                succeeded=False,
                completed_at=database_now,
            )
            session.add(
                ScheduledJobRunLog(
                    run_id=run.run_id,
                    log_level="error",
                    message="Expired final execution lease marked timed out.",
                    metadata_json={"attempt_number": run.attempt_number},
                )
            )
            return

        run.lease_token = uuid4()
        run.triggered_by = triggered_by
        if run.retry_not_before_at is not None:
            retry_at = _as_aware_utc(run.retry_not_before_at)
            if retry_at <= database_now:
                deadline = database_now + timedelta(seconds=timeout_seconds)
                run.started_at = database_now
                run.retry_not_before_at = None
                run.attempt_deadline_at = deadline
                run.lease_expires_at = min(
                    database_now + timedelta(seconds=lease_seconds),
                    deadline,
                )
                message = "Expired retry-wait lease reclaimed for execution."
            else:
                run.lease_expires_at = database_now + timedelta(seconds=lease_seconds)
                message = "Expired retry-wait lease reclaimed before its due time."
        else:
            run.attempt_number += 1
            run.attempt_deadline_at = None
            run.retry_not_before_at = database_now + timedelta(
                seconds=retry_interval_seconds
            )
            run.lease_expires_at = database_now + timedelta(seconds=lease_seconds)
            run.error_message = "Previous execution lease expired."
            message = "Expired execution lease reclaimed for retry."
        session.add(
            ScheduledJobRunLog(
                run_id=run.run_id,
                log_level="warning",
                message=message,
                metadata_json={"attempt_number": run.attempt_number},
            )
        )

    def recover_stale_run(
        self,
        *,
        run_id: UUID,
        stale_before: datetime | None = None,
        triggered_by: str,
        maximum_attempts: int,
        timeout_seconds: int | None = None,
        lease_seconds: int = 90,
        retry_interval_seconds: int | None = None,
    ) -> ScheduledJobRun | None:
        """Reclaim one expired run by identifier using its persisted lease."""

        del stale_before
        with self._session_factory() as session:
            locked = self._locked_job_and_run(session, run_id=run_id)
            if locked is None:
                return None
            job, run = locked
            if (
                job.status != ScheduledJobStatus.ACTIVE.value
                or not job.is_active
                or job.activated_at is None
            ):
                raise ValueError("Scheduled job is no longer active.")
            if run.status != ScheduledJobRunStatus.RUNNING.value:
                return None
            database_now = _database_utc_now(session)
            if run.activation_generation != job.activation_generation:
                self._cancel_locked_run(
                    session,
                    run,
                    completed_at=database_now,
                    error_message="Occurrence predates the job's current activation.",
                    log_message="Pre-activation occurrence cancelled during recovery.",
                )
            elif (
                run.lease_expires_at is not None
                and _as_aware_utc(run.lease_expires_at) > database_now
            ):
                return None
            else:
                timeout, lease, retry_interval = self._execution_policy(
                    job,
                    timeout_seconds=timeout_seconds,
                    lease_seconds=lease_seconds,
                    retry_interval_seconds=retry_interval_seconds,
                )
                self._recover_locked_run(
                    session,
                    job=job,
                    run=run,
                    database_now=database_now,
                    triggered_by=triggered_by,
                    maximum_attempts=maximum_attempts,
                    timeout_seconds=timeout,
                    lease_seconds=lease,
                    retry_interval_seconds=retry_interval,
                )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def record_retry(
        self,
        *,
        run_id: UUID,
        attempt_number: int,
        previous_error: str,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        timeout_seconds: int = 600,
        lease_seconds: int = 90,
    ) -> ScheduledJobRun | None:
        """Advance directly to a retry after the caller's completed backoff."""

        with self._session_factory() as session:
            locked = self._locked_job_and_run(session, run_id=run_id)
            if locked is None:
                return None
            job, run = locked
            database_now = _database_utc_now(session)
            self._require_fenced_running_run(
                run,
                expected_lease_token=expected_lease_token,
                expected_attempt_number=expected_attempt_number,
                database_now=database_now,
            )
            if attempt_number != run.attempt_number + 1:
                raise ValueError("Retry attempt must advance by exactly one.")
            timeout, lease, _ = self._execution_policy(
                job,
                timeout_seconds=timeout_seconds,
                lease_seconds=lease_seconds,
            )
            deadline = database_now + timedelta(seconds=timeout)
            run.attempt_number = attempt_number
            run.lease_token = uuid4()
            run.lease_expires_at = min(
                database_now + timedelta(seconds=lease),
                deadline,
            )
            run.attempt_deadline_at = deadline
            run.retry_not_before_at = None
            run.started_at = database_now
            run.error_message = previous_error
            session.add(
                ScheduledJobRunLog(
                    run_id=run.run_id,
                    log_level="warning",
                    message="Execution attempt failed; retrying.",
                    metadata_json={
                        "attempt_number": attempt_number,
                        "previous_error": previous_error,
                    },
                )
            )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def complete_run(
        self,
        *,
        run_id: UUID,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        output_type: str,
        content: str | None,
        content_json: dict[str, Any] | None,
        artifact_path: str | None,
        output_metadata: dict[str, Any],
    ) -> tuple[ScheduledJobRun, ScheduledJobOutput] | None:
        """Persist output and atomically retire a matching one-time job."""

        with self._session_factory() as session:
            locked = self._locked_job_and_run(session, run_id=run_id)
            if locked is None:
                return None
            job, run = locked
            completed_at = _database_utc_now(session)
            self._require_fenced_running_run(
                run,
                expected_lease_token=expected_lease_token,
                expected_attempt_number=expected_attempt_number,
                database_now=completed_at,
            )
            run.status = ScheduledJobRunStatus.COMPLETED.value
            run.completed_at = completed_at
            run.error_message = None
            self._clear_execution_lease(run)
            self._terminalize_one_time_job(
                job,
                run,
                succeeded=True,
                completed_at=completed_at,
            )
            output = ScheduledJobOutput(
                run_id=run.run_id,
                output_type=output_type,
                content=content,
                content_json=content_json,
                artifact_path=artifact_path,
                metadata_json=output_metadata,
            )
            session.add(output)
            session.add(
                ScheduledJobRunLog(
                    run_id=run.run_id,
                    log_level="info",
                    message="Scheduled-job execution completed.",
                    metadata_json={"attempt_number": run.attempt_number},
                )
            )
            session.commit()
            session.refresh(run)
            session.refresh(output)
            session.expunge(run)
            session.expunge(output)
            return run, output

    def cancel_run(
        self,
        *,
        run_id: UUID,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        reason: str,
    ) -> ScheduledJobRun | None:
        """Cooperatively cancel a fenced run and retire a one-time activation."""

        with self._session_factory() as session:
            locked = self._locked_job_and_run(session, run_id=run_id)
            if locked is None:
                return None
            job, run = locked
            completed_at = _database_utc_now(session)
            self._require_fenced_running_run(
                run,
                expected_lease_token=expected_lease_token,
                expected_attempt_number=expected_attempt_number,
                database_now=completed_at,
            )
            self._cancel_locked_run(
                session,
                run,
                completed_at=completed_at,
                error_message=reason,
                log_message="Scheduled-job execution cancelled cooperatively.",
            )
            self._terminalize_one_time_job(
                job,
                run,
                succeeded=False,
                completed_at=completed_at,
            )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def fail_run(
        self,
        *,
        run_id: UUID,
        expected_lease_token: UUID,
        expected_attempt_number: int,
        error_message: str,
        timed_out: bool = False,
    ) -> ScheduledJobRun | None:
        """Persist a fenced failure and retire a matching one-time job."""

        with self._session_factory() as session:
            locked = self._locked_job_and_run(session, run_id=run_id)
            if locked is None:
                return None
            job, run = locked
            completed_at = _database_utc_now(session)
            self._require_fenced_running_run(
                run,
                expected_lease_token=expected_lease_token,
                expected_attempt_number=expected_attempt_number,
                database_now=completed_at,
            )
            run.status = (
                ScheduledJobRunStatus.TIMED_OUT.value
                if timed_out
                else ScheduledJobRunStatus.FAILED.value
            )
            run.completed_at = completed_at
            run.error_message = error_message
            self._clear_execution_lease(run)
            self._terminalize_one_time_job(
                job,
                run,
                succeeded=False,
                completed_at=completed_at,
            )
            session.add(
                ScheduledJobRunLog(
                    run_id=run.run_id,
                    log_level="error",
                    message=(
                        "Scheduled-job execution timed out."
                        if timed_out
                        else "Scheduled-job execution failed."
                    ),
                    metadata_json={
                        "attempt_number": run.attempt_number,
                        "error": error_message,
                    },
                )
            )
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run
