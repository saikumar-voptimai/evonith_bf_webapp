"""Repository classes for PostgreSQL-backed relational persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timezone
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from sqlalchemy import MetaData, Table, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker, undefer

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
    User,
    UserRole,
    UserRoleAssignment,
    utc_now,
)


def _new_id(prefix: str) -> str:
    """Return a compact application id with a stable prefix."""
    return f"{prefix}_{uuid4().hex}"


_UNSET: Any = object()


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
            # These small columns are deferred on the model, but callers inspect
            # them after the rows are detached below. Load them while the session
            # is active to avoid a detached-instance lazy-load error.
            stmt = select(MemoryDocument).options(
                undefer(MemoryDocument.content_type),
                undefer(MemoryDocument.file_size),
                undefer(MemoryDocument.sha256),
            )
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
