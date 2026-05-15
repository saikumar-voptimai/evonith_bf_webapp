"""Repository classes for Neon-backed relational persistence."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    BURDEN_VALUE_COLUMNS,
    HOPPER_COLUMNS,
    BurdenHistory,
    Hopper,
    HopperRawMaterialHistory,
    Material,
    User,
    UserRole,
    UserRoleAssignment,
)


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class UserRepository:
    """User/auth repository operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
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
            session.add(
                UserRoleAssignment(user_id=user.id, role=UserRole.ADMIN.value)
            )
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
            stmt = select(User.id).where(User.username == username, User.is_active.is_(True))
            row = session.execute(stmt).first()
            return row[0] if row else None


class PlantMasterRepository:
    """Read-only plant master lookups used by app repositories and UI."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def list_active_hoppers(self) -> list[dict[str, Any]]:
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
        return {
            row["material_name"]: row["material_code"]
            for row in self.list_active_materials()
        }

    def material_name_by_code(self) -> dict[str, str]:
        return {
            row["material_code"]: row["material_name"]
            for row in self.list_active_materials()
        }

class HopperHistoryRepository:
    """Repository for wide hopper-material snapshot history."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @staticmethod
    def _row_to_codes(row: HopperRawMaterialHistory | None) -> dict[str, str | None]:
        if row is None:
            return {column: None for column in HOPPER_COLUMNS}
        return {column: getattr(row, column) for column in HOPPER_COLUMNS}

    def _latest_row(
        self, session: Session, ts: datetime | None = None
    ) -> HopperRawMaterialHistory | None:
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

    def get_hopper_material_code_at(
        self, hopper: str, ts: datetime
    ) -> str | None:
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
                    "user_modified": str(row.user_modified) if row.user_modified else None,
                }
                payload.update({column: getattr(row, column) for column in HOPPER_COLUMNS})
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
        self._session_factory = session_factory

    @staticmethod
    def burden_fields() -> list[str]:
        return list(BURDEN_VALUE_COLUMNS)

    def _latest_row(self, session: Session, ts: datetime | None = None) -> BurdenHistory | None:
        stmt = select(BurdenHistory)
        if ts is not None:
            stmt = stmt.where(BurdenHistory.date_time <= _as_aware_utc(ts))
        stmt = stmt.order_by(BurdenHistory.date_time.desc(), BurdenHistory.id.desc()).limit(1)
        return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def _row_to_values(row: BurdenHistory | None) -> dict[str, Any]:
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
                    "user_modified": str(row.user_modified) if row.user_modified else None,
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
            session.execute(delete(BurdenHistory).where(BurdenHistory.id.in_(record_ids)))
            session.commit()

    def fetch_distribution_frame(self, *, start_date: date, end_date: date) -> pd.DataFrame:
        """Return latest burden snapshots overlapping the date window."""
        window_start = datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc)
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
        file_path: str | None = None,
        summary: str | None = None,
        qdrant_collection: str | None = None,
        qdrant_point_ids: list | None = None,
        token_estimate: int | None = None,
        metadata: dict | None = None,
    ) -> MemoryDocument:
        """
        Create and return uploaded document metadata.

        Args:
             - user_id: str - User that owns the document.
             - filename: str - Original uploaded file name.
             - file_type: str | None - Optional file type or extension.
             - file_path: str | None - Optional local file path.
             - summary: str | None - Optional document summary.
             - qdrant_collection: str | None - Optional Qdrant collection name.
             - qdrant_point_ids: list | None - Optional Qdrant point ids for chunks.
             - token_estimate: int | None - Optional estimated token count.
             - metadata: dict | None - Optional JSON metadata for the document.

        Returns:
             - return: MemoryDocument - Created document ORM row.
        """
        now = utc_now()
        document = MemoryDocument(
            document_id=_new_id("doc"),
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_path=file_path,
            summary=summary,
            qdrant_collection=qdrant_collection,
            qdrant_point_ids=qdrant_point_ids or [],
            token_estimate=token_estimate,
            metadata_json=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._session_factory() as session:
            session.add(document)
            session.commit()
            session.refresh(document)
            session.expunge(document)
            return document

    def list_documents(
        self,
        *,
        user_id: str,
        active_only: bool = True,
    ) -> list[MemoryDocument]:
        """
        List uploaded documents for one user.

        Args:
             - user_id: str - User whose documents should be listed.
             - active_only: bool - Whether to return only active documents.

        Returns:
             - return: list[MemoryDocument] - Uploaded document rows.
        """
        with self._session_factory() as session:
            stmt = select(MemoryDocument).where(MemoryDocument.user_id == user_id)
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


class LongTermMemoryRepository:
    """Repository for durable FurnaceMind memories."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def create_memory(
        self,
        *,
        user_id: str,
        memory_text: str,
        qdrant_collection: str | None = None,
        qdrant_point_id: str | None = None,
        source_conversation_id: str | None = None,
        source_user_message_id: str | None = None,
        source_assistant_message_id: str | None = None,
        token_estimate: int | None = None,
        metadata: dict | None = None,
    ) -> LongTermMemory:
        """
        Create and return one long-term memory.

        Args:
             - user_id: str - User that owns the memory.
             - memory_text: str - Durable memory text to store.
             - qdrant_collection: str | None - Optional Qdrant collection name.
             - qdrant_point_id: str | None - Optional Qdrant point id.
             - source_conversation_id: str | None - Source conversation id.
             - source_user_message_id: str | None - Source user message id.
             - source_assistant_message_id: str | None - Source assistant message id.
             - token_estimate: int | None - Optional estimated token count.
             - metadata: dict | None - Optional JSON metadata for the memory.

        Returns:
             - return: LongTermMemory - Created memory ORM row.
        """
        now = utc_now()
        memory = LongTermMemory(
            memory_id=_new_id("ltm"),
            user_id=user_id,
            memory_text=memory_text,
            qdrant_collection=qdrant_collection,
            qdrant_point_id=qdrant_point_id,
            source_conversation_id=source_conversation_id,
            source_user_message_id=source_user_message_id,
            source_assistant_message_id=source_assistant_message_id,
            token_estimate=token_estimate,
            metadata_json=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._session_factory() as session:
            session.add(memory)
            session.commit()
            session.refresh(memory)
            session.expunge(memory)
            return memory

    def list_memories(self, *, user_id: str, limit: int = 50) -> list[LongTermMemory]:
        """
        List recent long-term memories for one user.

        Args:
             - user_id: str - User whose memories should be listed.
             - limit: int - Maximum number of memories to return.

        Returns:
             - return: list[LongTermMemory] - Recent long-term memory rows.
        """
        with self._session_factory() as session:
            stmt = (
                select(LongTermMemory)
                .where(LongTermMemory.user_id == user_id)
                .order_by(LongTermMemory.created_at.desc())
                .limit(limit)
            )
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows


class SkillRepository:
    """Repository for built-in and uploaded FurnaceMind skills."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def create_skill(
        self,
        *,
        name: str,
        instruction: str,
        icon: str | None = None,
        description: str | None = None,
        source_type: str = "uploaded",
        qdrant_collection: str | None = None,
        is_active: bool = True,
        created_by: str | None = None,
        metadata: dict | None = None,
    ) -> Skill:
        """
        Create and return a skill definition.

        Args:
             - name: str - Skill display name.
             - instruction: str - Prompt instruction used when the skill is selected.
             - icon: str | None - Optional icon label for the skill.
             - description: str | None - Optional skill description.
             - source_type: str - Skill source, such as built_in or uploaded.
             - qdrant_collection: str | None - Optional Qdrant collection name.
             - is_active: bool - Whether the skill is active.
             - created_by: str | None - User that created the skill.
             - metadata: dict | None - Optional JSON metadata for the skill.

        Returns:
             - return: Skill - Created skill ORM row.
        """
        now = utc_now()
        skill = Skill(
            skill_id=_new_id("skill"),
            name=name,
            icon=icon,
            description=description,
            instruction=instruction,
            source_type=source_type,
            qdrant_collection=qdrant_collection,
            is_active=is_active,
            created_by=created_by,
            metadata_json=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._session_factory() as session:
            session.add(skill)
            session.commit()
            session.refresh(skill)
            session.expunge(skill)
            return skill

    def list_skills(self, *, active_only: bool = False) -> list[Skill]:
        """
        List skills, optionally filtering to active skills.

        Args:
             - active_only: bool - Whether to return only active skills.

        Returns:
             - return: list[Skill] - Skill rows.
        """
        with self._session_factory() as session:
            stmt = select(Skill).order_by(Skill.created_at.asc())
            if active_only:
                stmt = stmt.where(Skill.is_active.is_(True))
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

    def update_skill(
        self,
        *,
        skill_id: str,
        name: str = _UNSET,
        icon: str | None = _UNSET,
        description: str | None = _UNSET,
        instruction: str = _UNSET,
        is_active: bool = _UNSET,
    ) -> Skill | None:
        """
        Update a skill definition and return the row.

        Args:
             - skill_id: str - Skill id to update.
             - name: str - New skill display name.
             - icon: str | None - New optional icon label.
             - description: str | None - New skill description.
             - instruction: str - New prompt instruction.
             - is_active: bool - New active state.

        Returns:
             - return: Skill | None - Updated skill row when found, otherwise None.
        """
        with self._session_factory() as session:
            skill = session.get(Skill, skill_id)
            if skill is None:
                return None
            if name is not _UNSET:
                skill.name = name
            if icon is not _UNSET:
                skill.icon = icon
            if description is not _UNSET:
                skill.description = description
            if instruction is not _UNSET:
                skill.instruction = instruction
            if is_active is not _UNSET:
                skill.is_active = is_active
            skill.updated_at = utc_now()
            session.commit()
            session.refresh(skill)
            session.expunge(skill)
            return skill


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
        prev_assistant_message: str | None = None,
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
             - prev_assistant_message: str | None - Assistant response being reviewed.
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
            prev_assistant_message=prev_assistant_message,
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
