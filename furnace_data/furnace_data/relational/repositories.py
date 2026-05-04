"""Repository classes for BF2 relational persistence."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import String, and_, cast, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    BurdenDistributionHistory,
    Conversation,
    ConversationMessage,
    FeedbackItem,
    HopperMaterialHistory,
    LongTermMemory,
    MemoryDocument,
    MemorySummary,
    Skill,
    User,
    UserRole,
    utc_now,
)


def _new_id(prefix: str) -> str:
    """
    Return a compact application id with a stable prefix.

    Args:
         - prefix: str - Prefix that identifies the entity type.

    Returns:
         - return: str - Generated application id.
    """
    return f"{prefix}_{uuid4().hex}"


class UserRepository:
    """User/auth repository operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def seed_admin_user(self, *, password_hash: str) -> None:
        """
        Seed the default admin user when it is missing.

        Args:
             - password_hash: str - Password hash to store for the default admin user.

        Returns:
             - return: None - This function does not return a value.
        """
        with self._session_factory() as session:
            exists_stmt = select(User.username).where(User.username == "admin").limit(1)
            if session.execute(exists_stmt).first():
                return

            session.add(
                User(
                    username="admin",
                    password_hash=password_hash,
                    role=UserRole.ADMIN,
                )
            )
            session.commit()

    def add_user(self, username: str, password_hash: str, role: str) -> None:
        """
        Create a user row.

        Args:
             - username: str - Username to create.
             - password_hash: str - Password hash for the user.
             - role: str - Role value for the user.

        Returns:
             - return: None - This function does not return a value.
        """
        with self._session_factory() as session:
            session.add(
                User(
                    username=username,
                    password_hash=password_hash,
                    role=UserRole(role),
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise

    def validate_user(
        self, username: str, password_hash: str
    ) -> tuple[str, str] | None:
        """
        Validate user credentials and return identity details when valid.

        Args:
             - username: str - Username to validate.
             - password_hash: str - Password hash to compare.

        Returns:
             - return: tuple[str, str] | None - Username and role when valid, otherwise None.
        """
        with self._session_factory() as session:
            stmt = select(User.username, User.role).where(
                and_(
                    User.username == username,
                    User.password_hash == password_hash,
                )
            )
            row = session.execute(stmt).first()
            if row is None:
                return None
            return row[0], row[1].value if isinstance(row[1], UserRole) else str(row[1])


class HopperHistoryRepository:
    """SCD Type-2 repository for hopper-material mapping."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def seed_hoppers_if_missing(self, hoppers: list[str], now: datetime) -> None:
        """
        Seed missing hoppers with UNASSIGNED records.

        Args:
             - hoppers: list[str] - Hopper names that should exist.
             - now: datetime - Timestamp to use as the valid-from time.

        Returns:
             - return: None - This function does not return a value.
        """
        with self._session_factory() as session:
            existing_stmt = select(HopperMaterialHistory.hopper).distinct()
            existing = {row[0] for row in session.execute(existing_stmt).all()}
            for hopper in hoppers:
                if hopper in existing:
                    continue
                session.add(
                    HopperMaterialHistory(
                        hopper=hopper,
                        material="UNASSIGNED",
                        valid_from=now,
                    )
                )
            session.commit()

    def update_hopper_material_with_time(
        self,
        *,
        hopper: str,
        material: str,
        from_time: datetime,
        modifier: str,
        ip_address: str,
    ) -> None:
        """
        Close the current hopper-material row and insert a new active record.

        Args:
             - hopper: str - Hopper name to update.
             - material: str - New material assigned to the hopper.
             - from_time: datetime - Time from which the new assignment is valid.
             - modifier: str - User or process making the change.
             - ip_address: str - Client IP address for audit context.

        Returns:
             - return: None - This function does not return a value.
        """
        with self._session_factory() as session:
            close_stmt = (
                update(HopperMaterialHistory)
                .where(
                    and_(
                        HopperMaterialHistory.hopper == hopper,
                        HopperMaterialHistory.valid_upto.is_(None),
                    )
                )
                .values(valid_upto=from_time - timedelta(seconds=1))
            )
            session.execute(close_stmt)

            session.add(
                HopperMaterialHistory(
                    hopper=hopper,
                    material=material,
                    valid_from=from_time,
                    modifier=modifier,
                    ip_address=ip_address,
                )
            )
            session.commit()

    def get_current_hopper_materials(self) -> dict[str, str]:
        """
        Return the current hopper-to-material map.

        Args:
             - None

        Returns:
             - return: dict[str, str] - Mapping from hopper name to current material.
        """
        with self._session_factory() as session:
            stmt = (
                select(HopperMaterialHistory.hopper, HopperMaterialHistory.material)
                .where(HopperMaterialHistory.valid_upto.is_(None))
                .order_by(HopperMaterialHistory.hopper.asc())
            )
            return {row[0]: row[1] for row in session.execute(stmt).all()}

    def get_hopper_material_at(self, hopper: str, ts: datetime) -> str | None:
        """
        Return the assigned material for a hopper at a timestamp.

        Args:
             - hopper: str - Hopper name to inspect.
             - ts: datetime - Timestamp for historical lookup.

        Returns:
             - return: str | None - Assigned material when found, otherwise None.
        """
        with self._session_factory() as session:
            stmt = (
                select(HopperMaterialHistory.material)
                .where(
                    and_(
                        HopperMaterialHistory.hopper == hopper,
                        HopperMaterialHistory.valid_from <= ts,
                        or_(
                            HopperMaterialHistory.valid_upto.is_(None),
                            HopperMaterialHistory.valid_upto >= ts,
                        ),
                    )
                )
                .order_by(HopperMaterialHistory.valid_from.desc())
                .limit(1)
            )
            row = session.execute(stmt).first()
            return row[0] if row else None

    def get_hopper_material_history(self) -> list[dict[str, Any]]:
        """
        Return complete hopper material history rows.

        Args:
             - None

        Returns:
             - return: list[dict[str, Any]] - Display-ready hopper material history rows.
        """
        with self._session_factory() as session:
            stmt = select(HopperMaterialHistory).order_by(
                HopperMaterialHistory.hopper.asc(),
                HopperMaterialHistory.valid_from.desc(),
                HopperMaterialHistory.id.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": row.id,
                    "hopper": row.hopper,
                    "material": row.material,
                    "valid_from": row.valid_from,
                    "valid_upto": row.valid_upto,
                    "modifier": row.modifier,
                    "ip_address": row.ip_address,
                }
                for row in rows
            ]

    def delete_hopper_material_history(self, record_ids: list[int]) -> None:
        """
        Delete hopper history rows by ids.

        Args:
             - record_ids: list[int] - Hopper history row ids to delete.

        Returns:
             - return: None - This function does not return a value.
        """
        if not record_ids:
            return
        with self._session_factory() as session:
            stmt = delete(HopperMaterialHistory).where(
                HopperMaterialHistory.id.in_(record_ids)
            )
            session.execute(stmt)
            session.commit()


class BurdenHistoryRepository:
    """SCD Type-2 repository for burden-distribution fields."""

    TEXT_FIELDS = frozenset(
        {
            "COKE_CHARGE_PATTERN",
            "NON_COKE_CHARGE_PATTERN",
            "BURDEN_CHANGING_PURPOSE",
        }
    )

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """
        Create the repository with a SQLAlchemy session factory.

        Args:
             - session_factory: sessionmaker[Session] - Factory used to open database sessions.

        Returns:
             - return: None - This function does not return a value.
        """
        self._session_factory = session_factory

    def update_burden_field(
        self,
        *,
        field_name: str,
        value: Any,
        valid_from: datetime,
        modifier: str = "system",
        ip: str = "",
    ) -> None:
        """
        Close the active burden value and append a new value row.

        Args:
             - field_name: str - Burden field name to update.
             - value: Any - New value to store for the field.
             - valid_from: datetime - Time from which the new value is valid.
             - modifier: str - User or process making the change.
             - ip: str - Client IP address for audit context.

        Returns:
             - return: None - This function does not return a value.
        """
        end_time = valid_from - timedelta(seconds=1)
        is_text_field = field_name in self.TEXT_FIELDS

        with self._session_factory() as session:
            close_stmt = (
                update(BurdenDistributionHistory)
                .where(
                    and_(
                        BurdenDistributionHistory.field_name == field_name,
                        BurdenDistributionHistory.valid_upto.is_(None),
                    )
                )
                .values(valid_upto=end_time)
            )
            session.execute(close_stmt)

            payload = {
                "field_name": field_name,
                "valid_from": valid_from,
                "valid_upto": None,
                "modifier": modifier,
                "ip_address": ip,
                "field_value_text": str(value) if is_text_field else None,
                "field_value_float": None if is_text_field else float(value),
            }
            session.add(BurdenDistributionHistory(**payload))
            session.commit()

    def update_burden_row(
        self,
        *,
        row_values: dict[str, Any],
        timestamp: datetime,
        burden_fields: list[str],
        modifier: str = "system",
        ip: str = "",
    ) -> None:
        """
        Bulk-update applicable burden fields from one row-like mapping.

        Args:
             - row_values: dict[str, Any] - Source values keyed by field name.
             - timestamp: datetime - Time from which the new values are valid.
             - burden_fields: list[str] - Allowed burden field names to update.
             - modifier: str - User or process making the change.
             - ip: str - Client IP address for audit context.

        Returns:
             - return: None - This function does not return a value.
        """
        for field, value in row_values.items():
            if field in burden_fields and value is not None:
                self.update_burden_field(
                    field_name=field,
                    value=value,
                    valid_from=timestamp,
                    modifier=modifier,
                    ip=ip,
                )

    def get_burden_history(self) -> list[dict[str, Any]]:
        """
        Return full burden history as display-ready dictionaries.

        Args:
             - None

        Returns:
             - return: list[dict[str, Any]] - Display-ready burden history rows.
        """
        with self._session_factory() as session:
            stmt = select(BurdenDistributionHistory).order_by(
                BurdenDistributionHistory.field_name.asc(),
                BurdenDistributionHistory.valid_from.desc(),
            )
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": row.id,
                    "field_name": row.field_name,
                    "value": (
                        row.field_value_text
                        if row.field_value_text is not None
                        else row.field_value_float
                    ),
                    "valid_from": row.valid_from,
                    "valid_upto": row.valid_upto,
                    "modifier": row.modifier,
                    "ip_address": row.ip_address,
                }
                for row in rows
            ]

    def get_all_current_burden_values(self, ts: datetime) -> dict[str, Any]:
        """
        Return active burden field values at a timestamp.

        Args:
             - ts: datetime - Timestamp for historical lookup.

        Returns:
             - return: dict[str, Any] - Mapping from burden field name to active value.
        """
        with self._session_factory() as session:
            ranked_subquery = (
                select(
                    BurdenDistributionHistory.field_name.label("field_name"),
                    BurdenDistributionHistory.field_value_float.label(
                        "field_value_float"
                    ),
                    BurdenDistributionHistory.field_value_text.label(
                        "field_value_text"
                    ),
                    func.row_number()
                    .over(
                        partition_by=BurdenDistributionHistory.field_name,
                        order_by=BurdenDistributionHistory.valid_from.desc(),
                    )
                    .label("row_num"),
                )
                .where(
                    and_(
                        BurdenDistributionHistory.valid_from <= ts,
                        or_(
                            BurdenDistributionHistory.valid_upto.is_(None),
                            BurdenDistributionHistory.valid_upto >= ts,
                        ),
                    )
                )
                .subquery()
            )

            stmt = select(
                ranked_subquery.c.field_name,
                ranked_subquery.c.field_value_float,
                ranked_subquery.c.field_value_text,
            ).where(ranked_subquery.c.row_num == 1)

            rows = session.execute(stmt).all()
            return {row[0]: (row[2] if row[2] is not None else row[1]) for row in rows}

    def delete_burden_history(self, record_ids: list[int]) -> None:
        """
        Delete burden history rows by ids.

        Args:
             - record_ids: list[int] - Burden history row ids to delete.

        Returns:
             - return: None - This function does not return a value.
        """
        if not record_ids:
            return
        with self._session_factory() as session:
            stmt = delete(BurdenDistributionHistory).where(
                BurdenDistributionHistory.id.in_(record_ids)
            )
            session.execute(stmt)
            session.commit()

    def list_distribution_rows_for_window(
        self,
        *,
        start_date: date,
        end_date: date,
    ) -> list[tuple[str, str | None, datetime, datetime | None]]:
        """
        Return burden rows overlapping a date window for analytics joins.

        Args:
             - start_date: date - Start date of the lookup window.
             - end_date: date - End date of the lookup window.

        Returns:
             - return: list[tuple[str, str | None, datetime, datetime | None]] - Burden rows overlapping the window.
        """
        window_start = datetime.combine(start_date, time.min)
        window_end = datetime.combine(end_date, time.max)

        with self._session_factory() as session:
            field_value = func.coalesce(
                cast(BurdenDistributionHistory.field_value_float, String),
                BurdenDistributionHistory.field_value_text,
            ).label("field_value")

            stmt = (
                select(
                    BurdenDistributionHistory.field_name,
                    field_value,
                    BurdenDistributionHistory.valid_from,
                    BurdenDistributionHistory.valid_upto,
                )
                .where(
                    and_(
                        BurdenDistributionHistory.valid_from <= window_end,
                        or_(
                            BurdenDistributionHistory.valid_upto.is_(None),
                            BurdenDistributionHistory.valid_upto >= window_start,
                        ),
                    )
                )
                .order_by(BurdenDistributionHistory.valid_from.asc())
            )
            return list(session.execute(stmt).all())


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

    def update_skill(self, *, skill_id: str, is_active: bool) -> Skill | None:
        """
        Update a skill active flag and return the row.

        Args:
             - skill_id: str - Skill id to update.
             - is_active: bool - New active state.

        Returns:
             - return: Skill | None - Updated skill row when found, otherwise None.
        """
        with self._session_factory() as session:
            skill = session.get(Skill, skill_id)
            if skill is None:
                return None
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
