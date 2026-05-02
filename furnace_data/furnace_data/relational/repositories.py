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
    """Return a compact unique identifier with a readable prefix."""
    return f"{prefix}_{uuid4().hex}"


class UserRepository:
    """User/auth repository operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def seed_admin_user(self, *, password_hash: str) -> None:
        """Seed default admin user if missing."""
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
        """Create a user row."""
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
        """Return ``(username, role)`` when credentials are valid."""
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
        self._session_factory = session_factory

    def seed_hoppers_if_missing(self, hoppers: list[str], now: datetime) -> None:
        """Seed missing hoppers with ``UNASSIGNED`` records."""
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
        """Close current active row and insert a new hopper-material record."""
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
        """Return current hopper to material map."""
        with self._session_factory() as session:
            stmt = (
                select(HopperMaterialHistory.hopper, HopperMaterialHistory.material)
                .where(HopperMaterialHistory.valid_upto.is_(None))
                .order_by(HopperMaterialHistory.hopper.asc())
            )
            return {row[0]: row[1] for row in session.execute(stmt).all()}

    def get_hopper_material_at(self, hopper: str, ts: datetime) -> str | None:
        """Return assigned material for hopper at timestamp."""
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
        """Return complete hopper material history rows."""
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
        """Delete hopper history rows by IDs."""
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
        """Close active value and append new value row for one burden field."""
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
        """Bulk-update all applicable burden fields from one row-like mapping."""
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
        """Return full burden history as display-ready dictionaries."""
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
        """Return active burden field values at timestamp."""
        with self._session_factory() as session:
            ranked_subquery = (
                select(
                    BurdenDistributionHistory.field_name.label("field_name"),
                    BurdenDistributionHistory.field_value_float.label("field_value_float"),
                    BurdenDistributionHistory.field_value_text.label("field_value_text"),
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
        """Delete burden-history rows by IDs."""
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
        """Return burden rows overlapping [start_date, end_date] for analytics joins."""
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
    """Repository for FurnaceMind conversations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def create_conversation(
        self,
        *,
        user_id: str,
        title: str | None = None,
        model_mode: str = "medium",
    ) -> Conversation:
        """Create and return a new conversation row."""
        now = utc_now()
        conversation = Conversation(
            conversation_id=_new_id("conv"),
            user_id=user_id,
            title=title,
            model_mode=model_mode,
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
        """Return one conversation by ID, or None when missing."""
        with self._session_factory() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                return None
            session.expunge(conversation)
            return conversation

    def list_conversations(self, *, user_id: str, limit: int = 30) -> list[Conversation]:
        """List a user's conversations ordered by most recently updated."""
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
        title: str | None = None,
    ) -> None:
        """Update conversation timestamp and optionally title."""
        values: dict[str, Any] = {"updated_at": utc_now()}
        if title:
            values["title"] = title
        with self._session_factory() as session:
            session.execute(
                update(Conversation)
                .where(Conversation.conversation_id == conversation_id)
                .values(**values)
            )
            session.commit()

    def update_model_mode(self, *, conversation_id: str, model_mode: str) -> None:
        """Persist the selected reasoning effort for a conversation."""
        with self._session_factory() as session:
            session.execute(
                update(Conversation)
                .where(Conversation.conversation_id == conversation_id)
                .values(model_mode=model_mode, updated_at=utc_now())
            )
            session.commit()


class ConversationMessageRepository:
    """Repository for FurnaceMind conversation messages."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def add_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        display: str | None = None,
        model: str | None = None,
        token_count: int | None = None,
        tool_calls: list | None = None,
        metadata: dict | None = None,
    ) -> ConversationMessage:
        """Create one message and assign the next conversation sequence number."""
        with self._session_factory() as session:
            max_stmt = select(func.max(ConversationMessage.sequence_num)).where(
                ConversationMessage.conversation_id == conversation_id
            )
            sequence_num = int(session.execute(max_stmt).scalar() or 0) + 1
            message = ConversationMessage(
                message_id=_new_id("msg"),
                conversation_id=conversation_id,
                user_id=user_id,
                role=role,
                content=content,
                model=model,
                sequence_num=sequence_num,
                token_count=token_count,
                tool_calls=tool_calls,
                metadata_json={**(metadata or {}), "display": display},
                created_at=utc_now(),
            )
            session.add(message)
            session.commit()
            session.refresh(message)
            session.expunge(message)
            return message

    def list_recent_messages(
        self,
        *,
        conversation_id: str,
        limit: int = 50,
    ) -> list[ConversationMessage]:
        """Return recent messages in chronological order."""
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

    def get_message(self, message_id: str) -> ConversationMessage | None:
        """Return one message by ID, or None when it is missing."""
        with self._session_factory() as session:
            message = session.get(ConversationMessage, message_id)
            if message is None:
                return None
            session.expunge(message)
            return message

    def list_messages_after(
        self,
        *,
        conversation_id: str,
        sequence_num: int,
    ) -> list[ConversationMessage]:
        """Return messages after a sequence number in chronological order."""
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
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def create_document(
        self,
        *,
        user_id: str,
        filename: str,
        file_type: str,
        file_path: str | None = None,
        summary: str | None = None,
        qdrant_collection: str | None = None,
        qdrant_point_ids: list | None = None,
        token_estimate: int | None = None,
        metadata: dict | None = None,
    ) -> MemoryDocument:
        """Create and return metadata for an uploaded memory document."""
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
        """List memory documents for one user."""
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
        """Mark a memory document inactive."""
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
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def create_summary(
        self,
        *,
        conversation_id: str,
        user_id: str,
        summary_text: str,
        token_count: int | None = None,
        source_message_id_start: str | None = None,
        source_message_id_end: str | None = None,
        metadata: dict | None = None,
    ) -> MemorySummary:
        """Create and return a conversation memory summary."""
        summary = MemorySummary(
            summary_id=_new_id("sum"),
            conversation_id=conversation_id,
            user_id=user_id,
            summary_text=summary_text,
            source_message_id_start=source_message_id_start,
            source_message_id_end=source_message_id_end,
            token_count=token_count,
            metadata_json=metadata or {},
            created_at=utc_now(),
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
        """List summaries for one conversation, newest first."""
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
    """Repository for durable FurnaceMind long-term memories."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
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
        """Create and return one long-term memory row."""
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

    def list_memories(self, *, user_id: str) -> list[LongTermMemory]:
        """List long-term memories for one user."""
        with self._session_factory() as session:
            stmt = select(LongTermMemory).where(LongTermMemory.user_id == user_id)
            stmt = stmt.order_by(LongTermMemory.created_at.desc())
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows


class SkillRepository:
    """Repository for built-in and uploaded FurnaceMind skills."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def upsert_skill(
        self,
        *,
        name: str,
        instruction: str,
        description: str | None = None,
        source_type: str = "custom",
        created_by: str | None = None,
        metadata: dict | None = None,
    ) -> Skill:
        """Create or update a skill by name/source and return the row."""
        with self._session_factory() as session:
            skill = session.execute(
                select(Skill)
                .where(
                    Skill.name == name,
                    Skill.source_type == source_type,
                    Skill.created_by.is_(created_by) if created_by is None else Skill.created_by == created_by,
                )
                .limit(1)
            ).scalar_one_or_none()
            now = utc_now()
            if skill is None:
                skill = Skill(
                    skill_id=_new_id("skill"),
                    name=name,
                    instruction=instruction,
                    description=description,
                    source_type=source_type,
                    created_by=created_by,
                    metadata_json=metadata or {},
                    created_at=now,
                    updated_at=now,
                )
                session.add(skill)
            else:
                skill.name = name
                skill.instruction = instruction
                skill.description = description
                skill.source_type = source_type
                skill.created_by = created_by
                skill.metadata_json = metadata or {}
                skill.updated_at = now
            session.commit()
            session.refresh(skill)
            session.expunge(skill)
            return skill

    def create_skill(
        self,
        *,
        name: str,
        instruction: str,
        description: str | None = None,
        source_type: str = "custom",
        created_by: str | None = None,
        metadata: dict | None = None,
    ) -> Skill:
        """Create and return a new skill row."""
        skill = Skill(
            skill_id=_new_id("skill"),
            name=name,
            instruction=instruction,
            description=description,
            source_type=source_type,
            created_by=created_by,
            metadata_json=metadata or {},
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        with self._session_factory() as session:
            session.add(skill)
            session.commit()
            session.refresh(skill)
            session.expunge(skill)
            return skill

    def list_skills(self, *, active_only: bool = False) -> list[Skill]:
        """List skills with an optional active filter."""
        with self._session_factory() as session:
            stmt = select(Skill)
            if active_only:
                stmt = stmt.where(Skill.is_active.is_(True))
            stmt = stmt.order_by(Skill.name.asc())
            rows = list(session.execute(stmt).scalars().all())
            for row in rows:
                session.expunge(row)
            return rows

    def update_skill(self, *, skill_id: str, is_active: bool) -> Skill:
        """Toggle a skill's active flag and return the row."""
        with self._session_factory() as session:
            skill = session.get(Skill, skill_id)
            if skill is None:
                raise ValueError(f"Skill {skill_id} not found.")
            skill.is_active = is_active
            skill.updated_at = utc_now()
            session.commit()
            session.refresh(skill)
            session.expunge(skill)
            return skill


class FeedbackItemRepository:
    """Repository for FurnaceMind response feedback and lessons."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create the repository with a SQLAlchemy session factory."""
        self._session_factory = session_factory

    def add_feedback(
        self,
        *,
        user_id: str,
        source: str,
        polarity: str,
        message_id: str,
        conversation_id: str,
        feedback_text: str | None = None,
        raw_user_message: str | None = None,
        prev_assistant_message: str | None = None,
        snapshot: dict | None = None,
        metadata: dict | None = None,
    ) -> FeedbackItem:
        """Create feedback or return the existing unique row."""
        with self._session_factory() as session:
            if message_id:
                existing = session.execute(
                    select(FeedbackItem).where(
                        FeedbackItem.message_id == message_id,
                        FeedbackItem.user_id == user_id,
                        FeedbackItem.source == source,
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    session.expunge(existing)
                    return existing
            now = utc_now()
            feedback = FeedbackItem(
                feedback_id=_new_id("fb"),
                message_id=message_id,
                conversation_id=conversation_id,
                user_id=user_id,
                source=source,
                polarity=polarity,
                feedback_text=feedback_text,
                raw_user_message=raw_user_message,
                prev_assistant_message=prev_assistant_message,
                snapshot=snapshot or {},
                metadata_json=metadata or {},
                created_at=now,
            )
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
            session.expunge(feedback)
            return feedback

    def get_feedback(self, *, message_id: str, user_id: str) -> FeedbackItem | None:
        """Return saved feedback for a message and user."""
        with self._session_factory() as session:
            row = session.execute(
                select(FeedbackItem)
                .where(FeedbackItem.message_id == message_id, FeedbackItem.user_id == user_id)
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            session.expunge(row)
            return row

    def list_pending_lessons(self, *, limit: int = 10) -> list[FeedbackItem]:
        """Return feedback rows that still need lesson extraction."""
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
        mem0_memory_id: str | None = None,
    ) -> None:
        """Save the extracted lesson for a feedback row."""
        with self._session_factory() as session:
            session.execute(
                update(FeedbackItem)
                .where(FeedbackItem.feedback_id == feedback_id)
                .values(
                    extracted_lesson=lesson,
                    lesson_extracted=True,
                    mem0_memory_id=mem0_memory_id,
                )
            )
            session.commit()
