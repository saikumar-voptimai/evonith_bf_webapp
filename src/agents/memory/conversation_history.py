"""PostgreSQL-backed FurnaceMind conversation, skill, and feedback helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

_FURNACE_DATA_SRC = Path(__file__).resolve().parents[3] / "furnace_data"
if str(_FURNACE_DATA_SRC) not in sys.path:
    sys.path.insert(0, str(_FURNACE_DATA_SRC))
_loaded_furnace_data = sys.modules.get("furnace_data")
_loaded_path = Path(getattr(_loaded_furnace_data, "__file__", "") or "")
if _loaded_furnace_data is not None and _FURNACE_DATA_SRC not in _loaded_path.parents:
    for _module_name in list(sys.modules):
        if _module_name == "furnace_data" or _module_name.startswith("furnace_data."):
            sys.modules.pop(_module_name, None)

from furnace_data.relational import (
    ConversationMessageRepository,
    ConversationRepository,
    FeedbackItemRepository,
    LongTermMemoryRepository,
    MemoryDocumentRepository,
    MemorySummaryRepository,
    SkillRepository,
    build_relational_engine,
    build_relational_session_factory,
)

from agents.memory.context_budget import estimate_text_tokens
from agents.memory.feedback_learning import generate_feedback_lesson
from agents.memory.long_term_memory import extract_long_term_memories
from agents.memory.semantic_memory import SemanticMemoryStore

_SUMMARY_KEEP_MESSAGES = 6


DEFAULT_SKILLS: tuple[dict[str, Any], ...] = (
    {
        "name": "Unit Cost",
        "description": "Analyse unit-cost gap and operating levers.",
        "instruction": "Run the built-in Unit Cost optimisation skill.",
        "source_type": "built_in",
        "metadata": {"button_order": 10, "handler": "optimise", "icon": "💰"},
    },
    {
        "name": "Shift to Best",
        "description": "Compare selected shift to best-shift targets.",
        "instruction": "Run the built-in Shift to Best skill.",
        "source_type": "built_in",
        "metadata": {"button_order": 20, "handler": "shift_to_best", "icon": "🎯"},
    },
    {
        "name": "Heatloads",
        "description": "Check recent furnace heatload behavior.",
        "instruction": "Run the built-in Heatloads skill.",
        "source_type": "built_in",
        "metadata": {"button_order": 30, "handler": "heatload", "icon": "🌡️"},
    },
)


class ConversationHistoryStore:
    """Facade over FurnaceMind PostgreSQL repositories."""

    def __init__(self) -> None:
        """Create repository instances using ``DATABASE_URL``."""
        self.engine = build_relational_engine()
        self.session_factory = build_relational_session_factory(self.engine)
        self.conversations = ConversationRepository(self.session_factory)
        self.messages = ConversationMessageRepository(self.session_factory)
        self.documents = MemoryDocumentRepository(self.session_factory)
        self.summaries = MemorySummaryRepository(self.session_factory)
        self.long_term_memories = LongTermMemoryRepository(self.session_factory)
        self.skills = SkillRepository(self.session_factory)
        self.feedback = FeedbackItemRepository(self.session_factory)

    def dispose(self) -> None:
        """Dispose the SQLAlchemy engine."""
        self.engine.dispose()

    def create_conversation(self, *, user_id: str, title: str | None = None) -> str:
        """Create a conversation and return its ID."""
        conversation = self.conversations.create_conversation(
            user_id=user_id,
            title=title or "FurnaceMind chat",
        )
        return conversation.conversation_id

    def get_or_create_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        title: str | None = None,
    ) -> str:
        """Return a valid conversation ID, creating one when needed."""
        if conversation_id:
            conversation = self.conversations.get_conversation(conversation_id)
            if conversation is not None and conversation.user_id == user_id:
                return conversation_id
        return self.create_conversation(user_id=user_id, title=title)

    def list_conversations(self, *, user_id: str) -> list[dict[str, Any]]:
        """Return recent conversations for one user."""
        return [
            {
                "conversation_id": row.conversation_id,
                "user_id": row.user_id,
                "title": row.title or "FurnaceMind chat",
                "model_mode": row.model_mode or "medium",
                "updated_at": row.updated_at,
            }
            for row in self.conversations.list_conversations(user_id=user_id)
        ]

    def get_conversation_model_mode(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> str:
        """Return the saved reasoning effort for a conversation."""
        conversation = self.conversations.get_conversation(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return "medium"
        return conversation.model_mode or "medium"

    def update_conversation_model_mode(
        self,
        *,
        conversation_id: str,
        user_id: str,
        model_mode: str,
    ) -> None:
        """Persist the selected reasoning effort for a conversation."""
        conversation = self.conversations.get_conversation(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return
        self.conversations.update_model_mode(
            conversation_id=conversation_id,
            model_mode=model_mode,
        )

    def add_user_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        content: str,
        display: str | None = None,
    ) -> str:
        """Persist a user message and return its ID."""
        message = self.messages.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=content,
            display=display,
            token_count=estimate_text_tokens(content),
        )
        self.conversations.touch_conversation(
            conversation_id=conversation_id,
            title=_title_from_message(content),
        )
        return message.message_id

    def add_assistant_message(
        self,
        *,
        conversation_id: str,
        user_id: str,
        content: str,
        model: str | None,
        metadata: dict | None = None,
    ) -> str:
        """Persist an assistant message and return its ID."""
        message = self.messages.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=content,
            model=model,
            token_count=estimate_text_tokens(content),
            metadata=metadata or {},
        )
        self.conversations.touch_conversation(conversation_id=conversation_id)
        return message.message_id

    def load_chat_history(self, *, conversation_id: str) -> list[dict[str, Any]]:
        """Load recent chat messages as Streamlit-compatible dictionaries."""
        rows = self.messages.list_recent_messages(conversation_id=conversation_id)
        summaries = self.summaries.list_summaries(conversation_id=conversation_id, limit=1)
        summary = summaries[0] if summaries else None
        summary_item: dict[str, Any] | None = None
        if summary is not None:
            summary_item = {
                "role": "assistant",
                "content": "CONVERSATION MEMORY SUMMARY:\n" + summary.summary_text,
                "display": "_Older conversation compressed into memory._",
                "message_id": None,
                "conversation_id": conversation_id,
                "compressed": True,
                "knowledge_sources": [],
            }
            end_message = (
                self.messages.get_message(summary.source_message_id_end)
                if summary.source_message_id_end
                else None
            )
            if end_message is not None:
                rows = self.messages.list_messages_after(
                    conversation_id=conversation_id,
                    sequence_num=end_message.sequence_num,
                )
        history: list[dict[str, Any]] = []
        if summary_item is not None:
            history.append(summary_item)
        for row in rows:
            metadata = row.metadata_json or {}
            history.append(
                {
                    "role": row.role,
                    "content": row.content,
                    "display": metadata.get("display") or row.content,
                    "message_id": row.message_id,
                    "conversation_id": row.conversation_id,
                    "knowledge_sources": metadata.get("knowledge_sources", []),
                }
            )
        return history

    def create_document(
        self,
        *,
        user_id: str,
        filename: str,
        file_type: str,
        file_path: str | None,
        summary: str | None,
        qdrant_collection: str | None,
        qdrant_point_ids: list | None,
        token_estimate: int | None,
        metadata: dict | None = None,
    ) -> str:
        """Persist uploaded memory document metadata and return its ID."""
        document = self.documents.create_document(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_path=file_path,
            summary=summary,
            qdrant_collection=qdrant_collection,
            qdrant_point_ids=qdrant_point_ids or [],
            token_estimate=token_estimate,
            metadata=metadata or {},
        )
        return document.document_id

    def create_long_term_memory(
        self,
        *,
        user_id: str,
        memory: str,
        qdrant_point_id: str,
        conversation_id: str | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        qdrant_collection: str = "furnacemind_long_term_memories",
    ) -> str:
        """Persist one extracted long-term memory in SQL metadata."""
        memory_row = self.long_term_memories.create_memory(
            user_id=user_id,
            memory_text=memory,
            qdrant_collection=qdrant_collection,
            qdrant_point_id=qdrant_point_id,
            source_conversation_id=conversation_id,
            source_user_message_id=user_message_id,
            source_assistant_message_id=assistant_message_id,
            token_estimate=estimate_text_tokens(memory),
            metadata={
                "source": "chat_turn",
            },
        )
        return memory_row.memory_id

    def list_documents(self, *, user_id: str) -> list[dict[str, Any]]:
        """Return active uploaded memory documents for one user."""
        return [_document_to_dict(row) for row in self.documents.list_documents(user_id=user_id)]

    def deactivate_document(self, *, document_id: str) -> None:
        """Deactivate one uploaded memory document."""
        self.documents.deactivate_document(document_id)

    def seed_default_skills(self) -> int:
        """Ensure built-in skills exist and return how many were inserted."""
        existing = {
            (skill.name, skill.source_type, skill.created_by)
            for skill in self.skills.list_skills()
        }
        count = 0
        for skill in DEFAULT_SKILLS:
            self.skills.upsert_skill(**skill)
            if (skill["name"], skill["source_type"], skill.get("created_by")) not in existing:
                count += 1
        return count

    def list_skills(self) -> list[dict[str, Any]]:
        """Return all skill rows as dictionaries."""
        return [_skill_to_dict(row) for row in self.skills.list_skills()]

    def list_button_skills(self) -> list[dict[str, Any]]:
        """Return active skills that should render as quick-action buttons."""
        skills = [_skill_to_dict(row) for row in self.skills.list_skills(active_only=True)]
        return sorted(
            skills,
            key=lambda item: (
                int((item.get("metadata") or {}).get("button_order", 999)),
                item["name"],
            ),
        )

    def update_skill(self, *, skill_id: str, is_active: bool) -> None:
        """Activate or deactivate a skill."""
        self.skills.update_skill(skill_id=skill_id, is_active=is_active)

    def create_uploaded_skill(
        self,
        *,
        user_id: str,
        filename: str,
        file_type: str,
        instruction: str,
    ) -> str:
        """Create a skill from an uploaded instruction document."""
        name = _skill_name_from_filename(filename)
        skill = self.skills.create_skill(
            name=name,
            description=f"Uploaded from {filename}.",
            instruction=instruction.strip(),
            source_type="uploaded",
            created_by=user_id,
            metadata={
                "button_order": 100,
                "handler": "prompt",
                "source_filename": filename,
                "file_type": file_type,
            },
        )
        return skill.skill_id

    def add_feedback(
        self,
        *,
        user_id: str,
        source: str,
        polarity: str,
        message_id: str,
        conversation_id: str,
        feedback_text: str | None,
        raw_user_message: str | None,
        prev_assistant_message: str | None,
        snapshot: dict | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Persist a feedback item and return its ID."""
        feedback = self.feedback.add_feedback(
            user_id=user_id,
            source=source,
            polarity=polarity,
            message_id=message_id,
            conversation_id=conversation_id,
            feedback_text=feedback_text,
            raw_user_message=raw_user_message,
            prev_assistant_message=prev_assistant_message,
            snapshot=snapshot or {},
            metadata=metadata or {},
        )
        return feedback.feedback_id

    def compress_conversation(
        self,
        *,
        conversation_id: str,
        user_id: str,
        llm_client: Any | None,
        keep_last_messages: int = _SUMMARY_KEEP_MESSAGES,
    ) -> str | None:
        """Summarize older conversation messages and persist the memory summary."""
        conversation = self.conversations.get_conversation(conversation_id)
        if conversation is None or conversation.user_id != user_id:
            return None
        rows = self.messages.list_recent_messages(conversation_id=conversation_id, limit=200)
        rows = [row for row in rows if row.role in {"user", "assistant"}]
        if len(rows) <= keep_last_messages:
            return None
        older_rows = rows[:-keep_last_messages]
        if not older_rows:
            return None
        prompt = _build_compression_prompt(older_rows)
        summary_text = ""
        if llm_client is not None:
            try:
                summary_text = llm_client.generate(
                    "You compress industrial assistant chats into concise working memory.",
                    prompt,
                ).strip()
            except Exception:
                summary_text = ""
        if not summary_text:
            summary_text = _fallback_summary(older_rows)
        self.summaries.create_summary(
            conversation_id=conversation_id,
            user_id=user_id,
            summary_text=summary_text,
            token_count=estimate_text_tokens(summary_text),
            source_message_id_start=older_rows[0].message_id,
            source_message_id_end=older_rows[-1].message_id,
            metadata={"compressed_message_count": len(older_rows)},
        )
        return summary_text

    def get_feedback(self, *, message_id: str, user_id: str) -> dict[str, Any] | None:
        """Return feedback for a message when it exists."""
        row = self.feedback.get_feedback(message_id=message_id, user_id=user_id)
        return None if row is None else _feedback_to_dict(row)

    def process_pending_feedback_lessons(
        self,
        *,
        semantic_memory_store: SemanticMemoryStore | None,
        lesson_llm_client: Any,
        limit: int = 5,
    ) -> int:
        """Generate lessons for pending feedback and store them in Qdrant."""
        if semantic_memory_store is None:
            return 0
        processed = 0
        for feedback in self.feedback.list_pending_lessons(limit=limit):
            lesson = generate_feedback_lesson(feedback, lesson_llm_client)
            point_id = semantic_memory_store.add_lesson(
                user_id=feedback.user_id,
                lesson=lesson,
                metadata={
                    "feedback_id": feedback.feedback_id,
                    "conversation_id": feedback.conversation_id,
                    "message_id": feedback.message_id,
                    "polarity": feedback.polarity,
                },
            )
            self.feedback.mark_lesson_extracted(
                feedback_id=feedback.feedback_id,
                lesson=lesson,
                mem0_memory_id=point_id,
            )
            processed += 1
        return processed

    def store_turn_long_term_memories(
        self,
        *,
        user_id: str,
        user_text: str,
        assistant_text: str,
        semantic_memory_store: SemanticMemoryStore | None,
        memory_llm_client: Any,
        conversation_id: str | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
    ) -> int:
        """Extract durable memories from a chat turn and store them."""
        if semantic_memory_store is None:
            return 0
        memories = extract_long_term_memories(
            user_text=user_text,
            assistant_text=assistant_text,
            llm_client=memory_llm_client,
        )
        existing = {
            " ".join(str(item.memory_text or "").lower().split())
            for item in self.long_term_memories.list_memories(user_id=user_id)
        }
        stored = 0
        for memory in memories:
            normalized = " ".join(memory.lower().split())
            if normalized in existing:
                continue
            point_id = semantic_memory_store.add_long_term_memory(
                user_id=user_id,
                memory=memory,
                metadata={
                    "conversation_id": conversation_id,
                    "user_message_id": user_message_id,
                    "assistant_message_id": assistant_message_id,
                },
            )
            self.create_long_term_memory(
                user_id=user_id,
                memory=memory,
                qdrant_point_id=point_id,
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                qdrant_collection=semantic_memory_store.long_term_collection_name,
            )
            stored += 1
            existing.add(normalized)
        return stored


def _title_from_message(content: str) -> str:
    """Return a compact conversation title from a user message."""
    clean = " ".join(content.strip().split())
    return clean[:60] or "FurnaceMind chat"


def _build_compression_prompt(messages: list[Any]) -> str:
    """Build the prompt used to compress older chat turns."""
    lines = [
        "Summarize these older FurnaceMind turns for future context.",
        "Keep operator intent, decisions, constraints, useful facts, and unresolved tasks.",
        "Do not include filler. Use 5 to 10 bullets.",
        "",
    ]
    for message in messages:
        role = str(message.role).upper()
        content = " ".join(str(message.content or "").split())
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _fallback_summary(messages: list[Any]) -> str:
    """Create a deterministic compression summary when the LLM is unavailable."""
    snippets: list[str] = []
    for message in messages[-8:]:
        content = " ".join(str(message.content or "").split())
        if content:
            snippets.append(f"- {message.role}: {content[:220]}")
    return "\n".join(snippets) or "Older conversation was compressed."


def _skill_name_from_filename(filename: str) -> str:
    """Return a readable skill name from a filename."""
    stem = Path(filename).stem
    clean = " ".join(stem.replace("_", " ").replace("-", " ").split())
    return clean.title() or "Uploaded Skill"


def _skill_to_dict(skill: Any) -> dict[str, Any]:
    """Convert a skill ORM row into a plain dictionary."""
    return {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "description": skill.description,
        "instruction": skill.instruction,
        "source_type": skill.source_type,
        "created_by": skill.created_by,
        "is_active": skill.is_active,
        "button_label": _skill_button_label(skill),
        "metadata": skill.metadata_json or {},
    }


def _skill_button_label(skill: Any) -> str:
    """Return a display label for a skill without storing UI columns."""
    metadata = skill.metadata_json or {}
    icon = str(metadata.get("icon") or "").strip()
    return f"{icon} {skill.name}".strip()


def _document_to_dict(document: Any) -> dict[str, Any]:
    """Convert a document ORM row into a plain dictionary."""
    return {
        "document_id": document.document_id,
        "filename": document.filename,
        "file_type": document.file_type,
        "file_path": document.file_path,
        "summary": document.summary,
        "qdrant_collection": document.qdrant_collection,
        "qdrant_point_ids": document.qdrant_point_ids or [],
        "token_estimate": document.token_estimate,
        "created_at": document.created_at,
        "metadata": document.metadata_json or {},
    }


def _feedback_to_dict(feedback: Any) -> dict[str, Any]:
    """Convert a feedback ORM row into a plain dictionary."""
    return {
        "feedback_id": feedback.feedback_id,
        "message_id": feedback.message_id,
        "polarity": feedback.polarity,
        "feedback_text": feedback.feedback_text,
        "lesson_extracted": feedback.lesson_extracted,
    }
