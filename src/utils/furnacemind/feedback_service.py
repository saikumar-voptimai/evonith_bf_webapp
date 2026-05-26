"""Feedback persistence, lesson generation, and retrieval for FurnaceMind.

This module supports two feedback paths:

1. Explicit feedback from the thumbs controls shown below assistant answers.
2. Chat-based feedback where the user corrects an answer in the next message.

Both paths store a feedback row in PostgreSQL, generate a reusable lesson with
the configured LLM, save that lesson back to SQL, and upsert it into a Qdrant
collection so future user questions can retrieve relevant lessons by semantic
similarity.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from agents.furnacemind import prompts
from furnace_data import relational
from utils.settings import settings

_DEFAULT_FEEDBACK_COLLECTION = "furnacemind_feedback_lessons"
_EXPLICIT_SOURCE = "explicit"
_CHAT_SOURCE = "chat_correction"


def latest_assistant_exchange(
    chat_history: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Find the latest assistant answer and the user message that caused it.

    Feedback must be anchored to the answer being reviewed and the original user
    request. This helper walks the in-session chat history backwards, skips
    non-text artifacts, and returns the last assistant text message plus the
    closest earlier user text message.

    Args:
         - chat_history: list[dict[str, Any]] - Streamlit chat-history items.

    Returns:
         - return: tuple[dict[str, Any] | None, dict[str, Any] | None] - Previous
           user message and latest assistant message when both are available.
    """
    assistant_item: dict[str, Any] | None = None
    assistant_index: int | None = None

    for index in range(len(chat_history) - 1, -1, -1):
        item = chat_history[index]
        if item.get("type", "text") != "text":
            continue
        if item.get("role") == "assistant" and str(item.get("content") or "").strip():
            assistant_item = item
            assistant_index = index
            break

    if assistant_item is None or assistant_index is None:
        return None, None

    for index in range(assistant_index - 1, -1, -1):
        item = chat_history[index]
        if item.get("type", "text") != "text":
            continue
        if item.get("role") == "user" and str(item.get("content") or "").strip():
            return item, assistant_item

    return None, assistant_item


def _json_from_llm(text: str) -> dict[str, Any]:
    """
    Parse a JSON object returned by a feedback helper prompt.

    The prompts request JSON-only output, but some models still wrap JSON in a
    small amount of prose or fenced code. This parser trims common wrappers and
    extracts the first object-shaped section before calling ``json.loads``.

    Args:
         - text: str - Raw LLM output.

    Returns:
         - return: dict[str, Any] - Parsed JSON object, or an empty dictionary.
    """
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _point_id(feedback_id: str) -> str:
    """
    Convert a feedback id into a stable Qdrant point id.

    Qdrant accepts UUID strings for point ids. Feedback ids are application
    strings, so this helper creates a deterministic UUID from each feedback id.

    Args:
         - feedback_id: str - PostgreSQL feedback id.

    Returns:
         - return: str - Deterministic UUID string for Qdrant.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, feedback_id))


class FeedbackLessonVectorStore:
    """Qdrant adapter for reusable FurnaceMind feedback lessons."""

    def __init__(self, embedding_client: Any) -> None:
        """
        Connect to Qdrant and prepare the feedback lesson collection.

        The feedback lesson collection uses the same Qdrant endpoint and
        embedding dimension as the knowledge collection, but keeps lessons in a
        separate collection so retrieval can be filtered and reasoned about
        independently.

        Args:
             - embedding_client: Any - Cloud embedding client with ``embed_text``.

        Returns:
             - return: None - This function does not return a value.
        """
        qcfg = settings.qdrant_knowledge
        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )
        self.embedding = embedding_client
        self.embedding_dim = embedding_client.dimension
        self.collection_name = os.getenv(
            "FEEDBACK_QDRANT_COLLECTION",
            _DEFAULT_FEEDBACK_COLLECTION,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """
        Create or validate the feedback lesson Qdrant collection.

        Args:
             - None

        Returns:
             - return: None - This function does not return a value.
        """
        collections = self.client.get_collections().collections
        existing = {collection.name for collection in collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            return

        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        if hasattr(vectors, "size") and vectors.size != self.embedding_dim:
            raise RuntimeError(
                f"Vector dimension mismatch for collection '{self.collection_name}': "
                f"expected {self.embedding_dim}, got {vectors.size}"
            )

    def upsert_lesson(
        self,
        *,
        feedback_id: str,
        user_id: str,
        lesson: str,
        metadata: dict[str, Any],
    ) -> str:
        """
        Embed and upsert one reusable feedback lesson.

        Args:
             - feedback_id: str - PostgreSQL feedback id.
             - user_id: str - User that owns the lesson.
             - lesson: str - Reusable lesson text.
             - metadata: dict[str, Any] - Extra payload metadata.

        Returns:
             - return: str - Qdrant point id used for the lesson.
        """
        point_id = _point_id(feedback_id)
        embedding = self.embedding.embed_text(lesson)
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match "
                f"expected {self.embedding_dim}"
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        **metadata,
                        "user_id": user_id,
                        "feedback_id": feedback_id,
                        "lesson": lesson,
                    },
                )
            ],
            wait=True,
        )
        return point_id

    def search_lessons(
        self,
        *,
        query: str,
        user_id: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve user-specific feedback lessons relevant to a query.

        Args:
             - query: str - Current user question.
             - user_id: str - User whose lessons should be retrieved.
             - top_k: int - Maximum lessons to return.

        Returns:
             - return: list[dict[str, Any]] - Matching Qdrant payloads and scores.
        """
        query_embedding = self.embedding.embed_text(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [
            {"score": point.score, "payload": point.payload or {}}
            for point in results.points
        ]


class FurnaceMindFeedbackService:
    """Service for detecting, storing, and retrieving FurnaceMind feedback."""

    def __init__(self, *, embedding_client: Any | None = None) -> None:
        """
        Build the SQL repository and optional Qdrant lesson store.

        SQL feedback storage is required. Qdrant lesson storage is optional so
        FurnaceMind can still capture feedback when the vector database or
        embedding provider is temporarily unavailable.

        Args:
             - embedding_client: Any | None - Optional cloud embedding client.

        Returns:
             - return: None - This function does not return a value.
        """
        engine = relational.build_relational_engine()
        session_factory = relational.build_relational_session_factory(engine)
        self._feedback = relational.FeedbackItemRepository(session_factory)
        self._lesson_store: FeedbackLessonVectorStore | None = None
        if embedding_client is not None:
            try:
                self._lesson_store = FeedbackLessonVectorStore(embedding_client)
            except Exception:
                self._lesson_store = None

    def detect_chat_feedback(
        self,
        *,
        user_message: str,
        raw_user_message: str,
        assistant_response: str,
        llm: Any,
    ) -> dict[str, Any] | None:
        """
        Detect whether a user message is feedback on the previous answer.

        Args:
             - user_message: str - Latest message typed by the user.
             - raw_user_message: str - Original question before the assistant answer.
             - assistant_response: str - Assistant answer being evaluated.
             - llm: Any - LLM client used for classification.

        Returns:
             - return: dict[str, Any] | None - Feedback details when detected.
        """
        if not user_message or not raw_user_message or not assistant_response:
            return None

        detector_prompt = (
            "Original user question:\n"
            f"{raw_user_message}\n\n"
            "Assistant answer:\n"
            f"{assistant_response}\n\n"
            "Latest user message:\n"
            f"{user_message}"
        )
        try:
            result = llm.generate(
                system_prompt=prompts.FEEDBACK_DETECTION_SYSTEM_PROMPT,
                user_prompt=detector_prompt,
            )
        except Exception:
            return None

        data = _json_from_llm(result)
        if data.get("is_feedback") is not True:
            return None

        feedback_text = str(data.get("feedback_text") or user_message).strip()
        polarity = str(data.get("polarity") or "negative").strip().lower()
        if polarity not in {"positive", "negative"}:
            polarity = "negative"
        return {"polarity": polarity, "feedback_text": feedback_text}

    def save_feedback(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
        source: str,
        polarity: str,
        feedback_text: str | None,
        raw_user_message: str,
        assistant_response: str,
        llm: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Store feedback, generate a reusable lesson, and index the lesson.

        Args:
             - user_id: str - User that submitted the feedback.
             - conversation_id: str - Conversation where feedback was submitted.
             - message_id: str - Assistant message receiving feedback.
             - source: str - Feedback source, such as explicit or chat correction.
             - polarity: str - Feedback polarity.
             - feedback_text: str | None - Optional user feedback text.
             - raw_user_message: str - Original user question.
             - assistant_response: str - Assistant answer being reviewed.
             - llm: Any - LLM client used to generate the reusable lesson.
             - metadata: dict[str, Any] | None - Additional metadata.

        Returns:
             - return: str | None - Feedback id when saved.
        """
        feedback = self._feedback.add_feedback(
            user_id=user_id,
            message_id=message_id,
            conversation_id=conversation_id,
            source=source,
            polarity=polarity,
            feedback_text=feedback_text,
            raw_user_message=raw_user_message,
            assistant_response=assistant_response,
            metadata=metadata or {},
        )

        lesson = self._generate_lesson(
            polarity=polarity,
            feedback_text=feedback_text,
            raw_user_message=raw_user_message,
            assistant_response=assistant_response,
            llm=llm,
        )
        if not lesson:
            return feedback.feedback_id

        qdrant_point_id: str | None = None
        qdrant_collection: str | None = None
        if self._lesson_store is not None:
            try:
                qdrant_point_id = self._lesson_store.upsert_lesson(
                    feedback_id=feedback.feedback_id,
                    user_id=user_id,
                    lesson=lesson,
                    metadata={
                        "source": source,
                        "polarity": polarity,
                        "conversation_id": conversation_id,
                        "message_id": message_id,
                    },
                )
                qdrant_collection = self._lesson_store.collection_name
            except Exception:
                qdrant_point_id = None
                qdrant_collection = None

        self._feedback.mark_lesson_extracted(
            feedback_id=feedback.feedback_id,
            lesson=lesson,
            qdrant_collection=qdrant_collection,
            qdrant_point_id=qdrant_point_id,
        )
        return feedback.feedback_id

    def save_explicit_feedback(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
        raw_user_message: str,
        assistant_response: str,
        polarity: str,
        feedback_text: str,
        llm: Any,
    ) -> str | None:
        """
        Save thumbs feedback from the assistant response UI.

        Args:
             - user_id: str - User that clicked the feedback control.
             - conversation_id: str - Conversation where feedback was submitted.
             - message_id: str - Assistant message receiving feedback.
             - raw_user_message: str - User question that caused the answer.
             - assistant_response: str - Assistant answer being reviewed.
             - polarity: str - ``positive`` for thumbs up or ``negative`` for thumbs down.
             - feedback_text: str - User-written feedback details.
             - llm: Any - LLM client used to generate the reusable lesson.

        Returns:
             - return: str | None - Feedback id when saved.
        """
        feedback_text = feedback_text.strip()
        if not feedback_text:
            feedback_text = "Looks good" if polarity == "positive" else "Needs work"
        return self.save_feedback(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            source=_EXPLICIT_SOURCE,
            polarity=polarity,
            feedback_text=feedback_text,
            raw_user_message=raw_user_message,
            assistant_response=assistant_response,
            llm=llm,
            metadata={"feedback_type": "thumbs"},
        )

    def save_chat_feedback(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message_id: str,
        raw_user_message: str,
        assistant_response: str,
        feedback_text: str,
        polarity: str,
        llm: Any,
    ) -> str | None:
        """
        Save correction feedback detected from a normal chat message.

        Args:
             - user_id: str - User whose correction was detected.
             - conversation_id: str - Conversation where correction happened.
             - message_id: str - Assistant message receiving feedback.
             - raw_user_message: str - User question that caused the answer.
             - assistant_response: str - Assistant answer being corrected.
             - feedback_text: str - User correction text.
             - polarity: str - Detected feedback polarity.
             - llm: Any - LLM client used to generate the reusable lesson.

        Returns:
             - return: str | None - Feedback id when saved.
        """
        return self.save_feedback(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            source=_CHAT_SOURCE,
            polarity=polarity,
            feedback_text=feedback_text,
            raw_user_message=raw_user_message,
            assistant_response=assistant_response,
            llm=llm,
            metadata={"feedback_type": "chat_correction"},
        )

    def feedback_context(
        self,
        *,
        query: str,
        user_id: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve relevant feedback lessons and format them for prompt injection.

        Args:
             - query: str - Current user question.
             - user_id: str - User whose lessons should be searched.
             - top_k: int - Maximum lessons to inject.

        Returns:
             - return: str - Prompt-ready feedback lesson context.
        """
        if self._lesson_store is None or not query.strip():
            return ""
        try:
            matches = self._lesson_store.search_lessons(
                query=query,
                user_id=user_id,
                top_k=top_k,
            )
        except Exception:
            return ""

        lessons: list[str] = []
        for match in matches:
            payload = match.get("payload") or {}
            lesson = str(payload.get("lesson") or "").strip()
            if lesson:
                lessons.append(lesson)

        if not lessons:
            return ""

        numbered = "\n".join(
            f"{index}. {lesson}" for index, lesson in enumerate(lessons, 1)
        )
        return (
            "RELEVANT USER FEEDBACK LESSONS:\n"
            "Use these lessons to avoid repeating previous answer mistakes.\n"
            f"{numbered}"
        )

    def _generate_lesson(
        self,
        *,
        polarity: str,
        feedback_text: str | None,
        raw_user_message: str,
        assistant_response: str,
        llm: Any,
    ) -> str:
        """
        Generate a reusable lesson from one feedback item.

        Args:
             - polarity: str - Feedback polarity.
             - feedback_text: str | None - User feedback or correction.
             - raw_user_message: str - Original user question.
             - assistant_response: str - Assistant answer being reviewed.
             - llm: Any - LLM client used for lesson extraction.

        Returns:
             - return: str - Lesson text, or an empty string on failure.
        """
        lesson_prompt = (
            "Original user question:\n"
            f"{raw_user_message}\n\n"
            "Assistant answer:\n"
            f"{assistant_response}\n\n"
            f"Feedback polarity: {polarity}\n"
            f"Feedback text: {feedback_text or ''}"
        )
        try:
            return llm.generate(
                system_prompt=prompts.FEEDBACK_LESSON_SYSTEM_PROMPT,
                user_prompt=lesson_prompt,
            ).strip()
        except Exception:
            return ""
