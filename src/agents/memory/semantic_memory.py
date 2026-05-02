"""Qdrant-backed semantic memory for FurnaceMind lessons and skills."""

from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from utils.settings import settings


class SemanticMemoryStore:
    """Store and retrieve FurnaceMind semantic objects in Qdrant."""

    def __init__(self, embedding_client: Any | None = None) -> None:
        """Create a Qdrant-backed semantic memory store."""
        from agents.embeddings.local_embedding import LocalEmbeddingClient

        qcfg = settings.qdrant_shift
        self.client = QdrantClient(
            url=qcfg.url,
            api_key=qcfg.api_key,
            timeout=qcfg.timeout,
        )
        self.embedding = embedding_client or LocalEmbeddingClient()
        self.collection_name = os.getenv(
            "FURNACEMIND_MEMORY_COLLECTION",
            "furnacemind_feedback_lessons",
        )
        self.skill_collection_name = os.getenv(
            "FURNACEMIND_SKILL_COLLECTION",
            "furnacemind_skills",
        )
        self.long_term_collection_name = os.getenv(
            "FURNACEMIND_LONG_TERM_MEMORY_COLLECTION",
            "furnacemind_long_term_memories",
        )
        self.embedding_dim = self.embedding.dimension
        self._ensure_collection(self.collection_name)
        self._ensure_collection(self.skill_collection_name)
        self._ensure_collection(self.long_term_collection_name)

    def _ensure_collection(self, collection_name: str) -> None:
        """Create a Qdrant collection when it is missing."""
        collections = self.client.get_collections().collections
        if collection_name in {item.name for item in collections}:
            self._ensure_user_index(collection_name)
            return
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
        )
        self._ensure_user_index(collection_name)

    def _ensure_user_index(self, collection_name: str) -> None:
        """Create the user_id payload index needed for filtered Qdrant searches."""
        info = self.client.get_collection(collection_name)
        payload_schema = info.payload_schema or {}
        if "user_id" in payload_schema:
            return
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def add_lesson(
        self,
        *,
        user_id: str,
        lesson: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store one feedback lesson and return its Qdrant point ID."""
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=self.embedding.embed_text(lesson),
                    payload={
                        "user_id": user_id,
                        "type": "feedback_lesson",
                        "memory": lesson,
                        **(metadata or {}),
                    },
                )
            ],
            wait=True,
        )
        return point_id

    def add_long_term_memory(
        self,
        *,
        user_id: str,
        memory: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store one long-term chat memory and return its Qdrant point ID."""
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.long_term_collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=self.embedding.embed_text(memory),
                    payload={
                        "user_id": user_id,
                        "type": "long_term_memory",
                        "memory": memory,
                        **(metadata or {}),
                    },
                )
            ],
            wait=True,
        )
        return point_id

    def upsert_skill(
        self,
        *,
        skill: dict[str, Any],
        user_id: str | None = None,
    ) -> str:
        """Store or update one skill vector and return its point ID."""
        skill_id = str(skill.get("skill_id") or skill.get("name") or uuid.uuid4())
        owner = str(skill.get("created_by") or user_id or "__global__")
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"furnacemind-skill:{owner}:{skill_id}"))
        skill_text = build_skill_embedding_text(skill)
        self.client.upsert(
            collection_name=self.skill_collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=self.embedding.embed_text(skill_text),
                    payload={
                        "type": "skill",
                        "user_id": owner,
                        "skill_id": skill.get("skill_id"),
                        "name": skill.get("name"),
                        "description": skill.get("description"),
                        "instruction": skill.get("instruction"),
                        "source_type": skill.get("source_type"),
                        "is_active": bool(skill.get("is_active", True)),
                        "skill_text": skill_text,
                    },
                )
            ],
            wait=True,
        )
        return point_id

    def search_skills(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search global and user-owned skills by semantic similarity."""
        if not query.strip():
            return []
        vector = self.embedding.embed_text(query)
        results: list[dict[str, Any]] = []
        for owner in ("__global__", user_id):
            response = self.client.query_points(
                collection_name=self.skill_collection_name,
                query=vector,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=owner))]
                ),
                limit=top_k,
                with_payload=True,
            )
            results.extend(
                {"score": item.score, "payload": item.payload or {}}
                for item in response.points
            )
        results.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
        return results[:top_k]

    def search_lessons(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 4,
    ) -> list[dict[str, Any]]:
        """Search lessons for one user."""
        if not query.strip():
            return []
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedding.embed_text(query),
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [{"score": item.score, "payload": item.payload or {}} for item in results.points]

    def search_long_term_memories(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search long-term memories for one user."""
        if not query.strip():
            return []
        results = self.client.query_points(
            collection_name=self.long_term_collection_name,
            query=self.embedding.embed_text(query),
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [{"score": item.score, "payload": item.payload or {}} for item in results.points]


def build_skill_embedding_text(skill: dict[str, Any]) -> str:
    """Return text used to embed a skill for semantic search."""
    parts = [
        f"Skill name: {skill.get('name') or ''}",
        f"Description: {skill.get('description') or ''}",
        f"Instruction: {skill.get('instruction') or ''}",
    ]
    return "\n".join(part for part in parts if part.strip())


def build_feedback_lessons_context(lessons: list[dict[str, Any]]) -> str:
    """Return prompt-ready text for retrieved feedback lessons."""
    if not lessons:
        return ""
    lines = ["FEEDBACK LESSONS FOR THIS USER:"]
    for item in lessons:
        payload = item.get("payload") or {}
        memory = payload.get("memory")
        if memory:
            lines.append(f"- {memory}")
    return "\n".join(lines)


def build_long_term_memory_context(memories: list[dict[str, Any]]) -> str:
    """Return prompt-ready text for retrieved long-term memories."""
    if not memories:
        return ""
    lines = ["RELEVANT LONG-TERM MEMORY FOR THIS USER:"]
    for item in memories:
        payload = item.get("payload") or {}
        memory = payload.get("memory")
        if memory:
            lines.append(f"- {memory}")
    return "\n".join(lines)
