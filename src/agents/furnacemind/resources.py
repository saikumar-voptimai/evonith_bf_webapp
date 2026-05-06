"""Cached runtime dependencies for the FurnaceMind page."""

from __future__ import annotations

import streamlit as st

from agents.embeddings.cloud_embedding import CloudEmbeddingClient
from agents.furnacemind.context import SystemPromptContext
from agents.furnacemind.skills import SkillEngine
from agents.memory.conversation_history import ConversationHistoryStore
from agents.memory.knowledge_vector_store import KnowledgeVectorStore
from agents.memory.vector_store import QdrantVectorStore
from agents.multimodal.ingestion import DocumentIngestionService


@st.cache_resource(show_spinner=False)
def cached_embedding_client() -> CloudEmbeddingClient:
    """
    Return the shared cloud embedding client.

    Args:
         - None

    Returns:
         - return: CloudEmbeddingClient - Cached embedding client.
    """
    return CloudEmbeddingClient()


@st.cache_resource(show_spinner=False)
def cached_knowledge_store() -> KnowledgeVectorStore:
    """
    Return the shared knowledge vector store.

    Args:
         - None

    Returns:
         - return: KnowledgeVectorStore - Cached knowledge vector store.
    """
    return KnowledgeVectorStore(cached_embedding_client())


@st.cache_resource(show_spinner=False)
def cached_shift_store() -> QdrantVectorStore:
    """
    Return the shared shift vector store.

    Args:
         - None

    Returns:
         - return: QdrantVectorStore - Cached shift vector store.
    """
    return QdrantVectorStore()


@st.cache_resource(show_spinner=False)
def cached_context() -> SystemPromptContext:
    """
    Return the shared FurnaceMind system prompt context.

    Args:
         - None

    Returns:
         - return: SystemPromptContext - Cached prompt context.
    """
    return SystemPromptContext()


@st.cache_resource(show_spinner=False)
def cached_skill_engine() -> SkillEngine:
    """
    Return the shared FurnaceMind skill engine.

    Args:
         - None

    Returns:
         - return: SkillEngine - Cached skill engine.
    """
    return SkillEngine()


@st.cache_resource(show_spinner=False)
def cached_ingestion_service() -> DocumentIngestionService | None:
    """
    Return the shared document ingestion service when available.

    Args:
         - None

    Returns:
         - return: DocumentIngestionService | None - Cached ingestion service.
    """
    try:
        return DocumentIngestionService(cached_knowledge_store())
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def cached_history_store() -> ConversationHistoryStore | None:
    """
    Return the shared PostgreSQL conversation history store when available.

    Args:
         - None

    Returns:
         - return: ConversationHistoryStore | None - Cached history store.
    """
    try:
        return ConversationHistoryStore()
    except Exception:
        return None
