"""Upload and chat-submission helpers for FurnaceMind."""

from __future__ import annotations

from typing import Any

from agents.embeddings.cloud_embedding import CloudEmbeddingClient
from agents.memory.knowledge_vector_store import KnowledgeVectorStore
from agents.multimodal.ingestion import DocumentIngestionService, process_file


def ingest_uploaded_knowledge_files(
    files: list[Any],
    *,
    user_id: str,
    knowledge_store: KnowledgeVectorStore,
    embedding_client: CloudEmbeddingClient,
    ingestion_service: DocumentIngestionService | None,
) -> None:
    """
    Ingest uploaded knowledge files into configured stores.

    Args:
         - files: list[Any] - Uploaded Streamlit file objects.
         - user_id: str - Owner of the uploaded files.
         - knowledge_store: KnowledgeVectorStore - Qdrant-backed knowledge store.
         - embedding_client: CloudEmbeddingClient - Embedding client fallback.
         - ingestion_service: DocumentIngestionService | None - SQL and Qdrant service.

    Returns:
         - return: None - Stores uploaded content and metadata.
    """
    for uploaded in files:
        if ingestion_service is not None:
            ingestion_service.ingest_knowledge_file(user_id=user_id, file=uploaded)
        else:
            process_file(uploaded, knowledge_store, embedding_client)


def extract_submission(
    chat_submission: object,
    *,
    user_id: str,
    knowledge_store: KnowledgeVectorStore,
    embedding_client: CloudEmbeddingClient,
    ingestion_service: DocumentIngestionService | None,
) -> tuple[str | None, str | None]:
    """
    Extract user text and ingest attached files from a chat submission.

    Args:
         - chat_submission: object - Streamlit chat input submission.
         - user_id: str - Owner of any attached files.
         - knowledge_store: KnowledgeVectorStore - Qdrant-backed knowledge store.
         - embedding_client: CloudEmbeddingClient - Embedding client fallback.
         - ingestion_service: DocumentIngestionService | None - SQL and Qdrant service.

    Returns:
         - return: tuple[str | None, str | None] - Agent query and display text.
    """
    if not chat_submission:
        return None, None

    if hasattr(chat_submission, "text"):
        typed_query = chat_submission.text
        files = getattr(chat_submission, "files", None) or []
    elif isinstance(chat_submission, dict):
        typed_query = chat_submission.get("text", "")
        files = chat_submission.get("files", []) or []
    else:
        typed_query = str(chat_submission)
        files = []

    if files:
        ingest_uploaded_knowledge_files(
            files,
            user_id=user_id,
            knowledge_store=knowledge_store,
            embedding_client=embedding_client,
            ingestion_service=ingestion_service,
        )

    if typed_query and str(typed_query).strip():
        text = str(typed_query).strip()
        return text, text
    if files:
        file_label = ", ".join(getattr(file_obj, "name", "file") for file_obj in files)
        display = f"Attached files: {file_label}"
        return display, display
    return None, None
