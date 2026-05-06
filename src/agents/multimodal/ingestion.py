"""Document ingestion pipeline for FurnaceMind knowledge and skill uploads."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct

from agents.memory.knowledge_vector_store import KnowledgeVectorStore
from agents.multimodal.parsers import parse_docx, parse_excel, parse_pdf, parse_pptx

_TEXT_EXTENSIONS = {"txt", "md", "csv", "json", "yaml", "yml"}


def _estimate_text_tokens(text: str | None) -> int:
    """
    Estimate the token count for extracted upload text.

    Args:
         - text: str | None - Text content to estimate.

    Returns:
         - return: int - Approximate token count.
    """
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 4)


def _file_name(file: Any) -> str:
    """
    Return a stable upload file name.

    Args:
         - file: Any - Uploaded file-like object.

    Returns:
         - return: str - File name or fallback upload name.
    """
    return str(getattr(file, "name", "") or "uploaded_file")


def _file_type(file: Any) -> str:
    """
    Return the lowercase extension for an uploaded file.

    Args:
         - file: Any - Uploaded file-like object.

    Returns:
         - return: str - Lowercase file extension without the dot.
    """
    return _file_name(file).rsplit(".", 1)[-1].lower()


def _reset_file(file: Any) -> None:
    """
    Reset a file-like object to the beginning when supported.

    Args:
         - file: Any - Uploaded file-like object.

    Returns:
         - return: None - This function does not return a value.
    """
    if hasattr(file, "seek"):
        file.seek(0)


def _read_plain_text(file: Any) -> str:
    """
    Read a plain text upload as a Unicode string.

    Args:
         - file: Any - Uploaded file-like object.

    Returns:
         - return: str - Decoded text content.
    """
    raw = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if isinstance(raw, str):
        return raw
    return bytes(raw).decode("utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
         - text: str - Input text to split.
         - chunk_size: int - Maximum characters per chunk.
         - overlap: int - Characters to overlap between consecutive chunks.

    Returns:
         - return: list[str] - Text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def extract_text(file: Any) -> str:
    """
    Extract text from a supported upload.

    Args:
         - file: Any - Uploaded file-like object.

    Returns:
         - return: str - Extracted text content.
    """
    file_type = _file_type(file)
    _reset_file(file)
    try:
        if file_type == "pdf":
            return parse_pdf(file)
        if file_type == "docx":
            return parse_docx(file)
        if file_type == "pptx":
            return parse_pptx(file)
        if file_type in {"xls", "xlsx"}:
            return parse_excel(file)
        if file_type in _TEXT_EXTENSIONS:
            return _read_plain_text(file)
        return ""
    finally:
        _reset_file(file)


def summarize_text(text: str, max_chars: int = 320) -> str:
    """
    Build a compact summary preview from extracted text.

    Args:
         - text: str - Extracted text content.
         - max_chars: int - Maximum characters to include.

    Returns:
         - return: str - Compact text preview.
    """
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


def process_file(file: Any, knowledge_store: Any, embedding_client: Any) -> None:
    """
    Parse, chunk, embed, and upsert a document or image into Qdrant.

    Args:
         - file: Any - Streamlit uploaded file object.
         - knowledge_store: Any - Qdrant-backed knowledge store.
         - embedding_client: Any - Embedding client for text and image content.

    Returns:
         - return: None - This function does not return a value.
    """
    file_name = _file_name(file)
    file_type = _file_type(file)

    if file_type in ["png", "jpg", "jpeg"]:
        image_bytes = file.read()
        upload_dir = Path("uploaded_images")
        upload_dir.mkdir(exist_ok=True)
        image_path = upload_dir / f"{uuid.uuid4()}_{file_name}"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        embedding = embedding_client.embed_image(image_bytes)
        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": file_name,
                        "type": "image",
                        "file_path": str(image_path),
                    },
                )
            ],
            wait=True,
        )
        return

    text = extract_text(file)
    if not text:
        return

    for chunk in chunk_text(text):
        embedding = embedding_client.embed_text(chunk)
        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": file_name,
                        "type": file_type,
                        "content": chunk,
                    },
                )
            ],
            wait=True,
        )


class DocumentIngestionService:
    """Service for storing uploaded knowledge and skill documents."""

    def __init__(self, knowledge_store: KnowledgeVectorStore) -> None:
        """
        Create the ingestion service with vector and relational stores.

        Args:
             - knowledge_store: KnowledgeVectorStore - Qdrant-backed knowledge store.

        Returns:
             - return: None - This function does not return a value.
        """
        from furnace_data.relational import (
            MemoryDocumentRepository,
            SkillRepository,
            build_relational_engine,
            build_relational_session_factory,
        )

        engine = build_relational_engine()
        session_factory = build_relational_session_factory(engine)
        self._knowledge_store = knowledge_store
        self._documents = MemoryDocumentRepository(session_factory)
        self._skills = SkillRepository(session_factory)

    def ingest_knowledge_file(self, *, user_id: str, file: Any) -> str | None:
        """
        Ingest one knowledge file into Qdrant and PostgreSQL metadata.

        Args:
             - user_id: str - User that owns the uploaded file.
             - file: Any - Uploaded file-like object.

        Returns:
             - return: str | None - Created memory document id when indexed.
        """
        file_name = _file_name(file)
        file_type = _file_type(file)
        text = extract_text(file)
        if not text.strip():
            return None

        chunks = chunk_text(text)
        point_ids = []
        for index, chunk in enumerate(chunks):
            point_ids.append(
                self._knowledge_store.add_document(
                    chunk,
                    {
                        "source": file_name,
                        "type": file_type,
                        "document_kind": "knowledge",
                        "chunk_index": index,
                        "user_id": user_id,
                    },
                )
            )

        document = self._documents.create_document(
            user_id=user_id,
            filename=file_name,
            file_type=file_type,
            file_path=None,
            summary=summarize_text(text),
            qdrant_collection=self._knowledge_store.collection_name,
            qdrant_point_ids=point_ids,
            token_estimate=_estimate_text_tokens(text),
            metadata={"chunk_count": len(chunks), "source": "upload"},
        )
        return document.document_id

    def ingest_skill_file(self, *, user_id: str, file: Any) -> str | None:
        """
        Ingest one uploaded skill file into Qdrant and PostgreSQL metadata.

        Args:
             - user_id: str - User that owns the uploaded skill.
             - file: Any - Uploaded file-like object.

        Returns:
             - return: str | None - Created skill id when indexed.
        """
        file_name = _file_name(file)
        file_type = _file_type(file)
        text = extract_text(file)
        if not text.strip():
            return None

        chunks = chunk_text(text)
        point_ids = []
        for index, chunk in enumerate(chunks):
            point_ids.append(
                self._knowledge_store.add_document(
                    chunk,
                    {
                        "source": file_name,
                        "type": file_type,
                        "document_kind": "skill",
                        "chunk_index": index,
                        "user_id": user_id,
                    },
                )
            )

        skill = self._skills.create_skill(
            name=Path(file_name).stem.replace("_", " ").strip().title(),
            instruction=text.strip(),
            description=summarize_text(text),
            source_type="uploaded",
            qdrant_collection=self._knowledge_store.collection_name,
            is_active=True,
            created_by=user_id,
            metadata={
                "file_name": file_name,
                "file_type": file_type,
                "chunk_count": len(chunks),
                "qdrant_point_ids": point_ids,
            },
        )
        return skill.skill_id
