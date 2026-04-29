"""Document and image ingestion pipeline for the Knowledge Hub.

Chunks text documents into overlapping windows and upserts them into the
Qdrant Knowledge collection via :class:`~memory.knowledge_vector_store.KnowledgeVectorStore`.
Images are handled via multimodal embeddings from
:class:`~embeddings.cloud_embedding.CloudEmbeddingClient`.

Entry point: :func:`process_file`.
"""

import hashlib
import uuid
from io import BytesIO
from pathlib import Path

from qdrant_client.models import PointStruct

from multimodal.parsers import (
    parse_docx,
    parse_excel,
    parse_pdf,
    parse_pptx,
    parse_text,
)

TEXT_FILE_TYPES = {"txt", "md", "csv", "json", "log"}


# ---------------------------------------------
# 🔹 Text Chunking
# ---------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split *text* into overlapping chunks of *chunk_size* characters.

    Args:
        text:       Input text to split.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters to overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def read_uploaded_file_bytes(file) -> bytes:
    """Return bytes from a Streamlit UploadedFile or generic file-like object."""
    if hasattr(file, "getvalue"):
        data = file.getvalue()
        return data if isinstance(data, bytes) else str(data).encode("utf-8")

    if hasattr(file, "seek"):
        file.seek(0)
    data = file.read()
    if hasattr(file, "seek"):
        file.seek(0)
    return data if isinstance(data, bytes) else str(data).encode("utf-8")


def compute_content_hash(file_bytes: bytes) -> str:
    """Return a stable SHA-256 hash for uploaded file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def parse_file_text(filename: str, file_bytes: bytes) -> str:
    """Extract text from a supported upload type."""
    file_type = Path(filename).suffix.lower().lstrip(".")
    file_obj = BytesIO(file_bytes)

    if file_type == "pdf":
        return parse_pdf(file_obj)
    if file_type == "docx":
        return parse_docx(file_obj)
    if file_type == "pptx":
        return parse_pptx(file_obj)
    if file_type in {"xls", "xlsx"}:
        return parse_excel(file_obj)
    if file_type in TEXT_FILE_TYPES:
        return parse_text(file_obj)

    return ""


# ---------------------------------------------
# 🔹 Main File Processor
# ---------------------------------------------
def process_file(
    file,
    knowledge_store,
    embedding_client,
    *,
    doc_id: str | None = None,
    file_bytes: bytes | None = None,
) -> dict | None:
    """Parse, chunk, embed, and upsert a document or image into the Knowledge Hub.

    Supports PDF, DOCX, PPTX, XLS/XLSX (text chunking + sentence embeddings)
    and PNG/JPG/JPEG (multimodal image embeddings).

    Args:
        file:            Streamlit ``UploadedFile`` object.
        knowledge_store: :class:`~memory.knowledge_vector_store.KnowledgeVectorStore`
                         instance to upsert chunks into.
        embedding_client: :class:`~embeddings.cloud_embedding.CloudEmbeddingClient`
                          used for both text and image embeddings.
    """
    file_bytes = file_bytes if file_bytes is not None else read_uploaded_file_bytes(file)
    file_type = Path(file.name).suffix.lower().lstrip(".")
    doc_id = doc_id or str(uuid.uuid4())
    point_ids: list[str] = []

    # ==================================================
    # 🖼 IMAGE HANDLING (Multimodal Embedding)
    # ==================================================
    if file_type in ["png", "jpg", "jpeg"]:

        image_bytes = file_bytes

        # Generate embedding
        embedding = embedding_client.embed_image(image_bytes)
        point_id = str(uuid.uuid4())

        # Insert into Qdrant
        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "doc_id": doc_id,
                        "source": file.name,
                        "type": "image",
                    },
                )
            ],
            wait=True,
        )
        point_ids.append(point_id)

        return {
            "doc_id": doc_id,
            "filename": file.name,
            "file_type": file_type,
            "content_hash": compute_content_hash(file_bytes),
            "file_size_bytes": len(file_bytes),
            "text": "",
            "text_preview": "Image uploaded for multimodal knowledge search.",
            "qdrant_collection": knowledge_store.collection_name,
            "qdrant_point_ids": point_ids,
        }

    # ==================================================
    # 📄 TEXT DOCUMENT HANDLING
    # ==================================================

    text = parse_file_text(file.name, file_bytes)

    if not text:
        return None

    chunks = chunk_text(text)

    for chunk in chunks:
        embedding = embedding_client.embed_text(chunk)
        point_id = str(uuid.uuid4())

        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "doc_id": doc_id,
                        "source": file.name,
                        "type": file_type,
                        "content": chunk,  # store text for retrieval / display
                    },
                )
            ],
            wait=True,
        )
        point_ids.append(point_id)

    compact_preview = " ".join(text.split())[:1200]
    return {
        "doc_id": doc_id,
        "filename": file.name,
        "file_type": file_type,
        "content_hash": compute_content_hash(file_bytes),
        "file_size_bytes": len(file_bytes),
        "text": text,
        "text_preview": compact_preview,
        "qdrant_collection": knowledge_store.collection_name,
        "qdrant_point_ids": point_ids,
    }
