"""Document and image ingestion pipeline for the Knowledge Hub.

Chunks text documents into overlapping windows and upserts them into the
Qdrant Knowledge collection via :class:`~memory.knowledge_vector_store.KnowledgeVectorStore`.
Images are handled via multimodal embeddings from
:class:`~embeddings.cloud_embedding.CloudEmbeddingClient`.

Entry point: :func:`process_file`.
"""

import os
import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from multimodal.parsers import (
    parse_docx,
    parse_excel,
    parse_pdf,
    parse_pptx,
)


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


# ---------------------------------------------
# 🔹 Main File Processor
# ---------------------------------------------
def process_file(file, knowledge_store, embedding_client) -> None:
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
    file_type = file.name.split(".")[-1].lower()

    file_type = file.name.split(".")[-1].lower()

    # ==================================================
    # 🖼 IMAGE HANDLING (Multimodal Embedding)
    # ==================================================
    if file_type in ["png", "jpg", "jpeg"]:

        image_bytes = file.read()

        # Save image locally (so we can display later)
        upload_dir = Path("uploaded_images")
        upload_dir.mkdir(exist_ok=True)

        image_path = upload_dir / f"{uuid.uuid4()}_{file.name}"

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # Generate embedding
        embedding = embedding_client.embed_image(image_bytes)

        # Insert into Qdrant
        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": file.name,
                        "type": "image",
                        "file_path": str(image_path),
                    },
                )
            ],
            wait=True,
        )

        return

    # ==================================================
    # 📄 TEXT DOCUMENT HANDLING
    # ==================================================

    if file_type == "pdf":
        text = parse_pdf(file)

    elif file_type == "docx":
        text = parse_docx(file)

    elif file_type == "pptx":
        text = parse_pptx(file)

    elif file_type in ["xls", "xlsx"]:
        text = parse_excel(file)

    else:
        # Unsupported file type
        return

    if not text:
        return

    chunks = chunk_text(text)

    for chunk in chunks:
        embedding = embedding_client.embed_text(chunk)

        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": file.name,
                        "type": file_type,
                        "content": chunk,  # store text for retrieval / display
                    },
                )
            ],
            wait=True,
        )
