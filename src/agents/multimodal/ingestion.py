"""Document and image ingestion helpers for FurnaceMind knowledge upload."""

from __future__ import annotations

import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from agents.multimodal.parsers import parse_docx, parse_excel, parse_pdf, parse_pptx


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping character chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def extract_text_from_file(file) -> tuple[str, str]:
    """Extract text from a supported uploaded file."""
    file_type = file.name.split(".")[-1].lower()
    if file_type == "pdf":
        return file_type, parse_pdf(file)
    if file_type == "docx":
        return file_type, parse_docx(file)
    if file_type == "pptx":
        return file_type, parse_pptx(file)
    if file_type in {"xls", "xlsx"}:
        return file_type, parse_excel(file)
    if file_type in {"txt", "md"}:
        return file_type, file.read().decode("utf-8", errors="ignore")
    return file_type, ""


def process_file(file, knowledge_store, embedding_client) -> dict:
    """Parse, chunk, embed, and upsert one uploaded file into Qdrant."""
    file_type = file.name.split(".")[-1].lower()
    if file_type in {"png", "jpg", "jpeg"}:
        return _process_image_file(file, knowledge_store, embedding_client, file_type)
    return _process_text_file(file, knowledge_store, embedding_client)


def _process_image_file(file, knowledge_store, embedding_client, file_type: str) -> dict:
    """Embed and upsert one image file into Qdrant."""
    image_bytes = file.read()
    upload_dir = Path("uploaded_images")
    upload_dir.mkdir(exist_ok=True)
    image_path = upload_dir / f"{uuid.uuid4()}_{file.name}"
    image_path.write_bytes(image_bytes)

    point_id = str(uuid.uuid4())
    knowledge_store.client.upsert(
        collection_name=knowledge_store.collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding_client.embed_image(image_bytes),
                payload={
                    "source": file.name,
                    "type": "image",
                    "file_path": str(image_path),
                },
            )
        ],
        wait=True,
    )
    return {
        "file_type": file_type,
        "qdrant_collection": knowledge_store.collection_name,
        "qdrant_point_ids": [point_id],
        "chunk_count": 1,
        "token_estimate": 0,
    }


def _process_text_file(file, knowledge_store, embedding_client) -> dict:
    """Extract text, chunk it, and upsert chunks into Qdrant."""
    file_type, text = extract_text_from_file(file)
    if not text:
        return {
            "file_type": file_type,
            "qdrant_collection": knowledge_store.collection_name,
            "qdrant_point_ids": [],
            "chunk_count": 0,
            "token_estimate": 0,
        }

    chunks = chunk_text(text)
    point_ids: list[str] = []
    for chunk in chunks:
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)
        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding_client.embed_text(chunk),
                    payload={
                        "source": file.name,
                        "type": file_type,
                        "content": chunk,
                    },
                )
            ],
            wait=True,
        )

    return {
        "file_type": file_type,
        "qdrant_collection": knowledge_store.collection_name,
        "qdrant_point_ids": point_ids,
        "chunk_count": len(chunks),
        "token_estimate": max(1, len(text) // 4),
    }
