# FurnaceMind/multimodal/ingestion.py
# Purpose: Secure file ingestion pipeline for Knowledge Hub
# Fixed: File size limits, content-type validation, safe filenames,
#        sentence-aware chunking, proper error handling

import re
import uuid
import logging
from pathlib import Path

from qdrant_client.models import PointStruct

from FurnaceMind.multimodal.parsers import (
    parse_pdf,
    parse_docx,
    parse_pptx,
    parse_excel,
)

logger = logging.getLogger(__name__)


# Configuration
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Magic bytes for content-type validation
MAGIC_SIGNATURES = {
    "pdf":  [b"%PDF"],
    "docx": [b"PK\x03\x04"],  # ZIP-based
    "pptx": [b"PK\x03\x04"],
    "xlsx": [b"PK\x03\x04"],
    "xls":  [b"\xd0\xcf\x11\xe0"],  # OLE2
    "png":  [b"\x89PNG"],
    "jpg":  [b"\xff\xd8\xff"],
    "jpeg": [b"\xff\xd8\xff"],
}



# Text Chunking (sentence-aware)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Split text into chunks respecting sentence boundaries where possible.
    Falls back to character-based splitting for very long sentences.
    """
    # Split into sentences / paragraphs
    sentences = _SENTENCE_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        # If a single sentence exceeds chunk_size, split it by characters
        if len(sentence) > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep last bit for overlap
                overlap_text = " ".join(current_chunk)[-overlap:]
                current_chunk = [overlap_text] if overlap_text else []
                current_len = len(overlap_text) if overlap_text else 0

            # Character-split the long sentence
            start = 0
            while start < len(sentence):
                end = start + chunk_size
                chunks.append(sentence[start:end])
                start = end - overlap
            continue

        # Normal flow: accumulate sentences
        if current_len + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Overlap: keep tail sentences
            overlap_text = " ".join(current_chunk)[-overlap:]
            current_chunk = [overlap_text] if overlap_text else []
            current_len = len(overlap_text) if overlap_text else 0

        current_chunk.append(sentence)
        current_len += len(sentence) + 1

    # Flush remaining
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



# Security helpers
def _sanitize_filename(name: str) -> str:
    """Remove path separators and other dangerous characters from filename."""
    # Take only the basename (strip any directory components)
    name = Path(name).name
    # Remove any remaining suspicious characters
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "unnamed_file"


def _validate_file_size(file) -> None:
    """Check file size doesn't exceed limit."""
    file.seek(0, 2)  # seek to end
    size = file.tell()
    file.seek(0)  # reset
    if size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large: {size / 1024 / 1024:.1f} MB "
            f"(max {MAX_FILE_SIZE_MB} MB)"
        )


def _validate_content_type(file, declared_type: str) -> None:
    """Verify file content matches declared type via magic bytes."""
    expected_signatures = MAGIC_SIGNATURES.get(declared_type)
    if not expected_signatures:
        return  # no signature to check (e.g., .txt)

    header = file.read(16)
    file.seek(0)

    if not any(header.startswith(sig) for sig in expected_signatures):
        raise ValueError(
            f"File content does not match declared type '{declared_type}'. "
            f"Possible file type mismatch or corruption."
        )



# Main File Processor
def process_file(file, knowledge_store, embedding_client) -> None:
    """
    Process and index an uploaded file into the knowledge vector store.
    Validates file size and content type before processing.
    """
    safe_name = _sanitize_filename(file.name)
    file_type = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""

    # Validate
    try:
        _validate_file_size(file)
        _validate_content_type(file, file_type)
    except ValueError as e:
        logger.warning(f"File validation failed for '{safe_name}': {e}")
        raise


    # Image Handling (Multimodal Embedding)
    if file_type in ("png", "jpg", "jpeg"):
        image_bytes = file.read()

        upload_dir = Path("uploaded_images")
        upload_dir.mkdir(exist_ok=True)

        # Safe filename with UUID prefix
        image_path = upload_dir / f"{uuid.uuid4()}_{safe_name}"

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
                        "source": safe_name,
                        "type": "image",
                        "file_path": str(image_path),
                    },
                )
            ],
            wait=True,
        )
        return


    # Text Document Handling
    parsers = {
        "pdf": parse_pdf,
        "docx": parse_docx,
        "pptx": parse_pptx,
        "xls": parse_excel,
        "xlsx": parse_excel,
        "txt": lambda f: f.read().decode("utf-8", errors="replace"),
    }

    parser = parsers.get(file_type)
    if parser is None:
        logger.warning(f"Unsupported file type: '{file_type}' for file '{safe_name}'")
        return

    try:
        text = parser(file)
    except Exception as e:
        logger.error(f"Failed to parse '{safe_name}': {e}")
        raise ValueError(f"Could not parse file '{safe_name}': {e}") from e

    if not text or not text.strip():
        logger.info(f"No text extracted from '{safe_name}'")
        return

    chunks = chunk_text(text)
    logger.info(f"Indexing '{safe_name}': {len(chunks)} chunks")

    for chunk in chunks:
        if not chunk.strip():
            continue

        embedding = embedding_client.embed_text(chunk)

        knowledge_store.client.upsert(
            collection_name=knowledge_store.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": safe_name,
                        "type": file_type,
                        "content": chunk,
                    },
                )
            ],
            wait=True,
        )