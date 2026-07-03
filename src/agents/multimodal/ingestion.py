"""Build and persist multimodal retrieval chunks for FurnaceMind MRAG.

This module owns the upload-to-vector-store ingestion path used by the
``Multimodal Knowledge`` sidebar. It converts a single uploaded file into a
sequence of :class:`DocumentPart` objects, embeds each part with the correct
modality, writes the vectors to Qdrant, and records best-effort SQL metadata for
library listing, active-document filtering, traceability, and deletion.

Pipeline stages:
1. Read the Streamlit upload bytes without permanently consuming the stream.
2. Generate a stable content hash based ``document_id`` for deduplication.
3. Parse the file by type and normalize it into text-backed or image-backed
   ``DocumentPart`` records.
4. Embed text-backed parts with ``embed_text(..., input_type="document")``.
5. Embed image-backed parts with ``embed_image(..., input_type="document")`` so
   images, rendered pages, rendered slides, and text share one multimodal vector
   space.
6. Upsert all vectors and rich payload metadata into the configured Qdrant
   knowledge collection.
7. Persist document/chunk metadata to PostgreSQL when repositories are provided.

Supported input families:
- Images: PNG, JPG/JPEG, WEBP, BMP, TIF/TIFF.
- Documents: PDF, DOCX, PPTX, TXT, Markdown.
- Tables: XLS/XLSX and CSV.

PDF files create both page text chunks and rendered page image chunks. PPTX files
create slide text chunks, embedded image chunks, and optional full-slide render
chunks when LibreOffice/``soffice`` is available on the host.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from qdrant_client.models import PointStruct

from furnace_data.runtime_paths import runtime_path
from agents.multimodal.parsers import (
    extract_csv_text,
    extract_docx_text,
    extract_excel_sheets,
    extract_pdf_pages,
    extract_pptx_slides,
    extract_text_file,
    render_pptx_slides,
)
from utils.logger import get_logger

_POINT_NAMESPACE = uuid.NAMESPACE_URL
logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentPart:
    """Normalized unit of knowledge that becomes one Qdrant point.

    ``DocumentPart`` is the common representation for every modality produced by
    ingestion. Text-bearing parts keep extracted text in ``content`` and leave
    ``image_path`` empty. Visual parts keep the original/rendered image on disk,
    store that path in ``image_path``, and may include a short textual caption or
    optional OCR text in ``content``.

    Attributes:
        document_id: Stable content hash id shared by all parts from one upload.
        chunk_id: Stable id unique within the document, such as ``page_text_0001``.
        source: Original uploaded filename used in citations.
        file_type: Lowercase file extension, for example ``pdf`` or ``pptx``.
        modality: Retrieval modality, for example ``text``, ``table``,
            ``page_image``, ``slide_text``, ``slide_image``, or ``slide_render``.
        chunk_index: Zero-based ordering across every part in the document.
        user_id: Optional owner id used for Qdrant filtering.
        content: Text/caption stored in payload and used for summaries.
        image_path: Local path to the visual source. When present, the part is
            embedded with ``embed_image``.
        page_number: Source PDF page number when applicable.
        slide_number: Source PPTX slide number when applicable.
        sheet_name: Source Excel sheet name when applicable.
        metadata: Additional parser-specific payload fields.
    """

    document_id: str
    chunk_id: str
    source: str
    file_type: str
    modality: str
    chunk_index: int
    user_id: str | None = None
    content: str = ""
    image_path: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def point_id(self) -> str:
        """Return the deterministic Qdrant point id for this part.

        The id includes owner, document id, and chunk id so re-indexing the same
        file for the same user overwrites the same points instead of creating
        duplicates.

        Returns:
            UUID string accepted by Qdrant as a point id.
        """
        owner = self.user_id or "shared"
        return str(
            uuid.uuid5(_POINT_NAMESPACE, f"{owner}:{self.document_id}:{self.chunk_id}")
        )

    def payload(self) -> dict[str, Any]:
        """Build the Qdrant payload stored beside this part vector.

        The payload carries both retrieval metadata and answer-citation metadata.
        ``user_id`` and ``document_id`` support scoped search and deletion.
        ``source``, ``page_number``, ``slide_number``, and ``sheet_name`` let the
        LLM cite exact locations. ``image_path`` lets the agent attach retrieved
        visual evidence to a vision-capable model after vector search.

        Returns:
            Dictionary suitable for ``PointStruct.payload``.
        """
        payload = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "file_type": self.file_type,
            "type": self.file_type,
            "modality": self.modality,
            "chunk_index": self.chunk_index,
            "content": self.content,
            **self.metadata,
        }
        if self.user_id:
            payload["user_id"] = self.user_id
        optional = {
            "image_path": self.image_path,
            "page_number": self.page_number,
            "slide_number": self.slide_number,
            "sheet_name": self.sheet_name,
        }
        payload.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return payload


def _read_upload_bytes(file) -> bytes:
    """Read bytes from a Streamlit upload or file-like object.

    Args:
        file: Object exposing ``read()`` and optionally ``seek()``. Streamlit
            uploaded files support both.

    Returns:
        Raw file bytes. When ``seek`` is available, the stream is rewound before
        and after reading so later callers can still inspect it.
    """
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass
    data = file.read()
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass
    return data


def _safe_name(name: str) -> str:
    """Sanitize a value for use in generated local filenames.

    Args:
        name: Arbitrary filename/id fragment.

    Returns:
        ASCII-safe fragment containing letters, numbers, underscores, dots, and
        hyphens. Returns ``"upload"`` if nothing usable remains.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "upload"


def _document_id(file_bytes: bytes) -> str:
    """Create the stable MRAG document id for an upload.

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Returns:
        ``doc_`` plus the first 24 hex characters of the SHA-256 digest. This is
        stable across repeated uploads of identical content.
    """
    digest = hashlib.sha256(file_bytes).hexdigest()[:24]
    return f"doc_{digest}"


def _image_suffix(image_bytes: bytes, fallback: str = "png") -> str:
    """Infer the best file extension for saved image bytes.

    Args:
        image_bytes: Raw image bytes.
        fallback: Extension used when PIL cannot identify the image format.

    Returns:
        Lowercase extension without a leading dot. ``jpeg`` is normalized to
        ``jpg`` for shorter generated filenames.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        fmt = (image.format or fallback).lower()
        return "jpg" if fmt == "jpeg" else fmt
    except Exception:
        return fallback


def _save_image_bytes(
    *,
    document_id: str,
    chunk_id: str,
    image_bytes: bytes,
    suffix: str = "png",
) -> str:
    """Save visual chunk bytes under the MRAG image storage directory.

    Args:
        document_id: Stable upload id used in the generated filename.
        chunk_id: Chunk id used in the generated filename.
        image_bytes: Raw image bytes to write.
        suffix: File extension without a leading dot.

    Returns:
        String path stored in Qdrant payload. Retrieval later reads this path and
        sends the image to the multimodal LLM as visual evidence.
    """
    path = runtime_path(
        "uploads",
        "furnacemind",
        "mrag_images",
        f"{_safe_name(document_id)}_{_safe_name(chunk_id)}.{suffix}",
        create_parent=True,
    )
    path.write_bytes(image_bytes)
    return str(path)


def _optional_ocr_text(image_bytes: bytes) -> str:
    """Extract optional OCR text from an image for payload context.

    OCR is not required for visual retrieval. The image vector is always produced
    from the original pixels. OCR text, when available, only enriches ``content``
    so answers can quote visible labels or table text.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        OCR text, or an empty string when pytesseract is unavailable or fails.
    """
    try:
        import pytesseract  # type: ignore[import-not-found]

        image = Image.open(BytesIO(image_bytes))
        return str(pytesseract.image_to_string(image) or "").strip()
    except ImportError:
        return ""
    except Exception as exc:
        logger.debug("Optional image OCR skipped: %s", exc)
        return ""


def _embed_text(embedding_client, text: str, *, input_type: str) -> list[float]:
    """Embed text with a client-compatible retrieval mode.

    Args:
        embedding_client: Embedding client exposing ``embed_text``. Newer clients
            accept ``input_type``; older test doubles may not.
        text: Text to embed.
        input_type: Retrieval mode such as ``"document"`` or ``"query"``.

    Returns:
        Dense embedding vector.
    """
    try:
        return embedding_client.embed_text(text, input_type=input_type)
    except TypeError:
        return embedding_client.embed_text(text)


def _embed_image(
    embedding_client, image_bytes: bytes, *, input_type: str
) -> list[float]:
    """Embed image bytes with a client-compatible retrieval mode.

    Args:
        embedding_client: Multimodal embedding client exposing ``embed_image``.
        image_bytes: Raw image bytes read from the saved visual source.
        input_type: Retrieval mode, normally ``"document"`` for indexed chunks.

    Returns:
        Dense image embedding vector in the same space as text embeddings.
    """
    try:
        return embedding_client.embed_image(image_bytes, input_type=input_type)
    except TypeError:
        return embedding_client.embed_image(image_bytes)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Split extracted text into overlapping character chunks.

    Args:
        text: Extracted text from a document, slide, sheet, or table.
        chunk_size: Maximum characters per chunk.
        overlap: Characters repeated between adjacent chunks to preserve context.

    Returns:
        Non-empty stripped chunks in source order.

    Raises:
        ValueError: If ``chunk_size`` is not positive or ``overlap`` is invalid.
    """
    normalized = str(text or "").strip()
    if not normalized:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = end - overlap
    return [chunk for chunk in chunks if chunk]


def _text_parts(
    *,
    document_id: str,
    source: str,
    file_type: str,
    modality: str,
    text: str,
    chunk_index_start: int,
    page_number: int | None = None,
    slide_number: int | None = None,
    sheet_name: str | None = None,
    user_id: str | None = None,
) -> tuple[list[DocumentPart], int]:
    """Convert extracted text into ordered ``DocumentPart`` objects.

    Args:
        document_id: Stable id shared by all parts from the upload.
        source: Uploaded filename for citations.
        file_type: Lowercase file extension.
        modality: Source modality label, such as ``page_text``, ``slide_text``,
            ``table``, or ``text``.
        text: Extracted text to chunk.
        chunk_index_start: Global chunk index to start from.
        page_number: Source PDF page number, if any.
        slide_number: Source PPTX slide number, if any.
        sheet_name: Source Excel sheet name, if any.
        user_id: Optional owner id for user-scoped retrieval.

    Returns:
        Tuple of ``(parts, next_chunk_index)``. ``parts`` may be empty when the
        text is blank.
    """
    parts: list[DocumentPart] = []
    chunk_index = chunk_index_start
    for text_index, chunk in enumerate(chunk_text(text), start=1):
        chunk_id = f"{modality}_{chunk_index:04d}"
        parts.append(
            DocumentPart(
                document_id=document_id,
                chunk_id=chunk_id,
                source=source,
                file_type=file_type,
                modality=modality,
                chunk_index=chunk_index,
                user_id=user_id,
                content=chunk,
                page_number=page_number,
                slide_number=slide_number,
                sheet_name=sheet_name,
                metadata={"text_chunk_index": text_index},
            )
        )
        chunk_index += 1
    return parts, chunk_index


def _image_part(
    *,
    document_id: str,
    source: str,
    file_type: str,
    modality: str,
    image_bytes: bytes,
    chunk_index: int,
    content: str,
    user_id: str | None = None,
    page_number: int | None = None,
    slide_number: int | None = None,
    suffix: str = "png",
    metadata: dict[str, Any] | None = None,
) -> DocumentPart:
    """Create an image-backed ``DocumentPart`` and save the visual bytes.

    Args:
        document_id: Stable id shared by all parts from the upload.
        source: Uploaded filename for citations.
        file_type: Lowercase file extension.
        modality: Visual modality label, such as ``image``, ``page_image``,
            ``slide_image``, or ``slide_render``.
        image_bytes: Raw image bytes to save and later embed.
        chunk_index: Global chunk index assigned to this image part.
        content: Caption/context stored in the payload beside the vector.
        user_id: Optional owner id for user-scoped retrieval.
        page_number: Source PDF page number, if any.
        slide_number: Source PPTX slide number, if any.
        suffix: File extension for the saved image.
        metadata: Optional parser-specific payload additions.

    Returns:
        A ``DocumentPart`` with ``image_path`` populated. Because ``image_path``
        is present, ``_embedding_for_part`` will embed it with ``embed_image``.
    """
    chunk_id = f"{modality}_{chunk_index:04d}"
    image_path = _save_image_bytes(
        document_id=document_id,
        chunk_id=chunk_id,
        image_bytes=image_bytes,
        suffix=suffix,
    )
    metadata_payload = {**(metadata or {})}
    ocr_text = _optional_ocr_text(image_bytes)
    if ocr_text:
        metadata_payload["ocr_engine"] = "pytesseract"
        metadata_payload["ocr_text"] = ocr_text[:4000]
        content = "\n\n".join([content, f"OCR text:\n{ocr_text[:4000]}"])
    return DocumentPart(
        document_id=document_id,
        chunk_id=chunk_id,
        source=source,
        file_type=file_type,
        modality=modality,
        chunk_index=chunk_index,
        user_id=user_id,
        content=content,
        image_path=image_path,
        page_number=page_number,
        slide_number=slide_number,
        metadata=metadata_payload,
    )


def build_document_parts(
    filename: str,
    file_bytes: bytes,
    *,
    user_id: str | None = None,
) -> list[DocumentPart]:
    """Parse an upload and normalize it into MRAG parts.

    This function does not call the embedding service and does not write to
    Qdrant. It only decides how the upload should be represented for retrieval.

    Args:
        filename: Original uploaded filename. The extension selects the parser.
        file_bytes: Raw uploaded file bytes.
        user_id: Optional owner id copied into every generated part.

    Returns:
        Ordered list of ``DocumentPart`` objects. The list may contain both text
        and image parts. Unknown or unsupported file extensions return ``[]``.

    File-type behavior:
        Images: One ``image`` part backed by the uploaded pixels.
        PDF: ``page_text`` parts plus ``page_image`` rendered page parts.
        DOCX: Chunked ``text`` parts from paragraphs and tables.
        PPTX: ``slide_text`` parts, ``slide_image`` embedded-image parts, and
            optional ``slide_render`` full-slide image parts when rendering is
            available.
        XLS/XLSX/CSV: ``table`` parts containing Markdown/CSV-like table text.
        TXT/MD/Markdown: Chunked ``text`` parts.
    """
    file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    document_id = _document_id(file_bytes)
    parts: list[DocumentPart] = []
    chunk_index = 0

    if file_type in {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}:
        parts.append(
            _image_part(
                document_id=document_id,
                source=filename,
                file_type=file_type,
                modality="image",
                image_bytes=file_bytes,
                chunk_index=chunk_index,
                user_id=user_id,
                content=(
                    f"Uploaded image from {filename}. Use the attached image for "
                    "visual evidence, OCR, diagrams, labels, and layout."
                ),
                suffix=_image_suffix(file_bytes, file_type if file_type else "png"),
            )
        )
        return parts

    if file_type == "pdf":
        for page in extract_pdf_pages(file_bytes, render_pages=True):
            page_number = int(page["page_number"])
            page_text = str(page.get("text") or "").strip()
            text_parts, chunk_index = _text_parts(
                document_id=document_id,
                source=filename,
                file_type=file_type,
                modality="page_text",
                text=page_text,
                chunk_index_start=chunk_index,
                page_number=page_number,
                user_id=user_id,
            )
            parts.extend(text_parts)
            image_bytes = page.get("image_bytes")
            if image_bytes:
                parts.append(
                    _image_part(
                        document_id=document_id,
                        source=filename,
                        file_type=file_type,
                        modality="page_image",
                        image_bytes=image_bytes,
                        chunk_index=chunk_index,
                        user_id=user_id,
                        page_number=page_number,
                        content=(
                            f"Rendered page {page_number} from {filename}. Use this "
                            "page image for scanned text, visual tables, charts, "
                            "diagrams, labels, and layout."
                        ),
                    )
                )
                chunk_index += 1
        return parts

    if file_type == "docx":
        text_parts, _ = _text_parts(
            document_id=document_id,
            source=filename,
            file_type=file_type,
            modality="text",
            text=extract_docx_text(file_bytes),
            chunk_index_start=chunk_index,
            user_id=user_id,
        )
        return text_parts

    if file_type == "pptx":
        rendered_slide_images = {
            int(rendered["slide_number"]): rendered.get("image_bytes")
            for rendered in render_pptx_slides(file_bytes)
            if rendered.get("image_bytes")
        }
        for slide in extract_pptx_slides(file_bytes):
            slide_number = int(slide["slide_number"])
            text_parts, chunk_index = _text_parts(
                document_id=document_id,
                source=filename,
                file_type=file_type,
                modality="slide_text",
                text=str(slide.get("text") or ""),
                chunk_index_start=chunk_index,
                slide_number=slide_number,
                user_id=user_id,
            )
            parts.extend(text_parts)
            rendered_image = rendered_slide_images.get(slide_number)
            if rendered_image:
                parts.append(
                    _image_part(
                        document_id=document_id,
                        source=filename,
                        file_type=file_type,
                        modality="slide_render",
                        image_bytes=rendered_image,
                        chunk_index=chunk_index,
                        user_id=user_id,
                        slide_number=slide_number,
                        content=(
                            f"Rendered full slide {slide_number} from {filename}. "
                            "Use this slide image for charts, layouts, labels, "
                            "tables, diagrams, and other visual evidence."
                        ),
                        suffix="png",
                        metadata={"render_source": "libreoffice"},
                    )
                )
                chunk_index += 1
            for image_blob in slide.get("image_blobs") or []:
                parts.append(
                    _image_part(
                        document_id=document_id,
                        source=filename,
                        file_type=file_type,
                        modality="slide_image",
                        image_bytes=image_blob,
                        chunk_index=chunk_index,
                        user_id=user_id,
                        slide_number=slide_number,
                        content=(
                            f"Embedded image from slide {slide_number} of {filename}. "
                            "Use this image for visual evidence from the presentation."
                        ),
                        suffix=_image_suffix(image_blob),
                    )
                )
                chunk_index += 1
        return parts

    if file_type in {"xls", "xlsx"}:
        for sheet in extract_excel_sheets(file_bytes):
            text_parts, chunk_index = _text_parts(
                document_id=document_id,
                source=filename,
                file_type=file_type,
                modality="table",
                text=str(sheet.get("text") or ""),
                chunk_index_start=chunk_index,
                sheet_name=str(sheet.get("sheet_name") or ""),
                user_id=user_id,
            )
            parts.extend(text_parts)
        return parts

    if file_type == "csv":
        text_parts, _ = _text_parts(
            document_id=document_id,
            source=filename,
            file_type=file_type,
            modality="table",
            text=extract_csv_text(file_bytes),
            chunk_index_start=chunk_index,
            user_id=user_id,
        )
        return text_parts

    if file_type in {"txt", "md", "markdown"}:
        text_parts, _ = _text_parts(
            document_id=document_id,
            source=filename,
            file_type=file_type,
            modality="text",
            text=extract_text_file(file_bytes),
            chunk_index_start=chunk_index,
            user_id=user_id,
        )
        return text_parts

    return []


def _embedding_for_part(part: DocumentPart, embedding_client) -> list[float]:
    """Create the embedding vector for a normalized part.

    Args:
        part: Normalized text-backed or image-backed MRAG part.
        embedding_client: Client exposing ``embed_text`` and ``embed_image``.

    Returns:
        Dense vector for Qdrant.

    Behavior:
        If ``part.image_path`` is set, the saved image bytes are read and passed
        to ``embed_image(..., input_type="document")``. Otherwise ``part.content``
        is passed to ``embed_text(..., input_type="document")``.
    """
    if part.image_path:
        return _embed_image(
            embedding_client,
            Path(part.image_path).read_bytes(),
            input_type="document",
        )
    return _embed_text(embedding_client, part.content, input_type="document")


def _document_metadata(parts: list[DocumentPart]) -> dict[str, Any]:
    """Build the JSON metadata stored for one uploaded document.

    Args:
        parts: Non-empty list of parts from a single upload.

    Returns:
        Metadata containing the stable document id, user id, chunk count,
        modalities, file type, source page/slide/sheet lists, and saved image
        paths. This supports the sidebar document library and delete cleanup.
    """
    return {
        "source": "furnacemind_knowledge",
        "document_id": parts[0].document_id,
        "user_id": parts[0].user_id or "",
        "chunk_count": len(parts),
        "modalities": sorted({part.modality for part in parts}),
        "file_type": parts[0].file_type,
        "pages": sorted(
            {part.page_number for part in parts if part.page_number is not None}
        ),
        "slides": sorted(
            {part.slide_number for part in parts if part.slide_number is not None}
        ),
        "sheets": sorted({part.sheet_name for part in parts if part.sheet_name}),
        "image_paths": [
            part.image_path for part in parts if part.image_path is not None
        ],
    }


def _persist_document_metadata(
    *,
    document_repository: Any | None,
    user_id: str | None,
    filename: str,
    knowledge_store,
    parts: list[DocumentPart],
) -> Any | None:
    """Create or reuse the SQL document row for an indexed upload.

    Args:
        document_repository: Repository with ``list_documents`` and
            ``create_document`` methods, or ``None`` when SQL persistence is not
            available.
        user_id: Current user id. Without this, no SQL document row is written.
        filename: Original uploaded filename.
        knowledge_store: Qdrant store providing ``collection_name``.
        parts: Parts successfully prepared for indexing.

    Returns:
        The created/reused document row, or ``None`` when persistence is skipped
        or fails. Failures are logged and do not block Qdrant indexing.
    """
    if document_repository is None or not user_id or not parts:
        return None

    try:
        metadata = _document_metadata(parts)
        if hasattr(document_repository, "list_documents"):
            existing_documents = document_repository.list_documents(user_id=user_id)
            for document in existing_documents:
                existing_metadata = getattr(document, "metadata_json", None)
                if not isinstance(existing_metadata, dict):
                    continue
                same_document = (
                    existing_metadata.get("document_id") == parts[0].document_id
                )
                same_collection = (
                    getattr(document, "qdrant_collection", None)
                    == knowledge_store.collection_name
                )
                if same_document and same_collection:
                    return document
        return document_repository.create_document(
            user_id=user_id,
            filename=filename,
            file_type=parts[0].file_type,
            qdrant_collection=knowledge_store.collection_name,
            qdrant_point_ids=[part.point_id for part in parts],
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning("MRAG document metadata save failed for %s: %s", filename, exc)
        return None


def _persist_chunk_metadata(
    *,
    chunk_repository: Any | None,
    document: Any | None,
    knowledge_store,
    parts: list[DocumentPart],
) -> None:
    """Persist SQL chunk rows for the indexed parts when available.

    Args:
        chunk_repository: Repository exposing ``create_chunks``, or ``None``.
        document: SQL document row returned by ``_persist_document_metadata``.
        knowledge_store: Qdrant store providing ``collection_name``.
        parts: Parts written to Qdrant.

    Returns:
        None. Failures are logged and do not roll back Qdrant indexing.
    """
    if chunk_repository is None or document is None or not parts:
        return
    try:
        chunk_repository.create_chunks(
            document=document,
            parts=parts,
            qdrant_collection=knowledge_store.collection_name,
        )
    except Exception as exc:
        logger.warning(
            "MRAG chunk metadata save failed for %s: %s",
            getattr(document, "document_id", "unknown"),
            exc,
        )


def process_file(
    file,
    knowledge_store,
    embedding_client,
    *,
    user_id: str | None = None,
    document_repository: Any | None = None,
    chunk_repository: Any | None = None,
) -> list[DocumentPart]:
    """Index one uploaded file into the multimodal knowledge collection.

    This is the public ingestion entrypoint used by the Streamlit ``Chunk &
    Index`` button. It performs parsing, modality-aware embedding, Qdrant upsert,
    and optional SQL metadata persistence.

    Args:
        file: Streamlit upload object. The object must expose ``read()`` and
            should expose ``name`` for source citations.
        knowledge_store: Knowledge vector store with ``client``,
            ``collection_name``, ``embedding_dim``, and optionally
            ``_ensure_collection``.
        embedding_client: Multimodal embedding client used for text and image
            vectors.
        user_id: Optional owner id copied to Qdrant payloads and SQL metadata.
        document_repository: Optional SQL repository for document-level metadata.
        chunk_repository: Optional SQL repository for chunk-level metadata.

    Returns:
        List of ``DocumentPart`` objects that were embedded and upserted. Empty
        when the file type is unsupported or no extractable content is found.

    Raises:
        ValueError: If any generated embedding dimension differs from the Qdrant
        collection dimension.
        Any exception raised by Qdrant upsert or the embedding provider. SQL
        metadata failures are caught and logged separately.
    """
    filename = getattr(file, "name", "upload")
    file_bytes = _read_upload_bytes(file)
    parts = build_document_parts(filename, file_bytes, user_id=user_id)
    if not parts:
        return []

    points: list[PointStruct] = []
    for part in parts:
        embedding = _embedding_for_part(part, embedding_client)
        if len(embedding) != knowledge_store.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match expected "
                f"{knowledge_store.embedding_dim}"
            )
        points.append(
            PointStruct(
                id=part.point_id,
                vector=embedding,
                payload=part.payload(),
            )
        )

    if hasattr(knowledge_store, "_ensure_collection"):
        knowledge_store._ensure_collection()

    knowledge_store.client.upsert(
        collection_name=knowledge_store.collection_name,
        points=points,
        wait=True,
    )
    document = _persist_document_metadata(
        document_repository=document_repository,
        user_id=user_id,
        filename=filename,
        knowledge_store=knowledge_store,
        parts=parts,
    )
    _persist_chunk_metadata(
        chunk_repository=chunk_repository,
        document=document,
        knowledge_store=knowledge_store,
        parts=parts,
    )
    return parts
