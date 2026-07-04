"""FurnaceMind document upload, extraction, and indexing service."""

from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import json
import logging
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from starlette.requests import Request

from app.core.config import BackendSettings, load_backend_settings
from app.core.errors import ApiError
from app.repositories.furnacemind_document_repository import (
    DocumentRecord,
    FurnaceMindDocumentRepository,
)
from app.services.furnacemind_safety_service import FurnaceMindSafetyService, warning
from furnace_data.runtime_paths import runtime_path

log = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ParsedDocumentUpload:
    filename: str
    content_type: str
    content: bytes


class FurnaceMindDocumentService:
    """Validate, store, and extract FurnaceMind documents under runtime."""

    def __init__(
        self,
        *,
        repository: FurnaceMindDocumentRepository | None = None,
        settings: BackendSettings | None = None,
        safety: FurnaceMindSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        database_url = self.settings.furnacemind_database_url.strip() or None
        self.repository = repository or FurnaceMindDocumentRepository(database_url=database_url)
        self.safety = safety or FurnaceMindSafetyService(self.settings)

    @property
    def upload_dir(self) -> Path:
        path = runtime_path("uploads", "furnacemind", "documents")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def extraction_dir(self) -> Path:
        path = runtime_path("furnacemind", "documents")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_size_bytes(self) -> int:
        return int(self.settings.furnacemind_max_document_mb) * 1024 * 1024

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        basename = Path(str(filename or "")).name.strip()
        if not basename or basename in {".", ".."}:
            raise ApiError("FURNACEMIND_DOCUMENT_PATH_INVALID", "Document filename is invalid.", 422)
        stem = Path(basename).stem
        suffix = Path(basename).suffix.lower()
        safe_stem = _SAFE_NAME_RE.sub("_", stem).strip("._") or "document"
        safe_suffix = re.sub(r"[^a-z0-9.]+", "", suffix)
        return f"{safe_stem}{safe_suffix}"[:180]

    def _validate_upload(self, safe_name: str, content_type: str, size_bytes: int) -> None:
        if not self.settings.furnacemind_documents_enabled:
            raise ApiError("FURNACEMIND_DOCUMENT_UPLOAD_FAILED", "FurnaceMind documents are disabled.", 403)
        if size_bytes > self.max_size_bytes:
            raise ApiError(
                "FURNACEMIND_DOCUMENT_TOO_LARGE",
                "Document exceeds the configured size limit.",
                status_code=413,
                details={"max_document_mb": self.settings.furnacemind_max_document_mb},
            )
        extension = Path(safe_name).suffix.lower()
        allowed_extensions = {
            ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
            for ext in self.settings.furnacemind_allowed_document_extensions
        }
        if extension not in allowed_extensions:
            raise ApiError(
                "FURNACEMIND_DOCUMENT_EXTENSION_NOT_ALLOWED",
                "Document extension is not allowed.",
                status_code=415,
                details={"extension": extension},
            )
        allowed_types = {item.lower() for item in self.settings.furnacemind_allowed_document_types}
        if content_type.lower() not in allowed_types:
            raise ApiError(
                "FURNACEMIND_DOCUMENT_TYPE_NOT_ALLOWED",
                "Document content type is not allowed.",
                status_code=415,
                details={"content_type": content_type},
            )

    async def parse_upload_request(self, request: Request) -> ParsedDocumentUpload:
        max_body = self.max_size_bytes + 1024 * 1024
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_body:
                raise ApiError("FURNACEMIND_DOCUMENT_TOO_LARGE", "Document exceeds the configured size limit.", 413)
            chunks.append(chunk)
        body = b"".join(chunks)
        content_type = request.headers.get("content-type", "application/octet-stream")
        if content_type.lower().startswith("multipart/form-data"):
            return self._parse_multipart(body=body, content_type=content_type)
        filename = request.headers.get("x-filename") or request.query_params.get("filename") or "document.txt"
        return ParsedDocumentUpload(
            filename=str(filename),
            content_type=content_type.split(";", 1)[0].strip() or "application/octet-stream",
            content=body,
        )

    @staticmethod
    def _parse_multipart(*, body: bytes, content_type: str) -> ParsedDocumentUpload:
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        for part in message.iter_parts():
            filename = part.get_filename()
            if not filename:
                continue
            payload = part.get_payload(decode=True) or b""
            return ParsedDocumentUpload(
                filename=filename,
                content_type=part.get_content_type(),
                content=payload,
            )
        raise ApiError("FURNACEMIND_DOCUMENT_UPLOAD_FAILED", "No document file was found.", 422)

    def store_document(
        self,
        *,
        upload: ParsedDocumentUpload,
        current_user: dict[str, Any] | None,
        request_id: str,
    ) -> dict[str, Any]:
        safe_name = self.sanitize_filename(upload.filename)
        content_type = (upload.content_type or "application/octet-stream").split(";", 1)[0].strip()
        size_bytes = len(upload.content)
        self._validate_upload(safe_name, content_type, size_bytes)

        document_id = f"fmd_{uuid4().hex}"
        stored_filename = f"{document_id}_{safe_name}"
        target_path = (self.upload_dir / stored_filename).resolve()
        target_path.relative_to(self.upload_dir.resolve())
        target_path.write_bytes(upload.content)

        extracted, extract_warnings = self.extract_text(content_type=content_type, filename=safe_name, content=upload.content)
        chunks = self.chunk_text(extracted)
        if extracted:
            text_path = (self.extraction_dir / f"{document_id}.txt").resolve()
            text_path.relative_to(self.extraction_dir.resolve())
            text_path.write_text(extracted, encoding="utf-8")
        record = self.repository.create_document_metadata(
            {
                "id": document_id,
                "filename": safe_name,
                "original_filename": Path(str(upload.filename)).name,
                "stored_filename": stored_filename,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "status": "extracted" if extracted else "uploaded",
                "indexed": False,
                "chunk_count": len(chunks),
                "owner_id": self._owner_id(current_user),
                "metadata": {
                    "warnings": extract_warnings,
                    "extracted_chars": len(extracted),
                },
            }
        )
        if chunks:
            self.repository.replace_chunks(document_id, chunks, {"source": safe_name})
        log.info(
            "furnacemind.document.uploaded request_id=%s document_id=%s owner_id=%s content_type=%s size_bytes=%s",
            request_id,
            record.id,
            record.owner_id,
            record.content_type,
            record.size_bytes,
        )
        return {**self.response(record), "warnings": extract_warnings}

    def extract_text(self, *, content_type: str, filename: str, content: bytes) -> tuple[str, list[dict[str, Any]]]:
        warnings: list[dict[str, Any]] = []
        extension = Path(filename).suffix.lower()
        if content_type in {"text/plain", "text/markdown", "text/csv", "application/json"} or extension in {".txt", ".md", ".csv", ".json"}:
            text = content.decode("utf-8", errors="replace")
            if content_type == "application/json" or extension == ".json":
                try:
                    text = json.dumps(json.loads(text), indent=2, sort_keys=True)
                except json.JSONDecodeError:
                    warnings.append(warning("FURNACEMIND_DOCUMENT_EXTRACTION_FAILED", "JSON document could not be normalized."))
            if len(text) > self.settings.furnacemind_max_extracted_chars:
                text = text[: self.settings.furnacemind_max_extracted_chars]
                warnings.append(
                    warning(
                        "FURNACEMIND_CONTEXT_TOO_LARGE",
                        "Extracted document text was capped.",
                        {"max_extracted_chars": self.settings.furnacemind_max_extracted_chars},
                    )
                )
            return self.safety.redact(text) if self.settings.furnacemind_enable_data_redaction else text, warnings
        if content_type == "application/pdf" or extension == ".pdf":
            warnings.append(
                warning(
                    "FURNACEMIND_DOCUMENT_EXTRACTION_UNSUPPORTED",
                    "PDF text extraction is not enabled in this backend runtime.",
                )
            )
            return "", warnings
        warnings.append(
            warning(
                "FURNACEMIND_DOCUMENT_EXTRACTION_UNSUPPORTED",
                "Document text extraction is unsupported for this content type.",
            )
        )
        return "", warnings

    @staticmethod
    def chunk_text(text: str, *, chunk_size: int = 2000) -> list[str]:
        clean = str(text or "").strip()
        if not clean:
            return []
        return [clean[index : index + chunk_size] for index in range(0, len(clean), chunk_size)]

    def list_documents(self, *, filters: dict[str, Any], current_user: dict[str, Any] | None) -> dict[str, Any]:
        query = dict(filters)
        if current_user is not None:
            query["owner_id"] = self._owner_id(current_user)
        items, total = self.repository.list_documents(query)
        return {
            "items": [self.response(item) for item in items],
            "total": total,
            "limit": min(200, max(1, int(query.get("limit") or 50))),
            "offset": max(0, int(query.get("offset") or 0)),
        }

    def get_document(self, document_id: str, *, current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self.repository.get_document(document_id)
        if record is None:
            raise ApiError("FURNACEMIND_DOCUMENT_NOT_FOUND", "FurnaceMind document not found.", 404)
        self.require_document_access(record, current_user)
        return self.response(record)

    def delete_document(self, document_id: str, *, current_user: dict[str, Any] | None) -> dict[str, Any]:
        record = self.repository.get_document(document_id)
        if record is None:
            raise ApiError("FURNACEMIND_DOCUMENT_NOT_FOUND", "FurnaceMind document not found.", 404)
        self.require_document_access(record, current_user)
        upload_path = (self.upload_dir / record.stored_filename).resolve()
        upload_path.relative_to(self.upload_dir.resolve())
        upload_path.unlink(missing_ok=True)
        text_path = (self.extraction_dir / f"{record.id}.txt").resolve()
        text_path.relative_to(self.extraction_dir.resolve())
        text_path.unlink(missing_ok=True)
        self.repository.delete_document_metadata(record.id)
        return {"deleted": True}

    def index_document(
        self,
        document_id: str,
        *,
        current_user: dict[str, Any] | None,
        memory_service: Any,
    ) -> dict[str, Any]:
        record = self.repository.get_document(document_id)
        if record is None:
            raise ApiError("FURNACEMIND_DOCUMENT_NOT_FOUND", "FurnaceMind document not found.", 404)
        self.require_document_access(record, current_user)
        chunks = self.repository.list_chunks(record.id)
        result = memory_service.index_document_chunks(
            document_id=record.id,
            chunks=[chunk.content for chunk in chunks],
            metadata={"filename": record.filename, "owner_id": record.owner_id},
        )
        updated = self.repository.update_document_status(
            record.id,
            status="indexed" if result.get("indexed") else record.status,
            indexed=bool(result.get("indexed")),
            chunk_count=len(chunks),
            metadata={**record.metadata, "index_result": result},
        )
        return {**self.response(updated or record), "warnings": result.get("warnings", [])}

    def require_document_access(self, document: DocumentRecord, current_user: dict[str, Any] | None) -> None:
        owner_id = self._owner_id(current_user)
        role = str((current_user or {}).get("role") or "").lower()
        if owner_id is None or role == "admin" or document.owner_id == owner_id:
            return
        raise ApiError("FURNACEMIND_FORBIDDEN", "You are not allowed to access this document.", 403)

    @staticmethod
    def _owner_id(current_user: dict[str, Any] | None) -> str | None:
        value = (current_user or {}).get("id")
        return str(value) if value is not None else None

    @staticmethod
    def response(record: DocumentRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "filename": record.filename,
            "original_filename": record.original_filename,
            "content_type": record.content_type,
            "size_bytes": record.size_bytes,
            "status": record.status,
            "indexed": record.indexed,
            "chunk_count": record.chunk_count,
            "owner_id": record.owner_id,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "metadata": record.metadata,
        }
