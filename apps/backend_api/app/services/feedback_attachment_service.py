
"""Attachment handling for backend feedback tickets."""

from __future__ import annotations

from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import hashlib
from io import BytesIO
import logging
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from starlette.requests import Request

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.feedback_repository import (
    FeedbackAttachmentRecord,
    FeedbackRepository,
)
from furnace_data.runtime_paths import get_feedback_upload_dir

log = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ParsedUpload:
    filename: str
    content_type: str
    content: bytes


class FeedbackAttachmentService:
    """Validate and store feedback attachments under the runtime directory."""

    def __init__(
        self,
        *,
        repository: FeedbackRepository,
        settings: BackendSettings | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings or load_backend_settings()

    @property
    def upload_dir(self) -> Path:
        path = get_feedback_upload_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_size_bytes(self) -> int:
        return int(self.settings.feedback_max_attachment_mb) * 1024 * 1024

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Return a safe basename preserving a simple extension."""
        basename = Path(str(filename or "")).name.strip()
        if not basename:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_INVALID_FILENAME",
                "Attachment filename is required.",
                status_code=422,
            )
        stem = Path(basename).stem
        suffix = Path(basename).suffix.lower()
        safe_stem = _SAFE_NAME_RE.sub("_", stem).strip("._") or "upload"
        safe_suffix = re.sub(r"[^a-z0-9.]+", "", suffix)
        safe_name = f"{safe_stem}{safe_suffix}"
        if safe_name in {"", ".", ".."}:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_INVALID_FILENAME",
                "Attachment filename is invalid.",
                status_code=422,
            )
        return safe_name[:180]

    def _validate_type(
        self,
        filename: str,
        content_type: str,
        size_bytes: int,
        content: bytes,
    ) -> None:
        if size_bytes > self.max_size_bytes:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_TOO_LARGE",
                "Attachment exceeds the configured size limit.",
                status_code=413,
                details={"max_attachment_mb": self.settings.feedback_max_attachment_mb},
            )
        extension = Path(filename).suffix.lower()
        allowed_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in self.settings.feedback_allowed_attachment_extensions
        }
        if extension not in allowed_extensions:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_EXTENSION_NOT_ALLOWED",
                "Attachment extension is not allowed.",
                status_code=415,
                details={"extension": extension},
            )
        allowed_types = {item.lower() for item in self.settings.feedback_allowed_attachment_types}
        if content_type.lower() not in allowed_types:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_TYPE_NOT_ALLOWED",
                "Attachment content type is not allowed.",
                status_code=415,
                details={"content_type": content_type},
            )
        self._validate_signature(extension=extension, content_type=content_type, content=content)

    @staticmethod
    def _validate_signature(*, extension: str, content_type: str, content: bytes) -> None:
        """Verify basic magic bytes for binary attachment types."""
        lower_type = content_type.lower()
        valid = True
        if lower_type == "image/png" or extension == ".png":
            valid = content.startswith(b"\x89PNG\r\n\x1a\n")
        elif lower_type == "image/jpeg" or extension in {".jpg", ".jpeg"}:
            valid = content.startswith(b"\xff\xd8")
        elif lower_type == "image/webp" or extension == ".webp":
            valid = len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP"
        elif lower_type == "application/pdf" or extension == ".pdf":
            valid = content.startswith(b"%PDF")
        if not valid:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_SIGNATURE_MISMATCH",
                "Attachment content does not match its declared type.",
                status_code=415,
                details={"content_type": content_type, "extension": extension},
            )

    async def parse_upload_request(self, request: Request) -> ParsedUpload:
        """Parse a bounded raw or multipart upload request."""
        max_body = self.max_size_bytes + 1024 * 1024
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_body:
                raise ApiError(
                    "FEEDBACK_ATTACHMENT_TOO_LARGE",
                    "Attachment exceeds the configured size limit.",
                    status_code=413,
                )
            chunks.append(chunk)
        body = b"".join(chunks)
        content_type = request.headers.get("content-type", "application/octet-stream")
        if content_type.lower().startswith("multipart/form-data"):
            return self._parse_multipart(body=body, content_type=content_type)
        filename = request.headers.get("x-filename") or request.query_params.get("filename") or "upload"
        return ParsedUpload(
            filename=str(filename),
            content_type=content_type.split(";", 1)[0].strip() or "application/octet-stream",
            content=body,
        )

    @staticmethod
    def _parse_multipart(*, body: bytes, content_type: str) -> ParsedUpload:
        """Parse first file part from a multipart body using stdlib email."""
        message = BytesParser(policy=policy.default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        for part in message.iter_parts():
            filename = part.get_filename()
            if not filename:
                continue
            payload = part.get_payload(decode=True) or b""
            return ParsedUpload(
                filename=filename,
                content_type=part.get_content_type(),
                content=payload,
            )
        raise ApiError(
            "FEEDBACK_ATTACHMENT_UPLOAD_FAILED",
            "No attachment file was found in the upload request.",
            status_code=422,
        )

    def store_attachment(
        self,
        *,
        ticket_id: str,
        upload: ParsedUpload,
        current_user: dict[str, Any] | None,
        request_id: str,
    ) -> FeedbackAttachmentRecord:
        """Validate, write, and persist one attachment."""
        existing = self.repository.list_attachments(ticket_id)
        if len(existing) >= self.settings.feedback_max_attachments_per_ticket:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_LIMIT_EXCEEDED",
                "Maximum attachments per ticket reached.",
                status_code=409,
            )
        safe_name = self.sanitize_filename(upload.filename)
        size_bytes = len(upload.content)
        content_type = (upload.content_type or "application/octet-stream").split(";", 1)[0].strip()
        self._validate_type(safe_name, content_type, size_bytes, upload.content)
        checksum_sha256 = hashlib.sha256(upload.content).hexdigest()
        attachment_id = f"fba_{uuid4().hex}"
        stored_filename = f"{ticket_id}_{attachment_id}_{safe_name}"
        target_path = (self.upload_dir / stored_filename).resolve()
        try:
            target_path.relative_to(self.upload_dir.resolve())
        except ValueError as exc:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_INVALID_FILENAME",
                "Attachment path is invalid.",
                status_code=400,
            ) from exc
        try:
            target_path.write_bytes(upload.content)
            record = self.repository.add_attachment_metadata(
                {
                    "id": attachment_id,
                    "ticket_id": ticket_id,
                    "filename": safe_name,
                    "original_filename": Path(str(upload.filename)).name,
                    "stored_filename": stored_filename,
                    "content_type": content_type,
                    "size_bytes": size_bytes,
                    "checksum_sha256": checksum_sha256,
                    "storage_status": "stored",
                    "created_by": str(current_user.get("id")) if current_user and current_user.get("id") else None,
                    "created_by_username": str(current_user.get("username")) if current_user and current_user.get("username") else None,
                }
            )
        except ApiError:
            target_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            target_path.unlink(missing_ok=True)
            raise ApiError(
                "FEEDBACK_ATTACHMENT_UPLOAD_FAILED",
                "Attachment upload failed.",
                status_code=500,
            ) from exc
        log.info(
            "feedback_attachment_uploaded request_id=%s ticket_id=%s attachment_id=%s content_type=%s size_bytes=%s",
            request_id,
            ticket_id,
            record.id,
            record.content_type,
            record.size_bytes,
        )
        return record

    def resolve_download_path(self, attachment: FeedbackAttachmentRecord) -> Path:
        """Return safe filesystem path for a stored attachment."""
        path = (self.upload_dir / attachment.stored_filename).resolve()
        try:
            path.relative_to(self.upload_dir.resolve())
        except ValueError as exc:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_INVALID_FILENAME",
                "Attachment path is invalid.",
                status_code=400,
            ) from exc
        if not path.exists() or not path.is_file():
            raise ApiError(
                "FEEDBACK_ATTACHMENT_NOT_FOUND",
                "Attachment file not found.",
                status_code=404,
            )
        return path

    def delete_attachment_file(self, attachment: FeedbackAttachmentRecord) -> None:
        """Remove attachment file if present."""
        path = self.resolve_download_path(attachment)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_DELETE_FAILED",
                "Attachment delete failed.",
                status_code=500,
            ) from exc

    def build_image_preview(self, attachment: FeedbackAttachmentRecord) -> tuple[bytes, str]:
        """Return a bounded image preview for an attachment."""
        if attachment.content_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ApiError(
                "FEEDBACK_ATTACHMENT_PREVIEW_UNSUPPORTED",
                "Preview is only available for image attachments.",
                status_code=415,
            )
        path = self.resolve_download_path(attachment)
        try:
            from PIL import Image
        except ImportError:
            data = path.read_bytes()
            if len(data) > min(self.max_size_bytes, 2 * 1024 * 1024):
                raise ApiError(
                    "FEEDBACK_ATTACHMENT_PREVIEW_UNSUPPORTED",
                    "Image preview generation is unavailable for this file.",
                    status_code=415,
                )
            return data, attachment.content_type
        with Image.open(path) as image:
            image.thumbnail((640, 640))
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="PNG", optimize=True)
            return output.getvalue(), "image/png"
