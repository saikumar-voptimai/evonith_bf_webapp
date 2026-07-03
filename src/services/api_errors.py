"""Frontend-safe API client exceptions."""

from __future__ import annotations

from typing import Any


class FrontendApiError(Exception):
    """Base class for frontend-safe API errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"code={self.error_code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " ".join(parts)


class BackendUnavailableError(FrontendApiError):
    """Raised when the backend cannot be reached."""


class BackendApiHTTPError(FrontendApiError):
    """Raised for non-successful backend HTTP responses."""


class BackendApiValidationError(BackendApiHTTPError):
    """Raised for backend validation errors."""


class BackendApiTimeoutError(FrontendApiError):
    """Raised when the backend request times out."""


class BackendApiDecodeError(FrontendApiError):
    """Raised when a JSON response cannot be decoded."""
