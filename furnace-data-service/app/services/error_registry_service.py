"""Safe error-code registry for operational diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ErrorCodeInfo:
    code: str
    category: str
    http_status: int
    severity: str
    message: str
    remediation: str


class ErrorRegistryService:
    """Document stable error-code families without exposing stack traces."""

    def __init__(self) -> None:
        self._codes = self._build()

    def list_codes(self) -> dict[str, Any]:
        items = [asdict(item) for item in sorted(self._codes.values(), key=lambda item: item.code)]
        return {"items": items, "total": len(items)}

    def get_code(self, code: str) -> dict[str, Any]:
        item = self._codes.get(str(code or "").upper())
        if item is None:
            item = ErrorCodeInfo(
                code=str(code or "UNKNOWN"),
                category="unknown",
                http_status=400,
                severity="warning",
                message="Unregistered error code.",
                remediation="Check the request ID in backend logs and update docs if this code is expected.",
            )
        return asdict(item)

    @staticmethod
    def _build() -> dict[str, ErrorCodeInfo]:
        families = {
            "AUTH": (401, "error", "Authenticate again or verify token configuration."),
            "ADMIN": (403, "error", "Use an admin token and verify the requested user resource."),
            "DATA": (400, "warning", "Check data source, query parameters, and export request shape."),
            "DATASET": (400, "warning", "Check dataset id, runtime artifact id, or refresh job status."),
            "FEEDBACK": (400, "warning", "Check ticket id, attachment type, ownership, and status."),
            "MATERIAL_BALANCE": (422, "warning", "Validate material balance input payload."),
            "RECOMMENDATION": (422, "warning", "Validate recommendation input payload."),
            "BLEND_OPTIMIZER": (422, "warning", "Validate blend optimizer payload and model availability."),
            "MODEL": (503, "error", "Verify model registry configuration and optional model artifacts."),
            "COPILOT": (409, "warning", "Check Copilot safety settings, provider configuration, or input caps."),
            "FURNACEMIND": (409, "warning", "Check FurnaceMind auth, memory, LLM, tool, or document settings."),
            "OPS": (403, "error", "Use an admin token for operational endpoints."),
            "RUNTIME": (503, "error", "Check EVONITH_RUNTIME_DIR, permissions, and disk space."),
            "DEPENDENCY": (503, "warning", "Check optional service configuration and dependency status."),
            "JOB": (404, "warning", "Check job id and workflow."),
            "AUDIT": (500, "error", "Check audit database availability and retention settings."),
            "CLEANUP": (409, "warning", "Check cleanup settings and dry-run mode."),
            "METRICS": (403, "warning", "Check metrics settings and admin token."),
        }
        codes: dict[str, ErrorCodeInfo] = {}
        for family, (status, severity, remediation) in families.items():
            code = f"{family}_*"
            codes[code] = ErrorCodeInfo(
                code=code,
                category=family.lower(),
                http_status=status,
                severity=severity,
                message=f"{family} error family.",
                remediation=remediation,
            )
        for code, category, status, severity, message, remediation in [
            ("AUTH_REQUIRED", "auth", 401, "error", "Authentication is required.", "Login and retry with a bearer token."),
            ("FORBIDDEN", "auth", 403, "error", "The current user is not allowed.", "Use an account with the required role."),
            ("VALIDATION_ERROR", "api", 422, "warning", "Request validation failed.", "Fix request body, path, or query parameters."),
            ("INTERNAL_SERVER_ERROR", "api", 500, "critical", "Internal server error.", "Use request ID to inspect redacted backend logs."),
            ("RUNTIME_NOT_READY", "runtime", 503, "error", "Runtime directory is not ready.", "Check runtime directory permissions and free space."),
            ("JOB_NOT_FOUND", "job", 404, "warning", "Job was not found.", "Verify job id and workflow."),
            ("AUDIT_UNAVAILABLE", "audit", 503, "error", "Audit storage is unavailable.", "Check audit SQLite/runtime status."),
            ("CLEANUP_DISABLED", "cleanup", 409, "warning", "Runtime cleanup is disabled.", "Enable EVONITH_CLEANUP_ENABLED for cleanup."),
            ("METRICS_DISABLED", "metrics", 404, "warning", "Metrics are disabled.", "Enable EVONITH_METRICS_ENABLED for metrics."),
        ]:
            codes[code] = ErrorCodeInfo(code, category, status, severity, message, remediation)
        return codes

