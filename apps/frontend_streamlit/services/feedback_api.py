"""Frontend adapter for API v1 feedback endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(token: str | None) -> dict[str, str]:
    clean = str(token or "").strip()
    return {"Authorization": f"Bearer {clean}"} if clean else {}


def get_feedback_config(
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get("/feedback/config", headers=_auth_headers(token))
    )


def list_tickets(
    filters: dict[str, Any] | None = None,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    params = {key: value for key, value in (filters or {}).items() if value not in (None, "", [])}
    return unwrap_api_response(
        api.get("/feedback/tickets", params=params, headers=_auth_headers(token))
    )


def create_ticket(
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post("/feedback/tickets", json=payload, headers=_auth_headers(token))
    )


def get_ticket(
    ticket_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(f"/feedback/tickets/{ticket_id}", headers=_auth_headers(token))
    )


def update_ticket(
    ticket_id: str,
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.patch(
            f"/feedback/tickets/{ticket_id}",
            json=payload,
            headers=_auth_headers(token),
        )
    )


def close_ticket(
    ticket_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            f"/feedback/tickets/{ticket_id}/close",
            json={},
            headers=_auth_headers(token),
        )
    )


def list_comments(
    ticket_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(
            f"/feedback/tickets/{ticket_id}/comments",
            headers=_auth_headers(token),
        )
    )


def add_comment(
    ticket_id: str,
    body: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.post(
            f"/feedback/tickets/{ticket_id}/comments",
            json={"body": body},
            headers=_auth_headers(token),
        )
    )


def list_attachments(
    ticket_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.get(
            f"/feedback/tickets/{ticket_id}/attachments",
            headers=_auth_headers(token),
        )
    )


def upload_attachment(
    ticket_id: str,
    *,
    filename: str,
    content: bytes,
    content_type: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.upload(
            f"/feedback/tickets/{ticket_id}/attachments",
            filename=filename,
            content=content,
            content_type=content_type,
            headers=_auth_headers(token),
        )
    )


def delete_attachment(
    attachment_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> Any:
    api = client or get_api_client()
    return unwrap_api_response(
        api.delete(
            f"/feedback/attachments/{attachment_id}",
            headers=_auth_headers(token),
        )
    )


def download_attachment(
    attachment_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> bytes:
    api = client or get_api_client()
    return api.download(
        f"/feedback/attachments/{attachment_id}/download",
        headers=_auth_headers(token),
    )


def get_attachment_download_url(
    attachment_id: str,
    client: ApiClient | None = None,
) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/feedback/attachments/{attachment_id}/download"
