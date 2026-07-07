"""Frontend adapter for API v1 FurnaceMind endpoints."""

from __future__ import annotations

from typing import Any

try:
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response
except ModuleNotFoundError:  # pragma: no cover - repo-root import compatibility
    from apps.frontend_streamlit.services.api_client import ApiClient, get_api_client, unwrap_api_response


def _auth_headers(token: str | None) -> dict[str, str]:
    clean = str(token or "").strip()
    return {"Authorization": f"Bearer {clean}"} if clean else {}


def get_furnacemind_config(token: str | None = None, client: ApiClient | None = None) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/furnacemind/config", headers=_auth_headers(token)))


def create_conversation(
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post("/furnacemind/conversations", json=payload or {}, headers=_auth_headers(token)))


def list_conversations(
    filters: dict[str, Any] | None = None,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/furnacemind/conversations", params=filters or {}, headers=_auth_headers(token)))


def get_conversation(
    conversation_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/furnacemind/conversations/{conversation_id}", headers=_auth_headers(token)))


def update_conversation(
    conversation_id: str,
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.patch(f"/furnacemind/conversations/{conversation_id}", json=payload, headers=_auth_headers(token)))


def archive_conversation(
    conversation_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post(f"/furnacemind/conversations/{conversation_id}/archive", headers=_auth_headers(token)))


def list_messages(
    conversation_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> list[dict[str, Any]]:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/furnacemind/conversations/{conversation_id}/messages", headers=_auth_headers(token)))


def send_message(
    conversation_id: str,
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post(f"/furnacemind/conversations/{conversation_id}/messages", json=payload, headers=_auth_headers(token)))


def start_run(
    conversation_id: str,
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post(f"/furnacemind/conversations/{conversation_id}/runs", json=payload, headers=_auth_headers(token)))


def get_run(
    run_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/furnacemind/runs/{run_id}", headers=_auth_headers(token)))


def get_run_events(
    run_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> list[dict[str, Any]]:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/furnacemind/runs/{run_id}/events", headers=_auth_headers(token)))


def list_documents(
    filters: dict[str, Any] | None = None,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/furnacemind/documents", params=filters or {}, headers=_auth_headers(token)))


def upload_document(
    file: Any,
    metadata: dict[str, Any] | None = None,  # reserved for future API contract
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    filename = getattr(file, "name", None) or getattr(file, "filename", None) or "document.txt"
    content_type = getattr(file, "type", None) or getattr(file, "content_type", None) or "text/plain"
    if hasattr(file, "getvalue"):
        content = file.getvalue()
    elif hasattr(file, "read"):
        content = file.read()
    else:
        content = bytes(file)
    return unwrap_api_response(
        api.upload(
            "/furnacemind/documents",
            filename=str(filename),
            content=content,
            content_type=str(content_type),
            headers=_auth_headers(token),
        )
    )


def get_document(
    document_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.get(f"/furnacemind/documents/{document_id}", headers=_auth_headers(token)))


def delete_document(
    document_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.delete(f"/furnacemind/documents/{document_id}", headers=_auth_headers(token)))


def index_document(
    document_id: str,
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post(f"/furnacemind/documents/{document_id}/index", headers=_auth_headers(token)))


def list_tools(token: str | None = None, client: ApiClient | None = None) -> list[dict[str, Any]]:
    api = client or get_api_client()
    return unwrap_api_response(api.get("/furnacemind/tools", headers=_auth_headers(token)))


def download_artifact_url(artifact_id: str, client: ApiClient | None = None) -> str:
    api = client or get_api_client()
    return f"{api.base_url}/furnacemind/artifacts/{artifact_id}/download"


def submit_message_feedback(
    message_id: str,
    payload: dict[str, Any],
    token: str | None = None,
    client: ApiClient | None = None,
) -> dict[str, Any]:
    api = client or get_api_client()
    return unwrap_api_response(api.post(f"/furnacemind/messages/{message_id}/feedback", json=payload, headers=_auth_headers(token)))
