"""Tests for FurnaceMind service components."""

from __future__ import annotations

import sys

import pytest

from apps.backend_api.app.core.config import BackendSettings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.repositories.furnacemind_conversation_repository import FurnaceMindConversationRepository
from apps.backend_api.app.repositories.furnacemind_document_repository import FurnaceMindDocumentRepository
from apps.backend_api.app.repositories.furnacemind_run_repository import FurnaceMindRunRepository
from apps.backend_api.app.services.furnacemind_document_service import FurnaceMindDocumentService, ParsedDocumentUpload
from apps.backend_api.app.services.furnacemind_event_service import FurnaceMindEventService
from apps.backend_api.app.services.furnacemind_llm_service import FurnaceMindLLMService
from apps.backend_api.app.services.furnacemind_memory_service import FurnaceMindMemoryService
from apps.backend_api.app.services.furnacemind_prompt_service import FurnaceMindPromptService
from apps.backend_api.app.services.furnacemind_retrieval_service import FurnaceMindRetrievalService
from apps.backend_api.app.services.furnacemind_safety_service import FurnaceMindSafetyService
from apps.backend_api.app.services.furnacemind_service import FurnaceMindService
from apps.backend_api.app.services.furnacemind_tool_executor import FurnaceMindToolExecutor
from apps.backend_api.app.services.furnacemind_tool_registry import FurnaceMindToolRegistry


def _settings(tmp_path, **overrides):
    return BackendSettings(
        backend_env="test",
        auth_secret_key="test-secret",
        furnacemind_database_url=f"sqlite:///{(tmp_path / 'fm.db').as_posix()}",
        **overrides,
    )


def test_safety_redacts_limits_and_blocks_unsafe(tmp_path, monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = _settings(tmp_path, furnacemind_max_message_chars=5, furnacemind_max_prompt_chars=10)
    safety = FurnaceMindSafetyService(settings)
    payload = {"password": "secret", "nested": {"token": "abc"}, "path": str(tmp_path / "runtime" / "x")}

    redacted = safety.redact(payload)

    assert redacted["password"] == "[REDACTED]"
    assert redacted["nested"]["token"] == "[REDACTED]"
    assert "[REDACTED]" in redacted["path"]
    with pytest.raises(ApiError, match="FurnaceMind message"):
        safety.enforce_message_limit("toolong")
    capped, warnings = safety.cap_prompt("01234567890")
    assert capped == "0123456789"
    assert warnings[0]["code"] == "FURNACEMIND_PROMPT_TOO_LARGE"
    with pytest.raises(ApiError) as exc_info:
        safety.block_unsafe_options({"enable_code_execution": True})
    assert exc_info.value.code == "FURNACEMIND_CODE_EXECUTION_DISABLED"
    with pytest.raises(ApiError) as shell_exc:
        safety.block_unsafe_options({"enable_shell_execution": True})
    assert shell_exc.value.code == "FURNACEMIND_SHELL_EXECUTION_DISABLED"
    with pytest.raises(ApiError) as raw_exc:
        safety.ensure_raw_docs_allowed(requested=True)
    assert raw_exc.value.code == "FURNACEMIND_RAW_DOCUMENTS_NOT_ALLOWED"
    assert payload["password"] == "secret"


def test_document_service_upload_extract_index_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = _settings(
        tmp_path,
        furnacemind_require_auth=False,
        furnacemind_memory_enabled=True,
        furnacemind_vector_backend="fake",
        furnacemind_max_extracted_chars=12,
    )
    repository = FurnaceMindDocumentRepository(settings.furnacemind_database_url)
    service = FurnaceMindDocumentService(repository=repository, settings=settings)
    memory = FurnaceMindMemoryService(settings)

    result = service.store_document(
        upload=ParsedDocumentUpload("..\\unsafe name.txt", "text/plain", b"blast furnace document"),
        current_user={"id": "u1"},
        request_id="req",
    )
    indexed = service.index_document(result["id"], current_user={"id": "u1"}, memory_service=memory)
    pdf = service.store_document(
        upload=ParsedDocumentUpload("manual.pdf", "application/pdf", b"%PDF"),
        current_user={"id": "u1"},
        request_id="req",
    )

    assert result["filename"] == "unsafe_name.txt"
    assert result["chunk_count"] == 1
    assert result["metadata"]["extracted_chars"] == 12
    assert indexed["indexed"] is True
    assert pdf["warnings"][0]["code"] == "FURNACEMIND_DOCUMENT_EXTRACTION_UNSUPPORTED"
    assert str(service.upload_dir).startswith(str(tmp_path))
    assert "stored_filename" not in result
    assert service.delete_document(result["id"], current_user={"id": "u1"}) == {"deleted": True}

    with pytest.raises(ApiError) as type_exc:
        service.store_document(
            upload=ParsedDocumentUpload("bad.exe", "application/octet-stream", b"x"),
            current_user={"id": "u1"},
            request_id="req",
        )
    assert type_exc.value.code in {"FURNACEMIND_DOCUMENT_EXTENSION_NOT_ALLOWED", "FURNACEMIND_DOCUMENT_TYPE_NOT_ALLOWED"}


def test_memory_retrieval_prompt_llm_tools_events_and_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    settings = _settings(
        tmp_path,
        furnacemind_require_auth=False,
        furnacemind_memory_enabled=True,
        furnacemind_vector_backend="fake",
        furnacemind_tools_enabled=True,
        furnacemind_llm_enabled=True,
        furnacemind_enable_provider_calls=True,
        furnacemind_provider="mock",
    )
    conversations = FurnaceMindConversationRepository(settings.furnacemind_database_url)
    documents = FurnaceMindDocumentRepository(settings.furnacemind_database_url)
    runs = FurnaceMindRunRepository(settings.furnacemind_database_url)
    safety = FurnaceMindSafetyService(settings)
    memory = FurnaceMindMemoryService(settings, safety=safety)
    document_service = FurnaceMindDocumentService(repository=documents, settings=settings, safety=safety)
    uploaded = document_service.store_document(
        upload=ParsedDocumentUpload("sop.md", "text/markdown", b"pressure stability evidence"),
        current_user={"id": "u1"},
        request_id="req",
    )
    document_service.index_document(uploaded["id"], current_user={"id": "u1"}, memory_service=memory)
    convo = conversations.create_conversation({"title": "C", "owner_id": "u1"})
    conversations.create_message({"conversation_id": convo.id, "role": "user", "content": "history"})
    retrieval = FurnaceMindRetrievalService(
        conversation_repository=conversations,
        document_repository=documents,
        memory_service=memory,
        settings=settings,
        safety=safety,
    )
    context = retrieval.build_context(
        conversation_id=convo.id,
        message="pressure",
        document_ids=[uploaded["id"]],
        owner_id="u1",
    )
    prompt, prompt_warnings = FurnaceMindPromptService(settings=settings, safety=safety).build_prompt(
        message="pressure",
        context=context,
        tool_results=[],
    )
    llm = FurnaceMindLLMService(settings)
    registry = FurnaceMindToolRegistry(settings)
    tool = FurnaceMindToolExecutor(registry=registry, settings=settings, safety=safety).execute(
        "data_summary",
        {"rows": [{"a": 1}]},
    )
    run = runs.create_run({"conversation_id": convo.id, "owner_id": "u1"})
    event_service = FurnaceMindEventService(repository=runs, settings=settings, safety=safety)
    event_service.append(run_id=run.id, conversation_id=convo.id, event_type="warning", payload={"token": "abc"})
    service = FurnaceMindService(settings=settings)
    created = service.create_conversation({"title": "API"}, {"id": "u1"})
    run_response = service.create_run(
        created["id"],
        {
            "message": "Summarise pressure",
            "document_ids": [uploaded["id"]],
            "allow_llm": True,
            "options": {"tool_calls": [{"name": "data_summary", "input": {"rows": [{"a": 1}]}}], "export": True},
        },
        {"id": "u1"},
        request_id="req",
    )

    assert context["evidence"]
    assert "OPENAI_API_KEY" not in prompt
    assert prompt_warnings == []
    assert llm.generate(prompt).provider_name == "mock"
    assert tool["output"]["row_count"] == 1
    assert event_service.list_events(run.id)[0]["payload"]["token"] == "[REDACTED]"
    assert run_response["status"] == "completed"
    status = service.get_run_status(run_response["id"], {"id": "u1"})
    assert status["result_message"]["role"] == "assistant"
    assert status["artifacts"]
    feedback = service.submit_message_feedback(status["result_message"]["id"], {"helpful": True}, {"id": "u1"})
    assert feedback["message_id"] == status["result_message"]["id"]


def test_llm_disabled_provider_errors_and_no_openai_import(tmp_path, monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("openai", None)
    settings = _settings(tmp_path)
    llm = FurnaceMindLLMService(settings)

    assert llm.is_available() is False
    with pytest.raises(ApiError) as exc_info:
        llm.generate("prompt")
    assert exc_info.value.code == "FURNACEMIND_LLM_DISABLED"
    assert "openai" not in sys.modules

    configured = _settings(
        tmp_path,
        furnacemind_llm_enabled=True,
        furnacemind_enable_provider_calls=True,
        furnacemind_provider="openai",
    )
    with pytest.raises(ApiError) as provider_exc:
        FurnaceMindLLMService(configured).generate("prompt")
    assert provider_exc.value.code == "FURNACEMIND_LLM_PROVIDER_NOT_CONFIGURED"


def test_tools_disabled_unknown_and_unsafe(tmp_path):
    disabled = FurnaceMindToolExecutor(settings=_settings(tmp_path))
    with pytest.raises(ApiError) as disabled_exc:
        disabled.execute("data_summary", {"rows": []})
    assert disabled_exc.value.code == "FURNACEMIND_TOOLS_DISABLED"

    enabled = FurnaceMindToolExecutor(settings=_settings(tmp_path, furnacemind_tools_enabled=True))
    with pytest.raises(ApiError) as unknown_exc:
        enabled.execute("unknown", {})
    assert unknown_exc.value.code == "FURNACEMIND_TOOL_NOT_ALLOWED"
    with pytest.raises(ApiError) as unsafe_exc:
        enabled.execute("execute_python_plot", {})
    assert unsafe_exc.value.code == "FURNACEMIND_TOOL_UNSAFE"
