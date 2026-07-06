"""Tests for FurnaceMind SQLite repositories."""

from __future__ import annotations

from pathlib import Path

from app.repositories.furnacemind_conversation_repository import FurnaceMindConversationRepository
from app.repositories.furnacemind_document_repository import FurnaceMindDocumentRepository
from app.repositories.furnacemind_run_repository import FurnaceMindRunRepository


def _url(path: Path) -> str:
    return f"sqlite:///{path.as_posix()}"


def test_furnacemind_repositories_persist_and_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("EVONITH_RUNTIME_DIR", str(tmp_path / "runtime"))
    database_url = _url(tmp_path / "runtime" / "furnacemind" / "furnacemind.db")
    conversations = FurnaceMindConversationRepository(database_url)
    runs = FurnaceMindRunRepository(database_url)
    documents = FurnaceMindDocumentRepository(database_url)

    convo = conversations.create_conversation({"title": "A", "owner_id": "user-1"})
    other = conversations.create_conversation({"title": "B", "owner_id": "user-2"})
    message = conversations.create_message(
        {"conversation_id": convo.id, "role": "user", "content": "hello"}
    )
    run = runs.create_run({"conversation_id": convo.id, "owner_id": "user-1", "user_message_id": message.id})
    first_event = runs.append_event({"run_id": run.id, "conversation_id": convo.id, "event_type": "run_created"})
    second_event = runs.append_event({"run_id": run.id, "conversation_id": convo.id, "event_type": "run_completed"})
    doc = documents.create_document_metadata(
        {
            "filename": "safe.txt",
            "original_filename": "safe.txt",
            "stored_filename": "fmd_safe.txt",
            "content_type": "text/plain",
            "size_bytes": 5,
            "owner_id": "user-1",
        }
    )

    scoped, total = conversations.list_conversations({"owner_id": "user-1"})

    assert conversations.db_path() is not None
    assert str(conversations.db_path()).startswith(str(tmp_path))
    assert [item.id for item in scoped] == [convo.id]
    assert total == 1
    assert conversations.get_conversation(other.id).owner_id == "user-2"
    assert conversations.archive_conversation(convo.id).archived is True
    assert conversations.list_messages(convo.id)[0].id == message.id
    assert runs.get_run(run.id).user_message_id == message.id
    assert [first_event.sequence, second_event.sequence] == [1, 2]
    assert [event.event_type for event in runs.list_events(run.id)] == ["run_created", "run_completed"]
    assert documents.get_document(doc.id).filename == "safe.txt"
    assert "stored_filename" not in {
        "id": doc.id,
        "filename": doc.filename,
        "owner_id": doc.owner_id,
    }
