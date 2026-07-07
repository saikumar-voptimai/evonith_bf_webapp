"""FurnaceMind run event service."""

from __future__ import annotations

from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.repositories.furnacemind_run_repository import FurnaceMindRunRepository, RunEventRecord
from apps.backend_api.app.services.furnacemind_safety_service import FurnaceMindSafetyService


class FurnaceMindEventService:
    """Append and list redacted run events for polling."""

    def __init__(
        self,
        *,
        repository: FurnaceMindRunRepository,
        settings: BackendSettings | None = None,
        safety: FurnaceMindSafetyService | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings or load_backend_settings()
        self.safety = safety or FurnaceMindSafetyService(self.settings)

    def append(
        self,
        *,
        run_id: str,
        conversation_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> RunEventRecord:
        event = self.repository.append_event(
            {
                "run_id": run_id,
                "conversation_id": conversation_id,
                "event_type": event_type,
                "payload": self.safety.redact(payload or {}),
            }
        )
        self.repository.trim_events(run_id, max_events=self.settings.furnacemind_max_events_per_run)
        return event

    def list_events(self, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return [
            self.response(event)
            for event in self.repository.list_events(
                run_id,
                limit=limit or self.settings.furnacemind_max_events_per_run,
            )
        ]

    @staticmethod
    def response(event: RunEventRecord) -> dict[str, Any]:
        return {
            "id": event.id,
            "run_id": event.run_id,
            "conversation_id": event.conversation_id,
            "event_type": event.event_type,
            "sequence": event.sequence,
            "payload": event.payload,
            "created_at": event.created_at,
        }
