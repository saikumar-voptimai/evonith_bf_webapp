"""Prompt construction for Copilot LLM calls."""

from __future__ import annotations

import json
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.services.copilot_safety_service import CopilotSafetyService

_SYSTEM = (
    "You are Evonith BF AI Copilot. Give concise, operationally safe blast "
    "furnace analysis using only the supplied context. Do not claim access to "
    "hidden plant systems. Do not request code execution."
)


class CopilotPromptService:
    """Build capped provider prompts from redacted context."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        safety: CopilotSafetyService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.safety = safety or CopilotSafetyService(self.settings)

    def build_prompt(self, context: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        payload = self.safety.redact(context)
        prompt = (
            f"{_SYSTEM}\n\n"
            "Return markdown with: Summary, Key Signals, Operator Notes, Risks.\n\n"
            f"Context JSON:\n{json.dumps(payload, indent=2, sort_keys=True, default=str)}"
        )
        return self.safety.enforce_prompt_limit(prompt)
