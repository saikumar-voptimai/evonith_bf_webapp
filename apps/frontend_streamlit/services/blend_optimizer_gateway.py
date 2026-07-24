"""Gateway selection for the rich Blend Optimizer page."""

from __future__ import annotations

import uuid
from typing import Any, Protocol

from apps.frontend_streamlit.config.frontend_settings import is_backend_api_enabled
from apps.frontend_streamlit.services.api_client import ApiClient
from apps.frontend_streamlit.services import blend_optimizer_api


class BlendOptimizerGateway(Protocol):
    def get_catalog(self) -> Any: ...
    def create_context(self, request: dict[str, Any], *, idempotency_key: str) -> Any: ...
    def get_context(self, context_id: str) -> Any: ...
    def get_diagnostics(self, context_id: str) -> Any: ...
    def get_preferences(self) -> Any: ...
    def update_preferences(self, request: dict[str, Any], *, idempotency_key: str | None = None) -> Any: ...
    def create_run(self, request: dict[str, Any], *, idempotency_key: str) -> Any: ...
    def get_run(self, run_id: str) -> Any: ...
    def get_run_events(self, run_id: str, *, after: int | None = None) -> Any: ...
    def cancel_run(self, run_id: str) -> Any: ...
    def evaluate_manual_blend(self, run_id: str, request: dict[str, Any], *, idempotency_key: str) -> Any: ...
    def download_artifact(self, artifact_id: str) -> bytes: ...
    def run_lp_baseline(self, context_id: str, context_version: str, scenario: dict[str, Any], *, idempotency_key: str | None = None) -> Any: ...
    def run_total_cost_optimizer(self, context_id: str, context_version: str, scenario: dict[str, Any], *, idempotency_key: str | None = None) -> Any: ...


class ApiBlendOptimizerGateway:
    def __init__(self, *, token: str | None = None, client: ApiClient | None = None) -> None:
        self.token = token
        self.client = client

    def get_catalog(self) -> Any:
        return blend_optimizer_api.get_blend_optimizer_catalog(token=self.token, client=self.client)

    def create_context(self, request: dict[str, Any], *, idempotency_key: str) -> Any:
        return blend_optimizer_api.create_blend_optimizer_context(request, idempotency_key=idempotency_key, token=self.token, client=self.client)

    def get_context(self, context_id: str) -> Any:
        return blend_optimizer_api.get_blend_optimizer_context_by_id(context_id, token=self.token, client=self.client)

    def get_diagnostics(self, context_id: str) -> Any:
        return blend_optimizer_api.get_blend_optimizer_context_diagnostics(context_id, token=self.token, client=self.client)

    def get_preferences(self) -> Any:
        return blend_optimizer_api.get_blend_optimizer_preferences(token=self.token, client=self.client)

    def update_preferences(self, request: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        return blend_optimizer_api.update_blend_optimizer_preferences(request, token=self.token, client=self.client)

    def create_run(self, request: dict[str, Any], *, idempotency_key: str) -> Any:
        return blend_optimizer_api.create_blend_optimizer_run(request, idempotency_key=idempotency_key, token=self.token, client=self.client)

    def get_run(self, run_id: str) -> Any:
        return blend_optimizer_api.get_blend_optimizer_run(run_id, token=self.token, client=self.client)

    def get_run_events(self, run_id: str, *, after: int | None = None) -> Any:
        return blend_optimizer_api.get_blend_optimizer_run_events(run_id, after=after, token=self.token, client=self.client)

    def cancel_run(self, run_id: str) -> Any:
        return blend_optimizer_api.cancel_blend_optimizer_run(run_id, token=self.token, client=self.client)

    def evaluate_manual_blend(self, run_id: str, request: dict[str, Any], *, idempotency_key: str) -> Any:
        return blend_optimizer_api.create_blend_optimizer_manual_evaluation(run_id, request, idempotency_key=idempotency_key, token=self.token, client=self.client)

    def download_artifact(self, artifact_id: str) -> bytes:
        return blend_optimizer_api.download_blend_optimizer_artifact(artifact_id, token=self.token, client=self.client)

    def run_lp_baseline(self, context_id: str, context_version: str, scenario: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        return self.create_run(
            {"mode": "lp_baseline", "context_id": context_id, "expected_context_version": context_version, "scenario": scenario, "options": {"algorithm_version": "bmo_lp_legacy_v1"}},
            idempotency_key=idempotency_key or f"bmo-lp-{uuid.uuid4().hex}",
        )

    def run_total_cost_optimizer(self, context_id: str, context_version: str, scenario: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        return self.create_run(
            {"mode": "total_cost", "context_id": context_id, "expected_context_version": context_version, "scenario": scenario, "options": {"algorithm_version": "bmo_total_cost_de_legacy_v1", "iteration_budget_id": "standard"}},
            idempotency_key=idempotency_key or f"bmo-total-{uuid.uuid4().hex}",
        )


class DirectBlendOptimizerGateway:
    """Temporary rollback gateway; imports canonical BMO modules lazily."""

    def get_catalog(self) -> Any:
        return {"catalog_version": "direct_bmo_legacy", "advisory_only": True, "operator_review_required": True}

    def create_context(self, request: dict[str, Any], *, idempotency_key: str) -> Any:
        from furnace_data.bmo.data import EvonithBmoContextProvider
        provider = EvonithBmoContextProvider()
        ores, diagnostics = provider.build_ore_inputs()
        return {"context_id": "direct", "context_version": "direct", "eligible_materials": ores, "source_provenance": diagnostics}

    def get_context(self, context_id: str) -> Any:
        return self.create_context({}, idempotency_key="direct")

    def get_diagnostics(self, context_id: str) -> Any:
        return {}

    def get_preferences(self) -> Any:
        return {"version": 0, "preferences": {}}

    def update_preferences(self, request: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        return {"version": 0, "preferences": request.get("preferences") or {}}

    def create_run(self, request: dict[str, Any], *, idempotency_key: str) -> Any:
        raise RuntimeError("Direct BMO run execution is owned by the rich page legacy path.")

    def get_run(self, run_id: str) -> Any:
        raise RuntimeError("Direct BMO run status is not available.")

    def get_run_events(self, run_id: str, *, after: int | None = None) -> Any:
        return []

    def cancel_run(self, run_id: str) -> Any:
        return {"run_id": run_id, "status": "cancelled"}

    def evaluate_manual_blend(self, run_id: str, request: dict[str, Any], *, idempotency_key: str) -> Any:
        raise RuntimeError("Direct manual evaluation is owned by the rich page legacy path.")

    def download_artifact(self, artifact_id: str) -> bytes:
        raise RuntimeError("Direct artifact downloads are not available.")

    def run_lp_baseline(self, context_id: str, context_version: str, scenario: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        raise RuntimeError("Direct BMO LP execution is owned by the rich page legacy path.")

    def run_total_cost_optimizer(self, context_id: str, context_version: str, scenario: dict[str, Any], *, idempotency_key: str | None = None) -> Any:
        raise RuntimeError("Direct BMO total-cost execution is owned by the rich page legacy path.")


def get_blend_optimizer_gateway(*, token: str | None = None, client: ApiClient | None = None) -> BlendOptimizerGateway:
    if is_backend_api_enabled("blend_optimizer"):
        return ApiBlendOptimizerGateway(token=token, client=client)
    return DirectBlendOptimizerGateway()
