"""AI Copilot orchestration service."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from apps.backend_api.app.core.config import BackendSettings, load_backend_settings
from apps.backend_api.app.core.errors import ApiError
from apps.backend_api.app.services.compute_artifact_service import ComputeArtifactService
from apps.backend_api.app.services.compute_job_service import compute_job_service
from apps.backend_api.app.services.copilot_anomaly_service import CopilotAnomalyService
from apps.backend_api.app.services.copilot_context_service import CopilotContextService
from apps.backend_api.app.services.copilot_data_service import CopilotDataService
from apps.backend_api.app.services.copilot_llm_service import CopilotLLMService
from apps.backend_api.app.services.copilot_prompt_service import CopilotPromptService
from apps.backend_api.app.services.copilot_safety_service import CopilotSafetyService, warning

log = logging.getLogger(__name__)


class CopilotService:
    """Coordinate Copilot data, anomaly, prompt, LLM, jobs, and artifacts."""

    def __init__(
        self,
        *,
        settings: BackendSettings | None = None,
        data_service: CopilotDataService | None = None,
        anomaly_service: CopilotAnomalyService | None = None,
        safety_service: CopilotSafetyService | None = None,
        context_service: CopilotContextService | None = None,
        prompt_service: CopilotPromptService | None = None,
        llm_service: CopilotLLMService | None = None,
        artifact_service: ComputeArtifactService | None = None,
    ) -> None:
        self.settings = settings or load_backend_settings()
        self.safety = safety_service or CopilotSafetyService(self.settings)
        self.data = data_service or CopilotDataService(settings=self.settings, safety=self.safety)
        self.anomaly = anomaly_service or CopilotAnomalyService(settings=self.settings, data_service=self.data)
        self.context = context_service or CopilotContextService(settings=self.settings, safety=self.safety)
        self.prompts = prompt_service or CopilotPromptService(settings=self.settings, safety=self.safety)
        self.llm = llm_service or CopilotLLMService(self.settings)
        self.artifacts = artifact_service or ComputeArtifactService(self.settings)

    def get_config(self) -> dict[str, Any]:
        provider = self.settings.copilot_provider or None
        warnings = []
        if not self.settings.copilot_llm_enabled:
            warnings.append(warning("COPILOT_LLM_DISABLED", "Copilot LLM calls are disabled by default."))
        return {
            "llm_enabled": self.settings.copilot_llm_enabled,
            "provider_configured": self.llm.is_available(),
            "provider_name": provider,
            "model_name": self.settings.copilot_model or None,
            "data_redaction_enabled": self.settings.copilot_enable_data_redaction,
            "raw_data_to_llm_allowed": self.settings.copilot_allow_raw_data_to_llm,
            "max_context_rows": self.settings.copilot_max_context_rows,
            "max_prompt_chars": self.settings.copilot_max_prompt_chars,
            "max_output_chars": self.settings.copilot_max_output_chars,
            "supported_analysis_modes": ["summary", "anomaly", "burden", "shift_handover"],
            "warnings": warnings,
        }

    def get_recent_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        log.info("copilot.recent_data.requested source=%s", payload.get("source") or "online")
        return self.data.fetch_recent_data(payload)

    def analyze_anomaly(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.anomaly.analyze(payload)
        log.info("copilot.anomaly.completed severity=%s signals=%s", result.get("severity"), len(result.get("signals") or []))
        return result

    def analyze_question(self, payload: dict[str, Any], *, request_id: str | None = None) -> dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ApiError("COPILOT_INPUT_INVALID", "question is required.", status_code=422)
        if bool((payload.get("options") or {}).get("enable_code_execution")):
            raise ApiError("COPILOT_UNSAFE_INPUT", "Copilot code execution is disabled.", status_code=403)

        warnings: list[dict[str, Any]] = []
        data_result = None
        if payload.get("include_recent_data", True):
            data_result = self._data_for_analysis(payload)
            warnings.extend(data_result.get("warnings", []))

        anomaly_result = None
        if payload.get("include_anomaly", True):
            try:
                anomaly_result = self.analyze_anomaly(
                    {
                        "source": payload.get("source"),
                        "start_time": payload.get("start_time"),
                        "end_time": payload.get("end_time"),
                        "input_data": payload.get("input_data"),
                        "columns": None,
                        "options": payload.get("options") or {},
                        "timezone": payload.get("timezone"),
                    }
                )
            except ApiError as exc:
                if exc.code in {"COPILOT_ANOMALY_DATA_EMPTY", "COPILOT_DATA_EMPTY"}:
                    warnings.append(warning(exc.code, exc.message))
                else:
                    raise

        context = self.context.build_context(
            question=question,
            data=data_result,
            anomaly=anomaly_result,
            analysis_mode=str(payload.get("analysis_mode") or "summary"),
            warnings=warnings,
        )
        prompt, prompt_warnings = self.prompts.build_prompt(context)
        warnings.extend(prompt_warnings)

        llm_used = False
        provider_name = None
        model_name = None
        wants_llm = payload.get("allow_llm") is True
        if wants_llm:
            llm_result = self.llm.generate(prompt, request_id=request_id, options=payload.get("options") or {})
            answer = llm_result.text
            provider_name = llm_result.provider_name
            model_name = llm_result.model_name
            llm_used = True
        else:
            if payload.get("allow_llm") is None and self.settings.copilot_llm_enabled and not self.llm.is_available():
                warnings.append(warning("COPILOT_LLM_PROVIDER_NOT_CONFIGURED", "LLM is enabled but no provider is available."))
            answer = self._fallback_answer(question, data_result, anomaly_result, warnings)

        tables = {}
        if data_result:
            tables["recent_data"] = {
                "columns": data_result.get("columns", []),
                "rows": data_result.get("rows", []),
                "row_count": data_result.get("row_count"),
                "returned_rows": data_result.get("returned_rows"),
                "truncated": data_result.get("truncated"),
            }
        if anomaly_result:
            tables.update(anomaly_result.get("tables") or {})

        result = {
            "answer": answer[: self.settings.copilot_max_output_chars],
            "analysis_mode": str(payload.get("analysis_mode") or "summary"),
            "llm_used": llm_used,
            "provider_name": provider_name,
            "model_name": model_name,
            "confidence": 0.75 if llm_used else 0.55,
            "anomaly": anomaly_result,
            "evidence": self._evidence(data_result, anomaly_result),
            "tables": tables,
            "charts": (anomaly_result or {}).get("charts", {}) if self.settings.copilot_enable_plots else {},
            "warnings": warnings,
            "artifacts": [],
            "computed_at": datetime.now(timezone.utc),
        }
        if payload.get("options", {}).get("export") or self._large_result(data_result):
            metadata = self.artifacts.create_json_artifact(
                workflow="copilot",
                filename_prefix="copilot_analysis",
                payload={key: value for key, value in result.items() if key != "artifacts"},
                ttl_hours=self.settings.copilot_artifact_ttl_hours,
            )
            result["artifacts"].append(self.artifacts.response(metadata, "/copilot"))
        log.info("copilot.analysis.completed llm_used=%s warnings=%s", llm_used, len(warnings))
        return result

    def start_analysis_job(self, payload: dict[str, Any], *, request_id: str | None = None):
        return compute_job_service.run_inline(
            workflow="copilot",
            fn=lambda: self.analyze_question(payload, request_id=request_id),
            message="Copilot analysis job queued",
        )

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        job = compute_job_service.get_job(job_id)
        if job is None or job.workflow != "copilot":
            raise ApiError("COPILOT_JOB_NOT_FOUND", "Copilot job not found.", status_code=404)
        return compute_job_service.status(job)

    def _data_for_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("input_data") is not None:
            rows = payload.get("input_data")
            if isinstance(rows, dict) and "rows" in rows:
                rows = rows["rows"]
            return self.data.fetch_recent_data(
                {
                    "source": "input_data",
                    "filters": {"rows": rows if isinstance(rows, list) else [rows]},
                    "limit": min(self.settings.copilot_max_context_rows, self.settings.copilot_max_json_rows),
                }
            )
        return self.data.fetch_recent_data(
            {
                "source": payload.get("source") or "online",
                "start_time": payload.get("start_time"),
                "end_time": payload.get("end_time"),
                "filters": payload.get("options") or {},
                "limit": min(self.settings.copilot_max_context_rows, self.settings.copilot_max_json_rows),
                "timezone": payload.get("timezone"),
            }
        )

    @staticmethod
    def _evidence(data_result: dict[str, Any] | None, anomaly_result: dict[str, Any] | None) -> list[dict[str, Any]]:
        evidence = []
        if data_result:
            summary = data_result.get("summary", {})
            evidence.append(
                {
                    "id": "recent_data",
                    "title": "Recent data summary",
                    "type": "data_summary",
                    "summary": f"{summary.get('row_count', 0)} rows and {summary.get('column_count', 0)} columns summarized.",
                    "metadata": {"truncated": data_result.get("truncated", False)},
                }
            )
        if anomaly_result:
            evidence.append(
                {
                    "id": "anomaly",
                    "title": "Anomaly signals",
                    "type": "anomaly_summary",
                    "summary": f"Overall severity: {anomaly_result.get('severity') or 'unknown'}.",
                    "metadata": {"signal_count": len(anomaly_result.get("signals") or [])},
                }
            )
        return evidence

    @staticmethod
    def _fallback_answer(
        question: str,
        data_result: dict[str, Any] | None,
        anomaly_result: dict[str, Any] | None,
        warnings: list[dict[str, Any]],
    ) -> str:
        data_summary = (data_result or {}).get("summary", {})
        anomaly_severity = (anomaly_result or {}).get("severity") or "not available"
        top_signal = None
        signals = (anomaly_result or {}).get("signals") or []
        if signals:
            top_signal = signals[0].get("name")
        parts = [
            "Copilot non-LLM analysis.",
            f"Question: {question}",
            f"Data window: {data_summary.get('row_count', 0)} rows, {data_summary.get('column_count', 0)} columns.",
            f"Anomaly severity: {anomaly_severity}.",
        ]
        if top_signal:
            parts.append(f"Top signal: {top_signal}.")
        if warnings:
            parts.append(f"Warnings: {len(warnings)} safety or data warning(s).")
        return "\n\n".join(parts)

    def _large_result(self, data_result: dict[str, Any] | None) -> bool:
        if not data_result:
            return False
        return int(data_result.get("row_count") or 0) > self.settings.copilot_job_threshold_rows
