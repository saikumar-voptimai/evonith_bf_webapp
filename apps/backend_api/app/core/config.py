"""Backend API settings loaded from environment variables."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_CORS_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")


class BackendSettings(BaseSettings):
    """Settings for the independently runnable FastAPI backend."""

    api_prefix: str = Field("/api/v1", validation_alias="EVONITH_API_PREFIX")
    backend_env: str = Field("local", validation_alias="EVONITH_BACKEND_ENV")
    runtime_profile: str = Field("local", validation_alias="EVONITH_RUNTIME_PROFILE")
    edge_mode: bool = Field(False, validation_alias="EVONITH_EDGE_MODE")
    edge_device_type: str = Field("", validation_alias="EVONITH_EDGE_DEVICE_TYPE")
    backend_profile: str = Field("backend-base", validation_alias="EVONITH_BACKEND_PROFILE")
    frontend_profile: str = Field("frontend", validation_alias="EVONITH_FRONTEND_PROFILE")
    uvicorn_workers: int = Field(1, validation_alias="EVONITH_UVICORN_WORKERS")
    uvicorn_host: str = Field("0.0.0.0", validation_alias="EVONITH_UVICORN_HOST")
    uvicorn_port: int = Field(8080, validation_alias="EVONITH_UVICORN_PORT")
    frontend_host: str = Field("0.0.0.0", validation_alias="EVONITH_FRONTEND_HOST")
    frontend_port: int = Field(8501, validation_alias="EVONITH_FRONTEND_PORT")
    enable_optional_ai: bool = Field(False, validation_alias="EVONITH_ENABLE_OPTIONAL_AI")
    enable_optional_ml: bool = Field(True, validation_alias="EVONITH_ENABLE_OPTIONAL_ML")
    enable_optional_vector: bool = Field(False, validation_alias="EVONITH_ENABLE_OPTIONAL_VECTOR")
    enable_optional_documents: bool = Field(
        True,
        validation_alias="EVONITH_ENABLE_OPTIONAL_DOCUMENTS",
    )
    enable_optional_local_llm: bool = Field(
        False,
        validation_alias="EVONITH_ENABLE_OPTIONAL_LOCAL_LLM",
    )
    backend_log_level: str = Field(
        "INFO",
        validation_alias=AliasChoices("EVONITH_LOG_LEVEL", "EVONITH_BACKEND_LOG_LEVEL"),
    )
    log_format: str = Field("json", validation_alias="EVONITH_LOG_FORMAT")
    access_log_enabled: bool = Field(True, validation_alias="EVONITH_ACCESS_LOG_ENABLED")
    access_log_include_query_params: bool = Field(
        False,
        validation_alias="EVONITH_ACCESS_LOG_INCLUDE_QUERY_PARAMS",
    )
    log_redaction_enabled: bool = Field(True, validation_alias="EVONITH_LOG_REDACTION_ENABLED")
    log_file_enabled: bool = Field(False, validation_alias="EVONITH_LOG_FILE_ENABLED")
    log_max_file_mb: int = Field(50, validation_alias="EVONITH_LOG_MAX_FILE_MB")
    log_backup_count: int = Field(5, validation_alias="EVONITH_LOG_BACKUP_COUNT")
    cors_origins: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CORS_ORIGINS),
        validation_alias="BACKEND_CORS_ORIGINS",
    )
    enable_legacy_routes: bool = Field(True, validation_alias="EVONITH_ENABLE_LEGACY_ROUTES")
    openapi_title: str = Field(
        "Evonith BF Backend API",
        validation_alias="EVONITH_OPENAPI_TITLE",
    )
    openapi_version: str = Field("0.1.0", validation_alias="EVONITH_OPENAPI_VERSION")
    openapi_description: str = Field(
        "Versioned backend API for Evonith BF web application",
        validation_alias="EVONITH_OPENAPI_DESCRIPTION",
    )
    auth_enabled: bool = Field(True, validation_alias="EVONITH_AUTH_ENABLED")
    auth_secret_key: str = Field("", validation_alias="EVONITH_AUTH_SECRET_KEY")
    auth_algorithm: str = Field("HS256", validation_alias="EVONITH_AUTH_ALGORITHM")
    auth_access_token_expire_minutes: int = Field(
        480,
        validation_alias="EVONITH_AUTH_ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    auth_require_secret_in_production: bool = Field(
        True,
        validation_alias="EVONITH_AUTH_REQUIRE_SECRET_IN_PRODUCTION",
    )
    auth_allow_legacy_password_hashes: bool = Field(
        True,
        validation_alias="EVONITH_AUTH_ALLOW_LEGACY_PASSWORD_HASHES",
    )
    auth_upgrade_legacy_hash_on_login: bool = Field(
        True,
        validation_alias="EVONITH_AUTH_UPGRADE_LEGACY_HASH_ON_LOGIN",
    )
    auth_password_hash_scheme: str = Field(
        "bcrypt",
        validation_alias="EVONITH_AUTH_PASSWORD_HASH_SCHEME",
    )
    auth_min_password_length: int = Field(
        8,
        validation_alias="EVONITH_AUTH_MIN_PASSWORD_LENGTH",
    )
    auth_bootstrap_admin_enabled: bool = Field(
        False,
        validation_alias="EVONITH_AUTH_BOOTSTRAP_ADMIN_ENABLED",
    )
    feedback_require_auth: bool = Field(
        True,
        validation_alias="EVONITH_FEEDBACK_REQUIRE_AUTH",
    )
    feedback_storage_backend: str = Field(
        "sqlite",
        validation_alias="EVONITH_FEEDBACK_STORAGE_BACKEND",
    )
    feedback_database_url: str = Field(
        "",
        validation_alias="EVONITH_FEEDBACK_DATABASE_URL",
    )
    feedback_max_attachment_mb: int = Field(
        10,
        validation_alias="EVONITH_FEEDBACK_MAX_ATTACHMENT_MB",
    )
    feedback_allowed_attachment_types: list[str] = Field(
        default_factory=lambda: [
            "image/png",
            "image/jpeg",
            "image/webp",
            "application/pdf",
            "text/plain",
            "text/csv",
        ],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_ATTACHMENT_TYPES",
    )
    feedback_allowed_attachment_extensions: list[str] = Field(
        default_factory=lambda: [
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
            ".txt",
            ".csv",
        ],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_ATTACHMENT_EXTENSIONS",
    )
    feedback_max_attachments_per_ticket: int = Field(
        10,
        validation_alias="EVONITH_FEEDBACK_MAX_ATTACHMENTS_PER_TICKET",
    )
    feedback_default_status: str = Field(
        "open",
        validation_alias="EVONITH_FEEDBACK_DEFAULT_STATUS",
    )
    feedback_allowed_statuses: list[str] = Field(
        default_factory=lambda: ["open", "in_progress", "resolved", "closed", "rejected"],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_STATUSES",
    )
    feedback_allowed_priorities: list[str] = Field(
        default_factory=lambda: ["low", "medium", "high", "critical"],
        validation_alias="EVONITH_FEEDBACK_ALLOWED_PRIORITIES",
    )
    feedback_ticket_id_prefix: str = Field(
        "FB",
        validation_alias="EVONITH_FEEDBACK_TICKET_ID_PREFIX",
    )
    feedback_enable_legacy_read_fallback: bool = Field(
        True,
        validation_alias="EVONITH_FEEDBACK_ENABLE_LEGACY_READ_FALLBACK",
    )
    compute_require_auth: bool = Field(
        True,
        validation_alias="EVONITH_COMPUTE_REQUIRE_AUTH",
    )
    compute_max_preview_rows: int = Field(
        500,
        validation_alias="EVONITH_COMPUTE_MAX_PREVIEW_ROWS",
    )
    compute_max_json_rows: int = Field(
        5000,
        validation_alias="EVONITH_COMPUTE_MAX_JSON_ROWS",
    )
    compute_max_input_rows: int = Field(
        50000,
        validation_alias="EVONITH_COMPUTE_MAX_INPUT_ROWS",
    )
    compute_job_threshold_rows: int = Field(
        5000,
        validation_alias="EVONITH_COMPUTE_JOB_THRESHOLD_ROWS",
    )
    compute_job_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_COMPUTE_JOB_TTL_HOURS",
    )
    compute_artifact_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_COMPUTE_ARTIFACT_TTL_HOURS",
    )
    compute_max_seconds: int = Field(
        180,
        validation_alias="EVONITH_COMPUTE_MAX_SECONDS",
    )
    compute_threadpool_workers: int = Field(
        1,
        validation_alias="EVONITH_COMPUTE_THREADPOOL_WORKERS",
    )
    compute_export_format: str = Field(
        "csv",
        validation_alias="EVONITH_COMPUTE_EXPORT_FORMAT",
    )
    model_dir: str = Field("", validation_alias="EVONITH_MODEL_DIR")
    ml_device: str = Field("auto", validation_alias="EVONITH_ML_DEVICE")
    xgboost_device: str = Field("auto", validation_alias="EVONITH_XGBOOST_DEVICE")
    cuda_required: bool = Field(False, validation_alias="EVONITH_CUDA_REQUIRED")
    model_lazy_load: bool = Field(True, validation_alias="EVONITH_MODEL_LAZY_LOAD")
    model_cache_max_items: int = Field(
        2,
        validation_alias="EVONITH_MODEL_CACHE_MAX_ITEMS",
    )
    model_load_timeout_seconds: int = Field(
        30,
        validation_alias="EVONITH_MODEL_LOAD_TIMEOUT_SECONDS",
    )
    model_allow_missing_optional_models: bool = Field(
        True,
        validation_alias="EVONITH_MODEL_ALLOW_MISSING_OPTIONAL_MODELS",
    )
    material_balance_config_source: str = Field(
        "file",
        validation_alias="EVONITH_MATERIAL_BALANCE_CONFIG_SOURCE",
    )
    material_balance_allow_runtime_config: bool = Field(
        False,
        validation_alias="EVONITH_MATERIAL_BALANCE_ALLOW_RUNTIME_CONFIG",
    )
    recommendations_enable_explanations: bool = Field(
        True,
        validation_alias="EVONITH_RECOMMENDATIONS_ENABLE_EXPLANATIONS",
    )
    recommendations_max_items: int = Field(
        50,
        validation_alias="EVONITH_RECOMMENDATIONS_MAX_ITEMS",
    )
    blend_optimizer_max_candidates: int = Field(
        100,
        validation_alias="EVONITH_BLEND_OPTIMIZER_MAX_CANDIDATES",
    )
    blend_optimizer_max_iterations: int = Field(
        1000,
        validation_alias="EVONITH_BLEND_OPTIMIZER_MAX_ITERATIONS",
    )
    blend_optimizer_timeout_seconds: int = Field(
        120,
        validation_alias="EVONITH_BLEND_OPTIMIZER_TIMEOUT_SECONDS",
    )
    blend_optimizer_enable_model_predictions: bool = Field(
        True,
        validation_alias="EVONITH_BLEND_OPTIMIZER_ENABLE_MODEL_PREDICTIONS",
    )
    copilot_require_auth: bool = Field(
        True,
        validation_alias="EVONITH_COPILOT_REQUIRE_AUTH",
    )
    copilot_llm_enabled: bool = Field(
        False,
        validation_alias="EVONITH_COPILOT_LLM_ENABLED",
    )
    copilot_provider: str = Field("", validation_alias="EVONITH_COPILOT_PROVIDER")
    copilot_model: str = Field("", validation_alias="EVONITH_COPILOT_MODEL")
    copilot_api_key_env: str = Field(
        "OPENAI_API_KEY",
        validation_alias="EVONITH_COPILOT_API_KEY_ENV",
    )
    copilot_timeout_seconds: int = Field(
        60,
        validation_alias="EVONITH_COPILOT_TIMEOUT_SECONDS",
    )
    copilot_max_seconds: int = Field(
        120,
        validation_alias="EVONITH_COPILOT_MAX_SECONDS",
    )
    copilot_max_context_rows: int = Field(
        1000,
        validation_alias="EVONITH_COPILOT_MAX_CONTEXT_ROWS",
    )
    copilot_max_json_rows: int = Field(
        5000,
        validation_alias="EVONITH_COPILOT_MAX_JSON_ROWS",
    )
    copilot_max_prompt_chars: int = Field(
        20000,
        validation_alias="EVONITH_COPILOT_MAX_PROMPT_CHARS",
    )
    copilot_max_output_chars: int = Field(
        8000,
        validation_alias="EVONITH_COPILOT_MAX_OUTPUT_CHARS",
    )
    copilot_job_threshold_rows: int = Field(
        1000,
        validation_alias="EVONITH_COPILOT_JOB_THRESHOLD_ROWS",
    )
    copilot_job_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_COPILOT_JOB_TTL_HOURS",
    )
    copilot_artifact_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_COPILOT_ARTIFACT_TTL_HOURS",
    )
    copilot_enable_data_redaction: bool = Field(
        True,
        validation_alias="EVONITH_COPILOT_ENABLE_DATA_REDACTION",
    )
    copilot_allow_raw_data_to_llm: bool = Field(
        False,
        validation_alias="EVONITH_COPILOT_ALLOW_RAW_DATA_TO_LLM",
    )
    copilot_enable_provider_calls: bool = Field(
        False,
        validation_alias="EVONITH_COPILOT_ENABLE_PROVIDER_CALLS",
    )
    copilot_enable_code_execution: bool = Field(
        False,
        validation_alias="EVONITH_COPILOT_ENABLE_CODE_EXECUTION",
    )
    copilot_enable_plots: bool = Field(
        True,
        validation_alias="EVONITH_COPILOT_ENABLE_PLOTS",
    )
    copilot_log_prompt_preview: bool = Field(
        False,
        validation_alias="EVONITH_COPILOT_LOG_PROMPT_PREVIEW",
    )
    furnacemind_require_auth: bool = Field(
        True,
        validation_alias="EVONITH_FURNACEMIND_REQUIRE_AUTH",
    )
    furnacemind_storage_backend: str = Field(
        "sqlite",
        validation_alias="EVONITH_FURNACEMIND_STORAGE_BACKEND",
    )
    furnacemind_database_url: str = Field(
        "",
        validation_alias="EVONITH_FURNACEMIND_DATABASE_URL",
    )
    furnacemind_llm_enabled: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_LLM_ENABLED",
    )
    furnacemind_provider: str = Field("", validation_alias="EVONITH_FURNACEMIND_PROVIDER")
    furnacemind_model: str = Field("", validation_alias="EVONITH_FURNACEMIND_MODEL")
    furnacemind_api_key_env: str = Field(
        "OPENAI_API_KEY",
        validation_alias="EVONITH_FURNACEMIND_API_KEY_ENV",
    )
    furnacemind_enable_provider_calls: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_ENABLE_PROVIDER_CALLS",
    )
    furnacemind_timeout_seconds: int = Field(
        120,
        validation_alias="EVONITH_FURNACEMIND_TIMEOUT_SECONDS",
    )
    furnacemind_max_message_chars: int = Field(
        8000,
        validation_alias="EVONITH_FURNACEMIND_MAX_MESSAGE_CHARS",
    )
    furnacemind_max_response_chars: int = Field(
        12000,
        validation_alias="EVONITH_FURNACEMIND_MAX_RESPONSE_CHARS",
    )
    furnacemind_max_prompt_chars: int = Field(
        30000,
        validation_alias="EVONITH_FURNACEMIND_MAX_PROMPT_CHARS",
    )
    furnacemind_max_history_messages: int = Field(
        20,
        validation_alias="EVONITH_FURNACEMIND_MAX_HISTORY_MESSAGES",
    )
    furnacemind_max_context_docs: int = Field(
        5,
        validation_alias="EVONITH_FURNACEMIND_MAX_CONTEXT_DOCS",
    )
    furnacemind_max_context_chars: int = Field(
        20000,
        validation_alias="EVONITH_FURNACEMIND_MAX_CONTEXT_CHARS",
    )
    furnacemind_enable_data_redaction: bool = Field(
        True,
        validation_alias="EVONITH_FURNACEMIND_ENABLE_DATA_REDACTION",
    )
    furnacemind_allow_raw_docs_to_llm: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_ALLOW_RAW_DOCS_TO_LLM",
    )
    furnacemind_log_prompt_preview: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_LOG_PROMPT_PREVIEW",
    )
    furnacemind_documents_enabled: bool = Field(
        True,
        validation_alias="EVONITH_FURNACEMIND_DOCUMENTS_ENABLED",
    )
    furnacemind_max_document_mb: int = Field(
        20,
        validation_alias="EVONITH_FURNACEMIND_MAX_DOCUMENT_MB",
    )
    furnacemind_allowed_document_types: list[str] = Field(
        default_factory=lambda: [
            "application/pdf",
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
        ],
        validation_alias="EVONITH_FURNACEMIND_ALLOWED_DOCUMENT_TYPES",
    )
    furnacemind_allowed_document_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".md", ".csv", ".json"],
        validation_alias="EVONITH_FURNACEMIND_ALLOWED_DOCUMENT_EXTENSIONS",
    )
    furnacemind_max_extracted_chars: int = Field(
        200000,
        validation_alias="EVONITH_FURNACEMIND_MAX_EXTRACTED_CHARS",
    )
    furnacemind_document_ttl_days: int = Field(
        0,
        validation_alias="EVONITH_FURNACEMIND_DOCUMENT_TTL_DAYS",
    )
    furnacemind_memory_enabled: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_MEMORY_ENABLED",
    )
    furnacemind_vector_backend: str = Field(
        "qdrant",
        validation_alias="EVONITH_FURNACEMIND_VECTOR_BACKEND",
    )
    furnacemind_qdrant_url: str = Field(
        "",
        validation_alias="EVONITH_FURNACEMIND_QDRANT_URL",
    )
    furnacemind_qdrant_api_key_env: str = Field(
        "QDRANT_API_KEY",
        validation_alias="EVONITH_FURNACEMIND_QDRANT_API_KEY_ENV",
    )
    furnacemind_qdrant_collection: str = Field(
        "evonith_furnacemind",
        validation_alias="EVONITH_FURNACEMIND_QDRANT_COLLECTION",
    )
    furnacemind_embeddings_enabled: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_EMBEDDINGS_ENABLED",
    )
    furnacemind_embedding_provider: str = Field(
        "",
        validation_alias="EVONITH_FURNACEMIND_EMBEDDING_PROVIDER",
    )
    furnacemind_embedding_model: str = Field(
        "",
        validation_alias="EVONITH_FURNACEMIND_EMBEDDING_MODEL",
    )
    furnacemind_embedding_api_key_env: str = Field(
        "OPENAI_API_KEY",
        validation_alias="EVONITH_FURNACEMIND_EMBEDDING_API_KEY_ENV",
    )
    furnacemind_vector_top_k: int = Field(
        5,
        validation_alias="EVONITH_FURNACEMIND_VECTOR_TOP_K",
    )
    furnacemind_vector_timeout_seconds: int = Field(
        10,
        validation_alias="EVONITH_FURNACEMIND_VECTOR_TIMEOUT_SECONDS",
    )
    furnacemind_tools_enabled: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_TOOLS_ENABLED",
    )
    furnacemind_allowed_tools: list[str] = Field(
        default_factory=lambda: [
            "data_summary",
            "anomaly_summary",
            "material_balance_summary",
            "recommendations_summary",
            "blend_optimizer_summary",
        ],
        validation_alias="EVONITH_FURNACEMIND_ALLOWED_TOOLS",
    )
    furnacemind_tool_timeout_seconds: int = Field(
        30,
        validation_alias="EVONITH_FURNACEMIND_TOOL_TIMEOUT_SECONDS",
    )
    furnacemind_max_tool_calls_per_run: int = Field(
        5,
        validation_alias="EVONITH_FURNACEMIND_MAX_TOOL_CALLS_PER_RUN",
    )
    furnacemind_enable_code_execution: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_ENABLE_CODE_EXECUTION",
    )
    furnacemind_enable_shell_execution: bool = Field(
        False,
        validation_alias="EVONITH_FURNACEMIND_ENABLE_SHELL_EXECUTION",
    )
    furnacemind_run_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_FURNACEMIND_RUN_TTL_HOURS",
    )
    furnacemind_artifact_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_FURNACEMIND_ARTIFACT_TTL_HOURS",
    )
    furnacemind_event_retention_hours: int = Field(
        24,
        validation_alias="EVONITH_FURNACEMIND_EVENT_RETENTION_HOURS",
    )
    furnacemind_max_events_per_run: int = Field(
        500,
        validation_alias="EVONITH_FURNACEMIND_MAX_EVENTS_PER_RUN",
    )
    furnacemind_streaming_enabled: bool = Field(
        True,
        validation_alias="EVONITH_FURNACEMIND_STREAMING_ENABLED",
    )
    furnacemind_polling_fallback_enabled: bool = Field(
        True,
        validation_alias="EVONITH_FURNACEMIND_POLLING_FALLBACK_ENABLED",
    )
    audit_log_enabled: bool = Field(True, validation_alias="EVONITH_AUDIT_LOG_ENABLED")
    audit_storage_backend: str = Field(
        "sqlite",
        validation_alias="EVONITH_AUDIT_STORAGE_BACKEND",
    )
    audit_database_url: str = Field("", validation_alias="EVONITH_AUDIT_DATABASE_URL")
    audit_retention_days: int = Field(
        90,
        validation_alias="EVONITH_AUDIT_RETENTION_DAYS",
    )
    audit_admin_read_enabled: bool = Field(
        True,
        validation_alias="EVONITH_AUDIT_ADMIN_READ_ENABLED",
    )
    status_public_health: bool = Field(
        True,
        validation_alias="EVONITH_STATUS_PUBLIC_HEALTH",
    )
    status_require_auth_for_details: bool = Field(
        True,
        validation_alias="EVONITH_STATUS_REQUIRE_AUTH_FOR_DETAILS",
    )
    dependency_check_timeout_seconds: int = Field(
        3,
        validation_alias="EVONITH_DEPENDENCY_CHECK_TIMEOUT_SECONDS",
    )
    dependency_check_cache_seconds: int = Field(
        30,
        validation_alias="EVONITH_DEPENDENCY_CHECK_CACHE_SECONDS",
    )
    runtime_min_free_mb: int = Field(1024, validation_alias="EVONITH_RUNTIME_MIN_FREE_MB")
    runtime_warn_free_mb: int = Field(4096, validation_alias="EVONITH_RUNTIME_WARN_FREE_MB")
    metrics_enabled: bool = Field(True, validation_alias="EVONITH_METRICS_ENABLED")
    metrics_require_auth: bool = Field(True, validation_alias="EVONITH_METRICS_REQUIRE_AUTH")
    metrics_format: str = Field("json", validation_alias="EVONITH_METRICS_FORMAT")
    metrics_reset_enabled: bool = Field(False, validation_alias="EVONITH_METRICS_RESET_ENABLED")
    unified_jobs_enabled: bool = Field(
        True,
        validation_alias="EVONITH_UNIFIED_JOBS_ENABLED",
    )
    cleanup_enabled: bool = Field(True, validation_alias="EVONITH_CLEANUP_ENABLED")
    cleanup_dry_run_default: bool = Field(
        True,
        validation_alias="EVONITH_CLEANUP_DRY_RUN_DEFAULT",
    )
    cleanup_require_admin: bool = Field(
        True,
        validation_alias="EVONITH_CLEANUP_REQUIRE_ADMIN",
    )
    cleanup_max_delete_per_run: int = Field(
        500,
        validation_alias="EVONITH_CLEANUP_MAX_DELETE_PER_RUN",
    )
    cleanup_include_logs: bool = Field(
        False,
        validation_alias="EVONITH_CLEANUP_INCLUDE_LOGS",
    )
    cleanup_include_uploads: bool = Field(
        False,
        validation_alias="EVONITH_CLEANUP_INCLUDE_UPLOADS",
    )
    cleanup_job_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_CLEANUP_JOB_TTL_HOURS",
    )
    cleanup_artifact_ttl_hours: int = Field(
        24,
        validation_alias="EVONITH_CLEANUP_ARTIFACT_TTL_HOURS",
    )
    cleanup_temp_ttl_hours: int = Field(
        6,
        validation_alias="EVONITH_CLEANUP_TEMP_TTL_HOURS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        prefix = (value or "/api/v1").strip()
        if not prefix.startswith("/"):
            prefix = f"/{prefix}"
        return prefix.rstrip("/") or "/api/v1"

    @field_validator("backend_log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return (value or "INFO").strip().upper()

    @field_validator("auth_algorithm")
    @classmethod
    def normalize_auth_algorithm(cls, value: str) -> str:
        return (value or "HS256").strip().upper()

    @field_validator("auth_password_hash_scheme")
    @classmethod
    def normalize_password_hash_scheme(cls, value: str) -> str:
        return (value or "bcrypt").strip().lower()

    @field_validator(
        "auth_access_token_expire_minutes",
        "auth_min_password_length",
        "uvicorn_workers",
        "uvicorn_port",
        "frontend_port",
        "log_max_file_mb",
        "log_backup_count",
        "feedback_max_attachment_mb",
        "feedback_max_attachments_per_ticket",
        "compute_max_preview_rows",
        "compute_max_json_rows",
        "compute_max_input_rows",
        "compute_job_threshold_rows",
        "compute_job_ttl_hours",
        "compute_artifact_ttl_hours",
        "compute_max_seconds",
        "compute_threadpool_workers",
        "model_cache_max_items",
        "model_load_timeout_seconds",
        "recommendations_max_items",
        "blend_optimizer_max_candidates",
        "blend_optimizer_max_iterations",
        "blend_optimizer_timeout_seconds",
        "copilot_timeout_seconds",
        "copilot_max_seconds",
        "copilot_max_context_rows",
        "copilot_max_json_rows",
        "copilot_max_prompt_chars",
        "copilot_max_output_chars",
        "copilot_job_threshold_rows",
        "copilot_job_ttl_hours",
        "copilot_artifact_ttl_hours",
        "furnacemind_timeout_seconds",
        "furnacemind_max_message_chars",
        "furnacemind_max_response_chars",
        "furnacemind_max_prompt_chars",
        "furnacemind_max_history_messages",
        "furnacemind_max_context_docs",
        "furnacemind_max_context_chars",
        "furnacemind_max_document_mb",
        "furnacemind_max_extracted_chars",
        "furnacemind_vector_top_k",
        "furnacemind_vector_timeout_seconds",
        "furnacemind_tool_timeout_seconds",
        "furnacemind_max_tool_calls_per_run",
        "furnacemind_run_ttl_hours",
        "furnacemind_artifact_ttl_hours",
        "furnacemind_event_retention_hours",
        "furnacemind_max_events_per_run",
        "audit_retention_days",
        "dependency_check_timeout_seconds",
        "dependency_check_cache_seconds",
        "runtime_min_free_mb",
        "runtime_warn_free_mb",
        "cleanup_max_delete_per_run",
        "cleanup_job_ttl_hours",
        "cleanup_artifact_ttl_hours",
        "cleanup_temp_ttl_hours",
    )
    @classmethod
    def require_positive_int(cls, value: int) -> int:
        return max(1, int(value))

    @field_validator(
        "feedback_storage_backend",
        "compute_export_format",
        "log_format",
        "runtime_profile",
        "backend_profile",
        "frontend_profile",
        "copilot_provider",
        "furnacemind_storage_backend",
        "furnacemind_provider",
        "furnacemind_vector_backend",
        "furnacemind_embedding_provider",
        "audit_storage_backend",
        "metrics_format",
        "ml_device",
        "xgboost_device",
    )
    @classmethod
    def normalize_lower_string(cls, value: str) -> str:
        return str(value or "").strip().lower()

    @field_validator("ml_device", "xgboost_device")
    @classmethod
    def validate_accelerator_device(cls, value: str) -> str:
        device = str(value or "auto").strip().lower()
        if device in {"auto", "cpu", "cuda"}:
            return device
        if device.startswith("cuda:") and device[5:].isdigit():
            return device
        raise ValueError("accelerator device must be auto, cpu, cuda, or cuda:<index>")

    @field_validator(
        "feedback_default_status",
        "feedback_ticket_id_prefix",
        "edge_device_type",
        "uvicorn_host",
        "frontend_host",
        "model_dir",
        "material_balance_config_source",
        "copilot_model",
        "copilot_api_key_env",
        "furnacemind_database_url",
        "audit_database_url",
        "furnacemind_model",
        "furnacemind_api_key_env",
        "furnacemind_qdrant_url",
        "furnacemind_qdrant_api_key_env",
        "furnacemind_qdrant_collection",
        "furnacemind_embedding_model",
        "furnacemind_embedding_api_key_env",
    )
    @classmethod
    def normalize_feedback_string(cls, value: str) -> str:
        return str(value or "").strip()

    @field_validator(
        "feedback_allowed_attachment_types",
        "feedback_allowed_attachment_extensions",
        "feedback_allowed_statuses",
        "feedback_allowed_priorities",
        "furnacemind_allowed_document_types",
        "furnacemind_allowed_document_extensions",
        "furnacemind_allowed_tools",
        mode="before",
    )
    @classmethod
    def parse_feedback_csv(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return list(_DEFAULT_CORS_ORIGINS)
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(origin).strip() for origin in value if str(origin).strip()]
        return list(_DEFAULT_CORS_ORIGINS)

    def safe_runtime_profile_summary(self) -> dict[str, Any]:
        """Return public-safe runtime profile metadata for status endpoints."""
        return {
            "runtime_profile": self.runtime_profile,
            "edge_mode": self.edge_mode,
            "edge_device_type": self.edge_device_type,
            "backend_profile": self.backend_profile,
            "frontend_profile": self.frontend_profile,
            "service": {
                "uvicorn_workers": self.uvicorn_workers,
                "uvicorn_host": self.uvicorn_host,
                "uvicorn_port": self.uvicorn_port,
                "frontend_host": self.frontend_host,
                "frontend_port": self.frontend_port,
            },
            "optional_features": {
                "ai": self.enable_optional_ai,
                "ml": self.enable_optional_ml,
                "vector": self.enable_optional_vector,
                "documents": self.enable_optional_documents,
                "local_llm": self.enable_optional_local_llm,
            },
            "acceleration": {
                "ml_device": self.ml_device,
                "xgboost_device": self.xgboost_device,
                "cuda_required": self.cuda_required,
            },
        }


def load_backend_settings() -> BackendSettings:
    """Load backend settings from the current environment."""
    return BackendSettings()
