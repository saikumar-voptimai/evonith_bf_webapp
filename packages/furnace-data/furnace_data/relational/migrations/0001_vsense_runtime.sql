-- V-Sense runtime persistence schema.
-- The backend applies the same statements idempotently for SQLite edge
-- deployments via apps.backend_api.app.repositories.vsense_repository.

CREATE TABLE IF NOT EXISTS vsense_contexts (
    context_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    optimization_type_id TEXT NOT NULL,
    catalog_version TEXT NOT NULL,
    algorithm_version TEXT NOT NULL,
    dataset_version TEXT,
    control_profile_version INTEGER,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    as_of TEXT NOT NULL,
    context_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_vsense_context_owner
    ON vsense_contexts(owner_user_id, created_at);

CREATE INDEX IF NOT EXISTS ix_vsense_context_expiry
    ON vsense_contexts(expires_at);

CREATE TABLE IF NOT EXISTS vsense_control_profiles (
    profile_id TEXT NOT NULL,
    optimization_type_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    catalog_version TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    updated_by_user_id TEXT,
    updated_by_username TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, optimization_type_id)
);

CREATE TABLE IF NOT EXISTS vsense_control_profile_history (
    profile_id TEXT NOT NULL,
    optimization_type_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    catalog_version TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    updated_by_user_id TEXT,
    updated_by_username TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, optimization_type_id, version)
);

CREATE TABLE IF NOT EXISTS vsense_runs (
    run_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    owner_username TEXT,
    optimization_type_id TEXT NOT NULL,
    context_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(context_id) REFERENCES vsense_contexts(context_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS ix_vsense_runs_owner
    ON vsense_runs(owner_user_id, created_at);

CREATE TABLE IF NOT EXISTS vsense_idempotency (
    owner_user_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    idempotency_key_hash TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (owner_user_id, scope, idempotency_key_hash)
);

