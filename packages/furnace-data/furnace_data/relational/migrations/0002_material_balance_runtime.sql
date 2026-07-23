-- Versioned runtime Material Balance configuration.
CREATE TABLE IF NOT EXISTS material_balance_config_revisions (
    revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_key TEXT NOT NULL,
    version TEXT NOT NULL UNIQUE,
    config_json TEXT NOT NULL,
    packaged_default_checksum TEXT NOT NULL,
    created_at TEXT NOT NULL,
    created_by TEXT,
    request_id TEXT,
    client_metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_material_balance_config_profile
    ON material_balance_config_revisions(profile_key, revision_id DESC);