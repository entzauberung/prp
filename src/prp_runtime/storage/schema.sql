-- Current PRP runtime SQLite schema.
--
-- This is the only schema. There is no migration framework and no backward
-- compatible variant: during pre-0.1 development an incompatible development
-- database is deleted and recreated.
--
-- Timestamps are ISO-8601 UTC strings. Enum-valued columns hold the native
-- domain enum values.

CREATE TABLE IF NOT EXISTS runs (
    run_id                    TEXT    PRIMARY KEY,
    status                    TEXT    NOT NULL,
    routing_policy            TEXT    NOT NULL,
    strategy                  TEXT,
    graph_version             INTEGER NOT NULL CHECK (graph_version >= 1),
    request_json              TEXT    NOT NULL,
    usage_input_tokens        INTEGER NOT NULL DEFAULT 0 CHECK (usage_input_tokens >= 0),
    usage_output_tokens       INTEGER NOT NULL DEFAULT 0 CHECK (usage_output_tokens >= 0),
    usage_strong_model_tokens INTEGER NOT NULL DEFAULT 0 CHECK (usage_strong_model_tokens >= 0),
    usage_elapsed_ms          INTEGER NOT NULL DEFAULT 0 CHECK (usage_elapsed_ms >= 0),
    error_category            TEXT,
    error_message             TEXT,
    created_at                TEXT    NOT NULL,
    started_at                TEXT,
    completed_at              TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status, created_at);

CREATE TABLE IF NOT EXISTS work_units (
    work_unit_id        TEXT    PRIMARY KEY,
    run_id              TEXT    NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    graph_version       INTEGER NOT NULL CHECK (graph_version >= 1),
    name                TEXT    NOT NULL,
    instruction         TEXT    NOT NULL,
    acceptance_criteria TEXT,
    output_json         TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    created_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_work_units_run
    ON work_units (run_id, graph_version, status);

CREATE TABLE IF NOT EXISTS work_unit_dependencies (
    work_unit_id  TEXT NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    depends_on_id TEXT NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    PRIMARY KEY (work_unit_id, depends_on_id),
    CHECK (work_unit_id <> depends_on_id)
);

CREATE INDEX IF NOT EXISTS idx_work_unit_dependencies_depends_on
    ON work_unit_dependencies (depends_on_id);

CREATE TABLE IF NOT EXISTS work_unit_resource_claims (
    work_unit_id TEXT NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    resource     TEXT NOT NULL,
    access       TEXT NOT NULL,
    PRIMARY KEY (work_unit_id, resource, access)
);

CREATE INDEX IF NOT EXISTS idx_work_unit_resource_claims_resource
    ON work_unit_resource_claims (resource, access);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id                TEXT    PRIMARY KEY,
    run_id                    TEXT    NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    work_unit_id              TEXT    NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    attempt_index             INTEGER NOT NULL CHECK (attempt_index >= 1),
    role                      TEXT    NOT NULL,
    provider                  TEXT    NOT NULL,
    model                     TEXT    NOT NULL,
    status                    TEXT    NOT NULL,
    provider_request_id       TEXT,
    usage_input_tokens        INTEGER CHECK (usage_input_tokens IS NULL OR usage_input_tokens >= 0),
    usage_output_tokens       INTEGER CHECK (usage_output_tokens IS NULL OR usage_output_tokens >= 0),
    usage_strong_model_tokens INTEGER CHECK (usage_strong_model_tokens IS NULL OR usage_strong_model_tokens >= 0),
    usage_elapsed_ms          INTEGER CHECK (usage_elapsed_ms IS NULL OR usage_elapsed_ms >= 0),
    error_category            TEXT,
    error_message             TEXT,
    created_at                TEXT    NOT NULL,
    started_at                TEXT,
    completed_at              TEXT,
    UNIQUE (work_unit_id, attempt_index)
);

CREATE INDEX IF NOT EXISTS idx_attempts_run_status ON attempts (run_id, status);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id  TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    work_unit_id TEXT NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    attempt_id   TEXT NOT NULL REFERENCES attempts (attempt_id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_work_unit ON artifacts (work_unit_id, name);
CREATE INDEX IF NOT EXISTS idx_artifacts_attempt ON artifacts (attempt_id);

-- ``result`` is the whole verdict and the only stored form. There is no boolean
-- column: PASS/FAIL/INCONCLUSIVE cannot be recovered from a boolean, because FAIL
-- and INCONCLUSIVE would share it, and a check that could not decide would read
-- back as a proven failure. The boolean is derived in the domain model instead.
--
-- A deterministic check must name the rule it applied; a model review need not,
-- but still has to carry a verdict.
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id  TEXT    PRIMARY KEY,
    run_id       TEXT    NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    work_unit_id TEXT    NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    artifact_id  TEXT    NOT NULL REFERENCES artifacts (artifact_id) ON DELETE CASCADE,
    kind         TEXT    NOT NULL CHECK (kind IN ('DETERMINISTIC_CHECK', 'MODEL_REVIEW')),
    rule         TEXT    CHECK (rule IS NULL OR TRIM(rule) <> ''),
    result       TEXT    NOT NULL CHECK (result IN ('PASS', 'FAIL', 'INCONCLUSIVE')),
    detail       TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    CHECK (kind <> 'DETERMINISTIC_CHECK' OR rule IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_evidence_artifact ON evidence (artifact_id);
CREATE INDEX IF NOT EXISTS idx_evidence_work_unit ON evidence (work_unit_id, result);

-- The append-only ledger. The primary key makes a duplicate sequence number
-- impossible, even under concurrent appends.
CREATE TABLE IF NOT EXISTS events (
    run_id       TEXT    NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    sequence     INTEGER NOT NULL CHECK (sequence >= 1),
    event_type   TEXT    NOT NULL,
    payload_json TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    PRIMARY KEY (run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_events_type ON events (run_id, event_type);
