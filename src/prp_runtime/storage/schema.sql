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
    final_work_unit_id        TEXT,
    request_json              TEXT    NOT NULL,
    usage_input_tokens        INTEGER NOT NULL DEFAULT 0 CHECK (usage_input_tokens >= 0),
    usage_output_tokens       INTEGER NOT NULL DEFAULT 0 CHECK (usage_output_tokens >= 0),
    usage_strong_model_tokens INTEGER NOT NULL DEFAULT 0 CHECK (usage_strong_model_tokens >= 0),
    usage_elapsed_ms          INTEGER NOT NULL DEFAULT 0 CHECK (usage_elapsed_ms >= 0),
    metrics_usage_observed    INTEGER NOT NULL DEFAULT 0 CHECK (metrics_usage_observed IN (0, 1)),
    metrics_usage_known       INTEGER NOT NULL DEFAULT 0 CHECK (metrics_usage_known IN (0, 1)),
    metrics_provider_elapsed_ms INTEGER CHECK (metrics_provider_elapsed_ms IS NULL OR metrics_provider_elapsed_ms >= 0),
    metrics_wall_clock_ms     INTEGER CHECK (metrics_wall_clock_ms IS NULL OR metrics_wall_clock_ms >= 0),
    metrics_cost_known        INTEGER NOT NULL DEFAULT 0 CHECK (metrics_cost_known IN (0, 1)),
    metrics_cost              TEXT CHECK (metrics_cost IS NULL OR TRIM(metrics_cost) <> ''),
    error_category            TEXT,
    error_message             TEXT,
    created_at                TEXT    NOT NULL,
    started_at                TEXT,
    completed_at              TEXT
    ,FOREIGN KEY (final_work_unit_id, run_id, graph_version)
        REFERENCES work_units (work_unit_id, run_id, graph_version)
);

CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status, created_at);

CREATE TABLE IF NOT EXISTS work_units (
    work_unit_id        TEXT    PRIMARY KEY,
    run_id              TEXT    NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    graph_version       INTEGER NOT NULL CHECK (graph_version >= 1),
    lineage_key         TEXT,
    dependency_fingerprint TEXT,
    content_fingerprint TEXT,
    name                TEXT    NOT NULL,
    instruction         TEXT    NOT NULL,
    acceptance_criteria TEXT,
    output_json         TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    created_at          TEXT    NOT NULL
    ,UNIQUE (work_unit_id, run_id, graph_version),
    CHECK (
        (lineage_key IS NULL AND dependency_fingerprint IS NULL AND content_fingerprint IS NULL)
        OR (lineage_key IS NOT NULL AND dependency_fingerprint IS NOT NULL AND content_fingerprint IS NOT NULL)
    ),
    CHECK (
        dependency_fingerprint IS NULL
        OR (length(dependency_fingerprint) = 64 AND dependency_fingerprint NOT GLOB '*[^0-9a-f]*')
    ),
    CHECK (
        content_fingerprint IS NULL
        OR (length(content_fingerprint) = 64 AND content_fingerprint NOT GLOB '*[^0-9a-f]*')
    ),
    UNIQUE (run_id, graph_version, lineage_key)
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
    cost_input_price_per_million_tokens TEXT,
    cost_output_price_per_million_tokens TEXT,
    cost_input              TEXT,
    cost_output             TEXT,
    cost_total              TEXT,
    error_category            TEXT,
    error_message             TEXT,
    created_at                TEXT    NOT NULL,
    started_at                TEXT,
    completed_at              TEXT,
    UNIQUE (work_unit_id, attempt_index)
    ,CHECK (
        (cost_total IS NULL AND cost_input_price_per_million_tokens IS NULL
         AND cost_output_price_per_million_tokens IS NULL AND cost_input IS NULL
         AND cost_output IS NULL)
        OR (cost_total IS NOT NULL AND cost_input_price_per_million_tokens IS NOT NULL
            AND cost_output_price_per_million_tokens IS NOT NULL AND cost_input IS NOT NULL
            AND cost_output IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_attempts_run_status ON attempts (run_id, status);

-- Public Agent history only stores the discriminated turn/result projection.
-- The raw Provider body, arguments secrets and hidden reasoning never cross
-- this row contract; sequence and idempotency are unique per attempt.
CREATE TABLE IF NOT EXISTS agent_history (
    run_id          TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    work_unit_id    TEXT NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    attempt_id      TEXT NOT NULL REFERENCES attempts (attempt_id) ON DELETE CASCADE,
    sequence        INTEGER NOT NULL CHECK (sequence >= 1 AND sequence <= 128),
    idempotency_key TEXT NOT NULL CHECK (
        idempotency_key = TRIM(idempotency_key)
        AND idempotency_key <> ''
        AND length(idempotency_key) <= 128
    ),
    item_kind       TEXT NOT NULL CHECK (item_kind IN ('turn', 'tool_result')),
    item_json       TEXT NOT NULL CHECK (
        json_valid(item_json) AND length(item_json) <= 131072
    ),
    created_at      TEXT NOT NULL,
    PRIMARY KEY (attempt_id, sequence),
    UNIQUE (attempt_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_agent_history_run
    ON agent_history (run_id, work_unit_id, attempt_id, sequence);

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

CREATE TABLE IF NOT EXISTS reservations (
    reservation_id                TEXT PRIMARY KEY,
    run_id                        TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    work_unit_id                  TEXT NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    dispatch_key                  TEXT NOT NULL,
    attempt_units                 INTEGER NOT NULL CHECK (attempt_units >= 1),
    estimated_input_tokens        INTEGER CHECK (estimated_input_tokens IS NULL OR estimated_input_tokens >= 0),
    estimated_output_tokens       INTEGER CHECK (estimated_output_tokens IS NULL OR estimated_output_tokens >= 0),
    token_upper_bound             INTEGER CHECK (token_upper_bound IS NULL OR token_upper_bound >= 0),
    strong_token_upper_bound      INTEGER CHECK (strong_token_upper_bound IS NULL OR strong_token_upper_bound >= 0),
    capacity_key                  TEXT,
    status                        TEXT NOT NULL CHECK (status IN ('PENDING', 'HELD', 'SETTLED', 'RELEASED', 'EXPIRED')),
    created_at                    TEXT NOT NULL,
    held_at                       TEXT,
    completed_at                  TEXT,
    measured_input_tokens         INTEGER CHECK (measured_input_tokens IS NULL OR measured_input_tokens >= 0),
    measured_output_tokens        INTEGER CHECK (measured_output_tokens IS NULL OR measured_output_tokens >= 0),
    measured_strong_model_tokens  INTEGER CHECK (measured_strong_model_tokens IS NULL OR measured_strong_model_tokens >= 0),
    measured_elapsed_ms           INTEGER CHECK (measured_elapsed_ms IS NULL OR measured_elapsed_ms >= 0),
    UNIQUE (run_id, dispatch_key),
    CHECK (capacity_key IS NULL OR TRIM(capacity_key) <> ''),
    CHECK (dispatch_key = TRIM(dispatch_key) AND dispatch_key <> ''),
    CHECK (status = 'PENDING' AND held_at IS NULL AND completed_at IS NULL
           AND measured_input_tokens IS NULL AND measured_output_tokens IS NULL
           AND measured_strong_model_tokens IS NULL AND measured_elapsed_ms IS NULL
           OR status = 'HELD' AND held_at IS NOT NULL AND completed_at IS NULL
           AND measured_input_tokens IS NULL AND measured_output_tokens IS NULL
           AND measured_strong_model_tokens IS NULL AND measured_elapsed_ms IS NULL
           OR status IN ('SETTLED', 'RELEASED', 'EXPIRED') AND held_at IS NOT NULL
           AND completed_at IS NOT NULL),
    CHECK (held_at IS NULL OR held_at >= created_at),
    CHECK (completed_at IS NULL OR held_at IS NOT NULL AND completed_at >= held_at),
    CHECK (strong_token_upper_bound IS NULL OR token_upper_bound IS NULL
           OR strong_token_upper_bound <= token_upper_bound),
    CHECK (estimated_input_tokens IS NULL OR estimated_output_tokens IS NULL
           OR token_upper_bound IS NULL
           OR estimated_input_tokens + estimated_output_tokens <= token_upper_bound),
    CHECK (status IN ('RELEASED', 'EXPIRED')
           OR measured_input_tokens IS NULL OR measured_output_tokens IS NOT NULL),
    CHECK (status IN ('RELEASED', 'EXPIRED')
           OR measured_input_tokens IS NULL OR measured_strong_model_tokens IS NOT NULL),
    CHECK (status IN ('RELEASED', 'EXPIRED')
           OR measured_input_tokens IS NULL OR measured_elapsed_ms IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_reservations_run_status
    ON reservations (run_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_reservations_work_unit
    ON reservations (work_unit_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_reservations_active_dispatch
    ON reservations (run_id, dispatch_key)
    WHERE status IN ('PENDING', 'HELD');

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

-- Workspaces are service-owned aliases or opaque bridge grants. No host path or
-- credential is persisted, and RESTRICT keeps snapshot audit history intact.
CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id    TEXT PRIMARY KEY,
    owner_id        TEXT NOT NULL CHECK (TRIM(owner_id) <> ''),
    alias           TEXT NOT NULL CHECK (TRIM(alias) <> ''),
    source_type     TEXT NOT NULL CHECK (source_type IN ('SERVER_ALIAS', 'BRIDGE_GRANT')),
    server_alias    TEXT,
    bridge_grant    TEXT,
    status          TEXT NOT NULL CHECK (status IN ('ACTIVE', 'SUSPENDED', 'REVOKED')),
    created_at      TEXT NOT NULL,
    closed_at       TEXT,
    UNIQUE (owner_id, alias),
    UNIQUE (workspace_id, owner_id),
    CHECK (
        (source_type = 'SERVER_ALIAS' AND TRIM(server_alias) <> '' AND bridge_grant IS NULL)
        OR (source_type = 'BRIDGE_GRANT' AND server_alias IS NULL AND TRIM(bridge_grant) <> '')
    ),
    CHECK (
        (status = 'REVOKED' AND closed_at IS NOT NULL)
        OR (status IN ('ACTIVE', 'SUSPENDED') AND closed_at IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_workspaces_owner ON workspaces (owner_id, status);

-- Sessions bind an authenticated principal to one server-owned workspace. The
-- request never supplies an owner; the API derives it from the bearer token.
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    principal_id     TEXT NOT NULL CHECK (TRIM(principal_id) <> ''),
    workspace_id     TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE RESTRICT,
    access_json      TEXT NOT NULL CHECK (length(access_json) <= 4096),
    agent_options_json TEXT NOT NULL CHECK (length(agent_options_json) <= 4096),
    status           TEXT NOT NULL CHECK (status IN ('ACTIVE', 'REVOKED')),
    created_at       TEXT NOT NULL,
    expires_at       TEXT,
    revoked_at       TEXT,
    CHECK (expires_at IS NULL OR expires_at > created_at),
    CHECK (
        (status = 'ACTIVE' AND revoked_at IS NULL)
        OR (status = 'REVOKED' AND revoked_at IS NOT NULL)
    ),
    UNIQUE (session_id, principal_id, workspace_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_principal
    ON sessions (principal_id, status, created_at, session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_workspace
    ON sessions (workspace_id, status, created_at, session_id);

-- A run belongs to at most one Session. This indirection keeps the existing
-- Run domain/storage contract stable while making API ownership durable.
CREATE TABLE IF NOT EXISTS session_runs (
    session_id   TEXT NOT NULL REFERENCES sessions (session_id) ON DELETE RESTRICT,
    run_id       TEXT NOT NULL UNIQUE REFERENCES runs (run_id) ON DELETE CASCADE,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (session_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_session_runs_session
    ON session_runs (session_id, created_at, run_id);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id     TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE RESTRICT,
    status          TEXT NOT NULL CHECK (status IN ('CREATING', 'READY', 'INVALIDATED')),
    manifest_hash   TEXT NOT NULL UNIQUE
                    CHECK (length(manifest_hash) = 64
                           AND manifest_hash NOT GLOB '*[^0-9a-f]*'),
    file_count      INTEGER CHECK (file_count IS NULL OR file_count >= 0),
    total_size      INTEGER CHECK (total_size IS NULL OR total_size >= 0),
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    CHECK (
        (status = 'CREATING' AND completed_at IS NULL)
        OR (status IN ('READY', 'INVALIDATED') AND completed_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_snapshots_workspace ON snapshots (workspace_id, status, created_at);

CREATE TABLE IF NOT EXISTS snapshot_files (
    snapshot_id     TEXT NOT NULL REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    path            TEXT NOT NULL CHECK (
                        path = TRIM(path)
                        AND path <> ''
                        AND path NOT GLOB '/*'
                        AND path NOT LIKE '%\\%' ESCAPE '\'
                        AND path NOT GLOB '*..*'
                    ),
    sha256          TEXT NOT NULL CHECK (
                        length(sha256) = 64
                        AND sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
    size            INTEGER NOT NULL CHECK (size >= 0 AND size <= 1073741824),
    entry_type      TEXT NOT NULL CHECK (entry_type IN ('FILE', 'DIRECTORY')),
    PRIMARY KEY (snapshot_id, path)
);

CREATE INDEX IF NOT EXISTS idx_snapshot_files_snapshot ON snapshot_files (snapshot_id, path);

-- ChangeSets make one approved patch transition auditable. File content is
-- represented only by bounded metadata; neither table persists host paths.
CREATE TABLE IF NOT EXISTS change_sets (
    change_set_id    TEXT NOT NULL PRIMARY KEY CHECK (change_set_id = TRIM(change_set_id)
                                                       AND change_set_id <> ''),
    run_id           TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    tool_call_id     TEXT NOT NULL UNIQUE REFERENCES tool_calls (call_id) ON DELETE RESTRICT,
    workspace_id     TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE RESTRICT,
    base_snapshot_id TEXT NOT NULL REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    new_snapshot_id  TEXT NOT NULL REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    patch_text       TEXT NOT NULL CHECK (length(patch_text) BETWEEN 1 AND 262144),
    patch_sha256     TEXT NOT NULL CHECK (length(patch_sha256) = 64
                                           AND patch_sha256 NOT GLOB '*[^0-9a-f]*'),
    created_at       TEXT NOT NULL,
    CHECK (base_snapshot_id <> new_snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_change_sets_run ON change_sets (run_id, created_at, change_set_id);
CREATE INDEX IF NOT EXISTS idx_change_sets_workspace
    ON change_sets (workspace_id, created_at, change_set_id);

CREATE TABLE IF NOT EXISTS change_set_files (
    change_set_id TEXT NOT NULL REFERENCES change_sets (change_set_id) ON DELETE CASCADE,
    path          TEXT NOT NULL CHECK (
                      path = TRIM(path)
                      AND path <> ''
                      AND path NOT GLOB '/*'
                      AND path NOT LIKE '%\\%' ESCAPE '\'
                      AND path NOT GLOB '*..*'
                  ),
    action        TEXT NOT NULL CHECK (action IN ('ADD', 'MODIFY', 'DELETE')),
    before_sha256 TEXT CHECK (before_sha256 IS NULL OR (length(before_sha256) = 64
                                                         AND before_sha256 NOT GLOB '*[^0-9a-f]*')),
    before_size   INTEGER CHECK (before_size IS NULL OR (before_size >= 0
                                                           AND before_size <= 1073741824)),
    after_sha256  TEXT CHECK (after_sha256 IS NULL OR (length(after_sha256) = 64
                                                        AND after_sha256 NOT GLOB '*[^0-9a-f]*')),
    after_size    INTEGER CHECK (after_size IS NULL OR (after_size >= 0
                                                          AND after_size <= 1073741824)),
    PRIMARY KEY (change_set_id, path),
    CHECK (
        (action = 'ADD' AND before_sha256 IS NULL AND before_size IS NULL
         AND after_sha256 IS NOT NULL AND after_size IS NOT NULL)
        OR (action = 'MODIFY' AND before_sha256 IS NOT NULL AND before_size IS NOT NULL
            AND after_sha256 IS NOT NULL AND after_size IS NOT NULL)
        OR (action = 'DELETE' AND before_sha256 IS NOT NULL AND before_size IS NOT NULL
            AND after_sha256 IS NULL AND after_size IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_change_set_files_path ON change_set_files (path, change_set_id);

-- Merge lifecycle facts contain only identities, bounded digests and status.
-- The staging root and promotion destination are intentionally not durable.
CREATE TABLE IF NOT EXISTS merge_ledger (
    merge_id              TEXT PRIMARY KEY CHECK (merge_id = TRIM(merge_id)
                                                   AND merge_id GLOB 'merge_*'),
    run_id                TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    workspace_id          TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE RESTRICT,
    base_snapshot_id      TEXT NOT NULL REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    change_set_ids_json   TEXT NOT NULL CHECK (
                              json_valid(change_set_ids_json)
                              AND json_type(change_set_ids_json) = 'array'
                              AND length(change_set_ids_json) <= 262144
                          ),
    input_digest          TEXT NOT NULL CHECK (length(input_digest) = 64
                                               AND input_digest NOT GLOB '*[^0-9a-f]*'),
    status                TEXT NOT NULL CHECK (
                              status IN ('PLANNED', 'RUNNING', 'MERGED', 'PROMOTED',
                                         'CONFLICT', 'UNKNOWN')
                          ),
    merged_snapshot_id    TEXT REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    merged_content_hash   TEXT CHECK (merged_content_hash IS NULL
                                      OR (length(merged_content_hash) = 64
                                          AND merged_content_hash NOT GLOB '*[^0-9a-f]*')),
    promoted_content_hash TEXT CHECK (promoted_content_hash IS NULL
                                      OR (length(promoted_content_hash) = 64
                                          AND promoted_content_hash NOT GLOB '*[^0-9a-f]*')),
    created_at            TEXT NOT NULL,
    completed_at          TEXT,
    UNIQUE (run_id, input_digest),
    CHECK (
        (status IN ('PLANNED', 'RUNNING')
         AND merged_snapshot_id IS NULL AND merged_content_hash IS NULL
         AND promoted_content_hash IS NULL AND completed_at IS NULL)
        OR (status = 'MERGED'
            AND merged_snapshot_id IS NOT NULL AND merged_content_hash IS NOT NULL
            AND promoted_content_hash IS NULL AND completed_at IS NOT NULL)
        OR (status = 'PROMOTED'
            AND merged_snapshot_id IS NOT NULL AND merged_content_hash IS NOT NULL
            AND promoted_content_hash IS NOT NULL AND completed_at IS NOT NULL)
        OR (status IN ('CONFLICT', 'UNKNOWN')
            AND merged_snapshot_id IS NULL AND merged_content_hash IS NULL
            AND promoted_content_hash IS NULL AND completed_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_merge_ledger_run
    ON merge_ledger (run_id, status, created_at, merge_id);
CREATE INDEX IF NOT EXISTS idx_merge_ledger_snapshot
    ON merge_ledger (base_snapshot_id, status, created_at, merge_id);

CREATE TRIGGER IF NOT EXISTS trg_merge_ledger_identity_immutable
BEFORE UPDATE OF merge_id, run_id, workspace_id, base_snapshot_id, change_set_ids_json,
                 input_digest, created_at
ON merge_ledger
BEGIN
    SELECT RAISE(ABORT, 'merge ledger identity is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_merge_ledger_terminal_immutable
BEFORE UPDATE ON merge_ledger
WHEN OLD.status IN ('PROMOTED', 'CONFLICT', 'UNKNOWN')
BEGIN
    SELECT RAISE(ABORT, 'terminal merge ledger is immutable');
END;

-- Progressive rounds are immutable recovery facts. A verified round is the only
-- state allowed to reference a merged snapshot, and round/index graph identities
-- are unique within one run so an old round cannot be rewritten in place. A
-- content-identical immutable base snapshot may be reused by later graph rounds.
CREATE TABLE IF NOT EXISTS progressive_rounds (
    round_id              TEXT PRIMARY KEY CHECK (round_id = TRIM(round_id)
                                                  AND round_id <> ''),
    run_id                TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    round_index           INTEGER NOT NULL CHECK (round_index >= 0),
    graph_version         INTEGER NOT NULL CHECK (graph_version >= 1),
    base_snapshot_id      TEXT NOT NULL REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    merged_snapshot_id    TEXT REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    change_set_ids_json   TEXT NOT NULL DEFAULT '[]' CHECK (length(change_set_ids_json) <= 262144),
    evidence_ids_json     TEXT NOT NULL DEFAULT '[]' CHECK (length(evidence_ids_json) <= 262144),
    status                TEXT NOT NULL CHECK (status IN ('PLANNED', 'VERIFIED', 'FAILED', 'CANCELLED')),
    revision_of_round_id  TEXT REFERENCES progressive_rounds (round_id) ON DELETE RESTRICT,
    revision_reason       TEXT CHECK (revision_reason IS NULL OR length(revision_reason) <= 512),
    failure_reason        TEXT CHECK (failure_reason IS NULL OR length(failure_reason) <= 512),
    created_at            TEXT NOT NULL,
    completed_at          TEXT,
    UNIQUE (run_id, round_index),
    UNIQUE (run_id, graph_version),
    CHECK (
        (status = 'PLANNED' AND merged_snapshot_id IS NULL AND evidence_ids_json = '[]'
         AND failure_reason IS NULL AND completed_at IS NULL)
        OR (status = 'VERIFIED' AND merged_snapshot_id IS NOT NULL AND evidence_ids_json <> '[]'
            AND failure_reason IS NULL AND completed_at IS NOT NULL)
        OR (status IN ('FAILED', 'CANCELLED') AND merged_snapshot_id IS NULL
            AND evidence_ids_json = '[]' AND failure_reason IS NOT NULL AND completed_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_progressive_rounds_run
    ON progressive_rounds (run_id, round_index, graph_version);

-- Tool calls are immutable requests tied to one run, work unit and workspace
-- snapshot. The idempotency key is scoped to a run so a retried transport
-- request cannot create a second durable call.
CREATE TABLE IF NOT EXISTS tool_calls (
    call_id          TEXT PRIMARY KEY CHECK (call_id = TRIM(call_id) AND call_id <> ''),
    run_id           TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    work_unit_id     TEXT NOT NULL REFERENCES work_units (work_unit_id) ON DELETE CASCADE,
    workspace_id     TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE RESTRICT,
    base_snapshot_id TEXT NOT NULL REFERENCES snapshots (snapshot_id) ON DELETE RESTRICT,
    idempotency_key  TEXT NOT NULL CHECK (idempotency_key = TRIM(idempotency_key)
                                          AND idempotency_key <> ''),
    tool_name        TEXT NOT NULL CHECK (tool_name = TRIM(tool_name)
                                          AND length(tool_name) BETWEEN 1 AND 64),
    effect           TEXT NOT NULL CHECK (effect IN ('READ', 'WRITE', 'COMMAND', 'NETWORK')),
    arguments_json   TEXT NOT NULL CHECK (length(arguments_json) <= 65536),
    status           TEXT NOT NULL CHECK (
        status IN (
            'REQUESTED', 'AWAITING_APPROVAL', 'RUNNING', 'SUCCEEDED', 'FAILED',
            'CANCELLED', 'REJECTED', 'INTERRUPTED', 'UNKNOWN'
        )
    ),
    requested_at     TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    UNIQUE (run_id, idempotency_key),
    UNIQUE (call_id, run_id, workspace_id),
    CHECK (
        (status IN ('REQUESTED', 'AWAITING_APPROVAL')
         AND started_at IS NULL AND completed_at IS NULL)
        OR (status = 'RUNNING' AND started_at IS NOT NULL AND completed_at IS NULL)
        OR (status IN ('SUCCEEDED', 'FAILED', 'CANCELLED', 'REJECTED', 'INTERRUPTED', 'UNKNOWN')
            AND completed_at IS NOT NULL
            AND (started_at IS NULL OR completed_at >= started_at))
    )
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_run_status
    ON tool_calls (run_id, status, requested_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_work_unit
    ON tool_calls (work_unit_id, status);
CREATE INDEX IF NOT EXISTS idx_tool_calls_workspace
    ON tool_calls (workspace_id, requested_at);

-- A result is the one terminal observation of one call. Making call_id the
-- primary key prevents a second conflicting terminal result at the DB layer.
CREATE TABLE IF NOT EXISTS tool_results (
    call_id          TEXT PRIMARY KEY REFERENCES tool_calls (call_id) ON DELETE CASCADE,
    status           TEXT NOT NULL CHECK (
        status IN ('SUCCEEDED', 'FAILED', 'CANCELLED', 'REJECTED', 'INTERRUPTED', 'UNKNOWN')
    ),
    result_json      TEXT CHECK (result_json IS NULL OR length(result_json) <= 262144),
    output           TEXT NOT NULL DEFAULT '' CHECK (length(output) <= 262144),
    truncated        INTEGER NOT NULL DEFAULT 0 CHECK (truncated IN (0, 1)),
    changed_paths_json TEXT NOT NULL DEFAULT '[]' CHECK (length(changed_paths_json) <= 262144),
    exit_code        INTEGER,
    error_category   TEXT CHECK (error_category IS NULL OR TRIM(error_category) <> ''),
    error_message    TEXT CHECK (error_message IS NULL OR length(error_message) <= 4096),
    completed_at     TEXT NOT NULL,
    CHECK (
        (status IN ('FAILED', 'REJECTED', 'INTERRUPTED', 'UNKNOWN')
         AND error_category IS NOT NULL AND error_message IS NOT NULL)
        OR (status IN ('SUCCEEDED', 'CANCELLED')
            AND error_category IS NULL AND error_message IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_tool_results_status
    ON tool_results (status, completed_at);

-- Native Bridge claims are lease-bearing facts, separate from approvals and
-- tool results. Composite foreign keys keep call, run, session, workspace and
-- owner scope aligned even when rows are inserted directly into SQLite.
CREATE TABLE IF NOT EXISTS bridge_claims (
    claim_id         TEXT PRIMARY KEY CHECK (claim_id = TRIM(claim_id)
                                              AND claim_id <> ''),
    call_id          TEXT NOT NULL,
    run_id           TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    workspace_id     TEXT NOT NULL,
    owner_id         TEXT NOT NULL CHECK (TRIM(owner_id) <> ''),
    claimant_id      TEXT NOT NULL CHECK (TRIM(claimant_id) <> ''
                                          AND length(claimant_id) <= 128),
    idempotency_key  TEXT NOT NULL CHECK (idempotency_key = TRIM(idempotency_key)
                                          AND idempotency_key <> ''
                                          AND length(idempotency_key) <= 128),
    fingerprint      TEXT NOT NULL CHECK (length(fingerprint) = 64
                                          AND fingerprint NOT GLOB '*[^0-9a-f]*'),
    status           TEXT NOT NULL CHECK (status IN ('ACTIVE', 'EXPIRED', 'SETTLED', 'RELEASED')),
    claimed_at       TEXT NOT NULL,
    expires_at       TEXT NOT NULL,
    closed_at        TEXT,
    UNIQUE (session_id, run_id, idempotency_key),
    FOREIGN KEY (call_id, run_id, workspace_id)
        REFERENCES tool_calls (call_id, run_id, workspace_id) ON DELETE RESTRICT,
    FOREIGN KEY (session_id, run_id)
        REFERENCES session_runs (session_id, run_id) ON DELETE RESTRICT,
    FOREIGN KEY (session_id, owner_id, workspace_id)
        REFERENCES sessions (session_id, principal_id, workspace_id) ON DELETE RESTRICT,
    FOREIGN KEY (workspace_id, owner_id)
        REFERENCES workspaces (workspace_id, owner_id) ON DELETE RESTRICT,
    CHECK (expires_at > claimed_at),
    CHECK (claimed_at LIKE '%T%Z' OR claimed_at LIKE '%T%+__:__'
           OR claimed_at LIKE '%T%-__:__'),
    CHECK (expires_at LIKE '%T%Z' OR expires_at LIKE '%T%+__:__'
           OR expires_at LIKE '%T%-__:__'),
    CHECK (closed_at IS NULL OR closed_at LIKE '%T%Z'
           OR closed_at LIKE '%T%+__:__' OR closed_at LIKE '%T%-__:__'),
    CHECK (closed_at IS NULL OR closed_at >= claimed_at),
    CHECK ((status = 'ACTIVE' AND closed_at IS NULL)
           OR (status = 'EXPIRED' AND closed_at IS NOT NULL AND closed_at >= expires_at)
           OR (status = 'SETTLED' AND closed_at IS NOT NULL AND closed_at < expires_at)
           OR (status = 'RELEASED' AND closed_at IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_bridge_claims_call
    ON bridge_claims (call_id, status, expires_at);
CREATE INDEX IF NOT EXISTS idx_bridge_claims_owner
    ON bridge_claims (owner_id, session_id, status, expires_at);
CREATE UNIQUE INDEX IF NOT EXISTS uq_bridge_claims_active_call
    ON bridge_claims (call_id)
    WHERE status = 'ACTIVE';

-- Identity and lease facts are immutable. The only permitted mutation is one
-- ACTIVE -> terminal status transition performed by the Store in a transaction.
CREATE TRIGGER IF NOT EXISTS trg_bridge_claims_identity_immutable
BEFORE UPDATE OF claim_id, call_id, run_id, session_id, workspace_id, owner_id,
                 claimant_id, idempotency_key, fingerprint, claimed_at, expires_at
ON bridge_claims
BEGIN
    SELECT RAISE(ABORT, 'bridge claim identity is immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_bridge_claims_terminal_immutable
BEFORE UPDATE ON bridge_claims
WHEN OLD.status <> 'ACTIVE'
BEGIN
    SELECT RAISE(ABORT, 'terminal bridge claim is immutable');
END;

-- Approval decisions are append-once facts on the request row. The Store only
-- fills the nullable decision columns once and treats a replay as idempotent.
CREATE TABLE IF NOT EXISTS approvals (
    request_id       TEXT PRIMARY KEY CHECK (request_id = TRIM(request_id)
                                             AND request_id <> ''),
    call_id          TEXT NOT NULL REFERENCES tool_calls (call_id) ON DELETE CASCADE,
    run_id           TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    workspace_id     TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE RESTRICT,
    owner_id         TEXT NOT NULL CHECK (TRIM(owner_id) <> ''),
    tool_name        TEXT NOT NULL CHECK (tool_name = TRIM(tool_name)
                                          AND length(tool_name) BETWEEN 1 AND 64),
    effect           TEXT NOT NULL CHECK (effect IN ('READ', 'WRITE', 'COMMAND', 'NETWORK')),
    scope_json       TEXT NOT NULL CHECK (length(scope_json) <= 65536),
    reason           TEXT NOT NULL CHECK (TRIM(reason) <> '' AND length(reason) <= 512),
    issuer           TEXT NOT NULL CHECK (issuer IN ('USER', 'SERVER')),
    requested_at     TEXT NOT NULL,
    outcome          TEXT CHECK (outcome IS NULL OR outcome IN ('ALLOW', 'DENY')),
    decision_issuer  TEXT CHECK (decision_issuer IS NULL OR decision_issuer IN ('USER', 'SERVER')),
    decision_reason  TEXT CHECK (decision_reason IS NULL OR length(decision_reason) <= 512),
    decided_at       TEXT,
    CHECK (
        (outcome IS NULL AND decision_issuer IS NULL AND decision_reason IS NULL
         AND decided_at IS NULL)
        OR (outcome IS NOT NULL AND decision_issuer IS NOT NULL AND decided_at IS NOT NULL
            AND (outcome = 'ALLOW' OR TRIM(decision_reason) <> ''))
    )
);

CREATE INDEX IF NOT EXISTS idx_approvals_owner ON approvals (owner_id, requested_at, request_id);
CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals (run_id, requested_at, request_id);
CREATE INDEX IF NOT EXISTS idx_approvals_call ON approvals (call_id, requested_at);

CREATE TABLE IF NOT EXISTS leases (
    lease_id             TEXT PRIMARY KEY CHECK (lease_id = TRIM(lease_id)
                                                  AND lease_id <> ''),
    request_id           TEXT NOT NULL REFERENCES approvals (request_id) ON DELETE RESTRICT,
    call_id              TEXT NOT NULL REFERENCES tool_calls (call_id) ON DELETE RESTRICT,
    workspace_id         TEXT NOT NULL REFERENCES workspaces (workspace_id) ON DELETE RESTRICT,
    owner_id             TEXT NOT NULL CHECK (TRIM(owner_id) <> ''),
    scope_json           TEXT NOT NULL CHECK (length(scope_json) <= 65536),
    issuer               TEXT NOT NULL CHECK (issuer IN ('USER', 'SERVER')),
    issued_at            TEXT NOT NULL,
    expires_at           TEXT NOT NULL,
    status               TEXT NOT NULL CHECK (status IN ('ACTIVE', 'REVOKED', 'EXPIRED')),
    closed_at            TEXT,
    close_reason         TEXT CHECK (close_reason IS NULL OR length(close_reason) <= 512),
    CHECK (
        (status = 'ACTIVE' AND closed_at IS NULL AND close_reason IS NULL)
        OR (status = 'REVOKED' AND closed_at IS NOT NULL AND TRIM(close_reason) <> '')
        OR (status = 'EXPIRED' AND closed_at IS NOT NULL)
    ),
    CHECK (expires_at > issued_at)
);

CREATE INDEX IF NOT EXISTS idx_leases_owner ON leases (owner_id, status, expires_at, lease_id);
CREATE INDEX IF NOT EXISTS idx_leases_run ON leases (call_id, status, expires_at);
