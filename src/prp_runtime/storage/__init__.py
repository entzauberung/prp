"""Persistence layer.

One concrete SQLite implementation. No ORM, no repository abstraction and no
migration framework.
"""

from prp_runtime.storage.recovery import (
    RECOVERY_REASON,
    RecoveryReport,
    recover_after_restart,
)
from prp_runtime.storage.sqlite import (
    SCHEMA_PATH,
    SCHEMA_VERSION,
    DanglingReferenceError,
    DuplicateEntityError,
    IncompatibleSchemaError,
    MissingEntityError,
    SequenceConflictError,
    SqliteStore,
)

__all__ = [
    "RECOVERY_REASON",
    "SCHEMA_PATH",
    "SCHEMA_VERSION",
    "DanglingReferenceError",
    "DuplicateEntityError",
    "IncompatibleSchemaError",
    "MissingEntityError",
    "RecoveryReport",
    "SequenceConflictError",
    "SqliteStore",
    "recover_after_restart",
]
