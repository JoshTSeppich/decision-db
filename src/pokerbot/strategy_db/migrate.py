"""Schema migrations for the SQLite backend (Spec.html §D).

Migrations are embedded as `(version, sql)` tuples and tracked in the `meta`
table via the `schema_version` key. Idempotent — calling `migrate_sqlite`
twice is a no-op.
"""

from __future__ import annotations

import sqlite3
from typing import Final

# Composite PK = (infoset_hash, version). The single-PK form in the spec's
# DDL conflicts with `test_version_filter`; this is the corrected layout.
_SCHEMA_V1: Final[str] = """
CREATE TABLE IF NOT EXISTS strategy (
    infoset_hash    BLOB        NOT NULL,
    version         INTEGER     NOT NULL,
    table_size      INTEGER     NOT NULL,
    street          INTEGER     NOT NULL,
    position        INTEGER     NOT NULL,
    stack_bucket    INTEGER     NOT NULL,
    card_bucket     INTEGER     NOT NULL,
    history_blob    BLOB        NOT NULL,
    action_mask     INTEGER     NOT NULL,
    action_probs    BLOB        NOT NULL,
    visit_count     INTEGER     NOT NULL DEFAULT 0,
    PRIMARY KEY (infoset_hash, version)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_lookup_compound
    ON strategy (table_size, street, card_bucket, position, stack_bucket, version);

CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_versions (
    version              INTEGER PRIMARY KEY,
    created_at           TEXT NOT NULL,
    training_iter        INTEGER NOT NULL,
    notes                TEXT,
    abstraction_checksum TEXT NOT NULL
);
"""

SCHEMA_MIGRATIONS: Final[list[tuple[int, str]]] = [
    (1, _SCHEMA_V1),
]

LATEST_SCHEMA_VERSION: Final[int] = SCHEMA_MIGRATIONS[-1][0]


def _read_current_version(conn: sqlite3.Connection) -> int:
    try:
        cur = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'")
    except sqlite3.OperationalError:
        return 0  # meta table not yet created
    row = cur.fetchone()
    return int(row[0]) if row else 0


def migrate_sqlite(conn: sqlite3.Connection) -> None:
    """Apply any pending schema migrations. Idempotent."""
    current = _read_current_version(conn)
    new_version = current
    for version, sql in SCHEMA_MIGRATIONS:
        if version > current:
            conn.executescript(sql)
            new_version = version
    if new_version != current or _read_current_version(conn) != LATEST_SCHEMA_VERSION:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(LATEST_SCHEMA_VERSION),),
        )
        conn.commit()


__all__ = ["LATEST_SCHEMA_VERSION", "SCHEMA_MIGRATIONS", "migrate_sqlite"]
