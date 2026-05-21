"""SQLite-backed StrategyDB (Spec.html §D)."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from pokerbot.strategy_db.base import (
    StrategyDB,
    StrategyRow,
    pack_probs,
    unpack_probs,
    validate_action_probs,
)
from pokerbot.strategy_db.migrate import migrate_sqlite

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from numpy.typing import NDArray

    from pokerbot.abstraction import InfoSet


_META_CURRENT_VERSION_KEY = "current_strategy_version"


class SQLiteStrategyDB(StrategyDB):
    """SQLite implementation. `path == ':memory:'` for an in-memory DB."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL") if path != ":memory:" else None
        migrate_sqlite(self._conn)

    # ───────── housekeeping ─────────

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteStrategyDB is closed")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ───────── reads ─────────

    def get(self, infoset: InfoSet, version: int | None = None) -> StrategyRow | None:
        conn = self._require_conn()
        v = self.current_version() if version is None else version
        cur = conn.execute(
            "SELECT action_mask, action_probs, visit_count, version "
            "FROM strategy WHERE infoset_hash = ? AND version = ?",
            (infoset.hash16(), v),
        )
        row = cur.fetchone()
        if row is None:
            return None
        mask, probs_blob, visit_count, ver = row
        return StrategyRow(
            action_mask=int(mask),
            action_probs=unpack_probs(int(mask), bytes(probs_blob)),
            visit_count=int(visit_count),
            version=int(ver),
        )

    def nearest_neighbor(self, infoset: InfoSet, version: int) -> StrategyRow | None:
        conn = self._require_conn()
        cur = conn.execute(
            """
            SELECT action_mask, action_probs, visit_count, version
            FROM strategy
            WHERE table_size = ? AND street = ? AND card_bucket = ?
              AND position = ? AND stack_bucket = ? AND version = ?
            ORDER BY visit_count DESC, infoset_hash ASC
            LIMIT 1
            """,
            (
                infoset.table_size,
                infoset.street,
                infoset.card_bucket,
                infoset.position,
                infoset.stack_bucket,
                version,
            ),
        )
        row = cur.fetchone()
        if row is None:
            return None
        mask, probs_blob, visit_count, ver = row
        return StrategyRow(
            action_mask=int(mask),
            action_probs=unpack_probs(int(mask), bytes(probs_blob)),
            visit_count=int(visit_count),
            version=int(ver),
        )

    # ───────── writes ─────────

    def _insert(
        self,
        conn: sqlite3.Connection,
        infoset: InfoSet,
        action_mask: int,
        action_probs: NDArray[np.float32],
        version: int,
        visit_count: int,
    ) -> None:
        validate_action_probs(action_mask, action_probs)
        conn.execute(
            """
            INSERT INTO strategy
              (infoset_hash, version, table_size, street, position, stack_bucket,
               card_bucket, history_blob, action_mask, action_probs, visit_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(infoset_hash, version) DO UPDATE SET
              table_size = excluded.table_size,
              street = excluded.street,
              position = excluded.position,
              stack_bucket = excluded.stack_bucket,
              card_bucket = excluded.card_bucket,
              history_blob = excluded.history_blob,
              action_mask = excluded.action_mask,
              action_probs = excluded.action_probs,
              visit_count = excluded.visit_count
            """,
            (
                infoset.hash16(),
                version,
                infoset.table_size,
                infoset.street,
                infoset.position,
                infoset.stack_bucket,
                infoset.card_bucket,
                infoset.history,
                action_mask,
                pack_probs(action_probs),
                visit_count,
            ),
        )

    def put(
        self,
        infoset: InfoSet,
        action_mask: int,
        action_probs: NDArray[np.float32],
        version: int,
        visit_count: int = 0,
    ) -> None:
        conn = self._require_conn()
        with conn:
            self._insert(conn, infoset, action_mask, action_probs, version, visit_count)

    def bulk_put(
        self,
        rows: Iterable[tuple[InfoSet, int, NDArray[np.float32]]],
        version: int,
    ) -> None:
        conn = self._require_conn()
        with conn:  # rolls back on any exception, including iteration failures
            for infoset, mask, probs in rows:
                self._insert(conn, infoset, mask, probs, version, 0)

    # ───────── version control ─────────

    def current_version(self) -> int:
        conn = self._require_conn()
        cur = conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (_META_CURRENT_VERSION_KEY,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                f"meta.{_META_CURRENT_VERSION_KEY!r} not set; "
                "call set_current_version() or pass version= explicitly"
            )
        return int(row[0])

    def set_current_version(self, version: int) -> None:
        conn = self._require_conn()
        with conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_META_CURRENT_VERSION_KEY, str(version)),
            )


__all__ = ["SQLiteStrategyDB"]
