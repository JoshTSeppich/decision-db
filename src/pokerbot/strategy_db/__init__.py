"""Strategy DB (Spec.html §D).

Public API:
    - `StrategyDB`        abstract base
    - `StrategyRow`       row dataclass
    - `SQLiteStrategyDB`  SQLite backend
    - `LMDBStrategyDB`    LMDB backend (msgpack values, prefix-index for NN)
    - `open_db(uri)`      factory: 'sqlite:///path.db' or 'lmdb:///path/'
"""

from __future__ import annotations

from pokerbot.strategy_db.base import (
    PROBS_SUM_TOL,
    StrategyDB,
    StrategyRow,
    pack_probs,
    popcount,
    unpack_probs,
    validate_action_probs,
)
from pokerbot.strategy_db.lmdb import LMDBStrategyDB
from pokerbot.strategy_db.migrate import LATEST_SCHEMA_VERSION, migrate_sqlite
from pokerbot.strategy_db.sqlite import SQLiteStrategyDB


def open_db(uri: str) -> StrategyDB:
    """Open or create a strategy DB by URI. Applies migrations as needed.

    Supported schemes:
        sqlite:///absolute/path.db
        sqlite:///:memory:          (in-memory; per-connection)
        lmdb:///absolute/path/      (directory; created if missing)
    """
    scheme, sep, rest = uri.partition("://")
    if not sep:
        raise ValueError(f"invalid uri (missing '://'): {uri!r}")
    if scheme == "sqlite":
        path = rest
        if path in (":memory:", "/:memory:"):
            return SQLiteStrategyDB(":memory:")
        return SQLiteStrategyDB(path)
    if scheme == "lmdb":
        return LMDBStrategyDB(rest)
    raise ValueError(f"unsupported scheme: {scheme!r} (use 'sqlite' or 'lmdb')")


__all__ = [
    "LATEST_SCHEMA_VERSION",
    "PROBS_SUM_TOL",
    "LMDBStrategyDB",
    "SQLiteStrategyDB",
    "StrategyDB",
    "StrategyRow",
    "migrate_sqlite",
    "open_db",
    "pack_probs",
    "popcount",
    "unpack_probs",
    "validate_action_probs",
]
