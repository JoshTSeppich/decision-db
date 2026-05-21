"""Spec.html §D tests for the strategy DB.

Parametrized across both backends (SQLite + LMDB). The 8 spec tests:
    1. test_put_get_roundtrip
    2. test_get_missing_returns_none
    3. test_nearest_neighbor_finds_when_history_differs
    4. test_bulk_put_atomic
    5. test_version_filter
    6. test_migrations_idempotent
    7. test_action_probs_sum_to_one
    8. test_sqlite_lmdb_equivalence
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pokerbot.abstraction import ActionType, InfoSet
from pokerbot.strategy_db import (
    LATEST_SCHEMA_VERSION,
    LMDBStrategyDB,
    SQLiteStrategyDB,
    StrategyDB,
    migrate_sqlite,
    open_db,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# ───────── fixtures ─────────


def _mk_infoset(card_bucket: int = 42, history: bytes = b"") -> InfoSet:
    return InfoSet(
        table_size=6,
        street=1,
        position=2,
        stack_bucket=4,
        card_bucket=card_bucket,
        history=history,
    )


def _mk_probs(values: list[float] | None = None) -> np.ndarray:
    if values is None:
        values = [0.4, 0.4, 0.2]
    return np.array(values, dtype=np.float32)


def _mk_mask(actions: list[ActionType] | None = None) -> int:
    if actions is None:
        actions = [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.BET_66]
    m = 0
    for a in actions:
        m |= 1 << int(a)
    return m


@pytest.fixture(params=["sqlite", "lmdb"])
def db(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[StrategyDB]:
    if request.param == "sqlite":
        instance: StrategyDB = SQLiteStrategyDB(str(tmp_path / "strategy.db"))
    else:
        instance = LMDBStrategyDB(str(tmp_path / "strategy.lmdb"))
    instance.set_current_version(1)
    try:
        yield instance
    finally:
        instance.close()


# ───────── 1. test_put_get_roundtrip ─────────


def test_put_get_roundtrip(db: StrategyDB) -> None:
    info = _mk_infoset()
    mask = _mk_mask()
    probs = _mk_probs()
    db.put(info, mask, probs, version=1, visit_count=7)

    row = db.get(info, version=1)
    assert row is not None
    assert row.action_mask == mask
    assert row.version == 1
    assert row.visit_count == 7
    np.testing.assert_array_almost_equal(row.action_probs, probs)


# ───────── 2. test_get_missing_returns_none ─────────


def test_get_missing_returns_none(db: StrategyDB) -> None:
    info = _mk_infoset()
    assert db.get(info, version=1) is None
    # Also: different version returns None even after a put.
    db.put(info, _mk_mask(), _mk_probs(), version=1)
    assert db.get(info, version=99) is None


# ───────── 3. test_nearest_neighbor_finds_when_history_differs ─────────


def test_nearest_neighbor_finds_when_history_differs(db: StrategyDB) -> None:
    info_stored = _mk_infoset(history=b"\x10\x02\xf0\x11")  # bet, call, boundary, bet_66
    info_query = _mk_infoset(history=b"\x12\x02\xf0\x11")  # different history, same prefix
    assert info_stored.hash16() != info_query.hash16()

    db.put(info_stored, _mk_mask(), _mk_probs(), version=1, visit_count=5)
    # exact lookup misses
    assert db.get(info_query, version=1) is None
    # nearest_neighbor finds the stored row by (t_size, street, card_bucket, position, stack_bucket)
    row = db.nearest_neighbor(info_query, version=1)
    assert row is not None
    assert row.visit_count == 5


def test_nearest_neighbor_picks_highest_visit_count(db: StrategyDB) -> None:
    info_a = _mk_infoset(history=b"\x10")
    info_b = _mk_infoset(history=b"\x11")
    db.put(info_a, _mk_mask(), _mk_probs(), version=1, visit_count=3)
    db.put(info_b, _mk_mask(), _mk_probs(), version=1, visit_count=42)

    info_query = _mk_infoset(history=b"\x12\x02")
    row = db.nearest_neighbor(info_query, version=1)
    assert row is not None
    assert row.visit_count == 42


def test_nearest_neighbor_returns_none_when_no_prefix_match(db: StrategyDB) -> None:
    info = _mk_infoset()
    db.put(info, _mk_mask(), _mk_probs(), version=1)
    # Different card_bucket prefix — no match.
    other = _mk_infoset(card_bucket=99)
    assert db.nearest_neighbor(other, version=1) is None


# ───────── 4. test_bulk_put_atomic ─────────


def test_bulk_put_atomic_rolls_back_on_exception(db: StrategyDB) -> None:
    infos = [_mk_infoset(history=bytes([i])) for i in range(5)]

    def gen() -> Iterator[tuple[InfoSet, int, np.ndarray]]:
        yield (infos[0], _mk_mask(), _mk_probs())
        yield (infos[1], _mk_mask(), _mk_probs())
        raise RuntimeError("simulated mid-batch failure")

    with pytest.raises(RuntimeError, match="simulated"):
        db.bulk_put(gen(), version=1)

    # None of the rows in the failed batch should be visible.
    for info in infos:
        assert db.get(info, version=1) is None


def test_bulk_put_commits_on_success(db: StrategyDB) -> None:
    infos = [_mk_infoset(history=bytes([i])) for i in range(10)]
    rows = [(info, _mk_mask(), _mk_probs()) for info in infos]
    db.bulk_put(iter(rows), version=1)
    for info in infos:
        assert db.get(info, version=1) is not None


# ───────── 5. test_version_filter ─────────


def test_version_filter_keeps_rows_distinct(db: StrategyDB) -> None:
    info = _mk_infoset()
    probs_v1 = _mk_probs([0.5, 0.3, 0.2])
    probs_v2 = _mk_probs([0.1, 0.6, 0.3])

    db.put(info, _mk_mask(), probs_v1, version=1, visit_count=1)
    db.put(info, _mk_mask(), probs_v2, version=2, visit_count=2)

    row1 = db.get(info, version=1)
    row2 = db.get(info, version=2)
    assert row1 is not None and row2 is not None
    assert row1.version == 1 and row2.version == 2
    np.testing.assert_array_almost_equal(row1.action_probs, probs_v1)
    np.testing.assert_array_almost_equal(row2.action_probs, probs_v2)
    assert row1.visit_count == 1 and row2.visit_count == 2


# ───────── 6. test_migrations_idempotent ─────────


def test_migrate_sqlite_idempotent(tmp_path: Path) -> None:
    path = str(tmp_path / "m.db")
    db = SQLiteStrategyDB(path)
    # Already migrated by ctor; re-running migrate should be a no-op.
    conn = sqlite3.connect(path)
    migrate_sqlite(conn)
    migrate_sqlite(conn)
    cur = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'")
    row = cur.fetchone()
    conn.close()
    assert int(row[0]) == LATEST_SCHEMA_VERSION
    db.close()


def test_open_db_applies_migrations(tmp_path: Path) -> None:
    """`open_db` is the supported entry point and should leave a usable DB."""
    db = open_db(f"sqlite:///{tmp_path / 'opened.db'}")
    db.set_current_version(1)
    info = _mk_infoset()
    db.put(info, _mk_mask(), _mk_probs(), version=1)
    assert db.get(info, version=1) is not None
    db.close()


# ───────── 7. test_action_probs_sum_to_one ─────────


def test_action_probs_must_sum_to_one(db: StrategyDB) -> None:
    info = _mk_infoset()
    bad = np.array([0.4, 0.4, 0.1], dtype=np.float32)  # sum = 0.9
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        db.put(info, _mk_mask(), bad, version=1)


def test_action_probs_length_must_match_mask(db: StrategyDB) -> None:
    info = _mk_infoset()
    mask = _mk_mask()  # 3 actions set
    too_short = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(ValueError, match="length"):
        db.put(info, mask, too_short, version=1)


def test_action_probs_must_be_float32(db: StrategyDB) -> None:
    info = _mk_infoset()
    bad_dtype = np.array([0.4, 0.4, 0.2], dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        db.put(info, _mk_mask(), bad_dtype, version=1)  # type: ignore[arg-type]


def test_action_probs_must_be_non_negative(db: StrategyDB) -> None:
    info = _mk_infoset()
    has_negative = np.array([1.2, -0.1, -0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="non-negative"):
        db.put(info, _mk_mask(), has_negative, version=1)


# ───────── 8. test_sqlite_lmdb_equivalence ─────────


def test_sqlite_lmdb_equivalence(tmp_path: Path) -> None:
    """Same writes → same reads across backends."""
    sq = SQLiteStrategyDB(str(tmp_path / "eq.db"))
    lm = LMDBStrategyDB(str(tmp_path / "eq.lmdb"))
    for d in (sq, lm):
        d.set_current_version(1)

    infos = [_mk_infoset(card_bucket=cb, history=bytes([cb])) for cb in range(5)]
    mask = _mk_mask()
    probs_list = [_mk_probs([0.1 * (cb + 1), 0.9 - 0.1 * cb, 0.0]) for cb in range(5)]
    # Normalize each row to sum exactly to 1.0 in float32.
    probs_list = [(p / p.sum()).astype(np.float32) for p in probs_list]

    for info, probs in zip(infos, probs_list, strict=True):
        sq.put(info, mask, probs, version=1, visit_count=int(info.card_bucket))
        lm.put(info, mask, probs, version=1, visit_count=int(info.card_bucket))

    for info, probs in zip(infos, probs_list, strict=True):
        rs = sq.get(info, version=1)
        rl = lm.get(info, version=1)
        assert rs is not None and rl is not None
        assert rs.action_mask == rl.action_mask
        assert rs.visit_count == rl.visit_count
        assert rs.version == rl.version
        np.testing.assert_array_almost_equal(rs.action_probs, rl.action_probs)
        np.testing.assert_array_almost_equal(rs.action_probs, probs)

    # nearest_neighbor on a never-seen-history infoset returns equivalent rows.
    probe = _mk_infoset(card_bucket=3, history=b"\xff\xff")
    rs_nn = sq.nearest_neighbor(probe, version=1)
    rl_nn = lm.nearest_neighbor(probe, version=1)
    assert rs_nn is not None and rl_nn is not None
    assert rs_nn.visit_count == rl_nn.visit_count == 3

    sq.close()
    lm.close()


# ───────── extras: version + lifecycle ─────────


def test_current_version_raises_when_unset(tmp_path: Path) -> None:
    db = SQLiteStrategyDB(str(tmp_path / "uv.db"))
    with pytest.raises(RuntimeError, match="not set"):
        db.current_version()
    db.close()


def test_get_uses_current_version_when_none(db: StrategyDB) -> None:
    info = _mk_infoset()
    db.put(info, _mk_mask(), _mk_probs(), version=1)
    # current_version was set to 1 by the fixture
    row = db.get(info)  # version=None
    assert row is not None
    assert row.version == 1


def test_set_current_version_persists(tmp_path: Path) -> None:
    path = str(tmp_path / "persist.db")
    db = SQLiteStrategyDB(path)
    db.set_current_version(7)
    db.close()
    db2 = SQLiteStrategyDB(path)
    assert db2.current_version() == 7
    db2.close()


def test_open_db_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match="unsupported scheme"):
        open_db("postgres:///foo")
    with pytest.raises(ValueError, match="missing"):
        open_db("not-a-uri")
