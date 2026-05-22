"""Tests for MultiSizeStrategyDB routing (per-table-size policy DBs)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pokerbot.abstraction import InfoSet
from pokerbot.strategy_db import open_db
from pokerbot.strategy_db.multi_size import SUPPORTED_TABLE_SIZES, MultiSizeStrategyDB

if TYPE_CHECKING:
    from pokerbot.strategy_db.base import StrategyDB


def _make_db(size: int) -> tuple[StrategyDB, InfoSet]:
    """Build an in-memory DB seeded with one row at table_size=`size`."""
    db = open_db("sqlite:///:memory:")
    db.set_current_version(1)
    info = InfoSet(table_size=size, street=0, position=0, stack_bucket=4, card_bucket=0)
    probs = np.array([0.5, 0.5], dtype=np.float32)
    db.put(info, action_mask=0b11, action_probs=probs, version=1, visit_count=10)
    return db, info


def _make_all_eight() -> tuple[dict[int, StrategyDB], dict[int, InfoSet]]:
    dbs: dict[int, StrategyDB] = {}
    infos: dict[int, InfoSet] = {}
    for size in sorted(SUPPORTED_TABLE_SIZES):
        db, info = _make_db(size)
        dbs[size] = db
        infos[size] = info
    return dbs, infos


def test_routes_every_supported_table_size_to_its_own_db() -> None:
    """For each size 2..9, get() routes to the matching DB and finds the row."""
    dbs, infos = _make_all_eight()
    multi = MultiSizeStrategyDB(dbs)
    for size in sorted(SUPPORTED_TABLE_SIZES):
        row = multi.get(infos[size], version=1)
        assert row is not None, f"size={size} should hit its own DB"
        assert row.visit_count == 10


def test_byte_identical_to_underlying_db_per_size() -> None:
    """For each size, multi.get(info) returns exactly what dbs[size].get(info) returns."""
    dbs, infos = _make_all_eight()
    multi = MultiSizeStrategyDB(dbs)
    for size, info in infos.items():
        underlying = dbs[size].get(info, version=1)
        routed = multi.get(info, version=1)
        assert underlying is not None and routed is not None
        assert underlying.action_mask == routed.action_mask
        assert underlying.visit_count == routed.visit_count
        assert underlying.version == routed.version
        np.testing.assert_array_equal(underlying.action_probs, routed.action_probs)


def test_missing_table_size_raises_keyerror() -> None:
    """If a size isn't registered, _route raises KeyError with the missing size."""
    db_6, _ = _make_db(6)
    multi = MultiSizeStrategyDB({6: db_6})
    probe = InfoSet(table_size=4, street=0, position=0, stack_bucket=4, card_bucket=0)
    with pytest.raises(KeyError, match="table_size=4"):
        multi.get(probe, version=1)


def test_partial_registration_works_for_registered_sizes() -> None:
    """A multi-DB with only some sizes still serves those sizes."""
    db_2, info_2 = _make_db(2)
    db_9, info_9 = _make_db(9)
    multi = MultiSizeStrategyDB({2: db_2, 9: db_9})
    assert multi.get(info_2, version=1) is not None
    assert multi.get(info_9, version=1) is not None


def test_constructor_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiSizeStrategyDB({})


def test_constructor_rejects_out_of_range_keys() -> None:
    db_6, _ = _make_db(6)
    with pytest.raises(ValueError, match="out-of-range"):
        MultiSizeStrategyDB({1: db_6})
    with pytest.raises(ValueError, match="out-of-range"):
        MultiSizeStrategyDB({10: db_6})


def test_constructor_rejects_version_mismatch() -> None:
    """All underlying DBs must agree on current_version()."""
    dbs, _ = _make_all_eight()
    dbs[3].set_current_version(2)
    with pytest.raises(ValueError, match="matching current_version"):
        MultiSizeStrategyDB(dbs)


def test_nearest_neighbor_routes_per_size() -> None:
    """NN lookups also route by table_size."""
    dbs, _ = _make_all_eight()
    multi = MultiSizeStrategyDB(dbs)
    for size in sorted(SUPPORTED_TABLE_SIZES):
        probe = InfoSet(
            table_size=size,
            street=0,
            position=0,
            stack_bucket=4,
            card_bucket=0,
            history=b"\x01\x02",
        )
        assert multi.nearest_neighbor(probe, version=1) is not None, (
            f"NN at size={size} should hit registered DB"
        )


def test_is_read_only() -> None:
    """put() and bulk_put() raise — MultiSizeStrategyDB is read-only."""
    db_6, _ = _make_db(6)
    multi = MultiSizeStrategyDB({6: db_6})
    info = InfoSet(table_size=6, street=0, position=0, stack_bucket=4, card_bucket=0)
    probs = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(RuntimeError, match="read-only"):
        multi.put(info, action_mask=0b11, action_probs=probs, version=1)
    with pytest.raises(RuntimeError, match="read-only"):
        multi.bulk_put([(info, 0b11, probs)], version=1)


def test_set_current_version_propagates_to_all_underlying() -> None:
    dbs, _ = _make_all_eight()
    multi = MultiSizeStrategyDB(dbs)
    multi.set_current_version(7)
    assert multi.current_version() == 7
    for db in dbs.values():
        assert db.current_version() == 7


def test_close_closes_all_underlying() -> None:
    """close() must call close on every underlying DB."""
    closed: list[int] = []

    class _RecordingDB:
        def __init__(self, size: int) -> None:
            self._size = size
            self._ver = 1

        def current_version(self) -> int:
            return self._ver

        def set_current_version(self, v: int) -> None:
            self._ver = v

        def close(self) -> None:
            closed.append(self._size)

    fakes = {s: _RecordingDB(s) for s in (2, 5, 9)}
    multi = MultiSizeStrategyDB(fakes)  # type: ignore[arg-type]
    multi.close()
    assert sorted(closed) == [2, 5, 9]


def test_supported_sizes_match_table_size_field_validation() -> None:
    """SUPPORTED_TABLE_SIZES should match the {2..9} range InfoSet accepts."""
    assert set(SUPPORTED_TABLE_SIZES) == {2, 3, 4, 5, 6, 7, 8, 9}
