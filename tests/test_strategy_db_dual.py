"""Tests for DualStrategyDB routing (Cairn 6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pokerbot.abstraction import InfoSet
from pokerbot.strategy_db import open_db
from pokerbot.strategy_db.dual import NINE_MAX_ROUTE, SIX_MAX_ROUTE, DualStrategyDB

if TYPE_CHECKING:
    from pokerbot.strategy_db.base import StrategyDB


def _make_db(size: int) -> tuple[StrategyDB, InfoSet]:
    """Build an in-memory DB with one row at table_size=`size`."""
    db = open_db("sqlite:///:memory:")
    db.set_current_version(1)
    info = InfoSet(table_size=size, street=0, position=0, stack_bucket=4, card_bucket=0)
    probs = np.array([0.5, 0.5], dtype=np.float32)
    db.put(info, action_mask=0b11, action_probs=probs, version=1, visit_count=10)
    return db, info


def test_dual_db_routes_table_size_6_to_primary() -> None:
    """table_size=6 → primary DB."""
    primary, info_6 = _make_db(6)
    secondary, _ = _make_db(9)
    dual = DualStrategyDB(primary, secondary)
    row = dual.get(info_6, version=1)
    assert row is not None, "primary should have the table_size=6 row"


def test_dual_db_routes_table_size_9_to_secondary() -> None:
    """table_size=9 → secondary DB."""
    primary, _ = _make_db(6)
    secondary, info_9 = _make_db(9)
    dual = DualStrategyDB(primary, secondary)
    row = dual.get(info_9, version=1)
    assert row is not None, "secondary should have the table_size=9 row"


def test_dual_db_routes_table_size_8_to_secondary() -> None:
    """table_size=8 → secondary (9-max DB; closer to 9)."""
    primary, _ = _make_db(6)
    secondary, info_8 = _make_db(8)
    dual = DualStrategyDB(primary, secondary)
    row = dual.get(info_8, version=1)
    assert row is not None


def test_dual_db_routes_table_size_7_to_secondary() -> None:
    """table_size=7 → secondary (closer to 9 than to 6)."""
    primary, _ = _make_db(6)
    secondary, info_7 = _make_db(7)
    dual = DualStrategyDB(primary, secondary)
    row = dual.get(info_7, version=1)
    assert row is not None


def test_dual_db_routes_small_table_sizes_to_secondary() -> None:
    """table_size in {2, 3, 4, 5} → secondary (more diverse postflop data)."""
    primary, _ = _make_db(6)
    for n in (2, 3, 4, 5):
        secondary, info_n = _make_db(n)
        dual = DualStrategyDB(primary, secondary)
        row = dual.get(info_n, version=1)
        assert row is not None, f"table_size={n} should route to secondary and hit"


def test_dual_db_byte_identical_for_table_size_6() -> None:
    """For table_size=6, dual.get returns exactly what primary.get returns."""
    primary, info_6 = _make_db(6)
    secondary, _ = _make_db(9)
    dual = DualStrategyDB(primary, secondary)
    row_primary = primary.get(info_6, version=1)
    row_dual = dual.get(info_6, version=1)
    assert row_primary is not None and row_dual is not None
    assert row_primary.action_mask == row_dual.action_mask
    assert row_primary.visit_count == row_dual.visit_count
    assert row_primary.version == row_dual.version
    np.testing.assert_array_equal(row_primary.action_probs, row_dual.action_probs)


def test_dual_db_byte_identical_for_table_size_9() -> None:
    """For table_size=9, dual.get returns exactly what secondary.get returns."""
    primary, _ = _make_db(6)
    secondary, info_9 = _make_db(9)
    dual = DualStrategyDB(primary, secondary)
    row_secondary = secondary.get(info_9, version=1)
    row_dual = dual.get(info_9, version=1)
    assert row_secondary is not None and row_dual is not None
    assert row_secondary.action_mask == row_dual.action_mask
    assert row_secondary.visit_count == row_dual.visit_count
    assert row_secondary.version == row_dual.version
    np.testing.assert_array_equal(row_secondary.action_probs, row_dual.action_probs)


def test_dual_db_rejects_version_mismatch() -> None:
    """Construction fails if the two DBs have different current_versions."""
    primary, _ = _make_db(6)
    secondary, _ = _make_db(9)
    secondary.set_current_version(2)
    with pytest.raises(ValueError, match="matching current_version"):
        DualStrategyDB(primary, secondary)


def test_dual_db_nearest_neighbor_routes_correctly() -> None:
    """nearest_neighbor() also routes by table_size."""
    primary, _info_6 = _make_db(6)
    secondary, _info_9 = _make_db(9)
    dual = DualStrategyDB(primary, secondary)
    # Build a probe infoset with a different history but same prefix as the seeded
    # row; nearest_neighbor falls back to (table_size, street, card_bucket,
    # position, stack_bucket) so it should still hit.
    probe_6 = InfoSet(
        table_size=6, street=0, position=0, stack_bucket=4, card_bucket=0,
        history=b"\x01\x02",
    )
    probe_9 = InfoSet(
        table_size=9, street=0, position=0, stack_bucket=4, card_bucket=0,
        history=b"\x01\x02",
    )
    row_6 = dual.nearest_neighbor(probe_6, version=1)
    row_9 = dual.nearest_neighbor(probe_9, version=1)
    assert row_6 is not None, "nearest_neighbor at table_size=6 should hit primary"
    assert row_9 is not None, "nearest_neighbor at table_size=9 should hit secondary"


def test_dual_db_is_read_only() -> None:
    """put() and bulk_put() raise — DualStrategyDB is read-only."""
    primary, _ = _make_db(6)
    secondary, _ = _make_db(9)
    dual = DualStrategyDB(primary, secondary)
    info = InfoSet(table_size=6, street=0, position=0, stack_bucket=4, card_bucket=0)
    probs = np.array([0.5, 0.5], dtype=np.float32)
    with pytest.raises(RuntimeError, match="read-only"):
        dual.put(info, action_mask=0b11, action_probs=probs, version=1)
    with pytest.raises(RuntimeError, match="read-only"):
        dual.bulk_put([(info, 0b11, probs)], version=1)


def test_route_sets_partition_all_supported_sizes() -> None:
    """SIX_MAX_ROUTE U NINE_MAX_ROUTE covers exactly {2..9}, no overlap."""
    assert SIX_MAX_ROUTE.isdisjoint(NINE_MAX_ROUTE)
    assert frozenset({2, 3, 4, 5, 6, 7, 8, 9}) == SIX_MAX_ROUTE | NINE_MAX_ROUTE
