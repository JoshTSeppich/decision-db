"""Tests for MultiSizeStrategyDB routing (per-table-size policy DBs)."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pokerbot.abstraction import InfoSet
from pokerbot.strategy_db import open_db
from pokerbot.strategy_db.multi_size import SUPPORTED_TABLE_SIZES, MultiSizeStrategyDB

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pokerbot.strategy_db.base import StrategyDB

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import serve  # noqa: E402  (after sys.path insert)

_SALVAGE_YAML = _REPO_ROOT / "config" / "multi-size-batch1-salvage.yaml"


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


# ─────────── Batch-1 salvage integration tests ───────────
#
# These exercise the actual config/multi-size-batch1-salvage.yaml against the
# real on-disk DBs (production 6-max + 9-max baselines and the salvaged
# pilot-v4-3-iter_0100.db). They skip cleanly when those files aren't
# present (CI / fresh checkouts that haven't built the salvage DB).


def _resolve_salvage_paths() -> dict[int, Path]:
    """Parse the salvage YAML and resolve relative paths against the repo root."""
    raw = serve._load_multi_size_config(str(_SALVAGE_YAML))
    out: dict[int, Path] = {}
    for size, p in raw.items():
        candidate = Path(p)
        if not candidate.is_absolute():
            candidate = _REPO_ROOT / candidate
        out[size] = candidate.resolve()
    return out


@pytest.fixture(scope="module")
def salvage_paths() -> dict[int, Path]:
    if not _SALVAGE_YAML.exists():
        pytest.skip(f"salvage YAML not present at {_SALVAGE_YAML}")
    paths = _resolve_salvage_paths()
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"salvage DB files not present: {missing}")
    return paths


@pytest.fixture(scope="module")
def salvage_multi_db(
    salvage_paths: dict[int, Path],
) -> Iterator[MultiSizeStrategyDB]:
    per_size_dbs = {size: open_db(f"sqlite:///{p}") for size, p in salvage_paths.items()}
    multi = MultiSizeStrategyDB(per_size_dbs)
    yield multi
    multi.close()


def test_salvage_yaml_loads_and_constructs_multi_size_db(
    salvage_multi_db: MultiSizeStrategyDB,
) -> None:
    """All 8 sizes resolve to a registered DB, all sub-DBs share a version."""
    for size in (2, 3, 4, 5, 6, 7, 8, 9):
        # Routing succeeds (no KeyError) — using the public nearest_neighbor
        # path with a probe that won't match anything is enough to confirm
        # the size is registered.
        probe = InfoSet(table_size=size, street=0, position=0, stack_bucket=0, card_bucket=0)
        salvage_multi_db.nearest_neighbor(probe, version=1)
    assert salvage_multi_db.current_version() == 1


def test_salvage_real_size_lookup_returns_trained_row(
    salvage_paths: dict[int, Path],
    salvage_multi_db: MultiSizeStrategyDB,
) -> None:
    """For each "real" size in the YAML, look up an infoset that actually
    exists in that DB and confirm the routed lookup returns a valid row.

    Pulls a real (table_size, street, position, stack_bucket, card_bucket,
    history) tuple from each underlying DB to construct the InfoSet — the
    blake2b hash must match exactly so it round-trips through the wrapper.
    """
    real_sizes = (3, 6, 9)
    for size in real_sizes:
        with sqlite3.connect(salvage_paths[size]) as conn:
            row = conn.execute(
                "SELECT table_size, street, position, stack_bucket, card_bucket, "
                "history_blob FROM strategy WHERE table_size = ? AND version = 1 "
                "LIMIT 1",
                (size,),
            ).fetchone()
        assert row is not None, f"no rows at table_size={size} in registered DB"
        info = InfoSet(
            table_size=int(row[0]),
            street=int(row[1]),
            position=int(row[2]),
            stack_bucket=int(row[3]),
            card_bucket=int(row[4]),
            history=bytes(row[5]),
        )
        result = salvage_multi_db.get(info, version=1)
        assert result is not None, f"size={size}: routed get() should find the row"
        assert np.isclose(float(result.action_probs.sum()), 1.0, atol=1e-5), (
            f"size={size}: action_probs must sum to 1.0, got {result.action_probs.sum()}"
        )


def test_salvage_fallback_size_lookup_returns_none(
    salvage_multi_db: MultiSizeStrategyDB,
) -> None:
    """Sizes 4, 5, 7, 8 have no real policy; both get() and nearest_neighbor()
    must miss because table_size is part of the hash and the NN SQL filter.
    The adapter handles this by falling through to default_policy_action()."""
    for fallback_size in (4, 5, 7, 8):
        probe = InfoSet(
            table_size=fallback_size,
            street=0,  # preflop
            position=0,
            stack_bucket=4,
            card_bucket=0,
            history=b"",
        )
        exact = salvage_multi_db.get(probe, version=1)
        nn = salvage_multi_db.nearest_neighbor(probe, version=1)
        assert exact is None, (
            f"size={fallback_size}: exact get() must miss against a proxy DB"
        )
        assert nn is None, (
            f"size={fallback_size}: nearest_neighbor must miss "
            f"(table_size SQL filter excludes the proxy's rows)"
        )
