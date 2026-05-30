"""Regression tests for the Component-4 DB lookup coverage bug.

The bug: training/export keys DB rows by EFFECTIVE stack = min(hero, max opp)
(nlhe_game.py / encoding.effective_stack), but make_db_spot_policy looked rows up by
the hero's RAW stack. They differ for the chip leader (the button at hand start: full
100bb vs blinds-posted ~99.5bb opponents), so the button's opens — stored at the
effective-stack bucket — were queried at an over-deep, empty bucket → miss → silent
CHECK_CALL fallback. That inflated VPIP and suppressed PFR. These tests pin the fix:
the lookup must bucket by effective stack, and a miss must be a detectable empty result.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.abstraction import AbstractionTables, ActionType, InfoSet
from pokerbot.abstraction.encoding import effective_stack, stack_bucket_from_eff_bb
from pokerbot.strategy_db import open_db
from zoom.agents import AgentSpot
from zoom.eval import ScreenCoverageError, catastrophic_screen
from zoom.eval.db_policy import make_db_spot_policy

_BB = 10
_TABLE_SIZE = 3
_POSITION = 2  # button (SB-relative) — the chip leader at hand start
_HOLE = (48, 49)  # Ac Ad


def _raise_row_db(abstraction: AbstractionTables, *, stack_bucket: int):
    """In-memory DB holding ONE distinctive RAISE_2_5X row for the button-open spot,
    keyed at `stack_bucket`. Returns (db, card_bucket)."""
    db = open_db("sqlite:///:memory:")
    card_bucket = int(abstraction.lookup(_HOLE, (), "preflop"))
    mask = 1 << int(ActionType.RAISE_2_5X)
    probs = np.array([1.0], dtype=np.float32)
    info = InfoSet(
        table_size=_TABLE_SIZE,
        street=0,
        position=_POSITION,
        stack_bucket=stack_bucket,
        card_bucket=card_bucket,
        history=b"",
    )
    db.put(info, mask, probs, version=1)
    db.set_current_version(1)
    return db


def _button_open_spot(*, stack: int, effective: int | None) -> AgentSpot:
    return AgentSpot(
        hole=_HOLE,
        board=(),
        street="preflop",
        position=_POSITION,
        pot=15,
        to_call=_BB,
        stack=stack,
        min_raise=_BB,
        table_size=_TABLE_SIZE,
        effective_stack=effective,
    )


def test_chip_leader_lookup_uses_effective_stack() -> None:
    """The chip-leader button-open must HIT the row stored at its EFFECTIVE-stack bucket.

    Hero raw stack 100bb (bucket 6, empty) but effective stack ~99.5bb (bucket 5, where
    the export wrote the row). Pre-fix: buckets by raw stack → miss → no RAISE mass (RED).
    Post-fix: buckets by effective stack → hits the RAISE row (GREEN).
    """
    abstraction = AbstractionTables()
    hero_stack = 100 * _BB  # 1000 = 100bb
    opp_stack = hero_stack - 5  # blinds-posted opponent, ~99.5bb
    eff = effective_stack(hero_stack, [opp_stack])
    eff_bucket = stack_bucket_from_eff_bb(eff // _BB)
    raw_bucket = stack_bucket_from_eff_bb(hero_stack // _BB)
    assert eff_bucket != raw_bucket, "test premise: chip-leader raw vs effective buckets differ"

    db = _raise_row_db(abstraction, stack_bucket=eff_bucket)
    policy = make_db_spot_policy(db, abstraction, bb=_BB, table_size=_TABLE_SIZE)
    dist = policy(_button_open_spot(stack=hero_stack, effective=eff))
    assert dist.get(ActionType.RAISE_2_5X, 0.0) > 0.5, (
        f"expected a HIT on the RAISE row at effective bucket {eff_bucket}, got {dist}"
    )


def test_effective_stack_none_falls_back_to_raw_stack() -> None:
    """effective_stack=None reproduces raw-stack bucketing (scripted-agent/back-compat path)."""
    abstraction = AbstractionTables()
    stack = 95 * _BB  # 950 → bucket from raw stack
    raw_bucket = stack_bucket_from_eff_bb(stack // _BB)
    db = _raise_row_db(abstraction, stack_bucket=raw_bucket)
    policy = make_db_spot_policy(db, abstraction, bb=_BB, table_size=_TABLE_SIZE)
    dist = policy(_button_open_spot(stack=stack, effective=None))
    assert dist.get(ActionType.RAISE_2_5X, 0.0) > 0.5, f"None should bucket by raw stack; got {dist}"


def test_lookup_miss_returns_empty_distribution() -> None:
    """A miss must be a DETECTABLE empty mapping (not a silent CHECK_CALL), so callers
    like the catastrophic screen can tell 'no data' from a real check-call policy."""
    abstraction = AbstractionTables()
    db = open_db("sqlite:///:memory:")
    db.set_current_version(1)  # empty DB → every lookup misses
    policy = make_db_spot_policy(db, abstraction, bb=_BB, table_size=_TABLE_SIZE)
    dist = policy(_button_open_spot(stack=100 * _BB, effective=995))
    assert dist == {}, f"miss must return an empty mapping, got {dist}"


def test_exact_history_lookup_beats_higher_visit_nearest() -> None:
    """The lookup must try an EXACT history match first (mirroring production's
    adapter: db.get → nearest_neighbor). Without it, a history-blind nearest_neighbor
    returns the highest-visit row in the cell regardless of the real betting context —
    the SB/BB history-blindness root. Here the spot's true history row (RAISE) must win
    over a different, higher-visit history row (FOLD) in the same cell.
    """
    abstraction = AbstractionTables()
    stack = 100 * _BB
    eff = effective_stack(stack, [stack - 5])
    sbk = stack_bucket_from_eff_bb(eff // _BB)
    cb = int(abstraction.lookup(_HOLE, (), "preflop"))
    db = open_db("sqlite:///:memory:")

    def info(hist: bytes):
        return InfoSet(table_size=_TABLE_SIZE, street=0, position=_POSITION,
                       stack_bucket=sbk, card_bucket=cb, history=hist)

    # The spot's true-history row: RAISE, low visit count.
    db.put(info(b"\x01"), 1 << int(ActionType.RAISE_2_5X), np.array([1.0], np.float32), 1, visit_count=1)
    # A different history in the SAME cell: FOLD, much higher visit count (what a
    # history-blind nearest_neighbor would return).
    db.put(info(b"\x02"), 1 << int(ActionType.FOLD), np.array([1.0], np.float32), 1, visit_count=99)
    db.set_current_version(1)

    spot = AgentSpot(hole=_HOLE, board=(), street="preflop", position=_POSITION,
                     pot=15, to_call=_BB, stack=stack, min_raise=_BB,
                     table_size=_TABLE_SIZE, effective_stack=eff, history=b"\x01")
    policy = make_db_spot_policy(db, abstraction, bb=_BB, table_size=_TABLE_SIZE)
    dist = policy(spot)
    assert dist.get(ActionType.RAISE_2_5X, 0.0) > 0.5, (
        f"exact-history row (RAISE) must win over higher-visit nearest row (FOLD); got {dist}"
    )


def test_screen_raises_on_uncovered_probe_instead_of_vacuous_clean() -> None:
    """A DB with no coverage at the screen probes must make catastrophic_screen RAISE,
    not return [] — closing the false-clean path (a screen that never saw the policy
    cannot certify it doesn't fold AA)."""
    abstraction = AbstractionTables()
    db = open_db("sqlite:///:memory:")
    db.set_current_version(1)  # empty DB → every probe misses
    policy = make_db_spot_policy(db, abstraction, bb=_BB, table_size=_TABLE_SIZE)
    with pytest.raises(ScreenCoverageError):
        catastrophic_screen(policy)
