"""Spec.html §F tests for the runtime adapter.

Spec tests:
    1. test_known_infoset_round_trip
    2. test_unknown_infoset_falls_back_to_nn
    3. test_no_neighbors_falls_back_to_default
    4. test_latency_p99_under_100ms
    5. test_illegal_action_never_emitted
    6. test_opponent_model_injection
    7. test_schema_v1_strict
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from pokerbot.abstraction import AbstractionTables, ActionType, InfoSet
from pokerbot.runtime import (
    ActionHistoryEntry,
    BlindsSchema,
    GameStateRequest,
    IdentityOpponentModel,
    ObservedHistory,
    OpponentModel,
    RuntimeAdapter,
)
from pokerbot.strategy_db import SQLiteStrategyDB

if TYPE_CHECKING:
    from pathlib import Path

    from pokerbot.strategy_db import StrategyDB


# ───────── fixtures + helpers ─────────


def _mk_request(
    *,
    board: list[str] | None = None,
    hero_seat: int = 2,
    button_seat: int = 0,
    table_size: int = 6,
    action_history: list[ActionHistoryEntry] | None = None,
    to_call: int = 0,
    pot_committed: int = 30,
    stacks: list[int] | None = None,
    min_raise: int = 10,
    max_raise: int | None = None,
) -> GameStateRequest:
    if stacks is None:
        stacks = [1000] * table_size
    if max_raise is None:
        max_raise = stacks[hero_seat]
    return GameStateRequest(
        schema_version=1,
        game_type="cash",
        table_size=table_size,  # type: ignore[arg-type]  # Literal[6,8,9] enforced at runtime
        blinds=BlindsSchema(sb=5, bb=10),
        ante=0,
        hero_seat=hero_seat,
        button_seat=button_seat,
        hero_hole=["As", "Kh"],
        board=board if board is not None else ["7c", "2d", "Jh"],
        stacks=stacks,
        current_bets=[0] * table_size,
        pot_committed=pot_committed,
        to_call=to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        action_history=action_history if action_history is not None else [],
    )


@pytest.fixture
def db(tmp_path: Path) -> StrategyDB:
    d = SQLiteStrategyDB(str(tmp_path / "rt.db"))
    d.set_current_version(1)
    return d


@pytest.fixture
def adapter(db: StrategyDB) -> RuntimeAdapter:
    abstraction = AbstractionTables()  # empty → placeholder bucketing
    return RuntimeAdapter(db=db, abstraction=abstraction, rng_seed=42)


# ───────── 1. test_known_infoset_round_trip ─────────


def test_known_infoset_round_trip(db: StrategyDB, adapter: RuntimeAdapter) -> None:
    """Put a row with a deterministic action; query via JSON; get back the same action."""
    request = _mk_request()
    infoset = adapter.build_infoset(request)

    mask = 1 << int(ActionType.BET_66)
    probs = np.array([1.0], dtype=np.float32)
    db.put(infoset, mask, probs, version=1, visit_count=10)

    response = adapter.decide(request)
    assert response.fallback_used == "exact"
    assert response.abstract_action == "BET_66"
    assert response.action == "bet"  # to_call == 0 → "bet"
    # 0.66 * pot = 0.66 * 30 ≈ 20; we expect a bet of that order, clamped to stack.
    assert response.amount > 0
    assert response.amount <= request.stacks[request.hero_seat]
    assert response.probability_sampled == pytest.approx(1.0)
    assert response.infoset_hash == infoset.hash16().hex()


# ───────── 2. test_unknown_infoset_falls_back_to_nn ─────────


def test_unknown_infoset_falls_back_to_nn(db: StrategyDB, adapter: RuntimeAdapter) -> None:
    """Different history hashes differ; nearest_neighbor finds a sibling."""
    request_a = _mk_request(
        action_history=[ActionHistoryEntry(seat=2, street=0, type="raise", amount=30)]
    )
    request_b = _mk_request(action_history=[])  # same prefix (cards/pos/stack), no history
    info_a = adapter.build_infoset(request_a)
    info_b = adapter.build_infoset(request_b)
    assert info_a.hash16() != info_b.hash16()

    mask = 1 << int(ActionType.CHECK_CALL)
    probs = np.array([1.0], dtype=np.float32)
    db.put(info_a, mask, probs, version=1, visit_count=5)

    response = adapter.decide(request_b)
    assert response.fallback_used == "nearest_neighbor"
    assert response.abstract_action == "CHECK_CALL"
    assert response.action == "check"


# ───────── 3. test_no_neighbors_falls_back_to_default ─────────


def test_no_neighbors_falls_back_to_default(adapter: RuntimeAdapter) -> None:
    """Empty DB → default-policy action; response is still a valid AbstractAction."""
    request = _mk_request(board=[])  # preflop
    response = adapter.decide(request)
    assert response.fallback_used == "default_policy"
    assert response.action in {"fold", "check", "call", "bet", "raise"}
    assert response.amount >= 0
    assert response.amount <= request.stacks[request.hero_seat]


def test_no_neighbors_postflop_returns_check_or_fold(adapter: RuntimeAdapter) -> None:
    """Postflop default with no showdown value + no call owed: check."""
    request = _mk_request(board=["7c", "2d", "9h"], to_call=0)
    response = adapter.decide(request)
    assert response.fallback_used == "default_policy"
    # AKh on 7-2-9 board has no pair → default checks postflop
    assert response.action == "check"


# ───────── 4. test_latency_p99_under_100ms ─────────


@pytest.mark.parametrize("seed", [123])
def test_latency_p99_under_100ms(db: StrategyDB, adapter: RuntimeAdapter, seed: int) -> None:
    """10k random requests, p99 latency < 100ms."""
    rng = random.Random(seed)
    # Pre-populate the DB with a handful of rows so some lookups hit "exact".
    for i in range(50):
        info = InfoSet(
            table_size=6,
            street=1,
            position=2,
            stack_bucket=4,
            card_bucket=i,
            history=bytes([i]),
        )
        mask = 1 << int(ActionType.CHECK_CALL)
        probs = np.array([1.0], dtype=np.float32)
        db.put(info, mask, probs, version=1)

    latencies: list[float] = []
    cards = [
        "2c",
        "3c",
        "4c",
        "5c",
        "6c",
        "7c",
        "8c",
        "9c",
        "Tc",
        "Jc",
        "Qc",
        "Kc",
        "Ac",
        "2d",
        "3d",
        "4d",
        "5d",
        "6d",
        "7d",
        "8d",
    ]
    for _ in range(10_000):
        rng.shuffle(cards)
        hole = cards[0:2]
        board = cards[2:5]
        req = _mk_request(board=board)
        # patch hero_hole via copy
        req = req.model_copy(update={"hero_hole": hole})
        t0 = time.perf_counter()
        adapter.decide(req)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    p99_ms = latencies[int(0.99 * len(latencies))] * 1000
    assert p99_ms < 100.0, f"p99 latency = {p99_ms:.2f}ms (budget 100ms)"


# ───────── 5. test_illegal_action_never_emitted ─────────


def test_illegal_action_clamped_to_stack(db: StrategyDB, adapter: RuntimeAdapter) -> None:
    """DB has BET_150 with stack < 1.5x-pot. Strategy lookup yields BET_150; emitted
    chip amount is clamped to the stack (= all-in).
    """
    stacks = [120] * 6  # stack = 120 = less than 1.5 * pot=100
    request = _mk_request(pot_committed=100, stacks=stacks, to_call=0)
    infoset = adapter.build_infoset(request)

    mask = 1 << int(ActionType.BET_150)
    probs = np.array([1.0], dtype=np.float32)
    db.put(infoset, mask, probs, version=1)

    response = adapter.decide(request)
    assert response.abstract_action == "BET_150"
    assert response.amount == 120, f"expected stack clamp to 120, got {response.amount}"


# ───────── 6. test_opponent_model_injection ─────────


class _ForceFoldModel(OpponentModel):
    def adjust(
        self,
        infoset: InfoSet,
        base_probs: dict[ActionType, float],
        observed_history: ObservedHistory,
    ) -> dict[ActionType, float]:
        return {ActionType.FOLD: 1.0}


def test_opponent_model_injection_overrides_to_fold(tmp_path: Path) -> None:
    db = SQLiteStrategyDB(str(tmp_path / "om.db"))
    db.set_current_version(1)
    abstraction = AbstractionTables()
    adapter = RuntimeAdapter(
        db=db, abstraction=abstraction, opponent_model=_ForceFoldModel(), rng_seed=0
    )

    request = _mk_request(to_call=10, board=["7c", "2d", "Jh"])
    infoset = adapter.build_infoset(request)
    mask = 1 << int(ActionType.CHECK_CALL)
    db.put(infoset, mask, np.array([1.0], dtype=np.float32), version=1)

    response = adapter.decide(request)
    assert response.abstract_action == "FOLD"
    assert response.action == "fold"
    assert response.amount == 0


def test_default_opponent_model_is_identity(db: StrategyDB) -> None:
    adapter = RuntimeAdapter(db=db, abstraction=AbstractionTables(), rng_seed=0)
    assert isinstance(adapter.opponent_model, IdentityOpponentModel)


# ───────── 7. test_schema_v1_strict ─────────


def test_schema_v1_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GameStateRequest(
            schema_version=1,
            game_type="cash",
            table_size=6,
            blinds=BlindsSchema(sb=5, bb=10),
            ante=0,
            hero_seat=2,
            button_seat=0,
            hero_hole=["As", "Kh"],
            board=[],
            stacks=[1000] * 6,
            current_bets=[0] * 6,
            pot_committed=0,
            to_call=0,
            min_raise=10,
            max_raise=1000,
            action_history=[],
            extra_field="boom",  # type: ignore[call-arg]
        )


def test_schema_v1_rejects_schema_version_2() -> None:
    with pytest.raises(ValidationError):
        GameStateRequest(
            schema_version=2,
            game_type="cash",
            table_size=6,
            blinds=BlindsSchema(sb=5, bb=10),
            hero_seat=0,
            button_seat=0,
            hero_hole=["As", "Kh"],
            board=[],
            stacks=[100, 100],
            current_bets=[0, 0],
            pot_committed=0,
            to_call=0,
            min_raise=10,
            max_raise=100,
        )


# ───────── extra sanity ─────────


def test_build_infoset_position_relative_to_sb(adapter: RuntimeAdapter) -> None:
    """SB-relative position: button=0, hero=2 in 6-max → SB is seat 1 → hero pos=1."""
    request = _mk_request(table_size=6, button_seat=0, hero_seat=2)
    info = adapter.build_infoset(request)
    assert info.position == 1  # (2 - 1) mod 6


def test_build_infoset_position_in_9max(adapter: RuntimeAdapter) -> None:
    """9-max: button=4 → SB=5 → hero at seat 8 → position = (8-5) mod 9 = 3."""
    request = _mk_request(
        table_size=9,
        button_seat=4,
        hero_seat=8,
        stacks=[1000] * 9,
        board=[],
    )
    info = adapter.build_infoset(request)
    assert info.position == 3


def test_schema_accepts_all_table_sizes_2_through_9() -> None:
    """table_size widened to {2..9} for tournament play (Cairn 5 Fix 2)."""
    for n in (2, 3, 4, 5, 6, 7, 8, 9):
        # Construct with each n; should not raise. hero_seat=0 keeps validity at n=2.
        _mk_request(table_size=n, hero_seat=0, button_seat=0, stacks=[1000] * n, board=[])
    with pytest.raises(ValidationError):
        _mk_request(table_size=10, hero_seat=0, button_seat=0, stacks=[1000] * 10, board=[])
    with pytest.raises(ValidationError):
        _mk_request(table_size=1, hero_seat=0, button_seat=0, stacks=[1000], board=[])


def test_build_infoset_stack_bucket_clamps_to_top(adapter: RuntimeAdapter) -> None:
    """Effective stack > 300 BB → bucket 9 (top)."""
    request = _mk_request(stacks=[5000] * 6, board=[])  # 5000 / BB(10) = 500 BB
    info = adapter.build_infoset(request)
    assert info.stack_bucket == 9


def test_build_infoset_history_includes_boundary_byte(adapter: RuntimeAdapter) -> None:
    request = _mk_request(
        board=["7c", "2d", "Jh"],
        action_history=[
            ActionHistoryEntry(seat=2, street=0, type="raise", amount=30),
            ActionHistoryEntry(seat=3, street=0, type="call", amount=30),
            ActionHistoryEntry(seat=2, street=1, type="bet", amount=20),
        ],
    )
    info = adapter.build_infoset(request)
    assert 0xF0 in info.history, "expected street boundary byte"
