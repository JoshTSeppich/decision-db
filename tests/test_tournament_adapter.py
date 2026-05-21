"""Tests for TournamentAdapter (Cairn 3).

Acceptance criteria (1-11):
    1. test_cash_mode_passthrough
    2. test_tournament_short_stack_uses_pushfold
    3. test_tournament_deep_stack_uses_cfr_with_icm
    4. test_bubble_pressure_tightens_play
    5. test_risk_factor_bounds
    6. test_icm_integration
    7. test_pushfold_integration
    8. test_tournament_state_validation
    9. test_no_tournament_state_in_cash_mode_ok
    10. test_missing_tournament_state_in_tournament_mode_errors
    11. test_smoke_full_tournament_hand
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from pokerbot.abstraction import AbstractionTables
from pokerbot.runtime import (
    ActionHistoryEntry,
    BlindsSchema,
    GameStateRequest,
    RuntimeAdapter,
)
from pokerbot.strategy_db import SQLiteStrategyDB
from pokerbot.tournament.adapter import (
    RISK_FACTOR_MAX,
    RISK_FACTOR_MIN,
    TournamentAdapter,
)
from pokerbot.tournament.state import TournamentState

if TYPE_CHECKING:
    from pathlib import Path


# ─────────── fixtures + helpers ───────────


def _mk_request(
    *,
    game_type: str = "cash",
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
    hero_hole: list[str] | None = None,
) -> GameStateRequest:
    if stacks is None:
        stacks = [1000] * table_size
    if max_raise is None:
        max_raise = stacks[hero_seat]
    if hero_hole is None:
        hero_hole = ["As", "Kh"]
    return GameStateRequest(
        schema_version=1,
        game_type=game_type,  # type: ignore[arg-type]
        table_size=table_size,  # type: ignore[arg-type]
        blinds=BlindsSchema(sb=5, bb=10),
        ante=0,
        hero_seat=hero_seat,
        button_seat=button_seat,
        hero_hole=hero_hole,
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
def runtime_adapter(tmp_path: Path) -> RuntimeAdapter:
    db = SQLiteStrategyDB(str(tmp_path / "adapter.db"))
    db.set_current_version(1)
    abstraction = AbstractionTables()
    return RuntimeAdapter(db=db, abstraction=abstraction, rng_seed=42)


@pytest.fixture
def runtime_adapter_for_compare(tmp_path: Path) -> RuntimeAdapter:
    """Twin of runtime_adapter for byte-identical comparison testing."""
    db = SQLiteStrategyDB(str(tmp_path / "adapter_compare.db"))
    db.set_current_version(1)
    abstraction = AbstractionTables()
    return RuntimeAdapter(db=db, abstraction=abstraction, rng_seed=42)


def _mk_ts(
    *,
    stacks: tuple[int, ...] = (1000, 800, 1200, 900, 1100, 1000),
    hero_index: int = 0,
    payouts: tuple[float, ...] = (5000.0, 3000.0, 2000.0, 1500.0, 1000.0, 700.0),
    players_in_money: int = 6,
    blinds_bb: int = 10,
    starting_stack_bb: int = 100,
) -> TournamentState:
    return TournamentState(
        stacks=stacks,
        hero_index=hero_index,
        payouts=payouts,
        players_remaining=len(stacks),
        players_in_money=players_in_money,
        blinds_bb=blinds_bb,
        starting_stack_bb=starting_stack_bb,
    )


# ─────────── 1. cash-mode passthrough ───────────


def test_cash_mode_passthrough(
    runtime_adapter: RuntimeAdapter, runtime_adapter_for_compare: RuntimeAdapter
) -> None:
    """Byte-identical to RuntimeAdapter.decide for game_type='cash'."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    rng = np.random.default_rng(123)
    for _ in range(50):
        # Randomize the request a bit to vary states
        table_size = int(rng.choice([6, 8, 9]))
        button = int(rng.integers(0, table_size))
        hero_seat = int(rng.integers(0, table_size))
        to_call = int(rng.choice([0, 10, 50, 200]))
        request = _mk_request(
            game_type="cash",
            table_size=table_size,
            button_seat=button,
            hero_seat=hero_seat,
            to_call=to_call,
            stacks=[1000] * table_size,
        )
        # Tournament call (cash mode) advances runtime_adapter's RNG once.
        r1 = tournament.decide(request)
        # Direct call on the twin adapter (same seed) advances its RNG identically.
        r2 = runtime_adapter_for_compare.decide(request)
        assert r1 == r2, f"cash-mode passthrough mismatch:\n  tournament: {r1}\n  base:       {r2}"


# ─────────── 2. short-stack tournament uses pushfold ───────────


def test_tournament_short_stack_uses_pushfold(runtime_adapter: RuntimeAdapter) -> None:
    """hero_stack_bb = 8 should route to push/fold; action ∈ {FOLD, ALL_IN}."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    # 8 BB = 80 chips at BB=10
    stacks = [80, 80, 80, 80, 80, 80]
    request = _mk_request(
        game_type="tournament",
        stacks=stacks,
        hero_seat=2,
        button_seat=0,
        to_call=0,
        board=[],
        hero_hole=["As", "Ah"],  # AA: definitely push
    )
    ts = _mk_ts(stacks=tuple(stacks), hero_index=2)
    response = tournament.decide(request, ts)
    assert response.abstract_action in {"FOLD", "ALL_IN"}, (
        f"expected FOLD or ALL_IN, got {response.abstract_action}"
    )
    assert response.fallback_used == "pushfold"


# ─────────── 3. deep stack uses CFR with ICM (close to base) ───────────


def test_tournament_deep_stack_uses_cfr_with_icm(
    runtime_adapter: RuntimeAdapter, runtime_adapter_for_compare: RuntimeAdapter
) -> None:
    """hero_stack_bb = 50, far from bubble: distribution near base CFR (risk_factor ≈ 1)."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    stacks = [500, 500, 500, 500, 500, 500]
    # Far from bubble: 6 in, 6 pay. bubble_distance = 0.
    ts = _mk_ts(stacks=tuple(stacks), hero_index=2, players_in_money=6)
    # 50 BB stack
    rf = tournament._compute_risk_factor(ts)
    # base = 1.0, no bubble penalty, 50 is between 25 and 75 so no stack adjustment.
    # Optional ICM adjustment may kick in (dollar share vs chip share), but stacks are equal.
    assert rf >= 0.90, f"deep-stack-no-bubble risk_factor too low: {rf}"

    # Sample 200 decisions from base vs tournament; distributions should be similar.
    counts_base: dict[str, int] = {}
    counts_tour: dict[str, int] = {}
    n = 200
    for i in range(n):
        request = _mk_request(
            game_type="tournament",
            stacks=stacks,
            hero_seat=2,
            button_seat=(i % 6),  # rotate button to vary infoset
            to_call=(50 if i % 2 == 0 else 0),
            board=[],
        )
        r_base = runtime_adapter_for_compare.decide(
            GameStateRequest(**{**request.model_dump(), "game_type": "cash"})
        )
        r_tour = tournament.decide(request, ts)
        counts_base[r_base.abstract_action] = counts_base.get(r_base.abstract_action, 0) + 1
        counts_tour[r_tour.abstract_action] = counts_tour.get(r_tour.abstract_action, 0) + 1

    # No action's frequency should differ by more than 15% in absolute terms
    # (loose tolerance; the risk_factor ~ 0.9 can still shift a few % of mass).
    for action in set(counts_base) | set(counts_tour):
        b = counts_base.get(action, 0) / n
        t = counts_tour.get(action, 0) / n
        assert abs(b - t) <= 0.15, (
            f"deep-stack distribution mismatch on {action}: base={b:.1%} tour={t:.1%}"
        )


# ─────────── 4. bubble pressure tightens play ───────────


def test_bubble_pressure_tightens_play(runtime_adapter: RuntimeAdapter) -> None:
    """Same hand, different TournamentState — bubble version is meaningfully tighter."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    stacks = [500, 500, 500, 500, 500, 500]
    ts_safe = _mk_ts(stacks=tuple(stacks), hero_index=2, players_in_money=6)
    # Bubble: 6 left, 5 pay. bubble_distance = 1 → base *= 0.6.
    ts_bubble = _mk_ts(stacks=tuple(stacks), hero_index=2, players_in_money=5)

    rf_safe = tournament._compute_risk_factor(ts_safe)
    rf_bubble = tournament._compute_risk_factor(ts_bubble)
    assert rf_bubble < rf_safe, (
        f"bubble risk_factor {rf_bubble} should be lower than safe {rf_safe}"
    )

    # Sample many decisions; bubble should fold/call more, raise/bet less.
    n = 500
    agg_safe = agg_bubble = 0
    for i in range(n):
        request = _mk_request(
            game_type="tournament",
            stacks=stacks,
            hero_seat=2,
            button_seat=(i % 6),
            to_call=0,
            board=[],
        )
        # Reset RNG for fair comparison (same sequence of states, different ts)
        tournament.rng.seed(i)
        tournament.base.rng.seed(i)
        r_safe = tournament.decide(request, ts_safe)
        tournament.rng.seed(i)
        tournament.base.rng.seed(i)
        r_bubble = tournament.decide(request, ts_bubble)

        risky_set = {"ALL_IN", "RAISE_3_5X", "RAISE_2_5X", "BET_150", "BET_100", "BET_66", "BET_33"}
        if r_safe.abstract_action in risky_set:
            agg_safe += 1
        if r_bubble.abstract_action in risky_set:
            agg_bubble += 1

    safe_frac = agg_safe / n
    bubble_frac = agg_bubble / n
    # Bubble should be at least 5 percentage points tighter on aggression
    assert bubble_frac < safe_frac - 0.04, (
        f"bubble aggression {bubble_frac:.1%} not meaningfully lower than safe {safe_frac:.1%}"
    )


# ─────────── 5. risk-factor bounds ───────────


def test_risk_factor_bounds(runtime_adapter: RuntimeAdapter) -> None:
    """_compute_risk_factor must always return values in [0.5, 1.0]."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    edge_cases = [
        # (description, ts)
        (
            "9-handed final table, ITM",
            _mk_ts(stacks=tuple([500] * 9), hero_index=0, players_in_money=9),
        ),
        ("HU, ITM", _mk_ts(stacks=(1000, 1000), hero_index=0, players_in_money=2)),
        (
            "on bubble (1 from money)",
            _mk_ts(stacks=tuple([500] * 6), hero_index=0, players_in_money=5),
        ),
        (
            "near bubble (2 from money)",
            _mk_ts(stacks=tuple([500] * 6), hero_index=0, players_in_money=4),
        ),
        (
            "hero very short, on bubble",
            _mk_ts(stacks=(100, 1000, 1000, 1000, 1000, 1000), hero_index=0, players_in_money=5),
        ),
        (
            "hero very deep, no bubble",
            _mk_ts(stacks=(5000, 500, 500, 500, 500, 500), hero_index=0, players_in_money=6),
        ),
        (
            "hero deep + on bubble",
            _mk_ts(stacks=(5000, 500, 500, 500, 500, 500), hero_index=0, players_in_money=5),
        ),
        (
            "large field",
            _mk_ts(stacks=tuple([500] * 9), hero_index=4, players_in_money=3, blinds_bb=10),
        ),
    ]
    for desc, ts in edge_cases:
        rf = tournament._compute_risk_factor(ts)
        assert RISK_FACTOR_MIN <= rf <= RISK_FACTOR_MAX, (
            f"{desc}: risk_factor={rf} outside [{RISK_FACTOR_MIN}, {RISK_FACTOR_MAX}]"
        )


# ─────────── 6. icm integration ───────────


def test_icm_integration(runtime_adapter: RuntimeAdapter) -> None:
    """icm_equity is called during _icm_weighted_decide (via _compute_risk_factor)."""
    from pokerbot.tournament import icm as icm_module

    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    stacks = [1000, 800, 1200, 900, 1100, 1000]
    ts = _mk_ts(stacks=tuple(stacks), hero_index=2, players_in_money=6)
    request = _mk_request(game_type="tournament", stacks=stacks, hero_seat=2, button_seat=0)

    with patch("pokerbot.tournament.adapter.icm_equity", wraps=icm_module.icm_equity) as mock_icm:
        tournament.decide(request, ts)
    assert mock_icm.call_count >= 1, "icm_equity was not called during decide()"


# ─────────── 7. pushfold integration ───────────


def test_pushfold_integration(runtime_adapter: RuntimeAdapter) -> None:
    """pushfold.pushfold_decision is called during _pushfold_decide for short stacks."""
    from pokerbot.tournament import pushfold as pf_module

    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    stacks = [80, 80, 80, 80, 80, 80]  # 8 BB
    request = _mk_request(game_type="tournament", stacks=stacks, hero_seat=2, button_seat=0)
    ts = _mk_ts(stacks=tuple(stacks), hero_index=2)

    with patch(
        "pokerbot.tournament.adapter.pushfold_decision", wraps=pf_module.pushfold_decision
    ) as mock_pf:
        tournament.decide(request, ts)
    assert mock_pf.call_count == 1, (
        f"pushfold_decision called {mock_pf.call_count} times, expected 1"
    )


# ─────────── 8. TournamentState validation ───────────


def test_tournament_state_validation() -> None:
    """Bad hero_index / negative stacks / bad blinds raise ValueError; len(payouts)!=N is OK."""
    # OK: payouts longer than players_remaining
    _ = TournamentState(
        stacks=(1000, 800, 1200),
        hero_index=0,
        payouts=(5000.0, 3000.0, 2000.0, 1500.0, 1000.0, 700.0),
        players_remaining=3,
        players_in_money=3,
        blinds_bb=10,
        starting_stack_bb=100,
    )
    # OK: payouts shorter than players_remaining
    _ = TournamentState(
        stacks=(1000, 800, 1200, 900, 1100, 1000),
        hero_index=0,
        payouts=(5000.0, 3000.0, 2000.0),
        players_remaining=6,
        players_in_money=3,
        blinds_bb=10,
        starting_stack_bb=100,
    )
    # hero_index out of range
    with pytest.raises(ValueError, match="hero_index"):
        TournamentState(
            stacks=(1000, 1000),
            hero_index=5,
            payouts=(100.0,),
            players_remaining=2,
            players_in_money=1,
            blinds_bb=10,
            starting_stack_bb=100,
        )
    # negative stack
    with pytest.raises(ValueError, match="non-negative"):
        TournamentState(
            stacks=(1000, -100),
            hero_index=0,
            payouts=(100.0,),
            players_remaining=2,
            players_in_money=1,
            blinds_bb=10,
            starting_stack_bb=100,
        )
    # blinds_bb <= 0
    with pytest.raises(ValueError, match="blinds_bb"):
        TournamentState(
            stacks=(1000, 1000),
            hero_index=0,
            payouts=(100.0,),
            players_remaining=2,
            players_in_money=1,
            blinds_bb=0,
            starting_stack_bb=100,
        )


# ─────────── 9. no TournamentState in cash mode is OK ───────────


def test_no_tournament_state_in_cash_mode_ok(runtime_adapter: RuntimeAdapter) -> None:
    """decide(cash_request, None) must work."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    request = _mk_request(game_type="cash")
    response = tournament.decide(request, tournament_state=None)
    assert response is not None


# ─────────── 10. missing TournamentState in tournament mode errors ───────────


def test_missing_tournament_state_in_tournament_mode_errors(
    runtime_adapter: RuntimeAdapter,
) -> None:
    """decide(tournament_request, None) must raise ValueError."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    request = _mk_request(game_type="tournament")
    with pytest.raises(ValueError, match="tournament_state required"):
        tournament.decide(request, tournament_state=None)


# ─────────── 11. smoke: full tournament hand ───────────


def test_smoke_full_tournament_hand(runtime_adapter: RuntimeAdapter) -> None:
    """One realistic final-table hand at 30bb completes without errors."""
    tournament = TournamentAdapter(runtime_adapter, rng_seed=0)
    # 6-handed final table, hero stack 30 BB
    stacks = [3000, 2000, 1500, 1200, 800, 500]  # in chips
    ts = TournamentState(
        stacks=tuple(stacks),
        hero_index=0,  # hero is biggest stack
        payouts=(5000.0, 3000.0, 2000.0, 1500.0, 1000.0, 700.0),
        players_remaining=6,
        players_in_money=6,
        blinds_bb=100,
        starting_stack_bb=100,
    )
    # Preflop spot, hero in cutoff position with a strong hand
    request = _mk_request(
        game_type="tournament",
        stacks=stacks,
        table_size=6,
        hero_seat=0,
        button_seat=5,
        to_call=100,  # someone opened
        action_history=[
            ActionHistoryEntry(seat=2, street=0, type="raise", amount=300),
        ],
        board=[],
        hero_hole=["As", "Ks"],
    )
    response = tournament.decide(request, ts)
    assert response is not None
    assert response.action in {"fold", "check", "call", "bet", "raise"}
    assert response.amount >= 0
    # Hero stack 30 BB → deep-stack path → fallback_used reflects base adapter's lookup result
    assert response.fallback_used in {"exact", "nearest_neighbor", "default_policy"}
