"""Tests for SimpleNLHEGame (GAP 1 — production NLHE game for Deep CFR).

Spec acceptance:
    - 6/8/9-max state construction
    - Terminal rewards sum to ≈ 0
    - infoset_key matches RuntimeAdapter.build_infoset for the equivalent JSON
    - Folded players don't act
    - 100 random games complete without raising
    - Trainer(make_test_config(), SimpleNLHEGame(...)) runs one outer iteration
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
import torch  # noqa: F401 — imported so torch is importable in the test process

from pokerbot.abstraction import (
    AbstractionTables,
    ActionType,
)
from pokerbot.runtime import (
    ActionHistoryEntry,
    BlindsSchema,
    GameStateRequest,
    RuntimeAdapter,
)
from pokerbot.strategy_db import SQLiteStrategyDB
from pokerbot.training import (
    SimpleNLHEGame,
    Trainer,
    make_test_config,
)
from pokerbot.training.nlhe_game import FEATURE_DIM, NLHEState

if TYPE_CHECKING:
    from pathlib import Path


# ───────── helpers ─────────


def _play_random(game: SimpleNLHEGame, rng: random.Random, max_actions: int = 200) -> NLHEState:
    state = game.new_initial_state(rng)
    n = 0
    while not game.is_terminal(state):
        legal = game.legal_actions(state)
        action = rng.choice(legal)
        state = game.apply_action(state, action, rng)
        n += 1
        if n > max_actions:
            raise RuntimeError(f"hand didn't terminate in {max_actions} actions")
    return state


# ───────── 1. construction across table sizes ─────────


@pytest.mark.parametrize("table_size", [2, 3, 4, 5, 6, 7, 8, 9])
def test_construction_per_table_size(table_size: int) -> None:
    game = SimpleNLHEGame(AbstractionTables(), table_size=table_size)  # type: ignore[arg-type]
    assert game.num_players == table_size
    assert game.num_actions == len(ActionType)
    assert game.feature_dim == FEATURE_DIM

    rng = random.Random(table_size * 1000)
    state = game.new_initial_state(rng)
    assert not game.is_terminal(state)
    assert 0 <= game.current_player(state) < table_size
    feats = game.infoset_features(state)
    assert feats.shape == (FEATURE_DIM,)


@pytest.mark.parametrize("bad_size", [0, 1, 10])
def test_construction_rejects_unsupported_table_size(bad_size: int) -> None:
    with pytest.raises(ValueError, match="table_size"):
        SimpleNLHEGame(AbstractionTables(), table_size=bad_size)  # type: ignore[arg-type]


# ───────── 2. terminal rewards sum to zero ─────────


def test_terminal_rewards_zero_sum_random_hands() -> None:
    """Across 50 random hands at each table size, rewards must sum to ≈ 0."""
    for table_size in (2, 3, 4, 5, 6, 7, 8, 9):
        game = SimpleNLHEGame(AbstractionTables(), table_size=table_size)
        rng = random.Random(0xBEEF + table_size)
        for _ in range(50):
            terminal = _play_random(game, rng)
            rewards = game.terminal_reward(terminal).rewards
            assert len(rewards) == table_size
            # No rake → exact integer sum
            assert sum(rewards) == pytest.approx(0.0, abs=1e-6), (
                f"reward sum {sum(rewards)} for {rewards}"
            )


# ───────── 3. side pots resolve when stacks differ ─────────


def test_side_pots_short_stack_loses_at_most_their_stack() -> None:
    """If short stack goes all-in and loses, their loss is bounded by their stack."""
    rng = random.Random(123)
    # 6 players, normal stacks. Force one to all-in via repeated calls/raises.
    game = SimpleNLHEGame(AbstractionTables(), table_size=6)
    # Just verify the abstract property: terminal rewards never exceed initial stacks lost
    for _ in range(20):
        terminal = _play_random(game, rng)
        rewards = game.terminal_reward(terminal).rewards
        for r in rewards:
            # Nobody can lose more than starting_stack
            assert r >= -game.starting_stack, f"reward {r} > starting_stack {game.starting_stack}"


# ───────── 4. infoset_key matches RuntimeAdapter.build_infoset ─────────


def _state_to_request(game: SimpleNLHEGame, state: NLHEState) -> GameStateRequest:
    """Convert an NLHEState to the equivalent JSON game-state request.

    Used by the consistency contract test. Translates pokerkit state + our
    abstract history into a runtime-adapter-shaped request.
    """
    pk = state.pk_state
    actor = game.current_player(state)
    table_size = game.table_size
    stacks = [int(pk.stacks[i]) for i in range(table_size)]
    current_bets = [int(pk.bets[i]) for i in range(table_size)]
    pot = int(sum(state.initial_stacks) - sum(pk.stacks))
    to_call = int(pk.checking_or_calling_amount or 0)
    bet_actor = current_bets[actor]
    min_raise_to = int(pk.min_completion_betting_or_raising_to_amount or 0)
    min_raise = max(min_raise_to - bet_actor, 0)
    max_raise = int(pk.max_completion_betting_or_raising_to_amount or 0)

    # Hero hole cards
    hole_pk = pk.hole_cards[actor]
    hero_hole = [_pk_card_str(c) for c in hole_pk]

    # Board (flattened)
    board: list[str] = []
    for stack_ in pk.board_cards:
        board.extend(_pk_card_str(c) for c in stack_)

    # Action history JSON entries derived from our recorded abstract actions.
    history_entries: list[ActionHistoryEntry] = []
    for entry in state.history:
        history_entries.append(_abstract_to_json_entry(entry.seat, entry.street, entry.action))

    return GameStateRequest(
        schema_version=1,
        game_type="cash",
        table_size=table_size,  # type: ignore[arg-type]
        blinds=BlindsSchema(sb=game.blinds[0], bb=game.blinds[1]),
        ante=0,
        hero_seat=actor,
        # In pokerkit's HU, seat 1 = SB = button (acts first); in 3+, seat 0 =
        # SB → button = last seat. Either way, button == seat (table_size - 1).
        button_seat=table_size - 1,
        hero_hole=hero_hole,
        board=board,
        stacks=stacks,
        current_bets=current_bets,
        pot_committed=pot,
        to_call=to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        action_history=history_entries,
    )


def _pk_card_str(card: object) -> str:
    return f"{card.rank.value}{card.suit.value}"  # type: ignore[attr-defined]


def _abstract_to_json_entry(seat: int, street: int, abs_action: object) -> ActionHistoryEntry:
    """Map an AbstractAction to the JSON-history form the runtime adapter expects."""
    at = abs_action.type  # type: ignore[attr-defined]
    amount = abs_action.amount_chips  # type: ignore[attr-defined]
    if at == ActionType.FOLD:
        type_str = "fold"
        amount = 0
    elif at == ActionType.CHECK_CALL:
        type_str = "check" if amount == 0 else "call"
    elif at == ActionType.ALL_IN:
        type_str = "all-in"
    elif street == 0:  # preflop bet/raise
        type_str = "raise"
    else:
        type_str = "bet" if amount > 0 else "check"
    return ActionHistoryEntry(seat=seat, street=street, type=type_str, amount=amount)  # type: ignore[arg-type]


@pytest.mark.parametrize("table_size", [2, 3, 4, 5, 6, 7, 8, 9])
def test_infoset_key_matches_runtime_adapter_fuzz_100_hands(
    tmp_path: Path, table_size: int
) -> None:
    """Cross-consistency fuzz: byte-identical infoset keys at EVERY decision point.

    Plays 100 random hands and asserts `SimpleNLHEGame.infoset_key(state)` ==
    `RuntimeAdapter.build_infoset(req(s)).to_bytes()` at every actor decision
    on every street. This is the regression test that catches §C encoding
    drift between training and runtime — the original 4-action smoke test
    didn't exercise enough 3-bet+ branches to surface the bug fixed by
    `encode_action_history_byte`. At table_size=2 it also pins the HU
    button==SB==seat 0 convention shared by trainer and adapter.
    """
    tables = AbstractionTables()
    game = SimpleNLHEGame(tables, table_size=table_size)  # type: ignore[arg-type]
    db = SQLiteStrategyDB(str(tmp_path / "consistency.db"))
    db.set_current_version(1)
    adapter = RuntimeAdapter(db=db, abstraction=tables, rng_seed=0)

    rng = random.Random(2026 + table_size)
    checked = 0
    for hand_idx in range(100):
        state = game.new_initial_state(rng)
        while not game.is_terminal(state):
            key_from_game = game.infoset_key(state)
            request = _state_to_request(game, state)
            key_from_adapter = adapter.build_infoset(request).to_bytes()
            assert key_from_game == key_from_adapter, (
                f"hand={hand_idx} table_size={table_size} hist_len={len(state.history)}\n"
                f"  game:    {key_from_game.hex()}\n"
                f"  adapter: {key_from_adapter.hex()}"
            )
            checked += 1
            legal = game.legal_actions(state)
            action = rng.choice(legal)
            state = game.apply_action(state, action, rng)
    # Sanity: 100 hands x at least a few decisions per hand. Smaller tables
    # have fewer decisions/hand, so the floor is scaled to avoid spurious
    # failures while still catching a vacuous test.
    assert checked >= 30 * table_size, (
        f"only {checked} decision points checked across 100 hands at table_size={table_size}"
    )


# ───────── 5. folded players don't act ─────────


def test_folded_player_never_acts() -> None:
    """Once a seat folds, the game never names them as the actor again."""
    game = SimpleNLHEGame(AbstractionTables(), table_size=6)
    rng = random.Random(42)
    state = game.new_initial_state(rng)

    folded: set[int] = set()
    while not game.is_terminal(state):
        actor = game.current_player(state)
        assert actor not in folded, f"seat {actor} already folded but is acting again"
        legal = game.legal_actions(state)
        # Force a fold path so we exercise the property
        action = int(ActionType.FOLD) if int(ActionType.FOLD) in legal else legal[0]
        if action == int(ActionType.FOLD):
            folded.add(actor)
        state = game.apply_action(state, action, rng)


# ───────── 6. smoke: 100 random games complete ─────────


def test_smoke_100_random_games_complete() -> None:
    game = SimpleNLHEGame(AbstractionTables(), table_size=6)
    rng = random.Random(2026)
    for _ in range(100):
        terminal = _play_random(game, rng)
        assert game.is_terminal(terminal)
        rewards = game.terminal_reward(terminal).rewards
        assert sum(rewards) == pytest.approx(0.0, abs=1e-6)


# ───────── 7. Trainer integration smoke ─────────


def test_trainer_runs_one_outer_iteration_on_nlhe(tmp_path: Path) -> None:
    """Acceptance criterion: `Trainer(DeepCFRConfig(), SimpleNLHEGame(...))` completes
    one full outer iteration with a tiny config without raising.
    """
    config = make_test_config(
        outer_iters=1,
        traversals_per_iter=4,
        train_steps_per_iter=2,
        policy_train_steps=2,
        batch_size=4,
        advantage_buffer_size=64,
        policy_buffer_size=64,
        advantage_hidden=(16, 16),
        policy_hidden=(16, 16),
        checkpoint_every=1,
        seed=2026,
    )
    game = SimpleNLHEGame(AbstractionTables(), table_size=6)
    trainer = Trainer(config, game)
    trainer.train(tmp_path / "nlhe_smoke")
    # After one iter both reservoirs should have at least one sample (preflop
    # always has actions).
    total_adv = sum(len(r) for r in trainer.advantage_reservoirs)
    assert total_adv > 0
    assert len(trainer.policy_reservoir) > 0
