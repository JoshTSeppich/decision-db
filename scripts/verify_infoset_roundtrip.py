"""Verify the training↔runtime infoset consistency contract on real game states.

The contract (Spec.html §C + §F): for every observable state `s` reachable in
training,

    SimpleNLHEGame.infoset_key(s)  ==  RuntimeAdapter.build_infoset(req(s)).to_bytes()

where `req(s)` is the GameStateRequest a JSON client would send for the same
state. If this drifts, DB rows written during training are unreachable from
runtime lookups, and `nearest_neighbor` returns rows for unrelated positions.
The existing `test_infoset_key_matches_runtime_adapter` only checks 4 random
transitions from one fresh deal — not enough to catch street-transition or
deep-history drift.

This script replays SimpleNLHEGame rollouts (the same game object the trainer
uses), samples 20 (state, key) pairs balanced across streets, and round-trips
each one through the runtime adapter. On any mismatch it prints both InfoSet
decompositions field-by-field so the broken field is obvious.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pokerbot.abstraction import AbstractionTables, ActionType
from pokerbot.runtime import (
    ActionHistoryEntry,
    BlindsSchema,
    GameStateRequest,
    RuntimeAdapter,
)
from pokerbot.strategy_db import open_db
from pokerbot.training import SimpleNLHEGame

if TYPE_CHECKING:
    from pokerbot.abstraction import AbstractAction, InfoSet
    from pokerbot.training.nlhe_game import NLHEState


# ───────── GameStateRequest construction (mirrors tests/test_nlhe_game.py) ─────────


def _pk_card_str(card: object) -> str:
    return f"{card.rank.value}{card.suit.value}"  # type: ignore[attr-defined]


def _abstract_to_json_entry(
    seat: int, street: int, abs_action: AbstractAction
) -> ActionHistoryEntry:
    at = abs_action.type
    amount = abs_action.amount_chips
    if at == ActionType.FOLD:
        type_str: str = "fold"
        amount = 0
    elif at == ActionType.CHECK_CALL:
        type_str = "check" if amount == 0 else "call"
    elif at == ActionType.ALL_IN:
        type_str = "all-in"
    elif street == 0:
        type_str = "raise"
    else:
        type_str = "bet" if amount > 0 else "check"
    return ActionHistoryEntry(seat=seat, street=street, type=type_str, amount=amount)  # type: ignore[arg-type]


def _state_to_request(game: SimpleNLHEGame, state: NLHEState) -> GameStateRequest:
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

    hero_hole = [_pk_card_str(c) for c in pk.hole_cards[actor]]
    board: list[str] = []
    for stack_ in pk.board_cards:
        board.extend(_pk_card_str(c) for c in stack_)

    history_entries = [
        _abstract_to_json_entry(entry.seat, entry.street, entry.action) for entry in state.history
    ]

    return GameStateRequest(
        schema_version=1,
        game_type="cash",
        table_size=table_size,  # type: ignore[arg-type]
        blinds=BlindsSchema(sb=game.blinds[0], bb=game.blinds[1]),
        ante=0,
        hero_seat=actor,
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


_STREET_NAMES = ("preflop", "flop", "turn", "river")


@dataclass(slots=True)
class Sample:
    street: int
    history_len: int  # number of recorded abstract actions before this decision
    state: NLHEState
    key_game: bytes
    info_game: InfoSet


def _collect_samples(
    game: SimpleNLHEGame,
    rng: random.Random,
    *,
    target_per_street: int = 20,
    max_hands: int = 5000,
) -> dict[int, list[Sample]]:
    """Play random hands; collect samples until every street has `target_per_street`."""
    by_street: dict[int, list[Sample]] = defaultdict(list)

    for _ in range(max_hands):
        state = game.new_initial_state(rng)
        while not game.is_terminal(state):
            info = game._build_infoset(state)
            key = info.to_bytes()
            by_street[info.street].append(
                Sample(
                    street=info.street,
                    history_len=len(state.history),
                    state=state,
                    key_game=key,
                    info_game=info,
                )
            )
            legal = game.legal_actions(state)
            action = rng.choice(legal)
            state = game.apply_action(state, action, rng)
        if all(len(by_street[s]) >= target_per_street for s in range(4)):
            break
    return by_street


def _format_infoset(info: InfoSet) -> str:
    return (
        f"table_size={info.table_size} street={info.street} "
        f"position={info.position} stack_bucket={info.stack_bucket} "
        f"card_bucket={info.card_bucket} "
        f"history({len(info.history)}B)={info.history.hex()}"
    )


def _decompose_diff(info_g: InfoSet, info_a: InfoSet) -> list[str]:
    diffs: list[str] = []
    for field in ("table_size", "street", "position", "stack_bucket", "card_bucket"):
        g = getattr(info_g, field)
        a = getattr(info_a, field)
        if g != a:
            diffs.append(f"  {field}: game={g}  adapter={a}")
    if info_g.history != info_a.history:
        diffs.append(
            f"  history: game({len(info_g.history)}B)={info_g.history.hex()}\n"
            f"           adapter({len(info_a.history)}B)={info_a.history.hex()}"
        )
    return diffs


def _verify(
    samples: list[Sample],
    game: SimpleNLHEGame,
    adapter: RuntimeAdapter,
) -> tuple[int, int]:
    """Return (n_ok, n_fail). Prints per-sample summary and any field diffs."""
    ok = 0
    fail = 0
    for i, sample in enumerate(samples, start=1):
        request = _state_to_request(game, sample.state)
        info_adapter = adapter.build_infoset(request)
        key_adapter = info_adapter.to_bytes()
        match = key_adapter == sample.key_game
        tag = "ok " if match else "FAIL"
        print(
            f"  [{i:>2}] {tag}  street={_STREET_NAMES[sample.street]:<7s}  "
            f"hist_len={sample.history_len:<3d}  "
            f"hash16(game)={sample.info_game.hash16().hex()[:12]}…"
        )
        if not match:
            fail += 1
            print("       game   :", _format_infoset(sample.info_game))
            print("       adapter:", _format_infoset(info_adapter))
            for line in _decompose_diff(sample.info_game, info_adapter):
                print(line)
        else:
            ok += 1
    return ok, fail


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--abstraction-dir", default="abstraction")
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--sb", type=int, default=5)
    p.add_argument("--bb", type=int, default=10)
    p.add_argument("--table-size", type=int, default=6, choices=[6, 8, 9])
    p.add_argument(
        "--per-street",
        type=int,
        default=5,
        help="samples per street (4 streets x per-street = total samples; default 5x4=20)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"Loading AbstractionTables from {args.abstraction_dir!r}…")
    abstraction = AbstractionTables(path=args.abstraction_dir)
    if set(abstraction.loaded_streets) != {"flop", "turn", "river"}:
        print(
            f"WARNING: only {abstraction.loaded_streets} loaded; "
            "miss-path uses placeholder hashes.",
            file=sys.stderr,
        )

    game = SimpleNLHEGame(
        abstraction,
        blinds=(args.sb, args.bb),
        starting_stack=args.starting_stack,
        table_size=args.table_size,
    )
    db = open_db("sqlite:///:memory:")
    db.set_current_version(1)
    adapter = RuntimeAdapter(db=db, abstraction=abstraction, rng_seed=args.seed)

    rng = random.Random(args.seed)
    print(
        f"\nPlaying random hands until each street has ≥{args.per_street} samples…",
    )
    by_street = _collect_samples(game, rng, target_per_street=args.per_street)
    for s in range(4):
        print(f"  {_STREET_NAMES[s]:<7s}: {len(by_street[s])} samples collected")

    # Take `per_street` from each street, in order.
    sample_rng = random.Random(args.seed ^ 0x5A5A)
    selected: list[Sample] = []
    for s in range(4):
        if not by_street[s]:
            print(f"  ⚠ no samples on {_STREET_NAMES[s]} — skipping")
            continue
        chosen = sample_rng.sample(by_street[s], k=min(args.per_street, len(by_street[s])))
        selected.extend(chosen)

    print(f"\nVerifying round-trip on {len(selected)} samples…")
    ok, fail = _verify(selected, game, adapter)

    print("\n── summary ──")
    print(f"  total: {ok + fail}   ok: {ok}   fail: {fail}")
    if fail:
        print("  ⚠ CONSISTENCY-CONTRACT DRIFT — see field diffs above.")
        return 1
    print("  ✓ contract holds on all sampled states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
