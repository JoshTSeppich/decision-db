"""Head-to-head eval: trained adapter (strategy-pilot.db) vs default-policy adapter.

Plays N hands of NLHE 6-max via `SimpleNLHEGame` (pokerkit rules engine), routing
each decision through one of two `RuntimeAdapter`s:

    - trained:  reads `strategy-pilot.db`
    - default:  reads an empty in-memory DB → every decision falls through to
                `default_policy_action`

Half the seats are trained, half default. The seat→adapter mapping flips every
hand to neutralize positional bias:

    hand i % 2 == 0  →  trained at {0, 2, 4}, default at {1, 3, 5}
    hand i % 2 == 1  →  trained at {1, 3, 5}, default at {0, 2, 4}

Pokerkit's button is fixed at the last seat (= table_size - 1) — see
`SimpleNLHEGame.new_initial_state`. Rotating the seat→adapter mapping (instead
of the button) gives each side every table position with equal frequency
across two hands, which is what we actually want.

Reports trained-side mbb/hand with a 95% CI, plus a fallback breakdown and a
20-row policy-collapse sanity check.
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

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
from pokerbot.strategy_db import StrategyDB, open_db, unpack_probs
from pokerbot.training import SimpleNLHEGame

if TYPE_CHECKING:
    from pokerbot.abstraction import AbstractAction
    from pokerbot.runtime.schema import FallbackUsed
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
        button_seat=table_size - 1,  # pokerkit convention: last seat = BTN
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


# ───────── one hand ─────────


_STREET_NAMES: Final = ("preflop", "flop", "turn", "river")
_BOARD_LEN_TO_STREET: Final = {0: 0, 3: 1, 4: 2, 5: 3}


def _empty_fallback_table() -> dict[int, dict[FallbackUsed, int]]:
    return {s: {"exact": 0, "nearest_neighbor": 0, "default_policy": 0} for s in range(4)}


@dataclass(slots=True)
class HandResult:
    trained_delta: int
    default_delta: int
    fallback_by_street: dict[int, dict[FallbackUsed, int]]
    decisions: int
    illegal_remaps: int  # times the adapter's sampled action wasn't pokerkit-legal


def _clamp_to_legal(chosen: ActionType, legal_ints: tuple[int, ...]) -> ActionType:
    """If `chosen` isn't pokerkit-legal, remap to the nearest semantic action.

    The runtime adapter trusts the DB row's `action_mask`, but pokerkit imposes
    extra gates (`can_fold` etc.) that don't always match — particularly via
    `nearest_neighbor`, which ignores history. Training-time `_legal_abstract_at`
    intersects with pokerkit's gates; in production we'd push that intersection
    into the adapter or championship-side translator. For the eval, we clamp
    here and count the remaps (high rate ⇒ consistency drift).
    """
    if int(chosen) in legal_ints:
        return chosen
    legal_set = {ActionType(i) for i in legal_ints}
    if chosen == ActionType.FOLD and ActionType.CHECK_CALL in legal_set:
        # Can't fold (no chips to call) → free check is the passive analog.
        return ActionType.CHECK_CALL
    # Any raise/bet that pokerkit rejects (cap, short stack) → call/check.
    if ActionType.CHECK_CALL in legal_set:
        return ActionType.CHECK_CALL
    if ActionType.ALL_IN in legal_set:
        return ActionType.ALL_IN
    # Last resort: anything legal.
    return next(iter(legal_set))


def _trained_seats_for_hand(hand_idx: int, table_size: int) -> set[int]:
    """Even hands → even seats trained; odd hands → odd seats trained."""
    parity = hand_idx % 2
    return {s for s in range(table_size) if s % 2 == parity}


def _play_one_hand(
    hand_idx: int,
    game: SimpleNLHEGame,
    trained: RuntimeAdapter,
    default: RuntimeAdapter,
    rng: random.Random,
    max_decisions: int = 400,
) -> HandResult:
    trained_seats = _trained_seats_for_hand(hand_idx, game.table_size)
    state = game.new_initial_state(rng)
    fallback_by_street = _empty_fallback_table()
    decisions = 0
    illegal_remaps = 0

    while not game.is_terminal(state):
        actor = game.current_player(state)
        request = _state_to_request(game, state)
        adapter = trained if actor in trained_seats else default
        response = adapter.decide(request)
        if adapter is trained:
            street_idx = _BOARD_LEN_TO_STREET[len(request.board)]
            fallback_by_street[street_idx][response.fallback_used] += 1
        decisions += 1
        if decisions > max_decisions:
            raise RuntimeError(f"hand {hand_idx} stuck after {max_decisions} decisions")

        chosen = ActionType[response.abstract_action]
        legal_ints = game.legal_actions(state)
        applied = _clamp_to_legal(chosen, legal_ints)
        if applied != chosen:
            illegal_remaps += 1
        state = game.apply_action(state, int(applied), rng)

    rewards = game.terminal_reward(state).rewards
    # Zero-sum guarantee from pokerkit: no rake here.
    total = sum(rewards)
    if abs(total) > 1.0:  # 1 chip tolerance
        raise AssertionError(
            f"hand {hand_idx} not zero-sum: sum(rewards)={total}, rewards={rewards}"
        )

    trained_delta = sum(int(rewards[s]) for s in trained_seats)
    default_delta = sum(int(rewards[s]) for s in range(game.table_size) if s not in trained_seats)
    return HandResult(
        trained_delta=trained_delta,
        default_delta=default_delta,
        fallback_by_street=fallback_by_street,
        decisions=decisions,
        illegal_remaps=illegal_remaps,
    )


# ───────── stats ─────────


def _mbb_stats(per_hand_deltas: list[int], bb_size: int) -> tuple[float, float, float]:
    """Return (mean_mbb_per_hand, ci_low_mbb, ci_high_mbb) using the per-hand
    chip-delta sample (one observation per hand)."""
    n = len(per_hand_deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(per_hand_deltas, dtype=np.float64)
    mean_chips = float(arr.mean())
    # std-of-mean via per-hand sample. ddof=1 for unbiased variance.
    stderr_chips = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0

    mean_mbb = mean_chips / bb_size * 1000.0
    half = 1.96 * stderr_chips / bb_size * 1000.0
    return mean_mbb, mean_mbb - half, mean_mbb + half


# ───────── sanity sampling ─────────


def _sample_db_rows(db_path: str, n: int, rng: random.Random) -> list[tuple[int, np.ndarray]]:
    """Pull `n` random (action_mask, probs) pairs straight from the DB.

    Uses reservoir-friendly random offsets rather than ORDER BY RANDOM() to
    avoid a full-table sort on 422k rows.
    """
    conn = sqlite3.connect(db_path)
    try:
        (total,) = conn.execute("SELECT COUNT(*) FROM strategy").fetchone()
        offsets = sorted(rng.sample(range(total), min(n, total)))
        rows: list[tuple[int, np.ndarray]] = []
        for off in offsets:
            cur = conn.execute(
                "SELECT action_mask, action_probs FROM strategy LIMIT 1 OFFSET ?",
                (off,),
            )
            row = cur.fetchone()
            if row is None:
                continue
            mask = int(row[0])
            probs = unpack_probs(mask, bytes(row[1]))
            rows.append((mask, probs))
        return rows
    finally:
        conn.close()


def _action_name(bit: int) -> str:
    return ActionType(bit).name


def _format_row(mask: int, probs: np.ndarray) -> str:
    parts: list[str] = []
    idx = 0
    for bit in range(mask.bit_length() + 1):
        if mask & (1 << bit):
            parts.append(f"{_action_name(bit)}={probs[idx]:.3f}")
            idx += 1
    return f"mask={mask:#05x}  " + ", ".join(parts)


def _print_sanity(
    db_path: str, n: int, rng: random.Random, one_hot_threshold: float = 0.95
) -> None:
    print(f"\n── sanity: {n} random strategy-pilot rows ──")
    rows = _sample_db_rows(db_path, n, rng)
    one_hot = 0
    for mask, probs in rows:
        print(f"  {_format_row(mask, probs)}")
        if float(probs.max()) >= one_hot_threshold:
            one_hot += 1
    frac = one_hot / len(rows) if rows else 0.0
    print(f"  one-hot (max p ≥ {one_hot_threshold}): {one_hot}/{len(rows)}  ({frac:.0%})")
    if frac > 0.80:
        print("  ⚠ FLAG: >80% of sampled rows look one-hot — possible policy collapse")


# ───────── main ─────────


def _open_opponent_db(opponent_db: str | None) -> StrategyDB:
    """Open the opponent's strategy DB.

    `None` → an empty in-memory DB with current version set, so every opponent
    decision falls through to `default_policy_action` (the original behavior).
    Otherwise the file-backed DB at `opponent_db`, opened like the trained DB.
    """
    if opponent_db is None:
        db = open_db("sqlite:///:memory:")
        db.set_current_version(1)
        return db
    return open_db(f"sqlite:///{Path(opponent_db).resolve()}")


def _comparison_header(db: str, opponent_db: str | None, n_hands: int, table_size: int) -> str:
    """Self-explanatory one-liner naming what's being compared, e.g.
    'Evaluation: strategy-v5-6max vs strategy-pilot-v2 (50000 hands, 6-max)'."""
    new_stem = Path(db).stem
    opp_stem = Path(opponent_db).stem if opponent_db is not None else "default policy"
    return f"Evaluation: {new_stem} vs {opp_stem} ({n_hands} hands, {table_size}-max)"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-hands", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--db", default="strategy-pilot.db")
    p.add_argument(
        "--opponent-db",
        default=None,
        help="opponent strategy DB file; when omitted the opponent uses an empty "
        "in-memory DB (every decision falls through to default policy)",
    )
    p.add_argument("--abstraction-dir", default="abstraction")
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--sb", type=int, default=5)
    p.add_argument("--bb", type=int, default=10)
    p.add_argument("--table-size", type=int, default=6, choices=[2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--sanity-n", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: trained DB not found at {db_path}", file=sys.stderr)
        return 2

    if args.opponent_db is not None and not Path(args.opponent_db).resolve().exists():
        print(
            f"ERROR: opponent DB not found at {Path(args.opponent_db).resolve()}",
            file=sys.stderr,
        )
        return 2

    print(f"Loading AbstractionTables from {args.abstraction_dir!r}…")
    t_load0 = time.perf_counter()
    abstraction = AbstractionTables(path=args.abstraction_dir)
    if set(abstraction.loaded_streets) != {"flop", "turn", "river"}:
        print(
            f"WARNING: AbstractionTables loaded only {abstraction.loaded_streets}; "
            "miss-path will fall back to placeholder hashes.",
            file=sys.stderr,
        )
    print(f"  loaded in {time.perf_counter() - t_load0:.2f}s")

    trained_db = open_db(f"sqlite:///{db_path}")
    default_db = _open_opponent_db(args.opponent_db)

    trained_adapter = RuntimeAdapter(db=trained_db, abstraction=abstraction, rng_seed=args.seed)
    default_adapter = RuntimeAdapter(
        db=default_db, abstraction=abstraction, rng_seed=args.seed ^ 0xDEADBEEF
    )

    game = SimpleNLHEGame(
        abstraction,
        blinds=(args.sb, args.bb),
        starting_stack=args.starting_stack,
        table_size=args.table_size,
    )

    rng = random.Random(args.seed)
    per_hand_trained: list[int] = []
    per_hand_default: list[int] = []
    fallback_by_street_totals = _empty_fallback_table()
    total_decisions = 0
    total_remaps = 0

    print(
        f"\nPlaying {args.n_hands} hands NLHE {args.table_size}-max "
        f"(SB={args.sb}, BB={args.bb}, stack={args.starting_stack})…"
    )
    t0 = time.perf_counter()
    for i in range(args.n_hands):
        result = _play_one_hand(i, game, trained_adapter, default_adapter, rng)
        per_hand_trained.append(result.trained_delta)
        per_hand_default.append(result.default_delta)
        for s, by_kind in result.fallback_by_street.items():
            for k, v in by_kind.items():
                fallback_by_street_totals[s][k] += v
        total_decisions += result.decisions
        total_remaps += result.illegal_remaps
        if (i + 1) % max(args.n_hands // 10, 1) == 0:
            elapsed = time.perf_counter() - t0
            hps = (i + 1) / elapsed
            print(f"  [{i + 1:>{len(str(args.n_hands))}}/{args.n_hands}] {hps:.1f} hands/sec")
    elapsed = time.perf_counter() - t0

    # ── results ──
    bb_size = args.bb
    trained_mean, trained_lo, trained_hi = _mbb_stats(per_hand_trained, bb_size)
    default_mean, default_lo, default_hi = _mbb_stats(per_hand_default, bb_size)

    sum_trained = sum(per_hand_trained)
    sum_default = sum(per_hand_default)
    if sum_trained + sum_default != 0:
        print(
            f"\n⚠ ZERO-SUM VIOLATION: trained={sum_trained}, default={sum_default}, "
            f"sum={sum_trained + sum_default}",
            file=sys.stderr,
        )
        # already asserted per-hand; this is a redundant guard
        raise AssertionError("aggregate not zero-sum — side-pot bug")

    header = _comparison_header(args.db, args.opponent_db, args.n_hands, args.table_size)
    opp_label = "opponent" if args.opponent_db is not None else "default"

    print(f"\n── results over {args.n_hands} hands ──")
    print(f"  {header}")
    print(f"  elapsed:  {elapsed:.1f}s  ({args.n_hands / elapsed:.1f} hands/sec)")
    print(f"  decisions: {total_decisions}  ({total_decisions / args.n_hands:.1f}/hand avg)")
    remap_frac = total_remaps / total_decisions if total_decisions else 0.0
    print(f"  illegal-action remaps: {total_remaps}  ({remap_frac:.1%} of decisions)")
    if remap_frac > 0.05:
        print("  ⚠ FLAG: >5% remap rate — likely consistency-contract drift between")
        print("    training-time pokerkit gates and runtime-side DB action_mask.")
    print(
        f"  trained mbb/hand:  {trained_mean:+8.2f}   95% CI [{trained_lo:+.2f}, {trained_hi:+.2f}]"
    )
    print(
        f"  {opp_label} mbb/hand:  {default_mean:+8.2f}   95% CI [{default_lo:+.2f}, {default_hi:+.2f}]"
    )
    print(
        f"  zero-sum check: trained+{opp_label} = {trained_mean + default_mean:+.4f} mbb/hand (should be 0)"
    )

    # Aggregate across streets for the headline summary.
    fallback_totals: dict[FallbackUsed, int] = {
        "exact": 0,
        "nearest_neighbor": 0,
        "default_policy": 0,
    }
    for s in range(4):
        for k, v in fallback_by_street_totals[s].items():
            fallback_totals[k] += v
    fb_total = sum(fallback_totals.values())

    print("\n── trained-side fallback breakdown (overall) ──")
    if fb_total == 0:
        print("  (no trained-side decisions recorded)")
    else:
        for k in ("exact", "nearest_neighbor", "default_policy"):
            c = fallback_totals[k]
            print(f"  {k:<18s} {c:>7d}   {c / fb_total:6.1%}")

    print("\n── trained-side fallback breakdown (by street) ──")
    print(f"  {'street':<8s} {'exact':>14s} {'nearest_nbr':>16s} {'default_pol':>16s}   total")
    for s in range(4):
        row = fallback_by_street_totals[s]
        row_total = sum(row.values())
        if row_total == 0:
            print(f"  {_STREET_NAMES[s]:<8s} {'—':>14s} {'—':>16s} {'—':>16s}       0")
            continue
        ex_pct = row["exact"] / row_total
        nn_pct = row["nearest_neighbor"] / row_total
        dp_pct = row["default_policy"] / row_total
        print(
            f"  {_STREET_NAMES[s]:<8s} "
            f"{row['exact']:>6d} ({ex_pct:5.1%})  "
            f"{row['nearest_neighbor']:>6d} ({nn_pct:5.1%})  "
            f"{row['default_policy']:>6d} ({dp_pct:5.1%})  "
            f"{row_total:>6d}"
        )

    _print_sanity(str(db_path), args.sanity_n, rng)

    # ── verdict ──
    print("\n── verdict ──")
    print(f"  {header}")
    if trained_lo > 0 and trained_mean > 50.0:
        print("  PASS: trained > +50 mbb/hand and CI lower bound > 0 → pipeline learned.")
        return 0
    if trained_hi < 0:
        print("  FAIL: trained CI upper bound < 0 → something is broken. STOP and investigate.")
        print("    Likely candidates: (a) consistency-contract drift, (b) policy collapse,")
        print("    (c) action-mask/probs unpacking bug in _row_to_probs.")
        return 1
    print(
        f"  AMBIGUOUS: trained mean {trained_mean:+.2f} mbb/hand, CI "
        f"[{trained_lo:+.2f}, {trained_hi:+.2f}] — re-run at higher N to tighten the interval."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
