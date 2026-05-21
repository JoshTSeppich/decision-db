"""Nash equilibrium push/fold tables for short-stack tournament play.

Solves the open-shove and call-vs-shove games for stack depths 2-15 BB and
2-9 players remaining via fictitious play (Brown 1951) against a 169x169
all-in equity matrix.

Architecture:
    `_solve_hu(stack_bb)`    HU push/fold Nash, the cornerstone primitive.
    `_solve_multi(stack_bb, n_players)`
                              Multi-way: solves all (position, opponent) pairs
                              iteratively, using HU best-response steps plus
                              a single-caller approximation for the pusher's
                              multi-way EV.
    Public API:
        `push_range(stack_bb, position, players_remaining)`
        `call_range(stack_bb, facing_shove_from, hero_position, players_remaining)`
        `pushfold_decision(hole_cards, stack_bb, position, players_remaining,
                           action_history)`

Position convention: position 0 = first to act preflop, position `players_remaining-1`
= last to act (the big blind in standard NLHE). In heads-up: position 0 = SB
(button, acts first), position 1 = BB.

Caching: solved tables are persisted to `~/.cache/pokerbot/pushfold_tables.npz`
on first import (~10s build, ~0.1s load).
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pokerbot.abstraction.preflop import NUM_PREFLOP_BUCKETS, preflop_bucket
from pokerbot.tournament.all_in_equity import (
    COMBOS_PER_BUCKET,
    load_equity_table,
)

if TYPE_CHECKING:
    from pokerbot.runtime.schema import ActionHistoryEntry

# ─────────── constants ───────────

MIN_STACK_BB: int = 2
MAX_STACK_BB: int = 15
MAX_PLAYERS: int = 9
NUM_HANDS: int = NUM_PREFLOP_BUCKETS

SB_BLIND: float = 0.5
BB_BLIND: float = 1.0
TOTAL_COMBOS: float = float(COMBOS_PER_BUCKET.sum())  # = 1326

CACHE_DIR: Path = Path(
    os.environ.get("POKERBOT_CACHE_DIR", str(Path.home() / ".cache" / "pokerbot"))
)
PUSHFOLD_CACHE_PATH: Path = CACHE_DIR / "pushfold_tables.npz"


# ─────────── best-response primitives ───────────


def _br_call(stack_bb: float, push_range: np.ndarray, equity_table: np.ndarray) -> np.ndarray:
    """BB's best-response call range given SB's push range.

    For each BB hand h: call iff stack_bb*(2*eq(h vs push range) - 1) + BB > 0.
    Returns a frequency vector in {0, 1} (deterministic best response).
    """
    push_weighted = COMBOS_PER_BUCKET * push_range
    total = push_weighted.sum()
    eq = (equity_table @ push_weighted) / total if total > 0 else np.full(NUM_HANDS, 0.5)
    delta = stack_bb * (2.0 * eq - 1.0) + BB_BLIND
    return (delta > 0).astype(np.float64)


def _br_push_hu(stack_bb: float, call_range: np.ndarray, equity_table: np.ndarray) -> np.ndarray:
    """SB's best-response push range vs BB's call range, heads-up."""
    call_weighted = COMBOS_PER_BUCKET * call_range
    total_call = call_weighted.sum()
    p_call = total_call / TOTAL_COMBOS  # card-removal ignored
    eq_called = (
        (equity_table @ call_weighted) / total_call if total_call > 0 else np.full(NUM_HANDS, 0.5)
    )
    # Δ = (1-p_call)*BB + p_call*stack_bb*(2*eq - 1) + SB  vs folding (-SB)
    # Push iff Δ > 0.
    delta = (1.0 - p_call) * BB_BLIND + p_call * stack_bb * (2.0 * eq_called - 1.0) + SB_BLIND
    result: np.ndarray = (delta > 0).astype(np.float64)
    return result


def _br_push_multi(
    stack_bb: float,
    caller_ranges: list[np.ndarray],
    equity_table: np.ndarray,
) -> np.ndarray:
    """Pusher's best-response range with multiple potential callers behind.

    Uses an "aggregate caller" approximation: each caller calls independently
    with probability `p_i` (over their hand distribution). Hero's EV combines:
        - P(all fold) * BB                                           (steal blinds)
        - (1 - P(all fold)) * stack_bb * (2*E[eq vs called-by-anyone] - 1)
    where E[eq vs called-by-anyone] is hero's equity against the combined
    range of all callers, weighted by each caller's call probability.

    This understates EV slightly in multi-caller spots (treats hero as facing
    one composite caller rather than the worst case of multiple callers), but
    is the standard simplification used by ICM solvers like HoldemResources
    for tractability.
    """
    if not caller_ranges:
        # No one behind — pure blind-steal, always push (positive EV).
        return np.ones(NUM_HANDS, dtype=np.float64)

    p_calls: list[float] = []
    weighted_ranges: list[np.ndarray] = []
    for cr in caller_ranges:
        weighted = COMBOS_PER_BUCKET * cr
        p_calls.append(weighted.sum() / TOTAL_COMBOS)
        weighted_ranges.append(weighted)

    p_all_fold = float(np.prod([1.0 - p for p in p_calls]))
    p_any_call = 1.0 - p_all_fold

    # Composite calling range = sum over callers of weighted_range
    # (= combos * P(any caller has hand h AND calls)). The composite total
    # weight is sum(p_calls).
    composite: np.ndarray = np.sum(weighted_ranges, axis=0)
    composite_total = float(composite.sum())
    eq_vs_called = (
        (equity_table @ composite) / composite_total
        if composite_total > 0
        else np.full(NUM_HANDS, 0.5)
    )

    delta = p_all_fold * BB_BLIND + p_any_call * stack_bb * (2.0 * eq_vs_called - 1.0) + SB_BLIND
    return (delta > 0).astype(np.float64)


# ─────────── HU solver ───────────


def _solve_hu(
    stack_bb: float,
    equity_table: np.ndarray,
    n_iters: int = 1000,
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """HU push/fold Nash via fictitious play.

    Returns (push_freq, call_freq), each length-169 vector in [0, 1] giving
    the Nash equilibrium frequency for each preflop hand class.
    """
    avg_push = np.full(NUM_HANDS, 0.5)
    avg_call = np.full(NUM_HANDS, 0.5)
    for it in range(1, n_iters + 1):
        br_call = _br_call(stack_bb, avg_push, equity_table)
        br_push = _br_push_hu(stack_bb, avg_call, equity_table)
        # Linear averaging update
        new_push = avg_push + (br_push - avg_push) / it
        new_call = avg_call + (br_call - avg_call) / it
        # Convergence: gap between current avg and best response (exploitability proxy)
        if it > 50:
            gap = max(
                float(np.abs(br_push - avg_push).max()),
                float(np.abs(br_call - avg_call).max()),
            )
            if gap < tol:
                avg_push, avg_call = new_push, new_call
                break
        avg_push, avg_call = new_push, new_call
    return avg_push, avg_call


# ─────────── multi-way solver ───────────


def _solve_multi(
    stack_bb: float,
    n_players: int,
    equity_table: np.ndarray,
    n_iters: int = 200,
    tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-way push/fold for `n_players`.

    Returns:
        push: shape (n_players, 169)  push[p] = position p's open-shove freq.
        call: shape (n_players, n_players, 169)
              call[c, p] = caller c's call freq vs pusher p (only used when p < c).

    Iterative fictitious play across all positions and (caller, pusher) pairs.
    """
    # Seed all positions and pairs with HU Nash as a warm start.
    hu_push, hu_call = _solve_hu(stack_bb, equity_table)
    push = np.tile(hu_push, (n_players, 1))
    call = np.tile(hu_call, (n_players, n_players, 1))

    for it in range(1, n_iters + 1):
        # Compute best responses (no in-place mutation).
        br_call = np.empty_like(call)
        br_push = np.empty_like(push)

        for c in range(1, n_players):
            for p in range(c):
                br_call[c, p] = _br_call(stack_bb, push[p], equity_table)

        for p in range(n_players):
            callers = [call[c, p] for c in range(p + 1, n_players)]
            br_push[p] = _br_push_multi(stack_bb, callers, equity_table)

        # Convergence check (against averages, before update).
        gap = 0.0
        if it > 20:
            for c in range(1, n_players):
                for p in range(c):
                    gap = max(gap, float(np.abs(br_call[c, p] - call[c, p]).max()))
            for p in range(n_players):
                gap = max(gap, float(np.abs(br_push[p] - push[p]).max()))

        # Linear-averaging update
        push = push + (br_push - push) / it
        for c in range(1, n_players):
            for p in range(c):
                call[c, p] = call[c, p] + (br_call[c, p] - call[c, p]) / it

        if it > 20 and gap < tol:
            break
    return push, call


# ─────────── load-or-build cached tables ───────────


def _build_all_tables(equity_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve all (stack_bb, n_players) combinations. Returns big arrays."""
    n_stacks = MAX_STACK_BB - MIN_STACK_BB + 1  # 14
    n_player_counts = MAX_PLAYERS - 1  # 2..9 → 8 values
    # push: (n_stacks, n_player_counts, max_players, 169)
    # call: (n_stacks, n_player_counts, max_players, max_players, 169)
    push_all = np.zeros((n_stacks, n_player_counts, MAX_PLAYERS, NUM_HANDS), dtype=np.float32)
    call_all = np.zeros(
        (n_stacks, n_player_counts, MAX_PLAYERS, MAX_PLAYERS, NUM_HANDS),
        dtype=np.float32,
    )
    for s_idx, stack_bb in enumerate(range(MIN_STACK_BB, MAX_STACK_BB + 1)):
        for p_idx, n_players in enumerate(range(2, MAX_PLAYERS + 1)):
            push, call = _solve_multi(float(stack_bb), n_players, equity_table)
            push_all[s_idx, p_idx, :n_players, :] = push
            call_all[s_idx, p_idx, :n_players, :n_players, :] = call
    return push_all, call_all


_PUSH_TABLE: np.ndarray
_CALL_TABLE: np.ndarray
_EQUITY_TABLE: np.ndarray


def _initialize_tables(force_rebuild: bool = False) -> None:
    global _PUSH_TABLE, _CALL_TABLE, _EQUITY_TABLE
    _EQUITY_TABLE = load_equity_table()
    if not force_rebuild and PUSHFOLD_CACHE_PATH.exists():
        with np.load(PUSHFOLD_CACHE_PATH) as data:
            _PUSH_TABLE = data["push"].astype(np.float32, copy=False)
            _CALL_TABLE = data["call"].astype(np.float32, copy=False)
            expected_push_shape = (
                MAX_STACK_BB - MIN_STACK_BB + 1,
                MAX_PLAYERS - 1,
                MAX_PLAYERS,
                NUM_HANDS,
            )
            if _PUSH_TABLE.shape == expected_push_shape:
                return
    print("[pushfold] solving Nash tables (cache miss)…")
    t0 = time.perf_counter()
    push, call = _build_all_tables(_EQUITY_TABLE)
    dt = time.perf_counter() - t0
    print(f"[pushfold] solved in {dt:.2f}s; caching to {PUSHFOLD_CACHE_PATH}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PUSHFOLD_CACHE_PATH, push=push, call=call)
    _PUSH_TABLE = push
    _CALL_TABLE = call


_initialize_tables()


# ─────────── public API ───────────


def _hand_key(bucket_id: int) -> tuple[int, int, bool]:
    """Bucket ID → (high_rank, low_rank, is_suited) tuple key for public API."""
    if bucket_id < 13:
        return (bucket_id, bucket_id, False)  # pair: high == low
    is_suited = bucket_id < 91
    pair_idx = bucket_id - 13 if is_suited else bucket_id - 91
    high = 1
    while (high * (high + 1)) // 2 <= pair_idx:
        high += 1
    low = pair_idx - (high * (high - 1)) // 2
    return (high, low, is_suited)


@lru_cache(maxsize=1)
def _all_hand_keys() -> list[tuple[int, int, bool]]:
    return [_hand_key(b) for b in range(NUM_HANDS)]


def _clamp_stack(stack_bb: int) -> int:
    if stack_bb < MIN_STACK_BB:
        return MIN_STACK_BB
    if stack_bb > MAX_STACK_BB:
        return MAX_STACK_BB
    return stack_bb


def push_range(
    stack_bb: int,
    position: int,
    players_remaining: int,
) -> dict[tuple[int, int, bool], float]:
    """Nash open-shove range as `{(high_rank, low_rank, is_suited): freq}`.

    Position 0 = first to act preflop. `players_remaining` ∈ [2, 9].
    """
    if not 2 <= players_remaining <= MAX_PLAYERS:
        raise ValueError(f"players_remaining must be in [2, 9], got {players_remaining}")
    if not 0 <= position < players_remaining:
        raise ValueError(f"position {position} out of range for {players_remaining} players")
    stack = _clamp_stack(stack_bb)
    s_idx = stack - MIN_STACK_BB
    p_idx = players_remaining - 2
    freqs = _PUSH_TABLE[s_idx, p_idx, position]  # shape (169,)
    return {_hand_key(b): float(freqs[b]) for b in range(NUM_HANDS)}


def call_range(
    stack_bb: int,
    facing_shove_from: int,
    hero_position: int,
    players_remaining: int,
) -> dict[tuple[int, int, bool], float]:
    """Nash calling range vs a shove from `facing_shove_from`."""
    if not 2 <= players_remaining <= MAX_PLAYERS:
        raise ValueError(f"players_remaining must be in [2, 9], got {players_remaining}")
    if not 0 <= facing_shove_from < hero_position < players_remaining:
        raise ValueError(
            f"invalid positions: hero={hero_position}, pusher={facing_shove_from}, "
            f"n={players_remaining} (require pusher < hero < n)"
        )
    stack = _clamp_stack(stack_bb)
    s_idx = stack - MIN_STACK_BB
    p_idx = players_remaining - 2
    freqs = _CALL_TABLE[s_idx, p_idx, hero_position, facing_shove_from]
    return {_hand_key(b): float(freqs[b]) for b in range(NUM_HANDS)}


def _canonical_hand_key(hole_cards: tuple[int, int]) -> tuple[int, int, bool]:
    """Card-int pair → (high, low, is_suited) lookup key."""
    c1, c2 = hole_cards
    r1, s1 = c1 >> 2, c1 & 3
    r2, s2 = c2 >> 2, c2 & 3
    high = max(r1, r2)
    low = min(r1, r2)
    is_suited = (s1 == s2) and (r1 != r2)
    return (high, low, is_suited)


def _hand_to_bucket(hole_cards: tuple[int, int]) -> int:
    c1, c2 = hole_cards
    return preflop_bucket((min(c1, c2), max(c1, c2)))


def pushfold_decision(
    hole_cards: tuple[int, int],
    stack_bb: int,
    position: int,
    players_remaining: int,
    action_history: list[ActionHistoryEntry],
) -> dict[str, float]:
    """Push/fold/call decision for hero with `hole_cards` at the given spot.

    Routes between `push_range` (no prior shove → open spot) and `call_range`
    (someone has shoved → call/fold spot). If hero is facing only folds, treats
    as open-shove spot.
    """
    bucket = _hand_to_bucket(hole_cards)
    pusher_pos = _find_shover(action_history)
    if pusher_pos is None or pusher_pos >= position:
        # Open spot — push_range applies (`pusher_pos >= position` shouldn't happen
        # in practice but defensively routes to open spot).
        pr = push_range(stack_bb, position, players_remaining)
        push_p = pr[_hand_key(bucket)]
        return {"push": push_p, "fold": 1.0 - push_p}
    # Facing a shove — call_range applies.
    cr = call_range(stack_bb, pusher_pos, position, players_remaining)
    call_p = cr[_hand_key(bucket)]
    return {"call": call_p, "fold": 1.0 - call_p}


def _find_shover(action_history: list[ActionHistoryEntry]) -> int | None:
    """Position of the player who shoved, or None if nobody has."""
    for entry in action_history:
        if entry.type == "all-in":
            return entry.seat
        # Heuristic for push/fold context: large preflop bet is treated as a shove.
        # (In real push/fold games actions are limited to fold/push/call.)
        if entry.type == "raise" and entry.amount > 0:
            return entry.seat
    return None


__all__ = [
    "MAX_PLAYERS",
    "MAX_STACK_BB",
    "MIN_STACK_BB",
    "call_range",
    "push_range",
    "pushfold_decision",
]
