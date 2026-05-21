"""169x169 preflop all-in equity table.

For every ordered pair of preflop hand classes `(i, j)` (0..168 each, using
the lossless 169-bucket map from `pokerbot.abstraction.preflop`), this module
provides `P(hand i beats hand j) + 0.5*P(tie)` at all-in showdown (random
5-card runout).

Generation uses a *shared-board* Monte Carlo: each random board is reused
across all 169 hand classes simultaneously. For one random board:

    1. For each hand class h, sample one specific (c1, c2) combo whose
       cards don't appear on the board.
    2. Evaluate the 7-card hand strength for each h once (169 phevaluator
       calls per board).
    3. For every (h1, h2) pair whose sampled combos also don't overlap each
       other, tally win/tie/loss vectorized via numpy.

This reduces ~170M phevaluator calls (28k pairs x 5k samples x 2 evals) to
~850k calls (5k boards x 169) — about 2-5 seconds wall clock.

The resulting matrix is cached at `~/.cache/pokerbot/all_in_equity.npz` and
loaded on subsequent imports.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
from phevaluator.evaluator import evaluate_cards

from pokerbot.abstraction.build.cards import card_int_to_str
from pokerbot.abstraction.preflop import NUM_PREFLOP_BUCKETS, preflop_bucket

DEFAULT_N_BOARDS: int = 5000
CACHE_DIR: Path = Path(
    os.environ.get("POKERBOT_CACHE_DIR", str(Path.home() / ".cache" / "pokerbot"))
)
CACHE_PATH: Path = CACHE_DIR / "all_in_equity.npz"


# ─────────── bucket → combos enumeration ───────────


def _bucket_combos(bucket_id: int) -> list[tuple[int, int]]:
    """All specific (c1, c2) card-int combos belonging to a preflop bucket.

    Card ints follow the project convention: `card = rank * 4 + suit`,
    `rank ∈ 0..12` (2..A), `suit ∈ 0..3` (c, d, h, s).
    """
    if bucket_id < 13:
        # Pocket pairs. 6 combos: C(4,2) pairs of suits.
        rank = bucket_id
        combos: list[tuple[int, int]] = []
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                combos.append((rank * 4 + s1, rank * 4 + s2))
        return combos

    # Non-pair: recover (high, low) from triangular index.
    is_suited = bucket_id < 91
    pair_idx = bucket_id - 13 if is_suited else bucket_id - 91
    high = 1
    while (high * (high + 1)) // 2 <= pair_idx:
        high += 1
    low = pair_idx - (high * (high - 1)) // 2

    combos = []
    if is_suited:
        for s in range(4):
            combos.append((low * 4 + s, high * 4 + s))
    else:
        for s_low in range(4):
            for s_high in range(4):
                if s_low != s_high:
                    combos.append((low * 4 + s_low, high * 4 + s_high))
    return combos


# Self-test: enumerate all combos, check we get exactly 1326.
_ALL_BUCKET_COMBOS: list[list[tuple[int, int]]] = [
    _bucket_combos(h) for h in range(NUM_PREFLOP_BUCKETS)
]
_TOTAL_COMBO_CHECK = sum(len(c) for c in _ALL_BUCKET_COMBOS)
if _TOTAL_COMBO_CHECK != 1326:
    raise AssertionError(
        f"combo enumeration broken: got {_TOTAL_COMBO_CHECK} combos, expected 1326"
    )

# Verify each combo's preflop_bucket matches its source bucket.
for _bid in range(NUM_PREFLOP_BUCKETS):
    for _c1, _c2 in _ALL_BUCKET_COMBOS[_bid]:
        _canon = (min(_c1, _c2), max(_c1, _c2))
        if preflop_bucket(_canon) != _bid:
            raise AssertionError(
                f"combo {(_c1, _c2)} for bucket {_bid} maps to {preflop_bucket(_canon)}"
            )


# Number of combos per bucket; used by pushfold to weight hand-distribution priors.
COMBOS_PER_BUCKET: np.ndarray = np.array(
    [len(_ALL_BUCKET_COMBOS[h]) for h in range(NUM_PREFLOP_BUCKETS)],
    dtype=np.float64,
)


# ─────────── shared-board MC equity generation ───────────


def _evaluate_one_board(board: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """For one random 5-card board, sample one valid combo per hand class
    (whose cards don't appear on the board) and evaluate its 7-card strength.

    Returns (ranks, cards_used) where:
      ranks[h]      = phevaluator rank (lower = stronger), or -1 if no
                      conflict-free combo could be sampled for h on this board.
      cards_used[h] = (c1, c2) used for h, or (-1, -1) if h was skipped.
    """
    board_set = set(board)
    board_strs = [card_int_to_str(c) for c in board]
    ranks = np.full(NUM_PREFLOP_BUCKETS, -1, dtype=np.int32)
    cards_used = np.full((NUM_PREFLOP_BUCKETS, 2), -1, dtype=np.int8)
    for h in range(NUM_PREFLOP_BUCKETS):
        options = [
            c for c in _ALL_BUCKET_COMBOS[h] if c[0] not in board_set and c[1] not in board_set
        ]
        if not options:
            continue
        c1, c2 = rng.choice(options)
        ranks[h] = evaluate_cards(
            card_int_to_str(c1),
            card_int_to_str(c2),
            *board_strs,
        )
        cards_used[h, 0] = c1
        cards_used[h, 1] = c2
    return ranks, cards_used


def _build_equity_table(n_boards: int, rng_seed: int = 0xEE) -> np.ndarray:
    """Generate the 169x169 equity matrix.

    Off-diagonal cells use *weighted* shared-board sampling: each random
    board is reused across all 169 hand classes, but each pairwise (i, j)
    sample is weighted by `|valid_combos_i_given_board| * |valid_combos_j_given_board|`.
    This corrects the bias that arises because boards containing rank-i
    or rank-j cards force a smaller pool of valid combos (making naive
    shared-board over-sample boards where hero/opp have few combo choices).

    Diagonal cells (`h` vs `h`) require sampling two distinct combos from
    the same hand class. Handled by a separate per-cell MC pass; expected
    near 0.5 (with tie fraction).

    Returns float32 array shape (169, 169).
    """
    rng = random.Random(rng_seed)
    # weighted accumulators
    num = np.zeros((NUM_PREFLOP_BUCKETS, NUM_PREFLOP_BUCKETS), dtype=np.float64)
    denom = np.zeros((NUM_PREFLOP_BUCKETS, NUM_PREFLOP_BUCKETS), dtype=np.float64)

    for _ in range(n_boards):
        board = rng.sample(range(52), 5)
        board_set = set(board)
        ranks, cards_used = _evaluate_one_board(board, rng)

        # For each hand, count how many of its combos are conflict-free with this board.
        valid_count = np.zeros(NUM_PREFLOP_BUCKETS, dtype=np.float64)
        for h in range(NUM_PREFLOP_BUCKETS):
            if ranks[h] < 0:
                continue
            valid_count[h] = sum(
                1 for c in _ALL_BUCKET_COMBOS[h] if c[0] not in board_set and c[1] not in board_set
            )

        valid_hand = ranks >= 0
        # Card-overlap mask for the SAMPLED combos (one per hand).
        presence = np.zeros((NUM_PREFLOP_BUCKETS, 52), dtype=np.int8)
        for h in range(NUM_PREFLOP_BUCKETS):
            if valid_hand[h]:
                presence[h, cards_used[h, 0]] = 1
                presence[h, cards_used[h, 1]] = 1
        conflict = (presence @ presence.T) > 0
        valid_pair = valid_hand[:, None] & valid_hand[None, :] & ~conflict

        # Outer product of valid counts → weight for each (i, j) pair.
        weights = np.outer(valid_count, valid_count)
        weights *= valid_pair  # zero out invalid pairs

        # Pairwise rank comparison; contribution is 1 (win), 0.5 (tie), or 0 (loss).
        rank_i = ranks[:, None]
        rank_j = ranks[None, :]
        contribution = np.where(
            (rank_i < rank_j) & valid_pair,
            1.0,
            np.where((rank_i == rank_j) & valid_pair, 0.5, 0.0),
        )

        num += contribution * weights
        denom += weights

    equity = np.zeros((NUM_PREFLOP_BUCKETS, NUM_PREFLOP_BUCKETS), dtype=np.float32)
    nz = denom > 0
    equity[nz] = (num[nz] / denom[nz]).astype(np.float32)

    # Diagonal: separate per-cell MC. AA vs AA, AKs vs AKs, etc.
    rng_diag = random.Random(rng_seed ^ 0x77)
    n_diag = max(n_boards // 4, 500)  # cheaper since only 169 cells
    for h in range(NUM_PREFLOP_BUCKETS):
        combos = _ALL_BUCKET_COMBOS[h]
        if len(combos) < 2:
            equity[h, h] = 0.5
            continue
        wins = ties = total = 0
        attempts = 0
        max_attempts = n_diag * 5
        while total < n_diag and attempts < max_attempts:
            attempts += 1
            hero_combo = rng_diag.choice(combos)
            opp_combo = rng_diag.choice(combos)
            if set(hero_combo) & set(opp_combo):
                continue
            used = set(hero_combo) | set(opp_combo)
            deck = [c for c in range(52) if c not in used]
            board = rng_diag.sample(deck, 5)
            bs = [card_int_to_str(b) for b in board]
            rh = evaluate_cards(
                card_int_to_str(hero_combo[0]),
                card_int_to_str(hero_combo[1]),
                *bs,
            )
            ro = evaluate_cards(
                card_int_to_str(opp_combo[0]),
                card_int_to_str(opp_combo[1]),
                *bs,
            )
            if rh < ro:
                wins += 1
            elif rh == ro:
                ties += 1
            total += 1
        equity[h, h] = (wins + 0.5 * ties) / total if total > 0 else 0.5

    return equity


# ─────────── load-or-build with caching ───────────


def load_equity_table(n_boards: int = DEFAULT_N_BOARDS, force_rebuild: bool = False) -> np.ndarray:
    """Return the 169x169 all-in equity matrix, building + caching if needed.

    Cache path: ~/.cache/pokerbot/all_in_equity.npz (overridable via
    `POKERBOT_CACHE_DIR`).
    """
    if not force_rebuild and CACHE_PATH.exists():
        with np.load(CACHE_PATH) as data:
            arr = data["equity"]
            if arr.shape == (NUM_PREFLOP_BUCKETS, NUM_PREFLOP_BUCKETS):
                cast_arr: np.ndarray = arr.astype(np.float32, copy=False)
                return cast_arr
    print(
        f"[all_in_equity] building 169x169 equity table "
        f"(n_boards={n_boards}, cache miss); this takes a few seconds…"
    )
    t0 = time.perf_counter()
    equity = _build_equity_table(n_boards)
    dt = time.perf_counter() - t0
    print(f"[all_in_equity] built in {dt:.2f}s; caching to {CACHE_PATH}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_PATH, equity=equity)
    return equity


__all__ = [
    "CACHE_PATH",
    "COMBOS_PER_BUCKET",
    "DEFAULT_N_BOARDS",
    "load_equity_table",
]
