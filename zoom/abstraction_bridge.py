"""Adapters from the frozen `pokerbot` abstraction into the zoom modules' seams.

Read-only against `pokerbot`: this module *wraps* `AbstractionTables` and the
`build.equity` Monte-Carlo helper; it never modifies them.

Two seams are bridged:

  * ``make_bucket_fn`` adapts ``AbstractionTables.lookup(hole, board, street)``
    into the 2-arg ``bucket_fn(hole, board) -> int`` callable the opponent model
    injects. The opponent model never passes a street, so we derive it from the
    board length. Card ints are pokerbot-compatible (``rank*4 + suit``, ranks
    ``23456789TJQKA``, suits ``cdhs``) in BOTH the staged modules and pokerbot,
    so no card-int translation is needed.

  * ``posterior_range_equity`` is the real-equity analogue of
    ``RangeTracker.range_strength()`` (which uses the self-contained Chen
    heuristic). It reports the posterior-weighted Monte-Carlo showdown equity of
    a villain's inferred range using phevaluator. Chen ``range_strength()`` stays
    as the no-board / no-``[train]``-deps fallback (it is not modified).

The phevaluator-backed import is lazy (inside the equity functions) so that
``make_bucket_fn`` remains usable with only the base deps installed.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from pokerbot.abstraction import AbstractionTables

if TYPE_CHECKING:
    from pokerbot.abstraction import Street

# Inverse of pokerbot.abstraction.cards._EXPECTED_BOARD_LEN (board length → street).
# Kept explicit (that mapping is private) and matched to the validated lengths.
_STREET_BY_BOARD_LEN: dict[int, "Street"] = {
    0: "preflop",
    3: "flop",
    4: "turn",
    5: "river",
}

BucketFn = Callable[[tuple[int, int], tuple[int, ...]], int]


def street_for_board(board: tuple[int, ...]) -> "Street":
    """Map a board (0/3/4/5 cards) to its abstraction `Street` label."""
    try:
        return _STREET_BY_BOARD_LEN[len(board)]
    except KeyError:
        raise ValueError(f"board must have 0, 3, 4, or 5 cards; got {len(board)}") from None


def make_bucket_fn(tables: AbstractionTables) -> BucketFn:
    """Adapt `AbstractionTables.lookup` into the `bucket_fn(hole, board) -> int`
    seam the opponent model expects, deriving the street from the board length.

    Wraps the abstraction; does not modify it. Preflop buckets are deterministic
    (lossless 169 classes) even without NPZ artifacts on disk.
    """

    def bucket_fn(hole: tuple[int, int], board: tuple[int, ...]) -> int:
        board_t = tuple(board)
        return tables.lookup(tuple(hole), board_t, street_for_board(board_t))

    return bucket_fn


def real_equity(
    hole: tuple[int, int],
    board: tuple[int, ...],
    *,
    num_samples: int = 200,
    rng: random.Random | None = None,
) -> float:
    """P(hero wins ± ties) at showdown via Monte Carlo, backed by phevaluator.

    Thin read-only wrapper over `pokerbot.abstraction.build.equity.hand_strength`.
    Requires the `[train]` extra (phevaluator); import is lazy so this module
    stays importable without it.
    """
    from pokerbot.abstraction.build.equity import hand_strength

    if rng is None:
        rng = random.Random(0)
    return hand_strength(tuple(hole), tuple(board), num_samples=num_samples, rng=rng)


def posterior_range_equity(
    posterior: dict[tuple[int, int], float],
    board: tuple[int, ...],
    *,
    num_samples: int = 200,
    rng: random.Random | None = None,
    top_k: int | None = None,
) -> float:
    """Posterior-weighted showdown equity of a villain's inferred range:
    ``sum_combo posterior[combo] * real_equity(combo, board)``.

    The real-equity analogue of `RangeTracker.range_strength()`. Intended for
    board-present use (a board sharpens equity); preflop it still works but the
    Chen `range_strength()` is the cheaper default there. `top_k` restricts the
    sum to the highest-mass combos (renormalized) to bound the MC cost — most of
    the posterior mass after a concentrating action lives in a few combos.
    Returns 0.5 (neutral) for an empty posterior.
    """
    if not posterior:
        return 0.5
    if rng is None:
        rng = random.Random(0)
    items = sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        items = items[:top_k]
    weight = sum(p for _, p in items) or 1.0
    return sum(
        (p / weight) * real_equity(combo, board, num_samples=num_samples, rng=rng)
        for combo, p in items
    )


__all__ = [
    "BucketFn",
    "make_bucket_fn",
    "posterior_range_equity",
    "real_equity",
    "street_for_board",
]
