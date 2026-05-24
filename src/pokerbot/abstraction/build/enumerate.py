"""Canonical (hole, board) class generators per street (Spec.html §A).

Each canonical class is a (hole_tuple, board_tuple) pair after suit
isomorphism via `canonical_hand` (in `pokerbot.abstraction.cards`).

For production builds the generators yield millions of classes; callers
typically materialise to a list/array and feed downstream stages.

`--max-classes` in the CLI sub-samples by yielding the first N (after
optional random shuffle) instead of the full enumeration — that's how a
multi-hour smoke build is achievable in minutes.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pokerbot.abstraction.cards import canonical_hand

if TYPE_CHECKING:
    from collections.abc import Iterator

CanonicalClass = tuple[tuple[int, int], tuple[int, ...]]


def _iter_5_card_combos() -> Iterator[tuple[tuple[int, int], tuple[int, int, int]]]:
    """All (hole, flop) combos with distinct cards. ~26M outputs — caller should
    canonicalize + dedupe.
    """
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            for b1 in range(52):
                if b1 in (c1, c2):
                    continue
                for b2 in range(b1 + 1, 52):
                    if b2 in (c1, c2):
                        continue
                    for b3 in range(b2 + 1, 52):
                        if b3 in (c1, c2):
                            continue
                        yield (c1, c2), (b1, b2, b3)


def enumerate_canonical_flops(
    *,
    max_classes: int | None = None,
    rng_seed: int = 0,
) -> list[CanonicalClass]:
    """All distinct canonical (hole, flop) classes in lexicographic order.

    With `max_classes`, returns a random sample of that many distinct classes
    (deterministic given `rng_seed`).
    """
    seen: set[CanonicalClass] = set()
    for hole, flop in _iter_5_card_combos():
        canon = canonical_hand(hole, flop)
        seen.add(canon)
    result = sorted(seen)
    if max_classes is None or max_classes >= len(result):
        return result
    rng = random.Random(rng_seed)
    rng.shuffle(result)
    return sorted(result[:max_classes])


def enumerate_canonical_turns(
    *,
    max_classes: int | None = None,
    rng_seed: int = 0,
) -> list[CanonicalClass]:
    """All canonical (hole, flop+turn) classes — len(board) == 4."""
    seen: set[CanonicalClass] = set()
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            for b1 in range(52):
                if b1 in (c1, c2):
                    continue
                for b2 in range(b1 + 1, 52):
                    if b2 in (c1, c2):
                        continue
                    for b3 in range(b2 + 1, 52):
                        if b3 in (c1, c2):
                            continue
                        for b4 in range(52):
                            if b4 in (c1, c2, b1, b2, b3):
                                continue
                            canon = canonical_hand((c1, c2), (b1, b2, b3, b4))
                            seen.add(canon)
    result = sorted(seen)
    if max_classes is None or max_classes >= len(result):
        return result
    rng = random.Random(rng_seed)
    rng.shuffle(result)
    return sorted(result[:max_classes])


def enumerate_canonical_rivers(
    *,
    max_classes: int | None = None,
    rng_seed: int = 0,
) -> list[CanonicalClass]:
    """All canonical (hole, full-board) classes — len(board) == 5.

    The full enumeration is ~10M classes. Most production builds use a
    subsample via `max_classes` because OCHS Monte Carlo across the full
    set is a multi-day job at our phevaluator throughput.
    """
    seen: set[CanonicalClass] = set()
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            for b1 in range(52):
                if b1 in (c1, c2):
                    continue
                for b2 in range(b1 + 1, 52):
                    if b2 in (c1, c2):
                        continue
                    for b3 in range(b2 + 1, 52):
                        if b3 in (c1, c2):
                            continue
                        for b4 in range(b3 + 1, 52):
                            if b4 in (c1, c2):
                                continue
                            for b5 in range(b4 + 1, 52):
                                if b5 in (c1, c2):
                                    continue
                                canon = canonical_hand((c1, c2), (b1, b2, b3, b4, b5))
                                seen.add(canon)
    result = sorted(seen)
    if max_classes is None or max_classes >= len(result):
        return result
    rng = random.Random(rng_seed)
    rng.shuffle(result)
    return sorted(result[:max_classes])


def sample_canonical_rivers(
    n: int,
    *,
    rng_seed: int = 0,
) -> list[CanonicalClass]:
    """Random sample without full enumeration — much faster for smoke runs.

    Draws random distinct 7-card combos until `n` distinct canonical classes
    are collected. Returns sorted result.
    """
    rng = random.Random(rng_seed)
    deck = list(range(52))
    seen: set[CanonicalClass] = set()
    while len(seen) < n:
        rng.shuffle(deck)
        hole = (min(deck[0], deck[1]), max(deck[0], deck[1]))
        board = tuple(deck[2:7])
        canon = canonical_hand(hole, board)
        seen.add(canon)
    return sorted(seen)


def sample_canonical_flops(n: int, *, rng_seed: int = 0) -> list[CanonicalClass]:
    rng = random.Random(rng_seed)
    deck = list(range(52))
    seen: set[CanonicalClass] = set()
    while len(seen) < n:
        rng.shuffle(deck)
        hole = (min(deck[0], deck[1]), max(deck[0], deck[1]))
        board = tuple(deck[2:5])
        canon = canonical_hand(hole, board)
        seen.add(canon)
    return sorted(seen)


def sample_canonical_turns(n: int, *, rng_seed: int = 0) -> list[CanonicalClass]:
    rng = random.Random(rng_seed)
    deck = list(range(52))
    seen: set[CanonicalClass] = set()
    while len(seen) < n:
        rng.shuffle(deck)
        hole = (min(deck[0], deck[1]), max(deck[0], deck[1]))
        board = tuple(deck[2:6])
        canon = canonical_hand(hole, board)
        seen.add(canon)
    return sorted(seen)


__all__ = [
    "CanonicalClass",
    "enumerate_canonical_flops",
    "enumerate_canonical_rivers",
    "enumerate_canonical_turns",
    "sample_canonical_flops",
    "sample_canonical_rivers",
    "sample_canonical_turns",
]
