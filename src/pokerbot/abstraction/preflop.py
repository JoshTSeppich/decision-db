"""Lossless preflop 169-bucket map (Spec.html §A).

Layout of bucket ids (closed-form, no NPZ artifact required):

    [0, 12]    pocket pairs, ordered by rank (22→0 … AA→12)
    [13, 90]   78 suited combos, ordered by (high rank, low rank)
    [91, 168]  78 offsuit combos, ordered by (high rank, low rank)

A canonical 2-card hole (see `cards.canonical_hand`) determines suited-ness from
its suit fields alone — both cards share canonical suit 0 iff the hand is suited.
"""

from __future__ import annotations

from typing import Final

NUM_PREFLOP_BUCKETS: Final[int] = 169
_NUM_PAIRS: Final[int] = 13
_NUM_NON_PAIR: Final[int] = 78  # C(13,2)


def preflop_bucket(canonical_hole: tuple[int, int]) -> int:
    """Map a canonicalized 2-card hole (cards 0..51, ascending) to a 169-bucket id."""
    c1, c2 = canonical_hole
    if c1 >= c2:
        raise ValueError(f"canonical_hole must be ascending: {canonical_hole!r}")
    r1, s1 = c1 >> 2, c1 & 3
    r2, s2 = c2 >> 2, c2 & 3

    if r1 == r2:
        return r2  # 0..12

    suited = s1 == s2
    # Triangular index for (high=r2, low=r1) with r2 > r1: 0..77
    pair_idx = (r2 * (r2 - 1)) // 2 + r1
    return _NUM_PAIRS + pair_idx + (0 if suited else _NUM_NON_PAIR)


__all__ = ["NUM_PREFLOP_BUCKETS", "preflop_bucket"]
