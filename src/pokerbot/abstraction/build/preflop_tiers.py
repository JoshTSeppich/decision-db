"""Group the 169 canonical preflop hands into 8 equity tiers (Spec.html §A).

Tiers are the *opponent* clusters that OCHS projects river hands onto:
    tier 0 = strongest preflop hands (premiums)
    tier 7 = weakest (trash)

Each tier is a tuple of (hole_card_int_pair, ...) — canonical representatives.
Building river OCHS features means computing equity against each tier.
"""

from __future__ import annotations

import random
from typing import Final

from pokerbot.abstraction.build.equity import hand_strength

NUM_PREFLOP_TIERS: Final[int] = 8


def _canonical_preflop_representatives() -> list[tuple[int, int]]:
    """169 canonical 2-card holes in sorted-by-id order."""
    reps: list[tuple[int, int]] = []
    # Pocket pairs: (rank*4+0, rank*4+1) for rank 0..12
    for r in range(13):
        reps.append((r * 4 + 0, r * 4 + 1))
    # Non-pairs (suited): both suit 0
    for hi in range(1, 13):
        for lo in range(hi):
            reps.append((lo * 4 + 0, hi * 4 + 0))
    # Non-pairs (offsuit): suits 0 and 1
    for hi in range(1, 13):
        for lo in range(hi):
            reps.append((lo * 4 + 0, hi * 4 + 1))
    return reps


def build_preflop_tiers(
    *,
    samples_per_hand: int = 2000,
    rng_seed: int = 0xC0FFEE,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Compute preflop equity for each canonical hand, then partition into 8
    equal-size tiers ordered by equity (descending).

    Returns a tuple of 8 tuples, each containing canonical hole-card pairs.
    Tier 0 = ~top 12% of hands, tier 7 = ~bottom 12%.
    """
    reps = _canonical_preflop_representatives()
    rng = random.Random(rng_seed)
    equities: list[tuple[float, tuple[int, int]]] = []
    for hand in reps:
        eq = hand_strength(hand, (), num_samples=samples_per_hand, rng=rng)
        equities.append((eq, hand))
    equities.sort(key=lambda x: -x[0])  # descending equity

    tiers: list[list[tuple[int, int]]] = [[] for _ in range(NUM_PREFLOP_TIERS)]
    for i, (_eq, hand) in enumerate(equities):
        tier_idx = (i * NUM_PREFLOP_TIERS) // len(equities)
        tiers[tier_idx].append(hand)
    return tuple(tuple(t) for t in tiers)


__all__ = [
    "NUM_PREFLOP_TIERS",
    "build_preflop_tiers",
]
