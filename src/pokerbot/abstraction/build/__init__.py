"""Build the postflop abstraction NPZ artifacts (Spec.html §A).

Production usage:
    pokerbot build-abstraction --out /path/to/abstraction --samples 1000

Internals (also importable for tests):
    - `cards`         int 0..51 ↔ phevaluator string
    - `equity`        EHS + OCHS Monte Carlo
    - `preflop_tiers` 8 preflop equity tiers for OCHS opponents
    - `clustering`    k-means with Euclidean (OCHS) and sorted-EMD (potential-aware)
    - `enumerate`     canonical (hole, board) class generators per street
    - `builder`       composes everything; writes buckets_*.npz
"""

from pokerbot.abstraction.build.builder import (
    BuildConfig,
    build_flop_buckets,
    build_river_buckets,
    build_turn_buckets,
)

__all__ = [
    "BuildConfig",
    "build_flop_buckets",
    "build_river_buckets",
    "build_turn_buckets",
]
