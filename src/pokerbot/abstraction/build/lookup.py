"""On-the-fly feature computation for AbstractionTables.lookup miss path.

The production builds sample ~200k canonical classes out of ~5M total per
street, so most runtime queries don't hit an exact NPZ entry. Instead of
falling back to the deterministic placeholder hash (which assigns arbitrary
buckets), we compute the queried hand's feature vector on demand and project
onto the saved centroids by nearest distance.

    river  → 8-d OCHS feature vector, L2 distance
    flop   → 50-bin EHS² histogram, then cumsum → L1 distance (= 1-D EMD)
    turn   → same as flop

Cost per miss: ~5-10ms (river OCHS at 30 samples x 8 tiers x 2 evals = 480
phevaluator calls). Tractable for runtime; the trainer should pre-cache.

Preflop tiers are deterministic-from-seed but expensive to rebuild (~6s for
2000 samples x 169 hands). We cache the result in a module-level singleton
so it amortises across the lifetime of the process.
"""

from __future__ import annotations

import hashlib
import random
import threading
from typing import Literal

import numpy as np
import numpy.typing as npt

from pokerbot.abstraction.build.equity import (
    hand_strength,
    hand_strength_vs_cluster,
)
from pokerbot.abstraction.build.preflop_tiers import build_preflop_tiers

Street = Literal["flop", "turn", "river"]


_TIERS_LOCK = threading.Lock()
_TIERS_CACHE: tuple[tuple[tuple[int, int], ...], ...] | None = None
_TIERS_SEED: int | None = None


def get_preflop_tiers(seed: int = 0xC0FFEE) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Thread-safe cached preflop-tier accessor. Rebuilds if seed changes."""
    global _TIERS_CACHE, _TIERS_SEED
    with _TIERS_LOCK:
        if _TIERS_CACHE is None or seed != _TIERS_SEED:
            _TIERS_CACHE = build_preflop_tiers(samples_per_hand=2000, rng_seed=seed)
            _TIERS_SEED = seed
        return _TIERS_CACHE


def _ehs2_histogram_query(
    hole: tuple[int, int],
    board: tuple[int, ...],
    *,
    bins: int,
    inner_samples: int,
    rng: random.Random,
) -> npt.NDArray[np.float32]:
    """Standalone copy of builder._ehs2_histogram for use at lookup time.

    Kept here rather than imported so this module's dependency graph stays
    self-contained: pokerbot.abstraction.cards → build.lookup → build.equity,
    not back through builder.
    """
    if len(board) not in (3, 4):
        raise ValueError(f"flop/turn boards only, got len={len(board)}")
    used = {*hole, *board}
    remaining = [c for c in range(52) if c not in used]
    histogram = np.zeros(bins, dtype=np.float32)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for _ in range(inner_samples):
        next_card = rng.choice(remaining)
        new_board = (*board, next_card)
        ehs = hand_strength(hole, new_board, num_samples=inner_samples, rng=rng)
        ehs_sq = ehs * ehs
        idx = min(int(np.searchsorted(bin_edges[1:], ehs_sq)), bins - 1)
        histogram[idx] += 1.0
    return (histogram / inner_samples).astype(np.float32, copy=False)


def query_features(
    canon_hole: tuple[int, int],
    canon_board: tuple[int, ...],
    street: Street,
    *,
    num_samples: int = 30,
    histogram_bins: int = 50,
    rng: random.Random | None = None,
    tiers_seed: int = 0xC0FFEE,
) -> npt.NDArray[np.float32]:
    """Compute the feature vector for a single (hole, board) query."""
    if rng is None:
        # Deterministic per-query rng. We use blake2b instead of Python's
        # built-in `hash()` because the latter randomises string hashes via
        # PYTHONHASHSEED — making bucket assignment differ across processes
        # for hands near a cluster boundary. blake2b is fixed-output, so
        # SimpleNLHEGame (training) and RuntimeAdapter (live) always agree.
        seed_bytes = hashlib.blake2b(
            bytes((*canon_hole, *canon_board)) + street.encode("ascii"),
            digest_size=8,
        ).digest()
        seed = int.from_bytes(seed_bytes, "little") & 0xFFFFFFFF
        rng = random.Random(seed)
    if street == "river":
        tiers = get_preflop_tiers(seed=tiers_seed)
        feats = np.zeros(len(tiers), dtype=np.float32)
        for i, tier in enumerate(tiers):
            feats[i] = hand_strength_vs_cluster(
                canon_hole, canon_board, tier, num_samples=num_samples, rng=rng
            )
        return feats
    return _ehs2_histogram_query(
        canon_hole, canon_board, bins=histogram_bins, inner_samples=num_samples, rng=rng
    )


def project_to_centroid(
    canon_hole: tuple[int, int],
    canon_board: tuple[int, ...],
    street: Street,
    centroids: npt.NDArray[np.float32],
    *,
    num_samples: int = 30,
    rng: random.Random | None = None,
) -> int:
    """Find the nearest bucket id for a hand not in the NPZ sample.

    For river (OCHS), L2 distance on 8-d equity vectors.
    For flop/turn, the centroids are stored in CDF space (built with
    `kmeans_cdf_emd`); query histogram → cumsum → L1 distance approximates
    1-D EMD.
    """
    feats = query_features(
        canon_hole,
        canon_board,
        street,
        num_samples=num_samples,
        histogram_bins=centroids.shape[1],
        rng=rng,
    )
    if street == "river":
        d = np.sum((centroids - feats) ** 2, axis=1)
    else:
        query_cdf = np.cumsum(feats).astype(np.float32)
        d = np.sum(np.abs(centroids - query_cdf), axis=1)
    return int(np.argmin(d))


__all__ = [
    "Street",
    "get_preflop_tiers",
    "project_to_centroid",
    "query_features",
]
