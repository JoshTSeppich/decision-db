"""Composes enumerate → features → cluster → save (Spec.html §A).

Three public entry points, one per postflop street:
    `build_flop_buckets(out_dir, cfg)`   — potential-aware k-means via turn EHS²
    `build_turn_buckets(out_dir, cfg)`   — potential-aware k-means via river EHS²
    `build_river_buckets(out_dir, cfg)`  — OCHS against 8 preflop equity tiers

All three write `buckets_<street>.npz` in the layout `AbstractionTables` expects:
    keys     : uint8 array of shape (N, K) — sorted-lex canonical class bytes
    buckets  : int32 array of shape (N,)   — bucket id for each class

A shared `centroids.npz` holds the k-means centroids for each street (for
debugging + a future nearest-bucket fallback in the runtime).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

from pokerbot.abstraction.build.clustering import (
    kmeans_cdf_emd,
    kmeans_euclidean,
)
from pokerbot.abstraction.build.enumerate import (
    CanonicalClass,
    sample_canonical_flops,
    sample_canonical_rivers,
    sample_canonical_turns,
)
from pokerbot.abstraction.build.equity import (
    hand_strength,
    hand_strength_vs_cluster,
)
from pokerbot.abstraction.build.preflop_tiers import (
    NUM_PREFLOP_TIERS,
    build_preflop_tiers,
)

_LOG = logging.getLogger("pokerbot.build")


# ────────── config ──────────


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Knobs the CLI exposes; tests use small values, production uses big ones."""

    num_buckets: int = 200  # postflop bucket count (spec §A)
    histogram_bins: int = 50  # 50-bin EHS² histogram (spec §A)
    samples_per_class: int = 1000  # MC samples driving hand_strength / OCHS
    inner_samples: int = 100  # nested MC for EHS² histogram (flop/turn)
    kmeans_iters: int = 20
    max_classes: int | None = None  # subsample canonical classes (smoke builds)
    preflop_tier_samples: int = 2000  # samples per hand when ranking preflop equity
    rng_seed: int = 0xC0FFEE
    progress_every: int = 1000  # log every N processed classes


# ────────── NPZ writers ──────────


def _canonical_to_bytes(canon: CanonicalClass) -> bytes:
    hole, board = canon
    return bytes((*hole, *board))


def _write_buckets_npz(
    out_path: Path,
    classes: list[CanonicalClass],
    assignments: npt.NDArray[np.int32],
    centroids: npt.NDArray[np.float32],
) -> None:
    if len(classes) != len(assignments):
        raise ValueError(f"size mismatch: {len(classes)} classes vs {len(assignments)} assigns")
    key_bytes = [_canonical_to_bytes(c) for c in classes]
    keys = np.array([list(b) for b in key_bytes], dtype=np.uint8)
    buckets = np.asarray(assignments, dtype=np.int32)
    # Ensure sorted-by-key order so AbstractionTables._binary_search_rows can find them.
    order = np.lexsort(keys.T[::-1])  # row-wise lex sort
    keys = keys[order]
    buckets = buckets[order]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, keys=keys, buckets=buckets, centroids=centroids)


# ────────── feature extractors ──────────


def _ehs2_histogram(
    hole: tuple[int, int],
    board: tuple[int, ...],
    *,
    bins: int,
    inner_samples: int,
    rng: random.Random,
) -> npt.NDArray[np.float32]:
    """For potential-aware bucketing: sample many future "next-street" deals
    and record each one's EHS². The resulting histogram captures hand
    *potential*, not just current strength.
    """
    if len(board) not in (3, 4):
        raise ValueError(f"_ehs2_histogram supports flop or turn boards only, got {len(board)}")
    used = {*hole, *board}
    remaining = [c for c in range(52) if c not in used]
    histogram = np.zeros(bins, dtype=np.float32)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for _ in range(inner_samples):
        # Pick one next-street card uniformly at random
        next_card = rng.choice(remaining)
        new_board = (*board, next_card)
        ehs = hand_strength(hole, new_board, num_samples=inner_samples, rng=rng)
        ehs_sq = ehs * ehs
        idx = min(int(np.searchsorted(bin_edges[1:], ehs_sq)), bins - 1)
        histogram[idx] += 1.0
    histogram /= inner_samples
    return histogram


def _ochs_features(
    hole: tuple[int, int],
    river_board: tuple[int, ...],
    *,
    tiers: tuple[tuple[tuple[int, int], ...], ...],
    samples_per_tier: int,
    rng: random.Random,
) -> npt.NDArray[np.float32]:
    """8-d feature vector: equity vs each preflop tier of opponent hands."""
    if len(river_board) != 5:
        raise ValueError(f"OCHS requires a 5-card board, got {len(river_board)}")
    out = np.zeros(len(tiers), dtype=np.float32)
    for i, tier in enumerate(tiers):
        out[i] = hand_strength_vs_cluster(
            hole, river_board, tier, num_samples=samples_per_tier, rng=rng
        )
    return out


# ────────── per-street drivers ──────────


def _compute_features(
    classes: list[CanonicalClass],
    feature_fn: Callable[[CanonicalClass, random.Random], npt.NDArray[np.float32]],
    cfg: BuildConfig,
    *,
    street: str,
) -> npt.NDArray[np.float32]:
    n = len(classes)
    if n == 0:
        raise ValueError(f"no canonical classes for street={street}")
    rng = random.Random(cfg.rng_seed)
    first = feature_fn(classes[0], rng)
    feat_dim = first.shape[0]
    features = np.zeros((n, feat_dim), dtype=np.float32)
    features[0] = first
    t0 = time.perf_counter()
    for i in range(1, n):
        features[i] = feature_fn(classes[i], rng)
        if cfg.progress_every > 0 and (i + 1) % cfg.progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta_s = (n - i - 1) / max(rate, 1e-6)
            _LOG.info(
                "[%s] %d/%d classes (%.0f/s, ETA %.1fmin)",
                street,
                i + 1,
                n,
                rate,
                eta_s / 60.0,
            )
    return features


def build_flop_buckets(out_dir: Path, cfg: BuildConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = cfg.max_classes if cfg.max_classes is not None else 200_000
    _LOG.info("[flop] sampling %d canonical classes", n)
    classes = sample_canonical_flops(n, rng_seed=cfg.rng_seed)
    _LOG.info("[flop] computing EHS² histograms (%d MC samples each)", cfg.inner_samples)

    def _feat(canon: CanonicalClass, rng: random.Random) -> npt.NDArray[np.float32]:
        return _ehs2_histogram(
            canon[0], canon[1], bins=cfg.histogram_bins, inner_samples=cfg.inner_samples, rng=rng
        )

    features = _compute_features(classes, _feat, cfg, street="flop")
    _LOG.info("[flop] k-means with sorted-EMD, k=%d", cfg.num_buckets)
    assignments, centroids = kmeans_cdf_emd(
        features, k=cfg.num_buckets, iters=cfg.kmeans_iters, seed=cfg.rng_seed
    )
    path = out_dir / "buckets_flop.npz"
    _write_buckets_npz(path, classes, assignments, centroids)
    _LOG.info("[flop] wrote %s (%d classes)", path, len(classes))
    return path


def build_turn_buckets(out_dir: Path, cfg: BuildConfig) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = cfg.max_classes if cfg.max_classes is not None else 200_000
    _LOG.info("[turn] sampling %d canonical classes", n)
    classes = sample_canonical_turns(n, rng_seed=cfg.rng_seed)
    _LOG.info("[turn] computing EHS² histograms (%d MC samples each)", cfg.inner_samples)

    def _feat(canon: CanonicalClass, rng: random.Random) -> npt.NDArray[np.float32]:
        return _ehs2_histogram(
            canon[0], canon[1], bins=cfg.histogram_bins, inner_samples=cfg.inner_samples, rng=rng
        )

    features = _compute_features(classes, _feat, cfg, street="turn")
    _LOG.info("[turn] k-means with sorted-EMD, k=%d", cfg.num_buckets)
    assignments, centroids = kmeans_cdf_emd(
        features, k=cfg.num_buckets, iters=cfg.kmeans_iters, seed=cfg.rng_seed
    )
    path = out_dir / "buckets_turn.npz"
    _write_buckets_npz(path, classes, assignments, centroids)
    _LOG.info("[turn] wrote %s (%d classes)", path, len(classes))
    return path


def build_river_buckets(
    out_dir: Path,
    cfg: BuildConfig,
    *,
    classes: list[CanonicalClass] | None = None,
) -> Path:
    """Build river bucketing NPZ.

    Set `classes` to override sampling — useful for tests that need specific
    canonical (hole, board) states present in the output NPZ.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info("[river] computing %d preflop equity tiers", NUM_PREFLOP_TIERS)
    tiers = build_preflop_tiers(samples_per_hand=cfg.preflop_tier_samples, rng_seed=cfg.rng_seed)
    if classes is None:
        n = cfg.max_classes if cfg.max_classes is not None else 200_000
        _LOG.info("[river] sampling %d canonical classes", n)
        classes = sample_canonical_rivers(n, rng_seed=cfg.rng_seed)
    else:
        _LOG.info("[river] using %d caller-supplied canonical classes", len(classes))
    _LOG.info(
        "[river] computing OCHS features (8 tiers x %d MC samples each)",
        cfg.samples_per_class,
    )

    def _feat(canon: CanonicalClass, rng: random.Random) -> npt.NDArray[np.float32]:
        return _ochs_features(
            canon[0],
            canon[1],
            tiers=tiers,
            samples_per_tier=cfg.samples_per_class,
            rng=rng,
        )

    features = _compute_features(classes, _feat, cfg, street="river")
    _LOG.info("[river] k-means with L2, k=%d", cfg.num_buckets)
    assignments, centroids = kmeans_euclidean(
        features, k=cfg.num_buckets, iters=cfg.kmeans_iters, seed=cfg.rng_seed
    )
    path = out_dir / "buckets_river.npz"
    _write_buckets_npz(path, classes, assignments, centroids)
    _LOG.info("[river] wrote %s (%d classes)", path, len(classes))
    return path


__all__ = [
    "BuildConfig",
    "build_flop_buckets",
    "build_river_buckets",
    "build_turn_buckets",
]
