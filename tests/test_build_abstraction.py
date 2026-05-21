"""Tests for GAP 2 — the real `build-abstraction` pipeline.

Unit tests run at small scale (50 samples, ~30 classes) so the whole file
completes in <60s. The spec acceptance test (`test_river_ochs_monotone` in
test_abstraction_cards.py) is marked `slow` and unskipped after the user
runs `pokerbot build-abstraction --samples 50 --max-classes 5000 --out …`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pokerbot.abstraction import AbstractionTables, parse_card, parse_hand
from pokerbot.abstraction.build.builder import (
    BuildConfig,
    build_river_buckets,
    build_turn_buckets,
)
from pokerbot.abstraction.build.cards import card_int_to_str, cards_to_strs
from pokerbot.abstraction.build.clustering import kmeans_cdf_emd, kmeans_euclidean
from pokerbot.abstraction.build.enumerate import (
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

if TYPE_CHECKING:
    from pathlib import Path


# ───────── card bridge ─────────


def test_card_int_to_str_roundtrip() -> None:
    for c in range(52):
        s = card_int_to_str(c)
        assert parse_card(s) == c


def test_card_int_to_str_examples() -> None:
    assert card_int_to_str(0) == "2c"
    assert card_int_to_str(51) == "As"
    assert card_int_to_str(parse_card("Kh")) == "Kh"


def test_cards_to_strs_keeps_order() -> None:
    hole = (parse_card("As"), parse_card("Kh"))
    assert cards_to_strs(hole) == ["As", "Kh"]


# ───────── equity ─────────


def test_hand_strength_aa_dominates_72o() -> None:
    rng = random.Random(0)
    aa = hand_strength(parse_hand("As Ah"), (), num_samples=300, rng=rng)  # type: ignore[arg-type]
    seven_two = hand_strength(parse_hand("7c 2d"), (), num_samples=300, rng=rng)  # type: ignore[arg-type]
    assert aa > 0.7, f"AA equity {aa:.3f} should be ≥ 0.7 (canonical ~85%)"
    assert seven_two < 0.4, f"72o equity {seven_two:.3f} should be ≤ 0.4"
    assert aa > seven_two + 0.3


def test_hand_strength_nut_flush_vs_random() -> None:
    # Ah Kh on Qh-Jh-Th board = royal flush. Should be ~1.0 vs random opp.
    rng = random.Random(1)
    eq = hand_strength(
        (parse_card("Ah"), parse_card("Kh")),
        (parse_card("Qh"), parse_card("Jh"), parse_card("Th")),
        num_samples=200,
        rng=rng,
    )
    assert eq > 0.97, f"royal flush eq {eq:.3f} should be ~1.0"


def test_hand_strength_vs_cluster_basic() -> None:
    rng = random.Random(2)
    # Cluster of premium pairs
    premiums = ((parse_card("As"), parse_card("Ad")), (parse_card("Ks"), parse_card("Kd")))
    # Hero is 2c 3c — should lose to premium pairs preflop
    eq = hand_strength_vs_cluster(
        (parse_card("2c"), parse_card("3c")),
        (),
        cluster=premiums,
        num_samples=200,
        rng=rng,
    )
    assert eq < 0.3


# ───────── preflop tiers ─────────


@pytest.mark.parametrize("samples", [200])
def test_preflop_tiers_partition_169_into_8(samples: int) -> None:
    tiers = build_preflop_tiers(samples_per_hand=samples)
    assert len(tiers) == NUM_PREFLOP_TIERS
    total = sum(len(t) for t in tiers)
    assert total == 169, f"expected 169 canonical preflops, got {total}"
    # No overlap
    seen = set()
    for tier in tiers:
        for h in tier:
            assert h not in seen, f"duplicate hand {h}"
            seen.add(h)


def test_preflop_tiers_strong_in_top_tier() -> None:
    tiers = build_preflop_tiers(samples_per_hand=200)
    top_tier = tiers[0]
    # AA representative is (rank=12, suits 0 and 1) = (48, 49)
    aa_canonical = (12 * 4 + 0, 12 * 4 + 1)
    assert aa_canonical in top_tier, "AA must be in top preflop tier"


# ───────── clustering ─────────


def test_kmeans_euclidean_separates_two_clusters() -> None:
    rng = np.random.default_rng(0)
    # 2 well-separated clusters in 4-d
    a = rng.normal(loc=0.0, scale=0.1, size=(50, 4)).astype(np.float32)
    b = rng.normal(loc=5.0, scale=0.1, size=(50, 4)).astype(np.float32)
    features = np.concatenate([a, b], axis=0)
    assignments, centroids = kmeans_euclidean(features, k=2, iters=10, seed=0)
    assert centroids.shape == (2, 4)
    first_half = set(assignments[:50])
    second_half = set(assignments[50:])
    assert first_half.isdisjoint(second_half), "clusters got mixed"


def test_kmeans_cdf_emd_separates_low_vs_high_bin_peaks() -> None:
    rng = np.random.default_rng(1)
    # Two "histogram" populations: one peaked at low bins, one at high bins
    dim = 10
    low = np.zeros((30, dim), dtype=np.float32)
    low[:, :3] = rng.dirichlet(np.ones(3), size=30).astype(np.float32)
    high = np.zeros((30, dim), dtype=np.float32)
    high[:, -3:] = rng.dirichlet(np.ones(3), size=30).astype(np.float32)
    features = np.concatenate([low, high], axis=0)
    assignments, _ = kmeans_cdf_emd(features, k=2, iters=10, seed=0)
    first = set(assignments[:30])
    second = set(assignments[30:])
    assert first.isdisjoint(second)


# ───────── enumeration ─────────


def test_sample_canonical_flops_yields_distinct_classes() -> None:
    classes = sample_canonical_flops(50, rng_seed=0)
    assert len(classes) == 50
    assert len(set(classes)) == 50


def test_sample_canonical_turns_distinct() -> None:
    classes = sample_canonical_turns(40, rng_seed=0)
    assert len(set(classes)) == 40


def test_sample_canonical_rivers_distinct() -> None:
    classes = sample_canonical_rivers(40, rng_seed=0)
    assert len(set(classes)) == 40
    # All rivers have 5-card boards
    for _, board in classes:
        assert len(board) == 5


# ───────── end-to-end smoke (small) ─────────


@pytest.mark.slow
def test_build_river_buckets_smoke(tmp_path: Path) -> None:
    """Tiny end-to-end build. Slow because OCHS Monte Carlo is the bottleneck."""
    cfg = BuildConfig(
        num_buckets=4,
        samples_per_class=30,
        kmeans_iters=5,
        max_classes=60,
        preflop_tier_samples=200,
        rng_seed=0,
        progress_every=0,  # silence logging in tests
    )
    path = build_river_buckets(tmp_path, cfg)
    assert path.exists()
    with np.load(path) as npz:
        keys = npz["keys"]
        buckets = npz["buckets"]
    assert keys.shape[0] == 60
    assert keys.shape[1] == 2 + 5  # hole + 5-card board
    assert buckets.shape == (60,)
    assert buckets.min() >= 0
    assert buckets.max() < 4
    # AbstractionTables should now be able to load this file:
    tables = AbstractionTables(path=tmp_path)
    assert "river" in tables.loaded_streets


@pytest.mark.slow
def test_build_turn_buckets_smoke(tmp_path: Path) -> None:
    cfg = BuildConfig(
        num_buckets=3,
        histogram_bins=8,
        samples_per_class=20,
        inner_samples=15,
        kmeans_iters=5,
        max_classes=30,
        rng_seed=0,
        progress_every=0,
    )
    path = build_turn_buckets(tmp_path, cfg)
    assert path.exists()
    with np.load(path) as npz:
        keys, buckets = npz["keys"], npz["buckets"]
    assert keys.shape[0] == 30
    assert keys.shape[1] == 2 + 4
    assert buckets.min() >= 0 and buckets.max() < 3


# ───────── CLI smoke ─────────


def test_cli_build_abstraction_help_lists_args() -> None:
    """`pokerbot build-abstraction --help` should exit 0 and mention key knobs."""
    from pokerbot.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["build-abstraction", "--help"])
    assert exc.value.code == 0
