"""Spec.html §A tests for the card abstraction module."""

from __future__ import annotations

import random
import time

import pytest

from pokerbot.abstraction import (
    AbstractionTables,
    bucket_hand,
    canonical_hand,
    num_buckets,
    parse_card,
    parse_hand,
)

# ─────────── test_preflop_169 ───────────


def test_preflop_169_all_holes_bucketed() -> None:
    """Every (52 choose 2) = 1326 starting hand maps to a bucket in [0, 169)."""
    buckets = set()
    for c1 in range(52):
        for c2 in range(c1 + 1, 52):
            b = bucket_hand((c1, c2), (), "preflop")
            assert 0 <= b < 169
            buckets.add(b)
    assert len(buckets) == 169, f"expected all 169 buckets used, got {len(buckets)}"


def test_preflop_aks_vs_ako_distinct() -> None:
    aks = bucket_hand(parse_hand("As Ks"), (), "preflop")  # type: ignore[arg-type]
    ako = bucket_hand(parse_hand("As Kh"), (), "preflop")  # type: ignore[arg-type]
    assert aks != ako


def test_preflop_aks_suit_rotations_collide() -> None:
    holes = [("As", "Ks"), ("Ah", "Kh"), ("Ad", "Kd"), ("Ac", "Kc")]
    buckets = {bucket_hand((parse_card(a), parse_card(b)), (), "preflop") for a, b in holes}
    assert len(buckets) == 1, f"AKs rotations should collide, got {buckets}"


# ─────────── test_isomorphism (§A, postflop) ───────────


def test_isomorphism_canonical_hand_equal() -> None:
    """Suit-rotated flops produce identical canonical (hole, board)."""
    h1 = (parse_card("As"), parse_card("Kh"))
    b1 = (parse_card("2c"), parse_card("7d"), parse_card("Jh"))
    h2 = (parse_card("Ah"), parse_card("Ks"))
    b2 = (parse_card("2d"), parse_card("7c"), parse_card("Js"))
    assert canonical_hand(h1, b1) == canonical_hand(h2, b2)


def test_isomorphism_bucket_hand_equal() -> None:
    """Suit-rotated flops bucket identically (placeholder respects canonicalization)."""
    h1 = (parse_card("As"), parse_card("Kh"))
    b1 = (parse_card("2c"), parse_card("7d"), parse_card("Jh"))
    h2 = (parse_card("Ah"), parse_card("Ks"))
    b2 = (parse_card("2d"), parse_card("7c"), parse_card("Js"))
    assert bucket_hand(h1, b1, "flop") == bucket_hand(h2, b2, "flop")


# ─────────── test_bucket_count ───────────


def test_bucket_count_per_street() -> None:
    assert num_buckets("preflop") == 169
    assert num_buckets("flop") == 200
    assert num_buckets("turn") == 200
    assert num_buckets("river") == 200


# ─────────── test_determinism ───────────


def test_determinism_10k_random_hands() -> None:
    rng = random.Random(0xDECA1)
    samples: list[tuple[tuple[int, int], tuple[int, ...], str]] = []
    deck = list(range(52))
    for _ in range(10_000):
        rng.shuffle(deck)
        h = (deck[0], deck[1])
        # rotate through streets
        n_board = rng.choice([0, 3, 4, 5])
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[n_board]
        b = tuple(deck[2 : 2 + n_board])
        samples.append((h, b, street))
    first = [bucket_hand(h, b, s) for h, b, s in samples]  # type: ignore[arg-type]
    second = [bucket_hand(h, b, s) for h, b, s in samples]  # type: ignore[arg-type]
    assert first == second


# ─────────── test_speed ───────────


def test_speed_10k_lookups_under_500ms() -> None:
    rng = random.Random(0xCAFE)
    deck = list(range(52))
    hands: list[tuple[tuple[int, int], tuple[int, ...], str]] = []
    for _ in range(10_000):
        rng.shuffle(deck)
        n_board = rng.choice([0, 3, 4, 5])
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[n_board]
        hands.append(((deck[0], deck[1]), tuple(deck[2 : 2 + n_board]), street))
    t0 = time.perf_counter()
    for h, b, s in hands:
        bucket_hand(h, b, s)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.5, f"10k lookups took {elapsed:.3f}s (>0.5s budget)"


# ─────────── test_river_ochs_monotone (deferred until real OCHS NPZ ships) ───────────


@pytest.mark.slow
def test_river_ochs_monotone_nut_top_pair_bottom(tmp_path: object) -> None:
    """Spec §A acceptance: nut hand lands in a top-quartile OCHS bucket; bottom
    pair lands in the bottom quartile.

    This is the real cross-cutting test: run a small `build_river_buckets`,
    then look up two curated hands via `AbstractionTables`, then rank buckets
    by their centroid's mean OCHS equity and check the assignment.
    """
    pytest.importorskip("phevaluator", reason="real OCHS bucketing needs phevaluator")
    import numpy as np

    from pokerbot.abstraction.build import BuildConfig, build_river_buckets
    from pokerbot.abstraction.build.enumerate import sample_canonical_rivers
    from pokerbot.abstraction.cards import canonical_hand

    # Curated targets:
    #   nut:    Ah Kh on Qh-Jh-Th-2c-3c (royal flush)
    #   bot:    2c 3d on As-Ks-Qs-7h-8d (no pair, no draw → near-zero equity)
    nut_hole = (parse_card("Ah"), parse_card("Kh"))
    nut_board = (
        parse_card("Qh"),
        parse_card("Jh"),
        parse_card("Th"),
        parse_card("2c"),
        parse_card("3c"),
    )
    bot_hole = (parse_card("2c"), parse_card("3d"))
    bot_board = (
        parse_card("As"),
        parse_card("Ks"),
        parse_card("Qs"),
        parse_card("7h"),
        parse_card("8d"),
    )
    nut_canon = canonical_hand(nut_hole, nut_board)
    bot_canon = canonical_hand(bot_hole, bot_board)

    # Mix the targets into a random sample so the clustering has variety.
    sampled = sample_canonical_rivers(200, rng_seed=2026)
    classes = sorted({*sampled, nut_canon, bot_canon})

    cfg = BuildConfig(
        num_buckets=8,  # coarse → quartile = 2 buckets per quartile
        samples_per_class=80,  # MC budget per OCHS feature
        preflop_tier_samples=200,
        kmeans_iters=10,
        rng_seed=2026,
        progress_every=0,
    )
    npz_path = build_river_buckets(tmp_path, cfg, classes=classes)  # type: ignore[arg-type]

    tables = AbstractionTables(path=tmp_path)  # type: ignore[arg-type]
    assert "river" in tables.loaded_streets
    nut_bucket = tables.lookup(nut_hole, nut_board, "river")
    bot_bucket = tables.lookup(bot_hole, bot_board, "river")

    # Rank buckets by their centroid's mean OCHS equity. Higher = stronger.
    with np.load(npz_path) as npz:
        centroids = npz["centroids"]  # shape (k, 8)
    mean_eq = centroids.mean(axis=1)
    ranking = mean_eq.argsort()  # bucket ids sorted weakest → strongest
    rank_of = {int(b): i for i, b in enumerate(ranking)}

    k = cfg.num_buckets
    top_quartile = set(int(b) for b in ranking[(3 * k) // 4 :])
    bot_quartile = set(int(b) for b in ranking[: k // 4 if k >= 4 else 1])

    assert nut_bucket in top_quartile, (
        f"nut hand bucket {nut_bucket} (rank {rank_of[nut_bucket]}/{k}) not in top quartile"
    )
    assert bot_bucket in bot_quartile, (
        f"bottom-pair bucket {bot_bucket} (rank {rank_of[bot_bucket]}/{k}) not in bottom quartile"
    )


# ─────────── extra sanity ───────────


def test_validate_inputs_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        bucket_hand((parse_card("As"), parse_card("As")), (), "preflop")


def test_validate_inputs_rejects_wrong_board_length() -> None:
    with pytest.raises(ValueError, match="board"):
        bucket_hand(
            (parse_card("As"), parse_card("Kh")),
            (parse_card("2c"),),
            "flop",
        )


def test_abstraction_tables_falls_back_without_npz(tmp_path: object) -> None:
    """Empty directory → AbstractionTables.lookup matches bucket_hand placeholder."""
    tables = AbstractionTables(path=tmp_path)  # type: ignore[arg-type]
    assert tables.loaded_streets == ()
    h = (parse_card("As"), parse_card("Kh"))
    b = (parse_card("2c"), parse_card("7d"), parse_card("Jh"))
    assert tables.lookup(h, b, "flop") == bucket_hand(h, b, "flop")
