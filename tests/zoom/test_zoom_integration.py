"""Integration test: drive L2 (opponent model) + L3 (range tracker) through the
REAL `pokerbot` abstraction.

The standalone proof harnesses inject a synthetic `chen_bucket`. Here we inject
the real `AbstractionTables` bucketing (via `make_bucket_fn`) and assert the same
qualitative result the standalone tests prove: when a modeled tight value-raiser
makes a `RAISE_2_5X`, the inferred range concentrates onto strong holdings.

Preflop buckets are lossless/deterministic without NPZ artifacts, so this runs
against the real abstraction with no on-disk build. A second test exercises the
phevaluator-backed `posterior_range_equity` readout with a board present.
"""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from pokerbot.abstraction import AbstractionTables
from zoom.abstraction_bridge import make_bucket_fn, posterior_range_equity
from zoom.opponent_model import DirichletOpponentModel, coarse_key, make_opponent_strategy
from zoom.range_tracker import PublicState, RangeTracker, chen_score, parse_cards

RAISE = "RAISE_2_5X"


def _teach_tight_raiser(bucket_fn, *, raise_threshold: float = 12.0, reps: int = 8):
    """Teach a modeled opponent a strength-dependent preflop strategy keyed on the
    REAL preflop bucket of each hand: raise strong hands, fold weak ones. All
    combos in a canonical preflop class share both a bucket and a Chen score, so
    the per-bucket action assignment is consistent."""
    model = DirichletOpponentModel()
    model.set_archetype("villain", "TAG")
    for combo in combinations(range(52), 2):
        bucket = bucket_fn(combo, ())
        key = coarse_key("preflop", position=2, card_bucket=bucket, to_call_bb=1.0)
        action = RAISE if chen_score(combo) >= raise_threshold else "FOLD"
        for _ in range(reps):
            model.observe("villain", key, action)
    return model


def test_real_abstraction_tight_raise_concentrates_range():
    """Same behavior as the standalone end-to-end check, but bucketing comes from
    the real AbstractionTables instead of the synthetic chen_bucket."""
    tables = AbstractionTables()  # no NPZ path → real, deterministic preflop buckets
    bucket_fn = make_bucket_fn(tables)
    model = _teach_tight_raiser(bucket_fn)
    opp_strat = make_opponent_strategy("villain", model, bucket_fn)

    rt = RangeTracker()
    st = PublicState(board=(), street="preflop", position=2, to_call_bb=1.0, facing="raise")
    strength_before, eff_before = rt.range_strength(), rt.effective_combos()

    rt.update(RAISE, st, opp_strat)

    strength_after, eff_after = rt.range_strength(), rt.effective_combos()
    assert strength_after > strength_before, (
        f"range strength should rise after a modeled raise: "
        f"{strength_before:.2f} -> {strength_after:.2f}"
    )
    assert eff_after < eff_before, (
        f"range should tighten (effective combos fall): {eff_before:.0f} -> {eff_after:.0f}"
    )

    post = rt.posterior()
    premium = sum(v for c, v in post.items() if chen_score(c) >= 14)
    trash = sum(v for c, v in post.items() if chen_score(c) <= 4)
    assert premium > trash, f"premium mass {premium:.3f} should exceed trash mass {trash:.3f}"


@pytest.mark.needs_train_deps
def test_real_equity_readout_reflects_range_concentration():
    """The phevaluator-backed posterior_range_equity readout (board present) ranks
    a concentrated (post-raise) range above a uniform range on the same flop."""
    pytest.importorskip("phevaluator", reason="real-equity readout needs the [train] extra")

    tables = AbstractionTables()
    bucket_fn = make_bucket_fn(tables)
    model = _teach_tight_raiser(bucket_fn)
    opp_strat = make_opponent_strategy("villain", model, bucket_fn)

    flop = tuple(parse_cards("2c 7d Ts"))  # low rainbow; does not block premiums

    concentrated = RangeTracker()
    concentrated.update(
        RAISE, PublicState(board=(), street="preflop", position=2, to_call_bb=1.0), opp_strat
    )
    concentrated.reveal_board(flop)

    uniform = RangeTracker()
    uniform.reveal_board(flop)

    eq_conc = posterior_range_equity(
        concentrated.posterior(), flop, num_samples=40, rng=random.Random(0)
    )
    eq_unif = posterior_range_equity(
        uniform.posterior(), flop, num_samples=40, rng=random.Random(0)
    )

    # A uniform range averages ~0.5 equity vs a random hand; a premium-weighted
    # range is clearly stronger. Margin is conservative vs the expected ~0.15 gap.
    assert eq_conc > eq_unif + 0.03, (
        f"concentrated range equity {eq_conc:.3f} should exceed uniform {eq_unif:.3f}"
    )
