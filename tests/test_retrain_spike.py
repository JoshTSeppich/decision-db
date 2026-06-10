"""Retrain-mechanisms spike: aggression-bias sampling + visit-depth instrument.

These tests verify what the two pieces DO mechanically — the bias knob shifts the
opponent sampling distribution toward bet/raise, the bias=0 path is unchanged, and
the per-(street x facing-bet) visit counter accumulates correctly. They do NOT
assert that aggression bias increases late-street facing-aggression visitation —
the tiny A/B verify run showed it does the OPPOSITE (aggressive opponents end hands
earlier). See the spike report.
"""

from __future__ import annotations

import random

from pokerbot.abstraction.actions import ActionType
from pokerbot.training.config import DeepCFRConfig
from pokerbot.training.kuhn import KuhnPokerGame
from pokerbot.training.deepcfr import Trainer
from pokerbot.training.traversal import TraversalStats, _aggression_biased_weights

FOLD = int(ActionType.FOLD)
CHECK_CALL = int(ActionType.CHECK_CALL)
BET_66 = int(ActionType.BET_66)
ALL_IN = int(ActionType.ALL_IN)


# ── Piece 1: the aggression-bias sampling weights ──────────────────────────
def test_aggression_biased_weights_shifts_mass_to_bet() -> None:
    legal = [FOLD, CHECK_CALL, BET_66]          # one aggressive action (BET_66)
    on_policy = [0.5, 0.4, 0.1]                  # BET is least likely on-policy
    w = _aggression_biased_weights(legal, on_policy, bias=0.5)
    # mixture: 0.5*normalized_on_policy + 0.5*uniform(aggressive)
    assert abs(w[2] - (0.5 * 0.1 + 0.5 * 1.0)) < 1e-9   # 0.55
    assert w[2] > on_policy[2]                          # bet weight rose
    assert w[0] < on_policy[0] and w[1] < on_policy[1]  # fold/call fell
    assert abs(sum(w) - 1.0) < 1e-9


def test_aggression_biased_weights_full_bias_is_all_aggressive() -> None:
    legal = [FOLD, CHECK_CALL, BET_66, ALL_IN]
    w = _aggression_biased_weights(legal, [0.7, 0.2, 0.05, 0.05], bias=1.0)
    assert w[0] == 0.0 and w[1] == 0.0           # non-aggressive zeroed
    assert abs(w[2] - 0.5) < 1e-9 and abs(w[3] - 0.5) < 1e-9  # split across 2 aggressive


def test_aggression_biased_weights_noop_without_aggressive_action() -> None:
    # fold/check-only node: no aggressive action → bias falls back to on-policy
    legal = [FOLD, CHECK_CALL]
    w = _aggression_biased_weights(legal, [0.6, 0.4], bias=0.9)
    assert abs(w[0] - 0.6) < 1e-9 and abs(w[1] - 0.4) < 1e-9


# ── Piece 2: the per-region visit counter ──────────────────────────────────
def test_traversalstats_record_visit_per_region() -> None:
    s = TraversalStats()
    s.record_visit(street=1, facing_bet=1, infoset_key=b"k1")
    s.record_visit(street=1, facing_bet=1, infoset_key=b"k1")   # repeat → depth 2
    s.record_visit(street=1, facing_bet=1, infoset_key=b"k2")
    s.record_visit(street=1, facing_bet=0, infoset_key=b"k3")   # different role
    s.record_visit(street=3, facing_bet=1, infoset_key=b"k4")   # different street
    flop_facing = s.region_visits[(1, 1)]
    assert flop_facing[b"k1"] == 2 and flop_facing[b"k2"] == 1
    assert len(flop_facing) == 2                                 # 2 distinct infosets
    assert (1, 0) in s.region_visits and (3, 1) in s.region_visits
    assert len(s.region_visits[(1, 0)]) == 1


def test_default_config_has_knobs_off() -> None:
    c = DeepCFRConfig()
    assert c.opp_aggression_bias == 0.0
    assert c.coverage_instrument is False


def test_coverage_instrument_off_means_no_stats() -> None:
    trainer = Trainer(DeepCFRConfig(coverage_instrument=False), KuhnPokerGame())
    assert trainer._cov_stats is None


def test_instrument_on_accumulates_during_traversals() -> None:
    cfg = DeepCFRConfig(coverage_instrument=True, opp_aggression_bias=0.0, lbr_every=0)
    trainer = Trainer(cfg, KuhnPokerGame())
    assert trainer._cov_stats is not None
    trainer._cfr_iteration(1)
    # Kuhn has a single street (0); some decision nodes must have been recorded.
    assert trainer._cov_stats.region_visits
    assert sum(sum(c.values()) for c in trainer._cov_stats.region_visits.values()) > 0
