"""Tests for OpponentStats / OpponentStatsTracker (Cairn 4)."""

from __future__ import annotations

from pokerbot.abstraction import ActionType
from pokerbot.opponent.stats import (
    ObservedAction,
    OpponentStats,
    OpponentStatsTracker,
)


def test_empty_stats_returns_zero() -> None:
    s = OpponentStats()
    assert s.hands_observed == 0
    # All rates default to 0 (numerator 0, denominator max(1, …) = 1).
    assert s.vpip() == 0.0
    assert s.pfr() == 0.0
    assert s.af() == 0.0
    assert s.three_bet() == 0.0
    assert s.cbet() == 0.0
    assert s.fold_to_cbet() == 0.0


def test_vpip_increments_on_preflop_call_or_raise() -> None:
    """VPIP counts hands with voluntary preflop action; BB's free check does not."""
    tracker = OpponentStatsTracker()

    # Hand 1: BB checks free option (involuntary) → VPIP=no
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.CHECK_CALL,
                to_call=0,
                voluntary_preflop=False,
            )
        ],
    )
    # Hand 2: limp (voluntary call) → VPIP=yes
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.CHECK_CALL,
                to_call=10,
                voluntary_preflop=True,
            )
        ],
    )
    # Hand 3: raise → VPIP=yes
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.RAISE_2_5X,
                to_call=10,
                voluntary_preflop=True,
            )
        ],
    )

    s = tracker.get("p1")
    assert s.hands_observed == 3
    assert s.preflop_voluntary_actions == 2
    # vpip = 2/3 ≈ 0.667
    assert abs(s.vpip() - (2 / 3)) < 1e-9


def test_pfr_increments_only_on_raise() -> None:
    tracker = OpponentStatsTracker()
    # Voluntary call → VPIP yes, PFR no
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.CHECK_CALL,
                to_call=10,
                voluntary_preflop=True,
            )
        ],
    )
    # Raise → both VPIP and PFR
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.RAISE_2_5X,
                to_call=10,
                voluntary_preflop=True,
            )
        ],
    )
    # All-in preflop also counts as PFR (aggressive)
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.ALL_IN,
                to_call=10,
                voluntary_preflop=True,
            )
        ],
    )
    # Fold → neither
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.FOLD,
                to_call=10,
                voluntary_preflop=False,
            )
        ],
    )

    s = tracker.get("p1")
    assert s.hands_observed == 4
    assert s.preflop_voluntary_actions == 3
    assert s.preflop_raises == 2
    assert s.pfr() == 0.5  # 2 of 4 hands


def test_af_postflop_only() -> None:
    """Preflop calling/raising must NOT touch AF counters; AF is postflop only."""
    tracker = OpponentStatsTracker()
    # 5 preflop voluntary calls — none should affect AF
    for _ in range(5):
        tracker.update_from_hand(
            "p1",
            [
                ObservedAction(
                    street=0,
                    action_type=ActionType.CHECK_CALL,
                    to_call=10,
                    voluntary_preflop=True,
                )
            ],
        )
    s = tracker.get("p1")
    assert s.postflop_bets == 0
    assert s.postflop_raises == 0
    assert s.postflop_calls == 0
    # max(1, 0) denominator means AF = 0/1 = 0
    assert s.af() == 0.0

    # Add a postflop bet (to_call=0) → AF numerator increments
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=1,
                action_type=ActionType.BET_66,
                to_call=0,
            )
        ],
    )
    s = tracker.get("p1")
    assert s.postflop_bets == 1
    assert s.af() == 1.0  # 1 bet / max(1, 0 calls) = 1

    # Add a postflop call (CHECK_CALL with to_call>0)
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=1,
                action_type=ActionType.CHECK_CALL,
                to_call=50,
            )
        ],
    )
    s = tracker.get("p1")
    assert s.postflop_calls == 1
    assert s.af() == 1.0  # (1 bet + 0 raise) / 1 call


def test_cbet_opportunity_requires_preflop_aggressor() -> None:
    """Only the preflop raiser (first to act on flop) has a cbet opportunity."""
    tracker = OpponentStatsTracker()

    # Player WAS the PF aggressor, first to act on flop, bets → cbet opportunity + cbet
    tracker.update_from_hand(
        "aggressor",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.RAISE_2_5X,
                to_call=10,
                voluntary_preflop=True,
            ),
            ObservedAction(
                street=1,
                action_type=ActionType.BET_66,
                to_call=0,
                is_cbet_opportunity=True,
            ),
        ],
    )
    s = tracker.get("aggressor")
    assert s.cbet_opportunities == 1
    assert s.cbets == 1

    # Different player — NOT the PF aggressor; their flop bet is not a cbet
    tracker.update_from_hand(
        "non_aggressor",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.CHECK_CALL,
                to_call=10,
                voluntary_preflop=True,
            ),
            ObservedAction(
                street=1,
                action_type=ActionType.BET_66,
                to_call=0,
                is_cbet_opportunity=False,
            ),
        ],
    )
    s2 = tracker.get("non_aggressor")
    assert s2.cbet_opportunities == 0
    assert s2.cbets == 0

    # PF aggressor who CHECKS the flop has an opportunity but didn't cbet
    tracker.update_from_hand(
        "aggressor2",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.RAISE_2_5X,
                to_call=10,
                voluntary_preflop=True,
            ),
            ObservedAction(
                street=1,
                action_type=ActionType.CHECK_CALL,
                to_call=0,
                is_cbet_opportunity=True,
            ),
        ],
    )
    s3 = tracker.get("aggressor2")
    assert s3.cbet_opportunities == 1
    assert s3.cbets == 0


def test_fold_to_cbet_requires_facing_cbet() -> None:
    tracker = OpponentStatsTracker()

    # Faced a cbet, folded → folds_to_cbet
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.CHECK_CALL,
                to_call=10,
                voluntary_preflop=True,
            ),
            ObservedAction(
                street=1,
                action_type=ActionType.FOLD,
                to_call=50,
                is_facing_cbet=True,
            ),
        ],
    )
    s = tracker.get("p1")
    assert s.faced_cbet == 1
    assert s.folds_to_cbet == 1

    # Faced a cbet, called → faced_cbet but no fold
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.CHECK_CALL,
                to_call=10,
                voluntary_preflop=True,
            ),
            ObservedAction(
                street=1,
                action_type=ActionType.CHECK_CALL,
                to_call=50,
                is_facing_cbet=True,
            ),
        ],
    )
    s = tracker.get("p1")
    assert s.faced_cbet == 2
    assert s.folds_to_cbet == 1
    assert s.fold_to_cbet() == 0.5

    # Postflop fold with NO is_facing_cbet flag → not counted
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=2,
                action_type=ActionType.FOLD,
                to_call=100,
                is_facing_cbet=False,
            ),
        ],
    )
    s = tracker.get("p1")
    assert s.faced_cbet == 2
    assert s.folds_to_cbet == 1


def test_multiple_opponents_tracked_separately() -> None:
    tracker = OpponentStatsTracker()
    tracker.update_from_hand(
        "p1",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.RAISE_2_5X,
                to_call=10,
                voluntary_preflop=True,
            )
        ],
    )
    tracker.update_from_hand(
        "p2",
        [
            ObservedAction(
                street=0,
                action_type=ActionType.FOLD,
                to_call=10,
                voluntary_preflop=False,
            )
        ],
    )
    s1 = tracker.get("p1")
    s2 = tracker.get("p2")
    assert s1.hands_observed == 1
    assert s2.hands_observed == 1
    assert s1.preflop_raises == 1
    assert s2.preflop_raises == 0
    assert s1.preflop_voluntary_actions == 1
    assert s2.preflop_voluntary_actions == 0
    # Snapshot returns a copy keyed correctly
    snap = tracker.opponents()
    assert set(snap.keys()) == {"p1", "p2"}


def test_stats_persist_across_hands() -> None:
    tracker = OpponentStatsTracker()
    # 10 hands of preflop raises
    for _ in range(10):
        tracker.update_from_hand(
            "p1",
            [
                ObservedAction(
                    street=0,
                    action_type=ActionType.RAISE_2_5X,
                    to_call=10,
                    voluntary_preflop=True,
                ),
                # 3-bet opportunity test: simulate two prior raises in hand
                ObservedAction(
                    street=0,
                    action_type=ActionType.RAISE_3_5X,
                    to_call=30,
                    voluntary_preflop=True,
                    pf_raises_before=1,
                ),
                ObservedAction(
                    street=1,
                    action_type=ActionType.BET_66,
                    to_call=0,
                ),
                ObservedAction(
                    street=2,
                    action_type=ActionType.CHECK_CALL,
                    to_call=100,
                ),
            ],
        )
    s = tracker.get("p1")
    assert s.hands_observed == 10
    assert s.preflop_voluntary_actions == 10  # VPIP per-hand
    assert s.preflop_raises == 10  # PFR per-hand
    assert s.preflop_3bet_opportunities == 10
    assert s.preflop_3bets == 10  # raised when pf_raises_before>=1
    assert s.three_bet() == 1.0
    assert s.postflop_bets == 10
    assert s.postflop_calls == 10
    assert s.af() == 1.0
