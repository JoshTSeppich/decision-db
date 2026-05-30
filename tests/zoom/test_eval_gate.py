"""Component 4 — Stage-1 eval harness self-validation (Approach-2).

Lock 1: the should-be-better acceptance test comes FIRST. It is the guard against
the v5-6max failure (a gate that measured the wrong thing and shipped a broken
policy on a green light). If the harness can't score an ALL_IN-zeroed policy
strictly better than the raw over-aggressive one, nothing it later says about the
fine-tune means anything.

Order: (1) should-be-better acceptance, (2) catastrophic screens, (3) band
evaluator PASS-iff-all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from zoom.eval import (
    BehavioralProfile,
    band_score,
    catastrophic_screen,
    evaluate_bands,
    profile_spot_policy,
    zero_all_in_at_deep_stacks,
)

from pokerbot.abstraction import ActionType
from pokerbot.training import SimpleNLHEGame

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zoom.agents import AgentSpot


def _ungated_3max() -> SimpleNLHEGame:
    """100bb 3-max on the UNGATED game, so an over-aggressive policy can actually
    shove and the ALL_IN% difference is real."""
    from pokerbot.abstraction import AbstractionTables

    return SimpleNLHEGame(AbstractionTables(), blinds=(5, 10), starting_stack=1000, table_size=3)


# ─────────── 1. LOCK 1: should-be-better acceptance ───────────


def _over_aggressive(spot: AgentSpot) -> Mapping[ActionType, float]:
    """A caricature of the v5 signature: shoves most hands preflop."""
    if spot.street == "preflop":
        return {ActionType.ALL_IN: 0.7, ActionType.CHECK_CALL: 0.3}
    return {ActionType.CHECK_CALL: 0.6, ActionType.BET_66: 0.4}


def test_should_be_better_allin_zeroed_scores_strictly_better() -> None:
    """ACCEPTANCE: the ALL_IN-zeroed-at-deep-stacks policy scores strictly better on
    the bands than the raw over-aggressive policy. Proves the harness can tell a good
    policy from a bad one — the binding Lock-1 guarantee.
    """
    game = _ungated_3max()
    raw_profile = profile_spot_policy(_over_aggressive, game, n_hands=200, seed=2026)
    fixed_profile = profile_spot_policy(
        zero_all_in_at_deep_stacks(_over_aggressive), game, n_hands=200, seed=2026
    )

    assert fixed_profile.all_in_preflop_pct < raw_profile.all_in_preflop_pct
    assert band_score(fixed_profile) < band_score(raw_profile), (
        f"harness cannot distinguish fixed from raw: "
        f"raw={band_score(raw_profile):.2f} fixed={band_score(fixed_profile):.2f}"
    )


def test_allin_metric_is_load_bearing_for_the_distinction() -> None:
    """Teeth check (maps to the v5 'measured the wrong thing' failure): the preflop
    ALL_IN% reduction is real and accounts for the bulk of the score improvement —
    a harness blind to ALL_IN% could not rank the fixed policy clearly better.
    """
    game = _ungated_3max()
    raw = profile_spot_policy(_over_aggressive, game, n_hands=200, seed=2026)
    fixed = profile_spot_policy(
        zero_all_in_at_deep_stacks(_over_aggressive), game, n_hands=200, seed=2026
    )
    allin_drop = raw.all_in_preflop_pct - fixed.all_in_preflop_pct
    total_improvement = band_score(raw) - band_score(fixed)
    assert allin_drop > 0
    assert total_improvement >= allin_drop - 1e-6, (
        "score improvement isn't explained by the ALL_IN band — harness may be "
        "measuring the wrong thing"
    )


# ─────────── 2. catastrophic screens ───────────


def _ranks(hole: tuple[int, int]) -> set[int]:
    return {hole[0] >> 2, hole[1] >> 2}


def _broken_policy(spot: AgentSpot) -> Mapping[ActionType, float]:
    """Folds AA and opens 72o — the catastrophes the screen must catch."""
    ranks = _ranks(spot.hole)
    if ranks == {12}:  # pocket aces
        return {ActionType.FOLD: 1.0}
    if ranks == {0, 5}:  # 72 offsuit
        return {ActionType.RAISE_2_5X: 1.0}
    return {ActionType.CHECK_CALL: 1.0}


def _sane_policy(spot: AgentSpot) -> Mapping[ActionType, float]:
    """Raises AA, never opens 72o."""
    ranks = _ranks(spot.hole)
    if ranks == {12}:
        return {ActionType.RAISE_2_5X: 1.0}
    return {ActionType.CHECK_CALL: 1.0}


def test_catastrophic_screen_flags_folds_aa_and_opens_72o() -> None:
    violations = catastrophic_screen(_broken_policy)
    assert any("AA" in v for v in violations), violations
    assert any("72o" in v for v in violations), violations


def test_catastrophic_screen_clean_on_sane_policy() -> None:
    assert catastrophic_screen(_sane_policy) == []


# ─────────── 3. band evaluator: PASS iff ALL metrics in range ───────────


def test_bands_pass_when_all_metrics_in_range() -> None:
    result = evaluate_bands(
        BehavioralProfile.from_pcts(vpip=26, pfr=22, all_in_preflop=0, fold_to_cbet=55)
    )
    assert result.passed
    assert result.score == 0.0
    assert result.failures == ()


@pytest.mark.parametrize(
    ("vpip", "pfr", "all_in", "fold_to_cbet", "bad_metric"),
    [
        (40, 22, 0, 55, "vpip_pct"),  # vpip too high
        (26, 5, 0, 55, "pfr_pct"),  # pfr too low
        (26, 22, 12, 55, "all_in_preflop_pct"),  # shoves too much
        (26, 22, 0, 20, "fold_to_cbet_pct"),  # folds too little
    ],
)
def test_single_out_of_range_metric_forces_fail(
    vpip: float, pfr: float, all_in: float, fold_to_cbet: float, bad_metric: str
) -> None:
    """A single out-of-range metric forces FAIL — no averaging a bad metric away."""
    result = evaluate_bands(
        BehavioralProfile.from_pcts(
            vpip=vpip, pfr=pfr, all_in_preflop=all_in, fold_to_cbet=fold_to_cbet
        )
    )
    assert not result.passed
    assert {m.name for m in result.failures} == {bad_metric}
    assert result.score > 0.0


def test_band_score_zero_iff_passed() -> None:
    passing = evaluate_bands(
        BehavioralProfile.from_pcts(vpip=25, pfr=20, all_in_preflop=0, fold_to_cbet=55)
    )
    failing = evaluate_bands(
        BehavioralProfile.from_pcts(vpip=50, pfr=20, all_in_preflop=0, fold_to_cbet=55)
    )
    assert passing.score == 0.0 and passing.passed
    assert failing.score > 0.0 and not failing.passed
