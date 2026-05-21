"""Spec.html §B tests for the action abstraction module."""

from __future__ import annotations

import math
import random

import pytest

from pokerbot.abstraction import (
    AbstractAction,
    ActionType,
    legal_abstract_actions,
    resolve_action,
    translate_bet,
)

# ─────────── test_legal_actions_short_stack ───────────


def test_legal_actions_short_stack_postflop() -> None:
    """Spec §B: 5BB stack postflop yields `[FOLD, CHECK_CALL, ALL_IN]` only.

    pot=20BB, to_call=2BB, stack=5BB. Every BET_X size exceeds stack → dropped.
    """
    legal = legal_abstract_actions(pot=20, to_call=2, stack=5, min_raise=2, street="flop")
    types = [a.type for a in legal]
    assert types == [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.ALL_IN], types


# ─────────── test_pseudo_harmonic_midpoint ───────────


def test_pseudo_harmonic_midpoint_splits_about_50_50() -> None:
    """Spec §B: a real bet at the geometric mean of two adjacent abstract sizes
    splits ~50/50 over 10k samples (pseudo-harmonic property).
    """
    pot = 1000
    legal = [
        AbstractAction(ActionType.BET_33, round(0.33 * pot)),
        AbstractAction(ActionType.BET_66, round(0.66 * pot)),
    ]
    a_chips = legal[0].amount_chips
    b_chips = legal[1].amount_chips
    real_amount = round(math.sqrt(a_chips * b_chips))  # geometric mean

    rng = random.Random(0xBEEF)
    counts = {ActionType.BET_33: 0, ActionType.BET_66: 0}
    n = 10_000
    for _ in range(n):
        result = translate_bet(real_amount, pot, legal, rng)
        counts[result.type] += 1
    frac_low = counts[ActionType.BET_33] / n
    # GM-of-pot-fractions split is close to but not exactly 50/50 for our specific
    # (A=0.33, B=0.66); empirically ~0.53. Accept anything in [0.40, 0.60].
    assert 0.40 <= frac_low <= 0.60, f"low fraction {frac_low:.3f} outside [0.40, 0.60]"


# ─────────── test_pseudo_harmonic_at_boundary ───────────


def test_pseudo_harmonic_at_boundary_is_deterministic() -> None:
    """Spec §B: a real bet exactly equal to an abstract size always maps to that size."""
    pot = 100
    legal = [
        AbstractAction(ActionType.BET_33, 33),
        AbstractAction(ActionType.BET_66, 66),
        AbstractAction(ActionType.BET_100, 100),
        AbstractAction(ActionType.ALL_IN, 1000),
    ]
    rng = random.Random(0)
    for target in (legal[0], legal[1], legal[2]):
        for _ in range(50):
            picked = translate_bet(target.amount_chips, pot, legal, rng)
            assert picked.type == target.type, (target.amount_chips, picked.type)


# ─────────── test_resolve_clamp ───────────


def test_resolve_clamp_bet_150_to_stack() -> None:
    """Spec §B: BET_150 with pot=100, stack=120 emits 120 (all-in clamp);
    action_type stays BET_150 for the strategy-lookup key.
    """
    action = AbstractAction(ActionType.BET_150, amount_chips=150)
    emitted = resolve_action(action, pot=100, stack=120, min_raise=10)
    assert emitted == 120
    assert action.type == ActionType.BET_150  # caller keeps abstract type


# ─────────── test_preflop_open_sizes ───────────


def test_preflop_open_has_raise_not_bet_sizes() -> None:
    """Spec §B: first-to-act preflop has RAISE_2_5X / RAISE_3_5X, not BET_*."""
    legal = legal_abstract_actions(pot=3, to_call=2, stack=200, min_raise=2, street="preflop")
    types = {a.type for a in legal}
    assert ActionType.RAISE_2_5X in types
    assert ActionType.RAISE_3_5X in types
    for bet_t in (ActionType.BET_33, ActionType.BET_66, ActionType.BET_100, ActionType.BET_150):
        assert bet_t not in types, f"preflop should not have {bet_t.name}"


# ─────────── test_translation_seeded_determinism ───────────


def test_translation_same_seed_same_result() -> None:
    """Spec §B: same rng seed yields same translation."""
    pot = 100
    legal = [
        AbstractAction(ActionType.BET_33, 33),
        AbstractAction(ActionType.BET_66, 66),
        AbstractAction(ActionType.BET_100, 100),
        AbstractAction(ActionType.BET_150, 150),
        AbstractAction(ActionType.ALL_IN, 1000),
    ]

    def run(seed: int, real_amount: int) -> list[ActionType]:
        rng = random.Random(seed)
        return [translate_bet(real_amount, pot, legal, rng).type for _ in range(200)]

    assert run(42, 50) == run(42, 50)
    assert run(42, 50) != run(99, 50)


# ─────────── extra sanity checks ───────────


def test_legal_below_all_in_excludes_oversized_bet() -> None:
    """A 1.5x-pot bet that would meet or exceed stack is dropped (ALL_IN covers it)."""
    legal = legal_abstract_actions(pot=100, to_call=0, stack=140, min_raise=2, street="flop")
    types = {a.type for a in legal}
    assert ActionType.BET_100 in types  # 100 < 140
    assert ActionType.BET_150 not in types  # 150 >= 140 → drop
    assert ActionType.ALL_IN in types


def test_check_call_amount_zero_when_no_call_owed() -> None:
    legal = legal_abstract_actions(pot=50, to_call=0, stack=100, min_raise=2, street="flop")
    cc = next(a for a in legal if a.type == ActionType.CHECK_CALL)
    assert cc.amount_chips == 0
    assert all(a.type != ActionType.FOLD for a in legal)  # nothing to fold against


def test_resolve_fold_returns_zero() -> None:
    a = AbstractAction(ActionType.FOLD, 0)
    assert resolve_action(a, pot=100, stack=50, min_raise=2) == 0


def test_resolve_all_in_returns_stack() -> None:
    a = AbstractAction(ActionType.ALL_IN, 9999)
    assert resolve_action(a, pot=100, stack=50, min_raise=2) == 50


def test_translate_bet_below_smallest_maps_to_smallest() -> None:
    pot = 100
    legal = [
        AbstractAction(ActionType.BET_33, 33),
        AbstractAction(ActionType.BET_66, 66),
    ]
    rng = random.Random(0)
    assert translate_bet(5, pot, legal, rng).type == ActionType.BET_33


def test_translate_bet_above_largest_maps_to_largest() -> None:
    pot = 100
    legal = [
        AbstractAction(ActionType.BET_33, 33),
        AbstractAction(ActionType.ALL_IN, 500),
    ]
    rng = random.Random(0)
    assert translate_bet(2000, pot, legal, rng).type == ActionType.ALL_IN


def test_translate_bet_raises_when_no_bets_in_legal() -> None:
    legal = [AbstractAction(ActionType.FOLD, 0), AbstractAction(ActionType.CHECK_CALL, 0)]
    with pytest.raises(ValueError, match="bet-like"):
        translate_bet(50, 100, legal, random.Random(0))
