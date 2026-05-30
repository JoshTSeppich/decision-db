"""Component 1 — stack-aware ALL_IN gate (Approach-2, Stage 1).

The v5 blueprints shoved 12-20% preflop at 100bb purely because `ALL_IN` was a
legal abstract action at every stack depth. `zoom.abstraction_gate` wraps the
frozen `pokerbot.abstraction.actions.legal_abstract_actions` (never modified) and
drops `ALL_IN` when the stack is too deep relative to the pot for a shove to be a
sane abstract size — expressing a continuum (pure-shove short / ~no-shove deep)
from one SPR rule.

Red-first ordering: `test_deep_stack_all_in_illegal_preflop` is the first failing
test — it pins the exact v5 bug the gate exists to kill.
"""

from __future__ import annotations

import random

from zoom.abstraction_gate import (
    DEFAULT_SPR_CAP,
    effective_spr,
    legal_abstract_actions_gated,
)

from pokerbot.abstraction import ActionType, translate_bet
from pokerbot.abstraction.actions import action_to_byte, byte_to_action_type, legal_abstract_actions

_AGGRO_NON_ALL_IN = {
    ActionType.BET_33,
    ActionType.BET_66,
    ActionType.BET_100,
    ActionType.BET_150,
    ActionType.RAISE_2_5X,
    ActionType.RAISE_3_5X,
}


# ─────────── the first red test: the exact v5 bug ───────────


def test_deep_stack_all_in_illegal_preflop() -> None:
    """100bb preflop open (SB=1/BB=2 → pot=3, to_call=2, stack=200): ALL_IN is
    NOT an available abstract action. This is the mechanical cause of the
    12-20% preflop shove rate the gate exists to eliminate.
    """
    gated = legal_abstract_actions_gated(pot=3, to_call=2, stack=200, min_raise=2, street="preflop")
    types = {a.type for a in gated}
    assert ActionType.ALL_IN not in types, types
    # aggression must still be possible — it just routes through raises.
    assert ActionType.RAISE_2_5X in types
    assert ActionType.RAISE_3_5X in types


# ─────────── short-stack: shove survives ───────────


def test_short_stack_all_in_legal_preflop() -> None:
    """5bb preflop open (stack=10 in BB=2 chips): ALL_IN survives — at the short
    end the equilibrium makes it dominant (push/fold zone).
    """
    gated = legal_abstract_actions_gated(pot=3, to_call=2, stack=10, min_raise=2, street="preflop")
    types = {a.type for a in gated}
    assert ActionType.ALL_IN in types, types


# ─────────── the continuum: one rule, monotone in depth ───────────


def test_all_in_availability_is_a_continuum_monotone_in_stack() -> None:
    """At a fixed preflop spot, sweeping stack from short→deep, ALL_IN is present
    while shallow and absent once deep, with a SINGLE transition (monotone). A
    flat threshold would also pass present/absent — monotonicity is what proves
    it's a continuum, not a special-case.
    """
    pot, to_call, min_raise = 3, 2, 2
    stacks = list(range(4, 401, 2))
    present = [
        ActionType.ALL_IN
        in {a.type for a in legal_abstract_actions_gated(pot, to_call, s, min_raise, "preflop")}
        for s in stacks
    ]
    # present at the short end, absent at the deep end
    assert present[0] is True, "ALL_IN should be available at the shortest stack"
    assert present[-1] is False, "ALL_IN should be gated off at 200bb"
    # monotone: once it turns off it never turns back on (no True after a False)
    first_off = present.index(False)
    assert all(p is False for p in present[first_off:]), "ALL_IN availability is not monotone"


def test_transition_matches_spr_cap_preflop() -> None:
    """The on/off transition sits where stack ≈ spr_cap·(pot+to_call)."""
    pot, to_call = 3, 2
    ref = pot + to_call  # 5
    boundary = DEFAULT_SPR_CAP * ref  # stacks below → keep, above → drop
    just_below = int(boundary) - ref
    just_above = int(boundary) + ref
    below = {a.type for a in legal_abstract_actions_gated(pot, to_call, just_below, 2, "preflop")}
    above = {a.type for a in legal_abstract_actions_gated(pot, to_call, just_above, 2, "preflop")}
    assert ActionType.ALL_IN in below
    assert ActionType.ALL_IN not in above


# ─────────── R2 guard: legitimate low-SPR jams survive ───────────


def test_river_all_in_survives_at_low_spr() -> None:
    """A built-up river pot (pot=120, stack=80 → SPR≈0.67) keeps ALL_IN — the
    gate must NOT strip legitimate low-SPR commitment shoves. This is the R2
    regression guard the plan requires as a real test, not a tuning note.
    """
    gated = legal_abstract_actions_gated(pot=120, to_call=0, stack=80, min_raise=2, street="river")
    assert ActionType.ALL_IN in {a.type for a in gated}


def test_river_all_in_dropped_at_high_spr() -> None:
    """River is NOT blanket-allowed: a tiny pot with a deep stack (pot=10,
    stack=200 → SPR=20) drops the spazzy overbet shove.
    """
    gated = legal_abstract_actions_gated(pot=10, to_call=0, stack=200, min_raise=2, street="river")
    assert ActionType.ALL_IN not in {a.type for a in gated}


# ─────────── forced-jam safety net ───────────


def test_forced_jam_keeps_all_in_when_no_other_aggression() -> None:
    """When the stack is so short every bet size is dropped (base =
    [FOLD, CHECK_CALL, ALL_IN]), the gate must keep ALL_IN — it's the only
    aggressive action; stripping it would leave a player unable to commit.
    """
    base = legal_abstract_actions(pot=1000, to_call=2, stack=5, min_raise=2, street="flop")
    base_types = {a.type for a in base}
    assert base_types == {ActionType.FOLD, ActionType.CHECK_CALL, ActionType.ALL_IN}
    gated = legal_abstract_actions_gated(pot=1000, to_call=2, stack=5, min_raise=2, street="flop")
    assert ActionType.ALL_IN in {a.type for a in gated}


# ─────────── the gate only ever removes ALL_IN ───────────


def test_gated_is_base_minus_at_most_all_in() -> None:
    """Across a sweep of spots the gated set is a subset of the base set, and the
    only type the gate may remove is ALL_IN — never FOLD/CHECK_CALL/bets/raises.
    """
    spots = [
        (3, 2, 200, "preflop"),
        (3, 2, 10, "preflop"),
        (50, 0, 100, "flop"),
        (120, 0, 80, "river"),
        (10, 0, 200, "turn"),
        (1000, 2, 5, "flop"),
    ]
    for pot, to_call, stack, street in spots:
        base = legal_abstract_actions(pot, to_call, stack, 2, street)  # type: ignore[arg-type]
        gated = legal_abstract_actions_gated(pot, to_call, stack, 2, street)  # type: ignore[arg-type]
        base_types = [a.type for a in base]
        gated_types = [a.type for a in gated]
        assert set(gated_types) <= set(base_types)
        removed = set(base_types) - set(gated_types)
        assert removed <= {ActionType.ALL_IN}, (pot, to_call, stack, street, removed)
        # non-all-in actions are preserved exactly (order and identity).
        assert [t for t in base_types if t != ActionType.ALL_IN] == [
            t for t in gated_types if t != ActionType.ALL_IN
        ]


# ─────────── spr_cap is the one tuned knob ───────────


def test_spr_cap_parameter_controls_threshold() -> None:
    """A generous spr_cap keeps a deep-stack ALL_IN; a tight one drops it — the
    knob Component 5 tunes against the 100bb ALL_IN<1% band.
    """
    spot = dict(pot=3, to_call=2, stack=200, min_raise=2, street="preflop")
    keep = legal_abstract_actions_gated(**spot, spr_cap=1000.0)  # type: ignore[arg-type]
    drop = legal_abstract_actions_gated(**spot, spr_cap=1.0)  # type: ignore[arg-type]
    assert ActionType.ALL_IN in {a.type for a in keep}
    assert ActionType.ALL_IN not in {a.type for a in drop}


# ─────────── effective_spr math ───────────


def test_effective_spr_values() -> None:
    assert effective_spr(pot=3, to_call=2, stack=200) == 200 / 5
    assert effective_spr(pot=120, to_call=0, stack=80) == 80 / 120
    assert effective_spr(pot=0, to_call=0, stack=100) == float("inf")  # undefined → inf


# ─────────── off-tree mapping still works on the gated set ───────────


def test_translate_bet_deterministic_on_gated_set() -> None:
    """Off-tree bet translation (reused unchanged) is deterministic for a seeded
    rng over the gated legal set — the gate doesn't disturb `translate_bet`.
    """
    gated = legal_abstract_actions_gated(pot=120, to_call=0, stack=80, min_raise=2, street="river")

    def run(seed: int) -> list[ActionType]:
        rng = random.Random(seed)
        return [translate_bet(55, 120, gated, rng).type for _ in range(100)]

    assert run(7) == run(7)


def test_round_trip_action_byte_on_gated_actions() -> None:
    """Every action surviving the gate round-trips through the history byte codec."""
    gated = legal_abstract_actions_gated(pot=50, to_call=0, stack=100, min_raise=2, street="flop")
    for a in gated:
        if a.type == ActionType.FOLD:
            assert byte_to_action_type(action_to_byte(a)) == ActionType.FOLD
        elif a.type == ActionType.CHECK_CALL:
            assert byte_to_action_type(action_to_byte(a)) == ActionType.CHECK_CALL
        else:
            assert byte_to_action_type(action_to_byte(a)) == a.type


def test_default_spr_cap_is_documented_constant() -> None:
    """The default lives in one place so Component 5 tunes a single knob."""
    default_gated = legal_abstract_actions_gated(
        pot=3, to_call=2, stack=200, min_raise=2, street="preflop"
    )
    explicit_gated = legal_abstract_actions_gated(
        pot=3, to_call=2, stack=200, min_raise=2, street="preflop", spr_cap=DEFAULT_SPR_CAP
    )
    assert [a.type for a in default_gated] == [a.type for a in explicit_gated]


def test_check_call_always_survives() -> None:
    """CHECK_CALL is always present post-gate (a player can always proceed)."""
    for pot, to_call, stack, street in [
        (3, 2, 200, "preflop"),
        (50, 10, 100, "flop"),
        (10, 0, 200, "river"),
    ]:
        gated = legal_abstract_actions_gated(pot, to_call, stack, 2, street)  # type: ignore[arg-type]
        assert ActionType.CHECK_CALL in {a.type for a in gated}
