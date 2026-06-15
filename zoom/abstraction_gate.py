"""Stack-aware ALL_IN gate (Approach-2, Stage 1, Component 1).

Wraps the frozen `pokerbot.abstraction.actions.legal_abstract_actions` (read-only,
never modified — the DO-NOT list keeps `src/pokerbot/abstraction/` untouched) and
drops ``ALL_IN`` from the abstract action set when the stack is too deep relative
to the pot for a shove to be a sane abstract size.

This is the *mechanical* half of the over-aggression fix. The v5 blueprints shoved
12-20% preflop at 100bb purely because ``ALL_IN`` was offered at every stack depth
(`actions._legal_preflop`/`_legal_postflop` add it whenever ``stack > 0``). That is
distinct from the *distributional* cause (pure self-play), which the mixed-opponent
fine-tune addresses separately.

The gate expresses a CONTINUUM from a single stack-to-pot-ratio (SPR) rule rather
than a hand-tuned per-depth table:

    ALL_IN stays available iff
        no other bet/raise size is legal   (forced jam — the stack is so short that
                                             every raise size meets-or-exceeds it, so
                                             ALL_IN is the only way to commit), OR
        the reference pot is non-positive   (SPR undefined — keep it, defensive), OR
        spr <= spr_cap                      (commitment is near: short stack, or a
                                             built-up pot with a low remaining SPR)

    where ``spr = stack / (pot + to_call)`` (chips behind, over the pot once the
    call is made — the pot the shove would be made into).

At 5bb preflop ``spr`` is small (~2) so ALL_IN survives and the equilibrium makes it
dominant (matching the Nash push/fold tables). At 100bb preflop ``spr`` is large
(~40) so ALL_IN is dropped and aggression routes through RAISE_2_5X/RAISE_3_5X. The
*same* ``spr_cap`` yields pure-shove at the short end and <1% shoves at 100bb — one
mechanism, the whole 2-200bb continuum.

``spr_cap`` is the single tuned knob. Component 5 tunes it against the 100bb
ALL_IN<1% behavioral band while keeping legitimate low-SPR river jams non-zero
(see `tests/zoom/test_abstraction_gate.py::test_river_all_in_survives_at_low_spr`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pokerbot.abstraction.actions import ActionType, legal_abstract_actions

if TYPE_CHECKING:
    from pokerbot.abstraction._types import Street
    from pokerbot.abstraction.actions import AbstractAction

# Default stack-to-pot ratio above which ALL_IN is no longer offered as an abstract
# action. ~10 keeps shoves through the push/fold zone (≲25bb preflop opens) and drops
# the spazzy deep-stack overbet shove. The one knob Component 5 tunes.
#
# NOTE (abstraction-fix Phase 1): SPR is the POSTFLOP gate. It is the WRONG lever for
# PREFLOP: facing a raise lowers SPR below the cap, so a 100bb facing-3bet 4bet-jam
# (SPR≈5) survives while a legitimate 25bb open-jam (SPR≈10) is the higher-SPR spot —
# SPR conflates depth and pot-commitment in opposite directions, so no single cap
# separates them (50k gated smoke: unopened-deep ALL_IN 100%→0% but facing-deep ~47%
# unchanged; nc-AI 6.9→6.6%). Preflop therefore uses `preflop_all_in_allowed` below,
# which gates on DEPTH (eff_bb) and stack-behind COMMITMENT as orthogonal axes.
DEFAULT_SPR_CAP: Final[float] = 10.0

# Preflop depth cap (effective stack in BB) at/below which ALL_IN stays legal — the
# push/fold zone where open-/3bet-jamming is genuine GTO. Above it a preflop ALL_IN is
# the discretionary deep overshove the nc-AI band forbids, UNLESS the actor is committed
# (see DEFAULT_PREFLOP_COMMIT_POTS). ~25bb is the standard top of the push/fold zone.
DEFAULT_MAX_PREFLOP_ALLIN_EFF_BB: Final[float] = 25.0

# Preflop commitment exception: keep ALL_IN even when deep iff the stack remaining AFTER
# calling is at most this many pot-sized bets — i.e. the actor is already in a 4bet/5bet
# shove-war where the jam is the real abstract size. Keyed on STACK-BEHIND (stack −
# to_call), NOT on to_call: a 100bb facing-3bet jam has a large to_call yet ~4.7 pots
# still behind (discretionary → dropped); a 100bb facing-4bet jam has ~1.4 pots behind
# (committed → kept). ~1.5 separates them.
DEFAULT_PREFLOP_COMMIT_POTS: Final[float] = 1.5

# Bet/raise actions other than ALL_IN. If any of these is legal, ALL_IN is not the
# *only* aggressive option, so the SPR rule is allowed to remove it; if none is, the
# forced-jam clause keeps ALL_IN regardless of SPR.
_AGGRESSIVE_NON_ALL_IN: Final[frozenset[ActionType]] = frozenset(
    {
        ActionType.BET_33,
        ActionType.BET_66,
        ActionType.BET_100,
        ActionType.BET_150,
        ActionType.RAISE_2_5X,
        ActionType.RAISE_3_5X,
    }
)


def effective_spr(pot: int, to_call: int, stack: int) -> float:
    """Stack-to-pot ratio used by the gate: ``stack / (pot + to_call)``.

    Returns ``inf`` when the reference pot is non-positive (SPR undefined); callers
    treat that as "cannot compute" and keep ALL_IN.
    """
    ref_pot = pot + to_call
    if ref_pot <= 0:
        return float("inf")
    return stack / ref_pot


def all_in_allowed(
    pot: int,
    to_call: int,
    stack: int,
    *,
    has_other_aggression: bool,
    spr_cap: float = DEFAULT_SPR_CAP,
) -> bool:
    """The single keep-ALL_IN predicate (one source of truth for the continuum).

    ALL_IN stays available iff it's the only aggression (forced jam), the reference
    pot is non-positive (SPR undefined), or SPR ≤ ``spr_cap``. Both the standalone
    `legal_abstract_actions_gated` and the training-time `GatedNLHEGame` call this,
    so the short-stack continuum is defined in exactly one place.
    """
    ref_pot = pot + to_call
    if not has_other_aggression or ref_pot <= 0:
        return True
    return stack <= spr_cap * ref_pot  # spr <= spr_cap, division-free


def preflop_all_in_allowed(
    pot: int,
    to_call: int,
    stack: int,
    bb: int,
    *,
    has_other_aggression: bool,
    max_eff_bb: float = DEFAULT_MAX_PREFLOP_ALLIN_EFF_BB,
    commit_pots: float = DEFAULT_PREFLOP_COMMIT_POTS,
) -> bool:
    """Keep-ALL_IN predicate for PREFLOP (one source of truth for the preflop gate).

    ALL_IN stays available iff ANY of:
      * it's the only aggression (forced jam — no non-ALL_IN raise fits the stack), or
      * the reference pot is non-positive (undefined — keep, defensive), or
      * the effective stack is short — ``eff_bb <= max_eff_bb`` — the push/fold zone
        where open-/3bet-jamming is genuine GTO (rescued by DEPTH regardless of pot,
        which is exactly what SPR alone could not do), or
      * the actor is COMMITTED — stack remaining after calling is at most
        ``commit_pots`` pot-sized bets — a real 4bet/5bet shove-war.

    Otherwise ALL_IN is DROPPED: deep, uncommitted, with a non-shove raise available —
    the discretionary deep overshove (incl. the facing-3bet-100bb 4bet-jam) the nc-AI
    band forbids. Depth (eff_bb) and commitment (stack-behind) are ORTHOGONAL axes; SPR
    conflated them, so this gate keys on each directly.
    """
    if not has_other_aggression:
        return True
    ref_pot = pot + to_call
    if ref_pot <= 0:
        return True
    eff_bb = stack / bb if bb > 0 else float("inf")
    if eff_bb <= max_eff_bb:
        return True  # push/fold zone — depth keeps the jam regardless of pot
    stack_behind_after_call = stack - to_call
    return stack_behind_after_call <= commit_pots * ref_pot  # committed shove-war


def legal_abstract_actions_gated(
    pot: int,
    to_call: int,
    stack: int,
    min_raise: int,
    street: Street,
    *,
    spr_cap: float = DEFAULT_SPR_CAP,
) -> list[AbstractAction]:
    """`legal_abstract_actions` with ALL_IN dropped at deep SPR.

    Identical to the wrapped function except that ALL_IN is removed when SPR exceeds
    ``spr_cap`` AND some other bet/raise size is legal. All non-ALL_IN actions are
    returned unchanged, in order — the gate only ever subtracts ALL_IN.
    """
    base = legal_abstract_actions(pot, to_call, stack, min_raise, street)
    has_other_aggression = any(a.type in _AGGRESSIVE_NON_ALL_IN for a in base)
    if all_in_allowed(
        pot, to_call, stack, has_other_aggression=has_other_aggression, spr_cap=spr_cap
    ):
        return base
    return [a for a in base if a.type != ActionType.ALL_IN]


__all__ = [
    "DEFAULT_MAX_PREFLOP_ALLIN_EFF_BB",
    "DEFAULT_PREFLOP_COMMIT_POTS",
    "DEFAULT_SPR_CAP",
    "all_in_allowed",
    "effective_spr",
    "legal_abstract_actions_gated",
    "preflop_all_in_allowed",
]
