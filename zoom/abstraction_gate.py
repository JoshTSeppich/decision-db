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
DEFAULT_SPR_CAP: Final[float] = 10.0

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
    "DEFAULT_SPR_CAP",
    "all_in_allowed",
    "effective_spr",
    "legal_abstract_actions_gated",
]
