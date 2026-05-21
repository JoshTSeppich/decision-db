"""Action abstraction (Spec.html §B).

Public API:
    - `ActionType`             IntEnum (0=FOLD, 1=CHECK_CALL, 2..6=BET_*, 7..8=RAISE_*X)
    - `AbstractAction`         frozen dataclass (type + amount_chips)
    - `legal_abstract_actions` what's allowed given pot/to_call/stack/street
    - `translate_bet`          pseudo-harmonic mapping of an opponent's real bet
    - `resolve_action`         abstract action → concrete chip count (clamped)

Plus the per-action byte encoding used by `infoset.encode_history` (§C):
    - `action_to_byte(action)`        → int (one of 0x00, 0x01, 0x02, 0x10..0x14, 0x20..0x21)
    - `byte_to_action_type(b)`        → ActionType
    - `STREET_BOUNDARY_BYTE = 0xF0`
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import random

    from pokerbot.abstraction._types import Street

# ─────────── enum + dataclass ───────────


class ActionType(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    BET_33 = 2
    BET_66 = 3
    BET_100 = 4
    BET_150 = 5
    ALL_IN = 6
    # Preflop-specific
    RAISE_2_5X = 7
    RAISE_3_5X = 8


@dataclass(frozen=True, slots=True)
class AbstractAction:
    type: ActionType
    amount_chips: int  # additional chips committed this action (0 for fold/check)


# Sizes for postflop bets, in pot fractions (Spec.html §B).
POSTFLOP_BET_SIZES: Final[tuple[tuple[ActionType, float], ...]] = (
    (ActionType.BET_33, 0.33),
    (ActionType.BET_66, 0.66),
    (ActionType.BET_100, 1.0),
    (ActionType.BET_150, 1.5),
)

# Sizes for preflop raises, as multipliers of `to_call` (≈ BB for an open).
PREFLOP_RAISE_RATIOS: Final[tuple[tuple[ActionType, float], ...]] = (
    (ActionType.RAISE_2_5X, 2.5),
    (ActionType.RAISE_3_5X, 3.5),
)

_BET_LIKE_TYPES: Final[frozenset[ActionType]] = frozenset(
    {
        ActionType.BET_33,
        ActionType.BET_66,
        ActionType.BET_100,
        ActionType.BET_150,
        ActionType.ALL_IN,
        ActionType.RAISE_2_5X,
        ActionType.RAISE_3_5X,
    }
)


# ─────────── legal_abstract_actions ───────────


def legal_abstract_actions(
    pot: int,
    to_call: int,
    stack: int,
    min_raise: int,  # noqa: ARG001  reserved for step 6 (resolve_action uses it)
    street: Street,
) -> list[AbstractAction]:
    """Abstract actions available for the current player.

    `min_raise` is accepted for signature parity with the spec; the spec defers
    min-raise legality enforcement to `resolve_action`/`translate_bet`. Bet
    sizes whose nominal target meets-or-exceeds `stack` are dropped here —
    `ALL_IN` always covers that branch.
    """
    if pot < 0 or to_call < 0 or stack < 0:
        raise ValueError(f"pot/to_call/stack must be non-negative: {pot=} {to_call=} {stack=}")

    if street == "preflop":
        return _legal_preflop(to_call, stack)
    return _legal_postflop(pot, to_call, stack)


def _legal_preflop(to_call: int, stack: int) -> list[AbstractAction]:
    actions: list[AbstractAction] = []
    if to_call > 0:
        actions.append(AbstractAction(ActionType.FOLD, 0))
    actions.append(AbstractAction(ActionType.CHECK_CALL, min(to_call, stack)))
    # The "previous raise" anchor for first-raise vs 3-bet+ is approximated by
    # `to_call` — BB when no one has raised yet, the prior raise-to amount when
    # someone has. Step 6 may refine using explicit raise-history.
    for at, ratio in PREFLOP_RAISE_RATIOS:
        target = round(ratio * to_call) if to_call > 0 else 0
        if 0 < target < stack:
            actions.append(AbstractAction(at, target))
    if stack > 0:
        actions.append(AbstractAction(ActionType.ALL_IN, stack))
    return actions


def _legal_postflop(pot: int, to_call: int, stack: int) -> list[AbstractAction]:
    actions: list[AbstractAction] = []
    if to_call > 0:
        actions.append(AbstractAction(ActionType.FOLD, 0))
    actions.append(AbstractAction(ActionType.CHECK_CALL, min(to_call, stack)))
    for at, frac in POSTFLOP_BET_SIZES:
        # Total chips committed this action = call portion + size·pot raise/bet.
        target = to_call + round(frac * pot)
        if to_call < target < stack:
            actions.append(AbstractAction(at, target))
    if stack > 0:
        actions.append(AbstractAction(ActionType.ALL_IN, stack))
    return actions


# ─────────── translate_bet (pseudo-harmonic) ───────────


def translate_bet(
    real_amount: int,
    pot: int,
    legal: list[AbstractAction],
    rng: random.Random,
) -> AbstractAction:
    """Pseudo-harmonic mapping of an opponent's real bet onto our abstract sizes.

    Picks two adjacent bet-like actions A < B from `legal` bracketing
    `real_amount`, then samples A with probability
        p(A) = (B - x)(1 + A) / ((B - A)(1 + x))
    where (A, B, x) are pot-scaled bet fractions. Real bets at or below the
    smallest abstract size map to that size; real bets at or above the largest
    map to ALL_IN (or the largest available).
    """
    if pot <= 0:
        raise ValueError(f"pot must be positive for translation: {pot}")
    candidates: list[tuple[int, AbstractAction]] = sorted(
        (a.amount_chips, a) for a in legal if a.type in _BET_LIKE_TYPES
    )
    if not candidates:
        raise ValueError("translate_bet requires at least one bet-like action in `legal`")

    # Below the smallest → smallest. At-or-above the largest → largest.
    if real_amount <= candidates[0][0]:
        return candidates[0][1]
    if real_amount >= candidates[-1][0]:
        return candidates[-1][1]

    # Locate adjacent (A, B) bracketing real_amount.
    for i in range(len(candidates) - 1):
        lo_chips, lo_action = candidates[i]
        hi_chips, hi_action = candidates[i + 1]
        if lo_chips <= real_amount <= hi_chips:
            if real_amount == lo_chips:
                return lo_action
            if real_amount == hi_chips:
                return hi_action
            a = lo_chips / pot
            b = hi_chips / pot
            x = real_amount / pot
            p_lo = (b - x) * (1.0 + a) / ((b - a) * (1.0 + x))
            return lo_action if rng.random() < p_lo else hi_action

    return candidates[-1][1]


# ─────────── resolve_action ───────────


def resolve_action(
    action: AbstractAction,
    pot: int,  # noqa: ARG001  reserved for future use; signature follows spec
    stack: int,
    min_raise: int,
) -> int:
    """Concrete chip amount to put in. Clamped to stack; raises clamped to ≥ min_raise.

    `BET_150` with `pot=100, stack=120` → 120 (all-in clamp); the caller keeps
    the abstract `BET_150` ActionType for strategy lookup but emits 120 on the wire.
    """
    if stack < 0:
        raise ValueError(f"stack must be non-negative: {stack}")
    if action.type == ActionType.FOLD:
        return 0
    if action.type == ActionType.CHECK_CALL:
        return min(action.amount_chips, stack)
    if action.type == ActionType.ALL_IN:
        return stack
    # bet / raise: clamp upward to min_raise, downward to stack
    amount = max(action.amount_chips, min_raise)
    return min(amount, stack)


# ─────────── byte encoding for history (used by infoset.encode_history) ───────────

STREET_BOUNDARY_BYTE: Final[int] = 0xF0

_BYTE_FOLD: Final[int] = 0x00
_BYTE_CHECK: Final[int] = 0x01
_BYTE_CALL: Final[int] = 0x02
_BYTE_BET_BASE: Final[int] = 0x10  # BET_33 → 0x10, …, ALL_IN → 0x14
_BYTE_RAISE_BASE: Final[int] = 0x20  # RAISE_2_5X → 0x20, RAISE_3_5X → 0x21

# precomputed for fast decode
_BYTE_TO_ACTION_TYPE: Final[dict[int, ActionType]] = {
    _BYTE_FOLD: ActionType.FOLD,
    _BYTE_CHECK: ActionType.CHECK_CALL,
    _BYTE_CALL: ActionType.CHECK_CALL,
    _BYTE_BET_BASE + 0: ActionType.BET_33,
    _BYTE_BET_BASE + 1: ActionType.BET_66,
    _BYTE_BET_BASE + 2: ActionType.BET_100,
    _BYTE_BET_BASE + 3: ActionType.BET_150,
    _BYTE_BET_BASE + 4: ActionType.ALL_IN,
    _BYTE_RAISE_BASE + 0: ActionType.RAISE_2_5X,
    _BYTE_RAISE_BASE + 1: ActionType.RAISE_3_5X,
}


def action_to_byte(action: AbstractAction) -> int:
    """One-byte tag for `action` per §C. CHECK_CALL splits into 0x01/0x02 by amount."""
    t = action.type
    if t == ActionType.FOLD:
        return _BYTE_FOLD
    if t == ActionType.CHECK_CALL:
        return _BYTE_CHECK if action.amount_chips == 0 else _BYTE_CALL
    if ActionType.BET_33 <= t <= ActionType.ALL_IN:
        return _BYTE_BET_BASE + (int(t) - int(ActionType.BET_33))
    if t in (ActionType.RAISE_2_5X, ActionType.RAISE_3_5X):
        return _BYTE_RAISE_BASE + (int(t) - int(ActionType.RAISE_2_5X))
    raise ValueError(f"unencodable action type: {t!r}")


def byte_to_action_type(b: int) -> ActionType:
    """Inverse of `action_to_byte` — loses the check-vs-call distinction (intentional)."""
    if b == STREET_BOUNDARY_BYTE:
        raise ValueError("street-boundary byte 0xF0 is not an action")
    try:
        return _BYTE_TO_ACTION_TYPE[b]
    except KeyError as e:
        raise ValueError(f"unknown action byte: 0x{b:02x}") from e


__all__ = [
    "POSTFLOP_BET_SIZES",
    "PREFLOP_RAISE_RATIOS",
    "STREET_BOUNDARY_BYTE",
    "AbstractAction",
    "ActionType",
    "action_to_byte",
    "byte_to_action_type",
    "legal_abstract_actions",
    "resolve_action",
    "translate_bet",
]
