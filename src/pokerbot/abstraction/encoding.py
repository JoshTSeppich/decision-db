"""Shared infoset-encoding helpers (Spec.html §C).

Both `pokerbot.runtime.adapter` (live decisions) and
`pokerbot.training.nlhe_game` (training-time game simulation) must produce
*byte-identical* infoset keys for the same observable state — otherwise DB
lookups silently miss in production. They both import from this module.

Stack bucket boundaries are pinned to spec §C:
    [<10, 10-20, 20-30, 30-50, 50-75, 75-100, 100-150, 150-200, 200-300, >300] BB

Position encoding is SB-relative: position 0 = SB, position table_size-1 = button.

`encode_action_history_byte` is the single source of truth for §C history-byte
encoding — both the runtime adapter (replaying JSON history forward) and the
trainer (snapshotting pokerkit state at the moment of the action) call it. By
construction, both sides emit byte-identical history blobs for the same
observable state. The bet/raise classifier is pure chip math: it does NOT use
the actor's chosen abstract action type. This trades expressiveness (we can't
distinguish a deliberate RAISE_2_5X from a RAISE_3_5X that happens to fall in
the same chip-math bucket) for consistency.
"""

from __future__ import annotations

from typing import Final, Literal

STACK_BUCKET_BOUNDARIES: Final[tuple[int, ...]] = (10, 20, 30, 50, 75, 100, 150, 200, 300)
NUM_STACK_BUCKETS: Final[int] = len(STACK_BUCKET_BOUNDARIES) + 1  # = 10


# Action-kind taxonomy for history encoding. The adapter derives these from
# the JSON `type` field; the trainer derives them from the recorded
# `AbstractAction.type`/amount pair. Bet/raise both map to "bet_or_raise"; the
# preflop-vs-postflop branch in `encode_action_history_byte` picks the right
# classifier from `street`.
ActionKind = Literal["fold", "check", "call", "all-in", "bet_or_raise"]


# Byte tags — kept aligned with `abstraction.actions._BYTE_*` so `decode_history`
# (which goes through `byte_to_action_type`) still produces valid ActionTypes.
_BYTE_FOLD: Final[int] = 0x00
_BYTE_CHECK: Final[int] = 0x01
_BYTE_CALL: Final[int] = 0x02
_BYTE_BET_33: Final[int] = 0x10
_BYTE_BET_66: Final[int] = 0x11
_BYTE_BET_100: Final[int] = 0x12
_BYTE_BET_150: Final[int] = 0x13
_BYTE_ALL_IN: Final[int] = 0x14
_BYTE_RAISE_2_5X: Final[int] = 0x20
_BYTE_RAISE_3_5X: Final[int] = 0x21


def position_from_seats(button_seat: int, hero_seat: int, table_size: int) -> int:
    """SB-relative position. In heads-up the button IS the SB; in 3+ the SB is button+1.

    Spec §C: 6-/8-/9-max only — table_size=2 is accepted defensively for HU
    correctness but the runtime schema rejects it upstream.
    """
    sb_seat = button_seat if table_size == 2 else (button_seat + 1) % table_size
    return (hero_seat - sb_seat) % table_size


def effective_stack(hero_stack: int, remaining_opp_stacks: list[int]) -> int:
    """min(hero, max opponent who can still call us off)."""
    if not remaining_opp_stacks:
        return hero_stack
    return min(hero_stack, max(remaining_opp_stacks))


def stack_bucket_from_eff_bb(eff_bb: int) -> int:
    """Bin an effective stack (in BB) into a §C bucket id, 0..9."""
    for i, boundary in enumerate(STACK_BUCKET_BOUNDARIES):
        if eff_bb < boundary:
            return i
    return len(STACK_BUCKET_BOUNDARIES)


def encode_action_history_byte(
    *,
    action_kind: ActionKind,
    amount_chips: int,
    to_call_at_decision: int,
    pot_at_decision: int,
    street: int,
    bb: int,
) -> int:
    """Map one observable action into its §C history byte.

    `action_kind` disambiguates fold/check/call/all-in (which chip math can't
    distinguish from each other). For bets and raises the function classifies
    purely by chip ratios:

      - preflop raise:   amount / max(to_call, bb)
          ≤ 2.95         → RAISE_2_5X (0x20)
          > 2.95         → RAISE_3_5X (0x21)
      - postflop bet:    (amount - to_call) / max(pot, 1)
          < 0.5          → BET_33  (0x10)
          < 0.83         → BET_66  (0x11)
          < 1.25         → BET_100 (0x12)
          ≥ 1.25         → BET_150 (0x13)
        The subtraction is necessary because `_legal_postflop` defines
        `target = to_call + round(frac * pot)`, so a check-raise BET_66 with
        to_call=20, pot=100 commits 86 chips (20 + 66). Dividing 86/100 lands
        in BET_100; (86 - 20)/100 = 0.66 correctly lands in BET_66.

    The preflop denominator uses `to_call_at_decision` (the chips the actor
    must put in to match the current bet), not `bb`. This is what fixes the
    3-bet+ classification: a RAISE_2_5X over a 25-chip open commits 62 chips,
    62/25 = 2.5 → RAISE_2_5X, as intended. The old `amount/bb` heuristic
    classified the same action as RAISE_3_5X.

    The 2.95 threshold (rather than 3.0) leaves the half-bet-of-headroom for
    rounding: round(2.5 * to_call) / to_call can hit 2.95… for to_call=20.
    Practical raises (2.5x, 3.5x) land cleanly on either side.
    """
    if action_kind == "fold":
        return _BYTE_FOLD
    if action_kind == "check":
        return _BYTE_CHECK
    if action_kind == "call":
        return _BYTE_CALL
    if action_kind == "all-in":
        return _BYTE_ALL_IN
    # bet_or_raise: classify by chip math.
    if street == 0:
        denom = max(to_call_at_decision, bb, 1)
        ratio = amount_chips / denom
        return _BYTE_RAISE_2_5X if ratio <= 2.95 else _BYTE_RAISE_3_5X
    raise_over_call = max(amount_chips - to_call_at_decision, 0)
    frac = raise_over_call / max(pot_at_decision, 1)
    if frac < 0.5:
        return _BYTE_BET_33
    if frac < 0.83:
        return _BYTE_BET_66
    if frac < 1.25:
        return _BYTE_BET_100
    return _BYTE_BET_150


__all__ = [
    "NUM_STACK_BUCKETS",
    "STACK_BUCKET_BOUNDARIES",
    "ActionKind",
    "effective_stack",
    "encode_action_history_byte",
    "position_from_seats",
    "stack_bucket_from_eff_bb",
]
