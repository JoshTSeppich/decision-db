"""Hand-rolled fallback policy when both exact + nearest-neighbor lookups miss.

Spec.html §F: "don't crash, don't bleed too much" floor. Expected to fire on
<0.1% of hands after a full training run.

    Preflop: open-raise top 25% of hands UTG, sliding to top 60% on the button;
             3-bet top 5%; fold otherwise.
    Postflop: c-bet 0.66-pot with showdown value, check otherwise;
              call ≤ 1/4 pot with showdown value.

Showdown-value detection without phevaluator: pair-or-better is a pair-or-better
on the board, or a hole card paired with the board, or a pocket pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pokerbot.abstraction import (
    AbstractAction,
    ActionType,
    canonical_hand,
)
from pokerbot.abstraction.preflop import preflop_bucket

if TYPE_CHECKING:
    from pokerbot.abstraction import Card, Street


# ─────────── preflop strength ranking ───────────


def _chen_score(canon_hole: tuple[int, int]) -> float:
    """Chen-formula-ish strength score. Higher = stronger.

    Not perfectly accurate (modern solvers re-rank), but monotonic enough for a
    rough percentile bucket on the 169 lossless preflop classes.
    """
    c1, c2 = canon_hole
    r1, s1 = c1 >> 2, c1 & 3
    r2, s2 = c2 >> 2, c2 & 3

    # Chen values: A=10, K=8, Q=7, J=6, T=5, 9=4.5, 8=4, …, 2=1.
    chen_value: dict[int, float] = {
        12: 10.0,
        11: 8.0,
        10: 7.0,
        9: 6.0,
        8: 5.0,
        7: 4.5,
        6: 4.0,
        5: 3.5,
        4: 3.0,
        3: 2.5,
        2: 2.0,
        1: 1.5,
        0: 1.0,
    }
    high = max(r1, r2)
    low = min(r1, r2)

    if r1 == r2:
        return max(chen_value[high] * 2.0, 5.0)

    score = chen_value[high]
    if s1 == s2:
        score += 2.0
    gap = high - low - 1
    score -= [0.0, 1.0, 2.0, 4.0][gap] if gap < 4 else 5.0
    if gap <= 1 and high < 10:  # low connector bonus
        score += 1.0
    return score


def _build_preflop_percentile() -> dict[int, float]:
    """Map preflop bucket id → percentile in [0, 1) where 0 = strongest."""
    scores: dict[int, float] = {}
    # 13 pairs
    for r in range(13):
        canon = (r * 4 + 0, r * 4 + 1)  # both lowest suit ids; will sort fine
        canon_sorted = (min(canon), max(canon))
        scores[preflop_bucket(canon_sorted)] = _chen_score(canon_sorted)
    # 78 suited
    for high in range(1, 13):
        for low in range(high):
            canon = (low * 4 + 0, high * 4 + 0)
            scores[preflop_bucket(canon)] = _chen_score(canon)
    # 78 offsuit
    for high in range(1, 13):
        for low in range(high):
            canon = (low * 4 + 0, high * 4 + 1)
            scores[preflop_bucket(canon)] = _chen_score(canon)
    sorted_buckets = sorted(scores, key=lambda b: -scores[b])
    return {b: i / 169.0 for i, b in enumerate(sorted_buckets)}


_PREFLOP_PERCENTILE: Final[dict[int, float]] = _build_preflop_percentile()


def preflop_percentile(canon_hole: tuple[int, int]) -> float:
    """Approximate percentile rank of a canonical hole (0 = strongest)."""
    return _PREFLOP_PERCENTILE[preflop_bucket(canon_hole)]


# ─────────── policy entry point ───────────


def _open_threshold(position: int, table_size: int) -> float:
    """Slide from top 25% in earliest position to top 60% on the button."""
    if table_size <= 2:
        return 0.60  # heads-up: button is wide
    # position 0 = SB, position table_size-1 = BTN
    last = table_size - 1
    frac = position / max(last, 1)
    return 0.25 + (0.60 - 0.25) * frac


_THREE_BET_THRESHOLD: Final[float] = 0.05


def _has_showdown_value(hole: tuple[Card, Card], board: tuple[Card, ...]) -> bool:
    """Cheap heuristic: any pair (pocket pair, or paired board, or hole+board)."""
    r1, r2 = hole[0] >> 2, hole[1] >> 2
    if r1 == r2:
        return True
    board_ranks = [c >> 2 for c in board]
    if r1 in board_ranks or r2 in board_ranks:
        return True
    return len(set(board_ranks)) < len(board_ranks)  # paired board


def default_policy_action(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    street: Street,
    *,
    table_size: int,
    position: int,
    pot: int,
    to_call: int,
    stack: int,
) -> AbstractAction:
    """Pick a single AbstractAction using Spec.html §F's fallback heuristic."""
    if street == "preflop":
        return _preflop(hole, table_size, position, to_call, stack)
    return _postflop(hole, board, pot, to_call, stack)


def _preflop(
    hole: tuple[Card, Card],
    table_size: int,
    position: int,
    to_call: int,
    stack: int,
) -> AbstractAction:
    canon_hole, _ = canonical_hand(hole, ())
    pct = preflop_percentile(canon_hole)

    if to_call == 0:
        # open: action folds to us
        if pct < _open_threshold(position, table_size):
            target = max(round(2.5 * to_call), 1)
            return AbstractAction(ActionType.RAISE_2_5X, min(target, stack))
        return AbstractAction(ActionType.CHECK_CALL, 0)  # BB option to check

    # facing a raise (or unopened BB completion)
    if pct < _THREE_BET_THRESHOLD:
        target = round(3.5 * to_call)
        return AbstractAction(ActionType.RAISE_3_5X, min(target, stack))
    if pct < _open_threshold(position, table_size) / 2:
        return AbstractAction(ActionType.CHECK_CALL, min(to_call, stack))
    return AbstractAction(ActionType.FOLD, 0)


def _postflop(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    pot: int,
    to_call: int,
    stack: int,
) -> AbstractAction:
    showdown = _has_showdown_value(hole, board)
    if to_call == 0:
        if showdown:
            target = to_call + round(0.66 * pot)
            return AbstractAction(ActionType.BET_66, min(target, stack))
        return AbstractAction(ActionType.CHECK_CALL, 0)
    if showdown and to_call <= max(pot // 4, 1):
        return AbstractAction(ActionType.CHECK_CALL, min(to_call, stack))
    return AbstractAction(ActionType.FOLD, 0)


__all__ = [
    "default_policy_action",
    "preflop_percentile",
]
