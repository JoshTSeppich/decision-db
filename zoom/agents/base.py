"""Shared scaffolding for the scripted archetype agents (Component 2).

`AgentSpot` is the minimal decision context an agent needs; `ScriptedAgent` is the
base every archetype implements. All legality flows through the Component 1 gate
(`zoom.abstraction_gate.legal_abstract_actions_gated`) — agents pick *among* gated
actions and never re-derive what's legal, so their short-stack behavior inherits
the SPR continuum for free.

Hand strength is deterministic and dependency-free:
  * preflop — `preflop_rank` reuses `default_policy.preflop_percentile` (Chen-based,
    0 = strongest), so agents don't ship a second preflop ranking.
  * postflop — `postflop_strength` is a cheap made-hand heuristic in [0, 1]. It is
    NOT a solver equity; it only needs to rank made hands well enough to drive
    value-bet / call-down / fold behavior and to be a stable, reproducible proxy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pokerbot.abstraction import ActionType, canonical_hand
from pokerbot.runtime.default_policy import preflop_percentile
from zoom.abstraction_gate import legal_abstract_actions_gated

if TYPE_CHECKING:
    import random

    from pokerbot.abstraction import AbstractAction, Street
    from pokerbot.opponent.archetype import Archetype


@dataclass(frozen=True, slots=True)
class AgentSpot:
    """One decision an agent faces.

    `position` is SB-relative (0 = SB) per the project's encoding. `to_call == 0`
    means no bet to face (a free check / the BB option); `to_call > 0` means facing
    a bet or raise.
    """

    hole: tuple[int, int]
    board: tuple[int, ...]
    street: Street
    position: int
    pot: int
    to_call: int
    stack: int
    min_raise: int
    table_size: int = 3
    # Effective stack = min(hero, max remaining opponent), in chips — the quantity the
    # training/export keys the DB by (see nlhe_game.py / encoding.effective_stack). The
    # DB lookup must bucket by THIS, not the hero's raw stack, or the chip leader's spots
    # land in an over-deep (empty) bucket and miss. None → fall back to `stack` (callers
    # that don't know opponent stacks, e.g. scripted agents, are unaffected).
    effective_stack: int | None = None
    # §C-encoded betting history for the decision (production's InfoSet history key).
    # The DB lookup must try an EXACT (history-aware) match first, mirroring the
    # production RuntimeAdapter — a history-blind lookup reads a non-representative row
    # for positions that have no empty-history infoset (SB/BB act after the button).
    # Empty (the default) means "no history supplied" → fall back to the history-agnostic
    # nearest-neighbor, as scripted agents / screen probes do.
    history: bytes = b""


# Aggressive (bet/raise) preferences, in the order an agent tries to use them.
_OPEN_RAISES: tuple[ActionType, ...] = (ActionType.RAISE_2_5X, ActionType.RAISE_3_5X)
_RERAISES: tuple[ActionType, ...] = (ActionType.RAISE_3_5X, ActionType.RAISE_2_5X)
_BETS: tuple[ActionType, ...] = (
    ActionType.BET_66,
    ActionType.BET_100,
    ActionType.BET_33,
    ActionType.BET_150,
)


def preflop_rank(hole: tuple[int, int]) -> float:
    """Percentile rank of `hole` in [0, 1); 0 = strongest. Reuses default_policy."""
    canon, _ = canonical_hand(hole, ())
    return preflop_percentile(canon)


def postflop_strength(hole: tuple[int, int], board: tuple[int, ...]) -> float:
    """Cheap made-hand strength in [0, 1] (higher = stronger).

    Tiers: set/quads ≥ 0.92, two pair 0.85, overpair 0.80, top pair 0.62, other
    pair 0.48, overcards 0.30, air 0.14. A rough but monotone proxy — enough to
    drive value-bet / call-down / fold decisions deterministically.
    """
    if not board:
        return 0.5
    hole_ranks = sorted((hole[0] >> 2, hole[1] >> 2), reverse=True)
    board_ranks = [c >> 2 for c in board]
    pocket_pair = hole_ranks[0] == hole_ranks[1]
    hole_hits = [r for r in set(hole_ranks) if r in board_ranks]
    counts = Counter([*hole_ranks, *board_ranks])
    top_board = max(board_ranks)

    # set/quads: a hole rank appears 3+ times total (pocket pair hitting, or trips).
    if any(r in counts and counts[r] >= 3 for r in set(hole_ranks)):
        return 0.92
    if len(hole_hits) >= 2:  # both hole cards paired the board
        return 0.85
    if pocket_pair:
        return 0.80 if hole_ranks[0] > top_board else 0.55
    if hole_hits:
        return 0.62 if max(hole_hits) == top_board else 0.48
    if hole_ranks[0] > top_board:  # two overcards / one overcard
        return 0.30
    return 0.14


class ScriptedAgent:
    """Base class for a deterministic, rule-based opponent.

    Subclasses set `archetype` and implement `_preflop` / `_postflop`, returning a
    *desired* line; `action` resolves that desire against the gated legal set.
    """

    archetype: ClassVar[Archetype]

    def action(self, spot: AgentSpot, rng: random.Random | None = None) -> AbstractAction:  # noqa: ARG002
        legal = legal_abstract_actions_gated(
            spot.pot, spot.to_call, spot.stack, spot.min_raise, spot.street
        )
        if spot.street == "preflop":
            return self._preflop(spot, legal)
        return self._postflop(spot, legal)

    # subclasses override
    def _preflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        raise NotImplementedError

    def _postflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        raise NotImplementedError


# ─────────── selection helpers (operate on the gated legal set) ───────────


def pick(legal: list[AbstractAction], preferred: tuple[ActionType, ...]) -> AbstractAction | None:
    """First legal action whose type is in `preferred` (in preference order)."""
    by_type = {a.type: a for a in legal}
    for t in preferred:
        if t in by_type:
            return by_type[t]
    return None


def raise_or_shove(
    legal: list[AbstractAction], preferred: tuple[ActionType, ...]
) -> AbstractAction:
    """An aggressive action: a preferred raise/bet size if legal, else ALL_IN if the
    gate left it as the only aggression (short stack), else fall back to calling.
    """
    chosen = pick(legal, preferred)
    if chosen is not None:
        return chosen
    all_in = pick(legal, (ActionType.ALL_IN,))
    if all_in is not None:
        return all_in
    return call_action(legal)


def call_action(legal: list[AbstractAction]) -> AbstractAction:
    """CHECK_CALL — always present in any legal set."""
    cc = pick(legal, (ActionType.CHECK_CALL,))
    assert cc is not None, "CHECK_CALL must always be legal"
    return cc


def fold_or_check(legal: list[AbstractAction]) -> AbstractAction:
    """FOLD if facing a bet (FOLD legal), else CHECK (free) — never an invalid fold."""
    folded = pick(legal, (ActionType.FOLD,))
    return folded if folded is not None else call_action(legal)


__all__ = [
    "_BETS",
    "_OPEN_RAISES",
    "_RERAISES",
    "AgentSpot",
    "ScriptedAgent",
    "call_action",
    "fold_or_check",
    "pick",
    "postflop_strength",
    "preflop_rank",
    "raise_or_shove",
]
