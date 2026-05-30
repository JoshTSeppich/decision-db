"""The three scripted opponents: NIT, STATION, TAG (Component 2).

Each is a deterministic function of the spot. Range thresholds are expressed as
preflop percentiles (0 = strongest) and postflop made-hand strengths in [0, 1].
The numbers are tuned so each agent lands inside the live classifier's signature
band (`pokerbot.opponent.archetype.ArchetypeClassifier`) — and is therefore
classified as its own archetype — over the documented spot distribution in
`tests/zoom/test_agents.py`. Lock 2: these bands are the gate; the thresholds here
serve the bands, never the other way around.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from pokerbot.opponent.archetype import Archetype
from zoom.agents.base import (
    _BETS,
    _OPEN_RAISES,
    _RERAISES,
    AgentSpot,
    ScriptedAgent,
    call_action,
    fold_or_check,
    postflop_strength,
    preflop_rank,
    raise_or_shove,
)

if TYPE_CHECKING:
    from pokerbot.abstraction import AbstractAction


class NitAgent(ScriptedAgent):
    """Weak-tight: enters few pots, but acts aggressively (value) when it does.

    VPIP < 18%, PFR < 12%, AF > 1.5. Opens a tight range, 3-bets only premiums,
    folds the rest; postflop it value-bets/raises strong hands and folds weak ones,
    calling only a thin made-hand band — so aggression outweighs calling (high AF).
    """

    archetype: ClassVar[Archetype] = Archetype.NIT

    def _preflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        r = preflop_rank(spot.hole)
        if spot.to_call > 0:  # facing a raise
            if r < 0.045:
                return raise_or_shove(legal, _RERAISES)
            if r < 0.075:
                return call_action(legal)
            return fold_or_check(legal)
        # first-in / BB option — nits don't limp; open the tight top, else check.
        if r < 0.105:
            return raise_or_shove(legal, _OPEN_RAISES)
        return fold_or_check(legal)

    def _postflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        s = postflop_strength(spot.hole, spot.board)
        if spot.to_call > 0:  # facing a bet
            if s >= 0.78:
                return raise_or_shove(legal, _BETS)
            if s >= 0.60:
                return call_action(legal)
            return fold_or_check(legal)
        # checked to: value-bet strong, otherwise check.
        if s >= 0.55:
            return raise_or_shove(legal, _BETS)
        return fold_or_check(legal)


class StationAgent(ScriptedAgent):
    """Calling station: plays many hands, calls down, never bluffs.

    VPIP > 35%, AF < 1.0, fold-to-cbet < 40%. Raises only premiums/monsters (never
    a bluff), calls a very wide range, and never folds a bet it is priced into.
    """

    archetype: ClassVar[Archetype] = Archetype.STATION

    def _preflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        r = preflop_rank(spot.hole)
        if r < 0.05:  # premium → raise for value (never a bluff)
            preferred = _RERAISES if spot.to_call > 0 else _OPEN_RAISES
            return raise_or_shove(legal, preferred)
        if spot.to_call > 0:  # facing a raise: call very wide, fold only trash
            return call_action(legal) if r < 0.82 else fold_or_check(legal)
        # first-in: passive — limp via the free option / check, no opening aggression.
        return fold_or_check(legal)

    def _postflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        s = postflop_strength(spot.hole, spot.board)
        if spot.to_call > 0:  # facing a bet
            if s >= 0.85:  # monster → raise for value (still not a bluff)
                return raise_or_shove(legal, _BETS)
            pot_odds = spot.to_call / (spot.pot + spot.to_call)
            if s >= pot_odds or s >= 0.20:  # priced in, or any showdown value → call
                return call_action(legal)
            return fold_or_check(legal)  # only pure air vs a big bet folds
        # checked to: almost always check; bet only a monster (thin, rare).
        if s >= 0.90:
            return raise_or_shove(legal, _BETS)
        return fold_or_check(legal)


class TagAgent(ScriptedAgent):
    """Tight-aggressive: a genuinely tight range, played aggressively.

    VPIP 18-26%, PFR 14-22%, AF > 2.0. Opens tight, 3-bets a real value/bluff mix,
    and postflop bets/raises far more than it calls.
    """

    archetype: ClassVar[Archetype] = Archetype.TAG

    def _preflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        r = preflop_rank(spot.hole)
        if spot.to_call > 0:  # facing a raise
            if r < 0.18:
                return raise_or_shove(legal, _RERAISES)
            if r < 0.24:
                return call_action(legal)
            return fold_or_check(legal)
        if r < 0.40:  # open tight, aggressively
            return raise_or_shove(legal, _OPEN_RAISES)
        return fold_or_check(legal)

    def _postflop(self, spot: AgentSpot, legal: list[AbstractAction]) -> AbstractAction:
        s = postflop_strength(spot.hole, spot.board)
        if spot.to_call > 0:  # facing a bet
            if s >= 0.55:
                return raise_or_shove(legal, _BETS)
            if s >= 0.45:
                return call_action(legal)
            return fold_or_check(legal)
        # checked to: c-bet a wide value+equity range.
        if s >= 0.40:
            return raise_or_shove(legal, _BETS)
        return fold_or_check(legal)


def build_archetype_pool() -> list[ScriptedAgent]:
    """The Stage-1 fine-tune opponent pool: real tight-passive players, no self-play."""
    return [NitAgent(), TagAgent(), StationAgent()]


__all__ = ["NitAgent", "StationAgent", "TagAgent", "build_archetype_pool"]
