"""GatedNLHEGame — SimpleNLHEGame with the Component 1 SPR ALL_IN gate applied.

The fine-tune must train the learner on the SAME gated action set the deployed
policy will use, so the blueprint can't learn to shove at 100bb. `SimpleNLHEGame`
is frozen and exposes the ungated legality, so this thin subclass overrides
`legal_actions` to drop ALL_IN at deep SPR via the single `all_in_allowed`
predicate (one source of truth for the continuum — never re-derived here).

It also exposes `agent_spot`, which projects the current pokerkit state into the
`AgentSpot` the Component 2 archetype agents consume — the bridge the opponent
adapter uses. Card/street extraction reuses the frozen game's own private helpers
(`_pk_card_to_int`, `_STREET_NAMES`) read-only, so both sides encode cards
identically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokerbot.abstraction import ActionType
from pokerbot.abstraction.encoding import position_from_seats
from pokerbot.training.nlhe_game import (
    _STREET_NAMES,  # read-only reuse: street-index → name, identical to the game
    SimpleNLHEGame,
    _pk_card_to_int,  # read-only reuse: pokerkit Card → 0..51 int, identical encoding
)
from zoom.abstraction_gate import DEFAULT_SPR_CAP, all_in_allowed
from zoom.agents import AgentSpot

if TYPE_CHECKING:
    from pokerbot.abstraction import AbstractionTables, Street
    from pokerbot.training.nlhe_game import NLHEState

_AGGRESSIVE_NON_ALL_IN = frozenset(
    {
        ActionType.BET_33,
        ActionType.BET_66,
        ActionType.BET_100,
        ActionType.BET_150,
        ActionType.RAISE_2_5X,
        ActionType.RAISE_3_5X,
    }
)


class GatedNLHEGame(SimpleNLHEGame):
    """`SimpleNLHEGame` whose `legal_actions` are SPR-gated (Component 1)."""

    def __init__(
        self,
        abstraction: AbstractionTables,
        *,
        blinds: tuple[int, int] = (5, 10),
        starting_stack: int = 1000,
        table_size: int = 6,
        spr_cap: float = DEFAULT_SPR_CAP,
    ) -> None:
        super().__init__(
            abstraction,
            blinds=blinds,
            starting_stack=starting_stack,
            table_size=table_size,  # type: ignore[arg-type]
        )
        self.spr_cap = spr_cap

    def _gate_context(self, state: NLHEState) -> tuple[int, int, int]:
        """(pot, to_call, stack) for the current actor — the gate's inputs."""
        pk = state.pk_state
        actor = self.current_player(state)
        stack = int(pk.stacks[actor])
        to_call = int(pk.checking_or_calling_amount or 0)
        pot = int(sum(state.initial_stacks) - sum(pk.stacks))
        return pot, to_call, stack

    def legal_actions(self, state: NLHEState) -> tuple[int, ...]:
        base = self._legal_abstract_at(state)  # ungated abstract actions (pk-gated)
        has_other_aggression = any(a.type in _AGGRESSIVE_NON_ALL_IN for a in base)
        pot, to_call, stack = self._gate_context(state)
        if all_in_allowed(
            pot,
            to_call,
            stack,
            has_other_aggression=has_other_aggression,
            spr_cap=self.spr_cap,
        ):
            return tuple(int(a.type) for a in base)
        return tuple(int(a.type) for a in base if a.type != ActionType.ALL_IN)

    def agent_spot(self, state: NLHEState) -> AgentSpot:
        """Project the current actor's view into an `AgentSpot` for a ScriptedAgent."""
        pk = state.pk_state
        actor = self.current_player(state)
        hole_pk = pk.hole_cards[actor]
        hole = (_pk_card_to_int(hole_pk[0]), _pk_card_to_int(hole_pk[1]))
        raw_board = pk.board_cards
        board = tuple(_pk_card_to_int(c) for inner in raw_board for c in inner) if raw_board else ()
        street: Street = _STREET_NAMES[int(pk.street_index)]
        pot, to_call, stack = self._gate_context(state)
        bet_actor = int(pk.bets[actor])
        min_raise_to = int(pk.min_completion_betting_or_raising_to_amount or 0)
        min_raise = max(min_raise_to - bet_actor, 0)
        position = position_from_seats(self.table_size - 1, actor, self.table_size)
        return AgentSpot(
            hole=hole,
            board=board,
            street=street,
            position=position,
            pot=pot,
            to_call=to_call,
            stack=stack,
            min_raise=min_raise,
            table_size=self.table_size,
        )


__all__ = ["GatedNLHEGame"]
