"""Adapter: a Component 2 archetype pool → an `OpponentPolicy` for the traversal.

At a non-traverser node the traversal asks the policy for a distribution over legal
action indices. This adapter projects the state into an `AgentSpot` (via
`GatedNLHEGame.agent_spot`), asks the seat's assigned ScriptedAgent for an action,
and returns it as a point mass on that action's index. The game stays the source of
truth for legality: if the agent's chosen type isn't currently legal, the adapter
falls back to CHECK_CALL (always legal).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pokerbot.abstraction import ActionType

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pokerbot.training.nlhe_game import NLHEState
    from zoom.agents import ScriptedAgent
    from zoom.train.game import GatedNLHEGame
    from zoom.train.traversal import OpponentPolicy


def make_archetype_opponent_policy(
    game: GatedNLHEGame,
    assignment: Mapping[int, ScriptedAgent],
) -> OpponentPolicy:
    """Build an `OpponentPolicy` that plays `assignment[actor]` at each seat.

    `assignment` maps a seat index to the ScriptedAgent occupying it this hand
    (the traverser seat is absent — it's the learner, not an opponent).
    """
    check_call = int(ActionType.CHECK_CALL)

    def policy(_game: Any, state: Any, actor: int) -> Mapping[int, float]:
        # `game` (captured, typed) is the GatedNLHEGame being traversed; `state` is
        # its NLHEState. Typed `Any` to match the OpponentPolicy protocol exactly.
        nlhe_state: NLHEState = state
        agent = assignment[actor]
        spot = game.agent_spot(nlhe_state)
        idx = int(agent.action(spot).type)
        legal = set(game.legal_actions(nlhe_state))
        return {idx: 1.0} if idx in legal else {check_call: 1.0}

    return policy


__all__ = ["make_archetype_opponent_policy"]
