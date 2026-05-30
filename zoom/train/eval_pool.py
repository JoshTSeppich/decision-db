"""EV-vs-pool measurement — the proof that fine-tuning actually exploits the pool.

Plays full hands where one seat (the hero) acts on the current advantage-net
strategy and the other seats are ScriptedAgents from the archetype pool, then
returns the hero's mean chip delta per hand in big blinds. Deterministic given a
seed. Used by the BR-improves-vs-pool gate: the value must strictly rise after a
few fine-tune iterations from a fixed-seed checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pokerbot.abstraction import ActionType
from pokerbot.training.nets import regret_match

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

    from torch import nn

    from zoom.agents import ScriptedAgent
    from zoom.train.game import GatedNLHEGame


def _advantage_action(
    game: GatedNLHEGame, advantage_nets: Sequence[nn.Module], state: object, rng: random.Random
) -> int:
    """Sample the hero's action index from the current advantage-net strategy."""
    actor = game.current_player(state)  # type: ignore[arg-type]
    features = game.infoset_features(state)  # type: ignore[arg-type]
    mask = torch.tensor(game.legal_mask(state), dtype=torch.float32)  # type: ignore[arg-type]
    with torch.no_grad():
        adv = advantage_nets[actor](features.unsqueeze(0)).squeeze(0)
    strategy = regret_match(adv.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
    legal = list(game.legal_actions(state))  # type: ignore[arg-type]
    weights = [float(strategy[a].item()) for a in legal]
    if sum(weights) <= 0:
        return rng.choice(legal)
    return rng.choices(legal, weights=weights, k=1)[0]


def evaluate_vs_pool(
    game: GatedNLHEGame,
    advantage_nets: Sequence[nn.Module],
    pool: Sequence[ScriptedAgent],
    *,
    n_hands: int,
    seed: int,
) -> float:
    """Mean hero chip delta per hand (in BB), hero seat rotating over the table.

    Hero plays the advantage-net strategy; the other seats play archetypes drawn
    from `pool`. Fully deterministic given `seed`.
    """
    import random

    rng = random.Random(seed)
    check_call = int(ActionType.CHECK_CALL)
    bb = float(game.blinds[1])
    total = 0.0
    for h in range(n_hands):
        hero = h % game.num_players
        assignment = {seat: rng.choice(pool) for seat in range(game.num_players) if seat != hero}
        state = game.new_initial_state(rng)
        while not game.is_terminal(state):
            actor = game.current_player(state)
            if actor == hero:
                action = _advantage_action(game, advantage_nets, state, rng)
            else:
                idx = int(assignment[actor].action(game.agent_spot(state)).type)
                legal = set(game.legal_actions(state))
                action = idx if idx in legal else check_call
            state = game.apply_action(state, action, rng)
        total += float(game.terminal_reward(state).rewards[hero])
    return total / n_hands / bb


__all__ = ["evaluate_vs_pool"]
