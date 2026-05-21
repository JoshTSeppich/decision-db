"""Local Best Response (Spec.html §E).

For 2-player games small enough to enumerate (Kuhn, Leduc subgames) we compute
an EXACT best response by recursive tree search — that's the test path.

For larger games (NLHE), the same recursion is used but the caller must supply
a sampled or depth-limited tree, plus a leaf-rollout heuristic. v1 ships the
exact version; depth-limited support is a v2 follow-up.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import torch

from pokerbot.training.nets import regret_match

if TYPE_CHECKING:
    from pokerbot.training.game import Game


StateT = TypeVar("StateT")

# Strategy function: (state, acting_player) -> probability vector over actions.
StrategyFn = Callable[..., torch.Tensor]


_DUMMY_RNG = random.Random(0)  # apply_action shouldn't need RNG for games with no in-play chance


def _best_response_value(
    game: Game[StateT],
    state: StateT,
    strategy_fn: StrategyFn,
    br_player: int,
) -> float:
    if game.is_terminal(state):
        return float(game.terminal_reward(state).rewards[br_player])
    actor = game.current_player(state)
    legal = game.legal_actions(state)
    if actor == br_player:
        best = float("-inf")
        for a in legal:
            child = game.apply_action(state, a, _DUMMY_RNG)
            v = _best_response_value(game, child, strategy_fn, br_player)
            best = max(best, v)
        return best
    # Opponent plays per strategy_fn — expected value over their actions.
    strategy = strategy_fn(state, actor)
    total = 0.0
    for a in legal:
        prob = float(strategy[a].item() if hasattr(strategy, "item") else strategy[a])
        if prob <= 0:
            continue
        child = game.apply_action(state, a, _DUMMY_RNG)
        total += prob * _best_response_value(game, child, strategy_fn, br_player)
    return total


def local_best_response(
    game: Game[StateT],
    strategy_fn: StrategyFn,
    br_player: int,
    initial_states: list[tuple[StateT, float]],
) -> float:
    """Expected value to `br_player` of best-responding to the opponent's strategy.

    `initial_states` is the chance distribution at the root (state, weight pairs;
    weights should sum to ≈ 1). Returned value is in *raw game-reward units* —
    callers convert to mbb/hand if they need that scale.
    """
    total = 0.0
    weight_sum = 0.0
    for state, weight in initial_states:
        total += weight * _best_response_value(game, state, strategy_fn, br_player)
        weight_sum += weight
    if weight_sum == 0:
        raise ValueError("initial_states had zero total weight")
    return total / weight_sum


def exploitability(
    game: Game[StateT],
    strategy_fn: StrategyFn,
    initial_states: list[tuple[StateT, float]],
) -> float:
    """Sum of BR values across all players (= 2 * NashConv for 2p zero-sum)."""
    return sum(
        local_best_response(game, strategy_fn, p, initial_states) for p in range(game.num_players)
    )


# ───────── strategy-fn builders ─────────


def make_advantage_strategy_fn(
    game: Game[StateT], advantage_nets: list[torch.nn.Module]
) -> StrategyFn:
    """Wrap a list of per-player advantage nets into a regret-matched StrategyFn."""

    def strategy_fn(state: StateT, actor: int) -> torch.Tensor:
        features = game.infoset_features(state)
        mask = torch.tensor(game.legal_mask(state), dtype=torch.float32)
        with torch.no_grad():
            adv = advantage_nets[actor](features.unsqueeze(0)).squeeze(0)
        return regret_match(adv.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

    return strategy_fn


def make_uniform_strategy_fn(game: Game[StateT]) -> StrategyFn:
    """Strategy that plays uniform-random over legal actions."""

    def strategy_fn(state: StateT, actor: int) -> torch.Tensor:  # noqa: ARG001
        mask = torch.tensor(game.legal_mask(state), dtype=torch.float32)
        total = mask.sum().clamp(min=1.0)
        return mask / total

    return strategy_fn


__all__ = [
    "exploitability",
    "local_best_response",
    "make_advantage_strategy_fn",
    "make_uniform_strategy_fn",
]
