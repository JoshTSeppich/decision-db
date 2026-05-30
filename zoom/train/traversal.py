"""External-sampling MCCFR traversal with an opponent-injection seam (Component 3).

This is a NEW traversal — `src/pokerbot/training/traversal.py` is frozen and has no
seam for substituting opponent actions, which is exactly what the mixed-opponent
fine-tune needs. In self-play mode (``opponent_policy=None``) this is line-for-line
equivalent to the reference `pokerbot.training.external_sampling_traversal` (proven
by the differential test), so Kuhn still converges to Nash. When an
``opponent_policy`` is supplied, non-traverser nodes draw their action from it
(the Component 2 archetype pool) instead of from self-play — that is the seam that
turns equilibrium self-play into a best-response fine-tune against fixed
tight-passive opponents.

Reuses `regret_match`, `Reservoir`, and the `Game` ABC read-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import torch

from pokerbot.training.nets import regret_match
from pokerbot.training.traversal import Reservoir, TraversalStats

if TYPE_CHECKING:
    import random
    from collections.abc import Mapping, Sequence

    from pokerbot.training.game import Game

__all__ = ["OpponentPolicy", "Reservoir", "TraversalStats", "external_sampling_traversal"]

StateT = TypeVar("StateT")


class OpponentPolicy(Protocol):
    """Maps a non-traverser decision to a probability distribution over legal
    action indices. The seam: when provided, opponent nodes sample from this
    instead of from the self-play regret-matched strategy.

    `game`/`state` are typed `Any` so concrete adapters (e.g. the NLHE archetype
    adapter, which narrows `state` to `NLHEState`) satisfy the protocol structurally
    without fighting the traversal's `StateT` generic.
    """

    def __call__(self, game: Any, state: Any, actor: int, /) -> Mapping[int, float]: ...


def external_sampling_traversal(
    game: Game[StateT],
    state: StateT,
    traverser: int,
    advantage_nets: Sequence[torch.nn.Module],
    advantage_reservoir: Reservoir,
    policy_reservoir: Reservoir,
    iter_t: int,
    rng: random.Random,
    *,
    opponent_policy: OpponentPolicy | None = None,
    stats: TraversalStats | None = None,
    opp_actions: list[tuple[int, int]] | None = None,
) -> float:
    """One external-sampling MCCFR traversal.

    Returns the traverser's expected reward from `state`. Adds advantage samples at
    traverser nodes. At opponent nodes: with ``opponent_policy=None`` it samples
    from the self-play strategy and records a policy sample (standard Deep CFR);
    with an ``opponent_policy`` it samples from that fixed policy and records NO
    policy sample (the opponent is not a learner). ``opp_actions`` (if given)
    collects ``(actor, action)`` at every opponent node for seam-verification tests.
    """
    if game.is_terminal(state):
        if stats is not None:
            stats.terminal_visits += 1
        return float(game.terminal_reward(state).rewards[traverser])

    actor = game.current_player(state)
    features = game.infoset_features(state)
    mask_tuple = game.legal_mask(state)
    mask = torch.tensor(mask_tuple, dtype=torch.float32)
    if mask.sum().item() == 0:
        raise RuntimeError(f"no legal actions at infoset {game.infoset_key(state)!r}")

    with torch.no_grad():
        advantages = advantage_nets[actor](features.unsqueeze(0)).squeeze(0)
    strategy = regret_match(advantages.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

    if actor == traverser:
        action_values = torch.zeros(game.num_actions, dtype=torch.float32)
        for a in game.legal_actions(state):
            child = game.apply_action(state, a, rng)
            action_values[a] = external_sampling_traversal(
                game,
                child,
                traverser,
                advantage_nets,
                advantage_reservoir,
                policy_reservoir,
                iter_t,
                rng,
                opponent_policy=opponent_policy,
                stats=stats,
                opp_actions=opp_actions,
            )
        expected = (strategy * action_values).sum().item()
        regret = (action_values - expected) * mask
        advantage_reservoir.add(
            features, mask, regret, float(iter_t), rng, infoset_key=game.infoset_key(state)
        )
        if stats is not None:
            stats.advantage_samples += 1
        return expected

    # ── Opponent node ──
    legal_list = list(game.legal_actions(state))
    sampled = _sample_opponent_action(
        game, state, actor, legal_list, strategy, opponent_policy, rng
    )
    if opponent_policy is None:
        # Self-play: the opponent is a learner; record its current strategy for the
        # policy-net average (matches the reference traversal exactly).
        policy_reservoir.add(
            features,
            mask,
            strategy.detach(),
            float(iter_t),
            rng,
            infoset_key=game.infoset_key(state),
        )
        if stats is not None:
            stats.policy_samples += 1
    if opp_actions is not None:
        opp_actions.append((actor, sampled))
    child = game.apply_action(state, sampled, rng)
    return external_sampling_traversal(
        game,
        child,
        traverser,
        advantage_nets,
        advantage_reservoir,
        policy_reservoir,
        iter_t,
        rng,
        opponent_policy=opponent_policy,
        stats=stats,
        opp_actions=opp_actions,
    )


def _sample_opponent_action(
    game: Game[StateT],
    state: StateT,
    actor: int,
    legal_list: list[int],
    strategy: torch.Tensor,
    opponent_policy: OpponentPolicy | None,
    rng: random.Random,
) -> int:
    if opponent_policy is not None:
        # SEAM: the opponent is a fixed (archetype) policy, not a learner. Sample
        # its action over the legal set; if it returns no mass on legal actions,
        # fall back to uniform-legal so the traversal never stalls.
        dist = opponent_policy(game, state, actor)
        weights = [max(0.0, float(dist.get(a, 0.0))) for a in legal_list]
        if sum(weights) <= 0:
            return rng.choice(legal_list)
        return rng.choices(legal_list, weights=weights, k=1)[0]

    # Self-play: sample from the current regret-matched strategy (matches the
    # reference traversal's rng-consumption order exactly).
    probs = [float(strategy[a].item()) for a in legal_list]
    total = sum(probs)
    if total <= 0:
        return rng.choice(legal_list)
    return rng.choices(legal_list, weights=probs, k=1)[0]
