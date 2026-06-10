"""External-sampling MCCFR traversal + reservoir buffers (Spec.html §E)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch

from pokerbot.abstraction.actions import ActionType
from pokerbot.training.nets import regret_match

# Aggressive actions for the opponent-aggression-bias knob (Piece 1). Reconstructed
# from the public enum (mirrors actions._BET_LIKE_TYPES) so we don't reach into a
# private. IntEnum members compare/hash equal to their int, so membership tests
# against the int action indices in `legal_actions` work directly.
_AGGRESSIVE_ACTIONS: frozenset[int] = frozenset(
    {
        int(ActionType.BET_33),
        int(ActionType.BET_66),
        int(ActionType.BET_100),
        int(ActionType.BET_150),
        int(ActionType.ALL_IN),
        int(ActionType.RAISE_2_5X),
        int(ActionType.RAISE_3_5X),
    }
)

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

    from pokerbot.training.game import Game


StateT = TypeVar("StateT")


# ───────── Reservoir ─────────


class Reservoir:
    """Standard reservoir sampling buffer (Vitter's Algorithm R)."""

    def __init__(self, capacity: int, feature_dim: int, num_actions: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive: {capacity}")
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.features = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.masks = np.zeros((capacity, num_actions), dtype=np.float32)
        self.targets = np.zeros((capacity, num_actions), dtype=np.float32)
        self.iter_weights = np.zeros(capacity, dtype=np.float32)
        self.infoset_keys: list[bytes] = []  # parallel to filled rows
        self.size = 0
        self.total_seen = 0

    def add(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        iter_weight: float,
        rng: random.Random,
        infoset_key: bytes = b"",
    ) -> None:
        self.total_seen += 1
        if self.size < self.capacity:
            idx = self.size
            self.size += 1
            self.infoset_keys.append(infoset_key)
        else:
            r = rng.randrange(self.total_seen)
            if r >= self.capacity:
                return
            idx = r
            self.infoset_keys[idx] = infoset_key
        self.features[idx] = features.detach().cpu().numpy()
        self.masks[idx] = mask.detach().cpu().numpy()
        self.targets[idx] = target.detach().cpu().numpy()
        self.iter_weights[idx] = iter_weight

    def sample_batch(
        self, batch_size: int, rng: random.Random
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.size == 0:
            return None
        k = min(batch_size, self.size)
        idx = [rng.randrange(self.size) for _ in range(k)]
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.iter_weights[idx]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "features": self.features[: self.size].copy(),
            "masks": self.masks[: self.size].copy(),
            "targets": self.targets[: self.size].copy(),
            "iter_weights": self.iter_weights[: self.size].copy(),
            "infoset_keys": list(self.infoset_keys),
            "size": self.size,
            "total_seen": self.total_seen,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"saved size {size} > capacity {self.capacity}")
        self.features[:size] = state["features"]
        self.masks[:size] = state["masks"]
        self.targets[:size] = state["targets"]
        self.iter_weights[:size] = state["iter_weights"]
        self.infoset_keys = list(state["infoset_keys"])
        self.size = size
        self.total_seen = int(state["total_seen"])

    # ── streaming npz (sidecar) save/load ──
    #
    # The monolithic torch.save path above pickles each reservoir, and
    # state_dict() .copy()s every sliced array first — peak memory during a
    # checkpoint write is roughly live + one full copy + pickle buffer, which
    # OOM-killed earlier runs under the cgroup limit. The pair below avoids
    # both: npz_arrays() returns prefix-slice *views* (no copy) so np.savez can
    # stream them array-by-array, and infoset_keys are packed into a numeric
    # blob + lengths so the npz never needs allow_pickle.

    def npz_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        """Return array *views* (no copy) for streaming into ``np.savez``.

        Keys are namespaced by ``prefix`` so several reservoirs can share one
        sidecar file. The float arrays are contiguous prefix slices, so
        ``np.savez`` writes them without materializing a copy.
        """
        keys = self.infoset_keys
        blob = (
            np.frombuffer(b"".join(keys), dtype=np.uint8)
            if keys
            else np.zeros(0, dtype=np.uint8)
        )
        lens = np.fromiter((len(k) for k in keys), dtype=np.int64, count=len(keys))
        return {
            f"{prefix}_features": self.features[: self.size],
            f"{prefix}_masks": self.masks[: self.size],
            f"{prefix}_targets": self.targets[: self.size],
            f"{prefix}_iter_weights": self.iter_weights[: self.size],
            f"{prefix}_keys_blob": blob,
            f"{prefix}_keys_lens": lens,
            f"{prefix}_meta": np.array(
                [self.size, self.total_seen, self.capacity], dtype=np.int64
            ),
        }

    def load_npz_arrays(self, npz: Any, prefix: str) -> None:
        """Restore reservoir state from an open ``np.load`` archive (no pickle)."""
        size, total_seen, _capacity = (int(x) for x in npz[f"{prefix}_meta"])
        if size > self.capacity:
            raise ValueError(f"saved size {size} > capacity {self.capacity}")
        self.features[:size] = npz[f"{prefix}_features"]
        self.masks[:size] = npz[f"{prefix}_masks"]
        self.targets[:size] = npz[f"{prefix}_targets"]
        self.iter_weights[:size] = npz[f"{prefix}_iter_weights"]
        blob = npz[f"{prefix}_keys_blob"].tobytes()
        keys: list[bytes] = []
        off = 0
        for n in npz[f"{prefix}_keys_lens"]:
            length = int(n)
            keys.append(blob[off : off + length])
            off += length
        self.infoset_keys = keys
        self.size = size
        self.total_seen = total_seen

    def __len__(self) -> int:
        return self.size


# ───────── External-sampling traversal ─────────


class TraversalStats:
    """Lightweight counter so tests can verify sample counts.

    When coverage instrumentation is on (Piece 2), also accumulates a per-region
    infoset visit counter: region_visits[(street, facing_bet)] is a Counter over
    infoset keys, so we can report single-visit fraction + visit-depth broken down
    by street AND aggressor role (facing_bet = FOLD is legal at this node).
    """

    def __init__(self) -> None:
        self.advantage_samples: int = 0
        self.policy_samples: int = 0
        self.terminal_visits: int = 0
        self.region_visits: dict[tuple[int, int], Counter[bytes]] = {}

    def record_visit(self, street: int, facing_bet: int, infoset_key: bytes) -> None:
        region = self.region_visits.get((street, facing_bet))
        if region is None:
            region = Counter()
            self.region_visits[(street, facing_bet)] = region
        region[infoset_key] += 1

    def reset_region_visits(self) -> None:
        self.region_visits = {}


def _aggression_biased_weights(
    legal_list: list[int], on_policy: list[float], bias: float
) -> list[float]:
    """Mixture sampling weights: (1-bias)*on_policy + bias*uniform(aggressive).

    bias==0 returns `on_policy` unchanged (callers guard on bias>0 anyway). If no
    aggressive action is legal at this node, the aggressive mass falls back onto
    the on-policy distribution so the result is still a valid weight vector.
    """
    total = sum(on_policy)
    base = [p / total for p in on_policy] if total > 0 else [1.0 / len(legal_list)] * len(legal_list)
    aggressive_idx = [i for i, a in enumerate(legal_list) if a in _AGGRESSIVE_ACTIONS]
    if not aggressive_idx:
        return base  # nothing aggressive to bias toward (e.g. fold/check-only node)
    agg = [0.0] * len(legal_list)
    for i in aggressive_idx:
        agg[i] = 1.0 / len(aggressive_idx)
    return [(1.0 - bias) * base[i] + bias * agg[i] for i in range(len(legal_list))]


def external_sampling_traversal(
    game: Game[StateT],
    state: StateT,
    traverser: int,
    advantage_nets: Sequence[torch.nn.Module],
    advantage_reservoir: Reservoir,
    policy_reservoir: Reservoir,
    iter_t: int,
    rng: random.Random,
    stats: TraversalStats | None = None,
    opp_aggression_bias: float = 0.0,
) -> float:
    """One external-sampling MCCFR traversal.

    Returns expected reward for `traverser` from `state` under the current
    strategies derived from `advantage_nets`. Adds advantage samples at
    traverser nodes and policy samples at opponent nodes (one per visited
    decision point).
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

    # Coverage instrumentation (Piece 2): record this decision node's visit,
    # keyed by (street, facing_bet). The infoset key is InfoSet.to_bytes() whose
    # byte 1 is the street (spec §C); facing_bet = FOLD (action 0) is legal here.
    if stats is not None:
        key = game.infoset_key(state)
        street = key[1] if len(key) > 1 and key[1] < 4 else 0
        facing_bet = 1 if mask_tuple[ActionType.FOLD] else 0
        stats.record_visit(street, facing_bet, key)

    with torch.no_grad():
        advantages = advantage_nets[actor](features.unsqueeze(0)).squeeze(0)
    strategy = regret_match(advantages.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

    if actor == traverser:
        action_values = torch.zeros(game.num_actions, dtype=torch.float32)
        legal = game.legal_actions(state)
        for a in legal:
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
                stats,
                opp_aggression_bias,
            )
        expected = (strategy * action_values).sum().item()
        # Counterfactual regret = (action_value - expected) for legal actions only.
        regret = (action_values - expected) * mask
        advantage_reservoir.add(
            features, mask, regret, float(iter_t), rng, infoset_key=game.infoset_key(state)
        )
        if stats is not None:
            stats.advantage_samples += 1
        return expected

    # Opponent: sample one action and record current strategy for the policy net.
    legal_list = list(game.legal_actions(state))
    probs = [float(strategy[a].item()) for a in legal_list]
    total = sum(probs)
    if opp_aggression_bias > 0.0:
        # Piece 1: mix the opponent's sampling distribution toward bet/raise.
        # ONLY the sampled action changes — `policy_reservoir.add(strategy, …)`
        # below still records the true on-policy strategy, so the policy net's
        # learning target is unchanged. (The traverser's regret estimator IS
        # biased by this; see config note / spike report.)
        weights = _aggression_biased_weights(legal_list, probs, opp_aggression_bias)
        sampled = rng.choices(legal_list, weights=weights, k=1)[0]
    elif total <= 0:
        sampled = rng.choice(legal_list)
    else:
        sampled = rng.choices(legal_list, weights=probs, k=1)[0]
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
        stats,
        opp_aggression_bias,
    )


__all__ = [
    "Reservoir",
    "TraversalStats",
    "external_sampling_traversal",
]
