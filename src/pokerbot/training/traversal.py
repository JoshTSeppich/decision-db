"""External-sampling MCCFR traversal + reservoir buffers (Spec.html §E)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch

from pokerbot.training.nets import regret_match

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
    """Lightweight counter so tests can verify sample counts."""

    def __init__(self) -> None:
        self.advantage_samples: int = 0
        self.policy_samples: int = 0
        self.terminal_visits: int = 0


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
    if total <= 0:
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
    )


__all__ = [
    "Reservoir",
    "TraversalStats",
    "external_sampling_traversal",
]
