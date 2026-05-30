"""Net-based best-response exporter (Approach-2, Component 5 prerequisite).

In a best-response fine-tune the deployable policy is `regret_match(advantage_nets)`
— the CURRENT best response to the fixed archetype pool, NOT a time-average
(averaging is for converging to Nash; this fine-tune deliberately is not). The
policy_reservoir is empty by construction (opponents are scripted agents, so the
self-play policy-recording branch never fires — the confirmed (a)-benign result),
so the standard `export_strategy_from_reservoir` (which reads the policy net) can't
emit this policy.

`export_best_response` walks the per-player ADVANTAGE reservoirs and writes
`regret_match(advantage_nets[p](features), mask)` to a `StrategyDB` in the EXACT
same row format the production exporter / RuntimeAdapter / Component 4 harness use
— it reuses `export_strategy_from_reservoir` itself, passing a thin per-net adapter
whose `forward_with_mask` returns the regret-matched strategy. So there is one row
format and one validated scoring path, not a parallel one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from pokerbot.training.export import decode_nlhe_infoset
from pokerbot.training.nets import regret_match

if TYPE_CHECKING:
    from torch import nn

    from pokerbot.strategy_db import StrategyDB
    from pokerbot.training.traversal import Reservoir


class _BestResponseStrategyNet:
    """Adapter exposing the `PolicyNet`-shaped surface `export_strategy_from_reservoir`
    calls (`eval()` + `forward_with_mask`), but returning the regret-matched current
    best-response strategy from an advantage net instead of a softmax over logits."""

    def __init__(self, advantage_net: nn.Module) -> None:
        self._net = advantage_net

    def eval(self) -> None:
        self._net.eval()

    def forward_with_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            advantages = self._net(x)
        return regret_match(advantages, mask.to(torch.float32))


def export_best_response(
    advantage_nets: list[nn.Module],
    advantage_reservoirs: list[Reservoir],
    db: StrategyDB,
    version: int,
) -> int:
    """Write regret_match(advantage_nets) for every visited infoset to `db`.

    One row per unique infoset across all per-player advantage reservoirs (infoset
    keys carry position, so per-player keys don't collide). Returns the total number
    of rows written. Sets `db` current version to `version` at the end.

    The row format mirrors `export_strategy_from_reservoir` exactly (int action_mask +
    packed legal probs, uniform fallback on zero-mass) so there is ONE scoring path —
    but it additionally writes per-infoset `visit_count` = the infoset's frequency in
    the advantage reservoir. That count is load-bearing: `nearest_neighbor` tiebreaks on
    `visit_count DESC`, so without it (the frozen exporter's `bulk_put` drops it to 0)
    a history-blind lookup returns a hash-arbitrary row instead of the representative
    (most-visited) one — the cause of the spurious 'folds AA' reading.
    """
    total = 0
    for net, reservoir in zip(advantage_nets, advantage_reservoirs, strict=True):
        strat = _BestResponseStrategyNet(net)
        strat.eval()
        # Per-key visit frequency + first occurrence (features/mask are identical for a
        # given infoset key, so the first index suffices for the net forward).
        counts: dict[bytes, int] = {}
        first_idx: dict[bytes, int] = {}
        for i in range(reservoir.size):
            key = reservoir.infoset_keys[i]
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            if key not in first_idx:
                first_idx[key] = i
        for key, idx in first_idx.items():
            info = decode_nlhe_infoset(key)
            features = torch.from_numpy(reservoir.features[idx : idx + 1])
            mask_np = reservoir.masks[idx]
            mask = torch.from_numpy(mask_np[None, :]).to(dtype=torch.bool)
            with torch.no_grad():
                probs = strat.forward_with_mask(features, mask).squeeze(0).cpu().numpy()
            # Compress to int mask + packed legal probs (identical to export.py).
            action_mask = 0
            packed: list[float] = []
            for a in range(reservoir.num_actions):
                if mask_np[a] > 0:
                    action_mask |= 1 << a
                    packed.append(float(probs[a]))
            arr = np.array(packed, dtype=np.float32)
            s = float(arr.sum())
            arr = np.full_like(arr, 1.0 / len(arr)) if s <= 0 else arr / s
            arr = (arr / float(arr.sum())).astype(np.float32, copy=False)
            db.put(info, action_mask, arr, version, visit_count=counts[key])
            total += 1
    db.set_current_version(version)
    return total


__all__ = ["export_best_response"]
