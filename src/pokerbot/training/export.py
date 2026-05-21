"""Walk a policy reservoir → write StrategyDB rows (Spec.html §E final phase)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from pokerbot.abstraction import InfoSet
from pokerbot.abstraction.infoset import PREFIX_LEN

if TYPE_CHECKING:
    from collections.abc import Callable

    from pokerbot.strategy_db import StrategyDB
    from pokerbot.training.nets import PolicyNet
    from pokerbot.training.traversal import Reservoir


# Default InfoSet decoder reverses the byte layout described in spec §C.
def decode_nlhe_infoset(blob: bytes) -> InfoSet:
    """Reverse `InfoSet.to_bytes()`. Caller is responsible for validity."""
    if len(blob) < PREFIX_LEN:
        raise ValueError(f"infoset blob too short: {len(blob)}B (need ≥ {PREFIX_LEN})")
    return InfoSet(
        table_size=blob[0],
        street=blob[1],
        position=blob[2],
        stack_bucket=blob[3],
        card_bucket=int.from_bytes(blob[4:6], "little"),
        history=blob[PREFIX_LEN:],
    )


def export_strategy_from_reservoir(
    reservoir: Reservoir,
    policy_net: PolicyNet,
    db: StrategyDB,
    version: int,
    *,
    infoset_decoder: Callable[[bytes], InfoSet] = decode_nlhe_infoset,
    bump_current_version: bool = True,
) -> int:
    """Convert the policy_reservoir into DB rows. Returns # distinct infosets written.

    For each unique `infoset_key` in `reservoir`, query `policy_net` on the
    cached feature row, pack the legal action distribution, and persist via
    `db.bulk_put` under `version`. If `bump_current_version`, also call
    `db.set_current_version(version)`.
    """
    seen: dict[bytes, int] = {}
    for i in range(reservoir.size):
        key = reservoir.infoset_keys[i]
        if key and key not in seen:
            seen[key] = i

    if not seen:
        if bump_current_version:
            db.set_current_version(version)
        return 0

    policy_net.eval()
    rows: list[tuple[InfoSet, int, np.ndarray]] = []
    for key, idx in seen.items():
        info = infoset_decoder(key)
        features = torch.from_numpy(reservoir.features[idx : idx + 1])
        mask_np = reservoir.masks[idx]
        mask = torch.from_numpy(mask_np[None, :]).to(dtype=torch.bool)
        with torch.no_grad():
            probs = policy_net.forward_with_mask(features, mask).squeeze(0).cpu().numpy()

        # Compress: build int mask + packed legal-action probs
        action_mask = 0
        packed: list[float] = []
        for a in range(reservoir.num_actions):
            if mask_np[a] > 0:
                action_mask |= 1 << a
                packed.append(float(probs[a]))
        arr = np.array(packed, dtype=np.float32)
        s = float(arr.sum())
        # Network output may collapse to zero on legal actions → fall back to uniform.
        arr = np.full_like(arr, 1.0 / len(arr)) if s <= 0 else arr / s
        # Guard against float32 sum drift: re-normalize to land within tol.
        arr = (arr / float(arr.sum())).astype(np.float32, copy=False)
        rows.append((info, action_mask, arr))

    db.bulk_put(iter(rows), version=version)
    if bump_current_version:
        db.set_current_version(version)
    return len(rows)


__all__ = ["decode_nlhe_infoset", "export_strategy_from_reservoir"]
