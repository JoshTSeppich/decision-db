"""PyTorch networks for Deep CFR (Spec.html §E).

Two heads:
    - `AdvantageNet`: outputs raw advantage per action. Regret-matching turns
       the positive part into a strategy.
    - `PolicyNet`: outputs a softmax over actions, masked by legal-action set.

Both are MLPs with LayerNorm + ReLU per spec.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from pokerbot.training.config import DeepCFRConfig


class _MLP(nn.Module):
    """Plain MLP with optional LayerNorm + configurable hidden stack."""

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
        *,
        layer_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if activation != "relu":
            raise ValueError(f"only 'relu' supported in v1, got {activation!r}")
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class AdvantageNet(_MLP):
    """Predicts per-action advantage. Regret-matching converts to strategy."""

    def __init__(self, in_dim: int, num_actions: int, config: DeepCFRConfig) -> None:
        super().__init__(
            in_dim,
            config.advantage_hidden,
            num_actions,
            layer_norm=config.layer_norm,
            activation=config.activation,
        )


class PolicyNet(_MLP):
    """Predicts per-action logits; softmax+mask for legal-action probs."""

    def __init__(self, in_dim: int, num_actions: int, config: DeepCFRConfig) -> None:
        super().__init__(
            in_dim,
            config.policy_hidden,
            num_actions,
            layer_norm=config.layer_norm,
            activation=config.activation,
        )

    def forward_with_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        # mask: 1 where legal, 0 where illegal
        very_neg = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(mask == 0, very_neg)
        return torch.softmax(masked, dim=-1)


def regret_match(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Vanilla regret matching: strategy ∝ max(0, advantage), uniform if all ≤ 0.

    `mask` is 1.0 where the action is legal, 0.0 elsewhere. The returned
    distribution has weight 0 on illegal actions, sums to 1 on legal ones.
    """
    if advantages.shape != mask.shape:
        raise ValueError(f"shape mismatch: adv={advantages.shape}, mask={mask.shape}")
    pos = torch.clamp(advantages, min=0.0) * mask
    total = pos.sum(dim=-1, keepdim=True)
    legal_count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    uniform_over_legal = mask / legal_count
    is_zero = total == 0
    return torch.where(is_zero, uniform_over_legal, pos / total.clamp(min=1e-12))


__all__ = [
    "AdvantageNet",
    "PolicyNet",
    "regret_match",
]
