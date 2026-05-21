"""Deep CFR training configuration (Spec.html §E).

All values here are spec-pinned — `test_config_pinned` is a regression guard
that breaks if anyone bumps a hyperparameter without a corresponding spec edit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepCFRConfig:
    # Networks
    advantage_hidden: tuple[int, ...] = (256, 256, 256)
    policy_hidden: tuple[int, ...] = (256, 256, 256)
    activation: str = "relu"
    layer_norm: bool = True

    # Optimization
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    grad_clip: float = 1.0
    batch_size: int = 4096

    # CFR loop
    outer_iters: int = 1000
    traversals_per_iter: int = 1000
    train_steps_per_iter: int = 4000
    policy_train_steps: int = 20000

    # Reservoirs
    advantage_buffer_size: int = 1_000_000
    policy_buffer_size: int = 1_000_000

    # CFR weighting (linear-CFR per spec)
    cfr_weighting: str = "linear"

    # Multi-player rotation
    num_players_train: tuple[int, ...] = (6, 8, 9)
    seat_randomization: bool = True

    # Determinism
    seed: int = 0xC0FFEE

    # Checkpoint cadence (spec §E "every 10 iterations")
    checkpoint_every: int = 10
    # LBR eval cadence. 0 disables LBR (used by --skip-lbr and tiny tests).
    lbr_every: int = 25
    # Each LBR eval is `lbr_samples` exact-tree-enumerations per BR player, so
    # cost scales linearly with (num_players * lbr_samples). Exact LBR expands
    # the full opponent-strategy tree, which on 6-handed NLHE is huge:
    # empirically lbr_samples=4 didn't finish in 16 min, lbr_samples=32 didn't
    # finish in 11 min. lbr.py's own docstring flags depth-limited LBR as the
    # proper v2 fix for NLHE. Until then, default lbr_samples=1 keeps the LBR
    # log line as a coarse but real convergence signal; cost is high variance
    # but bounded. Bump only on small games (Kuhn, heads-up) where the tree
    # actually enumerates.
    lbr_samples: int = 1


@dataclass(frozen=True)
class EvalResult:
    """Output of `Trainer.evaluate()` and `local_best_response()`."""

    iteration: int
    lbr_mbb_per_hand: float
    vs_opponent_mbb: dict[str, float]


__all__ = ["DeepCFRConfig", "EvalResult"]
