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

    # ── retrain-mechanisms spike (DEFAULT-OFF; default path is unchanged) ──
    # Piece 1: opponent-node aggression bias. 0.0 = pure on-policy sampling
    # (current behavior, byte-identical). >0 mixes the opponent's sampling
    # distribution toward bet/raise actions: sample ~ (1-b)*on_policy +
    # b*uniform(aggressive). Only the SAMPLED action changes — the recorded
    # policy-net target stays the true on-policy strategy. NOTE: with b>0 the
    # external-sampling regret estimator is biased (no importance-sampling
    # correction); this knob is an exploration/visitation lever, NOT a
    # correctness-preserving change. See the spike report.
    opp_aggression_bias: float = 0.0
    # Piece 2: per-(street × facing-aggression) visit-depth instrumentation.
    # False = no counter, no extra logging (unchanged). True = the trainer
    # accumulates a per-region infoset visit counter and logs single-visit
    # fraction + a depth histogram at each checkpoint.
    coverage_instrument: bool = False
    # Piece 3: deep-stack ALL_IN gate (abstraction-fix, Phase 1). The from-scratch
    # launcher builds a `GatedNLHEGame` with these so the trainer never explores
    # (never learns) the discretionary deep-stack shove that failed the nc-AI band
    # (POD_RUN_LOG close §③; ABSTRACTION_FIX_SPEC §1). Predicates live in ONE place,
    # `zoom.abstraction_gate` — NOT imported here (src/pokerbot must not import zoom),
    # so the literal defaults are kept in lockstep with that module by
    # `test_config_pinned`. Phase 2 sweeps these; ActionType enum is untouched.
    #
    # POSTFLOP gate: drop ALL_IN when SPR = stack/(pot+to_call) > allin_spr_cap and a
    # non-shove bet exists. 10.0 keeps low-SPR river jams, drops deep overbet shoves.
    allin_spr_cap: float = 10.0
    # PREFLOP gate (SPR is the WRONG preflop lever — facing a raise lowers SPR, so a
    # 100bb facing-3bet 4bet-jam survives an SPR cap while a 25bb open-jam, higher SPR,
    # would be cut; 50k smoke confirmed nc-AI stuck at 6.6%). Preflop instead gates on
    # two ORTHOGONAL axes:
    #   • DEPTH — keep ALL_IN iff effective stack ≤ max_preflop_allin_eff_bb (push/fold
    #     zone where open-/3bet-jamming is GTO). 25bb is the standard top of that zone.
    #   • COMMITMENT — keep iff stack-behind-after-calling ≤ preflop_allin_commit_pots
    #     pot-sized bets (real 4bet/5bet shove-war). Keyed on stack-behind, not to_call:
    #     facing-3bet-100bb has large to_call but ~4.7 pots behind (drop); facing-4bet
    #     has ~1.4 (keep). 1.5 separates them.
    # Deep + uncommitted + a non-shove raise available ⇒ ALL_IN dropped. Set eff_bb=inf
    # to disable the depth cut (reproduces SPR-only preflop behavior).
    max_preflop_allin_eff_bb: float = 25.0
    preflop_allin_commit_pots: float = 1.5


@dataclass(frozen=True)
class EvalResult:
    """Output of `Trainer.evaluate()` and `local_best_response()`."""

    iteration: int
    lbr_mbb_per_hand: float
    vs_opponent_mbb: dict[str, float]


__all__ = ["DeepCFRConfig", "EvalResult"]
