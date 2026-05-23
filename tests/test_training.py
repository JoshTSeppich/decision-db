"""Spec.html §E tests for the Deep CFR training pipeline.

Spec tests:
    1. test_config_pinned                  — exact-values regression guard
    2. test_traversal_single_hand          — one external-sampling traversal
    3. test_advantage_net_overfits_tiny_buffer
    4. test_checkpoint_resume              — train, save, reload, train more
    5. test_lbr_better_than_random         — 50 iters of CFR on Kuhn ≪ random
    6. test_export_strategy_writes_all_infosets

All tests run on Kuhn poker (tiny game) with `make_test_config` so the whole
file completes in seconds. Real NLHE training uses the spec-pinned defaults
and takes days — that path is exercised manually, not in CI.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pokerbot.abstraction import ActionType, InfoSet
from pokerbot.strategy_db import SQLiteStrategyDB
from pokerbot.training import (
    DeepCFRConfig,
    KuhnPokerGame,
    Reservoir,
    Trainer,
    TraversalStats,
    exploitability,
    external_sampling_traversal,
    make_advantage_strategy_fn,
    make_test_config,
    make_uniform_strategy_fn,
)
from pokerbot.training.export import export_strategy_from_reservoir
from pokerbot.training.kuhn import KuhnState
from pokerbot.training.nets import AdvantageNet, PolicyNet

if TYPE_CHECKING:
    from pathlib import Path


# ───────── 1. test_config_pinned ─────────


def test_config_pinned_values_match_spec() -> None:
    """Spec.html §E. ALL these values must change in lockstep with the spec doc."""
    c = DeepCFRConfig()
    assert c.advantage_hidden == (256, 256, 256)
    assert c.policy_hidden == (256, 256, 256)
    assert c.activation == "relu"
    assert c.layer_norm is True
    assert c.learning_rate == 1e-3
    assert c.optimizer == "adam"
    assert c.grad_clip == 1.0
    assert c.batch_size == 4096
    assert c.outer_iters == 1000
    assert c.traversals_per_iter == 1000
    assert c.train_steps_per_iter == 4000
    assert c.policy_train_steps == 20_000
    assert c.advantage_buffer_size == 1_000_000
    assert c.policy_buffer_size == 1_000_000
    assert c.cfr_weighting == "linear"
    assert c.num_players_train == (6, 8, 9)
    assert c.seat_randomization is True
    assert c.seed == 0xC0FFEE


# ───────── 2. test_traversal_single_hand ─────────


def test_traversal_single_hand_kuhn_sample_count() -> None:
    """One external-sampling traversal on Kuhn yields a deterministic # of samples."""
    game = KuhnPokerGame()
    config = make_test_config()
    nets = [
        AdvantageNet(game.feature_dim, game.num_actions, config) for _ in range(game.num_players)
    ]
    adv_res = Reservoir(100, game.feature_dim, game.num_actions)
    pol_res = Reservoir(100, game.feature_dim, game.num_actions)
    rng = random.Random(42)
    state = game.new_initial_state(rng)

    stats = TraversalStats()
    external_sampling_traversal(
        game=game,
        state=state,
        traverser=0,
        advantage_nets=nets,
        advantage_reservoir=adv_res,
        policy_reservoir=pol_res,
        iter_t=1,
        rng=rng,
        stats=stats,
    )
    # Bounds (Kuhn-specific): traverser enumerates both branches at each
    # decision node, so terminals & samples scale with the recursion shape.
    total = stats.advantage_samples + stats.policy_samples
    assert 1 <= total <= 5, f"expected 1..5 samples, got {total}"
    assert stats.advantage_samples >= 1, "traverser should make at least 1 decision"
    # Each traverser enumeration explores 2 actions and the opp samples 1 → between
    # 1 and 4 terminals per traversal depending on path.
    assert 1 <= stats.terminal_visits <= 4, f"terminal_visits = {stats.terminal_visits}"


# ───────── 3. test_advantage_net_overfits_tiny_buffer ─────────


def test_advantage_net_overfits_tiny_buffer() -> None:
    """Train a fresh AdvantageNet on a 100-row buffer for 1000 steps; MSE must be small."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    in_dim, n_actions = 8, 4
    n = 100

    features = torch.from_numpy(rng.standard_normal((n, in_dim)).astype(np.float32))
    targets = torch.from_numpy(rng.standard_normal((n, n_actions)).astype(np.float32))
    masks = torch.ones(n, n_actions)

    config = make_test_config(advantage_hidden=(64, 64), policy_hidden=(64, 64))
    net = AdvantageNet(in_dim, n_actions, config)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(1000):
        pred = net(features)
        loss = ((pred - targets) ** 2 * masks).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_mse = float(((net(features) - targets) ** 2).mean().item())
    assert final_mse < 1e-3, f"final MSE = {final_mse:.6f}, expected < 1e-3"


# ───────── 4. test_checkpoint_resume ─────────


def test_checkpoint_resume_matches_straight_training(tmp_path: Path) -> None:
    """Train 5 iters, checkpoint, resume, train 5 more.

    State (advantage net weights) should match training 10 straight iters from
    the same seed within numerical tolerance.
    """
    config = make_test_config(outer_iters=10, seed=12345)

    # Path A: train 10 iters straight.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    game_a = KuhnPokerGame()
    trainer_a = Trainer(config, game_a)
    trainer_a.train(tmp_path / "straight")
    final_state_a = trainer_a.advantage_nets[0].state_dict()

    # Path B: train 5 iters, checkpoint, resume, train 5 more.
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    half_config = make_test_config(outer_iters=5, seed=12345)
    game_b = KuhnPokerGame()
    trainer_b = Trainer(half_config, game_b)
    ckpt_path = tmp_path / "halfway" / "iter_0005.pt"
    trainer_b.train(tmp_path / "halfway")

    # Resume into a fresh trainer with the full 10-iter config.
    trainer_c = Trainer(config, KuhnPokerGame())
    trainer_c.train(tmp_path / "resumed", resume_from=ckpt_path)
    final_state_c = trainer_c.advantage_nets[0].state_dict()

    # Compare net params within tight tol.
    for k in final_state_a:
        diff = float((final_state_a[k] - final_state_c[k]).abs().max().item())
        assert diff < 1e-4, f"{k} drift {diff:.6f} between straight and resumed runs"


def test_reservoir_npz_roundtrip(tmp_path: Path) -> None:
    """Reservoir survives the streaming-npz save/load with variable-length keys."""
    rng = random.Random(7)
    src = Reservoir(capacity=64, feature_dim=5, num_actions=3)
    for i in range(40):  # partially filled, so size < capacity
        src.add(
            torch.rand(5),
            torch.tensor([1.0, 1.0, 0.0]),
            torch.rand(3),
            float(i),
            rng,
            infoset_key=bytes([i % 4]) * (i % 7 + 1),  # variable-length keys
        )

    arrays = src.npz_arrays("policy")
    np.savez(tmp_path / "r.npz", **arrays)  # type: ignore[arg-type]

    dst = Reservoir(capacity=64, feature_dim=5, num_actions=3)
    with np.load(tmp_path / "r.npz", allow_pickle=False) as npz:
        dst.load_npz_arrays(npz, "policy")

    assert dst.size == src.size
    assert dst.total_seen == src.total_seen
    assert dst.infoset_keys == src.infoset_keys
    np.testing.assert_array_equal(dst.features[: dst.size], src.features[: src.size])
    np.testing.assert_array_equal(dst.masks[: dst.size], src.masks[: src.size])
    np.testing.assert_array_equal(dst.targets[: dst.size], src.targets[: src.size])
    np.testing.assert_array_equal(
        dst.iter_weights[: dst.size], src.iter_weights[: src.size]
    )


def test_checkpoint_writes_split_format(tmp_path: Path) -> None:
    """save_checkpoint produces a small .pt (no reservoirs) plus a sidecar .npz."""
    config = make_test_config(outer_iters=2, seed=99)
    trainer = Trainer(config, KuhnPokerGame())
    trainer.train(tmp_path / "run")

    pt = tmp_path / "run" / "iter_0002.pt"
    npz = tmp_path / "run" / "iter_0002_reservoirs.npz"
    assert pt.exists() and npz.exists()

    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    assert "advantage_reservoirs" not in ckpt  # reservoirs moved to sidecar
    assert "policy_reservoir" not in ckpt


def test_checkpoint_loads_legacy_monolithic_format(tmp_path: Path) -> None:
    """Backward compat: resume from an old-style checkpoint with inline reservoirs."""
    config = make_test_config(outer_iters=3, seed=55)
    trainer = Trainer(config, KuhnPokerGame())
    trainer.train(tmp_path / "run")

    # Fabricate a legacy monolithic checkpoint (reservoirs pickled inline).
    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "iter": trainer.iter,
            "advantage_states": [net.state_dict() for net in trainer.advantage_nets],
            "policy_state": trainer.policy_net.state_dict(),
            "rng_state": trainer.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "advantage_reservoirs": [r.state_dict() for r in trainer.advantage_reservoirs],
            "policy_reservoir": trainer.policy_reservoir.state_dict(),
            "config": config,
        },
        legacy,
    )

    fresh = Trainer(config, KuhnPokerGame())
    fresh.load_checkpoint(legacy)
    assert fresh.iter == trainer.iter
    assert fresh.policy_reservoir.size == trainer.policy_reservoir.size
    assert fresh.policy_reservoir.infoset_keys == trainer.policy_reservoir.infoset_keys


def test_run_metadata_saved_in_checkpoint(tmp_path: Path) -> None:
    """run_metadata is written into the .pt alongside config (Fix 3)."""
    config = make_test_config(outer_iters=1, seed=1)
    meta = {"lru_max": 4_000_000, "advantage_buffer_size": 250_000, "policy_buffer_size": 1_000_000}
    trainer = Trainer(config, KuhnPokerGame(), run_metadata=meta)
    trainer.train(tmp_path / "run")

    ckpt = torch.load(tmp_path / "run" / "iter_0001.pt", map_location="cpu", weights_only=False)
    assert ckpt["run_metadata"] == meta
    assert ckpt["config"] is not None  # not replaced by run_metadata

    # Default (no run_metadata) stores None, not a crash.
    plain = Trainer(config, KuhnPokerGame())
    plain.train(tmp_path / "run2")
    ckpt2 = torch.load(tmp_path / "run2" / "iter_0001.pt", map_location="cpu", weights_only=False)
    assert ckpt2["run_metadata"] is None


# ───────── 5. test_lbr_better_than_random ─────────


def _enumerate_kuhn_deals() -> list[tuple[KuhnState, float]]:
    deals: list[tuple[KuhnState, float]] = []
    for c0 in range(3):
        for c1 in range(3):
            if c0 == c1:
                continue
            deals.append((KuhnState(cards=(c0, c1)), 1.0 / 6.0))
    return deals


def test_lbr_trained_better_than_uniform_random() -> None:
    """Spec §E test #5: trained policy is less exploitable than uniform random."""
    game = KuhnPokerGame()
    deals = _enumerate_kuhn_deals()
    uniform_strategy = make_uniform_strategy_fn(game)
    expl_uniform = exploitability(game, uniform_strategy, deals)

    config = make_test_config(
        outer_iters=50, traversals_per_iter=40, train_steps_per_iter=80, seed=2026
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    trainer = Trainer(config, KuhnPokerGame())
    trainer.train(output_dir=_tmp_dir() / "lbr_train")
    trained_strategy = make_advantage_strategy_fn(game, trainer.advantage_nets)
    expl_trained = exploitability(game, trained_strategy, deals)

    assert expl_trained < expl_uniform, (
        f"trained exploitability {expl_trained:.4f} ≥ uniform {expl_uniform:.4f}; "
        "CFR isn't learning anything"
    )


def _tmp_dir() -> Path:
    """Stable per-test tmp dir helper (pytest's tmp_path fixture isn't available here)."""
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="pokerbot-train-"))


# ───────── 6. test_export_strategy_writes_all_infosets ─────────


def test_export_writes_all_distinct_infosets(tmp_path: Path) -> None:
    """Populate a reservoir with valid InfoSet rows, export, verify every row landed."""
    db = SQLiteStrategyDB(str(tmp_path / "exp.db"))
    db.set_current_version(0)

    game = KuhnPokerGame()
    num_actions = 9  # match StrategyDB row layout (full ActionType space)
    config = make_test_config()
    feature_dim = game.feature_dim
    reservoir = Reservoir(capacity=200, feature_dim=feature_dim, num_actions=num_actions)
    rng = random.Random(0)

    # Build 30 distinct NLHE-style infosets + plausible legal masks.
    legal_actions = [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.BET_66]
    mask_t = torch.zeros(num_actions)
    for a in legal_actions:
        mask_t[int(a)] = 1.0
    target = torch.zeros(num_actions)
    for a in legal_actions:
        target[int(a)] = 1.0 / len(legal_actions)

    distinct = []
    for i in range(30):
        info = InfoSet(
            table_size=6, street=1, position=2, stack_bucket=4, card_bucket=i, history=b""
        )
        distinct.append(info)
        reservoir.add(
            features=torch.zeros(feature_dim),
            mask=mask_t,
            target=target,
            iter_weight=float(i + 1),
            rng=rng,
            infoset_key=info.to_bytes(),
        )

    policy_net = PolicyNet(feature_dim, num_actions, config)
    n_exported = export_strategy_from_reservoir(reservoir, policy_net, db, version=1)
    assert n_exported == 30

    db.set_current_version(1)
    for info in distinct:
        row = db.get(info, version=1)
        assert row is not None, f"missing row for {info}"
        assert row.action_mask != 0
        assert row.action_probs.sum() == pytest.approx(1.0, abs=1e-5)
