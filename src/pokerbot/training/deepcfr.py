"""Deep CFR Trainer (Spec.html §E).

Spec deviation: spec §E signature is `Trainer(config, abstraction)` and
assumes NLHE-on-the-abstraction. Step 5 ships with the trainer parametrized
over an arbitrary `Game` (default = the NLHE game built from the abstraction
when both are supplied), so tests can validate the pipeline on KuhnPoker in
seconds. Production training passes a SimpleNLHEGame.

Pipeline per spec:
    for t in 1..outer_iters:
        for traversal in 1..traversals_per_iter:
            external-sampling MCCFR  → adds samples to per-player advantage_buffer
                                       and shared policy_buffer
        train_advantage_nets        (batched, weighted MSE)
        if t % checkpoint_every == 0: checkpoint
    train_policy_net                (final phase)
    export_strategy                  (separate step → StrategyDB)
"""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from pokerbot.training.config import DeepCFRConfig
from pokerbot.training.export import (
    decode_nlhe_infoset,
    export_strategy_from_reservoir,
)
from pokerbot.training.lbr import local_best_response, make_advantage_strategy_fn
from pokerbot.training.nets import AdvantageNet, PolicyNet
from pokerbot.training.traversal import Reservoir, external_sampling_traversal

_LOG = logging.getLogger("pokerbot.training")

if TYPE_CHECKING:
    from pokerbot.strategy_db import StrategyDB
    from pokerbot.training.game import Game


class Trainer:
    """Run Deep CFR over an arbitrary `Game`."""

    def __init__(
        self,
        config: DeepCFRConfig,
        game: Game[Any],
        *,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.game = game
        self.device = torch.device(device)

        # Per-player advantage nets + reservoirs (standard Deep CFR layout).
        self.advantage_nets: list[nn.Module] = [
            AdvantageNet(game.feature_dim, game.num_actions, config).to(self.device)
            for _ in range(game.num_players)
        ]
        self.advantage_reservoirs: list[Reservoir] = [
            Reservoir(config.advantage_buffer_size, game.feature_dim, game.num_actions)
            for _ in range(game.num_players)
        ]
        # Shared policy net + reservoir (we mimic average strategy across iters).
        self.policy_net: PolicyNet = PolicyNet(game.feature_dim, game.num_actions, config).to(
            self.device
        )
        self.policy_reservoir: Reservoir = Reservoir(
            config.policy_buffer_size, game.feature_dim, game.num_actions
        )
        self.rng = random.Random(config.seed)
        self.iter = 0

    # ───────── public API ─────────

    def train(self, output_dir: Path, resume_from: Path | None = None) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            _LOG.info("resumed from %s at iter=%d", resume_from, self.iter)
        run_t0 = time.perf_counter()
        for t in range(self.iter + 1, self.config.outer_iters + 1):
            self.iter = t
            iter_t0 = time.perf_counter()
            self._cfr_iteration(t)
            trav_dt = time.perf_counter() - iter_t0
            train_t0 = time.perf_counter()
            self._train_advantage_nets(t)
            train_dt = time.perf_counter() - train_t0
            _LOG.info(
                "[iter %d/%d] %.1fs trav + %.1fs train (adv_buf=%d, policy_buf=%d, "
                "wall=%.1fmin, eta=%.1fmin)",
                t,
                self.config.outer_iters,
                trav_dt,
                train_dt,
                sum(len(r) for r in self.advantage_reservoirs),
                len(self.policy_reservoir),
                (time.perf_counter() - run_t0) / 60.0,
                (self.config.outer_iters - t) * (time.perf_counter() - run_t0) / max(t, 1) / 60.0,
            )
            if self.config.lbr_every > 0 and t % self.config.lbr_every == 0:
                self._log_lbr(t)
            if t % self.config.checkpoint_every == 0:
                self.save_checkpoint(output_dir / f"iter_{t:04d}.pt")
                _LOG.info("[iter %d] checkpoint -> iter_%04d.pt", t, t)
                self._log_per_street_unique_counts()
        _LOG.info("training advantage phase done; entering policy-net training")
        policy_t0 = time.perf_counter()
        self._train_policy_net()
        _LOG.info("policy-net training done in %.1fs", time.perf_counter() - policy_t0)
        _LOG.info(
            "TRAINING COMPLETE — total wall-clock %.2f min",
            (time.perf_counter() - run_t0) / 60.0,
        )

    def export_strategy(
        self,
        target_db: StrategyDB,
        version: int,
    ) -> int:
        return export_strategy_from_reservoir(
            self.policy_reservoir,
            self.policy_net,
            target_db,
            version,
            infoset_decoder=decode_nlhe_infoset,
        )

    # ───────── CFR inner loop ─────────

    def _cfr_iteration(self, t: int) -> None:
        for _ in range(self.config.traversals_per_iter):
            traverser = self.rng.randrange(self.game.num_players)
            state = self.game.new_initial_state(self.rng)
            external_sampling_traversal(
                game=self.game,
                state=state,
                traverser=traverser,
                advantage_nets=self.advantage_nets,
                advantage_reservoir=self.advantage_reservoirs[traverser],
                policy_reservoir=self.policy_reservoir,
                iter_t=t,
                rng=self.rng,
            )

    # ───────── diagnostics ─────────

    def _log_lbr(self, t: int) -> None:
        """Compute per-player LBR exploitability and log it (mbb/hand).

        Sampled root deals (lbr_samples of them, uniform weight) — LBR is exact
        tree enumeration from each root, so cost is linear in samples * players.
        Result is informational only; we never feed it back into training.
        """
        lbr_t0 = time.perf_counter()
        lbr_rng = random.Random(self.config.seed + t)
        n = self.config.lbr_samples
        initial_states = [(self.game.new_initial_state(lbr_rng), 1.0 / n) for _ in range(n)]
        strategy_fn = make_advantage_strategy_fn(self.game, self.advantage_nets)
        per_player_raw = [
            local_best_response(self.game, strategy_fn, p, initial_states)
            for p in range(self.game.num_players)
        ]
        # Raw rewards are chip deltas. NLHE stores BB in self.game.blinds[1];
        # other games (e.g. Kuhn in tests) don't have blinds, so default bb=1.0.
        bb = float(getattr(self.game, "blinds", (None, 1.0))[1])
        per_player_mbb = [r * 1000.0 / bb for r in per_player_raw]
        total_mbb = sum(per_player_mbb)
        if any(math.isnan(v) for v in per_player_mbb):
            raise RuntimeError(
                f"LBR produced NaN at iter {t}: per_player_raw={per_player_raw}"
            )
        lbr_dt = time.perf_counter() - lbr_t0
        per_player_str = ", ".join(f"{v:.2f}" for v in per_player_mbb)
        _LOG.info(
            "[iter %d/%d] LBR exploitability=%.2f mbb/hand "
            "(per_player=[%s], samples=%d, %.1fs)",
            t,
            self.config.outer_iters,
            total_mbb,
            per_player_str,
            n,
            lbr_dt,
        )

    def _log_per_street_unique_counts(self) -> None:
        """Log unique-policy-key counts in the reservoir, split by street.

        Lets us watch postflop coverage grow during training instead of
        finding out only at the end. Cost ~1-3s per call on a 5M reservoir;
        intended to be called at checkpoints, not every iter.

        The street byte sits at index 1 of each `InfoSet.to_bytes()` (spec §C).
        """
        keys = self.policy_reservoir.infoset_keys
        if not keys:
            return
        seen_per_street: list[set[bytes]] = [set() for _ in range(4)]
        for k in keys:
            if len(k) > 1 and k[1] < 4:
                seen_per_street[k[1]].add(k)
        counts = [len(s) for s in seen_per_street]
        total_unique = sum(counts)
        _LOG.info(
            "  policy_reservoir unique-key by street: "
            "preflop=%d flop=%d turn=%d river=%d (total unique=%d / size=%d / capacity=%d)",
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            total_unique,
            self.policy_reservoir.size,
            self.policy_reservoir.capacity,
        )

    # ───────── network training ─────────

    def _train_advantage_nets(self, t: int) -> None:  # noqa: ARG002  (t reserved for LR schedulers)
        for player in range(self.game.num_players):
            self._train_one_advantage(player)

    def _train_one_advantage(self, player: int) -> None:
        net = self.advantage_nets[player]
        if len(self.advantage_reservoirs[player]) == 0:
            return
        opt = torch.optim.Adam(net.parameters(), lr=self.config.learning_rate)
        net.train()
        for _ in range(self.config.train_steps_per_iter):
            batch = self.advantage_reservoirs[player].sample_batch(self.config.batch_size, self.rng)
            if batch is None:
                break
            features, masks, targets, weights = (b.to(self.device) for b in batch)
            pred = net(features)
            squared = (pred - targets) ** 2 * masks
            per_sample = squared.sum(dim=-1)
            loss = (per_sample * weights).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.config.grad_clip)
            opt.step()
        net.eval()

    def _train_policy_net(self) -> None:
        if len(self.policy_reservoir) == 0:
            return
        opt = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.policy_net.train()
        for _ in range(self.config.policy_train_steps):
            batch = self.policy_reservoir.sample_batch(self.config.batch_size, self.rng)
            if batch is None:
                break
            features, masks, targets, weights = (b.to(self.device) for b in batch)
            pred = self.policy_net.forward_with_mask(features, masks)
            squared = (pred - targets) ** 2 * masks
            per_sample = squared.sum(dim=-1)
            loss = (per_sample * weights).mean()
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip)
            opt.step()
        self.policy_net.eval()

    # ───────── checkpoints ─────────

    @staticmethod
    def _reservoir_sidecar_path(path: Path) -> Path:
        """Sidecar npz path for a given .pt checkpoint (iter_NNNN -> iter_NNNN_reservoirs.npz)."""
        return path.with_name(f"{path.stem}_reservoirs.npz")

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Reservoirs go to a sidecar .npz, written array-by-array as views so
        # peak memory holds no full copy of any reservoir (the OOM that killed
        # v4/v5 was in the monolithic torch.save path). The .pt keeps only the
        # small model/rng/config payload.
        npz_path = self._reservoir_sidecar_path(path)
        arrays: dict[str, np.ndarray] = {}
        for i, r in enumerate(self.advantage_reservoirs):
            arrays.update(r.npz_arrays(f"adv{i}"))
        arrays.update(self.policy_reservoir.npz_arrays("policy"))
        # mypy can't prove a **dict[str, ndarray] splat won't supply the
        # keyword-only `allow_pickle: bool`; every key here is a real array.
        np.savez(npz_path, **arrays)  # type: ignore[arg-type]
        torch.save(
            {
                "iter": self.iter,
                "advantage_states": [net.state_dict() for net in self.advantage_nets],
                "policy_state": self.policy_net.state_dict(),
                "rng_state": self.rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.iter = int(ckpt["iter"])
        for net, state in zip(self.advantage_nets, ckpt["advantage_states"], strict=True):
            net.load_state_dict(state)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.rng.setstate(ckpt["rng_state"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        np.random.set_state(ckpt["numpy_rng_state"])
        if "advantage_reservoirs" in ckpt:
            # Legacy monolithic format (pilot-v2 and earlier): reservoirs are
            # pickled inline in the .pt.
            for res, state in zip(
                self.advantage_reservoirs, ckpt["advantage_reservoirs"], strict=True
            ):
                res.load_state_dict(state)
            self.policy_reservoir.load_state_dict(ckpt["policy_reservoir"])
        else:
            # New split format: reservoirs live in the sidecar .npz.
            npz_path = self._reservoir_sidecar_path(path)
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"checkpoint {path} is split-format but sidecar {npz_path} is missing"
                )
            with np.load(npz_path, allow_pickle=False) as npz:
                for i, r in enumerate(self.advantage_reservoirs):
                    r.load_npz_arrays(npz, f"adv{i}")
                self.policy_reservoir.load_npz_arrays(npz, "policy")


def make_test_config(
    *,
    outer_iters: int = 5,
    traversals_per_iter: int = 10,
    train_steps_per_iter: int = 20,
    policy_train_steps: int = 50,
    batch_size: int = 32,
    advantage_buffer_size: int = 1000,
    policy_buffer_size: int = 1000,
    advantage_hidden: tuple[int, ...] = (32, 32),
    policy_hidden: tuple[int, ...] = (32, 32),
    checkpoint_every: int = 1,  # write often so tests can find a checkpoint quickly
    lbr_every: int = 0,  # LBR off by default in tests; tiny tree-walks aren't worth the seconds
    seed: int = 0xC0FFEE,
) -> DeepCFRConfig:
    """Build a tiny DeepCFRConfig for tests (full spec config takes hours)."""
    return DeepCFRConfig(
        outer_iters=outer_iters,
        traversals_per_iter=traversals_per_iter,
        train_steps_per_iter=train_steps_per_iter,
        policy_train_steps=policy_train_steps,
        batch_size=batch_size,
        advantage_buffer_size=advantage_buffer_size,
        policy_buffer_size=policy_buffer_size,
        advantage_hidden=advantage_hidden,
        policy_hidden=policy_hidden,
        checkpoint_every=checkpoint_every,
        lbr_every=lbr_every,
        seed=seed,
    )


__all__ = ["Trainer", "make_test_config"]
