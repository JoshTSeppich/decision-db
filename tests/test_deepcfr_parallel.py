"""Bit-identity + faithfulness gate for the Deep CFR fan-out/merge traversal port.

The serving package is frozen; the multi-worker traversal port lives on the
retrain spike. This suite is the proof the port did not drift reservoir
*composition* (the subtle hazard: reservoir sampling under multi-worker can change
which samples survive even when per-traversal outputs match). It asserts all three
pieces of net-new code, then a faithful-CFR convergence check:

  1. reservoir CONTENTS bit-identical: num_workers=1 (in-process serial reference)
     ≡ num_workers=N — size, total_seen, infoset_keys, features, masks, targets,
     iter_weights, for every advantage reservoir AND the policy reservoir (plus the
     trained nets, which follow deterministically from identical reservoirs);
  2. merged per-(street × facing-bet) COVERAGE counters identical 1 ≡ N (the Piece-2
     instrument the scaled run watches must reproduce single-process exactly);
  3. total sample count identical (guards a worker silently dropping/duplicating a
     traversal — e.g. a swallowed exception);
  4. self-play through the parallel engine still drives Kuhn exploitability below
     uniform (faithful Deep CFR, not merely self-consistent).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from pokerbot.training import exploitability, make_uniform_strategy_fn
from pokerbot.training.deepcfr import Trainer, make_test_config
from pokerbot.training.kuhn import KuhnPokerGame, KuhnState

if TYPE_CHECKING:
    from pokerbot.training.traversal import Reservoir


def _config(**over: Any) -> Any:
    base = make_test_config(
        traversals_per_iter=40,
        train_steps_per_iter=20,
        advantage_hidden=(32, 32),
        policy_hidden=(32, 32),
        advantage_buffer_size=4000,
        policy_buffer_size=4000,
        batch_size=64,
        seed=4242,
    )
    # coverage_instrument is not a make_test_config kwarg; set it on the frozen
    # dataclass so the port's coverage-merge path is exercised by every assertion.
    return dataclasses.replace(base, coverage_instrument=True, **over)


def _build_trainer(num_workers: int, *, seed: int = 4242) -> Trainer:
    """A trainer with identical initial net weights for a given seed."""
    config = _config(seed=seed)
    torch.manual_seed(seed)  # identical initial advantage-net weights across trainers
    np.random.seed(seed)
    return Trainer(config, KuhnPokerGame(), num_workers=num_workers)


def _run_iters(trainer: Trainer, k: int = 2) -> None:
    """Run k full iterations (parallel traversal + advantage train), then release the pool.

    Running >=2 iters is deliberate: iter 2's traversal uses iter-1-trained weights,
    so a worker that fails to load the pushed net state_dicts (stale-weights bug)
    diverges here even though iter 1 would have matched.
    """
    for t in range(1, k + 1):
        trainer.iter = t
        trainer._cfr_iteration_parallel(t)
        trainer._train_advantage_nets(t)
    trainer.close_pool()


def _assert_reservoir_bit_identical(a: Reservoir, b: Reservoir, label: str) -> None:
    assert a.size == b.size, f"{label}: size {a.size} != {b.size}"
    assert a.total_seen == b.total_seen, f"{label}: total_seen {a.total_seen} != {b.total_seen}"
    assert a.infoset_keys == b.infoset_keys, f"{label}: infoset_keys differ"
    np.testing.assert_array_equal(a.features[: a.size], b.features[: b.size], err_msg=f"{label} features")
    np.testing.assert_array_equal(a.masks[: a.size], b.masks[: b.size], err_msg=f"{label} masks")
    np.testing.assert_array_equal(a.targets[: a.size], b.targets[: b.size], err_msg=f"{label} targets")
    np.testing.assert_array_equal(
        a.iter_weights[: a.size], b.iter_weights[: b.size], err_msg=f"{label} iter_weights"
    )


def _assert_reservoirs_equal(a: Trainer, b: Trainer) -> None:
    for i, (ra, rb) in enumerate(
        zip(a.advantage_reservoirs, b.advantage_reservoirs, strict=True)
    ):
        _assert_reservoir_bit_identical(ra, rb, f"advantage_reservoir[{i}]")
    _assert_reservoir_bit_identical(a.policy_reservoir, b.policy_reservoir, "policy_reservoir")


def _coverage_snapshot(trainer: Trainer) -> dict[tuple[int, int], dict[bytes, int]]:
    cov = trainer._cov_stats
    assert cov is not None, "coverage_instrument must be on for the coverage assertions"
    return {region: dict(ctr) for region, ctr in cov.region_visits.items()}


# ─────────── Piece 1: reservoir contents bit-identical ───────────


def test_parallel_engine_reservoirs_bit_identical_to_serial() -> None:
    serial = _build_trainer(num_workers=1)
    parallel = _build_trainer(num_workers=4)
    _run_iters(serial, k=2)
    _run_iters(parallel, k=2)

    _assert_reservoirs_equal(serial, parallel)
    # Identical reservoirs + rng + start weights → identical training → identical nets.
    for i, (ns, npar) in enumerate(
        zip(serial.advantage_nets, parallel.advantage_nets, strict=True)
    ):
        for k_, v in ns.state_dict().items():
            torch.testing.assert_close(
                v, npar.state_dict()[k_], rtol=0, atol=0, msg=f"net[{i}].{k_} diverged"
            )


# ─────────── Piece 2: merged coverage counters bit-identical ───────────


def test_parallel_engine_coverage_stats_identical_to_serial() -> None:
    serial = _build_trainer(num_workers=1)
    parallel = _build_trainer(num_workers=4)
    _run_iters(serial, k=2)
    _run_iters(parallel, k=2)

    serial_cov = _coverage_snapshot(serial)
    # Guard against a vacuous pass: the instrument must actually have recorded visits.
    assert serial_cov, "no coverage recorded — the instrument is not exercising"
    assert sum(sum(c.values()) for c in serial_cov.values()) > 0
    assert serial_cov == _coverage_snapshot(parallel), "merged region visit counters diverged 1 vs N worker"

    for attr in ("advantage_samples", "policy_samples", "terminal_visits"):
        assert getattr(serial._cov_stats, attr) == getattr(parallel._cov_stats, attr), (
            f"coverage scalar {attr} diverged"
        )


# ─────────── Piece 3: no dropped / duplicated traversals ───────────


def test_total_sample_count_matches_serial() -> None:
    serial = _build_trainer(num_workers=1)
    parallel = _build_trainer(num_workers=4)
    _run_iters(serial, k=2)
    _run_iters(parallel, k=2)

    serial_total = (
        sum(r.total_seen for r in serial.advantage_reservoirs)
        + serial.policy_reservoir.total_seen
    )
    parallel_total = (
        sum(r.total_seen for r in parallel.advantage_reservoirs)
        + parallel.policy_reservoir.total_seen
    )
    assert serial_total == parallel_total > 0, (serial_total, parallel_total)


def test_two_parallel_runs_are_deterministic() -> None:
    a = _build_trainer(num_workers=4)
    b = _build_trainer(num_workers=4)
    _run_iters(a, k=2)
    _run_iters(b, k=2)
    _assert_reservoirs_equal(a, b)
    assert _coverage_snapshot(a) == _coverage_snapshot(b)


# ─────────── faithful CFR: self-play through the parallel engine converges ───────────


def _enumerate_kuhn_deals() -> list[tuple[KuhnState, float]]:
    return [
        (KuhnState(cards=(c0, c1)), 1.0 / 6.0)
        for c0 in range(3)
        for c1 in range(3)
        if c0 != c1
    ]


def test_parallel_engine_converges_to_kuhn_nash() -> None:
    """Self-play through the multi-worker engine drives the average policy's
    exploitability well below uniform on Kuhn — proves the port is faithful Deep
    CFR, not just self-consistent 1≡N. Uses num_workers=2 (real spawn pool), which
    also exercises picklability of the game across the process boundary."""
    game = KuhnPokerGame()
    deals = _enumerate_kuhn_deals()
    expl_uniform = exploitability(game, make_uniform_strategy_fn(game), deals)

    config = _config(
        outer_iters=150,
        traversals_per_iter=64,
        train_steps_per_iter=96,
        policy_train_steps=3000,
        advantage_buffer_size=40_000,
        policy_buffer_size=40_000,
        batch_size=128,
        seed=2026,
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    trainer = Trainer(config, KuhnPokerGame(), num_workers=2)
    for t in range(1, config.outer_iters + 1):
        trainer.iter = t
        trainer._cfr_iteration_parallel(t)
        trainer._train_advantage_nets(t)
    trainer._train_policy_net()
    trainer.close_pool()

    def policy_fn(state: object, actor: int) -> torch.Tensor:  # noqa: ARG001
        feats = game.infoset_features(state).unsqueeze(0)  # type: ignore[arg-type]
        mask = torch.tensor(game.legal_mask(state), dtype=torch.float32).unsqueeze(0)  # type: ignore[arg-type]
        with torch.no_grad():
            return trainer.policy_net.forward_with_mask(feats, mask).squeeze(0)

    expl_trained = exploitability(game, policy_fn, deals)
    assert expl_trained < 0.65 * expl_uniform, (
        f"parallel-engine avg-policy exploitability {expl_trained:.4f} not < 0.65·uniform "
        f"{expl_uniform:.4f} — the parallel self-play loop isn't converging toward Nash"
    )
