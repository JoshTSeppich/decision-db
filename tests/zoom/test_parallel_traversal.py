"""Equivalence + determinism proofs for the parallel traversal engine.

This is a concurrency change to the Deep CFR fine-tune loop — the highest
silent-corruption risk in the codebase. The bar is not "it runs faster" but
"provably equivalent to the serial version". The headline guard is
`test_parallel_engine_bit_identical_to_serial`: the new engine run serially
(`num_workers=1`) and in parallel (`num_workers=N`) must produce bit-identical
reservoirs — the only difference between the two paths is the executor, so any
divergence is a concurrency bug.

Why bit-identity is achievable here (and is NOT claimed against the *legacy*
single-RNG loop): each traversal draws from its own RNG derived from
`(master_seed, iter_t, traversal_index)`, writes to a private buffer, and the
main process merges the buffers into the shared reservoir in a fixed global order
with a single dedicated merge RNG. Serial and parallel share that seeding + merge,
so they agree to the bit. The legacy loop interleaves reservoir-eviction RNG into
the per-traversal stream, which is inherently sequential — equivalence to it is
statistical (same algorithm), evidenced by `test_parallel_engine_converges_to_kuhn_nash`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from zoom.agents import build_archetype_pool
from zoom.train import GatedNLHEGame
from zoom.train.finetune import FineTuneTrainer

from pokerbot.abstraction import AbstractionTables
from pokerbot.training import (
    exploitability,
    make_test_config,
    make_uniform_strategy_fn,
)
from pokerbot.training.kuhn import KuhnPokerGame

if TYPE_CHECKING:
    from pathlib import Path

    from pokerbot.training.traversal import Reservoir


def _enumerate_kuhn_deals() -> list[tuple[object, float]]:
    from pokerbot.training.kuhn import KuhnState

    return [
        (KuhnState(cards=(c0, c1)), 1.0 / 6.0)
        for c0 in range(3)
        for c1 in range(3)
        if c0 != c1
    ]


def _gated_3max() -> GatedNLHEGame:
    return GatedNLHEGame(AbstractionTables(), blinds=(5, 10), starting_stack=1000, table_size=3)


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


def _assert_trainers_reservoirs_equal(a: FineTuneTrainer, b: FineTuneTrainer) -> None:
    for i, (ra, rb) in enumerate(zip(a.advantage_reservoirs, b.advantage_reservoirs, strict=True)):
        _assert_reservoir_bit_identical(ra, rb, f"advantage_reservoir[{i}]")
    _assert_reservoir_bit_identical(a.policy_reservoir, b.policy_reservoir, "policy_reservoir")


def _build_trainer(num_workers: int, *, seed: int = 4242) -> FineTuneTrainer:
    """A fine-tune trainer with identical initial net weights for a given seed."""
    config = make_test_config(
        traversals_per_iter=40,
        train_steps_per_iter=20,
        advantage_hidden=(32, 32),
        advantage_buffer_size=4000,
        batch_size=64,
        seed=seed,
    )
    torch.manual_seed(seed)  # identical initial advantage-net weights across trainers
    np.random.seed(seed)
    game = _gated_3max()
    return FineTuneTrainer(config, game, pool=build_archetype_pool(), num_workers=num_workers)


def _run_parallel_iters(trainer: FineTuneTrainer, k: int) -> None:
    """Run k full iterations (parallel traversal + advantage train) then release the pool.

    Running >=2 iters is deliberate: iter 2's traversal uses iter-1-trained weights,
    so a worker that fails to load the pushed net state_dicts (stale-weights bug)
    diverges here even though iter 1 would have matched.
    """
    for t in range(1, k + 1):
        trainer.iter = t
        trainer._cfr_iteration_parallel(t)
        trainer._train_advantage_nets(t)
    trainer.close_pool()


# ─────────── THE safety case: serial ≡ parallel, bit-for-bit ───────────


def test_parallel_engine_bit_identical_to_serial() -> None:
    serial = _build_trainer(num_workers=1)
    parallel = _build_trainer(num_workers=4)

    _run_parallel_iters(serial, k=2)
    _run_parallel_iters(parallel, k=2)

    _assert_trainers_reservoirs_equal(serial, parallel)
    # Net weights must also match: identical reservoirs + rng + start weights → identical training.
    for i, (ns, np_) in enumerate(
        zip(serial.advantage_nets, parallel.advantage_nets, strict=True)
    ):
        for k_, v in ns.state_dict().items():
            torch.testing.assert_close(
                v, np_.state_dict()[k_], rtol=0, atol=0, msg=f"net[{i}].{k_} diverged"
            )


def test_two_parallel_runs_are_deterministic() -> None:
    a = _build_trainer(num_workers=4)
    b = _build_trainer(num_workers=4)
    _run_parallel_iters(a, k=2)
    _run_parallel_iters(b, k=2)
    _assert_trainers_reservoirs_equal(a, b)


def test_total_sample_count_matches_serial() -> None:
    """Sum of merged samples is identical serial vs parallel — guards a worker
    silently dropping a traversal (e.g. a swallowed exception)."""
    serial = _build_trainer(num_workers=1)
    parallel = _build_trainer(num_workers=4)
    _run_parallel_iters(serial, k=2)
    _run_parallel_iters(parallel, k=2)

    serial_total = sum(r.total_seen for r in serial.advantage_reservoirs) + serial.policy_reservoir.total_seen
    parallel_total = (
        sum(r.total_seen for r in parallel.advantage_reservoirs) + parallel.policy_reservoir.total_seen
    )
    assert serial_total == parallel_total > 0, (serial_total, parallel_total)


# ─────────── faithful CFR: the new path still converges on Kuhn ───────────


def test_parallel_engine_converges_to_kuhn_nash(tmp_path: Path) -> None:
    """The per-traversal-RNG refactor is faithful Deep CFR, not just self-consistent:
    self-play through the parallel engine drives the average policy's exploitability
    well below uniform on Kuhn — mirrors test_finetune's serial convergence gate."""
    game = KuhnPokerGame()
    deals = _enumerate_kuhn_deals()
    expl_uniform = exploitability(game, make_uniform_strategy_fn(game), deals)

    config = make_test_config(
        outer_iters=150,
        traversals_per_iter=64,
        train_steps_per_iter=96,
        policy_train_steps=3000,
        advantage_hidden=(32, 32),
        policy_hidden=(32, 32),
        advantage_buffer_size=40_000,
        policy_buffer_size=40_000,
        batch_size=128,
        seed=2026,
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    trainer = FineTuneTrainer(config, KuhnPokerGame(), pool=None, num_workers=2)
    for t in range(1, config.outer_iters + 1):
        trainer.iter = t
        trainer._cfr_iteration_parallel(t)
        trainer._train_advantage_nets(t)
    trainer._train_policy_net()
    trainer.close_pool()

    def policy_fn(state: object, actor: int) -> torch.Tensor:
        feats = game.infoset_features(state).unsqueeze(0)  # type: ignore[arg-type]
        mask = torch.tensor(game.legal_mask(state), dtype=torch.float32).unsqueeze(0)  # type: ignore[arg-type]
        with torch.no_grad():
            return trainer.policy_net.forward_with_mask(feats, mask).squeeze(0)

    expl_trained = exploitability(game, policy_fn, deals)
    assert expl_trained < 0.65 * expl_uniform, (
        f"parallel-engine avg-policy exploitability {expl_trained:.4f} not < 0.65·uniform "
        f"{expl_uniform:.4f} — the parallel loop isn't converging toward Nash"
    )
