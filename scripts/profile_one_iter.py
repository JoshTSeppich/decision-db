"""Profile one outer iteration of Deep CFR with the real abstraction.

Reports total wall-clock and percentage of time in each hot path:
    - AbstractionTables.lookup (incl. project_to_centroid)
    - copy.deepcopy on pk_state (game branching)
    - advantage_nets[actor] forward pass (regret-match input)

Usage: python scripts/profile_one_iter.py [--traversals 50]
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

from pokerbot.abstraction import AbstractionTables
from pokerbot.training import (
    SimpleNLHEGame,
    Trainer,
    make_test_config,
)
from pokerbot.training.nets import AdvantageNet

# ─────────── instrumentation ───────────


class Counter:
    __slots__ = ("calls", "total_time")

    def __init__(self) -> None:
        self.total_time = 0.0
        self.calls = 0

    def add(self, dt: float) -> None:
        self.total_time += dt
        self.calls += 1


def install_hooks() -> dict[str, Counter]:
    counters = {
        "lookup": Counter(),
        "deepcopy": Counter(),
        "net_forward": Counter(),
    }

    # Wrap AbstractionTables.lookup
    orig_lookup = AbstractionTables.lookup

    def wrapped_lookup(self, hole, board, street):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        try:
            return orig_lookup(self, hole, board, street)
        finally:
            counters["lookup"].add(time.perf_counter() - t0)

    AbstractionTables.lookup = wrapped_lookup  # type: ignore[assignment, method-assign]

    # Wrap copy.deepcopy
    orig_deepcopy = copy.deepcopy

    def wrapped_deepcopy(x, memo=None, _nil=None):  # type: ignore[no-untyped-def]
        if _nil is None:
            _nil = []
        t0 = time.perf_counter()
        try:
            return orig_deepcopy(x, memo, _nil)
        finally:
            counters["deepcopy"].add(time.perf_counter() - t0)

    copy.deepcopy = wrapped_deepcopy  # type: ignore[assignment]

    # Wrap AdvantageNet.forward
    orig_forward = AdvantageNet.forward

    def wrapped_forward(self, x):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        try:
            return orig_forward(self, x)
        finally:
            counters["net_forward"].add(time.perf_counter() - t0)

    AdvantageNet.forward = wrapped_forward  # type: ignore[assignment, method-assign]

    return counters


# ─────────── main ───────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traversals", type=int, default=50)
    parser.add_argument("--abstraction-dir", default="abstraction")
    args = parser.parse_args()

    print(f"Loading abstraction from {args.abstraction_dir}/ …")
    tables = AbstractionTables(path=Path(args.abstraction_dir))
    print(f"  loaded_streets: {tables.loaded_streets}")

    # Warm preflop tier cache (~6s) before profiling so the first lookup
    # doesn't dominate. project_to_centroid lazily builds tiers on first call.
    print("Warming preflop-tier cache (~6s) …")
    _ = tables.lookup(
        (51, 50),
        (32, 36, 40, 1, 5),
        "river",
    )

    config = make_test_config(
        outer_iters=1,
        traversals_per_iter=args.traversals,
        train_steps_per_iter=20,
        policy_train_steps=20,
        batch_size=32,
        advantage_buffer_size=2000,
        policy_buffer_size=2000,
        advantage_hidden=(64, 64),
        policy_hidden=(64, 64),
        checkpoint_every=1,
        seed=2026,
    )
    print("Game: SimpleNLHEGame(table_size=6, blinds=(5,10))")
    print(f"Config: outer=1, traversals={args.traversals}, train_steps_per_iter=20, nets=64x2")

    game = SimpleNLHEGame(tables, table_size=6)
    counters = install_hooks()
    trainer = Trainer(config, game)

    t0 = time.perf_counter()
    trainer.train(Path("/tmp/profile_one_iter"))
    total = time.perf_counter() - t0

    # The "iteration" body splits into: CFR traversal loop + advantage-net training.
    # The hooks count totals across both phases. For accuracy we surface
    # call counts so the user can sanity-check.
    print()
    print(f"=== one outer iteration ({args.traversals} traversals) ===")
    print(f"total wall-clock: {total:.2f}s")
    print()
    print(f"{'hot path':<14} {'calls':>10} {'total (s)':>10} {'%':>6} {'avg (μs)':>10}")
    print("-" * 56)
    for name, c in counters.items():
        pct = 100.0 * c.total_time / total if total > 0 else 0.0
        avg_us = (c.total_time / c.calls * 1e6) if c.calls else 0.0
        print(f"{name:<14} {c.calls:>10d} {c.total_time:>10.3f} {pct:>5.1f}% {avg_us:>9.0f}")

    other = total - sum(c.total_time for c in counters.values())
    print(f"{'other':<14} {'':>10} {other:>10.3f} {100 * other / total:>5.1f}%")

    print()
    print("=== infosets per second (proxy for training throughput) ===")
    n_decisions = counters["net_forward"].calls
    print(
        f"  {n_decisions} decision points across {args.traversals} traversals "
        f"= {n_decisions / args.traversals:.1f} decisions/hand"
    )
    print(f"  effective decision rate: {n_decisions / total:.1f}/s")


if __name__ == "__main__":
    main()
