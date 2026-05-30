"""Local serial-vs-parallel timing for the fine-tune traversal phase.

Measures the wall-clock of one CFR iteration's traversal phase (no neural train —
weights held constant) at num_workers=1 vs num_workers=N on this box. The first
iteration of the parallel run is a warm-up (pays the spawn-pool + per-worker game
build once); the timed iteration is the second, matching a steady-state run.

NOT a test — a measurement. Prints real observed seconds; asserts nothing.
"""

from __future__ import annotations

import sys
import time

import torch

from pokerbot.abstraction import AbstractionTables
from pokerbot.training import make_test_config
from zoom.agents import build_archetype_pool
from zoom.train import GatedNLHEGame
from zoom.train.finetune import FineTuneTrainer

TRAVERSALS = 200
TABLE_SIZE = 3


def _make_trainer(num_workers: int) -> FineTuneTrainer:
    cfg = make_test_config(traversals_per_iter=TRAVERSALS, seed=4242)
    torch.manual_seed(4242)
    game = GatedNLHEGame(AbstractionTables(), blinds=(5, 10), starting_stack=1000, table_size=TABLE_SIZE)
    return FineTuneTrainer(cfg, game, pool=build_archetype_pool(), num_workers=num_workers)


def _time_iter(trainer: FineTuneTrainer, t: int) -> float:
    t0 = time.perf_counter()
    trainer._cfr_iteration_parallel(t)
    return time.perf_counter() - t0


def main() -> int:
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    serial = _make_trainer(num_workers=1)
    _time_iter(serial, 1)  # warm-up (LRU caches etc.)
    serial_dt = _time_iter(serial, 2)
    serial.close_pool()
    print(f"serial  (num_workers=1): {serial_dt:.2f}s / iter ({TRAVERSALS} traversals)")

    par = _make_trainer(num_workers=n_workers)
    _time_iter(par, 1)  # warm-up (spawn pool + per-worker game build)
    par_dt = _time_iter(par, 2)
    par.close_pool()
    print(f"parallel(num_workers={n_workers}): {par_dt:.2f}s / iter ({TRAVERSALS} traversals)")

    if par_dt > 0:
        print(f"speedup: {serial_dt / par_dt:.2f}x on {n_workers} workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
