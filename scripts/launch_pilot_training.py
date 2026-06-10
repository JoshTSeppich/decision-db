"""Pilot Deep CFR training run on NLHE 6-max — ~4h target.

Configuration (post-pilot v2 — see scripts/diag_per_street_depth.py for the
diagnostic that motivated the policy_buffer_size bump):
    - 250 outer iters x 400 traversals = 100,000 traversals total
    - Advantage + policy nets at full spec width (256, 256, 256)
    - Batch 256
    - advantage_buffer 500k (per player); policy_buffer 3M total
      (was 500k — diagnostic showed the policy reservoir was the binding
      constraint at 9.5% acceptance, dropping 90% of postflop samples we
      paid CPU for)
    - policy_train_steps 40k (10x previous; keeps ~2 epochs over 3M)
    - Checkpoint every 25 iters; checkpoint log includes per-street unique
      infoset-key counts in the policy reservoir so postflop coverage can be
      watched during training rather than discovered at the end.
    - LRU on AbstractionTables.lookup miss path capped at 2M entries

Usage:
    python scripts/launch_pilot_training.py \
        [--checkpoint-dir /tmp/pokerbot-checkpoints] [--seed 2026]
    # resume a died run from its last good checkpoint:
    python scripts/launch_pilot_training.py \
        --checkpoint-dir /tmp/pokerbot-checkpoints \
        --resume /tmp/pokerbot-checkpoints/iter_1000.pt

Checkpoints land in --checkpoint-dir (a large LOCAL disk), never on a small
mounted network volume — a full volume killing a multi-GB write is what
motivated splitting checkpoints away from --out.

Logs to stdout; redirect with `tee logs/training-pilot.log` when launching.
After training, exports the policy reservoir into a SQLite strategy DB.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Pin PYTHONHASHSEED for defense-in-depth even though the lookup hash is
# already deterministic via blake2b. Restart the interpreter if it isn't set —
# the user shouldn't have to remember to export it.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


from pokerbot.abstraction import AbstractionTables
from pokerbot.strategy_db import open_db
from pokerbot.training import DeepCFRConfig, SimpleNLHEGame, Trainer

# Buffer/cache sizes must be sane before we spend ~6s warming caches and load
# the abstraction. Floor guards against fat-finger tiny values (a 100-entry
# reservoir trains on noise); ceiling guards against an accidental extra zero
# allocating tens of GB and OOM-killing the run.
_BUFFER_MIN = 1_000
_BUFFER_MAX = 100_000_000


def validate_buffer_args(args: argparse.Namespace) -> None:
    """Reject out-of-range buffer/cache sizes with a clear, flag-named error.

    Raises SystemExit(2) (the argparse convention) so the launcher exits
    non-zero before any heavy work begins.
    """
    checks = (
        ("--advantage-buffer-size", args.advantage_buffer_size),
        ("--policy-buffer-size", args.policy_buffer_size),
        # --lru-max and --lru-cache-size share dest `lru_max`.
        ("--lru-max/--lru-cache-size", args.lru_max),
    )
    for flag, value in checks:
        if value < _BUFFER_MIN or value > _BUFFER_MAX:
            msg = f"error: {flag}={value} out of range [{_BUFFER_MIN}, {_BUFFER_MAX}]"
            print(msg, file=sys.stderr)
            raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("training/pilot"),
        help=(
            "Legacy run dir. Kept for back-compat; checkpoints no longer land "
            "here — use --checkpoint-dir instead."
        ),
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/pokerbot-checkpoints"),
        help=(
            "Directory where training checkpoints (iter_NNNN.pt + "
            "iter_NNNN_reservoirs.npz) are written. Point this at LARGE LOCAL "
            "container disk, NOT a small mounted network volume — a multi-GB "
            "checkpoint write killed on a full 20 GB volume is what motivated "
            "this flag. Default is an absolute local path; created if missing."
        ),
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Path to an iter_NNNN.pt checkpoint to resume from. Its sidecar "
            "iter_NNNN_reservoirs.npz must sit alongside it. Training continues "
            "from the checkpoint's iter and writes new checkpoints into "
            "--checkpoint-dir."
        ),
    )
    p.add_argument("--abstraction-dir", type=Path, default=Path("abstraction"))
    p.add_argument("--db", default="sqlite:///strategy-pilot.db")
    p.add_argument("--strategy-version", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--outer-iters", type=int, default=250)
    p.add_argument("--traversals-per-iter", type=int, default=400)
    p.add_argument("--train-steps-per-iter", type=int, default=200)
    p.add_argument(
        "--policy-train-steps",
        type=int,
        default=40_000,
        help="~2 epochs over a 3M policy reservoir at batch 256",
    )
    p.add_argument(
        "--policy-buffer-size",
        type=int,
        default=3_000_000,
        help="Capacity of the policy reservoir (was 500k in pilot v1)",
    )
    p.add_argument(
        "--advantage-buffer-size",
        type=int,
        default=500_000,
        help="Per-player advantage reservoir capacity",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument(
        "--coverage-instrument",
        action="store_true",
        help=(
            "Piece 2: accumulate per-(street x facing-bet) infoset visit counters "
            "and log single-visit fraction + visit-depth histogram at each checkpoint. "
            "opp_aggression_bias is left at its 0.0 default (plain external-sampling MCCFR)."
        ),
    )
    p.add_argument(
        "--table-size",
        type=int,
        default=6,
        choices=[2, 3, 4, 5, 6, 7, 8, 9],
        help="SimpleNLHEGame table size; trained policy is keyed on this size.",
    )
    p.add_argument(
        "--lru-max",
        "--lru-cache-size",
        dest="lru_max",
        type=int,
        default=2_000_000,
        help=(
            "LRU cap on AbstractionTables miss-path cache. "
            "--lru-cache-size is an alias for --lru-max (last one specified "
            "wins); --lru-max is kept for back-compat with existing scripts."
        ),
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip the final policy_reservoir -> StrategyDB export",
    )
    p.add_argument(
        "--skip-lbr",
        action="store_true",
        help="Disable in-loop LBR exploitability eval (use in CI smoke tests)",
    )
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    validate_buffer_args(args)  # fail fast, before any heavy work

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pilot")

    # Build the abstraction (loads NPZ + centroids).
    log.info("loading abstraction from %s/", args.abstraction_dir)
    tables = AbstractionTables(path=args.abstraction_dir, lru_max=args.lru_max)
    log.info("  loaded_streets=%s, lru_max=%d", tables.loaded_streets, args.lru_max)
    if set(tables.loaded_streets) != {"flop", "turn", "river"}:
        log.error("missing one or more postflop NPZs in %s", args.abstraction_dir)
        return 2

    # Warm the preflop-tier cache so the first traversal doesn't pay its ~6s cost.
    log.info("warming preflop-tier cache (~6s)...")
    _ = tables.lookup(
        (51, 50),
        (32, 36, 40, 1, 5),
        "river",
    )

    # Pilot config — 256x3 nets per spec §E (user's choice over my 128x3 default).
    config = DeepCFRConfig(
        outer_iters=args.outer_iters,
        traversals_per_iter=args.traversals_per_iter,
        train_steps_per_iter=args.train_steps_per_iter,
        policy_train_steps=args.policy_train_steps,
        batch_size=args.batch_size,
        advantage_hidden=(256, 256, 256),
        policy_hidden=(256, 256, 256),
        advantage_buffer_size=args.advantage_buffer_size,
        policy_buffer_size=args.policy_buffer_size,
        checkpoint_every=args.checkpoint_every,
        lbr_every=0 if args.skip_lbr else DeepCFRConfig().lbr_every,
        coverage_instrument=args.coverage_instrument,  # opp_aggression_bias left at default 0.0
        seed=args.seed,
    )
    log.info("config: %s", config)

    game = SimpleNLHEGame(tables, table_size=args.table_size)
    log.info("training at table_size=%d", args.table_size)
    # lru_max lives in the abstraction layer, not DeepCFRConfig — record it
    # (and the buffer sizes) in checkpoint metadata for run provenance.
    run_metadata = {
        "lru_max": args.lru_max,
        "advantage_buffer_size": args.advantage_buffer_size,
        "policy_buffer_size": args.policy_buffer_size,
    }
    trainer = Trainer(config, game, run_metadata=run_metadata)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_from = Path(args.resume) if args.resume else None
    log.info("checkpoints -> %s/", args.checkpoint_dir)
    if resume_from is not None:
        log.info("resuming from %s", resume_from)
    t0 = time.perf_counter()
    trainer.train(args.checkpoint_dir, resume_from=resume_from)
    log.info("trainer.train returned after %.1fmin", (time.perf_counter() - t0) / 60.0)

    # Lookup stats after training — informs whether the cache was big enough.
    stats = tables.lookup_stats()
    total_lookups = stats["hits_exact"] + stats["hits_lru"] + stats["misses"]
    hit_rate = (stats["hits_exact"] + stats["hits_lru"]) / max(total_lookups, 1)
    log.info(
        "lookup stats: %s (total=%d, hit_rate=%.3f, lru_size=%d/%d)",
        {k: stats[k] for k in ("hits_exact", "hits_lru", "misses")},
        total_lookups,
        hit_rate,
        stats["lru_size"],
        args.lru_max,
    )

    if not args.skip_export:
        log.info("exporting strategy -> %s (version=%d)", args.db, args.strategy_version)
        db = open_db(args.db)
        n = trainer.export_strategy(db, version=args.strategy_version)
        log.info("exported %d distinct infosets", n)
        db.close()

    log.info("PILOT DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
