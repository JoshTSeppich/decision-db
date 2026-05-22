"""Convert a Deep CFR `.pt` checkpoint into a SQLite StrategyDB.

Extracted from the final phase of `scripts/launch_pilot_training.py` so partial
checkpoints (training stopped early or crashed) can still ship as policy DBs
without re-running training.

Usage:
    python scripts/checkpoint_to_db.py \\
        --checkpoint training/cloud-batch-1/pilot-v4-3-iter_0100.pt \\
        --db sqlite:///training/cloud-batch-1/pilot-v4-3-iter_0100.db \\
        --table-size 3 \\
        --abstraction-dir abstraction \\
        --strategy-version 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


from pokerbot.abstraction import AbstractionTables
from pokerbot.strategy_db import open_db
from pokerbot.training import SimpleNLHEGame, Trainer


def _to_db_uri(arg: str) -> str:
    """Mirror `scripts/serve.py:_to_db_url`: accept a path or a sqlite:// URI."""
    if arg.startswith(("sqlite:", "lmdb:")):
        return arg
    return f"sqlite:///{Path(arg).resolve()}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--db",
        required=True,
        help=(
            "Output DB. Either a path (relative paths are resolved to "
            "absolute) or a full sqlite:/// URI."
        ),
    )
    p.add_argument(
        "--table-size",
        type=int,
        required=True,
        choices=[2, 3, 4, 5, 6, 7, 8, 9],
    )
    p.add_argument("--abstraction-dir", type=Path, default=Path("abstraction"))
    p.add_argument("--strategy-version", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("ckpt2db")

    if not args.checkpoint.exists():
        log.error("checkpoint not found: %s", args.checkpoint)
        return 2

    import torch  # local import so the help screen is fast

    log.info("loading checkpoint %s", args.checkpoint)
    t0 = time.perf_counter()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    log.info("  loaded in %.1fs; iter=%d", time.perf_counter() - t0, int(ckpt["iter"]))

    config = ckpt["config"]
    log.info("  config: %s", config)

    n_advantage = len(ckpt["advantage_states"])
    if n_advantage != args.table_size:
        log.error(
            "checkpoint advantage_states count (%d) != --table-size (%d). "
            "Either the checkpoint was trained at a different table_size, or "
            "the wrong --table-size was passed.",
            n_advantage,
            args.table_size,
        )
        return 3

    log.info("loading abstraction from %s/", args.abstraction_dir)
    tables = AbstractionTables(path=args.abstraction_dir, lru_max=2_000_000)
    if set(tables.loaded_streets) != {"flop", "turn", "river"}:
        log.error("missing one or more postflop NPZs in %s", args.abstraction_dir)
        return 2

    game = SimpleNLHEGame(tables, table_size=args.table_size)
    trainer = Trainer(config, game)
    log.info("reconstructed trainer at table_size=%d", args.table_size)

    trainer.load_checkpoint(args.checkpoint)
    log.info(
        "  loaded checkpoint state; policy_reservoir size=%d / capacity=%d, total_seen=%d",
        trainer.policy_reservoir.size,
        trainer.policy_reservoir.capacity,
        trainer.policy_reservoir.total_seen,
    )

    db_uri = _to_db_uri(args.db)
    log.info("exporting strategy -> %s (version=%d)", db_uri, args.strategy_version)
    export_t0 = time.perf_counter()
    db = open_db(db_uri)
    try:
        n = trainer.export_strategy(db, version=args.strategy_version)
    finally:
        db.close()
    log.info(
        "exported %d distinct infosets in %.1fs",
        n,
        time.perf_counter() - export_t0,
    )

    log.info("DONE (total %.1fs)", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
