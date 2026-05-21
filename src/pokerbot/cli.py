"""Pokerbot CLI — entry points for every build step in Spec.html §I.

Subcommands that aren't wired yet stay in `_PENDING` and return rc=1 with a
pointer to the relevant build step.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pokerbot import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

_PROG = "pokerbot"

# Wired subcommands: name → handler. Anything not in here lives in _PENDING.
_PENDING: dict[str, str] = {
    "train": "step 5 — training.deepcfr (Spec.html §E)",
    "evaluate": "step 5 — training.deepcfr (Spec.html §E)",
    "export-strategy": "step 5 — training.export   (Spec.html §E)",
    "serve": "step 6 — runtime.adapter   (Spec.html §F)",
    "build-pushfold": "step 7 — tournament.build_pushfold (Spec.html §G)",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Decision DB pokerbot. See Spec.html for the architecture.",
    )
    parser.add_argument("--version", action="version", version=f"{_PROG} {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_build = sub.add_parser("build-abstraction", help="Build card abstraction NPZ artifacts (§A).")
    p_build.add_argument(
        "--out", required=True, type=Path, help="Output directory for buckets_*.npz"
    )
    p_build.add_argument(
        "--streets",
        default="flop,turn,river",
        help="Comma-separated subset of {flop,turn,river}. Default: all three.",
    )
    p_build.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="MC samples per canonical class (default 1000; smoke=50)",
    )
    p_build.add_argument(
        "--inner-samples",
        type=int,
        default=100,
        help="Inner MC samples for EHS² histograms on flop/turn (default 100)",
    )
    p_build.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Cap canonical classes per street (default: none, full)",
    )
    p_build.add_argument("--num-buckets", type=int, default=200)
    p_build.add_argument("--kmeans-iters", type=int, default=20)
    p_build.add_argument("--seed", type=int, default=0xC0FFEE)

    p_train = sub.add_parser("train", help="Run Deep CFR training loop (§E).")
    p_train.add_argument("--config", help="Path to DeepCFRConfig override (TOML/JSON)")
    p_train.add_argument("--resume", help="Checkpoint path to resume from")
    p_train.add_argument("--out", required=True, help="Checkpoints + logs directory")

    p_eval = sub.add_parser("evaluate", help="Run LBR + head-to-head evaluation (§E).")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--opponents", nargs="*", default=[])

    p_export = sub.add_parser("export-strategy", help="Export policy_reservoir → SQLite (§E).")
    p_export.add_argument("--from", dest="from_ckpt", required=True, metavar="CHECKPOINT")
    p_export.add_argument("--to", required=True, metavar="DB_URI")
    p_export.add_argument("--version", type=int, required=True, dest="strategy_version")

    p_serve = sub.add_parser("serve", help="Run runtime adapter as a service (§F).")
    p_serve.add_argument("--db", required=True, help="StrategyDB URI (sqlite:/// or lmdb:///)")
    p_serve.add_argument("--abstraction", required=True, help="Abstraction tables directory")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--seed", type=int, default=None)

    p_pf = sub.add_parser("build-pushfold", help="Solve push/fold Nash table (§G).")
    p_pf.add_argument("--out", required=True)

    return parser


def _run_build_abstraction(args: argparse.Namespace) -> int:
    from pokerbot.abstraction.build import (
        BuildConfig,
        build_flop_buckets,
        build_river_buckets,
        build_turn_buckets,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    streets = {s.strip() for s in args.streets.split(",") if s.strip()}
    unknown = streets - {"flop", "turn", "river"}
    if unknown:
        print(f"unknown streets: {sorted(unknown)}", file=sys.stderr)
        return 2

    cfg = BuildConfig(
        num_buckets=args.num_buckets,
        samples_per_class=args.samples,
        inner_samples=args.inner_samples,
        kmeans_iters=args.kmeans_iters,
        max_classes=args.max_classes,
        rng_seed=args.seed,
    )
    out_dir: Path = args.out
    builders = [
        ("flop", build_flop_buckets),
        ("turn", build_turn_buckets),
        ("river", build_river_buckets),
    ]
    for street, fn in builders:
        if street not in streets:
            continue
        fn(out_dir, cfg)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command: str = args.command

    if command == "build-abstraction":
        return _run_build_abstraction(args)

    pending = _PENDING.get(command)
    if pending is None:
        parser.error(f"unknown command {command!r}")
    print(f"{_PROG} {command!r} is not yet implemented — pending {pending}.", file=sys.stderr)
    print("See Spec.html in the project root for the full build plan.", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
