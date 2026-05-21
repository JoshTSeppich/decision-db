"""Verify the production OCHS / potential-aware NPZs after the overnight build.

Runs the user's spec-mandated checks:
    1. All three NPZs present + sizes
    2. AbstractionTables loads them
    3. OCHS quartile sanity (royal flush in top quartile; trash in bottom)
    4. Full pytest suite green

Usage: python scripts/verify_artifacts.py [--abstraction-dir abstraction/]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from pokerbot.abstraction import AbstractionTables
from pokerbot.abstraction.cards import parse_card


def _check_files(abstraction_dir: Path) -> None:
    print(f"== NPZ files in {abstraction_dir}/")
    for street in ("flop", "turn", "river"):
        p = abstraction_dir / f"buckets_{street}.npz"
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name}: {size_mb:.2f} MB")


def _check_load(abstraction_dir: Path) -> AbstractionTables:
    print("== AbstractionTables load")
    tables = AbstractionTables(path=abstraction_dir)
    print(f"  loaded_streets: {tables.loaded_streets}")
    if set(tables.loaded_streets) != {"flop", "turn", "river"}:
        raise SystemExit(f"missing streets: expected all 3, got {tables.loaded_streets}")
    return tables


def _ochs_sanity(abstraction_dir: Path, tables: AbstractionTables) -> None:
    print("== OCHS quartile sanity check")
    nut_hole = (parse_card("Ah"), parse_card("Kh"))
    nut_board = (
        parse_card("Qh"),
        parse_card("Jh"),
        parse_card("Th"),
        parse_card("2c"),
        parse_card("3c"),
    )
    bot_hole = (parse_card("2c"), parse_card("3d"))
    bot_board = (
        parse_card("As"),
        parse_card("Ks"),
        parse_card("Qs"),
        parse_card("7h"),
        parse_card("8d"),
    )
    nut_bucket = tables.lookup(nut_hole, nut_board, "river")
    bot_bucket = tables.lookup(bot_hole, bot_board, "river")

    with np.load(abstraction_dir / "buckets_river.npz") as npz:
        centroids = npz["centroids"]
    mean_eq = centroids.mean(axis=1)
    ranking = mean_eq.argsort()
    k = len(centroids)
    top_q = {int(b) for b in ranking[(3 * k) // 4 :]}
    bot_q = {int(b) for b in ranking[: k // 4]}

    rank_of = {int(b): i for i, b in enumerate(ranking)}
    print(
        f"  nut bucket: {nut_bucket} (rank {rank_of.get(nut_bucket, '?')}/{k})  in top quartile: {nut_bucket in top_q}"
    )
    print(
        f"  bot bucket: {bot_bucket} (rank {rank_of.get(bot_bucket, '?')}/{k})  in bot quartile: {bot_bucket in bot_q}"
    )

    failures = []
    if nut_bucket not in top_q:
        failures.append("nut hand not in top quartile")
    if bot_bucket not in bot_q:
        failures.append("bottom-pair not in bottom quartile")
    if failures:
        raise SystemExit("OCHS sanity FAILED: " + "; ".join(failures))
    print("  OCHS sanity check PASSED on production artifact")


def _run_pytest() -> None:
    print("== full pytest suite")
    rc = subprocess.call(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if rc != 0:
        raise SystemExit(f"pytest failed (rc={rc})")
    print("  pytest PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abstraction-dir",
        type=Path,
        default=Path("abstraction"),
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="skip the full pytest re-run (faster, useful for partial checks)",
    )
    args = parser.parse_args()

    abstraction_dir = args.abstraction_dir.resolve()
    _check_files(abstraction_dir)
    tables = _check_load(abstraction_dir)
    _ochs_sanity(abstraction_dir, tables)
    if not args.skip_pytest:
        _run_pytest()
    print("\nALL VERIFICATIONS PASSED")


if __name__ == "__main__":
    main()
