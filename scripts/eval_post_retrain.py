"""Post-retrain evaluation: 5 seeds x 5000 hands, with pass criteria.

Runs `scripts/eval_head_to_head.py` at five seeds (default 2026..2030) at
N=5000 each. Aggregates:
  - per-seed mbb/hand + 95% CI
  - pooled (across all 25k hands) mbb/hand + 95% CI
  - per-street fallback breakdown averaged across seeds
  - explicit verdict against the post-retrain pass criteria:
      preflop exact-hit ≥ 80%
      flop exact-hit    ≥ 50%
      turn exact-hit    ≥ 15%   (warn: 12-15%; escalate if < 12%)
      pooled trained mbb/hand > 0 with CI lower bound > 0

Usage:
    python scripts/eval_post_retrain.py [--db strategy-pilot.db]
                                        [--seeds 2026 2027 2028 2029 2030]
                                        [--n-hands 5000]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pokerbot.abstraction import AbstractionTables
from pokerbot.runtime import RuntimeAdapter
from pokerbot.strategy_db import open_db
from pokerbot.training import SimpleNLHEGame

# Import the eval primitives directly so we don't shell out per seed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_head_to_head import (  # type: ignore[import-not-found]
    _STREET_NAMES,
    _empty_fallback_table,
    _play_one_hand,
)

if TYPE_CHECKING:
    from pokerbot.runtime.schema import FallbackUsed


# ───────── single-seed driver ─────────


def _run_single_seed(
    *,
    seed: int,
    n_hands: int,
    abstraction: AbstractionTables,
    db_path: Path,
    starting_stack: int,
    sb: int,
    bb: int,
) -> tuple[list[int], dict[int, dict[FallbackUsed, int]], int, int, float]:
    """Return (per_hand_trained_deltas, fallback_by_street_totals,
    total_decisions, total_remaps, elapsed_s)."""
    trained_db = open_db(f"sqlite:///{db_path.resolve()}")
    default_db = open_db("sqlite:///:memory:")
    default_db.set_current_version(1)

    trained_adapter = RuntimeAdapter(db=trained_db, abstraction=abstraction, rng_seed=seed)
    default_adapter = RuntimeAdapter(
        db=default_db, abstraction=abstraction, rng_seed=seed ^ 0xDEADBEEF
    )

    game = SimpleNLHEGame(
        abstraction,
        blinds=(sb, bb),
        starting_stack=starting_stack,
        table_size=6,
    )

    rng = random.Random(seed)
    per_hand_trained: list[int] = []
    fallback_totals = _empty_fallback_table()
    total_decisions = 0
    total_remaps = 0

    t0 = time.perf_counter()
    for i in range(n_hands):
        result = _play_one_hand(i, game, trained_adapter, default_adapter, rng)
        per_hand_trained.append(result.trained_delta)
        for s, by_kind in result.fallback_by_street.items():
            for k, v in by_kind.items():
                fallback_totals[s][k] += v
        total_decisions += result.decisions
        total_remaps += result.illegal_remaps
    elapsed = time.perf_counter() - t0

    trained_db.close()
    default_db.close()
    return per_hand_trained, fallback_totals, total_decisions, total_remaps, elapsed


# ───────── stats ─────────


def _mbb_stats(per_hand_deltas: list[int], bb_size: int) -> tuple[float, float, float, float]:
    """Return (mean_mbb, ci_low_mbb, ci_high_mbb, stderr_mbb)."""
    n = len(per_hand_deltas)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(per_hand_deltas, dtype=np.float64)
    mean_chips = float(arr.mean())
    stderr_chips = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    mean_mbb = mean_chips / bb_size * 1000.0
    half = 1.96 * stderr_chips / bb_size * 1000.0
    stderr_mbb = stderr_chips / bb_size * 1000.0
    return mean_mbb, mean_mbb - half, mean_mbb + half, stderr_mbb


def _exact_hit_rate(fb: dict[int, dict[FallbackUsed, int]], street: int) -> float:
    row = fb[street]
    total = sum(row.values())
    return row["exact"] / total if total > 0 else 0.0


# ───────── report ─────────


def _print_seed_summary(
    seed: int,
    per_hand: list[int],
    fb: dict[int, dict[FallbackUsed, int]],
    decisions: int,
    remaps: int,
    elapsed: float,
    bb_size: int,
) -> tuple[float, float, float]:
    """Print one seed's summary and return (mean_mbb, ci_lo, ci_hi)."""
    mean, lo, hi, _se = _mbb_stats(per_hand, bb_size)
    remap_frac = remaps / decisions if decisions else 0.0
    print(
        f"  seed {seed}: {len(per_hand):>5d} hands in {elapsed:5.1f}s  "
        f"trained {mean:+8.2f} mbb/hand  CI [{lo:+8.2f}, {hi:+8.2f}]  "
        f"remaps {remap_frac:.1%}"
    )
    print(
        f"    fallback:  "
        f"PF exact {_exact_hit_rate(fb, 0):5.1%}  "
        f"FL exact {_exact_hit_rate(fb, 1):5.1%}  "
        f"TN exact {_exact_hit_rate(fb, 2):5.1%}  "
        f"RV exact {_exact_hit_rate(fb, 3):5.1%}"
    )
    return mean, lo, hi


def _print_pooled_summary(
    pooled_deltas: list[int],
    pooled_fb: dict[int, dict[FallbackUsed, int]],
    bb_size: int,
) -> tuple[float, float, float, dict[int, float]]:
    print(f"\n── pooled across all seeds ({len(pooled_deltas):,} hands) ──")
    mean, lo, hi, se = _mbb_stats(pooled_deltas, bb_size)
    print(f"  trained mbb/hand:  {mean:+8.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]   stderr {se:.2f}")
    print(f"  default mbb/hand:  {-mean:+8.2f}   95% CI [{-hi:+.2f}, {-lo:+.2f}]   (= -trained)")

    exact_by_street: dict[int, float] = {}
    print("\n  ── pooled per-street fallback ──")
    print(f"    {'street':<8s}  {'exact':>14s}  {'nearest_nbr':>14s}  {'default_pol':>14s}")
    for s in range(4):
        row = pooled_fb[s]
        total = sum(row.values())
        if total == 0:
            continue
        ex = row["exact"] / total
        nn = row["nearest_neighbor"] / total
        dp = row["default_policy"] / total
        exact_by_street[s] = ex
        print(
            f"    {_STREET_NAMES[s]:<8s}  "
            f"{row['exact']:>6d} ({ex:5.1%})  "
            f"{row['nearest_neighbor']:>6d} ({nn:5.1%})  "
            f"{row['default_policy']:>6d} ({dp:5.1%})"
        )
    return mean, lo, hi, exact_by_street


def _verdict(
    pooled_mean: float,
    pooled_lo: float,
    exact_by_street: dict[int, float],
) -> int:
    """Return process exit code. 0 = pass, 1 = soft-fail (warn), 2 = hard-fail."""
    print("\n── pass-criteria checks ──")
    fails: list[str] = []
    warns: list[str] = []

    pf = exact_by_street.get(0, 0.0)
    fl = exact_by_street.get(1, 0.0)
    tn = exact_by_street.get(2, 0.0)

    def _check(name: str, value: float, hard: float, soft: float | None = None) -> None:
        if value >= hard:
            print(f"  ✓ {name}: {value:.1%} (≥ {hard:.0%})")
        elif soft is not None and value >= soft:
            warns.append(f"{name}: {value:.1%} (between {soft:.0%} and {hard:.0%} — soft pass)")
            print(f"  ~ {name}: {value:.1%} (soft pass; below {hard:.0%} but ≥ {soft:.0%})")
        else:
            fails.append(f"{name}: {value:.1%} (< {hard:.0%})")
            print(
                f"  ✗ {name}: {value:.1%} (< {hard:.0%}"
                + (f", < soft {soft:.0%}" if soft is not None else "")
                + ")"
            )

    _check("preflop exact-hit", pf, hard=0.80)
    _check("flop    exact-hit", fl, hard=0.50)
    _check("turn    exact-hit", tn, hard=0.15, soft=0.12)

    if pooled_mean > 0 and pooled_lo > 0:
        print(
            f"  ✓ trained mbb/hand > 0 with CI excluding zero: "
            f"mean={pooled_mean:+.2f}, lo={pooled_lo:+.2f}"
        )
    elif pooled_mean > 0:
        warns.append(
            f"trained mbb/hand: mean={pooled_mean:+.2f} positive but CI lo={pooled_lo:+.2f} ≤ 0"
        )
        print(
            f"  ~ trained mbb/hand: mean={pooled_mean:+.2f} positive but "
            f"CI lo={pooled_lo:+.2f} crosses zero — run more hands"
        )
    else:
        fails.append(
            f"trained mbb/hand: mean={pooled_mean:+.2f} ≤ 0 (need > 0 with CI excluding zero)"
        )
        print(
            f"  ✗ trained mbb/hand: mean={pooled_mean:+.2f} ≤ 0 (need > 0 with CI excluding zero)"
        )

    print("\n── verdict ──")
    if fails:
        print("  FAIL — hard criteria missed:")
        for f in fails:
            print(f"    - {f}")
        if tn < 0.12:
            print(
                "  → escalation: turn exact-hit < 12%; bump traversals_per_iter "
                "(e.g., 400 → 800) and re-train."
            )
        return 2
    if warns:
        print("  SOFT PASS — review warnings:")
        for w in warns:
            print(f"    - {w}")
        return 1
    print("  PASS — all hard criteria met.")
    return 0


# ───────── main ─────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="strategy-pilot.db")
    p.add_argument("--abstraction-dir", default="abstraction")
    p.add_argument("--n-hands", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--sb", type=int, default=5)
    p.add_argument("--bb", type=int, default=10)
    args = p.parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: trained DB not found at {db_path}", file=sys.stderr)
        return 2

    print(f"Resolved DB path: {db_path}")
    print(f"DB file size:     {db_path.stat().st_size / 1e6:.1f} MB")
    print(f"Loading AbstractionTables from {args.abstraction_dir!r}…")
    abstraction = AbstractionTables(path=args.abstraction_dir)
    if set(abstraction.loaded_streets) != {"flop", "turn", "river"}:
        print(
            f"WARNING: AbstractionTables loaded only {abstraction.loaded_streets}; "
            "miss-path uses placeholder hashes.",
            file=sys.stderr,
        )

    print(
        f"\nRunning {len(args.seeds)} seeds x {args.n_hands} hands "
        f"({len(args.seeds) * args.n_hands:,} hands total)\n"
    )

    pooled_deltas: list[int] = []
    pooled_fb = _empty_fallback_table()
    for seed in args.seeds:
        per_hand, fb, decisions, remaps, elapsed = _run_single_seed(
            seed=seed,
            n_hands=args.n_hands,
            abstraction=abstraction,
            db_path=db_path,
            starting_stack=args.starting_stack,
            sb=args.sb,
            bb=args.bb,
        )
        _print_seed_summary(seed, per_hand, fb, decisions, remaps, elapsed, args.bb)
        pooled_deltas.extend(per_hand)
        for s in range(4):
            for k, v in fb[s].items():
                pooled_fb[s][k] += v

    pooled_mean, pooled_lo, _hi, exact_by_street = _print_pooled_summary(
        pooled_deltas, pooled_fb, args.bb
    )
    return _verdict(pooled_mean, pooled_lo, exact_by_street)


if __name__ == "__main__":
    raise SystemExit(main())
