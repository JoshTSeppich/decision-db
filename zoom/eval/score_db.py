"""Diagnostic: score an exported StrategyDB through the trustworthy Stage-1 gate.

Pure assembly of already-tested Component-4 pieces — NO new scoring logic:
  open_db -> make_db_spot_policy (history-aware + nearest-neighbor, visit_count
  tiebreak) -> profile_spot_policy (production tally loop, ungated SimpleNLHEGame so
  shoves are expressible) -> evaluate_bands. Reports VPIP / PFR / non-committed
  preflop ALL_IN (the GATED band) / raw preflop ALL_IN / fold-to-c-bet / AF, one
  line per seed plus the band PASS/FAIL.

Run: python -m zoom.eval.score_db --db sqlite:///path.db --abstraction-dir abstraction \
        --table-size 3 --n-hands 400 --seeds 1,2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pokerbot.abstraction import AbstractionTables
from pokerbot.strategy_db import open_db
from pokerbot.training import SimpleNLHEGame
from zoom.eval import (
    ScreenCoverageError,
    catastrophic_screen,
    evaluate_bands,
    make_db_spot_policy,
    profile_spot_policy,
)
from zoom.eval.bands import bands_for


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a StrategyDB through the Stage-1 gate")
    p.add_argument("--db", required=True, help="StrategyDB URL (sqlite:///path)")
    p.add_argument("--abstraction-dir", type=Path, default=Path("abstraction"))
    p.add_argument("--table-size", type=int, default=3)
    p.add_argument("--bb", type=int, default=10)
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--n-hands", type=int, default=400)
    p.add_argument("--seeds", default="1,2", help="comma-separated profiling seeds")
    a = p.parse_args(argv)

    abstraction = AbstractionTables(path=a.abstraction_dir)
    db = open_db(a.db)
    game = SimpleNLHEGame(
        abstraction, blinds=(a.bb // 2, a.bb), starting_stack=a.starting_stack,
        table_size=a.table_size,
    )
    policy = make_db_spot_policy(db, abstraction, bb=a.bb, table_size=a.table_size)
    bands = bands_for(a.table_size)

    print(f"DB={a.db}  version={db.current_version()}  table_size={a.table_size}  n_hands={a.n_hands}")
    print(f"bands={bands}")

    # Catastrophic screen (first-in open) — table-size aware. A coverage miss is loud.
    try:
        viols = catastrophic_screen(policy, table_size=a.table_size)
        print(f"screen: {'CLEAN' if not viols else 'VIOLATIONS ' + str(viols)}")
    except ScreenCoverageError as e:
        print(f"screen: COVERAGE-MISS ({e})")

    # 'f2cb*' is the CORRECTED fold-to-c-bet (ALL_IN c-bets + multiway facers) — ADVISORY
    # only (not in the 6-max gated bands; it doesn't separate sound from loose here). The
    # production fold_to_cbet_pct is a ~14x undercount in 6-max, so it is never gated there.
    print(f"{'seed':>5} | {'VPIP':>6} {'PFR':>6} | {'nc-AI%':>7} {'raw-AI%':>8} | "
          f"{'f2cb*':>7} {'(n)':>5} {'AF':>6} | band")
    for seed in (int(s) for s in a.seeds.split(",")):
        prof = profile_spot_policy(policy, game, n_hands=a.n_hands, seed=seed)
        res = evaluate_bands(prof, bands)
        af = prof.totals.af()
        af_s = "inf" if af == float("inf") else f"{af:.2f}"
        verdict = "PASS" if res.passed else "FAIL(" + ",".join(
            m.name.replace("_pct", "").replace("_preflop", "") for m in res.failures
        ) + ")"
        print(f"{seed:>5} | {prof.vpip_pct:>6.1f} {prof.pfr_pct:>6.1f} | "
              f"{prof.noncommitted_all_in_preflop_pct:>7.2f} {prof.all_in_preflop_pct:>8.2f} | "
              f"{prof.fold_to_cbet_corrected_pct:>7.1f} {prof.facing_cbet_corrected_count:>5} {af_s:>6} | {verdict}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
