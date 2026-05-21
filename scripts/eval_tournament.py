"""Cairn 5 evaluation harness: run N tournaments x (tournament-mode | cash-mode).

For each of N tournaments (deterministic seed per index), run the same
opponent field twice: once with the hero using TournamentAdapter
(game_type='tournament'), once with the hero using RuntimeAdapter
(game_type='cash'). Same RNG seed within each pair so the deal sequence
matches as closely as possible up to divergent decisions.

Outputs a JSON file with per-tournament results + summary stats. Prints a
human-readable summary report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pokerbot.abstraction import AbstractionTables
from pokerbot.opponent import (
    ArchetypeClassifier,
    ArchetypeOpponentModel,
    OpponentStatsTracker,
)
from pokerbot.runtime import RuntimeAdapter
from pokerbot.strategy_db import open_db
from pokerbot.strategy_db.dual import DualStrategyDB
from pokerbot.tournament.adapter import TournamentAdapter
from pokerbot.tournament.opponent_field import ParameterizedBot, generate_field
from pokerbot.tournament.simulator import (
    PAID_POSITIONS,
    PRIZE_POOL_DEFAULT,
    TournamentResult,
    TournamentSimulator,
)

if TYPE_CHECKING:
    from pokerbot.runtime.schema import GameStateRequest
    from pokerbot.tournament.state import TournamentState

BUY_IN: float = 100.0


def _run_one_tournament(
    *,
    mode: str,  # "tournament" or "cash"
    seed: int,
    abstraction: AbstractionTables,
    db_path: str,
    db_path_9max: str | None = None,
) -> TournamentResult:
    rng = random.Random(seed)
    field = generate_field(rng_seed=seed)

    # Bot adapters (in-memory empty DB → ParameterizedBot drives every decision)
    bot_db = open_db("sqlite:///:memory:")
    bot_db.set_current_version(1)
    bot_adapters: dict[int, RuntimeAdapter] = {
        pid: RuntimeAdapter(
            db=bot_db,
            abstraction=abstraction,
            opponent_model=ParameterizedBot(cfg),
            rng_seed=seed ^ (pid * 0x9E3779B9),
        )
        for pid, cfg in enumerate(field)
    }

    # Hero: trained policy DB + a fresh ArchetypeOpponentModel.
    # Tracker is per-tournament so cross-tournament data doesn't leak.
    # If --db-9max is provided, dual-route by table_size (Cairn 6).
    hero_db_primary = open_db(db_path)
    if db_path.endswith(":memory:"):
        hero_db_primary.set_current_version(1)
    if db_path_9max is not None:
        hero_db_secondary = open_db(db_path_9max)
        if db_path_9max.endswith(":memory:"):
            hero_db_secondary.set_current_version(1)
        hero_db = DualStrategyDB(hero_db_primary, hero_db_secondary)
    else:
        hero_db = hero_db_primary  # type: ignore[assignment]
    tracker = OpponentStatsTracker()
    archetype_model = ArchetypeOpponentModel(tracker, ArchetypeClassifier())
    hero_runtime = RuntimeAdapter(
        db=hero_db,
        abstraction=abstraction,
        opponent_model=archetype_model,
        rng_seed=seed,
    )

    if mode == "tournament":
        hero_tournament = TournamentAdapter(hero_runtime, rng_seed=seed)

        def hero_decide(
            request: GameStateRequest,
            ts: TournamentState | None,
            opp_ids: tuple[str, ...],
        ) -> str:
            archetype_model.set_active_opponents(opp_ids)
            return hero_tournament.decide(request, ts).abstract_action
    else:

        def hero_decide(
            request: GameStateRequest,
            ts: TournamentState | None,  # noqa: ARG001
            opp_ids: tuple[str, ...],
        ) -> str:
            archetype_model.set_active_opponents(opp_ids)
            return hero_runtime.decide(request).abstract_action

    def bot_decide(pid: int, request: GameStateRequest) -> str:
        return bot_adapters[pid].decide(request).abstract_action

    sim = TournamentSimulator(abstraction=abstraction)
    result = sim.run_tournament(
        hero_pid=0,
        bot_decide_fn=bot_decide,
        hero_decide_fn=hero_decide,
        rng=rng,
        opponent_tracker=tracker,
    )
    bot_db.close()
    hero_db.close()
    return result


def _summarize(results: list[TournamentResult]) -> dict[str, Any]:
    n = len(results)
    finishes = [r.finish_position for r in results]
    prizes = [r.prize for r in results]
    busts = [r.busted_at_hand for r in results if r.busted_at_hand is not None]
    elapsed = [r.elapsed_seconds for r in results]

    itm = sum(1 for f in finishes if f <= PAID_POSITIONS)
    final_table = sum(1 for f in finishes if f <= 9)
    won = sum(1 for f in finishes if f == 1)
    early_bust = sum(1 for f in finishes if f >= 61)  # bottom third
    mid_field = sum(1 for f in finishes if 31 <= f <= 60)
    deep_run = sum(1 for f in finishes if 13 <= f <= 30)  # deep but not ITM

    avg_prize = sum(prizes) / n
    avg_finish = sum(finishes) / n
    sorted_finishes = sorted(finishes)
    median_finish = sorted_finishes[n // 2] if n % 2 == 1 else (
        sorted_finishes[n // 2 - 1] + sorted_finishes[n // 2]
    ) / 2.0

    roi = (avg_prize - BUY_IN) / BUY_IN
    # Per-tournament ROI samples for CI
    roi_samples = [(p - BUY_IN) / BUY_IN for p in prizes]
    mean_roi = sum(roi_samples) / n
    if n > 1:
        var = sum((x - mean_roi) ** 2 for x in roi_samples) / (n - 1)
        stderr = math.sqrt(var / n)
    else:
        stderr = 0.0

    return {
        "n_tournaments": n,
        "itm_count": itm,
        "itm_pct": itm / n,
        "final_table_count": final_table,
        "final_table_pct": final_table / n,
        "won_count": won,
        "won_pct": won / n,
        "early_bust_count": early_bust,
        "early_bust_pct": early_bust / n,
        "mid_field_count": mid_field,
        "deep_run_count": deep_run,
        "avg_prize": avg_prize,
        "avg_finish": avg_finish,
        "median_finish": median_finish,
        "roi": roi,
        "roi_stderr": stderr,
        "roi_ci95_low": roi - 1.96 * stderr,
        "roi_ci95_high": roi + 1.96 * stderr,
        "avg_hero_hands_played": sum(r.hands_played for r in results) / n,
        "avg_busted_at_hand": (sum(busts) / len(busts)) if busts else None,
        "avg_elapsed_sec_per_tourney": sum(elapsed) / n,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-tournaments", type=int, default=100)
    p.add_argument("--db", default="strategy-pilot-v2.db")
    p.add_argument(
        "--db-9max",
        default=None,
        help="optional 9-max DB; if provided, dual-routes by table_size",
    )
    p.add_argument("--abstraction-dir", default="abstraction")
    p.add_argument("--output", required=True)
    p.add_argument("--seed-base", type=int, default=2026)
    args = p.parse_args(argv)

    def _to_url(arg: str) -> str:
        if arg.startswith("sqlite:"):
            return arg
        p = Path(arg).resolve()
        if not p.exists():
            print(f"ERROR: DB not found at {p}", file=sys.stderr)
            raise SystemExit(2)
        return f"sqlite:///{p}"

    db_url = _to_url(args.db)
    db_url_9max = _to_url(args.db_9max) if args.db_9max else None

    print(f"Loading AbstractionTables from {args.abstraction_dir!r}…")
    abstraction = AbstractionTables(path=args.abstraction_dir)
    print(f"Primary DB: {db_url}")
    if db_url_9max:
        print(f"9-max DB:   {db_url_9max}  (dual-routing by table_size)")
    print(f"Running {args.n_tournaments} tournaments x 2 modes = {args.n_tournaments * 2} runs")

    t0 = time.perf_counter()
    tournament_results: list[TournamentResult] = []
    cash_results: list[TournamentResult] = []
    for i in range(args.n_tournaments):
        seed = args.seed_base + i
        # Same field/seed for both modes
        t1 = time.perf_counter()
        tr = _run_one_tournament(
            mode="tournament", seed=seed, abstraction=abstraction,
            db_path=db_url, db_path_9max=db_url_9max,
        )
        cr = _run_one_tournament(
            mode="cash", seed=seed, abstraction=abstraction,
            db_path=db_url, db_path_9max=db_url_9max,
        )
        tournament_results.append(tr)
        cash_results.append(cr)
        print(
            f"  [{i + 1}/{args.n_tournaments}] seed={seed}  "
            f"tournament: finish={tr.finish_position:>3d} prize=${tr.prize:>7.2f}  |  "
            f"cash: finish={cr.finish_position:>3d} prize=${cr.prize:>7.2f}  "
            f"(this pair: {time.perf_counter() - t1:.1f}s)"
        )

    total_elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    summary_t = _summarize(tournament_results)
    summary_c = _summarize(cash_results)

    # Comparison: paired ROI delta
    roi_deltas = [
        (tr.prize - cr.prize) / BUY_IN
        for tr, cr in zip(tournament_results, cash_results, strict=True)
    ]
    mean_delta = sum(roi_deltas) / len(roi_deltas)
    if len(roi_deltas) > 1:
        var = sum((x - mean_delta) ** 2 for x in roi_deltas) / (len(roi_deltas) - 1)
        delta_stderr = math.sqrt(var / len(roi_deltas))
    else:
        delta_stderr = 0.0

    output = {
        "config": {
            "n_tournaments": args.n_tournaments,
            "buy_in": BUY_IN,
            "prize_pool": PRIZE_POOL_DEFAULT,
            "paid_positions": PAID_POSITIONS,
            "seed_base": args.seed_base,
        },
        "tournament_mode": {
            "summary": summary_t,
            "results": [asdict(r) for r in tournament_results],
        },
        "cash_mode": {
            "summary": summary_c,
            "results": [asdict(r) for r in cash_results],
        },
        "comparison": {
            "mean_roi_delta": mean_delta,
            "roi_delta_stderr": delta_stderr,
            "roi_delta_ci95_low": mean_delta - 1.96 * delta_stderr,
            "roi_delta_ci95_high": mean_delta + 1.96 * delta_stderr,
            "tournament_minus_cash": True,
        },
        "elapsed_seconds": total_elapsed,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    # Print summary
    print("\n══════════ TOURNAMENT-MODE summary ══════════")
    _print_summary(summary_t)
    print("\n══════════ CASH-MODE summary ══════════")
    _print_summary(summary_c)
    print("\n══════════ COMPARISON ══════════")
    print(f"  tournament-mode ROI:   {summary_t['roi']:+.4f}  (95% CI [{summary_t['roi_ci95_low']:+.4f}, {summary_t['roi_ci95_high']:+.4f}])")
    print(f"  cash-mode ROI:         {summary_c['roi']:+.4f}  (95% CI [{summary_c['roi_ci95_low']:+.4f}, {summary_c['roi_ci95_high']:+.4f}])")
    print(f"  delta (tourney - cash): {mean_delta:+.4f}  (95% CI [{mean_delta - 1.96 * delta_stderr:+.4f}, {mean_delta + 1.96 * delta_stderr:+.4f}])")
    significant = (mean_delta - 1.96 * delta_stderr > 0) or (mean_delta + 1.96 * delta_stderr < 0)
    print(f"  significant (CI excludes 0): {significant}")
    print(f"\nResults written to {out_path}")
    return 0


def _print_summary(s: dict[str, Any]) -> None:
    print(f"  n_tournaments:          {s['n_tournaments']}")
    print(f"  ITM%:                   {s['itm_pct']:.1%}  ({s['itm_count']}/{s['n_tournaments']})")
    print(f"  Final-table%:           {s['final_table_pct']:.1%}  ({s['final_table_count']})")
    print(f"  Won%:                   {s['won_pct']:.1%}  ({s['won_count']})")
    print(f"  Early-bust% (≥61st):    {s['early_bust_pct']:.1%}  ({s['early_bust_count']})")
    print(f"  Mid-field (31-60):      {s['mid_field_count']}")
    print(f"  Deep run (13-30):       {s['deep_run_count']}")
    print(f"  Avg prize:              ${s['avg_prize']:.2f}")
    print(f"  Avg finish:             {s['avg_finish']:.1f}")
    print(f"  Median finish:          {s['median_finish']}")
    print(f"  ROI:                    {s['roi']:+.4f}")
    print(f"  Avg hero hands played:  {s['avg_hero_hands_played']:.1f}")
    print(f"  Avg busted at hand:     {s['avg_busted_at_hand']}")
    print(f"  Avg sec per tourney:    {s['avg_elapsed_sec_per_tourney']:.1f}")


if __name__ == "__main__":
    raise SystemExit(main())
