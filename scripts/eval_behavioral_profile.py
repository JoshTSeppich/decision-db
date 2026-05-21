"""Behavioral profile of trained adapter vs default-policy adapter.

Plays N hands (default 25k = 5 seeds x 5000 to mirror the post-retrain eval)
and tracks side-by-side behavioral stats. Tests the hypothesis: 'trained
plays a more balanced/aggressive strategy than default-policy, which is why
default-policy beats it head-up even though the trained bot may be playing
genuinely solid poker'.

Stats reported per side (trained vs default):
  - Aggression factor (AF) = (raises + bets) / calls
  - VPIP                   = % (seat, hand) instances where the player
                             voluntarily put money in preflop (call or
                             raise, excluding the forced blinds)
  - PFR                    = % (seat, hand) instances where the player
                             raised preflop
  - 3-bet %                = of (seat, hand) instances where the player
                             faced a prior preflop raise, % responded
                             with another raise
  - C-bet %                = of (seat, hand) instances where the player
                             was the last preflop raiser AND got to act
                             first on the flop, % bet
  - Fold-to-c-bet %        = of (seat, hand) instances where the player
                             faced a c-bet (PF aggressor's flop bet),
                             % folded

Plus an action-distribution breakdown per side per street (FOLD / CHECK /
CALL / BET / RAISE / ALL_IN%) since AF doesn't capture frequency vs
magnitude.

The hands played here use the SAME seed schedule as `eval_post_retrain.py`
so the populations match.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pokerbot.abstraction import AbstractionTables, ActionType
from pokerbot.runtime import RuntimeAdapter
from pokerbot.strategy_db import open_db
from pokerbot.training import SimpleNLHEGame

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_head_to_head import (  # type: ignore[import-not-found]
    _BOARD_LEN_TO_STREET,
    _clamp_to_legal,
    _state_to_request,
    _trained_seats_for_hand,
)

if TYPE_CHECKING:
    from pokerbot.training.nlhe_game import NLHEState


# ───────── data structures ─────────


@dataclass(slots=True)
class SideTotals:
    """Aggregated counters across all (seat, hand) instances of one side."""

    # Per-decision action-type counters (used for AF and action distribution)
    folds: int = 0
    checks: int = 0  # CHECK_CALL with to_call == 0
    calls: int = 0  # CHECK_CALL with to_call > 0
    bets: int = 0  # any postflop aggression with to_call == 0
    raises: int = 0  # any aggression with to_call > 0 (preflop and postflop)
    all_ins: int = 0  # subset of bets+raises, broken out for visibility
    # Per-street action counts (for the action-distribution breakdown)
    per_street_decisions: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    per_street_aggressive: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    per_street_call: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    per_street_fold: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # Per (seat, hand) instance counters
    hand_seats: int = 0  # total (seat, hand) instances = 3 * n_hands
    vpip_yes: int = 0
    pfr_yes: int = 0
    faced_pf_raise: int = 0
    three_bet_yes: int = 0
    cbet_opportunities: int = 0
    cbets: int = 0
    facing_cbet: int = 0
    fold_to_cbet: int = 0

    def af(self) -> float:
        denom = self.calls
        return (self.raises + self.bets) / denom if denom > 0 else float("inf")

    def vpip_rate(self) -> float:
        return self.vpip_yes / self.hand_seats if self.hand_seats > 0 else 0.0

    def pfr_rate(self) -> float:
        return self.pfr_yes / self.hand_seats if self.hand_seats > 0 else 0.0

    def three_bet_rate(self) -> float:
        return self.three_bet_yes / self.faced_pf_raise if self.faced_pf_raise > 0 else 0.0

    def cbet_rate(self) -> float:
        return self.cbets / self.cbet_opportunities if self.cbet_opportunities > 0 else 0.0

    def fold_to_cbet_rate(self) -> float:
        return self.fold_to_cbet / self.facing_cbet if self.facing_cbet > 0 else 0.0


@dataclass(slots=True)
class _SeatHandFlags:
    """Per (seat, hand_id) per-event flags, reset every hand."""

    voluntary_preflop: bool = False
    raised_preflop: bool = False
    faced_pf_raise_chance: bool = False  # got a preflop decision when a prior PF raise existed
    three_bet: bool = False
    cbet_opportunity: bool = False
    cbet_made: bool = False
    faced_cbet: bool = False
    folded_to_cbet: bool = False


# ───────── one hand ─────────


def _categorize_action(chosen: ActionType, to_call: int) -> str:
    """Classify one action into {fold, check, call, bet, raise}."""
    if chosen == ActionType.FOLD:
        return "fold"
    if chosen == ActionType.CHECK_CALL:
        return "call" if to_call > 0 else "check"
    return "raise" if to_call > 0 else "bet"


def _is_pf_aggression(entry_type: str) -> bool:
    """Did this preflop action represent voluntary aggression (raise/bet/all-in)?"""
    return entry_type in {"bet", "raise", "all-in"}


def _last_preflop_aggressor_seat(action_history: list[object]) -> int | None:
    """Seat of the last preflop entry that was a raise/bet/all-in, else None."""
    last: int | None = None
    for e in action_history:
        if e.street == 0 and _is_pf_aggression(e.type):  # type: ignore[attr-defined]
            last = e.seat  # type: ignore[attr-defined]
    return last


def _count_pf_raises_in(action_history: list[object]) -> int:
    """How many preflop raise/bet/all-in entries exist in history."""
    n = 0
    for e in action_history:
        if e.street == 0 and _is_pf_aggression(e.type):  # type: ignore[attr-defined]
            n += 1
    return n


def _play_one_hand_behavioral(
    hand_idx: int,
    game: SimpleNLHEGame,
    trained: RuntimeAdapter,
    default: RuntimeAdapter,
    rng: random.Random,
    trained_totals: SideTotals,
    default_totals: SideTotals,
    max_decisions: int = 400,
) -> NLHEState:
    """Play one hand, updating side totals. Returns terminal state."""
    trained_seats = _trained_seats_for_hand(hand_idx, game.table_size)
    state = game.new_initial_state(rng)
    decisions = 0
    # Per-seat per-hand flags
    seat_flags: dict[int, _SeatHandFlags] = {s: _SeatHandFlags() for s in range(game.table_size)}
    # Track whether any decision was made by each seat in this hand
    seat_acted: set[int] = set()

    while not game.is_terminal(state):
        actor = game.current_player(state)
        request = _state_to_request(game, state)
        adapter = trained if actor in trained_seats else default
        is_trained = actor in trained_seats
        totals = trained_totals if is_trained else default_totals

        response = adapter.decide(request)
        decisions += 1
        if decisions > max_decisions:
            raise RuntimeError(f"hand {hand_idx} stuck after {max_decisions} decisions")

        chosen = ActionType[response.abstract_action]
        legal_ints = game.legal_actions(state)
        applied = _clamp_to_legal(chosen, legal_ints)

        street = _BOARD_LEN_TO_STREET[len(request.board)]
        to_call = request.to_call
        action_kind = _categorize_action(applied, to_call)

        # Per-decision counters
        totals.per_street_decisions[street] += 1
        if action_kind == "fold":
            totals.folds += 1
            totals.per_street_fold[street] += 1
        elif action_kind == "check":
            totals.checks += 1
        elif action_kind == "call":
            totals.calls += 1
            totals.per_street_call[street] += 1
        elif action_kind == "bet":
            totals.bets += 1
            totals.per_street_aggressive[street] += 1
            if applied == ActionType.ALL_IN:
                totals.all_ins += 1
        elif action_kind == "raise":
            totals.raises += 1
            totals.per_street_aggressive[street] += 1
            if applied == ActionType.ALL_IN:
                totals.all_ins += 1

        seat_acted.add(actor)
        flags = seat_flags[actor]

        # Preflop event tracking
        if street == 0:
            # Voluntary preflop = action other than fold or free check
            # (a "check" preflop happens for the BB when nobody raised — that's
            # NOT voluntary; explicit raise/call IS voluntary)
            if action_kind in ("call", "raise", "bet"):
                flags.voluntary_preflop = True
            if action_kind == "raise":
                flags.raised_preflop = True
            # 3-bet detection: prior raise existed when this decision started
            prior_pf_raises = _count_pf_raises_in(request.action_history)
            if prior_pf_raises >= 1:
                flags.faced_pf_raise_chance = True
                if action_kind == "raise":
                    flags.three_bet = True

        # Flop c-bet tracking
        if street == 1:
            pf_aggr = _last_preflop_aggressor_seat(request.action_history)
            # Look at flop sub-history so far
            flop_actions = [e for e in request.action_history if e.street == 1]
            # C-bet opportunity: actor is PF aggressor AND no flop action yet
            if pf_aggr is not None and actor == pf_aggr and len(flop_actions) == 0:
                flags.cbet_opportunity = True
                if action_kind in ("bet", "raise"):
                    flags.cbet_made = True
            # Facing c-bet: exactly one prior flop action, by PF aggressor, that's a bet/raise
            elif (
                pf_aggr is not None
                and len(flop_actions) == 1
                and flop_actions[0].seat == pf_aggr
                and flop_actions[0].type in ("bet", "raise")
                and actor != pf_aggr
            ):
                # We're the second-to-act facing the c-bet
                flags.faced_cbet = True
                if action_kind == "fold":
                    flags.folded_to_cbet = True

        state = game.apply_action(state, int(applied), rng)

    # Fold per-seat flags into side totals
    for seat in range(game.table_size):
        is_trained = seat in trained_seats
        totals = trained_totals if is_trained else default_totals
        totals.hand_seats += 1
        f = seat_flags[seat]
        if f.voluntary_preflop:
            totals.vpip_yes += 1
        if f.raised_preflop:
            totals.pfr_yes += 1
        if f.faced_pf_raise_chance:
            totals.faced_pf_raise += 1
            if f.three_bet:
                totals.three_bet_yes += 1
        if f.cbet_opportunity:
            totals.cbet_opportunities += 1
            if f.cbet_made:
                totals.cbets += 1
        if f.faced_cbet:
            totals.facing_cbet += 1
            if f.folded_to_cbet:
                totals.fold_to_cbet += 1

    return state


# ───────── report ─────────


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _print_side_by_side(
    trained: SideTotals,
    default: SideTotals,
) -> None:
    print("\n══════════ behavioral profile (trained vs default) ══════════")
    print(f"{'metric':<24s}  {'trained':>12s}  {'default':>12s}  {'Δ':>8s}")
    print("  " + "─" * 60)

    def _row(name: str, t_val: float, d_val: float, fmt: str = "{:.2f}") -> None:
        if fmt == "%":
            t_s = _pct(t_val)
            d_s = _pct(d_val)
            delta = f"{(t_val - d_val) * 100:+.1f}"
        else:
            t_s = fmt.format(t_val)
            d_s = fmt.format(d_val)
            delta = f"{t_val - d_val:+.2f}"
        print(f"  {name:<22s}  {t_s:>12s}  {d_s:>12s}  {delta:>8s}")

    _row("Aggression Factor", trained.af(), default.af())
    _row("VPIP", trained.vpip_rate(), default.vpip_rate(), fmt="%")
    _row("PFR", trained.pfr_rate(), default.pfr_rate(), fmt="%")
    _row("3-bet %", trained.three_bet_rate(), default.three_bet_rate(), fmt="%")
    _row("c-bet %", trained.cbet_rate(), default.cbet_rate(), fmt="%")
    _row("fold-to-c-bet %", trained.fold_to_cbet_rate(), default.fold_to_cbet_rate(), fmt="%")


def _print_raw_counts(label: str, t: SideTotals) -> None:
    print(f"\n  raw counts ({label}):")
    print(
        f"    folds={t.folds:,}  checks={t.checks:,}  calls={t.calls:,}  "
        f"bets={t.bets:,}  raises={t.raises:,}  all-ins={t.all_ins:,}"
    )
    print(
        f"    hand-seats={t.hand_seats:,}  vpip={t.vpip_yes:,}  pfr={t.pfr_yes:,}  "
        f"faced-pf-raise={t.faced_pf_raise:,}  3-bets={t.three_bet_yes:,}"
    )
    print(
        f"    c-bet-opps={t.cbet_opportunities:,}  c-bets={t.cbets:,}  "
        f"facing-c-bet={t.facing_cbet:,}  fold-to-c-bet={t.fold_to_cbet:,}"
    )


def _print_per_street(label: str, t: SideTotals) -> None:
    print(f"\n  per-street action mix ({label}):")
    street_names = ("preflop", "flop", "turn", "river")
    print(f"    {'street':<8s}  {'decisions':>10s}  {'agg %':>7s}  {'call %':>7s}  {'fold %':>7s}")
    for s in range(4):
        d = t.per_street_decisions[s]
        if d == 0:
            continue
        agg = t.per_street_aggressive[s] / d
        call = t.per_street_call[s] / d
        fold = t.per_street_fold[s] / d
        print(
            f"    {street_names[s]:<8s}  "
            f"{d:>10,d}  {_pct(agg):>7s}  {_pct(call):>7s}  {_pct(fold):>7s}"
        )


def _print_verdict(trained: SideTotals, default: SideTotals) -> None:
    print("\n══════════ verdict on the hypothesis ══════════")
    t_af = trained.af()
    d_af = default.af()
    t_vpip = trained.vpip_rate()
    d_vpip = default.vpip_rate()
    t_pfr = trained.pfr_rate()
    d_pfr = default.pfr_rate()

    print(f"  trained AF = {t_af:.2f}, default AF = {d_af:.2f}")
    if t_af > 2.0 and d_af < 1.0:
        print(
            "  ✓ HYPOTHESIS SUPPORTED: trained is aggressive (AF > 2.0), default is "
            "passive (AF < 1.0). The H2H loss is consistent with 'aggressive bot "
            "vs passive nit-bot' dynamics; the benchmark is biased."
        )
    elif t_af > d_af * 1.5:
        print(
            "  ~ PARTIAL SUPPORT: trained is meaningfully more aggressive than "
            f"default ({t_af / d_af:.1f}x), but not at extreme levels. Bias plausible."
        )
    elif abs(t_af - d_af) < 0.3:
        print(
            "  ✗ HYPOTHESIS NOT SUPPORTED: AF profiles are similar. The H2H loss is "
            "not explained by aggression mismatch — trained is genuinely getting "
            "out-played by the heuristic, or there's a different bug."
        )
    else:
        print("  ~ AMBIGUOUS: AF gap is moderate. Look at PFR/3-bet and c-bet stats too.")

    print(
        f"\n  trained PFR = {_pct(t_pfr)}, default PFR = {_pct(d_pfr)}  "
        f"(typical strong NLHE 6-max: 18-24%)"
    )
    print(f"  trained VPIP = {_pct(t_vpip)}, default VPIP = {_pct(d_vpip)}  (typical: 22-30%)")


# ───────── main ─────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="strategy-pilot-v2.db")
    p.add_argument("--abstraction-dir", default="abstraction")
    p.add_argument("--n-hands", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--sb", type=int, default=5)
    p.add_argument("--bb", type=int, default=10)
    args = p.parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2

    print(f"Resolved DB path: {db_path}")
    print(f"DB file size:     {db_path.stat().st_size / 1e6:.1f} MB")
    print(f"Loading AbstractionTables from {args.abstraction_dir!r}…")
    abstraction = AbstractionTables(path=args.abstraction_dir)

    print(
        f"\nRunning {len(args.seeds)} seeds x {args.n_hands} hands "
        f"({len(args.seeds) * args.n_hands:,} hands total)\n"
    )

    trained_totals = SideTotals()
    default_totals = SideTotals()

    for seed in args.seeds:
        trained_db = open_db(f"sqlite:///{db_path}")
        default_db = open_db("sqlite:///:memory:")
        default_db.set_current_version(1)
        trained_adapter = RuntimeAdapter(db=trained_db, abstraction=abstraction, rng_seed=seed)
        default_adapter = RuntimeAdapter(
            db=default_db, abstraction=abstraction, rng_seed=seed ^ 0xDEADBEEF
        )
        game = SimpleNLHEGame(
            abstraction,
            blinds=(args.sb, args.bb),
            starting_stack=args.starting_stack,
            table_size=6,
        )
        rng = random.Random(seed)

        t0 = time.perf_counter()
        for i in range(args.n_hands):
            _play_one_hand_behavioral(
                i, game, trained_adapter, default_adapter, rng, trained_totals, default_totals
            )
        elapsed = time.perf_counter() - t0
        print(f"  seed {seed}: {args.n_hands:,} hands in {elapsed:.1f}s")
        trained_db.close()
        default_db.close()

    _print_side_by_side(trained_totals, default_totals)
    _print_raw_counts("trained", trained_totals)
    _print_raw_counts("default", default_totals)
    _print_per_street("trained", trained_totals)
    _print_per_street("default", default_totals)
    _print_verdict(trained_totals, default_totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
