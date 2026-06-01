"""STEP-1 localizer: per-(position, stack_bucket) breakdown of the v5 preflop leak.

Additive diagnostic — adds NO new aggregate scoring. It drives the SAME production
tally loop the gate uses (`_play_one_hand_behavioral` via an instrumented adapter), so
the headline VPIP / PFR / nc-AI numbers it prints are bit-for-bit the gate's numbers
(read from the production `SideTotals`). On the side, the instrumented adapter records,
keyed by (position, stack_bucket):

  * per-hand-seat ENTRY cell (position + effective-stack bucket at the seat's first
    preflop decision) with VPIP / PFR flags  → "is the looseness position-concentrated?"
  * per nc-shove DECISION cell (position + stack bucket at the moment of the
    discretionary shove)  → "does the discretionary shoving concentrate at short
    stacks / the blinds / certain positions?"

The VPIP/PFR per-cell flags are reconstructed from the sampled (chosen) action — the
same action the gate's adapter samples — so the per-cell rates sum to the authoritative
aggregate (validated by printing both). nc-shove cells are exact: the gate already keys
its nc-AI metric off `chosen == ALL_IN`, so this breakdown partitions that exact count.

Run:
  python -m zoom.eval.cell_breakdown --db sqlite:///abs/path.db \
      --abstraction-dir abstraction --table-size 6 --n-hands 400 --seeds 1,2
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from pokerbot.abstraction import AbstractionTables, ActionType
from pokerbot.abstraction.encoding import (
    effective_stack,
    position_from_seats,
    stack_bucket_from_eff_bb,
)
from pokerbot.strategy_db import open_db
from pokerbot.training import SimpleNLHEGame
from zoom.eval import make_db_spot_policy
from zoom.eval.profile import (
    _preflop_forced_jam,
    _SpotPolicyAdapter,
)

if TYPE_CHECKING:
    from pokerbot.runtime.schema import GameStateRequest
    from zoom.eval.profile import SpotPolicy

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from eval_behavioral_profile import (  # noqa: E402
    SideTotals,
    _categorize_action,
    _play_one_hand_behavioral,
)

# SB-relative position labels (position_from_seats returns 0=SB, 1=BB, then around).
_POS_LABELS_6 = ("SB", "BB", "UTG", "MP", "CO", "BTN")


def _pos_label(position: int, table_size: int) -> str:
    if table_size == 6 and 0 <= position < 6:
        return _POS_LABELS_6[position]
    return f"P{position}"


def _bb_of(eff_chips: int, bb: int) -> int:
    return eff_chips // max(bb, 1)


class _CellAdapter(_SpotPolicyAdapter):
    """`_SpotPolicyAdapter` + per-(position, stack_bucket) tally on the side.

    The base class already records `preflop_shoves` / `preflop_noncommitted_shoves`
    keyed by (hand, seat); we add cell attribution without altering that behavior, so
    the aggregate read by `_play_one_hand_behavioral`/`SideTotals` is unchanged.
    """

    def __init__(self, spot_policy: SpotPolicy, *, seed: int, bb: int, table_size: int) -> None:
        super().__init__(spot_policy, seed=seed)
        self._bb = bb
        self._table_size = table_size
        # Per (hand, seat): entry cell (position, stack_bucket at first PF decision)
        self.entry_cell: dict[tuple[int, int], tuple[int, int]] = {}
        # Per (hand, seat): chosen-based voluntary/raised preflop flags
        self.vpip_flag: dict[tuple[int, int], bool] = defaultdict(bool)
        self.pfr_flag: dict[tuple[int, int], bool] = defaultdict(bool)
        # nc-shove decision cells: (position, stack_bucket) -> count
        self.nc_shove_cells: dict[tuple[int, int], int] = defaultdict(int)
        # raw-shove decision cells
        self.raw_shove_cells: dict[tuple[int, int], int] = defaultdict(int)

    def decide(self, request: GameStateRequest) -> object:
        # Defer to base for the canonical sampling + bookkeeping (advances self._rng
        # exactly once), then read back the action it chose for per-cell attribution.
        resp = super().decide(request)
        chosen = ActionType[resp.abstract_action]  # type: ignore[attr-defined]

        seat = request.hero_seat
        position = position_from_seats(request.button_seat, seat, request.table_size)
        opps = [
            request.stacks[s]
            for s in range(request.table_size)
            if s != seat and request.stacks[s] > 0
        ]
        eff = effective_stack(request.stacks[seat], opps)
        bucket = stack_bucket_from_eff_bb(_bb_of(eff, self._bb))

        if len(request.board) == 0:  # preflop
            key = (self.current_hand, seat)
            if key not in self.entry_cell:
                self.entry_cell[key] = (position, bucket)
            kind = _categorize_action(chosen, request.to_call)
            if kind in ("call", "raise", "bet"):
                self.vpip_flag[key] = True
            if kind == "raise":
                self.pfr_flag[key] = True
            if chosen == ActionType.ALL_IN:
                self.raw_shove_cells[(position, bucket)] += 1
                stack = request.stacks[seat]
                if not _preflop_forced_jam(
                    request.pot_committed, request.to_call, stack, request.min_raise
                ):
                    self.nc_shove_cells[(position, bucket)] += 1
        return resp


def _run(
    policy: SpotPolicy,
    game: SimpleNLHEGame,
    *,
    n_hands: int,
    seed: int,
    bb: int,
    table_size: int,
) -> tuple[_CellAdapter, SideTotals]:
    adapter = _CellAdapter(policy, seed=seed, bb=bb, table_size=table_size)
    trained, default = SideTotals(), SideTotals()
    rng = random.Random(seed)
    for i in range(n_hands):
        adapter.current_hand = i
        _play_one_hand_behavioral(i, game, adapter, adapter, rng, trained, default)  # type: ignore[arg-type]
    merged = SideTotals()
    import dataclasses

    for f in dataclasses.fields(SideTotals):
        va, vb = getattr(trained, f.name), getattr(default, f.name)
        if isinstance(va, list):
            setattr(merged, f.name, [x + y for x, y in zip(va, vb, strict=True)])
        else:
            setattr(merged, f.name, va + vb)
    return adapter, merged


def _report(adapter: _CellAdapter, totals: SideTotals, *, table_size: int, seed: int) -> None:
    # Per-position aggregation over hand-seats (denominator = entry cells at that position).
    pos_seats: dict[int, int] = defaultdict(int)
    pos_vpip: dict[int, int] = defaultdict(int)
    pos_pfr: dict[int, int] = defaultdict(int)
    pos_entry_bucket: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for key, (pos, bucket) in adapter.entry_cell.items():
        pos_seats[pos] += 1
        pos_entry_bucket[pos][bucket] += 1
        if adapter.vpip_flag.get(key):
            pos_vpip[pos] += 1
        if adapter.pfr_flag.get(key):
            pos_pfr[pos] += 1

    # nc-shoves attributed to position (summed over buckets at the shove decision)
    pos_nc: dict[int, int] = defaultdict(int)
    pos_raw: dict[int, int] = defaultdict(int)
    for (pos, _b), c in adapter.nc_shove_cells.items():
        pos_nc[pos] += c
    for (pos, _b), c in adapter.raw_shove_cells.items():
        pos_raw[pos] += c

    total_seats = sum(pos_seats.values())
    print(f"\n── seed {seed}: per-POSITION breakdown (denom = hand-seats dealt that position) ──")
    print(f"{'pos':>4} | {'seats':>6} | {'VPIP%':>6} | {'PFR%':>6} | {'nc-AI%':>7} | {'raw-AI%':>8}")
    for pos in sorted(pos_seats):
        n = pos_seats[pos]
        vp = pos_vpip[pos] / n * 100 if n else 0.0
        pf = pos_pfr[pos] / n * 100 if n else 0.0
        nc = pos_nc[pos] / n * 100 if n else 0.0
        rw = pos_raw[pos] / n * 100 if n else 0.0
        print(
            f"{_pos_label(pos, table_size):>4} | {n:>6} | {vp:>6.1f} | {pf:>6.1f} | "
            f"{nc:>7.2f} | {rw:>8.2f}"
        )

    # nc-shove concentration by (position, stack_bucket)
    print(f"\n── seed {seed}: nc-shove decisions by (position, stack_bucket@shove) ──")
    print("  (stack bucket boundaries bb: <10,<20,<30,<50,<75,<100,<150,<200,<300,>=300)")
    total_nc = sum(adapter.nc_shove_cells.values())
    if total_nc == 0:
        print("  (none)")
    else:
        print(f"{'pos':>4} | {'bucket':>6} | {'count':>6} | {'% of nc-shoves':>14}")
        for (pos, b), c in sorted(
            adapter.nc_shove_cells.items(), key=lambda kv: -kv[1]
        ):
            print(
                f"{_pos_label(pos, table_size):>4} | {b:>6} | {c:>6} | "
                f"{c / total_nc * 100:>13.1f}%"
            )
    print(
        f"  totals: nc-shoves={total_nc}  raw-shoves={sum(adapter.raw_shove_cells.values())}  "
        f"hand-seats={total_seats}"
    )
    # Cross-check: aggregate from SideTotals (authoritative gate numbers)
    print(
        f"  GATE aggregate (SideTotals): VPIP={totals.vpip_rate()*100:.1f}  "
        f"PFR={totals.pfr_rate()*100:.1f}  hand_seats={totals.hand_seats}  "
        f"(reconstructed VPIP from cells={sum(pos_vpip.values())/total_seats*100:.1f})"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="STEP-1 per-cell localizer for the v5 preflop leak")
    p.add_argument("--db", required=True)
    p.add_argument("--abstraction-dir", type=Path, default=Path("abstraction"))
    p.add_argument("--table-size", type=int, default=6)
    p.add_argument("--bb", type=int, default=10)
    p.add_argument("--starting-stack", type=int, default=1000)
    p.add_argument("--n-hands", type=int, default=400)
    p.add_argument("--seeds", default="1,2")
    a = p.parse_args(argv)

    abstraction = AbstractionTables(path=a.abstraction_dir)
    db = open_db(a.db)
    game = SimpleNLHEGame(
        abstraction, blinds=(a.bb // 2, a.bb), starting_stack=a.starting_stack,
        table_size=a.table_size,
    )
    policy = make_db_spot_policy(db, abstraction, bb=a.bb, table_size=a.table_size)
    print(f"DB={a.db}  version={db.current_version()}  table_size={a.table_size}  n_hands={a.n_hands}")
    for seed in (int(s) for s in a.seeds.split(",")):
        adapter, totals = _run(
            policy, game, n_hands=a.n_hands, seed=seed, bb=a.bb, table_size=a.table_size
        )
        _report(adapter, totals, table_size=a.table_size, seed=seed)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
