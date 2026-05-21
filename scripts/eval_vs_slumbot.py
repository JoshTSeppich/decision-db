"""Benchmark trained adapter against Slumbot (slumbot.com) HU NLHE bot.

Slumbot is a public 2018-era CFR-trained heads-up NLHE bot that exposes a
free HTTP API. It's the standard public benchmark for serious poker work.

Caveat (documented in docstring + printed at startup): our bot is trained
for 6-max, not heads-up. The mapping below puts the HU SB on our 6-max
seat 0 (= SB position) and HU BB on seat 1 (= BB position), with seats
2-5 inactive (stack=0). The bot will read this as a 6-max situation where
everyone but us+opp folded preflop — strategically wrong for HU (HU
ranges are MUCH wider) but the closest legal mapping in our schema.
Expect the bot to play too tight in HU.

API protocol (per https://slumbot.com/ docs + Gongsta/Poker-AI reference):
  POST https://slumbot.com/api/new_hand   {"token":?}
  POST https://slumbot.com/api/act        {"token":..., "incr":"<action>"}

Action format: 'k' check, 'c' call, 'b<N>' bet/raise to street-total N,
'f' fold. '/' separates streets in the cumulative action string.

Stakes: SB=50, BB=100, starting stack 20,000 (200bb).

Output:
  - per-hand chip result, running mbb/hand and 95% CI
  - per-street fallback breakdown
  - count of API rejections (would indicate translation bugs)

Don't run more than 200 hands until a smoke pass has been eyeballed.

═══════════════════════════════════════════════════════════════════════════
Baseline result — 200 hands, 2026-05-19 (post-bet-floor + real-stack fix):
═══════════════════════════════════════════════════════════════════════════

    mbb/hand:  -937.75    95% CI [-1592.59, -282.91]    stderr 334
    elapsed:    163.0s   (1.23 hands/sec)
    API rejections: 0    fallback: 100% default_policy across all streets

This number measures `default_policy + adapter infrastructure vs Slumbot`,
NOT the trained policy. Slumbot's 200 BB starting stacks place every infoset
in stack_bucket 8 (200-300 BB) where v2 has zero rows; both exact and NN
lookups miss, so every decision falls through to `default_policy_action`
(Chen-formula heuristic).

The trained 2.23M-row DB is irrelevant here. Treat -937 mbb/hand as a
baseline for the Chen heuristic against a competent CFR-trained bot in HU
NLHE. Future improvements to the default policy should be benchmarked
against this number.

To actually benchmark the trained policy, we'd need either:
  (a) Slumbot configured with 100 BB starting stacks. CONFIRMED unavailable:
      slumbot.com/sample_api.py hardcodes SMALL_BLIND=50, BIG_BLIND=100,
      STACK_SIZE=20000 as module-level constants; the API exposes no
      configuration parameters in either /api/new_hand or /api/act.
  (b) A different opponent we control — see `scripts/eval_archetypes.py`,
      which runs 6-max 100 BB head-to-head against hand-coded
      Nit/Maniac/Station/Default archetypes within our own simulator.

Verdict: Slumbot is permanently stuck at 200 BB. Use the archetype panel
for trained-policy benchmarking.
═══════════════════════════════════════════════════════════════════════════

Usage:
    python scripts/eval_vs_slumbot.py --n-hands 10 --debug
    python scripts/eval_vs_slumbot.py --n-hands 200 --db strategy-pilot-v2.db
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pokerbot.abstraction import AbstractionTables
from pokerbot.runtime import (
    ActionHistoryEntry,
    BlindsSchema,
    GameStateRequest,
    RuntimeAdapter,
)
from pokerbot.strategy_db import open_db

if TYPE_CHECKING:
    from pokerbot.runtime.schema import FallbackUsed

# ───────── constants ─────────

SLUMBOT_HOST = "https://slumbot.com"
SLUMBOT_SB = 50
SLUMBOT_BB = 100
SLUMBOT_STACK = 20_000
SLUMBOT_NUM_STREETS = 4

# 6-max mapping (documented assumption — see module docstring)
HERO_SB_SEAT = 0
HERO_BB_SEAT = 1
TABLE_SIZE = 6
BUTTON_SEAT = TABLE_SIZE - 1  # button at last seat → SB=(BTN+1)%6=0

# NOTE: Slumbot starts at 200 BB; our v2 DB has ZERO rows for stack_buckets
# 7-9 (covers 150-300+ BB) and ZERO rows at bucket 6 for positions 0/1.
# An earlier version of this script tried to "scale down" the visible stack
# to push the bot into bucket 5, but that broke legality at deep all-in spots
# (visible_stack going to 0 while real chips remained → API rejections).
# We now pass REAL chip counts and accept that most decisions will fall
# through to default_policy (no DB rows for 200 BB scenarios). The bot's
# legality gate handles every depth correctly.


# ───────── HTTP client ─────────


def _post_json(path: str, body: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    """POST a JSON body to Slumbot. Returns parsed JSON response.

    Raises RuntimeError with the server's error_msg on non-200 or error_msg
    field present.
    """
    url = f"{SLUMBOT_HOST}{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"{path} returned HTTP {resp.status}")
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
            return payload
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
        except Exception:
            err_body = {"status": e.code, "reason": e.reason}
        raise RuntimeError(f"{path} HTTP {e.code}: {err_body}") from e


def slumbot_new_hand(token: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if token:
        body["token"] = token
    return _post_json("/api/new_hand", body)


def slumbot_act(token: str, incr: str) -> dict[str, Any]:
    return _post_json("/api/act", {"token": token, "incr": incr})


# ───────── action-string parser ─────────


@dataclass(slots=True)
class ParsedState:
    """Snapshot of the betting state after parsing the action string.

    All chip counts are in Slumbot chips (1 BB = 100). Positions are
    Slumbot's: 0 = BB (postflop first), 1 = SB/button (preflop first).
    """

    history: list[ActionHistoryEntry]
    street: int  # 0..3
    committed_this_street: tuple[int, int]  # indexed by Slumbot pos: [BB, SB]
    total_committed: tuple[int, int]  # cumulative across all streets
    last_bet_to_street: int  # the bet-to target on the current street
    last_bet_size: int  # increment of the most recent raise
    pos_to_act: int  # 0 or 1, or -1 if hand is over
    is_terminal: bool


def parse_slumbot_action(action_str: str, sb_seat: int, bb_seat: int) -> ParsedState:
    """Walk a Slumbot action string and produce our action_history + state.

    Mirrors the reference parser in Gongsta/Poker-AI but streams per-step
    events into our `ActionHistoryEntry` format. Each `action` here is the
    cumulative string for the hand so far (preflop + flop + turn + river
    with '/' separators).
    """
    # State variables
    street = 0
    pos_to_act = 1  # SB acts first preflop
    # committed[pos] = chips committed THIS STREET, indexed by Slumbot pos
    committed = [SLUMBOT_BB, SLUMBOT_SB]  # BB has 100, SB has 50 from blinds
    total = [SLUMBOT_BB, SLUMBOT_SB]
    last_bet_to = SLUMBOT_BB  # the "bet level" on this street
    last_bet_size = SLUMBOT_BB - SLUMBOT_SB  # = 50; size of the BB "blind raise"
    check_or_call_ends_street = False
    is_terminal = False
    history: list[ActionHistoryEntry] = []

    def actor_seat(p: int) -> int:
        return sb_seat if p == 1 else bb_seat

    def advance_street() -> None:
        nonlocal street, pos_to_act, last_bet_to, last_bet_size, check_or_call_ends_street
        nonlocal committed
        if street == SLUMBOT_NUM_STREETS - 1:
            pos_to_act_local = -1
        else:
            pos_to_act_local = 0  # BB acts first postflop
            street += 1
            committed = [0, 0]
            last_bet_to = 0
            last_bet_size = 0
            check_or_call_ends_street = False
        pos_to_act = pos_to_act_local

    i = 0
    while i < len(action_str):
        c = action_str[i]
        i += 1
        if c == "/":
            # Slumbot inserts these between streets; advance_street already
            # handled the transition. Skip stray slashes.
            continue
        if c == "k":
            history.append(
                ActionHistoryEntry(
                    seat=actor_seat(pos_to_act),
                    street=street,
                    type="check",
                    amount=0,
                )
            )
            if check_or_call_ends_street:
                # Both players have checked — street ends
                if street == SLUMBOT_NUM_STREETS - 1:
                    pos_to_act = -1
                    is_terminal = True
                else:
                    advance_street()
                # consume optional '/'
                if i < len(action_str) and action_str[i] == "/":
                    i += 1
            else:
                pos_to_act = (pos_to_act + 1) % 2
                check_or_call_ends_street = True
        elif c == "c":
            delta = last_bet_to - committed[pos_to_act]
            committed[pos_to_act] += delta
            total[pos_to_act] += delta
            history.append(
                ActionHistoryEntry(
                    seat=actor_seat(pos_to_act),
                    street=street,
                    type="call",
                    amount=delta,
                )
            )
            # All-in call: the rest of the action string is empty/slashes
            if total[pos_to_act] == SLUMBOT_STACK:
                pos_to_act = -1
                is_terminal = True
                break
            if check_or_call_ends_street:
                if street == SLUMBOT_NUM_STREETS - 1:
                    pos_to_act = -1
                    is_terminal = True
                else:
                    advance_street()
                if i < len(action_str) and action_str[i] == "/":
                    i += 1
            else:
                # First action of the street was a call (e.g., SB limping
                # preflop); other player still has to act
                pos_to_act = (pos_to_act + 1) % 2
                check_or_call_ends_street = True
        elif c == "b":
            j = i
            while i < len(action_str) and action_str[i].isdigit():
                i += 1
            new_bet_to = int(action_str[j:i])
            delta = new_bet_to - committed[pos_to_act]
            committed[pos_to_act] = new_bet_to
            total[pos_to_act] += delta
            # Type: "bet" if no prior aggression on this street, else "raise".
            # Preflop's BB is a prior "bet level" of 100 (last_bet_size=50),
            # so the first 'b' preflop is a "raise".
            type_str = "raise" if last_bet_size > 0 else "bet"
            history.append(
                ActionHistoryEntry(
                    seat=actor_seat(pos_to_act),
                    street=street,
                    type=cast("Any", type_str),
                    amount=delta,
                )
            )
            last_bet_size = new_bet_to - last_bet_to
            last_bet_to = new_bet_to
            pos_to_act = (pos_to_act + 1) % 2
            check_or_call_ends_street = True
        elif c == "f":
            history.append(
                ActionHistoryEntry(
                    seat=actor_seat(pos_to_act),
                    street=street,
                    type="fold",
                    amount=0,
                )
            )
            pos_to_act = -1
            is_terminal = True
            break
        else:
            raise ValueError(f"unexpected char {c!r} in action {action_str!r}")

    return ParsedState(
        history=history,
        street=street,
        committed_this_street=(committed[0], committed[1]),
        total_committed=(total[0], total[1]),
        last_bet_to_street=last_bet_to,
        last_bet_size=last_bet_size,
        pos_to_act=pos_to_act,
        is_terminal=is_terminal,
    )


# ───────── state translator ─────────


def build_request(
    parsed: ParsedState,
    client_pos: int,
    hero_hole: list[str],
    board: list[str],
) -> GameStateRequest:
    """Convert Slumbot state into a GameStateRequest, with HU→6max mapping.

    Mapping (see module docstring):
      Slumbot pos 0 (BB) → seat 1 (BB position in 6-max)
      Slumbot pos 1 (SB) → seat 0 (SB position in 6-max)
      Seats 2-5         → stack=0 (inactive)
      button_seat       → 5
    """
    sb_seat = HERO_SB_SEAT  # 0
    bb_seat = HERO_BB_SEAT  # 1
    hero_seat = sb_seat if client_pos == 1 else bb_seat

    stacks = [0] * TABLE_SIZE
    # Pass REAL remaining chips (no visible-stack shim). With Slumbot's
    # 200 BB starting stacks most decisions will fall through to
    # default_policy (the v2 DB has no rows above stack_bucket=6) — that's
    # accepted as the tradeoff for correct legality at any stack depth.
    stacks[sb_seat] = max(SLUMBOT_STACK - parsed.total_committed[1], 0)
    stacks[bb_seat] = max(SLUMBOT_STACK - parsed.total_committed[0], 0)

    current_bets = [0] * TABLE_SIZE
    current_bets[sb_seat] = parsed.committed_this_street[1]
    current_bets[bb_seat] = parsed.committed_this_street[0]

    hero_stack = stacks[hero_seat]
    hero_committed_street = current_bets[hero_seat]
    to_call = max(parsed.last_bet_to_street - hero_committed_street, 0)
    # min_raise (chips hero must ADD over current commitment to legally raise):
    #   min_bet_to = last_bet_to + max(last_bet_size, BB)
    #   min_raise  = min_bet_to - hero_committed_street
    #             = to_call + max(last_bet_size, BB)
    # Capped at hero's remaining chips (cannot raise more than stack-to_call).
    raise_increment = max(parsed.last_bet_size, SLUMBOT_BB)
    min_raise = to_call + raise_increment
    if min_raise > hero_stack:
        # Only all-in fits — clamp to whatever's left
        min_raise = max(hero_stack, 0)
    # Bet-to TARGET that fills the rest of hero's stack
    max_raise = hero_committed_street + hero_stack

    pot_committed = (
        parsed.total_committed[0] + parsed.total_committed[1]
    )  # total chips in pot (incl. current street commitments)

    return GameStateRequest(
        schema_version=1,
        game_type="cash",
        table_size=cast("Any", TABLE_SIZE),
        blinds=BlindsSchema(sb=SLUMBOT_SB, bb=SLUMBOT_BB),
        ante=0,
        hero_seat=hero_seat,
        button_seat=BUTTON_SEAT,
        hero_hole=hero_hole,
        board=board,
        stacks=stacks,
        current_bets=current_bets,
        pot_committed=pot_committed,
        to_call=to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        action_history=parsed.history,
    )


# ───────── action translator ─────────


def adapter_to_slumbot(
    response_amount: int,
    response_action: str,
    hero_committed_street: int,
) -> str:
    """Translate our adapter's ActionResponse to Slumbot's 'incr' format.

    Our `amount` is the chips ADDED by this action. Slumbot's bet-to value
    is the actor's new total this-street commitment, i.e., committed_now + amount.
    """
    if response_action == "fold":
        return "f"
    if response_action == "check":
        return "k"
    if response_action == "call":
        return "c"
    # bet or raise → 'b<bet-to>'
    bet_to = hero_committed_street + response_amount
    return f"b{bet_to}"


# ───────── per-hand driver ─────────


@dataclass(slots=True)
class MatchStats:
    hands_played: int = 0
    winnings: list[int] = field(default_factory=list)
    fallback_by_street: dict[int, dict[FallbackUsed, int]] = field(
        default_factory=lambda: {
            s: {"exact": 0, "nearest_neighbor": 0, "default_policy": 0} for s in range(4)
        }
    )
    api_rejections: int = 0
    api_rejection_logs: list[str] = field(default_factory=list)
    decisions: int = 0


def play_one_hand(
    token: str | None,
    adapter: RuntimeAdapter,
    stats: MatchStats,
    debug: bool = False,
) -> tuple[str, int]:
    """Play one Slumbot hand. Returns (new_token, winnings_in_chips)."""
    r = slumbot_new_hand(token)
    if "error_msg" in r:
        raise RuntimeError(f"new_hand error: {r['error_msg']}")
    new_token = r.get("token") or token
    if new_token is None:
        raise RuntimeError("Slumbot didn't return a token in /new_hand response")
    token = new_token
    while True:
        action_str: str = r.get("action", "") or ""
        client_pos: int = r["client_pos"]
        hole_cards: list[str] = r["hole_cards"]
        board: list[str] = r.get("board", []) or []
        winnings = r.get("winnings")
        if winnings is not None:
            if debug:
                print(
                    f"    hand end: client_pos={client_pos} action={action_str!r} "
                    f"hole={hole_cards} board={board} winnings={winnings}"
                )
            return token, int(winnings)

        # It's our turn — parse and translate.
        parsed = parse_slumbot_action(action_str, sb_seat=HERO_SB_SEAT, bb_seat=HERO_BB_SEAT)
        if parsed.pos_to_act == -1:
            raise RuntimeError(
                f"parser says hand terminal but Slumbot didn't return winnings; "
                f"action={action_str!r}"
            )
        request = build_request(parsed, client_pos, hole_cards, board)

        response = adapter.decide(request)
        stats.fallback_by_street[parsed.street][response.fallback_used] += 1
        stats.decisions += 1

        hero_seat = request.hero_seat
        hero_committed_street = request.current_bets[hero_seat]
        incr = adapter_to_slumbot(response.amount, response.action, hero_committed_street)
        if debug:
            print(
                f"    state: pos={client_pos} street={parsed.street} "
                f"to_call={request.to_call} pot={request.pot_committed} "
                f"action_str={action_str!r}"
            )
            print(
                f"      bot: abstract={response.abstract_action} amount={response.amount} "
                f"fallback={response.fallback_used}  → slumbot incr={incr!r}"
            )

        r = slumbot_act(token, incr)
        if "error_msg" in r:
            stats.api_rejections += 1
            stats.api_rejection_logs.append(
                f"hand_action={action_str!r} our_incr={incr!r} err={r['error_msg']}"
            )
            # Try to recover by folding
            r_fold = slumbot_act(token, "f")
            if "error_msg" in r_fold:
                raise RuntimeError(
                    f"API rejected our action {incr!r} and recovery fold also failed: "
                    f"{r_fold['error_msg']}"
                )
            r = r_fold
        token = r.get("token") or token


# ───────── stats helpers ─────────


def _mbb_stats(
    winnings_chips: list[int], bb: int = SLUMBOT_BB
) -> tuple[float, float, float, float]:
    n = len(winnings_chips)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(winnings_chips, dtype=np.float64)
    mean_chips = float(arr.mean())
    stderr_chips = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    mean_mbb = mean_chips / bb * 1000.0
    half = 1.96 * stderr_chips / bb * 1000.0
    return mean_mbb, mean_mbb - half, mean_mbb + half, stderr_chips / bb * 1000.0


# ───────── main ─────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="strategy-pilot-v2.db")
    p.add_argument("--abstraction-dir", default="abstraction")
    p.add_argument("--n-hands", type=int, default=10)
    p.add_argument("--rng-seed", type=int, default=2026)
    p.add_argument("--inter-hand-sleep", type=float, default=0.3)
    p.add_argument("--debug", action="store_true", help="Print per-decision details")
    p.add_argument("--debug-hands", type=int, default=10, help="Print details for first N hands")
    args = p.parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2
    print(f"Resolved DB path: {db_path} ({db_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Loading AbstractionTables from {args.abstraction_dir!r}…")
    abstraction = AbstractionTables(path=args.abstraction_dir)

    print(
        "\nCAVEAT: bot trained for 6-max, deployed against HU Slumbot via "
        "seat-mapping shim. Expect tight HU play."
    )
    print(
        f"Mapping: HU SB → 6-max seat {HERO_SB_SEAT} (SB position), "
        f"HU BB → seat {HERO_BB_SEAT} (BB position), "
        f"seats 2-5 inactive (stack=0), button_seat={BUTTON_SEAT}."
    )

    trained_db = open_db(f"sqlite:///{db_path}")
    adapter = RuntimeAdapter(db=trained_db, abstraction=abstraction, rng_seed=args.rng_seed)

    print(f"\nPlaying {args.n_hands} hands vs Slumbot…\n")
    stats = MatchStats()
    token: str | None = None
    t0 = time.perf_counter()
    for i in range(args.n_hands):
        debug = args.debug or i < args.debug_hands
        if debug:
            print(f"── hand {i + 1}/{args.n_hands} ──")
        try:
            token, win = play_one_hand(token, adapter, stats, debug=debug)
        except RuntimeError as e:
            print(f"ERROR on hand {i + 1}: {e}", file=sys.stderr)
            return 3
        stats.hands_played += 1
        stats.winnings.append(win)
        if (i + 1) % max(args.n_hands // 10, 1) == 0 or debug:
            running_mean, lo, hi, _ = _mbb_stats(stats.winnings)
            print(
                f"  [{i + 1:>{len(str(args.n_hands))}}/{args.n_hands}]  "
                f"hand winnings={win:+d}  cum mbb/hand={running_mean:+.1f}  "
                f"CI [{lo:+.1f}, {hi:+.1f}]"
            )
        time.sleep(args.inter_hand_sleep)
    elapsed = time.perf_counter() - t0

    # ── final report ──
    mean_mbb, lo_mbb, hi_mbb, se_mbb = _mbb_stats(stats.winnings)
    total_chips = sum(stats.winnings)
    print(f"\n── results over {stats.hands_played} hands ──")
    print(f"  elapsed:        {elapsed:.1f}s  ({stats.hands_played / elapsed:.2f} hands/sec)")
    print(f"  total chips:    {total_chips:+,d}  ({total_chips / SLUMBOT_BB:+.1f} BB)")
    print(
        f"  mbb/hand:       {mean_mbb:+.2f}   95% CI [{lo_mbb:+.2f}, {hi_mbb:+.2f}]   stderr {se_mbb:.2f}"
    )
    print(
        f"  decisions:      {stats.decisions}  ({stats.decisions / max(stats.hands_played, 1):.1f}/hand avg)"
    )
    print(f"  API rejections: {stats.api_rejections}  ", end="")
    if stats.api_rejections > 0:
        print("⚠ (translation bugs — see log lines below)")
        for line in stats.api_rejection_logs[:20]:
            print(f"    - {line}")
    else:
        print("✓ none")

    # Per-street fallback
    print("\n  ── trained-side fallback by street ──")
    print(f"    {'street':<8s} {'exact':>14s} {'nearest_nbr':>14s} {'default_pol':>14s}")
    street_names = ("preflop", "flop", "turn", "river")
    for s in range(4):
        row = stats.fallback_by_street[s]
        tot = sum(row.values())
        if tot == 0:
            continue
        print(
            f"    {street_names[s]:<8s} "
            f"{row['exact']:>6d} ({row['exact'] / tot:5.1%})  "
            f"{row['nearest_neighbor']:>6d} ({row['nearest_neighbor'] / tot:5.1%})  "
            f"{row['default_policy']:>6d} ({row['default_policy'] / tot:5.1%})"
        )

    # ── verdict against the user's targets ──
    print("\n── verdict ──")
    if mean_mbb > 50:
        print(f"  STRONG: {mean_mbb:+.1f} mbb/hand > +50. Bot is genuinely good vs Slumbot.")
    elif mean_mbb > -200:
        print(
            f"  COMPETITIVE: {mean_mbb:+.1f} mbb/hand in [-200, +50]. Within "
            "expected range for a 6-max-trained bot mis-deployed in HU."
        )
    elif mean_mbb > -300:
        print(
            f"  WEAK: {mean_mbb:+.1f} mbb/hand in [-300, -200]. Significant loss, "
            "but plausibly attributable to HU mis-deployment."
        )
    else:
        print(
            f"  ALARMING: {mean_mbb:+.1f} mbb/hand < -300. Worse than a 2018 CFR bot; "
            "warrants investigation."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
