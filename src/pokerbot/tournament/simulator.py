"""Tournament simulator (Cairn 5).

Runs a 90-player multi-table NLHE tournament with a blind schedule, table
consolidation, hand-for-hand bubble play, and finish-position tracking.

Architecture:
    `BlindSchedule`   maps a hand number to (sb, bb) chip values.
    `payouts(...)`    returns the prize table for N paid spots.
    `TournamentResult` holds per-hero outcome data.
    `TournamentSimulator.run_tournament(...)` runs one tournament end-to-end.

Hand simulation uses pokerkit directly (per hand: fresh state with the
players' current stacks at that table; button rotates each hand).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from pokerkit import Automation, NoLimitTexasHoldem

from pokerbot.abstraction import (
    AbstractAction,
    ActionType,
)
from pokerbot.opponent.stats import ObservedAction, OpponentStatsTracker
from pokerbot.runtime.schema import (
    ActionHistoryEntry,
    BlindsSchema,
    GameStateRequest,
)
from pokerbot.tournament.state import TournamentState

if TYPE_CHECKING:
    from collections.abc import Callable

# ─────────── pokerkit automations (mirror SimpleNLHEGame) ───────────

_AUTOMATIONS: Final[tuple[Automation, ...]] = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_DEALING,
    Automation.BOARD_DEALING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


# ─────────── blind schedule ───────────


_BLIND_LEVELS: Final[tuple[tuple[int, int], ...]] = (
    (25, 50),
    (50, 100),
    (75, 150),
    (100, 200),
    (150, 300),
    (200, 400),
    (300, 600),
    (400, 800),
    (500, 1000),
    (750, 1500),
    (1000, 2000),
    (1500, 3000),
    (2000, 4000),
    (3000, 6000),
    (5000, 10000),
)
HANDS_PER_LEVEL: Final[int] = 20


class BlindSchedule:
    """Maps total hands played → current (SB, BB). Levels rise every 20 hands.

    Above the final level, the top blinds repeat (no infinite escalation).
    """

    levels: tuple[tuple[int, int], ...] = _BLIND_LEVELS
    hands_per_level: int = HANDS_PER_LEVEL

    def at_hand(self, hand_number: int) -> tuple[int, int]:
        idx = min(hand_number // self.hands_per_level, len(self.levels) - 1)
        return self.levels[idx]


# ─────────── payouts ───────────


_PAYOUT_FRACTIONS: Final[tuple[float, ...]] = (
    0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.015, 0.0125, 0.0125,
)
# Sum: 0.30 + 0.20 + 0.15 + 0.10 + 0.07 + 0.05 + 0.04 + 0.03 + 0.02 + 0.015
#    + 0.0125 + 0.0125 = 1.0
PRIZE_POOL_DEFAULT: Final[float] = 9000.0
PAID_POSITIONS: Final[int] = len(_PAYOUT_FRACTIONS)


def payouts(prize_pool: float = PRIZE_POOL_DEFAULT) -> tuple[float, ...]:
    """Dollar payout per finishing position (index 0 = 1st place)."""
    return tuple(prize_pool * f for f in _PAYOUT_FRACTIONS)


# ─────────── result types ───────────


@dataclass(slots=True)
class TournamentResult:
    """Per-hero outcome of one tournament."""

    finish_position: int  # 1=winner, 90=first out
    prize: float
    hands_played: int  # total hands the hero participated in (not tournament hands)
    busted_at_hand: int | None  # hand number when hero busted; None if survived
    elapsed_seconds: float
    field_size: int


# ─────────── seating / table ───────────


@dataclass(slots=True)
class _PlayerSeat:
    pid: int
    stack: int


@dataclass(slots=True)
class _Table:
    seats: list[_PlayerSeat] = field(default_factory=list)
    button_offset: int = 0  # # of hands played → rotates button

    def live_seats(self) -> list[_PlayerSeat]:
        return [s for s in self.seats if s.stack > 0]


# ─────────── hand runner ───────────


def _pk_card_str(card: object) -> str:
    return f"{card.rank.value}{card.suit.value}"  # type: ignore[attr-defined]


def _abstract_to_json_entry(seat: int, street: int, abs_action: AbstractAction) -> ActionHistoryEntry:
    """Convert an internal abstract action into an ActionHistoryEntry."""
    at = abs_action.type
    amount = abs_action.amount_chips
    if at == ActionType.FOLD:
        type_str: str = "fold"
        amount = 0
    elif at == ActionType.CHECK_CALL:
        type_str = "check" if amount == 0 else "call"
    elif at == ActionType.ALL_IN:
        type_str = "all-in"
    elif street == 0:
        type_str = "raise"
    else:
        type_str = "bet" if amount > 0 else "check"
    return ActionHistoryEntry(seat=seat, street=street, type=type_str, amount=amount)  # type: ignore[arg-type]


_STREET_NAMES: Final[tuple[str, ...]] = ("preflop", "flop", "turn", "river")


def _build_request(
    pk: Any,
    *,
    table_size: int,
    blinds: tuple[int, int],
    button_seat: int,
    game_type: str,
    action_history: list[ActionHistoryEntry],
    initial_stacks: tuple[int, ...],
) -> GameStateRequest:
    actor = pk.actor_index
    real_stacks = [int(pk.stacks[i]) for i in range(table_size)]
    real_bets = [int(pk.bets[i]) for i in range(table_size)]
    pot = sum(initial_stacks) - sum(real_stacks)
    to_call = int(pk.checking_or_calling_amount or 0)
    bet_actor = real_bets[actor]
    min_raise_to_attr = pk.min_completion_betting_or_raising_to_amount
    min_raise_to = int(min_raise_to_attr) if min_raise_to_attr is not None else 0
    min_raise = max(min_raise_to - bet_actor, 0)
    max_raise_attr = pk.max_completion_betting_or_raising_to_amount
    max_raise = int(max_raise_attr) if max_raise_attr is not None else 0

    hero_hole = [_pk_card_str(c) for c in pk.hole_cards[actor]]
    board: list[str] = []
    for stack_ in pk.board_cards:
        board.extend(_pk_card_str(c) for c in stack_)

    return GameStateRequest(
        schema_version=1,
        game_type=game_type,  # type: ignore[arg-type]
        table_size=table_size,  # type: ignore[arg-type]
        blinds=BlindsSchema(sb=blinds[0], bb=blinds[1]),
        ante=0,
        hero_seat=actor,
        button_seat=button_seat,
        hero_hole=hero_hole,
        board=board,
        stacks=real_stacks,
        current_bets=real_bets,
        pot_committed=pot,
        to_call=to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        action_history=list(action_history),
    )


def _clamp_to_legal(chosen: ActionType, legal_ints: tuple[int, ...]) -> ActionType:
    """If `chosen` isn't pokerkit-legal, remap to the nearest semantic action."""
    if int(chosen) in legal_ints:
        return chosen
    legal_set = {ActionType(i) for i in legal_ints}
    if chosen == ActionType.FOLD and ActionType.CHECK_CALL in legal_set:
        return ActionType.CHECK_CALL
    if ActionType.CHECK_CALL in legal_set:
        return ActionType.CHECK_CALL
    if ActionType.ALL_IN in legal_set:
        return ActionType.ALL_IN
    return next(iter(legal_set))


def _legal_at(pk: Any) -> tuple[int, ...]:
    """Pokerkit-legal abstract actions at the current state."""
    legal: list[int] = []
    if pk.can_fold():
        legal.append(int(ActionType.FOLD))
    if pk.can_check_or_call():
        legal.append(int(ActionType.CHECK_CALL))
    if pk.can_complete_bet_or_raise_to():
        street = int(pk.street_index)
        if street == 0:
            legal.extend([int(ActionType.RAISE_2_5X), int(ActionType.RAISE_3_5X)])
        else:
            legal.extend(
                [int(ActionType.BET_33), int(ActionType.BET_66), int(ActionType.BET_100), int(ActionType.BET_150)]
            )
        legal.append(int(ActionType.ALL_IN))
    return tuple(legal)


def _apply_to_pk(
    pk: Any, action: ActionType, *, initial_stacks: tuple[int, ...]
) -> AbstractAction:
    """Translate an ActionType to a pokerkit operation. Returns the abstract
    record (with chips committed) so the caller can append to history."""
    actor = pk.actor_index
    stack = int(pk.stacks[actor])
    bet_actor = int(pk.bets[actor])
    to_call = int(pk.checking_or_calling_amount or 0)
    min_raise_to_attr = pk.min_completion_betting_or_raising_to_amount
    min_raise_to = int(min_raise_to_attr) if min_raise_to_attr is not None else 0
    # Pot for bet sizing = total chips committed across all players.
    pot = sum(initial_stacks) - sum(int(s) for s in pk.stacks)

    if action == ActionType.FOLD:
        pk.fold()
        return AbstractAction(ActionType.FOLD, 0)
    if action == ActionType.CHECK_CALL:
        check_amount = min(to_call, stack)
        pk.check_or_call()
        return AbstractAction(ActionType.CHECK_CALL, check_amount)

    # Raise/bet variants — compute target bet-to and clamp to legal bounds.
    if action == ActionType.RAISE_2_5X:
        target_to = max(round(2.5 * max(to_call, 1)) + bet_actor, min_raise_to)
    elif action == ActionType.RAISE_3_5X:
        target_to = max(round(3.5 * max(to_call, 1)) + bet_actor, min_raise_to)
    elif action == ActionType.BET_33:
        target_to = max(round(0.33 * pot) + to_call + bet_actor, min_raise_to)
    elif action == ActionType.BET_66:
        target_to = max(round(0.66 * pot) + to_call + bet_actor, min_raise_to)
    elif action == ActionType.BET_100:
        target_to = max(round(1.0 * pot) + to_call + bet_actor, min_raise_to)
    elif action == ActionType.BET_150:
        target_to = max(round(1.5 * pot) + to_call + bet_actor, min_raise_to)
    elif action == ActionType.ALL_IN:
        target_to = bet_actor + stack
    else:
        raise ValueError(f"unknown action {action}")

    max_bet_to = bet_actor + stack
    max_pk_to_attr = pk.max_completion_betting_or_raising_to_amount
    max_pk_to = int(max_pk_to_attr) if max_pk_to_attr is not None else max_bet_to
    target_to = min(max_bet_to, max_pk_to, max(target_to, min_raise_to))
    chips_committed = target_to - bet_actor
    pk.complete_bet_or_raise_to(target_to)

    recorded_type = (
        ActionType.ALL_IN
        if target_to == max_bet_to and action != ActionType.ALL_IN
        else action
    )
    return AbstractAction(recorded_type, chips_committed)


def play_tournament_hand(
    *,
    seat_stacks: list[int],
    blinds: tuple[int, int],
    button_seat: int,
    decide_fn: Callable[[int, GameStateRequest], str],
    hero_seat: int | None,
    rng: random.Random,
) -> tuple[list[int], int, dict[int, list[ObservedAction]]]:
    """Play one hand of NLHE with the given per-seat stacks. Button is rotated
    by remapping seat indices so the pokerkit "last seat = BTN" convention is
    preserved. Returns:
      - final_stacks_in_input_seat_order
      - hero_decisions_count
      - per_seat_observed_actions (dict input_seat → list[ObservedAction])

    `decide_fn(actor_seat, request)` returns an `ActionType.name` string.
    """
    n = len(seat_stacks)
    if n < 2:
        return list(seat_stacks), 0, {i: [] for i in range(n)}

    # Remap input seats so the BTN ends up at pokerkit seat n-1.
    # pokerkit seat 0 corresponds to input seat (button+1) % n (the SB).
    rotation = (button_seat + 1) % n
    pk_to_input = [(rotation + i) % n for i in range(n)]
    input_to_pk = [0] * n
    for pk_seat, inp_seat in enumerate(pk_to_input):
        input_to_pk[inp_seat] = pk_seat

    pk_stacks = tuple(seat_stacks[pk_to_input[i]] for i in range(n))

    random.seed(rng.getrandbits(64))
    pk = NoLimitTexasHoldem.create_state(
        automations=_AUTOMATIONS,
        ante_trimming_status=True,
        raw_antes=0,
        raw_blinds_or_straddles=blinds,
        min_bet=blinds[1],
        raw_starting_stacks=pk_stacks,
        player_count=n,
    )

    action_history: list[ActionHistoryEntry] = []
    hero_decisions = 0
    per_seat_actions: dict[int, list[ObservedAction]] = {i: [] for i in range(n)}

    # Track per-hand state needed for ObservedAction flags.
    pf_raises_count = 0  # # of preflop raises so far
    pf_aggressor_pk_seat: int | None = None  # last preflop aggressor (pokerkit seat)
    flop_action_seats_pk: list[int] = []  # pokerkit seats of actors on the flop, in order

    aggressive_types = {
        ActionType.BET_33,
        ActionType.BET_66,
        ActionType.BET_100,
        ActionType.BET_150,
        ActionType.ALL_IN,
        ActionType.RAISE_2_5X,
        ActionType.RAISE_3_5X,
    }

    while pk.status and pk.actor_index is not None:
        actor_pk = pk.actor_index
        actor_input = pk_to_input[actor_pk]
        # Send 'tournament' for hero turns; bots always see 'cash'
        # (their ParameterizedBot ignores game_type anyway).
        is_hero_turn = hero_seat is not None and actor_input == hero_seat
        game_type = "tournament" if is_hero_turn else "cash"
        request = _build_request(
            pk,
            table_size=n,
            blinds=blinds,
            button_seat=n - 1,
            game_type=game_type,
            action_history=action_history,
            initial_stacks=pk_stacks,
        )
        # Caller's decide_fn is keyed on INPUT seat.
        action_name = decide_fn(actor_input, request)
        chosen = ActionType[action_name]
        legal = _legal_at(pk)
        applied = _clamp_to_legal(chosen, legal)

        street_before = int(pk.street_index) if pk.street_index is not None else 0
        to_call_before = int(pk.checking_or_calling_amount or 0)

        # Compute per-action flags for the stats tracker BEFORE applying.
        voluntary_pf = False
        if street_before == 0 and applied != ActionType.FOLD and not (
            applied == ActionType.CHECK_CALL and to_call_before == 0
        ):
            voluntary_pf = True
        # cbet-opportunity: street==1 (flop), this player was pf aggressor,
        # AND no flop actions yet.
        is_cbet_opp = (
            street_before == 1
            and pf_aggressor_pk_seat is not None
            and actor_pk == pf_aggressor_pk_seat
            and len(flop_action_seats_pk) == 0
        )
        # facing-cbet: street==1, pf aggressor already bet on the flop
        # (exactly one flop action prior, by pf aggressor, that was aggressive).
        is_facing_cbet = (
            street_before == 1
            and pf_aggressor_pk_seat is not None
            and len(flop_action_seats_pk) >= 1
            and flop_action_seats_pk[0] == pf_aggressor_pk_seat
            and actor_pk != pf_aggressor_pk_seat
        )

        per_seat_actions[actor_input].append(
            ObservedAction(
                street=street_before,
                action_type=applied,
                to_call=to_call_before,
                voluntary_preflop=voluntary_pf,
                pf_raises_before=pf_raises_count if street_before == 0 else 0,
                is_cbet_opportunity=is_cbet_opp,
                is_facing_cbet=is_facing_cbet,
            )
        )

        # Update per-hand trackers BEFORE applying.
        if street_before == 0 and applied in aggressive_types:
            pf_raises_count += 1
            pf_aggressor_pk_seat = actor_pk
        if street_before == 1:
            flop_action_seats_pk.append(actor_pk)

        recorded = _apply_to_pk(pk, applied, initial_stacks=pk_stacks)
        action_history.append(_abstract_to_json_entry(actor_pk, street_before, recorded))
        if actor_input == hero_seat:
            hero_decisions += 1

    # Read final stacks; remap back to input seat order.
    pk_final = [int(pk.stacks[i]) for i in range(n)]
    final_input = [pk_final[input_to_pk[i]] for i in range(n)]
    return final_input, hero_decisions, per_seat_actions


# ─────────── tournament orchestration ───────────


@dataclass(slots=True)
class _TournamentConfig:
    field_size: int = 90
    starting_stack: int = 5000
    tables: int = 10
    seats_per_table: int = 9
    paid_positions: int = PAID_POSITIONS
    prize_pool: float = PRIZE_POOL_DEFAULT
    bubble_threshold: int = 13  # live count at or below triggers hand-for-hand
    max_hands_per_tournament: int = 2000  # safety cap


def _initial_seating(
    pids: list[int], cfg: _TournamentConfig, rng: random.Random
) -> list[_Table]:
    """Randomly distribute pids across tables, each starting with `seats_per_table`."""
    shuffled = list(pids)
    rng.shuffle(shuffled)
    tables: list[_Table] = []
    per = cfg.seats_per_table
    for ti in range(cfg.tables):
        seats = [
            _PlayerSeat(pid=shuffled[ti * per + j], stack=cfg.starting_stack)
            for j in range(per)
        ]
        tables.append(_Table(seats=seats))
    return tables


def _consolidate(tables: list[_Table], cfg: _TournamentConfig) -> None:
    """Redistribute players so no table drops below 6 unless live <= 9 (final)."""
    live_total = sum(len(t.live_seats()) for t in tables)
    if live_total <= 9:
        # Merge all into the first table
        all_live: list[_PlayerSeat] = []
        for t in tables:
            all_live.extend(t.live_seats())
        # Clear all
        for t in tables:
            t.seats = []
        tables[0].seats = all_live[: cfg.seats_per_table]
        # If somehow >9 live (shouldn't happen because we checked), spill to next
        spill = all_live[cfg.seats_per_table:]
        if spill and len(tables) > 1:
            tables[1].seats = spill
        return

    # Otherwise: fill tables below 6 from largest table.
    while True:
        live_counts = [(i, len(t.live_seats())) for i, t in enumerate(tables) if t.live_seats()]
        if not live_counts:
            return
        smallest = min(live_counts, key=lambda x: x[1])
        if smallest[1] >= 6:
            return  # nothing to do
        # Donor: largest table that has more than 6 (room to give)
        donors = [(i, c) for i, c in live_counts if c > 6 and i != smallest[0]]
        if not donors:
            return  # can't consolidate further
        donor_idx = max(donors, key=lambda x: x[1])[0]
        donor = tables[donor_idx]
        recip = tables[smallest[0]]
        # Move one live player from donor → recip
        for seat in donor.seats:
            if seat.stack > 0:
                donor.seats.remove(seat)
                recip.seats.append(seat)
                break
        else:
            return


class TournamentSimulator:
    """Runs one 90-player tournament. Hero plays `hero_decide_fn`; others use
    `bot_decide_fn(pid, request)`."""

    def __init__(
        self,
        cfg: _TournamentConfig | None = None,
        abstraction: Any | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else _TournamentConfig()
        self.abstraction = abstraction  # only needed if hero's adapter uses it
        self.blind_schedule = BlindSchedule()

    def run_tournament(
        self,
        *,
        hero_pid: int,
        bot_decide_fn: Callable[[int, GameStateRequest], str],
        hero_decide_fn: Callable[
            [GameStateRequest, TournamentState | None, tuple[str, ...]], str
        ],
        rng: random.Random,
        opponent_tracker: OpponentStatsTracker | None = None,
    ) -> TournamentResult:
        cfg = self.cfg
        pids = list(range(cfg.field_size))
        tables = _initial_seating(pids, cfg, rng)
        bust_order: list[int] = []
        hand_number = 0
        hero_busted_at: int | None = None
        hero_hands_played = 0
        t0 = time.perf_counter()

        while True:
            live_total = sum(len(t.live_seats()) for t in tables)
            if live_total <= 1:
                break
            if hand_number >= cfg.max_hands_per_tournament:
                break

            # Consolidate if any table is too small or final-table threshold hit.
            _consolidate(tables, cfg)

            blinds = self.blind_schedule.at_hand(hand_number)
            hand_number += 1

            # Each table plays one hand (or skips if <2 live).
            for table in tables:
                live = table.live_seats()
                if len(live) < 2:
                    continue
                hero_in_table = any(s.pid == hero_pid for s in live)
                hero_seat_idx = (
                    next(i for i, s in enumerate(live) if s.pid == hero_pid)
                    if hero_in_table
                    else None
                )

                # Build tournament_state for hero. We use TABLE-LEVEL counts:
                # pushfold tables are indexed [2, 9] so tournament-wide live
                # counts (potentially up to 90) can't be passed; ICM equity
                # at the table is an approximation of full-tournament ICM
                # that's exact at the final table and reasonable earlier.
                stacks_for_ts = [s.stack for s in live]
                paid_pos_for_ts = min(cfg.paid_positions, len(live))

                # Active opponent ids (string-keyed for the tracker), excluding hero.
                active_opp_ids: tuple[str, ...] = tuple(
                    f"pid_{s.pid}" for s in live if s.pid != hero_pid
                )

                # Bind loop vars as default args so the closure captures them
                # by value (ruff B023 — also a subtle correctness guard).
                def decide(
                    actor_seat: int,
                    request: GameStateRequest,
                    _live: list[_PlayerSeat] = live,
                    _stacks: list[int] = stacks_for_ts,
                    _paid: int = paid_pos_for_ts,
                    _blinds: tuple[int, int] = blinds,
                    _opp_ids: tuple[str, ...] = active_opp_ids,
                ) -> str:
                    pid = _live[actor_seat].pid
                    if pid == hero_pid:
                        if request.game_type == "tournament":
                            ts = TournamentState(
                                stacks=tuple(_stacks),
                                hero_index=actor_seat,
                                payouts=payouts(cfg.prize_pool),
                                players_remaining=len(_live),
                                players_in_money=_paid,
                                blinds_bb=_blinds[1],
                                starting_stack_bb=cfg.starting_stack // _blinds[1],
                            )
                            return hero_decide_fn(request, ts, _opp_ids)
                        return hero_decide_fn(request, None, _opp_ids)
                    return bot_decide_fn(pid, request)

                # Button rotates: button_offset increments each hand.
                button_seat = table.button_offset % len(live)
                table.button_offset += 1

                stacks_in = [s.stack for s in live]
                final_stacks, _hero_dec, per_seat_actions = play_tournament_hand(
                    seat_stacks=stacks_in,
                    blinds=blinds,
                    button_seat=button_seat,
                    decide_fn=decide,
                    hero_seat=hero_seat_idx,
                    rng=rng,
                )

                # Write stacks back to table.seats. table.seats may have dead
                # seats interleaved that we don't touch.
                live_pids = [s.pid for s in live]
                for new_stack, pid in zip(final_stacks, live_pids, strict=True):
                    for seat in table.seats:
                        if seat.pid == pid:
                            seat.stack = new_stack
                            break

                # Feed opponent tracker (one per non-hero seat that participated).
                if opponent_tracker is not None:
                    for seat_idx, pid in enumerate(live_pids):
                        if pid == hero_pid:
                            continue
                        actions = per_seat_actions.get(seat_idx, [])
                        if actions:
                            opponent_tracker.update_from_hand(f"pid_{pid}", actions)

                if hero_in_table:
                    hero_hands_played += 1

                # Record any newly-busted players in bust order.
                for seat in table.seats:
                    if seat.stack <= 0 and seat.pid not in bust_order:
                        bust_order.append(seat.pid)
                        if seat.pid == hero_pid and hero_busted_at is None:
                            hero_busted_at = hand_number

            # End of this round
            # If only 1 player has chips, exit
            live_after = sum(len(t.live_seats()) for t in tables)
            if live_after <= 1:
                break

        elapsed = time.perf_counter() - t0

        # Determine finish position.
        all_pids_in_bust_order = list(bust_order)  # first bust → idx 0
        survivors = [
            pid
            for pid in pids
            if pid not in bust_order and any(s.pid == pid and s.stack > 0 for t in tables for s in t.seats)
        ]
        # Survivors are sorted by stack (largest = highest position = 1st place)
        survivor_stacks: dict[int, int] = {}
        for pid in survivors:
            for t in tables:
                for s in t.seats:
                    if s.pid == pid:
                        survivor_stacks[pid] = s.stack
                        break
        survivors.sort(key=lambda p: -survivor_stacks.get(p, 0))

        # Final ranking: survivors first (top), then reverse of bust order (recent bust = better finish).
        # All_pids_in_bust_order[0] = first to bust = position (field_size).
        # bust_order[-1] = last to bust = position (len(survivors) + 1).
        ranking: list[int] = list(survivors)
        for pid in reversed(all_pids_in_bust_order):
            ranking.append(pid)
        finish_position = ranking.index(hero_pid) + 1

        prize = (
            payouts(cfg.prize_pool)[finish_position - 1]
            if finish_position <= cfg.paid_positions
            else 0.0
        )
        return TournamentResult(
            finish_position=finish_position,
            prize=prize,
            hands_played=hero_hands_played,
            busted_at_hand=hero_busted_at,
            elapsed_seconds=elapsed,
            field_size=cfg.field_size,
        )


__all__ = [
    "HANDS_PER_LEVEL",
    "PAID_POSITIONS",
    "PRIZE_POOL_DEFAULT",
    "BlindSchedule",
    "TournamentResult",
    "TournamentSimulator",
    "_TournamentConfig",
    "payouts",
    "play_tournament_hand",
]
