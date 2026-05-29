"""Turn a `GameStateRequest`'s action history into abstract observed actions.

`GameStateRequest.action_history` carries `(seat, street, type, amount)` JSON
entries but NOT the chip context (pot / to-call) at each decision, which the
abstract bet-size classifier needs. We reconstruct that context by replaying the
history exactly the way `pokerbot.runtime.adapter._build_history_bytes` does —
reusing the canonical `encode_action_history_byte` classifier and
`byte_to_action_type` so the abstract labels we feed L2/L3 are byte-identical to
what the trained model sees. Read-only against `pokerbot`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pokerbot.abstraction import byte_to_action_type
from pokerbot.abstraction.encoding import ActionKind, encode_action_history_byte

# board cards visible when acting on each street index (0=preflop .. 3=river).
_BOARD_LEN_BY_STREET: tuple[int, ...] = (0, 3, 4, 5)
_STREET_NAME: tuple[str, ...] = ("preflop", "flop", "turn", "river")


@dataclass(frozen=True, slots=True)
class ReplayedAction:
    """One history entry resolved to its abstract action + reconstructed context."""

    index: int  # position in request.action_history
    seat: int
    abstract_label: str  # ActionType.name, e.g. "RAISE_2_5X" / "CHECK_CALL"
    street: str  # "preflop" | "flop" | "turn" | "river"
    board_at_street: tuple[int, ...]  # board ints visible when this action was taken
    to_call_bb: float  # chips-to-call / bb at the moment of the action


def _kind_from_json(type_str: str) -> ActionKind:
    """Mirror of adapter._kind_from_json (spec §C taxonomy)."""
    if type_str in ("fold", "check", "call", "all-in"):
        return type_str  # type: ignore[return-value]
    return "bet_or_raise"  # "bet" or "raise"


def _sb_seat(button_seat: int, table_size: int) -> int:
    return button_seat if table_size == 2 else (button_seat + 1) % table_size


def replay_actions(request, board_ints: tuple[int, ...]) -> list[ReplayedAction]:
    """Replay the full action history, returning one `ReplayedAction` per entry.

    `board_ints` is the parsed full board (callers pass the parsed
    `request.board`); per-street board slices are taken from it. The pot/to-call
    bookkeeping replicates `adapter._build_history_bytes` exactly.
    """
    history = request.action_history
    if not history:
        return []

    blinds = request.blinds
    table_size = request.table_size
    bb = max(blinds.bb, 1)
    pot_running = blinds.sb + blinds.bb + request.ante * table_size

    prev_street = history[0].street
    bets_this_street = [0] * table_size
    if prev_street == 0:
        sb_seat = _sb_seat(request.button_seat, table_size)
        bb_seat = (sb_seat + 1) % table_size
        bets_this_street[sb_seat] = blinds.sb
        bets_this_street[bb_seat] = blinds.bb

    out: list[ReplayedAction] = []
    for idx, entry in enumerate(history):
        if entry.street != prev_street:
            bets_this_street = [0] * table_size
            prev_street = entry.street

        actor_bet = bets_this_street[entry.seat] if entry.seat < table_size else 0
        max_bet = max(bets_this_street) if bets_this_street else 0
        to_call_at_decision = max(max_bet - actor_bet, 0)

        byte = encode_action_history_byte(
            action_kind=_kind_from_json(entry.type),
            amount_chips=entry.amount,
            to_call_at_decision=to_call_at_decision,
            pot_at_decision=pot_running,
            street=entry.street,
            bb=blinds.bb,
        )
        board_len = _BOARD_LEN_BY_STREET[entry.street]
        out.append(
            ReplayedAction(
                index=idx,
                seat=entry.seat,
                abstract_label=byte_to_action_type(byte).name,
                street=_STREET_NAME[entry.street],
                board_at_street=board_ints[:board_len],
                to_call_bb=to_call_at_decision / bb,
            )
        )

        if entry.seat < table_size:
            bets_this_street[entry.seat] += entry.amount
        pot_running += entry.amount

    return out


__all__ = ["ReplayedAction", "replay_actions"]
