"""Runtime adapter (Spec.html §F): JSON request → action_response.

Pipeline (latency budget per spec):
    1. Build InfoSet from request (cards → bucket, history → bytes, position)
    2. DB lookup chain: exact → nearest_neighbor → default policy
    3. Opponent-model adjustment on the probability vector
    4. Sample an ActionType weighted by adjusted probs
    5. Resolve the abstract action to a chip amount (clamped to legal range)
    6. Serialize the response
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Final

import numpy as np

from pokerbot.abstraction import (
    AbstractAction,
    ActionType,
    InfoSet,
    parse_card,
    resolve_action,
)
from pokerbot.abstraction.actions import STREET_BOUNDARY_BYTE
from pokerbot.abstraction.encoding import (
    ActionKind,
    effective_stack,
    encode_action_history_byte,
    position_from_seats,
    stack_bucket_from_eff_bb,
)
from pokerbot.runtime.default_policy import default_policy_action
from pokerbot.runtime.opponent import (
    IdentityOpponentModel,
    ObservedHistory,
    OpponentModel,
)
from pokerbot.runtime.schema import (
    ActionOut,
    ActionResponse,
    FallbackUsed,
    GameStateRequest,
)
from pokerbot.strategy_db.base import unpack_probs

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pokerbot.abstraction import Card, Street
    from pokerbot.strategy_db import StrategyDB
    from pokerbot.strategy_db.base import StrategyRow

# ─────────── constants ───────────

_STREET_BY_BOARD_LEN: Final[dict[int, Street]] = {
    0: "preflop",
    3: "flop",
    4: "turn",
    5: "river",
}
_STREET_INDEX: Final[dict[str, int]] = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

_BET_FRAC: Final[dict[ActionType, float]] = {
    ActionType.BET_33: 0.33,
    ActionType.BET_66: 0.66,
    ActionType.BET_100: 1.0,
    ActionType.BET_150: 1.5,
}
_RAISE_RATIO: Final[dict[ActionType, float]] = {
    ActionType.RAISE_2_5X: 2.5,
    ActionType.RAISE_3_5X: 3.5,
}


# ─────────── adapter ───────────


class RuntimeAdapter:
    """Drives one decision per JSON request. See module docstring + spec §F."""

    def __init__(
        self,
        db: StrategyDB,
        abstraction: object,  # AbstractionTables; typed loosely to avoid heavy import
        opponent_model: OpponentModel | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self.db = db
        self.abstraction = abstraction
        self.opponent_model = (
            opponent_model if opponent_model is not None else IdentityOpponentModel()
        )
        self.rng: random.Random = random.Random(rng_seed)

    # ─────────── public: infoset construction (testable) ───────────

    def build_infoset(self, request: GameStateRequest) -> InfoSet:
        """Construct the canonical InfoSet from a parsed GameStateRequest."""
        street_name = _street_name(request)
        hole = _parse_hole(request)
        board = _parse_board(request)
        card_bucket = self.abstraction.lookup(hole, board, street_name)  # type: ignore[attr-defined]
        stack_bucket = _stack_bucket(request)
        history = _build_history_bytes(request)
        position = position_from_seats(request.button_seat, request.hero_seat, request.table_size)
        return InfoSet(
            table_size=request.table_size,
            street=_STREET_INDEX[street_name],
            position=position,
            stack_bucket=stack_bucket,
            card_bucket=int(card_bucket),
            history=history,
        )

    # ─────────── public: decide ───────────

    def decide(self, request: GameStateRequest) -> ActionResponse:
        t0 = time.perf_counter()

        infoset = self.build_infoset(request)
        version = self.db.current_version()

        row: StrategyRow | None = self.db.get(infoset, version)
        fallback: FallbackUsed
        if row is not None:
            fallback = "exact"
        else:
            row = self.db.nearest_neighbor(infoset, version)
            fallback = "nearest_neighbor" if row is not None else "default_policy"

        if row is not None:
            base_probs = _row_to_probs(row)
        else:
            hole = _parse_hole(request)
            board = _parse_board(request)
            hero_stack = request.stacks[request.hero_seat]
            position = position_from_seats(
                request.button_seat, request.hero_seat, request.table_size
            )
            default_action = default_policy_action(
                hole=hole,
                board=board,
                street=_street_name(request),
                table_size=request.table_size,
                position=position,
                pot=request.pot_committed,
                to_call=request.to_call,
                stack=hero_stack,
            )
            base_probs = {default_action.type: 1.0}

        adjusted = self.opponent_model.adjust(infoset, dict(base_probs), ObservedHistory())
        legal_types = _legal_types_from_request(request)
        gated = _mask_probs_to_legal(adjusted, legal_types)
        sampled_type, sampled_prob = _sample_action_type(self.rng, gated)

        hero_stack = request.stacks[request.hero_seat]
        abstract = _build_abstract_with_amount(
            sampled_type,
            pot=request.pot_committed,
            to_call=request.to_call,
            stack=hero_stack,
            bb=request.blinds.bb,
        )
        emitted_amount = resolve_action(
            abstract, request.pot_committed, hero_stack, request.min_raise
        )
        emitted_amount = _enforce_legal_bet_floor(
            sampled_type,
            emitted_amount,
            to_call=request.to_call,
            bb=request.blinds.bb,
            stack=hero_stack,
        )
        action_str = _action_string(sampled_type, emitted_amount, request.to_call)

        return ActionResponse(
            action=action_str,
            amount=emitted_amount,
            abstract_action=sampled_type.name,
            probability_sampled=float(sampled_prob),
            infoset_hash=infoset.hash16().hex(),
            version=row.version if row is not None else version,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            fallback_used=fallback,
        )


# ─────────── helpers ───────────


def _street_name(request: GameStateRequest) -> Street:
    n = len(request.board)
    if n not in _STREET_BY_BOARD_LEN:
        raise ValueError(f"board must have 0, 3, 4, or 5 cards, got {n}")
    return _STREET_BY_BOARD_LEN[n]


def _parse_hole(request: GameStateRequest) -> tuple[Card, Card]:
    if len(request.hero_hole) != 2:
        raise ValueError(f"hero_hole must have 2 cards, got {len(request.hero_hole)}")
    a, b = request.hero_hole
    return (parse_card(a), parse_card(b))


def _parse_board(request: GameStateRequest) -> tuple[Card, ...]:
    return tuple(parse_card(c) for c in request.board)


def _stack_bucket(request: GameStateRequest) -> int:
    """Effective-stack bucket per §C: min(hero, max-remaining-opp) in BB."""
    hero_stack = request.stacks[request.hero_seat]
    folded = _folded_seats(request)
    remaining_opps = [
        s
        for seat, s in enumerate(request.stacks)
        if seat != request.hero_seat and seat not in folded and s > 0
    ]
    bb = max(request.blinds.bb, 1)
    eff = effective_stack(hero_stack, remaining_opps)
    return stack_bucket_from_eff_bb(eff // bb)


def _folded_seats(request: GameStateRequest) -> set[int]:
    return {a.seat for a in request.action_history if a.type == "fold"}


def _build_history_bytes(request: GameStateRequest) -> bytes:
    """§C history encoding by replaying JSON entries through the shared encoder.

    Tracks per-seat this-street commitments to compute `to_call_at_decision`
    accurately for each entry — critical for preflop 3-bet+ classification
    (see `encode_action_history_byte` doc). Caps at HISTORY_MAX=64 bytes.
    """
    if not request.action_history:
        return b""

    history = bytearray()
    prev_street = request.action_history[0].street
    blinds = request.blinds
    table_size = request.table_size
    pot_running = blinds.sb + blinds.bb + request.ante * table_size

    # Per-seat chips committed THIS STREET (resets on boundary).
    bets_this_street = [0] * table_size
    if prev_street == 0:
        # Preflop opens with blinds posted. SB and BB conventions per §C
        # (SB = (BTN+1)%N for 3+max, BB = SB+1).
        sb_seat = _sb_seat(request.button_seat, table_size)
        bb_seat = (sb_seat + 1) % table_size
        bets_this_street[sb_seat] = blinds.sb
        bets_this_street[bb_seat] = blinds.bb

    for entry in request.action_history:
        if entry.street != prev_street:
            history.append(STREET_BOUNDARY_BYTE)
            bets_this_street = [0] * table_size
            prev_street = entry.street

        actor_bet = bets_this_street[entry.seat] if entry.seat < table_size else 0
        max_bet = max(bets_this_street) if bets_this_street else 0
        to_call_at_decision = max(max_bet - actor_bet, 0)

        history.append(
            encode_action_history_byte(
                action_kind=_kind_from_json(entry.type),
                amount_chips=entry.amount,
                to_call_at_decision=to_call_at_decision,
                pot_at_decision=pot_running,
                street=entry.street,
                bb=blinds.bb,
            )
        )
        if entry.seat < table_size:
            bets_this_street[entry.seat] += entry.amount
        pot_running += entry.amount
        if len(history) >= 64:
            return bytes(history[:64])
    return bytes(history)


def _sb_seat(button_seat: int, table_size: int) -> int:
    """SB seat under our convention. HU: button = SB; otherwise: button+1."""
    return button_seat if table_size == 2 else (button_seat + 1) % table_size


def _kind_from_json(type_str: str) -> ActionKind:
    if type_str == "fold":
        return "fold"
    if type_str == "check":
        return "check"
    if type_str == "call":
        return "call"
    if type_str == "all-in":
        return "all-in"
    return "bet_or_raise"  # "bet" or "raise"


def _legal_types_from_request(request: GameStateRequest) -> set[ActionType]:
    """Derive pokerkit-legal action types from JSON chip context alone.

    Mirrors `SimpleNLHEGame._legal_abstract_at`'s post-pokerkit intersection
    so the runtime adapter doesn't sample actions that the rules engine would
    reject. Without this gate, `nearest_neighbor` rows (which match on
    card/position/stack bucket but ignore history) can carry probability mass
    on FOLD even when the current state is a free check, etc.

    Conservative — we approximate pokerkit's `can_complete_bet_or_raise_to`
    (which also enforces a raise-count cap) by checking `min_raise > 0 and
    max_raise > 0`. The JSON builder sets these to 0 when pokerkit reports the
    raise gate as closed.
    """
    legal: set[ActionType] = set()
    hero_stack = request.stacks[request.hero_seat]
    if hero_stack <= 0:
        return legal

    # Fold is legal iff there are chips at risk to call.
    if request.to_call > 0:
        legal.add(ActionType.FOLD)

    # Check/call always legal when actor has chips (CHECK_CALL clamps to stack).
    legal.add(ActionType.CHECK_CALL)

    raises_legal = request.min_raise > 0 and request.max_raise > 0
    if raises_legal:
        legal.add(ActionType.ALL_IN)
        street_idx = _STREET_INDEX[_street_name(request)]
        if street_idx == 0:
            legal.add(ActionType.RAISE_2_5X)
            legal.add(ActionType.RAISE_3_5X)
        else:
            legal.add(ActionType.BET_33)
            legal.add(ActionType.BET_66)
            legal.add(ActionType.BET_100)
            legal.add(ActionType.BET_150)
    return legal


def _mask_probs_to_legal(
    probs: dict[ActionType, float], legal: set[ActionType]
) -> dict[ActionType, float]:
    """Restrict the distribution to `legal` and renormalize.

    If `probs` has no mass on any legal action (e.g., a default-policy node
    chose an illegal action, or a stale NN row's mask is disjoint from the
    current legal set), fall back to uniform over `legal`. If `legal` itself
    is empty (actor has no chips), return the original distribution unchanged
    so downstream sampling raises the same error path it would have anyway.
    """
    if not legal:
        return probs
    masked = {t: p for t, p in probs.items() if t in legal and p > 0}
    total = sum(masked.values())
    if total > 0:
        return {t: p / total for t, p in masked.items()}
    u = 1.0 / len(legal)
    return dict.fromkeys(legal, u)


def _row_to_probs(row: StrategyRow) -> dict[ActionType, float]:
    """Expand a packed (mask, probs) into a dict keyed by ActionType."""
    probs = _ensure_float32(row.action_probs)
    out: dict[ActionType, float] = {}
    idx = 0
    mask = row.action_mask
    for bit in range(mask.bit_length() + 1):
        if mask & (1 << bit):
            out[ActionType(bit)] = float(probs[idx])
            idx += 1
    return out


def _ensure_float32(probs: NDArray[np.float32]) -> NDArray[np.float32]:
    if probs.dtype != np.float32:
        return probs.astype(np.float32)
    return probs


def _sample_action_type(
    rng: random.Random, probs: dict[ActionType, float]
) -> tuple[ActionType, float]:
    if not probs:
        raise ValueError("empty action distribution")
    types = list(probs.keys())
    weights = [probs[t] for t in types]
    total = sum(weights)
    if total <= 0:
        # All zero — fall back to uniform
        weights = [1.0] * len(types)
        total = float(len(types))
    sampled = rng.choices(types, weights=weights, k=1)[0]
    return sampled, probs[sampled] / total if probs[sampled] > 0 else 1.0 / len(types)


def _build_abstract_with_amount(
    at: ActionType, *, pot: int, to_call: int, stack: int, bb: int
) -> AbstractAction:
    if at == ActionType.FOLD:
        return AbstractAction(at, 0)
    if at == ActionType.CHECK_CALL:
        return AbstractAction(at, min(to_call, stack))
    if at == ActionType.ALL_IN:
        return AbstractAction(at, stack)
    if at in _RAISE_RATIO:
        # Anchor on the prior raise-to amount when facing a bet, else the
        # BB-equivalent open (to_call==0 — BB option / limped). Mirrors training
        # `_legal_preflop` so a sampled unopened raise emits a sane 2.5x/3.5x BB
        # open instead of a degenerate ~1-chip size both clamp to the same floor.
        anchor = to_call if to_call > 0 else max(bb, 1)
        target = round(_RAISE_RATIO[at] * anchor)
        return AbstractAction(at, target)
    frac = _BET_FRAC[at]
    target = to_call + round(frac * pot)
    return AbstractAction(at, target)


_NON_RAISE_TYPES: Final[frozenset[ActionType]] = frozenset(
    {ActionType.FOLD, ActionType.CHECK_CALL, ActionType.ALL_IN}
)


def _enforce_legal_bet_floor(
    sampled_type: ActionType,
    emitted_amount: int,
    *,
    to_call: int,
    bb: int,
    stack: int,
) -> int:
    """Defense-in-depth bet-size floor for bet/raise actions.

    Independent of the caller's `request.min_raise` (which may be miscomputed),
    enforce the absolute NLHE minimum: a legal raise increment is ≥ BB, so
    the delta over the actor's current commitment must be ≥ `to_call + bb`.
    For prior raises with `last_bet_size > bb` the caller's `min_raise` may be
    higher; `resolve_action` already clamps to that, so we only need the
    bb-based floor as a safety net here.

    Clamps the emitted amount UP to `to_call + bb`, capped at `stack`
    (going effectively all-in if the floor exceeds remaining stack).
    """
    if sampled_type in _NON_RAISE_TYPES:
        return emitted_amount
    floor = to_call + bb
    if emitted_amount >= floor:
        return emitted_amount
    return min(floor, stack)


def _action_string(at: ActionType, emitted_amount: int, to_call: int) -> ActionOut:
    if at == ActionType.FOLD:
        return "fold"
    if at == ActionType.CHECK_CALL:
        return "check" if emitted_amount == 0 else "call"
    # bets / raises / all-in
    return "raise" if to_call > 0 else "bet"


# Re-export for module consumers
__all__ = [
    "RuntimeAdapter",
    "unpack_probs",
]
