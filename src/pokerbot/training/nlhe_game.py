"""SimpleNLHEGame — pokerkit-backed NLHE for Deep CFR training (Spec.html §E + §A/B/C).

Wraps `pokerkit.NoLimitTexasHoldem` and exposes the `Game[NLHEState]` interface
the trainer expects. Critical contract: `infoset_key(state)` produces the same
bytes that `RuntimeAdapter.build_infoset(equivalent_request).to_bytes()` would,
so DB rows written during training are reachable from production lookups.

The contract is enforced by:
    - Shared encoding helpers in `abstraction.encoding`
    - Same card_bucket lookup (caller passes one AbstractionTables to both sides)
    - Same history-byte encoder (`encode_action_history_byte`) — both sides
      classify bets/raises purely from chip math against the same pot/to_call
      context, so they agree by construction

Pokerkit's state object is mutable; CFR needs branching, so `apply_action`
deep-copies the underlying state. That's slow but correct — production
training amortizes the copy cost across the gradient step.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

import torch
from pokerkit import Automation, NoLimitTexasHoldem
from pokerkit.utilities import Card as PKCard
from pokerkit.utilities import Rank, Suit

from pokerbot.abstraction import (
    AbstractAction,
    ActionType,
    InfoSet,
    legal_abstract_actions,
    resolve_action,
)
from pokerbot.abstraction.actions import STREET_BOUNDARY_BYTE
from pokerbot.abstraction.encoding import (
    NUM_STACK_BUCKETS,
    ActionKind,
    effective_stack,
    encode_action_history_byte,
    position_from_seats,
    stack_bucket_from_eff_bb,
)
from pokerbot.abstraction.infoset import HISTORY_MAX
from pokerbot.training.game import Game, TerminalReward

if TYPE_CHECKING:
    from pokerbot.abstraction import AbstractionTables, Card, Street


# ───────── card encoding bridge ─────────

# Pokerkit's Rank enum stringifies to "2"..."9","T","J","Q","K","A" via .value.
# Our 0..51 encoding: rank * 4 + suit, rank ∈ 0..12, suit ∈ {c,d,h,s} → 0..3.
_RANK_CHAR_TO_INT: Final[dict[str, int]] = {ch: i for i, ch in enumerate("23456789TJQKA")}
_SUIT_CHAR_TO_INT: Final[dict[str, int]] = {"c": 0, "d": 1, "h": 2, "s": 3}


def _pk_card_to_int(card: PKCard) -> int:
    return _RANK_CHAR_TO_INT[str(card.rank.value)] * 4 + _SUIT_CHAR_TO_INT[str(card.suit.value)]


def _int_to_pk_card(card_id: int) -> PKCard:
    """Inverse of `_pk_card_to_int`. Used only in deterministic tests."""
    rank_char = "23456789TJQKA"[card_id >> 2]
    suit_char = "cdhs"[card_id & 3]
    return PKCard(Rank(rank_char), Suit(suit_char))


_STREET_NAMES: Final[tuple[Street, Street, Street, Street]] = (
    "preflop",
    "flop",
    "turn",
    "river",
)

_RAISING_TYPES: Final[frozenset[ActionType]] = frozenset(
    {
        ActionType.BET_33,
        ActionType.BET_66,
        ActionType.BET_100,
        ActionType.BET_150,
        ActionType.ALL_IN,
        ActionType.RAISE_2_5X,
        ActionType.RAISE_3_5X,
    }
)


# ───────── state ─────────


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One past action plus the decision context needed for §C history bytes.

    `pot_at_decision` and `to_call_at_decision` are snapshotted from the
    pokerkit state at the moment the action was taken; they're what
    `encode_action_history_byte` needs to classify bets/raises consistently
    with the runtime adapter (which replays JSON history to derive the same
    quantities).
    """

    seat: int
    street: int
    action: AbstractAction
    pot_at_decision: int
    to_call_at_decision: int


@dataclass(frozen=True, slots=True)
class NLHEState:
    """Immutable snapshot of an NLHE game state.

    `pk_state` is pokerkit's mutable `State`; we treat it as opaque (typed
    `Any` here because pokerkit's State isn't stub-friendly) and deep-copy
    before any pokerkit method call. `history` records every abstract action
    taken so far so we can encode infoset bytes consistent with the runtime.
    """

    pk_state: Any  # pokerkit.state.State
    initial_stacks: tuple[int, ...]
    history: tuple[HistoryEntry, ...]


# ───────── game ─────────


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


# CFR branches by cloning the pokerkit State before each action. copy.deepcopy
# is correct but ~22x slower than necessary: it recursively copies immutable
# Card/Operation/Pot leaves and config tuples. `_clone_state` shallow-copies the
# State (no __init__/__post_init__) and then copies only the mutable container
# fields (list/deque/set), recursing one level for nested containers like
# board_cards / hole_cards. Cards/Operations/Pots and config tuples are shared
# by reference — validated safe in training/profiles/PHASE_1_5_SPIKE.md: State
# has no cyclic refs, those leaves are immutable/self-contained, and the clone
# is bit-exact vs deepcopy with zero parent-mutation leaks across 200 random
# games played to terminal. Flip _USE_LEGACY_DEEPCOPY to fall back to deepcopy
# (used by the bit-exact equivalence test and as a debugging escape hatch).
_USE_LEGACY_DEEPCOPY: bool = False


def _clone_state(pk: Any) -> Any:  # pokerkit State, kept opaque like NLHEState.pk_state
    """Fast branch-clone of a pokerkit State (see _USE_LEGACY_DEEPCOPY note)."""
    if _USE_LEGACY_DEEPCOPY:
        return copy.deepcopy(pk)
    new = copy.copy(pk)  # shallow: every field starts shared with the parent
    for name, val in vars(pk).items():
        # Replace each mutable container with a private copy so mutations on the
        # clone (and pokerkit's automations) never reach the parent. Scalars and
        # immutable tuples (config, antes/blinds/starting_stacks) stay shared.
        if isinstance(val, list):
            setattr(
                new,
                name,
                [
                    list(x)
                    if isinstance(x, (list, deque))
                    else set(x)
                    if isinstance(x, set)
                    else x
                    for x in val
                ],
            )
        elif isinstance(val, deque):
            setattr(new, name, deque(val))
        elif isinstance(val, set):
            setattr(new, name, set(val))
    return new


# Compact infoset-features layout — kept tiny so a 256x3 MLP fits the spec budget.
# Indexed as a flat float32 vector of length `FEATURE_DIM`.
_FEATURE_HISTORY_BYTES: Final[int] = HISTORY_MAX  # 64
FEATURE_DIM: Final[int] = 5 + _FEATURE_HISTORY_BYTES + 3  # struct + history + 3 ratios = 72


class SimpleNLHEGame(Game[NLHEState]):
    """NLHE 2-/3-/4-/5-/6-/7-/8-/9-max wrapped around pokerkit. Matching the spec narrative §A,§B,§C."""

    num_actions: int = len(ActionType)  # = 9
    feature_dim: int = FEATURE_DIM

    def __init__(
        self,
        abstraction: AbstractionTables,
        *,
        blinds: tuple[int, int] = (5, 10),
        starting_stack: int = 1000,
        table_size: Literal[2, 3, 4, 5, 6, 7, 8, 9] = 6,
    ) -> None:
        if table_size not in (2, 3, 4, 5, 6, 7, 8, 9):
            raise ValueError(
                f"table_size must be one of (2, 3, 4, 5, 6, 7, 8, 9), got {table_size}"
            )
        if blinds[0] <= 0 or blinds[1] <= 0 or blinds[0] >= blinds[1]:
            raise ValueError(f"blinds must be (SB>0, BB>SB), got {blinds}")
        if starting_stack < 10 * blinds[1]:
            raise ValueError(f"starting_stack too small: {starting_stack} < 10·BB={10 * blinds[1]}")
        self.abstraction = abstraction
        self.blinds = blinds
        self.starting_stack = starting_stack
        self.table_size: int = table_size
        self.num_players: int = table_size
        self._initial_stacks: tuple[int, ...] = tuple([starting_stack] * table_size)

    # ───────── lifecycle ─────────

    def new_initial_state(self, rng: random.Random) -> NLHEState:
        """Deal a fresh hand. Pokerkit uses its own RNG — we seed `random` first
        so deals are reproducible relative to `rng.getrandbits()`.
        """
        # Pokerkit's auto-dealing uses Python's `random.shuffle` internally.
        random.seed(rng.getrandbits(64))
        pk = NoLimitTexasHoldem.create_state(
            automations=_AUTOMATIONS,
            ante_trimming_status=True,
            raw_antes=0,
            raw_blinds_or_straddles=self.blinds,
            min_bet=self.blinds[1],
            raw_starting_stacks=self._initial_stacks,
            player_count=self.table_size,
        )
        return NLHEState(pk_state=pk, initial_stacks=self._initial_stacks, history=())

    def is_terminal(self, state: NLHEState) -> bool:
        pk = state.pk_state
        return not pk.status or pk.actor_index is None

    def current_player(self, state: NLHEState) -> int:
        pk = state.pk_state
        if self.is_terminal(state):
            raise ValueError("current_player called on terminal state")
        actor = pk.actor_index
        if actor is None:
            raise ValueError("pokerkit reports no actor on a non-terminal state")
        return int(actor)

    # ───────── legality + apply ─────────

    def _legal_abstract_at(self, state: NLHEState) -> list[AbstractAction]:
        pk = state.pk_state
        actor = self.current_player(state)
        stack = int(pk.stacks[actor])
        bet_actor = int(pk.bets[actor])
        to_call = int(pk.checking_or_calling_amount or 0)
        # pot at decision time = chips already committed across all players
        pot = int(sum(state.initial_stacks) - sum(pk.stacks))
        # min_raise as additional chips beyond current bet
        raise_to_attr = pk.min_completion_betting_or_raising_to_amount
        min_raise_to = int(raise_to_attr) if raise_to_attr is not None else 0
        min_raise = max(min_raise_to - bet_actor, 0)
        street_name = _STREET_NAMES[int(pk.street_index)]
        actions = legal_abstract_actions(pot, to_call, stack, min_raise, street_name)

        # Pokerkit can reject raises that our abstraction nominally allows
        # (insufficient stack to raise, opponents already all-in, raise count
        # capped). Mirror its `can_*` gates so apply_action never fails.
        if not pk.can_fold():
            actions = [a for a in actions if a.type != ActionType.FOLD]
        if not pk.can_check_or_call():
            actions = [a for a in actions if a.type != ActionType.CHECK_CALL]
        if not pk.can_complete_bet_or_raise_to():
            actions = [a for a in actions if a.type not in _RAISING_TYPES]
        return actions

    def legal_actions(self, state: NLHEState) -> tuple[int, ...]:
        return tuple(int(a.type) for a in self._legal_abstract_at(state))

    def apply_action(self, state: NLHEState, action: int, rng: random.Random) -> NLHEState:  # noqa: ARG002
        if self.is_terminal(state):
            raise ValueError("apply_action called on terminal state")
        actor = self.current_player(state)
        abstract_options = self._legal_abstract_at(state)
        target_type = ActionType(action)
        matching = next((a for a in abstract_options if a.type == target_type), None)
        if matching is None:
            raise ValueError(
                f"action {target_type.name} not in legal set "
                f"{[a.type.name for a in abstract_options]}"
            )

        pk_old = state.pk_state
        stack = int(pk_old.stacks[actor])
        bet_actor = int(pk_old.bets[actor])
        pot = int(sum(state.initial_stacks) - sum(pk_old.stacks))
        to_call_at_decision = int(pk_old.checking_or_calling_amount or 0)
        min_raise_to = int(pk_old.min_completion_betting_or_raising_to_amount or 0)
        min_raise = max(min_raise_to - bet_actor, 0)
        chips_committed = resolve_action(matching, pot, stack, min_raise)

        pk_new = _clone_state(pk_old)
        street_before = int(pk_new.street_index)
        if target_type == ActionType.FOLD:
            pk_new.fold()
            recorded = AbstractAction(ActionType.FOLD, 0)
        elif target_type == ActionType.CHECK_CALL:
            pk_new.check_or_call()
            recorded = AbstractAction(ActionType.CHECK_CALL, chips_committed)
        else:
            # bet / raise / all-in → translate to pokerkit bet-to amount.
            bet_to = bet_actor + chips_committed
            max_bet_to = bet_actor + stack
            bet_to = min(max_bet_to, max(bet_to, min_raise_to))
            pk_new.complete_bet_or_raise_to(bet_to)
            # If clamped to all-in, record as ALL_IN so the byte encoder treats
            # this the same way the adapter does for an "all-in" JSON event.
            recorded_type = (
                ActionType.ALL_IN
                if bet_to == max_bet_to and target_type != ActionType.ALL_IN
                else target_type
            )
            recorded = AbstractAction(recorded_type, chips_committed)

        entry = HistoryEntry(
            seat=actor,
            street=street_before,
            action=recorded,
            pot_at_decision=pot,
            to_call_at_decision=to_call_at_decision,
        )
        return NLHEState(
            pk_state=pk_new,
            initial_stacks=state.initial_stacks,
            history=(*state.history, entry),
        )

    # ───────── terminal payoff ─────────

    def terminal_reward(self, state: NLHEState) -> TerminalReward:
        if not self.is_terminal(state):
            raise ValueError("terminal_reward called on non-terminal state")
        pk = state.pk_state
        deltas = tuple(
            float(int(pk.stacks[i]) - state.initial_stacks[i]) for i in range(self.table_size)
        )
        return TerminalReward(rewards=deltas)

    # ───────── infoset construction ─────────

    def _build_infoset(self, state: NLHEState) -> InfoSet:
        """Build the canonical InfoSet for the current actor. Mirrors RuntimeAdapter.build_infoset."""
        pk = state.pk_state
        actor = self.current_player(state)

        hole_pk = pk.hole_cards[actor]
        if len(hole_pk) != 2:
            raise RuntimeError(f"hero seat {actor} has {len(hole_pk)} cards, expected 2")
        hole: tuple[Card, Card] = (_pk_card_to_int(hole_pk[0]), _pk_card_to_int(hole_pk[1]))

        # board_cards: list[list[Card]] in pokerkit (one inner list per board, for run-it-twice
        # support). Single-board NLHE: just the first inner list.
        raw_board = pk.board_cards
        board_flat = [c for stack_ in raw_board for c in stack_] if raw_board else []
        board: tuple[Card, ...] = tuple(_pk_card_to_int(c) for c in board_flat)

        street_idx = int(pk.street_index)
        street_name = _STREET_NAMES[street_idx]
        card_bucket = int(self.abstraction.lookup(hole, board, street_name))

        # Effective stack bucket: min(hero, max remaining opponent) in BB
        hero_stack = int(pk.stacks[actor])
        statuses = pk.statuses
        remaining_opps = [
            int(pk.stacks[s])
            for s in range(self.table_size)
            if s != actor and statuses[s] and int(pk.stacks[s]) > 0
        ]
        eff = effective_stack(hero_stack, remaining_opps)
        stack_bucket = stack_bucket_from_eff_bb(eff // max(self.blinds[1], 1))

        # SB-relative position. Pokerkit's seat-to-blind convention:
        #   - 3+ max: seat 0 = SB, seat (N-1) = button.
        #   - HU:     seat 1 = SB = button (SB acts first preflop), seat 0 = BB.
        # In both cases `button_seat = self.table_size - 1` lines up with
        # pokerkit, and `position_from_seats` handles HU's button==SB branch.
        position = position_from_seats(self.table_size - 1, actor, self.table_size)

        history_bytes = self._encode_history(state.history)
        return InfoSet(
            table_size=self.table_size,
            street=street_idx,
            position=position,
            stack_bucket=stack_bucket,
            card_bucket=card_bucket,
            history=history_bytes,
        )

    def _encode_history(self, history: tuple[HistoryEntry, ...]) -> bytes:
        """Emit §C history bytes through the shared encoder.

        The encoder uses pure chip math against `pot_at_decision` /
        `to_call_at_decision` snapshotted by `apply_action`. The runtime
        adapter reconstructs the same quantities by replaying JSON history,
        so both sides produce identical bytes by construction.
        """
        if not history:
            return b""
        out = bytearray()
        prev_street = history[0].street
        bb = self.blinds[1]
        for entry in history:
            if entry.street != prev_street:
                out.append(STREET_BOUNDARY_BYTE)
                prev_street = entry.street
            out.append(
                encode_action_history_byte(
                    action_kind=_kind_from_abstract(entry.action),
                    amount_chips=entry.action.amount_chips,
                    to_call_at_decision=entry.to_call_at_decision,
                    pot_at_decision=entry.pot_at_decision,
                    street=entry.street,
                    bb=bb,
                )
            )
            if len(out) >= HISTORY_MAX:
                return bytes(out[:HISTORY_MAX])
        return bytes(out)

    def infoset_key(self, state: NLHEState) -> bytes:
        return self._build_infoset(state).to_bytes()

    def infoset_features(self, state: NLHEState) -> torch.Tensor:
        info = self._build_infoset(state)
        pk = state.pk_state

        feats = torch.zeros(FEATURE_DIM, dtype=torch.float32)
        # Structured prefix (5 floats)
        feats[0] = info.table_size / 9.0
        feats[1] = info.street / 3.0
        feats[2] = info.position / max(self.table_size - 1, 1)
        feats[3] = info.stack_bucket / max(NUM_STACK_BUCKETS - 1, 1)
        feats[4] = info.card_bucket / 200.0

        # Padded history (64 floats — each byte normalised to [0, 1))
        hist = info.history[:HISTORY_MAX]
        for i, byte in enumerate(hist):
            feats[5 + i] = byte / 255.0

        # Side info (3 floats): pot/start_stack, to_call/pot, folded_count/table_size
        pot = int(sum(state.initial_stacks) - sum(pk.stacks))
        to_call = int(pk.checking_or_calling_amount or 0)
        statuses = pk.statuses
        folded = sum(1 for s in range(self.table_size) if not statuses[s])

        feats[5 + _FEATURE_HISTORY_BYTES + 0] = pot / max(self.starting_stack, 1)
        feats[5 + _FEATURE_HISTORY_BYTES + 1] = to_call / max(pot + 1, 1)
        feats[5 + _FEATURE_HISTORY_BYTES + 2] = folded / max(self.table_size, 1)
        return feats


def _kind_from_abstract(action: AbstractAction) -> ActionKind:
    """Map an `AbstractAction` to the kind taxonomy used by the shared encoder."""
    if action.type == ActionType.FOLD:
        return "fold"
    if action.type == ActionType.CHECK_CALL:
        return "check" if action.amount_chips == 0 else "call"
    if action.type == ActionType.ALL_IN:
        return "all-in"
    return "bet_or_raise"


__all__ = [
    "FEATURE_DIM",
    "HistoryEntry",
    "NLHEState",
    "SimpleNLHEGame",
]
