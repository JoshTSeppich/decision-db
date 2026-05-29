"""L5 advisory service core (stateful, advisory-only).

Holds the cross-hand opponent state and per-hand range trackers, wires L2+L3 to
the real abstraction, and renders a recommended action. The blueprint action is
produced by the frozen brain's read-only adapter path — that call site is the
explicit seam where the L4 subgame solver will plug in.

This module is the testable core; `scripts/serve_zoom.py` wraps it in a
WebSocket server. It NEVER fires keystrokes (the operator acts manually) and
NEVER modifies `pokerbot` or the frozen `serve.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokerbot.abstraction import parse_card
from pokerbot.abstraction.encoding import position_from_seats

from zoom.abstraction_bridge import make_bucket_fn
from zoom.archetype_bridge import set_opponent_archetype
from zoom.observe import ReplayedAction, replay_actions
from zoom.opponent_model import DirichletOpponentModel, coarse_key, make_opponent_strategy
from zoom.range_tracker import PublicState, RangeTracker
from zoom.zoom_schema import ZoomAdvice, ZoomObservation

if TYPE_CHECKING:
    from pokerbot.runtime.adapter import RuntimeAdapter
    from pokerbot.runtime.schema import GameStateRequest


class ZoomExploiterService:
    """Stateful advisory brain for 3-handed zoom.

    State lives per re-identified `opponent_id`:
      * the Dirichlet opponent model persists ACROSS hands (it learns each
        opponent's action frequencies, updated from showdown reveals);
      * the range tracker is rebuilt EACH hand (it tracks one villain's current
        hole-card distribution).
    """

    def __init__(self, adapter: RuntimeAdapter) -> None:
        self.adapter = adapter
        self.bucket_fn = make_bucket_fn(adapter.abstraction)
        # One model instance keyed internally by opponent_id → cross-hand memory.
        self.model = DirichletOpponentModel()

        self._range: dict[str, RangeTracker] = {}
        self._hand_sig: dict[str, tuple[str, ...]] = {}
        self._board_seen: dict[str, tuple[int, ...]] = {}
        self._hist_cursor: dict[str, int] = {}
        # Villain actions seen THIS hand (with the villain's position), buffered so
        # they can be folded into the model once the hole cards are revealed.
        self._pending: dict[str, list[tuple[ReplayedAction, int]]] = {}
        self._obs_count: dict[str, float] = {}

    # ── per-decision entry point ───────────────────────────────────────────
    def advise(self, obs: ZoomObservation) -> ZoomAdvice:
        request = obs.request
        opp_id = obs.opponent_id
        board_ints = tuple(parse_card(c) for c in request.board)

        self._maybe_start_new_hand(opp_id, request, board_ints)
        self._reveal_new_board(opp_id, board_ints)
        self._maybe_seed_archetype(opp_id, request)
        self._consume_new_actions(opp_id, request, board_ints)
        if obs.revealed_holes:
            self._learn_from_showdown(opp_id, request, obs.revealed_holes)

        # ── L4: subgame solver goes here — consumes range_tracker.posterior()
        # (self._range[opp_id]) + the per-opponent models (self.model). For now
        # we return the frozen blueprint via the read-only adapter path (the same
        # path the frozen serve.py adapter uses). The solver will replace this
        # single call; everything above stays.
        response = self.adapter.decide(request)

        tracker = self._range[opp_id]
        advice = f"BOT SAYS: {response.action.upper()} {response.amount}"
        return ZoomAdvice(
            advice=advice,
            opponent_id=opp_id,
            action=response.action,
            amount=response.amount,
            abstract_action=response.abstract_action,
            fallback_used=response.fallback_used,
            range_effective_combos=tracker.effective_combos(),
            range_top_combos=tracker.top_combos(5),
            opponent_observations=self._obs_count.get(opp_id, 0.0),
        )

    # ── hand lifecycle ─────────────────────────────────────────────────────
    def _maybe_start_new_hand(
        self, opp_id: str, request: GameStateRequest, board_ints: tuple[int, ...]
    ) -> None:
        # Heuristic hand boundary: hero is dealt new hole cards each hand.
        # TODO(eyes): use an explicit hand id from the eyes for robustness.
        sig = tuple(request.hero_hole)
        if self._hand_sig.get(opp_id) == sig:
            return
        hero_hole = [parse_card(c) for c in request.hero_hole]
        self._range[opp_id] = RangeTracker(dead_cards=[*hero_hole, *board_ints])
        self._hand_sig[opp_id] = sig
        self._board_seen[opp_id] = board_ints
        self._hist_cursor[opp_id] = 0
        self._pending[opp_id] = []

    def _reveal_new_board(self, opp_id: str, board_ints: tuple[int, ...]) -> None:
        seen = self._board_seen.get(opp_id, ())
        fresh = [c for c in board_ints if c not in seen]
        if fresh:
            self._range[opp_id].reveal_board(fresh)
            self._board_seen[opp_id] = board_ints

    def _maybe_seed_archetype(self, opp_id: str, request: GameStateRequest) -> None:
        """Seed the opponent's prior from the per-seat archetypes the caller may
        ship (reusing the real Archetype enum via the Task-3 bridge)."""
        archetypes = request.opponent_archetypes
        if not archetypes:
            return
        for seat, arch in enumerate(archetypes):
            if seat == request.hero_seat or arch is None:
                continue
            set_opponent_archetype(self.model, opp_id, arch)
            break  # single placeholder opponent_id → first classified villain

    def _consume_new_actions(
        self, opp_id: str, request: GameStateRequest, board_ints: tuple[int, ...]
    ) -> None:
        replay = replay_actions(request, board_ints)
        cursor = self._hist_cursor.get(opp_id, 0)
        opp_strat = make_opponent_strategy(opp_id, self.model, self.bucket_fn)
        tracker = self._range[opp_id]
        for ra in replay[cursor:]:
            if ra.seat == request.hero_seat:
                continue  # hero's own actions don't inform the villain range
            # TODO(eyes): with per-seat ids, route to that seat's tracker/model.
            villain_pos = position_from_seats(request.button_seat, ra.seat, request.table_size)
            ps = PublicState(
                board=ra.board_at_street,
                street=ra.street,
                position=villain_pos,
                to_call_bb=ra.to_call_bb,
                facing="bet" if ra.to_call_bb > 0 else "none",
            )
            tracker.update(ra.abstract_label, ps, opp_strat)
            self._pending[opp_id].append((ra, villain_pos))
        self._hist_cursor[opp_id] = len(replay)

    def _learn_from_showdown(
        self, opp_id: str, request: GameStateRequest, revealed: dict[int, list[str]]
    ) -> None:
        """Fold this hand's villain actions into the cross-hand Dirichlet model,
        keyed by the now-revealed card bucket. This is the only path that moves
        L2 past its archetype prior, so it requires showdown reveals."""
        for seat, cards in revealed.items():
            if seat == request.hero_seat or len(cards) != 2:
                continue
            hole = (parse_card(cards[0]), parse_card(cards[1]))
            for ra, villain_pos in self._pending[opp_id]:
                if ra.seat != seat:
                    continue
                bucket = self.bucket_fn(hole, ra.board_at_street)
                key = coarse_key(ra.street, villain_pos, bucket, ra.to_call_bb)
                self.model.observe(opp_id, key, ra.abstract_label)
                self._obs_count[opp_id] = self._obs_count.get(opp_id, 0.0) + 1.0


__all__ = ["ZoomExploiterService"]
