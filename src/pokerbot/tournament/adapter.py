"""Tournament-aware decision adapter.

Wraps a `RuntimeAdapter` with three operating regimes:

  1. Cash mode (`game_type == "cash"`): pure passthrough to the base adapter.
     The cash bot is unchanged.

  2. Tournament + short stack (`hero_stack_bb < 15`): defer to the push/fold
     Nash tables in `pokerbot.tournament.pushfold`. Below 15 BB the only sane
     actions are push or fold (or call-vs-shove), and an unexploitative
     ChipEV solver is far more accurate than a 6-max CFR policy trained on
     100 BB starting stacks.

  3. Tournament + deep stack (`hero_stack_bb >= 15`): use the base CFR policy
     plus ICM risk-aversion scaling. The base produces an action distribution;
     `_apply_risk_scaling` shifts mass from "risky" actions (RAISE_3_5X,
     BET_150, ALL_IN) toward "safe" actions (FOLD, CHECK_CALL) by an amount
     proportional to `1 - risk_factor`. The risk factor itself is computed
     from bubble proximity, stack size, and the ICM dollar-vs-chip share
     gap (via `pokerbot.tournament.icm.icm_equity`).

Design constraints honored:
  - `RuntimeAdapter` is NOT modified. We use a temporary opponent-model swap
    (`_ICMOpponentModel`) to inject risk scaling into `base.decide()`. The
    swap is reverted via `try/finally`. Single-threaded callers only.
  - Risk factor is always in [0.5, 1.0]; this is a hard design contract.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

from pokerbot.abstraction import ActionType, parse_card
from pokerbot.abstraction.encoding import position_from_seats
from pokerbot.runtime.opponent import ObservedHistory, OpponentModel
from pokerbot.runtime.schema import ActionOut, ActionResponse, GameStateRequest
from pokerbot.tournament.icm import icm_equity
from pokerbot.tournament.pushfold import pushfold_decision

if TYPE_CHECKING:
    from pokerbot.abstraction import InfoSet
    from pokerbot.runtime.adapter import RuntimeAdapter
    from pokerbot.tournament.state import TournamentState

# ─────────── action-risk classification ───────────

_RISKY_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.ALL_IN, ActionType.RAISE_3_5X, ActionType.BET_150}
)
_MID_RISK_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.RAISE_2_5X, ActionType.BET_66, ActionType.BET_100}
)
# Safe actions ({FOLD, CHECK_CALL, BET_33}) receive the redistributed mass.
# BET_33 is left as-is (already a "safe" sizing); redistribution goes to
# FOLD + CHECK_CALL evenly. The legality gate in `RuntimeAdapter.decide`
# masks out the illegal one at sampling time.

SHORT_STACK_BB_THRESHOLD: float = 15.0
RISK_FACTOR_MIN: float = 0.5
RISK_FACTOR_MAX: float = 1.0


# ─────────── ICM opponent-model wrapper ───────────


class _ICMOpponentModel(OpponentModel):
    """Wraps another OpponentModel; applies risk-aversion scaling after.

    Composed: `adjust = base.adjust → apply_scaling`. Used as a temporary
    drop-in for `RuntimeAdapter.opponent_model` during a single decision.
    """

    def __init__(
        self,
        base_model: OpponentModel,
        risk_factor: float,
        apply_fn: object,  # callable; typing kept loose to avoid circular import
    ) -> None:
        self._base_model = base_model
        self._risk_factor = risk_factor
        self._apply_fn = apply_fn

    def adjust(
        self,
        infoset: InfoSet,
        base_probs: dict[ActionType, float],
        observed_history: ObservedHistory,
    ) -> dict[ActionType, float]:
        adjusted = self._base_model.adjust(infoset, base_probs, observed_history)
        result: dict[ActionType, float] = self._apply_fn(adjusted, self._risk_factor)  # type: ignore[operator]
        return result


# ─────────── TournamentAdapter ───────────


class TournamentAdapter:
    """Wraps a RuntimeAdapter with tournament-aware decision logic."""

    def __init__(self, base_adapter: RuntimeAdapter, rng_seed: int = 0) -> None:
        self.base = base_adapter
        self.rng: random.Random = random.Random(rng_seed)

    # ── public API ──

    def decide(
        self,
        request: GameStateRequest,
        tournament_state: TournamentState | None = None,
    ) -> ActionResponse:
        if request.game_type == "cash":
            return self.base.decide(request)
        if tournament_state is None:
            raise ValueError("tournament_state required when game_type='tournament'")
        if tournament_state.hero_stack_bb < SHORT_STACK_BB_THRESHOLD:
            return self._pushfold_decide(request, tournament_state)
        return self._icm_weighted_decide(request, tournament_state)

    # ── short-stack path ──

    def _pushfold_decide(self, request: GameStateRequest, ts: TournamentState) -> ActionResponse:
        t0 = time.perf_counter()
        hole_cards = (
            parse_card(request.hero_hole[0]),
            parse_card(request.hero_hole[1]),
        )
        stack_bb_int = max(2, min(15, round(ts.hero_stack_bb)))

        # Translate schema position (SB-relative) → pushfold position (first-to-act order).
        # In HU the two conventions coincide; for 3+ players, preflop UTG = schema_pos 2.
        schema_pos = position_from_seats(request.button_seat, request.hero_seat, request.table_size)
        if ts.players_remaining == 2:
            pushfold_pos = schema_pos
        else:
            pushfold_pos = (schema_pos - 2) % ts.players_remaining

        decision = pushfold_decision(
            hole_cards=hole_cards,
            stack_bb=stack_bb_int,
            position=pushfold_pos,
            players_remaining=ts.players_remaining,
            action_history=list(request.action_history),
        )

        # Sample
        labels = list(decision.keys())
        probs = [decision[lbl] for lbl in labels]
        total = sum(probs)
        sampled_label = self.rng.choices(labels, weights=probs, k=1)[0] if total > 0 else "fold"
        sampled_freq = decision.get(sampled_label, 0.0)

        # Translate label → ActionType + chip amount + action string
        hero_stack = request.stacks[request.hero_seat]
        action_str: ActionOut
        if sampled_label == "fold":
            action_type = ActionType.FOLD
            emitted_amount = 0
            action_str = "fold"
        elif sampled_label == "push":
            action_type = ActionType.ALL_IN
            emitted_amount = hero_stack
            action_str = "raise" if request.to_call > 0 else "bet"
        elif sampled_label == "call":
            action_type = ActionType.CHECK_CALL
            emitted_amount = min(request.to_call, hero_stack)
            action_str = "call" if emitted_amount > 0 else "check"
        else:
            raise RuntimeError(f"unknown pushfold label: {sampled_label!r}")

        infoset = self.base.build_infoset(request)
        version = self.base.db.current_version()

        return ActionResponse(
            action=action_str,
            amount=emitted_amount,
            abstract_action=action_type.name,
            probability_sampled=float(sampled_freq),
            infoset_hash=infoset.hash16().hex(),
            version=version,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            fallback_used="pushfold",
        )

    # ── deep-stack path ──

    def _icm_weighted_decide(
        self, request: GameStateRequest, ts: TournamentState
    ) -> ActionResponse:
        risk_factor = self._compute_risk_factor(ts)
        original_opp = self.base.opponent_model
        self.base.opponent_model = _ICMOpponentModel(
            original_opp, risk_factor, self._apply_risk_scaling
        )
        try:
            return self.base.decide(request)
        finally:
            self.base.opponent_model = original_opp

    # ── risk-factor computation ──

    def _compute_risk_factor(self, ts: TournamentState) -> float:
        """Risk factor in [0.5, 1.0]. 1.0 = no scaling; 0.5 = max risk-aversion.

        Combines:
          - Bubble proximity (`bubble_distance` of 1 or 2 is high pressure)
          - Pay-jump proximity (within 20% of `players_in_money` is moderate)
          - Stack-size adjustment (very short = play to survive anyway;
            very deep = ICM pressure from being the big stack near pay jumps)
          - ICM-equity check (if hero's dollar-share >> chip-share, ICM pressure)
        """
        base = 1.0

        # Bubble proximity penalty
        if ts.bubble_distance == 1:
            base *= 0.6
        elif ts.bubble_distance == 2:
            base *= 0.8
        elif 0 < ts.bubble_distance <= max(int(0.2 * ts.players_in_money), 1):
            base *= 0.9

        # Stack-size adjustment
        if ts.hero_stack_bb < 25:
            base = min(RISK_FACTOR_MAX, base + 0.15)
        elif ts.hero_stack_bb > 75:
            base = max(RISK_FACTOR_MIN, base - 0.1)

        # ICM-based adjustment: detect "stack pressure" via dollar-vs-chip share.
        if ts.players_remaining <= len(ts.payouts) and sum(ts.payouts) > 0:
            equity = icm_equity(list(ts.stacks), list(ts.payouts))
            total_chips = sum(ts.stacks)
            if total_chips > 0:
                chip_share = ts.stacks[ts.hero_index] / total_chips
                dollar_share = equity[ts.hero_index] / sum(ts.payouts)
                if dollar_share > 1.1 * chip_share:
                    base *= 0.95

        return max(RISK_FACTOR_MIN, min(RISK_FACTOR_MAX, base))

    # ── risk scaling ──

    def _apply_risk_scaling(
        self,
        base_probs: dict[ActionType, float],
        risk_factor: float,
    ) -> dict[ActionType, float]:
        """Shift mass from risky/mid-risk actions to safe actions.

        Risky actions (ALL_IN, RAISE_3_5X, BET_150) lose `(1 - risk_factor)` of
        their mass. Mid-risk actions (RAISE_2_5X, BET_66, BET_100) lose half
        that fraction. Moved mass is redistributed evenly to FOLD and CHECK_CALL;
        the downstream legality gate masks out the illegal one (FOLD vs no
        bet, CHECK_CALL vs ill-defined call).
        """
        if risk_factor >= 1.0:
            return dict(base_probs)

        new_probs: dict[ActionType, float] = dict(base_probs)
        moved_mass = 0.0
        for action in list(new_probs.keys()):
            p = new_probs[action]
            if p <= 0:
                continue
            if action in _RISKY_ACTIONS:
                shift = p * (1.0 - risk_factor)
            elif action in _MID_RISK_ACTIONS:
                shift = p * (1.0 - risk_factor) * 0.5
            else:
                continue
            new_probs[action] = p - shift
            moved_mass += shift

        if moved_mass > 0:
            half = moved_mass * 0.5
            new_probs[ActionType.FOLD] = new_probs.get(ActionType.FOLD, 0.0) + half
            new_probs[ActionType.CHECK_CALL] = new_probs.get(ActionType.CHECK_CALL, 0.0) + half

        # Renormalize (paranoia; should already sum to 1 within float noise).
        total = sum(new_probs.values())
        if total > 0:
            new_probs = {k: v / total for k, v in new_probs.items()}
        return new_probs


__all__ = ["RISK_FACTOR_MAX", "RISK_FACTOR_MIN", "SHORT_STACK_BB_THRESHOLD", "TournamentAdapter"]
