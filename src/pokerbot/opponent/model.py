"""ArchetypeOpponentModel: classifies active opponents, applies static adjustments.

The adjustment magnitudes are heuristic-light per Cairn 4 spec:

    vs Station: -40% ALL_IN, -40% RAISE_3_5X, -30% BET_150; removed mass goes
                to FOLD when facing a bet (FOLD present in base_probs) else
                to CHECK_CALL. Rationale: stations don't fold; bluffs lose.

    vs Maniac:  +25% relative to CHECK_CALL (mass shifted from FOLD), and
                -20% RAISE_2_5X and RAISE_3_5X (removed mass to CHECK_CALL).
                Rationale: maniacs over-bluff; calling down extracts value.

    vs Nit:     -30% CHECK_CALL when facing bet (mass to FOLD); -20% BET_33
                and BET_66 when not facing bet (mass to CHECK_CALL). Rationale:
                nits value-bet only; calling down + thin value-betting both lose.

    vs TAG/LAG/Unknown: passthrough.

Multi-way pots use the most-extreme archetype across active opponents in
priority Station > Maniac > Nit > LAG > TAG > Unknown — the exploitable
lines that fail worst against those types win the routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokerbot.abstraction import ActionType
from pokerbot.opponent.archetype import Archetype, ArchetypeClassifier
from pokerbot.runtime.opponent import ObservedHistory, OpponentModel

if TYPE_CHECKING:
    from pokerbot.abstraction import InfoSet
    from pokerbot.opponent.stats import OpponentStatsTracker


_ARCHETYPE_PRIORITY: tuple[Archetype, ...] = (
    Archetype.STATION,
    Archetype.MANIAC,
    Archetype.NIT,
    Archetype.LAG,
    Archetype.TAG,
    Archetype.UNKNOWN,
)


class ArchetypeOpponentModel(OpponentModel):
    """OpponentModel that classifies active opponents into archetypes
    and applies static adjustments to the base policy's action distribution.

    Active opponent identifiers are read from `observed_history.active_opponent_ids`.
    Unknown identifiers (no stats yet) classify as Unknown → no adjustment.
    """

    def __init__(
        self,
        tracker: OpponentStatsTracker,
        classifier: ArchetypeClassifier,
    ) -> None:
        self.tracker = tracker
        self.classifier = classifier
        # Caller may set this directly before each decision when ObservedHistory
        # cannot be plumbed through (e.g., the runtime simulator calls
        # `RuntimeAdapter.decide()` which constructs an empty ObservedHistory
        # internally — see RuntimeAdapter.decide). When non-empty, this takes
        # precedence over observed_history.active_opponent_ids.
        self._active_opponent_ids: tuple[str, ...] = ()

    def set_active_opponents(self, opp_ids: tuple[str, ...]) -> None:
        """Set the active opponents seen by the next `adjust()` call. Single-
        threaded use only; the caller is responsible for ordering this with
        the corresponding `RuntimeAdapter.decide()`."""
        self._active_opponent_ids = opp_ids

    def adjust(
        self,
        infoset: InfoSet,  # noqa: ARG002 - infoset not used by static archetype adjustment
        base_probs: dict[ActionType, float],
        observed_history: ObservedHistory,
    ) -> dict[ActionType, float]:
        active_ids = self._active_opponent_ids or observed_history.active_opponent_ids
        if not active_ids:
            return dict(base_probs)

        archetype = self._pick_archetype(active_ids)

        if archetype == Archetype.STATION:
            adjusted = _adjust_vs_station(base_probs)
        elif archetype == Archetype.MANIAC:
            adjusted = _adjust_vs_maniac(base_probs)
        elif archetype == Archetype.NIT:
            adjusted = _adjust_vs_nit(base_probs)
        else:
            # TAG, LAG, Unknown → passthrough
            return dict(base_probs)

        return _renormalize(adjusted)

    def _pick_archetype(self, active_ids: tuple[str, ...]) -> Archetype:
        observed: set[Archetype] = set()
        existing = self.tracker.opponents()
        for opp_id in active_ids:
            if opp_id in existing:
                observed.add(self.classifier.classify(existing[opp_id]))
            else:
                observed.add(Archetype.UNKNOWN)
        for candidate in _ARCHETYPE_PRIORITY:
            if candidate in observed:
                return candidate
        return Archetype.UNKNOWN


# ─────────── adjustment primitives ───────────


def _adjust_vs_station(base: dict[ActionType, float]) -> dict[ActionType, float]:
    """Suppress bluffs (ALL_IN, RAISE_3_5X, BET_150)."""
    out = dict(base)
    cut_all_in = out.get(ActionType.ALL_IN, 0.0) * 0.40
    cut_raise = out.get(ActionType.RAISE_3_5X, 0.0) * 0.40
    cut_bet150 = out.get(ActionType.BET_150, 0.0) * 0.30
    out[ActionType.ALL_IN] = out.get(ActionType.ALL_IN, 0.0) - cut_all_in
    out[ActionType.RAISE_3_5X] = out.get(ActionType.RAISE_3_5X, 0.0) - cut_raise
    out[ActionType.BET_150] = out.get(ActionType.BET_150, 0.0) - cut_bet150

    removed = cut_all_in + cut_raise + cut_bet150
    sink = _safe_sink(prefer_fold=ActionType.FOLD in base)
    out[sink] = out.get(sink, 0.0) + removed
    return out


def _adjust_vs_maniac(base: dict[ActionType, float]) -> dict[ActionType, float]:
    """Call down (+CHECK_CALL from FOLD), don't escalate (-RAISE_2_5X/RAISE_3_5X)."""
    out = dict(base)
    # +25% relative to CHECK_CALL, mass from FOLD (capped at FOLD's mass).
    cc = out.get(ActionType.CHECK_CALL, 0.0)
    fold = out.get(ActionType.FOLD, 0.0)
    shift_from_fold = min(cc * 0.25, fold)
    out[ActionType.CHECK_CALL] = cc + shift_from_fold
    out[ActionType.FOLD] = fold - shift_from_fold

    # -20% RAISE_2_5X and RAISE_3_5X, mass to CHECK_CALL.
    cut_r25 = out.get(ActionType.RAISE_2_5X, 0.0) * 0.20
    cut_r35 = out.get(ActionType.RAISE_3_5X, 0.0) * 0.20
    out[ActionType.RAISE_2_5X] = out.get(ActionType.RAISE_2_5X, 0.0) - cut_r25
    out[ActionType.RAISE_3_5X] = out.get(ActionType.RAISE_3_5X, 0.0) - cut_r35
    out[ActionType.CHECK_CALL] = out.get(ActionType.CHECK_CALL, 0.0) + cut_r25 + cut_r35
    return out


def _adjust_vs_nit(base: dict[ActionType, float]) -> dict[ActionType, float]:
    """Fold more when facing bet; check more (don't thin-value) when not."""
    out = dict(base)
    facing_bet = ActionType.FOLD in base and base.get(ActionType.FOLD, 0.0) > 0.0
    if facing_bet:
        # -30% CHECK_CALL when facing bet; mass to FOLD.
        cut_cc = out.get(ActionType.CHECK_CALL, 0.0) * 0.30
        out[ActionType.CHECK_CALL] = out.get(ActionType.CHECK_CALL, 0.0) - cut_cc
        out[ActionType.FOLD] = out.get(ActionType.FOLD, 0.0) + cut_cc
    else:
        # -20% BET_33 and BET_66 when not facing bet; mass to CHECK_CALL.
        cut_b33 = out.get(ActionType.BET_33, 0.0) * 0.20
        cut_b66 = out.get(ActionType.BET_66, 0.0) * 0.20
        out[ActionType.BET_33] = out.get(ActionType.BET_33, 0.0) - cut_b33
        out[ActionType.BET_66] = out.get(ActionType.BET_66, 0.0) - cut_b66
        out[ActionType.CHECK_CALL] = out.get(ActionType.CHECK_CALL, 0.0) + cut_b33 + cut_b66
    return out


def _safe_sink(*, prefer_fold: bool) -> ActionType:
    """Pick the destination action for redistributed mass.

    When facing a bet (FOLD legal), spill into FOLD. Otherwise spill into
    CHECK_CALL — always legal.
    """
    return ActionType.FOLD if prefer_fold else ActionType.CHECK_CALL


def _renormalize(probs: dict[ActionType, float]) -> dict[ActionType, float]:
    """Clamp at zero and rescale to sum 1.0; preserves zero-only inputs."""
    clipped = {k: max(0.0, v) for k, v in probs.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        return clipped
    return {k: v / total for k, v in clipped.items()}


__all__ = ["ArchetypeOpponentModel"]
