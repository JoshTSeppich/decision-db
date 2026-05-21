"""Running per-opponent stat tracker for archetype classification.

`OpponentStats` holds per-opponent counters; rates (VPIP/PFR/AF/3-bet/c-bet/
fold-to-c-bet) are computed on demand. Tracker is fed `ObservedAction`s one
hand at a time via `update_from_hand`, which converts per-action events into
per-hand binaries (the standard poker convention: VPIP/PFR/3-bet/c-bet/
fold-to-c-bet are per-hand rates; AF is the per-action ratio).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokerbot.abstraction import ActionType

# Action sets used for AF / preflop-raise classification.
# Imported lazily inside `update_from_hand` to avoid circular imports at
# package import time (this module is loaded before pokerbot.abstraction in
# some startup paths).


@dataclass(frozen=True, slots=True)
class ObservedAction:
    """One observed opponent action with pre-computed context flags.

    The caller is responsible for setting flags accurately; the tracker treats
    them as ground truth.

        street               0=preflop, 1=flop, 2=turn, 3=river
        action_type          chosen ActionType
        to_call              chips needed to call at decision time (>0 = facing a bet)
        voluntary_preflop    True iff street==0 and this action counts toward VPIP
                             (any non-FOLD preflop action that isn't BB's free
                             CHECK_CALL with to_call==0)
        pf_raises_before     # of preflop raises observed before this action this
                             hand (>=1 means a 3-bet opportunity exists)
        is_cbet_opportunity  True iff street==1 and this player was the preflop
                             aggressor AND first to act on the flop
        is_facing_cbet       True iff this action responds to a c-bet (preflop
                             aggressor's flop bet)
    """

    street: int
    action_type: ActionType
    to_call: int
    voluntary_preflop: bool = False
    pf_raises_before: int = 0
    is_cbet_opportunity: bool = False
    is_facing_cbet: bool = False


@dataclass(slots=True)
class OpponentStats:
    """Running statistics for a single opponent across multiple hands.

    Per-hand counters (VPIP/PFR/3-bet/cbet/fold-to-cbet) treat each hand as a
    binary event — a player who calls preflop twice in one hand counts as one
    VPIP increment, not two. AF counters (bets/raises/calls) are per-action.
    """

    hands_observed: int = 0
    preflop_voluntary_actions: int = 0  # hands with VPIP=yes (per-hand binary)
    preflop_raises: int = 0  # hands with PFR=yes (per-hand binary)
    preflop_3bets: int = 0
    preflop_3bet_opportunities: int = 0
    postflop_bets: int = 0  # per-action
    postflop_raises: int = 0  # per-action
    postflop_calls: int = 0  # per-action (CHECK_CALL with to_call > 0)
    cbets: int = 0
    cbet_opportunities: int = 0
    folds_to_cbet: int = 0
    faced_cbet: int = 0

    def vpip(self) -> float:
        return self.preflop_voluntary_actions / max(1, self.hands_observed)

    def pfr(self) -> float:
        return self.preflop_raises / max(1, self.hands_observed)

    def af(self) -> float:
        return (self.postflop_bets + self.postflop_raises) / max(1, self.postflop_calls)

    def three_bet(self) -> float:
        return self.preflop_3bets / max(1, self.preflop_3bet_opportunities)

    def cbet(self) -> float:
        return self.cbets / max(1, self.cbet_opportunities)

    def fold_to_cbet(self) -> float:
        return self.folds_to_cbet / max(1, self.faced_cbet)


class OpponentStatsTracker:
    """Keyed bag of OpponentStats, one per opponent identifier.

    Opponent identifiers are caller-assigned strings (e.g., seat IDs across a
    stable table). The tracker does not interpret them.
    """

    def __init__(self) -> None:
        self._opponents: dict[str, OpponentStats] = {}

    def get(self, opponent_id: str) -> OpponentStats:
        if opponent_id not in self._opponents:
            self._opponents[opponent_id] = OpponentStats()
        return self._opponents[opponent_id]

    def opponents(self) -> dict[str, OpponentStats]:
        """Snapshot of all tracked opponents (returns a copy of the mapping)."""
        return dict(self._opponents)

    def update_from_hand(
        self,
        opponent_id: str,
        hand_actions: list[ObservedAction],
    ) -> None:
        """Increment the named opponent's stats from one hand's actions.

        Per-hand binaries (VPIP/PFR/3-bet/cbet/fold-to-cbet) collapse multiple
        events into one increment; AF counters accumulate per action.
        """
        from pokerbot.abstraction import ActionType  # local import: see header

        aggressive: set[ActionType] = {
            ActionType.BET_33,
            ActionType.BET_66,
            ActionType.BET_100,
            ActionType.BET_150,
            ActionType.ALL_IN,
            ActionType.RAISE_2_5X,
            ActionType.RAISE_3_5X,
        }

        stats = self.get(opponent_id)
        stats.hands_observed += 1

        voluntary_pf = False
        raised_pf = False
        three_bet_opp = False
        three_bet_made = False
        cbet_opp = False
        cbet_made = False
        faced_cbet = False
        folded_to_cbet = False

        for a in hand_actions:
            if a.street == 0:
                if a.voluntary_preflop:
                    voluntary_pf = True
                if a.action_type in aggressive:
                    raised_pf = True
                if a.pf_raises_before >= 1:
                    three_bet_opp = True
                    if a.action_type in aggressive:
                        three_bet_made = True
            else:
                # Postflop AF + cbet tracking.
                if a.is_cbet_opportunity:
                    cbet_opp = True
                    if a.action_type in aggressive and a.to_call == 0:
                        cbet_made = True
                if a.is_facing_cbet:
                    faced_cbet = True
                    if a.action_type == ActionType.FOLD:
                        folded_to_cbet = True
                # AF per-action:
                if a.action_type in aggressive:
                    if a.to_call > 0:
                        stats.postflop_raises += 1
                    else:
                        stats.postflop_bets += 1
                elif a.action_type == ActionType.CHECK_CALL and a.to_call > 0:
                    stats.postflop_calls += 1
                # CHECK_CALL with to_call==0 is a check; doesn't affect AF.

        if voluntary_pf:
            stats.preflop_voluntary_actions += 1
        if raised_pf:
            stats.preflop_raises += 1
        if three_bet_opp:
            stats.preflop_3bet_opportunities += 1
            if three_bet_made:
                stats.preflop_3bets += 1
        if cbet_opp:
            stats.cbet_opportunities += 1
            if cbet_made:
                stats.cbets += 1
        if faced_cbet:
            stats.faced_cbet += 1
            if folded_to_cbet:
                stats.folds_to_cbet += 1


__all__ = ["ObservedAction", "OpponentStats", "OpponentStatsTracker"]
