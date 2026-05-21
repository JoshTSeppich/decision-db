"""Archetype classification from running opponent stats (threshold rules)."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokerbot.opponent.stats import OpponentStats

MIN_HANDS_FOR_CLASSIFICATION: int = 20


class Archetype(StrEnum):
    NIT = "nit"
    TAG = "tag"
    LAG = "lag"
    MANIAC = "maniac"
    STATION = "station"
    UNKNOWN = "unknown"


class ArchetypeClassifier:
    """Threshold-based classifier with hand-count-aware margins.

    Base thresholds (locked):
        Nit     VPIP < 18%   AND PFR < 12%   AND AF > 1.5
        TAG     VPIP 18-26%  AND PFR 14-22%  AND AF > 2.0
        LAG     VPIP 26-40%  AND PFR 22-35%  AND AF > 2.5
        Maniac  VPIP > 40%   AND PFR > 30%   AND AF > 3.0
        Station VPIP > 35%   AND AF < 1.0    AND fold-to-cbet < 40%
        Unknown otherwise OR fewer than `MIN_HANDS_FOR_CLASSIFICATION` hands.

    Margins tighten the "greater-than" gates (VPIP/PFR/AF lower bounds)
    multiplicatively by `(1 + margin)` and the "less-than" gates for
    well-sampled stats (AF<1.0) multiplicatively by `(1 - margin)`. The
    fold-to-cbet gate is RELAXED at low hand counts (its denominator is
    the smallest of any stat and is dominated by noise) by `(1 + margin)`
    — at low confidence we don't want a noisy fcbet reading to block
    classification of an otherwise-clear Station.

    Band-archetype bounds (LAG/TAG) shrink inward symmetrically so the
    moderate-stat archetypes default to Unknown at low hand counts.

    Station is checked first because its VPIP band overlaps with LAG/Maniac;
    the AF<1.0 and fold-to-cbet<40% gates pick it out.
    """

    def _margin(self, hands_observed: int) -> float:
        """Margin shrinks as observations accumulate.

           <  20 hands: classifier returns Unknown (early-out before _margin)
           20-49 hands: margin = 0.15 (high uncertainty)
           50-99 hands: margin = 0.08 (moderate uncertainty)
          100-199 hands: margin = 0.04 (low uncertainty)
          200+      hands: margin = 0.0  (use base thresholds)
        """
        if hands_observed < MIN_HANDS_FOR_CLASSIFICATION:
            return float("inf")
        if hands_observed < 50:
            return 0.15
        if hands_observed < 100:
            return 0.08
        if hands_observed < 200:
            return 0.04
        return 0.0

    def classify(self, stats: OpponentStats) -> Archetype:
        if stats.hands_observed < MIN_HANDS_FOR_CLASSIFICATION:
            return Archetype.UNKNOWN

        vpip = stats.vpip()
        pfr = stats.pfr()
        af = stats.af()
        ftc = stats.fold_to_cbet()
        m = self._margin(stats.hands_observed)

        # Station: VPIP > 35% tightened, AF < 1.0 tightened, ftc < 40% relaxed.
        if (
            vpip > 0.35 * (1 + m)
            and af < 1.0 * (1 - m)
            and ftc < 0.40 * (1 + m)
        ):
            return Archetype.STATION

        # Maniac: all three thresholds tightened.
        if (
            vpip > 0.40 * (1 + m)
            and pfr > 0.30 * (1 + m)
            and af > 3.0 * (1 + m)
        ):
            return Archetype.MANIAC

        # LAG: band shrinks inward; AF lower bound tightens.
        if (
            0.26 * (1 + m) <= vpip <= 0.40 * (1 - m)
            and 0.22 * (1 + m) <= pfr <= 0.35 * (1 - m)
            and af > 2.5 * (1 + m)
        ):
            return Archetype.LAG

        # TAG: band shrinks inward; AF lower bound tightens.
        if (
            0.18 * (1 + m) <= vpip <= 0.26 * (1 - m)
            and 0.14 * (1 + m) <= pfr <= 0.22 * (1 - m)
            and af > 2.0 * (1 + m)
        ):
            return Archetype.TAG

        # Nit: VPIP/PFR upper bounds tighten; AF lower bound tightens.
        if (
            vpip < 0.18 * (1 - m)
            and pfr < 0.12 * (1 - m)
            and af > 1.5 * (1 + m)
        ):
            return Archetype.NIT

        return Archetype.UNKNOWN


__all__ = ["MIN_HANDS_FOR_CLASSIFICATION", "Archetype", "ArchetypeClassifier"]
