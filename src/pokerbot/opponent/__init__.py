"""Opponent modeling (Cairn 4): archetype classification + static adjustment.

Public surface:
    - `OpponentStats`              per-opponent running counters
    - `OpponentStatsTracker`       keyed bag of stats with `update_from_hand`
    - `ObservedAction`             one event passed to `update_from_hand`
    - `Archetype`                  StrEnum of {nit, tag, lag, maniac, station, unknown}
    - `ArchetypeClassifier`        threshold-based classifier
    - `ArchetypeOpponentModel`     OpponentModel that classifies + adjusts probs
"""

from pokerbot.opponent.archetype import Archetype, ArchetypeClassifier
from pokerbot.opponent.model import ArchetypeOpponentModel
from pokerbot.opponent.stats import (
    ObservedAction,
    OpponentStats,
    OpponentStatsTracker,
)

__all__ = [
    "Archetype",
    "ArchetypeClassifier",
    "ArchetypeOpponentModel",
    "ObservedAction",
    "OpponentStats",
    "OpponentStatsTracker",
]
