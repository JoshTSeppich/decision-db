"""Opponent-modeling seam (Spec.html §F).

v1 ships `IdentityOpponentModel`; the runtime adapter takes an `OpponentModel`
by injection so v2+ exploitative models can plug in without touching the rest
of the runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pokerbot.abstraction import ActionType, InfoSet


@dataclass(frozen=True, slots=True)
class ObservedHistory:
    """Cross-hand opponent observations. Empty placeholder in v1; v2 populates
    `active_opponent_ids` so an exploit model can look up per-opponent stats.

    v2+ will populate aggregate stats (VPIP, PFR, AF, tendencies by board
    texture). For v1 callers that don't supply identifiers, the default empty
    tuple keeps the existing `ObservedHistory()` callsites byte-identical.
    """

    active_opponent_ids: tuple[str, ...] = ()


class OpponentModel(ABC):
    """Adjust base action probabilities given an observed opponent profile."""

    @abstractmethod
    def adjust(
        self,
        infoset: InfoSet,
        base_probs: dict[ActionType, float],
        observed_history: ObservedHistory,
    ) -> dict[ActionType, float]: ...


class IdentityOpponentModel(OpponentModel):
    """Returns base_probs unchanged. The v1 default."""

    def adjust(
        self,
        infoset: InfoSet,  # noqa: ARG002
        base_probs: dict[ActionType, float],
        observed_history: ObservedHistory,  # noqa: ARG002
    ) -> dict[ActionType, float]:
        return base_probs


__all__ = ["IdentityOpponentModel", "ObservedHistory", "OpponentModel"]
