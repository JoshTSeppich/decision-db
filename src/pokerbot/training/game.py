"""Game-agnostic interface for the Deep CFR trainer (Spec.html §E).

The trainer never knows whether it's solving NLHE 6-max or Kuhn poker — it
talks to a `Game` instance through this ABC. The two concrete impls:

    `KuhnPokerGame`  — 3-card, 1-betting-round HU game; tests run on it because
                       it converges in seconds.
    `SimpleNLHEGame` — abstracted NLHE 6/8/9-max; the production target.

Each `Game` exposes both an "infoset key" (hashable identifier for the
information set) and an "infoset feature tensor" (fixed-size float vector for
the networks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    import random

    import torch


# State type is opaque to the trainer; subclasses pick their own.
StateT = TypeVar("StateT")


@dataclass(frozen=True, slots=True)
class TerminalReward:
    """Per-player chip delta at terminal nodes (zero-sum: sum ≈ 0 modulo rake)."""

    rewards: tuple[float, ...]


class Game(ABC, Generic[StateT]):
    """Minimal interface a game must expose for the CFR trainer."""

    num_players: int
    num_actions: int  # output dimensionality of the networks (max actions)
    feature_dim: int  # input dimensionality of the networks

    @abstractmethod
    def new_initial_state(self, rng: random.Random) -> StateT:
        """Sample a fresh game state (deals chance outcomes as needed)."""

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool: ...

    @abstractmethod
    def current_player(self, state: StateT) -> int:
        """0..num_players-1. Caller ensures `is_terminal` is False first."""

    @abstractmethod
    def legal_actions(self, state: StateT) -> tuple[int, ...]:
        """Indices into [0, num_actions) — which output slots are legal here."""

    @abstractmethod
    def apply_action(self, state: StateT, action: int, rng: random.Random) -> StateT:
        """Return the successor state. RNG used only for chance sampling."""

    @abstractmethod
    def terminal_reward(self, state: StateT) -> TerminalReward: ...

    @abstractmethod
    def infoset_key(self, state: StateT) -> bytes:
        """Hashable identifier for the acting player's information set."""

    @abstractmethod
    def infoset_features(self, state: StateT) -> torch.Tensor:
        """Fixed-size float tensor for the network input."""

    def legal_mask(self, state: StateT) -> tuple[bool, ...]:
        """Convenience: bool tuple of length `num_actions`."""
        legal = set(self.legal_actions(state))
        return tuple(i in legal for i in range(self.num_actions))


__all__ = ["Game", "TerminalReward"]
