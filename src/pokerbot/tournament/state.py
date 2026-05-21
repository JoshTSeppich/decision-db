"""Tournament-context state for ICM / push-fold decisions.

Distinct from `GameStateRequest`, which describes the current HAND.
`TournamentState` describes the surrounding tournament situation: who's
still alive, what they're stacked at, what the payouts are.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TournamentState:
    """Tournament-context state needed for ICM / push-fold decisions.

    Fields:
        stacks               Each live player's chip count.
        hero_index           Index of hero in `stacks`.
        payouts              Payout structure for the full tournament; index k
                             = prize for finishing in (k+1)th place. Length need
                             not match `players_remaining` (payouts describes
                             the whole structure, not the current snapshot).
        players_remaining    = `len(stacks)`; tracked separately for clarity.
        players_in_money     How many places pay; everyone else gets $0.
        blinds_bb            Current BB level in chips.
        starting_stack_bb    Starting stack in BBs (for bubble-distance
                             computation; advisory).

    Validation: `hero_index` in range, all stacks non-negative, `blinds_bb > 0`,
    `players_remaining > 0`, `players_in_money > 0`. `len(payouts) !=
    players_remaining` is ALLOWED.
    """

    stacks: tuple[int, ...]
    hero_index: int
    payouts: tuple[float, ...]
    players_remaining: int
    players_in_money: int
    blinds_bb: int
    starting_stack_bb: int

    def __post_init__(self) -> None:
        if not self.stacks:
            raise ValueError("stacks cannot be empty")
        if any(s < 0 for s in self.stacks):
            raise ValueError(f"stacks must all be non-negative: {self.stacks}")
        if not 0 <= self.hero_index < len(self.stacks):
            raise ValueError(
                f"hero_index {self.hero_index} out of range for {len(self.stacks)} stacks"
            )
        if self.blinds_bb <= 0:
            raise ValueError(f"blinds_bb must be positive: {self.blinds_bb}")
        if self.players_remaining <= 0:
            raise ValueError(f"players_remaining must be positive: {self.players_remaining}")
        if self.players_in_money <= 0:
            raise ValueError(f"players_in_money must be positive: {self.players_in_money}")

    @property
    def hero_stack_bb(self) -> float:
        return self.stacks[self.hero_index] / self.blinds_bb

    @property
    def in_money(self) -> bool:
        return self.players_remaining <= self.players_in_money

    @property
    def bubble_distance(self) -> int:
        """How many players need to bust before ITM. 0 means ITM already."""
        return max(0, self.players_remaining - self.players_in_money)


__all__ = ["TournamentState"]
