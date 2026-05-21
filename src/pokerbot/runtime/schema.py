"""Pydantic v2 input + output schemas for the runtime adapter (Spec.html §F).

`GameStateRequest` is the v1 best-guess JSON contract — the real championship
interface will land later (§H). When it does, a `championship_adapter` module
will map championship JSON → `GameStateRequest`; this internal schema stays
stable.

All models use `extra='forbid'` so unknown fields fail loud at the boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pokerbot.opponent.archetype import Archetype  # noqa: TC001  Pydantic needs runtime access

_STRICT_FROZEN = ConfigDict(extra="forbid", frozen=True)

ActionTypeJSON = Literal["fold", "check", "call", "bet", "raise", "all-in"]
GameType = Literal["cash", "tournament"]
FallbackUsed = Literal["exact", "nearest_neighbor", "default_policy", "pushfold"]
ActionOut = Literal["fold", "check", "call", "bet", "raise"]


class BlindsSchema(BaseModel):
    model_config = _STRICT_FROZEN
    sb: int = Field(ge=0)
    bb: int = Field(gt=0)


class ActionHistoryEntry(BaseModel):
    model_config = _STRICT_FROZEN
    seat: int = Field(ge=0)
    street: int = Field(ge=0, le=3)
    type: ActionTypeJSON
    amount: int = Field(ge=0)


class GameStateRequest(BaseModel):
    """v1 game state. See spec §F for field semantics."""

    model_config = _STRICT_FROZEN

    schema_version: int = Field(ge=1, le=1)
    game_type: GameType
    table_size: Literal[2, 3, 4, 5, 6, 7, 8, 9]  # widened to 2..9 for tournament play
    blinds: BlindsSchema
    ante: int = Field(default=0, ge=0)

    hero_seat: int = Field(ge=0)
    button_seat: int = Field(ge=0)

    hero_hole: list[str]  # parsed/validated by `parse_card` downstream
    board: list[str]

    stacks: list[int]
    current_bets: list[int]
    pot_committed: int = Field(ge=0)
    to_call: int = Field(ge=0)
    min_raise: int = Field(ge=0)
    max_raise: int = Field(ge=0)

    action_history: list[ActionHistoryEntry] = Field(default_factory=list)

    # Optional per-seat opponent archetype classifications shipped by the caller.
    # Length, when present, MUST equal table_size; per-slot None means
    # "no classification for this seat" (insufficient hands, hero's own seat,
    # sitting-out, etc.). Default None ⇒ no opponent-model adjustment is applied.
    opponent_archetypes: tuple[Archetype | None, ...] | None = None

    @model_validator(mode="after")
    def _check_opponent_archetypes_length(self) -> GameStateRequest:
        if self.opponent_archetypes is not None and len(self.opponent_archetypes) != self.table_size:
            raise ValueError(
                f"opponent_archetypes length {len(self.opponent_archetypes)} "
                f"must equal table_size {self.table_size}"
            )
        return self


class ActionResponse(BaseModel):
    model_config = _STRICT_FROZEN

    action: ActionOut
    amount: int = Field(ge=0)
    abstract_action: str  # ActionType.name, e.g. "BET_66"
    probability_sampled: float = Field(ge=0.0, le=1.0)
    infoset_hash: str  # hex, 32 chars
    version: int
    latency_ms: int = Field(ge=0)
    fallback_used: FallbackUsed


__all__ = [
    "ActionHistoryEntry",
    "ActionOut",
    "ActionResponse",
    "ActionTypeJSON",
    "BlindsSchema",
    "FallbackUsed",
    "GameStateRequest",
    "GameType",
]
