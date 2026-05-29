"""Zoom-side wire schema.

The frozen brain's contract (`INTEGRATION_CONTRACT.md`) is a stateless
`{seq, request: GameStateRequest}` exchange. The zoom exploiter REUSES that
`GameStateRequest` envelope unchanged (the eyes keep speaking it), and wraps it
in a NEW zoom-side message that adds an `opponent_id` for cross-hand
re-identification. `GameStateRequest` itself is never modified or extended.

The eyes do not emit `opponent_id` yet, so it defaults to a single placeholder —
enough to exercise the cross-hand plumbing end-to-end. See the TODOs for the
hand-off when the eyes supply stable ids / fingerprints and showdown reveals.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pokerbot.runtime.schema import GameStateRequest

_STRICT = ConfigDict(extra="forbid")

# TODO(eyes): replace with the sister project's stable per-seat opponent id, or a
# fingerprint-based re-id when names rotate. Until then every villain collapses
# onto this single id — fine for plumbing, not for real multi-villain profiling.
PLACEHOLDER_OPPONENT_ID: str = "zoom-villain-0"


class ZoomObservation(BaseModel):
    """A zoom decision request: the frozen-brain `GameStateRequest` plus zoom
    state needed for stateful, exploitative play.

    A plain `{seq, request}` envelope (what the eyes send today) validates here
    with `opponent_id` defaulting to the placeholder and no showdown reveals.
    """

    model_config = _STRICT

    seq: int
    request: GameStateRequest
    opponent_id: str = PLACEHOLDER_OPPONENT_ID
    # TODO(eyes): supply at showdown as {seat: ["As","Kd"]}. Enables L2 cross-hand
    # learning (we can only attribute observed actions to a card bucket once the
    # villain's hole cards are known). Absent today ⇒ L2 stays at its prior.
    revealed_holes: dict[int, list[str]] | None = None


class ZoomAdvice(BaseModel):
    """One advisory decision out. `advice` is the operator-facing render; the
    rest are structured fields + per-opponent diagnostics proving statefulness.
    """

    model_config = _STRICT

    advice: str  # operator-facing, e.g. "BOT SAYS: RAISE 75"
    opponent_id: str

    # Blueprint action (placeholder for the future L4 subgame solver output).
    action: str
    amount: int = Field(ge=0)
    abstract_action: str
    fallback_used: str

    # Stateful diagnostics.
    range_effective_combos: float
    range_top_combos: list[tuple[str, float]]
    opponent_observations: float


__all__ = ["PLACEHOLDER_OPPONENT_ID", "ZoomAdvice", "ZoomObservation"]
