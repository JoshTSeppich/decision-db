"""Scripted tight-passive archetype agents — the Stage-1 fine-tune opponent pool.

The v5 over-aggression came from a pure-self-play training distribution that never
contained tight-passive opponents to exploit. These agents (NIT/STATION/TAG) are
that missing distribution: deterministic, rule-based, and validated to match the
live `pokerbot.opponent.archetype` signatures. They draw legality from the
Component 1 SPR gate, so their short-stack play inherits the same all-in continuum.
"""

from zoom.agents.archetypes import (
    LagAgent,
    NitAgent,
    StationAgent,
    TagAgent,
    build_archetype_pool,
)
from zoom.agents.base import AgentSpot, ScriptedAgent, postflop_strength, preflop_rank

__all__ = [
    "AgentSpot",
    "LagAgent",
    "NitAgent",
    "ScriptedAgent",
    "StationAgent",
    "TagAgent",
    "build_archetype_pool",
    "postflop_strength",
    "preflop_rank",
]
