"""Bridge the real `pokerbot` Archetype enum onto the opponent model's prior seam.

The opponent model (L2) keys its archetype priors on uppercase strings
("NIT"/"TAG"/"LAG"/"MANIAC"/"STATION"). Those strings are *exactly the names* of
the real `pokerbot.opponent.archetype.Archetype` enum members, so this bridge
references the real enum (no duplication) and feeds its members in.

Read-only against `pokerbot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokerbot.opponent.archetype import Archetype

if TYPE_CHECKING:
    from zoom.opponent_model import DirichletOpponentModel


def archetype_prior_key(archetype: Archetype) -> str:
    """The opponent-model prior key for a real Archetype member.

    Equals the enum member name (e.g. ``Archetype.TAG`` → ``"TAG"``). The model's
    ``_prior_alpha`` uppercases its key and falls back to a uniform prior for any
    name it has no profile for, so ``Archetype.UNKNOWN`` correctly yields a
    uniform prior.
    """
    return archetype.name


def set_opponent_archetype(
    model: DirichletOpponentModel, opp_id: str, archetype: Archetype
) -> None:
    """Seed an opponent's prior in the Dirichlet model from a real Archetype."""
    model.set_archetype(opp_id, archetype_prior_key(archetype))


__all__ = ["archetype_prior_key", "set_opponent_archetype"]
