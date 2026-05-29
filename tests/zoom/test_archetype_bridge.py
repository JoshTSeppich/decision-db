"""Task 3: the opponent model's archetype priors key off the REAL Archetype enum.

Ties the staged uppercase profile keys to `pokerbot.opponent.archetype.Archetype`
without duplicating the profiles, and confirms UNKNOWN falls back to uniform.
"""

from __future__ import annotations

from pokerbot.opponent.archetype import Archetype
from zoom.archetype_bridge import archetype_prior_key, set_opponent_archetype
from zoom.opponent_model import (
    _POSTFLOP_PROFILES,
    _PREFLOP_PROFILES,
    DirichletOpponentModel,
    coarse_key,
)


def test_profiles_correspond_exactly_to_the_real_enum():
    non_unknown = {a.name for a in Archetype} - {"UNKNOWN"}
    # Every classifiable archetype has a prior profile...
    assert non_unknown <= set(_PREFLOP_PROFILES)
    assert non_unknown <= set(_POSTFLOP_PROFILES)
    # ...and the profiles introduce no key that isn't a real archetype.
    real_names = {a.name for a in Archetype}
    assert set(_PREFLOP_PROFILES) <= real_names
    assert set(_POSTFLOP_PROFILES) <= real_names


def test_known_archetype_seeds_its_profile():
    m = DirichletOpponentModel()
    set_opponent_archetype(m, "v", Archetype.NIT)
    d = m.strategy("v", coarse_key("preflop", 2, 5, 1.0), "preflop")
    assert archetype_prior_key(Archetype.NIT) == "NIT"
    assert d["FOLD"] > 0.6  # NIT folds the vast majority preflop — not uniform


def test_unknown_archetype_falls_back_to_uniform():
    m = DirichletOpponentModel()
    set_opponent_archetype(m, "v", Archetype.UNKNOWN)
    d = m.strategy("v", coarse_key("preflop", 2, 5, 1.0), "preflop")
    probs = list(d.values())
    assert max(probs) - min(probs) < 1e-9  # uniform over the preflop action set
