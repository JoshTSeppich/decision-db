"""
Per-opponent Bayesian action model (layer 2 of the search-based zoom bot).

WHAT THIS IS
------------
For each re-identified opponent, maintain a Dirichlet posterior over their
action frequencies at each (coarsened) infoset. Seed the prior from the
project's archetype classifier; update conjugately as actions are observed:

    posterior_alpha(infoset) = prior_alpha(archetype, street)
                             + decayed_observed_counts(opponent, infoset)
    strategy = posterior_alpha / posterior_alpha.sum()   (street-legal masked)

This is the Bayes' Bluff design (Southey et al. 2005) the project's own
research doc specifies. Two things make it the right layer-2 here:

  * Fixed re-identifiable pool -> hundreds of hands per opponent accumulate,
    so the posterior moves well past the prior and pins each bot's actual
    leaks (not a 5-way bucket).
  * The archetype classifier you already ship becomes the *prior mean*, so a
    freshly-seen opponent still plays a sensible distribution before data,
    and the conjugate updates refine it per-opponent over the session.

KEYING (important)
------------------
The model keys on a *coarsened* infoset: (street, position, card_bucket,
facing_bucket). It deliberately does NOT include the 64-byte action-history
blob the full project infoset uses -- keying on full history would fragment
per-opponent data so badly the posterior never concentrates. Coarse keys
trade infoset resolution for sample concentration, which is the correct trade
on limited per-opponent hand counts.

DECOUPLING
----------
Card -> bucket mapping is injected as `bucket_fn(hole, board) -> int`. In
production you pass a callable backed by the project's `AbstractionTables`
(200 postflop buckets / preflop buckets). For testing we inject a synthetic
chen-based bucketer. The model itself has zero dependency on the frozen
`pokerbot` package.

ACTION SPACE
------------
Matches the project's `ActionType` names exactly (probe-confirmed):
preflop  = FOLD, CHECK_CALL, RAISE_2_5X, RAISE_3_5X, ALL_IN
postflop = FOLD, CHECK_CALL, BET_33, BET_66, BET_100, BET_150, ALL_IN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ── action vocabulary (project ActionType names) ─────────────────────────── #
PREFLOP_ACTIONS: tuple[str, ...] = (
    "FOLD",
    "CHECK_CALL",
    "RAISE_2_5X",
    "RAISE_3_5X",
    "ALL_IN",
)
POSTFLOP_ACTIONS: tuple[str, ...] = (
    "FOLD",
    "CHECK_CALL",
    "BET_33",
    "BET_66",
    "BET_100",
    "BET_150",
    "ALL_IN",
)


def actions_for_street(street: str) -> tuple[str, ...]:
    return PREFLOP_ACTIONS if street == "preflop" else POSTFLOP_ACTIONS


# ── archetype priors (illustrative; production fits these from data) ──────── #
# Each profile is a frequency distribution over that street's action set.
# These map 1:1 to the project's Archetype enum (nit/tag/lag/maniac/station).
_PREFLOP_PROFILES: dict[str, dict[str, float]] = {
    "NIT":     {"FOLD": .80, "CHECK_CALL": .08, "RAISE_2_5X": .09, "RAISE_3_5X": .02, "ALL_IN": .01},
    "TAG":     {"FOLD": .70, "CHECK_CALL": .08, "RAISE_2_5X": .15, "RAISE_3_5X": .05, "ALL_IN": .02},
    "LAG":     {"FOLD": .55, "CHECK_CALL": .10, "RAISE_2_5X": .22, "RAISE_3_5X": .10, "ALL_IN": .03},
    "MANIAC":  {"FOLD": .25, "CHECK_CALL": .12, "RAISE_2_5X": .33, "RAISE_3_5X": .22, "ALL_IN": .08},
    "STATION": {"FOLD": .30, "CHECK_CALL": .60, "RAISE_2_5X": .07, "RAISE_3_5X": .02, "ALL_IN": .01},
}
_POSTFLOP_PROFILES: dict[str, dict[str, float]] = {
    "NIT":     {"FOLD": .55, "CHECK_CALL": .30, "BET_33": .06, "BET_66": .05, "BET_100": .02, "BET_150": .01, "ALL_IN": .01},
    "TAG":     {"FOLD": .35, "CHECK_CALL": .33, "BET_33": .12, "BET_66": .12, "BET_100": .05, "BET_150": .02, "ALL_IN": .01},
    "LAG":     {"FOLD": .25, "CHECK_CALL": .30, "BET_33": .15, "BET_66": .16, "BET_100": .08, "BET_150": .04, "ALL_IN": .02},
    "MANIAC":  {"FOLD": .12, "CHECK_CALL": .20, "BET_33": .18, "BET_66": .22, "BET_100": .15, "BET_150": .09, "ALL_IN": .04},
    "STATION": {"FOLD": .10, "CHECK_CALL": .72, "BET_33": .08, "BET_66": .06, "BET_100": .02, "BET_150": .01, "ALL_IN": .01},
}

# Total pseudo-counts the prior is worth. Small so ~tens of real observations
# dominate (matches "20 hands to classify, 50-500 to concentrate").
PRIOR_STRENGTH: float = 8.0


def _prior_alpha(archetype: str, street: str) -> np.ndarray:
    acts = actions_for_street(street)
    table = _PREFLOP_PROFILES if street == "preflop" else _POSTFLOP_PROFILES
    prof = table.get(archetype.upper())
    if prof is None:  # UNKNOWN / unrecognized -> uniform prior
        return np.full(len(acts), PRIOR_STRENGTH / len(acts))
    return np.array([prof[a] for a in acts], dtype=float) * PRIOR_STRENGTH


# ── infoset keying ────────────────────────────────────────────────────────── #
InfosetKey = tuple[str, int, int, int]  # (street, position, card_bucket, facing_bucket)


def facing_bucket(to_call_bb: float) -> int:
    """Coarse facing context: 0 = can act for free, 1 = facing a bet."""
    return 1 if to_call_bb > 0 else 0


def coarse_key(street: str, position: int, card_bucket: int, to_call_bb: float) -> InfosetKey:
    return (street, position, card_bucket, facing_bucket(to_call_bb))


# ── the model ─────────────────────────────────────────────────────────────── #
@dataclass
class _OppState:
    archetype: str = "UNKNOWN"
    observed: dict[InfosetKey, np.ndarray] = field(default_factory=dict)


class DirichletOpponentModel:
    """Per-opponent Dirichlet action model.

        m = DirichletOpponentModel()
        m.set_archetype("villain_A", "NIT")          # from your classifier
        m.observe("villain_A", key, "RAISE_2_5X")    # per observed action
        dist = m.strategy("villain_A", key, street)  # posterior-mean dist

    `decay` (0<decay<=1) is applied to that opponent/infoset's observed counts
    on each observation, so recent play dominates for adaptive opponents. Set
    decay=1.0 for stationary opponents (no forgetting).
    """

    def __init__(self, decay: float = 1.0) -> None:
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self.decay = decay
        self._opps: dict[str, _OppState] = {}

    def _state(self, opp_id: str) -> _OppState:
        if opp_id not in self._opps:
            self._opps[opp_id] = _OppState()
        return self._opps[opp_id]

    def set_archetype(self, opp_id: str, archetype: str) -> None:
        """Set the prior archetype for an opponent (call when your classifier
        fires / updates). Changing it re-seeds the prior; observed counts are
        kept, so data already gathered is not lost."""
        self._state(opp_id).archetype = archetype

    def observe(self, opp_id: str, key: InfosetKey, action: str) -> None:
        st = self._state(opp_id)
        acts = actions_for_street(key[0])
        if action not in acts:
            return  # action illegal for this street; ignore defensively
        vec = st.observed.get(key)
        if vec is None:
            vec = np.zeros(len(acts))
        else:
            vec = vec * self.decay
        vec[acts.index(action)] += 1.0
        st.observed[key] = vec

    def strategy(self, opp_id: str, key: InfosetKey, street: str) -> dict[str, float]:
        """Posterior-mean action distribution at `key` (street-legal)."""
        acts = actions_for_street(street)
        st = self._state(opp_id)
        alpha = _prior_alpha(st.archetype, street)
        obs = st.observed.get(key)
        if obs is not None and obs.shape == alpha.shape:
            alpha = alpha + obs
        p = alpha / alpha.sum()
        return {a: float(pi) for a, pi in zip(acts, p)}

    def n_observations(self, opp_id: str, key: InfosetKey) -> float:
        obs = self._state(opp_id).observed.get(key)
        return float(obs.sum()) if obs is not None else 0.0


# ── adapter to the range tracker's OpponentStrategy seam ──────────────────── #
# bucket_fn maps (hole_cards, board) -> card_bucket. Inject the project's
# AbstractionTables.lookup in production; a synthetic bucketer in tests.
BucketFn = Callable[[tuple[int, int], tuple[int, ...]], int]


def make_opponent_strategy(
    opp_id: str,
    model: DirichletOpponentModel,
    bucket_fn: BucketFn,
):
    """Produce the `opponent_strategy(hole, public_state) -> {action: prob}`
    callable that `RangeTracker.update` consumes. Closes over one opponent."""

    def strat(hole: tuple[int, int], public_state) -> dict[str, float]:  # noqa: ANN001
        bucket = bucket_fn(hole, tuple(public_state.board))
        key = coarse_key(
            public_state.street, public_state.position, bucket, public_state.to_call_bb
        )
        return model.strategy(opp_id, key, public_state.street)

    return strat


__all__ = [
    "PREFLOP_ACTIONS",
    "POSTFLOP_ACTIONS",
    "PRIOR_STRENGTH",
    "DirichletOpponentModel",
    "InfosetKey",
    "actions_for_street",
    "coarse_key",
    "facing_bucket",
    "make_opponent_strategy",
]
