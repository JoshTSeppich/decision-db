"""
Bayesian range tracker for the search-based zoom bot.

WHAT THIS IS
------------
Real-time subgame search needs the villains' *ranges* -- a probability
distribution over what hole cards they hold given the actions they've taken --
not just their action frequencies. This module maintains that distribution and
updates it by Bayes' rule as each villain action is observed.

    posterior(combo | action) ∝ P(action | combo, public_state) · prior(combo)

The likelihood P(action | combo, public_state) is supplied by an injected
callable -- the *opponent model*. This is the seam that keeps the tracker
decoupled from the rest of the system:

    opponent_strategy(hole_cards, public_state) -> {action_label: probability}

In production you pass a callable backed by the per-opponent Dirichlet model
(seeded by the archetype classifier) composed with the project's infoset
abstraction. For testing here we pass synthetic strategies with known shape and
verify the posterior concentrates the way Bayes says it must.

This file deliberately has NO dependency on the frozen `pokerbot` package, the
Pydantic schema, or the abstraction encoding. It needs only: a card
representation, an action history, and the opponent-model callable. That is why
it can be built and proven in isolation, ahead of the integration work.

CARD ENCODING
-------------
Cards are ints 0..51.  rank = card // 4  (0='2' .. 12='A');  suit = card % 4
('c','d','h','s').  A combo is a sorted 2-tuple of distinct card ints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"


# --------------------------------------------------------------------------- #
# Card / combo utilities
# --------------------------------------------------------------------------- #
def make_card(rank: int, suit: int) -> int:
    return rank * 4 + suit


def card_rank(card: int) -> int:
    return card // 4


def card_suit(card: int) -> int:
    return card % 4


def parse_card(s: str) -> int:
    """'As' -> int.  Rank char then suit char."""
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"bad card {s!r}")
    r = RANKS.index(s[0].upper())
    su = SUITS.index(s[1].lower())
    return make_card(r, su)


def card_str(card: int) -> str:
    return f"{RANKS[card_rank(card)]}{SUITS[card_suit(card)]}"


def parse_cards(s: str) -> list[int]:
    """'As Kd' or 'AsKd' -> [int, int]."""
    s = s.replace(" ", "")
    return [parse_card(s[i : i + 2]) for i in range(0, len(s), 2)]


def combo_str(combo: tuple[int, int]) -> str:
    return f"{card_str(combo[0])}{card_str(combo[1])}"


# --------------------------------------------------------------------------- #
# Chen-formula preflop strength.
# Used ONLY by (a) the synthetic opponents in the tests and (b) range-strength
# reporting. Production reporting should use real equity (pokerkit). The Bayes
# filter itself does not depend on this function at all.
# --------------------------------------------------------------------------- #
def chen_score(combo: tuple[int, int]) -> float:
    r1, r2 = card_rank(combo[0]), card_rank(combo[1])
    s1, s2 = card_suit(combo[0]), card_suit(combo[1])
    hi, lo = max(r1, r2), min(r1, r2)

    def base(rank: int) -> float:
        # rank 12='A'..0='2'.  A=10,K=8,Q=7,J=6, T..2 = (rank+2)/2
        if rank == 12:
            return 10.0
        if rank == 11:
            return 8.0
        if rank == 10:
            return 7.0
        if rank == 9:
            return 6.0
        return (rank + 2) / 2.0  # T(rank8)->5 ... 2(rank0)->1

    if r1 == r2:  # pair
        score = max(base(hi) * 2.0, 5.0)
    else:
        score = base(hi)
        if s1 == s2:
            score += 2.0  # suited
        gap = hi - lo - 1
        if gap == 1:
            score -= 1.0
        elif gap == 2:
            score -= 2.0
        elif gap == 3:
            score -= 4.0
        elif gap >= 4:
            score -= 5.0
        # straight bonus: 0/1-gap and both below Q
        if gap <= 1 and hi < 10:
            score += 1.0
    return float(np.ceil(score))


# --------------------------------------------------------------------------- #
# Public state -- opaque to the tracker; forwarded verbatim to the opponent
# model. Holds whatever the model needs (board, street, betting context...).
# Kept minimal here; the production version is the project's accumulator view.
# --------------------------------------------------------------------------- #
@dataclass
class PublicState:
    board: tuple[int, ...] = ()
    street: str = "preflop"          # preflop|flop|turn|river
    pot_bb: float = 1.5
    to_call_bb: float = 0.0
    facing: str = "none"             # none|bet|raise -- what villain faced
    position: str = "unknown"        # villain's position
    extra: dict = field(default_factory=dict)


# Opponent model contract:
#   (hole_cards: tuple[int,int], public_state: PublicState) -> dict[str,float]
# A mapping action_label -> probability. Need not be normalized; the tracker
# only reads the probability of the *observed* action.
OpponentStrategy = Callable[[tuple[int, int], PublicState], dict[str, float]]


# --------------------------------------------------------------------------- #
# The tracker
# --------------------------------------------------------------------------- #
class RangeTracker:
    """
    Maintains a posterior over a single villain's hole-card combo.

    Lifecycle per hand:
        rt = RangeTracker(dead_cards=hero_hole + initial_board)
        rt.update("raise", public_state, opp_model)   # each villain action
        rt.reveal_board([flop1, flop2, flop3])          # as streets come
        ...
        rt.posterior()        # {combo: prob}
        rt.range_strength()   # expected Chen score under posterior
    """

    def __init__(self, dead_cards: Iterable[int] = ()):
        self._dead: set[int] = set(dead_cards)
        self._combos: list[tuple[int, int]] = []
        self._logp: np.ndarray = np.array([])  # log-probabilities, unnormalized
        self._rebuild_support()

    # -- support management --------------------------------------------------
    def _rebuild_support(self) -> None:
        """(Re)build the combo list excluding dead cards, preserving current
        beliefs for combos that survive."""
        old = self.posterior() if self._combos else {}
        live = [c for c in range(52) if c not in self._dead]
        self._combos = list(combinations(live, 2))
        if old:
            # carry forward surviving mass; uniform for any new combo (none here)
            probs = np.array([old.get(c, 0.0) for c in self._combos], dtype=float)
            if probs.sum() <= 0:
                probs = np.ones(len(self._combos))
        else:
            probs = np.ones(len(self._combos))
        probs = probs / probs.sum()
        self._logp = np.log(probs)

    def add_dead_cards(self, cards: Iterable[int]) -> None:
        new = {c for c in cards if c not in self._dead}
        if not new:
            return
        self._dead |= new
        self._rebuild_support()

    def reveal_board(self, board_cards: Iterable[int]) -> None:
        """Board cards are public and removed from every villain combo."""
        self.add_dead_cards(board_cards)

    # -- the Bayesian update -------------------------------------------------
    def update(
        self,
        action: str,
        public_state: PublicState,
        opponent_strategy: OpponentStrategy,
        eps: float = 1e-9,
    ) -> None:
        """
        Multiply each combo's belief by the likelihood of the observed action
        under the opponent model, then renormalize.

        posterior ∝ P(action | combo, state) · prior
        """
        like = np.empty(len(self._combos))
        for i, combo in enumerate(self._combos):
            dist = opponent_strategy(combo, public_state)
            total = sum(dist.values()) or 1.0
            p = dist.get(action, 0.0) / total
            like[i] = max(p, eps)  # floor avoids -inf and total collapse

        new_logp = self._logp + np.log(like)

        # Guard: if every likelihood floored to eps (model assigns ~0 mass to
        # this action for all hands), the action is unexplained by the model.
        # Keep the prior rather than corrupting the belief with noise.
        if not np.isfinite(new_logp).any() or (like <= eps).all():
            return

        self._logp = new_logp - new_logp.max()  # stabilize before exp

    # -- readouts ------------------------------------------------------------
    def posterior(self) -> dict[tuple[int, int], float]:
        if len(self._combos) == 0:
            return {}
        p = np.exp(self._logp)
        s = p.sum()
        if s <= 0:
            p = np.ones_like(p)
            s = p.sum()
        p = p / s
        return {c: float(pi) for c, pi in zip(self._combos, p)}

    def n_combos(self) -> int:
        return len(self._combos)

    def prob_mass(self, combos: Iterable[tuple[int, int]]) -> float:
        post = self.posterior()
        wanted = {tuple(sorted(c)) for c in combos}
        return sum(v for c, v in post.items() if c in wanted)

    def top_combos(self, k: int = 10) -> list[tuple[str, float]]:
        post = self.posterior()
        ordered = sorted(post.items(), key=lambda kv: kv[1], reverse=True)
        return [(combo_str(c), round(v, 4)) for c, v in ordered[:k]]

    def range_strength(self) -> float:
        """Expected Chen score under the posterior. Higher = stronger range.
        A clean single-number health check; production uses equity instead."""
        post = self.posterior()
        return float(sum(chen_score(c) * v for c, v in post.items()))

    def effective_combos(self) -> float:
        """Perplexity of the posterior: how many combos the range 'effectively'
        spans. Drops as the range tightens. exp(entropy)."""
        post = self.posterior()
        p = np.array(list(post.values()))
        p = p[p > 0]
        ent = -np.sum(p * np.log(p))
        return float(np.exp(ent))


# --------------------------------------------------------------------------- #
# Synthetic opponent models for testing (NOT production).
# These stand in for the Dirichlet-model-backed callable. Their job is to have
# a known, checkable shape so we can prove the filter updates correctly.
# --------------------------------------------------------------------------- #
def tight_value_opponent(strength_for_max_raise: float = 18.0) -> OpponentStrategy:
    """Raises proportionally to preflop strength; folds weak hands.
    Feeding 'raise' should pull the posterior toward strong starting hands."""

    def strat(hole: tuple[int, int], st: PublicState) -> dict[str, float]:
        sc = chen_score(hole)
        # logistic raise frequency in Chen score, centered ~12
        raise_p = 1.0 / (1.0 + np.exp(-(sc - 12.0)))
        return {"raise": float(raise_p), "fold": float(1.0 - raise_p)}

    return strat


def polarized_opponent() -> OpponentStrategy:
    """Bets with very strong OR very weak hands, checks the middle.

    Note on calibration: there are far more weak combos than premium ones, so
    to get a posterior with *comparable* mass in both peaks the per-combo bet
    frequency for air must be much lower than for nuts. We bet ~all nutted
    hands, only a thin slice of air (a realistic bluff frequency), and almost
    never the middle. Feeding 'bet' then yields two visible peaks with the
    middle as the smallest bucket."""

    def strat(hole: tuple[int, int], st: PublicState) -> dict[str, float]:
        sc = chen_score(hole)
        strong = 1.0 / (1.0 + np.exp(-3.0 * (sc - 15.0)))   # steep high gate
        air_gate = 1.0 / (1.0 + np.exp(3.0 * (sc - 5.0)))   # steep low gate
        weak = 0.08 * air_gate                               # thin bluff slice
        bet_p = float(min(1.0, strong + weak))
        return {"bet": bet_p, "check": 1.0 - bet_p}

    return strat


def calling_station() -> OpponentStrategy:
    """Calls almost anything, rarely raises. Feeding 'call' barely moves the
    range -- the correct behavior when an action is nearly uninformative."""

    def strat(hole: tuple[int, int], st: PublicState) -> dict[str, float]:
        return {"call": 0.9, "fold": 0.05, "raise": 0.05}

    return strat
