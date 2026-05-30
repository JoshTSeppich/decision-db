"""Behavioral profiling of a policy + the ALL_IN-zeroed transform (Component 4).

`SpotPolicy` is the harness's policy form: a function from an `AgentSpot` to a
distribution over `ActionType`s — the same currency Components 2/3 use. The
behavioral profile is computed by driving the PRODUCTION tally loop
`scripts/eval_behavioral_profile.py::_play_one_hand_behavioral` (reused read-only):
a tiny `.decide(request)` adapter wraps the `SpotPolicy`, so VPIP / PFR /
fold-to-c-bet come out of `SideTotals` with exactly the production definitions —
no re-implemented tally that could drift from what the gate measures (the v5
"measured the wrong thing" failure).

The one metric the production `SideTotals` doesn't isolate — preflop ALL_IN% — is
tracked in the adapter (it sees the board length and the chosen action), keyed per
(hand, seat) to match the per-hand-seat band semantics.

`zero_all_in_at_deep_stacks` builds the should-be-better reference policy: it
removes ALL_IN at deep SPR and renormalizes — the "iter_400 with ALL_IN zeroed at
deep stacks" construction the acceptance test uses.
"""

from __future__ import annotations

import dataclasses
import random
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from pokerbot.abstraction import ActionType, parse_card
from pokerbot.abstraction.actions import legal_abstract_actions
from pokerbot.abstraction.encoding import effective_stack, position_from_seats
from pokerbot.runtime.adapter import _build_history_bytes
from zoom.abstraction_gate import DEFAULT_SPR_CAP, effective_spr

# Reuse the production profiler read-only. It lives in scripts/, which the project
# treats as importable (mypy_path includes it; the eval tests add it to sys.path the
# same way). Re-implementing its tally would risk the metrics drifting from what the
# gate measures — the exact v5 "measured the wrong thing" failure.
_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from eval_behavioral_profile import (  # noqa: E402  (needs the sys.path insert above)
    SideTotals,
    _play_one_hand_behavioral,
)

if TYPE_CHECKING:
    from pokerbot.runtime.schema import GameStateRequest
    from pokerbot.training import SimpleNLHEGame
    from zoom.agents import AgentSpot

# A policy: AgentSpot → distribution over action types (need not sum to 1; the
# adapter restricts to the dist's positive-mass actions and the loop clamps to legal).
SpotPolicy = Callable[["AgentSpot"], "Mapping[ActionType, float]"]

_STREETS: Final[tuple[str, str, str, str]] = ("preflop", "flop", "turn", "river")
_BOARD_LEN_TO_STREET_IDX: Final[dict[int, int]] = {0: 0, 3: 1, 4: 2, 5: 3}

# Aggressive actions OTHER than ALL_IN. If any is legal, ALL_IN is a discretionary choice
# (there was a real raise/bet alternative); if none is, ALL_IN is the ONLY legal aggression
# (pot-committed forced jam) and shoving is structurally forced, not a leak.
_NONALLIN_AGGRESSION: Final[frozenset[ActionType]] = frozenset(
    {
        ActionType.RAISE_2_5X,
        ActionType.RAISE_3_5X,
        ActionType.BET_33,
        ActionType.BET_66,
        ActionType.BET_100,
        ActionType.BET_150,
    }
)


def _preflop_forced_jam(pot: int, to_call: int, stack: int, min_raise: int) -> bool:
    """True iff ALL_IN is the ONLY legal aggression preflop (pot-committed: the stack is
    too short for any non-ALL_IN raise). Such shoves are structurally forced — folding is
    clearly -EV — so they are NOT the indiscriminate-shoving leak the band targets, and are
    excluded from the gated non-committed metric (raw ALL_IN% still reported)."""
    base = legal_abstract_actions(pot, to_call, stack, min_raise, "preflop")
    return not any(a.type in _NONALLIN_AGGRESSION for a in base)


@dataclass(frozen=True)
class BehavioralProfile:
    """The four gated metrics, read from the production `SideTotals` (+ preflop ALL_IN)."""

    totals: SideTotals
    preflop_all_in_count: int  # raw: all preflop shoves (reported, NOT gated)
    hand_seats: int
    noncommitted_all_in_count: int = 0  # preflop shoves that had a non-ALL_IN alternative

    @property
    def vpip_pct(self) -> float:
        return self.totals.vpip_rate() * 100.0

    @property
    def pfr_pct(self) -> float:
        return self.totals.pfr_rate() * 100.0

    @property
    def fold_to_cbet_pct(self) -> float:
        return self.totals.fold_to_cbet_rate() * 100.0

    @property
    def all_in_preflop_pct(self) -> float:
        """Raw preflop ALL_IN rate over all hand-seats (reported for transparency, NOT gated)."""
        return (self.preflop_all_in_count / self.hand_seats * 100.0) if self.hand_seats else 0.0

    @property
    def noncommitted_all_in_preflop_pct(self) -> float:
        """Preflop shoves at spots that HAD a non-ALL_IN alternative, over all hand-seats —
        the discretionary/indiscriminate-shoving leak. This is the GATED ALL_IN metric;
        structurally-forced (pot-committed) jams are excluded."""
        return (self.noncommitted_all_in_count / self.hand_seats * 100.0) if self.hand_seats else 0.0

    @classmethod
    def from_pcts(
        cls,
        *,
        vpip: float,
        pfr: float,
        all_in_preflop: float,
        fold_to_cbet: float,
        noncommitted_all_in: float = 0.0,
        hands: int = 100,
    ) -> BehavioralProfile:
        """Construct a profile with exactly the given pcts (for band unit tests)."""
        totals = SideTotals(
            hand_seats=hands,
            vpip_yes=round(vpip / 100.0 * hands),
            pfr_yes=round(pfr / 100.0 * hands),
            facing_cbet=hands,
            fold_to_cbet=round(fold_to_cbet / 100.0 * hands),
        )
        return cls(
            totals=totals,
            preflop_all_in_count=round(all_in_preflop / 100.0 * hands),
            hand_seats=hands,
            noncommitted_all_in_count=round(noncommitted_all_in / 100.0 * hands),
        )


@dataclass(frozen=True)
class _Response:
    abstract_action: str  # the only field `_play_one_hand_behavioral` reads


def _spot_from_request(request: GameStateRequest) -> AgentSpot:
    from zoom.agents import AgentSpot

    seat = request.hero_seat
    hole = [parse_card(c) for c in request.hero_hole]
    board = tuple(parse_card(c) for c in request.board)
    street = _STREETS[_BOARD_LEN_TO_STREET_IDX[len(request.board)]]
    position = position_from_seats(request.button_seat, seat, request.table_size)
    # Effective stack = min(hero, max remaining opp), matching how the export keys rows
    # (nlhe_game.py:388). Without this the chip leader buckets by its raw stack and misses.
    opps = [request.stacks[s] for s in range(request.table_size) if s != seat and request.stacks[s] > 0]
    eff = effective_stack(request.stacks[seat], opps)
    # Real §C betting history via the SAME encoder production uses (adapter._build_history_bytes),
    # so the DB lookup tries the exact history-aware row first, exactly like the deployed
    # RuntimeAdapter. Without this the gate reads history-blind (non-representative) rows.
    history = _build_history_bytes(request)
    return AgentSpot(
        hole=(hole[0], hole[1]),
        board=board,
        street=street,  # type: ignore[arg-type]
        position=position,
        pot=request.pot_committed,
        to_call=request.to_call,
        stack=request.stacks[seat],
        min_raise=request.min_raise,
        table_size=request.table_size,
        effective_stack=eff,
        history=history,
    )


class _SpotPolicyAdapter:
    """Duck-typed `.decide(request)` so a `SpotPolicy` plugs into the production loop.

    Also records preflop ALL_IN per (hand, seat) for the all_in_preflop_pct metric.
    `current_hand` is set by the profiler before each hand.
    """

    def __init__(self, spot_policy: SpotPolicy, *, seed: int) -> None:
        self._policy = spot_policy
        self._rng = random.Random(seed)
        self.current_hand = 0
        self.preflop_shoves: set[tuple[int, int]] = set()  # ALL preflop shoves (raw)
        # Preflop shoves at NON-committed spots (a non-ALL_IN raise was legal) — the
        # discretionary-shoving leak the band gates on; forced jams are excluded here.
        self.preflop_noncommitted_shoves: set[tuple[int, int]] = set()

    def decide(self, request: GameStateRequest) -> _Response:
        chosen = self._sample(self._policy(_spot_from_request(request)))
        if len(request.board) == 0 and chosen == ActionType.ALL_IN:
            key = (self.current_hand, request.hero_seat)
            self.preflop_shoves.add(key)
            stack = request.stacks[request.hero_seat]
            if not _preflop_forced_jam(request.pot_committed, request.to_call, stack, request.min_raise):
                self.preflop_noncommitted_shoves.add(key)
        return _Response(abstract_action=chosen.name)

    def _sample(self, dist: Mapping[ActionType, float]) -> ActionType:
        items = [(a, p) for a, p in dist.items() if p > 0]
        if not items:
            return ActionType.CHECK_CALL
        types, weights = zip(*items, strict=True)
        return cast("ActionType", self._rng.choices(list(types), weights=list(weights), k=1)[0])


def _merge(a: SideTotals, b: SideTotals) -> SideTotals:
    merged = SideTotals()
    for f in dataclasses.fields(SideTotals):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, list):
            setattr(merged, f.name, [x + y for x, y in zip(va, vb, strict=True)])
        else:
            setattr(merged, f.name, va + vb)
    return merged


def profile_spot_policy(
    spot_policy: SpotPolicy,
    game: SimpleNLHEGame,
    *,
    n_hands: int,
    seed: int,
) -> BehavioralProfile:
    """Behavioral profile of `spot_policy` via self-play on `game` (production tally)."""
    adapter = _SpotPolicyAdapter(spot_policy, seed=seed)
    trained, default = SideTotals(), SideTotals()
    rng = random.Random(seed)
    for i in range(n_hands):
        adapter.current_hand = i
        # Same policy in every seat (self-play); _play_one_hand_behavioral splits the
        # tally by seat parity, so we merge both sides back into one profile.
        _play_one_hand_behavioral(i, game, adapter, adapter, rng, trained, default)  # type: ignore[arg-type]
    merged = _merge(trained, default)
    return BehavioralProfile(
        totals=merged,
        preflop_all_in_count=len(adapter.preflop_shoves),
        hand_seats=merged.hand_seats,
        noncommitted_all_in_count=len(adapter.preflop_noncommitted_shoves),
    )


def zero_all_in_at_deep_stacks(
    spot_policy: SpotPolicy, *, spr_cap: float = DEFAULT_SPR_CAP
) -> SpotPolicy:
    """Return `spot_policy` with ALL_IN removed at deep SPR and the rest renormalized.

    The should-be-better reference construction: deep-stack shoves (the v5
    over-aggression) are zeroed and their mass redistributed; shallow spots (low
    SPR, where a shove is legitimate) are untouched.
    """

    def transformed(spot: AgentSpot) -> Mapping[ActionType, float]:
        dist = dict(spot_policy(spot))
        is_deep = effective_spr(spot.pot, spot.to_call, spot.stack) > spr_cap
        if is_deep and dist.get(ActionType.ALL_IN, 0.0) > 0.0:
            del dist[ActionType.ALL_IN]
            total = sum(p for p in dist.values() if p > 0)
            if total > 0:
                return {a: p / total for a, p in dist.items() if p > 0}
            return {ActionType.CHECK_CALL: 1.0}
        return dist

    return transformed


__all__ = [
    "BehavioralProfile",
    "SpotPolicy",
    "profile_spot_policy",
    "zero_all_in_at_deep_stacks",
]
