"""Component 2 — scripted NIT/STATION/TAG archetype agents (Approach-2, Stage 1).

Lock 2: the entire premise of the Stage-1 fine-tune is that the opponent pool
contains *real* tight-passive players — the v5 failure was a training distribution
that never did. So these tests are the hard gate. They measure each agent's stats
with the SAME definitions the live classifier uses (`opponent.stats` /
`opponent.archetype`) and assert the agent both lands in its signature band AND is
classified as its own archetype by the real `ArchetypeClassifier`. If an agent's
sampled stats don't match its signature, that is a FAIL to fix in the agent — the
bands are never softened to make a loose agent pass.

Red-first ordering: `test_nit_vpip_below_18` is the first failing test (the analog
of Component 1's deep-stack-illegal property) — it pins the core tight-passive
premise.

Stat definitions mirror `pokerbot.opponent.stats.OpponentStats` exactly:
  VPIP  per-hand: any non-FOLD preflop action that isn't a free (to_call==0)
        CHECK_CALL.
  PFR   per-hand: any aggressive preflop action (raise/all-in).
  AF    per-action, postflop only: (bets + raises) / calls, where
        bet = aggressive with to_call==0, raise = aggressive with to_call>0,
        call = CHECK_CALL with to_call>0.
  fold-to-cbet: fraction folding when facing a (post-flop) bet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import pytest
from zoom.abstraction_gate import legal_abstract_actions_gated
from zoom.agents import (
    AgentSpot,
    LagAgent,
    NitAgent,
    ScriptedAgent,
    StationAgent,
    TagAgent,
    build_archetype_pool,
    postflop_strength,
)

from pokerbot.abstraction import ActionType
from pokerbot.opponent.archetype import Archetype, ArchetypeClassifier
from pokerbot.opponent.stats import OpponentStats

_AGGRESSIVE = {
    ActionType.BET_33,
    ActionType.BET_66,
    ActionType.BET_100,
    ActionType.BET_150,
    ActionType.ALL_IN,
    ActionType.RAISE_2_5X,
    ActionType.RAISE_3_5X,
}

_BB = 2  # big blind in chips
_DEEP_STACK = 200  # 100bb — Stage 1 trains at 100bb fixed


# ─────────── spot samplers (deterministic, seeded) ───────────


def _deal(rng: random.Random, n: int, dead: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Draw `n` distinct card ints 0..51 avoiding `dead`."""
    out: list[int] = []
    pool = [c for c in range(52) if c not in dead]
    rng.shuffle(pool)
    out = pool[:n]
    return tuple(out)


def _sample_preflop_spot(rng: random.Random) -> AgentSpot:
    """A 100bb 3-max preflop decision: ~55% facing a 2.5bb open, ~45% first-in.

    The mix is fixed and documented so 'VPIP in band' is a meaningful, reproducible
    statement. Facing-a-raise spots dominate slightly (you face action more often
    than you open in 3-max).
    """
    hole = _deal(rng, 2)
    position = rng.randint(0, 2)
    if rng.random() < 0.55:  # facing a 2.5bb open
        pot = round(3.5 * _BB)  # blinds + opener's 2.5bb, roughly
        to_call = round(2.5 * _BB)
    else:  # first-in / BB option
        pot = 3  # 1.5bb of blinds
        # BB with the option faces no bet; everyone else must at least call the BB.
        to_call = 0 if position == 1 else _BB
    return AgentSpot(
        hole=hole,
        board=(),
        street="preflop",
        position=position,
        pot=pot,
        to_call=to_call,
        stack=_DEEP_STACK,
        min_raise=_BB,
    )


def _sample_postflop_spot(rng: random.Random) -> AgentSpot:
    """A 100bb 3-max postflop decision: ~50% checked-to, ~50% facing a 0.66-pot bet."""
    street = rng.choice(["flop", "turn", "river"])
    n_board = {"flop": 3, "turn": 4, "river": 5}[street]
    hole = _deal(rng, 2)
    board = _deal(rng, n_board, dead=hole)
    pot = rng.randint(8, 40)
    facing = rng.random() < 0.5
    to_call = round(0.66 * pot) if facing else 0
    return AgentSpot(
        hole=hole,
        board=board,
        street=street,  # type: ignore[arg-type]
        position=rng.randint(0, 2),
        pot=pot,
        to_call=to_call,
        stack=_DEEP_STACK,
        min_raise=_BB,
    )


# ─────────── stat measurement (mirrors opponent.stats) ───────────


@dataclass
class MeasuredStats:
    vpip: float
    pfr: float
    af: float
    fold_to_cbet: float
    as_opponent_stats: OpponentStats


def _measure(agent: ScriptedAgent, *, seed: int = 2026, n: int = 6000) -> MeasuredStats:
    rng = random.Random(seed)
    vpip_yes = pfr_yes = 0
    bets = raises = calls = 0
    faced_cbet = folds_to_cbet = 0

    for _ in range(n):
        spot = _sample_preflop_spot(rng)
        a = agent.action(spot, rng)
        voluntary = a.type != ActionType.FOLD and not (
            a.type == ActionType.CHECK_CALL and spot.to_call == 0
        )
        if voluntary:
            vpip_yes += 1
        if a.type in _AGGRESSIVE:
            pfr_yes += 1

    for _ in range(n):
        spot = _sample_postflop_spot(rng)
        a = agent.action(spot, rng)
        if a.type in _AGGRESSIVE:
            if spot.to_call > 0:
                raises += 1
            else:
                bets += 1
        elif a.type == ActionType.CHECK_CALL and spot.to_call > 0:
            calls += 1
        if spot.to_call > 0:  # treat every facing-a-bet postflop spot as a c-bet faced
            faced_cbet += 1
            if a.type == ActionType.FOLD:
                folds_to_cbet += 1

    stats = OpponentStats(
        hands_observed=n,
        preflop_voluntary_actions=vpip_yes,
        preflop_raises=pfr_yes,
        postflop_bets=bets,
        postflop_raises=raises,
        postflop_calls=calls,
        faced_cbet=faced_cbet,
        folds_to_cbet=folds_to_cbet,
    )
    return MeasuredStats(
        vpip=stats.vpip(),
        pfr=stats.pfr(),
        af=stats.af(),
        fold_to_cbet=stats.fold_to_cbet(),
        as_opponent_stats=stats,
    )


# ─────────── the first red test: NIT is genuinely tight ───────────


def test_nit_vpip_below_18() -> None:
    """The core tight-passive premise: a NIT voluntarily enters <18% of hands."""
    stats = _measure(NitAgent())
    assert stats.vpip < 0.18, f"NIT VPIP {stats.vpip:.3f} not < 0.18 — agent is too loose"


# ─────────── full archetype signatures (classify-as-self) ───────────


def test_nit_full_signature_classifies_as_nit() -> None:
    stats = _measure(NitAgent())
    assert stats.vpip < 0.18, stats.vpip
    assert stats.pfr < 0.12, stats.pfr
    assert stats.af > 1.5, stats.af
    assert ArchetypeClassifier().classify(stats.as_opponent_stats) == Archetype.NIT


def test_tag_full_signature_classifies_as_tag() -> None:
    stats = _measure(TagAgent())
    assert 0.18 <= stats.vpip <= 0.26, stats.vpip
    assert 0.14 <= stats.pfr <= 0.22, stats.pfr
    assert stats.af > 2.0, stats.af
    assert ArchetypeClassifier().classify(stats.as_opponent_stats) == Archetype.TAG


def test_station_full_signature_classifies_as_station() -> None:
    stats = _measure(StationAgent())
    assert stats.vpip > 0.35, stats.vpip
    assert stats.af < 1.0, stats.af
    assert stats.fold_to_cbet < 0.40, stats.fold_to_cbet
    assert ArchetypeClassifier().classify(stats.as_opponent_stats) == Archetype.STATION


def test_lag_full_signature_classifies_as_lag() -> None:
    """The aggressive archetype the pool was missing: loose-aggressive, lands in the
    classifier's LAG band (VPIP 26-40%, PFR 22-35%, AF > 2.5)."""
    stats = _measure(LagAgent())
    assert 0.26 <= stats.vpip <= 0.40, stats.vpip
    assert 0.22 <= stats.pfr <= 0.35, stats.pfr
    assert stats.af > 2.5, stats.af
    assert ArchetypeClassifier().classify(stats.as_opponent_stats) == Archetype.LAG


# ─────────── STATION: the two halves that define a station ───────────


def test_station_never_folds_a_bet_it_is_priced_into() -> None:
    """A station calls whenever it has the odds: for any postflop spot where the
    station's own equity estimate ≥ pot odds, it must NOT fold. A station that
    folds a priced-in bet is not a station.
    """
    agent = StationAgent()
    rng = random.Random(7)
    checked = 0
    for _ in range(4000):
        spot = _sample_postflop_spot(rng)
        if spot.to_call <= 0:
            continue
        pot_odds = spot.to_call / (spot.pot + spot.to_call)
        if postflop_strength(spot.hole, spot.board) >= pot_odds:
            checked += 1
            a = agent.action(spot, rng)
            assert a.type != ActionType.FOLD, (spot, pot_odds)
    assert checked > 100, "sampler produced too few priced-in spots to be meaningful"


def test_station_never_bluff_shoves() -> None:
    """A station never makes an aggressive action (raise/all-in) without a strong
    made hand — no bluffs, preflop or postflop. A station that bluffs is not a
    station.
    """
    agent = StationAgent()
    rng = random.Random(11)
    for _ in range(5000):
        spot = _sample_preflop_spot(rng) if rng.random() < 0.5 else _sample_postflop_spot(rng)
        a = agent.action(spot, rng)
        if a.type in _AGGRESSIVE:
            # aggression is only ever with a premium/strong hand, never a bluff.
            if spot.street == "preflop":
                from pokerbot.abstraction import canonical_hand
                from pokerbot.runtime.default_policy import preflop_percentile

                canon, _ = canonical_hand(spot.hole, ())
                assert preflop_percentile(canon) < 0.10, ("preflop bluff-shove", spot)
            else:
                assert postflop_strength(spot.hole, spot.board) >= 0.80, ("postflop bluff", spot)


# ─────────── TAG is tight, not loose-aggressive ───────────


def test_tag_is_tight_not_loose() -> None:
    """TAG's range is genuinely tight: its VPIP sits below the LAG floor (26%) and
    well below the STATION's, so it is tight-aggressive, not loose wearing a label.
    """
    tag = _measure(TagAgent())
    station = _measure(StationAgent())
    assert tag.vpip < 0.26
    assert tag.vpip < station.vpip


def test_relative_tightness_ordering() -> None:
    """NIT tightest, then TAG, then STATION loosest — the pool spans the spectrum."""
    nit = _measure(NitAgent()).vpip
    tag = _measure(TagAgent()).vpip
    station = _measure(StationAgent()).vpip
    assert nit < tag < station


# ─────────── Lock-2 wiring: legality comes from the Component 1 gate ───────────


def test_agents_only_emit_gated_legal_actions() -> None:
    """Every agent action is a member of the Component 1 gated legal set — agents
    never re-derive legality. This is what makes their short-stack behavior inherit
    the SPR continuum for free.
    """
    rng = random.Random(99)
    agents = build_archetype_pool()
    for _ in range(3000):
        base_spot = _sample_preflop_spot(rng) if rng.random() < 0.5 else _sample_postflop_spot(rng)
        # vary stack depth too, so the gate's continuum is exercised.
        spot = replace(base_spot, stack=rng.choice([8, 20, 60, 200]))
        for agent in agents:
            a = agent.action(spot, rng)
            legal = legal_abstract_actions_gated(
                spot.pot, spot.to_call, spot.stack, spot.min_raise, spot.street
            )
            legal_types = {x.type for x in legal}
            assert a.type in legal_types, (agent.archetype, spot, a.type, legal_types)


def test_short_stack_premium_shoves_via_gate() -> None:
    """At a stack so short that the gate drops every raise size (forced jam), an
    agent that wants to raise a premium emits ALL_IN — legality from the gate, not
    a hardcoded shove rule.
    """
    # pot huge vs stack so all RAISE/BET sizes are dropped; base = FOLD/CHECK_CALL/ALL_IN.
    spot = AgentSpot(
        hole=(48, 49),  # premium pair (aces)
        board=(),
        street="preflop",
        position=2,
        pot=1000,
        to_call=2,
        stack=5,
        min_raise=2,
    )
    legal = legal_abstract_actions_gated(
        spot.pot, spot.to_call, spot.stack, spot.min_raise, "preflop"
    )
    assert {a.type for a in legal} == {ActionType.FOLD, ActionType.CHECK_CALL, ActionType.ALL_IN}
    a = NitAgent().action(spot, random.Random(0))
    assert a.type == ActionType.ALL_IN


# ─────────── pool + determinism ───────────


def test_build_archetype_pool_includes_weighted_aggressive_archetype() -> None:
    from collections import Counter

    pool = build_archetype_pool()
    arches = {a.archetype for a in pool}
    assert arches == {Archetype.NIT, Archetype.TAG, Archetype.STATION, Archetype.LAG}
    # LAG weighted up; STATION (the passivity-reward) no heavier than LAG.
    counts = Counter(type(a).__name__ for a in pool)
    assert counts["LagAgent"] >= 2, counts
    assert counts["StationAgent"] <= counts["LagAgent"], counts


@pytest.mark.parametrize("agent", build_archetype_pool())
def test_agents_are_deterministic(agent: ScriptedAgent) -> None:
    """Same spot → same action (scripted agents are pure functions of the spot)."""
    rng = random.Random(3)
    spot = _sample_postflop_spot(rng)
    first = agent.action(spot, random.Random(0))
    for _ in range(20):
        again = agent.action(spot, random.Random(0))
        assert again.type == first.type and again.amount_chips == first.amount_chips
