"""Smoke test for the L5 advisory service core (zoom.service.ZoomExploiterService).

Drives the service directly (no socket) with synthetic observations:
  * renders "BOT SAYS: ..." and is stateful (cursor avoids double-counting);
  * cross-hand L2 learning from showdown reveals shifts the opponent model;
  * a learned (bucket-conditional) model concentrates the per-hand range, where
    an untaught uniform model leaves it untouched.

Uses an in-memory StrategyDB so the blueprint path runs without trained data
(it falls through to the default policy) — the focus here is the stateful
plumbing, not the blueprint quality.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_runtime_deps  # pydantic / runtime schema

from pokerbot.abstraction import AbstractionTables, parse_card
from pokerbot.abstraction.encoding import position_from_seats
from pokerbot.runtime.adapter import RuntimeAdapter
from pokerbot.strategy_db import open_db
from zoom.opponent_model import DirichletOpponentModel, coarse_key
from zoom.service import ZoomExploiterService
from zoom.zoom_schema import PLACEHOLDER_OPPONENT_ID, ZoomObservation

OPP = PLACEHOLDER_OPPONENT_ID
RAISE_HIST = [{"seat": 1, "street": 0, "type": "raise", "amount": 25}]


def _service() -> ZoomExploiterService:
    db = open_db("sqlite:///:memory:")
    db.set_current_version(1)
    adapter = RuntimeAdapter(db=db, abstraction=AbstractionTables(), opponent_model=None, rng_seed=0)
    return ZoomExploiterService(adapter)


def _obs(seq, hero_hole, *, board=(), history=None, revealed=None, opp_id=OPP):
    return ZoomObservation.model_validate(
        {
            "seq": seq,
            "opponent_id": opp_id,
            "revealed_holes": revealed,
            "request": {
                "schema_version": 1,
                "game_type": "cash",
                "table_size": 3,
                "blinds": {"sb": 5, "bb": 10},
                "hero_seat": 0,
                "button_seat": 0,
                "hero_hole": list(hero_hole),
                "board": list(board),
                "stacks": [1000, 1000, 1000],
                "current_bets": [0, 0, 0],
                "pot_committed": 0,
                "to_call": 25,
                "min_raise": 50,
                "max_raise": 1000,
                "action_history": history if history is not None else [],
            },
        }
    )


def test_advise_renders_and_does_not_double_count():
    svc = _service()
    a1 = svc.advise(_obs(1, ("As", "Ks"), history=RAISE_HIST))
    assert a1.advice.startswith("BOT SAYS:")
    assert a1.action in ("fold", "check", "call", "bet", "raise")
    assert a1.opponent_id == OPP
    # Same hand, same history replayed → cursor consumed all entries already, so
    # no further range updates (effective combos unchanged).
    a2 = svc.advise(_obs(2, ("As", "Ks"), history=RAISE_HIST))
    assert svc._hist_cursor[OPP] == len(RAISE_HIST)
    assert a2.range_effective_combos == a1.range_effective_combos


def test_cross_hand_learning_shifts_opponent_model():
    svc = _service()
    premiums = [("Ah", "Ad"), ("Kh", "Kd"), ("Qh", "Qd"), ("Ah", "Kh"), ("As", "Ks")]
    heroes = [("2c", "3d"), ("2h", "4s"), ("5c", "6d"), ("7h", "8s"), ("9c", "Td")]
    rounds = 6
    for r in range(rounds):
        for i, vill in enumerate(premiums):
            # distinct hero cards per (round, hand) so each is detected as a new hand
            hero = (heroes[i][0], heroes[(i + r) % len(heroes)][1])
            svc.advise(_obs(100 + r * 10 + i, hero, history=RAISE_HIST, revealed={1: list(vill)}))

    assert svc._obs_count[OPP] >= rounds * len(premiums)

    # Model now favors RAISE at the (premium) bucket the villain kept raising.
    villain_pos = position_from_seats(0, 1, 3)  # seat 1 = SB → position 0
    aa_bucket = svc.bucket_fn((parse_card("Ah"), parse_card("Ad")), ())
    key = coarse_key("preflop", villain_pos, aa_bucket, 1.0)
    learned = svc.model.strategy(OPP, key, "preflop")["RAISE_2_5X"]
    prior = DirichletOpponentModel().strategy("fresh", key, "preflop")["RAISE_2_5X"]
    assert learned > prior + 0.1, f"learned RAISE prob {learned:.3f} should exceed prior {prior:.3f}"


def test_learned_model_concentrates_range_untaught_does_not():
    taught = _service()
    premiums = [("Ah", "Ad"), ("Kh", "Kd"), ("Qh", "Qd"), ("Ah", "Kh"), ("As", "Ks")]
    heroes = [("2c", "3d"), ("2h", "4s"), ("5c", "6d"), ("7h", "8s"), ("9c", "Td")]
    for r in range(6):
        for i, vill in enumerate(premiums):
            hero = (heroes[i][0], heroes[(i + r) % len(heroes)][1])
            taught.advise(_obs(200 + r * 10 + i, hero, history=RAISE_HIST, revealed={1: list(vill)}))

    untaught = _service()
    final = _obs(999, ("2c", "2d"), history=RAISE_HIST)  # fresh hand, villain raises
    a_taught = taught.advise(final)
    a_untaught = untaught.advise(_obs(999, ("2c", "2d"), history=RAISE_HIST))

    # The untaught (uniform) model is bucket-independent → the raise carries no
    # info → range stays at the full per-hand prior. The taught model concentrates.
    assert a_taught.range_effective_combos < a_untaught.range_effective_combos
