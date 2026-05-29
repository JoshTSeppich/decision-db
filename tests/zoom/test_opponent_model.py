"""
Proof harness for opponent_model.py + its composition with range_tracker.py.
Runs as a plain script. Asserts the Bayesian behavior and prints it legibly.
"""

import numpy as np

from zoom.opponent_model import (
    DirichletOpponentModel,
    coarse_key,
    make_opponent_strategy,
)
from zoom.range_tracker import PublicState, RangeTracker, chen_score, parse_cards

results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")


# A synthetic abstraction: map a combo to a coarse strength bucket 0..10 by
# Chen score. Stands in for AbstractionTables.lookup in tests. Board ignored
# (these tests are preflop), which is fine for the seam check.
def chen_bucket(hole, board):  # noqa: ANN001, ARG001
    return int(min(10, max(0, round(chen_score(tuple(hole)) / 2))))


print("=" * 70)
print("DIRICHLET OPPONENT MODEL — proof of correctness")
print("=" * 70)

# --------------------------------------------------------------------------- #
print("\n[1] Archetype prior seeds the distribution before any data")
m = DirichletOpponentModel()
m.set_archetype("nit_guy", "NIT")
m.set_archetype("maniac_guy", "MANIAC")
key = coarse_key("preflop", position=2, card_bucket=5, to_call_bb=1.0)
nit_d = m.strategy("nit_guy", key, "preflop")
man_d = m.strategy("maniac_guy", key, "preflop")
check("NIT prior folds most of the time preflop", nit_d["FOLD"] > 0.6,
      f"NIT FOLD={nit_d['FOLD']:.2f}")
check("MANIAC prior raises/jams far more than NIT",
      (man_d["RAISE_2_5X"] + man_d["RAISE_3_5X"] + man_d["ALL_IN"])
      > (nit_d["RAISE_2_5X"] + nit_d["RAISE_3_5X"] + nit_d["ALL_IN"]),
      f"MANIAC raise+jam={(man_d['RAISE_2_5X']+man_d['RAISE_3_5X']+man_d['ALL_IN']):.2f} "
      f"vs NIT={(nit_d['RAISE_2_5X']+nit_d['RAISE_3_5X']+nit_d['ALL_IN']):.2f}")
check("distribution is normalized", abs(sum(nit_d.values()) - 1.0) < 1e-9)

# --------------------------------------------------------------------------- #
print("\n[2] Conjugate updates converge to OBSERVED frequencies (prior washes out)")
# A 'NIT'-labeled opponent who is actually a tricky raiser at this infoset.
m = DirichletOpponentModel()
m.set_archetype("liar", "NIT")  # wrong prior on purpose
k = coarse_key("preflop", position=2, card_bucket=8, to_call_bb=1.0)
before = m.strategy("liar", k, "preflop")["RAISE_2_5X"]
for _ in range(300):
    m.observe("liar", k, "RAISE_2_5X")
after = m.strategy("liar", k, "preflop")["RAISE_2_5X"]
check("posterior overrides a wrong prior given enough data",
      before < 0.2 and after > 0.9,
      f"P(RAISE_2_5X): prior {before:.3f} -> posterior {after:.3f} after 300 obs")

# how fast does it cross 50%? (sample-efficiency sanity)
m2 = DirichletOpponentModel()
m2.set_archetype("liar2", "NIT")
crossed = None
for n in range(1, 200):
    m2.observe("liar2", k, "RAISE_2_5X")
    if m2.strategy("liar2", k, "preflop")["RAISE_2_5X"] > 0.5 and crossed is None:
        crossed = n
check("crosses 50% within a few dozen observations",
      crossed is not None and crossed <= 40,
      f"P(RAISE) exceeded 0.5 after {crossed} observations")

# --------------------------------------------------------------------------- #
print("\n[3] Decay lets the model track an opponent who CHANGES strategy")
m = DirichletOpponentModel(decay=0.95)
k = coarse_key("flop", position=1, card_bucket=5, to_call_bb=0.0)
for _ in range(120):  # phase 1: villain bets relentlessly
    m.observe("drifter", k, "BET_66")
mid = m.strategy("drifter", k, "flop")["BET_66"]
for _ in range(120):  # phase 2: villain switches to passive checking/calling
    m.observe("drifter", k, "CHECK_CALL")
end = m.strategy("drifter", k, "flop")
check("model follows the switch (BET_66 mass drops, CHECK_CALL rises)",
      mid > 0.7 and end["CHECK_CALL"] > end["BET_66"],
      f"BET_66 {mid:.2f} -> {end['BET_66']:.2f}; CHECK_CALL now {end['CHECK_CALL']:.2f}")

# contrast: with no decay, the same sequence stays stuck near 50/50
m_nodecay = DirichletOpponentModel(decay=1.0)
for _ in range(120):
    m_nodecay.observe("d2", k, "BET_66")
for _ in range(120):
    m_nodecay.observe("d2", k, "CHECK_CALL")
nd = m_nodecay.strategy("d2", k, "flop")
check("no-decay model is sluggish (BET_66 still substantial)",
      nd["BET_66"] > 0.35,
      f"no-decay BET_66={nd['BET_66']:.2f}, CHECK_CALL={nd['CHECK_CALL']:.2f}")

# --------------------------------------------------------------------------- #
print("\n[4] END-TO-END: opponent model -> range tracker concentrates the range")
# Build an opponent whose play is strength-dependent: raises strong buckets,
# folds weak ones. Feed those observations, then drive the range tracker with
# the model's strategy and confirm 'raise' tightens onto strong hands.
m = DirichletOpponentModel()
m.set_archetype("villain_A", "TAG")
# teach the model: high buckets -> raise, low buckets -> fold (preflop, facing)
for b in range(11):
    kb = coarse_key("preflop", position=2, card_bucket=b, to_call_bb=1.0)
    action = "RAISE_2_5X" if b >= 6 else "FOLD"
    for _ in range(60):
        m.observe("villain_A", kb, action)

opp_strat = make_opponent_strategy("villain_A", m, chen_bucket)

rt = RangeTracker()  # clean 1326-combo prior
st = PublicState(street="preflop", position=2, to_call_bb=1.0, facing="raise")
strength_before = rt.range_strength()
eff_before = rt.effective_combos()
rt.update("RAISE_2_5X", st, opp_strat)
strength_after = rt.range_strength()
eff_after = rt.effective_combos()

check("range strengthens after the modeled opponent raises",
      strength_after > strength_before,
      f"expected Chen {strength_before:.2f} -> {strength_after:.2f}")
check("range tightens (effective combos fall)",
      eff_after < eff_before,
      f"eff. combos {eff_before:.0f} -> {eff_after:.0f}")
print("        top of inferred range after the modeled raise:")
for label, p in rt.top_combos(8):
    print(f"          {label}   {p:.4f}")

# sanity: premium mass should now dominate trash mass
post = rt.posterior()
premium = sum(v for c, v in post.items() if chen_score(c) >= 14)
trash = sum(v for c, v in post.items() if chen_score(c) <= 4)
check("premium-hand mass now exceeds trash mass",
      premium > trash,
      f"premium={premium:.3f}  trash={trash:.3f}")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
total, passed = len(results), sum(results)
print(f"RESULT: {passed}/{total} checks passed")
print("=" * 70)
if passed != total:
    raise SystemExit(1)
