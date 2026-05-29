"""
Proof harness for range_tracker.py — runs as a plain script (no pytest needed).
Each check asserts the mathematically-required behavior and prints what happened
so the update is legible, not just green.
"""

from zoom.range_tracker import (
    PublicState,
    RangeTracker,
    calling_station,
    chen_score,
    combo_str,
    parse_cards,
    polarized_opponent,
    tight_value_opponent,
)
from itertools import combinations

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"{PASS if cond else FAIL}  {name}")
    if detail:
        print(f"        {detail}")


print("=" * 70)
print("RANGE TRACKER — proof of correctness")
print("=" * 70)

# --------------------------------------------------------------------------- #
print("\n[1] Prior is uniform over the right number of combos")
rt = RangeTracker()
n = rt.n_combos()
post = rt.posterior()
probs = list(post.values())
check("1326 combos with no dead cards", n == 1326, f"n_combos = {n}")
check(
    "prior is uniform (all probs equal)",
    max(probs) - min(probs) < 1e-12,
    f"p = {probs[0]:.3e} for every combo",
)
check("prior sums to 1", abs(sum(probs) - 1.0) < 1e-9, f"sum = {sum(probs):.6f}")

# --------------------------------------------------------------------------- #
print("\n[2] Blockers: hero's cards are removed from villain's range")
hero = parse_cards("As Ah")  # hero holds two aces
rt = RangeTracker(dead_cards=hero)
n = rt.n_combos()
# combos using As or Ah must be absent entirely
post = rt.posterior()
uses_hero = any((hero[0] in c) or (hero[1] in c) for c in post)
check("combo count drops by exactly the blocked combos",
      n == len(list(combinations([c for c in range(52) if c not in set(hero)], 2))),
      f"n_combos = {n} (was 1326)")
check("no surviving combo contains As or Ah", not uses_hero,
      "villain cannot hold a card hero holds")

# --------------------------------------------------------------------------- #
print("\n[3] Board reveal removes board cards from the range")
rt = RangeTracker(dead_cards=parse_cards("As Ah"))
before = rt.n_combos()
flop = parse_cards("Kd 7c 2s")
rt.reveal_board(flop)
after = rt.n_combos()
post = rt.posterior()
uses_board = any(any(card in c for card in flop) for c in post)
check("combo count shrinks after flop", after < before,
      f"{before} -> {after} combos")
check("no surviving combo contains a board card", not uses_board)
check("posterior still normalized after reveal",
      abs(sum(post.values()) - 1.0) < 1e-9, f"sum = {sum(post.values()):.6f}")

# --------------------------------------------------------------------------- #
print("\n[4] Concentration: a tight value-raiser's 'raise' tightens the range")
rt = RangeTracker()  # no dead cards, clean prior
opp = tight_value_opponent()
st = PublicState(facing="none", position="btn")

strength_before = rt.range_strength()
eff_before = rt.effective_combos()
mass_premium_before = rt.prob_mass(
    [parse_cards(h) for h in ["As Ah", "Ks Kh", "Qs Qh", "As Ks"]]
)

rt.update("raise", st, opp)

strength_after = rt.range_strength()
eff_after = rt.effective_combos()
mass_premium_after = rt.prob_mass(
    [parse_cards(h) for h in ["As Ah", "Ks Kh", "Qs Qh", "As Ks"]]
)

check("range strength rises after a raise",
      strength_after > strength_before,
      f"expected Chen {strength_before:.2f} -> {strength_after:.2f}")
check("effective combo count falls (range tightens)",
      eff_after < eff_before,
      f"eff. combos {eff_before:.0f} -> {eff_after:.0f}")
check("premium-hand mass increases",
      mass_premium_after > mass_premium_before,
      f"P(top premiums) {mass_premium_before:.4f} -> {mass_premium_after:.4f}")
print("        top of posterior range after raise:")
for label, p in rt.top_combos(8):
    print(f"          {label}   {p:.4f}")

# --------------------------------------------------------------------------- #
print("\n[5] Two raises tighten further than one (monotone concentration)")
rt1 = RangeTracker()
rt1.update("raise", st, opp)
s1, e1 = rt1.range_strength(), rt1.effective_combos()
rt1.update("raise", st, opp)  # villain raises again on a later street
s2, e2 = rt1.range_strength(), rt1.effective_combos()
check("second raise raises strength further", s2 > s1,
      f"strength {s1:.2f} -> {s2:.2f}")
check("second raise tightens range further", e2 < e1,
      f"eff. combos {e1:.0f} -> {e2:.0f}")

# --------------------------------------------------------------------------- #
print("\n[6] Polarization: a polarized bettor yields nuts + air, thin middle")
rt = RangeTracker()
rt.update("bet", PublicState(facing="none"), polarized_opponent())
post = rt.posterior()
# bucket posterior mass by Chen score into low / mid / high thirds
lows = sum(v for c, v in post.items() if chen_score(c) <= 6)
mids = sum(v for c, v in post.items() if 7 <= chen_score(c) <= 13)
highs = sum(v for c, v in post.items() if chen_score(c) >= 14)
check("range is bimodal: middle is smallest, both tails carry real mass",
      mids < lows and mids < highs and lows > 0.05 and highs > 0.05,
      f"low={lows:.3f}  mid={mids:.3f}  high={highs:.3f}")

# --------------------------------------------------------------------------- #
print("\n[7] Uninformative action barely moves the range (a station's call)")
rt = RangeTracker()
opp = calling_station()
before = rt.range_strength()
rt.update("call", PublicState(facing="bet"), opp)
after = rt.range_strength()
check("a near-uniform 'call' leaves range strength ~unchanged",
      abs(after - before) < 0.05,
      f"strength {before:.3f} -> {after:.3f} (delta {after-before:+.4f})")

# --------------------------------------------------------------------------- #
print("\n[8] Defensive: an action the model never takes keeps the prior")
rt = RangeTracker()
before = dict(rt.posterior())
# tight_value_opponent only emits 'raise'/'fold'; feed an impossible 'allin'
rt.update("allin", PublicState(), tight_value_opponent())
after = rt.posterior()
unchanged = all(abs(before[c] - after[c]) < 1e-12 for c in before)
check("unexplained action does not corrupt the belief", unchanged,
      "posterior identical to prior (no NaNs, no collapse)")

# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
total, passed = len(results), sum(results)
print(f"RESULT: {passed}/{total} checks passed")
print("=" * 70)
if passed != total:
    raise SystemExit(1)
