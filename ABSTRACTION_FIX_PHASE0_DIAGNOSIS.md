# Abstraction-Fix — Phase 0 Diagnosis (read-only) + stale-attribution correction

**Date:** 2026-06-10/11 · **Branch:** `abstraction-fix` (off `retrain-mechanisms-spike`) · **Verdict: A — abstraction confirmed**, corrected mechanism.

## Verdict

ALL_IN is over-admitted preflop — **but not via the mechanism the brief and the run log named.**
The live, confirmed cause is the *unconditional* admission of ALL_IN at every stack depth.

## Evidence

### 0a — Legal-set audit (code)
`src/pokerbot/abstraction/actions.py:116-117`, `_legal_preflop`:
```python
if stack > 0:
    actions.append(AbstractAction(ActionType.ALL_IN, stack))
```
ALL_IN is appended **unconditionally whenever `stack > 0`** — both the unopened (`to_call==0`) and
facing-bet branches, at *every* depth incl. 100bb deep. No jam-threshold / SPR guard. External-sampling
MCCFR then explores and reinforces it in every preflop infoset.

### Timeline finding that corrects the stale attribution
- The `to_call==0 → ALL_IN-only` collapse was **already fixed** by commit `172c243` (2026-06-01): the
  anchor (`anchor = to_call if to_call>0 else min_raise`) offers RAISE_2_5X/3_5X in unopened pots.
- `172c243` **is an ancestor of the spike HEAD `e5fdbbd`** the 2M campaign trained from (it carries the
  four campaign features: multi-worker port `3a55da0`, `opp_aggression_bias` `b97771c`,
  `coverage_instrument` `fc29d19`, `--lbr-every` `e5fdbbd`).
- ∴ the 2M campaign trained **with** that fix in, yet still failed bands (VPIP 37.4 / nc-AI 6.9%). The run
  log's attribution to "`to_call==0 → ALL_IN`" (`POD_RUN_LOG.md` line ~136) was therefore **stale** — that
  path was neutralized and the looseness persisted anyway. Corrected in the run log on 2026-06-11.
  `ABSTRACTION_FIX_SPEC.md` §1 was already correct.

### 0b — Strategy-mass audit (`strategy-2m-parallel.db`, sha `8432cc77…`, 459,936 preflop infosets)
Row-mean ALL_IN probability by effective-stack bucket (visit_count=0 in this DB — known export bug — so
row-mean over infosets; cross-validated against the bands self-play nc-AI which agrees in magnitude):

| eff stack | rows | ALL_IN mass | GTO expectation |
|---|---|---|---|
| 30-50bb | 21,770 | 0.204 | small |
| 50-75bb | 55,960 | 0.185 | ~0 |
| **75-100bb (facing)** | **348,339** | **0.101** | **~0** |
| 75-100bb (to_call==0) | 3,711 | 0.066 | ~0 |
| 100-150bb | 4,717 | 0.068 | ~0 |

ALL_IN mass is spread across depths incl. deep — **not** confined to the short-stack buckets where jamming
is correct. The best-trained cell (75-100bb, 348k rows) sits at 0.101, below the 0.20 uniform-fallback floor
(⇒ trained, not artifact) yet ~10× the <1% band, and ≈ the self-play nc-AI of 6.9%.

### 0c — Alternatives
- **(i) linear CFR weighting** (`config.py:37`, spec-pinned): **low plausibility as primary.** VPIP flat
  iter_450→final (38→37) ⇒ the fixed point is loose at every iteration; weighting isn't creating it.
- **(ii) self-play opponent regime:** **plausible secondary, cannot be excluded.** Entangled with the
  abstraction — the Phase 1 smoke train is the separating probe (nc-AI drops toward band ⇒ abstraction was
  the lever; persists ⇒ regime implicated, Phase 2 scope changes).

## Phase 1 fix (record)
Compose, don't edit frozen `actions.py` (spec §3 option (a)): the from-scratch/parallel launcher builds a
`GatedNLHEGame` (existing tested gate module `zoom.abstraction_gate`) instead of `SimpleNLHEGame`. ActionType
enum untouched (Gate C parity). The fix went through TWO iterations:

**Iteration 1 — SPR gate (`spr_cap=10`): partial, superseded.** First wired the existing SPR gate
(`all_in_allowed`, SPR = stack/(pot+to_call)). 50k smoke result (structural, robust to training amount):
75-100bb **unopened** preflop ALL_IN-in-mask **100%→0%** (the unprovoked open-jam, fixed), but **facing-a-bet**
deep ALL_IN **53.8%→46.9%** (≈unchanged) — and facing cells are 99.4% of deep preflop infosets (557k vs 3.6k).
Self-play nc-AI **6.9%→6.6%** (bands still FAIL). **Cause:** facing a raise lowers SPR below the cap, so the
dominant facing-3bet/4bet deep overshove survives; and SPR conflates depth with commitment in OPPOSITE
directions — a 25bb open-jam (legit) has SPR≈10 while a 100bb facing-3bet overshove (illegit) has SPR≈5, so
**no single SPR cap separates them**. SPR is the right POSTFLOP lever, the wrong PREFLOP one.

**Iteration 2 — preflop depth + commitment gate (current).** Preflop now uses `preflop_all_in_allowed`
keying on two ORTHOGONAL axes: **depth** (keep iff eff_bb ≤ `max_preflop_allin_eff_bb`, default 25 — push/fold
zone, rescues short jams regardless of pot, which SPR couldn't) and **commitment** (keep iff stack-behind-
after-call ≤ `preflop_allin_commit_pots` pot-sized bets, default 1.5 — real 4bet/5bet shove-war; keyed on
stack-behind, NOT to_call). Postflop unchanged (`allin_spr_cap`, default 10). Six-spot probe at 100bb (all
pass): unopened/facing-open/**facing-3bet** → DROP; **facing-4bet (committed)**/25bb-open/12bb-open → KEEP.
Knobs in `DeepCFRConfig` (`allin_spr_cap`, `max_preflop_allin_eff_bb`, `preflop_allin_commit_pots`), lockstep
with the gate module via `test_config_pinned`; Phase 2 sweeps them. `max_preflop_allin_eff_bb=inf` reproduces
ungated preflop.

**0c-(ii) regime read — 200k corrected-gate smoke: abstraction was the MAJOR lever (not regime).**
Structural (vs 2M ungated): 75-100bb facing ALL_IN-in-mask **53.8%→30.5%** and **100-150bb facing 75.0%→0.0%**
(the deepest facing cells fully gated; the 30.5% residual at 75-100bb is the committed/forced-jam spots the
commitment exception correctly keeps), unopened-deep **100%→0%**. Behavioral self-play (3 seeds, table-size 6,
100bb): **nc-AI 6.9% (2M) / 6.6% (50k SPR) → 2.7%** corrected, raw-AI 8.6%→4.35%. nc-AI more than halved the
moment the facing-bet deep ALL_IN was actually removed ⇒ the abstraction over-admission was the PRIMARY driver
of nc-AI; the self-play regime is NOT implicated as the main cause. **Residual:** 2.7% is still above the <1%
band — not yet attributable, because 200k is undertrained (VPIP 43.8 vs the 2M's converged 37.4; VPIP is
trending down with training, 50k→200k = 50→44, exactly the undertraining signature). Whether the residual
reaches <1% at convergence, or floors at ~2-3% (a finer-abstraction card-bucket effect per spec §4, or a small
regime contribution), is the question the **Phase 2 converged rescale** answers — it is NOT decidable at smoke
scale. Phase 2 remains a separate Orch go.

**Phase 2 costing note.** Anchor the burn-rate estimate to the FULL-RESERVOIR per-iter rate, not the opening
pace: the 200k smoke showed per-iter ballooning from ~20s to ~75-100s once the 3M advantage/policy reservoirs
capped (same lesson as the 2M campaign). Otherwise reuse the 2M recipe + every operational gotcha (secure
A100-SXM, ulimit 1048576, `pip install -e ".[all]"`, pokerkit==0.7.3, key via env, terminate-not-stop, iter-100
pre-flight gate, auto-shutdown). Gate defaults (eff_bb 25 / commit 1.5 / spr 10) stay as-is; Phase 2 sweeps them.
