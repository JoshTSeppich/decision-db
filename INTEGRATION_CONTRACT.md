# Integration Contract — Decision-DB WebSocket Policy Server

> Authoritative wire-level contract for the trained CFR poker policy
> served over localhost WebSocket. Server source: `scripts/serve.py`.
> Underlying adapter source: `src/pokerbot/runtime/adapter.py`.

This document describes ONLY the server interface (request/response
shape, semantics, errors). It does not document what is or isn't
observable from the consumer side — that is the consumer's concern.

---

## A. CONFIRMED vs ASSUMED

**CONFIRMED (read directly from `src/pokerbot/`):**

- `GameStateRequest` is Pydantic v2 with `extra='forbid'` and
  `frozen=True` (`runtime/schema.py:17, 42`). Unknown JSON fields raise
  `ValidationError`.
- `schema_version` is locked to `1` (`schema.py:44`,
  `Field(ge=1, le=1)`).
- `table_size` accepts any integer in `{2,3,4,5,6,7,8,9}`
  (`schema.py:46`).
- `DualStrategyDB` routes `table_size == 6` → primary (6-max DB); all
  other sizes → secondary (9-max DB) (`strategy_db/dual.py:32-33, 53-54`).
- `ActionResponse` returns ONE sampled action plus the resolved chip
  `amount` — not a distribution (`runtime/adapter.py:158, 179-188`).
- Postflop bet sizing fractions: BET_33=0.33, BET_66=0.66, BET_100=1.0,
  BET_150=1.5 × pot (`abstraction/actions.py:50-56`,
  `runtime/adapter.py:66-71`).
- Preflop raise ratios: RAISE_2_5X=2.5, RAISE_3_5X=3.5 × max(to_call,1)
  (`abstraction/actions.py:58-61`, `runtime/adapter.py:72-75`).
- Position is SB-relative: `(hero_seat - sb_seat) % table_size`, with
  HU convention `sb_seat == button_seat` and otherwise
  `sb_seat = (button_seat + 1) % table_size` (`runtime/adapter.py:284-286`,
  `abstraction/encoding.py:53-60`).
- Archetype enum: `nit/tag/lag/maniac/station/unknown`
  (`opponent/archetype.py:14-20`).
- Archetype priority for multi-way pots:
  `STATION > MANIAC > NIT > LAG > TAG > UNKNOWN`
  (`opponent/model.py:37-44`).
- Minimum hand count for archetype classification: 20
  (`opponent/archetype.py:11`).

**ASSUMED (extrapolated or inferred):**

- The sister project will run on localhost; no authentication
  is required. (Standard for localhost integration.)
- The sister project maintains opponent stats across hands on its side
  and ships the per-decision snapshot via `opponent_archetypes`. (The
  schema's `opponent_archetypes` is optional; consumers can omit it.)
- `current_bets` represents chips committed THIS STREET (not cumulative
  across the hand). This matches `_build_history_bytes` in
  `runtime/adapter.py:240-281` but is not explicitly tested in isolation.
- `to_call`, `min_raise`, `max_raise` are caller-computed from the
  rules engine and set to 0 when the corresponding action is gated
  (raise cap reached, free check available, etc.). Server treats these
  values as authoritative.
- Latency targets (p50 < 50ms, p95 < 200ms) are validated by a smoke
  client but not bound to specific hardware.

---

## B. Server Protocol

- **URL:** `ws://127.0.0.1:8765` (default port; configurable via `--port`).
- **Wire format:** one JSON object per WebSocket text message, both
  directions. Binary frames are not supported.
- **Request envelope:**
  ```json
  { "seq": <int>, "request": <GameStateRequest object> }
  ```
- **Response envelope (success):**
  ```json
  { "seq": <same int>, "response": <ActionResponse object> }
  ```
- **Response envelope (error):**
  ```json
  { "seq": <int|null>, "error": "<string>", "details": <object|null> }
  ```
  `seq` is `null` only when the inbound payload was unparseable JSON or
  was not a JSON object.
- **`seq` round-trips:** the server echoes the same `seq` value the
  caller sent. Callers may use `seq` to multiplex / match responses to
  requests on a shared connection.
- **Concurrency:** multiple simultaneous connections are allowed. The
  server serializes `RuntimeAdapter.decide()` calls process-wide so
  per-request opponent-model state never interleaves.
- **Auth:** none. Localhost only.
- **Statelessness:** the server holds NO session state across requests.
  Per-request opponent modeling is achieved by the caller shipping the
  `opponent_archetypes` snapshot in each request payload.

---

## C. `GameStateRequest` Schema

All fields are required unless marked optional. `extra="forbid"`: any
extra field raises `validation_error`.

| Field | Type | Constraints | Default | Notes |
|---|---|---|---|---|
| `schema_version` | `int` | `== 1` | — | Version-locked; do not send 2. |
| `game_type` | `"cash"` \| `"tournament"` | — | — | This server expects `"cash"`. |
| `table_size` | `int` | `2..9` | — | Players currently at the table (excludes folded / sitting out). Routes the DB lookup; see §H. |
| `blinds` | `{sb: int, bb: int}` | `sb ≥ 0`, `bb > 0` | — | Chip amounts. |
| `ante` | `int` | `≥ 0` | `0` | Optional; per-seat ante in chips. |
| `hero_seat` | `int` | `≥ 0` | — | 0-indexed seat of the bot. Must be `< table_size`. |
| `button_seat` | `int` | `≥ 0` | — | 0-indexed dealer button seat. SB derived as: `button_seat` (heads-up) or `(button_seat + 1) % table_size` (3+). |
| `hero_hole` | `list[str]` | `len == 2` | — | Two card strings like `"As"`, `"Th"` (rank: 2-9, T, J, Q, K, A; suit: c, d, h, s). |
| `board` | `list[str]` | `len ∈ {0,3,4,5}` | — | Community cards. Length encodes street: 0=preflop, 3=flop, 4=turn, 5=river. |
| `stacks` | `list[int]` | `len == table_size` | — | Per-seat chip stacks. `hero_stack = stacks[hero_seat]`. |
| `current_bets` | `list[int]` | `len == table_size` | — | Chips each seat has committed THIS STREET (resets at street boundary). |
| `pot_committed` | `int` | `≥ 0` | — | Total pot before hero's decision (chips). |
| `to_call` | `int` | `≥ 0` | — | Chips hero must put in to match current bet. 0 if free check. |
| `min_raise` | `int` | `≥ 0` | — | Minimum raise size (chips). Set to 0 when raise gate is closed by the rules engine. |
| `max_raise` | `int` | `≥ 0` | — | Maximum raise size (chips). Set to 0 when raise gate is closed. |
| `action_history` | `list[ActionHistoryEntry]` | — | `[]` | Chronological prior actions; see below. |
| `opponent_archetypes` | `tuple[Archetype \| null, ...] \| null` | `len == table_size` when present | `null` | OPTIONAL. Per-seat archetype classification (or null for unknown / hero's own seat). When omitted or all-null, opponent modeling is not engaged. See §J. |

**`ActionHistoryEntry`:**

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `seat` | `int` | `≥ 0` | 0-indexed seat that took the action. |
| `street` | `int` | `0..3` | 0=preflop, 1=flop, 2=turn, 3=river. |
| `type` | `str` | `"fold" \| "check" \| "call" \| "bet" \| "raise" \| "all-in"` | — |
| `amount` | `int` | `≥ 0` | Chips committed BY THAT ACTION (not cumulative). |

**Position encoding (critical):**
- Internally the policy uses SB-relative position:
  `position = (hero_seat - sb_seat) % table_size`.
- Heads-up: `sb_seat == button_seat` (so the button posts SB).
- 3+ players: `sb_seat = (button_seat + 1) % table_size`.
- Seat indices in the request are independent of this; the server
  derives position from `button_seat` and `hero_seat`.

**Stack semantics:**
- All stacks in `stacks` are CHIP AMOUNTS, not BB. The server converts
  to BB internally for bucketing
  (`min(hero, max-remaining-opp) // bb → bucket`).

---

## D. `ActionResponse` Schema

| Field | Type | Notes |
|---|---|---|
| `action` | `"fold" \| "check" \| "call" \| "bet" \| "raise"` | Emitted action verb. `"check"` vs `"call"` is `to_call == 0` vs `> 0`. `"bet"` vs `"raise"` is likewise distinguished. |
| `amount` | `int` | Resolved chip amount to commit. For fold = 0, check = 0, call = `min(to_call, stack)`, bet/raise = sized per the table below and clamped to `[min_raise, stack]`. |
| `abstract_action` | `str` | Name of the sampled `ActionType` (e.g. `"BET_66"`, `"RAISE_2_5X"`). See §E. |
| `probability_sampled` | `float` | `[0.0, 1.0]`. Probability of the sampled action under the post-opponent-adjustment distribution. |
| `infoset_hash` | `str` | 32-char hex of the canonical InfoSet hash. Diagnostic only. |
| `version` | `int` | DB version used. Pinned by the server at boot. |
| `latency_ms` | `int` | `≥ 0`. Wall-clock decision latency for this request. |
| `fallback_used` | `"exact" \| "nearest_neighbor" \| "default_policy" \| "pushfold"` | Which lookup path produced the decision. See §F. |

**The response is a SINGLE sampled action, not a distribution.** If the
caller needs ensemble behavior, sample by sending the same request
multiple times (the adapter's RNG seed advances per call).

### D.1 Bet-sizing translation (CALLER MAY COPY VERBATIM)

The server returns `amount` already resolved. If your client needs to
compute the resolved amount independently, the rule is (from
`abstraction/actions.py:50-61` and `runtime/adapter.py:398-412, 420-445,
abstraction/actions.py:186-207`):

```
FOLD                  →  0
CHECK_CALL            →  min(to_call, stack)
ALL_IN                →  stack
RAISE_2_5X (preflop)  →  round(2.5 * max(to_call, 1))     [then clamped]
RAISE_3_5X (preflop)  →  round(3.5 * max(to_call, 1))     [then clamped]
BET_33  (postflop)    →  to_call + round(0.33 * pot)      [then clamped]
BET_66  (postflop)    →  to_call + round(0.66 * pot)      [then clamped]
BET_100 (postflop)    →  to_call + round(1.0  * pot)      [then clamped]
BET_150 (postflop)    →  to_call + round(1.5  * pot)      [then clamped]

Clamp rule for bet/raise:
    amount = max(amount, min_raise)                       [resolve_action]
    amount = max(amount, to_call + bb)                    [adapter floor]
    amount = min(amount, stack)                           [final clamp]
```

`pot` is `request.pot_committed`; `to_call`, `min_raise`, `stack` are
from the request.

---

## E. `ActionType` Enum

Values emitted via `abstract_action`:

| Name | Street legality | Semantic |
|---|---|---|
| `FOLD` | any (when `to_call > 0`) | Surrender hand. |
| `CHECK_CALL` | any | Check if `to_call == 0`, call otherwise. |
| `BET_33` | postflop only | Bet 33% pot over `to_call`. |
| `BET_66` | postflop only | Bet 66% pot over `to_call`. |
| `BET_100` | postflop only | Pot-sized bet over `to_call`. |
| `BET_150` | postflop only | 1.5× pot over `to_call`. |
| `RAISE_2_5X` | preflop only | 2.5× current bet. |
| `RAISE_3_5X` | preflop only | 3.5× current bet. |
| `ALL_IN` | any (when raise gate open) | Stack-sized commitment. |

The server enforces street legality before sampling
(`runtime/adapter.py:301-339`), so a postflop request will never receive
`RAISE_*` and a preflop request will never receive `BET_*`.

---

## F. Error Modes

The server NEVER crashes the connection in response to a client error.
All failures produce an error envelope; the connection stays open.

| Inbound condition | Envelope | `seq` |
|---|---|---|
| Malformed JSON (unparseable) | `{"error": "parse_error", "details": {"message": ...}}` | `null` |
| Valid JSON but not an object | `{"error": "envelope_error", "details": {...}}` | `null` |
| Missing or non-int `seq` | `{"error": "envelope_error", "details": {...}}` | `null` |
| Missing `request` object | `{"error": "envelope_error", "details": {...}}` | echoed |
| Schema validation failure (extra/missing/wrong-typed) | `{"error": "validation_error", "details": {"errors": [Pydantic errors]}}` | echoed |
| Adapter raises (unexpected bug, e.g. bad card string) | `{"error": "adapter_exception", "details": {"type": "...", "message": "..."}}` | echoed |

The `details.errors` list under `validation_error` is the output of
`pydantic.ValidationError.errors(include_url=False)`. Each entry has
`type`, `loc`, `msg`, `input`.

**`fallback_used` semantics in successful responses:**
- `"exact"` — DB row matched on the full infoset hash. Highest fidelity.
- `"nearest_neighbor"` — no exact match; row matched on
  `(table_size, street, card_bucket, position, stack_bucket)` prefix
  (history ignored). Reasonable fidelity for well-trained sizes.
- `"default_policy"` — no neighbor; the server fell through to the
  built-in Chen-formula default. Decorative — opponent-model adjustment
  applied to a single-action distribution is a no-op.
- `"pushfold"` — short-stack push/fold table override (tournament mode
  only). Cash callers should never see this.

**Server crash:** if the server process dies, all WebSocket connections
drop. The client should reconnect with exponential backoff.

---

## G. Example Request / Response Pairs

### G.1 Preflop, hero UTG, facing BB only, 100bb stacks, 8-max

> `stacks` are CHIPS-BEHIND: the SB (seat 1) and BB (seat 2) blinds are deducted
> from their stacks (995 / 990) and held in `current_bets`. This matches the
> training convention (`pk.stacks`); do NOT report gross stacks. See §H.5.

Request:
```json
{
  "seq": 1,
  "request": {
    "schema_version": 1,
    "game_type": "cash",
    "table_size": 8,
    "blinds": {"sb": 5, "bb": 10},
    "ante": 0,
    "hero_seat": 3,
    "button_seat": 0,
    "hero_hole": ["As", "Kh"],
    "board": [],
    "stacks": [1000, 995, 990, 1000, 1000, 1000, 1000, 1000],
    "current_bets": [0, 5, 10, 0, 0, 0, 0, 0],
    "pot_committed": 15,
    "to_call": 10,
    "min_raise": 20,
    "max_raise": 1000,
    "action_history": []
  }
}
```

Response (illustrative):
```json
{
  "seq": 1,
  "response": {
    "action": "raise",
    "amount": 35,
    "abstract_action": "RAISE_3_5X",
    "probability_sampled": 0.62,
    "infoset_hash": "4f8e2a91bc73d104",
    "version": 1,
    "latency_ms": 4,
    "fallback_used": "exact"
  }
}
```

### G.2 Flop, hero IP, facing 1/2-pot c-bet, top pair

Request:
```json
{
  "seq": 2,
  "request": {
    "schema_version": 1,
    "game_type": "cash",
    "table_size": 8,
    "blinds": {"sb": 5, "bb": 10},
    "ante": 0,
    "hero_seat": 3,
    "button_seat": 3,
    "hero_hole": ["As", "Kh"],
    "board": ["Ac", "7d", "2h"],
    "stacks": [950, 950, 950, 950, 1000, 1000, 1000, 1000],
    "current_bets": [0, 0, 50, 0, 0, 0, 0, 0],
    "pot_committed": 100,
    "to_call": 50,
    "min_raise": 100,
    "max_raise": 950,
    "action_history": [
      {"seat": 3, "street": 0, "type": "raise", "amount": 30},
      {"seat": 2, "street": 0, "type": "call",  "amount": 25},
      {"seat": 2, "street": 1, "type": "bet",   "amount": 50}
    ]
  }
}
```

### G.3 River, hero OOP, drawing dead, facing pot-sized bet

Request:
```json
{
  "seq": 3,
  "request": {
    "schema_version": 1,
    "game_type": "cash",
    "table_size": 8,
    "blinds": {"sb": 5, "bb": 10},
    "ante": 0,
    "hero_seat": 2,
    "button_seat": 5,
    "hero_hole": ["7c", "8c"],
    "board": ["Ac", "Kd", "2h", "5s", "Jh"],
    "stacks": [500, 500, 400, 500, 500, 500, 500, 500],
    "current_bets": [0, 0, 0, 0, 0, 200, 0, 0],
    "pot_committed": 400,
    "to_call": 200,
    "min_raise": 400,
    "max_raise": 400,
    "action_history": []
  }
}
```

A response of `{"action": "fold", "amount": 0, "abstract_action": "FOLD", ...}`
is the expected baseline here.

---

## H. Table-Size Routing and Calibration

**Competition target is 6-max.** The live zoom advisory service is wired as a
PURE-6-max single DB: `--db training/v5-6max-fix.db` (the de-biased v5-6max RNR
blueprint, `table_size=6` rows only), **no `--db-9max`**. With the secondary
omitted, `build_zoom_adapter` uses the single-DB path (`db = primary`), so a
`table_size=6` request keys directly into the blueprint — no `DualStrategyDB`
routing. See `scripts/serve_6max_blueprint.sh`. `table_size` is pass-through from
the eyes (never hardcoded).

If a `--db-9max` secondary IS supplied (for serving non-6 sizes), the
`DualStrategyDB` router selects the policy DB by `table_size`:

| `table_size` | Routes to | DB training distribution |
|---|---|---|
| `6` | `--db` (the v5-6max blueprint) | 6-max trained (only `table_size=6` rows present) |
| `2, 3, 4, 5, 7, 8, 9` | `--db-9max` (e.g. a 9-max DB) | 9-max trained (only `table_size=9` rows present) |

The server **accepts every `table_size ∈ {2..9}` with no error** (so a non-6
request against the pure-6-max wiring returns `default_policy`, NOT an error).

### H.1 Empirical coverage (measured 2026-05-20, PILOT DBs)

> NOTE: this table was measured against the PILOT DBs (`strategy-pilot-v2` +
> `strategy-pilot-v3-9max`), not the live v5-6max blueprint. It documents the
> `table_size`-routing behaviour (only `{6, 9}` hit), which is unchanged. The
> 6-max row applies to the blueprint too, with one caveat now tracked
> separately: a 100bb effective stack rounds to `stack_bucket=6`, which is
> near-empty at canonical preflop cells, so 100bb spots hit ONLY if the eyes
> report chips-behind (post-blind) stacks → `stack_bucket=5`. See §H.5.

`StrategyDB.nearest_neighbor` requires an EXACT `table_size` match in
SQL (`strategy_db/sqlite.py:78`). The trained DBs contain only
`table_size ∈ {6, 9}` rows. As a result, NN lookup hits ONLY at the
trained sizes:

| `table_size` | exact_hit | nn_hit | default_policy | Status |
|---|---|---|---|---|
| 2 | 0.0% | 0.0% | **100.0%** | POOR |
| 3 | 0.0% | 0.0% | **100.0%** | POOR |
| 4 | 0.0% | 0.0% | **100.0%** | POOR |
| 5 | 0.0% | 0.0% | **100.0%** | POOR |
| **6** | 0.0% | **76.0%** | 24.0% | **GOOD** |
| 7 | 0.0% | 0.0% | **100.0%** | POOR |
| 8 | 0.0% | 0.0% | **100.0%** | POOR |
| **9** | 0.0% | **84.0%** | 16.0% | **GOOD** |

(N=25 per size, randomly generated `(card, board, position, stack)`
combinations across streets. `exact_hit` is 0% because we don't replay
specific training histories; production traffic with realistic
histories will see some `exact_hit` mass shift in from `nn_hit`.)

### H.2 Empirical groupings

- **GOOD** (CFR primary, `default_policy < 20%`): `table_size ∈ {6, 9}`.
- **MARGINAL** (20-50%): none in this configuration.
- **POOR** (>50% default_policy, client-side fallback recommended):
  `table_size ∈ {2, 3, 4, 5, 7, 8}`.

### H.3 Caller guidance (UPDATED from empirical findings)

**Strong recommendation:** the sister project should keep its own
decision engine for any `table_size ∉ {6, 9}`. At those sizes the
server returns valid `ActionResponse` payloads, but `fallback_used`
will be `default_policy` ~100% of the time — meaning the decision is
made by the built-in Chen-formula heuristic, NOT the trained CFR
policy. That heuristic is reasonable but not what you're paying for
the CFR DB for.

For `table_size ∈ {6, 9}`: CFR-backed. ~76-84% of decisions come from
NN-matched DB rows; the remaining 16-24% still fall through to
`default_policy`. Inspect `response.fallback_used` per request if you
want a quality signal.

**Dynamic table size:**
The live competition table fluctuates as players sit, leave, or are
reseated. Send the CURRENT `table_size` (count of non-folded,
non-sitting-out players) on every request. The router reselects the DB
per call; no session state required. **In practice the policy is only
trained at 6 and 9, so plan your fallback routing around those two
anchor sizes.**

### H.4 Note on per-street coverage at GOOD sizes

Within `table_size=6` and `table_size=9`, NN coverage degrades on
later streets (random board buckets are less likely to match a
trained row):

```
size=6:  preflop 100% nn  |  flop 89% nn  |  turn 50% nn  |  river 63% nn
size=9:  preflop 100% nn  |  flop 100% nn |  turn 57% nn  |  river 80% nn
```

Preflop and flop decisions are robust; turn and river decisions have a
non-trivial default-policy share.

**Dynamic table size:**
The live competition table fluctuates as players sit, leave, or are
reseated. Send the CURRENT `table_size` (count of non-folded,
non-sitting-out players) on every request. The router reselects the DB
per call; no session state required.

### H.5 100bb effective-stack boundary (LIVE — confirm before trusting 6-max)

`stack_bucket` boundaries are `(10,20,30,50,75,100,150,200,300)` BB, binned by
`eff_bb < boundary` (`abstraction/encoding.py:70-75`). So an effective stack of
**exactly 100bb → `stack_bucket=6`**, whereas **99bb → `stack_bucket=5`**.

There are THREE conventions in play, and two of them are verified to CONFLICT:

- **Training (VERIFIED — chips-behind).** The export keys `stack_bucket` off
  pokerkit `pk.stacks` (`nlhe_game.py:381-386`), which holds chips BEHIND —
  posted blinds and the current bet have already left the stack (confirmed by
  `pot = sum(initial_stacks) − sum(pk.stacks)`, `nlhe_game.py:267`). So at a
  100bb table the BB's effective stack at the first decision is ~99bb →
  `stack_bucket=5`. The v5-6max blueprint trained the canonical preflop spots
  there: measured coverage in `training/v5-6max-fix.db` at the AKo preflop
  SB-relative cell has `stack_bucket` 0-5 populated, **bucket 6 absent**
  (overall preflop: 18,644 rows at bucket 5 vs only 1,307 at bucket 6).
- **Contract example G.1 (VERIFIED — gross; WRONG, contradicts training).** §G.1
  shows the BB with `stacks=1000` and `current_bets=10` — the blind is in
  `current_bets`, NOT deducted from `stacks`. That is GROSS, the opposite of the
  chips-behind convention training used. Fed verbatim it gives 100bb →
  `stack_bucket=6` → the near-empty bucket → `default_policy` MISS. **G.1 has
  been corrected below (BB now shows `stacks=990`) so the contract stops
  demonstrating the wrong convention.**
- **Adapter (VERIFIED — consumes `request.stacks` raw).** `_stack_bucket`
  computes `min(hero, max-remaining-opp) // bb` from `request.stacks` directly
  and **never references `current_bets`** (`adapter.py:213-224`). It therefore
  assumes the caller already sent chips-behind (to match training); it does NOT
  normalize gross→behind.

**The ONLY remaining unknown is which convention the EYES actually emit** — that
is not observable from this repo (no eyes source / captured frames). Resolve by
capturing ONE real `GameStateRequest` at a known 100bb 6-max preflop spot and
checking a committed seat (e.g. the BB): `stacks[seat] == starting`
(GROSS — adapter mis-buckets to 6, MISS) vs `stacks[seat] == starting −
current_bets[seat]` (chips-behind — hits bucket 5). If the eyes are gross, the
fix is `chips-behind = stacks − current_bets` (a field the eyes already send),
normalized in the zoom bridge before `adapter.decide` — gated on this
confirmation, since applying it when the eyes already report chips-behind would
double-subtract. Same keying-seam class as the `to_call==0` and folded-seats
bugs: align the stack convention, do not guess.

---

## I. Game Type

This server targets CASH play. Send `game_type: "cash"` on every
request. The adapter chain is `RuntimeAdapter.decide(request)` directly
(no `TournamentAdapter` wrap).

If `game_type: "tournament"` is sent, the server still routes it
through `RuntimeAdapter` (which ignores the tournament-specific ICM /
push-fold paths). For true tournament play, a separate server endpoint
would be needed; that is out of scope for this contract.

---

## J. Opponent Modeling

**Field:** `GameStateRequest.opponent_archetypes` (optional, default
`null`). When present, it is a tuple of length `table_size` whose
elements are one of:

```
"nit" | "tag" | "lag" | "maniac" | "station" | "unknown" | null
```

(`Archetype` enum, `opponent/archetype.py:14-20`. JSON encoding is the
StrEnum string value.)

Per-slot:
- `null` (JSON) → no classification for that seat (insufficient
  observations, hero's own seat, sitting-out player). Skipped by the
  server.
- Hero's own seat: send `null` (the server strips it regardless, but
  explicit `null` is clearest).

### J.1 Multi-way pot resolution

The server applies the same priority ordering as
`ArchetypeOpponentModel._pick_archetype` (`opponent/model.py:37-44`):

```
STATION > MANIAC > NIT > LAG > TAG > UNKNOWN
```

Among all non-null, non-hero archetypes in the request, the highest-priority
one wins. The chosen archetype's static adjustment is applied to the policy's
base action distribution before sampling.

Rationale (from `opponent/model.py` module docstring): the priority is
ordered by the EXPLOITABILITY of the line that fails worst against that
type — Stations bust bluffs the worst, Maniacs bust folds the worst, etc.

### J.2 Per-archetype adjustments (informational)

The adjustments are static (do not depend on infoset) and are applied
to the base policy's probability distribution
(`opponent/model.py:116-167`):

| Archetype | Adjustment summary |
|---|---|
| `station` | −40% ALL_IN, −40% RAISE_3_5X, −30% BET_150 mass → FOLD (if facing bet) else CHECK_CALL. Bluffs are bad vs stations. |
| `maniac` | +25% relative CHECK_CALL mass from FOLD; −20% RAISE_2_5X / RAISE_3_5X → CHECK_CALL. Call down vs over-bluffers. |
| `nit` | When facing bet: −30% CHECK_CALL → FOLD. When not facing bet: −20% BET_33 / BET_66 → CHECK_CALL. Don't thin-value nits. |
| `tag` / `lag` / `unknown` | Passthrough (no adjustment). |

### J.3 Caller responsibility

If the caller wants its archetype classifier to match the server's,
mirror these thresholds from `opponent/archetype.py:23-118`:

```
Nit      VPIP < 18%   AND PFR < 12%   AND AF > 1.5
TAG      VPIP 18-26%  AND PFR 14-22%  AND AF > 2.0
LAG      VPIP 26-40%  AND PFR 22-35%  AND AF > 2.5
Maniac   VPIP > 40%   AND PFR > 30%   AND AF > 3.0
Station  VPIP > 35%   AND AF < 1.0    AND fold-to-cbet < 40%
```

Margin schedule (`ArchetypeClassifier._margin`,
`opponent/archetype.py:49-66`) — applied as multiplicative tightening:

| Hands observed | Margin |
|---|---|
| `< 20` | `∞` (always returns `UNKNOWN`) |
| `20–49` | `0.15` |
| `50–99` | `0.08` |
| `100–199` | `0.04` |
| `≥ 200` | `0.0` (base thresholds) |

Send `"unknown"` (or `null`) for any seat with `< 20` hands observed.
The server treats `"unknown"` as passthrough regardless of multi-way
priority.

### J.4 What the server does NOT do

- The server does NOT track opponent state across requests. It does
  NOT remember an opponent across hands. The caller owns this entirely.
- The server does NOT compute archetypes from raw stats. The caller
  must pre-classify and ship the archetype enum.
- The server does NOT validate that the caller's classifier matches
  the server's thresholds. If you classify differently, the server
  will still apply the named archetype's adjustment.
