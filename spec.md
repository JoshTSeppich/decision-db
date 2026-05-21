# Poker Bot Architecture Spec

**Target:** No-Limit Texas Hold'em · 6/8/9-max · cash + tournament · ~10k bots · long-run unexploitability

**Stack:** Python 3.11+ · OpenSpiel Deep CFR · PokerKit · phevaluator · SQLite + LMDB · PyTorch

**Layout:** `src/pokerbot/{abstraction,training,strategy_db,runtime,tournament,cli.py}`

This spec is opinionated. Every "DECIDED:" line is a commitment, not a suggestion. Build steps that need to revisit a decision are flagged in §I.

---

## A. Card Abstraction Strategy

### Bucket counts (committed)

| Street  | Buckets | Method                                  |
|---------|---------|-----------------------------------------|
| Preflop | 169     | Lossless (canonical isomorphic classes) |
| Flop    | 200     | Potential-aware k-means on EHS²·dist    |
| Turn    | 200     | Potential-aware k-means on EHS²·dist    |
| River   | 200     | OCHS with 8 opponent clusters           |

**Total infoset cards space:** 169 × 200 × 200 × 200 ≈ 1.35B card-paths, but per-street lookup is O(1).

### Clustering method — DECIDED: Potential-aware (flop, turn) + OCHS (river)

- **Preflop** needs no clustering; 169 is small enough to keep lossless. Hashing 2-card hands to {0..168} via suit-isomorphism is trivial and removes a whole class of bugs.
- **Flop & Turn** use **potential-aware imperfect-recall abstraction** (Ganzfried & Sandholm 2014): each hand maps to a 50-bin EHS-squared histogram over the next street, and we k-means over those histograms with **Earth Mover's Distance** as the metric (approximated via 1D-sort EMD on sorted bin values — exact EMD is too slow).
- **River** uses **Opponent Cluster Hand Strength (OCHS)**: project each river hand onto its win-rate vector against 8 opponent hand clusters (the 8 clusters are derived from preflop equity tiers). This is more informative than raw EHS on the river because river decisions are about who's beating whom, not raw equity.

**Trade-off vs EHS-only:** Potential-aware costs ~20× more compute to build (one-time, ~6h on a workstation CPU) and produces materially better postflop play on the flop/turn where draws matter. EHS-only would conflate "78s on a J84 board" with "AT on K94" — both have ~30% equity but very different futures. Worth the compute.

### Trade-offs vs table size — DECIDED: same buckets across 6/8/9-max

Card bucket depends only on cards + board, not table size. Multi-way pots change the *optimal action* in a bucket, not the bucket boundary. Table size enters through the infoset key (§C), not the bucketer.

> Some literature argues for finer river buckets in multi-way pots because the "what beats me" distribution shifts. We don't do this in v1 — it doubles bucket counts and complicates DB schema. Revisit after first training run (§I).

### Abstraction build artifacts

- `abstraction/buckets_flop.npz` — array shape `(num_canonical_flops, 200)` giving bucket id for each (hole, flop) canonical class.
- `abstraction/buckets_turn.npz`
- `abstraction/buckets_river.npz`
- `abstraction/centroids.npz` — k-means centroids (for debugging + nearest-neighbor fallback in runtime, §F).
- All built once, shipped with the repo, checksummed. Rebuild via `cli.py build-abstraction`.

> **CC HANDOFF — Section A**
>
> **Build step:** 2 (Card abstraction module)
> **Module:** `src/pokerbot/abstraction/cards.py`
>
> **Public API:**
> ```python
> from typing import Literal
> Street = Literal["preflop", "flop", "turn", "river"]
>
> def bucket_hand(
>     hole: tuple[int, int],          # two card ids 0..51
>     board: tuple[int, ...],         # 0, 3, 4, or 5 card ids
>     street: Street,
> ) -> int:                           # 0..N-1 where N is street's bucket count
>     """Deterministic. Suit-isomorphism canonicalized internally."""
>
> def num_buckets(street: Street) -> int: ...
>
> def canonical_hand(
>     hole: tuple[int, int],
>     board: tuple[int, ...],
> ) -> tuple[tuple[int, int], tuple[int, ...]]:
>     """Apply suit-isomorphism canonicalization for cache keys."""
>
> class AbstractionTables:
>     """Loads buckets_*.npz files. Singleton, ~80MB resident."""
>     def __init__(self, path: Path) -> None: ...
>     def lookup(self, hole, board, street) -> int: ...
> ```
>
> **Tests CC must write:**
> 1. `test_preflop_169` — every starting hand maps to one of 169 buckets; AKs and AKo are distinct; suit-rotations of AKs collide.
> 2. `test_isomorphism` — `bucket_hand((As,Kh), (2c,7d,Jh), "flop")` equals `bucket_hand((Ah,Ks), (2d,7c,Js), "flop")` after canonicalization.
> 3. `test_river_ochs_monotone` — nut hand on dry board is in top OCHS bucket; bottom pair is in bottom quartile.
> 4. `test_bucket_count` — `num_buckets("flop") == 200` etc.
> 5. `test_determinism` — 10k random hands bucketed twice yields identical results.
> 6. `test_speed` — 10k lookups in <500ms on CI hardware (regression guard).

---

## B. Action Abstraction Strategy

### Bet sizing (committed, uniform across all postflop streets)

**Preflop (open / first raise):** `[fold, call, 2.5×BB, 3.5×BB, all-in]`
**Preflop (3-bet+):** `[fold, call, 2.5× previous raise, 3.5× previous raise, all-in]`
**Postflop (all streets):** `[fold, check/call, 0.33× pot, 0.66× pot, 1.0× pot, 1.5× pot, all-in]`

That's **7 abstract actions postflop, 5 preflop**. Fold is folded out (heh) when not legal; check folds into call when there's nothing to call. Min-raise legality is checked at translation time, not in the abstraction.

**Why these exact sizes:**
- 0.33-pot covers thin value / weak bluffs (common modern small-bet line).
- 0.66-pot is the donk/cbet sweet spot in solver outputs.
- 1.0-pot is the canonical polarized bet.
- 1.5-pot is overbet territory for the polar end.
- All-in always present so the abstraction never *forces* a non-shove on a short stack.

Adding more sizes (e.g. 0.25, 0.75, 2× pot) bloats the tree multiplicatively for marginal EV gain. 5 postflop sizes + check + fold is the established sweet spot in Brown & Sandholm's Libratus/Pluribus follow-ups for single-workstation budgets.

### Off-tree bet translation — DECIDED: pseudo-harmonic mapping

When the runtime sees an opponent bet of, say, 0.5-pot (not in our abstraction):

**Pseudo-harmonic mapping** (Ganzfried & Sandholm 2013) treats two adjacent abstract bets `A < B` and maps a real bet `x` to abstract bet `A` with probability

```
p(A) = (B - x)(1 + A) / ((B - A)(1 + x))
```

and to `B` otherwise. Deterministic given a seeded RNG; randomized to defeat exploit attempts that probe boundaries.

**Why pseudo-harmonic vs nearest:** Nearest is deterministic-exploitable — an opponent who knows our abstraction can bet ε below a boundary to consistently force us into the smaller bucket. Pseudo-harmonic randomizes near boundaries with a well-defined probability and has provable bounds on exploitability gap vs the unabstracted game.

**On our own bets going out:** we always bet exactly one of the abstract sizes (no translation needed outbound).

### Discretization vs street/stack — DECIDED: uniform postflop, separate preflop

Same 7 actions on flop/turn/river. Preflop is treated specially (BB-relative sizing) because pot-relative preflop is degenerate (pot ≈ blinds before action).

Stack depth changes *which* of the 7 are legal (a 50BB stack can't make a 1.5-pot bet into a 100BB pot), but doesn't change the abstraction. All-in absorbs the "I want to bet more than legal pot fractions allow" case.

> **CC HANDOFF — Section B**
>
> **Build step:** 3 (Action abstraction module)
> **Module:** `src/pokerbot/abstraction/actions.py`
>
> **Public API:**
> ```python
> from dataclasses import dataclass
> from enum import IntEnum
>
> class ActionType(IntEnum):
>     FOLD = 0
>     CHECK_CALL = 1
>     BET_33 = 2
>     BET_66 = 3
>     BET_100 = 4
>     BET_150 = 5
>     ALL_IN = 6
>     # Preflop-specific:
>     RAISE_2_5X = 7
>     RAISE_3_5X = 8
>
> @dataclass(frozen=True)
> class AbstractAction:
>     type: ActionType
>     amount_chips: int     # resolved against current pot/stack at call time
>
> def legal_abstract_actions(
>     pot: int, to_call: int, stack: int, min_raise: int,
>     street: Street,
> ) -> list[AbstractAction]: ...
>
> def translate_bet(
>     real_amount: int, pot: int, legal: list[AbstractAction],
>     rng: random.Random,
> ) -> AbstractAction:
>     """Pseudo-harmonic mapping. Deterministic given rng state."""
>
> def resolve_action(
>     action: AbstractAction, pot: int, stack: int, min_raise: int,
> ) -> int:
>     """Abstract action -> concrete chip amount, clamped to legal range."""
> ```
>
> **Tests CC must write:**
> 1. `test_legal_actions_short_stack` — 5BB stack postflop yields `[FOLD, CHECK_CALL, ALL_IN]` only.
> 2. `test_pseudo_harmonic_midpoint` — real bet at geometric mean of two abstract sizes splits ~50/50 over 10k samples.
> 3. `test_pseudo_harmonic_at_boundary` — real bet exactly equal to abstract size always maps to that size.
> 4. `test_resolve_clamp` — `BET_150` with `pot=100, stack=120` resolves to `120` (all-in clamp), action_type stays `BET_150` for strategy lookup but emitted amount is 120.
> 5. `test_preflop_open_sizes` — first-to-act preflop has `RAISE_2_5X` and `RAISE_3_5X`, not `BET_*`.
> 6. `test_translation_seeded_determinism` — same rng seed yields same translation.

---

## C. Information Set Encoding

### Key layout (committed, byte-exact)

```
+--------+--------+--------+--------+--------+--------+----...---+
| t_size | street | pos    | stack  |  card_bucket    | history  |
| u8     | u8     | u8     | u8     |  u16 LE         | var      |
+--------+--------+--------+--------+--------+--------+----...---+
   1B       1B       1B       1B          2B            ≤64B
```

Fixed prefix = **6 bytes**. Variable history capped at 64 bytes (more than enough for any realistic NLHE hand). Total key ≤ 70 bytes.

**Field semantics:**
- `t_size`: 6, 8, or 9
- `street`: 0=preflop, 1=flop, 2=turn, 3=river
- `pos`: 0..t_size−1, where 0 = SB. Hero's seat relative to the button is computed at runtime; the *infoset* uses absolute seat to capture positional asymmetries.
- `stack`: effective-stack bucket id 0..9 (see below)
- `card_bucket`: 0..65535 (we use 0..199 postflop, 0..168 preflop; u16 for headroom)
- `history`: see compression below

**Effective-stack buckets (BB):** `[<10, 10-20, 20-30, 30-50, 50-75, 75-100, 100-150, 150-200, 200-300, >300]`. "Effective" = min(hero_stack, max_remaining_opponent_stack).

### Betting history compression

Each action in the hand is one byte:

```
0x00 = fold        (terminates branch; rare in history since folded players don't act)
0x01 = check
0x02 = call
0x10-0x16 = bet sized as ActionType.BET_33 .. ALL_IN (postflop)
0x20-0x21 = raise sized as RAISE_2_5X, RAISE_3_5X  (preflop)
0xF0 = street boundary
```

History is the concatenation of per-action bytes from start of hand to current decision point, **excluding the current player's pending action**. Folded players' fold bytes ARE included (they affect remaining-player count and pot).

**Why bytes, not action-indices-per-street:** It's flat, fast to hash, fast to compare, and the street boundary marker makes parsing trivial. A 4-street hand averages ~12 actions = 12-16 bytes. 64-byte cap is generous.

### Single shared strategy across 6/8/9-max — DECIDED: yes

Three separate tables would triple training cost for marginal benefit. `t_size` in the key partitions the lookup space naturally; the network sees `t_size` as a feature and learns size-specific behavior where it matters (mostly preflop). The DB row count grows ~linearly with `t_size` count regardless of whether the model is unified or split.

> A future optimization is fine-tuning a single shared trunk into three heads if multi-way play looks weak after step 5. Not in v1.

### Canonical InfoSet type

```python
@dataclass(frozen=True)
class InfoSet:
    table_size: int          # 6, 8, 9
    street: int              # 0..3
    position: int            # 0..table_size-1
    stack_bucket: int        # 0..9
    card_bucket: int         # 0..65535
    history: bytes           # ≤64 bytes

    def to_bytes(self) -> bytes:
        """Byte-exact layout above. Used as primary key."""

    def hash16(self) -> bytes:
        """BLAKE2b-128 of to_bytes(). Used as SQLite primary key (BLOB)."""
```

Hash is BLAKE2b at digest_size=16. 128 bits is collision-safe for ~2^60 rows (we'll never approach 10^9 rows). Cheaper than SHA-256, no Python overhead vs builtins.

> **CC HANDOFF — Section C**
>
> **Build steps:** 2, 3, 4 (shared between abstraction/ and strategy_db/)
> **Module:** `src/pokerbot/abstraction/infoset.py` (the type) — imported by both `strategy_db/` and `runtime/`.
>
> **Public API:** as the `InfoSet` dataclass above, plus:
> ```python
> def encode_history(actions: list[AbstractAction], street_boundaries: list[int]) -> bytes: ...
> def decode_history(blob: bytes) -> list[tuple[int, ActionType]]:
>     """Returns (street, action_type) pairs."""
> ```
>
> **Tests CC must write:**
> 1. `test_byte_layout` — `InfoSet(6, 0, 2, 4, 99, b"").to_bytes()` equals exactly `bytes([6,0,2,4,99,0])`.
> 2. `test_hash_collision_free` — 100k random infosets produce 100k unique hashes.
> 3. `test_history_roundtrip` — encode then decode is identity for 1000 random sequences.
> 4. `test_history_cap` — encoding a 65-action sequence raises `ValueError`.
> 5. `test_endianness` — `card_bucket=256` is bytes `[0x00, 0x01]` (little-endian).

---

## D. Database Schema

### SQLite DDL (committed)

```sql
-- strategy_db/schema/001_init.sql

CREATE TABLE strategy (
    infoset_hash    BLOB(16) NOT NULL,
    version         INTEGER  NOT NULL,
    table_size      INTEGER  NOT NULL,
    street          INTEGER  NOT NULL,
    position        INTEGER  NOT NULL,
    stack_bucket    INTEGER  NOT NULL,
    card_bucket     INTEGER  NOT NULL,
    history_blob    BLOB     NOT NULL,           -- variable-length, exact bytes
    action_mask     INTEGER  NOT NULL,           -- bitmask of legal ActionType ids
    action_probs    BLOB     NOT NULL,           -- packed float32 array, len = popcount(action_mask)
    visit_count     INTEGER  NOT NULL DEFAULT 0,
    PRIMARY KEY (infoset_hash, version)          -- composite: same infoset, distinct versions for A/B
) WITHOUT ROWID;

CREATE INDEX idx_lookup_compound
    ON strategy(table_size, street, card_bucket, position, stack_bucket, version);

CREATE TABLE meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
-- Required meta keys: schema_version, current_strategy_version, training_iter,
-- abstraction_checksum_flop, abstraction_checksum_turn, abstraction_checksum_river

CREATE TABLE strategy_versions (
    version         INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,             -- ISO-8601
    training_iter   INTEGER NOT NULL,
    notes           TEXT,
    abstraction_checksum TEXT NOT NULL
);
```

**Rationale:**
- `WITHOUT ROWID` — strategy is keyed on the `(infoset_hash, version)` composite PK; we don't want SQLite to maintain a separate ROWID index over the composite. `version` is part of the key (not just a filter column) because a single infoset legitimately has multiple rows under different strategy versions for A/B testing.
- **action_mask + packed action_probs** — variable-length probabilities packed dense; reader expands using popcount of mask. Saves ~30 bytes per row vs storing 7 floats with padding.
- **Compound index** is for the nearest-neighbor fallback at runtime (§F): when an exact infoset hash misses, we scan rows with the same (table_size, street, card_bucket, position, stack_bucket) prefix, ignoring history.
- **version column on every row** — supports A/B testing two strategy versions live by storing both and filtering on version at lookup time. Drop old versions with a single `DELETE WHERE version < N`.

### LMDB layout (equivalent, for the hot-path-read alternative backend)

```
key   = infoset_hash (16 bytes BLOB) ++ version (uint32 big-endian, 4 bytes)
value = msgpack-encoded:
        {
          "ts": uint8 table_size,
          "st": uint8 street,
          "ps": uint8 position,
          "sb": uint8 stack_bucket,
          "cb": uint16 card_bucket,
          "hi": bytes history,
          "am": uint8 action_mask,
          "ap": bytes packed float32 probs,
          "vc": uint32 visit_count,
          "v":  uint16 version,
        }
```

A second LMDB sub-DB `meta` mirrors the SQLite `meta` table. LMDB is read-mmap'd; the runtime hits it without any deserialization overhead beyond msgpack-unpacking the small value blob (~50 bytes typical).

### Row count + disk size estimates

**Assumptions (state them, since CC needs them defensible):**
- 3 table sizes × 4 streets × {169 or 200} card buckets × 9 positions (avg) × 10 stack buckets × ~50 distinct histories actually visited per (street, card_bucket, position, stack_bucket) tuple in training.
- "50 distinct histories" comes from: most of the betting tree is pruned by training — only frequently-visited branches get filled in. Deep CFR with 1k iterations on a 7-action tree explores ~10^4 unique action sequences per street, but most collapse onto the same card_bucket × position rows.

**Estimate:**
- Preflop: 3 × 169 × 9 × 10 × ~30 histories ≈ 137k rows
- Per postflop street: 3 × 200 × 9 × 10 × ~60 histories ≈ 324k rows
- **Total: ~1.1M rows for v1.** Likely 2–5M after a week of training as the tree fills out.

**Disk size at 1M rows:**
- Avg row: 16B hash + 6B prefix fields + 12B history + 4B mask + 28B probs (7 floats) + 4B visits + 2B version ≈ 72 bytes data, ~100 bytes with SQLite overhead.
- **~100 MB at 1M rows. ~500 MB at 5M rows. ~2 GB at 20M rows.** All comfortable on a workstation.

### Migration / versioning

- **Schema versions:** `strategy_db/schema/NNN_*.sql` files applied in order. `meta.schema_version` is the latest applied. Pure SQL forward migrations only; no downgrades.
- **Strategy versions:** the `version` column on every row + `strategy_versions` table. New training run = new version. Runtime defaults to `meta.current_strategy_version`; can be overridden per-lookup for A/B.

> **CC HANDOFF — Section D**
>
> **Build step:** 4 (Strategy database)
> **Module:** `src/pokerbot/strategy_db/{base.py,sqlite.py,lmdb.py,migrate.py}`
>
> **Public API:**
> ```python
> from abc import ABC, abstractmethod
> import numpy as np
>
> class StrategyDB(ABC):
>     @abstractmethod
>     def get(self, infoset: InfoSet, version: int | None = None) -> StrategyRow | None: ...
>
>     @abstractmethod
>     def put(self, infoset: InfoSet, action_mask: int,
>             action_probs: np.ndarray, version: int) -> None: ...
>
>     @abstractmethod
>     def nearest_neighbor(self, infoset: InfoSet, version: int) -> StrategyRow | None:
>         """Match on (table_size, street, card_bucket, position, stack_bucket),
>         return any row (caller decides how to use). None if no match."""
>
>     @abstractmethod
>     def bulk_put(self, rows: Iterable[tuple[InfoSet, int, np.ndarray]],
>                  version: int) -> None: ...
>
>     @abstractmethod
>     def current_version(self) -> int: ...
>
>     @abstractmethod
>     def set_current_version(self, version: int) -> None: ...
>
> @dataclass(frozen=True)
> class StrategyRow:
>     action_mask: int
>     action_probs: np.ndarray  # float32, len = popcount(action_mask)
>     visit_count: int
>     version: int
>
> class SQLiteStrategyDB(StrategyDB): ...
> class LMDBStrategyDB(StrategyDB): ...
>
> def open_db(uri: str) -> StrategyDB:
>     """uri: 'sqlite:///path.db' or 'lmdb:///path/'. Applies migrations."""
> ```
>
> **Tests CC must write:**
> 1. `test_put_get_roundtrip` — both backends.
> 2. `test_get_missing_returns_none`.
> 3. `test_nearest_neighbor_finds_when_history_differs`.
> 4. `test_bulk_put_atomic` — interrupt mid-bulk-put leaves DB unchanged (transaction).
> 5. `test_version_filter` — same infoset, two versions, `get(version=1)` and `get(version=2)` return different rows.
> 6. `test_migrations_idempotent` — running migrations twice is a no-op.
> 7. `test_action_probs_sum_to_one` — `put` rejects probs that don't sum to 1.0 ± 1e-6.
> 8. `test_sqlite_lmdb_equivalence` — same writes to both produce equivalent reads.

---

## E. Training Pipeline

### Deep CFR hyperparameters (committed)

```python
# training/config.py
@dataclass(frozen=True)
class DeepCFRConfig:
    # Networks
    advantage_hidden: tuple[int, ...] = (256, 256, 256)
    policy_hidden:    tuple[int, ...] = (256, 256, 256)
    activation:       str = "relu"
    layer_norm:       bool = True

    # Optimization
    learning_rate:    float = 1e-3
    optimizer:        str = "adam"
    grad_clip:        float = 1.0
    batch_size:       int = 4096

    # CFR loop
    outer_iters:      int = 1000              # CFR iterations
    traversals_per_iter: int = 1000            # external-sampling traversals per iter
    train_steps_per_iter: int = 4000           # SGD steps on advantage net per iter
    policy_train_steps:   int = 20000          # SGD steps on policy net (final phase)

    # Reservoirs
    advantage_buffer_size: int = 1_000_000
    policy_buffer_size:    int = 1_000_000

    # Discounting (linear CFR)
    cfr_weighting: str = "linear"

    # Multi-player
    num_players_train: tuple[int, ...] = (6, 8, 9)   # rotated per iteration
    seat_randomization: bool = True

    # Determinism
    seed: int = 0xC0FFEE
```

**Why these numbers:**
- 256×3 MLP is the OpenSpiel Deep CFR default that converges on Leduc and HUNL subgames; large enough to fit our infoset distribution, small enough to train on a single GPU.
- 1k iters × 1k traversals × 4k steps = ~4M gradient updates per training run. On an RTX 4090 with batch 4096 this is ~6–10h pure GPU time; the bottleneck will be CPU-side traversals.
- Linear CFR (weight iteration `t` by `t`) converges materially faster than vanilla CFR in practice without the implementation cost of Discounted CFR.
- Layer norm because Deep CFR's regression targets shift across iterations and BN behaves badly with that.

### Self-play loop structure

```
for t in 1..outer_iters:
    table_size = rotate([6, 8, 9])
    for traversal in 1..traversals_per_iter:
        deck = shuffle(seed=hash(t, traversal))
        state = PokerKit.new_hand(table_size, blinds, stacks)
        external_sampling_traversal(state, advantage_nets, t)
            # adds samples to advantage_reservoir
    train_advantage_net(samples=advantage_reservoir, steps=train_steps_per_iter)
    if t % 10 == 0:
        checkpoint(t)

# final policy training
train_policy_net(samples=policy_reservoir, steps=policy_train_steps)
export_strategy_table(policy_net) -> SQLite
```

**External-sampling MCCFR** is the standard Deep CFR sampler — sample one action for the opponents, sample all for the traverser, regret-match at traverser nodes.

### Convergence metric — DECIDED: NashConv via Local Best Response + frozen-opponent win-rate

Exploitability isn't defined for n>2. Two proxies, both tracked:

1. **Local Best Response (LBR)** (Lisý & Bowling 2017): a depth-limited best-responder against the current strategy. Lower LBR-exploitability = closer to equilibrium. Compute every 25 iterations; expensive (~30 min) but the only principled metric we have.
2. **Head-to-head mbb/hand vs frozen checkpoint** (e.g. iter 0, iter 100, iter 500). Plot is the "are we still improving?" signal between LBR evaluations.

**Stop condition:** LBR-exploitability plateau (Δ < 5% over 100 iters) OR compute budget exhausted. The expectation is "compute budget exhausted first" — this is fine; we ship the best checkpoint by LBR.

### Compute budget on a single workstation

Assumed hardware: 16-core CPU, 64GB RAM, 1× RTX 4090 (24GB).

| Budget | Iterations | LBR-eval count | Realistic strength                |
|--------|------------|----------------|-----------------------------------|
| 1 day  | ~80–120    | 3–4            | Weak HU baseline; loses to humans |
| 3 days | ~250–400   | 10–16          | Solid 6-max, weak 9-max           |
| 1 week | ~700–1000  | 28–40          | Strong baseline, ships to v1      |

These ranges (not point estimates) acknowledge that traversal speed varies with how often hands go to showdown — bigger pots = more traversals/sec. Assumption: ~3–5k traversals/sec sustained on the CPU.

### Single shared model vs separate per table size — DECIDED: single shared model

- The CFR networks are conditioned on `table_size` as an input feature.
- Training rotates `table_size ∈ {6, 8, 9}` per outer iteration with stratified seat assignment.
- Rationale: 3 separate models triple training cost. Transfer between table sizes is high (preflop ranges are correlated, postflop principles share). A shared trunk + 3 heads is a future v2 if multi-way play measurably underperforms (§I).

### Checkpoint format and cadence

- **Cadence:** every 10 iterations.
- **Format:** `training/checkpoints/iter_NNNN.pt` containing:
  ```python
  {"iter": int, "advantage_state": state_dict, "policy_state": state_dict,
   "rng_state": ..., "reservoir_advantage": list, "reservoir_policy": list,
   "config": DeepCFRConfig, "git_sha": str, "abstraction_checksums": dict}
  ```
- **Resumable.** `cli.py train --resume training/checkpoints/iter_0500.pt`.
- **Strategy export:** at the end of training, walk all infosets encountered in policy_reservoir, query the policy net, write to SQLite via `strategy_db.bulk_put`. This is a separate step (`cli.py export-strategy --from iter_1000.pt --to db/strategy_v3.db`).

> **CC HANDOFF — Section E**
>
> **Build step:** 5 (Training pipeline)
> **Module:** `src/pokerbot/training/{config.py,deepcfr.py,traversal.py,nets.py,export.py}`
>
> **Public API:**
> ```python
> class Trainer:
>     def __init__(
>         self,
>         config: DeepCFRConfig,
>         game: Game[Any],                # production: SimpleNLHEGame(abstraction, blinds, …)
>         *, device: str = "cpu",
>     ) -> None: ...
>     def train(self, output_dir: Path, resume_from: Path | None = None) -> None: ...
>     def evaluate(self, opponents: list[Path]) -> EvalResult: ...
>     def export_strategy(self, target_db: StrategyDB, version: int) -> None: ...
>
> def local_best_response(
>     policy: PolicyNet, opponent_seats: list[int], depth: int = 2,
> ) -> float:
>     """Returns LBR-exploitability in milli-big-blinds per hand."""
> ```
>
> The Trainer is **game-agnostic** — it talks to its environment through the
> `Game[StateT]` ABC (`new_initial_state`, `is_terminal`, `legal_actions`,
> `apply_action`, `terminal_reward`, `infoset_key`, `infoset_features`). Tests
> use a tiny `KuhnPokerGame` so the CFR loop converges in seconds; production
> training instantiates `SimpleNLHEGame(abstraction=tables, blinds=(SB, BB),
> starting_stack=1000, table_size=6)` and passes it in. The abstraction is a
> SimpleNLHEGame ctor argument, not a Trainer one, because the Trainer never
> needs to know it's solving NLHE.
>
> **Tests CC must write:**
> 1. `test_config_pinned` — DeepCFRConfig() values match this spec exactly (regression guard).
> 2. `test_traversal_single_hand` — one external-sampling traversal on a fixed seed produces expected sample count.
> 3. `test_advantage_net_overfits_tiny_buffer` — sanity check: train on a 100-sample buffer for 1000 steps, MSE < 1e-3.
> 4. `test_checkpoint_resume` — train 5 iters, checkpoint, resume, train 5 more; state_dict matches training 10 iters straight (within numeric tolerance, same seed).
> 5. `test_lbr_better_than_random` — LBR-exploitability of a uniform-random policy > LBR of a trained-50-iter policy.
> 6. `test_export_strategy_writes_all_infosets` — every infoset in policy_reservoir appears in target DB.

---

## F. Runtime Adapter

### JSON input schema (committed v1)

```json
{
  "schema_version": 1,
  "game_type": "cash",
  "table_size": 6,
  "blinds": {"sb": 5, "bb": 10},
  "ante": 0,
  "hero_seat": 2,
  "button_seat": 0,
  "hero_hole": ["As", "Kh"],
  "board": ["7c", "2d", "Jh"],
  "stacks": [1000, 950, 800, 1200, 900, 0],
  "current_bets": [0, 0, 50, 0, 100, 0],
  "pot_committed": 1500,
  "to_call": 50,
  "min_raise": 100,
  "max_raise": 800,
  "action_history": [
    {"seat": 2, "street": 0, "type": "raise", "amount": 30},
    {"seat": 3, "street": 0, "type": "call",  "amount": 30},
    {"seat": 2, "street": 1, "type": "bet",   "amount": 50}
  ]
}
```

### Output schema

```json
{
  "action": "raise",
  "amount": 150,
  "abstract_action": "BET_100",
  "probability_sampled": 0.42,
  "infoset_hash": "9f4a...e1",
  "version": 7,
  "latency_ms": 14,
  "fallback_used": "exact"
}
```

`fallback_used` is one of: `"exact"`, `"nearest_neighbor"`, `"default_policy"`.

### State translation pipeline

```
parse_json  → 1ms     (pydantic v2, strict mode)
build_infoset:
  card_bucket  via AbstractionTables.lookup           → 3-5ms
  history_bytes via encode_history                    → <1ms
  stack_bucket via simple binning                     → <1ms
db_lookup:
  exact hash get                                       → 2-4ms
  nearest_neighbor fallback if miss                    → 5-15ms
sample_action:
  rng-weighted choice over action_probs               → <1ms
translate_outbound:
  resolve_action to chip amount + legality clamp      → <1ms
opponent_model.adjust  (v1: identity)                  → <1ms
serialize_response                                     → <1ms
```

**Latency budget total: <30ms typical, <100ms p99.** The p99 spike comes from the nearest-neighbor scan when a history is unseen — bounded by the size of the compound index range, which is ~50 rows worst case.

### Missing-infoset fallback — DECIDED: nearest-neighbor → default policy

1. **Exact lookup** by `infoset.hash16()`.
2. **Miss → nearest-neighbor**: scan rows matching `(table_size, street, card_bucket, position, stack_bucket)` ignoring history. Pick the row with the highest `visit_count` (most-trained sibling) and use its action_probs.
3. **Still miss → default policy**: hand-rolled heuristic in `runtime/default_policy.py`:
   - Preflop: open-raise top 25% of hands UTG sliding to top 60% on the button (single committed range table); 3-bet top 5%; fold otherwise.
   - Postflop: c-bet 0.66-pot with showdown value + draws; check otherwise; call ≤ 1/4 pot with showdown value.
   - This is a "don't crash, don't bleed too much" floor — not meant to play well. It should fire <0.1% of hands after a full training run.

**Why nearest-neighbor over interpolation:** interpolation requires defining a metric over histories, which we haven't done; nearest-neighbor by (cards, position, stack) is robust and bounded-cost.

### Opponent modeling seam

```python
class OpponentModel(ABC):
    @abstractmethod
    def adjust(
        self,
        infoset: InfoSet,
        base_probs: dict[ActionType, float],
        observed_history: ObservedHistory,
    ) -> dict[ActionType, float]: ...

class IdentityOpponentModel(OpponentModel):
    """v1 default. Returns base_probs unchanged."""
    def adjust(self, infoset, base_probs, observed_history):
        return base_probs
```

The runtime takes an injected `OpponentModel`. v1 ships `IdentityOpponentModel`. v2+ can plug in exploitative models without touching the rest of the runtime.

> **CC HANDOFF — Section F**
>
> **Build step:** 6 (Runtime adapter)
> **Module:** `src/pokerbot/runtime/{adapter.py,schema.py,default_policy.py,opponent.py}`
>
> **Public API:**
> ```python
> from pydantic import BaseModel, ConfigDict, Field
> from typing import Literal
>
> class GameStateRequest(BaseModel):
>     """Pydantic v2 model matching the JSON schema above. `extra='forbid'`."""
>     model_config = ConfigDict(extra="forbid", frozen=True)
>     schema_version: int                = Field(ge=1, le=1)
>     game_type:      Literal["cash", "tournament"]
>     table_size:     Literal[6, 8, 9]   # §C: 6-/8-/9-max only — no heads-up, no 7-max
>     blinds:         BlindsSchema       # {"sb": int>=0, "bb": int>0}
>     # … remaining fields per the JSON example above (hero/board/stacks/action_history)
>
> class ActionResponse(BaseModel):
>     action: Literal["fold", "check", "call", "bet", "raise"]
>     amount: int
>     abstract_action: str
>     probability_sampled: float
>     infoset_hash: str
>     version: int
>     latency_ms: int
>     fallback_used: Literal["exact", "nearest_neighbor", "default_policy"]
>
> class RuntimeAdapter:
>     def __init__(self, db: StrategyDB, abstraction: AbstractionTables,
>                  opponent_model: OpponentModel = IdentityOpponentModel(),
>                  rng_seed: int | None = None) -> None: ...
>     def decide(self, request: GameStateRequest) -> ActionResponse: ...
> ```
>
> **Tests CC must write:**
> 1. `test_known_infoset_round_trip` — put a fake row, query via JSON, get back a legal action.
> 2. `test_unknown_infoset_falls_back_to_nn` — `fallback_used == "nearest_neighbor"`.
> 3. `test_no_neighbors_falls_back_to_default` — empty DB, request returns valid default-policy action.
> 4. `test_latency_p99_under_100ms` — 10k random requests against a populated DB, p99 < 100ms.
> 5. `test_illegal_action_never_emitted` — `BET_150` clamped when stack < 1.5×pot.
> 6. `test_opponent_model_injection` — custom model that overrides to FOLD is respected.
> 7. `test_schema_v1_strict` — extra JSON fields raise validation error.

---

## G. Tournament-Mode Additions

### ICM model — DECIDED: Malmuth-Harville (exact for ≤8 players, MC for 9+)

ICM gives the EV-in-prize-money of a chip stack given remaining stacks and payout structure. Malmuth-Harville is the standard recursive computation; exact computation is O(n!) over remaining players but n ≤ 8 is fine on a CPU in <1ms. For 9-handed tables we use a **Monte Carlo MH approximation** with 1000 samples (still <5ms).

```python
# tournament/icm.py
def icm_equity(
    stacks: list[int],
    payouts: list[int],   # prize $ for finishes 1..len(payouts)
    mc_samples: int = 0,  # 0 = exact; >0 = MC
) -> list[float]:
    """Returns expected prize $ for each seat. Sums to sum(payouts)."""
```

### Push/fold thresholds by stack depth — DECIDED: generate ourselves, do not source externally

For effective stacks ≤ 10BB, we solve a small 2-player Nash push/fold game ourselves and store the resulting ranges as a precomputed table.

```python
# tournament/pushfold.py
PUSHFOLD_RANGES: dict[tuple[int, int], frozenset[int]]
# key: (effective_bb_int, position_id) → set of preflop bucket ids that push
# generated by tournament/build_pushfold.py via a 2-player LP solver on
# the simplified push/fold game tree.
```

**Why generate vs source:** publicly available push/fold charts have provenance and license issues, vary by author, and were computed for HU SNG payout structures that may not match our targets. Building our own — it's a 2-player Nash with a tractable state space (169 × 169 strategy matrices), takes <1 minute to solve. Reproducible, ours.

For 10–15BB we use a hybrid: push/fold ranges as a *prior* on the cash strategy, blended via softmax temperature that decays with stack depth.

### Layered on cash strategy vs separately trained — DECIDED: separately fine-tuned from cash baseline

- Cash baseline trains first (§E).
- **Tournament fine-tune** = continue training from the final cash checkpoint with a modified reward function: instead of chips, the terminal reward is `Δ ICM-equity in $`. Same network architecture, same loop, fewer iterations (~200 outer, ~3 days of compute).
- This gives a tournament-aware policy without throwing away cash-game learning. Push/fold table is a separate runtime lookup that overrides the policy net when `effective_stack_bb ≤ 10` (configurable threshold).

The runtime adapter takes `game_type: "cash" | "tournament"` and routes accordingly:

```
if game_type == "cash":          use cash strategy DB
elif game_type == "tournament":
    if eff_stack_bb <= 10:        use PUSHFOLD_RANGES
    else:                          use tournament strategy DB (ICM-fine-tuned)
```

> **CC HANDOFF — Section G**
>
> **Build step:** 7 (Tournament additions)
> **Module:** `src/pokerbot/tournament/{icm.py,pushfold.py,reward.py,build_pushfold.py}`
>
> **Public API:**
> ```python
> def icm_equity(stacks: list[int], payouts: list[int], mc_samples: int = 0) -> list[float]: ...
>
> def icm_reward(
>     final_stacks: list[int], initial_stacks: list[int], payouts: list[int],
> ) -> list[float]:
>     """Per-seat reward = (ICM_after - ICM_before) / bb_value. For training."""
>
> def pushfold_action(
>     hole: tuple[int, int], position: int, eff_stack_bb: int, table_size: int,
> ) -> ActionType:
>     """Returns FOLD or ALL_IN. Raises if eff_stack_bb > 10."""
>
> def build_pushfold_table(output_path: Path) -> None:
>     """Run 2-player Nash solver, write PUSHFOLD_RANGES dump."""
> ```
>
> **Tests CC must write:**
> 1. `test_icm_two_player_known` — equal stacks, [70, 30] payout → each seat expects $50.
> 2. `test_icm_three_player_chip_chip_leader` — chip leader's $ < chip share (ICM pressure).
> 3. `test_icm_mc_converges_to_exact` — 8-player MC with 10k samples matches exact within 1%.
> 4. `test_pushfold_tight_under_gun` — UTG 5BB pushes only top ~15% of hands.
> 5. `test_pushfold_wide_button` — Button 5BB pushes top ~50%.
> 6. `test_icm_reward_zero_sum` — sum across seats == 0.

---

## H. Open Questions and Risks

### What CC will need from the championship interface to finalize

1. **Exact JSON schema** the championship uses. Our §F schema is our v1 best guess; the real one almost certainly differs in field names, card encoding, and action representation. Build an adapter layer (`runtime/championship_adapter.py`) that maps championship JSON → our `GameStateRequest`. Stub it for now; fill it in once the spec lands.
2. **Latency SLA** beyond "fast." We targeted <100ms p99; if the real budget is 50ms or 200ms it changes whether nearest-neighbor fallback is viable as-is.
3. **Concurrency** — single bot per process, or do we share one DB across N concurrent bots? LMDB read-mmap handles concurrent reads fine; SQLite WAL mode does too but worse. **Default to LMDB in production deployment if N > 32 concurrent bots.**
4. **Whether the championship interface gives us opponent-action history across hands** (for opponent modeling) or just per-hand. We assume per-hand in v1.

### Where the strategy will be weakest

- **9-max early-stage play**, especially limped multi-way pots. Our abstraction collapses too many hands postflop multi-way.
- **River overbets** with a polarized range against thinking opponents. The 1.5×pot bucket is the cap; real GTO sometimes wants 2×+.
- **Short stack play (10–25BB) outside push/fold territory** — the boundary between "use cash strategy" and "use push/fold table" is fuzzy and will produce visible seams in play.
- **Adaptive opponents**. v1 has no opponent modeling. Anyone who notices we always 0.66-pot c-bet on dry boards will exploit this.

### What's most likely to fail at scale

1. **Convergence in 9-max.** Deep CFR on >3 players doesn't have hard guarantees, and 9-handed has the biggest tree. If LBR-exploitability stops improving at 200 iters in 9-max, we may need to either: (a) train 6-max separately and use it for all sizes (worse 9-max but stable), or (b) increase advantage net capacity to 512×4.
2. **Reservoir buffer overflow with low-quality samples.** If early-iteration policy is so bad that buffer fills with garbage, learning stalls. Mitigation: warm-start advantage net with supervised pretraining on a uniform policy.
3. **Bucket boundary artifacts.** Two near-identical hands ending up in different flop buckets and getting wildly different action probs. Mitigation: at runtime, blend the top-2 nearest buckets if their centroid distance is close.
4. **SQLite write contention during `bulk_put`.** 1M rows is fine; 20M might need batched transactions + `PRAGMA synchronous=NORMAL` during writes only.

### Decisions most likely to need revision after step 5's first training run

| Decision | Trigger to revisit |
|---|---|
| Bucket counts (200 postflop) | If LBR-exploitability plateaus high and bucket-boundary regret is large in inspection. Go to 500. |
| Single shared model across table sizes | If 9-max underperforms 6-max by >50 mbb/hand vs frozen baseline. Split into shared trunk + 3 heads. |
| Action set (7 postflop) | If solver-derived hand histories show consistent off-tree bet-sizing exploitation. Add 0.25 and 2.0 pot. |
| Pseudo-harmonic translation | If empirical exploit testing shows boundary attack; switch to randomized-by-default for all translations. |
| Fine-tune (vs from-scratch) for tournament | If ICM fine-tune diverges (catastrophic forgetting), train tournament from scratch with ICM reward. |

---

## I. Build-Order Compatibility Check

### Dependency map

| CC Step | Depends on Spec Sections | Notes |
|---------|--------------------------|-------|
| 1. Scaffolding | — | pyproject.toml, ruff config, mypy strict, pytest. No spec dependency. |
| 2. Card abstraction | A, C | Needs `InfoSet` type from C for downstream test signatures; needs A for bucket logic. |
| 3. Action abstraction | B, C | C is shared. |
| 4. Strategy DB | C, D | C defines the key; D defines the schema and interface. |
| 5. Training | A, B, C, E | All abstractions plus E for the loop. Uses StrategyDB (step 4) for export only — declared as a dependency interface, not a write target until export phase. |
| 6. Runtime adapter | A, B, C, D, F | Reads from DB; uses abstractions to build infoset; F defines I/O. |
| 7. Tournament | C, D, E (config), G | ICM in pure-Python; pushfold solver uses card abstraction (A) for the 169 preflop classes; fine-tune uses §E pipeline with G's reward. |
| 8. CLI | All | Thin wrapper. |

### Circular dependencies

**None.** Every section is decided in this spec; no spec section says "decide this after step N."

The one subtle point: **step 5 (training) needs step 4 (StrategyDB) only at the export phase**, not during the CFR loop. Reservoirs and nets live in-process; only the final `export_strategy` call hits the DB. CC can build step 5 with a no-op StrategyDB stub if step 4 isn't ready, then wire the real one in.

Similarly, **step 6 (runtime) does not depend on step 5 (training)**. The runtime just reads a populated DB. CC can build and test the runtime against a hand-written tiny DB before training finishes — this is encouraged.

### Decisions CC will likely revisit after early training

These are flagged in §H. They are all *parameters*, not *interfaces* — the public APIs in this spec stay stable across these revisions:
- Bucket count (changes a constant, requires rebuilding abstraction NPZ files; runtime API unchanged).
- Action set (adds entries to `ActionType` enum; `action_mask` already supports it).
- Single vs split model (training-internal; DB schema and runtime unchanged).
- Translation algorithm (internal to `translate_bet`).

**No revision in §H requires breaking changes to a public API in another step.** This is the property we needed.

---

*End of spec. Total committed decisions: 47. Open questions: 4. Build is unblocked.*
