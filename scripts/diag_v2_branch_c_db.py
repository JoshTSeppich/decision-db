"""Branch C diagnostics #1 + #2 for strategy-pilot-v2.db.

#1: Sample 50 random rows from the v2 DB. Print (action_mask, action_probs)
    per row and report:
      - % one-hot   (max prob ≥ 0.95)
      - % near-uniform (max prob ≤ 0.4 — uniform over k legal actions has
        max prob = 1/k, so this is roughly "no strong preference")
      - distribution of policy entropy (in nats), summarized as min/median/p95/max
        and per-quartile counts

#2: Spot-check the 20 sample reservoir entries' features through their
    seat's advantage net (regret-matched) and compare to the stored
    policy targets (= what the policy net was trained against, very close
    to what got exported as action_probs).

Loads `training/pilot-v2/iter_0250.pt` for the advantage nets and the
policy reservoir's features.
"""

from __future__ import annotations

import argparse
import math
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pokerbot.training.nets import AdvantageNet, regret_match


def _print_section(title: str) -> None:
    print(f"\n══════════ {title} ══════════")


def _action_name(bit: int) -> str:
    names = [
        "FOLD",
        "CHECK_CALL",
        "BET_33",
        "BET_66",
        "BET_100",
        "BET_150",
        "ALL_IN",
        "RAISE_2_5X",
        "RAISE_3_5X",
    ]
    return names[bit] if 0 <= bit < len(names) else f"BIT_{bit}"


def _unpack_probs(action_mask: int, blob: bytes) -> np.ndarray:
    expected = int(action_mask).bit_count()
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if len(arr) != expected:
        raise ValueError(f"length mismatch: {len(arr)} vs popcount={expected}")
    return arr


def _entropy_nats(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _format_row(mask: int, probs: np.ndarray) -> str:
    parts: list[str] = []
    idx = 0
    for bit in range(mask.bit_length() + 1):
        if mask & (1 << bit):
            parts.append(f"{_action_name(bit)}={probs[idx]:.3f}")
            idx += 1
    return f"mask={mask:#05x}  " + ", ".join(parts)


# ───────── DB row sampling ─────────


def _sample_db_rows(db_path: Path, n: int, rng: random.Random) -> list[tuple[int, np.ndarray]]:
    conn = sqlite3.connect(str(db_path))
    try:
        (total,) = conn.execute("SELECT COUNT(*) FROM strategy WHERE version=1").fetchone()
        if total == 0:
            return []
        offsets = sorted(rng.sample(range(total), min(n, total)))
        rows: list[tuple[int, np.ndarray]] = []
        for off in offsets:
            cur = conn.execute(
                "SELECT action_mask, action_probs FROM strategy WHERE version=1 LIMIT 1 OFFSET ?",
                (off,),
            )
            row = cur.fetchone()
            if row is None:
                continue
            mask = int(row[0])
            probs = _unpack_probs(mask, bytes(row[1]))
            rows.append((mask, probs))
        return rows
    finally:
        conn.close()


def _report_db_sample(rows: list[tuple[int, np.ndarray]]) -> None:
    _print_section(f"Diag #1: {len(rows)} random rows from strategy-pilot-v2.db")
    entropies: list[float] = []
    max_probs: list[float] = []
    one_hot = 0
    near_uniform = 0
    for i, (mask, probs) in enumerate(rows, start=1):
        ent = _entropy_nats(probs)
        max_p = float(probs.max()) if probs.size else 0.0
        entropies.append(ent)
        max_probs.append(max_p)
        if max_p >= 0.95:
            one_hot += 1
        if max_p <= 0.4:
            near_uniform += 1
        print(f"  [{i:>2}] H={ent:.3f}  max_p={max_p:.3f}  {_format_row(mask, probs)}")

    n = len(rows)
    ent_arr = np.asarray(entropies)
    print("\n  ── summary ──")
    print(f"    one-hot  (max p ≥ 0.95):    {one_hot:>2d} / {n}  ({one_hot / n:.0%})")
    print(f"    near-uniform (max p ≤ 0.4): {near_uniform:>2d} / {n}  ({near_uniform / n:.0%})")
    print(
        f"    entropy  min={ent_arr.min():.3f}  "
        f"median={np.median(ent_arr):.3f}  "
        f"p95={np.percentile(ent_arr, 95):.3f}  "
        f"max={ent_arr.max():.3f}"
    )
    # Bucket entropy: < 0.5 = peaky, 0.5-1.0 = mixed, > 1.0 = high-mix
    peaky = int((ent_arr < 0.5).sum())
    mixed = int(((ent_arr >= 0.5) & (ent_arr < 1.0)).sum())
    spread = int((ent_arr >= 1.0).sum())
    print(
        f"    entropy buckets:  "
        f"peaky H<0.5: {peaky} ({peaky / n:.0%})  "
        f"mixed 0.5-1: {mixed} ({mixed / n:.0%})  "
        f"spread H≥1: {spread} ({spread / n:.0%})"
    )
    # Anything weird?
    flags: list[str] = []
    if one_hot / n > 0.25:
        flags.append(f">25% one-hot rows ({one_hot / n:.0%}) — possible policy collapse")
    if near_uniform / n > 0.5:
        flags.append(
            f">50% near-uniform rows ({near_uniform / n:.0%}) — policy may have undertrained"
        )
    if ent_arr.min() == 0.0:
        n_zero = int((ent_arr == 0.0).sum())
        flags.append(f"{n_zero} row(s) have H=0 exactly (pure one-hot)")
    if flags:
        print("\n    ⚠ FLAGS:")
        for f in flags:
            print(f"      - {f}")
    else:
        print("\n    no anomalies flagged")


# ───────── Advantage-net spot check ─────────


def _load_advantage_nets(
    ckpt_path: Path, feature_dim: int, num_actions: int
) -> tuple[list[torch.nn.Module], dict[str, Any]]:
    from pokerbot.training.config import DeepCFRConfig

    # Match pilot v2 config (architecture only matters here)
    cfg = DeepCFRConfig(
        outer_iters=1,
        traversals_per_iter=1,
        train_steps_per_iter=1,
        policy_train_steps=1,
        batch_size=1,
        advantage_hidden=(256, 256, 256),
        policy_hidden=(256, 256, 256),
        advantage_buffer_size=1,
        policy_buffer_size=1,
        checkpoint_every=1,
        seed=0,
    )
    nets: list[torch.nn.Module] = [
        AdvantageNet(feature_dim, num_actions, cfg) for _ in range(6)
    ]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for net, state in zip(nets, ckpt["advantage_states"], strict=True):
        net.load_state_dict(state)
        net.eval()
    return nets, ckpt["policy_reservoir"]


def _net_strategy(net: torch.nn.Module, features: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Run advantage net + regret_match → action distribution (full 9-vector)."""
    feats_t = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
    mask_t = torch.from_numpy(legal_mask.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        adv = net(feats_t)
    return regret_match(adv, mask_t).squeeze(0).numpy()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def _spot_check_advantage_vs_db(
    db_path: Path,
    ckpt_path: Path,
    n_samples: int,
    rng: random.Random,
) -> None:
    _print_section(f"Diag #2: advantage net vs DB policy probs on {n_samples} reservoir entries")
    # Load the advantage nets + the reservoir state
    nets, reservoir = _load_advantage_nets(ckpt_path, feature_dim=72, num_actions=9)
    keys: list[bytes] = reservoir["infoset_keys"]
    features_all: np.ndarray = reservoir["features"]
    masks_all: np.ndarray = reservoir["masks"]
    targets_all: np.ndarray = reservoir["targets"]
    size = int(reservoir["size"])

    sample_idx = sorted(rng.sample(range(size), min(n_samples, size)))

    # Open DB for matching infoset lookups
    conn = sqlite3.connect(str(db_path))
    try:
        pearson_per: list[float] = []
        l1_per: list[float] = []
        kl_advantage_vs_db: list[float] = []
        for n, idx in enumerate(sample_idx, start=1):
            key = keys[idx]
            features = features_all[idx]
            mask = masks_all[idx]
            target = targets_all[idx]
            if len(key) < 4:
                continue
            position = key[2]
            net = nets[position % len(nets)]
            adv_strategy = _net_strategy(net, features, mask)
            adv_strategy_masked = adv_strategy * mask
            s = adv_strategy_masked.sum()
            if s > 0:
                adv_strategy_masked = adv_strategy_masked / s

            # Look up DB row for this infoset_hash
            import hashlib

            hash16 = hashlib.blake2b(key, digest_size=16).digest()
            cur = conn.execute(
                "SELECT action_mask, action_probs FROM strategy "
                "WHERE infoset_hash = ? AND version = 1",
                (hash16,),
            )
            row = cur.fetchone()
            if row is None:
                print(f"  [{n:>2}] pos={position} key={key.hex()[:20]}… DB MISS (skipping)")
                continue
            db_mask = int(row[0])
            db_probs = _unpack_probs(db_mask, bytes(row[1]))
            # Expand db_probs to a 9-vector aligned by action bit
            db_full = np.zeros(9, dtype=np.float32)
            j = 0
            for bit in range(9):
                if db_mask & (1 << bit):
                    db_full[bit] = db_probs[j]
                    j += 1

            # Restrict comparison to legal actions
            legal = mask.astype(bool)
            adv_legal = adv_strategy_masked[legal]
            db_legal = db_full[legal]
            target_legal = target[legal]
            pearson = _pearson(adv_legal, db_legal)
            l1 = _l1(adv_legal, db_legal)
            # KL(db || adv) for additional insight
            eps = 1e-8
            kl = float(np.sum(db_legal * (np.log(db_legal + eps) - np.log(adv_legal + eps))))
            pearson_per.append(pearson)
            l1_per.append(l1)
            kl_advantage_vs_db.append(kl)

            print(
                f"  [{n:>2}] pos={position} street={key[1]} mask={db_mask:#05x}  "
                f"pearson(adv,db)={pearson:+.3f}  L1={l1:.3f}  KL(db||adv)={kl:.3f}"
            )
            print(
                "        adv:    "
                + ", ".join(
                    f"{_action_name(bit)}={adv_strategy_masked[bit]:.3f}"
                    for bit in range(9)
                    if mask[bit]
                )
            )
            print(
                "        db:     "
                + ", ".join(
                    f"{_action_name(bit)}={db_full[bit]:.3f}" for bit in range(9) if mask[bit]
                )
            )
            print(
                "        target: "
                + ", ".join(
                    f"{_action_name(bit)}={target_legal[i]:.3f}"
                    for i, bit in enumerate([b for b in range(9) if mask[b]])
                )
            )

        print("\n  ── summary ──")
        if pearson_per:
            valid = [p for p in pearson_per if not math.isnan(p)]
            print(
                f"    pearson(adv_strategy, db_probs):  "
                f"mean={np.mean(valid):+.3f}  median={np.median(valid):+.3f}  "
                f"min={min(valid):+.3f}  max={max(valid):+.3f}  "
                f"(n={len(valid)} valid of {len(pearson_per)})"
            )
            print(
                f"    L1(adv, db):                       "
                f"mean={np.mean(l1_per):.3f}  median={np.median(l1_per):.3f}  "
                f"min={min(l1_per):.3f}  max={max(l1_per):.3f}"
            )
            print(
                f"    KL(db || adv):                     "
                f"mean={np.mean(kl_advantage_vs_db):.3f}  median={np.median(kl_advantage_vs_db):.3f}"
            )
            # Flag if correlation is too low
            if np.mean(valid) < 0.3:
                print(
                    "\n    ⚠ FLAG: mean Pearson < 0.3 — advantage and policy disagree more "
                    "than expected. The policy net may have learned something quite different "
                    "from the current advantage net (which is fine since they're trained on "
                    "different targets), but extreme disagreement warrants a look."
                )
        else:
            print("    no valid samples (all DB-misses?)")
    finally:
        conn.close()


# ───────── main ─────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="strategy-pilot-v2.db")
    p.add_argument("--checkpoint", type=Path, default=Path("training/pilot-v2/iter_0250.pt"))
    p.add_argument("--n-rows", type=int, default=50, help="rows to sample for diag #1")
    p.add_argument("--n-spot-check", type=int, default=20, help="reservoir samples for diag #2")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found at {args.checkpoint}", file=sys.stderr)
        return 2

    print(f"DB:         {db_path}")
    print(f"checkpoint: {args.checkpoint.resolve()}")

    rng = random.Random(args.seed)
    rows = _sample_db_rows(db_path, args.n_rows, rng)
    _report_db_sample(rows)

    rng2 = random.Random(args.seed ^ 0xCAFE)
    _spot_check_advantage_vs_db(db_path, args.checkpoint, args.n_spot_check, rng2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
