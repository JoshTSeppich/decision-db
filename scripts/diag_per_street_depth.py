"""Diagnostic: per-street decision-point depth in external-sampling MCCFR traversals.

Question: how many advantage samples (traverser decisions) and policy samples
(opponent decisions) does ONE traversal generate per street? The pilot DB's
postflop coverage is sparse — we want to know whether the bottleneck is

    (a) traversals not reaching postflop often enough (algorithmic), or
    (b) the postflop state space being inherently larger than preflop's, so
        the same number of writes-per-street still gives poor coverage.

This script wraps `external_sampling_traversal` with per-street counters and
runs N traversals on the production NLHE 6-max game + abstraction. Reports
mean / median / p95 of advantage- and policy-samples by street, plus the
ratio of "policy samples per street" to "unique policy samples per street"
(approximation — actually counts non-distinct samples since uniqueness is
expensive across a Reservoir, but it's the relevant metric for write
volume into the strategy DB).

Compares both:
    - Untrained nets (uniform regret_match → ~uniform action probs); this
      is the BREADTH-optimal exploration pattern that approximates iter-1
      of training.
    - Loaded pilot checkpoint (final iter 250) — what actually trained on.

Usage:
    python scripts/diag_per_street_depth.py [--n-traversals 1000]
                                            [--checkpoint training/pilot/iter_0250.pt]
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from pokerbot.abstraction import AbstractionTables
from pokerbot.training import SimpleNLHEGame
from pokerbot.training.nets import AdvantageNet, regret_match

if TYPE_CHECKING:
    from pokerbot.training.config import DeepCFRConfig
    from pokerbot.training.nlhe_game import NLHEState


_STREET_NAMES = ("preflop", "flop", "turn", "river")


@dataclass(slots=True)
class StreetCounters:
    """Per-traversal counts of advantage + policy samples, indexed by street."""

    advantage: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    policy: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def reset(self) -> None:
        self.advantage[:] = [0] * 4
        self.policy[:] = [0] * 4


# ───────── instrumented traversal (mirrors external_sampling_traversal) ─────────


def _instrumented_traversal(
    game: SimpleNLHEGame,
    state: NLHEState,
    traverser: int,
    advantage_nets: list[torch.nn.Module],
    rng: random.Random,
    counters: StreetCounters,
) -> float:
    if game.is_terminal(state):
        return float(game.terminal_reward(state).rewards[traverser])

    actor = game.current_player(state)
    street_idx = int(state.pk_state.street_index)
    features = game.infoset_features(state)
    legal = game.legal_actions(state)
    if not legal:
        raise RuntimeError("no legal actions")
    mask = torch.zeros(game.num_actions, dtype=torch.float32)
    for a in legal:
        mask[a] = 1.0

    with torch.no_grad():
        advantages = advantage_nets[actor](features.unsqueeze(0)).squeeze(0)
    strategy = regret_match(advantages.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

    if actor == traverser:
        counters.advantage[street_idx] += 1
        for a in legal:
            child = game.apply_action(state, a, rng)
            _instrumented_traversal(game, child, traverser, advantage_nets, rng, counters)
        return 0.0  # we don't need the actual EV for this diagnostic

    counters.policy[street_idx] += 1
    probs = [float(strategy[a].item()) for a in legal]
    total = sum(probs)
    sampled = rng.choice(legal) if total <= 0 else rng.choices(legal, weights=probs, k=1)[0]
    child = game.apply_action(state, sampled, rng)
    return _instrumented_traversal(game, child, traverser, advantage_nets, rng, counters)


# ───────── setup helpers ─────────


def _build_untrained_nets(game: SimpleNLHEGame) -> tuple[list[torch.nn.Module], DeepCFRConfig]:
    """Build per-player AdvantageNets with the same architecture as the pilot."""
    from pokerbot.training.config import DeepCFRConfig

    config = DeepCFRConfig(
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
        AdvantageNet(game.feature_dim, game.num_actions, config) for _ in range(game.num_players)
    ]
    for net in nets:
        net.eval()
    return nets, config


def _load_pilot_nets(
    game: SimpleNLHEGame, checkpoint_path: Path
) -> tuple[list[torch.nn.Module], DeepCFRConfig]:
    """Load the final pilot checkpoint and return advantage nets + config."""
    nets, config = _build_untrained_nets(game)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for net, state in zip(nets, ckpt["advantage_states"], strict=True):
        net.load_state_dict(state)
        net.eval()
    return nets, config


# ───────── run + report ─────────


def _run_traversals(
    game: SimpleNLHEGame,
    advantage_nets: list[torch.nn.Module],
    n_traversals: int,
    rng: random.Random,
) -> tuple[list[StreetCounters], dict[bytes, int]]:
    """Return per-traversal counters and a frequency table of policy infoset keys.

    The frequency table is what tells us how many UNIQUE infosets get
    visited (across all `n_traversals` traversals) per street — that's
    what bounds the strategy-DB row count.
    """
    per_traversal: list[StreetCounters] = []
    unique_policy_keys_by_street: list[set[bytes]] = [set() for _ in range(4)]

    for _ in range(n_traversals):
        traverser = rng.randrange(game.num_players)
        state = game.new_initial_state(rng)
        counters = StreetCounters()
        # Walk the same tree but also record (street, infoset_key) for policy nodes.
        _walk_with_key_capture(
            game,
            state,
            traverser,
            advantage_nets,
            rng,
            counters,
            unique_policy_keys_by_street,
        )
        per_traversal.append(counters)
    flat_counts: dict[bytes, int] = {
        _STREET_NAMES[s].encode(): len(unique_policy_keys_by_street[s]) for s in range(4)
    }
    return per_traversal, flat_counts


def _walk_with_key_capture(
    game: SimpleNLHEGame,
    state: NLHEState,
    traverser: int,
    advantage_nets: list[torch.nn.Module],
    rng: random.Random,
    counters: StreetCounters,
    unique_policy_keys: list[set[bytes]],
) -> float:
    if game.is_terminal(state):
        return float(game.terminal_reward(state).rewards[traverser])
    actor = game.current_player(state)
    street_idx = int(state.pk_state.street_index)
    features = game.infoset_features(state)
    legal = game.legal_actions(state)
    if not legal:
        raise RuntimeError("no legal actions")
    mask = torch.zeros(game.num_actions, dtype=torch.float32)
    for a in legal:
        mask[a] = 1.0
    with torch.no_grad():
        advantages = advantage_nets[actor](features.unsqueeze(0)).squeeze(0)
    strategy = regret_match(advantages.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)

    if actor == traverser:
        counters.advantage[street_idx] += 1
        for a in legal:
            child = game.apply_action(state, a, rng)
            _walk_with_key_capture(
                game, child, traverser, advantage_nets, rng, counters, unique_policy_keys
            )
        return 0.0

    counters.policy[street_idx] += 1
    unique_policy_keys[street_idx].add(game.infoset_key(state))
    probs = [float(strategy[a].item()) for a in legal]
    total = sum(probs)
    sampled = rng.choice(legal) if total <= 0 else rng.choices(legal, weights=probs, k=1)[0]
    child = game.apply_action(state, sampled, rng)
    return _walk_with_key_capture(
        game, child, traverser, advantage_nets, rng, counters, unique_policy_keys
    )


def _pct(arr: list[int], p: float) -> int:
    if not arr:
        return 0
    s = sorted(arr)
    idx = int(p * (len(s) - 1))
    return s[idx]


def _report(
    label: str, per_traversal: list[StreetCounters], unique_counts: dict[bytes, int]
) -> None:
    print(f"\n══════════ {label} ══════════")
    n = len(per_traversal)
    if n == 0:
        print("  (no traversals)")
        return

    print(f"  {n} traversals")
    print(
        f"  {'street':<8s}  {'mean adv':>10s}  {'med':>5s}  {'p95':>5s}  "
        f"{'mean pol':>10s}  {'med':>5s}  {'p95':>5s}  {'unique pol':>11s}"
    )
    for s in range(4):
        adv_vals = [c.advantage[s] for c in per_traversal]
        pol_vals = [c.policy[s] for c in per_traversal]
        u = unique_counts.get(_STREET_NAMES[s].encode(), 0)
        print(
            f"  {_STREET_NAMES[s]:<8s}  "
            f"{np.mean(adv_vals):>10.2f}  "
            f"{_pct(adv_vals, 0.5):>5d}  "
            f"{_pct(adv_vals, 0.95):>5d}  "
            f"{np.mean(pol_vals):>10.2f}  "
            f"{_pct(pol_vals, 0.5):>5d}  "
            f"{_pct(pol_vals, 0.95):>5d}  "
            f"{u:>11d}"
        )

    total_adv = sum(sum(c.advantage) for c in per_traversal)
    total_pol = sum(sum(c.policy) for c in per_traversal)
    print(
        f"\n  totals: {total_adv} advantage samples, {total_pol} policy samples ({total_pol / n:.1f}/traversal avg)"
    )
    # Per-street total policy samples (write volume into reservoir before reservoir-sampling decimates)
    per_street_pol = [sum(c.policy[s] for c in per_traversal) for s in range(4)]
    print(f"  per-street policy samples: {dict(zip(_STREET_NAMES, per_street_pol, strict=True))}")


def _project_to_pilot_scale(
    per_traversal: list[StreetCounters], pilot_traversals: int = 100_000
) -> None:
    """Extrapolate the per-traversal counts to the pilot's 100k traversals."""
    n = len(per_traversal)
    if n == 0:
        return
    scale = pilot_traversals / n
    print(f"\n  projection to {pilot_traversals:,} traversals (pilot scale):")
    print(f"    {'street':<8s}  {'proj adv samples':>16s}  {'proj pol samples':>16s}")
    for s in range(4):
        adv_total = sum(c.advantage[s] for c in per_traversal)
        pol_total = sum(c.policy[s] for c in per_traversal)
        print(
            f"    {_STREET_NAMES[s]:<8s}  "
            f"{int(adv_total * scale):>16,d}  "
            f"{int(pol_total * scale):>16,d}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-traversals", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--abstraction-dir", default="abstraction")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("training/pilot/iter_0250.pt"),
        help="If exists, also run a pass with the loaded pilot nets",
    )
    args = p.parse_args()

    print(f"Loading abstraction from {args.abstraction_dir!r}…")
    tables = AbstractionTables(path=args.abstraction_dir)
    game = SimpleNLHEGame(tables, table_size=6)
    print(f"  loaded streets: {tables.loaded_streets}")

    # ── untrained pass ──
    nets_untrained, _cfg = _build_untrained_nets(game)
    print(f"\nRunning {args.n_traversals} untrained-net traversals…")
    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    per_t_u, uniques_u = _run_traversals(game, nets_untrained, args.n_traversals, rng)
    print(f"  elapsed {time.perf_counter() - t0:.1f}s")
    _report("UNTRAINED nets", per_t_u, uniques_u)
    _project_to_pilot_scale(per_t_u)

    # ── trained pass (if checkpoint available) ──
    if args.checkpoint.exists():
        print(f"\nLoading pilot checkpoint {args.checkpoint}…")
        nets_pilot, _cfg = _load_pilot_nets(game, args.checkpoint)
        # smaller N so we don't wait too long; the pilot nets concentrate exploration
        n_trained = max(200, args.n_traversals // 5)
        print(f"Running {n_trained} pilot-trained-net traversals (subset to save time)…")
        rng2 = random.Random(args.seed ^ 0xBEEF)
        t0 = time.perf_counter()
        per_t_p, uniques_p = _run_traversals(game, nets_pilot, n_trained, rng2)
        print(f"  elapsed {time.perf_counter() - t0:.1f}s")
        _report("PILOT-TRAINED nets (iter 250)", per_t_p, uniques_p)
        _project_to_pilot_scale(per_t_p)
    else:
        print(f"\n(no pilot checkpoint at {args.checkpoint} — skipping trained-net pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
