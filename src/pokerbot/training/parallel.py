"""Fan-out / merge parallel engine for the Deep CFR self-play traversal phase.

Mirrors the validated zoom/train/parallel_traversal.py engine but drives the
from-scratch self-play trainer (``pokerbot.training.deepcfr.Trainer``):

  * No scripted-opponent pool — opponents play on-policy via the advantage nets
    *inside* ``external_sampling_traversal`` (self-play MCCFR), so there is no
    ``pool`` argument to thread through.
  * The per-(street × facing-bet) coverage instrument (Piece 2) is threaded
    through each traversal and merged back, so ``num_workers > 1`` reproduces the
    single-process visit-depth readout exactly.

Safety model (identical to the validated zoom engine):
  1. each traversal draws from its OWN RNG derived from
     ``(master_seed, iter_t, index)`` — traversals are independent of one another
     and of the executor;
  2. workers collect samples into a private ``CollectingBuffer`` that IGNORES the
     eviction RNG — no shared reservoir is ever touched inside a worker;
  3. the main process replays the collected samples into the shared reservoirs in
     a fixed ``(index, DFS)`` order with ONE dedicated ``merge_rng`` — the sole
     writer to the reservoirs.

Therefore ``num_workers=1`` (in-process serial reference) and ``num_workers=N``
produce **bit-identical reservoirs** AND **bit-identical merged coverage
counters** — proven by tests/test_deepcfr_parallel.py.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from pokerbot.training.traversal import TraversalStats, external_sampling_traversal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pokerbot.training.game import Game
    from pokerbot.training.traversal import Reservoir

__all__ = [
    "CollectingBuffer",
    "TraversalResult",
    "cfr_iteration",
    "derive_seed",
    "make_worker_pool",
    "merge_coverage",
    "merge_results",
    "run_one_traversal",
]

# A reservoir-ready sample: (features, mask, target, iter_weight, infoset_key).
# Stored as numpy (not torch) so it pickles cheaply across the spawn boundary.
Sample = tuple[np.ndarray, np.ndarray, np.ndarray, float, bytes]


def derive_seed(master_seed: int, iter_t: int, tag: int | str) -> int:
    """Stable, process-independent seed for ``(master_seed, iter_t, tag)``.

    Uses sha256 rather than the builtin ``hash`` because ``hash`` of a str is
    randomized per process (PYTHONHASHSEED), which would make spawned workers
    disagree with the main process and break reproducibility.
    """
    digest = hashlib.sha256(f"{master_seed}:{iter_t}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class CollectingBuffer:
    """Private per-traversal sink, duck-typed to ``Reservoir``.

    Implements the exact ``.add`` signature the traversal calls, but appends the
    sample to a list and IGNORES ``rng`` — no reservoir-sampling eviction happens
    here. That is what keeps the eviction RNG out of the per-traversal stream so
    each traversal stays independent of every other one.
    """

    def __init__(self) -> None:
        self.samples: list[Sample] = []

    def add(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        iter_weight: float,
        rng: random.Random,  # noqa: ARG002 — intentionally ignored (no eviction in workers)
        infoset_key: bytes = b"",
    ) -> None:
        self.samples.append(
            (
                features.detach().cpu().numpy(),
                mask.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                float(iter_weight),
                infoset_key,
            )
        )

    def __len__(self) -> int:
        return len(self.samples)


@dataclass
class TraversalResult:
    """One traversal's private output, tagged for deterministic merge ordering."""

    index: int
    traverser: int
    advantage_samples: list[Sample]
    policy_samples: list[Sample]
    # Coverage (Piece 2) — None when instrumentation is off (zero overhead).
    region_visits: dict[tuple[int, int], Counter[bytes]] | None
    advantage_count: int
    policy_count: int
    terminal_count: int


def run_one_traversal(
    game: Game[Any],
    nets: Sequence[torch.nn.Module],
    iter_t: int,
    index: int,
    master_seed: int,
    opp_aggression_bias: float,
    coverage: bool,
) -> TraversalResult:
    """Run a single self-play traversal into private buffers. Pure (no shared state)."""
    rng = random.Random(derive_seed(master_seed, iter_t, index))
    traverser = rng.randrange(game.num_players)
    state = game.new_initial_state(rng)
    adv_buf = CollectingBuffer()
    pol_buf = CollectingBuffer()
    stats = TraversalStats() if coverage else None
    external_sampling_traversal(
        game=game,
        state=state,
        traverser=traverser,
        advantage_nets=nets,
        advantage_reservoir=adv_buf,  # type: ignore[arg-type]
        policy_reservoir=pol_buf,  # type: ignore[arg-type]
        iter_t=iter_t,
        rng=rng,
        stats=stats,
        opp_aggression_bias=opp_aggression_bias,
    )
    return TraversalResult(
        index=index,
        traverser=traverser,
        advantage_samples=adv_buf.samples,
        policy_samples=pol_buf.samples,
        region_visits=(stats.region_visits if stats is not None else None),
        advantage_count=(stats.advantage_samples if stats is not None else 0),
        policy_count=(stats.policy_samples if stats is not None else 0),
        terminal_count=(stats.terminal_visits if stats is not None else 0),
    )


def merge_results(
    results: list[TraversalResult],
    advantage_reservoirs: Sequence[Reservoir],
    policy_reservoir: Reservoir,
    merge_rng: random.Random,
) -> None:
    """Replay private samples into the shared reservoirs, single-threaded.

    Deterministic by construction: results are processed in traversal-index order
    and samples within a traversal in DFS order, all eviction decisions drawn from
    the one ``merge_rng``. This is the SOLE writer to the shared reservoirs.
    """
    for res in sorted(results, key=lambda r: r.index):
        adv_res = advantage_reservoirs[res.traverser]
        for features, mask, target, weight, key in res.advantage_samples:
            adv_res.add(
                torch.from_numpy(features),
                torch.from_numpy(mask),
                torch.from_numpy(target),
                weight,
                merge_rng,
                infoset_key=key,
            )
        for features, mask, target, weight, key in res.policy_samples:
            policy_reservoir.add(
                torch.from_numpy(features),
                torch.from_numpy(mask),
                torch.from_numpy(target),
                weight,
                merge_rng,
                infoset_key=key,
            )


def merge_coverage(
    results: list[TraversalResult], cov_stats: TraversalStats | None
) -> None:
    """Fold per-traversal coverage counters into the shared ``TraversalStats``.

    Counter addition is associative + commutative, so the merged result equals
    exactly what a single shared stats object would have accumulated in-line during
    a serial traversal — independent of the executor or the chunking. Processed in
    index order so dict insertion order is stable across runs (the visit-depth
    readout re-sorts by region anyway, so only the counts matter for correctness).
    """
    if cov_stats is None:
        return
    for res in sorted(results, key=lambda r: r.index):
        cov_stats.advantage_samples += res.advantage_count
        cov_stats.policy_samples += res.policy_count
        cov_stats.terminal_visits += res.terminal_count
        if res.region_visits:
            for region, ctr in res.region_visits.items():
                dst = cov_stats.region_visits.get(region)
                if dst is None:
                    dst = Counter()
                    cov_stats.region_visits[region] = dst
                dst.update(ctr)


# ───────── multiprocessing worker plumbing (spawn-safe) ─────────
#
# Workers are built once with a copy of the game + advantage-net skeletons (passed
# via the Pool initializer). Per iteration the current net weights are pushed as
# state_dicts (cheap — a few hundred KB each) so workers never run on stale weights.

_WORKER_GAME: Game[Any] | None = None
_WORKER_NETS: list[torch.nn.Module] | None = None


def _init_worker(game: Game[Any], nets: list[torch.nn.Module]) -> None:
    global _WORKER_GAME, _WORKER_NETS
    torch.set_num_threads(1)  # one BLAS thread per worker — avoid oversubscription
    _WORKER_GAME = game
    _WORKER_NETS = nets
    for net in _WORKER_NETS:
        net.eval()


def _worker_run(
    iter_t: int,
    indices: list[int],
    net_states: list[dict[str, Any]],
    master_seed: int,
    opp_aggression_bias: float,
    coverage: bool,
) -> list[TraversalResult]:
    assert _WORKER_GAME is not None and _WORKER_NETS is not None
    for net, state in zip(_WORKER_NETS, net_states, strict=True):
        net.load_state_dict(state)
    return [
        run_one_traversal(
            _WORKER_GAME, _WORKER_NETS, iter_t, i, master_seed, opp_aggression_bias, coverage
        )
        for i in indices
    ]


def make_worker_pool(
    num_workers: int, game: Game[Any], nets: Sequence[torch.nn.Module]
) -> Any:
    """Create a spawn-based process pool with workers pre-loaded with game + nets."""
    ctx = mp.get_context("spawn")
    return ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(game, list(nets)),
    )


def _chunk_indices(n: int, num_chunks: int) -> list[list[int]]:
    """Split range(n) into <=num_chunks round-robin chunks (stable, executor-free)."""
    chunks: list[list[int]] = [[] for _ in range(num_chunks)]
    for i in range(n):
        chunks[i % num_chunks].append(i)
    return [c for c in chunks if c]


def cfr_iteration(
    *,
    game: Game[Any],
    nets: Sequence[torch.nn.Module],
    advantage_reservoirs: Sequence[Reservoir],
    policy_reservoir: Reservoir,
    iter_t: int,
    traversals_per_iter: int,
    master_seed: int,
    num_workers: int,
    opp_aggression_bias: float,
    coverage: bool,
    cov_stats: TraversalStats | None,
    worker_pool: Any | None = None,
) -> None:
    """Run one CFR iteration's traversal phase and merge it into the reservoirs.

    ``num_workers <= 1``: traversals run in-process (the serial reference).
    ``num_workers > 1``: traversals fan out across ``worker_pool`` and merge in the
    main process. Both paths share the per-traversal seeds and the merge, so they
    produce bit-identical reservoirs and coverage counters.
    """
    if num_workers <= 1 or worker_pool is None:
        results = [
            run_one_traversal(game, nets, iter_t, i, master_seed, opp_aggression_bias, coverage)
            for i in range(traversals_per_iter)
        ]
    else:
        # One chunk per worker. net_states (a few hundred KB/net) is pickled once
        # per chunk, so we keep chunk count == worker count: finer splits re-pickle
        # the weights more times and that IPC cost outweighs any load-balancing
        # gain. Correctness is independent of the split — the merge re-sorts by
        # traversal index.
        net_states = [net.state_dict() for net in nets]
        chunks = _chunk_indices(traversals_per_iter, num_workers)
        async_results = [
            worker_pool.apply_async(
                _worker_run,
                (iter_t, chunk, net_states, master_seed, opp_aggression_bias, coverage),
            )
            for chunk in chunks
        ]
        results = []
        for ar in async_results:
            results.extend(ar.get())

    merge_rng = random.Random(derive_seed(master_seed, iter_t, "merge"))
    merge_results(results, advantage_reservoirs, policy_reservoir, merge_rng)
    merge_coverage(results, cov_stats)
