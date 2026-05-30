"""Fan-out / merge parallel engine for the Deep CFR fine-tune traversal phase.

The traversal phase dominates per-iteration wall-clock: `traversals_per_iter`
pure-Python pokerkit traversals run one at a time, GIL-bound, on one core. This
module fans those traversals across CPU cores (separate PROCESSES — the GIL means
threads would not parallelize pure-Python work) and merges the results back into
the shared reservoirs single-threaded.

The correctness contract — equivalence to the serial version — is structural:

  1. PER-TRAVERSAL RNG. Traversal ``i`` of iteration ``t`` draws from its own
     ``random.Random`` seeded by ``derive_seed(master_seed, t, i)`` (a stable
     sha256 hash, NOT the process-randomized builtin ``hash``). Every random
     decision in that traversal — traverser pick, initial deal, chance/opponent
     sampling, archetype seat assignment — flows from that one RNG, so traversals
     are independent and reproducible by index, which is exactly what
     external-sampling MCCFR assumes.

  2. PRIVATE BUFFERS, NO SHARED-RESERVOIR RACE. Each traversal writes its samples
     into a private `CollectingBuffer` (a duck-typed reservoir sink with the same
     ``.add`` signature that appends and *ignores* the eviction RNG). No worker
     ever touches the shared reservoir, so the corruption hazard is eliminated by
     construction, and removing eviction RNG from the per-traversal stream is what
     makes (1) sufficient.

  3. DETERMINISTIC SERIAL MERGE. After all traversals finish, the MAIN process
     replays the collected samples into the real reservoirs in a fixed global
     order (by traversal index, then DFS order within a traversal) using a single
     dedicated ``merge_rng``. This is the only writer to the shared reservoirs.
     Algorithm R yields a uniform random k-subset of a stream regardless of
     insertion order, so the merge is distributionally identical to the legacy
     loop; because the order and RNG are fixed, serial and parallel agree bit-for-bit.

`src/pokerbot` is frozen: this wraps the unchanged
`zoom.train.traversal.external_sampling_traversal` read-only.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from zoom.train.opponents import make_archetype_opponent_policy
from zoom.train.traversal import external_sampling_traversal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pokerbot.training.game import Game
    from pokerbot.training.traversal import Reservoir
    from zoom.agents import ScriptedAgent

__all__ = [
    "CollectingBuffer",
    "TraversalResult",
    "cfr_iteration",
    "derive_seed",
    "make_worker_pool",
    "merge_results",
    "run_one_traversal",
]

# A reservoir-ready sample: (features, mask, target, iter_weight, infoset_key).
Sample = tuple[np.ndarray, np.ndarray, np.ndarray, float, bytes]


def derive_seed(master_seed: int, iter_t: int, tag: int | str) -> int:
    """Stable, process-independent seed for ``(master_seed, iter_t, tag)``.

    Uses sha256 rather than the builtin ``hash`` because ``hash`` of str is
    randomized per process (PYTHONHASHSEED), which would make spawned workers
    disagree with the main process and break reproducibility.
    """
    digest = hashlib.sha256(f"{master_seed}:{iter_t}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class CollectingBuffer:
    """Private per-traversal sink, duck-typed to `Reservoir`.

    Implements the exact ``.add`` signature the traversal calls, but appends the
    sample to a list and IGNORES ``rng`` — no reservoir-sampling eviction happens
    here. This is what keeps the eviction RNG out of the per-traversal stream so
    each traversal is independent of every other.
    """

    def __init__(self) -> None:
        self.samples: list[Sample] = []

    def add(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        iter_weight: float,
        rng: random.Random,  # noqa: ARG002 — intentionally ignored (no eviction in a private buffer)
        infoset_key: bytes = b"",
    ) -> None:
        self.samples.append(
            (
                features.detach().cpu().numpy().copy(),
                mask.detach().cpu().numpy().copy(),
                target.detach().cpu().numpy().copy(),
                float(iter_weight),
                bytes(infoset_key),
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


def _opponent_policy_for(
    game: Game[Any],
    traverser: int,
    pool: Sequence[ScriptedAgent] | None,
    rng: random.Random,
) -> Any:
    """Assign an archetype to each non-traverser seat for this hand (drawn from the
    per-traversal ``rng`` for reproducibility), or None for self-play."""
    if pool is None:
        return None
    assignment = {
        seat: rng.choice(list(pool))
        for seat in range(game.num_players)
        if seat != traverser
    }
    return make_archetype_opponent_policy(game, assignment)  # type: ignore[arg-type]


def run_one_traversal(
    game: Game[Any],
    nets: Sequence[torch.nn.Module],
    iter_t: int,
    index: int,
    master_seed: int,
    pool: Sequence[ScriptedAgent] | None,
) -> TraversalResult:
    """Run a single traversal into private buffers. Pure (no shared state)."""
    rng = random.Random(derive_seed(master_seed, iter_t, index))
    traverser = rng.randrange(game.num_players)
    state = game.new_initial_state(rng)
    opponent_policy = _opponent_policy_for(game, traverser, pool, rng)
    adv_buf = CollectingBuffer()
    pol_buf = CollectingBuffer()
    external_sampling_traversal(
        game=game,
        state=state,
        traverser=traverser,
        advantage_nets=nets,
        advantage_reservoir=adv_buf,  # type: ignore[arg-type]
        policy_reservoir=pol_buf,  # type: ignore[arg-type]
        iter_t=iter_t,
        rng=rng,
        opponent_policy=opponent_policy,
    )
    return TraversalResult(index, traverser, adv_buf.samples, pol_buf.samples)


def merge_results(
    results: Iterable[TraversalResult],
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


# ───────── multiprocessing worker plumbing (spawn-safe) ─────────
#
# Workers are built once with a copy of the game + opponent pool + advantage-net
# skeletons (passed via the Pool initializer). Per iteration the current net
# weights are pushed as state_dicts (cheap — a few hundred KB each) so workers
# never run on stale weights.

_WORKER_GAME: Game[Any] | None = None
_WORKER_NETS: list[torch.nn.Module] | None = None
_WORKER_POOL: list[ScriptedAgent] | None = None


def _init_worker(
    game: Game[Any],
    nets: list[torch.nn.Module],
    pool: list[ScriptedAgent] | None,
) -> None:
    global _WORKER_GAME, _WORKER_NETS, _WORKER_POOL
    torch.set_num_threads(1)  # one BLAS thread per worker — avoid oversubscription
    _WORKER_GAME = game
    _WORKER_NETS = nets
    for net in _WORKER_NETS:
        net.eval()
    _WORKER_POOL = pool


def _worker_run(
    iter_t: int,
    indices: list[int],
    net_states: list[dict[str, Any]],
    master_seed: int,
) -> list[TraversalResult]:
    assert _WORKER_GAME is not None and _WORKER_NETS is not None
    for net, state in zip(_WORKER_NETS, net_states, strict=True):
        net.load_state_dict(state)
    return [
        run_one_traversal(_WORKER_GAME, _WORKER_NETS, iter_t, i, master_seed, _WORKER_POOL)
        for i in indices
    ]


def make_worker_pool(
    num_workers: int,
    game: Game[Any],
    nets: Sequence[torch.nn.Module],
    pool: Sequence[ScriptedAgent] | None,
) -> Any:
    """Create a spawn-based process pool with workers pre-loaded with game + nets."""
    ctx = mp.get_context("spawn")
    return ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(game, list(nets), list(pool) if pool is not None else None),
    )


def _chunk_indices(n: int, num_chunks: int) -> list[list[int]]:
    """Split range(n) into <=num_chunks contiguous chunks (round-robin-free, stable)."""
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
    pool: Sequence[ScriptedAgent] | None,
    num_workers: int,
    worker_pool: Any | None = None,
) -> None:
    """Run one CFR iteration's traversal phase and merge it into the reservoirs.

    ``num_workers == 1``: traversals run in-process (the serial reference).
    ``num_workers > 1``: traversals fan out across ``worker_pool`` and merge in the
    main process. Both paths share the per-traversal seeds and the merge, so they
    produce bit-identical reservoirs.
    """
    if num_workers == 1 or worker_pool is None:
        results = [
            run_one_traversal(game, nets, iter_t, i, master_seed, pool)
            for i in range(traversals_per_iter)
        ]
    else:
        # One chunk per worker. net_states (a few hundred KB/net) is pickled once per
        # chunk, so we keep chunk count == worker count: finer splits re-pickle the
        # weights more times and that IPC cost outweighs any load-balancing gain
        # (measured). Correctness is independent of the split — the merge re-sorts by
        # traversal index.
        net_states = [net.state_dict() for net in nets]
        chunks = _chunk_indices(traversals_per_iter, num_workers)
        async_results = [
            worker_pool.apply_async(_worker_run, (iter_t, chunk, net_states, master_seed))
            for chunk in chunks
        ]
        results = []
        for ar in async_results:
            results.extend(ar.get())

    merge_rng = random.Random(derive_seed(master_seed, iter_t, "merge"))
    merge_results(results, advantage_reservoirs, policy_reservoir, merge_rng)
