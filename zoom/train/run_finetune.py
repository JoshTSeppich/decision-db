"""Stage-1 fine-tune driver (Approach-2, Component 5).

Pure assembly of already-tested pieces — NO new training logic:

  1. Load nets-only from a checkpoint (the `advantage_states` from iter_0400.pt);
     fresh reservoirs (does NOT need the 1.75GB reservoir sidecar).
  2. Build `FineTuneTrainer(pool=build_archetype_pool())` on a `GatedNLHEGame` at
     100bb fixed — opponent-injection ON (the Component 3 seam), NOT self-play.
  3. Run N iters (N is a CLI arg, so the same driver runs 100 or 200): each iter is
     `_cfr_iteration` (injected traversal) + `_train_advantage_nets`. We loop these
     directly rather than `Trainer.train()` so we (a) log per-iter for live tailing
     and (b) skip the policy-net phase, which is empty by construction in a
     best-response fine-tune (the (a)-benign empty policy_reservoir).
  4. `export_best_response(...)` — regret_match of the advantage nets (current
     best-response, NOT a time-average) — to a StrategyDB at `--out-db`.

Run as: `python -m zoom.train.run_finetune --checkpoint training/v5-3max/iter_0400.pt
         --out-db sqlite:///training/v5-3max/strategy-v5-3max-finetuned.db --iters 100`
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from pokerbot.abstraction import AbstractionTables
from pokerbot.strategy_db import open_db
from pokerbot.training.config import DeepCFRConfig
from zoom.abstraction_gate import DEFAULT_SPR_CAP
from zoom.agents import build_archetype_pool
from zoom.train.export_br import export_best_response
from zoom.train.finetune import FineTuneTrainer
from zoom.train.game import GatedNLHEGame

_LOG = logging.getLogger("zoom.finetune")


@dataclass(frozen=True)
class FineTuneArgs:
    checkpoint: Path
    out_db: str
    abstraction_dir: Path | None
    iters: int
    traversals_per_iter: int
    train_steps_per_iter: int
    table_size: int
    seed: int
    strategy_version: int
    blinds: tuple[int, int] = (5, 10)
    starting_stack: int = 1000  # 100bb at bb=10 — Stage 1 trains at 100bb fixed
    # Component-1 SPR shove gate: ALL_IN is offered only when spr <= spr_cap (or it's a
    # forced jam / undefined pot). Lower = fewer deep-stack preflop shoves. Passed into the
    # GatedNLHEGame so the learner trains on the gated action set; tuned to the preflop
    # ALL_IN<1% band while the R2 unit tests keep low-SPR/short-stack jams alive.
    spr_cap: float = DEFAULT_SPR_CAP
    # Survivability: write a lightweight resume checkpoint (advantage-net states +
    # iter + rng) every `checkpoint_every` iters to `checkpoint_dir` (the persistent
    # volume on the pod). On restart the driver resumes from the latest, so an
    # interruption costs ≤checkpoint_every iters instead of the whole run. None =
    # off (the test-default for the tiny scorable-DB tests).
    checkpoint_dir: Path | None = None
    checkpoint_every: int = 20
    # Traversal-phase parallelism: fan the per-iteration MCCFR traversals across this
    # many worker processes (1 = serial). The parallel path is provably equivalent to
    # serial (see zoom.train.parallel_traversal); use it to cut per-iter wall-clock on
    # multi-core boxes. Future runs only — does not change results, only speed.
    num_workers: int = 1


_CKPT_GLOB = "finetune_iter_*.pt"


def _ckpt_path(checkpoint_dir: Path, t: int) -> Path:
    return checkpoint_dir / f"finetune_iter_{t:04d}.pt"


def resume_iter_for(checkpoint_dir: Path | None) -> int:
    """Latest completed iter with a checkpoint on disk, or 0 if none/dir absent."""
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return 0
    iters = [int(p.stem.split("_")[-1]) for p in checkpoint_dir.glob(_CKPT_GLOB)]
    return max(iters, default=0)


def _save_resume_checkpoint(trainer: FineTuneTrainer, checkpoint_dir: Path, t: int) -> None:
    """Lightweight checkpoint: advantage-net states + iter + rng. Deliberately does
    NOT write the (multi-GB) reservoir sidecar — the learned policy lives in the
    nets; reservoirs re-fill on resume. Written atomically (tmp + rename) so an
    interruption mid-write can't leave a corrupt checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final = _ckpt_path(checkpoint_dir, t)
    tmp = final.with_suffix(".pt.tmp")
    torch.save(
        {
            "iter": t,
            "advantage_states": [n.state_dict() for n in trainer.advantage_nets],
            "rng_state": trainer.rng.getstate(),
        },
        tmp,
    )
    tmp.replace(final)


def _load_resume_checkpoint(trainer: FineTuneTrainer, checkpoint_dir: Path, t: int) -> None:
    ckpt = torch.load(_ckpt_path(checkpoint_dir, t), map_location="cpu", weights_only=False)
    for net, state in zip(trainer.advantage_nets, ckpt["advantage_states"], strict=True):
        net.load_state_dict(state)
    trainer.rng.setstate(ckpt["rng_state"])


def _load_nets_only(trainer: FineTuneTrainer, checkpoint: Path) -> None:
    """Load ONLY the advantage-net weights from `checkpoint` — fresh reservoirs.

    Reads `advantage_states` (the same key `Trainer.save_checkpoint` writes) and
    loads each into the corresponding net. Deliberately ignores the reservoir
    sidecar: a best-response fine-tune starts from the blueprint policy with fresh
    buffers, not the old self-play regrets (which would dilute the archetype-BR
    signal — Component 3's rationale).
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    states = ckpt["advantage_states"]
    if len(states) != len(trainer.advantage_nets):
        raise ValueError(
            f"checkpoint has {len(states)} advantage nets but game has "
            f"{len(trainer.advantage_nets)} seats (table_size mismatch?)"
        )
    for net, state in zip(trainer.advantage_nets, states, strict=True):
        net.load_state_dict(state)


def run_finetune(args: FineTuneArgs) -> int:
    """Run the fine-tune and export the best-response DB. Returns rows exported."""
    abstraction = AbstractionTables(path=args.abstraction_dir)
    game = GatedNLHEGame(
        abstraction,
        blinds=args.blinds,
        starting_stack=args.starting_stack,
        table_size=args.table_size,
        spr_cap=args.spr_cap,
    )
    cfg = DeepCFRConfig(
        outer_iters=args.iters,
        traversals_per_iter=args.traversals_per_iter,
        train_steps_per_iter=args.train_steps_per_iter,
        seed=args.seed,
    )
    trainer = FineTuneTrainer(
        cfg, game, pool=build_archetype_pool(), num_workers=args.num_workers
    )

    # Resume from the latest mid-run checkpoint if one exists, else nets-only from
    # the blueprint. This is what makes an interrupted run survivable: a host reset
    # / OOM / preemption costs ≤checkpoint_every iters, not the whole run.
    resume_from = resume_iter_for(args.checkpoint_dir)
    if resume_from > 0 and args.checkpoint_dir is not None:
        _load_resume_checkpoint(trainer, args.checkpoint_dir, resume_from)
        _LOG.info("RESUMED from %s at iter %d", args.checkpoint_dir, resume_from)
    else:
        _load_nets_only(trainer, args.checkpoint)
        _LOG.info(
            "loaded nets-only from %s | table_size=%d stack=%dbb iters=%d trav/iter=%d",
            args.checkpoint,
            args.table_size,
            args.starting_stack // args.blinds[1],
            args.iters,
            args.traversals_per_iter,
        )

    run_t0 = time.perf_counter()
    for t in range(resume_from + 1, args.iters + 1):
        trainer.iter = t
        it0 = time.perf_counter()
        if args.num_workers > 1:
            trainer._cfr_iteration_parallel(t)  # fan-out/merge (equivalent to serial)
        else:
            trainer._cfr_iteration(t)  # injected traversal (opponent pool ON)
        trainer._train_advantage_nets(t)
        dt = time.perf_counter() - it0
        elapsed = (time.perf_counter() - run_t0) / 60.0
        eta = (args.iters - t) * dt / 60.0
        _LOG.info(
            "[iter %d/%d] %.1fs/iter adv_buf=%d (wall=%.1fmin eta=%.1fmin)",
            t,
            args.iters,
            dt,
            sum(len(r) for r in trainer.advantage_reservoirs),
            elapsed,
            eta,
        )
        if args.checkpoint_dir is not None and t % args.checkpoint_every == 0:
            _save_resume_checkpoint(trainer, args.checkpoint_dir, t)
            _LOG.info("  checkpoint -> %s", _ckpt_path(args.checkpoint_dir, t))

    # Save a FULL checkpoint INCLUDING the reservoir sidecar so this run's DB is fully
    # re-exportable later (the prior run saved nets-only, which is why its DB couldn't be
    # re-exported when the export logic changed). The lightweight resume checkpoints above
    # deliberately omit reservoirs; this one does not.
    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = args.checkpoint_dir / "finetune_final.pt"
        trainer.save_checkpoint(final_ckpt)
        _LOG.info("saved full checkpoint (advantage+policy reservoirs) -> %s", final_ckpt)

    trainer.close_pool()  # release worker processes before the (single-threaded) export
    db = open_db(args.out_db)
    n_rows = export_best_response(
        trainer.advantage_nets, trainer.advantage_reservoirs, db, version=args.strategy_version
    )
    db.close()
    _LOG.info(
        "export complete: %d rows -> %s (version=%d)", n_rows, args.out_db, args.strategy_version
    )
    return n_rows


def build_args(argv: list[str] | None = None) -> FineTuneArgs:
    p = argparse.ArgumentParser(description="Stage-1 archetype-pool best-response fine-tune")
    p.add_argument("--checkpoint", type=Path, required=True, help="iter_NNNN.pt (nets-only load)")
    p.add_argument("--out-db", required=True, help="output StrategyDB URL (sqlite:///path)")
    p.add_argument("--abstraction-dir", type=Path, default=None, help="dir with buckets_*.npz")
    p.add_argument("--iters", type=int, default=100, help="fine-tune iterations (100 or 200)")
    p.add_argument("--traversals-per-iter", type=int, default=200)
    p.add_argument(
        "--train-steps-per-iter",
        type=int,
        default=200,
        help="gradient steps/iter (200 = proven pilot/blueprint value; NOT the from-scratch "
        "DeepCFRConfig default of 4000, which is 20x the neural work for no fine-tune reason)",
    )
    p.add_argument("--table-size", type=int, default=3)
    p.add_argument(
        "--spr-cap",
        type=float,
        default=DEFAULT_SPR_CAP,
        help="Component-1 ALL_IN gate: offer ALL_IN only when spr<=spr_cap (or forced jam). "
        "Lower drops deep-stack preflop shoves (tune to preflop ALL_IN<1%%).",
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--strategy-version", type=int, default=1)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="dir for mid-run resume checkpoints (use the persistent volume, e.g. /workspace/ckpts)",
    )
    p.add_argument("--checkpoint-every", type=int, default=20, help="save a resume checkpoint every N iters")
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="parallelize the traversal phase across this many worker processes "
        "(1 = serial; the parallel path is provably equivalent to serial)",
    )
    a = p.parse_args(argv)
    return FineTuneArgs(
        checkpoint=a.checkpoint,
        out_db=a.out_db,
        abstraction_dir=a.abstraction_dir,
        iters=a.iters,
        traversals_per_iter=a.traversals_per_iter,
        train_steps_per_iter=a.train_steps_per_iter,
        table_size=a.table_size,
        spr_cap=a.spr_cap,
        seed=a.seed,
        strategy_version=a.strategy_version,
        checkpoint_dir=a.checkpoint_dir,
        checkpoint_every=a.checkpoint_every,
        num_workers=a.num_workers,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    n_rows = run_finetune(build_args(argv))
    return 0 if n_rows > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
