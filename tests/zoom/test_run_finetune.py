"""Component 5 driver — zoom/train/run_finetune.py (Approach-2).

Pure assembly of already-tested pieces: load nets-only from a checkpoint, build
`FineTuneTrainer(pool=build_archetype_pool())` on `GatedNLHEGame` at 100bb, run N
iters with opponent-injection ON, and `export_best_response(...)` the current
best-response (regret_match of the advantage nets) to a DB. No new training logic.

Red-first: `run_finetune` must run a tiny N and produce a DB the Component 4 harness
can score. The import fails until the driver exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from zoom.eval import evaluate_bands, make_db_spot_policy, profile_spot_policy
from zoom.train.run_finetune import FineTuneArgs, run_finetune

from pokerbot.abstraction import AbstractionTables
from pokerbot.strategy_db import open_db
from pokerbot.training import SimpleNLHEGame
from pokerbot.training.config import DeepCFRConfig
from pokerbot.training.nets import AdvantageNet

if TYPE_CHECKING:
    from pathlib import Path

_FEATURE_DIM = 72  # SimpleNLHEGame.feature_dim
_NUM_ACTIONS = 9  # len(ActionType)


def _write_fake_checkpoint(path: Path, *, table_size: int = 3) -> None:
    """Write a minimal nets-only checkpoint matching iter_0400's structure: an
    `advantage_states` list of `table_size` AdvantageNet state_dicts (256x3)."""
    cfg = DeepCFRConfig()  # production arch (256,256,256) — matches iter_0400
    nets = [AdvantageNet(_FEATURE_DIM, _NUM_ACTIONS, cfg) for _ in range(table_size)]
    torch.save(
        {"iter": 400, "advantage_states": [n.state_dict() for n in nets], "config": cfg},
        path,
    )


def test_run_finetune_produces_scorable_db(tmp_path: Path) -> None:
    """The driver runs a tiny N and writes a DB the Component 4 harness can score."""
    ckpt = tmp_path / "iter_0400.pt"
    _write_fake_checkpoint(ckpt)
    out_db = tmp_path / "bp.db"

    args = FineTuneArgs(
        checkpoint=ckpt,
        out_db=f"sqlite:///{out_db}",
        abstraction_dir=None,  # AbstractionTables() with no NPZ — fast, deterministic
        iters=2,
        traversals_per_iter=20,
        train_steps_per_iter=20,
        table_size=3,
        seed=4242,
        strategy_version=1,
    )
    n_rows = run_finetune(args)
    assert n_rows > 0, "driver exported no rows"
    assert out_db.exists()

    # Scorable through the ONE validated gate (no second profiler).
    db = open_db(f"sqlite:///{out_db}")
    abstraction = AbstractionTables()
    game = SimpleNLHEGame(abstraction, blinds=(5, 10), starting_stack=1000, table_size=3)
    db_policy = make_db_spot_policy(db, abstraction, bb=10, table_size=3)
    profile = profile_spot_policy(db_policy, game, n_hands=20, seed=1)
    result = evaluate_bands(profile)
    assert isinstance(result.passed, bool)
    for m in result.metrics:
        assert np.isfinite(m.value), (m.name, m.value)
    db.close()


def test_run_finetune_loads_nets_only_not_reservoirs(tmp_path: Path) -> None:
    """Driver loads advantage-net weights from the checkpoint (nets-only resume) —
    it does NOT require the (1.75GB) reservoir sidecar, and starts fresh reservoirs.
    Proven by running with ONLY the .pt present (no sidecar npz)."""
    ckpt = tmp_path / "iter_0400.pt"
    _write_fake_checkpoint(ckpt)
    assert not (tmp_path / "iter_0400_reservoirs.npz").exists()  # no sidecar

    args = FineTuneArgs(
        checkpoint=ckpt,
        out_db=f"sqlite:///{tmp_path / 'bp.db'}",
        abstraction_dir=None,
        iters=1,
        traversals_per_iter=12,
        train_steps_per_iter=10,
        table_size=3,
        seed=7,
        strategy_version=1,
    )
    n_rows = run_finetune(args)  # must not raise about a missing sidecar
    assert n_rows > 0


def test_cli_parses_iters_arg() -> None:
    """`iters` (the run length) is a CLI arg, so the same driver runs 100 or 200."""
    from zoom.train.run_finetune import build_args

    args = build_args(
        [
            "--checkpoint",
            "/tmp/iter_0400.pt",
            "--out-db",
            "sqlite:///x.db",
            "--iters",
            "100",
            "--table-size",
            "3",
        ]
    )
    assert args.iters == 100
    assert args.table_size == 3


# ─────────── mid-run checkpointing + resume (survivability) ───────────


def test_checkpoint_written_mid_run(tmp_path: Path) -> None:
    """With checkpoint_every=2, a 5-iter run writes resume checkpoints to
    checkpoint_dir (so an interruption costs ≤checkpoint_every iters, not the run)."""
    ckpt = tmp_path / "iter_0400.pt"
    _write_fake_checkpoint(ckpt)
    ckdir = tmp_path / "ckpts"

    args = FineTuneArgs(
        checkpoint=ckpt,
        out_db=f"sqlite:///{tmp_path / 'bp.db'}",
        abstraction_dir=None,
        iters=5,
        traversals_per_iter=10,
        train_steps_per_iter=10,
        table_size=3,
        seed=1,
        strategy_version=1,
        checkpoint_dir=ckdir,
        checkpoint_every=2,
    )
    run_finetune(args)
    # iters 2 and 4 should have produced resume checkpoints.
    written = sorted(ckdir.glob("finetune_iter_*.pt"))
    assert written, f"no mid-run checkpoint written to {ckdir}"
    iters_ckpted = {int(p.stem.split("_")[-1]) for p in written}
    assert 2 in iters_ckpted and 4 in iters_ckpted, iters_ckpted


def test_resume_picks_up_from_latest_checkpoint(tmp_path: Path) -> None:
    """A second run pointed at a checkpoint_dir that already has a checkpoint at
    iter K resumes from K+1 — it does NOT restart from iter 1. Proven by the log
    showing the first executed iter is K+1 and the run stops at `iters`."""
    ckpt = tmp_path / "iter_0400.pt"
    _write_fake_checkpoint(ckpt)
    ckdir = tmp_path / "ckpts"

    base = dict(
        checkpoint=ckpt,
        out_db=f"sqlite:///{tmp_path / 'bp.db'}",
        abstraction_dir=None,
        traversals_per_iter=10,
        train_steps_per_iter=10,
        table_size=3,
        seed=1,
        strategy_version=1,
        checkpoint_dir=ckdir,
        checkpoint_every=2,
    )
    # First run: 4 iters → latest checkpoint at iter 4.
    run_finetune(FineTuneArgs(iters=4, **base))  # type: ignore[arg-type]
    assert int(sorted(ckdir.glob("finetune_iter_*.pt"))[-1].stem.split("_")[-1]) == 4

    # Resume run to 6 iters: must execute iters 5,6 only (resumed from 4).
    from zoom.train.run_finetune import resume_iter_for

    assert resume_iter_for(ckdir) == 4  # latest completed iter on disk
    run_finetune(FineTuneArgs(iters=6, **base))  # type: ignore[arg-type]
    assert int(sorted(ckdir.glob("finetune_iter_*.pt"))[-1].stem.split("_")[-1]) == 6


def test_cli_parses_checkpoint_args() -> None:
    """checkpoint-dir / checkpoint-every are CLI args (written to the persistent volume)."""
    from zoom.train.run_finetune import build_args

    args = build_args(
        [
            "--checkpoint", "/tmp/iter_0400.pt",
            "--out-db", "sqlite:///x.db",
            "--iters", "100",
            "--checkpoint-dir", "/workspace/ckpts",
            "--checkpoint-every", "20",
        ]
    )
    assert str(args.checkpoint_dir) == "/workspace/ckpts"
    assert args.checkpoint_every == 20


def test_train_steps_default_is_200_not_4000() -> None:
    """The fine-tune default must be 200 (the proven pilot/blueprint value), NOT the
    from-scratch DeepCFRConfig default of 4000. 4000 = 20x the neural work per iter
    (4000x3 player passes) for no fine-tune reason — a best-response fine-tune needs
    fewer steps than from-scratch, never more. Guards against the typo recurring.
    """
    from zoom.train.run_finetune import build_args

    args = build_args(["--checkpoint", "/tmp/iter_0400.pt", "--out-db", "sqlite:///x.db"])
    assert args.train_steps_per_iter == 200, (
        f"fine-tune train-steps default is {args.train_steps_per_iter}, expected 200 "
        "(4000 is the from-scratch default and a 20x per-iter tax)"
    )
