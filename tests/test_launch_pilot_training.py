"""Tests for scripts/launch_pilot_training.py CLI: buffer validation + lru alias.

The launcher re-execs the interpreter at import time unless PYTHONHASHSEED=0,
so we pin it before importing the module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ["PYTHONHASHSEED"] = "0"  # suppress the launcher's re-exec guard
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import launch_pilot_training as lpt


def test_lru_cache_size_is_alias_for_lru_max() -> None:
    parser = lpt.build_parser()
    assert parser.parse_args(["--lru-max", "4000000"]).lru_max == 4_000_000
    assert parser.parse_args(["--lru-cache-size", "4000000"]).lru_max == 4_000_000
    # Both passed → argparse last-one-wins, no error.
    both = parser.parse_args(["--lru-max", "1000000", "--lru-cache-size", "4000000"])
    assert both.lru_max == 4_000_000


@pytest.mark.parametrize(
    ("flag", "value", "needle"),
    [
        ("--advantage-buffer-size", "500", "--advantage-buffer-size"),
        ("--policy-buffer-size", "999", "--policy-buffer-size"),
        ("--lru-cache-size", "200000000", "--lru-max/--lru-cache-size"),
    ],
)
def test_validate_rejects_out_of_range(
    flag: str, value: str, needle: str, capsys: pytest.CaptureFixture[str]
) -> None:
    args = lpt.build_parser().parse_args([flag, value])
    with pytest.raises(SystemExit) as exc:
        lpt.validate_buffer_args(args)
    assert exc.value.code == 2  # argparse usage-error convention, non-zero
    err = capsys.readouterr().err
    assert needle in err
    assert "out of range" in err


def test_validate_accepts_in_range_boundaries() -> None:
    args = lpt.build_parser().parse_args(
        [
            "--advantage-buffer-size", "1000",  # floor
            "--policy-buffer-size", "100000000",  # ceiling
            "--lru-cache-size", "2000000",
        ]
    )
    lpt.validate_buffer_args(args)  # must not raise


def test_checkpoint_dir_default_is_local_not_volume() -> None:
    default = lpt.build_parser().parse_args([]).checkpoint_dir
    # Must be an absolute local container-disk path, never the network volume
    # that filled up and killed the v5 run.
    assert default.is_absolute()
    assert "workspace" not in default.parts
    assert default == Path("/tmp/pokerbot-checkpoints")


def test_checkpoint_dir_parses() -> None:
    args = lpt.build_parser().parse_args(["--checkpoint-dir", "/mnt/big/ckpts"])
    assert args.checkpoint_dir == Path("/mnt/big/ckpts")


def test_resume_defaults_none_and_parses_to_path() -> None:
    assert lpt.build_parser().parse_args([]).resume is None
    args = lpt.build_parser().parse_args(["--resume", "/tmp/ckpts/iter_1000.pt"])
    assert args.resume == Path("/tmp/ckpts/iter_1000.pt")


def test_main_passes_checkpoint_dir_and_resume_to_train(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """main() must hand --checkpoint-dir to train() as output_dir and --resume
    as resume_from, with no real training / GPU / network / DB export."""
    ckpt_dir = tmp_path / "ckpts"
    resume_pt = tmp_path / "ckpts" / "iter_1000.pt"
    captured: dict[str, object] = {}

    # Stub the heavy collaborators so main() runs in milliseconds.
    monkeypatch.setattr(lpt, "AbstractionTables", _StubTables)

    def fake_train(self: object, output_dir: Path, resume_from: Path | None = None) -> None:
        captured["output_dir"] = output_dir
        captured["resume_from"] = resume_from

    monkeypatch.setattr(lpt.Trainer, "train", fake_train)

    argv = [
        "launch_pilot_training.py",
        "--checkpoint-dir", str(ckpt_dir),
        "--resume", str(resume_pt),
        "--advantage-buffer-size", "1000",  # tiny reservoirs → fast, low memory
        "--policy-buffer-size", "1000",
        "--skip-export",  # no DB
        "--skip-lbr",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    rc = lpt.main()

    assert rc == 0
    assert captured["output_dir"] == ckpt_dir
    assert captured["resume_from"] == resume_pt
    # The launcher must create the checkpoint dir.
    assert ckpt_dir.is_dir()


class _StubTables:
    """Minimal stand-in for AbstractionTables: all three postflop streets loaded,
    cheap lookup, trivial stats. Avoids loading any NPZ/centroid files."""

    loaded_streets = ("flop", "turn", "river")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def lookup(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def lookup_stats(self) -> dict[str, int]:
        return {"hits_exact": 0, "hits_lru": 0, "misses": 0, "lru_size": 0}
