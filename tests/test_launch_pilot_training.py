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
