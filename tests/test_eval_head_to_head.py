"""Tests for scripts/eval_head_to_head.py: the --opponent-db flag + report header.

Hermetic — builds any DB it needs as a throwaway in tmp_path via the project's
own strategy_db write path (open_db / set_current_version / put). Does NOT depend
on any *.db file in the repo root (all *.db are gitignored).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pokerbot.abstraction import ActionType, InfoSet
from pokerbot.strategy_db import open_db

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import eval_head_to_head as ehh

if TYPE_CHECKING:
    import pytest

# ───────── helpers ─────────


def _build_tiny_db(path: Path, *, base_bucket: int = 40) -> Path:
    """Write a handful of table_size-6 rows to a throwaway SQLite strategy DB."""
    db = open_db(f"sqlite:///{path}")
    db.set_current_version(1)
    mask = (
        (1 << int(ActionType.FOLD))
        | (1 << int(ActionType.CHECK_CALL))
        | (1 << int(ActionType.BET_66))
    )
    probs = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    for cb in range(base_bucket, base_bucket + 5):
        info = InfoSet(
            table_size=6,
            street=1,
            position=2,
            stack_bucket=4,
            card_bucket=cb,
            history=bytes([cb % 256]),
        )
        db.put(info, mask, probs, version=1, visit_count=cb)
    db.close()
    return path


# ───────── arg parsing ─────────


def test_opponent_db_defaults_to_none() -> None:
    args = ehh._parse_args(["--db", "x.db"])
    assert args.opponent_db is None


def test_opponent_db_parses_path() -> None:
    args = ehh._parse_args(["--db", "x.db", "--opponent-db", "/some/where/opp.db"])
    assert args.opponent_db == "/some/where/opp.db"


# ───────── _open_opponent_db ─────────


def test_open_opponent_db_none_is_empty_default_policy() -> None:
    db = ehh._open_opponent_db(None)
    try:
        assert db.current_version() == 1
        # No strategy rows → any lookup misses (falls through to default policy).
        probe = InfoSet(
            table_size=6, street=1, position=2, stack_bucket=4, card_bucket=40, history=b"\x28"
        )
        assert db.get(probe, version=1) is None
        assert db.nearest_neighbor(probe, version=1) is None
    finally:
        db.close()


def test_open_opponent_db_path_opens_file_backed(tmp_path: Path) -> None:
    db_file = _build_tiny_db(tmp_path / "opp.db", base_bucket=40)
    db = ehh._open_opponent_db(str(db_file))
    try:
        probe = InfoSet(
            table_size=6, street=1, position=2, stack_bucket=4, card_bucket=40, history=b"\x28"
        )
        row = db.get(probe, version=1)
        assert row is not None
        assert row.visit_count == 40
    finally:
        db.close()


# ───────── _comparison_header ─────────


def test_comparison_header_with_opponent() -> None:
    h = ehh._comparison_header("strategy-v5-6max.db", "strategy-pilot-v2.db", 50000, 6)
    assert h == "Evaluation: strategy-v5-6max vs strategy-pilot-v2 (50000 hands, 6-max)"


def test_comparison_header_default_policy() -> None:
    h = ehh._comparison_header("strategy-v5-6max.db", None, 1000, 6)
    assert h == "Evaluation: strategy-v5-6max vs default policy (1000 hands, 6-max)"


# ───────── end-to-end smoke ─────────


def test_main_db_vs_db_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trained = _build_tiny_db(tmp_path / "new.db", base_bucket=40)
    opponent = _build_tiny_db(tmp_path / "strategy-baseline.db", base_bucket=80)
    rc = ehh.main(
        [
            "--db",
            str(trained),
            "--opponent-db",
            str(opponent),
            "--n-hands",
            "5",
            "--table-size",
            "6",
            "--abstraction-dir",
            "abstraction",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Evaluation: new vs strategy-baseline (5 hands, 6-max)" in out
    assert "opponent mbb/hand:" in out


def test_main_default_policy_header(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trained = _build_tiny_db(tmp_path / "new.db", base_bucket=40)
    rc = ehh.main(
        [
            "--db",
            str(trained),
            "--n-hands",
            "5",
            "--table-size",
            "6",
            "--abstraction-dir",
            "abstraction",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "vs default policy" in out
    assert "default mbb/hand:" in out


def test_main_missing_opponent_db_returns_2(tmp_path: Path) -> None:
    trained = _build_tiny_db(tmp_path / "new.db", base_bucket=40)
    rc = ehh.main(
        [
            "--db",
            str(trained),
            "--opponent-db",
            str(tmp_path / "nonexistent.db"),
            "--n-hands",
            "5",
            "--abstraction-dir",
            "abstraction",
        ]
    )
    assert rc == 2
