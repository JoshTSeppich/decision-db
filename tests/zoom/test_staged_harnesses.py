"""Pytest gate over the L2/L3 proof harnesses.

The harnesses (`test_opponent_model.py`, `test_range_tracker.py`) are kept as
standalone scripts whose assertion logic is byte-for-byte the original proof of
correctness for the two staged modules. Here we run each as a subprocess and
assert it exits 0 and reports the full check count, so the pytest suite gates on
them without importing (and thus executing) them during collection.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = Path(__file__).resolve().parent


def _run_harness(filename: str) -> subprocess.CompletedProcess[str]:
    """Run a harness script with the repo root importable (so `import zoom` works
    when the script is executed directly, not via pytest's rootdir injection)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    return subprocess.run(
        [sys.executable, str(HARNESS_DIR / filename)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("filename", "expected_checks"),
    [
        ("test_opponent_model.py", 10),
        ("test_range_tracker.py", 16),
    ],
)
def test_staged_proof_harness(filename: str, expected_checks: int) -> None:
    result = _run_harness(filename)
    assert result.returncode == 0, (
        f"{filename} failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert f"{expected_checks}/{expected_checks} checks passed" in result.stdout, (
        f"{filename} did not report {expected_checks}/{expected_checks}:\n{result.stdout}"
    )
