"""Pytest config for the zoom test package.

`test_opponent_model.py` and `test_range_tracker.py` are the original proof
harnesses for L2/L3. They are plain scripts (top-level `check()` calls +
`raise SystemExit(1)` on failure), NOT pytest test functions — importing them
during collection would execute the proofs at import time. We keep their
assertion logic byte-for-byte and instead run them as subprocesses from
`test_staged_harnesses.py`, so they stay the canonical correctness proof while
the pytest suite still drives (and gates on) them.
"""

collect_ignore = [
    "test_opponent_model.py",
    "test_range_tracker.py",
]
