"""Strategy DB abstract base + shared helpers (Spec.html §D).

Spec-faithfulness note: the spec's published DDL has `infoset_hash` as the
single-column primary key, but the §D `test_version_filter` requires storing
two distinct rows for the same infoset under different `version`s. The
implementation here uses a composite primary key `(infoset_hash, version)` —
the minimum change that makes the spec internally consistent.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import TracebackType

    from numpy.typing import NDArray

    from pokerbot.abstraction import InfoSet


PROBS_SUM_TOL = 1e-6


@dataclass(frozen=True, slots=True)
class StrategyRow:
    action_mask: int
    action_probs: NDArray[np.float32]
    visit_count: int
    version: int


def popcount(mask: int) -> int:
    if mask < 0:
        raise ValueError(f"action_mask must be non-negative: {mask}")
    return mask.bit_count()


def validate_action_probs(action_mask: int, probs: np.ndarray) -> None:
    """Reject probs that don't match the action mask or aren't a proper distribution."""
    if probs.dtype != np.float32:
        raise ValueError(f"action_probs dtype must be float32, got {probs.dtype}")
    if probs.ndim != 1:
        raise ValueError(f"action_probs must be 1-D, got ndim={probs.ndim}")
    expected = popcount(action_mask)
    if len(probs) != expected:
        raise ValueError(f"action_probs length {len(probs)} != popcount(action_mask)={expected}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("action_probs must be finite")
    if (probs < 0).any():
        raise ValueError("action_probs must be non-negative")
    total = float(probs.sum())
    if not math.isclose(total, 1.0, abs_tol=PROBS_SUM_TOL):
        raise ValueError(f"action_probs must sum to 1.0 ± {PROBS_SUM_TOL}, got {total}")


def pack_probs(probs: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(probs, dtype=np.float32)
    return arr.tobytes()


def unpack_probs(action_mask: int, blob: bytes) -> NDArray[np.float32]:
    expected = popcount(action_mask)
    arr: NDArray[np.float32] = np.frombuffer(blob, dtype=np.float32).copy()
    if len(arr) != expected:
        raise ValueError(f"unpacked probs length {len(arr)} != popcount(action_mask)={expected}")
    return arr


class StrategyDB(ABC):
    """Abstract strategy storage. Concrete backends: SQLite, LMDB.

    Lookup: `(infoset_hash, version)` is the primary key.
    Nearest-neighbor: prefix on `(table_size, street, card_bucket, position,
    stack_bucket, version)` — ignores betting history.
    """

    @abstractmethod
    def get(self, infoset: InfoSet, version: int | None = None) -> StrategyRow | None:
        """Exact hash lookup. `version=None` uses `current_version()`."""

    @abstractmethod
    def put(
        self,
        infoset: InfoSet,
        action_mask: int,
        action_probs: NDArray[np.float32],
        version: int,
        visit_count: int = 0,
    ) -> None:
        """Insert-or-replace one row. Validates `action_probs`."""

    @abstractmethod
    def nearest_neighbor(self, infoset: InfoSet, version: int) -> StrategyRow | None:
        """Highest-visit_count row matching (t_size, street, card_bucket, position, stack_bucket)."""

    @abstractmethod
    def bulk_put(
        self,
        rows: Iterable[tuple[InfoSet, int, NDArray[np.float32]]],
        version: int,
    ) -> None:
        """Atomic batch insert. Any exception during iteration rolls the whole batch back."""

    @abstractmethod
    def current_version(self) -> int:
        """The current strategy version (set via `set_current_version`). Raises if unset."""

    @abstractmethod
    def set_current_version(self, version: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> StrategyDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "PROBS_SUM_TOL",
    "StrategyDB",
    "StrategyRow",
    "pack_probs",
    "popcount",
    "unpack_probs",
    "validate_action_probs",
]
