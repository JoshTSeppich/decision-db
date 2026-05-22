"""MultiSizeStrategyDB: routes lookups by `infoset.table_size` to one of N
underlying strategy DBs (one per trained table size).

Generalises `DualStrategyDB` (which routes between two DBs) so each
`table_size in {2..9}` can have its own dedicated CFR-trained policy DB.
Read-only at runtime: `put` and `bulk_put` raise. All underlying DBs must
share the same `current_version()`, asserted at construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pokerbot.strategy_db.base import StrategyDB

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from numpy.typing import NDArray

    from pokerbot.abstraction import InfoSet
    from pokerbot.strategy_db.base import StrategyRow


SUPPORTED_TABLE_SIZES: frozenset[int] = frozenset({2, 3, 4, 5, 6, 7, 8, 9})


class MultiSizeStrategyDB(StrategyDB):
    """Routes by `infoset.table_size` to a per-size StrategyDB. Read-only at runtime."""

    def __init__(self, dbs: dict[int, StrategyDB]) -> None:
        if not dbs:
            raise ValueError("MultiSizeStrategyDB requires at least one underlying DB")
        bad_keys = [k for k in dbs if k not in SUPPORTED_TABLE_SIZES]
        if bad_keys:
            raise ValueError(
                f"table_size keys must be in {sorted(SUPPORTED_TABLE_SIZES)}, "
                f"got out-of-range keys: {sorted(bad_keys)}"
            )
        versions = {size: db.current_version() for size, db in dbs.items()}
        unique_versions = set(versions.values())
        if len(unique_versions) > 1:
            raise ValueError(
                f"MultiSizeStrategyDB requires matching current_version across all "
                f"underlying DBs; got {versions}"
            )
        self._dbs = dict(dbs)
        self._version = next(iter(unique_versions))

    def _route(self, table_size: int) -> StrategyDB:
        try:
            return self._dbs[table_size]
        except KeyError:
            raise KeyError(
                f"MultiSizeStrategyDB has no DB registered for table_size={table_size}; "
                f"registered sizes: {sorted(self._dbs)}"
            ) from None

    def get(self, infoset: InfoSet, version: int | None = None) -> StrategyRow | None:
        return self._route(infoset.table_size).get(infoset, version)

    def nearest_neighbor(self, infoset: InfoSet, version: int) -> StrategyRow | None:
        return self._route(infoset.table_size).nearest_neighbor(infoset, version)

    def current_version(self) -> int:
        return self._version

    def set_current_version(self, version: int) -> None:
        for db in self._dbs.values():
            db.set_current_version(version)
        self._version = version

    def put(
        self,
        infoset: InfoSet,  # noqa: ARG002
        action_mask: int,  # noqa: ARG002
        action_probs: NDArray[np.float32],  # noqa: ARG002
        version: int,  # noqa: ARG002
        visit_count: int = 0,  # noqa: ARG002
    ) -> None:
        raise RuntimeError(
            "MultiSizeStrategyDB is read-only; write to the underlying DB directly"
        )

    def bulk_put(
        self,
        rows: Iterable[tuple[InfoSet, int, NDArray[np.float32]]],  # noqa: ARG002
        version: int,  # noqa: ARG002
    ) -> None:
        raise RuntimeError(
            "MultiSizeStrategyDB is read-only; write to the underlying DB directly"
        )

    def close(self) -> None:
        for db in self._dbs.values():
            db.close()


__all__ = ["SUPPORTED_TABLE_SIZES", "MultiSizeStrategyDB"]
