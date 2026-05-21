"""DualStrategyDB: routes lookups by `infoset.table_size` to one of two
underlying strategy DBs.

Per Cairn 6 routing:
  - table_size == 6           → primary_db (6-max trained policy)
  - table_size in {2..5, 7..9} → secondary_db (9-max trained policy)

Read-only wrapper: `put` and `bulk_put` raise. Used at runtime by the
tournament hero adapter to consult the appropriate trained DB by table
size. Versions on both DBs are expected to match; `current_version()`
returns the primary's version (the secondary's is asserted-equal at
construction).
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


# Routing table: which table_size goes to which DB.
SIX_MAX_ROUTE: frozenset[int] = frozenset({6})
NINE_MAX_ROUTE: frozenset[int] = frozenset({2, 3, 4, 5, 7, 8, 9})


class DualStrategyDB(StrategyDB):
    """Routes by `infoset.table_size`. Read-only at runtime."""

    def __init__(self, primary_db: StrategyDB, secondary_db: StrategyDB) -> None:
        # Versions must match — RuntimeAdapter caches `current_version()` and
        # passes it across calls; mismatched versions would route a lookup
        # against the wrong policy generation.
        p_ver = primary_db.current_version()
        s_ver = secondary_db.current_version()
        if p_ver != s_ver:
            raise ValueError(
                f"DualStrategyDB requires matching current_version: "
                f"primary={p_ver}, secondary={s_ver}"
            )
        self._primary = primary_db
        self._secondary = secondary_db

    def _route(self, table_size: int) -> StrategyDB:
        return self._primary if table_size in SIX_MAX_ROUTE else self._secondary

    def get(self, infoset: InfoSet, version: int | None = None) -> StrategyRow | None:
        return self._route(infoset.table_size).get(infoset, version)

    def nearest_neighbor(self, infoset: InfoSet, version: int) -> StrategyRow | None:
        return self._route(infoset.table_size).nearest_neighbor(infoset, version)

    def current_version(self) -> int:
        return self._primary.current_version()

    def set_current_version(self, version: int) -> None:
        self._primary.set_current_version(version)
        self._secondary.set_current_version(version)

    def put(
        self,
        infoset: InfoSet,  # noqa: ARG002
        action_mask: int,  # noqa: ARG002
        action_probs: NDArray[np.float32],  # noqa: ARG002
        version: int,  # noqa: ARG002
        visit_count: int = 0,  # noqa: ARG002
    ) -> None:
        raise RuntimeError("DualStrategyDB is read-only; write to the underlying DB directly")

    def bulk_put(
        self,
        rows: Iterable[tuple[InfoSet, int, NDArray[np.float32]]],  # noqa: ARG002
        version: int,  # noqa: ARG002
    ) -> None:
        raise RuntimeError("DualStrategyDB is read-only; write to the underlying DB directly")

    def close(self) -> None:
        self._primary.close()
        self._secondary.close()


__all__ = ["NINE_MAX_ROUTE", "SIX_MAX_ROUTE", "DualStrategyDB"]
