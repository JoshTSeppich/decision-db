"""LMDB-backed StrategyDB (Spec.html §D).

Layout:
    strategy sub-DB:  key = hash(16) + version_BE(4)   →  msgpack value
    nn_index sub-DB:  key = t_size(1) + street(1) + card_bucket_BE(2)
                          + position(1) + stack_bucket(1) + version_BE(4)
                          + hash(16)                     →  b""
    meta     sub-DB:  key = ascii bytes                 →  ascii bytes

`nn_index` lets `nearest_neighbor` do a prefix range-scan over the lookup
key, avoiding a full DB walk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import lmdb
import msgpack

from pokerbot.strategy_db.base import (
    StrategyDB,
    StrategyRow,
    pack_probs,
    unpack_probs,
    validate_action_probs,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from pokerbot.abstraction import InfoSet


_META_CURRENT_VERSION_KEY: Final[bytes] = b"current_strategy_version"
_META_SCHEMA_VERSION_KEY: Final[bytes] = b"schema_version"
_LMDB_SCHEMA_VERSION: Final[int] = 1
_DEFAULT_MAP_SIZE: Final[int] = 1 << 30  # 1 GiB; LMDB grows lazily within this cap


def _nn_prefix(infoset: InfoSet, version: int) -> bytes:
    return (
        bytes([infoset.table_size, infoset.street])
        + infoset.card_bucket.to_bytes(2, "big")
        + bytes([infoset.position, infoset.stack_bucket])
        + version.to_bytes(4, "big", signed=False)
    )


def _main_key(infoset_hash: bytes, version: int) -> bytes:
    return infoset_hash + version.to_bytes(4, "big", signed=False)


class LMDBStrategyDB(StrategyDB):
    """LMDB implementation. `path` is a directory (created if missing)."""

    def __init__(self, path: str | Path, map_size: int = _DEFAULT_MAP_SIZE) -> None:
        self.path = str(path)
        self._env: lmdb.Environment | None = lmdb.open(
            self.path,
            max_dbs=3,
            map_size=map_size,
            subdir=True,
            create=True,
        )
        self._strategy = self._env.open_db(b"strategy")
        self._nn_index = self._env.open_db(b"nn_index")
        self._meta = self._env.open_db(b"meta")
        self._initialize_schema_version()

    # ───────── housekeeping ─────────

    def _require_env(self) -> lmdb.Environment:
        if self._env is None:
            raise RuntimeError("LMDBStrategyDB is closed")
        return self._env

    def _initialize_schema_version(self) -> None:
        env = self._require_env()
        with env.begin(write=True, db=self._meta) as txn:
            existing = txn.get(_META_SCHEMA_VERSION_KEY)
            if existing is None:
                txn.put(_META_SCHEMA_VERSION_KEY, str(_LMDB_SCHEMA_VERSION).encode("ascii"))

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ───────── reads ─────────

    def get(self, infoset: InfoSet, version: int | None = None) -> StrategyRow | None:
        env = self._require_env()
        v = self.current_version() if version is None else version
        key = _main_key(infoset.hash16(), v)
        with env.begin(db=self._strategy) as txn:
            blob = txn.get(key)
        if blob is None:
            return None
        return _unpack_row(bytes(blob))

    def nearest_neighbor(self, infoset: InfoSet, version: int) -> StrategyRow | None:
        env = self._require_env()
        prefix = _nn_prefix(infoset, version)
        best_row: StrategyRow | None = None
        with env.begin() as txn:
            cur = txn.cursor(db=self._nn_index)
            if not cur.set_range(prefix):
                return None
            for raw_k, _ in cur:
                k = bytes(raw_k)
                if not k.startswith(prefix):
                    break
                hash_bytes = k[-16:]
                blob = txn.get(_main_key(hash_bytes, version), db=self._strategy)
                if blob is None:
                    continue  # index entry orphaned (shouldn't happen)
                row = _unpack_row(bytes(blob))
                if best_row is None or row.visit_count > best_row.visit_count:
                    best_row = row
        return best_row

    # ───────── writes ─────────

    def _write_one(
        self,
        txn: lmdb.Transaction,
        infoset: InfoSet,
        action_mask: int,
        action_probs: NDArray[np.float32],
        version: int,
        visit_count: int,
    ) -> None:
        validate_action_probs(action_mask, action_probs)
        h = infoset.hash16()
        value: dict[str, Any] = {
            "ts": infoset.table_size,
            "st": infoset.street,
            "ps": infoset.position,
            "sb": infoset.stack_bucket,
            "cb": infoset.card_bucket,
            "hi": infoset.history,
            "am": action_mask,
            "ap": pack_probs(action_probs),
            "vc": visit_count,
            "v": version,
        }
        txn.put(_main_key(h, version), msgpack.packb(value, use_bin_type=True), db=self._strategy)
        txn.put(_nn_prefix(infoset, version) + h, b"", db=self._nn_index)

    def put(
        self,
        infoset: InfoSet,
        action_mask: int,
        action_probs: NDArray[np.float32],
        version: int,
        visit_count: int = 0,
    ) -> None:
        env = self._require_env()
        with env.begin(write=True) as txn:
            self._write_one(txn, infoset, action_mask, action_probs, version, visit_count)

    def bulk_put(
        self,
        rows: Iterable[tuple[InfoSet, int, NDArray[np.float32]]],
        version: int,
    ) -> None:
        env = self._require_env()
        txn = env.begin(write=True)
        try:
            for infoset, mask, probs in rows:
                self._write_one(txn, infoset, mask, probs, version, 0)
        except BaseException:
            txn.abort()
            raise
        else:
            txn.commit()

    # ───────── version control ─────────

    def current_version(self) -> int:
        env = self._require_env()
        with env.begin(db=self._meta) as txn:
            blob = txn.get(_META_CURRENT_VERSION_KEY)
        if blob is None:
            raise RuntimeError(
                f"meta[{_META_CURRENT_VERSION_KEY!r}] not set; "
                "call set_current_version() or pass version= explicitly"
            )
        return int(bytes(blob).decode("ascii"))

    def set_current_version(self, version: int) -> None:
        env = self._require_env()
        with env.begin(write=True, db=self._meta) as txn:
            txn.put(_META_CURRENT_VERSION_KEY, str(version).encode("ascii"))


def _unpack_row(blob: bytes) -> StrategyRow:
    data = msgpack.unpackb(blob, raw=False)
    mask = int(data["am"])
    return StrategyRow(
        action_mask=mask,
        action_probs=unpack_probs(mask, bytes(data["ap"])),
        visit_count=int(data["vc"]),
        version=int(data["v"]),
    )


__all__ = ["LMDBStrategyDB"]
