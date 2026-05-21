"""Card abstraction — public entry point for Spec.html §A.

Card encoding (0..51):
    card_id = rank * 4 + suit
    rank: 0..12   == '2','3','4','5','6','7','8','9','T','J','Q','K','A'
    suit: 0..3    == 'c','d','h','s'

Public API (per spec):
    - `bucket_hand(hole, board, street) -> int`
    - `num_buckets(street) -> int`
    - `canonical_hand(hole, board) -> (canon_hole, canon_board)`
    - `AbstractionTables(path)` — loads `buckets_*.npz`

Postflop bucketing is currently a deterministic canonical-hash placeholder
(BLAKE2b of the canonical card-tuple, mod 200). It respects suit isomorphism
but does not capture equity / potential / OCHS — the real k-means/OCHS clusters
ship as NPZ artifacts once `cli.py build-abstraction` lands (requires phevaluator
and ~6h of one-time compute). When the NPZ files are present in the supplied
directory, `AbstractionTables` reads them; otherwise it falls back to the hash.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

from pokerbot.abstraction._types import Card, Street
from pokerbot.abstraction.preflop import NUM_PREFLOP_BUCKETS, preflop_bucket

_RANKS: Final[str] = "23456789TJQKA"
_SUITS: Final[str] = "cdhs"

_BUCKET_COUNTS: Final[dict[Street, int]] = {
    "preflop": NUM_PREFLOP_BUCKETS,
    "flop": 200,
    "turn": 200,
    "river": 200,
}

_EXPECTED_BOARD_LEN: Final[dict[Street, int]] = {
    "preflop": 0,
    "flop": 3,
    "turn": 4,
    "river": 5,
}

_NPZ_FILENAMES: Final[dict[Street, str]] = {
    "flop": "buckets_flop.npz",
    "turn": "buckets_turn.npz",
    "river": "buckets_river.npz",
}


# ─────────── card parsing ───────────


def parse_card(token: str) -> Card:
    if len(token) != 2:
        raise ValueError(f"card token must be 2 chars: {token!r}")
    rank = _RANKS.find(token[0].upper())
    suit = _SUITS.find(token[1].lower())
    if rank < 0 or suit < 0:
        raise ValueError(f"unparseable card: {token!r}")
    return rank * 4 + suit


def card_str(card: Card) -> str:
    if not 0 <= card < 52:
        raise ValueError(f"card out of range [0,52): {card}")
    return _RANKS[card >> 2] + _SUITS[card & 3]


def parse_hand(spec: str) -> tuple[Card, ...]:
    """Parse a whitespace-separated card string: 'As Kh' -> (51, 50)."""
    return tuple(parse_card(t) for t in spec.split())


# ─────────── canonicalization ───────────


def canonical_hand(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
) -> tuple[tuple[Card, Card], tuple[Card, ...]]:
    """Suit-isomorphic canonicalization.

    Relabel suits in the order they first appear in `hole + board`, so any two
    hands that are suit-permutations of each other produce identical output.
    Hole cards are returned in ascending order; board cards keep position
    (flop slots are then sorted within the flop, turn/river kept at the end).
    """
    suit_map: dict[int, int] = {}
    next_id = 0
    for c in (*hole, *board):
        s = c & 3
        if s not in suit_map:
            suit_map[s] = next_id
            next_id += 1

    def remap(c: Card) -> Card:
        return (c & ~3) | suit_map[c & 3]

    canon_hole = tuple(sorted(remap(c) for c in hole))
    if len(canon_hole) != 2:
        raise ValueError(f"hole must be exactly 2 cards: {hole!r}")

    canon_board_list = [remap(c) for c in board]
    # Within the flop the 3 cards are dealt simultaneously — order doesn't matter.
    if len(canon_board_list) >= 3:
        flop_sorted = sorted(canon_board_list[:3])
        canon_board: tuple[Card, ...] = (*flop_sorted, *canon_board_list[3:])
    else:
        canon_board = tuple(canon_board_list)

    return ((canon_hole[0], canon_hole[1]), canon_board)


# ─────────── bucket counts ───────────


def num_buckets(street: Street) -> int:
    return _BUCKET_COUNTS[street]


# ─────────── bucket lookups ───────────


def _validate_inputs(hole: tuple[Card, Card], board: tuple[Card, ...], street: Street) -> None:
    if street not in _BUCKET_COUNTS:
        raise ValueError(f"unknown street: {street!r}")
    if len(hole) != 2:
        raise ValueError(f"hole must be 2 cards, got {len(hole)}")
    expected = _EXPECTED_BOARD_LEN[street]
    if len(board) != expected:
        raise ValueError(f"street={street!r} expects {expected} board cards, got {len(board)}")
    seen: set[Card] = set()
    for c in (*hole, *board):
        if not 0 <= c < 52:
            raise ValueError(f"card out of range [0,52): {c}")
        if c in seen:
            raise ValueError(f"duplicate card: {card_str(c)}")
        seen.add(c)


def _canonical_bytes(canon_hole: tuple[Card, Card], canon_board: tuple[Card, ...]) -> bytes:
    return bytes((*canon_hole, *canon_board))


def _placeholder_postflop_bucket(
    canon_hole: tuple[Card, Card],
    canon_board: tuple[Card, ...],
    street: Street,
) -> int:
    """Deterministic, isomorphism-respecting placeholder until real NPZs ship.

    Produces a stable bucket in [0, num_buckets(street)) for any canonical
    (hole, board) class. Will be replaced by `AbstractionTables` once the
    `cli.py build-abstraction` artifact is on disk.
    """
    key = _canonical_bytes(canon_hole, canon_board) + bytes(street, "ascii")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "little") % _BUCKET_COUNTS[street]


def bucket_hand(
    hole: tuple[Card, Card],
    board: tuple[Card, ...],
    street: Street,
) -> int:
    """Bucket id in [0, num_buckets(street)).

    Preflop is lossless (169 classes). Postflop uses the placeholder until
    `AbstractionTables` is loaded with real NPZ data — see module docstring.
    """
    _validate_inputs(hole, board, street)
    canon_hole, canon_board = canonical_hand(hole, board)
    if street == "preflop":
        return preflop_bucket(canon_hole)
    return _placeholder_postflop_bucket(canon_hole, canon_board, street)


# ─────────── loadable tables (NPZ-backed) ───────────


class AbstractionTables:
    """Loads pre-built bucket NPZ artifacts for postflop streets (Spec.html §A).

    Expected directory layout:
        {path}/buckets_flop.npz   — int32 array, len = num canonical (hole, flop)
        {path}/buckets_turn.npz   — same shape for turn
        {path}/buckets_river.npz  — same shape for river (OCHS)
        {path}/centroids.npz      — optional, for debugging

    The NPZ files are produced by `cli.py build-abstraction`. Until they exist,
    `lookup` transparently uses the same placeholder bucketing as `bucket_hand`.
    """

    # Bounded LRU for the miss path. project_to_centroid is ~5ms per call;
    # a CFR run repeats the same canonical (hole, board, street) thousands of
    # times across traversals, so caching is a 5-10x speedup for training.
    # 500k entries x ~64 bytes = ~32 MB — cheap.
    DEFAULT_LRU_MAX: int = 500_000

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        miss_samples: int = 30,
        lru_max: int | None = None,
    ) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self.miss_samples = miss_samples
        self._lru_max: int = lru_max if lru_max is not None else self.DEFAULT_LRU_MAX
        self._tables: dict[Street, npt.NDArray[np.int32] | None] = {
            "flop": None,
            "turn": None,
            "river": None,
        }
        self._class_id_keys: dict[Street, npt.NDArray[np.uint8] | None] = {
            "flop": None,
            "turn": None,
            "river": None,
        }
        # Loaded centroids enable on-the-fly nearest-centroid projection when an
        # arbitrary query misses the NPZ's sampled canonical classes. Production
        # builds sample ~4% of the canonical space, so misses are the common case.
        self._centroids: dict[Street, npt.NDArray[np.float32] | None] = {
            "flop": None,
            "turn": None,
            "river": None,
        }
        # LRU cache for the miss path's projected buckets, keyed on canonical
        # (hole, board, street). OrderedDict gives us move-to-end + popitem(0).
        self._miss_cache: OrderedDict[tuple[tuple[Card, Card], tuple[Card, ...], Street], int] = (
            OrderedDict()
        )
        # Stats so tests + the profile script can measure hit rate.
        self.lookup_hits_exact: int = 0
        self.lookup_hits_lru: int = 0
        self.lookup_misses: int = 0
        if self.path is not None:
            self._maybe_load_all()

    def _maybe_load_all(self) -> None:
        assert self.path is not None
        for street, fname in _NPZ_FILENAMES.items():
            fpath = self.path / fname
            if not fpath.exists():
                continue
            with np.load(fpath) as npz:
                if "buckets" not in npz.files or "keys" not in npz.files:
                    raise ValueError(f"{fpath} missing required arrays 'buckets' and 'keys'")
                self._tables[street] = npz["buckets"].astype(np.int32, copy=False)
                self._class_id_keys[street] = npz["keys"].astype(np.uint8, copy=False)
                if "centroids" in npz.files:
                    self._centroids[street] = npz["centroids"].astype(np.float32, copy=False)

    @property
    def loaded_streets(self) -> tuple[Street, ...]:
        return tuple(s for s, t in self._tables.items() if t is not None)

    def lookup(
        self,
        hole: tuple[Card, Card],
        board: tuple[Card, ...],
        street: Street,
    ) -> int:
        _validate_inputs(hole, board, street)
        canon_hole, canon_board = canonical_hand(hole, board)
        if street == "preflop":
            return preflop_bucket(canon_hole)
        buckets = self._tables.get(street)
        keys = self._class_id_keys.get(street)
        if buckets is None or keys is None:
            return _placeholder_postflop_bucket(canon_hole, canon_board, street)
        canon_key = np.frombuffer(_canonical_bytes(canon_hole, canon_board), dtype=np.uint8)
        idx = _binary_search_rows(keys, canon_key)
        if idx >= 0:
            self.lookup_hits_exact += 1
            return int(buckets[idx])
        # Miss path. Cache projected buckets by canonical state to avoid paying
        # the ~5ms project_to_centroid cost on every repeat query.
        cache_key = (canon_hole, canon_board, street)
        cached = self._miss_cache.get(cache_key)
        if cached is not None:
            self._miss_cache.move_to_end(cache_key)
            self.lookup_hits_lru += 1
            return cached
        # Cache miss → compute features on-the-fly and project to nearest centroid.
        # Requires the [train] extras for phevaluator + the build helper module;
        # if either is unavailable we fall back to the deterministic placeholder.
        centroids = self._centroids.get(street)
        if centroids is None:
            return _placeholder_postflop_bucket(canon_hole, canon_board, street)
        try:
            from pokerbot.abstraction.build.lookup import project_to_centroid
        except ImportError:
            return _placeholder_postflop_bucket(canon_hole, canon_board, street)
        bucket = project_to_centroid(
            canon_hole, canon_board, street, centroids, num_samples=self.miss_samples
        )
        self._miss_cache[cache_key] = bucket
        if len(self._miss_cache) > self._lru_max:
            self._miss_cache.popitem(last=False)  # evict oldest
        self.lookup_misses += 1
        return bucket

    def lookup_stats(self) -> dict[str, int]:
        """Diagnostic counters reset by `reset_lookup_stats()`."""
        return {
            "hits_exact": self.lookup_hits_exact,
            "hits_lru": self.lookup_hits_lru,
            "misses": self.lookup_misses,
            "lru_size": len(self._miss_cache),
        }

    def reset_lookup_stats(self) -> None:
        self.lookup_hits_exact = 0
        self.lookup_hits_lru = 0
        self.lookup_misses = 0


def _binary_search_rows(keys: npt.NDArray[np.uint8], target: npt.NDArray[np.uint8]) -> int:
    """Find row in `keys` (sorted lexicographically) matching `target`, or -1."""
    if keys.ndim != 2 or keys.shape[1] != target.shape[0]:
        return -1
    n = int(keys.shape[0])
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        row = keys[mid]
        cmp = _cmp_rows(row, target)
        if cmp == 0:
            return mid
        if cmp < 0:
            lo = mid + 1
        else:
            hi = mid
    return -1


def _cmp_rows(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> int:
    for x, y in zip(a, b, strict=True):
        if x < y:
            return -1
        if x > y:
            return 1
    return 0


__all__ = [
    "AbstractionTables",
    "Card",
    "Street",
    "bucket_hand",
    "canonical_hand",
    "card_str",
    "num_buckets",
    "parse_card",
    "parse_hand",
]
