"""InfoSet — byte-exact key for the strategy DB.

Spec.html §C pins the layout:

    +--------+--------+--------+--------+--------+--------+----...---+
    | t_size | street | pos    | stack  |  card_bucket    | history  |
    | u8     | u8     | u8     | u8     |  u16 LE         | var≤64   |
    +--------+--------+--------+--------+--------+--------+----...---+

`encode_history` / `decode_history` are deferred to step 3 because they need
`ActionType` from `abstraction.actions` (which step 3 owns).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Final

from pokerbot.abstraction.actions import (
    STREET_BOUNDARY_BYTE,
    AbstractAction,
    ActionType,
    action_to_byte,
    byte_to_action_type,
)

PREFIX_FMT: Final[str] = "<BBBBH"  # u8 t_size, u8 street, u8 pos, u8 stack, u16 LE card_bucket
PREFIX_LEN: Final[int] = struct.calcsize(PREFIX_FMT)  # = 6
HISTORY_MAX: Final[int] = 64
HASH_DIGEST_BYTES: Final[int] = 16  # BLAKE2b-128

VALID_TABLE_SIZES: Final[frozenset[int]] = frozenset({2, 3, 4, 5, 6, 7, 8, 9})
VALID_STREETS: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
STACK_BUCKETS: Final[int] = 10  # 0..9
CARD_BUCKET_MAX: Final[int] = 65535  # u16


@dataclass(frozen=True, slots=True)
class InfoSet:
    """Canonical infoset identifier. Equality + hashing follow the byte layout."""

    table_size: int  # 6, 8, 9
    street: int  # 0=preflop, 1=flop, 2=turn, 3=river
    position: int  # 0..table_size-1 (0 = SB)
    stack_bucket: int  # 0..9
    card_bucket: int  # 0..65535 (uses 0..168 preflop, 0..199 postflop)
    history: bytes = field(default=b"")

    def __post_init__(self) -> None:
        if self.table_size not in VALID_TABLE_SIZES:
            raise ValueError(
                f"table_size must be one of {sorted(VALID_TABLE_SIZES)}, got {self.table_size}"
            )
        if self.street not in VALID_STREETS:
            raise ValueError(f"street must be 0..3, got {self.street}")
        if not 0 <= self.position < self.table_size:
            raise ValueError(f"position must be in [0, {self.table_size}), got {self.position}")
        if not 0 <= self.stack_bucket < STACK_BUCKETS:
            raise ValueError(
                f"stack_bucket must be in [0, {STACK_BUCKETS}), got {self.stack_bucket}"
            )
        if not 0 <= self.card_bucket <= CARD_BUCKET_MAX:
            raise ValueError(
                f"card_bucket must be in [0, {CARD_BUCKET_MAX}], got {self.card_bucket}"
            )
        if len(self.history) > HISTORY_MAX:
            raise ValueError(f"history exceeds {HISTORY_MAX} bytes: {len(self.history)}")

    def to_bytes(self) -> bytes:
        prefix = struct.pack(
            PREFIX_FMT,
            self.table_size,
            self.street,
            self.position,
            self.stack_bucket,
            self.card_bucket,
        )
        return prefix + self.history

    def hash16(self) -> bytes:
        return hashlib.blake2b(self.to_bytes(), digest_size=HASH_DIGEST_BYTES).digest()


# ─────────── betting history encode/decode (Spec.html §C) ───────────


def encode_history(actions: list[AbstractAction], street_boundaries: list[int]) -> bytes:
    """Encode an action sequence with street markers into ≤ `HISTORY_MAX` bytes.

    `street_boundaries` is a strictly increasing list of indices where a NEW
    street begins. If `actions = [a,b,c,d,e]` and `street_boundaries = [2, 4]`:
        street 0: a, b      street 1: c, d      street 2: e
        bytes  : a  b  0xF0  c  d  0xF0  e

    Empty `street_boundaries` means the whole sequence is on one street.
    """
    n = len(actions)
    bnd = list(street_boundaries)
    if bnd != sorted(set(bnd)):
        raise ValueError(f"street_boundaries must be strictly increasing: {street_boundaries!r}")
    if bnd and (bnd[0] <= 0 or bnd[-1] > n):
        raise ValueError(f"street_boundaries out of range [1, {n}]: {street_boundaries!r}")

    out = bytearray()
    bnd_set = set(bnd)
    for i, a in enumerate(actions):
        if i in bnd_set:
            out.append(STREET_BOUNDARY_BYTE)
        out.append(action_to_byte(a))
    if len(out) > HISTORY_MAX:
        raise ValueError(f"encoded history exceeds {HISTORY_MAX} bytes: {len(out)}")
    return bytes(out)


def decode_history(blob: bytes) -> list[tuple[int, ActionType]]:
    """Decode `(street, action_type)` pairs from a history blob.

    The check-vs-call distinction (bytes 0x01 vs 0x02) is collapsed back to
    `ActionType.CHECK_CALL`; the abstract type is what matters for strategy
    lookup, and the chip-amount information lives elsewhere.
    """
    pairs: list[tuple[int, ActionType]] = []
    street = 0
    for b in blob:
        if b == STREET_BOUNDARY_BYTE:
            street += 1
            continue
        pairs.append((street, byte_to_action_type(b)))
    return pairs


__all__ = [
    "CARD_BUCKET_MAX",
    "HASH_DIGEST_BYTES",
    "HISTORY_MAX",
    "PREFIX_FMT",
    "PREFIX_LEN",
    "STACK_BUCKETS",
    "VALID_STREETS",
    "VALID_TABLE_SIZES",
    "InfoSet",
    "decode_history",
    "encode_history",
]
