"""Spec.html §C tests for the InfoSet type + history encode/decode."""

from __future__ import annotations

import os
import random

import pytest

from pokerbot.abstraction import (
    AbstractAction,
    ActionType,
    InfoSet,
    decode_history,
    encode_history,
)


def test_byte_layout_matches_spec() -> None:
    """Spec.html §C: `InfoSet(6, 0, 2, 4, 99, b"").to_bytes()` == `bytes([6,0,2,4,99,0])`."""
    info = InfoSet(table_size=6, street=0, position=2, stack_bucket=4, card_bucket=99)
    assert info.to_bytes() == bytes([6, 0, 2, 4, 99, 0])


def test_endianness_card_bucket_u16_little() -> None:
    """`card_bucket=256` → bytes [0x00, 0x01] in the u16 LE slot."""
    info = InfoSet(table_size=6, street=0, position=0, stack_bucket=0, card_bucket=256)
    blob = info.to_bytes()
    assert blob[4:6] == bytes([0x00, 0x01])


def test_hash_collision_free_over_100k_random_infosets() -> None:
    """Use a wide enough source space that duplicate InfoSets are vanishingly rare."""
    rng = os.urandom
    table_sizes = (6, 8, 9)
    seen_hashes: set[bytes] = set()
    seen_inputs: set[bytes] = set()
    for _ in range(100_000):
        t = table_sizes[int.from_bytes(rng(1), "little") % 3]
        info = InfoSet(
            table_size=t,
            street=int.from_bytes(rng(1), "little") & 3,
            position=int.from_bytes(rng(1), "little") % t,
            stack_bucket=int.from_bytes(rng(1), "little") % 10,
            card_bucket=int.from_bytes(rng(2), "little"),  # full u16 range
            history=rng(int.from_bytes(rng(1), "little") % 64 + 1),
        )
        seen_inputs.add(info.to_bytes())
        seen_hashes.add(info.hash16())
    # No hash collision should occur beyond what's already due to duplicate inputs.
    assert len(seen_hashes) == len(seen_inputs)
    # And duplicates from the random source itself should be effectively zero.
    assert len(seen_inputs) >= 99_990, (
        f"random source produced too many duplicates: {len(seen_inputs)}"
    )


def test_invalid_table_size_rejected() -> None:
    """VALID_TABLE_SIZES = {2..9} (Cairn 5 Fix 2). Out-of-range sizes still rejected."""
    # 2..9 should all construct cleanly.
    for n in (2, 3, 4, 5, 6, 7, 8, 9):
        InfoSet(table_size=n, street=0, position=0, stack_bucket=0, card_bucket=0)
    # Out-of-range still rejected.
    with pytest.raises(ValueError, match="table_size"):
        InfoSet(table_size=10, street=0, position=0, stack_bucket=0, card_bucket=0)
    with pytest.raises(ValueError, match="table_size"):
        InfoSet(table_size=1, street=0, position=0, stack_bucket=0, card_bucket=0)


def test_history_cap_64_bytes() -> None:
    with pytest.raises(ValueError, match="history exceeds"):
        InfoSet(
            table_size=6,
            street=3,
            position=2,
            stack_bucket=4,
            card_bucket=199,
            history=b"\x00" * 65,
        )


def test_position_out_of_range() -> None:
    with pytest.raises(ValueError, match="position"):
        InfoSet(table_size=6, street=0, position=6, stack_bucket=0, card_bucket=0)


def test_card_bucket_max_u16() -> None:
    InfoSet(table_size=6, street=3, position=0, stack_bucket=0, card_bucket=65535)
    with pytest.raises(ValueError, match="card_bucket"):
        InfoSet(table_size=6, street=3, position=0, stack_bucket=0, card_bucket=65536)


# ─────────── §C history encode/decode ───────────


def _derive_pairs(
    actions: list[AbstractAction],
    boundaries: list[int],
) -> list[tuple[int, ActionType]]:
    pairs: list[tuple[int, ActionType]] = []
    cur_street = 0
    bnd_idx = 0
    for i, a in enumerate(actions):
        while bnd_idx < len(boundaries) and i == boundaries[bnd_idx]:
            cur_street += 1
            bnd_idx += 1
        pairs.append((cur_street, a.type))
    return pairs


def _random_action(rng: random.Random) -> AbstractAction:
    t = rng.choice(list(ActionType))
    if t == ActionType.CHECK_CALL:
        return AbstractAction(t, rng.choice([0, rng.randint(1, 50)]))
    if t == ActionType.FOLD:
        return AbstractAction(t, 0)
    return AbstractAction(t, rng.randint(1, 500))


def test_history_roundtrip_1000_random_sequences() -> None:
    """Spec §C: encode → decode is identity for 1000 random (street, ActionType) sequences."""
    rng = random.Random(0xC0DE)
    for _ in range(1000):
        n = rng.randint(1, 20)
        actions = [_random_action(rng) for _ in range(n)]
        # 0..min(3, n-1) random street boundaries, strictly increasing in [1, n].
        max_b = min(3, max(n - 1, 0))
        k = rng.randint(0, max_b)
        boundaries = sorted(rng.sample(range(1, n), k)) if (n > 1 and k > 0) else []
        blob = encode_history(actions, boundaries)
        assert decode_history(blob) == _derive_pairs(actions, boundaries)


def test_history_encode_65_actions_raises() -> None:
    """Spec §C: encoding a 65-action sequence exceeds the 64-byte cap and raises."""
    actions = [AbstractAction(ActionType.CHECK_CALL, 0) for _ in range(65)]
    with pytest.raises(ValueError, match="exceeds 64 bytes"):
        encode_history(actions, [])


def test_history_boundary_byte_emitted_between_streets() -> None:
    actions = [
        AbstractAction(ActionType.RAISE_2_5X, 5),
        AbstractAction(ActionType.CHECK_CALL, 5),
        AbstractAction(ActionType.BET_66, 7),
        AbstractAction(ActionType.CHECK_CALL, 7),
    ]
    blob = encode_history(actions, [2])
    # raise(0x20), call(0x02), boundary(0xF0), bet_66(0x11), call(0x02)
    assert blob == bytes([0x20, 0x02, 0xF0, 0x11, 0x02])


def test_history_check_vs_call_byte_distinction() -> None:
    actions = [
        AbstractAction(ActionType.CHECK_CALL, 0),  # check
        AbstractAction(ActionType.CHECK_CALL, 10),  # call
    ]
    blob = encode_history(actions, [])
    assert blob[0] == 0x01  # check
    assert blob[1] == 0x02  # call


def test_history_invalid_boundaries_raise() -> None:
    actions = [AbstractAction(ActionType.CHECK_CALL, 0) for _ in range(3)]
    with pytest.raises(ValueError, match="strictly increasing"):
        encode_history(actions, [2, 2])
    with pytest.raises(ValueError, match="strictly increasing"):
        encode_history(actions, [2, 1])
    with pytest.raises(ValueError, match="out of range"):
        encode_history(actions, [0])
    with pytest.raises(ValueError, match="out of range"):
        encode_history(actions, [4])
