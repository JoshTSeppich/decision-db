"""Card encoding bridge between our int 0..51 representation and phevaluator strings."""

from __future__ import annotations

from functools import lru_cache
from typing import Final

_RANK_CHARS: Final[str] = "23456789TJQKA"
_SUIT_CHARS: Final[str] = "cdhs"


@lru_cache(maxsize=52)
def card_int_to_str(card: int) -> str:
    """0..51 → 'As' / 'Td' / etc.  phevaluator-compatible."""
    if not 0 <= card < 52:
        raise ValueError(f"card out of range [0,52): {card}")
    return _RANK_CHARS[card >> 2] + _SUIT_CHARS[card & 3]


def cards_to_strs(cards: tuple[int, ...]) -> list[str]:
    return [card_int_to_str(c) for c in cards]


__all__ = ["card_int_to_str", "cards_to_strs"]
