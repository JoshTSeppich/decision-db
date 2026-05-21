"""Shared type aliases for the abstraction modules.

Lives in its own module so `actions.py` (lightweight) and `cards.py` (numpy)
can share a `Street` literal without bringing numpy along for the ride.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

Street: TypeAlias = Literal["preflop", "flop", "turn", "river"]
Card: TypeAlias = int  # 0..51 ; card_id = rank*4 + suit

__all__ = ["Card", "Street"]
