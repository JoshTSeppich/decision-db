"""Catastrophic-bug screens (Component 4).

Mechanically checkable from the policy itself — no external GTO source needed. A
sane NLHE policy never folds AA preflop and never opens 72o from early position;
either is an unambiguous catastrophe. The screen queries the policy at the relevant
constructed spots and flags violations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pokerbot.abstraction import ActionType

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from zoom.agents import AgentSpot
    from zoom.eval.profile import SpotPolicy

# Card ints (rank*4 + suit; A=rank12, 7=rank5, 2=rank0).
_AA: Final[tuple[int, int]] = (48, 49)  # Ac Ad
_SEVEN_TWO_OFF: Final[tuple[int, int]] = (1, 20)  # 2d 7c (offsuit)

_RAISE_TYPES: Final[frozenset[ActionType]] = frozenset(
    {
        ActionType.RAISE_2_5X,
        ActionType.RAISE_3_5X,
        ActionType.ALL_IN,
        ActionType.BET_33,
        ActionType.BET_66,
        ActionType.BET_100,
        ActionType.BET_150,
    }
)

_BB: Final[int] = 2
_DEEP_STACK: Final[int] = 200  # 100bb — catastrophes are most clearly wrong deep


class ScreenCoverageError(LookupError):
    """A screen probe hit no DB row, so the policy could not be evaluated there.

    Raised instead of silently returning "clean": a screen that can't distinguish a real
    pass from a vacuous one (the policy was never actually seen) is not a screen. Build
    the probe to land on a populated infoset, but never DEPEND on that guess — verify the
    hit and fail loud on a miss.
    """


def _prob_of(dist: Mapping[ActionType, float], types: Iterable[ActionType]) -> float:
    total = sum(p for p in dist.values() if p > 0)
    if total <= 0:
        return 0.0
    return sum(max(0.0, dist.get(t, 0.0)) for t in types) / total


def _spot(hole: tuple[int, int], *, position: int, to_call: int, pot: int) -> AgentSpot:
    from zoom.agents import AgentSpot

    return AgentSpot(
        hole=hole,
        board=(),
        street="preflop",
        position=position,
        pot=pot,
        to_call=to_call,
        stack=_DEEP_STACK,
        min_raise=_BB,
        # Effective stack a blinds-posted ~100bb table actually produces (opponent posted
        # the BB), so the probe queries the bucket the export wrote rather than the empty
        # full-stack bucket. Best-effort: a wrong guess yields a miss, which `_probe`
        # turns into a loud ScreenCoverageError — never a silent vacuous "clean".
        effective_stack=_DEEP_STACK - _BB,
    )


def _probe(policy: SpotPolicy, spot: AgentSpot, label: str) -> Mapping[ActionType, float]:
    """Query `policy` at `spot`, asserting a real (non-empty) distribution came back.

    A DB-backed policy returns an empty mapping on a coverage miss; treat that as
    failure-to-evaluate and raise, so the screen can never silently pass a policy it
    never actually saw at the probe infoset.
    """
    dist = policy(spot)
    if not dist or sum(p for p in dist.values() if p > 0) <= 0:
        raise ScreenCoverageError(
            f"no DB coverage at screen probe '{label}' "
            f"(table_size={spot.table_size} position={spot.position} "
            f"eff_stack={spot.effective_stack}) — cannot certify the policy here"
        )
    return dist


_BUTTON: Final[int] = 2  # SB-relative button position in 3-max (the first actor / opener)


def catastrophic_screen(policy: SpotPolicy) -> list[str]:
    """Return a list of catastrophic-bug descriptions (empty = clean).

    Probes the BUTTON OPEN (`history=b""`, first-in) — the one spot a history-aware DB
    lookup reads faithfully, because only the first actor has an empty-history infoset.
    SB/BB "facing a bet" probes are deliberately NOT synthesized: their real rows carry
    a betting history this constructed spot can't reproduce, so the lookup would read a
    non-representative row. `_probe` fails loud (`ScreenCoverageError`) on a miss, so the
    screen can never silently pass a policy it didn't actually observe.
    """
    violations: list[str] = []

    # AA must never be FOLDED as a first-in open.
    aa = _probe(policy, _spot(_AA, position=_BUTTON, to_call=0, pot=3), "AA button-open")
    if _prob_of(aa, {ActionType.FOLD}) > 0.5:
        violations.append("folds AA preflop (button open)")

    # 72o must never be OPENED (raised) first-in.
    junk = _probe(policy, _spot(_SEVEN_TWO_OFF, position=_BUTTON, to_call=0, pot=3), "72o button-open")
    if _prob_of(junk, _RAISE_TYPES) > 0.5:
        violations.append("opens 72o UTG (button open)")

    return violations


__all__ = ["ScreenCoverageError", "catastrophic_screen"]
