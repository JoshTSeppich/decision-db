"""Field generation for tournament simulation.

Produces 90 bots (45 archetype + 45 randomized) per the Cairn 5 spec. Each
bot is a `BotConfig` (parameters) paired with a `ParameterizedBot` adapter
that maps the params to action distributions via the existing infoset
abstraction (no DB lookup, no CFR — pure rule logic).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pokerbot.abstraction import ActionType
from pokerbot.runtime.default_policy import _PREFLOP_PERCENTILE
from pokerbot.runtime.opponent import ObservedHistory, OpponentModel

if TYPE_CHECKING:
    from pokerbot.abstraction import InfoSet

# Action history byte ranges (mirror eval_archetypes.py).
_STREET_BOUNDARY: int = 0xF0
_AGGRESSION_BYTES: frozenset[int] = frozenset({0x10, 0x11, 0x12, 0x13, 0x14, 0x20, 0x21})


def _bucket_pct(bucket_id: int) -> float:
    return _PREFLOP_PERCENTILE.get(bucket_id, 1.0)


def _split_streets(history: bytes) -> list[bytes]:
    if not history:
        return [b""]
    parts: list[bytearray] = [bytearray()]
    for b in history:
        if b == _STREET_BOUNDARY:
            parts.append(bytearray())
        else:
            parts[-1].append(b)
    return [bytes(p) for p in parts]


def _count_pf_raises_in_history(history: bytes) -> int:
    streets = _split_streets(history)
    return sum(1 for b in streets[0] if b in _AGGRESSION_BYTES)


def _facing_aggression_this_street(history: bytes) -> bool:
    streets = _split_streets(history)
    if not streets:
        return False
    current = streets[-1]
    return len(current) > 0 and current[-1] in _AGGRESSION_BYTES


def _postflop_strength(card_bucket: int) -> float:
    return min(max(card_bucket, 0), 199) / 199.0


# ─────────── data structures ───────────


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Behavioral parameters that drive a `ParameterizedBot`.

    Fields:
        archetype       label: 'nit', 'default', 'tag', 'lag', 'maniac',
                        'station', or 'randomized'
        vpip            fraction of preflop card-buckets played voluntarily
        pfr             fraction of preflop card-buckets raised
        af              postflop aggression factor target (bets+raises)/calls
        cbet            cbet frequency as preflop aggressor
        fold_to_cbet    fraction of flop bets folded to as caller
        three_bet       fraction of card-buckets that 3-bet a prior raise
    """

    archetype: str
    vpip: float
    pfr: float
    af: float
    cbet: float = 0.60
    fold_to_cbet: float = 0.50
    three_bet: float = 0.08


# ─────────── ParameterizedBot ───────────


class ParameterizedBot(OpponentModel):
    """OpponentModel driven by a `BotConfig`. No DB lookups — fast enough for
    tournament-scale self-play across 89 non-hero seats per table."""

    def __init__(self, config: BotConfig) -> None:
        self.cfg = config

    def adjust(
        self,
        infoset: InfoSet,
        base_probs: dict[ActionType, float],  # noqa: ARG002
        observed_history: ObservedHistory,  # noqa: ARG002
    ) -> dict[ActionType, float]:
        if infoset.street == 0:
            return self._preflop(infoset)
        return self._postflop(infoset)

    def _preflop(self, infoset: InfoSet) -> dict[ActionType, float]:
        pct = _bucket_pct(infoset.card_bucket)
        n_raises = _count_pf_raises_in_history(infoset.history)
        if n_raises == 0:
            # First-in: raise top PFR%, limp PFR%..VPIP%, fold rest.
            if pct < self.cfg.pfr:
                return {ActionType.RAISE_2_5X: 1.0}
            if pct < self.cfg.vpip:
                return {ActionType.CHECK_CALL: 1.0}
            return {ActionType.FOLD: 1.0}
        # Facing 1+ raises.
        if pct < self.cfg.three_bet:
            return {ActionType.RAISE_2_5X: 1.0}
        # Flat-call window: half of VPIP-3bet gap to keep multi-way pots open.
        flat_cap = self.cfg.three_bet + max(self.cfg.vpip - self.cfg.three_bet, 0.0) * 0.5
        if pct < flat_cap:
            return {ActionType.CHECK_CALL: 1.0}
        return {ActionType.FOLD: 1.0}

    def _postflop(self, infoset: InfoSet) -> dict[ActionType, float]:
        strength = _postflop_strength(infoset.card_bucket)
        facing = _facing_aggression_this_street(infoset.history)
        if not facing:
            # No bet to face — bet at cbet rate weighted by strength.
            if strength > (1.0 - self.cfg.cbet):
                return {ActionType.BET_66: 1.0}
            return {ActionType.CHECK_CALL: 1.0}
        # Facing a bet: fold the weakest fold_to_cbet fraction; otherwise
        # continue. Of those that continue, raise the top by `agg_share`,
        # call the rest.
        if strength < self.cfg.fold_to_cbet:
            return {ActionType.FOLD: 1.0}
        # Aggression share converts AF target into a raise/call split among
        # continuing hands. AF = raises / calls → raise_frac = AF / (AF + 1).
        agg_share = self.cfg.af / (self.cfg.af + 1.0)
        if strength > (1.0 - agg_share):
            return {ActionType.RAISE_2_5X: 1.0}
        return {ActionType.CHECK_CALL: 1.0}


# ─────────── archetype roster ───────────

ARCHETYPE_CONFIGS: dict[str, BotConfig] = {
    "nit": BotConfig(
        archetype="nit", vpip=0.13, pfr=0.09, af=1.8, cbet=0.45, fold_to_cbet=0.65, three_bet=0.03
    ),
    "default": BotConfig(
        archetype="default", vpip=0.20, pfr=0.14, af=2.0, cbet=0.55, fold_to_cbet=0.50, three_bet=0.06
    ),
    "tag": BotConfig(
        archetype="tag", vpip=0.22, pfr=0.18, af=2.3, cbet=0.65, fold_to_cbet=0.45, three_bet=0.09
    ),
    "lag": BotConfig(
        archetype="lag", vpip=0.32, pfr=0.26, af=2.8, cbet=0.75, fold_to_cbet=0.35, three_bet=0.14
    ),
    "maniac": BotConfig(
        archetype="maniac", vpip=0.50, pfr=0.38, af=3.5, cbet=0.85, fold_to_cbet=0.25, three_bet=0.22
    ),
    "station": BotConfig(
        archetype="station", vpip=0.45, pfr=0.06, af=0.6, cbet=0.30, fold_to_cbet=0.20, three_bet=0.02
    ),
}


def _sample_randomized(rng: random.Random) -> BotConfig:
    return BotConfig(
        archetype="randomized",
        vpip=rng.uniform(0.12, 0.38),
        pfr=rng.uniform(0.08, 0.30),
        af=rng.uniform(0.8, 3.5),
        cbet=rng.uniform(0.35, 0.85),
        fold_to_cbet=rng.uniform(0.25, 0.65),
        three_bet=rng.uniform(0.03, 0.18),
    )


def generate_field(
    *,
    archetype_counts: dict[str, int] | None = None,
    n_randomized: int = 45,
    rng_seed: int = 0,
) -> list[BotConfig]:
    """Produce a 90-bot field. Spec calls for 45 archetypes + 45 randomized.

    Default archetype distribution (9 each of nit/default/lag/maniac/station)
    chosen so the archetypes sum to 45. The spec's literal "12 Nit, 12 Default,
    12 LAG, 9 Maniac, 12 Station" actually sums to 57 — the spec was
    internally inconsistent with its "45 archetypes" header. Adjudicated to
    9 each per the "45 + 45 = 90" test target.
    """
    if archetype_counts is None:
        archetype_counts = {"nit": 9, "default": 9, "lag": 9, "maniac": 9, "station": 9}
    rng = random.Random(rng_seed)
    field: list[BotConfig] = []
    for label, count in archetype_counts.items():
        cfg = ARCHETYPE_CONFIGS[label]
        for _ in range(count):
            field.append(cfg)
    for _ in range(n_randomized):
        field.append(_sample_randomized(rng))
    rng.shuffle(field)
    return field


__all__ = [
    "ARCHETYPE_CONFIGS",
    "BotConfig",
    "ParameterizedBot",
    "generate_field",
]
