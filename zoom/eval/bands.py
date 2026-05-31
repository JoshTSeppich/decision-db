"""Stage-1 behavioral band gate (Approach-2, Component 4).

The hard stop-gate the fine-tuned blueprint must clear. The bands are the absolute
targets from the plan — NOT a head-to-head delta against the old blueprint (that
relative-to-a-worse-baseline measure is the v5-6max failure). PASS requires EVERY
metric in range; a single out-of-range metric forces FAIL (no averaging a bad
metric away). `band_score` is the total distance outside the bands (0 = all in
range, lower = closer) — used to rank two policies (the should-be-better test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zoom.eval.profile import BehavioralProfile

# Absolute Stage-1 bands (percent), inclusive ranges:
#   VPIP 22-30, PFR 18-26, NON-COMMITTED preflop ALL_IN <1%, fold-to-c-bet 50-60.
#
# The ALL_IN band gates `noncommitted_all_in_preflop_pct` — shoves at spots that HAD a
# non-ALL_IN raise/bet alternative — NOT the raw rate. Structurally-forced (pot-committed)
# jams, where ALL_IN is the only legal aggression and folding is clearly -EV, are sound
# play, not the v5 indiscriminate-shoving leak; counting them conflated forced correct
# play with spew (measured the wrong thing). Raw `all_in_preflop_pct` is still reported.
BANDS: Final[dict[str, tuple[float, float]]] = {
    "vpip_pct": (22.0, 30.0),
    "pfr_pct": (18.0, 26.0),
    "noncommitted_all_in_preflop_pct": (0.0, 1.0),
    "fold_to_cbet_pct": (50.0, 60.0),
}

# 6-max GATED bands. These are NOT the 3-max numbers: with 3 more players to act
# behind, sound play folds more from early position, so VPIP/PFR run LOWER. They are
# also calibrated to THIS self-play harness, not HUD/field conventions — a known-sound
# scripted TAG (zoom.agents.archetypes.TagAgent) reads VPIP ~17 / PFR ~12 / nc-AI 0.0
# in 6-handed self-play here (a full table of tight players makes few pots, which
# compresses aggregate VPIP below the 22-30 HUD figures measured against a mixed field).
# Validated against the scripted archetypes (zoom.agents.archetypes): TAG PASSES all
# seeds; NIT (VPIP ~5, too tight), LAG (VPIP ~32, too loose), and STATION (VPIP ~67)
# all FAIL — and the loose v5-6max blueprint (VPIP 40-51, nc-AI ~8) FAILs on vpip+nc-AI.
#   * VPIP 15-28, PFR 11-22 — anchored on the TAG reference ± a reg spread (TAG ~17/12
#     in band; LAG ~32 and v5 ~40-51 above the ceiling; NIT ~5 below the floor).
#   * NON-COMMITTED preflop ALL_IN <1% — SEAT-COUNT-INDEPENDENT (discretionary
#     deep-stack shoving is a leak at any table size). The sharpest discriminator:
#     every scripted archetype reads 0.0; only the v5 blueprint spews (~8).
#
# fold-to-c-bet is DELIBERATELY NOT gated in 6-max. Validation showed it does not
# separate the classes: the tight scripted TAG over-folds (corrected 72-81%) while the
# loose v5 blueprint sits at a near-GTO 33-50%, so any band that passes TAG would pass
# v5 too. It is REPORTED as advisory via profile.fold_to_cbet_corrected_pct (the
# production fold_to_cbet_pct is a ~14x undercount in 6-max — it drops ALL_IN c-bets and
# every multiway facer — and must not be used). Revisit only with a solver-grounded
# reference and far more facing-c-bet samples.
BANDS_6MAX: Final[dict[str, tuple[float, float]]] = {
    "vpip_pct": (15.0, 28.0),
    "pfr_pct": (11.0, 22.0),
    "noncommitted_all_in_preflop_pct": (0.0, 1.0),
}


def bands_for(table_size: int) -> dict[str, tuple[float, float]]:
    """Behavioral bands appropriate to the table size (6-max is tighter than 3-max)."""
    return dict(BANDS_6MAX if table_size >= 6 else BANDS)


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    low: float
    high: float

    @property
    def in_band(self) -> bool:
        return self.low <= self.value <= self.high

    @property
    def distance(self) -> float:
        if self.value < self.low:
            return self.low - self.value
        if self.value > self.high:
            return self.value - self.high
        return 0.0


@dataclass(frozen=True)
class BandResult:
    metrics: tuple[MetricResult, ...]

    @property
    def passed(self) -> bool:
        """PASS iff EVERY metric is in band — a single failure forces FAIL."""
        return all(m.in_band for m in self.metrics)

    @property
    def score(self) -> float:
        """Total distance outside the bands (lower is better; 0 means PASS)."""
        return sum(m.distance for m in self.metrics)

    @property
    def failures(self) -> tuple[MetricResult, ...]:
        return tuple(m for m in self.metrics if not m.in_band)


def _metric_values(profile: BehavioralProfile, bands: Mapping[str, tuple[float, float]]) -> Mapping[str, float]:
    return {name: float(getattr(profile, name)) for name in bands}


def evaluate_bands(
    profile: BehavioralProfile,
    bands: Mapping[str, tuple[float, float]] | None = None,
) -> BandResult:
    """Evaluate a profile against the Stage-1 bands (PASS-iff-all).

    `bands` defaults to the 3-max `BANDS`; pass `bands_for(table_size)` (or
    `BANDS_6MAX`) to gate a 6-max policy against the tighter, harness-calibrated bands.
    """
    active = BANDS if bands is None else bands
    values = _metric_values(profile, active)
    return BandResult(
        metrics=tuple(
            MetricResult(name=name, value=values[name], low=lo, high=hi)
            for name, (lo, hi) in active.items()
        )
    )


def band_score(
    profile: BehavioralProfile,
    bands: Mapping[str, tuple[float, float]] | None = None,
) -> float:
    """Total out-of-band distance for `profile` (lower = better; 0 = all in band)."""
    return evaluate_bands(profile, bands).score


__all__ = [
    "BANDS",
    "BANDS_6MAX",
    "BandResult",
    "MetricResult",
    "band_score",
    "bands_for",
    "evaluate_bands",
]
