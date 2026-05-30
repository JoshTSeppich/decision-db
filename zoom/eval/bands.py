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
#   VPIP 22-30, PFR 18-26, preflop ALL_IN <1%, fold-to-c-bet 50-60.
BANDS: Final[dict[str, tuple[float, float]]] = {
    "vpip_pct": (22.0, 30.0),
    "pfr_pct": (18.0, 26.0),
    "all_in_preflop_pct": (0.0, 1.0),
    "fold_to_cbet_pct": (50.0, 60.0),
}


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


def _metric_values(profile: BehavioralProfile) -> Mapping[str, float]:
    return {name: float(getattr(profile, name)) for name in BANDS}


def evaluate_bands(profile: BehavioralProfile) -> BandResult:
    """Evaluate a profile against the Stage-1 bands (PASS-iff-all)."""
    values = _metric_values(profile)
    return BandResult(
        metrics=tuple(
            MetricResult(name=name, value=values[name], low=lo, high=hi)
            for name, (lo, hi) in BANDS.items()
        )
    )


def band_score(profile: BehavioralProfile) -> float:
    """Total out-of-band distance for `profile` (lower = better; 0 = all in band)."""
    return evaluate_bands(profile).score


__all__ = ["BANDS", "BandResult", "MetricResult", "band_score", "evaluate_bands"]
