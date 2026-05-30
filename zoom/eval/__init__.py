"""Stage-1 evaluation harness (Approach-2, Component 4).

Self-validated independently of the training pipeline (so a training change can't
quietly validate itself):

  * `bands`   — the absolute behavioral gate (VPIP/PFR/ALL_IN%/fold-to-c-bet),
                PASS only when EVERY metric is in range; `band_score` ranks policies.
  * `profile` — behavioral profiling of a `SpotPolicy` via the production tally loop
                (reused read-only), and the ALL_IN-zeroed should-be-better transform.
  * `screens` — catastrophic-bug screens (folds AA / opens 72o), mechanical.

Lock 1: the should-be-better acceptance test (ALL_IN-zeroed policy scores strictly
better than raw) must be green before any fine-tune verdict means anything.
"""

from zoom.eval.bands import BANDS, BandResult, MetricResult, band_score, evaluate_bands
from zoom.eval.db_policy import make_db_spot_policy
from zoom.eval.profile import (
    BehavioralProfile,
    SpotPolicy,
    profile_spot_policy,
    zero_all_in_at_deep_stacks,
)
from zoom.eval.screens import ScreenCoverageError, catastrophic_screen

__all__ = [
    "ScreenCoverageError",
    "BANDS",
    "BandResult",
    "BehavioralProfile",
    "MetricResult",
    "SpotPolicy",
    "band_score",
    "catastrophic_screen",
    "evaluate_bands",
    "make_db_spot_policy",
    "profile_spot_policy",
    "zero_all_in_at_deep_stacks",
]
