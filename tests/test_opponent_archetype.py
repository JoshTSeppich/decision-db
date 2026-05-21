"""Tests for ArchetypeClassifier threshold rules."""

from __future__ import annotations

from pokerbot.opponent.archetype import Archetype, ArchetypeClassifier
from pokerbot.opponent.stats import OpponentStats


def _stats(
    *,
    hands: int = 50,
    vpip: float,
    pfr: float,
    af: float,
    ftc: float = 0.5,
    cbet: float = 0.5,
    three_bet: float = 0.1,
) -> OpponentStats:
    """Build an OpponentStats with the requested rates.

    Sets counters so the rate methods return exactly the requested values
    (rounded to integer counters at `hands` denominator). AF and fold-to-cbet
    use separate denominators (postflop_calls and faced_cbet respectively).
    """
    pf_vol = round(vpip * hands)
    pf_raise = round(pfr * hands)
    pf_calls = 20
    pf_bets_plus_raises = round(af * pf_calls)
    # Split arbitrarily; classifier only sums them.
    pf_bets = pf_bets_plus_raises
    faced = 20
    folded = round(ftc * faced)
    cbet_opps = 20
    cbets = round(cbet * cbet_opps)
    pf_3bet_opps = 10
    pf_3bets = round(three_bet * pf_3bet_opps)
    return OpponentStats(
        hands_observed=hands,
        preflop_voluntary_actions=pf_vol,
        preflop_raises=pf_raise,
        preflop_3bets=pf_3bets,
        preflop_3bet_opportunities=pf_3bet_opps,
        postflop_bets=pf_bets,
        postflop_raises=0,
        postflop_calls=pf_calls,
        cbets=cbets,
        cbet_opportunities=cbet_opps,
        folds_to_cbet=folded,
        faced_cbet=faced,
    )


def test_unknown_below_threshold() -> None:
    c = ArchetypeClassifier()
    # 19 hands → Unknown regardless of other stats
    s = _stats(hands=19, vpip=0.30, pfr=0.20, af=2.5)
    assert c.classify(s) == Archetype.UNKNOWN


def test_station_classified() -> None:
    """VPIP=55%, AF=0.3, ftc=10% — well above margin at hands=100."""
    c = ArchetypeClassifier()
    s = _stats(hands=100, vpip=0.55, pfr=0.05, af=0.3, ftc=0.10)
    assert c.classify(s) == Archetype.STATION


def test_maniac_classified() -> None:
    """VPIP=60%, PFR=45%, AF=4.5 — well above margin at hands=100.

    ftc=0.60 keeps Station rule from matching (AF=4.5 NOT < 1.0).
    """
    c = ArchetypeClassifier()
    s = _stats(hands=100, vpip=0.60, pfr=0.45, af=4.5, ftc=0.60)
    assert c.classify(s) == Archetype.MANIAC


def test_nit_classified() -> None:
    """VPIP=10%, PFR=6%, AF=2.5 — well below upper bounds with margin."""
    c = ArchetypeClassifier()
    s = _stats(hands=100, vpip=0.10, pfr=0.06, af=2.5)
    assert c.classify(s) == Archetype.NIT


def test_tag_classified() -> None:
    """VPIP=22%, PFR=18%, AF=2.5 — center of TAG band (margin=0.04 leaves it inside)."""
    c = ArchetypeClassifier()
    s = _stats(hands=100, vpip=0.22, pfr=0.18, af=2.5)
    assert c.classify(s) == Archetype.TAG


def test_lag_classified() -> None:
    """VPIP=33%, PFR=28%, AF=3.0 — center of LAG band (margin=0.04 leaves it inside)."""
    c = ArchetypeClassifier()
    s = _stats(hands=100, vpip=0.33, pfr=0.28, af=3.0)
    assert c.classify(s) == Archetype.LAG


def test_station_priority_over_maniac() -> None:
    """A high-VPIP/high-PFR opponent who also has AF<1.0 and ftc<40% is a
    Station, not a Maniac. The Station rule (checked first) must claim them."""
    c = ArchetypeClassifier()
    s = _stats(hands=100, vpip=0.60, pfr=0.45, af=0.3, ftc=0.10)
    assert c.classify(s) == Archetype.STATION


# ───────── margin tests ─────────


def test_margin_progression() -> None:
    """Verify _margin returns the schedule values at representative hand counts."""
    c = ArchetypeClassifier()
    assert c._margin(15) == float("inf")
    assert c._margin(25) == 0.15
    assert c._margin(75) == 0.08
    assert c._margin(150) == 0.04
    assert c._margin(300) == 0.0


def test_margin_low_hands_requires_extreme_stats() -> None:
    """At 25 hands (margin=0.15), Maniac requires AF > 3.45 strictly.

    Pick boundary stats: AF=3.3 (with VPIP/PFR clearing) → Unknown;
    AF=3.6 → Maniac.
    """
    c = ArchetypeClassifier()
    s_low_af = _stats(hands=25, vpip=0.55, pfr=0.42, af=3.3, ftc=0.60)
    assert c.classify(s_low_af) == Archetype.UNKNOWN
    s_high_af = _stats(hands=25, vpip=0.55, pfr=0.42, af=3.6, ftc=0.60)
    assert c.classify(s_high_af) == Archetype.MANIAC


def test_margin_high_hands_uses_base_threshold() -> None:
    """At 300 hands (margin=0.0), AF=3.1 with VPIP/PFR clearing classifies Maniac."""
    c = ArchetypeClassifier()
    s = _stats(hands=300, vpip=0.41, pfr=0.31, af=3.1, ftc=0.60)
    assert c.classify(s) == Archetype.MANIAC


def test_borderline_observations_classify_as_unknown() -> None:
    """AF exactly at the threshold (=3.0) is Unknown at every hand count.

    At hands<200 the margin requires AF strictly above 3.0*(1+m) > 3.0;
    at hands>=200 the base threshold is `af > 3.0` (strict) which also fails
    for AF=3.0 exactly. VPIP and PFR are picked outside other archetype
    bands (Station gate's AF<1.0 also fails for AF=3.0).
    """
    c = ArchetypeClassifier()
    for hands in (50, 100, 199, 300):
        s = _stats(hands=hands, vpip=0.50, pfr=0.35, af=3.0, ftc=0.60)
        result = c.classify(s)
        assert result == Archetype.UNKNOWN, (
            f"hands={hands}: AF=3.0 exact should be Unknown, got {result}"
        )
