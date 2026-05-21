"""Tests for ICM equity (Malmuth-Harville). Cairn 1 deliverable.

Acceptance criteria from the cairn brief:
    1. test_two_player_winner_takes_all
    2. test_two_player_proportional
    3. test_three_player_published_example     ← strict-mode tripwire
    4. test_nine_player_final_table_conservation
    5. test_cache_speedup
    6. test_zero_stack_excluded
    7. test_order_invariance
"""

from __future__ import annotations

import time

import pytest

from pokerbot.tournament.icm import _icm_canonical, icm_equity

# ───────── 1. two-player winner-takes-all, equal stacks ─────────


def test_two_player_winner_takes_all() -> None:
    """Two equal stacks, all prize on 1st. Each player's equity = 50."""
    equities = icm_equity([1000, 1000], [100.0, 0.0])
    assert equities[0] == pytest.approx(50.0, abs=1e-9)
    assert equities[1] == pytest.approx(50.0, abs=1e-9)
    assert sum(equities) == pytest.approx(100.0, abs=1e-9)


# ───────── 2. two-player proportional ─────────


def test_two_player_proportional() -> None:
    """Two players, stacks [70, 30], payouts [100, 0]. Equities = [70, 30]."""
    equities = icm_equity([70, 30], [100.0, 0.0])
    assert equities[0] == pytest.approx(70.0, abs=1e-9)
    assert equities[1] == pytest.approx(30.0, abs=1e-9)


# ───────── 3. three-player published example ─────────


def test_three_player_published_example() -> None:
    """Three players, stacks [5000, 3000, 2000], payouts [50, 30, 20].

    Expected M-H equities: [38.3929, 32.7500, 28.8571] (analytical, abs tol 0.001).

    Original brief quoted [37.8, 31.2, 31.0] which appears to be from a
    non-M-H ICM variant (possibly HRC-style future-game simulation or
    ITM-adjusted model). Pure Malmuth-Harville produces [38.39, 32.75, 28.86]
    for these inputs. Verified by hand:
        P(player 0 finishes 1st) = 5000/10000 = 0.5;
        P(2nd) = (3000/10000)(5000/7000) + (2000/10000)(5000/8000) = 0.3393;
        P(3rd) = 0.1607;
        equity = 50(0.5) + 30(0.3393) + 20(0.1607) = 38.393.
    """
    equities = icm_equity([5000, 3000, 2000], [50.0, 30.0, 20.0])
    expected = [38.3929, 32.7500, 28.8571]
    for i, (got, want) in enumerate(zip(equities, expected, strict=True)):
        assert got == pytest.approx(want, abs=0.001), (
            f"player {i}: got {got:.4f}, expected {want}; diff {got - want:+.4f}"
        )


# ───────── 4. nine-player conservation ─────────


def test_nine_player_final_table_conservation() -> None:
    """Sum of equities equals sum of payouts within $0.01."""
    stacks = [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 500]
    # Standard tournament payout structure for top 9
    payouts = [10000.0, 6000.0, 4000.0, 2500.0, 1750.0, 1250.0, 850.0, 600.0, 400.0]
    equities = icm_equity(stacks, payouts)
    assert len(equities) == len(stacks)
    assert sum(equities) == pytest.approx(sum(payouts), abs=0.01)


# ───────── 5. cache speedup ─────────


def test_cache_speedup() -> None:
    """1000 identical calls; last 999 should be cache hits (well under 10ms total)."""
    stacks = [10000, 8000, 6000, 5000, 4000, 3000, 2000, 1500, 500]
    payouts = [10000.0, 6000.0, 4000.0, 2500.0, 1750.0, 1250.0, 850.0, 600.0, 400.0]
    _icm_canonical.cache_clear()
    # warm
    _ = icm_equity(stacks, payouts)
    t0 = time.perf_counter()
    for _ in range(999):
        _ = icm_equity(stacks, payouts)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 10.0, f"999 cached calls took {elapsed_ms:.2f}ms (expected < 10ms)"


# ───────── 6. zero-stack excluded ─────────


def test_zero_stack_excluded() -> None:
    """Player with 0 chips contributes 0 to others and has 0 equity themselves."""
    # 4 players, one with zero stack. Expected: zero-stack gets 0, the other
    # three split equity as if the zero-stack didn't exist.
    stacks_with_zero = [5000, 3000, 2000, 0]
    payouts = [50.0, 30.0, 20.0, 0.0]
    eq_with_zero = icm_equity(stacks_with_zero, payouts)
    eq_three_only = icm_equity([5000, 3000, 2000], [50.0, 30.0, 20.0])
    assert eq_with_zero[3] == pytest.approx(0.0, abs=1e-9)
    for i in range(3):
        assert eq_with_zero[i] == pytest.approx(eq_three_only[i], abs=1e-6), (
            f"player {i}: with-zero {eq_with_zero[i]} vs three-only {eq_three_only[i]}"
        )


# ───────── 7. order invariance ─────────


def test_order_invariance() -> None:
    """Equities for permuted stacks must permute correspondingly."""
    eq_a = icm_equity([5000, 3000, 2000], [50.0, 30.0, 20.0])
    eq_b = icm_equity([2000, 5000, 3000], [50.0, 30.0, 20.0])
    # eq_b is the permutation where player 0 has 2000, player 1 has 5000, player 2 has 3000.
    # So eq_b[0] should equal eq_a[2] (the 2000-stack equity), etc.
    assert eq_b[0] == pytest.approx(eq_a[2], abs=1e-9)
    assert eq_b[1] == pytest.approx(eq_a[0], abs=1e-9)
    assert eq_b[2] == pytest.approx(eq_a[1], abs=1e-9)
    # And the multiset of equities is identical.
    assert sorted(eq_a) == pytest.approx(sorted(eq_b), abs=1e-9)
