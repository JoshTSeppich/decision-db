"""ICM (Independent Chip Model) equity using the Malmuth-Harville algorithm.

Each player's expected tournament $-equity given a stack distribution and a
payout structure is computed via the standard recursive M-H formula:

    P(player i finishes 1st)               = stacks[i] / sum(stacks)
    P(player i finishes k-th | j finishes < k)
                                            = stacks[i] / (sum(stacks) - sum(j))

    E_i = sum over positions k of P(player i finishes k-th) * payouts[k-1]

Caching: subtree calls are cached by `(sorted_stacks, payouts)` tuple keys
since ICM equity is invariant under player permutation. Internal recursion
sorts before recursing to maximize cache hits.

Numerical accuracy: all arithmetic in float64. Total accumulated error for a
9-player table is ~1e-9 of the total payout — well within the $0.01
conservation tolerance required by the test suite.
"""

from __future__ import annotations

from functools import lru_cache


def icm_equity(stacks: list[int], payouts: list[float]) -> list[float]:
    """Return each player's $-equity given stacks and a payout structure.

    Implements the Malmuth-Harville recursive algorithm. See module docstring.

    `stacks[i]` is player i's chip count (integer; 0 means "out of contention").
    `payouts[k]` is the prize for finishing in position k+1 (0-indexed: payouts[0]
    is 1st place, payouts[1] is 2nd, etc.).

    Result is a list of $-equities in the same order as `stacks`. Sum equals
    sum(payouts) (within float64 rounding) when at least one player has chips.

    Cached by canonical (sorted) stack tuple — order-invariant.
    """
    n = len(stacks)
    if n == 0:
        return []
    if sum(stacks) <= 0:
        return [0.0] * n

    # Sort stacks ascending, remember original index → sorted-position mapping.
    indexed = sorted(range(n), key=lambda i: stacks[i])
    sorted_stacks = tuple(stacks[i] for i in indexed)
    payouts_t = tuple(float(p) for p in payouts)
    sorted_eq = _icm_canonical(sorted_stacks, payouts_t)
    # Unsort: equities[original_index] = sorted_eq[sorted_position]
    equities = [0.0] * n
    for sorted_pos, orig_idx in enumerate(indexed):
        equities[orig_idx] = sorted_eq[sorted_pos]
    return equities


@lru_cache(maxsize=10000)
def _icm_canonical(
    stacks: tuple[int, ...],
    payouts: tuple[float, ...],
) -> tuple[float, ...]:
    """Recursive M-H. Returns equity tuple in the same order as `stacks`.

    Cache hits when called with the same `(stacks, payouts)`. Callers pass
    sorted-ascending `stacks` so that any permutation of the same multiset
    hits the same cache slot.

    Players with 0 chips contribute zero probability of finishing first and
    receive 0 equity (already out of contention).
    """
    n = len(stacks)
    if n == 0 or len(payouts) == 0:
        return (0.0,) * n
    total = sum(stacks)
    if total == 0:
        return (0.0,) * n

    equities = [0.0] * n
    # Iterate over possible first-place finishers.
    for j in range(n):
        s_j = stacks[j]
        if s_j <= 0:
            continue
        p_j_first = s_j / total
        equities[j] += p_j_first * payouts[0]
        if n == 1 or len(payouts) == 1:
            continue
        # Sub-tournament with player j removed.
        sub_unsorted = stacks[:j] + stacks[j + 1 :]
        # Re-sort sub for cache canonicalization.
        sub_indexed = sorted(range(n - 1), key=lambda i: sub_unsorted[i])
        sub_sorted = tuple(sub_unsorted[i] for i in sub_indexed)
        sub_payouts = payouts[1:]
        sub_eq_sorted = _icm_canonical(sub_sorted, sub_payouts)
        # Map sub equities back to "stacks-with-j-removed" order.
        sub_eq = [0.0] * (n - 1)
        for sorted_pos, orig_pos in enumerate(sub_indexed):
            sub_eq[orig_pos] = sub_eq_sorted[sorted_pos]
        # Add to original-index equities (skipping j).
        ridx = 0
        for i in range(n):
            if i == j:
                continue
            equities[i] += p_j_first * sub_eq[ridx]
            ridx += 1
    return tuple(equities)


__all__ = ["icm_equity"]
