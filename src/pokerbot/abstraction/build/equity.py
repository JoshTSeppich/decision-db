"""Monte Carlo equity computation backed by phevaluator (Spec.html §A).

`hand_strength`: P(hero wins at showdown | random opp hand + random remaining board).
                 Used for EHS² histograms (potential-aware bucketing on flop+turn).

`hand_strength_vs_cluster`: P(hero wins | opp hand from a given preflop cluster).
                           Used for OCHS (river bucketing).

Both functions take ints (0..51) and use phevaluator strings internally. Lower
phevaluator rank = stronger hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phevaluator.evaluator import evaluate_cards

from pokerbot.abstraction.build.cards import card_int_to_str

if TYPE_CHECKING:
    import random


def _remaining_deck(used: tuple[int, ...]) -> list[int]:
    used_set = set(used)
    return [c for c in range(52) if c not in used_set]


def hand_strength(
    hole: tuple[int, int],
    board: tuple[int, ...],
    *,
    num_samples: int,
    rng: random.Random,
) -> float:
    """P(hero wins) ± ties at showdown, given a random opp hand and random
    remaining board cards. `board` may be 0, 3, 4, or 5 cards. `num_samples`
    is the number of MC samples.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive: {num_samples}")
    used = (*hole, *board)
    if len(set(used)) != len(used):
        raise ValueError(f"duplicate cards in hole+board: {used}")
    deck = _remaining_deck(used)
    cards_to_draw_board = 5 - len(board)
    needed = 2 + cards_to_draw_board  # opp hole + remaining board

    hole_strs = (card_int_to_str(hole[0]), card_int_to_str(hole[1]))
    board_strs = tuple(card_int_to_str(c) for c in board)

    wins = 0.0
    for _ in range(num_samples):
        sample = rng.sample(deck, needed)
        opp = sample[:2]
        rest_board = sample[2:]
        full_board = (*board_strs, *(card_int_to_str(c) for c in rest_board))
        my_rank = evaluate_cards(*hole_strs, *full_board)
        opp_rank = evaluate_cards(card_int_to_str(opp[0]), card_int_to_str(opp[1]), *full_board)
        if my_rank < opp_rank:
            wins += 1.0
        elif my_rank == opp_rank:
            wins += 0.5
    return wins / num_samples


def hand_strength_vs_cluster(
    hole: tuple[int, int],
    board: tuple[int, ...],
    cluster: tuple[tuple[int, int], ...],
    *,
    num_samples: int,
    rng: random.Random,
) -> float:
    """P(hero wins) ± ties when opp hand is drawn from `cluster` (a tuple of
    canonical hole-card pairs in 0..51 form). Hands that conflict with the
    hero's hole/board are skipped — falls back to uniform if all conflict.
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive: {num_samples}")
    if not cluster:
        raise ValueError("cluster must contain at least one hand")
    used = set(hole) | set(board)
    eligible = [pair for pair in cluster if used.isdisjoint(pair)]
    if not eligible:
        return 0.5  # no representative in cluster — neutral
    deck = _remaining_deck((*hole, *board))
    cards_to_draw_board = 5 - len(board)

    hole_strs = (card_int_to_str(hole[0]), card_int_to_str(hole[1]))
    board_strs = tuple(card_int_to_str(c) for c in board)

    wins = 0.0
    for _ in range(num_samples):
        opp = rng.choice(eligible)
        remaining = [c for c in deck if c not in opp]
        rest_board = rng.sample(remaining, cards_to_draw_board) if cards_to_draw_board > 0 else []
        full_board = (*board_strs, *(card_int_to_str(c) for c in rest_board))
        my_rank = evaluate_cards(*hole_strs, *full_board)
        opp_rank = evaluate_cards(card_int_to_str(opp[0]), card_int_to_str(opp[1]), *full_board)
        if my_rank < opp_rank:
            wins += 1.0
        elif my_rank == opp_rank:
            wins += 0.5
    return wins / num_samples


__all__ = [
    "hand_strength",
    "hand_strength_vs_cluster",
]
