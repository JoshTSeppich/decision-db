"""Kuhn poker — tiny 2-player game used to validate the trainer pipeline.

Rules (standard):
    - 3-card deck (J=0, Q=1, K=2).
    - Each player antes 1, is dealt 1 card.
    - Player 0 acts first: pass (0) or bet (1).
    - If pass: P1 may pass (showdown, higher card wins 1) or bet.
    - If a bet faces, the other player may fold (lose 1) or call (showdown, higher card wins 2).

We choose Kuhn for tests because:
    - 12 information sets total; CFR converges in <100 iterations.
    - Closed-form game-theoretic optimal exists, so LBR is meaningful.
    - 2 players is the simplest non-trivial CFR case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from pokerbot.training.game import Game, TerminalReward

if TYPE_CHECKING:
    import random


PASS: int = 0
BET: int = 1

NUM_CARDS: int = 3
NUM_ACTIONS: int = 2
NUM_PLAYERS: int = 2


@dataclass(frozen=True, slots=True)
class KuhnState:
    """Kuhn-poker state. `history` is a tuple of action ids in order taken."""

    cards: tuple[int, int]  # cards[i] = player i's private card
    history: tuple[int, ...] = ()

    @property
    def turn(self) -> int:
        return len(self.history) % NUM_PLAYERS


class KuhnPokerGame(Game[KuhnState]):
    """2-player Kuhn poker."""

    num_players: int = NUM_PLAYERS
    num_actions: int = NUM_ACTIONS

    # 3 (card one-hot) + 4 (history one-hot: empty/pass/bet/pass-bet) per player slot.
    # We use a flat encoding: own_card_onehot(3) + history_indicator(4).
    feature_dim: int = NUM_CARDS + 4

    def new_initial_state(self, rng: random.Random) -> KuhnState:
        deck = [0, 1, 2]
        rng.shuffle(deck)
        return KuhnState(cards=(deck[0], deck[1]))

    def is_terminal(self, state: KuhnState) -> bool:
        h = state.history
        if len(h) < 2:
            return False
        # Terminal patterns: pass-pass, bet-pass, bet-call, pass-bet-pass, pass-bet-call
        return h in {
            (PASS, PASS),
            (BET, PASS),
            (BET, BET),
            (PASS, BET, PASS),
            (PASS, BET, BET),
        }

    def current_player(self, state: KuhnState) -> int:
        return state.turn

    def legal_actions(self, state: KuhnState) -> tuple[int, ...]:  # noqa: ARG002
        return (PASS, BET)

    def apply_action(self, state: KuhnState, action: int, rng: random.Random) -> KuhnState:  # noqa: ARG002
        if action not in (PASS, BET):
            raise ValueError(f"illegal action: {action}")
        return KuhnState(cards=state.cards, history=(*state.history, action))

    def terminal_reward(self, state: KuhnState) -> TerminalReward:
        if not self.is_terminal(state):
            raise ValueError("state is not terminal")
        h = state.history
        p0_card, p1_card = state.cards

        # Stake (per-player chips at risk beyond the ante)
        if h == (PASS, PASS):
            showdown_pot = 1
            folded = None
        elif h == (BET, PASS):
            showdown_pot = 0  # P1 folded; P0 wins the 1-chip ante
            folded = 1
        elif h == (BET, BET):
            showdown_pot = 2  # both put in 1 ante + 1 bet
            folded = None
        elif h == (PASS, BET, PASS):
            showdown_pot = 0  # P0 folded
            folded = 0
        else:  # (PASS, BET, BET)
            showdown_pot = 2
            folded = None

        if folded is not None:
            winner = 1 - folded
            payoff_winner = 1.0  # win the opponent's ante
            return TerminalReward(
                rewards=tuple(payoff_winner if i == winner else -1.0 for i in range(NUM_PLAYERS))
            )

        winner = 0 if p0_card > p1_card else 1
        payoff_winner = float(showdown_pot)
        return TerminalReward(
            rewards=tuple(
                payoff_winner if i == winner else -payoff_winner for i in range(NUM_PLAYERS)
            )
        )

    def infoset_key(self, state: KuhnState) -> bytes:
        actor = self.current_player(state)
        return bytes([state.cards[actor], *state.history])

    def infoset_features(self, state: KuhnState) -> torch.Tensor:
        actor = self.current_player(state)
        feats = [0.0] * self.feature_dim
        feats[state.cards[actor]] = 1.0  # own card one-hot
        # History indicator: bit per encountered action up to depth 4
        history_slot = NUM_CARDS
        history_idx = 0
        for a in state.history:
            history_idx = (history_idx << 1) | a
        if history_idx < 4:
            feats[history_slot + history_idx] = 1.0
        return torch.tensor(feats, dtype=torch.float32)


__all__ = [
    "BET",
    "PASS",
    "KuhnPokerGame",
    "KuhnState",
]
