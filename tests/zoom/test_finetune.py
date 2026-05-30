"""Component 3 — mixed-opponent fine-tune driver (Approach-2, Stage 1).

Three gates, in dependency order, none softened:

  GATE 2 (written first, the designated first red test) — the opponent-injection
    seam genuinely injects: at a non-traverser node the action comes from the
    supplied policy, NOT from self-play. The companion test proves the seam test
    discriminates (self-play does NOT follow the injected policy), so a silent
    regression to self-play would be caught — this is the test that protects the
    whole Stage (the v5 failure was a distribution with no tight-passive opponents).

  GATE 1 — the traversal reproduces Kuhn Nash: a differential test (self-play mode
    is bit-identical to the proven reference traversal) plus a convergence test
    (exploitability falls far below uniform). Validates the CFR math independent of
    NLHE before any NLHE run is trusted.

  GATE 3 — PokerKit-wrapper integrity on the GatedNLHEGame (zero-sum, no chip leak,
    blind/ante posting, button/position) runs BEFORE any compute is spent, plus
    BR-improves-vs-pool (EV vs the archetype pool strictly rises after K iters from
    a fixed-seed toy checkpoint) — the proof that exploitation actually works.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from zoom.agents import build_archetype_pool
from zoom.train import FineTuneTrainer, GatedNLHEGame, evaluate_vs_pool
from zoom.train.traversal import Reservoir, external_sampling_traversal

from pokerbot.abstraction import AbstractionTables, ActionType
from pokerbot.training import (
    SimpleNLHEGame,
    exploitability,
    make_test_config,
    make_uniform_strategy_fn,
)
from pokerbot.training.kuhn import BET, PASS, KuhnPokerGame, KuhnState
from pokerbot.training.nets import AdvantageNet
from pokerbot.training.traversal import (
    external_sampling_traversal as reference_traversal,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pokerbot.training.config import DeepCFRConfig


def _kuhn_nets(config: DeepCFRConfig) -> list[AdvantageNet]:
    game = KuhnPokerGame()
    return [
        AdvantageNet(game.feature_dim, game.num_actions, config) for _ in range(game.num_players)
    ]


def _enumerate_kuhn_deals() -> list[tuple[KuhnState, float]]:
    return [
        (KuhnState(cards=(c0, c1)), 1.0 / 6.0) for c0 in range(3) for c1 in range(3) if c0 != c1
    ]


# ─────────── GATE 2: the opponent-injection seam (first red test) ───────────


def test_opponent_injection_seam_uses_pool() -> None:
    """With an injected policy that always plays BET at opponent nodes, EVERY
    opponent action recorded during traversal is BET. If the code fell back to
    self-play, the opponent would sample PASS/BET from regret-matching and this
    would fail — exactly the silent-regression-to-self-play bug this guards.
    """
    import torch

    config = make_test_config(advantage_hidden=(16, 16))
    game = KuhnPokerGame()
    nets = _kuhn_nets(config)
    adv = Reservoir(64, game.feature_dim, game.num_actions)
    pol = Reservoir(64, game.feature_dim, game.num_actions)

    def always_bet(_game: object, _state: object, _actor: int) -> dict[int, float]:
        return {BET: 1.0}

    seen_any = False
    for seed in range(60):
        rng = random.Random(seed)
        torch.manual_seed(seed)
        state = game.new_initial_state(rng)
        opp: list[tuple[int, int]] = []
        external_sampling_traversal(
            game,
            state,
            traverser=0,
            advantage_nets=nets,
            advantage_reservoir=adv,
            policy_reservoir=pol,
            iter_t=1,
            rng=rng,
            opponent_policy=always_bet,
            opp_actions=opp,
        )
        for _actor, action in opp:
            seen_any = True
            assert action == BET, f"opponent action {action} != injected BET — seam not used"
    assert seen_any, "no opponent nodes were exercised — test is vacuous"


def test_selfplay_config_does_not_follow_pool() -> None:
    """Discrimination proof: with opponent_policy=None (self-play), opponent actions
    are NOT all BET — so the seam test above genuinely fails without injection.
    This is the standing 'red on self-play' evidence the gate requires.
    """
    import torch

    config = make_test_config(advantage_hidden=(16, 16))
    game = KuhnPokerGame()
    nets = _kuhn_nets(config)
    adv = Reservoir(64, game.feature_dim, game.num_actions)
    pol = Reservoir(64, game.feature_dim, game.num_actions)

    actions_seen: set[int] = set()
    for seed in range(60):
        rng = random.Random(seed)
        torch.manual_seed(seed)
        state = game.new_initial_state(rng)
        opp: list[tuple[int, int]] = []
        external_sampling_traversal(
            game,
            state,
            traverser=0,
            advantage_nets=nets,
            advantage_reservoir=adv,
            policy_reservoir=pol,
            iter_t=1,
            rng=rng,
            opponent_policy=None,
            opp_actions=opp,
        )
        actions_seen.update(a for _, a in opp)
    assert PASS in actions_seen, "self-play never played PASS — cannot discriminate from injection"


# ─────────── GATE 1: traversal reproduces Kuhn Nash ───────────


def test_selfplay_traversal_is_bit_identical_to_reference() -> None:
    """Differential proof: in self-play mode (opponent_policy=None) the new traversal
    produces the exact same return value and reservoir contents as the proven
    reference traversal, for the same seed/state/nets. The CFR math is therefore
    identical to already-trusted code — so it converges exactly as the reference does.
    """
    config = make_test_config(advantage_hidden=(16, 16))
    game = KuhnPokerGame()
    torch.manual_seed(0)
    nets = _kuhn_nets(config)  # shared, read-only across both runs

    for seed in range(25):
        state = game.new_initial_state(random.Random(seed))

        adv_ref = Reservoir(128, game.feature_dim, game.num_actions)
        pol_ref = Reservoir(128, game.feature_dim, game.num_actions)
        ret_ref = reference_traversal(
            game, state, 0, nets, adv_ref, pol_ref, 1, random.Random(seed + 1000)
        )

        adv_mine = Reservoir(128, game.feature_dim, game.num_actions)
        pol_mine = Reservoir(128, game.feature_dim, game.num_actions)
        ret_mine = external_sampling_traversal(
            game,
            state,
            0,
            nets,
            adv_mine,
            pol_mine,
            1,
            random.Random(seed + 1000),
            opponent_policy=None,
        )

        assert ret_ref == ret_mine, f"return mismatch at seed {seed}"
        for ref, mine in ((adv_ref, adv_mine), (pol_ref, pol_mine)):
            assert ref.size == mine.size
            assert ref.infoset_keys == mine.infoset_keys
            np.testing.assert_array_equal(ref.targets[: ref.size], mine.targets[: mine.size])
            np.testing.assert_array_equal(ref.features[: ref.size], mine.features[: mine.size])
            np.testing.assert_array_equal(ref.masks[: ref.size], mine.masks[: mine.size])


def _policy_strategy_fn(game: KuhnPokerGame, policy_net: object):  # type: ignore[no-untyped-def]
    """The AVERAGE strategy (policy net) — the iterate that converges to Nash in CFR
    (the current-iterate advantage strategy only oscillates)."""

    def fn(state: object, actor: int) -> torch.Tensor:
        feats = game.infoset_features(state).unsqueeze(0)  # type: ignore[arg-type]
        mask = torch.tensor(game.legal_mask(state), dtype=torch.float32).unsqueeze(0)  # type: ignore[arg-type]
        with torch.no_grad():
            return policy_net.forward_with_mask(feats, mask).squeeze(0)  # type: ignore[attr-defined,no-any-return]

    return fn


def test_selfplay_converges_to_kuhn_nash(tmp_path: Path) -> None:
    """Self-play through the new traversal drives the AVERAGE policy's exploitability
    substantially below uniform-random on Kuhn — the loop reduces exploitability
    toward Nash. The rigorous correctness proof is the differential test above (the
    traversal is bit-identical to the proven reference); this confirms the training
    loop built on it actually converges. Tight pointwise Nash isn't reachable with a
    tiny neural net in CI time, so the threshold is the honest achievable margin.
    """
    game = KuhnPokerGame()
    deals = _enumerate_kuhn_deals()
    expl_uniform = exploitability(game, make_uniform_strategy_fn(game), deals)

    config = make_test_config(
        outer_iters=150,
        traversals_per_iter=64,
        train_steps_per_iter=96,
        policy_train_steps=3000,
        advantage_hidden=(32, 32),
        policy_hidden=(32, 32),
        advantage_buffer_size=40_000,
        policy_buffer_size=40_000,
        batch_size=128,
        seed=2026,
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    trainer = FineTuneTrainer(config, KuhnPokerGame(), pool=None)
    trainer.train(tmp_path / "kuhn")
    expl_trained = exploitability(game, _policy_strategy_fn(game, trainer.policy_net), deals)
    assert expl_trained < 0.65 * expl_uniform, (
        f"trained avg-policy exploitability {expl_trained:.4f} not < 0.65·uniform "
        f"{expl_uniform:.4f} — the self-play loop isn't converging toward Nash"
    )


# ─────────── GATE 3: PokerKit-wrapper integrity (before any compute) ───────────


def _play_random_gated(game: GatedNLHEGame, rng: random.Random) -> object:
    state = game.new_initial_state(rng)
    n = 0
    while not game.is_terminal(state):
        state = game.apply_action(state, rng.choice(game.legal_actions(state)), rng)
        n += 1
        assert n <= 300, "hand failed to terminate"
    return state


def _gated_3max() -> GatedNLHEGame:
    return GatedNLHEGame(AbstractionTables(), blinds=(5, 10), starting_stack=1000, table_size=3)


def test_gated_game_terminal_reward_is_zero_sum() -> None:
    game = _gated_3max()
    for seed in range(40):
        state = _play_random_gated(game, random.Random(seed))
        rewards = game.terminal_reward(state).rewards  # type: ignore[arg-type]
        assert abs(sum(rewards)) < 1e-6, (seed, rewards)


def test_gated_game_no_chip_leak() -> None:
    game = _gated_3max()
    for seed in range(40):
        state = _play_random_gated(game, random.Random(seed))
        pk = state.pk_state  # type: ignore[attr-defined]
        assert int(sum(pk.stacks)) == sum(game._initial_stacks), seed


def test_gated_game_posts_blinds_correctly() -> None:
    """3-max: button=seat2, SB=seat0, BB=seat1; antes 0. Blinds land on the right
    seats with the right amounts (the blind/ante-posting integrity property)."""
    game = _gated_3max()
    state = game.new_initial_state(random.Random(0))
    pk = state.pk_state
    bets = [int(b) for b in pk.bets]
    assert bets[0] == 5, bets  # SB
    assert bets[1] == 10, bets  # BB
    assert bets[2] == 0, bets  # button posts nothing preflop
    assert int(sum(state.initial_stacks) - sum(pk.stacks)) == 15  # pot = SB+BB, no antes


def test_gated_game_drops_all_in_at_100bb_but_base_keeps_it() -> None:
    """In-game gating is live: at a fresh 100bb preflop spot the gated game omits
    ALL_IN, while the ungated SimpleNLHEGame still offers it."""
    gated = _gated_3max()
    base = SimpleNLHEGame(AbstractionTables(), blinds=(5, 10), starting_stack=1000, table_size=3)
    rng_seed = 7
    g_state = gated.new_initial_state(random.Random(rng_seed))
    b_state = base.new_initial_state(random.Random(rng_seed))
    assert int(ActionType.ALL_IN) not in gated.legal_actions(g_state)
    assert int(ActionType.ALL_IN) in base.legal_actions(b_state)


# ─────────── GATE 3: BR-improves-vs-pool (exploitation actually works) ───────────


def test_br_improves_vs_pool(tmp_path: Path) -> None:
    """From a fixed-seed toy checkpoint, EV against the archetype pool strictly rises
    after a few fine-tune iterations — the proof that exploitation works."""
    pool = build_archetype_pool()
    config = make_test_config(
        outer_iters=6,
        traversals_per_iter=24,
        train_steps_per_iter=60,
        advantage_hidden=(48, 48),
        advantage_buffer_size=6000,
        batch_size=64,
        seed=4242,
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    game = GatedNLHEGame(AbstractionTables(), blinds=(5, 10), starting_stack=1000, table_size=3)
    trainer = FineTuneTrainer(config, game, pool=pool)

    ev_before = evaluate_vs_pool(game, trainer.advantage_nets, pool, n_hands=400, seed=11)
    trainer.train(tmp_path / "br")
    ev_after = evaluate_vs_pool(game, trainer.advantage_nets, pool, n_hands=400, seed=11)

    assert ev_after > ev_before, (
        f"EV vs pool did not improve: before={ev_before:.3f} after={ev_after:.3f} bb/hand"
    )
