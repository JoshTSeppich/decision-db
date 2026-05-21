"""Deep CFR training pipeline (Spec.html §E).

Production target: 6/8/9-max NLHE. Test path: KuhnPokerGame. The Trainer is
game-agnostic — pass any `Game` impl. See `deepcfr.make_test_config` for
the tiny config used by the §E tests.
"""

from pokerbot.training.config import DeepCFRConfig, EvalResult
from pokerbot.training.deepcfr import Trainer, make_test_config
from pokerbot.training.export import (
    decode_nlhe_infoset,
    export_strategy_from_reservoir,
)
from pokerbot.training.game import Game, TerminalReward
from pokerbot.training.kuhn import KuhnPokerGame
from pokerbot.training.lbr import (
    exploitability,
    local_best_response,
    make_advantage_strategy_fn,
    make_uniform_strategy_fn,
)
from pokerbot.training.nets import AdvantageNet, PolicyNet, regret_match
from pokerbot.training.nlhe_game import NLHEState, SimpleNLHEGame
from pokerbot.training.traversal import (
    Reservoir,
    TraversalStats,
    external_sampling_traversal,
)

__all__ = [
    "AdvantageNet",
    "DeepCFRConfig",
    "EvalResult",
    "Game",
    "KuhnPokerGame",
    "NLHEState",
    "PolicyNet",
    "Reservoir",
    "SimpleNLHEGame",
    "TerminalReward",
    "Trainer",
    "TraversalStats",
    "decode_nlhe_infoset",
    "exploitability",
    "export_strategy_from_reservoir",
    "external_sampling_traversal",
    "local_best_response",
    "make_advantage_strategy_fn",
    "make_test_config",
    "make_uniform_strategy_fn",
    "regret_match",
]
