"""Mixed-opponent best-response fine-tune driver (Approach-2, Stage 1, Component 3).

A NEW training path built alongside the frozen `pokerbot.training` package (which
it imports read-only). It reuses the proven `Trainer`, `Reservoir`, nets, and the
`Game` ABC, but swaps in a traversal with an opponent-injection seam so the
blueprint can be fine-tuned as a best response to the Component 2 archetype pool —
the tight-passive distribution pure self-play never contained.

This component builds and proves the driver on toy / fixed-seed checkpoints only.
The real Stage-1 fine-tune run is Component 5 (gated on Component 4's validated
eval harness).
"""

from zoom.train.eval_pool import evaluate_vs_pool
from zoom.train.export_br import export_best_response
from zoom.train.finetune import FineTuneTrainer
from zoom.train.game import GatedNLHEGame
from zoom.train.opponents import make_archetype_opponent_policy
from zoom.train.traversal import OpponentPolicy, external_sampling_traversal

__all__ = [
    "FineTuneTrainer",
    "GatedNLHEGame",
    "OpponentPolicy",
    "evaluate_vs_pool",
    "export_best_response",
    "external_sampling_traversal",
    "make_archetype_opponent_policy",
]
