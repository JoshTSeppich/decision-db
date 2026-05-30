"""FineTuneTrainer — Deep CFR fine-tune against a fixed archetype pool.

Subclasses the frozen `pokerbot.training.Trainer` and overrides ONLY the inner CFR
loop to use the seam traversal. Everything else — per-player advantage nets and
reservoirs, weighted-MSE training, checkpoint/resume, policy export — is the proven
machinery, reused unchanged.

Two modes:
  * ``pool=None`` → self-play (the reference Deep CFR objective). Used to validate
    the traversal on Kuhn (Gate 1).
  * ``pool=[agents]`` → each non-traverser seat is occupied by a ScriptedAgent drawn
    from the pool for that hand, so the learner trains a best response to the
    tight-passive distribution. This is the Stage-1 fine-tune objective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pokerbot.training.deepcfr import Trainer
from zoom.train.opponents import make_archetype_opponent_policy
from zoom.train.traversal import external_sampling_traversal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pokerbot.training.config import DeepCFRConfig
    from pokerbot.training.game import Game
    from zoom.agents import ScriptedAgent
    from zoom.train.traversal import OpponentPolicy


class FineTuneTrainer(Trainer):
    """Deep CFR over a `Game`, optionally with a fixed archetype opponent pool."""

    def __init__(
        self,
        config: DeepCFRConfig,
        game: Game[Any],
        *,
        pool: Sequence[ScriptedAgent] | None = None,
        device: str = "cpu",
        run_metadata: dict[str, Any] | None = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(config, game, device=device, run_metadata=run_metadata)
        self.pool: list[ScriptedAgent] | None = list(pool) if pool is not None else None
        # Traversal-phase parallelism (see zoom.train.parallel_traversal). 1 = the
        # serial reference path; >1 fans traversals across that many spawned worker
        # processes and merges single-threaded. The worker pool is created lazily on
        # first parallel iteration and torn down by close_pool().
        self.num_workers = num_workers
        self._worker_pool: Any | None = None

    def _cfr_iteration(self, t: int) -> None:
        for _ in range(self.config.traversals_per_iter):
            traverser = self.rng.randrange(self.game.num_players)
            state = self.game.new_initial_state(self.rng)
            opponent_policy = self._opponent_policy_for(traverser)
            external_sampling_traversal(
                game=self.game,
                state=state,
                traverser=traverser,
                advantage_nets=self.advantage_nets,
                advantage_reservoir=self.advantage_reservoirs[traverser],
                policy_reservoir=self.policy_reservoir,
                iter_t=t,
                rng=self.rng,
                opponent_policy=opponent_policy,
            )

    def _cfr_iteration_parallel(self, t: int) -> None:
        """Parallel traversal phase: fan out across worker processes (or run the
        serial reference when num_workers==1), then merge into the reservoirs.

        Provably equivalent to the serial path — see zoom.train.parallel_traversal.
        Uses the engine's per-traversal derived RNG, NOT self.rng, so the traversal
        phase consumes no self.rng draws (the train phase still does)."""
        from zoom.train.parallel_traversal import cfr_iteration, make_worker_pool

        if self.num_workers > 1 and self._worker_pool is None:
            self._worker_pool = make_worker_pool(
                self.num_workers, self.game, self.advantage_nets, self.pool
            )
        cfr_iteration(
            game=self.game,
            nets=self.advantage_nets,
            advantage_reservoirs=self.advantage_reservoirs,
            policy_reservoir=self.policy_reservoir,
            iter_t=t,
            traversals_per_iter=self.config.traversals_per_iter,
            master_seed=self.config.seed,
            pool=self.pool,
            num_workers=self.num_workers,
            worker_pool=self._worker_pool,
        )

    def close_pool(self) -> None:
        """Tear down the worker pool, if one was created. Safe to call repeatedly."""
        if self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool.join()
            self._worker_pool = None

    def _opponent_policy_for(self, traverser: int) -> OpponentPolicy | None:
        """Assign an archetype to each non-traverser seat for one hand, or None
        (self-play) when no pool is configured."""
        if self.pool is None:
            return None
        assignment = {
            seat: self.rng.choice(self.pool)
            for seat in range(self.game.num_players)
            if seat != traverser
        }
        # self.game is a GatedNLHEGame in the fine-tune; agent_spot lives there.
        return make_archetype_opponent_policy(self.game, assignment)  # type: ignore[arg-type]


__all__ = ["FineTuneTrainer"]
