"""DB-backed `SpotPolicy` — scores an exported StrategyDB through the Component 4 gate.

Wraps a `StrategyDB` as the harness's `SpotPolicy` (AgentSpot → action distribution)
so a fine-tuned policy exported by `export_best_response` is scored through the ONE
validated profiler (`profile_spot_policy`), not a second one. An `AgentSpot` carries
no betting history, so lookup uses `nearest_neighbor` (history-agnostic: keyed by
table_size/street/card_bucket/position/stack_bucket) and `action_distribution` to
expand the row. The stack key is the EFFECTIVE stack (min hero/opp) the export wrote,
carried on `AgentSpot.effective_stack`; bucketing by the hero's raw stack would make the
chip leader miss. A miss returns an EMPTY mapping ("no data here"): the band profiler's
adapter maps that to CHECK_CALL so it never stalls, while a coverage-sensitive caller
(the catastrophic screen) can detect the miss and refuse to certify.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pokerbot.abstraction import ActionType, InfoSet
from pokerbot.abstraction.encoding import stack_bucket_from_eff_bb

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pokerbot.abstraction import AbstractionTables
    from pokerbot.strategy_db import StrategyDB
    from zoom.agents import AgentSpot
    from zoom.eval.profile import SpotPolicy

_STREET_IDX: Final[dict[str, int]] = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def make_db_spot_policy(
    db: StrategyDB,
    abstraction: AbstractionTables,
    *,
    bb: int,
    table_size: int,
) -> SpotPolicy:
    """Build a `SpotPolicy` that reads `db` (nearest-neighbor) for each spot."""

    version = db.current_version()

    def policy(spot: AgentSpot) -> Mapping[ActionType, float]:
        card_bucket = int(abstraction.lookup(spot.hole, spot.board, spot.street))
        # Bucket by EFFECTIVE stack (min(hero, max opp)) — the quantity the export keyed
        # rows by (nlhe_game.py:388). `nearest_neighbor` matches stack_bucket exactly, so
        # bucketing by the hero's raw stack makes the chip leader (button at hand start,
        # 100bb vs blinds-posted opponents) query an over-deep, empty bucket and miss.
        # effective_stack is None for callers that don't know opponent stacks → use stack.
        eff = spot.effective_stack if spot.effective_stack is not None else spot.stack
        info = InfoSet(
            table_size=table_size,
            street=_STREET_IDX[spot.street],
            position=spot.position,
            stack_bucket=stack_bucket_from_eff_bb(eff // max(bb, 1)),
            card_bucket=card_bucket,
            history=spot.history,
        )
        # Mirror the production RuntimeAdapter lookup chain (adapter.py): EXACT
        # history-aware match first, then the history-agnostic nearest-neighbor. A
        # nearest-only lookup returns the highest-visit row in the cell regardless of
        # the real betting context — wrong for SB/BB, which have no empty-history rows.
        row = db.get(info, version)
        if row is None:
            row = db.nearest_neighbor(info, version)
        if row is None:
            # MISS = no data at this spot → empty mapping (NOT a silent CHECK_CALL). The
            # band profiler's adapter maps an empty dist to CHECK_CALL so it never stalls
            # (profile.py:_SpotPolicyAdapter._sample), while a coverage-sensitive caller
            # (the catastrophic screen) can detect the miss and refuse to certify.
            return {}
        # Expand (action_mask bitmask + packed legal probs) → {ActionType: prob}.
        # `action_probs` is in ascending-action-index order, matching the export.
        dist: dict[ActionType, float] = {}
        j = 0
        for a in range(len(ActionType)):
            if row.action_mask & (1 << a):
                dist[ActionType(a)] = float(row.action_probs[j])
                j += 1
        return dist

    return policy


__all__ = ["make_db_spot_policy"]
