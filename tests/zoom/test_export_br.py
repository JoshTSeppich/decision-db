"""Component 5 prerequisite — net-based best-response exporter (Approach-2).

The injected fine-tune leaves the deployable policy in the advantage nets
(regret_match(advantage_nets) = current best response); the policy_reservoir is
empty by construction (confirmed (a)-benign). So `export_strategy_from_reservoir`
(which reads the policy net) can't export it. `export_best_response` walks the
ADVANTAGE reservoirs and writes regret_match(advantage_nets) — the current best
response, NOT a time-average — to a StrategyDB in the SAME format the Component 4
harness / RuntimeAdapter reads, so the fine-tuned policy is scored through the one
validated gate.

Red-first: the exported DB must round-trip AND be scorable by the Component 4
harness (via a DB-backed SpotPolicy). These imports fail until the code exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zoom.agents import build_archetype_pool
from zoom.eval import evaluate_bands, profile_spot_policy
from zoom.eval.db_policy import make_db_spot_policy
from zoom.train import GatedNLHEGame
from zoom.train.export_br import export_best_response
from zoom.train.finetune import FineTuneTrainer

from pokerbot.abstraction import AbstractionTables
from pokerbot.strategy_db import open_db
from pokerbot.training import make_test_config
from pokerbot.training.export import decode_nlhe_infoset

if TYPE_CHECKING:
    from pathlib import Path


def _trained_trainer() -> tuple[FineTuneTrainer, GatedNLHEGame, AbstractionTables]:
    abstraction = AbstractionTables()
    game = GatedNLHEGame(abstraction, blinds=(5, 10), starting_stack=1000, table_size=3)
    cfg = make_test_config(outer_iters=1, traversals_per_iter=40, advantage_hidden=(32, 32))
    tr = FineTuneTrainer(cfg, game, pool=build_archetype_pool())
    tr._cfr_iteration(1)  # populate advantage reservoirs (the BR signal source)
    return tr, game, abstraction


def test_export_best_response_writes_rows(tmp_path: Path) -> None:
    """The exporter writes >0 rows from the populated advantage reservoirs."""
    tr, _game, _abs = _trained_trainer()
    db = open_db(f"sqlite:///{tmp_path / 'br.db'}")
    n = export_best_response(tr.advantage_nets, tr.advantage_reservoirs, db, version=1)
    assert n > 0, "exporter wrote no rows despite populated advantage reservoirs"
    db.close()


def test_exported_db_round_trips(tmp_path: Path) -> None:
    """A row written for an advantage-reservoir infoset is readable back, and its
    probabilities form a valid distribution over the legal actions."""
    tr, _game, _abs = _trained_trainer()
    db = open_db(f"sqlite:///{tmp_path / 'br.db'}")
    export_best_response(tr.advantage_nets, tr.advantage_reservoirs, db, version=1)

    # Pick a real exported key from a populated reservoir and read it back.
    checked = 0
    for res in tr.advantage_reservoirs:
        for i in range(res.size):
            key = res.infoset_keys[i]
            if not key:
                continue
            info = decode_nlhe_infoset(key)
            row = db.get(info, version=1)
            if row is None:
                continue
            probs = row.action_probs
            assert probs.sum() == __import__("pytest").approx(1.0, abs=1e-5)
            assert (probs >= 0).all()
            checked += 1
            if checked >= 5:
                break
        if checked >= 5:
            break
    assert checked > 0, "no exported row was readable back"
    db.close()


def test_exported_db_is_scorable_by_harness(tmp_path: Path) -> None:
    """The exported DB is scorable through the validated Component 4 gate via a
    DB-backed SpotPolicy — no second profiler. evaluate_bands runs on the result."""
    tr, game, abstraction = _trained_trainer()
    db = open_db(f"sqlite:///{tmp_path / 'br.db'}")
    export_best_response(tr.advantage_nets, tr.advantage_reservoirs, db, version=1)

    db_policy = make_db_spot_policy(db, abstraction, bb=10, table_size=3)
    profile = profile_spot_policy(db_policy, game, n_hands=30, seed=7)
    result = evaluate_bands(profile)

    # Scorability, not pass: every gated metric must be a finite number the gate
    # can evaluate.
    import math

    for m in result.metrics:
        assert math.isfinite(m.value), (m.name, m.value)
    assert isinstance(result.passed, bool)
    db.close()


def test_export_carries_visit_counts(tmp_path: Path) -> None:
    """The BR export must carry per-infoset visit counts (reservoir frequency).

    `nearest_neighbor` tiebreaks on `visit_count DESC, infoset_hash ASC`. An all-zero
    export makes it return a hash-arbitrary row for a (table/street/pos/stack/card) cell
    instead of the representative (most-visited) one — the root of the spurious
    'folds AA' reading. Red-first: today's export writes visit_count=0 everywhere.
    """
    from collections import Counter

    from pokerbot.training.export import decode_nlhe_infoset

    tr, _game, _abs = _trained_trainer()
    db = open_db(f"sqlite:///{tmp_path / 'br.db'}")
    export_best_response(tr.advantage_nets, tr.advantage_reservoirs, db, version=1)

    # Reservoir frequency per infoset key (keys are position-scoped → unique per cell).
    freq: Counter[bytes] = Counter()
    for res in tr.advantage_reservoirs:
        for i in range(res.size):
            k = res.infoset_keys[i]
            if k:
                freq[k] += 1
    key, cnt = freq.most_common(1)[0]

    row = db.get(decode_nlhe_infoset(key), version=1)
    assert row is not None, "most-frequent reservoir infoset was not exported"
    assert row.visit_count == cnt, (
        f"visit_count must equal reservoir frequency: got {row.visit_count}, expected {cnt}"
    )
    db.close()
