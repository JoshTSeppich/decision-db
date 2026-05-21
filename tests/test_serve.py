"""Tests for scripts/serve.py — WebSocket server wrapping RuntimeAdapter.

Tests use plain pytest with asyncio.run() (no pytest-asyncio dep). Each test
boots an in-process server on a random port, drives it with a websocket
client, and shuts down.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as ws_serve

from pokerbot.abstraction import AbstractionTables, ActionType
from pokerbot.opponent.archetype import Archetype
from pokerbot.runtime import (
    ActionHistoryEntry,
    BlindsSchema,
    GameStateRequest,
    RuntimeAdapter,
)
from pokerbot.strategy_db import SQLiteStrategyDB
from pokerbot.strategy_db.dual import NINE_MAX_ROUTE, SIX_MAX_ROUTE, DualStrategyDB

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import serve

# ─────────── helpers ───────────


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = int(s.getsockname()[1])
    s.close()
    return port


def _mk_request(
    *,
    table_size: int = 6,
    board: list[str] | None = None,
    hero_seat: int = 2,
    button_seat: int = 0,
    action_history: list[ActionHistoryEntry] | None = None,
    to_call: int = 0,
    pot_committed: int = 30,
    stacks: list[int] | None = None,
    min_raise: int = 10,
    max_raise: int | None = None,
    opponent_archetypes: tuple[Archetype | None, ...] | None = None,
) -> GameStateRequest:
    if stacks is None:
        stacks = [1000] * table_size
    if max_raise is None:
        max_raise = stacks[hero_seat]
    return GameStateRequest(
        schema_version=1,
        game_type="cash",
        table_size=table_size,  # type: ignore[arg-type]
        blinds=BlindsSchema(sb=5, bb=10),
        ante=0,
        hero_seat=hero_seat,
        button_seat=button_seat,
        hero_hole=["As", "Kh"],
        board=board if board is not None else ["7c", "2d", "Jh"],
        stacks=stacks,
        current_bets=[0] * table_size,
        pot_committed=pot_committed,
        to_call=to_call,
        min_raise=min_raise,
        max_raise=max_raise,
        action_history=action_history if action_history is not None else [],
        opponent_archetypes=opponent_archetypes,
    )


def _build_test_adapter(tmp_path: Path, *, rng_seed: int = 0) -> RuntimeAdapter:
    db = SQLiteStrategyDB(str(tmp_path / "rt.db"))
    db.set_current_version(1)
    return RuntimeAdapter(
        db=db, abstraction=AbstractionTables(), opponent_model=None, rng_seed=rng_seed
    )


class _ServerHandle:
    def __init__(self, state: serve.ServeState, port: int, task: asyncio.Task[None]) -> None:
        self.state = state
        self.port = port
        self.task = task
        self.url = f"ws://127.0.0.1:{port}"


async def _start_server(adapter: RuntimeAdapter) -> _ServerHandle:
    """Boot a serve.ServeState bound to the given adapter on a free port. Returns
    a handle whose `task` must be cancelled to shut the server down.
    """
    state = serve.ServeState(adapter)

    async def handler(conn: ServerConnection) -> None:
        await serve._connection_handler(state, conn)

    port = _free_port()
    started = asyncio.Event()

    async def _runner() -> None:
        async with ws_serve(handler, "127.0.0.1", port):
            started.set()
            # Run until cancelled.
            await asyncio.Future()

    task = asyncio.create_task(_runner())
    await asyncio.wait_for(started.wait(), timeout=2.0)
    return _ServerHandle(state, port, task)


async def _stop_server(handle: _ServerHandle) -> None:
    handle.task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await handle.task


async def _send_request(url: str, seq: int, request_payload: dict[str, Any]) -> dict[str, Any]:
    async with connect(url) as ws:
        await ws.send(json.dumps({"seq": seq, "request": request_payload}))
        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
    assert isinstance(raw, str)
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    return parsed


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ─────────── 1. boots + accepts connection ───────────


def test_serve_boots_and_accepts_connection(tmp_path: Path) -> None:
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            async with connect(handle.url) as ws:
                # Connection established; close cleanly.
                await ws.close()
        finally:
            await _stop_server(handle)

    _run(body())


# ─────────── 2. valid request returns response ───────────


def test_valid_request_returns_response(tmp_path: Path) -> None:
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            request = _mk_request(board=[])  # preflop → default_policy path
            payload = request.model_dump(mode="json")
            envelope = await _send_request(handle.url, seq=42, request_payload=payload)
            assert envelope["seq"] == 42
            assert "response" in envelope, f"got error envelope: {envelope}"
            resp = envelope["response"]
            assert resp["action"] in {"fold", "check", "call", "bet", "raise"}
            assert isinstance(resp["amount"], int)
            assert isinstance(resp["abstract_action"], str)
            assert 0.0 <= resp["probability_sampled"] <= 1.0
            assert resp["fallback_used"] in {
                "exact",
                "nearest_neighbor",
                "default_policy",
                "pushfold",
            }
        finally:
            await _stop_server(handle)

    _run(body())


# ─────────── 3. invalid request returns error envelope ───────────


def test_malformed_json_returns_parse_error(tmp_path: Path) -> None:
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            async with connect(handle.url) as ws:
                await ws.send("not-json-at-all{")
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            envelope = json.loads(raw)
            assert envelope["error"] == "parse_error"
            assert envelope["seq"] is None
        finally:
            await _stop_server(handle)

    _run(body())


def test_extra_field_returns_validation_error(tmp_path: Path) -> None:
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            request_payload = _mk_request().model_dump(mode="json")
            request_payload["surprise_field"] = "boom"
            envelope = await _send_request(handle.url, seq=1, request_payload=request_payload)
            assert envelope["seq"] == 1
            assert envelope["error"] == "validation_error"
            assert "details" in envelope
        finally:
            await _stop_server(handle)

    _run(body())


def test_missing_required_field_returns_validation_error(tmp_path: Path) -> None:
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            request_payload = _mk_request().model_dump(mode="json")
            del request_payload["hero_hole"]
            envelope = await _send_request(handle.url, seq=2, request_payload=request_payload)
            assert envelope["seq"] == 2
            assert envelope["error"] == "validation_error"
        finally:
            await _stop_server(handle)

    _run(body())


def test_server_survives_after_error(tmp_path: Path) -> None:
    """An error envelope must not crash the connection — a follow-up valid
    request should succeed."""
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            async with connect(handle.url) as ws:
                # Bad
                await ws.send("not-json{")
                err = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                assert err["error"] == "parse_error"

                # Good
                payload = _mk_request().model_dump(mode="json")
                await ws.send(json.dumps({"seq": 99, "request": payload}))
                ok = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
                assert ok["seq"] == 99 and "response" in ok
        finally:
            await _stop_server(handle)

    _run(body())


# ─────────── 4. seq round trips ───────────


def test_seq_round_trips(tmp_path: Path) -> None:
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            payload = _mk_request().model_dump(mode="json")
            async with connect(handle.url) as ws:
                for seq in (1, 2, 3):
                    await ws.send(json.dumps({"seq": seq, "request": payload}))
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    envelope = json.loads(raw)
                    assert envelope["seq"] == seq
                    assert "response" in envelope
        finally:
            await _stop_server(handle)

    _run(body())


# ─────────── 5. adapter singleton ───────────


_ABS_CONSTRUCTOR_CALLS: list[int] = []


class _CountingAbstraction(AbstractionTables):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ABS_CONSTRUCTOR_CALLS.append(1)
        super().__init__(*args, **kwargs)


def test_adapter_singleton_load_once(tmp_path: Path) -> None:
    """build_adapter loads the AbstractionTables + DB once. Subsequent
    decisions against the same adapter must not reinstantiate either."""
    db_calls: list[str] = []
    original_open_db = serve.open_db  # type: ignore[attr-defined]

    def fake_open_db(url: str) -> Any:
        db_calls.append(url)
        return original_open_db(url)

    _ABS_CONSTRUCTOR_CALLS.clear()

    with patch.object(serve, "open_db", side_effect=fake_open_db), \
         patch.object(serve, "AbstractionTables", _CountingAbstraction):
        sqlite_path = tmp_path / "single.db"
        sqlite_url = f"sqlite:///{sqlite_path}"
        # Seed an empty DB so open_db succeeds.
        original_open_db(sqlite_url).set_current_version(1)
        adapter, db = serve.build_adapter(
            db_path=sqlite_url,
            db_path_9max=None,
            abstraction_path=None,
            rng_seed=0,
        )

    # build_adapter should have made exactly one DB open call and one abstraction load.
    assert len(db_calls) == 1, f"expected 1 open_db call, got {len(db_calls)}"
    assert len(_ABS_CONSTRUCTOR_CALLS) == 1, (
        f"expected 1 AbstractionTables init, got {len(_ABS_CONSTRUCTOR_CALLS)}"
    )

    # Multiple decisions through the same adapter must not retrigger either.
    for _ in range(3):
        adapter.decide(_mk_request(board=[]))
    assert len(db_calls) == 1, "open_db must not be called per-decision"
    assert len(_ABS_CONSTRUCTOR_CALLS) == 1, "AbstractionTables must not re-instantiate"
    db.close()


# ─────────── 6. multiple concurrent connections ───────────


def test_multiple_concurrent_connections(tmp_path: Path) -> None:
    async def body() -> None:
        adapter = _build_test_adapter(tmp_path)
        handle = await _start_server(adapter)
        try:
            payload = _mk_request().model_dump(mode="json")

            async def one_client(seq: int) -> dict[str, Any]:
                async with connect(handle.url) as ws:
                    await ws.send(json.dumps({"seq": seq, "request": payload}))
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                parsed = json.loads(raw)
                assert isinstance(parsed, dict)
                return parsed

            envelopes = await asyncio.gather(one_client(10), one_client(20))
            assert {e["seq"] for e in envelopes} == {10, 20}
            for e in envelopes:
                assert "response" in e
        finally:
            await _stop_server(handle)

    _run(body())


# ─────────── 7. opponent intel changes decision ───────────


def _seed_mixed_distribution(db: SQLiteStrategyDB, adapter: RuntimeAdapter,
                              request: GameStateRequest) -> None:
    """Seed an exact-match row for `request`'s infoset with mass on actions
    Station suppresses (BET_150, ALL_IN) plus CHECK_CALL fallback."""
    infoset = adapter.build_infoset(request)
    actions = [ActionType.CHECK_CALL, ActionType.BET_66, ActionType.BET_150, ActionType.ALL_IN]
    mask = 0
    for a in actions:
        mask |= 1 << int(a)
    # Heavy mass on BET_150 + ALL_IN (Station kills these).
    probs = np.array([0.15, 0.15, 0.50, 0.20], dtype=np.float32)
    db.put(infoset, mask, probs, version=1)


def test_opponent_intel_shifts_distribution(tmp_path: Path) -> None:
    """Send the same flop request with and without Station archetypes pinned.
    Empirical action frequencies must shift away from BET_150 / ALL_IN."""
    async def body() -> None:
        db = SQLiteStrategyDB(str(tmp_path / "intel.db"))
        db.set_current_version(1)
        adapter = RuntimeAdapter(
            db=db, abstraction=AbstractionTables(), opponent_model=None, rng_seed=0
        )
        # Flop request with raise gate open so all 4 seeded actions are legal.
        base_request = _mk_request(
            board=["7c", "2d", "Jh"], to_call=0, pot_committed=30,
            min_raise=10, max_raise=1000,
        )
        _seed_mixed_distribution(db, adapter, base_request)

        handle = await _start_server(adapter)
        try:
            request_no_intel = base_request.model_dump(mode="json")
            stations = tuple(Archetype.STATION for _ in range(6))
            # Set hero's own seat to None.
            stations_list: list[Archetype | None] = list(stations)
            stations_list[base_request.hero_seat] = None
            request_with_intel = _mk_request(
                board=["7c", "2d", "Jh"], to_call=0, pot_committed=30,
                min_raise=10, max_raise=1000,
                opponent_archetypes=tuple(stations_list),
            ).model_dump(mode="json")

            n_samples = 60

            async def sample_n(payload: dict[str, Any], seq_base: int) -> list[str]:
                actions: list[str] = []
                async with connect(handle.url) as ws:
                    for i in range(n_samples):
                        await ws.send(json.dumps({"seq": seq_base + i, "request": payload}))
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        env = json.loads(raw)
                        assert "response" in env, f"unexpected error: {env}"
                        actions.append(env["response"]["abstract_action"])
                return actions

            # Reset adapter rng for each batch so the comparison is fair.
            adapter.rng.seed(0)
            actions_no_intel = await sample_n(request_no_intel, seq_base=1000)
            adapter.rng.seed(0)
            actions_station = await sample_n(request_with_intel, seq_base=2000)
        finally:
            await _stop_server(handle)

        suppressed = {"BET_150", "ALL_IN"}
        share_no_intel = sum(1 for a in actions_no_intel if a in suppressed) / n_samples
        share_station = sum(1 for a in actions_station if a in suppressed) / n_samples
        # Station kills 30-40% of BET_150 / ALL_IN. Expect a meaningful drop
        # (slack to absorb sampling noise; underlying probs differ by ~25 pp).
        assert share_station < share_no_intel - 0.10, (
            f"expected Station to suppress BET_150/ALL_IN; "
            f"no_intel share={share_no_intel:.2f}, station share={share_station:.2f}"
        )

    _run(body())


# ─────────── 8. table_size range accepted ───────────


def test_all_table_sizes_accepted(tmp_path: Path) -> None:
    async def body() -> None:
        # Build a dual DB so all sizes route somewhere.
        primary = SQLiteStrategyDB(str(tmp_path / "primary.db"))
        primary.set_current_version(1)
        secondary = SQLiteStrategyDB(str(tmp_path / "secondary.db"))
        secondary.set_current_version(1)
        dual = DualStrategyDB(primary, secondary)
        adapter = RuntimeAdapter(
            db=dual, abstraction=AbstractionTables(), opponent_model=None, rng_seed=0
        )
        handle = await _start_server(adapter)
        try:
            for table_size in (2, 3, 4, 5, 6, 7, 8, 9):
                # Hero/button must fit the table_size; SB seat exists.
                request = _mk_request(
                    table_size=table_size,
                    hero_seat=min(2, table_size - 1),
                    button_seat=0,
                    stacks=[1000] * table_size,
                    board=[],
                )
                envelope = await _send_request(handle.url, seq=table_size,
                                               request_payload=request.model_dump(mode="json"))
                assert envelope["seq"] == table_size
                assert "response" in envelope, f"size={table_size} got error: {envelope}"
        finally:
            await _stop_server(handle)
            dual.close()

    _run(body())


# ─────────── 9. table_size routes correctly ───────────


def test_dual_routing_selects_correct_db() -> None:
    """Confirm DualStrategyDB routing exactly matches Section H of the contract."""
    assert frozenset({6}) == SIX_MAX_ROUTE
    assert frozenset({2, 3, 4, 5, 7, 8, 9}) == NINE_MAX_ROUTE


def test_dual_db_routes_table_size_via_lookups(tmp_path: Path) -> None:
    """Black-box verify routing by seeding distinct rows in each DB and
    asserting which DB serves which table_size."""
    primary = SQLiteStrategyDB(str(tmp_path / "p.db"))
    primary.set_current_version(1)
    secondary = SQLiteStrategyDB(str(tmp_path / "s.db"))
    secondary.set_current_version(1)
    dual = DualStrategyDB(primary, secondary)
    abstraction = AbstractionTables()
    adapter = RuntimeAdapter(db=dual, abstraction=abstraction, rng_seed=0)

    # Seed primary with a row that responds CHECK_CALL; secondary with BET_66.
    for table_size, db, expected_action in (
        (6, primary, ActionType.CHECK_CALL),  # primary route
        (8, secondary, ActionType.BET_66),     # secondary route
        (2, secondary, ActionType.BET_66),     # heads-up → secondary
        (9, secondary, ActionType.BET_66),     # 9-max → secondary
    ):
        request = _mk_request(
            table_size=table_size,
            hero_seat=min(2, table_size - 1),
            button_seat=0,
            stacks=[1000] * table_size,
            board=["7c", "2d", "Jh"],
            to_call=0,
            pot_committed=30,
            min_raise=10,
        )
        infoset = adapter.build_infoset(request)
        mask = 1 << int(expected_action)
        # Only put the row in the EXPECTED-route DB. If routing is wrong, it
        # would fall through to default_policy.
        db.put(infoset, mask, np.array([1.0], dtype=np.float32), version=1)

        response = adapter.decide(request)
        assert response.fallback_used == "exact", (
            f"size={table_size} routed to wrong DB: fallback={response.fallback_used}"
        )
        assert response.abstract_action == expected_action.name

    dual.close()


# ─────────── schema extension tests ───────────


def test_opponent_archetypes_default_none_round_trips() -> None:
    request = _mk_request()
    assert request.opponent_archetypes is None
    # JSON round-trip preserves None.
    raw = request.model_dump(mode="json")
    assert raw["opponent_archetypes"] is None
    rebuilt = GameStateRequest.model_validate(raw)
    assert rebuilt.opponent_archetypes is None


def test_opponent_archetypes_length_mismatch_raises() -> None:
    with pytest.raises(Exception, match="length"):
        _mk_request(
            table_size=6,
            opponent_archetypes=(Archetype.NIT, Archetype.TAG),  # length 2 != 6
        )


def test_opponent_archetypes_valid_tuple_round_trips() -> None:
    pinned: tuple[Archetype | None, ...] = (
        Archetype.NIT,
        Archetype.TAG,
        None,
        Archetype.LAG,
        Archetype.MANIAC,
        Archetype.STATION,
    )
    request = _mk_request(table_size=6, opponent_archetypes=pinned)
    raw = request.model_dump(mode="json")
    rebuilt = GameStateRequest.model_validate(raw)
    assert rebuilt.opponent_archetypes is not None
    assert tuple(rebuilt.opponent_archetypes) == pinned


# ─────────── PinnedArchetypeOpponentModel direct tests ───────────


def test_pinned_archetype_picks_highest_priority() -> None:
    # Priority order is STATION > MANIAC > NIT > LAG > TAG > UNKNOWN.
    model = serve.PinnedArchetypeOpponentModel((Archetype.TAG, Archetype.NIT, Archetype.LAG))
    assert model.picked_archetype() == Archetype.NIT  # NIT beats LAG and TAG.

    model_station = serve.PinnedArchetypeOpponentModel(
        (Archetype.TAG, Archetype.STATION, Archetype.LAG)
    )
    assert model_station.picked_archetype() == Archetype.STATION


def test_build_opponent_model_strips_hero_and_none() -> None:
    request = _mk_request(
        table_size=6,
        hero_seat=2,
        opponent_archetypes=(
            Archetype.NIT,
            None,
            Archetype.STATION,  # hero's seat — should be stripped
            None,
            Archetype.TAG,
            Archetype.LAG,
        ),
    )
    model = serve._build_opponent_model(request)
    assert isinstance(model, serve.PinnedArchetypeOpponentModel)
    # Hero's STATION is stripped; remaining {NIT, TAG, LAG} → NIT wins by priority.
    assert model.picked_archetype() == Archetype.NIT


def test_build_opponent_model_all_none_returns_no_model() -> None:
    request = _mk_request(
        table_size=6,
        opponent_archetypes=(None, None, None, None, None, None),
    )
    assert serve._build_opponent_model(request) is None
