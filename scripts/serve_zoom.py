"""Stateful WebSocket advisory server for 6-max zoom (layer L5 skeleton).

Runs ALONGSIDE the frozen brain (`scripts/serve.py`) — it does not import or
modify it. It reuses the frozen brain's `GameStateRequest` envelope and wraps it
in a `ZoomObservation` (adds `opponent_id`; never modifies `GameStateRequest`).
For each decision it maintains per-opponent Dirichlet state + a per-hand range
tracker, then renders an advisory action. It is ADVISORY ONLY — it prints/returns
"BOT SAYS: <action> <amount>" and never fires keystrokes.

Wire format (one JSON message in, one out):
    in :  {"seq": <int>, "request": <GameStateRequest>, "opponent_id"?: <str>,
           "revealed_holes"?: {<seat>: ["As","Kd"]}}
    out:  {"seq": <int>, "response": <ZoomAdvice>}   |   {"seq", "error", "details"}

The blueprint action is the frozen DB policy (read-only); the L4 subgame solver
will replace that call inside `ZoomExploiterService.advise`.

Run (6-max competition target — single 6-max blueprint, no 9-max secondary):
    python scripts/serve_zoom.py --port 8766 \\
        --db training/v5-6max-fix.db \\
        --abstraction-path abstraction
    # or use the committed wrapper: bash scripts/serve_6max_blueprint.sh
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

# `zoom` is not pip-installed (unlike `pokerbot`), so make the repo root
# importable when this file is run directly as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import websockets
from pydantic import ValidationError
from websockets.asyncio.server import ServerConnection, serve

from pokerbot.abstraction import AbstractionTables
from pokerbot.runtime.adapter import RuntimeAdapter
from pokerbot.strategy_db import open_db
from pokerbot.strategy_db.dual import DualStrategyDB
from zoom.service import ZoomExploiterService
from zoom.zoom_schema import ZoomObservation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pokerbot.strategy_db.base import StrategyDB

log = logging.getLogger("zoom.serve")


def _to_db_url(arg: str) -> str:
    if arg.startswith("sqlite:"):
        return arg
    p = Path(arg).resolve()
    if not p.exists():
        raise SystemExit(f"ERROR: DB not found at {p}")
    return f"sqlite:///{p}"


def build_zoom_adapter(
    *,
    db_path: str,
    db_path_9max: str | None,
    abstraction_path: str | None,
    rng_seed: int | None,
) -> tuple[RuntimeAdapter, StrategyDB]:
    """Boot the read-only blueprint adapter. Mirrors `serve.build_adapter`'s
    single/dual routing without importing the frozen serve.py. For the 6-max
    competition target, pass only `--db` (the v5-6max blueprint): with
    `db_path_9max=None` this uses the single-DB path (`db = primary`), so
    table_size=6 keys directly into it. A `--db-9max` is only needed if non-6
    table sizes must be served (they route to the 9-max DB under DualStrategyDB)."""
    abstraction = AbstractionTables(path=abstraction_path)
    primary = open_db(_to_db_url(db_path))
    db: StrategyDB
    if db_path_9max is not None:
        db = DualStrategyDB(primary, open_db(_to_db_url(db_path_9max)))
    else:
        db = primary
    adapter = RuntimeAdapter(db=db, abstraction=abstraction, opponent_model=None, rng_seed=rng_seed)
    return adapter, db


class ServeState:
    """Process-wide shared state. Decisions are serialized: the adapter's SQLite
    connection is single-thread and the service mutates per-opponent state."""

    def __init__(self, service: ZoomExploiterService) -> None:
        self.service = service
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def locked(self) -> AsyncIterator[None]:
        async with self.lock:
            yield


async def _handle_one_message(state: ServeState, raw: str | bytes) -> dict[str, Any]:
    """Parse one inbound message into a response envelope. Never raises."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"seq": None, "error": "parse_error", "details": {"message": str(e)}}
    if not isinstance(payload, dict):
        return {"seq": None, "error": "envelope_error", "details": {"message": "payload must be an object"}}

    seq = payload.get("seq")
    try:
        obs = ZoomObservation.model_validate(payload)
    except ValidationError as e:
        return {"seq": seq, "error": "validation_error", "details": {"errors": e.errors(include_url=False)}}

    t0 = time.perf_counter()
    try:
        async with state.locked():
            advice = state.service.advise(obs)
    except Exception as e:  # noqa: BLE001 — never crash the connection on one bad message
        log.error("[seq=%s] advise exception: %s\n%s", seq, e, traceback.format_exc())
        return {"seq": seq, "error": "service_exception", "details": {"type": type(e).__name__, "message": str(e)}}

    log.info(
        "[seq=%s] %s  opp=%s eff_combos=%.1f obs=%.0f latency=%dms",
        seq, advice.advice, advice.opponent_id, advice.range_effective_combos,
        advice.opponent_observations, int((time.perf_counter() - t0) * 1000),
    )
    return {"seq": obs.seq, "response": advice.model_dump()}


async def _connection_handler(state: ServeState, conn: ServerConnection) -> None:
    try:
        async for raw in conn:
            await conn.send(json.dumps(await _handle_one_message(state, raw)))
    except websockets.ConnectionClosed:
        pass
    except Exception as e:  # noqa: BLE001
        log.error("connection error: %s\n%s", e, traceback.format_exc())


async def run_server(
    *,
    host: str,
    port: int,
    db_path: str,
    db_path_9max: str | None,
    abstraction_path: str,
    rng_seed: int | None,
    ready_event: asyncio.Event | None = None,
) -> None:
    t0 = time.perf_counter()
    adapter, db = build_zoom_adapter(
        db_path=db_path, db_path_9max=db_path_9max,
        abstraction_path=abstraction_path, rng_seed=rng_seed,
    )
    log.info("cold start complete in %.2fs", time.perf_counter() - t0)
    state = ServeState(ZoomExploiterService(adapter))

    async def handler(conn: ServerConnection) -> None:
        await _connection_handler(state, conn)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(getattr(signal, sig_name), stop_event.set)

    try:
        async with serve(handler, host, port):
            log.info("zoom advisory server listening on ws://%s:%d", host, port)
            if ready_event is not None:
                ready_event.set()
            await stop_event.wait()
    finally:
        with contextlib.suppress(Exception):
            db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--db", required=True, help="blueprint StrategyDB (read-only)")
    parser.add_argument("--db-9max", default=None, help="optional 9-max DB for dual routing")
    parser.add_argument("--abstraction-path", default="abstraction")
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(
        run_server(
            host=args.host, port=args.port, db_path=args.db, db_path_9max=args.db_9max,
            abstraction_path=args.abstraction_path, rng_seed=args.rng_seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
