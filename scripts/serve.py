"""WebSocket server wrapping the trained CFR policy.

One JSON message per request, one JSON message per response. Read-only and
stateless across requests; per-request opponent modeling is supported via
the optional `opponent_archetypes` field on GameStateRequest (sister project
classifies opponents across hands and ships the snapshot per decision).

Run:
    python scripts/serve.py \\
        --port 8765 \\
        --db strategy-pilot-v2.db \\
        --db-9max strategy-pilot-v3-9max.db \\
        --abstraction-path abstraction \\
        --rng-seed 0

See INTEGRATION_CONTRACT.md for the wire-level contract.
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

import websockets
from pydantic import ValidationError
from websockets.asyncio.server import ServerConnection, serve

from pokerbot.abstraction import AbstractionTables
from pokerbot.opponent.archetype import Archetype, ArchetypeClassifier
from pokerbot.opponent.model import _ARCHETYPE_PRIORITY, ArchetypeOpponentModel
from pokerbot.opponent.stats import OpponentStatsTracker
from pokerbot.runtime.adapter import RuntimeAdapter
from pokerbot.runtime.opponent import IdentityOpponentModel, OpponentModel
from pokerbot.runtime.schema import GameStateRequest
from pokerbot.strategy_db import open_db
from pokerbot.strategy_db.dual import DualStrategyDB
from pokerbot.strategy_db.multi_size import MultiSizeStrategyDB

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pokerbot.strategy_db.base import StrategyDB

log = logging.getLogger("pokerbot.serve")


# ─────────── per-request pinned opponent model ───────────


class PinnedArchetypeOpponentModel(ArchetypeOpponentModel):
    """Per-request opponent model whose archetype selection is fixed from the
    request payload (sister project pre-classifies; we apply adjustments).

    Reuses the priority order and per-archetype adjustment math from
    `ArchetypeOpponentModel`; only `_pick_archetype` is overridden so the
    tracker/classifier pair is unused (passed but never queried).
    """

    def __init__(self, pinned: tuple[Archetype, ...]) -> None:
        super().__init__(OpponentStatsTracker(), ArchetypeClassifier())
        self._pinned_archetypes = pinned
        # adjust() short-circuits when active_opponent_ids is empty; we set a
        # synthetic non-empty tuple so the body runs and our _pick_archetype
        # override is consulted.
        self._active_opponent_ids = tuple(f"_pin_{i}" for i in range(max(1, len(pinned))))

    def _pick_archetype(self, active_ids: tuple[str, ...]) -> Archetype:  # noqa: ARG002
        return self.picked_archetype()

    def picked_archetype(self) -> Archetype:
        """Resolved archetype that adjust() will apply (priority-based selection
        across pinned non-None entries). Exposed for logging/diagnostics.
        """
        observed = set(self._pinned_archetypes)
        for candidate in _ARCHETYPE_PRIORITY:
            if candidate in observed:
                return candidate
        return Archetype.UNKNOWN


def _build_opponent_model(request: GameStateRequest) -> OpponentModel | None:
    """Return a pinned model if the request carries usable archetype intel,
    else None (caller will fall back to IdentityOpponentModel).
    """
    if request.opponent_archetypes is None:
        return None
    # Exclude hero seat from pin set (hero's own classification is meaningless)
    # and strip None entries (insufficient observations).
    non_none: list[Archetype] = []
    for seat, archetype in enumerate(request.opponent_archetypes):
        if seat == request.hero_seat:
            continue
        if archetype is None:
            continue
        non_none.append(archetype)
    if not non_none:
        return None
    return PinnedArchetypeOpponentModel(tuple(non_none))


# ─────────── server ───────────


class ServeState:
    """Process-wide shared state: adapter, lock for serialized decide() calls."""

    def __init__(self, adapter: RuntimeAdapter) -> None:
        self.adapter = adapter
        # decide() mutates adapter.rng and we hot-swap opponent_model per
        # request. Serialize all decisions so concurrent connections don't
        # interleave opponent_model state.
        self.lock = asyncio.Lock()
        self._identity = IdentityOpponentModel()

    @asynccontextmanager
    async def use_opponent_model(self, model: OpponentModel | None) -> AsyncIterator[None]:
        """Temporarily swap adapter.opponent_model under the lock. Restores
        on exit even if decide() raises.
        """
        async with self.lock:
            previous = self.adapter.opponent_model
            self.adapter.opponent_model = model if model is not None else self._identity
            try:
                yield
            finally:
                self.adapter.opponent_model = previous


async def _handle_one_message(
    state: ServeState, raw: str | bytes
) -> dict[str, Any]:
    """Parse one inbound message and produce a response envelope.

    Returns a dict ready to be JSON-encoded. Never raises — every error
    path produces an error envelope.
    """
    seq: int | None = None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"seq": None, "error": "parse_error", "details": {"message": str(e)}}

    if not isinstance(payload, dict):
        return {"seq": None, "error": "envelope_error", "details": {"message": "payload must be a JSON object"}}

    seq = payload.get("seq")
    if not isinstance(seq, int):
        return {"seq": None, "error": "envelope_error", "details": {"message": "missing or non-int 'seq'"}}

    raw_request = payload.get("request")
    if not isinstance(raw_request, dict):
        return {"seq": seq, "error": "envelope_error", "details": {"message": "missing or non-object 'request'"}}

    try:
        request = GameStateRequest.model_validate(raw_request)
    except ValidationError as e:
        return {
            "seq": seq,
            "error": "validation_error",
            "details": {"errors": e.errors(include_url=False)},
        }

    opponent_model = _build_opponent_model(request)
    archetype_label = "none"
    if isinstance(opponent_model, PinnedArchetypeOpponentModel):
        archetype_label = opponent_model.picked_archetype().value

    t_start = time.perf_counter()
    try:
        async with state.use_opponent_model(opponent_model):
            # SQLite connections are bound to their creating thread; run
            # decide() synchronously on the event-loop thread. The lock above
            # serializes calls so two connections cannot interleave.
            response = state.adapter.decide(request)
    except Exception as e:
        log.error("[seq=%s] adapter exception: %s\n%s", seq, e, traceback.format_exc())
        return {
            "seq": seq,
            "error": "adapter_exception",
            "details": {"type": type(e).__name__, "message": str(e)},
        }

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    log.info(
        "[seq=%s] decided=%s fallback=%s archetype=%s table_size=%d latency=%dms",
        seq,
        response.abstract_action,
        response.fallback_used,
        archetype_label,
        request.table_size,
        elapsed_ms,
    )
    return {"seq": seq, "response": response.model_dump()}


async def _connection_handler(state: ServeState, conn: ServerConnection) -> None:
    """Read messages from one client until the connection closes."""
    peer = getattr(conn, "remote_address", "?")
    log.debug("connection opened from %s", peer)
    try:
        async for raw in conn:
            envelope = await _handle_one_message(state, raw)
            await conn.send(json.dumps(envelope))
    except websockets.ConnectionClosed:
        log.debug("connection closed from %s", peer)
    except Exception as e:
        # Don't crash the server on per-connection errors.
        log.error("connection error from %s: %s\n%s", peer, e, traceback.format_exc())


# ─────────── boot ───────────


def _to_db_url(arg: str) -> str:
    if arg.startswith("sqlite:"):
        return arg
    p = Path(arg).resolve()
    if not p.exists():
        raise SystemExit(f"ERROR: DB not found at {p}")
    return f"sqlite:///{p}"


def _load_multi_size_config(path: str) -> dict[int, str]:
    """Parse a multi-size YAML config -> {table_size: absolute_db_path}.

    Expected schema:
        sizes:
          2: /abs/path/strategy-pilot-v4-2.db
          3: /abs/path/strategy-pilot-v4-3.db
          ...
    """
    import yaml

    cfg_path = Path(path).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"ERROR: multi-size config not found at {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict) or "sizes" not in raw:
        raise SystemExit(
            f"ERROR: multi-size config {cfg_path} must contain a top-level 'sizes' mapping"
        )
    sizes_raw = raw["sizes"]
    if not isinstance(sizes_raw, dict) or not sizes_raw:
        raise SystemExit(f"ERROR: 'sizes' in {cfg_path} must be a non-empty mapping")
    out: dict[int, str] = {}
    for key, value in sizes_raw.items():
        try:
            size = int(key)
        except (TypeError, ValueError) as e:
            raise SystemExit(f"ERROR: size key {key!r} in {cfg_path} is not an int") from e
        out[size] = str(value)
    return out


def build_adapter(
    *,
    db_path: str | None,
    db_path_9max: str | None,
    abstraction_path: str | None,
    rng_seed: int | None,
    multi_size_config_path: str | None = None,
) -> tuple[RuntimeAdapter, StrategyDB]:
    """Boot the adapter. Returns the adapter and the (top-level) StrategyDB
    so the caller can close it on shutdown.

    Routing:
      - `multi_size_config_path` set        -> MultiSizeStrategyDB (per-size DBs).
      - `db_path_9max` set                  -> DualStrategyDB (6-max + 9-max).
      - `db_path` only                      -> single underlying DB.
    `multi_size_config_path` takes precedence over both other flags.
    """
    abstraction = AbstractionTables(path=abstraction_path)
    db: StrategyDB
    if multi_size_config_path is not None:
        size_to_path = _load_multi_size_config(multi_size_config_path)
        per_size_dbs = {size: open_db(_to_db_url(p)) for size, p in size_to_path.items()}
        db = MultiSizeStrategyDB(per_size_dbs)
    else:
        if db_path is None:
            raise SystemExit("ERROR: --db is required unless --multi-size-config is given")
        primary = open_db(_to_db_url(db_path))
        if db_path_9max is not None:
            secondary = open_db(_to_db_url(db_path_9max))
            db = DualStrategyDB(primary, secondary)
        else:
            db = primary
    adapter = RuntimeAdapter(
        db=db,
        abstraction=abstraction,
        opponent_model=None,  # IdentityOpponentModel default; swapped per-request
        rng_seed=rng_seed,
    )
    return adapter, db


async def run_server(
    *,
    host: str,
    port: int,
    db_path: str | None,
    db_path_9max: str | None,
    abstraction_path: str,
    rng_seed: int | None,
    multi_size_config_path: str | None = None,
    ready_event: asyncio.Event | None = None,
) -> None:
    log.info("loading abstraction from %s", abstraction_path)
    if multi_size_config_path is not None:
        log.info(
            "loading multi-size DB config %s  (per-table_size routing)",
            multi_size_config_path,
        )
    else:
        log.info("loading primary DB %s", db_path)
        if db_path_9max:
            log.info("loading 9-max DB  %s  (dual-routing by table_size)", db_path_9max)
    t0 = time.perf_counter()
    adapter, db = build_adapter(
        db_path=db_path,
        db_path_9max=db_path_9max,
        multi_size_config_path=multi_size_config_path,
        abstraction_path=abstraction_path,
        rng_seed=rng_seed,
    )
    log.info("cold start complete in %.2fs", time.perf_counter() - t0)
    state = ServeState(adapter)

    async def handler(conn: ServerConnection) -> None:
        await _connection_handler(state, conn)

    stop_event = asyncio.Event()

    def _trigger_stop(*_: object) -> None:
        log.info("shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig_name in ("SIGINT", "SIGTERM"):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            # NotImplementedError: Windows; RuntimeError: nested loop / non-main thread.
            loop.add_signal_handler(getattr(signal, sig_name), _trigger_stop)

    try:
        async with serve(handler, host, port):
            log.info("listening on ws://%s:%d", host, port)
            if ready_event is not None:
                ready_event.set()
            await stop_event.wait()
    finally:
        try:
            db.close()
        except Exception:
            log.exception("error closing DB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--db",
        default=None,
        help="primary StrategyDB (6-max). Required unless --multi-size-config is given.",
    )
    parser.add_argument(
        "--db-9max",
        default=None,
        help="optional 9-max StrategyDB for dual routing (ignored if --multi-size-config is set)",
    )
    parser.add_argument(
        "--multi-size-config",
        default=None,
        help=(
            "path to YAML mapping {table_size: db_path} for per-size routing via "
            "MultiSizeStrategyDB. Overrides --db / --db-9max when set."
        ),
    )
    parser.add_argument("--abstraction-path", default="abstraction")
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.db is None and args.multi_size_config is None:
        parser.error("either --db or --multi-size-config must be provided")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    asyncio.run(
        run_server(
            host=args.host,
            port=args.port,
            db_path=args.db,
            db_path_9max=args.db_9max,
            multi_size_config_path=args.multi_size_config,
            abstraction_path=args.abstraction_path,
            rng_seed=args.rng_seed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
