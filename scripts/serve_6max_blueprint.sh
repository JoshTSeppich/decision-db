#!/usr/bin/env bash
# Serve the live zoom advisory service against the real de-biased 6-max RNR
# blueprint (training/v5-6max-fix.db).
#
# Competition target is 6-MAX. This is a PURE-6-max wiring: a single 6-max DB,
# NO --db-9max secondary. With the secondary omitted, build_zoom_adapter uses
# the single-DB path (db = primary), so every table_size=6 request keys directly
# into the v5-6max blueprint with no DualStrategyDB routing or version-match
# constraint.
#
# NOTE: table_size is pass-through from the eyes' GameStateRequest (not
# hardcoded). With a 6-max-only DB and no secondary, a NON-6 request (e.g. a
# table that drops to 5-handed after a bust) will fall through to default_policy.
# For strictly-6-max play that is correct; if short-handed tables must be served,
# add a real short-handed DB as --db-9max.
#
# Run:
#     bash scripts/serve_6max_blueprint.sh
#     PORT=8770 DB=training/v5-6max-fix.db bash scripts/serve_6max_blueprint.sh
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
exec python scripts/serve_zoom.py \
  --port "${PORT:-8766}" \
  --db "${DB:-training/v5-6max-fix.db}" \
  --abstraction-path abstraction
