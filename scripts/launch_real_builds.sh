#!/usr/bin/env bash
# Launch the three production abstraction builds in parallel.
# Each tee's its stdout to a per-stream log; sentinel files mark completion.
#
# Usage: scripts/launch_real_builds.sh
# Run from project root.
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p abstraction/ logs/
# Wipe any prior sentinels so we can detect this run's completion.
rm -f logs/.river.done logs/.flop.done logs/.turn.done
rm -f logs/build-river.log logs/build-flop.log logs/build-turn.log

# shellcheck disable=SC1091
source .venv/bin/activate

date "+launch: %Y-%m-%d %H:%M:%S"

start_stream() {
  local street="$1"
  (
    pokerbot build-abstraction \
      --out abstraction/ \
      --streets "$street" \
      --samples 1000 \
      --max-classes 200000 \
      2>&1 | tee "logs/build-${street}.log"
    rc=$?
    echo "$rc" > "logs/.${street}.done"
    exit "$rc"
  ) &
  echo "started $street (pid=$!)"
}

start_stream river
start_stream flop
start_stream turn

wait
echo "all three streams completed"
date "+finished: %Y-%m-%d %H:%M:%S"
