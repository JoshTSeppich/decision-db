#!/usr/bin/env bash
# Prepare-only launcher for the L1 3-handed (numPlayers=3) Deep CFR blueprint.
#
# Mirrors the v5 production runs (training/v5-6max, v5-7max, v5-8max): same
# launcher (scripts/launch_pilot_training.py), same thread cap, same local
# checkpoint dir, same seed — only --table-size and the output paths change.
#
# NOT launched automatically. Run it yourself on the RunPod A100 pod:
#     bash scripts/launch_3max_blueprint.sh
# Resume a died run from its last checkpoint:
#     bash scripts/launch_3max_blueprint.sh --resume /tmp/pokerbot-ckpt-v5-3max/iter_NNNN.pt
#
# ── Settings mirrored from the v5 runs ───────────────────────────────────────
#   --table-size 3              numPlayers=3 (the only change vs v5-6max/7max/8max)
#   --outer-iters 1000          matches the v5-7max / v5-8max checkpoint depth
#   --traversals-per-iter 200   the user-approved production value (v5-6max Run 1)
#   OMP/MKL = 32                pins torch threads to the 32-vCPU pod (the fix for
#                               the thread-oversubscription that made attempt 1 ~60h)
#   --checkpoint-dir local      LARGE local container disk, never a small mounted
#                               volume (a full 20 GB volume killed an earlier run)
#   buffers/batch/policy-steps  left at launcher defaults (3M policy reservoir,
#                               500k per-player advantage, batch 256) — same as v5.
#
# ── Estimated wall-clock + cost (extrapolated from the v5-6max Run-1 actuals) ──
#   v5-6max actual: 2500 iters @ ~40-45s/iter (32-vCPU A100-SXM, $1.49/hr) ≈ 25-28h, ~$39.
#   The hot path is single-threaded pokerkit/traversal that scales ~linearly with
#   player count, so 3-max per-iter ≈ 3/6 of 6-max traversal + ~unchanged ~10s train
#   ≈ 25-35s/iter. At 1000 iters: ~7-10h wall-clock, ~$11-15 at $1.49/hr.
#   (Dominant uncertainty: per-iter traversal speed varies with showdown frequency.
#    For full parity with the 6-max depth, set OUTER_ITERS=2500 → ~18-25h, ~$27-37.)
set -euo pipefail
cd "$(dirname "$0")/.."

OUTER_ITERS="${OUTER_ITERS:-1000}"
TRAVERSALS="${TRAVERSALS:-200}"
SEED="${SEED:-2026}"
OUT_DIR="training/v5-3max"
DB_URL="sqlite:///${OUT_DIR}/strategy-v5-3max.db"
CKPT_DIR="${CKPT_DIR:-/tmp/pokerbot-ckpt-v5-3max}"

export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

mkdir -p "${OUT_DIR}" logs
# shellcheck disable=SC1091
source .venv/bin/activate

echo "launch (3-max blueprint): $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
set -x
python scripts/launch_pilot_training.py \
  --table-size 3 \
  --outer-iters "${OUTER_ITERS}" \
  --traversals-per-iter "${TRAVERSALS}" \
  --seed "${SEED}" \
  --db "${DB_URL}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --abstraction-dir abstraction \
  "$@" \
  2>&1 | tee "logs/training-v5-3max.log"
set +x
echo "finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
