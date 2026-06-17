#!/bin/bash
# Batch runner for HardMove TTPI reproduction experiments.
# Usage:
#   bash repro/scripts/run_hardmove_batch.sh                    # all experiments
#   bash repro/scripts/run_hardmove_batch.sh --smoke            # smoke test only
#   bash repro/scripts/run_hardmove_batch.sh --group hm8        # HM(8) only
#   bash repro/scripts/run_hardmove_batch.sh --group hm12       # HM(12) only
#   bash repro/scripts/run_hardmove_batch.sh --group hm16       # HM(16) only

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$ROOT/repro/scripts/run_hardmove.py"
LOG_DIR="$ROOT/repro/logs"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Default parameters (matching paper / Phase A baseline)
# ---------------------------------------------------------------------------
N_STATE=100
N_ACTION=100
N_ITER=100
N_ITER_V=1
DT=0.01
GAMMA=0.99
TARGET_RADIUS=0.02
N_TEST=100
DEVICE="cuda"

# TT-Cross parameters
RMAX_V=100
RMAX_A=100
NSWP_V=5
NSWP_A=10
KICKRANK_V=10
KICKRANK_A=10
EPS_CROSS_V=1e-3
EPS_CROSS_A=1e-3
EPS_ROUND_V=1e-3
EPS_ROUND_A=1e-3
N_SAMPLES=50

CALLBACK_FREQ=10
MAX_BATCH_V=10000
MAX_BATCH_A=100000

EARLY_STOP_SUCCESS=0.95
EARLY_STOP_MU=0.70
EARLY_STOP_AFTER=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            MODE="smoke"
            shift
            ;;
        --group)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--smoke | --group hm8|hm12|hm16]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
run_one() {
    local n_actuator=$1
    local seed=$2
    local extra_flags="${3:-}"

    local task="HM${n_actuator}_state${N_STATE}_action${N_ACTION}_iter${N_ITER}_seed${seed}"
    local logfile="$LOG_DIR/${task}.log"

    echo "=============================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: $task"
    echo "=============================================================================="

    python "$SCRIPT" \
        --n-actuator "$n_actuator" \
        --n-state "$N_STATE" \
        --n-action "$N_ACTION" \
        --seed "$seed" \
        --n-iter "$N_ITER" \
        --n-iter-v "$N_ITER_V" \
        --dt "$DT" \
        --gamma "$GAMMA" \
        --target-radius "$TARGET_RADIUS" \
        --n-test "$N_TEST" \
        --device "$DEVICE" \
        --rmax-v "$RMAX_V" \
        --rmax-a "$RMAX_A" \
        --nswp-v "$NSWP_V" \
        --nswp-a "$NSWP_A" \
        --kickrank-v "$KICKRANK_V" \
        --kickrank-a "$KICKRANK_A" \
        --eps-cross-v "$EPS_CROSS_V" \
        --eps-cross-a "$EPS_CROSS_A" \
        --eps-round-v "$EPS_ROUND_V" \
        --eps-round-a "$EPS_ROUND_A" \
        --n-samples "$N_SAMPLES" \
        --callback-freq "$CALLBACK_FREQ" \
        --max-batch-v "$MAX_BATCH_V" \
        --max-batch-a "$MAX_BATCH_A" \
        --early-stop-success "$EARLY_STOP_SUCCESS" \
        --early-stop-mu "$EARLY_STOP_MU" \
        --early-stop-after-callback "$EARLY_STOP_AFTER" \
        $extra_flags \
        2>&1 | tee "$logfile"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished: $task"
}

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
run_smoke() {
    echo "Running SMOKE TEST (fast, low-res, few iters)..."

    python "$SCRIPT" \
        --n-actuator 8 \
        --n-state 30 \
        --n-action 50 \
        --seed 0 \
        --n-iter 5 \
        --n-iter-v 1 \
        --dt 0.01 \
        --gamma 0.99 \
        --target-radius 0.02 \
        --n-test 20 \
        --device "$DEVICE" \
        --rmax-v 100 \
        --rmax-a 100 \
        --nswp-v 5 \
        --nswp-a 10 \
        --kickrank-v 10 \
        --kickrank-a 10 \
        --eps-cross-v 1e-3 \
        --eps-cross-a 1e-3 \
        --eps-round-v 1e-3 \
        --eps-round-a 1e-3 \
        --n-samples 50 \
        --callback-freq 1 \
        --max-batch-v 10000 \
        --max-batch-a 100000 \
        --early-stop-success 0.0 \
        --early-stop-mu 0.0 \
        2>&1 | tee "$LOG_DIR/HM8_smoke.log"

    echo "[SMOKE] Done."
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
if [[ "$MODE" == "smoke" ]]; then
    run_smoke
    exit 0
fi

SEEDS=(0 1 2 3 4)

declare -A ACTUATORS
if [[ "$MODE" == "all" ]]; then
    ACTUATORS=([8]=1 [12]=1 [16]=1)
elif [[ "$MODE" == "hm8" ]]; then
    ACTUATORS=([8]=1)
elif [[ "$MODE" == "hm12" ]]; then
    ACTUATORS=([12]=1)
elif [[ "$MODE" == "hm16" ]]; then
    ACTUATORS=([16]=1)
else
    echo "Unknown mode: $MODE"
    exit 1
fi

echo "============================================"
echo " Phase A: HardMove Reproduction"
echo " n_state=$N_STATE  n_action=$N_ACTION  n_iter=$N_ITER"
echo " Seeds: ${SEEDS[*]}"
echo " Actuators: ${!ACTUATORS[*]}"
echo " Early stop: S>=$EARLY_STOP_SUCCESS, mu>=$EARLY_STOP_MU"
echo "============================================"

for n_act in "${!ACTUATORS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_one "$n_act" "$seed"
        echo ""
    done
done

echo ""
echo "============================================"
echo " All experiments done. Summary:"
echo "============================================"
python "$ROOT/repro/scripts/summarize_hardmove.py"
