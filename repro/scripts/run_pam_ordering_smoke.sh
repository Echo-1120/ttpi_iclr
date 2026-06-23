#!/usr/bin/env bash
set -euo pipefail

# Smoke-test generated PAM ordering against key baselines.
# Run this on the GPU server from the repository root.

DEVICE="${DEVICE:-cuda}"
N_ACTUATOR="${N_ACTUATOR:-8}"
SEED="${SEED:-0}"
N_ITER="${N_ITER:-30}"
N_STATE="${N_STATE:-40}"
N_ACTION="${N_ACTION:-50}"
N_TEST="${N_TEST:-50}"

COMMON_ARGS=(
  --n-actuator "${N_ACTUATOR}"
  --n-state "${N_STATE}"
  --n-action "${N_ACTION}"
  --n-iter "${N_ITER}"
  --callback-freq 10
  --n-test "${N_TEST}"
  --seed "${SEED}"
  --rmax-v 60
  --rmax-a 60
  --max-batch-v 5000
  --max-batch-a 20000
  --eps-cross-v 1e-3
  --eps-cross-a 1e-3
  --nswp-v 5
  --nswp-a 10
  --n-samples 50
  --device "${DEVICE}"
)

mkdir -p repro/logs

python repro/scripts/build_pam_ordering.py --n-actuator "${N_ACTUATOR}" \
  | tee "repro/logs/pam_ordering_HM${N_ACTUATOR}.log"

for order in local pam_greedy pam_spectral opposite_pair badsplit random; do
  echo "=== HM${N_ACTUATOR} seed=${SEED} order=${order} ==="
  python repro/scripts/run_hardmove.py "${COMMON_ARGS[@]}" --action-order "${order}" \
    2>&1 | tee "repro/logs/HM${N_ACTUATOR}_${order}_seed${SEED}_smoke.log"
done
