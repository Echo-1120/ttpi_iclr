#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

stage="${1:-all}"
PYTHON_BIN="${PYTHON:-python3}"

run_analysis() {
  "$PYTHON_BIN" -m analysis.prestudy_5080 --stage "$1"
}

run_manifest() {
  local manifest="$1"
  "$PYTHON_BIN" repro/scripts/run_pam_ablation.py \
    --manifest "$manifest" \
    --execute \
    --resume
}

case "$stage" in
  audit)
    run_analysis audit
    ;;
  analyze-existing)
    run_analysis analyze-existing
    ;;
  smoke)
    run_manifest repro/experiments/prestudy_5080_smoke_manifest.json
    ;;
  sensitivity-lite)
    run_manifest repro/experiments/prestudy_5080_sensitivity_lite_manifest.json
    ;;
  hybrid-ablation)
    run_manifest repro/experiments/prestudy_5080_hybrid_ablation_manifest.json
    ;;
  report)
    run_analysis report
    ;;
  all)
    run_analysis audit
    run_analysis analyze-existing
    run_manifest repro/experiments/prestudy_5080_smoke_manifest.json
    run_manifest repro/experiments/prestudy_5080_sensitivity_lite_manifest.json
    run_manifest repro/experiments/prestudy_5080_hybrid_ablation_manifest.json
    run_analysis report
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    echo "Usage: bash scripts/run_prestudy_5080.sh {audit|analyze-existing|smoke|sensitivity-lite|hybrid-ablation|report|all}" >&2
    exit 2
    ;;
esac
