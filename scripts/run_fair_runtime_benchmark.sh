#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PWD}"
OUT_ROOT="${OUT_ROOT:-$CODE_ROOT/outputs/fair_runtime_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-2026}"
MAX_CASES="${MAX_CASES:-20}"
REPEATS="${REPEATS:-7}"
export PYTHONPATH="$DATA_ROOT/src:$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for FOLD in region_01 region_02 region_03 region_04 region_05; do
    DATASET="$DATA_ROOT/outputs/jiangxi_province_validation/folds/$FOLD/cmf_peno_rf_s32.npz"
    CHECKPOINT="$DATA_ROOT/outputs/cmf_peno_rf_v3_province/${FOLD}_seed${SEED}/full_v3/best.pt"
    if [[ ! -f "$DATASET" || ! -f "$CHECKPOINT" ]]; then
        printf 'Missing dataset or checkpoint for %s:\n%s\n%s\n' "$FOLD" "$DATASET" "$CHECKPOINT" >&2
        exit 2
    fi
    python "$CODE_ROOT/scripts/benchmark_reference_vs_network.py" \
        --dataset "$DATASET" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUT_ROOT/$FOLD" \
        --split test \
        --max-cases "$MAX_CASES" \
        --repeats "$REPEATS" \
        --seed "$SEED" \
        --threads 1 \
        --cuda
done

python "$CODE_ROOT/scripts/summarize_reference_runtime.py" --root "$OUT_ROOT"
printf 'Saved benchmark to %s\n' "$OUT_ROOT"
