#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-outputs/cmf_peno_rf_v3_loso}
BASE_ROOT=${BASE_ROOT:-outputs/nc_multiscene_loso}
RF_ROOT=${RF_ROOT:-outputs/cmf_peno_rf_loso/datasets}
SCENES=${SCENES:-"center east2 north west"}
SEEDS=${SEEDS:-"2026 2027 2028"}
EPOCHS=${EPOCHS:-60}
VARIANCE_EPOCHS=${VARIANCE_EPOCHS:-20}
DEVICE=${DEVICE:-cuda}
PYTHON_BIN=${PYTHON_BIN:-python}
EVENT_SAMPLES=${EVENT_SAMPLES:-32}

mkdir -p "$ROOT" "$RF_ROOT"

for SCENE in $SCENES; do
  BASE="$BASE_ROOT/heldout_${SCENE}.npz"
  RF="$RF_ROOT/${SCENE}_rf_v2_s${EVENT_SAMPLES}.npz"
  [[ -f "$BASE" ]] || { echo "missing heldout pack: $BASE" >&2; exit 2; }
  if [[ ! -f "$RF" ]]; then
    PYTHONPATH=src "$PYTHON_BIN" scripts/build_cmf_peno_rf_dataset.py \
      --base-dataset "$BASE" --output "$RF" --samples "$EVENT_SAMPLES" \
      --max-events 4 --frequency-mhz 2500
  fi
  for SEED in $SEEDS; do
    RUN="$ROOT/${SCENE}_seed${SEED}"
    MODEL="$RUN/full_v3"
    mkdir -p "$RUN"
    if [[ ! -f "$MODEL/training.json" ]]; then
      PYTHONPATH=src "$PYTHON_BIN" scripts/train_cmf_peno_rf_v3.py \
        --dataset "$RF" --output-dir "$MODEL" --epochs "$EPOCHS" \
        --variance-epochs "$VARIANCE_EPOCHS" \
        --seed "$SEED" --device "$DEVICE" 2>&1 | tee "$RUN/train_v3.log"
    fi
    if [[ ! -f "$MODEL/eval_val/metrics.json" ]]; then
      PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_cmf_peno_rf_v3.py \
        --dataset "$RF" --checkpoint "$MODEL/best.pt" --split val \
        --output-dir "$MODEL/eval_val" --device "$DEVICE"
    fi
    if [[ ! -f "$MODEL/variance_calibration.json" ]]; then
      PYTHONPATH=src "$PYTHON_BIN" scripts/fit_cmf_peno_variance_temperature.py \
        --validation-predictions "$MODEL/eval_val/predictions.npz" \
        --output "$MODEL/variance_calibration.json"
    fi
    if [[ ! -f "$MODEL/eval_test/metrics.json" ]]; then
      PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_cmf_peno_rf_v3.py \
        --dataset "$RF" --checkpoint "$MODEL/best.pt" --split test \
        --output-dir "$MODEL/eval_test" --device "$DEVICE" \
        --calibration-json "$MODEL/variance_calibration.json"
    fi
    PYTHONPATH=src "$PYTHON_BIN" scripts/summarize_cmf_peno_rf.py \
      --run-root "$RUN" --split test --output "$RUN/summary_test.csv"
  done
done

PYTHONPATH=src "$PYTHON_BIN" scripts/summarize_cmf_peno_rf_loso.py \
  --root "$ROOT" --reference-variant full_v3 --output "$ROOT/loso_summary.csv"

echo "CMF-PENO-RF v3 LOSO outputs written to $ROOT"
