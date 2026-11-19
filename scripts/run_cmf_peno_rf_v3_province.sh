#!/usr/bin/env bash
set -euo pipefail

# Resumable province-scale grouped validation and same-version ablation run.
FOLD_ROOT=${FOLD_ROOT:-outputs/jiangxi_province_validation/folds}
ROOT=${ROOT:-outputs/cmf_peno_rf_v3_province}
SEEDS=${SEEDS:-"2026 2027 2028"}
VARIANTS=${VARIANTS:-"full_v3 cnn_residual fno_residual no_multilevel no_events no_local no_spectral no_event_consistency deterministic"}
EPOCHS=${EPOCHS:-60}
VARIANCE_EPOCHS=${VARIANCE_EPOCHS:-20}
DEVICE=${DEVICE:-cuda}
PYTHON_BIN=${PYTHON_BIN:-python}
EVENT_SAMPLES=${EVENT_SAMPLES:-32}

mkdir -p "$ROOT"
if [[ -n "${FOLDS:-}" ]]; then
  read -r -a FOLD_LIST <<< "$FOLDS"
else
  mapfile -t FOLD_LIST < <(find "$FOLD_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi
[[ ${#FOLD_LIST[@]} -ge 1 ]] || { echo "no grouped folds found under $FOLD_ROOT" >&2; exit 2; }

for FOLD in "${FOLD_LIST[@]}"; do
  DATA="$FOLD_ROOT/$FOLD/cmf_peno_rf_s${EVENT_SAMPLES}.npz"
  [[ -f "$DATA" ]] || { echo "missing fold dataset: $DATA" >&2; exit 2; }
  for SEED in $SEEDS; do
    RUN="$ROOT/${FOLD}_seed${SEED}"
    for VARIANT in $VARIANTS; do
      MODEL="$RUN/$VARIANT"
      if [[ ! -f "$MODEL/training.json" || ! -f "$MODEL/best.pt" ]]; then
        if [[ -e "$MODEL" ]]; then
          echo "incomplete model directory requires review: $MODEL" >&2
          exit 2
        fi
        PYTHONPATH=src "$PYTHON_BIN" scripts/train_cmf_peno_rf_v3.py \
          --dataset "$DATA" --output-dir "$MODEL" --variant "$VARIANT" \
          --epochs "$EPOCHS" --variance-epochs "$VARIANCE_EPOCHS" \
          --seed "$SEED" --device "$DEVICE"
      fi
      if [[ ! -f "$MODEL/eval_val/metrics.json" ]]; then
        PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_cmf_peno_rf_v3.py \
          --dataset "$DATA" --checkpoint "$MODEL/best.pt" --split val \
          --output-dir "$MODEL/eval_val" --device "$DEVICE"
      fi
      CALIBRATION_ARGS=()
      if [[ "$VARIANT" != "deterministic" ]]; then
        if [[ ! -f "$MODEL/eval_calibration/metrics.json" ]]; then
          PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_cmf_peno_rf_v3.py \
            --dataset "$DATA" --checkpoint "$MODEL/best.pt" --split calibration \
            --output-dir "$MODEL/eval_calibration" --device "$DEVICE"
        fi
        if [[ ! -f "$MODEL/variance_calibration.json" ]]; then
          PYTHONPATH=src "$PYTHON_BIN" scripts/fit_cmf_peno_variance_temperature.py \
            --validation-predictions "$MODEL/eval_calibration/predictions.npz" \
            --output "$MODEL/variance_calibration.json"
        fi
        CALIBRATION_ARGS=(--calibration-json "$MODEL/variance_calibration.json")
      fi
      if [[ ! -f "$MODEL/eval_test/metrics.json" ]]; then
        PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_cmf_peno_rf_v3.py \
          --dataset "$DATA" --checkpoint "$MODEL/best.pt" --split test \
          --output-dir "$MODEL/eval_test" --device "$DEVICE" \
          "${CALIBRATION_ARGS[@]}"
      fi
    done
    if [[ ! -f "$RUN/summary_test.csv" ]]; then
      PYTHONPATH=src "$PYTHON_BIN" scripts/summarize_cmf_peno_rf.py \
        --run-root "$RUN" --split test --output "$RUN/summary_test.csv"
    fi
  done
  BASELINES="$ROOT/${FOLD}_baselines"
  if [[ ! -e "$BASELINES" ]]; then
    PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_jiangxi_field_baselines.py \
      --dataset "$DATA" --output-dir "$BASELINES"
  fi
done

if [[ ! -f "$ROOT/grouped_summary.csv" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/summarize_cmf_peno_rf_loso.py \
    --root "$ROOT" --reference-variant full_v3 --output "$ROOT/grouped_summary.csv"
fi

SUMMARY_DIR="$ROOT/province_statistics"
if [[ ${#FOLD_LIST[@]} -ge 4 && ! -e "$SUMMARY_DIR" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/summarize_jiangxi_province_experiments.py \
    --root "$ROOT" --reference-variant full_v3 \
    --baseline-glob '*_baselines/case_metrics.csv' --output-dir "$SUMMARY_DIR"
fi
echo "Province validation and ablation outputs written to $ROOT"
