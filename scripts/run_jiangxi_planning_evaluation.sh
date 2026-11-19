#!/usr/bin/env bash
set -euo pipefail

# Select on each model's own field and evaluate all plans on independent truth.
FOLD_ROOT=${FOLD_ROOT:-outputs/jiangxi_province_validation/folds}
MODEL_ROOT=${MODEL_ROOT:-outputs/cmf_peno_rf_v3_province}
REFERENCE_ROOT=${REFERENCE_ROOT:-outputs/jiangxi_independent_reference}
ROOT=${ROOT:-outputs/jiangxi_planning_independent}
SEEDS=${SEEDS:-"2026 2027 2028"}
VARIANTS=${VARIANTS:-"full_v3 cnn_residual fno_residual no_events no_local no_spectral deterministic"}
BUDGETS=${BUDGETS:-"1 2 3 4 6 8"}
PYTHON_BIN=${PYTHON_BIN:-python}
EVENT_SAMPLES=${EVENT_SAMPLES:-32}
THRESHOLD_DBM=${THRESHOLD_DBM:--100}
SHADOW_STD_DB=${SHADOW_STD_DB:-6}
PROBABILITY_THRESHOLD=${PROBABILITY_THRESHOLD:-0.9}
RANDOM_REPEATS=${RANDOM_REPEATS:-100}
EXACT_BUDGET_MAX=${EXACT_BUDGET_MAX:-3}

mkdir -p "$ROOT"
if [[ -n "${FOLDS:-}" ]]; then
  read -r -a FOLD_LIST <<< "$FOLDS"
else
  mapfile -t FOLD_LIST < <(find "$FOLD_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

for FOLD in "${FOLD_LIST[@]}"; do
  DATA="$FOLD_ROOT/$FOLD/cmf_peno_rf_s${EVENT_SAMPLES}.npz"
  REFERENCE="$REFERENCE_ROOT/$FOLD.npz"
  [[ -f "$DATA" && -f "$REFERENCE" ]] || {
    echo "missing dataset or independent reference for $FOLD" >&2; exit 2;
  }
  for SEED in $SEEDS; do
    OUT="$ROOT/${FOLD}_seed${SEED}"
    if [[ ! -f "$OUT/planning_metadata.json" ]]; then
      PREDICTION_ARGS=()
      for VARIANT in $VARIANTS; do
        PREDICTION="$MODEL_ROOT/${FOLD}_seed${SEED}/$VARIANT/eval_test/predictions.npz"
        [[ -f "$PREDICTION" ]] || { echo "missing prediction: $PREDICTION" >&2; exit 2; }
        PREDICTION_ARGS+=(--prediction "$VARIANT=$PREDICTION")
      done
      PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_jiangxi_candidate_planning.py \
        --dataset "$DATA" "${PREDICTION_ARGS[@]}" \
        --reference "$REFERENCE" --require-independent-reference \
        --planning-scope both \
        --budgets $BUDGETS --threshold-dbm "$THRESHOLD_DBM" \
        --shadow-std-db "$SHADOW_STD_DB" \
        --probability-threshold "$PROBABILITY_THRESHOLD" \
        --random-repeats "$RANDOM_REPEATS" --exact-budget-max "$EXACT_BUDGET_MAX" \
        --seed "$SEED" --output-dir "$OUT"
    fi
  done
done

if [[ ! -e "$ROOT/summary" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" scripts/summarize_jiangxi_planning.py \
    --root "$ROOT" --output-dir "$ROOT/summary"
fi
echo "Independent planning results written to $ROOT"
