#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Pipeline scheme
# ============================================================
#
# For each seed:
#
#   SASRec training
#        ↓
#   unique log file:
#        logs/<method>_<dataset>_seed<seed>.log
#
# ============================================================

PIPELINE_FAILED=0

run_step () {
  local step_name="$1"
  shift

  # After first failure, skip all remaining steps.
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0

  echo -e "\n=== ${step_name} ==="

  if "$@"; then
    echo "✅ Step OK: ${step_name}"
  else
    local status=$?
    echo "❌ Step FAILED (exit code ${status}): ${step_name}"
    PIPELINE_FAILED=1
  fi
}

DATASET="AmazonReviews2014"
# CATEGORY="Musical_Instruments"
CATEGORY="Beauty"
# CATEGORY="Toys_and_Games"
# CATEGORY="Sports_and_Outdoors"
# CATEGORY="Books"

# DATASET="Yelp"
# DATASET="MIND"

METHOD_NAME="SASRec_BPR"
SEEDS=(42 43 44)

DEVICE_NO=0
LOG_DIR="logs"

mkdir -p "${LOG_DIR}"

for SEED in "${SEEDS[@]}"; do
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && break

  # Build optional arguments
  DATASET_ARG=""
  if [[ -n "${DATASET:-}" ]]; then
    DATASET_ARG="--dataset='${DATASET}'"
    DATASET_CATEGORY="${DATASET}"
  fi

  CATEGORY_ARG=""
  if [[ -n "${CATEGORY:-}" ]]; then
    CATEGORY_ARG="--category='${CATEGORY}'"
    DATASET_CATEGORY="${DATASET}-${CATEGORY}"
  fi

  # Alias identifies every artifact from this run.
  ALIAS="${METHOD_NAME}_${DATASET_CATEGORY}_seed${SEED}"
  FINAL_METRICS_FILE="results/${ALIAS,,}.json"
  LOG_FILE="${LOG_DIR}/${ALIAS,,}.log"

  echo -e "\n########################################"
  echo "Dataset: ${DATASET}"
  echo "Method:  ${METHOD_NAME}"
  echo "Seed:    ${SEED}"
  echo "Alias:   ${ALIAS}"
  echo "########################################"

  # =========================
  # 1. Train SASRec
  # =========================

  run_step "Run SASRec | ${ALIAS}" bash -o pipefail -c "
  CUDA_VISIBLE_DEVICES='${DEVICE_NO}' python main_genrec.py \
    --final_metrics_file='${FINAL_METRICS_FILE}' \
    --model=SASRec \
    --n_head=4 \
    --n_embd=32 \
    ${DATASET_ARG} \
    ${CATEGORY_ARG} \
    --rand_seed='${SEED}' \
    --max_item_seq_len=20 \
    --train_batch_size=512 \
    --eval_batch_size=32 \
    --loss_type=basic \
    > '${LOG_FILE}' 2>&1
  "
done

echo
if [[ "${PIPELINE_FAILED}" -ne 0 ]]; then
  echo "🎯 Pipeline stopped after first failure"
else
  echo "🎯 Pipeline finished successfully"
fi
