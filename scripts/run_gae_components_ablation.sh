#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Pipeline scheme
# ============================================================
#
# For each dataset × seed:
#
#   0) Pretrain CF model     → Extrac embeddings from model checkpoint
#   1) gensid/main.py        → SID mapping (gnn_sid_mapping.json)
#   2) Copy SID mapping      → gensid to genrec cache
#   3) gensid_to_genrec      → .sem_ids mapping + env file
#   4) main_genrec.py        → finetune MemGen-GR / TIGER
#
# ============================================================

PIPELINE_FAILED=0

run_step() {
  local step_name="$1"; shift
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0
  echo -e "\n=== ${step_name} ==="
  if "$@" >> "$LOG_FILE" 2>&1; then
    echo "✅ Step OK: ${step_name}"
  else
    local status=$?
    echo "❌ Step FAILED (exit code ${status}): ${step_name}"
    PIPELINE_FAILED=1
  fi
}

# ============================================================
# Configuration
# ============================================================

# DATASETS=("Beauty" "Toys_and_Games" "Sports_and_Outdoors" "MIND")
DATASETS=("Beauty")
SEEDS=(42)
METHOD_NAME="GAE_t5"
EMB_MODEL="sentence-t5-base"
# METHOD_NAME="GAE_qwen"
# EMB_MODEL="Qwen3-Embedding-0.6B"

GENSID_SRC_DIR="./gensid"
GENSID_CKPT_DIR="./ckpt/gensid"
GENSID_CACHE_DIR="./cache/gensid"
GR_CACHE_DIR="./cache"

DEVICE_NO=0

mkdir -p "logs"

# ============================================================
# RQ-VAE training argument presets
#
# Define argument sets as functions that print flags to stdout.
# Pass the desired preset name to run_pipeline as the third arg.
#
# Available presets:
#   rqvae_args_full       - all flags enabled (default) -- RQ-GAE
#   rqvae_args_no_graph   - ablation: remove graph by setting num_prop to zero
#   rqvae_args_no_edge    - ablation: remove --use_edge_reconstruction_loss
#   rqvae_args_minimal    - ablation: no graph signal, no edge reconstruction -- TIGER (RQ-VAE)
# ============================================================

rqvae_args_full() {
  echo \
    --quant_loss_weight 0.25 \
    --epochs 150 \
    --alpha 0.02 \
    --beta 0.0 \
    --layers 512 256 128 64 \
    --eval_step 100 \
    --num_prop 1 \
    --appnp_alpha 0.5 \
    --use_edge_weight \
    --graph_signal \
    --disable_vq_init \
    --cf_loss_use_codebook_rep \
    --use_edge_reconstruction_loss
}

rqvae_args_no_graph() {
  echo \
    --quant_loss_weight 0.25 \
    --epochs 150 \
    --alpha 0.02 \
    --beta 0.0 \
    --layers 512 256 128 64 \
    --eval_step 100 \
    --num_prop 0 \
    --appnp_alpha 1.0 \
    --use_edge_weight \
    --graph_signal \
    --disable_vq_init \
    --cf_loss_use_codebook_rep \
    --use_edge_reconstruction_loss
}

rqvae_args_no_edge() {
  echo \
    --quant_loss_weight 0.25 \
    --epochs 150 \
    --alpha 0.0 \
    --beta 0.0 \
    --layers 512 256 128 64 \
    --eval_step 100 \
    --num_prop 1 \
    --appnp_alpha 0.5 \
    --disable_vq_init \
    --disable_cf_loss \
    --use_edge_weight \
    --graph_signal
}

rqvae_args_minimal() {
  echo \
    --quant_loss_weight 0.25 \
    --epochs 150 \
    --alpha 0.0 \
    --beta 0.0 \
    --layers 512 256 128 64 \
    --eval_step 100 \
    --disable_cf_loss
}

# ============================================================
# Helper functions
# ============================================================

get_dataset_prefix() {
  local dataset="$1"
  case "${dataset}" in
    Musical_Instruments|Beauty|Toys_and_Games|Sports_and_Outdoors|Books) echo AmazonReviews2014/ ;;
    *) echo "" ;;
  esac
}

# Returns the correct dataset/category flags for the GR finetuning step
get_gr_dataset_args() {
  local dataset="$1"
  case "${dataset}" in
    Musical_Instruments|Beauty|Toys_and_Games|Sports_and_Outdoors|Books)
      echo "--dataset=AmazonReviews2014 --category=${dataset}"
      ;;
    *)
      echo "--dataset=${dataset}"
      ;;
  esac
}

# ============================================================
# Core pipeline function
# Usage: run_pipeline <dataset> <seed> <rqvae_preset>
# ============================================================

# Parse --skip=step1,step2,... from CLI args
SKIP_STEPS=()
for arg in "$@"; do
  case "${arg}" in
    --skip=*) IFS=',' read -r -a SKIP_STEPS <<< "${arg#--skip=}" ;;
  esac
done

# Check membership
should_skip() {
  local step="$1"
  for s in "${SKIP_STEPS[@]}"; do
    [[ "${s}" == "${step}" ]] && return 0
  done
  return 1
}

run_pipeline() {
  local DATASET="$1"
  local SEED="$2"
  local RQVAE_PRESET="${3:-letter}"

  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0

  local DATA_PATH="${GENSID_CACHE_DIR}/${DATASET}/${DATASET}.emb-${EMB_MODEL}-td.npy"
  local INTERACTIONS_PATH="${GENSID_CACHE_DIR}/${DATASET}/${DATASET}.inter.json"
  local CF_EMB="${GENSID_CKPT_DIR}/${DATASET}-32d-sasrec.pt"
  local GR_PATH="${GR_CACHE_DIR}/$(get_dataset_prefix "${DATASET}")${DATASET}/processed"

  local ALIAS="${METHOD_NAME}_${RQVAE_PRESET}_${DATASET}_seed${SEED}"
  local CKPT_BASE="./${GENSID_CKPT_DIR}/quantizer"
  local SAVE_MODEL_DIR="${ALIAS}/full_run_model"
  local GR_SID_MAPPING_PATH="${GENSID_CACHE_DIR}/${DATASET}.${ALIAS}.json"
  local MAP_ENV="logs/env/${ALIAS}_sid_mapping.env"
  local LOG_FILE="logs/${ALIAS,,}.log"
  local FINAL_METRICS_FILE="results/${ALIAS,,}.json"

  mkdir -p "results"
  mkdir -p "logs/env/"

  echo -e "\n########################################"
  echo "Dataset:  ${DATASET}"
  echo "Method:   ${METHOD_NAME}"
  echo "Seed:     ${SEED}"
  echo "Preset:   ${RQVAE_PRESET}"
  echo "Alias:    ${ALIAS}"
  echo "########################################"

  # Read args from the preset function into an array
  local -a RQVAE_EXTRA_ARGS
  read -r -a RQVAE_EXTRA_ARGS <<< "$("${RQVAE_PRESET}")"

  # ------------------------------------------------------------------
  # 1. Train RQ-VAE
  # ------------------------------------------------------------------
  echo "${RQVAE_EXTRA_ARGS[@]}"
  should_skip "gensid" || run_step "Train-Predict RQ-VAE [${ALIAS}]" \
    python "main_gensid.py" \
      --device="cuda:${DEVICE_NO}" \
      --dataset="${DATASET}" \
      --embedder="${EMB_MODEL}" \
      --saved_model_dir="${SAVE_MODEL_DIR}" \
      --global_seed="${SEED}" \
      "${RQVAE_EXTRA_ARGS[@]}"

  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0

  # ------------------------------------------------------------------
  # 2. Copy SID mapping to GR LETTER cache
  # ------------------------------------------------------------------
  local INDEX_FILE
  INDEX_FILE=$(python - <<PY
root_path = "${SAVE_MODEL_DIR}"
ckpt_id = root_path.replace("..", "").replace("/", "_")
print(f".index.{ckpt_id}.json")
PY
)

  should_skip "copy" || run_step "Copy SID mapping [${ALIAS}]" \
    cp "${GENSID_CACHE_DIR}/${DATASET}/${DATASET}${INDEX_FILE}" \
       "${GR_SID_MAPPING_PATH}"

  # ------------------------------------------------------------------
  # 3. Remap LETTER → GR .sem_ids
  # ------------------------------------------------------------------
  should_skip "remap" || run_step "Remap SID mapping [${ALIAS}]" bash -c "
    python map_gensid_to_genrec.py \
      --sid_mapping_path '${GR_SID_MAPPING_PATH}' \
      --gr_path '${GR_PATH}' \
      --dataset '${DATASET}' \
      --env_file '${MAP_ENV}'
  "

  [[ "${PIPELINE_FAILED}" -ne 0 ]] && return 0

  # shellcheck disable=SC1090
  source "${MAP_ENV}"
  : "${GR_SEM_IDS_PATH:?Missing GR_SEM_IDS_PATH in ${MAP_ENV}}"
  : "${GR_SEM_IDS_DEPTH:?Missing GR_SEM_IDS_DEPTH in ${MAP_ENV}}"
  : "${GR_SEM_IDS_CODEBOOK_SIZES:?Missing GR_SEM_IDS_CODEBOOK_SIZES in ${MAP_ENV}}"

  echo "Using fixed GR SID mapping: ${GR_SEM_IDS_PATH}"
  echo "SID depth: ${GR_SEM_IDS_DEPTH}"
  echo "Codebook sizes: ${GR_SEM_IDS_CODEBOOK_SIZES}"

  # ------------------------------------------------------------------
  # 4. Finetune MemGen-GR
  # ------------------------------------------------------------------
  local GR_DATASET_FLAGS
  GR_DATASET_FLAGS="$(get_gr_dataset_args "${DATASET}")"
  should_skip "finetune" || run_step "Finetune MemGen-GR [${ALIAS}]" bash -c "
    CUDA_VISIBLE_DEVICES=${DEVICE_NO} accelerate launch \
      --mixed_precision bf16 \
      --dynamo_backend inductor \
      --num_processes 1 main_genrec.py \
      --final_metrics_file='${FINAL_METRICS_FILE}' \
      --model=TIGER \
      ${GR_DATASET_FLAGS} \
      --max_item_seq_len=20 \
      --val_eval_user_limit=0.2 \
      --train_batch_size=512 \
      --eval_batch_size=32 \
      --num_proc=1 \
      --num_workers=8 \
      --epochs=5 \
      --rand_seed='${SEED}' \
      --sem_ids_path='${GR_SEM_IDS_PATH}' \
      --rq_n_codebooks='${GR_SEM_IDS_DEPTH}' \
      --rq_codebook_size='${GR_SEM_IDS_CODEBOOK_SIZES}' \
      --config=aux_configs/mmgrec.yaml
  "
}

# ============================================================
# Ablation grid: datasets × seeds × presets
# ============================================================

# Edit this array to select which presets to run
RQVAE_PRESETS=(
  rqvae_args_full
  rqvae_args_no_graph
  rqvae_args_no_edge
  rqvae_args_minimal
)

# ============================================================
# Parallelism control
# ============================================================

MAX_PARALLEL=3   # max concurrent runs on this device

# Semaphore via a bounded background job pool
_sem_wait() {
  while [[ "$(jobs -rp | wc -l)" -ge "${MAX_PARALLEL}" ]]; do
    sleep 2
  done
}

for SEED in "${SEEDS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRESET in "${RQVAE_PRESETS[@]}"; do
      [[ "${PIPELINE_FAILED}" -ne 0 ]] && break 3
      _sem_wait
      run_pipeline "${DATASET}" "${SEED}" "${PRESET}" &
    done
  done
done

# Wait for all background jobs to finish
wait

echo
if [[ "${PIPELINE_FAILED}" -ne 0 ]]; then
  echo "🎯 Pipeline stopped after first failure"
else
  echo "🎯 Pipeline finished successfully"
fi
