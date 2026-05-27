#!/usr/bin/env bash
set -euo pipefail

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

DATASET="Beauty"
# METHOD_NAME="MMGRec_qwen"
# EMB_MODEL="Qwen3-Embedding-0.6B"
METHOD_NAME="MMGRec_t5"
EMB_MODEL="sentence-t5-base"
SEEDS=(42 43 44)

GENSID_DIR="./gensid"

get_dataset_prefix() {
  local dataset="$1"
  case "${dataset}" in
    Musical_Instruments|Beauty|Toys_and_Games|Sports_and_Outdoors|Books) echo AmazonReviews2014/ ;;
    *) echo "" ;;
  esac
}

GENSID_CACHE_DIR="cache/gensid"
GR_CACHE_DIR="cache"
DATA_PATH="${GENSID_CACHE_DIR}/${DATASET}/${DATASET}.emb-${EMB_MODEL}-td.npy"
INTERACTIONS_PATH="${GENSID_CACHE_DIR}/${DATASET}/${DATASET}.inter.json"
GENSID_CKPT_PATH="ckpt/gensid"
CF_EMB="${GENSID_CKPT_PATH}/${DATASET}-32d-SASRec.pt"

GR_PATH="${GR_CACHE_DIR}/$(get_dataset_prefix "${DATASET}")${DATASET}/processed"

DEVICE_NO=0
DEVICE="cuda:${DEVICE_NO}"

mkdir -p "logs/env"

for SEED in "${SEEDS[@]}"; do
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && break

  # Alias identifies every artifact from this run.
  ALIAS="${METHOD_NAME}_${DATASET}_seed${SEED}"
  INDEX_FILE=".${ALIAS}.index.json"

  OUT_DIR="${GENSID_CACHE_DIR}/mincut_runs/${ALIAS}"
  LABELS_PATH="${OUT_DIR}/graph_hierarchy_labels.json"

  # Path is relative to GR_CODEBASE.
  GR_SID_MAPPING_PATH="${GENSID_CACHE_DIR}/${DATASET}.${ALIAS}.json"

  # Mapper writes shell variables here; parent script sources it later.
  MAP_ENV="logs/env/${ALIAS}_sid_mapping.env"

  LOG_FILE="logs/${ALIAS,,}.log"
  FINAL_METRICS_FILE="results/${ALIAS,,}.json"

  echo -e "\n########################################"
  echo "Dataset: ${DATASET}"
  echo "Method:  ${METHOD_NAME}"
  echo "Seed:    ${SEED}"
  echo "Alias:   ${ALIAS}"
  echo "########################################"

  # =========================
  # 1. Generate SID mapping with MMGRec-style GNN + RQ
  # =========================
  
  run_step "Generate MMGRec GNN SID mapping" \
  python gensid/mmgrec_gnn_sid_adapter.py \
    --interactions_path "${INTERACTIONS_PATH}" \
    --embeddings_path "${DATA_PATH}" \
    --output_dir "${GENSID_CACHE_DIR}/${DATASET}" \
    --output_name "${DATASET}${INDEX_FILE}" \
    --skip_n_last 2 \
    --device "${DEVICE}" \
    --num_workers 4 \
    --seed "${SEED}" \
    \
    --gnn_epochs 1000 \
    --gnn_batch_size 3000 \
    --gnn_hidden_dim 128 \
    --gnn_out_dim 64 \
    --gnn_dropout 0.5 \
    --gnn_lr 0.001 \
    --gnn_weight_decay 1e-5 \
    \
    --rq_epochs 2000 \
    --rq_batch_size 1024 \
    --rq_eval_batch_size 4096 \
    --rq_hidden_dim 32 \
    --rq_latent_dim 4 \
    --rq_lr 0.001 \
    --rq_weight_decay 1e-5 \
    --codebook_sizes 256 256 256 256 \
    --embedding_weight 1.0 \
    --commitment_weight 0.25 \
    \
    --append_collision_token \
    --collision_sort popularity

  # =========================
  # 2. Copy SID mapping to GR / LETTER cache
  # =========================

  run_step "Copy SID mapping to GR LETTER cache | ${ALIAS}" \
  cp "${GENSID_CACHE_DIR}/${DATASET}/${DATASET}${INDEX_FILE}" "${GR_SID_MAPPING_PATH}"

  # =========================
  # 3. Remap LETTER-style SID mapping to GR .sem_ids
  # =========================

  run_step "Remap SID mapping to GR format | ${ALIAS}" bash -c "
  python map_gensid_to_genrec.py \
    --sid_mapping_path '${GR_SID_MAPPING_PATH}' \
    --gr_path '${GR_PATH}' \
    --dataset '${DATASET}' \
    --env_file '${MAP_ENV}'
  "

  if [[ "${PIPELINE_FAILED}" -eq 0 ]]; then
    # Import mapper outputs into this shell.
    # shellcheck disable=SC1090
    source "${MAP_ENV}"

    # Fail early if mapper did not write required variables.
    : "${GR_SEM_IDS_PATH:?Missing GR_SEM_IDS_PATH in ${MAP_ENV}}"
    : "${GR_SEM_IDS_DEPTH:?Missing GR_SEM_IDS_DEPTH in ${MAP_ENV}}"
    : "${GR_SEM_IDS_CODEBOOK_SIZES:?Missing GR_SEM_IDS_CODEBOOK_SIZES in ${MAP_ENV}}"

    echo "Using fixed GR SID mapping: ${GR_SEM_IDS_PATH}"
    echo "SID depth: ${GR_SEM_IDS_DEPTH}"
    echo "Codebook sizes: ${GR_SEM_IDS_CODEBOOK_SIZES}"
  fi

  # =========================
  # 4. Finetune GR
  # =========================

  if [[ "${PIPELINE_FAILED}" -eq 0 ]]; then
    run_step "Finetune MemGen-GR with mapped SIDs | ${ALIAS}" bash -c "
    CUDA_VISIBLE_DEVICES=${DEVICE_NO} accelerate launch \
      --mixed_precision bf16 \
      --dynamo_backend inductor \
      --num_processes 1 main_genrec.py \
      --final_metrics_file='${FINAL_METRICS_FILE}'\
      --model=TIGER \
      --dataset=AmazonReviews2014 \
      --category='${DATASET}' \
      --max_item_seq_len=20 \
      --val_eval_user_limit=0.2 \
      --train_batch_size=512 \
      --eval_batch_size=32 \
      --seed='${SEED}' \
      --sem_ids_path='${GR_SEM_IDS_PATH}' \
      --rq_n_codebooks='${GR_SEM_IDS_DEPTH}' \
      --rq_codebook_size='${GR_SEM_IDS_CODEBOOK_SIZES}' \
      --config=aux_configs/mmgrec.yaml \
      2>&1 | tee '${LOG_FILE}'
    "
  fi
done

echo
if [[ "${PIPELINE_FAILED}" -ne 0 ]]; then
  echo "🎯 Pipeline stopped after first failure"
else
  echo "🎯 Pipeline finished successfully"
fi
