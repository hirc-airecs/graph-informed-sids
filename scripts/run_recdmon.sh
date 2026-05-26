#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Pipeline scheme
# ============================================================
#
# For each seed:
#
#   1) RQ-VAE/mmgrec_gnn_sid_adapter.py
#        interactions + embeddings
#              ↓
#        LETTER-style SID mapping:
#        sid_out/.../gnn_sid_mapping.json
#
#   2) Copy SID mapping into GR/LETTER_cache
#              ↓
#        GR/LETTER_cache/<dataset>.<alias>.json
#
#   3) GR/gr_letter_mapper.py
#        LETTER-style SID mapping
#              ↓
#        GR-style .sem_ids mapping
#              ↓
#        writes tiny env file with:
#          GR_SEM_IDS_PATH
#          GR_SEM_IDS_DEPTH
#          GR_SEM_IDS_CODEBOOK_SIZES
#
#   4) GR/main.py
#        trains MemGen-GR / TIGER using mapped .sem_ids
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

DATASET="Beauty"
# METHOD_NAME="DMON_qwen"
# EMB_MODEL="Qwen3-Embedding-0.6B"
METHOD_NAME="RecDMoN_t5"
EMB_MODEL="sentence-t5-base"
SEEDS=(42)

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

DEVICE_NO=1
DEVICE="cuda:${DEVICE_NO}"

mkdir -p "logs/env"

for SEED in "${SEEDS[@]}"; do
  [[ "${PIPELINE_FAILED}" -ne 0 ]] && break

  # Alias identifies every artifact from this run.
  ALIAS="${METHOD_NAME}_${DATASET}_seed${SEED}"
  INDEX_FILE=".${ALIAS}.index.json"

  OUT_DIR="${GENSID_CACHE_DIR}/mincut_runs/${ALIAS}"
  LABELS_PATH="${OUT_DIR}/graph_hierarchy_labels.json"

  SID_OUTPUT_DIR="${GENSID_CACHE_DIR}/sid_out/${ALIAS}/${DATASET}"
  SID_MAPPING_PATH="${SID_OUTPUT_DIR}/gnn_sid_mapping.json"

  # Path is relative to GR_CODEBASE.
  GR_SID_MAPPING_PATH="${GENSID_CACHE_DIR}/${DATASET}.${ALIAS}.json"

  # Mapper writes shell variables here; parent script sources it later.
  MAP_ENV="logs/env/${ALIAS}_sid_mapping.env"

  LOG_FILE="logs/${ALIAS,,}.log"
  FINAL_METRICS_FILE="results/${ALIAS,,}.json"

  mkdir -p "${SID_OUTPUT_DIR}"

  echo -e "\n########################################"
  echo "Dataset: ${DATASET}"
  echo "Method:  ${METHOD_NAME}"
  echo "Seed:    ${SEED}"
  echo "Alias:   ${ALIAS}"
  echo "########################################"

# =========================
# STEP 1.1: GRAPH CLUSTERING
# =========================
run_step "Graph clustering (MinCut)" \
  python "${GENSID_DIR}/hierarchical_mincut_clustering.py" \
    --interactions_path "${INTERACTIONS_PATH}" \
    --embeddings_path "${DATA_PATH}" \
    --method recursive_dmon \
    --graph_type adjacent_cooc \
    --recursive_factor 6 \
    --recursive_levels 4 \
    --lr 3e-4 \
    --assign_hidden_dim 256 \
    --dropout 0.0 \
    --recursive_epochs 15 \
    --device "${DEVICE}" \
    --output_dir "${OUT_DIR}" \
    --output_prefix graph_hierarchy \
    --max_dense_nodes 100000

# =========================
# STEP 1.2: INDEX GENERATION
# =========================
run_step "Generate graph-cluster SID index" \
  python "${GENSID_DIR}/generate_graph_cluster_indices.py" \
    --labels_path "${LABELS_PATH}" \
    --output_file "${GENSID_CACHE_DIR}/${DATASET}/${DATASET}${INDEX_FILE}" \
    --label_group coarse_to_fine_item_labels \
    --levels level_1 level_2 level_3 level_4 \
    --unclustered_strategy special \
    --append_unique_leaf

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
      --epochs=5 \
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
