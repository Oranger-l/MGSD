#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

API_CONFIG="${API_CONFIG:-${REPO_ROOT}/api_config_files/api_config_vllm.json}"
DATASET_JSON="${DATASET_JSON:-${REPO_ROOT}/LlamaFactory/data/vsp_tasks_perception_sft.json}"
MODEL_NAME="${MODEL_NAME:-qwen3vl-vsp-perception}"
TASKS="${TASKS:-all}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/vsp_perception_final_all_tasks.jsonl}"
SAMPLES_PER_LEVEL="${SAMPLES_PER_LEVEL:-10}"
WORKERS="${WORKERS:-64}"
REPLICAS="${REPLICAS:-8}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-localhost}"
API_KEY="${API_KEY:-}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TIMEOUT="${TIMEOUT:-120}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
PRINT_JSON="${PRINT_JSON:-0}"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

read -r -a TASK_ARGS <<< "${TASKS}"

EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ "${RESUME}" == "1" ]]; then
    EXTRA_ARGS+=(--resume)
fi
if [[ "${PRINT_JSON}" == "1" ]]; then
    EXTRA_ARGS+=(--print-json)
fi

python "${SCRIPT_DIR}/query_vsp_perception.py" \
    --api-config "${API_CONFIG}" \
    --dataset-json "${DATASET_JSON}" \
    --model "${MODEL_NAME}" \
    --tasks "${TASK_ARGS[@]}" \
    --replicas "${REPLICAS}" \
    --base-port "${BASE_PORT}" \
    --host "${HOST}" \
    --api-key "${API_KEY}" \
    --workers "${WORKERS}" \
    --samples-per-level "${SAMPLES_PER_LEVEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --timeout "${TIMEOUT}" \
    --output "${OUTPUT}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
