#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

API_CONFIG="${API_CONFIG:-${REPO_ROOT}/api_config_files/api_config_vllm.json}"
MODEL_NAME="${MODEL_NAME:-frozenlake-opcd-level5-step120}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/VisualPlanning/dataset}"
FROZENLAKE_BENCH_ROOT="${FROZENLAKE_BENCH_ROOT:-${REPO_ROOT}/data/DiffThinker/FrozenLake/VSP/maps}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/results/visualplanning_ckpt_eval.jsonl}"
IMAGE_CACHE="${IMAGE_CACHE:-${SCRIPT_DIR}/rendered_images}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-0}"
WORKERS="${WORKERS:-128}"
# OpenAI API `max_tokens`, equivalent to HuggingFace/vLLM `max_new_tokens`.
MAX_TOKENS="${MAX_TOKENS:-4096}"
# vLLM request-side prompt truncation budget. Keep MAX_PROMPT_TOKENS + MAX_TOKENS <= server MAX_MODEL_LEN.
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-}"
MIN_P="${MIN_P:-}"
REPETITION_PENALTY="${REPETITION_PENALTY:-}"
SEED="${SEED:-}"
API_KEY="${API_KEY:-}"
BASE_URLS="${BASE_URLS:-http://localhost:8000/v1 http://localhost:8001/v1 http://localhost:8002/v1 http://localhost:8003/v1 http://localhost:8004/v1 http://localhost:8005/v1 http://localhost:8006/v1 http://localhost:8007/v1}"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

read -r -a BASE_URL_ARGS <<< "${BASE_URLS}"

EXTRA_GENERATION_ARGS=()
if [[ -n "${TOP_K}" ]]; then
    EXTRA_GENERATION_ARGS+=(--top-k "${TOP_K}")
fi
if [[ -n "${MIN_P}" ]]; then
    EXTRA_GENERATION_ARGS+=(--min-p "${MIN_P}")
fi
if [[ -n "${REPETITION_PENALTY}" ]]; then
    EXTRA_GENERATION_ARGS+=(--repetition-penalty "${REPETITION_PENALTY}")
fi
if [[ -n "${SEED}" ]]; then
    EXTRA_GENERATION_ARGS+=(--seed "${SEED}")
fi

python "${SCRIPT_DIR}/evaluate_visualplanning_ckpt.py" \
    --api-config "${API_CONFIG}" \
    --base-urls "${BASE_URL_ARGS[@]}" \
    --api-key "${API_KEY}" \
    --model "${MODEL_NAME}" \
    --dataset-root "${DATASET_ROOT}" \
    --frozenlake-bench-root "${FROZENLAKE_BENCH_ROOT}" \
    --tasks frozenlake maze minibehaviour \
    --samples-per-task "${SAMPLES_PER_TASK}" \
    --output "${OUTPUT}" \
    --image-cache "${IMAGE_CACHE}" \
    --workers "${WORKERS}" \
    --max-tokens "${MAX_TOKENS}" \
    --truncate-prompt-tokens "${MAX_PROMPT_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    "${EXTRA_GENERATION_ARGS[@]}" \
    "$@"
