#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

API_KEY_FILE="${API_KEY_FILE:-${REPO_ROOT}/api_config_files/api_config_openai.json}"
MODELS="${MODELS:-gpt-4o gpt-5 gemini-2.5-flash gemini-2.5-pro gemini-3-flash-preview}"
TASKS="${TASKS:-frozenlake maze minibehaviour}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/VisualPlanning/dataset}"
FROZENLAKE_BENCH_ROOT="${FROZENLAKE_BENCH_ROOT:-${REPO_ROOT}/data/DiffThinker/FrozenLake/VSP/maps}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
IMAGE_CACHE="${IMAGE_CACHE:-${SCRIPT_DIR}/rendered_images}"
PROMPT_DIR="${PROMPT_DIR:-${SCRIPT_DIR}/prompts}"
FROZENLAKE_PROMPT="${FROZENLAKE_PROMPT:-${PROMPT_DIR}/opcd_student_Frozenlake_direct.txt}"

# Closed-source APIs are expensive. Default to a smoke-sized selection.
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1}"
# API eval is network-bound. Increase these if the gateway rate limit allows it.
WORKERS="${WORKERS:-32}"
MODEL_WORKERS="${MODEL_WORKERS:-3}"
MAX_TOKENS="${MAX_TOKENS:-10240}"
TOKEN_FIELD="${TOKEN_FIELD:-max_tokens}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
TIMEOUT="${TIMEOUT:-180}"
RETRIES="${RETRIES:-2}"
RETRY_SLEEP="${RETRY_SLEEP:-5}"
ALLOW_UNLISTED="${ALLOW_UNLISTED:-0}"
NO_FETCH_MODEL_LIST="${NO_FETCH_MODEL_LIST:-0}"
DRY_RUN="${DRY_RUN:-0}"
RENDER_ONLY="${RENDER_ONLY:-0}"
RESUME="${RESUME:-1}"
PRINT_JSON="${PRINT_JSON:-0}"

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

read -r -a MODEL_ARGS <<< "${MODELS}"
read -r -a TASK_ARGS <<< "${TASKS}"

EXTRA_ARGS=()
if [[ "${ALLOW_UNLISTED}" == "1" ]]; then
    EXTRA_ARGS+=(--allow-unlisted)
fi
if [[ "${NO_FETCH_MODEL_LIST}" == "1" ]]; then
    EXTRA_ARGS+=(--no-fetch-model-list)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    EXTRA_ARGS+=(--dry-run)
fi
if [[ "${RENDER_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--render-only)
fi
if [[ "${RESUME}" == "1" ]]; then
    EXTRA_ARGS+=(--resume)
else
    EXTRA_ARGS+=(--no-resume)
fi
if [[ "${PRINT_JSON}" == "1" ]]; then
    EXTRA_ARGS+=(--print-json)
fi
if [[ -n "${TOP_P}" ]]; then
    EXTRA_ARGS+=(--top-p "${TOP_P}")
fi

python "${SCRIPT_DIR}/evaluate_api_models.py" \
    --api-key-file "${API_KEY_FILE}" \
    --models "${MODEL_ARGS[@]}" \
    --tasks "${TASK_ARGS[@]}" \
    --dataset-root "${DATASET_ROOT}" \
    --frozenlake-bench-root "${FROZENLAKE_BENCH_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-cache "${IMAGE_CACHE}" \
    --prompt-dir "${PROMPT_DIR}" \
    --frozenlake-prompt "${FROZENLAKE_PROMPT}" \
    --samples-per-task "${SAMPLES_PER_TASK}" \
    --workers "${WORKERS}" \
    --model-workers "${MODEL_WORKERS}" \
    --max-tokens "${MAX_TOKENS}" \
    --token-field "${TOKEN_FIELD}" \
    --temperature "${TEMPERATURE}" \
    --timeout "${TIMEOUT}" \
    --retries "${RETRIES}" \
    --retry-sleep "${RETRY_SLEEP}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
