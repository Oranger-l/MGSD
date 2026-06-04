#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/ckpts/Qwen3-VL-4B-VSP-Tasks-OPCD-Mixed}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-vsp-opcd}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
# vLLM max_model_len is prompt tokens + generated tokens.
# Default budget: prompt <= 8192, generation <= 4096.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":10}}"
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"min_pixels\":65536,\"max_pixels\":262144}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

EXTRA_ARGS=(
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    EXTRA_ARGS+=(--enable-prefix-caching)
fi
if [[ -n "${API_KEY}" ]]; then
    EXTRA_ARGS+=(--api-key "${API_KEY}")
fi

echo "[serve] model: ${MODEL_PATH}"
echo "[serve] name: ${SERVED_MODEL_NAME}"
echo "[serve] url: http://${HOST}:${PORT}/v1"
echo "[serve] max_model_len: ${MAX_MODEL_LEN}"

vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
    --mm-processor-kwargs "${MM_PROCESSOR_KWARGS}" \
    "${EXTRA_ARGS[@]}"
