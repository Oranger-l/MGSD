#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_KIND="${MODEL_KIND:-base}"
case "${MODEL_KIND}" in
    base)
        DEFAULT_MODEL_PATH="${REPO_ROOT}/models/Qwen3-VL-8B-Instruct"
        DEFAULT_SERVED_MODEL_NAME="qwen3vl8b-base"
        ;;
    opcd)
        DEFAULT_MODEL_PATH="${REPO_ROOT}/models/ckpts/Qwen3-VL-4B-VSP-Tasks-OPCD-Mixed"
        DEFAULT_SERVED_MODEL_NAME="qwen3vl4b-opcd-v2-step1400"
        ;;
    *)
        echo "Unsupported MODEL_KIND=${MODEL_KIND}. Use base or opcd." >&2
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${DEFAULT_SERVED_MODEL_NAME}}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
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

echo "[serve] kind: ${MODEL_KIND}"
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
