#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_KIND="${MODEL_KIND:-base}"
REPLICAS="${REPLICAS:-8}"
BASE_PORT="${BASE_PORT:-8000}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3 4 5 6 7}"
API_KEY="${API_KEY:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"

case "${MODEL_KIND}" in
    base)
        DEFAULT_MODEL_PATH="${REPO_ROOT}/models/Qwen3-VL-8B-Instruct"
        DEFAULT_SERVED_MODEL_NAME="qwen3vl8b-base"
        DEFAULT_LOG_LABEL="base8b"
        ;;
    opcd)
        DEFAULT_MODEL_PATH="${REPO_ROOT}/models/ckpts/Qwen3-VL-4B-VSP-Tasks-OPCD-Mixed"
        DEFAULT_SERVED_MODEL_NAME="qwen3vl4b-opcd-v2-step1400"
        DEFAULT_LOG_LABEL="opcd"
        ;;
    *)
        echo "Unsupported MODEL_KIND=${MODEL_KIND}. Use base or opcd." >&2
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${DEFAULT_SERVED_MODEL_NAME}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${DEFAULT_LOG_LABEL}}"

mkdir -p "${LOG_DIR}"

read -r -a DEVICE_ARRAY <<< "${CUDA_DEVICES}"
if (( ${#DEVICE_ARRAY[@]} < REPLICAS )); then
    echo "[serve-8x] CUDA_DEVICES has fewer entries than REPLICAS=${REPLICAS}" >&2
    exit 1
fi

pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM EXIT

for ((i = 0; i < REPLICAS; i++)); do
    device="${DEVICE_ARRAY[$i]}"
    port="$((BASE_PORT + i))"
    log_file="${LOG_DIR}/vllm_replica_${i}_port_${port}.log"
    echo "[serve-8x] kind=${MODEL_KIND} replica=${i} cuda=${device} port=${port} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${device}" \
    MODEL_KIND="${MODEL_KIND}" \
    MODEL_PATH="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
    PORT="${port}" \
    API_KEY="${API_KEY}" \
    MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
        bash "${SCRIPT_DIR}/serve_model.sh" >"${log_file}" 2>&1 &
    pids+=("$!")
    sleep 2
done

echo "[serve-8x] launched ${#pids[@]} ${MODEL_KIND} replicas on ports ${BASE_PORT}..$((BASE_PORT + REPLICAS - 1))"
echo "[serve-8x] served model name: ${SERVED_MODEL_NAME}"
echo "[serve-8x] press Ctrl-C to stop all replicas"
wait
