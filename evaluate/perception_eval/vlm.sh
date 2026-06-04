#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# One vLLM replica per GPU gives better throughput for many independent
# one-image perception requests than a single tensor-parallel server.
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/ckpts/Qwen3-VL-8B-VSP-Tasks-Perception-SFT-Final}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3vl-vsp-perception}"
VLLM_BIN="${VLLM_BIN:-vllm}"
REPLICAS="${REPLICAS:-8}"
BASE_PORT="${BASE_PORT:-8000}"
CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3 4 5 6 7}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":1}}"
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"min_pixels\":65536,\"max_pixels\":262144}}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

mkdir -p "${LOG_DIR}"

read -r -a DEVICE_ARRAY <<< "${CUDA_DEVICES}"
if (( ${#DEVICE_ARRAY[@]} < REPLICAS )); then
  echo "[vsp-perception-serve] CUDA_DEVICES has fewer entries than REPLICAS=${REPLICAS}" >&2
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
  log_file="${LOG_DIR}/vsp_perception_vllm_replica_${i}_port_${port}.log"
  echo "[vsp-perception-serve] replica=${i} cuda=${device} port=${port} log=${log_file}"

  extra_args=(
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${port}"
    --api-key "${API_KEY}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --trust-remote-code
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
    --mm-processor-kwargs "${MM_PROCESSOR_KWARGS}"
    --tensor-parallel-size 1
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  )

  if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
    extra_args+=(--enable-prefix-caching)
  fi
  if [[ -n "${API_KEY}" ]]; then
    extra_args+=(--api-key "${API_KEY}")
  fi

  CUDA_VISIBLE_DEVICES="${device}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" "${extra_args[@]}" >"${log_file}" 2>&1 &
  pids+=("$!")
  sleep 2
done

echo "[vsp-perception-serve] launched ${#pids[@]} replicas on ports ${BASE_PORT}..$((BASE_PORT + REPLICAS - 1))"
echo "[vsp-perception-serve] model=${MODEL_PATH}"
echo "[vsp-perception-serve] served_model=${SERVED_MODEL_NAME}"
echo "[vsp-perception-serve] vllm=${VLLM_BIN}"
echo "[vsp-perception-serve] mm_processor_kwargs=${MM_PROCESSOR_KWARGS}"
echo "[vsp-perception-serve] press Ctrl-C to stop all replicas"
wait
