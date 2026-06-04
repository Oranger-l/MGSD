#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${LLAMAFACTORY_ROOT}/.." && pwd)}"

usage() {
  cat <<'EOF'
Merge a VSP-Tasks perception-SFT LoRA checkpoint into its Qwen3-VL base model.

Usage:
  bash LlamaFactory/examples/merge_lora/merge_vsp_tasks_perception_sft.sh <adapter_checkpoint>

Common overrides:
  MODEL_SIZE=4b|8b       Infer base/export defaults. Auto-detected from path when unset.
  BASE_MODEL=/path       Base Qwen3-VL model path.
  EXPORT_DIR=/path       Output merged HF checkpoint directory.
  PYTHON_ENV=/path       Optional conda/env prefix containing llamafactory-cli.
  LLAMAFACTORY_CLI=cmd   LLaMA Factory CLI command. Default: llamafactory-cli.
  EXPORT_DEVICE=cpu      Export device. Default: cpu.
  EXPORT_SIZE=5          Shard size for export. Default: 5.

Examples:
  bash LlamaFactory/examples/merge_lora/merge_vsp_tasks_perception_sft.sh \
    LlamaFactory/saves/qwen3-vl-4b/vsp_tasks_perception_lora_sft/checkpoint-828

  MODEL_SIZE=8b EXPORT_DIR=models/ckpts/Qwen3-VL-8B-VSP-Tasks-Perception-SFT \
    bash LlamaFactory/examples/merge_lora/merge_vsp_tasks_perception_sft.sh \
    LlamaFactory/saves/qwen3-vl-8b/vsp_tasks_perception_lora_sft/checkpoint-828
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ADAPTER_PATH="${1:-${ADAPTER_PATH:-}}"
if [[ -z "${ADAPTER_PATH}" ]]; then
  usage >&2
  exit 1
fi

if [[ -n "${PYTHON_ENV:-}" ]]; then
  export PATH="${PYTHON_ENV}/bin:${PATH}"
fi
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"

case "${MODEL_SIZE:-}" in
  4b|4B) MODEL_SIZE="4b" ;;
  8b|8B) MODEL_SIZE="8b" ;;
  "")
    adapter_lower=$(echo "${ADAPTER_PATH}" | tr '[:upper:]' '[:lower:]')
    if [[ "${adapter_lower}" == *"8b"* ]]; then
      MODEL_SIZE="8b"
    else
      MODEL_SIZE="4b"
    fi
    ;;
  *)
    echo "MODEL_SIZE must be 4b or 8b, got: ${MODEL_SIZE}" >&2
    exit 1
    ;;
esac

if [[ "${MODEL_SIZE}" == "8b" ]]; then
  DEFAULT_BASE="${REPO_ROOT}/models/Qwen3-VL-8B-Instruct"
  DEFAULT_EXPORT="${REPO_ROOT}/models/ckpts/Qwen3-VL-8B-VSP-Tasks-Perception-SFT"
else
  DEFAULT_BASE="${REPO_ROOT}/models/Qwen3-VL-4B-Instruct"
  DEFAULT_EXPORT="${REPO_ROOT}/models/ckpts/Qwen3-VL-4B-VSP-Tasks-Perception-SFT"
fi

BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE}}"
EXPORT_DIR="${EXPORT_DIR:-${DEFAULT_EXPORT}}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
EXPORT_SIZE="${EXPORT_SIZE:-5}"

if [[ ! -d "${ADAPTER_PATH}" ]]; then
  echo "Adapter checkpoint directory not found: ${ADAPTER_PATH}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "Base model directory not found: ${BASE_MODEL}" >&2
  exit 1
fi

ADAPTER_PATH=$(realpath "${ADAPTER_PATH}")
BASE_MODEL=$(realpath "${BASE_MODEL}")
EXPORT_DIR=$(realpath -m "${EXPORT_DIR}")

mkdir -p "$(dirname "${EXPORT_DIR}")"
RUN_CONFIG_PATH="$(mktemp /tmp/vsp_tasks_sft_merge.XXXXXX.yaml)"
cat > "${RUN_CONFIG_PATH}" <<EOF
model_name_or_path: ${BASE_MODEL}
adapter_name_or_path: ${ADAPTER_PATH}
template: qwen3_vl_nothink
trust_remote_code: true
infer_dtype: bfloat16
export_dir: ${EXPORT_DIR}
export_size: ${EXPORT_SIZE}
export_device: ${EXPORT_DEVICE}
export_legacy_format: false
EOF

cd "${LLAMAFACTORY_ROOT}"
"${LLAMAFACTORY_CLI}" export "${RUN_CONFIG_PATH}"
echo "Merged SFT checkpoint saved to: ${EXPORT_DIR}"
