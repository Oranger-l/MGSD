#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"

usage() {
  cat <<'EOF'
Merge an EasyR1 VSP-Tasks OPCD checkpoint into Hugging Face format.

Usage:
  bash EasyR1/examples/merge_vsp_tasks_opcd_checkpoint.sh <global_step_or_actor_dir>

The script accepts either:
  - checkpoints/.../global_step_N
  - checkpoints/.../global_step_N/actor

Default output:
  <global_step_N>/actor/huggingface

Optional overrides:
  EXPORT_DIR=/path       Copy the merged Hugging Face checkpoint to this directory.
  PYTHON_BIN=python3     Python executable.
  HF_UPLOAD_PATH=repo    Optional Hugging Face repo id passed to model_merger.py.

Example:
  bash EasyR1/examples/merge_vsp_tasks_opcd_checkpoint.sh \
    EasyR1/checkpoints/VSP_Tasks_OPCD/exp/global_step_1000

  EXPORT_DIR=models/ckpts/Qwen3-VL-4B-VSP-Tasks-OPCD \
    bash EasyR1/examples/merge_vsp_tasks_opcd_checkpoint.sh \
    EasyR1/checkpoints/VSP_Tasks_OPCD/exp/global_step_1000
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CKPT_PATH="${1:-${CKPT_PATH:-}}"
if [[ -z "${CKPT_PATH}" ]]; then
  usage >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d "${CKPT_PATH}/actor" ]]; then
  ACTOR_DIR="${CKPT_PATH}/actor"
elif [[ "$(basename "${CKPT_PATH}")" == "actor" && -d "${CKPT_PATH}" ]]; then
  ACTOR_DIR="${CKPT_PATH}"
else
  echo "Checkpoint path must be a global_step directory with actor/ or an actor directory: ${CKPT_PATH}" >&2
  exit 1
fi

if ! compgen -G "${ACTOR_DIR}/model_world_size_*_rank_0.pt" >/dev/null; then
  echo "No EasyR1 model shard found under: ${ACTOR_DIR}" >&2
  exit 1
fi

ACTOR_DIR=$(realpath "${ACTOR_DIR}")
if [[ -n "${EXPORT_DIR:-}" ]]; then
  EXPORT_DIR=$(realpath -m "${EXPORT_DIR}")
fi

cd "${PROJECT_ROOT}"
merge_cmd=("${PYTHON_BIN}" scripts/model_merger.py --local_dir "${ACTOR_DIR}")
if [[ -n "${HF_UPLOAD_PATH:-}" ]]; then
  merge_cmd+=(--hf_upload_path "${HF_UPLOAD_PATH}")
fi
"${merge_cmd[@]}"

MERGED_DIR="${ACTOR_DIR}/huggingface"
if [[ -n "${EXPORT_DIR:-}" ]]; then
  mkdir -p "$(dirname "${EXPORT_DIR}")"
  rm -rf "${EXPORT_DIR}"
  cp -a "${MERGED_DIR}" "${EXPORT_DIR}"
  echo "Merged OPCD checkpoint copied to: ${EXPORT_DIR}"
else
  echo "Merged OPCD checkpoint saved to: ${MERGED_DIR}"
fi
