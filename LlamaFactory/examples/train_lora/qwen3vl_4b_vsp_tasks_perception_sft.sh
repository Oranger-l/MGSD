#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${LLAMAFACTORY_ROOT}/.." && pwd)}"

# =========================
# Common Config: Hardware / Distributed
# =========================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"  # empty means auto-count CUDA_VISIBLE_DEVICES
SINGLE_NODE="${SINGLE_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"

# =========================
# Common Config: Paths / Train Config
# =========================
PYTHON_ENV="${PYTHON_ENV:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
CONFIG_PATH="${CONFIG_PATH:-examples/train_lora/qwen3vl_4b_vsp_tasks_perception_sft.yaml}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${REPO_ROOT}/models/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${LLAMAFACTORY_ROOT}/saves/qwen3-vl-4b/vsp_tasks_perception_lora_sft}"
DATASET_DIR="${DATASET_DIR:-${LLAMAFACTORY_ROOT}/data}"
MEDIA_DIR="${MEDIA_DIR:-${REPO_ROOT}}"

# =========================
# Common Config: Logging
# =========================
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-VSP-Tasks}"
WANDB_NAME="${WANDB_NAME:-qwen3vl_vsp_tasks_perception_sft}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-VSP-Tasks}"
SWANLAB_RUN_NAME="${SWANLAB_RUN_NAME:-qwen3vl4b_vsp_tasks_perception_sft_final}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-${LLAMAFACTORY_ROOT}/saves/swanlab}"
ENABLE_SWANLAB="${ENABLE_SWANLAB:-auto}"

# =========================
# Common Config: Runtime
# =========================
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export FORCE_TORCHRUN CUDA_VISIBLE_DEVICES
IFS=',' read -ra VISIBLE_GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
if [[ -z "$NPROC_PER_NODE" ]]; then
  NPROC_PER_NODE="${#VISIBLE_GPU_LIST[@]}"
fi
export NPROC_PER_NODE NNODES NODE_RANK MASTER_ADDR MASTER_PORT

if [[ "$SINGLE_NODE" == "1" ]]; then
  export NNODES=1
  export NODE_RANK=0
  unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
  unset TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
fi

export TOKENIZERS_PARALLELISM
export WANDB_MODE WANDB_PROJECT WANDB_NAME
export SWANLAB_API_KEY SWANLAB_PROJECT SWANLAB_RUN_NAME SWANLAB_MODE SWANLAB_LOGDIR
export MODEL_NAME_OR_PATH DATASET_DIR MEDIA_DIR OUTPUT_DIR
if [[ -n "$PYTHON_ENV" ]]; then
  PYTHON_BIN="$PYTHON_ENV/bin/python"
  LLAMAFACTORY_CLI="$PYTHON_ENV/bin/llamafactory-cli"
  export PATH="$PYTHON_ENV/bin:$PATH"
fi
export PYTHONPATH="$LLAMAFACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$LLAMAFACTORY_ROOT"

RUN_CONFIG_PATH="$(mktemp /tmp/qwen3vl4b_vsp_tasks_perception_sft.XXXXXX.yaml)"
ENABLE_SWANLAB_RUNTIME=false
if [[ "$ENABLE_SWANLAB" != "false" ]]; then
  if "$PYTHON_BIN" -c "import swanlab" >/dev/null 2>&1; then
    ENABLE_SWANLAB_RUNTIME=true
  else
    echo "SwanLab requested but package 'swanlab' is not installed; running with local/file logs only."
  fi
fi

"$PYTHON_BIN" - "$CONFIG_PATH" "$RUN_CONFIG_PATH" "$ENABLE_SWANLAB_RUNTIME" <<'PY'
import os
import sys
import yaml

src, dst, enable_swanlab = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true"
with open(src, encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["model_name_or_path"] = os.environ["MODEL_NAME_OR_PATH"]
config["dataset_dir"] = os.environ["DATASET_DIR"]
config["media_dir"] = os.environ["MEDIA_DIR"]
config["output_dir"] = os.environ["OUTPUT_DIR"]

if enable_swanlab:
    config["use_swanlab"] = True
    config["swanlab_project"] = os.environ["SWANLAB_PROJECT"]
    config["swanlab_run_name"] = os.environ["SWANLAB_RUN_NAME"]
    config["swanlab_mode"] = os.environ["SWANLAB_MODE"]
    if os.environ.get("SWANLAB_API_KEY"):
        config["swanlab_api_key"] = os.environ["SWANLAB_API_KEY"]
    config["swanlab_logdir"] = os.environ["SWANLAB_LOGDIR"]

with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
PY
if [[ "$ENABLE_SWANLAB_RUNTIME" == "true" ]]; then
    echo "SwanLab enabled: project=${SWANLAB_PROJECT:-VSP-Tasks}, run=${SWANLAB_RUN_NAME:-qwen3vl_vsp_tasks_perception_sft}"
fi

"$LLAMAFACTORY_CLI" train "$RUN_CONFIG_PATH"
