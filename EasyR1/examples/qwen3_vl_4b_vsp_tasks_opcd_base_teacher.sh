#!/bin/bash

export WANDB_MODE=offline
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"

set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${PROJECT_ROOT}/.." && pwd)

# Student starts from the final perception-SFT model; Teacher/ref starts from the base model.
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/ckpts/Qwen3-VL-4B-VSP-Tasks-Perception-SFT-Final}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${REPO_ROOT}/models/Qwen3-VL-4B-Instruct}

DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data/VSP-Tasks/vsp_tasks_opcd_json/mixed/train}
VAL_DIR=${VAL_DIR:-${REPO_ROOT}/data/VSP-Tasks/vsp_tasks_opcd_json/mixed/val}
TEACHER_PROMPT=${TEACHER_PROMPT:-./examples/system_prompt/opcd_teacher_VSPTasks.txt}
TEACHER_USE_GT=${TEACHER_USE_GT:-1}
REWARD_FN=${REWARD_FN:-./examples/reward_function/vsp_tasks.py:compute_score}
IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-65536}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-262144}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}

# swanlab
PROJECT_NAME=${PROJECT_NAME:-VSP_Tasks_OPCD}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3vl4b_vsp_tasks_mixed_opcd_base_teacher_sft_student_final}
DATA_SEED=${DATA_SEED:-1}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Student model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [ ! -d "${TEACHER_MODEL_PATH}" ]; then
    echo "Teacher model directory not found: ${TEACHER_MODEL_PATH}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${VAL_DIR}/test.parquet \
    data.format_prompt=null \
    data.system_prompt=null \
    data.return_raw_chat=true \
    data.max_prompt_length=5120 \
    data.max_response_length=2048 \
    data.rollout_batch_size=32 \
    data.val_batch_size=100 \
    data.seed=${DATA_SEED} \
    data.min_pixels=${IMAGE_MIN_PIXELS} \
    data.max_pixels=${IMAGE_MAX_PIXELS} \
    algorithm.use_mm_context_distillation=true \
    algorithm.only_reverse_kl_advantages=true \
    algorithm.teacher_system_prompt=${TEACHER_PROMPT} \
    algorithm.teacher_append_ground_truth=${TEACHER_USE_GT} \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.ref.model.model_path=${TEACHER_MODEL_PATH} \
    worker.ref.model.tokenizer_path=${MODEL_PATH} \
    worker.ref.model.trust_remote_code=false \
    worker.ref.model.enable_gradient_checkpointing=false \
    worker.ref.fsdp.torch_dtype=bf16 \
    worker.ref.sync_with_actor=false \
    worker.actor.global_batch_size=32 \
    worker.rollout.n=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.90 \
    worker.reward.reward_function=${REWARD_FN} \
    worker.val_reward.reward_function=${REWARD_FN} \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.micro_batch_size_per_device_for_update=4 \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.model.enable_gradient_checkpointing=false \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.save_freq=20 \
    trainer.val_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.find_last_checkpoint=false \
    trainer.logger='["file","swanlab"]' \
    trainer.n_gpus_per_node=8
