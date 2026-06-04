#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd "${PROJECT_ROOT}/.." && pwd)

# Raw data roots
DATA_ROOT=${DATA_ROOT:-${REPO_ROOT}/data/VSP-Tasks}
VAL_RAW_ROOT=${VAL_RAW_ROOT:-${REPO_ROOT}/data/VSP-Tasks-OPCD-Val}

# Output roots
JSON_ROOT=${JSON_ROOT:-${DATA_ROOT}/vsp_tasks_opcd_json/mixed}
TRAIN_JSON=${TRAIN_JSON:-${JSON_ROOT}/train_normalized.json}
VAL_JSON=${VAL_JSON:-${JSON_ROOT}/val_normalized.json}
TRAIN_SAVE_DIR=${TRAIN_SAVE_DIR:-${JSON_ROOT}/train}
VAL_SAVE_DIR=${VAL_SAVE_DIR:-${JSON_ROOT}/val}

# Levels
FROZENLAKE_TRAIN_LEVELS=${FROZENLAKE_TRAIN_LEVELS:-3,4,5,6,7,8}
MAZE_TRAIN_LEVELS=${MAZE_TRAIN_LEVELS:-3,4,5,6}
MINIBEHAVIOUR_TRAIN_LEVELS=${MINIBEHAVIOUR_TRAIN_LEVELS:-5,6}
FROZENLAKE_VAL_LEVELS=${FROZENLAKE_VAL_LEVELS:-3,4,5,6,7,8}
MAZE_VAL_LEVELS=${MAZE_VAL_LEVELS:-3,4,5,6}
MINIBEHAVIOUR_VAL_LEVELS=${MINIBEHAVIOUR_VAL_LEVELS:-5,6}

# Conversion controls
SEED=${SEED:-42}
LIMIT_PER_LEVEL=${LIMIT_PER_LEVEL:-}
SAMPLE_PER_LEVEL=${SAMPLE_PER_LEVEL:-}
TEST_SAMPLE_PER_LEVEL=${TEST_SAMPLE_PER_LEVEL:-}
OPCD_DUPLICATE_TO=${OPCD_DUPLICATE_TO:-0}
TEACHER_USE_GT=${TEACHER_USE_GT:-1}

if [ ! -d "${DATA_ROOT}" ]; then
    echo "Missing training raw data directory: ${DATA_ROOT}" >&2
    exit 1
fi
if [ ! -d "${VAL_RAW_ROOT}" ]; then
    echo "Missing validation raw data directory: ${VAL_RAW_ROOT}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

normalize_cmd=(
    python examples/data_preprocess/vsp_tasks_mixed_to_opcd_json.py
    --data_root "${DATA_ROOT}"
    --test_data_root "${VAL_RAW_ROOT}"
    --train_save_path "${TRAIN_JSON}"
    --test_save_path "${VAL_JSON}"
    --frozenlake_train_levels "${FROZENLAKE_TRAIN_LEVELS}"
    --maze_train_levels "${MAZE_TRAIN_LEVELS}"
    --minibehaviour_train_levels "${MINIBEHAVIOUR_TRAIN_LEVELS}"
    --frozenlake_test_levels "${FROZENLAKE_VAL_LEVELS}"
    --maze_test_levels "${MAZE_VAL_LEVELS}"
    --minibehaviour_test_levels "${MINIBEHAVIOUR_VAL_LEVELS}"
    --seed "${SEED}"
    --teacher_include_ground_truth "${TEACHER_USE_GT}"
    --path_root "${REPO_ROOT}"
    --shuffle_train
)

if [ -n "${LIMIT_PER_LEVEL}" ]; then
    normalize_cmd+=(--limit_per_level "${LIMIT_PER_LEVEL}")
fi
if [ -n "${SAMPLE_PER_LEVEL}" ]; then
    normalize_cmd+=(--sample_per_level "${SAMPLE_PER_LEVEL}")
fi
if [ -n "${TEST_SAMPLE_PER_LEVEL}" ]; then
    normalize_cmd+=(--test_sample_per_level "${TEST_SAMPLE_PER_LEVEL}")
fi

"${normalize_cmd[@]}"

python examples/data_preprocess/opcd_json.py \
    --input "${TRAIN_JSON}" \
    --save_dir "${TRAIN_SAVE_DIR}" \
    --duplicate_to "${OPCD_DUPLICATE_TO}"

python examples/data_preprocess/opcd_json.py \
    --input "${VAL_JSON}" \
    --save_dir "${VAL_SAVE_DIR}" \
    --duplicate_to 0

mv -f "${VAL_SAVE_DIR}/train.parquet" "${VAL_SAVE_DIR}/test.parquet"

echo "Prepared training parquet: ${TRAIN_SAVE_DIR}/train.parquet"
echo "Prepared validation parquet: ${VAL_SAVE_DIR}/test.parquet"
