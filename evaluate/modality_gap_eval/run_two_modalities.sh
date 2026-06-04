#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_KIND="${MODEL_KIND:-base}"

INPUT_MODALITY=image bash "${SCRIPT_DIR}/run_condition.sh" "$@"
INPUT_MODALITY=text bash "${SCRIPT_DIR}/run_condition.sh" "$@"

python "${SCRIPT_DIR}/summarize_modality_gap.py" \
    --results-dir "${SCRIPT_DIR}/results" \
    --output-prefix "${SCRIPT_DIR}/results/modality_gap_comparison"
