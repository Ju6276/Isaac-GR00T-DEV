#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" != 1 ]]; then
  echo "Usage: $0 <double_desk|football|pp_box|boxing|sit_sofa|vision_navi>" >&2
  exit 2
fi

TASK="$1"
case "${TASK}" in
  double_desk|football|pp_box|boxing|sit_sofa|vision_navi) ;;
  *) echo "Unsupported HumanoidArena task: ${TASK}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${REPO_ROOT}/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/humanoidarena_sonic_v31_${TASK}}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/humanoidarena_sonic_${TASK}_groot_n17_100k}"
EXPERIMENT_NAME="humanoidarena-sonic-${TASK}-canonical40-h30-100k"

export NUM_GPUS="${NUM_GPUS:-1}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
# Six final checkpoints fit on this workstation; retaining five per task does not.
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
export USE_WANDB=1
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-ju-dong6276-technical-university-of-munich}"

cd "${REPO_ROOT}"
uv run python - "${DATASET_PATH}" <<'PY'
import sys
from pathlib import Path
from examples.HumanoidArena.prepare_dataset import validate_dataset

info = validate_dataset(Path(sys.argv[1]))
print(
    f"Validated {sys.argv[1]}: episodes={info.get('total_episodes')} "
    f"frames={info.get('total_frames')} state=64D action=40D"
)
PY
uv run python gr00t/data/stats.py \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag HUMANOIDARENA_G1 \
  --modality-config-path examples/HumanoidArena/humanoidarena_config.py

exec uv run bash examples/finetune.sh \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag HUMANOIDARENA_G1 \
  --modality-config-path examples/HumanoidArena/humanoidarena_config.py \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --wandb-project "${WANDB_PROJECT:-HumanoidArena}"
