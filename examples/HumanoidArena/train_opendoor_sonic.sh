#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${REPO_ROOT}/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/humanoidarena_sonic_v31_opendoor}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/humanoidarena_sonic_opendoor_groot_n17_100k}"
NUM_GPUS="${NUM_GPUS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"

export NUM_GPUS
export MAX_STEPS=100000
export GLOBAL_BATCH_SIZE
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export USE_WANDB=1
export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-ju-dong6276-technical-university-of-munich}"

cd "${REPO_ROOT}"

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
  --experiment-name humanoidarena-sonic-opendoor-canonical40-h30-100k \
  --wandb-project "${WANDB_PROJECT:-HumanoidArena}"
