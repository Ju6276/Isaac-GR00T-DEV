#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA_ROOT="${ARENA_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/HumanoidArena}"
EVAL_ROOT="${ARENA_ROOT}/isaaclab_twist2_g1"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
SERVER_PYTHON="${SERVER_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SONIC_POLICY_ROOT="${SONIC_POLICY_ROOT:?Set SONIC_POLICY_ROOT to the SONIC WBC policy/release directory}"
SERVER_SCRIPT="${REPO_ROOT}/examples/HumanoidArena/serve_eval_stack.py"
STATUS_DIR="${REPO_ROOT}/outputs/humanoidarena_sonic_6task_queue"
mkdir -p "${STATUS_DIR}"

TASKS=(double_desk football pp_box boxing sit_sofa vision_navi)

task_settings() {
  case "$1" in
    double_desk) echo "HOI_double_desk doubledesk_sonic_test.yaml 2000" ;;
    football) echo "HOI_football football_single_sonic_test.yaml 2000" ;;
    pp_box) echo "HOI_pp_box pp_box_sonic_test.yaml 1450" ;;
    boxing) echo "HSI_boxing boxing_sonic_test.yaml 900" ;;
    sit_sofa) echo "HSI_sit_sofa sit_sofa_sonic_test.yaml 2000" ;;
    vision_navi) echo "HSI_vision_navi vision_navi_sonic_test.yaml 1800" ;;
  esac
}

evaluate_task() {
  local task="$1" task_id config_name max_steps
  read -r task_id config_name max_steps < <(task_settings "${task}")
  local checkpoint="${REPO_ROOT}/outputs/humanoidarena_sonic_${task}_groot_n17_100k/humanoidarena-sonic-${task}-canonical40-h30-100k/checkpoint-100000"
  local launcher="${EVAL_ROOT}/script/eval_scripts/sonic_pi05/${task_id}_run_vla_eval_parallel.sh"
  local result_dir="${ARENA_ROOT}/eval_results/groot_n17_sonic_${task}_100000_base_$(date +%Y%m%d_%H%M%S)"
  local smoke_dir="${STATUS_DIR}/${task}_smoke_$(date +%Y%m%d_%H%M%S)"
  [[ -d "${checkpoint}" ]] || { echo "Missing final checkpoint: ${checkpoint}" >&2; return 1; }

  local common_env=(
    AUTO_ACTIVATE_CONDA=0 EVAL_PYTHON="${EVAL_PYTHON}"
    SERVER_PYTHON="${SERVER_PYTHON}" SERVER_SCRIPT="${SERVER_SCRIPT}"
    ENABLE_MEMORY_LIMIT=0
    SONIC_ENCODER_PATH="${SONIC_POLICY_ROOT}/model_encoder.onnx"
    SONIC_DECODER_PATH="${SONIC_POLICY_ROOT}/model_decoder.onnx"
    MODEL_PATHS_CSV="${checkpoint}"
    ENV_CONFIG_YAML="tasks/common_test_config/base_test/${config_name}"
    NUM_WORKERS=1 WORKERS_PER_DEVICE=1 SERVER_GPU_IDS=0 ISAAC_DEVICE=cuda:0
    MAX_STEPS="${max_steps}" HEADLESS=1 RECORD_VIDEO_EVERY_N=1
    SONIC_VLA_ACTION_FORMAT=semantic_v3 LEROBOT_VLA_RECORD_OUTPUTS=1
  )

  timeout --signal=TERM --kill-after=30s 30m \
    env "${common_env[@]}" RESULTS_DIR="${smoke_dir}" SEEDS_OVERRIDE=0 REPEATS_PER_SEED=1 \
      RESUME_LATEST=0 bash "${launcher}"
  "${EVAL_PYTHON}" - "${smoke_dir}/summary.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if len(rows) != 1 or rows[0]["episode_steps"] <= 0 or rows[0]["failure_reason"] in {"process_error", "worker_error"}:
    raise SystemExit(f"invalid GR00T smoke result: {rows}")
PY
  if command -v gio >/dev/null 2>&1; then
    gio trash "${smoke_dir}"
  else
    echo "[queue] gio is unavailable; keeping smoke results at ${smoke_dir}"
  fi

  env "${common_env[@]}" RESULTS_DIR="${result_dir}" SEEDS_OVERRIDE='0 1 2' REPEATS_PER_SEED=20 \
    RESUME_LATEST=1 bash "${launcher}"
  printf '%s\t%s\n' "${task}" "${result_dir}" >> "${STATUS_DIR}/completed_evaluations.tsv"
}

for task in "${TASKS[@]}"; do
  final_checkpoint="${REPO_ROOT}/outputs/humanoidarena_sonic_${task}_groot_n17_100k/humanoidarena-sonic-${task}-canonical40-h30-100k/checkpoint-100000"
  if [[ ! -d "${final_checkpoint}" ]]; then
    printf '%s\ttraining\t%s\n' "$(date --iso-8601=seconds)" "${task}" > "${STATUS_DIR}/current.tsv"
    bash "${REPO_ROOT}/examples/HumanoidArena/train_sonic_task.sh" "${task}"
  else
    echo "[queue] reusing completed checkpoint: ${final_checkpoint}"
  fi
  printf '%s\tevaluating\t%s\n' "$(date --iso-8601=seconds)" "${task}" > "${STATUS_DIR}/current.tsv"
  evaluate_task "${task}"
  printf '%s\tcomplete\t%s\n' "$(date --iso-8601=seconds)" "${task}" >> "${STATUS_DIR}/completed_tasks.tsv"
done

printf '%s\tall_complete\n' "$(date --iso-8601=seconds)" > "${STATUS_DIR}/current.tsv"
