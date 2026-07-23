#!/bin/bash
#SBATCH --job-name=evgnn_controlled_smoke_matrix
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail
set -E

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/controlled_multiscale_smoke_matrix"
OUT_DIR="/projects/fr57/cche0357/EV-GNN_outputs"

MAX_TIMESTEPS=512
START_TIMESTEPS=64
EVAL_FREQ=256
TRAIN_EVAL_EPISODES=1
BATCH_SIZE=32
REPLAY_BUFFER_SIZE=5000
DEVICE="cpu"
LOG_TO_WANDB="false"
SMOKE_EVAL_EPISODES=3
DETERMINISTIC="true"
EVAL_EXPL_NOISE="0.0"
REQUESTED_MEMORY="64G"
REQUESTED_WALL_TIME="04:00:00"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is required and must be 0-7" >&2
  exit 2
fi

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be an integer in 0-7; got ${TASK_ID}" >&2
  exit 2
fi

case "${TASK_ID}" in
  0)
    SCALE="25CP"
    SCALE_LABEL="25cp"
    CONFIG_PATH="./config_files/PublicPST_25cp.yaml"
    ALGORITHM="actiongnn"
    SEED=0
    EVAL_SEED_OFFSET=610000
    ;;
  1)
    SCALE="25CP"
    SCALE_LABEL="25cp"
    CONFIG_PATH="./config_files/PublicPST_25cp.yaml"
    ALGORITHM="hierarchical"
    SEED=0
    EVAL_SEED_OFFSET=610000
    ;;
  2)
    SCALE="100CP"
    SCALE_LABEL="100cp"
    CONFIG_PATH="./config_files/PublicPST_100.yaml"
    ALGORITHM="actiongnn"
    SEED=0
    EVAL_SEED_OFFSET=620000
    ;;
  3)
    SCALE="100CP"
    SCALE_LABEL="100cp"
    CONFIG_PATH="./config_files/PublicPST_100.yaml"
    ALGORITHM="hierarchical"
    SEED=0
    EVAL_SEED_OFFSET=620000
    ;;
  4)
    SCALE="500CP"
    SCALE_LABEL="500cp"
    CONFIG_PATH="./config_files/PublicPST_500.yaml"
    ALGORITHM="actiongnn"
    SEED=0
    EVAL_SEED_OFFSET=630000
    ;;
  5)
    SCALE="500CP"
    SCALE_LABEL="500cp"
    CONFIG_PATH="./config_files/PublicPST_500.yaml"
    ALGORITHM="hierarchical"
    SEED=0
    EVAL_SEED_OFFSET=630000
    ;;
  6)
    SCALE="1000CP"
    SCALE_LABEL="1000cp"
    CONFIG_PATH="./config_files/PublicPST_1000.yaml"
    ALGORITHM="actiongnn"
    SEED=0
    EVAL_SEED_OFFSET=640000
    ;;
  7)
    SCALE="1000CP"
    SCALE_LABEL="1000cp"
    CONFIG_PATH="./config_files/PublicPST_1000.yaml"
    ALGORITHM="hierarchical"
    SEED=0
    EVAL_SEED_OFFSET=640000
    ;;
  *)
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-7" >&2
    exit 2
    ;;
esac

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
JOB_NAME="${SLURM_JOB_NAME:-evgnn_controlled_smoke_matrix}"

RUN_NAME="m3_controlled_smoke_matrix_${SCALE_LABEL}_${ALGORITHM}_seed${SEED}_job${ARRAY_JOB_ID}_task${TASK_ID}"
RUN_DIR="${RUN_ROOT}/${SCALE_LABEL}/${ALGORITHM}/seed${SEED}/${RUN_NAME}"
SAVE_DIR="${RUN_DIR}/training"
MODEL_DIR="${SAVE_DIR}/${RUN_NAME}"
EVAL_DIR="${RUN_DIR}/smoke_eval"
METADATA_DIR="${RUN_DIR}/runtime_metadata"
SMOKE_EVAL_CSV="${EVAL_DIR}/${RUN_NAME}_eval3.csv"
CHECKPOINT_PREFIX="${MODEL_DIR}/model.best"
PACKAGE_PATH="${OUT_DIR}/m3_controlled_smoke_matrix_${SCALE_LABEL}_${ALGORITHM}_seed${SEED}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
STDOUT_LOG="${RUN_DIR}/stdout.log"
STDERR_LOG="${RUN_DIR}/stderr.log"
SLURM_STDOUT_PATH="${OUT_DIR}/${JOB_NAME}_${ARRAY_JOB_ID}_${TASK_ID}.out"
SLURM_STDERR_PATH="${OUT_DIR}/${JOB_NAME}_${ARRAY_JOB_ID}_${TASK_ID}.err"
PACKAGE_CREATED=0
RUN_DIR_CREATED=0

format_seconds() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

print_shell_command() {
  printf '%q ' "$@"
  printf '\n'
}

resolve_simulation_length() {
  local config_path="$1"
  if [[ ! -s "${config_path}" ]]; then
    echo "ERROR: expected config path is missing or empty before training: ${config_path}" >&2
    return 3
  fi

  python - "${config_path}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

if not isinstance(config, dict):
    print(f"ERROR: config did not parse as a YAML mapping: {config_path}", file=sys.stderr)
    raise SystemExit(3)

simulation_length = config.get("simulation_length")
if isinstance(simulation_length, bool):
    print(f"ERROR: simulation_length must be a positive integer in {config_path}", file=sys.stderr)
    raise SystemExit(3)

try:
    simulation_length_int = int(simulation_length)
except (TypeError, ValueError):
    print(
        f"ERROR: simulation_length is missing or invalid in {config_path}: {simulation_length!r}",
        file=sys.stderr,
    )
    raise SystemExit(3)

if simulation_length_int <= 0 or str(simulation_length_int) != str(simulation_length).strip():
    print(
        f"ERROR: simulation_length must be a positive integer in {config_path}: {simulation_length!r}",
        file=sys.stderr,
    )
    raise SystemExit(3)

print(simulation_length_int)
PY
}

build_train_command() {
  TRAIN_COMMAND=(
    python train_td3_gnn.py
    --algorithm "${ALGORITHM}"
    --config "${CONFIG_PATH}"
    --seed "${SEED}"
    --device "${DEVICE}"
    --run_name "${RUN_NAME}"
    --max_timesteps "${MAX_TIMESTEPS}"
    --start_timesteps "${START_TIMESTEPS}"
    --eval_freq "${EVAL_FREQ}"
    --eval_episodes "${TRAIN_EVAL_EPISODES}"
    --batch_size "${BATCH_SIZE}"
    --replay_buffer_size "${REPLAY_BUFFER_SIZE}"
    --save_dir "${SAVE_DIR}"
    --log_to_wandb "${LOG_TO_WANDB}"
  )
}

build_eval_command() {
  EVAL_COMMAND=(
    python evaluate_td3_gnn.py
    --algorithm "${ALGORITHM}"
    --config "${CONFIG_PATH}"
    --seed "${SEED}"
    --eval_episodes "${SMOKE_EVAL_EPISODES}"
    --checkpoint "${CHECKPOINT_PREFIX}"
    --device "${DEVICE}"
    --output_csv "${SMOKE_EVAL_CSV}"
    --run_name "${RUN_NAME}_eval3"
    --max_episode_steps "${MAX_EPISODE_STEPS}"
    --deterministic "${DETERMINISTIC}"
    --eval_expl_noise "${EVAL_EXPL_NOISE}"
    --eval_seed_offset "${EVAL_SEED_OFFSET}"
  )
}

print_task_mapping() {
  echo "task_id=${TASK_ID}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "seed=${SEED}"
  echo "config_path=${CONFIG_PATH}"
  echo "eval_seed_offset=${EVAL_SEED_OFFSET}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "model_dir=${MODEL_DIR}"
  echo "smoke_eval_csv=${SMOKE_EVAL_CSV}"
  echo "package_path=${PACKAGE_PATH}"
}

package_run() {
  PACKAGE_CREATED=1
  mkdir -p "${METADATA_DIR}" "${OUT_DIR}"

  if [[ -f "${SLURM_STDOUT_PATH}" ]]; then
    cp "${SLURM_STDOUT_PATH}" "${METADATA_DIR}/slurm_stdout_snapshot.out" || true
  fi
  if [[ -f "${SLURM_STDERR_PATH}" ]]; then
    cp "${SLURM_STDERR_PATH}" "${METADATA_DIR}/slurm_stderr_snapshot.err" || true
  fi

  find "${RUN_DIR}" -maxdepth 8 -type f | sort > "${METADATA_DIR}/packaged_files.txt"
  tar -czf "${PACKAGE_PATH}" -C "${RUN_DIR}" .
  tar -tzf "${PACKAGE_PATH}" >/dev/null
  ls -lh "${PACKAGE_PATH}"
}

failure_trap() {
  local exit_status="$?"
  trap - EXIT
  if [[ "${exit_status}" -ne 0 && "${PACKAGE_CREATED}" -ne 1 && "${RUN_DIR_CREATED}" == "1" ]]; then
    echo "ERROR: controlled smoke task failed with exit status ${exit_status}; packaging available artifacts." >&2
    {
      echo "date_failure=$(date -Iseconds)"
      echo "exit_status=${exit_status}"
      echo "task_id=${TASK_ID}"
      echo "scale=${SCALE}"
      echo "algorithm=${ALGORITHM}"
      echo "seed=${SEED}"
      echo "package_path=${PACKAGE_PATH}"
    } >> "${METADATA_DIR}/runtime_failure.txt"
    package_run || true
  fi
  exit "${exit_status}"
}

if [[ "${EV_GNN_SMOKE_DRY_RUN:-0}" == "1" ]]; then
  echo "CONTROLLED_SMOKE_TASK_START"
  echo "dry_run=1"
  print_task_mapping
  MAX_EPISODE_STEPS="$(resolve_simulation_length "${CONFIG_PATH}")"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  build_train_command
  build_eval_command
  echo "TRAINING_COMMAND"
  print_shell_command "${TRAIN_COMMAND[@]}"
  echo "EVALUATION_COMMAND"
  print_shell_command "${EVAL_COMMAND[@]}"
  echo "DRY_RUN_NO_TRAINING_OR_EVALUATION"
  exit 0
fi

THREAD_COUNT="${SLURM_CPUS_PER_TASK:-4}"
if [[ "${THREAD_COUNT}" != "4" ]]; then
  echo "ERROR: expected SLURM_CPUS_PER_TASK=4, got ${THREAD_COUNT}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREAD_COUNT}"
export MKL_NUM_THREADS="${THREAD_COUNT}"
export NUMEXPR_NUM_THREADS="${THREAD_COUNT}"

mkdir -p "${SAVE_DIR}" "${EVAL_DIR}" "${METADATA_DIR}" "${OUT_DIR}"
RUN_DIR_CREATED=1

exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)
trap failure_trap EXIT

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

cd "${REPO_DIR}"

TASK_START_EPOCH="$(date +%s)"
TASK_START_ISO="$(date -Iseconds)"

echo "CONTROLLED_SMOKE_TASK_START"
print_task_mapping
echo "date_start=${TASK_START_ISO}"
echo "hostname=$(hostname)"
echo "slurm_job_id=${JOB_ID}"
echo "slurm_array_job_id=${ARRAY_JOB_ID}"
echo "slurm_array_task_id=${TASK_ID}"
echo "slurm_job_name=${JOB_NAME}"
echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-}"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "memory_request=${REQUESTED_MEMORY}"
echo "wall_time_request=${REQUESTED_WALL_TIME}"
echo "device=${DEVICE}"
echo "max_timesteps=${MAX_TIMESTEPS}"
echo "start_timesteps=${START_TIMESTEPS}"
echo "eval_freq=${EVAL_FREQ}"
echo "train_eval_episodes=${TRAIN_EVAL_EPISODES}"
echo "batch_size=${BATCH_SIZE}"
echo "replay_buffer_size=${REPLAY_BUFFER_SIZE}"
echo "smoke_eval_episodes=${SMOKE_EVAL_EPISODES}"
echo "deterministic=${DETERMINISTIC}"

for source_path in "${CONFIG_PATH}" "train_td3_gnn.py" "evaluate_td3_gnn.py"; do
  if [[ ! -s "${source_path}" ]]; then
    echo "ERROR: missing or empty source/config file before training: ${source_path}" >&2
    exit 3
  fi
  echo "OK: ${source_path}"
done

MAX_EPISODE_STEPS="$(resolve_simulation_length "${CONFIG_PATH}")"
echo "max_episode_steps=${MAX_EPISODE_STEPS}"
build_train_command
build_eval_command

{
  echo "date_start=${TASK_START_ISO}"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${JOB_ID}"
  echo "slurm_array_job_id=${ARRAY_JOB_ID}"
  echo "slurm_array_task_id=${TASK_ID}"
  echo "slurm_job_name=${JOB_NAME}"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
  echo "repo_dir=${REPO_DIR}"
  echo "run_root=${RUN_ROOT}"
  echo "out_dir=${OUT_DIR}"
  echo "task_id=${TASK_ID}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "seed=${SEED}"
  echo "config_path=${CONFIG_PATH}"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "eval_seed_offset=${EVAL_SEED_OFFSET}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "model_dir=${MODEL_DIR}"
  echo "smoke_eval_csv=${SMOKE_EVAL_CSV}"
  echo "checkpoint_prefix=${CHECKPOINT_PREFIX}"
  echo "package_path=${PACKAGE_PATH}"
  echo "formal_evidence_boundary=Stage 1 smoke validation only; no formal result is claimed."
} > "${METADATA_DIR}/runtime_start.txt"

{
  echo "source_manifest_env=${EV_GNN_SOURCE_MANIFEST:-unprovided}"
  echo "source_update_note=No git pull is performed by this job; validate source with EV_GNN_SOURCE_MANIFEST when available."
} > "${METADATA_DIR}/source_provenance.txt"

if [[ -n "${EV_GNN_SOURCE_MANIFEST:-}" ]]; then
  if [[ ! -s "${EV_GNN_SOURCE_MANIFEST}" ]]; then
    echo "ERROR: EV_GNN_SOURCE_MANIFEST is set but missing or empty: ${EV_GNN_SOURCE_MANIFEST}" >&2
    exit 3
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum is required when EV_GNN_SOURCE_MANIFEST is provided" >&2
    exit 3
  fi
  if ! sha256sum -c "${EV_GNN_SOURCE_MANIFEST}" | tee "${METADATA_DIR}/source_manifest_validation.txt"; then
    echo "ERROR: M3 source does not match EV_GNN_SOURCE_MANIFEST" >&2
    exit 3
  fi
  cp "${EV_GNN_SOURCE_MANIFEST}" "${METADATA_DIR}/source_manifest.txt"
  sha256sum "${METADATA_DIR}/source_manifest.txt" > "${METADATA_DIR}/source_manifest.sha256"
else
  echo "WARNING: EV_GNN_SOURCE_MANIFEST not provided; source manifest validation skipped." >&2
fi

{
  echo "python_executable=$(command -v python)"
  python - <<'PY'
import platform
import sys

import numpy as np
import torch
import torch_geometric
import yaml

print(f"python_version={platform.python_version()}")
print(f"python_executable={sys.executable}")
print(f"numpy_version={np.__version__}")
print(f"pytorch_version={torch.__version__}")
print(f"pytorch_cuda_version={torch.version.cuda}")
print(f"pytorch_cuda_available={torch.cuda.is_available()}")
print(f"pytorch_num_threads={torch.get_num_threads()}")
print(f"pytorch_num_interop_threads={torch.get_num_interop_threads()}")
print(f"torch_geometric_version={torch_geometric.__version__}")
print(f"pyyaml_version={yaml.__version__}")
PY
} > "${METADATA_DIR}/python_environment.txt"

echo "TRAINING_COMMAND"
print_shell_command "${TRAIN_COMMAND[@]}" | tee "${METADATA_DIR}/training_command.txt"

TRAIN_START_EPOCH="$(date +%s)"
set +e
"${TRAIN_COMMAND[@]}"
TRAIN_EXIT_STATUS="$?"
set -e
TRAIN_END_EPOCH="$(date +%s)"
TRAIN_WALL_SECONDS=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))

{
  echo "date_train_start=$(date -Iseconds --date="@${TRAIN_START_EPOCH}" 2>/dev/null || date -r "${TRAIN_START_EPOCH}" -Iseconds 2>/dev/null || date -Iseconds)"
  echo "date_train_end=$(date -Iseconds)"
  echo "training_exit_status=${TRAIN_EXIT_STATUS}"
  echo "training_wall_time_seconds=${TRAIN_WALL_SECONDS}"
  echo "training_wall_time_hhmmss=$(format_seconds "${TRAIN_WALL_SECONDS}")"
} > "${METADATA_DIR}/runtime_training.txt"

if [[ "${TRAIN_EXIT_STATUS}" -ne 0 ]]; then
  echo "ERROR: training command failed with exit status ${TRAIN_EXIT_STATUS}" >&2
  exit "${TRAIN_EXIT_STATUS}"
fi

echo "=== Validate training artefacts ==="
REQUIRED_TRAINING_FILES=(
  "${MODEL_DIR}/model.best_actor"
  "${MODEL_DIR}/model.best_critic"
  "${MODEL_DIR}/config.yaml"
  "${MODEL_DIR}/kwargs.yaml"
  "${MODEL_DIR}/run_args.yaml"
  "${MODEL_DIR}/training_log.csv"
)

for required_path in "${REQUIRED_TRAINING_FILES[@]}"; do
  if [[ ! -s "${required_path}" ]]; then
    echo "ERROR: missing or empty required training artefact: ${required_path}" >&2
    exit 4
  fi
  echo "OK: ${required_path}"
done

echo "EVALUATION_COMMAND"
print_shell_command "${EVAL_COMMAND[@]}" | tee "${METADATA_DIR}/evaluation_command.txt"

EVAL_START_EPOCH="$(date +%s)"
set +e
"${EVAL_COMMAND[@]}"
EVAL_EXIT_STATUS="$?"
set -e
EVAL_END_EPOCH="$(date +%s)"
EVAL_WALL_SECONDS=$((EVAL_END_EPOCH - EVAL_START_EPOCH))

{
  echo "date_eval_start=$(date -Iseconds --date="@${EVAL_START_EPOCH}" 2>/dev/null || date -r "${EVAL_START_EPOCH}" -Iseconds 2>/dev/null || date -Iseconds)"
  echo "date_eval_end=$(date -Iseconds)"
  echo "evaluation_exit_status=${EVAL_EXIT_STATUS}"
  echo "evaluation_wall_time_seconds=${EVAL_WALL_SECONDS}"
  echo "evaluation_wall_time_hhmmss=$(format_seconds "${EVAL_WALL_SECONDS}")"
} > "${METADATA_DIR}/runtime_evaluation.txt"

if [[ "${EVAL_EXIT_STATUS}" -ne 0 ]]; then
  echo "ERROR: evaluation command failed with exit status ${EVAL_EXIT_STATUS}" >&2
  exit "${EVAL_EXIT_STATUS}"
fi

echo "=== Validate smoke eval CSV ==="
python - \
  "${SMOKE_EVAL_CSV}" \
  "${ALGORITHM}" \
  "${SEED}" \
  "${SMOKE_EVAL_EPISODES}" \
  "${MAX_EPISODE_STEPS}" \
  "${METADATA_DIR}/csv_validation_summary.env" <<'PY'
import csv
import sys
from pathlib import Path

(
    csv_path_text,
    expected_algorithm,
    expected_seed_text,
    expected_episode_count_text,
    expected_episode_steps_text,
    validation_summary_path_text,
) = sys.argv[1:]

csv_path = Path(csv_path_text)
expected_seed = int(expected_seed_text)
expected_episode_count = int(expected_episode_count_text)
expected_episode_steps = int(expected_episode_steps_text)
validation_summary_path = Path(validation_summary_path_text)

required_columns = {
    "row_type",
    "algorithm",
    "seed",
    "episode_index",
    "episode_reward",
    "episode_steps",
    "done",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_transformer_overload",
    "action_fraction_at_max",
}

if not csv_path.is_file() or csv_path.stat().st_size == 0:
    raise SystemExit(f"CSV validation failed: missing or empty CSV: {csv_path}")

with csv_path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if reader.fieldnames is None:
        raise SystemExit("CSV validation failed: CSV has no header")
    missing_columns = sorted(required_columns - set(reader.fieldnames))
    if missing_columns:
        raise SystemExit(f"CSV validation failed: missing columns: {', '.join(missing_columns)}")
    rows = list(reader)

if len(rows) != expected_episode_count + 1:
    raise SystemExit(
        f"CSV validation failed: expected {expected_episode_count + 1} data rows, got {len(rows)}"
    )

episode_rows = [row for row in rows if row.get("row_type") == "episode"]
summary_rows = [row for row in rows if row.get("row_type") == "summary"]
unexpected_row_types = sorted({row.get("row_type", "") for row in rows} - {"episode", "summary"})

if unexpected_row_types:
    raise SystemExit(
        f"CSV validation failed: unexpected row_type values: {', '.join(unexpected_row_types)}"
    )

if len(episode_rows) != expected_episode_count:
    raise SystemExit(
        f"CSV validation failed: expected {expected_episode_count} episode rows, got {len(episode_rows)}"
    )

if len(summary_rows) != 1:
    raise SystemExit(f"CSV validation failed: expected 1 summary row, got {len(summary_rows)}")

episode_indexes = [int(row["episode_index"]) for row in episode_rows]
expected_episode_indexes = list(range(expected_episode_count))
if episode_indexes != expected_episode_indexes:
    raise SystemExit(
        f"CSV validation failed: episode indexes {episode_indexes} != {expected_episode_indexes}"
    )

for row in episode_rows:
    if row["algorithm"] != expected_algorithm:
        raise SystemExit(
            f"CSV validation failed: episode {row['episode_index']} algorithm {row['algorithm']} != {expected_algorithm}"
        )
    if int(row["seed"]) != expected_seed:
        raise SystemExit(
            f"CSV validation failed: episode {row['episode_index']} seed {row['seed']} != {expected_seed}"
        )
    if int(row["episode_steps"]) != expected_episode_steps:
        raise SystemExit(
            f"CSV validation failed: episode {row['episode_index']} steps {row['episode_steps']} != {expected_episode_steps}"
        )
    if row["done"].strip().lower() != "true":
        raise SystemExit(f"CSV validation failed: episode {row['episode_index']} done is not true")

summary_row = summary_rows[0]
if summary_row["algorithm"] != expected_algorithm:
    raise SystemExit(
        f"CSV validation failed: summary algorithm {summary_row['algorithm']} != {expected_algorithm}"
    )
if int(summary_row["seed"]) != expected_seed:
    raise SystemExit(f"CSV validation failed: summary seed {summary_row['seed']} != {expected_seed}")

validation_summary_path.write_text(
    "\n".join([
        f"ACTUAL_ROW_COUNT={len(rows)}",
        f"ACTUAL_EPISODE_ROW_COUNT={len(episode_rows)}",
        f"ACTUAL_SUMMARY_ROW_COUNT={len(summary_rows)}",
        f"ACTUAL_EPISODE_STEPS={expected_episode_steps}",
    ]) + "\n",
    encoding="utf-8",
)

print("SMOKE_CSV_VALIDATION_OK")
print(f"row_count={len(rows)}")
print(f"episode_rows={len(episode_rows)}")
print(f"summary_rows={len(summary_rows)}")
print(f"episode_steps={expected_episode_steps}")
PY

if [[ ! -s "${SMOKE_EVAL_CSV}" ]]; then
  echo "ERROR: missing or empty smoke eval CSV after validation: ${SMOKE_EVAL_CSV}" >&2
  exit 5
fi

sha256sum \
  "${MODEL_DIR}/model.best_actor" \
  "${MODEL_DIR}/model.best_critic" \
  "${MODEL_DIR}/config.yaml" \
  "${MODEL_DIR}/kwargs.yaml" \
  "${MODEL_DIR}/run_args.yaml" \
  "${MODEL_DIR}/training_log.csv" \
  "${SMOKE_EVAL_CSV}" \
  > "${METADATA_DIR}/controlled_smoke_checksums.sha256"

TASK_END_EPOCH="$(date +%s)"
TASK_WALL_SECONDS=$((TASK_END_EPOCH - TASK_START_EPOCH))

{
  echo "date_end=$(date -Iseconds)"
  echo "task_exit_status=0"
  echo "task_wall_time_seconds=${TASK_WALL_SECONDS}"
  echo "task_wall_time_hhmmss=$(format_seconds "${TASK_WALL_SECONDS}")"
  echo "training_wall_time_seconds=${TRAIN_WALL_SECONDS}"
  echo "evaluation_wall_time_seconds=${EVAL_WALL_SECONDS}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "seed=${SEED}"
  echo "config_path=${CONFIG_PATH}"
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "smoke_eval_csv=${SMOKE_EVAL_CSV}"
  echo "package_path=${PACKAGE_PATH}"
  echo "formal_evidence_boundary=Stage 1 smoke validation only; no formal result is claimed."
} > "${METADATA_DIR}/runtime_final.txt"

find "${RUN_DIR}" -maxdepth 8 -type f | sort > "${METADATA_DIR}/generated_files.txt"

echo "=== Package controlled smoke task artefacts ==="
package_run

echo "CONTROLLED_SMOKE_TASK_COMPLETED"
echo "task_wall_time=$(format_seconds "${TASK_WALL_SECONDS}")"
echo "package_path=${PACKAGE_PATH}"
