#!/bin/bash
#SBATCH --job-name=evgnn_controlled_formal_matrix
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT=/projects/fr57/cche0357/EV-GNN
RUN_ROOT=/scratch2/fr57/cche0357/EV-GNN_runs/controlled_multiscale_formal
OUTPUT_ROOT=/projects/fr57/cche0357/EV-GNN_outputs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LOCAL_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

MAX_TIMESTEPS=50000
START_TIMESTEPS=1000
EVAL_FREQ=5000
TRAIN_EVAL_EPISODES=5
BATCH_SIZE=64
REPLAY_BUFFER_SIZE=100000
DEVICE=cpu
LOG_TO_WANDB=false
FORMAL_EVAL_EPISODES=30
DETERMINISTIC=true
EVAL_EXPL_NOISE=0.0
REQUESTED_MEMORY=96G
REQUESTED_WALL_TIME=48:00:00

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is required and must be 0-39" >&2
  exit 2
fi

if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be an integer in 0-39; got ${TASK_ID}" >&2
  exit 2
fi

if (( TASK_ID < 0 || TASK_ID > 39 )); then
  echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-39" >&2
  exit 2
fi

SCALES=(25cp 100cp 500cp 1000cp)
ALGORITHMS=(actiongnn hierarchical)
SEED_COUNT=5
TASKS_PER_SCALE=10

SCALE_INDEX=$((TASK_ID / TASKS_PER_SCALE))
ALGORITHM_INDEX=$(((TASK_ID % TASKS_PER_SCALE) / SEED_COUNT))
SEED=$((TASK_ID % SEED_COUNT))
SCALE="${SCALES[$SCALE_INDEX]}"
ALGORITHM="${ALGORITHMS[$ALGORITHM_INDEX]}"

case "${SCALE}" in
  25cp)
    CONFIG_PATH=config_files/PublicPST_25cp.yaml
    EVAL_BASE_SEED=710000
    ;;
  100cp)
    CONFIG_PATH=config_files/PublicPST_100.yaml
    EVAL_BASE_SEED=720000
    ;;
  500cp)
    CONFIG_PATH=config_files/PublicPST_500.yaml
    EVAL_BASE_SEED=730000
    ;;
  1000cp)
    CONFIG_PATH=config_files/PublicPST_1000.yaml
    EVAL_BASE_SEED=740000
    ;;
  *)
    echo "ERROR: unsupported scale mapping for task ${TASK_ID}: ${SCALE}" >&2
    exit 2
    ;;
esac

EXPECTED_FIRST_EPISODE_SEED=$((EVAL_BASE_SEED + SEED * 1000))
EXPECTED_LAST_EPISODE_SEED=$((EXPECTED_FIRST_EPISODE_SEED + FORMAL_EVAL_EPISODES - 1))
EVAL_SEED_OFFSET=$((EXPECTED_FIRST_EPISODE_SEED - SEED))

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
JOB_NAME="${SLURM_JOB_NAME:-evgnn_controlled_formal_matrix}"
RUN_NAME="controlled_multiscale_formal_${SCALE}_${ALGORITHM}_seed${SEED}"
TASK_RUN_DIR="${RUN_ROOT}/${SCALE}/${ALGORITHM}/seed${SEED}"
TRAIN_DIR="${TASK_RUN_DIR}/train"
MODEL_DIR="${TRAIN_DIR}/${RUN_NAME}"
EVAL_DIR="${TASK_RUN_DIR}/eval"
METADATA_DIR="${TASK_RUN_DIR}/runtime_metadata"
CONFIG_COPY_DIR="${TASK_RUN_DIR}/config"
CONFIG_COPY_PATH="${CONFIG_COPY_DIR}/${SCALE}_${ALGORITHM}_seed${SEED}_config.yaml"
CANONICAL_EVAL_CSV_NAME="${SCALE}_${ALGORITHM}_seed${SEED}_eval30.csv"
EVAL_CSV="${EVAL_DIR}/${CANONICAL_EVAL_CSV_NAME}"
CHECKPOINT_PREFIX="${MODEL_DIR}/model.best"
PACKAGE_PATH="${OUTPUT_ROOT}/m3_controlled_multiscale_formal_${SCALE}_${ALGORITHM}_seed${SEED}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
STDOUT_LOG="${TASK_RUN_DIR}/stdout.log"
STDERR_LOG="${TASK_RUN_DIR}/stderr.log"

TRAIN_EXIT_STATUS=not_started
EVAL_EXIT_STATUS=not_started
TASK_START_ISO=
TASK_END_ISO=
MAX_EPISODE_STEPS=
TRAIN_COMMAND_TEXT=
EVAL_COMMAND_TEXT=

iso_now() {
  date -Iseconds 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ"
}

format_seconds() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

print_shell_command() {
  printf "%q " "$@"
  printf "\n"
}

resolve_simulation_length() {
  local config_path="$1"
  if [[ ! -s "${config_path}" ]]; then
    echo "ERROR: config missing or empty: ${config_path}" >&2
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
        f"ERROR: simulation_length missing or invalid in {config_path}: {simulation_length!r}",
        file=sys.stderr,
    )
    raise SystemExit(3)

if simulation_length_int <= 0:
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
    --save_dir "${TRAIN_DIR}"
    --log_to_wandb "${LOG_TO_WANDB}"
  )
  TRAIN_COMMAND_TEXT="$(print_shell_command "${TRAIN_COMMAND[@]}")"
}

build_eval_command() {
  EVAL_COMMAND=(
    python evaluate_td3_gnn.py
    --algorithm "${ALGORITHM}"
    --config "${CONFIG_PATH}"
    --seed "${SEED}"
    --eval_episodes "${FORMAL_EVAL_EPISODES}"
    --checkpoint "${CHECKPOINT_PREFIX}"
    --device "${DEVICE}"
    --output_csv "${EVAL_CSV}"
    --run_name "${RUN_NAME}_eval30"
    --max_episode_steps "${MAX_EPISODE_STEPS}"
    --deterministic "${DETERMINISTIC}"
    --eval_expl_noise "${EVAL_EXPL_NOISE}"
    --eval_seed_offset "${EVAL_SEED_OFFSET}"
  )
  EVAL_COMMAND_TEXT="$(print_shell_command "${EVAL_COMMAND[@]}")"
}

print_task_mapping() {
  echo "task_id=${TASK_ID}"
  echo "slurm_job_id=${JOB_ID}"
  echo "slurm_array_job_id=${ARRAY_JOB_ID}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "seed=${SEED}"
  echo "config_path=${CONFIG_PATH}"
  echo "eval_base_seed=${EVAL_BASE_SEED}"
  echo "eval_seed_offset=${EVAL_SEED_OFFSET}"
  echo "expected_first_episode_seed=${EXPECTED_FIRST_EPISODE_SEED}"
  echo "expected_last_episode_seed=${EXPECTED_LAST_EPISODE_SEED}"
  echo "run_name=${RUN_NAME}"
  echo "train_dir=${TRAIN_DIR}"
  echo "model_dir=${MODEL_DIR}"
  echo "eval_dir=${EVAL_DIR}"
  echo "canonical_eval30_csv=${CANONICAL_EVAL_CSV_NAME}"
  echo "eval30_csv=${EVAL_CSV}"
  echo "package_path=${PACKAGE_PATH}"
}

write_runtime_metadata() {
  mkdir -p "${METADATA_DIR}"
  {
    echo "task_id=${TASK_ID}"
    echo "slurm_job_id=${JOB_ID}"
    echo "slurm_array_job_id=${ARRAY_JOB_ID}"
    echo "scale=${SCALE}"
    echo "algorithm=${ALGORITHM}"
    echo "seed=${SEED}"
    echo "config_path=${CONFIG_PATH}"
    echo "train_command=${TRAIN_COMMAND_TEXT}"
    echo "eval_command=${EVAL_COMMAND_TEXT}"
    echo "start_timestamp=${TASK_START_ISO}"
    echo "end_timestamp=${TASK_END_ISO}"
    echo "training_exit_status=${TRAIN_EXIT_STATUS}"
    echo "evaluation_exit_status=${EVAL_EXIT_STATUS}"
    echo "package_path=${PACKAGE_PATH}"
  } > "${METADATA_DIR}/task_runtime_metadata.env"
}

validate_required_files_exist() {
  local checked_path
  for checked_path in "$@"; do
    if [[ ! -s "${checked_path}" ]]; then
      echo "ERROR: required artefact missing or empty: ${checked_path}" >&2
      exit 4
    fi
    echo "OK: ${checked_path}"
  done
}

generate_source_manifest() {
  local source_manifest_path="${METADATA_DIR}/source_manifest.sha256"
  local source_manifest_files=(
    train_td3_gnn.py
    evaluate_td3_gnn.py
    TD3/TD3_ActionGNN_Controlled.py
    TD3/TD3_HierarchicalActionGNN.py
    utils/ev2gym_training_utils.py
    utils/state_public_pst_gnn.py
    "${CONFIG_PATH}"
    m3_jobs/17_controlled_multiscale_formal_train_eval.slurm
  )

  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum is required to generate the source manifest" >&2
    exit 3
  fi

  validate_required_files_exist "${source_manifest_files[@]}"
  printf "%s\n" "${source_manifest_files[@]}" > "${METADATA_DIR}/source_manifest_files.txt"
  sha256sum "${source_manifest_files[@]}" > "${source_manifest_path}"
}

validate_eval30_csv() {
  python - \
    "${EVAL_CSV}" \
    "${FORMAL_EVAL_EPISODES}" \
    "${EXPECTED_FIRST_EPISODE_SEED}" \
    "${MAX_EPISODE_STEPS}" \
    "${ALGORITHM}" \
    "${SEED}" \
    "${METADATA_DIR}/eval30_csv_validation.env" <<'PY'
import csv
import sys
from pathlib import Path

(
    csv_path_text,
    expected_episode_count_text,
    expected_first_episode_seed_text,
    expected_episode_steps_text,
    expected_algorithm,
    expected_seed_text,
    validation_summary_path_text,
) = sys.argv[1:]

csv_path = Path(csv_path_text)
expected_episode_count = int(expected_episode_count_text)
expected_first_episode_seed = int(expected_first_episode_seed_text)
expected_episode_steps = int(expected_episode_steps_text)
expected_seed = int(expected_seed_text)
validation_summary_path = Path(validation_summary_path_text)

required_columns = {
    "row_type",
    "algorithm",
    "seed",
    "episode_index",
    "episode_seed",
    "episode_steps",
    "done",
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_transformer_overload",
    "total_ev_served",
    "total_energy_charged",
    "total_energy_discharged",
    "battery_degradation",
    "battery_degradation_calendar",
    "battery_degradation_cycling",
    "total_profits",
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "action_fraction_zero",
    "action_fraction_at_max",
    "active_action_count_mean",
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

expected_data_row_count = expected_episode_count + 1
if len(rows) != expected_data_row_count:
    raise SystemExit(
        f"CSV validation failed: expected {expected_data_row_count} data rows, got {len(rows)}"
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

episode_seeds = [int(row["episode_seed"]) for row in episode_rows]
expected_episode_seeds = [
    expected_first_episode_seed + episode_index
    for episode_index in range(expected_episode_count)
]
if episode_seeds != expected_episode_seeds:
    raise SystemExit(
        f"CSV validation failed: episode seeds {episode_seeds} != {expected_episode_seeds}"
    )

for row in episode_rows:
    episode_index = row["episode_index"]
    if row["algorithm"] != expected_algorithm:
        raise SystemExit(
            f"CSV validation failed: episode {episode_index} algorithm {row['algorithm']} != {expected_algorithm}"
        )
    if int(row["seed"]) != expected_seed:
        raise SystemExit(
            f"CSV validation failed: episode {episode_index} seed {row['seed']} != {expected_seed}"
        )
    if int(row["episode_steps"]) != expected_episode_steps:
        raise SystemExit(
            f"CSV validation failed: episode {episode_index} steps {row['episode_steps']} != {expected_episode_steps}"
        )
    if row["done"].strip().lower() != "true":
        raise SystemExit(f"CSV validation failed: episode {episode_index} done is not true")

validation_summary_path.write_text(
    "\n".join([
        f"data_rows={len(rows)}",
        f"episode_rows={len(episode_rows)}",
        f"summary_rows={len(summary_rows)}",
        f"first_episode_seed={episode_seeds[0]}",
        f"last_episode_seed={episode_seeds[-1]}",
        f"episode_steps={expected_episode_steps}",
    ]) + "\n",
    encoding="utf-8",
)

print("FORMAL_EVAL30_CSV_VALIDATION_OK")
print(f"row_count={len(rows)}")
print(f"episode_rows={len(episode_rows)}")
print(f"summary_rows={len(summary_rows)}")
print(f"first_episode_seed={episode_seeds[0]}")
print(f"last_episode_seed={episode_seeds[-1]}")
PY
}

generate_package_checksum_manifest() {
  local checksum_manifest_path="${METADATA_DIR}/package_file_checksums.sha256"
  python - "${TASK_RUN_DIR}" "${checksum_manifest_path}" <<'PY'
import hashlib
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
checksum_manifest_path = Path(sys.argv[2])

package_files = sorted(path for path in run_dir.rglob("*") if path.is_file())
with checksum_manifest_path.open("w", encoding="utf-8") as checksum_manifest:
    for package_file in package_files:
        if package_file == checksum_manifest_path:
            continue
        digest = hashlib.sha256(package_file.read_bytes()).hexdigest()
        relative_path = package_file.relative_to(run_dir).as_posix()
        checksum_manifest.write(f"{digest}  {relative_path}\n")
PY
}

package_task() {
  generate_package_checksum_manifest
  tar -czf "${PACKAGE_PATH}" -C "${TASK_RUN_DIR}" .
  tar -tzf "${PACKAGE_PATH}" >/dev/null
  echo "TASK_PACKAGE_OK"
  echo "package_path=${PACKAGE_PATH}"
}

if [[ "${EV_GNN_FORMAL_DRY_RUN:-0}" == "1" ]]; then
  cd "${LOCAL_REPO_ROOT}"
  MAX_EPISODE_STEPS="$(resolve_simulation_length "${CONFIG_PATH}")"
  build_train_command
  build_eval_command
  echo "CONTROLLED_FORMAL_TASK_START"
  echo "dry_run=1"
  echo "TASK_MAPPING"
  print_task_mapping
  echo "max_episode_steps=${MAX_EPISODE_STEPS}"
  echo "TRAINING_COMMAND"
  echo "${TRAIN_COMMAND_TEXT}"
  echo "EVALUATION_COMMAND"
  echo "${EVAL_COMMAND_TEXT}"
  echo "expected_package_path_pattern=${OUTPUT_ROOT}/m3_controlled_multiscale_formal_${SCALE}_${ALGORITHM}_seed${SEED}_job<jobid>_task${TASK_ID}.tar.gz"
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

mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}" "${METADATA_DIR}" "${CONFIG_COPY_DIR}" "${OUTPUT_ROOT}"
exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

cd "${REPO_ROOT}"

MAX_EPISODE_STEPS="$(resolve_simulation_length "${CONFIG_PATH}")"
build_train_command
build_eval_command

TASK_START_EPOCH="$(date +%s)"
TASK_START_ISO="$(iso_now)"

echo "CONTROLLED_FORMAL_TASK_START"
echo "TASK_MAPPING"
print_task_mapping
echo "start_timestamp=${TASK_START_ISO}"
echo "hostname=$(hostname)"
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
echo "formal_eval_episodes=${FORMAL_EVAL_EPISODES}"
echo "deterministic=${DETERMINISTIC}"
echo "eval_expl_noise=${EVAL_EXPL_NOISE}"
echo "max_episode_steps=${MAX_EPISODE_STEPS}"

validate_required_files_exist "${CONFIG_PATH}" train_td3_gnn.py evaluate_td3_gnn.py
cp "${CONFIG_PATH}" "${CONFIG_COPY_PATH}"
generate_source_manifest
write_runtime_metadata

echo "TRAINING_COMMAND"
echo "${TRAIN_COMMAND_TEXT}" | tee "${METADATA_DIR}/training_command.txt"

TRAIN_START_EPOCH="$(date +%s)"
set +e
"${TRAIN_COMMAND[@]}"
TRAIN_EXIT_STATUS="$?"
set -e
TRAIN_END_EPOCH="$(date +%s)"
TRAIN_WALL_SECONDS=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))

{
  echo "training_exit_status=${TRAIN_EXIT_STATUS}"
  echo "training_wall_time_seconds=${TRAIN_WALL_SECONDS}"
  echo "training_wall_time_hhmmss=$(format_seconds "${TRAIN_WALL_SECONDS}")"
} > "${METADATA_DIR}/runtime_training.env"
write_runtime_metadata

if [[ "${TRAIN_EXIT_STATUS}" -ne 0 ]]; then
  echo "ERROR: train command exited non-zero with status ${TRAIN_EXIT_STATUS}" >&2
  exit "${TRAIN_EXIT_STATUS}"
fi

echo "=== Validate training artefacts ==="
validate_required_files_exist \
  "${MODEL_DIR}/model.best_actor" \
  "${MODEL_DIR}/model.best_critic" \
  "${MODEL_DIR}/config.yaml" \
  "${MODEL_DIR}/kwargs.yaml" \
  "${MODEL_DIR}/run_args.yaml" \
  "${MODEL_DIR}/training_log.csv"

echo "EVALUATION_COMMAND"
echo "${EVAL_COMMAND_TEXT}" | tee "${METADATA_DIR}/evaluation_command.txt"

EVAL_START_EPOCH="$(date +%s)"
set +e
"${EVAL_COMMAND[@]}"
EVAL_EXIT_STATUS="$?"
set -e
EVAL_END_EPOCH="$(date +%s)"
EVAL_WALL_SECONDS=$((EVAL_END_EPOCH - EVAL_START_EPOCH))

{
  echo "evaluation_exit_status=${EVAL_EXIT_STATUS}"
  echo "evaluation_wall_time_seconds=${EVAL_WALL_SECONDS}"
  echo "evaluation_wall_time_hhmmss=$(format_seconds "${EVAL_WALL_SECONDS}")"
} > "${METADATA_DIR}/runtime_evaluation.env"
write_runtime_metadata

if [[ "${EVAL_EXIT_STATUS}" -ne 0 ]]; then
  echo "ERROR: eval command exited non-zero with status ${EVAL_EXIT_STATUS}" >&2
  exit "${EVAL_EXIT_STATUS}"
fi

validate_eval30_csv
validate_required_files_exist "${EVAL_CSV}"

TASK_END_EPOCH="$(date +%s)"
TASK_END_ISO="$(iso_now)"
TASK_WALL_SECONDS=$((TASK_END_EPOCH - TASK_START_EPOCH))

{
  echo "task_wall_time_seconds=${TASK_WALL_SECONDS}"
  echo "task_wall_time_hhmmss=$(format_seconds "${TASK_WALL_SECONDS}")"
} > "${METADATA_DIR}/runtime_final.env"
write_runtime_metadata

package_task

echo "CONTROLLED_FORMAL_TASK_COMPLETED"
echo "task_wall_time=$(format_seconds "${TASK_WALL_SECONDS}")"
echo "package_path=${PACKAGE_PATH}"
