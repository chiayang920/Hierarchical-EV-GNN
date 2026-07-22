#!/bin/bash
#SBATCH --job-name=evgnn_500cp_eval30
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-9
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail
set -E

if [[ "${SLURM_CPUS_PER_TASK:-}" != "4" ]]; then
  echo "ERROR: expected SLURM_CPUS_PER_TASK=4, got '${SLURM_CPUS_PER_TASK:-unset}'" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
OUT_DIR="/projects/fr57/cche0357/EV-GNN_outputs"
TRAIN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/500cp_formal_train_50k"
EVAL_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/500cp_formal_eval30"
CANONICAL_EVAL_DIR="${EVAL_ROOT}/eval"

CONFIG_PATH="./config_files/PublicPST_500.yaml"
EVAL_EPISODES=30
MAX_EPISODE_STEPS=112
DEVICE="cpu"
DETERMINISTIC="true"
EVAL_EXPL_NOISE="0.0"
BASE_EVAL_SEED=520000
REQUESTED_MEMORY="32G"
REQUESTED_WALL_TIME="06:00:00"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
SLURM_JOB_ID_VALUE="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

case "${TASK_ID}" in
  0)
    ALGORITHM="actiongnn"
    ALGORITHM_LABEL="baseline"
    SEED=0
    TRAIN_ARRAY_JOB_ID="58417948"
    TRAIN_TASK_ID=0
    ;;
  1)
    ALGORITHM="actiongnn"
    ALGORITHM_LABEL="baseline"
    SEED=1
    TRAIN_ARRAY_JOB_ID="58441541"
    TRAIN_TASK_ID=1
    ;;
  2)
    ALGORITHM="actiongnn"
    ALGORITHM_LABEL="baseline"
    SEED=2
    TRAIN_ARRAY_JOB_ID="58458540"
    TRAIN_TASK_ID=2
    ;;
  3)
    ALGORITHM="actiongnn"
    ALGORITHM_LABEL="baseline"
    SEED=3
    TRAIN_ARRAY_JOB_ID="58441541"
    TRAIN_TASK_ID=3
    ;;
  4)
    ALGORITHM="actiongnn"
    ALGORITHM_LABEL="baseline"
    SEED=4
    TRAIN_ARRAY_JOB_ID="58441541"
    TRAIN_TASK_ID=4
    ;;
  5)
    ALGORITHM="hierarchical"
    ALGORITHM_LABEL="hierarchical"
    SEED=0
    TRAIN_ARRAY_JOB_ID="58417948"
    TRAIN_TASK_ID=5
    ;;
  6)
    ALGORITHM="hierarchical"
    ALGORITHM_LABEL="hierarchical"
    SEED=1
    TRAIN_ARRAY_JOB_ID="58441541"
    TRAIN_TASK_ID=6
    ;;
  7)
    ALGORITHM="hierarchical"
    ALGORITHM_LABEL="hierarchical"
    SEED=2
    TRAIN_ARRAY_JOB_ID="58441541"
    TRAIN_TASK_ID=7
    ;;
  8)
    ALGORITHM="hierarchical"
    ALGORITHM_LABEL="hierarchical"
    SEED=3
    TRAIN_ARRAY_JOB_ID="58441541"
    TRAIN_TASK_ID=8
    ;;
  9)
    ALGORITHM="hierarchical"
    ALGORITHM_LABEL="hierarchical"
    SEED=4
    TRAIN_ARRAY_JOB_ID="58441541"
    TRAIN_TASK_ID=9
    ;;
  *)
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-9" >&2
    exit 2
    ;;
esac

TRAIN_RUN_NAME="m3_500cp_formal_train_${ALGORITHM_LABEL}_${ALGORITHM}_cpu4_seed${SEED}_50000steps_job${TRAIN_ARRAY_JOB_ID}_task${TRAIN_TASK_ID}"
MODEL_DIR="${TRAIN_ROOT}/${ALGORITHM_LABEL}/seed${SEED}/${TRAIN_RUN_NAME}/saved_models/${TRAIN_RUN_NAME}"
CHECKPOINT_PREFIX="${MODEL_DIR}/model.best"

EVAL_SEED_OFFSET=$((BASE_EVAL_SEED + SEED * 1000 - SEED))
EXPECTED_EPISODE_SEED_MIN=$((SEED + EVAL_SEED_OFFSET))
EXPECTED_EPISODE_SEED_MAX=$((EXPECTED_EPISODE_SEED_MIN + EVAL_EPISODES - 1))

EVAL_RUN_NAME="m3_500cp_formal_eval30_${ALGORITHM}_seed${SEED}_job${ARRAY_JOB_ID}_task${TASK_ID}"
TASK_DIR="${EVAL_ROOT}/tasks/${ALGORITHM}/seed${SEED}/${EVAL_RUN_NAME}"
METADATA_DIR="${TASK_DIR}/runtime_metadata"
RAW_OUTPUT_CSV="${TASK_DIR}/${ALGORITHM}_500cp_seed${SEED}_eval30.csv"
CANONICAL_OUTPUT_CSV="${CANONICAL_EVAL_DIR}/${ALGORITHM}_500cp_seed${SEED}_eval30.csv"
PACKAGE_PATH="${OUT_DIR}/m3_500cp_formal_eval30_${ALGORITHM}_seed${SEED}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
VALIDATION_SUMMARY_ENV="${METADATA_DIR}/csv_validation_summary.env"

TASK_DIR_CREATED=0
CANONICAL_CREATED=0
CANONICAL_TEMP_PATH="${CANONICAL_OUTPUT_CSV}.tmp.${ARRAY_JOB_ID}.${TASK_ID}"

failure_trap() {
  local exit_status="$?"
  trap - EXIT
  set +e
  if [[ "${exit_status}" -ne 0 && "${TASK_DIR_CREATED:-0}" == "1" ]]; then
    echo "ERROR: M3 500CP formal eval30 task failed with exit status ${exit_status}" >&2
    if [[ -e "${CANONICAL_TEMP_PATH}" ]]; then
      rm -f "${CANONICAL_TEMP_PATH}"
      echo "Rolled back temporary canonical CSV: ${CANONICAL_TEMP_PATH}" >&2
    fi
    if [[ "${CANONICAL_CREATED:-0}" == "1" ]]; then
      rm -f "${CANONICAL_OUTPUT_CSV}"
      echo "Rolled back canonical CSV created by this task: ${CANONICAL_OUTPUT_CSV}" >&2
    fi
    {
      echo "iso_failure_time=$(date -Iseconds)"
      echo "exit_status=${exit_status}"
      echo "algorithm=${ALGORITHM}"
      echo "seed=${SEED}"
      echo "task_dir=${TASK_DIR}"
      echo "package_path=${PACKAGE_PATH}"
    } >> "${METADATA_DIR}/runtime_failure.txt"
    mkdir -p "${OUT_DIR}"
    tar -czf "${PACKAGE_PATH}" -C "${TASK_DIR}" .
    if tar -tzf "${PACKAGE_PATH}" >/dev/null; then
      echo "Packaged available failure artifacts: ${PACKAGE_PATH}" >&2
    else
      echo "ERROR: failed to verify failure artifact package: ${PACKAGE_PATH}" >&2
    fi
  fi
  exit "${exit_status}"
}

trap failure_trap EXIT

mkdir -p "${TASK_DIR}" "${METADATA_DIR}" "${CANONICAL_EVAL_DIR}" "${OUT_DIR}"
TASK_DIR_CREATED=1

exec > >(tee -a "${TASK_DIR}/stdout.log") 2> >(tee -a "${TASK_DIR}/stderr.log" >&2)

cd "${REPO_DIR}"

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

if [[ -z "${EV_GNN_SOURCE_PROVENANCE:-}" ]]; then
  echo "ERROR: EV_GNN_SOURCE_PROVENANCE is required for formal eval30 provenance" >&2
  exit 3
fi

if [[ -z "${EV_GNN_SOURCE_MANIFEST:-}" ]]; then
  echo "ERROR: EV_GNN_SOURCE_MANIFEST is required for formal eval30 provenance" >&2
  exit 3
fi

if [[ ! -s "${EV_GNN_SOURCE_MANIFEST}" ]]; then
  echo "ERROR: EV_GNN_SOURCE_MANIFEST is missing or empty: ${EV_GNN_SOURCE_MANIFEST}" >&2
  exit 3
fi

MANIFEST_VALIDATION_PATH="${METADATA_DIR}/source_manifest_validation.txt"

if ! sha256sum -c "${EV_GNN_SOURCE_MANIFEST}" \
  | tee "${MANIFEST_VALIDATION_PATH}"; then
  echo "ERROR: M3 source does not match EV_GNN_SOURCE_MANIFEST" >&2
  exit 3
fi

if [[ -e "${CANONICAL_OUTPUT_CSV}" ]]; then
  echo "ERROR: canonical CSV already exists and will not be overwritten: ${CANONICAL_OUTPUT_CSV}" >&2
  exit 4
fi

START_ISO="$(date -Iseconds)"
START_SECONDS="$(date +%s)"

{
  echo "iso_start_time=${START_ISO}"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID_VALUE}"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "task_id=${TASK_ID}"
  echo "node_list=${SLURM_JOB_NODELIST:-unset}"
  echo "cpu_count=${SLURM_CPUS_PER_TASK}"
  echo "memory_request=${REQUESTED_MEMORY}"
  echo "wall_time_request=${REQUESTED_WALL_TIME}"
  echo "algorithm=${ALGORITHM}"
  echo "algorithm_label=${ALGORITHM_LABEL}"
  echo "training_seed=${SEED}"
  echo "training_array_job=${TRAIN_ARRAY_JOB_ID}"
  echo "training_task=${TRAIN_TASK_ID}"
  echo "training_run_name=${TRAIN_RUN_NAME}"
  echo "model_directory=${MODEL_DIR}"
  echo "checkpoint_prefix=${CHECKPOINT_PREFIX}"
  echo "evaluation_episode_count=${EVAL_EPISODES}"
  echo "deterministic_mode=${DETERMINISTIC}"
  echo "evaluation_seed_offset=${EVAL_SEED_OFFSET}"
  echo "expected_episode_seed_min=${EXPECTED_EPISODE_SEED_MIN}"
  echo "expected_episode_seed_max=${EXPECTED_EPISODE_SEED_MAX}"
  echo "raw_csv_path=${RAW_OUTPUT_CSV}"
  echo "canonical_csv_path=${CANONICAL_OUTPUT_CSV}"
  echo "package_path=${PACKAGE_PATH}"
} > "${METADATA_DIR}/runtime_start.txt"

{
  echo "source_provenance=${EV_GNN_SOURCE_PROVENANCE}"
  echo "source_manifest_original_path=${EV_GNN_SOURCE_MANIFEST}"
  echo "eval_task=${TASK_ID}"
  echo "algorithm=${ALGORITHM}"
  echo "algorithm_label=${ALGORITHM_LABEL}"
  echo "training_seed=${SEED}"
  echo "training_array_job=${TRAIN_ARRAY_JOB_ID}"
  echo "training_task=${TRAIN_TASK_ID}"
  echo "training_run_name=${TRAIN_RUN_NAME}"
  echo "resolved_checkpoint_prefix=${CHECKPOINT_PREFIX}"
} > "${METADATA_DIR}/source_provenance.txt"

cp "${EV_GNN_SOURCE_MANIFEST}" "${METADATA_DIR}/source_manifest.txt"
sha256sum "${METADATA_DIR}/source_manifest.txt" > "${METADATA_DIR}/source_manifest.sha256"

{
  echo "python_executable=$(command -v python)"
  python - <<'PY'
import sys

import numpy as np
import torch
import torch_geometric

print(f"python_version={sys.version.replace(chr(10), ' ')}")
print(f"pytorch_version={torch.__version__}")
print(f"pytorch_cuda_version={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"pytorch_thread_count={torch.get_num_threads()}")
print(f"torch_geometric_version={torch_geometric.__version__}")
print(f"numpy_version={np.__version__}")
PY
} > "${METADATA_DIR}/python_environment.txt"

echo "=== M3 500CP formal eval30 task start ==="
echo "Task: ${TASK_ID}"
echo "Algorithm: ${ALGORITHM}"
echo "Algorithm label: ${ALGORITHM_LABEL}"
echo "Seed: ${SEED}"
echo "Training job/task: ${TRAIN_ARRAY_JOB_ID}_${TRAIN_TASK_ID}"
echo "Training run: ${TRAIN_RUN_NAME}"
echo "Checkpoint prefix: ${CHECKPOINT_PREFIX}"
echo "Expected episode seeds: ${EXPECTED_EPISODE_SEED_MIN}-${EXPECTED_EPISODE_SEED_MAX}"
echo "Raw CSV: ${RAW_OUTPUT_CSV}"
echo "Canonical CSV: ${CANONICAL_OUTPUT_CSV}"
echo "Package: ${PACKAGE_PATH}"

echo "=== Checkpoint and source preflight ==="
REQUIRED_PREFLIGHT_FILES=(
  "${CHECKPOINT_PREFIX}_actor"
  "${CHECKPOINT_PREFIX}_critic"
  "${MODEL_DIR}/config.yaml"
  "${MODEL_DIR}/kwargs.yaml"
  "${MODEL_DIR}/run_args.yaml"
  "${MODEL_DIR}/training_log.csv"
)

for path in "${REQUIRED_PREFLIGHT_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing required file: ${path}" >&2
    exit 5
  fi
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: required file is empty: ${path}" >&2
    exit 5
  fi
  echo "OK: ${path}"
done

for path in "${CONFIG_PATH}" "evaluate_td3_gnn.py"; do
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: missing or empty evaluation source file: ${path}" >&2
    exit 5
  fi
  echo "OK: ${path}"
done

if ! cmp -s "${MODEL_DIR}/config.yaml" "${CONFIG_PATH}"; then
  echo "ERROR: checkpoint config differs from ${CONFIG_PATH}: ${MODEL_DIR}/config.yaml" >&2
  exit 6
fi

sha256sum \
  "${CHECKPOINT_PREFIX}_actor" \
  "${CHECKPOINT_PREFIX}_critic" \
  "${MODEL_DIR}/config.yaml" \
  "${MODEL_DIR}/kwargs.yaml" \
  "${MODEL_DIR}/run_args.yaml" \
  "${MODEL_DIR}/training_log.csv" \
  "${CONFIG_PATH}" \
  "evaluate_td3_gnn.py" \
  > "${METADATA_DIR}/checkpoint_and_eval_source_checksums.sha256"

EVAL_COMMAND=(
  python evaluate_td3_gnn.py
  --algorithm "${ALGORITHM}"
  --config "${CONFIG_PATH}"
  --seed "${SEED}"
  --eval_episodes "${EVAL_EPISODES}"
  --checkpoint "${CHECKPOINT_PREFIX}"
  --device "${DEVICE}"
  --output_csv "${RAW_OUTPUT_CSV}"
  --run_name "${EVAL_RUN_NAME}"
  --max_episode_steps "${MAX_EPISODE_STEPS}"
  --deterministic "${DETERMINISTIC}"
  --eval_expl_noise "${EVAL_EXPL_NOISE}"
  --eval_seed_offset "${EVAL_SEED_OFFSET}"
)

printf '%q ' "${EVAL_COMMAND[@]}" > "${METADATA_DIR}/evaluation_command.txt"
printf '\n' >> "${METADATA_DIR}/evaluation_command.txt"

echo "=== Controlled deterministic eval30 ==="
EVAL_START_SECONDS="$(date +%s)"
set +e
"${EVAL_COMMAND[@]}"
EVAL_EXIT_STATUS="$?"
set -e
EVAL_END_SECONDS="$(date +%s)"
EVAL_WALL_SECONDS=$((EVAL_END_SECONDS - EVAL_START_SECONDS))
EVAL_WALL_FORMATTED="$(printf '%02d:%02d:%02d' $((EVAL_WALL_SECONDS / 3600)) $(((EVAL_WALL_SECONDS % 3600) / 60)) $((EVAL_WALL_SECONDS % 60)))"

if [[ "${EVAL_EXIT_STATUS}" -ne 0 ]]; then
  echo "ERROR: evaluation command failed with exit status ${EVAL_EXIT_STATUS}" >&2
  exit "${EVAL_EXIT_STATUS}"
fi

echo "=== Validate raw eval CSV ==="
python - \
  "${RAW_OUTPUT_CSV}" \
  "${ALGORITHM}" \
  "${SEED}" \
  "${CHECKPOINT_PREFIX}" \
  "${EXPECTED_EPISODE_SEED_MIN}" \
  "${EXPECTED_EPISODE_SEED_MAX}" \
  "${EVAL_EPISODES}" \
  "${MAX_EPISODE_STEPS}" \
  "${VALIDATION_SUMMARY_ENV}" <<'PY'
import csv
import math
import sys
from pathlib import Path

(
    raw_output_csv,
    expected_algorithm,
    expected_seed_text,
    expected_checkpoint,
    expected_seed_min_text,
    expected_seed_max_text,
    expected_episode_count_text,
    expected_episode_steps_text,
    validation_summary_env,
) = sys.argv[1:]

expected_seed = int(expected_seed_text)
expected_seed_min = int(expected_seed_min_text)
expected_seed_max = int(expected_seed_max_text)
expected_episode_count = int(expected_episode_count_text)
expected_episode_steps = int(expected_episode_steps_text)
raw_output_path = Path(raw_output_csv)

required_columns = {
    "run_name",
    "algorithm",
    "config",
    "seed",
    "episode_seed",
    "checkpoint",
    "episode_index",
    "episode_reward",
    "episode_steps",
    "done",
    "mean_reward",
    "std_reward",
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "action_fraction_zero",
    "action_fraction_at_max",
    "active_action_count_mean",
    "row_type",
}

if not raw_output_path.is_file() or raw_output_path.stat().st_size == 0:
    raise SystemExit(f"CSV validation failed: missing or empty CSV: {raw_output_path}")

with raw_output_path.open(newline="") as csv_file:
    reader = csv.DictReader(csv_file)
    if reader.fieldnames is None:
        raise SystemExit("CSV validation failed: CSV has no header")
    missing_columns = sorted(required_columns - set(reader.fieldnames))
    if missing_columns:
        raise SystemExit(f"CSV validation failed: missing columns: {', '.join(missing_columns)}")
    rows = list(reader)

episode_rows = [row for row in rows if row.get("row_type") == "episode"]
summary_rows = [row for row in rows if row.get("row_type") == "summary"]

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
expected_episode_seeds = list(range(expected_seed_min, expected_seed_max + 1))
if episode_seeds != expected_episode_seeds:
    raise SystemExit(
        f"CSV validation failed: episode seeds {episode_seeds} != {expected_episode_seeds}"
    )

for row in episode_rows:
    if int(row["episode_steps"]) != expected_episode_steps:
        raise SystemExit(
            f"CSV validation failed: episode {row['episode_index']} steps {row['episode_steps']} != {expected_episode_steps}"
        )
    if row["done"].strip().lower() != "true":
        raise SystemExit(f"CSV validation failed: episode {row['episode_index']} done is not true")
    if row["algorithm"] != expected_algorithm:
        raise SystemExit(
            f"CSV validation failed: episode {row['episode_index']} algorithm {row['algorithm']} != {expected_algorithm}"
        )
    if int(row["seed"]) != expected_seed:
        raise SystemExit(
            f"CSV validation failed: episode {row['episode_index']} seed {row['seed']} != {expected_seed}"
        )
    if row["checkpoint"] != expected_checkpoint:
        raise SystemExit(
            f"CSV validation failed: episode {row['episode_index']} checkpoint {row['checkpoint']} != {expected_checkpoint}"
        )
    if not math.isfinite(float(row["episode_reward"])):
        raise SystemExit(f"CSV validation failed: episode {row['episode_index']} reward is not finite")

summary_row = summary_rows[0]
if summary_row["algorithm"] != expected_algorithm:
    raise SystemExit(
        f"CSV validation failed: summary algorithm {summary_row['algorithm']} != {expected_algorithm}"
    )
if int(summary_row["seed"]) != expected_seed:
    raise SystemExit(f"CSV validation failed: summary seed {summary_row['seed']} != {expected_seed}")
if summary_row["checkpoint"] != expected_checkpoint:
    raise SystemExit(
        f"CSV validation failed: summary checkpoint {summary_row['checkpoint']} != {expected_checkpoint}"
    )

mean_reward = float(summary_row["mean_reward"])
if not math.isfinite(mean_reward):
    raise SystemExit("CSV validation failed: summary mean_reward is not finite")

episode_seed_min = min(episode_seeds)
episode_seed_max = max(episode_seeds)

print("CSV_VALIDATION_OK")
print(f"algorithm={expected_algorithm}")
print(f"seed={expected_seed}")
print(f"episode_count={len(episode_rows)}")
print(f"episode_seed_min={episode_seed_min}")
print(f"episode_seed_max={episode_seed_max}")
print(f"mean_reward={mean_reward}")

with Path(validation_summary_env).open("w") as summary_file:
    summary_file.write(f"ACTUAL_EPISODE_COUNT={len(episode_rows)}\n")
    summary_file.write(f"ACTUAL_EPISODE_SEED_MIN={episode_seed_min}\n")
    summary_file.write(f"ACTUAL_EPISODE_SEED_MAX={episode_seed_max}\n")
    summary_file.write(f"SUMMARY_MEAN_REWARD={mean_reward}\n")
PY

source "${VALIDATION_SUMMARY_ENV}"

if [[ -e "${CANONICAL_OUTPUT_CSV}" ]]; then
  echo "ERROR: canonical CSV already exists and will not be overwritten: ${CANONICAL_OUTPUT_CSV}" >&2
  exit 7
fi

cp "${RAW_OUTPUT_CSV}" "${CANONICAL_TEMP_PATH}"

if ! cmp -s "${RAW_OUTPUT_CSV}" "${CANONICAL_TEMP_PATH}"; then
  echo "ERROR: temporary canonical CSV differs from raw CSV" >&2
  exit 8
fi

if ! ln "${CANONICAL_TEMP_PATH}" "${CANONICAL_OUTPUT_CSV}"; then
  echo "ERROR: canonical CSV appeared before publication and will not be overwritten: ${CANONICAL_OUTPUT_CSV}" >&2
  exit 7
fi
CANONICAL_CREATED=1
rm -f "${CANONICAL_TEMP_PATH}"

sha256sum "${RAW_OUTPUT_CSV}" "${CANONICAL_OUTPUT_CSV}" > "${METADATA_DIR}/evaluation_csv_checksums.sha256"

if ! cmp -s "${RAW_OUTPUT_CSV}" "${CANONICAL_OUTPUT_CSV}"; then
  echo "ERROR: canonical CSV differs from raw CSV after copy: ${CANONICAL_OUTPUT_CSV}" >&2
  exit 8
fi

END_ISO="$(date -Iseconds)"
{
  echo "iso_end_time=${END_ISO}"
  echo "evaluation_exit_status=${EVAL_EXIT_STATUS}"
  echo "evaluation_wall_time_seconds=${EVAL_WALL_SECONDS}"
  echo "formatted_wall_time=${EVAL_WALL_FORMATTED}"
  echo "algorithm=${ALGORITHM}"
  echo "seed=${SEED}"
  echo "actual_30_episode_seed_range=${ACTUAL_EPISODE_SEED_MIN}-${ACTUAL_EPISODE_SEED_MAX}"
  echo "raw_csv_path=${RAW_OUTPUT_CSV}"
  echo "canonical_csv_path=${CANONICAL_OUTPUT_CSV}"
  echo "package_path=${PACKAGE_PATH}"
  echo "formal_evidence_boundary=eval30 is formal evaluation evidence, but cross-seed superiority requires aggregation."
} > "${METADATA_DIR}/runtime_final.txt"

echo "=== Package eval artefacts ==="
tar -czf "${PACKAGE_PATH}" -C "${TASK_DIR}" .
tar -tzf "${PACKAGE_PATH}" >/dev/null
ls -lh "${PACKAGE_PATH}"

echo "=== M3 500CP formal eval30 task completed ==="
