#!/bin/bash
#SBATCH --job-name=evgnn_500cp_10k_pilot
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

# Runtime pilot only: reward rows are smoke information, not formal behavioural evidence.

THREAD_COUNT="${SLURM_CPUS_PER_TASK:-4}"
if [[ "${THREAD_COUNT}" != "4" ]]; then
  echo "This pilot is controlled for CPU4 only; got SLURM_CPUS_PER_TASK=${THREAD_COUNT}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${THREAD_COUNT}"
export MKL_NUM_THREADS="${THREAD_COUNT}"
export NUMEXPR_NUM_THREADS="${THREAD_COUNT}"

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
OUT_DIR="/projects/fr57/cche0357/EV-GNN_outputs"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/500cp_runtime_pilot_10k"

CONFIG_PATH="./config_files/PublicPST_500.yaml"
TRAIN_SCRIPT="train_td3_gnn.py"
SEED=0

MAX_TIMESTEPS="${EVGNN_500CP_RUNTIME_PILOT_MAX_TIMESTEPS:-${EVGNN_RUNTIME_PILOT_MAX_TIMESTEPS:-10000}}"
START_TIMESTEPS=1000
EVAL_FREQ=5000
TRAIN_EVAL_EPISODES=5
BATCH_SIZE=64
REPLAY_BUFFER_SIZE=100000
DEVICE="cpu"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${TASK_ID}" == "0" ]]; then
  ALGORITHM="actiongnn"
  ALGORITHM_LABEL="baseline"
elif [[ "${TASK_ID}" == "1" ]]; then
  ALGORITHM="hierarchical"
  ALGORITHM_LABEL="hierarchical"
else
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}" >&2
  exit 2
fi

if ! [[ "${MAX_TIMESTEPS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_TIMESTEPS must be a positive integer; got ${MAX_TIMESTEPS}" >&2
  exit 2
fi
if (( MAX_TIMESTEPS <= 0 )); then
  echo "MAX_TIMESTEPS must be positive; got ${MAX_TIMESTEPS}" >&2
  exit 2
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_NAME="${SLURM_JOB_NAME:-evgnn_500cp_10k_pilot}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
RUN_NAME="m3_500cp_runtime_pilot_${ALGORITHM_LABEL}_${ALGORITHM}_cpu4_seed${SEED}_${MAX_TIMESTEPS}steps_job${ARRAY_JOB_ID}_${TASK_ID}"
RUN_DIR="${RUN_ROOT}/${ALGORITHM_LABEL}/${RUN_NAME}"
SAVE_DIR="${RUN_DIR}/saved_models"
MODEL_DIR="${SAVE_DIR}/${RUN_NAME}"
METADATA_DIR="${RUN_DIR}/runtime_metadata"
PACKAGE_PATH="${OUT_DIR}/${RUN_NAME}.tar.gz"
STDOUT_LOG="${RUN_DIR}/stdout.log"
STDERR_LOG="${RUN_DIR}/stderr.log"
SLURM_STDOUT_PATH="${OUT_DIR}/${JOB_NAME}_${ARRAY_JOB_ID}_${TASK_ID}.out"
SLURM_STDERR_PATH="${OUT_DIR}/${JOB_NAME}_${ARRAY_JOB_ID}_${TASK_ID}.err"

format_seconds() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

package_run() {
  if [[ -f "${SLURM_STDOUT_PATH}" ]]; then
    cp "${SLURM_STDOUT_PATH}" "${METADATA_DIR}/slurm_stdout_snapshot.out" || true
  fi
  if [[ -f "${SLURM_STDERR_PATH}" ]]; then
    cp "${SLURM_STDERR_PATH}" "${METADATA_DIR}/slurm_stderr_snapshot.err" || true
  fi
  find "${RUN_DIR}" -maxdepth 5 -type f | sort > "${METADATA_DIR}/packaged_files.txt"
  tar -czf "${PACKAGE_PATH}" -C "${RUN_DIR}" .
  ls -lh "${PACKAGE_PATH}"
}

mkdir -p "${SAVE_DIR}" "${METADATA_DIR}" "${OUT_DIR}"

exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

cd "${REPO_DIR}"

JOB_START_EPOCH="$(date +%s)"

echo "=== M3 500CP baseline-vs-hierarchical 10k runtime pilot ==="
echo "Date: $(date -Is)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-}"
echo "Array task ID: ${TASK_ID}"
echo "Job name: ${JOB_NAME}"
echo "Node list: ${SLURM_JOB_NODELIST:-}"
echo "Node ID: ${SLURM_NODEID:-}"
echo "CPU count requested: ${THREAD_COUNT}"
echo "CPU count online: $(getconf _NPROCESSORS_ONLN || true)"
echo "Algorithm label: ${ALGORITHM_LABEL}"
echo "Algorithm: ${ALGORITHM}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Config: ${CONFIG_PATH}"
echo "Seed: ${SEED}"
echo "Max timesteps: ${MAX_TIMESTEPS}"
echo "Start timesteps: ${START_TIMESTEPS}"
echo "Eval frequency: ${EVAL_FREQ}"
echo "Train eval episodes: ${TRAIN_EVAL_EPISODES}"
echo "Batch size: ${BATCH_SIZE}"
echo "Replay buffer size: ${REPLAY_BUFFER_SIZE}"
echo "Device: ${DEVICE}"
echo "Run name: ${RUN_NAME}"
echo "Run directory: ${RUN_DIR}"
echo "Model directory: ${MODEL_DIR}"
echo "Package path: ${PACKAGE_PATH}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS}"
echo "Python: $(which python)"
python --version
echo "MaxRSS note: MaxRSS is not known inside the running job; use sacct/jobstats after completion."
echo "Source update note: this script does not run git pull; validate source with EV_GNN_SOURCE_MANIFEST."
echo "Reward note: pilot rewards are smoke information only, not formal behavioural evidence."

{
  echo "date_start=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "slurm_array_job_id=${SLURM_ARRAY_JOB_ID:-}"
  echo "slurm_array_task_id=${TASK_ID}"
  echo "slurm_job_name=${JOB_NAME}"
  echo "slurm_job_nodelist=${SLURM_JOB_NODELIST:-}"
  echo "slurm_node_id=${SLURM_NODEID:-}"
  echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
  echo "online_cpu_count=$(getconf _NPROCESSORS_ONLN || true)"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "mkl_num_threads=${MKL_NUM_THREADS}"
  echo "numexpr_num_threads=${NUMEXPR_NUM_THREADS}"
  echo "repo_dir=${REPO_DIR}"
  echo "config_path=${CONFIG_PATH}"
  echo "algorithm_label=${ALGORITHM_LABEL}"
  echo "algorithm=${ALGORITHM}"
  echo "seed=${SEED}"
  echo "total_training_steps=${MAX_TIMESTEPS}"
  echo "train_script=${TRAIN_SCRIPT}"
  echo "device=${DEVICE}"
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "save_dir=${SAVE_DIR}"
  echo "model_dir=${MODEL_DIR}"
  echo "stdout_log=${STDOUT_LOG}"
  echo "stderr_log=${STDERR_LOG}"
  echo "slurm_stdout_path=${SLURM_STDOUT_PATH}"
  echo "slurm_stderr_path=${SLURM_STDERR_PATH}"
  echo "package_path=${PACKAGE_PATH}"
  echo "maxrss_note=MaxRSS unavailable inside job; query sacct/jobstats after completion."
} | tee "${METADATA_DIR}/runtime_start.txt"

{
  echo "source_provenance=${EV_GNN_SOURCE_PROVENANCE:-unprovided}"
  echo "source_manifest=${EV_GNN_SOURCE_MANIFEST:-unprovided}"
  echo "source_update_note=No git pull is performed by this job; validate M3 source from the manifest."
} | tee "${METADATA_DIR}/source_provenance.txt"

if [[ -n "${EV_GNN_SOURCE_MANIFEST:-}" ]]; then
  if [[ ! -s "${EV_GNN_SOURCE_MANIFEST}" ]]; then
    echo "EV_GNN_SOURCE_MANIFEST is set but missing or empty: ${EV_GNN_SOURCE_MANIFEST}" >&2
    package_run
    exit 2
  fi
  cp "${EV_GNN_SOURCE_MANIFEST}" "${METADATA_DIR}/source_manifest.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${METADATA_DIR}/source_manifest.txt" | tee "${METADATA_DIR}/source_manifest.sha256"
  fi
else
  echo "WARNING: EV_GNN_SOURCE_MANIFEST not provided; source manifest validation skipped." >&2
fi

echo "=== Dependency preflight ==="
python - <<'PY' | tee "${METADATA_DIR}/python_environment.txt"
import platform
import sys

import torch
import torch_geometric

print(f"python_executable={sys.executable}")
print(f"python_version={platform.python_version()}")
print(f"pytorch_version={torch.__version__}")
print(f"pytorch_cuda_version={torch.version.cuda}")
print(f"pytorch_cuda_available={torch.cuda.is_available()}")
print(f"pytorch_num_threads={torch.get_num_threads()}")
print(f"pytorch_num_interop_threads={torch.get_num_interop_threads()}")
print(f"pyg_version={torch_geometric.__version__}")
PY
python -c 'from ev2gym.models.ev2gym_env import EV2Gym; print("EV2GYM_ENV_IMPORT_OK")'

echo "=== Training command ==="
printf 'python %q --algorithm %q --config %q --seed %q --device %q --run_name %q --max_timesteps %q --start_timesteps %q --eval_freq %q --eval_episodes %q --batch_size %q --replay_buffer_size %q --save_dir %q --log_to_wandb false\n' \
  "${TRAIN_SCRIPT}" \
  "${ALGORITHM}" \
  "${CONFIG_PATH}" \
  "${SEED}" \
  "${DEVICE}" \
  "${RUN_NAME}" \
  "${MAX_TIMESTEPS}" \
  "${START_TIMESTEPS}" \
  "${EVAL_FREQ}" \
  "${TRAIN_EVAL_EPISODES}" \
  "${BATCH_SIZE}" \
  "${REPLAY_BUFFER_SIZE}" \
  "${SAVE_DIR}" | tee "${METADATA_DIR}/training_command.txt"

TRAIN_START_EPOCH="$(date +%s)"
TRAIN_START_ISO="$(date -Is)"
set +e
python "${TRAIN_SCRIPT}" \
  --algorithm "${ALGORITHM}" \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --run_name "${RUN_NAME}" \
  --max_timesteps "${MAX_TIMESTEPS}" \
  --start_timesteps "${START_TIMESTEPS}" \
  --eval_freq "${EVAL_FREQ}" \
  --eval_episodes "${TRAIN_EVAL_EPISODES}" \
  --batch_size "${BATCH_SIZE}" \
  --replay_buffer_size "${REPLAY_BUFFER_SIZE}" \
  --save_dir "${SAVE_DIR}" \
  --log_to_wandb false
TRAIN_STATUS="$?"
set -e
TRAIN_END_EPOCH="$(date +%s)"
TRAIN_END_ISO="$(date -Is)"
TRAIN_WALL_SECONDS=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))
JOB_WALL_SECONDS=$((TRAIN_END_EPOCH - JOB_START_EPOCH))

{
  echo "date_train_start=${TRAIN_START_ISO}"
  echo "date_train_end=${TRAIN_END_ISO}"
  echo "date_end=$(date -Is)"
  echo "training_exit_status=${TRAIN_STATUS}"
  echo "training_wall_time_seconds=${TRAIN_WALL_SECONDS}"
  echo "training_wall_time_hhmmss=$(format_seconds "${TRAIN_WALL_SECONDS}")"
  echo "job_elapsed_seconds=${JOB_WALL_SECONDS}"
  echo "job_elapsed_hhmmss=$(format_seconds "${JOB_WALL_SECONDS}")"
  echo "total_training_steps=${MAX_TIMESTEPS}"
  echo "model_dir=${MODEL_DIR}"
  echo "training_log=${MODEL_DIR}/training_log.csv"
  echo "run_args=${MODEL_DIR}/run_args.yaml"
  echo "copied_config=${MODEL_DIR}/config.yaml"
  echo "package_path=${PACKAGE_PATH}"
  echo "maxrss_note=MaxRSS unavailable inside job; query sacct/jobstats after completion."
} | tee "${METADATA_DIR}/runtime_final.txt"

if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
  echo "Training command failed with status ${TRAIN_STATUS}; packaging available logs and metadata." >&2
  package_run
  exit "${TRAIN_STATUS}"
fi

echo "=== Verify generated 500CP runtime-pilot artefacts ==="
REQUIRED_FILES=(
  "${MODEL_DIR}/config.yaml"
  "${MODEL_DIR}/kwargs.yaml"
  "${MODEL_DIR}/run_args.yaml"
  "${MODEL_DIR}/training_log.csv"
  "${MODEL_DIR}/model.best_actor"
  "${MODEL_DIR}/model.best_critic"
  "${MODEL_DIR}/model.last_actor"
  "${MODEL_DIR}/model.last_critic"
  "${METADATA_DIR}/runtime_start.txt"
  "${METADATA_DIR}/runtime_final.txt"
  "${METADATA_DIR}/source_provenance.txt"
  "${METADATA_DIR}/python_environment.txt"
  "${METADATA_DIR}/training_command.txt"
  "${STDOUT_LOG}"
)

for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing or empty required file: ${path}" >&2
    package_run
    exit 3
  fi
  echo "OK: ${path}"
done

if [[ ! -f "${STDERR_LOG}" ]]; then
  echo "Missing stderr log file: ${STDERR_LOG}" >&2
  package_run
  exit 3
fi
echo "OK: ${STDERR_LOG}"

echo "=== Generated files ==="
find "${RUN_DIR}" -maxdepth 5 -type f | sort

echo "=== Package 500CP runtime pilot artefacts ==="
package_run

echo "Training wall time: $(format_seconds "${TRAIN_WALL_SECONDS}") (${TRAIN_WALL_SECONDS}s)"
echo "Output package: ${PACKAGE_PATH}"
echo "=== M3 500CP ${ALGORITHM_LABEL} runtime pilot completed ==="
