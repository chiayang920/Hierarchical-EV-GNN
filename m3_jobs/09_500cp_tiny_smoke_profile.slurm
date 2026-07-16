#!/bin/bash
#SBATCH --job-name=evgnn_500cp_tiny_profile
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/profiling_500cp_tiny"
PACKAGE_ROOT="/projects/fr57/cche0357/EV-GNN_outputs"

CONFIG_PATH="./config_files/PublicPST_500.yaml"
PROFILE_SCRIPT="profiling/profile_td3_gnn_runtime.py"
SEED=0
PROFILE_STEPS=200
START_TIMESTEPS=50
BATCH_SIZE=64
REPLAY_BUFFER_SIZE=100000
DEVICE="cpu"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
if [[ "${TASK_ID}" == "0" ]]; then
  ALGORITHM="actiongnn"
elif [[ "${TASK_ID}" == "1" ]]; then
  ALGORITHM="hierarchical"
else
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}"
  exit 2
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_NAME="m3_500cp_tiny_smoke_profile_${ALGORITHM}_seed${SEED}_${JOB_TAG}"
OUTPUT_DIR="${RUN_ROOT}/${RUN_NAME}"
CSV_PATH="${OUTPUT_DIR}/${RUN_NAME}_runtime_profile_steps.csv"
JSON_PATH="${OUTPUT_DIR}/${RUN_NAME}_runtime_profile_summary.json"
PACKAGE_PATH="${PACKAGE_ROOT}/m3_500cp_tiny_smoke_profile_${ALGORITHM}_seed${SEED}_${JOB_TAG}.tar.gz"

mkdir -p "${OUTPUT_DIR}" "${PACKAGE_ROOT}"

cd "${REPO_DIR}"

echo "=== M3 500CP tiny TD3-GNN smoke profile ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Algorithm: ${ALGORITHM}"
echo "Run name: ${RUN_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Seed: ${SEED}"
echo "Profile steps: ${PROFILE_STEPS}"
echo "Start timesteps: ${START_TIMESTEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Replay buffer size: ${REPLAY_BUFFER_SIZE}"
echo "Device: ${DEVICE}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "Python: $(which python)"
python --version

echo "=== Dependency preflight ==="
python -c 'import torch, torch_geometric; print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("pyg", torch_geometric.__version__)'
python -c 'from ev2gym.models.ev2gym_env import EV2Gym; print("EV2GYM_ENV_IMPORT_OK")'
python -c 'from utils.ev2gym_training_utils import make_env, reset_env, normalise_step_result, resolve_device, set_global_seed; from utils.state_public_pst_gnn import PublicPST_GNN; from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer; from TD3.TD3_ActionGNN_Controlled import TD3_ActionGNN; from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN; print("KEY_IMPORTS_OK")'

echo "=== Run tiny smoke profiler ==="
python "${PROFILE_SCRIPT}" \
  --algorithm "${ALGORITHM}" \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --profile_steps "${PROFILE_STEPS}" \
  --start_timesteps "${START_TIMESTEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --replay_buffer_size "${REPLAY_BUFFER_SIZE}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --eval_episodes 0 \
  --eval_freq 0 \
  --log_every 50

echo "=== Verify profiling outputs ==="
if [[ ! -s "${CSV_PATH}" ]]; then
  echo "Missing or empty CSV output: ${CSV_PATH}"
  exit 3
fi
echo "OK: ${CSV_PATH}"

if [[ ! -s "${JSON_PATH}" ]]; then
  echo "Missing or empty JSON output: ${JSON_PATH}"
  exit 3
fi
echo "OK: ${JSON_PATH}"

echo "=== Generated profiling files ==="
find "${OUTPUT_DIR}" -maxdepth 1 -type f | sort

echo "=== Package profiling output ==="
tar -czf "${PACKAGE_PATH}" -C "${OUTPUT_DIR}" .
ls -lh "${PACKAGE_PATH}"

echo "=== M3 500CP tiny smoke profile completed ==="
