#!/bin/bash
#SBATCH --job-name=evgnn_actor_breakdown
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.err

set -euo pipefail

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/hierarchical_actor_breakdown"
PACKAGE_ROOT="/projects/fr57/cche0357/EV-GNN_outputs"
PROFILE_SCRIPT="profiling/profile_hierarchical_actor_breakdown.py"

SEED=0
DEVICE="cpu"
WARMUP_REPETITIONS=3
MEASURED_REPETITIONS=20
REPLAY_BUFFER_SIZE=100000
MAX_COLLECTION_STEPS=5000

JOB_TAG="${SLURM_JOB_ID}"
OUTPUT_DIR="${RUN_ROOT}/m3_hierarchical_actor_breakdown_seed${SEED}_${JOB_TAG}"
PACKAGE_PATH="${PACKAGE_ROOT}/m3_hierarchical_actor_breakdown_seed${SEED}_${JOB_TAG}.tar.gz"

mkdir -p "${OUTPUT_DIR}" "${PACKAGE_ROOT}"

cd "${REPO_DIR}"

echo "=== M3 hierarchical actor breakdown profile ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Repository: ${REPO_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Package path: ${PACKAGE_PATH}"
echo "Seed: ${SEED}"
echo "Device: ${DEVICE}"
echo "Warm-up repetitions: ${WARMUP_REPETITIONS}"
echo "Measured repetitions: ${MEASURED_REPETITIONS}"
echo "Replay buffer size: ${REPLAY_BUFFER_SIZE}"
echo "Max collection steps: ${MAX_COLLECTION_STEPS}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS}"
echo "Python: $(which python)"
python --version

echo "=== Dependency preflight ==="
python -c 'import torch, torch_geometric; print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("pyg", torch_geometric.__version__)'
python -c 'from ev2gym.models.ev2gym_env import EV2Gym; print("EV2GYM_ENV_IMPORT_OK")'
python -c 'from utils.ev2gym_training_utils import make_env, reset_env, normalise_step_result, resolve_device, set_global_seed; from utils.state_public_pst_gnn import PublicPST_GNN; from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer; from TD3.TD3_ActionGNN_Controlled import TD3_ActionGNN; from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN; print("KEY_IMPORTS_OK")'

GENERATED_FILES=()

run_breakdown() {
  local scale_name="$1"
  local config_path="$2"
  local run_name="m3_${scale_name}_hierarchical_actor_breakdown_seed${SEED}_${JOB_TAG}"
  local csv_path="${OUTPUT_DIR}/${run_name}_hierarchical_actor_breakdown_steps.csv"
  local json_path="${OUTPUT_DIR}/${run_name}_hierarchical_actor_breakdown_summary.json"

  echo "=== Run ${scale_name} actor breakdown ==="
  echo "Config: ${config_path}"
  echo "Run name: ${run_name}"

  python "${PROFILE_SCRIPT}" \
    --config "${config_path}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${run_name}" \
    --batch_sizes 1 64 \
    --warmup_repetitions "${WARMUP_REPETITIONS}" \
    --measured_repetitions "${MEASURED_REPETITIONS}" \
    --replay_buffer_size "${REPLAY_BUFFER_SIZE}" \
    --max_collection_steps "${MAX_COLLECTION_STEPS}" \
    --log_every 500

  echo "=== Verify ${scale_name} outputs ==="
  if [[ ! -s "${csv_path}" ]]; then
    echo "Missing or empty CSV output: ${csv_path}"
    exit 3
  fi
  echo "OK: ${csv_path}"

  if [[ ! -s "${json_path}" ]]; then
    echo "Missing or empty JSON output: ${json_path}"
    exit 3
  fi
  echo "OK: ${json_path}"

  GENERATED_FILES+=("$(basename "${csv_path}")")
  GENERATED_FILES+=("$(basename "${json_path}")")
}

run_breakdown "100cp" "./config_files/PublicPST_100.yaml"
run_breakdown "500cp" "./config_files/PublicPST_500.yaml"

echo "=== Generated profiling files ==="
find "${OUTPUT_DIR}" -maxdepth 1 -type f | sort

echo "=== Package CSV and JSON outputs only ==="
tar -czf "${PACKAGE_PATH}" -C "${OUTPUT_DIR}" "${GENERATED_FILES[@]}"

if [[ ! -s "${PACKAGE_PATH}" ]]; then
  echo "Missing or empty package output: ${PACKAGE_PATH}"
  exit 4
fi
ls -lh "${PACKAGE_PATH}"

echo "=== M3 hierarchical actor breakdown profile completed ==="
