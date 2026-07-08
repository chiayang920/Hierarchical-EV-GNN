#!/bin/bash
#SBATCH --job-name=evgnn_100cp_gpu_profile
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

module load miniforge3/24.3.0-0
module load cuda/12.2.0
conda activate evgnn_m3_cpu

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
OUT_DIR="/projects/fr57/cche0357/EV-GNN_outputs"
RESOURCE_LABEL="gpu1_4c32g"

CONFIG_PATH="./config_files/PublicPST_100.yaml"
SEED=0
MAX_TIMESTEPS=10000
START_TIMESTEPS=1000
EVAL_FREQ=5000
TRAIN_EVAL_EPISODES=3
FINAL_EVAL_EPISODES=3
BATCH_SIZE=64
REPLAY_BUFFER_SIZE=100000
DEVICE="cuda"

TASK_ID="${SLURM_ARRAY_TASK_ID}"

if [[ "${TASK_ID}" == "0" ]]; then
  ALGORITHM="actiongnn"
  TRAIN_SCRIPT="train_td3_gnn.py"
elif [[ "${TASK_ID}" == "1" ]]; then
  ALGORITHM="hierarchical"
  TRAIN_SCRIPT="train_td3_gnn.py"
else
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}"
  exit 2
fi

JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_profile_10k/${RESOURCE_LABEL}/${ALGORITHM}_seed${SEED}_${JOB_TAG}"
SAVE_DIR="${RUN_ROOT}/saved_models"
EVAL_DIR="${RUN_ROOT}/eval"
PACKAGE_DIR="${RUN_ROOT}/package"

mkdir -p "${SAVE_DIR}" "${EVAL_DIR}" "${PACKAGE_DIR}" "${OUT_DIR}"

cd "${REPO_DIR}"

RUN_NAME="phase2d_profile_${RESOURCE_LABEL}_${ALGORITHM}_100cp_seed${SEED}_10k"

echo "=== Phase 2D.1 100 CP GPU resource-profile pilot ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Resource label: ${RESOURCE_LABEL}"
echo "Algorithm: ${ALGORITHM}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Config: ${CONFIG_PATH}"
echo "Seed: ${SEED}"
echo "Max timesteps: ${MAX_TIMESTEPS}"
echo "Device: ${DEVICE}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "Python: $(which python)"
python --version

echo "=== GPU visibility ==="
nvidia-smi || true

echo "=== Dependency preflight ==="
python -c 'import setuptools; print("SETUPTOOLS", setuptools.__version__)'
python -c 'import pkg_resources; print("PKG_RESOURCES_OK")'
python -c 'from ev2gym.models.ev2gym_env import EV2Gym; print("EV2GYM_ENV_IMPORT_OK")'
python -c 'import torch, torch_geometric, ev2gym; print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("pyg", torch_geometric.__version__); assert torch.cuda.is_available(), "CUDA_AVAILABLE_CHECK_FAILED"'

echo "=== Training ==="
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

CHECKPOINT_PATH="${SAVE_DIR}/${RUN_NAME}/model.best"
OUTPUT_CSV="${EVAL_DIR}/${ALGORITHM}_100cp_seed${SEED}_eval${FINAL_EVAL_EPISODES}.csv"

echo "=== Controlled evaluator ==="
python evaluate_td3_gnn.py \
  --algorithm "${ALGORITHM}" \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --eval_episodes "${FINAL_EVAL_EPISODES}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device "${DEVICE}" \
  --output_csv "${OUTPUT_CSV}" \
  --run_name "${RUN_NAME}_eval${FINAL_EVAL_EPISODES}" \
  --max_episode_steps 112 \
  --deterministic true \
  --eval_seed_offset 410000

echo "=== Generated files ==="
find "${RUN_ROOT}" -maxdepth 5 -type f | sort

echo "=== Package profile artefacts ==="
tar -czf "${OUT_DIR}/phase2d_100cp_profile_${RESOURCE_LABEL}_${ALGORITHM}_${JOB_TAG}.tar.gz" \
  -C "${RUN_ROOT}" .

ls -lh "${OUT_DIR}/phase2d_100cp_profile_${RESOURCE_LABEL}_${ALGORITHM}_${JOB_TAG}.tar.gz"

echo "=== Phase 2D.1 GPU profile task completed ==="
