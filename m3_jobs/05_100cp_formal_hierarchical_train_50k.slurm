#!/bin/bash
#SBATCH --job-name=evgnn_100cp_hier50k
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

module load miniforge3/24.3.0-0
conda activate evgnn_m3_cpu

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
OUT_DIR="/projects/fr57/cche0357/EV-GNN_outputs"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_train"
SAVE_DIR="${RUN_ROOT}/saved_models"

CONFIG_PATH="./config_files/PublicPST_100.yaml"
ALGORITHM="hierarchical"
TRAIN_SCRIPT="train_td3_gnn.py"
SEED="${SLURM_ARRAY_TASK_ID}"

MAX_TIMESTEPS=50000
START_TIMESTEPS=1000
EVAL_FREQ=5000
TRAIN_EVAL_EPISODES=5
BATCH_SIZE=64
REPLAY_BUFFER_SIZE=100000
DEVICE="cpu"

RUN_NAME="phase2d_formal_hierarchical_100cp_seed${SEED}_50k"
MODEL_DIR="${SAVE_DIR}/${RUN_NAME}"

mkdir -p "${SAVE_DIR}" "${OUT_DIR}"

cd "${REPO_DIR}"

echo "=== Phase 2D.2B 100 CP formal hierarchical training ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Algorithm: ${ALGORITHM}"
echo "Run name: ${RUN_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Seed: ${SEED}"
echo "Max timesteps: ${MAX_TIMESTEPS}"
echo "Device: ${DEVICE}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "Python: $(which python)"
python --version

echo "=== Dependency preflight ==="
python -c 'import setuptools; print("SETUPTOOLS", setuptools.__version__)'
python -c 'import pkg_resources; print("PKG_RESOURCES_OK")'
python -c 'from ev2gym.models.ev2gym_env import EV2Gym; print("EV2GYM_ENV_IMPORT_OK")'
python -c 'import torch, torch_geometric, ev2gym; print("torch", torch.__version__); print("torch_cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("pyg", torch_geometric.__version__)'

echo "=== Training hierarchical formal 100 CP ==="
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

echo "=== Verify generated hierarchical artefacts ==="
REQUIRED_FILES=(
  "${MODEL_DIR}/config.yaml"
  "${MODEL_DIR}/kwargs.yaml"
  "${MODEL_DIR}/run_args.yaml"
  "${MODEL_DIR}/training_log.csv"
  "${MODEL_DIR}/model.best_actor"
  "${MODEL_DIR}/model.best_critic"
  "${MODEL_DIR}/model.last_actor"
  "${MODEL_DIR}/model.last_critic"
)

for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}"
    exit 3
  fi
  echo "OK: ${path}"
done

echo "=== Generated files ==="
find "${MODEL_DIR}" -maxdepth 2 -type f | sort

echo "=== Package hierarchical formal artefacts ==="
tar -czf "${OUT_DIR}/phase2d_100cp_formal_train_${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tar.gz" \
  -C "${MODEL_DIR}" .

ls -lh "${OUT_DIR}/phase2d_100cp_formal_train_${RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tar.gz"

echo "=== Phase 2D.2B hierarchical formal training completed ==="
