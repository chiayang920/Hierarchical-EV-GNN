#!/bin/bash
#SBATCH --job-name=evgnn_100cp_smoke
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.err

set -euo pipefail

module load miniforge3/24.3.0-0
conda activate evgnn_m3_cpu

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

REPO_DIR="/projects/fr57/cche0357/EV-GNN"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_tiny_smoke"
SAVE_DIR="${RUN_ROOT}/saved_models"
EVAL_DIR="${RUN_ROOT}/eval"

mkdir -p "${SAVE_DIR}" "${EVAL_DIR}"

cd "${REPO_DIR}"

echo "=== M3 100 CP tiny smoke ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
python --version
python -c 'import torch, torch_geometric, ev2gym; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("pyg", torch_geometric.__version__)'
python -c 'import pkg_resources; print("PKG_RESOURCES_OK")'
python -c 'from ev2gym.models.ev2gym_env import EV2Gym; print("EV2GYM_ENV_IMPORT_OK")'

echo "=== ActionGNN 100 CP tiny training smoke ==="
python train_td3_gnn.py \
  --algorithm actiongnn \
  --config ./config_files/PublicPST_100.yaml \
  --seed 0 \
  --device cpu \
  --run_name phase2d_smoke_actiongnn_100cp_seed0 \
  --max_timesteps 112 \
  --start_timesteps 16 \
  --eval_freq 112 \
  --eval_episodes 1 \
  --batch_size 8 \
  --replay_buffer_size 1000 \
  --save_dir "${SAVE_DIR}" \
  --log_to_wandb false

echo "=== Hierarchical 100 CP tiny training smoke ==="
python train_td3_gnn.py \
  --algorithm hierarchical \
  --config ./config_files/PublicPST_100.yaml \
  --seed 0 \
  --device cpu \
  --run_name phase2d_smoke_hierarchical_100cp_seed0 \
  --max_timesteps 112 \
  --start_timesteps 16 \
  --eval_freq 112 \
  --eval_episodes 1 \
  --batch_size 8 \
  --replay_buffer_size 1000 \
  --save_dir "${SAVE_DIR}" \
  --log_to_wandb false

echo "=== Controlled evaluator smoke: ActionGNN ==="
python evaluate_td3_gnn.py \
  --algorithm actiongnn \
  --config ./config_files/PublicPST_100.yaml \
  --seed 0 \
  --eval_episodes 1 \
  --checkpoint "${SAVE_DIR}/phase2d_smoke_actiongnn_100cp_seed0/model.best" \
  --device cpu \
  --output_csv "${EVAL_DIR}/actiongnn_100cp_seed0_eval1.csv" \
  --run_name phase2d_smoke_actiongnn_100cp_seed0_eval1 \
  --max_episode_steps 112 \
  --deterministic true \
  --eval_seed_offset 400000

echo "=== Controlled evaluator smoke: hierarchical ==="
python evaluate_td3_gnn.py \
  --algorithm hierarchical \
  --config ./config_files/PublicPST_100.yaml \
  --seed 0 \
  --eval_episodes 1 \
  --checkpoint "${SAVE_DIR}/phase2d_smoke_hierarchical_100cp_seed0/model.best" \
  --device cpu \
  --output_csv "${EVAL_DIR}/hierarchical_100cp_seed0_eval1.csv" \
  --run_name phase2d_smoke_hierarchical_100cp_seed0_eval1 \
  --max_episode_steps 112 \
  --deterministic true \
  --eval_seed_offset 400000

echo "=== Generated files ==="
find "${RUN_ROOT}" -maxdepth 4 -type f | sort

echo "=== 100 CP tiny smoke completed ==="
