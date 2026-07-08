#!/bin/bash
#SBATCH --job-name=evgnn_100cp_eval30
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-9
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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

TRAIN_MODEL_PARENT="/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_train/saved_models"
EVAL_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_eval30"
EVAL_DIR="${EVAL_ROOT}/eval"
PACKAGE_DIR="${EVAL_ROOT}/packages"

CONFIG_PATH="./config_files/PublicPST_100.yaml"
EVAL_EPISODES=30
MAX_EPISODE_STEPS=112
DEVICE="cpu"
DETERMINISTIC="true"
BASE_EVAL_SEED=420000

TASK_ID="${SLURM_ARRAY_TASK_ID}"

if [[ "${TASK_ID}" -ge 0 && "${TASK_ID}" -le 4 ]]; then
  ALG_SHORT="actiongnn"
  ALGORITHM="actiongnn"
  SEED="${TASK_ID}"
elif [[ "${TASK_ID}" -ge 5 && "${TASK_ID}" -le 9 ]]; then
  ALG_SHORT="hierarchical"
  ALGORITHM="hierarchical"
  SEED=$((TASK_ID - 5))
else
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}"
  exit 2
fi

RUN_NAME="phase2d_formal_eval30_${ALG_SHORT}_100cp_seed${SEED}"
MODEL_DIR="${TRAIN_MODEL_PARENT}/phase2d_formal_${ALG_SHORT}_100cp_seed${SEED}_50k"
CHECKPOINT_PREFIX="${MODEL_DIR}/model.best"

# evaluate_td3_gnn.py internally uses:
# episode_seed = args.seed + args.eval_seed_offset + episode_index
# This offset keeps actual evaluation episode seeds as:
# actiongnn/hierarchical seed0: 420000..420029
# actiongnn/hierarchical seed1: 421000..421029
# ...
EVAL_SEED_OFFSET=$((BASE_EVAL_SEED + SEED * 1000 - SEED))

OUTPUT_CSV="${EVAL_DIR}/${ALG_SHORT}_100cp_seed${SEED}_eval30.csv"
TASK_PACKAGE="${OUT_DIR}/phase2d_100cp_formal_eval30_${ALG_SHORT}_seed${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tar.gz"

mkdir -p "${EVAL_DIR}" "${PACKAGE_DIR}" "${OUT_DIR}"

cd "${REPO_DIR}"

echo "=== Phase 2D.3 100 CP formal controlled evaluation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Algorithm: ${ALGORITHM}"
echo "Seed: ${SEED}"
echo "Run name: ${RUN_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint prefix: ${CHECKPOINT_PREFIX}"
echo "Eval episodes: ${EVAL_EPISODES}"
echo "Max episode steps: ${MAX_EPISODE_STEPS}"
echo "Device: ${DEVICE}"
echo "Deterministic: ${DETERMINISTIC}"
echo "Eval seed offset: ${EVAL_SEED_OFFSET}"
echo "Output CSV: ${OUTPUT_CSV}"
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

echo "=== Check checkpoint files ==="
REQUIRED_FILES=(
  "${CHECKPOINT_PREFIX}_actor"
  "${CHECKPOINT_PREFIX}_critic"
  "${MODEL_DIR}/config.yaml"
  "${MODEL_DIR}/kwargs.yaml"
  "${MODEL_DIR}/run_args.yaml"
)

for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}"
    exit 3
  fi
  echo "OK: ${path}"
done

echo "=== Checkpoint checksums ==="
sha256sum "${CHECKPOINT_PREFIX}_actor" "${CHECKPOINT_PREFIX}_critic"

echo "=== Controlled eval30 ==="
rm -f "${OUTPUT_CSV}"

python evaluate_td3_gnn.py \
  --algorithm "${ALGORITHM}" \
  --config "${CONFIG_PATH}" \
  --seed "${SEED}" \
  --eval_episodes "${EVAL_EPISODES}" \
  --checkpoint "${CHECKPOINT_PREFIX}" \
  --device "${DEVICE}" \
  --output_csv "${OUTPUT_CSV}" \
  --run_name "${RUN_NAME}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" \
  --deterministic "${DETERMINISTIC}" \
  --eval_seed_offset "${EVAL_SEED_OFFSET}"

echo "=== Verify eval CSV ==="
if [[ ! -s "${OUTPUT_CSV}" ]]; then
  echo "Missing or empty output CSV: ${OUTPUT_CSV}"
  exit 4
fi

wc -l "${OUTPUT_CSV}"
head -n 5 "${OUTPUT_CSV}"

echo "=== Package eval artefacts ==="
tar -czf "${TASK_PACKAGE}" \
  -C "${EVAL_DIR}" "$(basename "${OUTPUT_CSV}")"

ls -lh "${TASK_PACKAGE}"

echo "=== Phase 2D.3 eval30 task completed ==="
