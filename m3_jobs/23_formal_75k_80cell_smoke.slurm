#!/bin/bash
#SBATCH --job-name=evgnn_formal75k_smoke
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${EV_GNN_FORMAL_75K_REPO_ROOT:-/projects/fr57/cche0357/EV-GNN}"
RUN_ROOT="${EV_GNN_FORMAL_75K_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/formal_75k_80cell_smoke}"
OUTPUT_ROOT="${EV_GNN_FORMAL_75K_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
PYTHON_BIN="${EV_GNN_FORMAL_75K_PYTHON:-python}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"

[[ "${TASK_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: SLURM_ARRAY_TASK_ID must be 0 or 1" >&2; exit 2; }
(( TASK_ID >= 0 && TASK_ID <= 1 )) || { echo "ERROR: SLURM_ARRAY_TASK_ID must be 0 or 1" >&2; exit 2; }

MAPPING="$("${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" smoke-task-mapping --task-id "${TASK_ID}")"
eval "${MAPPING// /; }"

THREAD_COUNT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREAD_COUNT}"
export MKL_NUM_THREADS="${THREAD_COUNT}"
export NUMEXPR_NUM_THREADS="${THREAD_COUNT}"

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
RUN_NAME="formal75k_smoke_${scale}_${algorithm}_seed${seed}_job${ARRAY_JOB_ID}_task${TASK_ID}"
TASK_ROOT="${RUN_ROOT}/job${ARRAY_JOB_ID}/task${TASK_ID}"
TRAIN_DIR="${TASK_ROOT}/train"
MODEL_DIR="${TRAIN_DIR}/${RUN_NAME}"
CHECKPOINT_PREFIX="${MODEL_DIR}/model.best"
EVAL_DIR="${TASK_ROOT}/smoke_eval"
EVAL_CSV="${EVAL_DIR}/${scale}_${algorithm}_seed${seed}_smoke_eval3.csv"
PACKAGE_PATH="${OUTPUT_ROOT}/m3_formal75k_smoke_${scale}_${algorithm}_seed${seed}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"

if [[ "${EV_GNN_FORMAL_75K_SMOKE_DRY_RUN:-0}" == "1" ]]; then
  echo "FORMAL_75K_SMOKE_DRY_RUN"
  echo "${MAPPING}"
  echo "run_name=${RUN_NAME}"
  echo "train_dir=${TRAIN_DIR}"
  echo "checkpoint_prefix=${CHECKPOINT_PREFIX}"
  echo "smoke_eval_csv=${EVAL_CSV}"
  echo "package_path=${PACKAGE_PATH}"
  echo "DRY_RUN_NO_TRAINING_EVALUATION_OR_PACKAGING"
  exit 0
fi

cd "${REPO_ROOT}"
mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}" "${OUTPUT_ROOT}" "${TASK_ROOT}/runtime_metadata"
echo "FORMAL_75K_SMOKE_TASK_START"
echo "${MAPPING}"
"${PYTHON_BIN}" train_td3_gnn.py \
  --algorithm "${algorithm}" \
  --config "${config}" \
  --seed "${seed}" \
  --device cpu \
  --run_name "${RUN_NAME}" \
  --max_timesteps "${training_steps}" \
  --start_timesteps 64 \
  --eval_freq 256 \
  --eval_episodes 1 \
  --batch_size 32 \
  --replay_buffer_size 5000 \
  --save_dir "${TRAIN_DIR}" \
  --log_to_wandb false
"${PYTHON_BIN}" evaluate_td3_gnn.py \
  --algorithm "${algorithm}" \
  --config "${config}" \
  --seed "${seed}" \
  --eval_episodes 3 \
  --checkpoint "${CHECKPOINT_PREFIX}" \
  --device cpu \
  --output_csv "${EVAL_CSV}" \
  --run_name "${RUN_NAME}_eval3" \
  --max_episode_steps 112 \
  --deterministic true \
  --eval_expl_noise 0.0 \
  --eval_seed_offset 710000
tar -czf "${PACKAGE_PATH}" -C "${TASK_ROOT}" .
tar -tzf "${PACKAGE_PATH}" >/dev/null
echo "FORMAL_75K_SMOKE_TASK_COMPLETED"
