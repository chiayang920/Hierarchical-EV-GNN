#!/bin/bash
#SBATCH --job-name=evgnn_formal75k_train
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-79
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${EV_GNN_FORMAL_75K_REPO_ROOT:-/projects/fr57/cche0357/EV-GNN}"
RUN_ROOT="${EV_GNN_FORMAL_75K_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/formal_75k_80cell}"
OUTPUT_ROOT="${EV_GNN_FORMAL_75K_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
PYTHON_BIN="${EV_GNN_FORMAL_75K_PYTHON:-python}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"

[[ "${TASK_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: SLURM_ARRAY_TASK_ID must be 0-79" >&2; exit 2; }
(( TASK_ID >= 0 && TASK_ID <= 79 )) || { echo "ERROR: SLURM_ARRAY_TASK_ID must be 0-79" >&2; exit 2; }

MAPPING="$("${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" formal-task-mapping --task-id "${TASK_ID}")"
eval "${MAPPING// /; }"

THREAD_COUNT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREAD_COUNT}"
export MKL_NUM_THREADS="${THREAD_COUNT}"
export NUMEXPR_NUM_THREADS="${THREAD_COUNT}"

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
RUN_NAME="formal75k_${scale}_${algorithm}_seed${seed}"
TASK_ROOT="${RUN_ROOT}/job${ARRAY_JOB_ID}/${scale}/${algorithm}/seed${seed}"
TRAIN_DIR="${TASK_ROOT}/train"
PACKAGE_PATH="${OUTPUT_ROOT}/m3_formal75k_80cell_${scale}_${algorithm}_seed${seed}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"

case "${algorithm}" in
  actiongnn_nonnegative) ACTOR_OUTPUT_TRANSFORM="shifted_tanh_v1" ;;
  hierarchical) ACTOR_OUTPUT_TRANSFORM="hierarchical_nonnegative_action_contract_v1" ;;
  *) echo "ERROR: unsupported formal algorithm: ${algorithm}" >&2; exit 2 ;;
esac

if [[ "${EV_GNN_FORMAL_75K_TRAIN_DRY_RUN:-0}" == "1" ]]; then
  echo "FORMAL_75K_TRAIN_DRY_RUN"
  echo "${MAPPING}"
  echo "run_name=${RUN_NAME}"
  echo "train_dir=${TRAIN_DIR}"
  echo "package_path=${PACKAGE_PATH}"
  echo "DRY_RUN_NO_TRAINING_OR_PACKAGING"
  exit 0
fi

[[ ! -e "${TASK_ROOT}" ]] || { echo "ERROR: task output already exists: ${TASK_ROOT}" >&2; exit 3; }
[[ ! -e "${PACKAGE_PATH}" ]] || { echo "ERROR: task package already exists: ${PACKAGE_PATH}" >&2; exit 3; }

cd "${REPO_ROOT}"
mkdir -p "${TRAIN_DIR}" "${OUTPUT_ROOT}" "${TASK_ROOT}/runtime_metadata"
{
  echo "protocol_version=formal_75k_80cell_v1"
  echo "task_id=${TASK_ID}"
  echo "slurm_array_job_id=${ARRAY_JOB_ID}"
  echo "scale=${scale}"
  echo "algorithm=${algorithm}"
  echo "seed=${seed}"
  echo "training_steps=75000"
  echo "evaluation_cadence=5000"
  echo "evaluation_episodes=5"
  echo "expected_scheduled_evaluations=15"
  echo "actor_output_transform=${ACTOR_OUTPUT_TRANSFORM}"
  echo "requested_cpu=${SLURM_CPUS_PER_TASK:-}"
  echo "requested_memory=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-}}"
  echo "requested_walltime=${SLURM_TIMELIMIT:-}"
} > "${TASK_ROOT}/runtime_metadata/task_runtime_metadata.env"

"${PYTHON_BIN}" train_td3_gnn.py \
  --algorithm "${algorithm}" \
  --config "${config}" \
  --seed "${seed}" \
  --device cpu \
  --run_name "${RUN_NAME}" \
  --max_timesteps 75000 \
  --start_timesteps 1000 \
  --eval_freq 5000 \
  --eval_episodes 5 \
  --batch_size 64 \
  --replay_buffer_size 100000 \
  --save_dir "${TRAIN_DIR}" \
  --log_to_wandb false

tar -czf "${PACKAGE_PATH}" -C "${TASK_ROOT}" .
tar -tzf "${PACKAGE_PATH}" >/dev/null
echo "FORMAL_75K_TRAIN_TASK_COMPLETED"
