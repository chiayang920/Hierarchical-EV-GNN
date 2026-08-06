#!/bin/bash
#SBATCH --job-name=evgnn_formal75k_diag
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-79
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${EV_GNN_FORMAL_75K_REPO_ROOT:-/projects/fr57/cche0357/EV-GNN}"
RUN_ROOT="${EV_GNN_FORMAL_75K_DIAGNOSTIC_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/formal_75k_80cell_diagnostics}"
PYTHON_BIN="${EV_GNN_FORMAL_75K_PYTHON:-python}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
FORMAL_ARRAY_JOB_ID="${EV_GNN_FORMAL_75K_ARRAY_JOB_ID:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"

[[ "${TASK_ID}" =~ ^[0-9]+$ ]] || { echo "ERROR: SLURM_ARRAY_TASK_ID must be 0-79" >&2; exit 2; }
(( TASK_ID >= 0 && TASK_ID <= 79 )) || { echo "ERROR: SLURM_ARRAY_TASK_ID must be 0-79" >&2; exit 2; }
[[ -n "${FORMAL_ARRAY_JOB_ID}" ]] || { echo "ERROR: EV_GNN_FORMAL_75K_ARRAY_JOB_ID is required" >&2; exit 2; }

MAPPING="$("${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" formal-task-mapping --task-id "${TASK_ID}")"
eval "${MAPPING// /; }"

THREAD_COUNT="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${THREAD_COUNT}"
export MKL_NUM_THREADS="${THREAD_COUNT}"
export NUMEXPR_NUM_THREADS="${THREAD_COUNT}"

RUN_NAME="formal75k_diag_${scale}_${algorithm}_seed${seed}_job${FORMAL_ARRAY_JOB_ID}_task${TASK_ID}"
TASK_ROOT="${RUN_ROOT}/job${FORMAL_ARRAY_JOB_ID}/task${TASK_ID}"
CHECKPOINT_PREFIX="${TASK_ROOT}/checkpoint/model.best"
OUTPUT_DIR="${TASK_ROOT}/diagnostics"

case "${scale}" in
  25cp) EVAL_BASE_SEED=710000 ;;
  100cp) EVAL_BASE_SEED=720000 ;;
  500cp) EVAL_BASE_SEED=730000 ;;
  1000cp) EVAL_BASE_SEED=740000 ;;
  *) echo "ERROR: unsupported scale: ${scale}" >&2; exit 2 ;;
esac
EVAL_SEED_OFFSET=$((EVAL_BASE_SEED + (999 * seed)))

if [[ "${EV_GNN_FORMAL_75K_DIAGNOSTIC_DRY_RUN:-0}" == "1" ]]; then
  echo "FORMAL_75K_DIAGNOSTIC_DRY_RUN"
  echo "${MAPPING}"
  echo "run_name=${RUN_NAME}"
  echo "checkpoint_prefix=${CHECKPOINT_PREFIX}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "eval_seed_offset=${EVAL_SEED_OFFSET}"
  echo "DRY_RUN_NO_DIAGNOSTICS_OR_PACKAGING"
  exit 0
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${TASK_ROOT}/runtime_metadata"
"${PYTHON_BIN}" evaluate_td3_gnn_infrastructure_diagnostics.py \
  --algorithm "${algorithm}" \
  --config "${config}" \
  --seed "${seed}" \
  --eval_episodes 30 \
  --checkpoint "${CHECKPOINT_PREFIX}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --device cpu \
  --max_episode_steps 112 \
  --deterministic true \
  --eval_expl_noise 0.0 \
  --eval_seed_offset "${EVAL_SEED_OFFSET}" \
  --matrix_job_id "${FORMAL_ARRAY_JOB_ID}"
echo "FORMAL_75K_DIAGNOSTIC_TASK_COMPLETED"
