#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=lib_transformer_constraint_runtime.sh
source "${SCRIPT_DIR}/lib_transformer_constraint_runtime.sh"
transformer_constraint_require_python311

WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/transformer_constraint_40cell_workflow.py"
SMOKE_SCRIPT="${SCRIPT_DIR}/31_transformer_constraint_40cell_smoke.slurm"
TRAIN_SCRIPT="${SCRIPT_DIR}/32_transformer_constraint_40cell_train.slurm"
EVAL_SCRIPT="${SCRIPT_DIR}/33_transformer_constraint_40cell_eval30.slurm"
DIAG_SCRIPT="${SCRIPT_DIR}/34_transformer_constraint_40cell_diagnostics.slurm"

DRY_RUN=0
ACTION=""
RESOURCE_PROFILE=""
ARRAY_JOB_ID=""
SMALL_JOB_ID=""
CP500_JOB_ID=""
CP1000_JOB_ID=""
TRAINING_SET_ID=""
EVAL_JOB_ID=""
DIAGNOSTIC_JOB_ID=""
PACKAGE_ROOT="${EV_GNN_TRANSFORMER_CONSTRAINT_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs/transformer_constraint_40cell}"
STAGE_ROOT="${EV_GNN_TRANSFORMER_CONSTRAINT_STAGE_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/transformer_constraint_40cell_stage}"
ANALYSIS_ROOT="${EV_GNN_TRANSFORMER_CONSTRAINT_ANALYSIS_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs/transformer_constraint_40cell_analysis}"
SOURCE_IDENTITY="${EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_IDENTITY:-}"
SOURCE_BUNDLE_IDENTITY="${EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_BUNDLE_IDENTITY:-}"

set_action() {
  [[ -z "${ACTION}" ]] || transformer_constraint_die "select only one workflow action"
  ACTION="$1"
}

require_source_identity() {
  [[ -n "${SOURCE_IDENTITY}" ]] || transformer_constraint_die "--source-identity is required"
  [[ -n "${SOURCE_BUNDLE_IDENTITY}" ]] || transformer_constraint_die "--source-bundle-identity is required"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --submit-smoke) set_action submit-smoke ;;
    --run-smoke-gate) set_action smoke-gate ;;
    --submit-formal-small) set_action submit-formal-small ;;
    --submit-formal-500cp) set_action submit-formal-500cp ;;
    --submit-formal-1000cp) set_action submit-formal-1000cp ;;
    --run-training-gate) set_action training-gate ;;
    --submit-eval) set_action submit-eval ;;
    --submit-diagnostics) set_action submit-diagnostics ;;
    --run-eval-gate) set_action eval-gate ;;
    --run-diagnostic-gate) set_action diagnostic-gate ;;
    --resource-profile) RESOURCE_PROFILE="$2"; shift ;;
    --array-job-id) ARRAY_JOB_ID="$2"; shift ;;
    --small-job-id) SMALL_JOB_ID="$2"; shift ;;
    --cp500-job-id) CP500_JOB_ID="$2"; shift ;;
    --cp1000-job-id) CP1000_JOB_ID="$2"; shift ;;
    --training-set-id) TRAINING_SET_ID="$2"; shift ;;
    --eval-job-id) EVAL_JOB_ID="$2"; shift ;;
    --diagnostic-job-id) DIAGNOSTIC_JOB_ID="$2"; shift ;;
    --package-root) PACKAGE_ROOT="$2"; shift ;;
    --stage-root) STAGE_ROOT="$2"; shift ;;
    --analysis-root) ANALYSIS_ROOT="$2"; shift ;;
    --source-identity) SOURCE_IDENTITY="$2"; shift ;;
    --source-bundle-identity) SOURCE_BUNDLE_IDENTITY="$2"; shift ;;
    *) transformer_constraint_die "unknown argument: $1" ;;
  esac
  shift
done

training_set_id_from_args() {
  if [[ -n "${TRAINING_SET_ID}" ]]; then
    transformer_constraint_require_safe_id "training set ID" "${TRAINING_SET_ID}"
    printf '%s\n' "${TRAINING_SET_ID}"
    return 0
  fi
  if [[ -n "${ARRAY_JOB_ID}" ]]; then
    transformer_constraint_require_numeric_id "training array job ID" "${ARRAY_JOB_ID}"
    printf 'trainjob%s\n' "${ARRAY_JOB_ID}"
    return 0
  fi
  if [[ -n "${SMALL_JOB_ID}" || -n "${CP500_JOB_ID}" || -n "${CP1000_JOB_ID}" ]]; then
    transformer_constraint_require_numeric_id "small training job ID" "${SMALL_JOB_ID}"
    transformer_constraint_require_numeric_id "500CP training job ID" "${CP500_JOB_ID}"
    transformer_constraint_require_numeric_id "1000CP training job ID" "${CP1000_JOB_ID}"
    printf 'trainset_%s_%s_%s\n' "${SMALL_JOB_ID}" "${CP500_JOB_ID}" "${CP1000_JOB_ID}"
    return 0
  fi
  transformer_constraint_die "training set identity requires --array-job-id, --training-set-id, or all split training job IDs"
}

for file in "${WORKFLOW_SCRIPT}" "${SMOKE_SCRIPT}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${DIAG_SCRIPT}"; do
  [[ -s "${file}" ]] || transformer_constraint_die "missing workflow file: ${file}"
done

echo "PYTHON_VERSION=${PYTHON_VERSION}"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "RESOURCE_PROFILE_STATUS=${RESOURCE_PROFILE:-RESOURCE_PROFILE_PENDING}"
  "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" dry-run-summary
  echo "INTENDED_SBATCH_SMOKE=sbatch --array=0-3 ${SMOKE_SCRIPT}"
  echo "INTENDED_SBATCH_FORMAL_SMALL=sbatch --array=0-19 ${TRAIN_SCRIPT}"
  echo "INTENDED_SBATCH_FORMAL_500CP=sbatch --array=20-29 ${TRAIN_SCRIPT}"
  echo "INTENDED_SBATCH_FORMAL_1000CP=sbatch --array=30-39 ${TRAIN_SCRIPT}"
  echo "INTENDED_SBATCH_EVAL30=sbatch --array=0-39 ${EVAL_SCRIPT}"
  echo "INTENDED_SBATCH_DIAGNOSTICS=sbatch --array=0-39 ${DIAG_SCRIPT}"
  exit 0
fi

case "${ACTION}" in
  submit-smoke)
    require_source_identity
    transformer_constraint_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" smoke
    sbatch --parsable \
      --array=0-3 \
      --cpus-per-task="${SMOKE_CPUS_PER_TASK}" \
      --mem="${SMOKE_MEM}" \
      --time="${SMOKE_TIME}" \
      --export="ALL,EV_GNN_TRANSFORMER_CONSTRAINT_REPO_ROOT=${REPO_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON=${EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON},EV_GNN_TRANSFORMER_CONSTRAINT_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_TRANSFORMER_CONSTRAINT_OUTPUT_ROOT=${PACKAGE_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_IDENTITY=${SOURCE_IDENTITY},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_BUNDLE_IDENTITY=${SOURCE_BUNDLE_IDENTITY}" \
      "${SMOKE_SCRIPT}"
    ;;
  smoke-gate)
    require_source_identity
    transformer_constraint_require_numeric_id "smoke job ID" "${ARRAY_JOB_ID}"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" validate-smoke-packages \
      --package-root "${PACKAGE_ROOT}" \
      --job-id "${ARRAY_JOB_ID}" \
      --stage-root "${STAGE_ROOT}/smoke" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    ;;
  submit-formal-small)
    require_source_identity
    transformer_constraint_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" train-small
    sbatch --parsable \
      --array=0-19 \
      --cpus-per-task="${TRAIN_SMALL_CPUS_PER_TASK}" \
      --mem="${TRAIN_SMALL_MEM}" \
      --time="${TRAIN_SMALL_TIME}" \
      --export="ALL,EV_GNN_TRANSFORMER_CONSTRAINT_REPO_ROOT=${REPO_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON=${EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON},EV_GNN_TRANSFORMER_CONSTRAINT_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_TRANSFORMER_CONSTRAINT_OUTPUT_ROOT=${PACKAGE_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_IDENTITY=${SOURCE_IDENTITY},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_BUNDLE_IDENTITY=${SOURCE_BUNDLE_IDENTITY}" \
      "${TRAIN_SCRIPT}"
    ;;
  submit-formal-500cp)
    require_source_identity
    transformer_constraint_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" train-500cp
    sbatch --parsable \
      --array=20-29 \
      --cpus-per-task="${TRAIN_500CP_CPUS_PER_TASK}" \
      --mem="${TRAIN_500CP_MEM}" \
      --time="${TRAIN_500CP_TIME}" \
      --export="ALL,EV_GNN_TRANSFORMER_CONSTRAINT_REPO_ROOT=${REPO_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON=${EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON},EV_GNN_TRANSFORMER_CONSTRAINT_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_TRANSFORMER_CONSTRAINT_OUTPUT_ROOT=${PACKAGE_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_IDENTITY=${SOURCE_IDENTITY},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_BUNDLE_IDENTITY=${SOURCE_BUNDLE_IDENTITY}" \
      "${TRAIN_SCRIPT}"
    ;;
  submit-formal-1000cp)
    require_source_identity
    transformer_constraint_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" train-1000cp
    sbatch --parsable \
      --array=30-39 \
      --cpus-per-task="${TRAIN_1000CP_CPUS_PER_TASK}" \
      --mem="${TRAIN_1000CP_MEM}" \
      --time="${TRAIN_1000CP_TIME}" \
      --export="ALL,EV_GNN_TRANSFORMER_CONSTRAINT_REPO_ROOT=${REPO_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON=${EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON},EV_GNN_TRANSFORMER_CONSTRAINT_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_TRANSFORMER_CONSTRAINT_OUTPUT_ROOT=${PACKAGE_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_IDENTITY=${SOURCE_IDENTITY},EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_BUNDLE_IDENTITY=${SOURCE_BUNDLE_IDENTITY}" \
      "${TRAIN_SCRIPT}"
    ;;
  training-gate)
    require_source_identity
    [[ -n "${SMALL_JOB_ID}" && -n "${CP500_JOB_ID}" && -n "${CP1000_JOB_ID}" ]] || transformer_constraint_die "split training job IDs are required for the formal 40-cell training gate"
    RESOLVED_TRAINING_SET_ID="$(training_set_id_from_args)"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" aggregate-training \
      --package-root "${PACKAGE_ROOT}" \
      --small-job-id "${SMALL_JOB_ID}" \
      --cp500-job-id "${CP500_JOB_ID}" \
      --cp1000-job-id "${CP1000_JOB_ID}" \
      --stage-root "${STAGE_ROOT}" \
      --output-dir "${ANALYSIS_ROOT}/${RESOLVED_TRAINING_SET_ID}" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    ;;
  submit-eval)
    RESOLVED_TRAINING_SET_ID="$(training_set_id_from_args)"
    [[ -z "${ARRAY_JOB_ID}" || "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]] || transformer_constraint_die "training array job ID must contain decimal digits only"
    transformer_constraint_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" eval
    sbatch --parsable \
      --array=0-39 \
      --cpus-per-task="${EVAL_CPUS_PER_TASK}" \
      --mem="${EVAL_MEM}" \
      --time="${EVAL_TIME}" \
      --export="ALL,EV_GNN_TRANSFORMER_CONSTRAINT_REPO_ROOT=${REPO_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON=${EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON},EV_GNN_TRANSFORMER_CONSTRAINT_ARRAY_JOB_ID=${ARRAY_JOB_ID},EV_GNN_TRANSFORMER_CONSTRAINT_TRAINING_SET_ID=${RESOLVED_TRAINING_SET_ID},EV_GNN_TRANSFORMER_CONSTRAINT_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_TRANSFORMER_CONSTRAINT_STAGE_ROOT=${STAGE_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_OUTPUT_ROOT=${PACKAGE_ROOT}" \
      "${EVAL_SCRIPT}"
    ;;
  submit-diagnostics)
    RESOLVED_TRAINING_SET_ID="$(training_set_id_from_args)"
    [[ -z "${ARRAY_JOB_ID}" || "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]] || transformer_constraint_die "training array job ID must contain decimal digits only"
    transformer_constraint_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" diagnostic
    sbatch --parsable \
      --array=0-39 \
      --cpus-per-task="${DIAGNOSTIC_CPUS_PER_TASK}" \
      --mem="${DIAGNOSTIC_MEM}" \
      --time="${DIAGNOSTIC_TIME}" \
      --export="ALL,EV_GNN_TRANSFORMER_CONSTRAINT_REPO_ROOT=${REPO_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON=${EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON},EV_GNN_TRANSFORMER_CONSTRAINT_ARRAY_JOB_ID=${ARRAY_JOB_ID},EV_GNN_TRANSFORMER_CONSTRAINT_TRAINING_SET_ID=${RESOLVED_TRAINING_SET_ID},EV_GNN_TRANSFORMER_CONSTRAINT_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_TRANSFORMER_CONSTRAINT_STAGE_ROOT=${STAGE_ROOT},EV_GNN_TRANSFORMER_CONSTRAINT_OUTPUT_ROOT=${PACKAGE_ROOT}" \
      "${DIAG_SCRIPT}"
    ;;
  eval-gate)
    require_source_identity
    RESOLVED_TRAINING_SET_ID="$(training_set_id_from_args)"
    transformer_constraint_require_numeric_id "eval job ID" "${EVAL_JOB_ID}"
    OUT="${ANALYSIS_ROOT}/${RESOLVED_TRAINING_SET_ID}"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" aggregate-eval30 \
      --package-root "${PACKAGE_ROOT}" \
      --array-job-id "${ARRAY_JOB_ID}" \
      --training-set-id "${RESOLVED_TRAINING_SET_ID}" \
      --eval-job-id "${EVAL_JOB_ID}" \
      --output-dir "${OUT}" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    ;;
  diagnostic-gate)
    require_source_identity
    RESOLVED_TRAINING_SET_ID="$(training_set_id_from_args)"
    transformer_constraint_require_numeric_id "diagnostic job ID" "${DIAGNOSTIC_JOB_ID}"
    OUT="${ANALYSIS_ROOT}/${RESOLVED_TRAINING_SET_ID}"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" aggregate-diagnostics \
      --package-root "${PACKAGE_ROOT}" \
      --array-job-id "${ARRAY_JOB_ID}" \
      --training-set-id "${RESOLVED_TRAINING_SET_ID}" \
      --diagnostic-job-id "${DIAGNOSTIC_JOB_ID}" \
      --output-dir "${OUT}" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    ;;
  "")
    echo "No action selected. Use --dry-run first."
    ;;
  *) transformer_constraint_die "unhandled action" ;;
esac
