#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=lib_formal_75k_runtime.sh
source "${SCRIPT_DIR}/lib_formal_75k_runtime.sh"
formal75k_require_python311

WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
SMOKE_SCRIPT="${SCRIPT_DIR}/23_formal_75k_80cell_smoke.slurm"
TRAIN_SCRIPT="${SCRIPT_DIR}/24_formal_75k_80cell_train.slurm"
EVAL_SCRIPT="${SCRIPT_DIR}/25_formal_75k_80cell_eval30.slurm"
DIAG_SCRIPT="${SCRIPT_DIR}/26_formal_75k_80cell_diagnostics.slurm"

DRY_RUN=0
PRINT_MATRIX=0
ACTION=""
RESOURCE_PROFILE=""
ARRAY_JOB_ID=""
EVAL_JOB_ID=""
DIAGNOSTIC_JOB_ID=""
PACKAGE_ROOT="${EV_GNN_FORMAL_75K_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
STAGE_ROOT="${EV_GNN_FORMAL_75K_STAGE_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/formal_75k_80cell_stage}"
ANALYSIS_ROOT="${EV_GNN_FORMAL_75K_ANALYSIS_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs/formal_75k_80cell_analysis}"
SOURCE_IDENTITY="${EV_GNN_FORMAL_75K_SOURCE_IDENTITY:-}"
SOURCE_BUNDLE_IDENTITY="${EV_GNN_FORMAL_75K_SOURCE_BUNDLE_IDENTITY:-}"
SERVICE_POLICY=""

set_action() {
  [[ -z "${ACTION}" ]] || formal75k_die "select only one workflow action"
  ACTION="$1"
}

require_source_identity() {
  [[ -n "${SOURCE_IDENTITY}" ]] || formal75k_die "--source-identity is required"
  [[ -n "${SOURCE_BUNDLE_IDENTITY}" ]] || formal75k_die "--source-bundle-identity is required"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --print-matrix) PRINT_MATRIX=1 ;;
    --submit-smoke) set_action submit-smoke ;;
    --run-smoke-gate) set_action smoke-gate ;;
    --submit-formal) set_action submit-formal ;;
    --run-training-gate) set_action training-gate ;;
    --submit-eval) set_action submit-eval ;;
    --submit-diagnostics) set_action submit-diagnostics ;;
    --run-eval-diagnostic-gate) set_action eval-diagnostic-gate ;;
    --run-reducer) set_action reducer ;;
    --resource-profile) RESOURCE_PROFILE="$2"; shift ;;
    --array-job-id) ARRAY_JOB_ID="$2"; shift ;;
    --eval-job-id) EVAL_JOB_ID="$2"; shift ;;
    --diagnostic-job-id) DIAGNOSTIC_JOB_ID="$2"; shift ;;
    --package-root) PACKAGE_ROOT="$2"; shift ;;
    --stage-root) STAGE_ROOT="$2"; shift ;;
    --analysis-root) ANALYSIS_ROOT="$2"; shift ;;
    --source-identity) SOURCE_IDENTITY="$2"; shift ;;
    --source-bundle-identity) SOURCE_BUNDLE_IDENTITY="$2"; shift ;;
    --service-policy) SERVICE_POLICY="$2"; shift ;;
    *) formal75k_die "unknown argument: $1" ;;
  esac
  shift
done

for file in "${WORKFLOW_SCRIPT}" "${SMOKE_SCRIPT}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${DIAG_SCRIPT}"; do
  [[ -s "${file}" ]] || formal75k_die "missing workflow file: ${file}"
done

echo "PYTHON_VERSION=${PYTHON_VERSION}"
if [[ "${PRINT_MATRIX}" -eq 1 ]]; then
  "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" --print-matrix
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  if [[ -n "${RESOURCE_PROFILE}" ]]; then
    formal75k_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" all
  fi
  echo "PHASE=FORMAL_75K_80CELL_WORKFLOW_DRY_RUN"
  echo "STAGES=S,A,B,C,D,E,F"
  echo "STAGE_S=two-cell smoke and exact package gate"
  echo "STAGE_A=80-cell fresh formal training array"
  echo "STAGE_B=exact-job training package validation and atomic staging"
  echo "STAGE_C=80-cell model.best eval30 array"
  echo "STAGE_D=80-cell infrastructure diagnostic array"
  echo "STAGE_E=independent eval30 and diagnostic validation gates"
  echo "STAGE_F=paired-seed scientific reducer and claim assessment"
  echo "MANUAL_JOB_ID_GOVERNANCE=REQUIRED"
  echo "STAGE_B_COMMAND=${WORKFLOW_SCRIPT} aggregate-training --array-job-id <exact>"
  echo "STAGE_E_COMMAND=${WORKFLOW_SCRIPT} aggregate-eval30/aggregate-diagnostics with exact IDs"
  echo "STAGE_F_COMMAND=${WORKFLOW_SCRIPT} reduce with real training_curve_long.csv"
  echo "DRY_RUN_NO_JOBS_SUBMITTED"
  exit 0
fi

case "${ACTION}" in
  submit-smoke)
    require_source_identity
    formal75k_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" smoke
    sbatch --parsable \
      --array=0-1 \
      --cpus-per-task="${SMOKE_CPUS_PER_TASK}" \
      --mem="${SMOKE_MEM}" \
      --time="${SMOKE_TIME}" \
      --export="ALL,EV_GNN_FORMAL_75K_REPO_ROOT=${REPO_ROOT},EV_GNN_FORMAL_75K_PYTHON=${EV_GNN_FORMAL_75K_PYTHON},EV_GNN_FORMAL_75K_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_FORMAL_75K_SOURCE_IDENTITY=${SOURCE_IDENTITY},EV_GNN_FORMAL_75K_SOURCE_BUNDLE_IDENTITY=${SOURCE_BUNDLE_IDENTITY}" \
      "${SMOKE_SCRIPT}"
    ;;
  smoke-gate)
    require_source_identity
    formal75k_require_numeric_id "smoke job ID" "${ARRAY_JOB_ID}"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" validate-smoke-packages \
      --package-root "${PACKAGE_ROOT}" \
      --job-id "${ARRAY_JOB_ID}" \
      --stage-root "${STAGE_ROOT}/smoke" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    ;;
  submit-formal)
    require_source_identity
    formal75k_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" formal
    sbatch --parsable \
      --array=0-79 \
      --cpus-per-task="${TRAIN_CPUS_PER_TASK}" \
      --mem="${TRAIN_MEM}" \
      --time="${TRAIN_TIME}" \
      --export="ALL,EV_GNN_FORMAL_75K_REPO_ROOT=${REPO_ROOT},EV_GNN_FORMAL_75K_PYTHON=${EV_GNN_FORMAL_75K_PYTHON},EV_GNN_FORMAL_75K_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_FORMAL_75K_SOURCE_IDENTITY=${SOURCE_IDENTITY},EV_GNN_FORMAL_75K_SOURCE_BUNDLE_IDENTITY=${SOURCE_BUNDLE_IDENTITY}" \
      "${TRAIN_SCRIPT}"
    ;;
  training-gate)
    require_source_identity
    formal75k_require_numeric_id "training array job ID" "${ARRAY_JOB_ID}"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" aggregate-training \
      --package-root "${PACKAGE_ROOT}" \
      --array-job-id "${ARRAY_JOB_ID}" \
      --stage-root "${STAGE_ROOT}" \
      --output-dir "${ANALYSIS_ROOT}/trainjob${ARRAY_JOB_ID}" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    ;;
  submit-eval)
    formal75k_require_numeric_id "training array job ID" "${ARRAY_JOB_ID}"
    formal75k_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" formal
    sbatch --parsable \
      --array=0-79 \
      --cpus-per-task="${EVAL_CPUS_PER_TASK}" \
      --mem="${EVAL_MEM}" \
      --time="${EVAL_TIME}" \
      --export="ALL,EV_GNN_FORMAL_75K_REPO_ROOT=${REPO_ROOT},EV_GNN_FORMAL_75K_PYTHON=${EV_GNN_FORMAL_75K_PYTHON},EV_GNN_FORMAL_75K_ARRAY_JOB_ID=${ARRAY_JOB_ID},EV_GNN_FORMAL_75K_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_FORMAL_75K_STAGE_ROOT=${STAGE_ROOT}" \
      "${EVAL_SCRIPT}"
    ;;
  submit-diagnostics)
    formal75k_require_numeric_id "training array job ID" "${ARRAY_JOB_ID}"
    formal75k_validate_resource_profile "${WORKFLOW_SCRIPT}" "${RESOURCE_PROFILE}" formal
    sbatch --parsable \
      --array=0-79 \
      --cpus-per-task="${DIAGNOSTIC_CPUS_PER_TASK}" \
      --mem="${DIAGNOSTIC_MEM}" \
      --time="${DIAGNOSTIC_TIME}" \
      --export="ALL,EV_GNN_FORMAL_75K_REPO_ROOT=${REPO_ROOT},EV_GNN_FORMAL_75K_PYTHON=${EV_GNN_FORMAL_75K_PYTHON},EV_GNN_FORMAL_75K_ARRAY_JOB_ID=${ARRAY_JOB_ID},EV_GNN_FORMAL_75K_RESOURCE_PROFILE=${RESOURCE_PROFILE},EV_GNN_FORMAL_75K_STAGE_ROOT=${STAGE_ROOT}" \
      "${DIAG_SCRIPT}"
    ;;
  eval-diagnostic-gate)
    require_source_identity
    formal75k_require_numeric_id "training array job ID" "${ARRAY_JOB_ID}"
    formal75k_require_numeric_id "eval job ID" "${EVAL_JOB_ID}"
    formal75k_require_numeric_id "diagnostic job ID" "${DIAGNOSTIC_JOB_ID}"
    OUT="${ANALYSIS_ROOT}/trainjob${ARRAY_JOB_ID}"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" aggregate-eval30 \
      --package-root "${PACKAGE_ROOT}" \
      --array-job-id "${ARRAY_JOB_ID}" \
      --eval-job-id "${EVAL_JOB_ID}" \
      --output-dir "${OUT}" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" aggregate-diagnostics \
      --package-root "${PACKAGE_ROOT}" \
      --array-job-id "${ARRAY_JOB_ID}" \
      --diagnostic-job-id "${DIAGNOSTIC_JOB_ID}" \
      --output-dir "${OUT}" \
      --source-identity "${SOURCE_IDENTITY}" \
      --source-bundle-identity "${SOURCE_BUNDLE_IDENTITY}"
    ;;
  reducer)
    formal75k_require_numeric_id "training array job ID" "${ARRAY_JOB_ID}"
    OUT="${ANALYSIS_ROOT}/trainjob${ARRAY_JOB_ID}"
    REDUCER_ARGS=(
      reduce
      --training-csv "${OUT}/formal_training_matrix_summary.csv"
      --training-curve-csv "${OUT}/training_curve_long.csv"
      --eval30-csv "${OUT}/canonical_eval30_episode_rows.csv"
      --diagnostic-csv "${OUT}/diagnostic_episode_rows.csv"
      --out-dir "${OUT}/scientific_results"
    )
    if [[ -n "${SERVICE_POLICY}" ]]; then
      REDUCER_ARGS+=(--service-policy "${SERVICE_POLICY}")
    fi
    "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" "${REDUCER_ARGS[@]}"
    ;;
  "")
    echo "No action selected. Use --dry-run first."
    ;;
  *) formal75k_die "unhandled action" ;;
esac
