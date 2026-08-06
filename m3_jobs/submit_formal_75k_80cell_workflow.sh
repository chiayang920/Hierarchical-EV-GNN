#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${EV_GNN_FORMAL_75K_PYTHON:-python}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
SMOKE_SCRIPT="${REPO_ROOT}/m3_jobs/23_formal_75k_80cell_smoke.slurm"
TRAIN_SCRIPT="${REPO_ROOT}/m3_jobs/24_formal_75k_80cell_train.slurm"
EVAL_SCRIPT="${REPO_ROOT}/m3_jobs/25_formal_75k_80cell_eval30.slurm"
DIAGNOSTIC_SCRIPT="${REPO_ROOT}/m3_jobs/26_formal_75k_80cell_diagnostics.slurm"
SOURCE_BUNDLE_SCRIPT="${REPO_ROOT}/m3_jobs/create_formal_75k_80cell_source_bundle.sh"

DRY_RUN=0
PRINT_MATRIX=0
SUBMIT_SMOKE=0
SUBMIT_FORMAL=0
RESOURCE_PROFILE=""
ARRAY_JOB_ID=""
EVAL_JOB_ID=""
DIAGNOSTIC_JOB_ID=""

die() {
  echo "ERROR: $*" >&2
  exit 2
}

print_stages() {
  echo "Stage S - two-cell smoke"
  echo "Stage A - 80-cell formal training array"
  echo "Stage B - training-matrix validation gate"
  echo "Stage C - 80-cell model.best eval30 array"
  echo "Stage D - 80-cell infrastructure-diagnostic array"
  echo "Stage E - evaluation/diagnostic validation gate"
  echo "Stage F - final reducer and scientific analysis"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --print-matrix) PRINT_MATRIX=1 ;;
    --submit-smoke) SUBMIT_SMOKE=1 ;;
    --submit-formal) SUBMIT_FORMAL=1 ;;
    --resource-profile)
      [[ "$#" -ge 2 ]] || die "--resource-profile requires a path"
      RESOURCE_PROFILE="$2"
      shift
      ;;
    --array-job-id)
      [[ "$#" -ge 2 ]] || die "--array-job-id requires a numeric ID"
      ARRAY_JOB_ID="$2"
      shift
      ;;
    --eval-job-id)
      [[ "$#" -ge 2 ]] || die "--eval-job-id requires a numeric ID"
      EVAL_JOB_ID="$2"
      shift
      ;;
    --diagnostic-job-id)
      [[ "$#" -ge 2 ]] || die "--diagnostic-job-id requires a numeric ID"
      DIAGNOSTIC_JOB_ID="$2"
      shift
      ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

for script in "${WORKFLOW_SCRIPT}" "${SMOKE_SCRIPT}" "${TRAIN_SCRIPT}" "${EVAL_SCRIPT}" "${DIAGNOSTIC_SCRIPT}" "${SOURCE_BUNDLE_SCRIPT}"; do
  [[ -s "${script}" ]] || die "required workflow file is missing or empty: ${script}"
done

if [[ "${PRINT_MATRIX}" == "1" ]]; then
  "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" --print-matrix
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "PHASE=FORMAL_75K_80CELL_WORKFLOW_DRY_RUN"
  print_stages
  echo "SMOKE_MATRIX"
  "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" --print-smoke
  echo "FORMAL_MATRIX_COUNT=80"
  echo "SOURCE_BUNDLE_DRY_RUN_COMMAND=EV_GNN_FORMAL_75K_SOURCE_DRY_RUN=1 ${SOURCE_BUNDLE_SCRIPT}"
  echo "SMOKE_SUBMIT_COMMAND=sbatch --parsable --array=0-1 ${SMOKE_SCRIPT}"
  echo "FORMAL_SUBMIT_COMMAND=sbatch --parsable --array=0-79 --cpus-per-task=<reviewed_train_cpus> --mem=<reviewed_train_mem> --time=<reviewed_train_time> ${TRAIN_SCRIPT}"
  echo "TRAINING_GATE_COMMAND=${PYTHON_BIN} ${WORKFLOW_SCRIPT} validate-training --csv <formal_training_matrix_summary.csv>"
  echo "EVAL30_SUBMIT_COMMAND=sbatch --parsable --dependency=afterok:<training_gate_job_id> --array=0-79 ${EVAL_SCRIPT}"
  echo "DIAGNOSTIC_SUBMIT_COMMAND=sbatch --parsable --dependency=afterok:<eval_job_id> --array=0-79 ${DIAGNOSTIC_SCRIPT}"
  echo "FINAL_REDUCER_COMMAND=${PYTHON_BIN} ${WORKFLOW_SCRIPT} reduce --training-csv <formal_training_matrix_summary.csv> --eval30-csv <canonical_eval30_episode_rows.csv> --diagnostic-csv <diagnostic_episode_rows.csv> --out-dir <analysis_output_dir>"
  if [[ -n "${ARRAY_JOB_ID}" ]]; then
    echo "manual_array_job_id=${ARRAY_JOB_ID}"
  fi
  if [[ -n "${EVAL_JOB_ID}" ]]; then
    echo "manual_eval_job_id=${EVAL_JOB_ID}"
  fi
  if [[ -n "${DIAGNOSTIC_JOB_ID}" ]]; then
    echo "manual_diagnostic_job_id=${DIAGNOSTIC_JOB_ID}"
  fi
  echo "DRY_RUN_NO_JOBS_SUBMITTED"
  exit 0
fi

if [[ "${SUBMIT_FORMAL}" == "1" ]]; then
  [[ -n "${RESOURCE_PROFILE}" ]] || die "--submit-formal requires --resource-profile"
  "${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" --validate-resource-profile "${RESOURCE_PROFILE}"
  # shellcheck disable=SC1090
  source "${RESOURCE_PROFILE}"
fi

if [[ "${SUBMIT_SMOKE}" == "1" ]]; then
  sbatch --parsable --array=0-1 "${SMOKE_SCRIPT}"
fi

if [[ "${SUBMIT_FORMAL}" == "1" ]]; then
  sbatch --parsable \
    --array=0-79 \
    --cpus-per-task="${TRAIN_CPUS_PER_TASK}" \
    --mem="${TRAIN_MEM}" \
    --time="${TRAIN_TIME}" \
    "${TRAIN_SCRIPT}"
fi

if [[ "${SUBMIT_SMOKE}" != "1" && "${SUBMIT_FORMAL}" != "1" && "${PRINT_MATRIX}" != "1" ]]; then
  print_stages
  echo "No submission flag provided. Use --dry-run first."
fi
