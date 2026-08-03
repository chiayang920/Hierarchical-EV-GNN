#!/bin/bash
#SBATCH --job-name=evgnn_full_infra_diag_reduce
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
RUN_ROOT="${EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/full_infrastructure_diagnostic_eval30}"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
TASK_PACKAGE_ROOT="${EV_GNN_FULL_DIAGNOSTIC_TASK_PACKAGE_ROOT:-${OUTPUT_ROOT}}"
SLURM_LOG_ROOT="${EV_GNN_FULL_DIAGNOSTIC_SLURM_LOG_ROOT:-${OUTPUT_ROOT}}"
ARRAY_JOB_ID="${EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID:-}"
EXPECTED_SOURCE_COMMIT="${EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT:-}"
SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
VALIDATOR="${REPO_ROOT}/scripts/validate_full_infrastructure_diagnostic_eval30.py"
M3_CONDA_ENV="${EV_GNN_FULL_DIAGNOSTIC_M3_CONDA_ENV:-/scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu}"
PYTHON_BIN="${M3_CONDA_ENV}/bin/python"
REDUCER_JOB_ID="${SLURM_JOB_ID:-manual}"
WORK_ROOT_BASE="${EV_GNN_FULL_DIAGNOSTIC_REDUCER_WORK_ROOT:-${RUN_ROOT}/reducer}"
WORK_ROOT="${WORK_ROOT_BASE}/reducer_job${REDUCER_JOB_ID}"
JOB_WORK_ROOT="${WORK_ROOT}/job${ARRAY_JOB_ID:-unknown}"
INTERNAL_STDOUT_LOG="${JOB_WORK_ROOT}/reducer_stdout.log"
INTERNAL_STDERR_LOG="${JOB_WORK_ROOT}/reducer_stderr.log"
SACCT_COMMAND="${EV_GNN_FULL_DIAGNOSTIC_SACCT_COMMAND:-sacct}"
SACCT_ATTEMPTS="${EV_GNN_FULL_DIAGNOSTIC_SACCT_ATTEMPTS:-6}"
SACCT_DELAY_SECONDS="${EV_GNN_FULL_DIAGNOSTIC_SACCT_DELAY_SECONDS:-10}"
SACCT_FIELDS="JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU"
FINAL_BUNDLE_PATH="${OUTPUT_ROOT}/full_infrastructure_diagnostic_eval30_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"
FINAL_BUNDLE_CHECKSUM_PATH="${FINAL_BUNDLE_PATH}.sha256"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

STAGE_D_PYTHON_RUNTIME_OUTPUT=""

require_python_runtime() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Stage D Python interpreter is required and executable at ${PYTHON_BIN}" >&2
    exit 2
  fi

  local runtime_output
  if ! runtime_output="$("${PYTHON_BIN}" - "${PYTHON_BIN}" <<'PY'
import sys
from pathlib import Path
from typing import TypeAlias

expected = Path(sys.argv[1]).resolve()
actual = Path(sys.executable).resolve()
if actual != expected:
    raise SystemExit(
        f"Stage D Python interpreter mismatch: expected {expected}, got {actual}"
    )
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Stage D Python >=3.10 is required, got {sys.version.split()[0]}"
    )
print(f"python_bin={sys.argv[1]}")
print(f"python_executable={sys.executable}")
print(f"python_executable_resolved={actual}")
print(f"python_version={sys.version.split()[0]}")
PY
  )"; then
    echo "ERROR: Stage D Python interpreter contract failed for ${PYTHON_BIN}" >&2
    if [[ -n "${runtime_output}" ]]; then
      printf '%s\n' "${runtime_output}" >&2
    fi
    exit 2
  fi

  STAGE_D_PYTHON_RUNTIME_OUTPUT="${runtime_output}"
}

report_python_runtime() {
  echo "m3_conda_env=${M3_CONDA_ENV}"
  printf '%s\n' "${STAGE_D_PYTHON_RUNTIME_OUTPUT}"
}

if [[ -z "${ARRAY_JOB_ID}" ]]; then
  echo "ERROR: EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID is required" >&2
  exit 2
fi
if [[ ! "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID must be numeric" >&2
  exit 2
fi

require_python_runtime

print_reduce_command() {
  printf "%q " \
    "${PYTHON_BIN}" "${VALIDATOR}" \
    validate-complete-workflow \
    --array-job-id "${ARRAY_JOB_ID}" \
    --task-package-root "${TASK_PACKAGE_ROOT}" \
    --slurm-log-root "${SLURM_LOG_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --work-root "${WORK_ROOT}" \
    --source-commit-sha "${EXPECTED_SOURCE_COMMIT}" \
    --reducer-job-id "${REDUCER_JOB_ID}" \
    --sacct-command "${SACCT_COMMAND}" \
    --sacct-attempts "${SACCT_ATTEMPTS}" \
    --sacct-delay-seconds "${SACCT_DELAY_SECONDS}" \
    --reducer-stdout-log "${INTERNAL_STDOUT_LOG}" \
    --reducer-stderr-log "${INTERNAL_STDERR_LOG}"
  printf "\n"
}

if [[ "${EV_GNN_FULL_DIAGNOSTIC_REDUCER_DRY_RUN:-0}" == "1" ]]; then
  echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_REDUCER_DRY_RUN"
  echo "dry_run=1"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "reducer_job_id=${REDUCER_JOB_ID}"
  echo "repo_root=${REPO_ROOT}"
  report_python_runtime
  echo "task_package_root=${TASK_PACKAGE_ROOT}"
  echo "slurm_log_root=${SLURM_LOG_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "work_root=${WORK_ROOT}"
  echo "expected_final_bundle=${FINAL_BUNDLE_PATH}"
  echo "expected_final_bundle_sha256=${FINAL_BUNDLE_CHECKSUM_PATH}"
  echo "task_package_count=8"
  echo "checkpoint_group_count=40"
  echo "episode_count=1200"
  echo "schema_version=3"
  echo "reconciliation_contract_version=2"
  echo "same_pass_summary=summaries/same_pass_canonical_reconciliation_summary.csv"
  echo "historical_drift_summary=summaries/historical_canonical_drift_summary.csv"
  echo "reconciliation_inventory=summaries/reconciliation_summary_inventory.csv"
  echo "SACCT_COMMAND=sacct -j ${ARRAY_JOB_ID} --parsable2 --noheader --format=${SACCT_FIELDS}"
  echo "REDUCER_COMMAND=$(print_reduce_command)"
  echo "DRY_RUN_NO_REDUCTION_OR_PACKAGING"
  exit 0
fi

if [[ -z "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT is required" >&2
  exit 2
fi
if [[ ! -s "${SOURCE_COMMIT_FILE}" ]]; then
  echo "ERROR: SOURCE_COMMIT_SHA.txt is required at ${SOURCE_COMMIT_FILE}" >&2
  exit 2
fi
SOURCE_COMMIT_SHA="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
if [[ "${SOURCE_COMMIT_SHA}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: source commit ${SOURCE_COMMIT_SHA} does not match ${EXPECTED_SOURCE_COMMIT}" >&2
  exit 3
fi

mkdir -p "${JOB_WORK_ROOT}" "${OUTPUT_ROOT}"
exec > >(tee -a "${INTERNAL_STDOUT_LOG}") 2> >(tee -a "${INTERNAL_STDERR_LOG}" >&2)

if command -v module >/dev/null 2>&1; then
  module load miniforge3/24.3.0-0
  conda activate "${M3_CONDA_ENV}"
fi

echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_REDUCER_START"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "reducer_job_id=${REDUCER_JOB_ID}"
echo "source_commit_sha=${SOURCE_COMMIT_SHA}"
report_python_runtime
echo "task_package_root=${TASK_PACKAGE_ROOT}"
echo "slurm_log_root=${SLURM_LOG_ROOT}"
echo "output_root=${OUTPUT_ROOT}"
echo "work_root=${WORK_ROOT}"

"${PYTHON_BIN}" "${VALIDATOR}" \
  validate-complete-workflow \
  --array-job-id "${ARRAY_JOB_ID}" \
  --task-package-root "${TASK_PACKAGE_ROOT}" \
  --slurm-log-root "${SLURM_LOG_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --work-root "${WORK_ROOT}" \
  --source-commit-sha "${SOURCE_COMMIT_SHA}" \
  --reducer-job-id "${REDUCER_JOB_ID}" \
  --sacct-command "${SACCT_COMMAND}" \
  --sacct-attempts "${SACCT_ATTEMPTS}" \
  --sacct-delay-seconds "${SACCT_DELAY_SECONDS}" \
  --reducer-stdout-log "${INTERNAL_STDOUT_LOG}" \
  --reducer-stderr-log "${INTERNAL_STDERR_LOG}"

echo "TASK_PACKAGE_COUNT=8"
echo "CHECKPOINT_GROUP_COUNT=40"
echo "EPISODE_COUNT=1200"
echo "FORMAL_SOURCE_JOB=58513929"
echo "SCHEMA_VERSION=3"
echo "COMPLETE_DIAGNOSTIC_EVAL30_BUNDLE_OK=1"
echo "COMPLETE_DIAGNOSTIC_EVAL30_BUNDLE_PATH=${FINAL_BUNDLE_PATH}"
echo "COMPLETE_DIAGNOSTIC_EVAL30_BUNDLE_SHA256_PATH=${FINAL_BUNDLE_CHECKSUM_PATH}"
echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_REDUCER_COMPLETED"
