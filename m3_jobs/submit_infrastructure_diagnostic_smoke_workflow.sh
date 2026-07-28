#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CURRENT_DIR="$(pwd -P)"
SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
EXPECTED_SOURCE_COMMIT="${EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT:-}"
OUTPUT_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
RUN_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/infrastructure_diagnostic_smoke}"
FORMAL_PACKAGE_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_PACKAGE_ROOT:-${OUTPUT_ROOT}}"
FORMAL_COMPLETE_BUNDLE="${EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_COMPLETE_BUNDLE:-${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz}"
ARRAY_SCRIPT="${REPO_ROOT}/m3_jobs/19_infrastructure_diagnostic_smoke_eval.slurm"
REDUCER_SCRIPT="${REPO_ROOT}/m3_jobs/20_infrastructure_diagnostic_smoke_reduce_bundle.slurm"
VALIDATOR="${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py"
FORMAL_JOB_ID="58513929"
SACCT_FIELDS="JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

print_task_mapping() {
  echo "task_id=0 scale=25cp algorithm=actiongnn formal_task_id=0 episode_seed=710000"
  echo "task_id=1 scale=25cp algorithm=hierarchical formal_task_id=5 episode_seed=710000"
  echo "task_id=2 scale=100cp algorithm=actiongnn formal_task_id=10 episode_seed=720000"
  echo "task_id=3 scale=100cp algorithm=hierarchical formal_task_id=15 episode_seed=720000"
  echo "task_id=4 scale=500cp algorithm=actiongnn formal_task_id=20 episode_seed=730000"
  echo "task_id=5 scale=500cp algorithm=hierarchical formal_task_id=25 episode_seed=730000"
  echo "task_id=6 scale=1000cp algorithm=actiongnn formal_task_id=30 episode_seed=740000"
  echo "task_id=7 scale=1000cp algorithm=hierarchical formal_task_id=35 episode_seed=740000"
}

array_export_vars() {
  printf "%s" "ALL"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_REPO_ROOT=%s" "${REPO_ROOT}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT=%s" "${EXPECTED_SOURCE_COMMIT}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_JOB_ID=%s" "${FORMAL_JOB_ID}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_PACKAGE_ROOT=%s" "${FORMAL_PACKAGE_ROOT}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_COMPLETE_BUNDLE=%s" "${FORMAL_COMPLETE_BUNDLE}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_OUTPUT_ROOT=%s" "${OUTPUT_ROOT}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_RUN_ROOT=%s" "${RUN_ROOT}"
}

reducer_export_vars() {
  local array_job_id="$1"
  printf "%s" "$(array_export_vars)"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_ARRAY_JOB_ID=%s" "${array_job_id}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_TASK_PACKAGE_ROOT=%s" "${OUTPUT_ROOT}"
  printf ",EV_GNN_DIAGNOSTIC_SMOKE_SLURM_LOG_ROOT=%s" "${OUTPUT_ROOT}"
}

normalise_sbatch_job_id() {
  local raw="$1"
  local first_line="${raw%%$'\n'*}"
  local numeric="${first_line%%;*}"
  if ! [[ "${numeric}" =~ ^[0-9]+$ ]]; then
    die "sbatch --parsable returned non-numeric job ID: ${raw}"
  fi
  printf "%s" "${numeric}"
}

if [[ "${EV_GNN_DIAGNOSTIC_SMOKE_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  echo "INFRASTRUCTURE_DIAGNOSTIC_SMOKE_SUBMIT_DRY_RUN"
  echo "dry_run=1"
  echo "repo_root=${REPO_ROOT}"
  echo "expected_source_commit=${EXPECTED_SOURCE_COMMIT:-<required-in-real-mode>}"
  echo "formal_job_id=${FORMAL_JOB_ID}"
  echo "formal_package_root=${FORMAL_PACKAGE_ROOT}"
  echo "formal_complete_bundle=${FORMAL_COMPLETE_BUNDLE}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "run_root=${RUN_ROOT}"
  print_task_mapping
  echo "SBATCH_ARRAY_COMMAND=sbatch --parsable --export=$(array_export_vars) ${ARRAY_SCRIPT}"
  echo "SBATCH_REDUCER_COMMAND=sbatch --parsable --dependency=afterok:<array_job_id> --export=$(reducer_export_vars "<array_job_id>") ${REDUCER_SCRIPT}"
  echo "SQUEUE_COMMAND=squeue -j <array_job_id>,<reducer_job_id>"
  echo "SACCT_COMMAND=sacct -j <array_job_id> --parsable2 --noheader --format=${SACCT_FIELDS}"
  echo "M3_TAR_VALIDATION_COMMAND=python ${VALIDATOR} validate-complete-bundle --bundle ${OUTPUT_ROOT}/infrastructure_diagnostic_smoke_complete_evidence_job<array_job_id>.tar.gz"
  echo "M3_CHECKSUM_VALIDATION_COMMAND=cd ${OUTPUT_ROOT} && sha256sum -c infrastructure_diagnostic_smoke_complete_evidence_job<array_job_id>.tar.gz.sha256"
  echo "LOCAL_SCP_COMMAND=scp cche0357@m3.massive.org.au:${OUTPUT_ROOT}/infrastructure_diagnostic_smoke_complete_evidence_job<array_job_id>.tar.gz cche0357@m3.massive.org.au:${OUTPUT_ROOT}/infrastructure_diagnostic_smoke_complete_evidence_job<array_job_id>.tar.gz.sha256 ."
  echo "DRY_RUN_NO_SBATCH_CALLED"
  exit 0
fi

[[ "${CURRENT_DIR}" == "${REPO_ROOT}" ]] || die "run from physical extracted source root: ${REPO_ROOT}"
[[ -n "${EXPECTED_SOURCE_COMMIT}" ]] || die "EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT is required"
[[ -s "${SOURCE_COMMIT_FILE}" ]] || die "SOURCE_COMMIT_SHA.txt is required at ${SOURCE_COMMIT_FILE}"
[[ -s "${ARRAY_SCRIPT}" ]] || die "missing array script: ${ARRAY_SCRIPT}"
[[ -s "${REDUCER_SCRIPT}" ]] || die "missing reducer script: ${REDUCER_SCRIPT}"
[[ -s "${VALIDATOR}" ]] || die "missing validator: ${VALIDATOR}"

SOURCE_COMMIT_SHA="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
[[ "${SOURCE_COMMIT_SHA}" == "${EXPECTED_SOURCE_COMMIT}" ]] || die "source commit ${SOURCE_COMMIT_SHA} does not match ${EXPECTED_SOURCE_COMMIT}"

PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evgnn_diag_smoke_preflight.XXXXXX")"
trap 'rm -rf "${PREFLIGHT_DIR}"' EXIT

for task_id in 0 1 2 3 4 5 6 7; do
  python "${VALIDATOR}" \
    resolve-package \
    --task-id "${task_id}" \
    --individual-package-root "${FORMAL_PACKAGE_ROOT}" \
    --complete-bundle "${FORMAL_COMPLETE_BUNDLE}" \
    --staging-dir "${PREFLIGHT_DIR}/resolved" \
    > "${PREFLIGHT_DIR}/task${task_id}_resolution.json"
  PACKAGE_PATH="$(
    python - "${PREFLIGHT_DIR}/task${task_id}_resolution.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["package_path"])
PY
  )"
  python "${VALIDATOR}" \
    validate-formal-package \
    --task-id "${task_id}" \
    --package "${PACKAGE_PATH}" \
    --extract-dir "${PREFLIGHT_DIR}/extract_task${task_id}" \
    > "${PREFLIGHT_DIR}/task${task_id}_formal_validation.json"
done

RAW_ARRAY_JOB_ID="$(
  sbatch \
    --parsable \
    --export="$(array_export_vars)" \
    "${ARRAY_SCRIPT}"
)"
ARRAY_JOB_ID="$(normalise_sbatch_job_id "${RAW_ARRAY_JOB_ID}")"
RAW_REDUCER_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --export="$(reducer_export_vars "${ARRAY_JOB_ID}")" \
    "${REDUCER_SCRIPT}"
)"
REDUCER_JOB_ID="$(normalise_sbatch_job_id "${RAW_REDUCER_JOB_ID}")"
FINAL_BUNDLE="${OUTPUT_ROOT}/infrastructure_diagnostic_smoke_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"
FINAL_BUNDLE_SHA256="${FINAL_BUNDLE}.sha256"

echo "INFRASTRUCTURE_DIAGNOSTIC_SMOKE_WORKFLOW_SUBMITTED"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "reducer_job_id=${REDUCER_JOB_ID}"
echo "expected_final_bundle=${FINAL_BUNDLE}"
echo "expected_final_bundle_sha256=${FINAL_BUNDLE_SHA256}"
echo "SQUEUE_COMMAND=squeue -j ${ARRAY_JOB_ID},${REDUCER_JOB_ID}"
echo "SACCT_COMMAND=sacct -j ${ARRAY_JOB_ID} --parsable2 --noheader --format=${SACCT_FIELDS}"
echo "M3_TAR_VALIDATION_COMMAND=python ${VALIDATOR} validate-complete-bundle --bundle ${FINAL_BUNDLE}"
echo "M3_CHECKSUM_VALIDATION_COMMAND=cd ${OUTPUT_ROOT} && sha256sum -c $(basename "${FINAL_BUNDLE_SHA256}")"
echo "LOCAL_SCP_COMMAND=scp cche0357@m3.massive.org.au:${FINAL_BUNDLE} cche0357@m3.massive.org.au:${FINAL_BUNDLE_SHA256} ."
