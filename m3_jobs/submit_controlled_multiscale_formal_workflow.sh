#!/bin/bash

set -euo pipefail

REPO_ROOT=/projects/fr57/cche0357/EV-GNN
OUTPUT_ROOT=/projects/fr57/cche0357/EV-GNN_outputs
ARRAY_SCRIPT=m3_jobs/17_controlled_multiscale_formal_train_eval.slurm
REDUCER_SCRIPT=m3_jobs/18_controlled_multiscale_formal_reduce_bundle.slurm
CURRENT_DIR="$(pwd -P)"

print_final_commands() {
  local array_job_id="$1"
  local reducer_job_id="$2"
  local final_bundle="${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job${array_job_id}.tar.gz"
  local bundle_filename
  bundle_filename="$(basename "${final_bundle}")"

  echo "squeue -j ${array_job_id},${reducer_job_id}"
  echo "sacct -j ${array_job_id},${reducer_job_id} --format=JobID,JobName%40,State,ExitCode,Elapsed,AllocCPUS,MaxRSS"
  echo "ls -lh ${final_bundle}"
  echo "tar -tzf ${final_bundle} >/dev/null && echo \"M3_FINAL_BUNDLE_READABLE_OK\""
  echo "mkdir -p ~/Downloads/EVGNN_Formal_Evidence"
  echo "scp cche0357@m3.massive.org.au:${final_bundle} ~/Downloads/EVGNN_Formal_Evidence/"
  echo "tar -tzf ~/Downloads/EVGNN_Formal_Evidence/${bundle_filename} >/dev/null && echo \"LOCAL_FINAL_BUNDLE_READABLE_OK\""
}

if [[ "${EV_GNN_FORMAL_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  if [[ "${CURRENT_DIR}" != "${REPO_ROOT}" ]]; then
    echo "dry_run_note=real submission must be run from ${REPO_ROOT}; current_dir=${CURRENT_DIR}"
  fi
  echo "ARRAY_SUBMIT_COMMAND=sbatch --parsable ${ARRAY_SCRIPT}"
  echo "REDUCER_SUBMIT_COMMAND=sbatch --parsable --dependency=afterok:<array_jobid> --export=ALL,EV_GNN_FORMAL_ARRAY_JOB_ID=<array_jobid> ${REDUCER_SCRIPT}"
  echo "expected_final_bundle_path_template=${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job<array_jobid>.tar.gz"
  echo "EXPECTED_MONITORING_COMMANDS"
  print_final_commands "<array_jobid>" "<reducer_jobid>"
  echo "DRY_RUN_NO_JOBS_SUBMITTED"
  exit 0
fi

if [[ "${CURRENT_DIR}" != "${REPO_ROOT}" ]]; then
  echo "ERROR: submit helper must be run from ${REPO_ROOT}; current directory is ${CURRENT_DIR}" >&2
  exit 2
fi

ARRAY_JOB_ID="$(sbatch --parsable "${ARRAY_SCRIPT}")"
REDUCER_JOB_ID="$(
  sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --export=ALL,EV_GNN_FORMAL_ARRAY_JOB_ID=${ARRAY_JOB_ID} \
    "${REDUCER_SCRIPT}"
)"
FINAL_BUNDLE="${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"

echo "ARRAY_JOB_ID=${ARRAY_JOB_ID}"
echo "REDUCER_JOB_ID=${REDUCER_JOB_ID}"
echo "FINAL_EXPECTED_BUNDLE_PATH=${FINAL_BUNDLE}"
echo "MONITORING_COMMANDS"
print_final_commands "${ARRAY_JOB_ID}" "${REDUCER_JOB_ID}"
