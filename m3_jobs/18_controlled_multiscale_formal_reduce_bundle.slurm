#!/bin/bash
#SBATCH --job-name=evgnn_controlled_formal_reduce
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%j.err

set -euo pipefail

REPO_ROOT=/projects/fr57/cche0357/EV-GNN
OUTPUT_ROOT=/projects/fr57/cche0357/EV-GNN_outputs
REDUCE_ROOT=/scratch2/fr57/cche0357/EV-GNN_runs/controlled_multiscale_formal_reduce

EXPECTED_TASK_COUNT=40
ARRAY_JOB_ID="${EV_GNN_FORMAL_ARRAY_JOB_ID:-}"
REDUCER_JOB_ID="${SLURM_JOB_ID:-manual}"
START_TIMESTAMP=
END_TIMESTAMP=
PACKAGE_COUNT=0
STDOUT_LOG_COUNT=0
STDERR_LOG_COUNT=0
CANONICAL_CSV_COUNT=0
AGGREGATION_EXIT_STATUS=not_started

if [[ -z "${ARRAY_JOB_ID}" ]]; then
  echo "ERROR: EV_GNN_FORMAL_ARRAY_JOB_ID=<array_jobid> is required" >&2
  exit 2
fi

REDUCER_JOB_DIR="${REDUCE_ROOT}/job${ARRAY_JOB_ID}"
BUNDLE_ROOT="${REDUCER_JOB_DIR}/controlled_multiscale_formal_complete_evidence_job${ARRAY_JOB_ID}"
TASK_PACKAGES_DIR="${BUNDLE_ROOT}/task_packages"
SLURM_STDOUT_DIR="${BUNDLE_ROOT}/slurm_logs/stdout"
SLURM_STDERR_DIR="${BUNDLE_ROOT}/slurm_logs/stderr"
CANONICAL_CSV_DIR="${BUNDLE_ROOT}/canonical_eval30_csv"
AGGREGATION_OUTPUTS_DIR="${BUNDLE_ROOT}/aggregation_outputs"
SOURCE_MANIFESTS_DIR="${BUNDLE_ROOT}/source_manifests"
TASK_METADATA_DIR="${BUNDLE_ROOT}/task_metadata"
REDUCER_METADATA_DIR="${BUNDLE_ROOT}/reducer_metadata"
COMPLETE_EVIDENCE_BUNDLE_PATH="${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"

SCALES=(25cp 100cp 500cp 1000cp)
ALGORITHMS=(actiongnn hierarchical)
SEEDS=(0 1 2 3 4)

print_shell_command() {
  printf "%q " "$@"
  printf "\n"
}

iso_now() {
  date -Iseconds 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ"
}

aggregation_command=(
  python scripts/aggregate_controlled_multiscale_eval30.py
  --csv-dir "${CANONICAL_CSV_DIR}"
  --out-dir "${AGGREGATION_OUTPUTS_DIR}"
)

if [[ "${EV_GNN_FORMAL_REDUCER_DRY_RUN:-0}" == "1" ]]; then
  echo "CONTROLLED_FORMAL_REDUCER_START"
  echo "dry_run=1"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "expected_task_package_pattern=${OUTPUT_ROOT}/m3_controlled_multiscale_formal_<scale>_<algorithm>_seed<seed>_job${ARRAY_JOB_ID}_task<taskid>.tar.gz"
  echo "expected_stdout_log_pattern=${OUTPUT_ROOT}/evgnn_controlled_formal_matrix_${ARRAY_JOB_ID}_<taskid>.out"
  echo "expected_stderr_log_pattern=${OUTPUT_ROOT}/evgnn_controlled_formal_matrix_${ARRAY_JOB_ID}_<taskid>.err"
  echo "expected_final_bundle_path=${COMPLETE_EVIDENCE_BUNDLE_PATH}"
  echo "expected_aggregation_command=$(print_shell_command "${aggregation_command[@]}")"
  echo "DRY_RUN_NO_PACKAGE_OR_LOG_REQUIREMENTS"
  exit 0
fi

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

cd "${REPO_ROOT}"

START_TIMESTAMP="$(iso_now)"

echo "CONTROLLED_FORMAL_REDUCER_START"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "reducer_job_id=${REDUCER_JOB_ID}"
echo "start_timestamp=${START_TIMESTAMP}"

shopt -s nullglob

task_package_matches=("${OUTPUT_ROOT}/m3_controlled_multiscale_formal_"*"_job${ARRAY_JOB_ID}_task"*.tar.gz)
stdout_log_matches=("${OUTPUT_ROOT}/evgnn_controlled_formal_matrix_${ARRAY_JOB_ID}_"*.out)
stderr_log_matches=("${OUTPUT_ROOT}/evgnn_controlled_formal_matrix_${ARRAY_JOB_ID}_"*.err)

PACKAGE_COUNT="${#task_package_matches[@]}"
STDOUT_LOG_COUNT="${#stdout_log_matches[@]}"
STDERR_LOG_COUNT="${#stderr_log_matches[@]}"

if [[ "${PACKAGE_COUNT}" -ne "${EXPECTED_TASK_COUNT}" ]]; then
  echo "ERROR: expected ${EXPECTED_TASK_COUNT} task packages for array job ${ARRAY_JOB_ID}, found ${PACKAGE_COUNT}" >&2
  exit 3
fi
if [[ "${STDOUT_LOG_COUNT}" -ne "${EXPECTED_TASK_COUNT}" ]]; then
  echo "ERROR: expected ${EXPECTED_TASK_COUNT} stdout logs for array job ${ARRAY_JOB_ID}, found ${STDOUT_LOG_COUNT}" >&2
  exit 3
fi
if [[ "${STDERR_LOG_COUNT}" -ne "${EXPECTED_TASK_COUNT}" ]]; then
  echo "ERROR: expected ${EXPECTED_TASK_COUNT} stderr logs for array job ${ARRAY_JOB_ID}, found ${STDERR_LOG_COUNT}" >&2
  exit 3
fi

echo "TASK_PACKAGE_COUNT=${PACKAGE_COUNT}"
echo "STDOUT_LOG_COUNT=${STDOUT_LOG_COUNT}"
echo "STDERR_LOG_COUNT=${STDERR_LOG_COUNT}"

mkdir -p "${REDUCER_JOB_DIR}"
if [[ -e "${BUNDLE_ROOT}" ]]; then
  echo "ERROR: reducer staging root already exists: ${BUNDLE_ROOT}" >&2
  exit 3
fi

mkdir -p \
  "${TASK_PACKAGES_DIR}" \
  "${SLURM_STDOUT_DIR}" \
  "${SLURM_STDERR_DIR}" \
  "${CANONICAL_CSV_DIR}" \
  "${AGGREGATION_OUTPUTS_DIR}" \
  "${SOURCE_MANIFESTS_DIR}" \
  "${TASK_METADATA_DIR}" \
  "${REDUCER_METADATA_DIR}"

require_readable_file() {
  local required_path="$1"
  if [[ ! -s "${required_path}" ]]; then
    echo "ERROR: required file missing or empty: ${required_path}" >&2
    exit 4
  fi
}

require_existing_file() {
  local required_path="$1"
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: required file missing: ${required_path}" >&2
    exit 4
  fi
}

extract_package_member() {
  local package_path="$1"
  local package_member="$2"
  local destination_path="$3"
  local matching_members=()

  mapfile -t matching_members < <(
    tar -tzf "${package_path}" |
      awk -v expected_member="${package_member}" '
        {
          member_name = $0
          sub(/^\.\//, "", member_name)
          if (member_name == expected_member) {
            print $0
          }
        }
      '
  )

  if [[ "${#matching_members[@]}" -ne 1 ]]; then
    echo "ERROR: expected exactly one tar member ${package_member} in ${package_path}, found ${#matching_members[@]}" >&2
    exit 5
  fi

  tar -xOf "${package_path}" "${matching_members[0]}" > "${destination_path}"
  require_readable_file "${destination_path}"
}

task_id=0
for scale in "${SCALES[@]}"; do
  for algorithm in "${ALGORITHMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      task_label="${scale}_${algorithm}_seed${seed}"
      package_path="${OUTPUT_ROOT}/m3_controlled_multiscale_formal_${scale}_${algorithm}_seed${seed}_job${ARRAY_JOB_ID}_task${task_id}.tar.gz"
      stdout_log_path="${OUTPUT_ROOT}/evgnn_controlled_formal_matrix_${ARRAY_JOB_ID}_${task_id}.out"
      stderr_log_path="${OUTPUT_ROOT}/evgnn_controlled_formal_matrix_${ARRAY_JOB_ID}_${task_id}.err"
      canonical_csv_name="${scale}_${algorithm}_seed${seed}_eval30.csv"

      require_readable_file "${package_path}"
      require_existing_file "${stdout_log_path}"
      require_existing_file "${stderr_log_path}"
      tar -tzf "${package_path}" >/dev/null

      cp "${package_path}" "${TASK_PACKAGES_DIR}/"
      cp "${stdout_log_path}" "${SLURM_STDOUT_DIR}/"
      cp "${stderr_log_path}" "${SLURM_STDERR_DIR}/"

      extract_package_member \
        "${package_path}" \
        "eval/${canonical_csv_name}" \
        "${CANONICAL_CSV_DIR}/${canonical_csv_name}"
      extract_package_member \
        "${package_path}" \
        "runtime_metadata/source_manifest.sha256" \
        "${SOURCE_MANIFESTS_DIR}/${task_label}_source_manifest.sha256"
      extract_package_member \
        "${package_path}" \
        "runtime_metadata/task_runtime_metadata.env" \
        "${TASK_METADATA_DIR}/${task_label}_task_runtime_metadata.env"
      extract_package_member \
        "${package_path}" \
        "runtime_metadata/eval30_csv_validation.env" \
        "${TASK_METADATA_DIR}/${task_label}_eval30_csv_validation.env"
      extract_package_member \
        "${package_path}" \
        "runtime_metadata/package_file_checksums.sha256" \
        "${TASK_METADATA_DIR}/${task_label}_package_file_checksums.sha256"

      task_id=$((task_id + 1))
    done
  done
done

canonical_csv_matches=("${CANONICAL_CSV_DIR}/"*"_eval30.csv")
CANONICAL_CSV_COUNT="${#canonical_csv_matches[@]}"
if [[ "${CANONICAL_CSV_COUNT}" -ne "${EXPECTED_TASK_COUNT}" ]]; then
  echo "ERROR: expected ${EXPECTED_TASK_COUNT} extracted canonical CSV files, found ${CANONICAL_CSV_COUNT}" >&2
  exit 6
fi
echo "CANONICAL_CSV_COUNT=${CANONICAL_CSV_COUNT}"

echo "AGGREGATION_COMMAND"
print_shell_command "${aggregation_command[@]}"
set +e
"${aggregation_command[@]}"
AGGREGATION_EXIT_STATUS="$?"
set -e

if [[ "${AGGREGATION_EXIT_STATUS}" -ne 0 ]]; then
  END_TIMESTAMP="$(iso_now)"
  {
    echo "array_job_id=${ARRAY_JOB_ID}"
    echo "reducer_job_id=${REDUCER_JOB_ID}"
    echo "start_timestamp=${START_TIMESTAMP}"
    echo "end_timestamp=${END_TIMESTAMP}"
    echo "package_count=${PACKAGE_COUNT}"
    echo "stdout_log_count=${STDOUT_LOG_COUNT}"
    echo "stderr_log_count=${STDERR_LOG_COUNT}"
    echo "canonical_csv_count=${CANONICAL_CSV_COUNT}"
    echo "aggregation_exit_status=${AGGREGATION_EXIT_STATUS}"
    echo "complete_evidence_bundle_path=${COMPLETE_EVIDENCE_BUNDLE_PATH}"
  } > "${REDUCER_METADATA_DIR}/reducer_runtime_metadata.env"
  echo "ERROR: aggregation failed with exit status ${AGGREGATION_EXIT_STATUS}" >&2
  exit "${AGGREGATION_EXIT_STATUS}"
fi

required_aggregation_outputs=(
  multiscale_per_seed_metric_summary.csv
  multiscale_paired_metric_tests.csv
  multiscale_metric_direction_summary.csv
  multiscale_scale_level_summary.csv
  controlled_multiscale_formal_summary.md
)

for aggregation_output in "${required_aggregation_outputs[@]}"; do
  require_readable_file "${AGGREGATION_OUTPUTS_DIR}/${aggregation_output}"
done
echo "AGGREGATION_OK"

END_TIMESTAMP="$(iso_now)"
{
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "reducer_job_id=${REDUCER_JOB_ID}"
  echo "start_timestamp=${START_TIMESTAMP}"
  echo "end_timestamp=${END_TIMESTAMP}"
  echo "package_count=${PACKAGE_COUNT}"
  echo "stdout_log_count=${STDOUT_LOG_COUNT}"
  echo "stderr_log_count=${STDERR_LOG_COUNT}"
  echo "canonical_csv_count=${CANONICAL_CSV_COUNT}"
  echo "aggregation_exit_status=${AGGREGATION_EXIT_STATUS}"
  echo "complete_evidence_bundle_path=${COMPLETE_EVIDENCE_BUNDLE_PATH}"
} > "${REDUCER_METADATA_DIR}/reducer_runtime_metadata.env"

python - "${BUNDLE_ROOT}" <<'PY'
import hashlib
import sys
from pathlib import Path

bundle_root = Path(sys.argv[1])
metadata_dir = bundle_root / "reducer_metadata"
file_list_path = metadata_dir / "complete_evidence_file_list.txt"
sha_manifest_path = metadata_dir / "complete_evidence_sha256_manifest.sha256"

payload_paths = sorted(
    path
    for path in bundle_root.rglob("*")
    if path.is_file() and path not in {file_list_path, sha_manifest_path}
)

file_list_paths = [
    path.relative_to(bundle_root).as_posix()
    for path in payload_paths
]
file_list_paths.append(file_list_path.relative_to(bundle_root).as_posix())
file_list_paths.append(sha_manifest_path.relative_to(bundle_root).as_posix())
file_list_path.write_text("\n".join(file_list_paths) + "\n", encoding="utf-8")

manifest_paths = sorted(
    path
    for path in bundle_root.rglob("*")
    if path.is_file() and path != sha_manifest_path
)
with sha_manifest_path.open("w", encoding="utf-8") as sha_manifest:
    for path in manifest_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative_path = path.relative_to(bundle_root).as_posix()
        sha_manifest.write(f"{digest}  {relative_path}\n")
PY

tar -czf \
  "${COMPLETE_EVIDENCE_BUNDLE_PATH}" \
  -C "${REDUCER_JOB_DIR}" \
  "$(basename "${BUNDLE_ROOT}")"
tar -tzf "${COMPLETE_EVIDENCE_BUNDLE_PATH}" >/dev/null

echo "COMPLETE_EVIDENCE_BUNDLE_OK"
echo "COMPLETE_EVIDENCE_BUNDLE_PATH=${COMPLETE_EVIDENCE_BUNDLE_PATH}"
echo "CONTROLLED_FORMAL_REDUCER_COMPLETED"
