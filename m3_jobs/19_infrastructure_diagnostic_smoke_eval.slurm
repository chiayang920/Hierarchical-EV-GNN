#!/bin/bash
#SBATCH --job-name=evgnn_infra_diag_smoke
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
RUN_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/infrastructure_diagnostic_smoke}"
OUTPUT_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
FORMAL_PACKAGE_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_PACKAGE_ROOT:-${OUTPUT_ROOT}}"
FORMAL_COMPLETE_BUNDLE="${EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_COMPLETE_BUNDLE:-${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz}"
FORMAL_JOB_ID="58513929"
SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
EXPECTED_SOURCE_COMMIT="${EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT:-}"
DEVICE="cpu"
DETERMINISTIC="true"
EVAL_EXPL_NOISE="0.0"
MAX_EPISODE_STEPS=""
REQUESTED_MEMORY="64G"
REQUESTED_WALL_TIME="06:00:00"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is required and must be 0-7" >&2
  exit 2
fi
if ! [[ "${TASK_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be an integer in 0-7; got ${TASK_ID}" >&2
  exit 2
fi

case "${TASK_ID}" in
  0) SCALE="25cp"; ALGORITHM="actiongnn"; FORMAL_TASK_ID=0; EPISODE_SEED=710000; CONFIG_NAME="PublicPST_25cp.yaml" ;;
  1) SCALE="25cp"; ALGORITHM="hierarchical"; FORMAL_TASK_ID=5; EPISODE_SEED=710000; CONFIG_NAME="PublicPST_25cp.yaml" ;;
  2) SCALE="100cp"; ALGORITHM="actiongnn"; FORMAL_TASK_ID=10; EPISODE_SEED=720000; CONFIG_NAME="PublicPST_100.yaml" ;;
  3) SCALE="100cp"; ALGORITHM="hierarchical"; FORMAL_TASK_ID=15; EPISODE_SEED=720000; CONFIG_NAME="PublicPST_100.yaml" ;;
  4) SCALE="500cp"; ALGORITHM="actiongnn"; FORMAL_TASK_ID=20; EPISODE_SEED=730000; CONFIG_NAME="PublicPST_500.yaml" ;;
  5) SCALE="500cp"; ALGORITHM="hierarchical"; FORMAL_TASK_ID=25; EPISODE_SEED=730000; CONFIG_NAME="PublicPST_500.yaml" ;;
  6) SCALE="1000cp"; ALGORITHM="actiongnn"; FORMAL_TASK_ID=30; EPISODE_SEED=740000; CONFIG_NAME="PublicPST_1000.yaml" ;;
  7) SCALE="1000cp"; ALGORITHM="hierarchical"; FORMAL_TASK_ID=35; EPISODE_SEED=740000; CONFIG_NAME="PublicPST_1000.yaml" ;;
  *) echo "ERROR: invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0-7" >&2; exit 2 ;;
esac

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
JOB_NAME="${SLURM_JOB_NAME:-evgnn_infra_diag_smoke}"
TASK_DIR="${RUN_ROOT}/job${ARRAY_JOB_ID}/${SCALE}/${ALGORITHM}/seed0"
FORMAL_STAGING_DIR="${TASK_DIR}/formal_package_staging"
FORMAL_EXTRACT_DIR="${TASK_DIR}/formal_extract"
CHECKPOINT_STAGING_DIR="${TASK_DIR}/checkpoint_staging"
DIAGNOSTIC_DIR="${TASK_DIR}/diagnostics"
CANONICAL_DIR="${TASK_DIR}/canonical"
CONFIG_DIR="${TASK_DIR}/config"
VALIDATION_DIR="${TASK_DIR}/validation"
METADATA_DIR="${TASK_DIR}/runtime_metadata"
PACKAGE_STAGING_DIR="${TASK_DIR}/package_staging"
PACKAGE_PATH="${OUTPUT_ROOT}/m3_infrastructure_diagnostic_smoke_${SCALE}_${ALGORITHM}_seed0_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
PACKAGE_BASENAME="$(basename "${PACKAGE_PATH}")"
RUN_NAME="infrastructure_diagnostic_smoke_${SCALE}_${ALGORITHM}_seed0_job${ARRAY_JOB_ID}_task${TASK_ID}"
STDOUT_LOG="${TASK_DIR}/stdout.log"
STDERR_LOG="${TASK_DIR}/stderr.log"

FORMAL_RUN_NAME="controlled_multiscale_formal_${SCALE}_${ALGORITHM}_seed0"
FORMAL_CONFIG_MEMBER="config/${SCALE}_${ALGORITHM}_seed0_config.yaml"
FORMAL_CANONICAL_MEMBER="eval/${SCALE}_${ALGORITHM}_seed0_eval30.csv"
FORMAL_TRAIN_DIR="train/${FORMAL_RUN_NAME}"

format_seconds() {
  local total_seconds="$1"
  printf "%02d:%02d:%02d" \
    $((total_seconds / 3600)) \
    $(((total_seconds % 3600) / 60)) \
    $((total_seconds % 60))
}

print_shell_command() {
  printf "%q " "$@"
  printf "\n"
}

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "ERROR: required file missing or empty: ${path}" >&2
    exit 4
  fi
}

write_single_hash() {
  local path="$1"
  local label="$2"
  local output_path="$3"
  python - "${path}" "${label}" "${output_path}" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
label = sys.argv[2]
output = Path(sys.argv[3])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(f"{digest}  {label}\n", encoding="utf-8")
PY
}

write_checkpoint_hashes() {
  local output_path="$1"
  shift
  python - "${output_path}" "$@" <<'PY'
import hashlib
import sys
from pathlib import Path

output = Path(sys.argv[1])
pairs = sys.argv[2:]
if len(pairs) % 2:
    raise SystemExit("checkpoint hash arguments must be path/label pairs")
with output.open("w", encoding="utf-8") as handle:
    for index in range(0, len(pairs), 2):
        path = Path(pairs[index])
        label = pairs[index + 1]
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {label}\n")
PY
}

write_canonical_episode0() {
  python - "${CANONICAL_DIR}/complete_eval30.csv" "${CANONICAL_DIR}/canonical_episode0.csv" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
with source.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = [
        row for row in reader
        if row.get("row_type") == "episode" and str(row.get("episode_index")) == "0"
    ]
    if reader.fieldnames is None:
        raise SystemExit("canonical CSV has no header")
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one canonical episode 0 row, found {len(rows)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerow(rows[0])
PY
}

prepare_package_staging() {
  rm -rf "${PACKAGE_STAGING_DIR}"
  mkdir -p "${PACKAGE_STAGING_DIR}"
  cp "${STDOUT_LOG}" "${PACKAGE_STAGING_DIR}/stdout.log"
  cp "${STDERR_LOG}" "${PACKAGE_STAGING_DIR}/stderr.log"
  cp -R "${DIAGNOSTIC_DIR}" "${PACKAGE_STAGING_DIR}/diagnostics"
  cp -R "${CANONICAL_DIR}" "${PACKAGE_STAGING_DIR}/canonical"
  cp -R "${CONFIG_DIR}" "${PACKAGE_STAGING_DIR}/config"
  cp -R "${VALIDATION_DIR}" "${PACKAGE_STAGING_DIR}/validation"
  cp -R "${METADATA_DIR}" "${PACKAGE_STAGING_DIR}/runtime_metadata"
}

generate_package_manifest() {
  python - "${PACKAGE_STAGING_DIR}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata_dir = root / "runtime_metadata"
file_list_path = metadata_dir / "package_file_list.txt"
manifest_path = metadata_dir / "package_file_checksums.sha256"

payload_files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path not in {file_list_path, manifest_path}
)
file_names = [path.relative_to(root).as_posix() for path in payload_files]
file_names.append(file_list_path.relative_to(root).as_posix())
file_names.append(manifest_path.relative_to(root).as_posix())
file_list_path.write_text("\n".join(file_names) + "\n", encoding="utf-8")

manifest_files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path != manifest_path
)
with manifest_path.open("w", encoding="utf-8") as manifest:
    for path in manifest_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write(f"{digest}  {path.relative_to(root).as_posix()}\n")
PY
}

package_task() {
  prepare_package_staging
  generate_package_manifest
  local temp_package_path
  temp_package_path="$(mktemp "${OUTPUT_ROOT}/.${PACKAGE_BASENAME}.tmp.XXXXXX")"
  if ! tar -czf "${temp_package_path}" -C "${PACKAGE_STAGING_DIR}" \
      stdout.log \
      stderr.log \
      diagnostics \
      canonical \
      config \
      validation \
      runtime_metadata; then
    rm -f "${temp_package_path}"
    return 1
  fi
  if ! tar -tzf "${temp_package_path}" >/dev/null; then
    rm -f "${temp_package_path}"
    return 1
  fi
  python "${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py" \
    validate-task-package \
    --task-id "${TASK_ID}" \
    --package "${temp_package_path}" || {
      rm -f "${temp_package_path}"
      return 1
    }
  mv "${temp_package_path}" "${PACKAGE_PATH}"
}

if [[ "${EV_GNN_DIAGNOSTIC_SMOKE_DRY_RUN:-0}" == "1" ]]; then
  echo "INFRASTRUCTURE_DIAGNOSTIC_SMOKE_TASK_START"
  echo "dry_run=1"
  echo "task_id=${TASK_ID}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "formal_job_id=${FORMAL_JOB_ID}"
  echo "formal_task_id=${FORMAL_TASK_ID}"
  echo "episode_seed=${EPISODE_SEED}"
  echo "config_name=${CONFIG_NAME}"
  echo "repo_root=${REPO_ROOT}"
  echo "task_dir=${TASK_DIR}"
  echo "package_path=${PACKAGE_PATH}"
  echo "VALIDATOR_COMMAND=python ${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py task-mapping --task-id ${TASK_ID}"
  echo "DIAGNOSTIC_COMMAND=python evaluate_td3_gnn_infrastructure_diagnostics.py --algorithm ${ALGORITHM} --scale ${SCALE} --config <extracted_formal_config> --seed 0 --eval_episodes 1 --checkpoint <staged_model.best> --output_dir ${DIAGNOSTIC_DIR} --device cpu --deterministic true --eval_expl_noise 0.0 --max_episode_steps <validated_simulation_length> --eval_seed_offset ${EPISODE_SEED} --matrix_job_id ${ARRAY_JOB_ID}"
  echo "DRY_RUN_NO_EVALUATION_OR_PACKAGING"
  exit 0
fi

THREAD_COUNT="${SLURM_CPUS_PER_TASK:-4}"
if [[ "${THREAD_COUNT}" != "4" ]]; then
  echo "ERROR: expected SLURM_CPUS_PER_TASK=4, got ${THREAD_COUNT}" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

if [[ -z "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT must be exported in real mode" >&2
  exit 2
fi

if [[ -e "${PACKAGE_PATH}" ]]; then
  echo "ERROR: final task package already exists: ${PACKAGE_PATH}" >&2
  exit 3
fi

EXPECTED_JOB_RUN_ROOT="${RUN_ROOT}/job${ARRAY_JOB_ID}"
if [[ -e "${TASK_DIR}" ]]; then
  case "${TASK_DIR}" in
    "${EXPECTED_JOB_RUN_ROOT}"/*) ;;
    *) echo "ERROR: refusing to clean task directory outside job run root: ${TASK_DIR}" >&2; exit 3 ;;
  esac
  if [[ "${EV_GNN_DIAGNOSTIC_SMOKE_CLEAN_EXISTING_TASK_DIR:-0}" != "1" ]]; then
    echo "ERROR: task directory already exists: ${TASK_DIR}" >&2
    echo "Set EV_GNN_DIAGNOSTIC_SMOKE_CLEAN_EXISTING_TASK_DIR=1 to delete and recreate it." >&2
    exit 3
  fi
  rm -rf "${TASK_DIR}"
fi

mkdir -p \
  "${TASK_DIR}" \
  "${FORMAL_STAGING_DIR}" \
  "${FORMAL_EXTRACT_DIR}" \
  "${CHECKPOINT_STAGING_DIR}" \
  "${DIAGNOSTIC_DIR}" \
  "${CANONICAL_DIR}" \
  "${CONFIG_DIR}" \
  "${VALIDATION_DIR}" \
  "${METADATA_DIR}" \
  "${OUTPUT_ROOT}"

exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

cd "${REPO_ROOT}"

TASK_START_EPOCH="$(date +%s)"
TASK_START_ISO="$(date -Iseconds)"

echo "INFRASTRUCTURE_DIAGNOSTIC_SMOKE_TASK_START"
echo "task_id=${TASK_ID}"
echo "scale=${SCALE}"
echo "algorithm=${ALGORITHM}"
echo "formal_job_id=${FORMAL_JOB_ID}"
echo "formal_task_id=${FORMAL_TASK_ID}"
echo "episode_seed=${EPISODE_SEED}"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "slurm_job_id=${JOB_ID}"
echo "slurm_job_name=${JOB_NAME}"
echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
echo "memory_request=${REQUESTED_MEMORY}"
echo "wall_time_request=${REQUESTED_WALL_TIME}"
echo "repo_root=${REPO_ROOT}"

require_file "${SOURCE_COMMIT_FILE}"
SOURCE_COMMIT_SHA="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
if [[ "${SOURCE_COMMIT_SHA}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: source commit ${SOURCE_COMMIT_SHA} does not match ${EXPECTED_SOURCE_COMMIT}" >&2
  exit 3
fi
printf "%s\n" "${SOURCE_COMMIT_SHA}" > "${METADATA_DIR}/source_commit_sha.txt"

python "${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py" \
  resolve-package \
  --task-id "${TASK_ID}" \
  --individual-package-root "${FORMAL_PACKAGE_ROOT}" \
  --complete-bundle "${FORMAL_COMPLETE_BUNDLE}" \
  --staging-dir "${FORMAL_STAGING_DIR}" \
  > "${METADATA_DIR}/source_package_resolution.json"

FORMAL_PACKAGE_PATH="$(
  python - "${METADATA_DIR}/source_package_resolution.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["package_path"])
PY
)"
SOURCE_MODE="$(
  python - "${METADATA_DIR}/source_package_resolution.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["source_mode"])
PY
)"
EXPECTED_PACKAGE_NAME="$(
  python - "${METADATA_DIR}/source_package_resolution.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["expected_package_name"])
PY
)"
BUNDLE_MEMBER="$(
  python - "${METADATA_DIR}/source_package_resolution.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("bundle_member", ""))
PY
)"

{
  echo "formal_job_id=${FORMAL_JOB_ID}"
  echo "formal_task_id=${FORMAL_TASK_ID}"
  echo "formal_scale=${SCALE}"
  echo "formal_algorithm=${ALGORITHM}"
  echo "formal_episode_seed=${EPISODE_SEED}"
} > "${METADATA_DIR}/source_formal_job.env"

{
  echo "source_mode=${SOURCE_MODE}"
  echo "package_path=${FORMAL_PACKAGE_PATH}"
  echo "expected_package_name=${EXPECTED_PACKAGE_NAME}"
  if [[ -n "${BUNDLE_MEMBER}" ]]; then
    echo "bundle_member=${BUNDLE_MEMBER}"
  fi
  echo "formal_package_path=${FORMAL_PACKAGE_PATH}"
  echo "formal_package_root=${FORMAL_PACKAGE_ROOT}"
  echo "formal_complete_bundle=${FORMAL_COMPLETE_BUNDLE}"
} > "${METADATA_DIR}/source_package.env"
write_single_hash \
  "${FORMAL_PACKAGE_PATH}" \
  "$(basename "${FORMAL_PACKAGE_PATH}")" \
  "${METADATA_DIR}/source_package.sha256"

python "${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py" \
  validate-formal-package \
  --task-id "${TASK_ID}" \
  --package "${FORMAL_PACKAGE_PATH}" \
  --extract-dir "${FORMAL_EXTRACT_DIR}" \
  > "${METADATA_DIR}/formal_package_validation.json"

MAX_EPISODE_STEPS="$(
  python - "${METADATA_DIR}/formal_package_validation.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["simulation_length"])
PY
)"

FORMAL_CONFIG="${FORMAL_EXTRACT_DIR}/${FORMAL_CONFIG_MEMBER}"
FORMAL_CANONICAL_CSV="${FORMAL_EXTRACT_DIR}/${FORMAL_CANONICAL_MEMBER}"
CHECKPOINT_PREFIX="${CHECKPOINT_STAGING_DIR}/model.best"

require_file "${FORMAL_CONFIG}"
require_file "${FORMAL_CANONICAL_CSV}"

cp "${FORMAL_CONFIG}" "${CONFIG_DIR}/formal_config.yaml"
cp "${FORMAL_CANONICAL_CSV}" "${CANONICAL_DIR}/complete_eval30.csv"
write_canonical_episode0

cp "${FORMAL_EXTRACT_DIR}/${FORMAL_TRAIN_DIR}/model.best_actor" "${CHECKPOINT_PREFIX}_actor"
cp "${FORMAL_EXTRACT_DIR}/${FORMAL_TRAIN_DIR}/model.best_actor_optimizer" "${CHECKPOINT_PREFIX}_actor_optimizer"
cp "${FORMAL_EXTRACT_DIR}/${FORMAL_TRAIN_DIR}/model.best_critic" "${CHECKPOINT_PREFIX}_critic"
cp "${FORMAL_EXTRACT_DIR}/${FORMAL_TRAIN_DIR}/model.best_critic_optimizer" "${CHECKPOINT_PREFIX}_critic_optimizer"
cp "${FORMAL_EXTRACT_DIR}/${FORMAL_TRAIN_DIR}/kwargs.yaml" "${CHECKPOINT_STAGING_DIR}/kwargs.yaml"

write_checkpoint_hashes \
  "${METADATA_DIR}/checkpoint_member_hashes.sha256" \
  "${CHECKPOINT_PREFIX}_actor" "${FORMAL_TRAIN_DIR}/model.best_actor" \
  "${CHECKPOINT_PREFIX}_actor_optimizer" "${FORMAL_TRAIN_DIR}/model.best_actor_optimizer" \
  "${CHECKPOINT_PREFIX}_critic" "${FORMAL_TRAIN_DIR}/model.best_critic" \
  "${CHECKPOINT_PREFIX}_critic_optimizer" "${FORMAL_TRAIN_DIR}/model.best_critic_optimizer" \
  "${CHECKPOINT_STAGING_DIR}/kwargs.yaml" "${FORMAL_TRAIN_DIR}/kwargs.yaml"
cp "${FORMAL_EXTRACT_DIR}/runtime_metadata/source_manifest.sha256" "${METADATA_DIR}/original_source_manifest.sha256"
cp "${FORMAL_EXTRACT_DIR}/runtime_metadata/task_runtime_metadata.env" "${METADATA_DIR}/original_task_runtime_metadata.env"

DIAGNOSTIC_COMMAND=(
  python evaluate_td3_gnn_infrastructure_diagnostics.py
  --algorithm "${ALGORITHM}"
  --scale "${SCALE}"
  --config "${CONFIG_DIR}/formal_config.yaml"
  --seed 0
  --eval_episodes 1
  --checkpoint "${CHECKPOINT_PREFIX}"
  --output_dir "${DIAGNOSTIC_DIR}"
  --run_name "${RUN_NAME}"
  --device "${DEVICE}"
  --max_episode_steps "${MAX_EPISODE_STEPS}"
  --deterministic "${DETERMINISTIC}"
  --eval_expl_noise "${EVAL_EXPL_NOISE}"
  --eval_seed_offset "${EPISODE_SEED}"
  --matrix_job_id "${ARRAY_JOB_ID}"
)
print_shell_command "${DIAGNOSTIC_COMMAND[@]}" > "${METADATA_DIR}/diagnostic_command.txt"

EVAL_START_EPOCH="$(date +%s)"
set +e
/usr/bin/time -v "${DIAGNOSTIC_COMMAND[@]}" > "${METADATA_DIR}/evaluator_stdout.txt" 2> "${METADATA_DIR}/evaluator_time_verbose.txt"
EVAL_EXIT_STATUS="$?"
set -e
EVAL_END_EPOCH="$(date +%s)"
EVAL_WALL_SECONDS=$((EVAL_END_EPOCH - EVAL_START_EPOCH))
if [[ "${EVAL_EXIT_STATUS}" -ne 0 ]]; then
  echo "ERROR: diagnostic evaluator failed with exit status ${EVAL_EXIT_STATUS}" >&2
  exit "${EVAL_EXIT_STATUS}"
fi

python "${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py" \
  validate-diagnostics \
  --task-id "${TASK_ID}" \
  --diagnostic-dir "${DIAGNOSTIC_DIR}" \
  --validation-dir "${VALIDATION_DIR}" \
  --matrix-job-id "${ARRAY_JOB_ID}"

python "${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py" \
  reconcile-canonical \
  --task-id "${TASK_ID}" \
  --episode-diagnostics "${DIAGNOSTIC_DIR}/episode_diagnostics.csv" \
  --canonical-csv "${CANONICAL_DIR}/complete_eval30.csv" \
  --validation-dir "${VALIDATION_DIR}"

TASK_END_EPOCH="$(date +%s)"
TASK_WALL_SECONDS=$((TASK_END_EPOCH - TASK_START_EPOCH))
{
  echo "task_id=${TASK_ID}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "formal_job_id=${FORMAL_JOB_ID}"
  echo "formal_task_id=${FORMAL_TASK_ID}"
  echo "episode_seed=${EPISODE_SEED}"
  echo "slurm_job_id=${JOB_ID}"
  echo "slurm_array_job_id=${ARRAY_JOB_ID}"
  echo "start_timestamp=${TASK_START_ISO}"
  echo "end_timestamp=$(date -Iseconds)"
  echo "evaluation_exit_status=${EVAL_EXIT_STATUS}"
  echo "evaluation_wall_time_seconds=${EVAL_WALL_SECONDS}"
  echo "evaluation_wall_time_hhmmss=$(format_seconds "${EVAL_WALL_SECONDS}")"
  echo "task_wall_time_seconds=${TASK_WALL_SECONDS}"
  echo "task_wall_time_hhmmss=$(format_seconds "${TASK_WALL_SECONDS}")"
  echo "package_path=${PACKAGE_PATH}"
} > "${METADATA_DIR}/task_runtime_metadata.env"

package_task

echo "INFRASTRUCTURE_DIAGNOSTIC_SMOKE_TASK_COMPLETED"
echo "task_wall_time=$(format_seconds "${TASK_WALL_SECONDS}")"
echo "package_path=${PACKAGE_PATH}"
