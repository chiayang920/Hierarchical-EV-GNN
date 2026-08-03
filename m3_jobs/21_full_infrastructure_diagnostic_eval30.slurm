#!/bin/bash
#SBATCH --job-name=evgnn_full_infra_diag_eval30
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

REPO_ROOT="${EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
RUN_ROOT="${EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/full_infrastructure_diagnostic_eval30}"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
FORMAL_PACKAGE_ROOT="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT:-${OUTPUT_ROOT}}"
FORMAL_COMPLETE_BUNDLE="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE:-${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz}"
EXPECTED_SOURCE_COMMIT="${EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT:-}"
SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
VALIDATOR="${REPO_ROOT}/scripts/validate_full_infrastructure_diagnostic_eval30.py"
EVALUATOR_SCRIPT="${EV_GNN_FULL_DIAGNOSTIC_EVALUATOR_SCRIPT:-${REPO_ROOT}/evaluate_td3_gnn_infrastructure_diagnostics.py}"
M3_CONDA_ENV="${EV_GNN_FULL_DIAGNOSTIC_M3_CONDA_ENV:-/scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu}"
PYTHON_BIN="${M3_CONDA_ENV}/bin/python"

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

FORMAL_JOB_ID="58513929"
DEVICE="cpu"
DETERMINISTIC="true"
EVAL_EXPL_NOISE="0.0"
EXPECTED_TRAINING_SEEDS="0,1,2,3,4"
EXPECTED_EVAL_EPISODES="30"
EXPECTED_EPISODE_COUNT="150"

TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${TASK_ID}" ]] || [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]] || (( TASK_ID < 0 || TASK_ID > 7 )); then
  echo "ERROR: Stage D task ID must be an integer from 0 to 7; observed '${TASK_ID}'" >&2
  exit 2
fi

if [[ ! -f "${VALIDATOR}" ]]; then
  echo "ERROR: Stage D validator not found: ${VALIDATOR}" >&2
  exit 3
fi

require_python_runtime
report_python_runtime

MAPPING_OUTPUT="$("${PYTHON_BIN}" "${VALIDATOR}" task-mapping --task-id "${TASK_ID}")"

MAPPED_TASK_ID=""
SCALE=""
ALGORITHM=""
TRAINING_SEEDS=""
FORMAL_TASK_IDS=""
EVAL_EPISODES_FROM_MAPPING=""
SCALE_BASE_SEED_OFFSET=""
EVAL_SEED_OFFSETS=""
FIRST_EPISODE_SEEDS=""

while IFS='=' read -r key value; do
  case "${key}" in
    task_id) MAPPED_TASK_ID="${value}" ;;
    scale) SCALE="${value}" ;;
    algorithm) ALGORITHM="${value}" ;;
    training_seeds) TRAINING_SEEDS="${value}" ;;
    formal_task_ids) FORMAL_TASK_IDS="${value}" ;;
    eval_episodes) EVAL_EPISODES_FROM_MAPPING="${value}" ;;
    scale_base_seed_offset) SCALE_BASE_SEED_OFFSET="${value}" ;;
    eval_seed_offsets) EVAL_SEED_OFFSETS="${value}" ;;
    first_episode_seeds) FIRST_EPISODE_SEEDS="${value}" ;;
    *)
      echo "ERROR: unsupported task-mapping field '${key}'" >&2
      exit 4
      ;;
  esac
done <<< "${MAPPING_OUTPUT}"

if [[ "${MAPPED_TASK_ID}" != "${TASK_ID}" ]]; then
  echo "ERROR: task-mapping task ID mismatch" >&2
  exit 5
fi
if [[ "${TRAINING_SEEDS}" != "${EXPECTED_TRAINING_SEEDS}" ]]; then
  echo "ERROR: task-mapping training seeds mismatch" >&2
  exit 6
fi
if [[ "${EVAL_EPISODES_FROM_MAPPING}" != "${EXPECTED_EVAL_EPISODES}" ]]; then
  echo "ERROR: task-mapping eval episode count mismatch" >&2
  exit 7
fi
if [[ ! "${FORMAL_TASK_IDS}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo "ERROR: task-mapping formal task IDs are invalid" >&2
  exit 8
fi
if [[ ! "${SCALE_BASE_SEED_OFFSET}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: task-mapping scale base seed offset is invalid" >&2
  exit 9
fi

IFS=',' read -r -a TRAINING_SEED_ARRAY <<< "${TRAINING_SEEDS}"
IFS=',' read -r -a FORMAL_TASK_ID_ARRAY <<< "${FORMAL_TASK_IDS}"
IFS=',' read -r -a EVAL_SEED_OFFSET_ARRAY <<< "${EVAL_SEED_OFFSETS}"
IFS=',' read -r -a FIRST_EPISODE_SEED_ARRAY <<< "${FIRST_EPISODE_SEEDS}"

if [[ "${#TRAINING_SEED_ARRAY[@]}" -ne 5 ]] \
  || [[ "${#FORMAL_TASK_ID_ARRAY[@]}" -ne 5 ]] \
  || [[ "${#EVAL_SEED_OFFSET_ARRAY[@]}" -ne 5 ]] \
  || [[ "${#FIRST_EPISODE_SEED_ARRAY[@]}" -ne 5 ]]; then
  echo "ERROR: Stage D task mapping must contain exactly five seeds, formal task IDs, offsets, and first episode seeds" >&2
  exit 10
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${JOB_ID}}"
TASK_DIR="${RUN_ROOT}/job${ARRAY_JOB_ID}/task${TASK_ID}"
PACKAGE_NAME="m3_full_infrastructure_diagnostic_eval30_${SCALE}_${ALGORITHM}_seeds0-4_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
PACKAGE_PATH="${OUTPUT_ROOT}/${PACKAGE_NAME}"
PACKAGE_SIDECAR_PATH="${PACKAGE_PATH}.sha256"

print_shell_command() {
  printf "%q " "$@"
  printf "\n"
}

json_get() {
  "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get(sys.argv[2], ""))' "$1" "$2"
}

if [[ "${EV_GNN_FULL_DIAGNOSTIC_DRY_RUN:-0}" == "1" ]]; then
  echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_TASK_DRY_RUN"
  echo "task_id=${TASK_ID}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "formal_job_id=${FORMAL_JOB_ID}"
  echo "training_seeds=${TRAINING_SEEDS}"
  echo "formal_task_ids=${FORMAL_TASK_IDS}"
  echo "eval_episodes=${EVAL_EPISODES_FROM_MAPPING}"
  echo "scale_base_seed_offset=${SCALE_BASE_SEED_OFFSET}"
  echo "eval_seed_offsets=${EVAL_SEED_OFFSETS}"
  echo "first_episode_seeds=${FIRST_EPISODE_SEEDS}"
  echo "expected_episode_count=${EXPECTED_EPISODE_COUNT}"
  echo "repo_root=${REPO_ROOT}"
  echo "formal_package_root=${FORMAL_PACKAGE_ROOT}"
  echo "formal_complete_bundle=${FORMAL_COMPLETE_BUNDLE}"
  echo "task_dir=${TASK_DIR}"
  echo "package_path=${PACKAGE_PATH}"
  echo "same_pass_canonical_eval30=diagnostics/same_pass_canonical_eval30.csv"
  echo "same_pass_reconciliation=validation/same_pass_canonical_reconciliation.csv"
  echo "historical_drift=validation/historical_canonical_drift.csv"
  echo "reconciliation_summary=runtime_metadata/reconciliation_summary.json"
  echo "reconciliation_contract_version=2"
  for index in 0 1 2 3 4; do
    training_seed="${TRAINING_SEED_ARRAY[${index}]}"
    formal_task_id="${FORMAL_TASK_ID_ARRAY[${index}]}"
    eval_seed_offset="${EVAL_SEED_OFFSET_ARRAY[${index}]}"
    first_episode_seed="${FIRST_EPISODE_SEED_ARRAY[${index}]}"
    seed_root="${TASK_DIR}/seed${training_seed}"
    checkpoint_prefix="${seed_root}/checkpoint/model.best"
    config_path="${seed_root}/config/formal_config.yaml"
    diagnostic_dir="${seed_root}/diagnostics"
    run_name="full_infrastructure_diagnostic_eval30_${SCALE}_${ALGORITHM}_seed${training_seed}_job${ARRAY_JOB_ID}_task${TASK_ID}"
    echo "EVALUATOR_COMMAND_SEED_${training_seed}=${PYTHON_BIN} ${EVALUATOR_SCRIPT} --algorithm ${ALGORITHM} --scale ${SCALE} --config ${config_path} --seed ${training_seed} --eval_episodes 30 --checkpoint ${checkpoint_prefix} --output_dir ${diagnostic_dir} --run_name ${run_name} --device ${DEVICE} --max_episode_steps <validated_simulation_length> --deterministic ${DETERMINISTIC} --eval_expl_noise ${EVAL_EXPL_NOISE} --eval_seed_offset ${eval_seed_offset} --matrix_job_id ${ARRAY_JOB_ID} # formal_task_id=${formal_task_id} first_episode_seed=${first_episode_seed}"
  done
  echo "DRY_RUN_NO_EVALUATION_OR_PACKAGING"
  exit 0
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

if command -v module >/dev/null 2>&1; then
  module load miniforge3/24.3.0-0
  conda activate "${M3_CONDA_ENV}"
fi

if [[ -z "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT must be exported in real mode" >&2
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

if [[ -e "${PACKAGE_PATH}" ]]; then
  echo "ERROR: final task package already exists: ${PACKAGE_PATH}" >&2
  exit 3
fi
if [[ -e "${PACKAGE_SIDECAR_PATH}" ]]; then
  echo "ERROR: final task package checksum already exists: ${PACKAGE_SIDECAR_PATH}" >&2
  exit 3
fi
if [[ -e "${TASK_DIR}" ]]; then
  echo "ERROR: task directory already exists: ${TASK_DIR}" >&2
  exit 3
fi

mkdir -p "${TASK_DIR}/logs" "${OUTPUT_ROOT}"
: > "${TASK_DIR}/logs/stdout.log"
: > "${TASK_DIR}/logs/stderr.log"

log_line() {
  echo "$*"
  echo "$*" >> "${TASK_DIR}/logs/stdout.log"
}

log_line "FULL_INFRASTRUCTURE_DIAGNOSTIC_TASK_START"
log_line "task_id=${TASK_ID}"
log_line "scale=${SCALE}"
log_line "algorithm=${ALGORITHM}"
log_line "formal_job_id=${FORMAL_JOB_ID}"
log_line "array_job_id=${ARRAY_JOB_ID}"
log_line "source_commit_sha=${SOURCE_COMMIT_SHA}"

cd "${REPO_ROOT}"

for index in 0 1 2 3 4; do
  training_seed="${TRAINING_SEED_ARRAY[${index}]}"
  formal_task_id="${FORMAL_TASK_ID_ARRAY[${index}]}"
  eval_seed_offset="${EVAL_SEED_OFFSET_ARRAY[${index}]}"
  seed_root="${TASK_DIR}/seed${training_seed}"
  formal_staging_dir="${seed_root}/formal_package_staging"
  formal_extract_dir="${seed_root}/formal_extract"
  checkpoint_dir="${seed_root}/checkpoint"
  config_dir="${seed_root}/config"
  canonical_dir="${seed_root}/canonical"
  diagnostic_dir="${seed_root}/diagnostics"
  validation_dir="${seed_root}/validation"
  logs_dir="${seed_root}/logs"
  runtime_dir="${seed_root}/runtime_metadata"
  mkdir -p \
    "${formal_staging_dir}" \
    "${checkpoint_dir}" \
    "${config_dir}" \
    "${canonical_dir}" \
    "${diagnostic_dir}" \
    "${validation_dir}" \
    "${logs_dir}" \
    "${runtime_dir}"
  : > "${logs_dir}/stderr.log"

  "${PYTHON_BIN}" "${VALIDATOR}" \
    resolve-formal-package \
    --task-id "${TASK_ID}" \
    --training-seed "${training_seed}" \
    --individual-package-root "${FORMAL_PACKAGE_ROOT}" \
    --complete-bundle "${FORMAL_COMPLETE_BUNDLE}" \
    --staging-dir "${formal_staging_dir}" \
    > "${runtime_dir}/source_package_resolution.json"
  formal_package_path="$(json_get "${runtime_dir}/source_package_resolution.json" package_path)"

  "${PYTHON_BIN}" "${VALIDATOR}" \
    validate-formal-package \
    --task-id "${TASK_ID}" \
    --training-seed "${training_seed}" \
    --package "${formal_package_path}" \
    --extract-dir "${formal_extract_dir}" \
    > "${runtime_dir}/formal_package_validation.json"
  max_episode_steps="$(json_get "${runtime_dir}/formal_package_validation.json" simulation_length)"

  formal_run_name="controlled_multiscale_formal_${SCALE}_${ALGORITHM}_seed${training_seed}"
  formal_train_dir="${formal_extract_dir}/train/${formal_run_name}"
  formal_config="${formal_extract_dir}/config/${SCALE}_${ALGORITHM}_seed${training_seed}_config.yaml"
  formal_canonical="${formal_extract_dir}/eval/${SCALE}_${ALGORITHM}_seed${training_seed}_eval30.csv"
  checkpoint_prefix="${checkpoint_dir}/model.best"
  cp "${formal_config}" "${config_dir}/formal_config.yaml"
  cp "${formal_canonical}" "${canonical_dir}/complete_eval30.csv"
  cp "${formal_train_dir}/model.best_actor" "${checkpoint_prefix}_actor"
  cp "${formal_train_dir}/model.best_actor_optimizer" "${checkpoint_prefix}_actor_optimizer"
  cp "${formal_train_dir}/model.best_critic" "${checkpoint_prefix}_critic"
  cp "${formal_train_dir}/model.best_critic_optimizer" "${checkpoint_prefix}_critic_optimizer"
  cp "${formal_train_dir}/kwargs.yaml" "${checkpoint_dir}/kwargs.yaml"

  run_name="full_infrastructure_diagnostic_eval30_${SCALE}_${ALGORITHM}_seed${training_seed}_job${ARRAY_JOB_ID}_task${TASK_ID}"
  diagnostic_command=(
    "${PYTHON_BIN}" "${EVALUATOR_SCRIPT}"
    --algorithm "${ALGORITHM}"
    --scale "${SCALE}"
    --config "${config_dir}/formal_config.yaml"
    --seed "${training_seed}"
    --eval_episodes "${EXPECTED_EVAL_EPISODES}"
    --checkpoint "${checkpoint_prefix}"
    --output_dir "${diagnostic_dir}"
    --run_name "${run_name}"
    --device "${DEVICE}"
    --max_episode_steps "${max_episode_steps}"
    --deterministic "${DETERMINISTIC}"
    --eval_expl_noise "${EVAL_EXPL_NOISE}"
    --eval_seed_offset "${eval_seed_offset}"
    --matrix_job_id "${ARRAY_JOB_ID}"
  )
  print_shell_command "${diagnostic_command[@]}" > "${runtime_dir}/diagnostic_command.txt"
  set +e
  "${diagnostic_command[@]}" > "${logs_dir}/stdout.log" 2> "${logs_dir}/stderr.log"
  eval_status="$?"
  set -e
  if [[ "${eval_status}" -ne 0 ]]; then
    echo "ERROR: diagnostic evaluator failed for seed ${training_seed} with exit status ${eval_status}" >&2
    exit "${eval_status}"
  fi
  if [[ -s "${logs_dir}/stderr.log" ]]; then
    echo "ERROR: seed stderr log is non-empty for seed ${training_seed}" >&2
    exit 4
  fi

  "${PYTHON_BIN}" "${VALIDATOR}" \
    prepare-seed-validation \
    --diagnostic-dir "${diagnostic_dir}" \
    --historical-canonical-csv "${canonical_dir}/complete_eval30.csv" \
    --validation-dir "${validation_dir}" \
    --runtime-metadata-dir "${runtime_dir}" \
    --formal-validation-json "${runtime_dir}/formal_package_validation.json" \
    --stage-d-source-commit-sha "${SOURCE_COMMIT_SHA}" \
    --config-path "${config_dir}/formal_config.yaml" \
    --checkpoint-prefix "${checkpoint_prefix}" \
    --task-id "${TASK_ID}" \
    --training-seed "${training_seed}" \
    > "${runtime_dir}/seed_validation_preparation.json"

  "${PYTHON_BIN}" "${VALIDATOR}" \
    validate-seed-output \
    --directory "${seed_root}" \
    --task-id "${TASK_ID}" \
    --training-seed "${training_seed}" \
    > "${runtime_dir}/seed_output_validation.json"

  log_line "seed${training_seed}_completed=formal_task${formal_task_id}"
done

"${PYTHON_BIN}" "${VALIDATOR}" \
  publish-task-package \
  --task-root "${TASK_DIR}" \
  --task-id "${TASK_ID}" \
  --array-job-id "${ARRAY_JOB_ID}" \
  --source-commit-sha "${SOURCE_COMMIT_SHA}" \
  --package "${PACKAGE_PATH}"

log_line "FULL_INFRASTRUCTURE_DIAGNOSTIC_TASK_COMPLETED"
log_line "package_path=${PACKAGE_PATH}"
