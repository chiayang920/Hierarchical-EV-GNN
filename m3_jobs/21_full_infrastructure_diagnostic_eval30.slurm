#!/bin/bash
#SBATCH --job-name=evgnn_infra_eval30
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-7
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/full_infra_eval30_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/full_infra_eval30_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
RUN_ROOT="${EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/full_infrastructure_diagnostics_eval30}"
FORMAL_PACKAGE_ROOT="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT:-${OUTPUT_ROOT}}"
FORMAL_COMPLETE_BUNDLE="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE:-${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz}"
FORMAL_JOB_ID="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID:-58513929}"
EXPECTED_SOURCE_COMMIT="${EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT:-}"
SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
VALIDATOR="${REPO_ROOT}/scripts/validate_full_infrastructure_diagnostic_eval30.py"
PACKAGE_VALIDATOR="${REPO_ROOT}/scripts/full_infrastructure_diagnostic_eval30_packages.py"
SMOKE_VALIDATOR="${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py"

: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
: "${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID}"
JOB_ID="${SLURM_JOB_ID}"

if ! [[ "${TASK_ID}" =~ ^[0-7]$ ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be 0-7; got ${TASK_ID}" >&2
  exit 2
fi
if ! [[ "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]] || ! [[ "${JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: Slurm job identifiers must be numeric" >&2
  exit 2
fi
if [[ "${FORMAL_JOB_ID}" != "58513929" ]]; then
  echo "ERROR: formal job ID must equal 58513929" >&2
  exit 2
fi

MAPPING_OUTPUT="$(python "${VALIDATOR}" task-mapping --task-id "${TASK_ID}")"
value_for() {
  local key="$1"
  printf "%s\n" "${MAPPING_OUTPUT}" | awk -F= -v key="${key}" '$1 == key {print substr($0, index($0, "=") + 1)}'
}
SCALE="$(value_for scale)"
ALGORITHM="$(value_for algorithm)"
TRAINING_SEEDS="$(value_for training_seeds)"
FORMAL_TASK_IDS="$(value_for formal_task_ids)"
EVAL_EPISODES="$(value_for eval_episodes)"
EVAL_SEED_OFFSET="$(value_for eval_seed_offset)"

if [[ "${TRAINING_SEEDS}" != "0,1,2,3,4" ]] || [[ "${EVAL_EPISODES}" != "30" ]]; then
  echo "ERROR: Stage D mapping contract mismatch" >&2
  exit 2
fi

case "${SCALE}" in
  25cp) CONFIG_NAME="PublicPST_25cp.yaml"; EXPECTED_CHARGERS=25; EXPECTED_TRANSFORMERS=3 ;;
  100cp) CONFIG_NAME="PublicPST_100.yaml"; EXPECTED_CHARGERS=100; EXPECTED_TRANSFORMERS=7 ;;
  500cp) CONFIG_NAME="PublicPST_500.yaml"; EXPECTED_CHARGERS=500; EXPECTED_TRANSFORMERS=35 ;;
  1000cp) CONFIG_NAME="PublicPST_1000.yaml"; EXPECTED_CHARGERS=1000; EXPECTED_TRANSFORMERS=70 ;;
  *) echo "ERROR: unsupported scale from mapping: ${SCALE}" >&2; exit 2 ;;
esac

TASK_DIR="${RUN_ROOT}/job${SLURM_ARRAY_JOB_ID}/task${SLURM_ARRAY_TASK_ID}"
PACKAGE_BASENAME="full_infrastructure_diagnostic_eval30_task${TASK_ID}_job${ARRAY_JOB_ID}.tar.gz"
PACKAGE_PATH="${OUTPUT_ROOT}/${PACKAGE_BASENAME}"
PACKAGE_SIDECAR="${PACKAGE_PATH}.sha256"
TEMP_PACKAGE="${OUTPUT_ROOT}/.${PACKAGE_BASENAME}.tmp.${SLURM_JOB_ID}"
TEMP_SIDECAR="${TEMP_PACKAGE}.sha256"
PACKAGE_STAGING="${TASK_DIR}/package_staging"

print_evaluator_command() {
  local training_seed="$1"
  local formal_task_id="$2"
  local seed_root="${TASK_DIR}/seed${training_seed}"
  printf "EVALUATOR_COMMAND_SEED_%s=python evaluate_td3_gnn_infrastructure_diagnostics.py --algorithm %s --scale %s --config %s/config/formal_config.yaml --seed %s --eval_episodes 30 --checkpoint %s/checkpoint/model.best --output_dir %s/diagnostics --run_name full_infrastructure_diagnostic_eval30_%s_%s_seed%s_job%s_task%s --device cpu --max_episode_steps '<validated_simulation_length>' --deterministic true --eval_expl_noise 0.0 --eval_seed_offset %s --matrix_job_id %s # formal_task_id=%s\n" \
    "${training_seed}" "${ALGORITHM}" "${SCALE}" "${seed_root}" "${training_seed}" \
    "${seed_root}" "${seed_root}" "${SCALE}" "${ALGORITHM}" "${training_seed}" \
    "${ARRAY_JOB_ID}" "${TASK_ID}" "${EVAL_SEED_OFFSET}" "${ARRAY_JOB_ID}" "${formal_task_id}"
}

if [[ "${EV_GNN_FULL_DIAGNOSTIC_DRY_RUN:-0}" == "1" ]]; then
  echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_TASK_DRY_RUN"
  echo "task_id=${TASK_ID}"
  echo "scale=${SCALE}"
  echo "algorithm=${ALGORITHM}"
  echo "training_seeds=${TRAINING_SEEDS}"
  echo "formal_task_ids=${FORMAL_TASK_IDS}"
  echo "eval_episodes=${EVAL_EPISODES}"
  echo "eval_seed_offset=${EVAL_SEED_OFFSET}"
  echo "expected_episode_count=150"
  echo "config_name=${CONFIG_NAME}"
  echo "repo_root=${REPO_ROOT}"
  echo "task_dir=${TASK_DIR}"
  echo "package_path=${PACKAGE_PATH}"
  IFS=',' read -r -a FORMAL_ID_ARRAY <<< "${FORMAL_TASK_IDS}"
  for training_seed in 0 1 2 3 4; do
    print_evaluator_command "${training_seed}" "${FORMAL_ID_ARRAY[${training_seed}]}"
  done
  echo "DRY_RUN_NO_EVALUATION_OR_PACKAGING"
  exit 0
fi

require_regular_file() {
  local path="$1"
  if [[ ! -f "${path}" ]] || [[ -L "${path}" ]] || [[ ! -s "${path}" ]]; then
    echo "ERROR: required non-empty regular file missing: ${path}" >&2
    exit 3
  fi
}

if [[ -z "${EXPECTED_SOURCE_COMMIT}" ]] || ! [[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT must be a 40-character lowercase SHA" >&2
  exit 2
fi
require_regular_file "${SOURCE_COMMIT_FILE}"
require_regular_file "${VALIDATOR}"
require_regular_file "${PACKAGE_VALIDATOR}"
require_regular_file "${SMOKE_VALIDATOR}"
require_regular_file "${FORMAL_COMPLETE_BUNDLE}"

SOURCE_COMMIT_SHA="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
if [[ "${SOURCE_COMMIT_SHA}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: source commit mismatch: ${SOURCE_COMMIT_SHA} != ${EXPECTED_SOURCE_COMMIT}" >&2
  exit 3
fi
if [[ -e "${PACKAGE_PATH}" ]] || [[ -e "${PACKAGE_SIDECAR}" ]]; then
  echo "ERROR: refusing to overwrite existing task evidence: ${PACKAGE_PATH}" >&2
  exit 3
fi
if [[ -e "${TASK_DIR}" ]]; then
  echo "ERROR: refusing stale task directory: ${TASK_DIR}" >&2
  exit 3
fi

mkdir -p "${TASK_DIR}" "${OUTPUT_ROOT}" "${PACKAGE_STAGING}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONWARNINGS="ignore:pkg_resources is deprecated as an API"

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu
cd "${REPO_ROOT}"

IFS=',' read -r -a FORMAL_ID_ARRAY <<< "${FORMAL_TASK_IDS}"
CHECKPOINT_INVENTORY="${TASK_DIR}/checkpoint_inventory.csv"
RUNTIME_SUMMARY="${TASK_DIR}/runtime_summary.csv"
printf "task_id,scale,algorithm,training_seed,formal_task_id,status\n" > "${CHECKPOINT_INVENTORY}"
printf "training_seed,state,exit_code,elapsed_seconds\n" > "${RUNTIME_SUMMARY}"

for training_seed in 0 1 2 3 4; do
  FORMAL_TASK_ID="${FORMAL_ID_ARRAY[${training_seed}]}"
  SEED_ROOT="${TASK_DIR}/seed${training_seed}"
  FORMAL_STAGING="${SEED_ROOT}/formal_staging"
  FORMAL_EXTRACT="${SEED_ROOT}/formal_extract"
  CHECKPOINT_DIR="${SEED_ROOT}/checkpoint"
  CONFIG_DIR="${SEED_ROOT}/config"
  CANONICAL_DIR="${SEED_ROOT}/canonical"
  DIAGNOSTIC_DIR="${SEED_ROOT}/diagnostics"
  VALIDATION_DIR="${SEED_ROOT}/validation"
  METADATA_DIR="${SEED_ROOT}/runtime_metadata"
  LOG_DIR="${SEED_ROOT}/logs"
  mkdir -p \
    "${FORMAL_STAGING}" "${FORMAL_EXTRACT}" "${CHECKPOINT_DIR}" \
    "${CONFIG_DIR}" "${CANONICAL_DIR}" "${DIAGNOSTIC_DIR}" \
    "${VALIDATION_DIR}" "${METADATA_DIR}" "${LOG_DIR}"

  EXPECTED_PACKAGE_NAME="m3_controlled_multiscale_formal_${SCALE}_${ALGORITHM}_seed${training_seed}_job${FORMAL_JOB_ID}_task${FORMAL_TASK_ID}.tar.gz"
  FORMAL_PACKAGE_PATH="${FORMAL_STAGING}/${EXPECTED_PACKAGE_NAME}"

  python - \
    "${FORMAL_PACKAGE_ROOT}" "${FORMAL_COMPLETE_BUNDLE}" \
    "${EXPECTED_PACKAGE_NAME}" "${FORMAL_PACKAGE_PATH}" <<'PY'
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

individual_root = Path(sys.argv[1])
complete_bundle = Path(sys.argv[2])
expected_name = sys.argv[3]
output_path = Path(sys.argv[4])
individual = individual_root / expected_name


def validate_members(tar):
    seen = set()
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe tar path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise SystemExit(f"unsupported tar member: {member.name}")
        normalised = path.as_posix()
        if normalised in seen:
            raise SystemExit(f"duplicate tar member: {normalised}")
        seen.add(normalised)


if individual.is_file():
    with tarfile.open(individual, "r:gz") as tar:
        validate_members(tar)
    shutil.copy2(individual, output_path)
    raise SystemExit(0)

matches = []
with tarfile.open(complete_bundle, "r:gz") as tar:
    validate_members(tar)
    for member in tar.getmembers():
        parts = PurePosixPath(member.name).parts
        if (
            member.isfile()
            and PurePosixPath(member.name).name == expected_name
            and "task_packages" in parts[:-1]
        ):
            matches.append(member)
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one formal package {expected_name}, found {len(matches)}"
        )
    payload = tar.extractfile(matches[0])
    if payload is None:
        raise SystemExit(f"unable to read formal package: {matches[0].name}")
    with output_path.open("wb") as output:
        shutil.copyfileobj(payload, output)

with tarfile.open(output_path, "r:gz") as tar:
    validate_members(tar)
PY

  FORMAL_VALIDATION_JSON="${METADATA_DIR}/formal_package_validation.json"
  python - \
    "${FORMAL_PACKAGE_PATH}" "${FORMAL_EXTRACT}" "${FORMAL_VALIDATION_JSON}" \
    "${TASK_ID}" "${SCALE}" "${ALGORITHM}" "${training_seed}" "${FORMAL_TASK_ID}" \
    "${EXPECTED_CHARGERS}" "${EXPECTED_TRANSFORMERS}" "${EVAL_SEED_OFFSET}" <<'PY'
import json
import sys
from pathlib import Path

from scripts import validate_infrastructure_diagnostic_smoke as smoke

package = Path(sys.argv[1])
extract_dir = Path(sys.argv[2])
output_json = Path(sys.argv[3])
task_id = int(sys.argv[4])
scale = sys.argv[5]
algorithm = sys.argv[6]
training_seed = int(sys.argv[7])
formal_task_id = int(sys.argv[8])
expected_chargers = int(sys.argv[9])
expected_transformers = int(sys.argv[10])
seed_offset = int(sys.argv[11])

run_name = f"controlled_multiscale_formal_{scale}_{algorithm}_seed{training_seed}"
config_member = f"config/{scale}_{algorithm}_seed{training_seed}_config.yaml"
canonical_member = f"eval/{scale}_{algorithm}_seed{training_seed}_eval30.csv"
train_dir = f"train/{run_name}"
required = [
    f"{train_dir}/model.best_actor",
    f"{train_dir}/model.best_actor_optimizer",
    f"{train_dir}/model.best_critic",
    f"{train_dir}/model.best_critic_optimizer",
    f"{train_dir}/kwargs.yaml",
    config_member,
    canonical_member,
    "runtime_metadata/source_manifest.sha256",
    "runtime_metadata/task_runtime_metadata.env",
    "runtime_metadata/package_file_checksums.sha256",
]
root = smoke.safe_extract_tar(package, extract_dir)
smoke.require_files(root, required)
smoke.verify_extracted_manifest(
    root,
    "runtime_metadata/package_file_checksums.sha256",
    required_coverage=[
        item for item in required
        if item != "runtime_metadata/package_file_checksums.sha256"
    ],
)
mapping = {
    "task_id": task_id,
    "scale": scale,
    "algorithm": algorithm,
    "formal_task_id": formal_task_id,
    "expected_charger_rows": expected_chargers,
    "expected_transformer_rows": expected_transformers,
}
config_values = smoke.validate_formal_config(root / config_member, mapping)
rows, _ = smoke.read_rows(root / canonical_member)
episodes = [row for row in rows if row.get("row_type") == "episode"]
if len(episodes) != 30:
    raise SystemExit(f"canonical eval CSV must contain 30 episode rows, got {len(episodes)}")
seen = set()
for row in episodes:
    episode_index = smoke.integer_value(row.get("episode_index"), "episode_index")
    if episode_index in seen or not 0 <= episode_index < 30:
        raise SystemExit(f"invalid canonical episode index: {episode_index}")
    seen.add(episode_index)
    if str(row.get("algorithm")) != algorithm:
        raise SystemExit("canonical algorithm mismatch")
    if smoke.integer_value(row.get("seed"), "seed") != training_seed:
        raise SystemExit("canonical training seed mismatch")
    expected_episode_seed = seed_offset + training_seed + episode_index
    if smoke.integer_value(row.get("episode_seed"), "episode_seed") != expected_episode_seed:
        raise SystemExit("canonical episode seed mismatch")
if seen != set(range(30)):
    raise SystemExit("canonical episode inventory incomplete")

output_json.write_text(
    json.dumps(
        {
            "status": "ok",
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": training_seed,
            "formal_task_id": formal_task_id,
            "extract_root": str(root),
            "config_member": config_member,
            "canonical_member": canonical_member,
            "train_dir": train_dir,
            **config_values,
        },
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

  read_json_field() {
    local path="$1"
    local field="$2"
    python - "${path}" "${field}" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload[sys.argv[2]])
PY
  }

  FORMAL_CONFIG_MEMBER="$(read_json_field "${FORMAL_VALIDATION_JSON}" config_member)"
  FORMAL_CANONICAL_MEMBER="$(read_json_field "${FORMAL_VALIDATION_JSON}" canonical_member)"
  FORMAL_TRAIN_DIR="$(read_json_field "${FORMAL_VALIDATION_JSON}" train_dir)"
  MAX_EPISODE_STEPS="$(read_json_field "${FORMAL_VALIDATION_JSON}" simulation_length)"

  cp "${FORMAL_EXTRACT}/${FORMAL_CONFIG_MEMBER}" "${CONFIG_DIR}/formal_config.yaml"
  cp "${FORMAL_EXTRACT}/${FORMAL_CANONICAL_MEMBER}" "${CANONICAL_DIR}/complete_eval30.csv"
  CHECKPOINT_PREFIX="${CHECKPOINT_DIR}/model.best"
  cp "${FORMAL_EXTRACT}/${FORMAL_TRAIN_DIR}/model.best_actor" "${CHECKPOINT_PREFIX}_actor"
  cp "${FORMAL_EXTRACT}/${FORMAL_TRAIN_DIR}/model.best_actor_optimizer" "${CHECKPOINT_PREFIX}_actor_optimizer"
  cp "${FORMAL_EXTRACT}/${FORMAL_TRAIN_DIR}/model.best_critic" "${CHECKPOINT_PREFIX}_critic"
  cp "${FORMAL_EXTRACT}/${FORMAL_TRAIN_DIR}/model.best_critic_optimizer" "${CHECKPOINT_PREFIX}_critic_optimizer"
  cp "${FORMAL_EXTRACT}/${FORMAL_TRAIN_DIR}/kwargs.yaml" "${CHECKPOINT_DIR}/kwargs.yaml"

  RUN_NAME="full_infrastructure_diagnostic_eval30_${SCALE}_${ALGORITHM}_seed${training_seed}_job${ARRAY_JOB_ID}_task${TASK_ID}"
  DIAGNOSTIC_COMMAND=(
    python evaluate_td3_gnn_infrastructure_diagnostics.py
    --algorithm "${ALGORITHM}"
    --scale "${SCALE}"
    --config "${CONFIG_DIR}/formal_config.yaml"
    --seed "${training_seed}"
    --eval_episodes 30
    --checkpoint "${CHECKPOINT_PREFIX}"
    --output_dir "${DIAGNOSTIC_DIR}"
    --run_name "${RUN_NAME}"
    --device cpu
    --max_episode_steps "${MAX_EPISODE_STEPS}"
    --deterministic true
    --eval_expl_noise 0.0
    --eval_seed_offset "${EVAL_SEED_OFFSET}"
    --matrix_job_id "${ARRAY_JOB_ID}"
  )

  printf "%q " "${DIAGNOSTIC_COMMAND[@]}" > "${METADATA_DIR}/evaluator_command.txt"
  printf "\n" >> "${METADATA_DIR}/evaluator_command.txt"
  EVAL_START="$(date +%s)"
  set +e
  /usr/bin/time -v -o "${METADATA_DIR}/evaluator_time_verbose.txt" \
    "${DIAGNOSTIC_COMMAND[@]}" \
    > "${LOG_DIR}/stdout.log" \
    2> "${LOG_DIR}/stderr.log"
  EVAL_STATUS="$?"
  set -e
  EVAL_END="$(date +%s)"
  if [[ "${EVAL_STATUS}" -ne 0 ]]; then
    echo "ERROR: evaluator failed for seed ${training_seed}: ${EVAL_STATUS}" >&2
    exit "${EVAL_STATUS}"
  fi

  python - \
    "${SEED_ROOT}" "${CANONICAL_DIR}/complete_eval30.csv" \
    "${TASK_ID}" "${training_seed}" "${ARRAY_JOB_ID}" \
    "${EXPECTED_CHARGERS}" "${EXPECTED_TRANSFORMERS}" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from scripts import validate_infrastructure_diagnostic_smoke as smoke
from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    SCHEMA_VERSION,
    episode_seed,
    stage_d_task,
    validate_seed_output_directory,
)

seed_root = Path(sys.argv[1])
canonical_csv = Path(sys.argv[2])
task_id = int(sys.argv[3])
training_seed = int(sys.argv[4])
matrix_job_id = sys.argv[5]
expected_chargers = int(sys.argv[6])
expected_transformers = int(sys.argv[7])
mapping = stage_d_task(task_id)
scale = str(mapping["scale"])
algorithm = str(mapping["algorithm"])
diagnostics = seed_root / "diagnostics"
validation = seed_root / "validation"

episode_rows, episode_fields = smoke.read_rows(diagnostics / "episode_diagnostics.csv")
seed_rows, seed_fields = smoke.read_rows(diagnostics / "seed_summary_diagnostics.csv")
charger_rows, charger_fields = smoke.read_rows(diagnostics / "charger_diagnostics.csv")
transformer_rows, transformer_fields = smoke.read_rows(diagnostics / "transformer_diagnostics.csv")
if len(episode_rows) != 30 or len(seed_rows) != 1:
    raise SystemExit("diagnostic episode or seed-summary row count mismatch")
smoke.require_columns(episode_fields, smoke.EPISODE_REQUIRED_COLUMNS, "episode diagnostics")
smoke.require_columns(seed_fields, smoke.SEED_SUMMARY_REQUIRED_COLUMNS, "seed summary diagnostics")
smoke.require_columns(charger_fields, smoke.CHARGER_REQUIRED_COLUMNS, "charger diagnostics")
smoke.require_columns(transformer_fields, smoke.TRANSFORMER_REQUIRED_COLUMNS, "transformer diagnostics")
all_rows = episode_rows + seed_rows + charger_rows + transformer_rows
smoke.validate_schema(episode_rows, "episode")
smoke.validate_schema(seed_rows, "seed summary")
smoke.validate_schema(charger_rows, "charger")
smoke.validate_schema(transformer_rows, "transformer")
smoke.validate_v2g_false(all_rows)
smoke.validate_signed_rows(episode_rows, "episode")
smoke.validate_signed_rows(charger_rows, "charger")
smoke.validate_signed_rows(transformer_rows, "transformer")
dynamic_mapping = {"algorithm": algorithm}
smoke.validate_no_hierarchical_negative(
    dynamic_mapping, episode_rows, seed_rows, charger_rows, transformer_rows
)

for row in all_rows:
    if row.get("scale") != scale or row.get("algorithm") != algorithm:
        raise SystemExit("diagnostic scale/algorithm identity mismatch")
    if smoke.integer_value(row.get("training_seed"), "training_seed") != training_seed:
        raise SystemExit("diagnostic training seed identity mismatch")
    if str(row.get("matrix_job_id")) != matrix_job_id:
        raise SystemExit("diagnostic matrix job identity mismatch")

episode_by_index = {}
for row in episode_rows:
    index = smoke.integer_value(row.get("episode_index"), "episode_index")
    if index in episode_by_index:
        raise SystemExit(f"duplicate diagnostic episode index: {index}")
    if smoke.integer_value(row.get("episode_seed"), "episode_seed") != episode_seed(
        scale, training_seed, index
    ):
        raise SystemExit(f"diagnostic episode seed mismatch: {index}")
    smoke.validate_episode_environment_bounds(dynamic_mapping, row)
    episode_by_index[index] = row
if set(episode_by_index) != set(range(30)):
    raise SystemExit("diagnostic episode inventory incomplete")


def group_by_episode(rows):
    grouped = defaultdict(list)
    for row in rows:
        index = smoke.integer_value(row.get("episode_index"), "episode_index")
        if smoke.integer_value(row.get("episode_seed"), "episode_seed") != episode_seed(
            scale, training_seed, index
        ):
            raise SystemExit(f"infrastructure episode seed mismatch: {index}")
        grouped[index].append(row)
    return grouped


charger_by_episode = group_by_episode(charger_rows)
transformer_by_episode = group_by_episode(transformer_rows)
canonical_rows = []
service_rows = []
for index in range(30):
    episode = episode_by_index[index]
    chargers = charger_by_episode[index]
    transformers = transformer_by_episode[index]
    if len(chargers) != expected_chargers or len(transformers) != expected_transformers:
        raise SystemExit(f"infrastructure row cardinality mismatch at episode {index}")
    smoke.validate_service_reconciliation(episode, chargers, transformers)
    canonical = smoke.canonical_episode_row(canonical_csv, index)
    for diagnostic_field, canonical_field, absolute_tolerance, relative_tolerance in (
        smoke.FLOAT_RECONCILIATION
    ):
        comparison = smoke.floating_reconciliation_row(
            diagnostic_field,
            canonical.get(canonical_field),
            episode.get(diagnostic_field),
            absolute_tolerance,
            relative_tolerance,
        )
        status = "pass" if comparison["pass"] == "True" else "fail"
        canonical_rows.append(
            {
                "episode_index": str(index),
                "field": diagnostic_field,
                "status": status,
                **comparison,
            }
        )
        if status != "pass":
            raise SystemExit(
                f"canonical floating reconciliation failed: episode={index}, field={diagnostic_field}"
            )
    for label, canonical_field, diagnostic_field, value_type in smoke.EXACT_RECONCILIATION:
        comparison = smoke.exact_reconciliation_row(
            label,
            canonical.get(canonical_field),
            episode.get(diagnostic_field),
            value_type,
        )
        status = "pass" if comparison["pass"] == "True" else "fail"
        canonical_rows.append(
            {
                "episode_index": str(index),
                "field": label,
                "status": status,
                **comparison,
            }
        )
        if status != "pass":
            raise SystemExit(
                f"canonical exact reconciliation failed: episode={index}, field={label}"
            )
    service_rows.append(
        {
            "episode_index": str(index),
            "served_count_reconciliation_status": "pass",
            "satisfaction_sum_reconciliation_status": "pass",
            "charged_energy_reconciliation_status": "pass",
            "discharged_energy_reconciliation_status": "pass",
        }
    )

validation.mkdir(parents=True, exist_ok=True)
with (validation / "canonical_reconciliation.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    fieldnames = list(canonical_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(canonical_rows)
with (validation / "service_reconciliation.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    fieldnames = list(service_rows[0])
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(service_rows)
(validation / "mapping_validation.json").write_text(
    json.dumps(
        {
            "status": "ok",
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": training_seed,
            "episodes": 30,
            "charger_rows_per_episode": expected_chargers,
            "transformer_rows_per_episode": expected_transformers,
            "diagnostic_schema_version": SCHEMA_VERSION,
        },
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
result = validate_seed_output_directory(seed_root, task_id, training_seed)
(validation / "seed_validation.json").write_text(
    json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
)
PY

  printf "task_id=%s\ntraining_seed=%s\nformal_task_id=%s\nstate=COMPLETED\nexit_code=0:0\nelapsed_seconds=%s\n" \
    "${TASK_ID}" "${training_seed}" "${FORMAL_TASK_ID}" "$((EVAL_END - EVAL_START))" \
    > "${METADATA_DIR}/seed_runtime.env"
  printf "%s,%s,%s,%s,%s,ok\n" \
    "${TASK_ID}" "${SCALE}" "${ALGORITHM}" "${training_seed}" "${FORMAL_TASK_ID}" \
    >> "${CHECKPOINT_INVENTORY}"
  printf "%s,COMPLETED,0:0,%s\n" \
    "${training_seed}" "$((EVAL_END - EVAL_START))" >> "${RUNTIME_SUMMARY}"
done

rm -rf "${PACKAGE_STAGING}"
mkdir -p \
  "${PACKAGE_STAGING}/task_metadata" "${PACKAGE_STAGING}/summaries" \
  "${PACKAGE_STAGING}/validation" "${PACKAGE_STAGING}/checksums" \
  "${PACKAGE_STAGING}/logs"
cp "${CHECKPOINT_INVENTORY}" \
  "${PACKAGE_STAGING}/task_metadata/formal_checkpoint_inventory.csv"
cp "${CHECKPOINT_INVENTORY}" \
  "${PACKAGE_STAGING}/summaries/checkpoint_inventory.csv"
cp "${RUNTIME_SUMMARY}" "${PACKAGE_STAGING}/summaries/runtime_summary.csv"
printf "task_id=%s\nscale=%s\nalgorithm=%s\narray_job_id=%s\n" \
  "${TASK_ID}" "${SCALE}" "${ALGORITHM}" "${ARRAY_JOB_ID}" \
  > "${PACKAGE_STAGING}/task_metadata/task.env"
printf "%s\n" "${SOURCE_COMMIT_SHA}" \
  > "${PACKAGE_STAGING}/task_metadata/source_commit_sha.txt"
printf "Stage D task %s completed five checkpoint groups.\n" "${TASK_ID}" \
  > "${PACKAGE_STAGING}/logs/stdout.log"
: > "${PACKAGE_STAGING}/logs/stderr.log"

for training_seed in 0 1 2 3 4; do
  mkdir -p "${PACKAGE_STAGING}/seed${training_seed}"
  cp -R "${TASK_DIR}/seed${training_seed}/diagnostics" \
    "${PACKAGE_STAGING}/seed${training_seed}/diagnostics"
  cp -R "${TASK_DIR}/seed${training_seed}/validation" \
    "${PACKAGE_STAGING}/seed${training_seed}/validation"
  cp -R "${TASK_DIR}/seed${training_seed}/runtime_metadata" \
    "${PACKAGE_STAGING}/seed${training_seed}/runtime_metadata"
  cp -R "${TASK_DIR}/seed${training_seed}/logs" \
    "${PACKAGE_STAGING}/seed${training_seed}/logs"
done

python - "${PACKAGE_STAGING}" "${TASK_ID}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
task_id = int(sys.argv[2])
episode_fields = [
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "diagnostic_schema_version",
]
episode_rows = []
for seed in range(5):
    path = root / f"seed{seed}/diagnostics/episode_diagnostics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            episode_rows.append({field: row[field] for field in episode_fields})
with (root / "summaries/episode_inventory.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=episode_fields)
    writer.writeheader()
    writer.writerows(episode_rows)
(root / "validation/task_validation.json").write_text(
    json.dumps(
        {
            "status": "ok",
            "task_id": task_id,
            "seed_groups": 5,
            "checkpoint_groups": 5,
            "episode_count": 150,
            "diagnostic_schema_version": "3",
        },
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

python - "${PACKAGE_STAGING}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "checksums/task_file_checksums.sha256"
files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
with manifest.open("w", encoding="utf-8") as handle:
    for path in files:
        handle.write(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
        )
PY

rm -f "${TEMP_PACKAGE}" "${TEMP_SIDECAR}"
tar -czf "${TEMP_PACKAGE}" -C "${PACKAGE_STAGING}" .
python - "${TEMP_PACKAGE}" "${TASK_ID}" <<'PY'
import json
import sys
from pathlib import Path
from scripts.full_infrastructure_diagnostic_eval30_packages import (
    validate_stage_d_task_package,
)
result = validate_stage_d_task_package(Path(sys.argv[1]), int(sys.argv[2]))
print(json.dumps(result, sort_keys=True))
PY
sha256sum "${TEMP_PACKAGE}" | sed "s#${TEMP_PACKAGE}#${PACKAGE_BASENAME}#" \
  > "${TEMP_SIDECAR}"
mv "${TEMP_PACKAGE}" "${PACKAGE_PATH}"
mv "${TEMP_SIDECAR}" "${PACKAGE_SIDECAR}"

echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_TASK_OK"
echo "task_id=${TASK_ID}"
echo "checkpoint_groups=5"
echo "episodes=150"
echo "package_path=${PACKAGE_PATH}"
echo "package_sha256=${PACKAGE_SIDECAR}"
