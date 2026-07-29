#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT_INPUT="${EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
if [[ ! -d "${REPO_ROOT_INPUT}" ]]; then
  echo "ERROR: repository root does not exist: ${REPO_ROOT_INPUT}" >&2
  exit 2
fi
REPO_ROOT="$(cd "${REPO_ROOT_INPUT}" && pwd -P)"

EXPECTED_SOURCE_COMMIT="${EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT:-}"
FORMAL_JOB_ID="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID:-58513929}"
FORMAL_PACKAGE_ROOT="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
FORMAL_COMPLETE_BUNDLE="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE:-${FORMAL_PACKAGE_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz}"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
RUN_ROOT="${EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/full_infrastructure_diagnostics_eval30}"

SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
VALIDATOR="${REPO_ROOT}/scripts/validate_full_infrastructure_diagnostic_eval30.py"
PACKAGE_VALIDATOR="${REPO_ROOT}/scripts/full_infrastructure_diagnostic_eval30_packages.py"
SMOKE_VALIDATOR="${REPO_ROOT}/scripts/validate_infrastructure_diagnostic_smoke.py"
ARRAY_SCRIPT="${REPO_ROOT}/m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm"
REDUCER_SCRIPT="${REPO_ROOT}/m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm"
SACCT_FIELDS="JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU"

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
if [[ "${FORMAL_JOB_ID}" != "58513929" ]]; then
  echo "ERROR: formal job ID must equal 58513929" >&2
  exit 2
fi

require_regular_file "${SOURCE_COMMIT_FILE}"
require_regular_file "${VALIDATOR}"
require_regular_file "${PACKAGE_VALIDATOR}"
require_regular_file "${SMOKE_VALIDATOR}"
require_regular_file "${ARRAY_SCRIPT}"
require_regular_file "${REDUCER_SCRIPT}"
require_regular_file "${FORMAL_COMPLETE_BUNDLE}"

SOURCE_COMMIT_SHA="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
if [[ "${SOURCE_COMMIT_SHA}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: source commit mismatch: ${SOURCE_COMMIT_SHA} != ${EXPECTED_SOURCE_COMMIT}" >&2
  exit 3
fi

cd "${REPO_ROOT}"

echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_PREFLIGHT"
echo "repo_root=${REPO_ROOT}"
echo "expected_source_commit=${EXPECTED_SOURCE_COMMIT}"
echo "formal_job_id=${FORMAL_JOB_ID}"
echo "formal_package_root=${FORMAL_PACKAGE_ROOT}"
echo "formal_complete_bundle=${FORMAL_COMPLETE_BUNDLE}"
echo "output_root=${OUTPUT_ROOT}"
echo "run_root=${RUN_ROOT}"

for task_id in 0 1 2 3 4 5 6 7; do
  python "${VALIDATOR}" task-mapping --task-id "${task_id}"
done

python - \
  "${FORMAL_PACKAGE_ROOT}" "${FORMAL_COMPLETE_BUNDLE}" \
  "${FORMAL_JOB_ID}" <<'PY'
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from scripts import validate_infrastructure_diagnostic_smoke as smoke
from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    EVAL_EPISODES,
    SCALE_SEED_OFFSETS,
    TRAINING_SEEDS,
    formal_task_id,
    stage_d_task,
)

individual_root = Path(sys.argv[1])
complete_bundle = Path(sys.argv[2])
formal_job_id = sys.argv[3]
if formal_job_id != "58513929":
    raise SystemExit("formal job identity mismatch")


def validate_outer_members(tar):
    seen = set()
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe outer formal bundle path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise SystemExit(f"unsupported outer formal bundle member: {member.name}")
        normalised = path.as_posix()
        if normalised in seen:
            raise SystemExit(f"duplicate outer formal bundle member: {normalised}")
        seen.add(normalised)


with tarfile.open(complete_bundle, "r:gz") as outer:
    validate_outer_members(outer)
    nested_members = {}
    for member in outer.getmembers():
        if not member.isfile():
            continue
        parts = PurePosixPath(member.name).parts
        if "task_packages" in parts[:-1]:
            nested_members.setdefault(PurePosixPath(member.name).name, []).append(member)

    validated = 0
    seen_formal_ids = set()
    for task_id in range(8):
        mapping = stage_d_task(task_id)
        scale = str(mapping["scale"])
        algorithm = str(mapping["algorithm"])
        expected_chargers, expected_transformers = {
            "25cp": (25, 3),
            "100cp": (100, 7),
            "500cp": (500, 35),
            "1000cp": (1000, 70),
        }[scale]
        for training_seed in TRAINING_SEEDS:
            formal_id = formal_task_id(task_id, training_seed)
            if formal_id in seen_formal_ids:
                raise SystemExit(f"duplicate formal task ID: {formal_id}")
            seen_formal_ids.add(formal_id)
            expected_name = (
                f"m3_controlled_multiscale_formal_{scale}_{algorithm}_"
                f"seed{training_seed}_job{formal_job_id}_task{formal_id}.tar.gz"
            )
            individual = individual_root / expected_name
            if individual.is_symlink():
                raise SystemExit(f"formal package must not be a symbolic link: {individual}")
            with tempfile.TemporaryDirectory(
                prefix=f"stage_d_formal_{formal_id}_"
            ) as temp_dir:
                temp_root = Path(temp_dir)
                if individual.is_file():
                    package = individual
                else:
                    matches = nested_members.get(expected_name, [])
                    if len(matches) != 1:
                        raise SystemExit(
                            f"expected exactly one nested formal package {expected_name}, "
                            f"found {len(matches)}"
                        )
                    payload = outer.extractfile(matches[0])
                    if payload is None:
                        raise SystemExit(f"unable to read nested formal package: {expected_name}")
                    package = temp_root / expected_name
                    package.write_bytes(payload.read())

                extract_root = smoke.safe_extract_tar(package, temp_root / "extract")
                run_name = (
                    f"controlled_multiscale_formal_{scale}_{algorithm}_seed{training_seed}"
                )
                train_dir = f"train/{run_name}"
                config_member = (
                    f"config/{scale}_{algorithm}_seed{training_seed}_config.yaml"
                )
                canonical_member = (
                    f"eval/{scale}_{algorithm}_seed{training_seed}_eval30.csv"
                )
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
                smoke.require_files(extract_root, required)
                smoke.verify_extracted_manifest(
                    extract_root,
                    "runtime_metadata/package_file_checksums.sha256",
                    required_coverage=[
                        item
                        for item in required
                        if item != "runtime_metadata/package_file_checksums.sha256"
                    ],
                )
                smoke.validate_formal_config(
                    extract_root / config_member,
                    {
                        "expected_charger_rows": expected_chargers,
                        "expected_transformer_rows": expected_transformers,
                    },
                )
                rows, _ = smoke.read_rows(extract_root / canonical_member)
                episodes = [row for row in rows if row.get("row_type") == "episode"]
                if len(episodes) != EVAL_EPISODES:
                    raise SystemExit(
                        f"formal task {formal_id} must contain 30 canonical episodes"
                    )
                seen_episodes = set()
                for row in episodes:
                    episode_index = smoke.integer_value(
                        row.get("episode_index"), "episode_index"
                    )
                    if episode_index in seen_episodes:
                        raise SystemExit(
                            f"duplicate canonical episode for formal task {formal_id}"
                        )
                    seen_episodes.add(episode_index)
                    if str(row.get("algorithm")) != algorithm:
                        raise SystemExit(
                            f"canonical algorithm mismatch for formal task {formal_id}"
                        )
                    if smoke.integer_value(row.get("seed"), "seed") != training_seed:
                        raise SystemExit(
                            f"canonical seed mismatch for formal task {formal_id}"
                        )
                    expected_episode_seed = (
                        SCALE_SEED_OFFSETS[scale] + training_seed + episode_index
                    )
                    if (
                        smoke.integer_value(row.get("episode_seed"), "episode_seed")
                        != expected_episode_seed
                    ):
                        raise SystemExit(
                            f"canonical episode seed mismatch for formal task {formal_id}"
                        )
                if seen_episodes != set(range(EVAL_EPISODES)):
                    raise SystemExit(
                        f"canonical episode inventory incomplete for formal task {formal_id}"
                    )
                validated += 1

if seen_formal_ids != set(range(40)):
    raise SystemExit(
        f"formal task IDs must equal 0..39, got {sorted(seen_formal_ids)}"
    )
if validated != 40:
    raise SystemExit(f"expected 40 validated formal checkpoints, got {validated}")
print("FORMAL_PREFLIGHT_OK")
print("validated_formal_checkpoints=40")
PY

ARRAY_EXPORTS="ALL,EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT=${REPO_ROOT},EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT},EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID=${FORMAL_JOB_ID},EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT=${FORMAL_PACKAGE_ROOT},EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE=${FORMAL_COMPLETE_BUNDLE},EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT=${OUTPUT_ROOT},EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT=${RUN_ROOT}"

if [[ "${EV_GNN_FULL_DIAGNOSTIC_SUBMIT_DRY_RUN:-0}" == "1" ]]; then
  echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_SUBMIT_DRY_RUN"
  echo "SBATCH_ARRAY_COMMAND=sbatch --parsable --export=${ARRAY_EXPORTS} ${ARRAY_SCRIPT}"
  echo "SBATCH_REDUCER_COMMAND=sbatch --parsable --dependency=afterok:<array_job_id> --export=ALL,EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT=${REPO_ROOT},EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT},EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID=<array_job_id>,EV_GNN_FULL_DIAGNOSTIC_TASK_PACKAGE_ROOT=${OUTPUT_ROOT},EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT=${OUTPUT_ROOT} ${REDUCER_SCRIPT}"
  echo "SQUEUE_COMMAND=squeue -j <array_job_id>,<reducer_job_id>"
  echo "SACCT_COMMAND=sacct -j <array_job_id> --parsable2 --noheader --format=${SACCT_FIELDS}"
  echo "FINAL_BUNDLE=${OUTPUT_ROOT}/full_infrastructure_diagnostics_complete_evidence_job<array_job_id>.tar.gz"
  echo "LOCAL_SCP_COMMAND=scp cche0357@m3.massive.org.au:${OUTPUT_ROOT}/full_infrastructure_diagnostics_complete_evidence_job<array_job_id>.tar.gz cche0357@m3.massive.org.au:${OUTPUT_ROOT}/full_infrastructure_diagnostics_complete_evidence_job<array_job_id>.tar.gz.sha256 ."
  echo "DRY_RUN_NO_SBATCH_CALLED"
  exit 0
fi

normalise_job_id() {
  local raw="$1"
  local normalised="${raw%%;*}"
  printf "%s" "${normalised}"
}

ARRAY_RESULT="$(sbatch --parsable --export="${ARRAY_EXPORTS}" "${ARRAY_SCRIPT}")"
ARRAY_JOB_ID="$(normalise_job_id "${ARRAY_RESULT}")"
if ! [[ "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: sbatch did not return a numeric array job ID: ${ARRAY_RESULT}" >&2
  exit 5
fi
echo "array_job_id=${ARRAY_JOB_ID}"

REDUCER_EXPORTS="ALL,EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT=${REPO_ROOT},EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT=${EXPECTED_SOURCE_COMMIT},EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID=${ARRAY_JOB_ID},EV_GNN_FULL_DIAGNOSTIC_TASK_PACKAGE_ROOT=${OUTPUT_ROOT},EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT=${OUTPUT_ROOT}"
REDUCER_RESULT="$(
  sbatch \
    --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    --export="${REDUCER_EXPORTS}" \
    "${REDUCER_SCRIPT}"
)"
REDUCER_JOB_ID="$(normalise_job_id "${REDUCER_RESULT}")"
if ! [[ "${REDUCER_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: sbatch did not return a numeric reducer job ID: ${REDUCER_RESULT}" >&2
  echo "PRESERVE_ARRAY_JOB_ID=${ARRAY_JOB_ID}" >&2
  exit 6
fi

FINAL_BUNDLE="${OUTPUT_ROOT}/full_infrastructure_diagnostics_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"
FINAL_SIDECAR="${FINAL_BUNDLE}.sha256"

echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_WORKFLOW_SUBMITTED"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "reducer_job_id=${REDUCER_JOB_ID}"
echo "expected_final_bundle=${FINAL_BUNDLE}"
echo "expected_final_bundle_sha256=${FINAL_SIDECAR}"
echo "SQUEUE_COMMAND=squeue -j ${ARRAY_JOB_ID},${REDUCER_JOB_ID}"
echo "SACCT_COMMAND=sacct -j ${ARRAY_JOB_ID} --parsable2 --noheader --format=${SACCT_FIELDS}"
echo "LOCAL_SCP_COMMAND=scp cche0357@m3.massive.org.au:${FINAL_BUNDLE} cche0357@m3.massive.org.au:${FINAL_SIDECAR} ."
