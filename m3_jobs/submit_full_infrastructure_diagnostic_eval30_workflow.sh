#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CURRENT_DIR="$(pwd -P)"
SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
EXPECTED_SOURCE_COMMIT="${EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT:-}"
SOURCE_ARCHIVE="${EV_GNN_FULL_DIAGNOSTIC_SOURCE_ARCHIVE:-}"
SOURCE_ARCHIVE_SHA256="${EV_GNN_FULL_DIAGNOSTIC_SOURCE_ARCHIVE_SHA256:-${SOURCE_ARCHIVE}.sha256}"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
RUN_ROOT="${EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/full_infrastructure_diagnostic_eval30}"
FORMAL_PACKAGE_ROOT="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT:-${OUTPUT_ROOT}}"
FORMAL_COMPLETE_BUNDLE="${EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE:-${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz}"
ARRAY_SCRIPT="${REPO_ROOT}/m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm"
REDUCER_SCRIPT="${REPO_ROOT}/m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm"
VALIDATOR="${REPO_ROOT}/scripts/validate_full_infrastructure_diagnostic_eval30.py"
FORMAL_JOB_ID="58513929"
SACCT_FIELDS="JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU"
EXECUTE=0
PREFLIGHT_DIR=""

SOURCE_BUNDLE_REQUIRED_MEMBERS=(
  evaluate_td3_gnn.py
  evaluate_td3_gnn_infrastructure_diagnostics.py
  TD3/TD3_ActionGNN_Controlled.py
  TD3/TD3_HierarchicalActionGNN.py
  config_files/PublicPST_25cp.yaml
  config_files/PublicPST_100.yaml
  config_files/PublicPST_500.yaml
  config_files/PublicPST_1000.yaml
  m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm
  m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm
  m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh
  m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh
  scripts/validate_full_infrastructure_diagnostic_eval30.py
  utils/ev2gym_training_utils.py
  utils/infrastructure_diagnostics.py
  utils/state_public_pst_gnn.py
  SOURCE_COMMIT_SHA.txt
  runtime_metadata/source_file_checksums.sha256
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1 ;;
    --dry-run) EXECUTE=0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

array_export_vars() {
  printf "%s" "ALL"
  printf ",EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT=%s" "${REPO_ROOT}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT=%s" "${EXPECTED_SOURCE_COMMIT}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID=%s" "${FORMAL_JOB_ID}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT=%s" "${FORMAL_PACKAGE_ROOT}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE=%s" "${FORMAL_COMPLETE_BUNDLE}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT=%s" "${OUTPUT_ROOT}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT=%s" "${RUN_ROOT}"
}

reducer_export_vars() {
  local array_job_id="$1"
  printf "%s" "$(array_export_vars)"
  printf ",EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID=%s" "${array_job_id}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_TASK_PACKAGE_ROOT=%s" "${OUTPUT_ROOT}"
  printf ",EV_GNN_FULL_DIAGNOSTIC_SLURM_LOG_ROOT=%s" "${OUTPUT_ROOT}"
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

validate_source_sidecar() {
  [[ -s "${SOURCE_ARCHIVE}" ]] || die "source bundle is missing: ${SOURCE_ARCHIVE}"
  [[ -s "${SOURCE_ARCHIVE_SHA256}" ]] || die "source bundle sidecar is missing: ${SOURCE_ARCHIVE_SHA256}"
  python - "${SOURCE_ARCHIVE}" "${SOURCE_ARCHIVE_SHA256}" "${EXPECTED_SOURCE_COMMIT}" "${SOURCE_BUNDLE_REQUIRED_MEMBERS[@]}" <<'PY'
import hashlib
import tarfile
import tempfile
import sys
from pathlib import Path
from pathlib import PurePosixPath


def safe_member_name(member):
    raw_name = member.name
    if not raw_name or "\\" in raw_name:
        raise SystemExit(f"unsafe source bundle path: {raw_name!r}")
    path = PurePosixPath(raw_name)
    parts = [part for part in path.parts if part not in {"", "."}]
    if path.is_absolute() or ".." in path.parts or not parts:
        raise SystemExit(f"unsafe source bundle path: {raw_name!r}")
    if member.issym() or member.islnk() or member.isdev():
        raise SystemExit(f"unsafe source bundle member type: {raw_name!r}")
    if not member.isfile() and not member.isdir():
        raise SystemExit(f"unsupported source bundle member type: {raw_name!r}")
    return PurePosixPath(*parts).as_posix()

archive_path = Path(sys.argv[1])
sidecar_path = Path(sys.argv[2])
expected_commit = sys.argv[3]
required_members = sorted(sys.argv[4:])
line = sidecar_path.read_text(encoding="utf-8").strip()
digest, basename = line.split("  ", 1)
if basename != archive_path.name:
    raise SystemExit(f"source sidecar basename mismatch: {basename} != {archive_path.name}")
observed = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if observed != digest:
    raise SystemExit("source sidecar checksum mismatch")
with tempfile.TemporaryDirectory() as temporary:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = []
        seen = set()
        for member in members:
            name = safe_member_name(member)
            if name in seen:
                raise SystemExit(f"duplicate source bundle member: {name}")
            seen.add(name)
            names.append((member, name))
        roots = {name.split("/", 1)[0] for _member, name in names}
        if len(roots) != 1:
            raise SystemExit("source bundle must contain exactly one top-level root")
        top_root = next(iter(roots))
        relative_files = sorted(
            name.split("/", 1)[1]
            for member, name in names
            if member.isfile() and name.startswith(top_root + "/")
        )
        if relative_files != required_members:
            missing = sorted(set(required_members) - set(relative_files))
            unexpected = sorted(set(relative_files) - set(required_members))
            raise SystemExit(
                "source bundle file set mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        prohibited_parts = {
            ".git",
            "saved_models",
            "checkpoints",
            "EVGNN_Formal_Evidence",
            "task_packages",
            "__pycache__",
            ".pytest_cache",
            "__MACOSX",
        }
        prohibited_suffixes = (
            ".tar",
            ".tar.gz",
            ".gz",
            ".pyc",
            ".pt",
            ".pth",
            ".ckpt",
            ".csv",
            ".zip",
            ".tgz",
        )
        for relative_name in relative_files:
            parts = PurePosixPath(relative_name).parts
            if any(part in prohibited_parts for part in parts):
                raise SystemExit(f"source bundle contains prohibited path: {relative_name}")
            if PurePosixPath(relative_name).name in {".DS_Store"}:
                raise SystemExit(f"source bundle contains prohibited metadata: {relative_name}")
            if any(part.startswith("._") for part in parts):
                raise SystemExit(f"source bundle contains AppleDouble metadata: {relative_name}")
            if relative_name.endswith(prohibited_suffixes):
                raise SystemExit(f"source bundle contains prohibited suffix: {relative_name}")
            if "model.best" in relative_name or "model.last" in relative_name:
                raise SystemExit(f"source bundle contains checkpoint path: {relative_name}")
        for member, name in names:
            output = Path(temporary) / name
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"cannot read source bundle member: {name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(source.read())
    root = Path(temporary) / next(iter(roots))
    source_commit = (root / "SOURCE_COMMIT_SHA.txt").read_text(encoding="utf-8").strip()
    if source_commit != expected_commit:
        raise SystemExit(f"source bundle commit {source_commit} != {expected_commit}")
    manifest = root / "runtime_metadata/source_file_checksums.sha256"
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        item_digest, name = line.split("  ", 1)
        entries[name] = item_digest
    expected = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    )
    if sorted(entries) != expected:
        raise SystemExit("source bundle manifest coverage mismatch")
    for name, item_digest in entries.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != item_digest:
            raise SystemExit(f"source bundle manifest checksum mismatch: {name}")
PY
}

validate_git_or_source_commit() {
  [[ -n "${EXPECTED_SOURCE_COMMIT}" ]] || die "EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT is required"
  [[ -s "${SOURCE_COMMIT_FILE}" ]] || die "SOURCE_COMMIT_SHA.txt is required at ${SOURCE_COMMIT_FILE}"
  local source_commit
  source_commit="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
  [[ "${source_commit}" == "${EXPECTED_SOURCE_COMMIT}" ]] || die "source commit ${source_commit} does not match ${EXPECTED_SOURCE_COMMIT}"
  if [[ -d "${REPO_ROOT}/.git" ]]; then
    local head
    head="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    [[ "${head}" == "${EXPECTED_SOURCE_COMMIT}" ]] || die "git HEAD ${head} does not match ${EXPECTED_SOURCE_COMMIT}"
    [[ -z "$(git -C "${REPO_ROOT}" status --porcelain=v1 --untracked-files=all)" ]] || die "git worktree is not clean"
  fi
}

print_task_mapping() {
  local task_id
  for task_id in 0 1 2 3 4 5 6 7; do
    python "${VALIDATOR}" task-mapping --task-id "${task_id}" | tr '\n' ' '
    printf "\n"
  done
}

run_preflight() {
  [[ "${CURRENT_DIR}" == "${REPO_ROOT}" ]] || die "run from physical source root: ${REPO_ROOT}"
  [[ -s "${ARRAY_SCRIPT}" ]] || die "missing array script: ${ARRAY_SCRIPT}"
  [[ -s "${REDUCER_SCRIPT}" ]] || die "missing reducer script: ${REDUCER_SCRIPT}"
  [[ -s "${VALIDATOR}" ]] || die "missing validator: ${VALIDATOR}"
  [[ "${FORMAL_JOB_ID}" == "58513929" ]] || die "formal source job must be 58513929"
  validate_git_or_source_commit
  validate_source_sidecar
  bash -n "${ARRAY_SCRIPT}"
  bash -n "${REDUCER_SCRIPT}"
  local task_id seed resolution package_path
  PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evgnn_full_submit_preflight.XXXXXX")"
  trap 'rm -rf "${PREFLIGHT_DIR}"' EXIT
  for task_id in 0 1 2 3 4 5 6 7; do
    for seed in 0 1 2 3 4; do
      python "${VALIDATOR}" \
        resolve-formal-package \
        --task-id "${task_id}" \
        --training-seed "${seed}" \
        --individual-package-root "${FORMAL_PACKAGE_ROOT}" \
        --complete-bundle "${FORMAL_COMPLETE_BUNDLE}" \
        --staging-dir "${PREFLIGHT_DIR}/resolved" \
        > "${PREFLIGHT_DIR}/task${task_id}_seed${seed}_resolution.json"
      resolution="${PREFLIGHT_DIR}/task${task_id}_seed${seed}_resolution.json"
      package_path="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["package_path"])' "${resolution}")"
      python "${VALIDATOR}" \
        validate-formal-package \
        --task-id "${task_id}" \
        --training-seed "${seed}" \
        --package "${package_path}" \
        --extract-dir "${PREFLIGHT_DIR}/extract_task${task_id}_seed${seed}" \
        > "${PREFLIGHT_DIR}/task${task_id}_seed${seed}_formal_validation.json"
    done
    EV_GNN_FULL_DIAGNOSTIC_DRY_RUN=1 \
      SLURM_ARRAY_TASK_ID="${task_id}" \
      SLURM_ARRAY_JOB_ID=123456 \
      SLURM_JOB_ID="123456_${task_id}" \
      bash "${ARRAY_SCRIPT}" > "${PREFLIGHT_DIR}/runner_task${task_id}_dry_run.out"
    grep -q "expected_episode_count=150" "${PREFLIGHT_DIR}/runner_task${task_id}_dry_run.out"
    [[ "$(grep -c '^EVALUATOR_COMMAND_SEED_' "${PREFLIGHT_DIR}/runner_task${task_id}_dry_run.out")" == "5" ]] || die "runner dry-run task ${task_id} did not print five evaluator commands"
  done
  EV_GNN_FULL_DIAGNOSTIC_REDUCER_DRY_RUN=1 \
    EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID=123456 \
    SLURM_JOB_ID=789012 \
    bash "${REDUCER_SCRIPT}" > "${PREFLIGHT_DIR}/reducer_dry_run.out"
  grep -q "episode_count=1200" "${PREFLIGHT_DIR}/reducer_dry_run.out"
}

run_preflight

if [[ "${EXECUTE}" -eq 0 ]]; then
  echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_SUBMIT_DRY_RUN"
  echo "dry_run=1"
  echo "repo_root=${REPO_ROOT}"
  echo "expected_source_commit=${EXPECTED_SOURCE_COMMIT}"
  echo "source_archive=${SOURCE_ARCHIVE}"
  echo "source_archive_sha256=${SOURCE_ARCHIVE_SHA256}"
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
  echo "M3_TAR_VALIDATION_COMMAND=python ${VALIDATOR} validate-complete-bundle --bundle ${OUTPUT_ROOT}/full_infrastructure_diagnostic_eval30_complete_evidence_job<array_job_id>.tar.gz"
  echo "M3_CHECKSUM_VALIDATION_COMMAND=cd ${OUTPUT_ROOT} && sha256sum -c full_infrastructure_diagnostic_eval30_complete_evidence_job<array_job_id>.tar.gz.sha256"
  echo "LOCAL_SCP_COMMAND=scp cche0357@m3.massive.org.au:${OUTPUT_ROOT}/full_infrastructure_diagnostic_eval30_complete_evidence_job<array_job_id>.tar.gz cche0357@m3.massive.org.au:${OUTPUT_ROOT}/full_infrastructure_diagnostic_eval30_complete_evidence_job<array_job_id>.tar.gz.sha256 ."
  echo "DRY_RUN_NO_SBATCH_CALLED"
  exit 0
fi

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
FINAL_BUNDLE="${OUTPUT_ROOT}/full_infrastructure_diagnostic_eval30_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"
FINAL_BUNDLE_SHA256="${FINAL_BUNDLE}.sha256"

echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_WORKFLOW_SUBMITTED"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "reducer_job_id=${REDUCER_JOB_ID}"
echo "expected_final_bundle=${FINAL_BUNDLE}"
echo "expected_final_bundle_sha256=${FINAL_BUNDLE_SHA256}"
echo "SQUEUE_COMMAND=squeue -j ${ARRAY_JOB_ID},${REDUCER_JOB_ID}"
echo "SACCT_COMMAND=sacct -j ${ARRAY_JOB_ID} --parsable2 --noheader --format=${SACCT_FIELDS}"
echo "M3_TAR_VALIDATION_COMMAND=python ${VALIDATOR} validate-complete-bundle --bundle ${FINAL_BUNDLE}"
echo "M3_CHECKSUM_VALIDATION_COMMAND=cd ${OUTPUT_ROOT} && sha256sum -c $(basename "${FINAL_BUNDLE_SHA256}")"
echo "LOCAL_SCP_COMMAND=scp cche0357@m3.massive.org.au:${FINAL_BUNDLE} cche0357@m3.massive.org.au:${FINAL_BUNDLE_SHA256} ."
