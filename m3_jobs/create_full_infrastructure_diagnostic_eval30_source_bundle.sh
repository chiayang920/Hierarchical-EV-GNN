#!/bin/bash
set -euo pipefail

SOURCE_EXPECTED_HEAD_SHA="${EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXPECTED_HEAD_SHA:-}"
BRANCH_REQUIRED="main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CURRENT_DIR="$(pwd -P)"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_SOURCE_OUTPUT_ROOT:-${HOME}/Downloads/EVGNN_Formal_Evidence}"
RECORDED_HEAD_SHA=""
TOP_LEVEL=""
ARCHIVE_PATH=""
SHA_PATH=""
export COPYFILE_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1

ALLOWLIST=(
  evaluate_td3_gnn.py
  evaluate_td3_gnn_infrastructure_diagnostics.py
  TD3/TD3_ActionGNN_Controlled.py
  TD3/TD3_HierarchicalActionGNN.py
  config_files/PublicPST_25cp.yaml
  config_files/PublicPST_100.yaml
  config_files/PublicPST_500.yaml
  config_files/PublicPST_1000.yaml
  m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm
  m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm
  m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh
  scripts/validate_infrastructure_diagnostic_smoke.py
  scripts/validate_full_infrastructure_diagnostic_eval30.py
  scripts/full_infrastructure_diagnostic_eval30_packages.py
  scripts/full_infrastructure_diagnostic_eval30_accounting.py
  utils/ev2gym_training_utils.py
  utils/infrastructure_diagnostics.py
  utils/state_public_pst_gnn.py
)


die() {
  echo "ERROR: $*" >&2
  exit 1
}


set_archive_paths() {
  TOP_LEVEL="EV-GNN-full-infrastructure-diagnostics-eval30-${RECORDED_HEAD_SHA}"
  ARCHIVE_PATH="${OUTPUT_ROOT}/${TOP_LEVEL}.tar.gz"
  SHA_PATH="${ARCHIVE_PATH}.sha256"
}


write_sha256_file() {
  local path="$1"
  local output_path="$2"
  local digest
  if command -v sha256sum >/dev/null 2>&1; then
    digest="$(sha256sum "${path}" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    digest="$(shasum -a 256 "${path}" | awk '{print $1}')"
  else
    die "sha256sum or shasum is required"
  fi
  printf "%s  %s\n" "${digest}" "$(basename "${path}")" > "${output_path}"
}


is_prohibited_path() {
  local path="$1"
  case "${path}" in
    .git|.git/*|*/.git|*/.git/*) return 0 ;;
    saved_models|saved_models/*|*/saved_models|*/saved_models/*) return 0 ;;
    task_packages|task_packages/*|*/task_packages|*/task_packages/*) return 0 ;;
    __pycache__|__pycache__/*|*/__pycache__|*/__pycache__/*) return 0 ;;
    .pytest_cache|.pytest_cache/*|*/.pytest_cache|*/.pytest_cache/*) return 0 ;;
    __MACOSX|__MACOSX/*|*/__MACOSX|*/__MACOSX/*) return 0 ;;
    ._*|*/._*) return 0 ;;
    .DS_Store|*/.DS_Store) return 0 ;;
    *.tar|*.tar.gz|*.tgz|*.zip|*.gz|*.pt|*.pth) return 0 ;;
    *model.best*|*model.last*) return 0 ;;
  esac
  return 1
}


validate_allowlist() {
  local path
  local absolute
  for path in "${ALLOWLIST[@]}"; do
    if is_prohibited_path "${path}"; then
      die "prohibited source-bundle path in allowlist: ${path}"
    fi
    absolute="${REPO_ROOT}/${path}"
    if [[ ! -e "${absolute}" ]]; then
      die "allowlisted source path is missing: ${path}"
    fi
    if [[ -L "${absolute}" ]]; then
      die "allowlisted source path must not be a symbolic link: ${path}"
    fi
    if [[ ! -f "${absolute}" ]]; then
      die "allowlisted source path must be a regular file: ${path}"
    fi
  done
}


validate_archive() {
  local archive="$1"
  local expected_root="$2"
  python - "${archive}" "${expected_root}" "${RECORDED_HEAD_SHA}" "${ALLOWLIST[@]}" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

archive = sys.argv[1]
expected_root = sys.argv[2]
expected_sha = sys.argv[3]
allowlist = set(sys.argv[4:])
expected_files = {f"{expected_root}/{path}" for path in allowlist}
expected_files.add(f"{expected_root}/SOURCE_COMMIT_SHA.txt")

with tarfile.open(archive, "r:gz") as tar:
    members = tar.getmembers()
    seen = set()
    files = set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SystemExit(f"unsafe archive path: {member.name}")
        if member.name in seen:
            raise SystemExit(f"duplicate archive member: {member.name}")
        seen.add(member.name)
        if member.issym() or member.islnk():
            raise SystemExit(f"archive link is prohibited: {member.name}")
        if member.isdev():
            raise SystemExit(f"archive device member is prohibited: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported archive member: {member.name}")
        if member.isfile():
            files.add(member.name)

    if files != expected_files:
        missing = sorted(expected_files - files)
        unexpected = sorted(files - expected_files)
        raise SystemExit(
            f"archive file-set mismatch: missing={missing}, unexpected={unexpected}"
        )

    source_member = tar.extractfile(f"{expected_root}/SOURCE_COMMIT_SHA.txt")
    if source_member is None:
        raise SystemExit("SOURCE_COMMIT_SHA.txt is unreadable")
    recorded = source_member.read().decode("utf-8").strip()
    if recorded != expected_sha:
        raise SystemExit(
            f"source commit member mismatch: {recorded} != {expected_sha}"
        )
PY
}


print_transfer_commands() {
  local archive_name
  archive_name="$(basename "${ARCHIVE_PATH}")"
  echo "SOURCE_ARCHIVE=${ARCHIVE_PATH}"
  echo "SOURCE_ARCHIVE_SHA256=${SHA_PATH}"
  echo "SCP_COMMAND=scp ${ARCHIVE_PATH} ${SHA_PATH} cche0357@m3.massive.org.au:/projects/fr57/cche0357/EV-GNN_sources/"
  echo "M3_CHECKSUM_COMMAND=cd /projects/fr57/cche0357/EV-GNN_sources && sha256sum -c ${archive_name}.sha256"
  echo "M3_EXTRACT_COMMAND=test ! -e /scratch2/fr57/cche0357/EV-GNN_sources/${TOP_LEVEL} && mkdir -p /scratch2/fr57/cche0357/EV-GNN_sources && tar -xzf /projects/fr57/cche0357/EV-GNN_sources/${archive_name} -C /scratch2/fr57/cche0357/EV-GNN_sources"
  echo "M3_REPO_ROOT=/scratch2/fr57/cche0357/EV-GNN_sources/${TOP_LEVEL}"
  echo "M3_SOURCE_MANIFEST=/projects/fr57/cche0357/EV-GNN_sources/${archive_name}.sha256"
  echo "M3_EXPECTED_SOURCE_COMMIT_ENV=EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT=${RECORDED_HEAD_SHA}"
}


validate_allowlist

if [[ "${EV_GNN_FULL_DIAGNOSTIC_SOURCE_DRY_RUN:-0}" == "1" ]]; then
  RECORDED_HEAD_SHA="${SOURCE_EXPECTED_HEAD_SHA:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf "DRY_RUN_HEAD_UNKNOWN")}"
  set_archive_paths
  echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_SOURCE_DRY_RUN"
  echo "repo_root=${REPO_ROOT}"
  echo "current_dir=${CURRENT_DIR}"
  echo "recorded_head_sha=${RECORDED_HEAD_SHA}"
  echo "top_level=${TOP_LEVEL}"
  echo "PROHIBITED_PATH_GUARD_OK"
  echo "ALLOWLIST"
  printf "%s\n" "${ALLOWLIST[@]}"
  print_transfer_commands
  echo "DRY_RUN_NO_ARCHIVE_CREATED"
  exit 0
fi

[[ "${CURRENT_DIR}" == "${REPO_ROOT}" ]] || die "run from physical repository root: ${REPO_ROOT}"
cd "${REPO_ROOT}"

CURRENT_BRANCH="$(git branch --show-current)"
[[ "${CURRENT_BRANCH}" == "${BRANCH_REQUIRED}" ]] || \
  die "source bundle must be created on branch ${BRANCH_REQUIRED}; got ${CURRENT_BRANCH}"

HEAD_SHA="$(git rev-parse HEAD)"
[[ "${HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]] || die "HEAD is not a 40-character commit SHA"
if [[ -n "${SOURCE_EXPECTED_HEAD_SHA}" && "${HEAD_SHA}" != "${SOURCE_EXPECTED_HEAD_SHA}" ]]; then
  die "HEAD ${HEAD_SHA} does not match required ${SOURCE_EXPECTED_HEAD_SHA}"
fi
RECORDED_HEAD_SHA="${HEAD_SHA}"
set_archive_paths

if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  die "worktree is dirty; tracked and untracked files must be clean"
fi

mkdir -p "${OUTPUT_ROOT}"
[[ ! -e "${ARCHIVE_PATH}" ]] || die "target source archive already exists: ${ARCHIVE_PATH}"
[[ ! -e "${SHA_PATH}" ]] || die "target source archive checksum already exists: ${SHA_PATH}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evgnn_full_diag_source.XXXXXX")"
RAW_TAR="${TMP_DIR}/${TOP_LEVEL}.tar"
STAGING_ROOT="${TMP_DIR}/${TOP_LEVEL}"
trap 'rm -rf "${TMP_DIR}"' EXIT
mkdir -p "${STAGING_ROOT}"
printf "%s\n" "${RECORDED_HEAD_SHA}" > "${STAGING_ROOT}/SOURCE_COMMIT_SHA.txt"

for path in "${ALLOWLIST[@]}"; do
  mkdir -p "${STAGING_ROOT}/$(dirname "${path}")"
  cp -p "${REPO_ROOT}/${path}" "${STAGING_ROOT}/${path}"
done

(
  cd "${TMP_DIR}"
  tar -cf "${RAW_TAR}" "${TOP_LEVEL}"
)
gzip -c "${RAW_TAR}" > "${ARCHIVE_PATH}"
validate_archive "${ARCHIVE_PATH}" "${TOP_LEVEL}"
write_sha256_file "${ARCHIVE_PATH}" "${SHA_PATH}"

echo "SOURCE_BUNDLE_OK"
print_transfer_commands
