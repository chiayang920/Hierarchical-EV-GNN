#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CURRENT_DIR="$(pwd -P)"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_SOURCE_OUTPUT_ROOT:-${HOME}/Downloads/EVGNN_Formal_Evidence}"
SOURCE_EXPECTED_HEAD_SHA="${EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXPECTED_HEAD_SHA:-}"
BRANCH_REQUIRED="${EV_GNN_FULL_DIAGNOSTIC_SOURCE_REQUIRED_BRANCH:-impl/full-per-infrastructure-diagnostics-eval30-compact}"
RECORDED_HEAD_SHA=""
TOP_LEVEL=""
ARCHIVE_PATH=""
SHA_PATH=""
export COPYFILE_DISABLE=1

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
  m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm
  m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh
  m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh
  scripts/validate_full_infrastructure_diagnostic_eval30.py
  utils/ev2gym_training_utils.py
  utils/infrastructure_diagnostics.py
  utils/state_public_pst_gnn.py
)

if [[ -n "${EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXTRA_PATH:-}" ]]; then
  ALLOWLIST+=("${EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXTRA_PATH}")
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

set_archive_paths() {
  TOP_LEVEL="EV-GNN-full-infrastructure-diagnostics-eval30-${RECORDED_HEAD_SHA}"
  ARCHIVE_PATH="${OUTPUT_ROOT}/${TOP_LEVEL}.tar.gz"
  SHA_PATH="${ARCHIVE_PATH}.sha256"
}

is_prohibited_path() {
  local path="$1"
  case "${path}" in
    .git|.git/*|*/.git|*/.git/*) return 0 ;;
    saved_models|saved_models/*|*/saved_models|*/saved_models/*) return 0 ;;
    checkpoints|checkpoints/*|*/checkpoints|*/checkpoints/*) return 0 ;;
    optimizer|optimizer/*|*/optimizer|*/optimizer/*) return 0 ;;
    optimizers|optimizers/*|*/optimizers|*/optimizers/*) return 0 ;;
    EVGNN_Formal_Evidence|EVGNN_Formal_Evidence/*|*/EVGNN_Formal_Evidence|*/EVGNN_Formal_Evidence/*) return 0 ;;
    task_packages|task_packages/*|*/task_packages|*/task_packages/*) return 0 ;;
    __pycache__|__pycache__/*|*/__pycache__|*/__pycache__/*) return 0 ;;
    .pytest_cache|.pytest_cache/*|*/.pytest_cache|*/.pytest_cache/*) return 0 ;;
    __MACOSX|__MACOSX/*|*/__MACOSX|*/__MACOSX/*) return 0 ;;
    ._*|*/._*) return 0 ;;
    .DS_Store|*/.DS_Store) return 0 ;;
    *.pyc|*.tar|*.tar.gz|*.tgz|*.zip|*.gz|*.csv|*.pt|*.pth|*.ckpt) return 0 ;;
    *model.best*|*model.last*) return 0 ;;
  esac
  return 1
}

validate_allowlist() {
  local path
  for path in "${ALLOWLIST[@]}"; do
    if is_prohibited_path "${path}"; then
      die "prohibited source-bundle path in allowlist: ${path}"
    fi
    if [[ ! -e "${REPO_ROOT}/${path}" ]]; then
      die "allowlisted source path is missing: ${path}"
    fi
  done
}

write_sha256_file() {
  local path="$1"
  local output_path="$2"
  local digest
  if command -v sha256sum >/dev/null 2>&1; then
    digest="$(sha256sum "${path}" | awk '{print $1}')"
    printf "%s  %s\n" "${digest}" "$(basename "${path}")" > "${output_path}"
    return
  fi
  if command -v shasum >/dev/null 2>&1; then
    digest="$(shasum -a 256 "${path}" | awk '{print $1}')"
    printf "%s  %s\n" "${digest}" "$(basename "${path}")" > "${output_path}"
    return
  fi
  die "sha256sum or shasum is required"
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
  echo "M3_EXPECTED_SOURCE_COMMIT_ENV=EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT=${RECORDED_HEAD_SHA}"
}

validate_allowlist

if [[ "${EV_GNN_FULL_DIAGNOSTIC_SOURCE_DRY_RUN:-0}" == "1" ]]; then
  RECORDED_HEAD_SHA="${SOURCE_EXPECTED_HEAD_SHA:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf "DRY_RUN_HEAD_UNKNOWN")}"
  set_archive_paths
  echo "dry_run=1"
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
[[ "${CURRENT_BRANCH}" == "${BRANCH_REQUIRED}" ]] || die "source bundle must be created on branch ${BRANCH_REQUIRED}; got ${CURRENT_BRANCH}"

HEAD_SHA="$(git rev-parse HEAD)"
if [[ -n "${SOURCE_EXPECTED_HEAD_SHA}" && "${HEAD_SHA}" != "${SOURCE_EXPECTED_HEAD_SHA}" ]]; then
  die "HEAD ${HEAD_SHA} does not match required ${SOURCE_EXPECTED_HEAD_SHA}"
fi
git diff-index --quiet HEAD -- || die "tracked worktree is dirty"
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  die "worktree contains untracked or modified files"
fi

RECORDED_HEAD_SHA="${HEAD_SHA}"
set_archive_paths
mkdir -p "${OUTPUT_ROOT}"
[[ ! -e "${ARCHIVE_PATH}" ]] || die "target source archive already exists: ${ARCHIVE_PATH}"
[[ ! -e "${SHA_PATH}" ]] || die "target source archive checksum already exists: ${SHA_PATH}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evgnn_full_source_bundle.XXXXXX")"
STAGING_ROOT="${TMP_DIR}/${TOP_LEVEL}"
RAW_TAR="${TMP_DIR}/${TOP_LEVEL}.tar"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${STAGING_ROOT}"
git archive --format=tar --prefix="${TOP_LEVEL}/" HEAD -- "${ALLOWLIST[@]}" > "${RAW_TAR}"
tar -xf "${RAW_TAR}" -C "${TMP_DIR}"
printf "%s\n" "${RECORDED_HEAD_SHA}" > "${STAGING_ROOT}/SOURCE_COMMIT_SHA.txt"
mkdir -p "${STAGING_ROOT}/runtime_metadata"

python - "${STAGING_ROOT}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "runtime_metadata/source_file_checksums.sha256"
paths = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
with manifest.open("w", encoding="utf-8") as handle:
    for path in paths:
        handle.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}\n")
PY

tar -cf "${RAW_TAR}" -C "${TMP_DIR}" "${TOP_LEVEL}"
gzip -n -c "${RAW_TAR}" > "${ARCHIVE_PATH}"
tar -tzf "${ARCHIVE_PATH}" >/dev/null

if tar -tzf "${ARCHIVE_PATH}" | grep -E '(^|/)(\.git|saved_models|checkpoints|EVGNN_Formal_Evidence|task_packages|__pycache__|\.pytest_cache|__MACOSX)(/|$)|(^|/)(\._|\.DS_Store$)|\.(pyc|tar|tar\.gz|tgz|zip|gz|csv|pt|pth|ckpt)$|model\.best|model\.last' >/dev/null; then
  die "archive contains a prohibited path"
fi

python - "${ARCHIVE_PATH}" "${TOP_LEVEL}" <<'PY'
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
top_level = sys.argv[2]
with tempfile.TemporaryDirectory() as temporary:
    with tarfile.open(archive_path, "r:gz") as archive:
        seen = set()
        for member in archive.getmembers():
            name = safe_member_name(member)
            if name in seen:
                raise SystemExit(f"duplicate source bundle member: {name}")
            seen.add(name)
            output = Path(temporary) / name
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(f"cannot read source bundle member: {name}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(source.read())
    root = Path(temporary) / top_level
    source_sha = (root / "SOURCE_COMMIT_SHA.txt").read_text(encoding="utf-8").strip()
    if not source_sha:
        raise SystemExit("SOURCE_COMMIT_SHA.txt is empty")
    manifest = root / "runtime_metadata/source_file_checksums.sha256"
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    expected = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    )
    if sorted(entries) != expected:
        raise SystemExit("source manifest coverage mismatch")
    for name, digest in entries.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != digest:
            raise SystemExit(f"source manifest checksum mismatch: {name}")
PY

write_sha256_file "${ARCHIVE_PATH}" "${SHA_PATH}"

echo "SOURCE_BUNDLE_OK"
print_transfer_commands
