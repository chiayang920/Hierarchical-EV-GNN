#!/bin/bash

set -euo pipefail

SOURCE_EXPECTED_HEAD_SHA="${EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_EXPECTED_HEAD_SHA:-}"
BRANCH_REQUIRED="main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
CURRENT_DIR="$(pwd -P)"
OUTPUT_ROOT="${EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_OUTPUT_ROOT:-${HOME}/Downloads/EVGNN_Formal_Evidence}"
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
  m3_jobs/19_infrastructure_diagnostic_smoke_eval.slurm
  scripts/validate_infrastructure_diagnostic_smoke.py
  utils/ev2gym_training_utils.py
  utils/infrastructure_diagnostics.py
  utils/state_public_pst_gnn.py
)

if [[ -n "${EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_EXTRA_PATH:-}" ]]; then
  ALLOWLIST+=("${EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_EXTRA_PATH}")
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

set_archive_paths() {
  TOP_LEVEL="EV-GNN-infrastructure-diagnostic-smoke-${RECORDED_HEAD_SHA}"
  ARCHIVE_PATH="${OUTPUT_ROOT}/${TOP_LEVEL}.tar.gz"
  SHA_PATH="${ARCHIVE_PATH}.sha256"
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
  die "sha256sum or shasum is required to write ${output_path}"
}

is_prohibited_path() {
  local path="$1"
  case "${path}" in
    .git|.git/*|*/.git|*/.git/*) return 0 ;;
    saved_models|saved_models/*|*/saved_models|*/saved_models/*) return 0 ;;
    Downloads|Downloads/*|*/Downloads|*/Downloads/*) return 0 ;;
    task_packages|task_packages/*|*/task_packages|*/task_packages/*) return 0 ;;
    __pycache__|__pycache__/*|*/__pycache__|*/__pycache__/*) return 0 ;;
    .pytest_cache|.pytest_cache/*|*/.pytest_cache|*/.pytest_cache/*) return 0 ;;
    __MACOSX|__MACOSX/*|*/__MACOSX|*/__MACOSX/*) return 0 ;;
    ._*|*/._*) return 0 ;;
    .DS_Store|*/.DS_Store) return 0 ;;
    *.tar|*.tar.gz|*.tgz|*.zip|*.gz|*.csv|*.pt|*.pth) return 0 ;;
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
  echo "M3_EXPECTED_SOURCE_COMMIT_ENV=EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT=${RECORDED_HEAD_SHA}"
}

validate_allowlist

if [[ "${EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_DRY_RUN:-0}" == "1" ]]; then
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
RECORDED_HEAD_SHA="${HEAD_SHA}"
set_archive_paths

git diff-index --quiet HEAD -- || die "tracked worktree is dirty"

mkdir -p "${OUTPUT_ROOT}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evgnn_source_bundle.XXXXXX")"
RAW_TAR="${TMP_DIR}/${TOP_LEVEL}.tar"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${TMP_DIR}/${TOP_LEVEL}"
printf "%s\n" "${RECORDED_HEAD_SHA}" > "${TMP_DIR}/${TOP_LEVEL}/SOURCE_COMMIT_SHA.txt"

git archive --format=tar --prefix="${TOP_LEVEL}/" HEAD -- "${ALLOWLIST[@]}" > "${RAW_TAR}"
tar -rf "${RAW_TAR}" -C "${TMP_DIR}" "${TOP_LEVEL}/SOURCE_COMMIT_SHA.txt"
gzip -c "${RAW_TAR}" > "${ARCHIVE_PATH}"
tar -tzf "${ARCHIVE_PATH}" >/dev/null

if tar -tzf "${ARCHIVE_PATH}" | grep -E '(^|/)(\.git|saved_models|Downloads|task_packages|__pycache__|\.pytest_cache|__MACOSX)(/|$)|(^|/)(\._|\.DS_Store$)|\.(tar|tar\.gz|tgz|zip|gz|csv|pt|pth)$|model\.best|model\.last' >/dev/null; then
  die "archive contains a prohibited path"
fi

write_sha256_file "${ARCHIVE_PATH}" "${SHA_PATH}"

echo "SOURCE_BUNDLE_OK"
print_transfer_commands
