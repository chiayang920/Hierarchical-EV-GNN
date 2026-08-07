#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
source "${SCRIPT_DIR}/lib_formal_75k_runtime.sh"
formal75k_require_python311
OUTPUT_ROOT="${EV_GNN_FORMAL_75K_SOURCE_OUTPUT_ROOT:-${HOME}/Downloads/EVGNN_Formal_Evidence}"
EXPECTED_HEAD_SHA="${EV_GNN_FORMAL_75K_SOURCE_EXPECTED_HEAD_SHA:-}"
REQUIRED_BRANCH="${EV_GNN_FORMAL_75K_SOURCE_REQUIRED_BRANCH:-exp/formal-75k-nonnegative-comparison-v1}"
RECORDED_HEAD_SHA=""
TOP_LEVEL=""
ARCHIVE_PATH=""
SHA_PATH=""
export COPYFILE_DISABLE=1

ALLOWLIST=(
  train_td3_gnn.py
  evaluate_td3_gnn.py
  evaluate_td3_gnn_infrastructure_diagnostics.py
  TD3/TD3_ActionGNN_NonNegative.py
  TD3/TD3_ActionGNN_Controlled.py
  TD3/TD3_HierarchicalActionGNN.py
  config_files/PublicPST_25cp.yaml
  config_files/PublicPST_100.yaml
  config_files/PublicPST_500.yaml
  config_files/PublicPST_1000.yaml
  m3_jobs/23_formal_75k_80cell_smoke.slurm
  m3_jobs/24_formal_75k_80cell_train.slurm
  m3_jobs/25_formal_75k_80cell_eval30.slurm
  m3_jobs/26_formal_75k_80cell_diagnostics.slurm
  m3_jobs/submit_formal_75k_80cell_workflow.sh
  m3_jobs/create_formal_75k_80cell_source_bundle.sh
  scripts/formal_75k_80cell_workflow.py
  scripts/formal_75k_80cell_artifacts.py
  scripts/formal_75k_80cell_statistics.py
  scripts/formal_75k_80cell_diagnostic_validation.py
  m3_jobs/lib_formal_75k_runtime.sh
  scripts/aggregate_controlled_multiscale_eval30.py
  utils/ev2gym_training_utils.py
  utils/infrastructure_diagnostics.py
  utils/replay_buffer_actiongnn.py
  utils/state_public_pst_gnn.py
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

set_archive_paths() {
  TOP_LEVEL="EV-GNN-formal-75k-80cell-${RECORDED_HEAD_SHA}"
  ARCHIVE_PATH="${OUTPUT_ROOT}/${TOP_LEVEL}.tar.gz"
  SHA_PATH="${ARCHIVE_PATH}.sha256"
}

is_prohibited_path() {
  local path="$1"
  case "${path}" in
    .git|.git/*|*/.git|*/.git/*) return 0 ;;
    saved_models|saved_models/*|*/saved_models|*/saved_models/*) return 0 ;;
    checkpoints|checkpoints/*|*/checkpoints|*/checkpoints/*) return 0 ;;
    EVGNN_Formal_Evidence|EVGNN_Formal_Evidence/*|*/EVGNN_Formal_Evidence|*/EVGNN_Formal_Evidence/*) return 0 ;;
    task_packages|task_packages/*|*/task_packages|*/task_packages/*) return 0 ;;
    __pycache__|__pycache__/*|*/__pycache__|*/__pycache__/*) return 0 ;;
    .pytest_cache|.pytest_cache/*|*/.pytest_cache|*/.pytest_cache/*) return 0 ;;
    __MACOSX|__MACOSX/*|*/__MACOSX) return 0 ;;
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
    is_prohibited_path "${path}" && die "prohibited source-bundle path in allowlist: ${path}"
    [[ -e "${REPO_ROOT}/${path}" ]] || die "allowlisted source path is missing: ${path}"
  done
}

write_sha256_file() {
  local path="$1"
  local output_path="$2"
  local digest=""
  if command -v sha256sum >/dev/null 2>&1; then
    digest="$(sha256sum "${path}" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    digest="$(shasum -a 256 "${path}" | awk '{print $1}')"
  else
    die "sha256sum or shasum is required"
  fi
  printf "%s  %s\n" "${digest}" "$(basename "${path}")" > "${output_path}"
}

validate_allowlist

if [[ "${EV_GNN_FORMAL_75K_SOURCE_DRY_RUN:-0}" == "1" ]]; then
  RECORDED_HEAD_SHA="${EXPECTED_HEAD_SHA:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf "DRY_RUN_HEAD_UNKNOWN")}"
  set_archive_paths
  echo "dry_run=1"
  echo "repo_root=${REPO_ROOT}"
  echo "recorded_head_sha=${RECORDED_HEAD_SHA}"
  echo "required_branch=${REQUIRED_BRANCH}"
  echo "top_level=${TOP_LEVEL}"
  echo "source_archive=${ARCHIVE_PATH}"
  echo "source_archive_sha256=${SHA_PATH}"
  echo "ALLOWLIST"
  printf "%s\n" "${ALLOWLIST[@]}"
  echo "DRY_RUN_NO_ARCHIVE_CREATED"
  exit 0
fi

cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == "${REQUIRED_BRANCH}" ]] || die "source bundle must be created on branch ${REQUIRED_BRANCH}"
[[ -n "${EXPECTED_HEAD_SHA}" ]] || die "EV_GNN_FORMAL_75K_SOURCE_EXPECTED_HEAD_SHA is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_HEAD_SHA}" ]] || die "HEAD does not match EV_GNN_FORMAL_75K_SOURCE_EXPECTED_HEAD_SHA"
git diff-index --quiet HEAD -- || die "tracked worktree is dirty"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die "worktree contains untracked or modified files"

RECORDED_HEAD_SHA="${EXPECTED_HEAD_SHA}"
set_archive_paths
mkdir -p "${OUTPUT_ROOT}"
[[ ! -e "${ARCHIVE_PATH}" ]] || die "target source archive already exists: ${ARCHIVE_PATH}"
[[ ! -e "${SHA_PATH}" ]] || die "target source archive checksum already exists: ${SHA_PATH}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/evgnn_formal75k_source.XXXXXX")"
STAGING_ROOT="${TMP_DIR}/${TOP_LEVEL}"
RAW_TAR="${TMP_DIR}/${TOP_LEVEL}.tar"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${STAGING_ROOT}/runtime_metadata"
for path in "${ALLOWLIST[@]}"; do
  mkdir -p "${STAGING_ROOT}/$(dirname "${path}")"
  cp "${REPO_ROOT}/${path}" "${STAGING_ROOT}/${path}"
done
printf "%s\n" "${RECORDED_HEAD_SHA}" > "${STAGING_ROOT}/SOURCE_COMMIT_SHA.txt"
tar -cf "${RAW_TAR}" -C "${TMP_DIR}" "${TOP_LEVEL}"
gzip -n -c "${RAW_TAR}" > "${ARCHIVE_PATH}"
tar -tzf "${ARCHIVE_PATH}" >/dev/null
write_sha256_file "${ARCHIVE_PATH}" "${SHA_PATH}"

echo "SOURCE_ARCHIVE=${ARCHIVE_PATH}"
echo "SOURCE_ARCHIVE_SHA256=${SHA_PATH}"
echo "SOURCE_BUNDLE_CREATED"
