#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=lib_transformer_constraint_runtime.sh
source "${SCRIPT_DIR}/lib_transformer_constraint_runtime.sh"
transformer_constraint_require_python311

OUTPUT_ROOT="${EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_OUTPUT_ROOT:-${HOME}/Downloads/EVGNN_Transformer_Constraint_Evidence}"
EXPECTED_HEAD_SHA="${EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_EXPECTED_HEAD_SHA:-}"
RECORDED_HEAD_SHA=""
TOP_LEVEL=""
ARCHIVE_PATH=""
SHA_PATH=""
SOURCE_SHA_PATH=""
export COPYFILE_DISABLE=1

ALLOWLIST=(
  train_td3_gnn.py
  evaluate_td3_gnn.py
  evaluate_transformer_constraint_projection_diagnostics.py
  evaluate_td3_gnn_infrastructure_diagnostics.py
  TD3/TD3_ActionGNN_NonNegative.py
  TD3/TD3_ActionGNN_Controlled.py
  TD3/TD3_HierarchicalActionGNN.py
  TD3/TD3_HierarchicalActionGNN_TransformerConstraint.py
  config_files/PublicPST_25cp.yaml
  config_files/PublicPST_100.yaml
  config_files/PublicPST_500.yaml
  config_files/PublicPST_1000.yaml
  m3_jobs/31_transformer_constraint_40cell_smoke.slurm
  m3_jobs/32_transformer_constraint_40cell_train.slurm
  m3_jobs/33_transformer_constraint_40cell_eval30.slurm
  m3_jobs/34_transformer_constraint_40cell_diagnostics.slurm
  m3_jobs/submit_transformer_constraint_40cell_workflow.sh
  m3_jobs/create_transformer_constraint_40cell_source_bundle.sh
  m3_jobs/lib_transformer_constraint_runtime.sh
  scripts/transformer_constraint_40cell_workflow.py
  scripts/transformer_constraint_40cell_artifacts.py
  scripts/transformer_constraint_40cell_diagnostic_validation.py
  scripts/transformer_constraint_comparator_gate.py
  scripts/transformer_constraint_projection_diagnostics.py
  scripts/validate_full_infrastructure_diagnostic_eval30.py
  utils/ev2gym_training_utils.py
  utils/infrastructure_diagnostics.py
  utils/replay_buffer_actiongnn.py
  utils/state_public_pst_gnn.py
  utils/transformer_feasibility_projection.py
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

set_archive_paths() {
  TOP_LEVEL="EV-GNN-transformer-constraint-40cell-${RECORDED_HEAD_SHA}"
  ARCHIVE_PATH="${OUTPUT_ROOT}/${TOP_LEVEL}.tar.gz"
  SHA_PATH="${ARCHIVE_PATH}.sha256"
  SOURCE_SHA_PATH="${OUTPUT_ROOT}/${TOP_LEVEL}.SOURCE_COMMIT_SHA.txt"
}

is_prohibited_path() {
  local path="$1"
  case "${path}" in
    .git|.git/*|*/.git|*/.git/*) return 0 ;;
    .superpowers|.superpowers/*|*/.superpowers|*/.superpowers/*) return 0 ;;
    EVGNN_Research_Artefacts|EVGNN_Research_Artefacts/*|*/EVGNN_Research_Artefacts|*/EVGNN_Research_Artefacts/*) return 0 ;;
    evidence|evidence/*|*/evidence|*/evidence/*) return 0 ;;
    saved_models|saved_models/*|*/saved_models|*/saved_models/*) return 0 ;;
    results|results/*|*/results|*/results/*) return 0 ;;
    logs|logs/*|*/logs|*/logs/*) return 0 ;;
    artifacts|artifacts/*|*/artifacts|*/artifacts/*) return 0 ;;
    outputs|outputs/*|*/outputs|*/outputs/*) return 0 ;;
    checkpoints|checkpoints/*|*/checkpoints|*/checkpoints/*) return 0 ;;
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

validate_allowlist_filesystem() {
  local path
  for path in "${ALLOWLIST[@]}"; do
    is_prohibited_path "${path}" && die "prohibited source-bundle path in allowlist: ${path}"
    [[ -e "${REPO_ROOT}/${path}" ]] || die "allowlisted source path is missing: ${path}"
  done
}

validate_allowlist_commit() {
  local path
  for path in "${ALLOWLIST[@]}"; do
    is_prohibited_path "${path}" && die "prohibited source-bundle path in allowlist: ${path}"
    git -C "${REPO_ROOT}" cat-file -e "${EXPECTED_HEAD_SHA}:${path}" || die "allowlisted source path is missing from expected commit: ${path}"
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

validate_allowlist_filesystem

if [[ "${EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_DRY_RUN:-0}" == "1" ]]; then
  RECORDED_HEAD_SHA="${EXPECTED_HEAD_SHA:-$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf "DRY_RUN_HEAD_UNKNOWN")}"
  set_archive_paths
  echo "dry_run=1"
  echo "repo_root=${REPO_ROOT}"
  echo "recorded_head_sha=${RECORDED_HEAD_SHA}"
  echo "top_level=${TOP_LEVEL}"
  echo "source_archive=${ARCHIVE_PATH}"
  echo "source_archive_sha256=${SHA_PATH}"
  echo "source_commit_sha_file=${SOURCE_SHA_PATH}"
  echo "ALLOWLIST"
  printf "%s\n" "${ALLOWLIST[@]}"
  echo "DRY_RUN_NO_ARCHIVE_CREATED"
  exit 0
fi

cd "${REPO_ROOT}"
[[ -n "${EXPECTED_HEAD_SHA}" ]] || die "EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_EXPECTED_HEAD_SHA is required"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_HEAD_SHA}" ]] || die "HEAD does not match EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_EXPECTED_HEAD_SHA"
git diff-index --quiet HEAD -- || die "tracked worktree is dirty"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || die "worktree contains untracked or modified files"
validate_allowlist_commit

RECORDED_HEAD_SHA="${EXPECTED_HEAD_SHA}"
set_archive_paths
mkdir -p "${OUTPUT_ROOT}"
[[ ! -e "${ARCHIVE_PATH}" ]] || die "target source archive already exists: ${ARCHIVE_PATH}"
[[ ! -e "${SHA_PATH}" ]] || die "target source archive checksum already exists: ${SHA_PATH}"
[[ ! -e "${SOURCE_SHA_PATH}" ]] || die "target source identity file already exists: ${SOURCE_SHA_PATH}"

git archive \
  --format=tar \
  --prefix="${TOP_LEVEL}/" \
  --mtime="1970-01-01T00:00:00Z" \
  --add-virtual-file="${TOP_LEVEL}/SOURCE_COMMIT_SHA.txt:${RECORDED_HEAD_SHA}" \
  "${EXPECTED_HEAD_SHA}" \
  -- "${ALLOWLIST[@]}" | gzip -n > "${ARCHIVE_PATH}"
tar -tzf "${ARCHIVE_PATH}" >/dev/null
printf "%s\n" "${RECORDED_HEAD_SHA}" > "${SOURCE_SHA_PATH}"
write_sha256_file "${ARCHIVE_PATH}" "${SHA_PATH}"

echo "SOURCE_HEAD=${RECORDED_HEAD_SHA}"
echo "SOURCE_ARCHIVE=${ARCHIVE_PATH}"
echo "SOURCE_ARCHIVE_SHA256=${SHA_PATH}"
echo "SOURCE_COMMIT_SHA_FILE=${SOURCE_SHA_PATH}"
echo "SOURCE_BUNDLE_CREATED"
