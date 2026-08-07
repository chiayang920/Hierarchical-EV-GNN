#!/bin/bash
set -Eeuo pipefail

PYTHON_BIN="/opt/anaconda3/envs/evgnn_core/bin/python3.11"
SOURCE_HEAD_EXPECTED="b5f0c2fd1b6895f71699f134aa04b1b9492ded4e"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${1:-${DEFAULT_REPO_ROOT}}"
HISTORICAL_FORMAL_BUNDLE="${2:-}"
HISTORICAL_DIAGNOSTIC_BUNDLE="${3:-}"
LOG_ROOT="${4:-${REPO_ROOT}/local_verification_logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_ROOT}/formal75k_repair_verification_${TIMESTAMP}.log"
FINAL_STATUS="BLOCKED"
BLOCK_REASON="verification did not complete"
TMP_ROOT=""

finish() {
  local rc=$?
  set +e
  if [[ "${FINAL_STATUS}" != "PASS" && "${rc}" -eq 0 ]]; then rc=1; fi
  echo "PHASE=FORMAL_75K_WORKFLOW_LOCAL_REPAIR_VERIFICATION"
  echo "FINAL_STATUS=${FINAL_STATUS}"
  echo "BLOCK_REASON=${BLOCK_REASON}"
  echo "SOURCE_HEAD_EXPECTED=${SOURCE_HEAD_EXPECTED}"
  echo "CONSOLIDATED_LOG=${LOG_PATH}"
  [[ -z "${TMP_ROOT}" ]] || rm -rf "${TMP_ROOT}"
  exit "${rc}"
}
trap finish EXIT
trap 'BLOCK_REASON="command failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

mkdir -p "${LOG_ROOT}"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

[[ -d "${REPO_ROOT}" ]] || { BLOCK_REASON="repaired repository root is missing: ${REPO_ROOT}"; exit 2; }
[[ -f "${HISTORICAL_FORMAL_BUNDLE}" ]] || { BLOCK_REASON="historical formal evidence bundle is required as argument 2"; exit 2; }
[[ -f "${HISTORICAL_DIAGNOSTIC_BUNDLE}" ]] || { BLOCK_REASON="historical diagnostic evidence bundle is required as argument 3"; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { BLOCK_REASON="required interpreter is missing: ${PYTHON_BIN}"; exit 2; }

PYTHON_VERSION="$(${PYTHON_BIN} -I -c 'import platform,sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info[:2] == (3,11) else 11)')"
[[ "${PYTHON_VERSION}" == 3.11.* ]] || { BLOCK_REASON="required interpreter is not Python 3.11: ${PYTHON_VERSION}"; exit 2; }
echo "PYTHON_3_11_BOOTSTRAP=PASS"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "PYTHON_VERSION=${PYTHON_VERSION}"

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/formal75k_repair_verify.XXXXXX")"
WORKFLOW="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
RUNTIME_LIB="${REPO_ROOT}/m3_jobs/lib_formal_75k_runtime.sh"
SUBMIT_HELPER="${REPO_ROOT}/m3_jobs/submit_formal_75k_80cell_workflow.sh"
SOURCE_BUNDLE_HELPER="${REPO_ROOT}/m3_jobs/create_formal_75k_80cell_source_bundle.sh"
HISTORICAL_VALIDATOR="${REPO_ROOT}/scripts/formal_75k_80cell_historical_evidence.py"
for file in "${WORKFLOW}" "${RUNTIME_LIB}" "${SUBMIT_HELPER}" "${SOURCE_BUNDLE_HELPER}" "${HISTORICAL_VALIDATOR}"; do
  [[ -s "${file}" ]] || { BLOCK_REASON="required repair file is missing: ${file}"; exit 2; }
done

set +e
MISSING_OUTPUT="$(/bin/bash -c "source '${RUNTIME_LIB}'; unset EV_GNN_FORMAL_75K_PYTHON; formal75k_require_python311" 2>&1)"
MISSING_STATUS=$?
set -e
[[ "${MISSING_STATUS}" -ne 0 && "${MISSING_OUTPUT}" == *"no bare-python fallback"* ]] || {
  BLOCK_REASON="missing-Python bootstrap test did not fail correctly"; exit 2;
}

WRONG_PYTHON="${TMP_ROOT}/python310"
cat > "${WRONG_PYTHON}" <<'PYWRONG'
#!/bin/bash
if [[ "${1:-}" == "-I" && "${2:-}" == "-c" ]]; then echo 3.10.14; exit 0; fi
echo 3.10.14
PYWRONG
chmod +x "${WRONG_PYTHON}"
set +e
WRONG_OUTPUT="$(EV_GNN_FORMAL_75K_PYTHON="${WRONG_PYTHON}" /bin/bash -c "source '${RUNTIME_LIB}'; formal75k_require_python311" 2>&1)"
WRONG_STATUS=$?
set -e
[[ "${WRONG_STATUS}" -ne 0 && "${WRONG_OUTPUT}" == *"Python 3.11"* ]] || {
  BLOCK_REASON="wrong-Python-version bootstrap test did not fail correctly"; exit 2;
}
EV_GNN_FORMAL_75K_PYTHON="${PYTHON_BIN}" /bin/bash -c "source '${RUNTIME_LIB}'; formal75k_require_python311"
echo "MISSING_WRONG_EXACT_PYTHON_TESTS=PASS"

mkdir -p "${TMP_ROOT}/clean_cwd"
CLEAN_MATRIX="$(cd "${TMP_ROOT}/clean_cwd" && env -i HOME="${HOME}" PATH="/usr/bin:/bin" "${PYTHON_BIN}" "${WORKFLOW}" --print-matrix)"
MATRIX_COUNT="$(printf '%s\n' "${CLEAN_MATRIX}" | awk 'NF{count++} END{print count+0}')"
[[ "${MATRIX_COUNT}" -eq 80 ]] || { BLOCK_REASON="clean-path matrix count is ${MATRIX_COUNT}, expected 80"; exit 2; }
FIRST_MATRIX="$(printf '%s\n' "${CLEAN_MATRIX}" | sed -n '1p')"
LAST_MATRIX="$(printf '%s\n' "${CLEAN_MATRIX}" | sed -n '80p')"
[[ "${FIRST_MATRIX}" == *"task_id=0 scale=25cp algorithm=actiongnn_nonnegative seed=0"* ]] || {
  BLOCK_REASON="formal matrix first cell is incorrect"; exit 2;
}
[[ "${LAST_MATRIX}" == *"task_id=79 scale=1000cp algorithm=hierarchical seed=9"* ]] || {
  BLOCK_REASON="formal matrix final cell is incorrect"; exit 2;
}
printf '%s\n' "${CLEAN_MATRIX}"
echo "CLEAN_PATH_CLI=PASS"
echo "FORMAL_MATRIX=PASS_80_UNIQUE_CELLS"

SENTINEL_BIN="${TMP_ROOT}/hostile_bin"
mkdir -p "${SENTINEL_BIN}"
SUBMISSION_MARKER="${TMP_ROOT}/submission_attempted"
BARE_PYTHON_MARKER="${TMP_ROOT}/bare_python_called"
cat > "${SENTINEL_BIN}/sbatch" <<SHADOW
#!/bin/bash
touch '${SUBMISSION_MARKER}'
echo 'submission command must never run during local verification' >&2
exit 97
SHADOW
cat > "${SENTINEL_BIN}/python" <<SHADOW
#!/bin/bash
touch '${BARE_PYTHON_MARKER}'
echo 'bare python must never be used' >&2
exit 98
SHADOW
chmod +x "${SENTINEL_BIN}/sbatch" "${SENTINEL_BIN}/python"
HOSTILE_OUTPUT="$(cd "${TMP_ROOT}/clean_cwd" && env -i \
  HOME="${HOME}" \
  PATH="${SENTINEL_BIN}:/usr/bin:/bin" \
  EV_GNN_FORMAL_75K_PYTHON="${PYTHON_BIN}" \
  /bin/bash "${SUBMIT_HELPER}" --dry-run --print-matrix)"
[[ "${HOSTILE_OUTPUT}" == *"DRY_RUN_NO_JOBS_SUBMITTED"* ]] || {
  BLOCK_REASON="hostile-PATH workflow dry-run did not complete"; exit 2;
}
[[ ! -e "${SUBMISSION_MARKER}" ]] || { BLOCK_REASON="a submission command was invoked"; exit 2; }
[[ ! -e "${BARE_PYTHON_MARKER}" ]] || { BLOCK_REASON="a bare Python command was invoked"; exit 2; }
echo "HOSTILE_PATH=PASS"
echo "DRY_RUN_NO_JOBS_SUBMITTED"

cd "${REPO_ROOT}"
FOCUSED_TESTS=(
  tests/test_train_td3_gnn_logging.py
  tests/test_formal_75k_80cell_workflow.py
  tests/test_formal_75k_80cell_repair.py
  tests/test_formal_75k_80cell_artifacts.py
  tests/test_formal_75k_80cell_statistics.py
  tests/test_formal_75k_80cell_shell_contracts.py
  tests/test_formal_75k_80cell_historical_evidence.py
  tests/test_formal_75k_local_run_scripts.py
)
"${PYTHON_BIN}" -m pytest -q "${FOCUSED_TESTS[@]}"
echo "FOCUSED_TEST_RESULT=PASS"

AFFECTED_TESTS=(
  tests/test_controlled_evaluator_contract.py
  tests/test_td3_actiongnn_nonnegative_contracts.py
  tests/test_td3_hierarchical_actiongnn_contracts.py
  tests/test_hierarchical_action_projection.py
  tests/test_infrastructure_diagnostics.py
  tests/test_full_infrastructure_diagnostic_eval30_workflow.py
  tests/test_infrastructure_diagnostic_smoke_workflow.py
)
"${PYTHON_BIN}" -m pytest -q "${AFFECTED_TESTS[@]}"
echo "AFFECTED_TEST_RESULT=PASS"

"${PYTHON_BIN}" -m pytest -q
echo "FULL_REPOSITORY_TEST_RESULT=PASS"

"${PYTHON_BIN}" - train_td3_gnn.py \
  scripts/formal_75k_80cell_workflow.py \
  scripts/formal_75k_80cell_artifacts.py \
  scripts/formal_75k_80cell_statistics.py \
  scripts/formal_75k_80cell_diagnostic_validation.py \
  scripts/formal_75k_80cell_historical_evidence.py <<'PY'
import py_compile
import sys
for name in sys.argv[1:]:
    py_compile.compile(name, doraise=True)
print(f"PYTHON_COMPILE_OK files={len(sys.argv)-1}")
PY
echo "PYTHON_COMPILE_CHECK=PASS"

BASH_FILES=(
  m3_jobs/lib_formal_75k_runtime.sh
  m3_jobs/23_formal_75k_80cell_smoke.slurm
  m3_jobs/24_formal_75k_80cell_train.slurm
  m3_jobs/25_formal_75k_80cell_eval30.slurm
  m3_jobs/26_formal_75k_80cell_diagnostics.slurm
  m3_jobs/submit_formal_75k_80cell_workflow.sh
  m3_jobs/create_formal_75k_80cell_source_bundle.sh
  local_tools/run_local_formal75k_repair_verification.sh
  local_tools/run_local_formal75k_functional_smoke.sh
)
for file in "${BASH_FILES[@]}"; do /bin/bash -n "${file}"; done
echo "BASH_SYNTAX_CHECK=PASS"

"${PYTHON_BIN}" "${HISTORICAL_VALIDATOR}" \
  --formal-bundle "${HISTORICAL_FORMAL_BUNDLE}" \
  --diagnostic-bundle "${HISTORICAL_DIAGNOSTIC_BUNDLE}" \
  --require-authoritative-diagnostic-sha
echo "HISTORICAL_EVIDENCE_SCHEMA_VALIDATION=PASS"

SOURCE_DRY_ROOT="${TMP_ROOT}/source_dry_run"
mkdir -p "${SOURCE_DRY_ROOT}"
SOURCE_DRY_OUTPUT="$(env \
  EV_GNN_FORMAL_75K_PYTHON="${PYTHON_BIN}" \
  EV_GNN_FORMAL_75K_SOURCE_DRY_RUN=1 \
  EV_GNN_FORMAL_75K_SOURCE_EXPECTED_HEAD_SHA="${SOURCE_HEAD_EXPECTED}" \
  EV_GNN_FORMAL_75K_SOURCE_OUTPUT_ROOT="${SOURCE_DRY_ROOT}" \
  /bin/bash "${SOURCE_BUNDLE_HELPER}")"
[[ "${SOURCE_DRY_OUTPUT}" == *"DRY_RUN_NO_ARCHIVE_CREATED"* ]] || {
  BLOCK_REASON="source-bundle dry-run did not report no archive"; exit 2;
}
[[ -z "$(find "${SOURCE_DRY_ROOT}" -mindepth 1 -print -quit)" ]] || {
  BLOCK_REASON="source-bundle dry-run created an artefact"; exit 2;
}
[[ ! -e "${SUBMISSION_MARKER}" ]] || { BLOCK_REASON="submission sentinel was triggered"; exit 2; }
echo "SOURCE_BUNDLE_DRY_RUN=PASS_NO_ARCHIVE_CREATED"

FINAL_STATUS="PASS"
BLOCK_REASON="NONE"
