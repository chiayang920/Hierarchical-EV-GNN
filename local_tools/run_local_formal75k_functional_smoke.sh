#!/bin/bash
set -Eeuo pipefail

PYTHON_BIN="/opt/anaconda3/envs/evgnn_core/bin/python3.11"
SOURCE_IDENTITY="b5f0c2fd1b6895f71699f134aa04b1b9492ded4e"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${1:-${DEFAULT_REPO_ROOT}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EVIDENCE_ROOT="${2:-${REPO_ROOT}/local_functional_smoke_evidence/${TIMESTAMP}}"
LOG_PATH="${EVIDENCE_ROOT}/formal75k_local_functional_smoke.log"
RUN_ROOT="${EVIDENCE_ROOT}/runs"
PACKAGE_ROOT="${EVIDENCE_ROOT}/packages"
STAGE_ROOT="${EVIDENCE_ROOT}/validated_stage"
RESOURCE_PROFILE="${EVIDENCE_ROOT}/LOCAL_FUNCTIONAL_SMOKE_ONLY_resource_profile.env"
JOB_ID="$(date +%m%d%H%M%S)"
FINAL_STATUS="BLOCKED"
BLOCK_REASON="local functional smoke did not complete"

finish() {
  local rc=$?
  set +e
  if [[ "${FINAL_STATUS}" != "PASS" && "${rc}" -eq 0 ]]; then rc=1; fi
  echo "PHASE=FORMAL_75K_LOCAL_FUNCTIONAL_SMOKE"
  echo "FINAL_STATUS=${FINAL_STATUS}"
  echo "BLOCK_REASON=${BLOCK_REASON}"
  echo "SMOKE_JOB_ID=${JOB_ID}"
  echo "SMOKE_CELL_COUNT=2"
  echo "SMOKE_CELLS=25cp_actiongnn_nonnegative_seed0,25cp_hierarchical_seed0"
  echo "SCIENTIFIC_CLAIM_GENERATED=NO"
  echo "EVIDENCE_ROOT=${EVIDENCE_ROOT}"
  echo "CONSOLIDATED_LOG=${LOG_PATH}"
  exit "${rc}"
}
trap finish EXIT
trap 'BLOCK_REASON="command failed at line ${LINENO}: ${BASH_COMMAND}"' ERR

mkdir -p "${EVIDENCE_ROOT}" "${RUN_ROOT}" "${PACKAGE_ROOT}"
touch "${LOG_PATH}"
exec > >(tee -a "${LOG_PATH}") 2>&1

[[ -d "${REPO_ROOT}" ]] || { BLOCK_REASON="repaired repository root is missing: ${REPO_ROOT}"; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { BLOCK_REASON="required interpreter is missing: ${PYTHON_BIN}"; exit 2; }
PYTHON_VERSION="$(${PYTHON_BIN} -I -c 'import platform,sys; print(platform.python_version()); raise SystemExit(0 if sys.version_info[:2] == (3,11) else 11)')"
[[ "${PYTHON_VERSION}" == 3.11.* ]] || { BLOCK_REASON="Python 3.11 is required"; exit 2; }
SMOKE_LAUNCHER="${REPO_ROOT}/m3_jobs/23_formal_75k_80cell_smoke.slurm"
WORKFLOW="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
[[ -s "${SMOKE_LAUNCHER}" && -s "${WORKFLOW}" ]] || { BLOCK_REASON="smoke workflow files are missing"; exit 2; }

cat > "${RESOURCE_PROFILE}" <<'PROFILE'
# This profile is for LOCAL_FUNCTIONAL_SMOKE_ONLY. It is not an M3 resource approval.
RESOURCE_PROFILE_CONTEXT=LOCAL_FUNCTIONAL_SMOKE_ONLY
RESOURCE_PROFILE_APPROVED=YES
SMOKE_CPUS_PER_TASK=1
SMOKE_MEM=LOCAL_FUNCTIONAL_SMOKE_ONLY
SMOKE_TIME=LOCAL_FUNCTIONAL_SMOKE_ONLY
PROFILE

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m pytest -q \
  tests/test_td3_actiongnn_nonnegative_contracts.py \
  tests/test_td3_hierarchical_actiongnn_contracts.py \
  tests/test_hierarchical_action_projection.py

echo "POLICY_ACTION_CONTRACT_TESTS=PASS"

COMMON_ENV=(
  EV_GNN_FORMAL_75K_PYTHON="${PYTHON_BIN}"
  EV_GNN_FORMAL_75K_REPO_ROOT="${REPO_ROOT}"
  EV_GNN_FORMAL_75K_RUN_ROOT="${RUN_ROOT}"
  EV_GNN_FORMAL_75K_OUTPUT_ROOT="${PACKAGE_ROOT}"
  EV_GNN_FORMAL_75K_RESOURCE_PROFILE="${RESOURCE_PROFILE}"
  EV_GNN_FORMAL_75K_SOURCE_IDENTITY="${SOURCE_IDENTITY}"
  EV_GNN_FORMAL_75K_SOURCE_BUNDLE_IDENTITY="LOCAL_FUNCTIONAL_SMOKE_ONLY_${SOURCE_IDENTITY}"
  SLURM_ARRAY_JOB_ID="${JOB_ID}"
  SLURM_JOB_ID="${JOB_ID}"
  SLURM_CPUS_PER_TASK=1
)

env "${COMMON_ENV[@]}" SLURM_ARRAY_TASK_ID=0 /bin/bash "${SMOKE_LAUNCHER}"
env "${COMMON_ENV[@]}" SLURM_ARRAY_TASK_ID=1 /bin/bash "${SMOKE_LAUNCHER}"

SCHEMA_CHECK_OUTPUT="${EVIDENCE_ROOT}/training_log_schema_check.txt"
: > "${SCHEMA_CHECK_OUTPUT}"
for task_id in 0 1; do
  TRAINING_LOG="$(find "${RUN_ROOT}/job${JOB_ID}/task${task_id}/train" -type f -name training_log.csv -print -quit)"
  [[ -s "${TRAINING_LOG}" ]] || {
    BLOCK_REASON="task ${task_id} training_log.csv is missing"; exit 2;
  }
  "${PYTHON_BIN}" - "${task_id}" "${TRAINING_LOG}" <<'PYSCHEMA' | tee -a "${SCHEMA_CHECK_OUTPUT}"
import csv
import math
import sys
from pathlib import Path


task_id = sys.argv[1]
path = Path(sys.argv[2])
with path.open(newline="", encoding="utf-8") as handle:
    raw_rows = list(csv.reader(handle))
if len(raw_rows) < 2:
    raise SystemExit(f"task {task_id}: training log has no data rows: {path}")
header = raw_rows[0]
if "eval/mean_reward" not in header:
    raise SystemExit(f"task {task_id}: eval/mean_reward is absent from CSV header")
for line_number, raw_row in enumerate(raw_rows[1:], start=2):
    if len(raw_row) != len(header):
        raise SystemExit(
            f"task {task_id}: row {line_number} has {len(raw_row)} columns; "
            f"header has {len(header)}"
        )

with path.open(newline="", encoding="utf-8") as handle:
    dict_rows = list(csv.DictReader(handle))
evaluation_rows = [
    row for row in dict_rows
    if row.get("type") in {"evaluation", "final_evaluation"}
]
if not evaluation_rows:
    raise SystemExit(f"task {task_id}: no evaluation rows found")
for row in evaluation_rows:
    if None in row:
        raise SystemExit(f"task {task_id}: unnamed extra CSV columns remain")
    value = row.get("eval/mean_reward")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"task {task_id}: eval/mean_reward must be numeric; got {value!r}"
        ) from exc
    if not math.isfinite(numeric_value):
        raise SystemExit(
            f"task {task_id}: eval/mean_reward must be finite; got {numeric_value!r}"
        )
print(f"TRAINING_LOG_SCHEMA_TASK_{task_id}=PASS")
print(f"TRAINING_LOG_HEADER_COLUMNS_TASK_{task_id}={len(header)}")
print(f"TRAINING_LOG_EVALUATION_ROWS_TASK_{task_id}={len(evaluation_rows)}")
PYSCHEMA
done
echo "TRAINING_LOG_SCHEMA=PASS_2_OF_2"

PACKAGE_COUNT="$(find "${PACKAGE_ROOT}" -maxdepth 1 -type f -name "m3_formal75k_smoke_*_job${JOB_ID}_task*.tar.gz" | wc -l | tr -d ' ')"
[[ "${PACKAGE_COUNT}" -eq 2 ]] || { BLOCK_REASON="expected two smoke packages; observed ${PACKAGE_COUNT}"; exit 2; }

GATE_OUTPUT="${EVIDENCE_ROOT}/smoke_gate_output.txt"
"${PYTHON_BIN}" "${WORKFLOW}" validate-smoke-packages \
  --package-root "${PACKAGE_ROOT}" \
  --job-id "${JOB_ID}" \
  --stage-root "${STAGE_ROOT}" \
  --source-identity "${SOURCE_IDENTITY}" \
  --source-bundle-identity "LOCAL_FUNCTIONAL_SMOKE_ONLY_${SOURCE_IDENTITY}" | tee "${GATE_OUTPUT}"
grep -q '^STATUS=PASS$' "${GATE_OUTPUT}"
grep -q '^SMOKE_CELL_COUNT=2$' "${GATE_OUTPUT}"
grep -q '^SCIENTIFIC_CLAIM_GENERATED=NO$' "${GATE_OUTPUT}"

for task_id in 0 1; do
  TASK_ROOT="${RUN_ROOT}/job${JOB_ID}/task${task_id}"
  [[ -s "${TASK_ROOT}/stdout.log" ]] || { BLOCK_REASON="task ${task_id} stdout log is missing"; exit 2; }
  [[ -f "${TASK_ROOT}/stderr.log" ]] || { BLOCK_REASON="task ${task_id} stderr log is missing"; exit 2; }
  if grep -Eqi 'Traceback \(most recent call last\)|ModuleNotFoundError|ImportError|SyntaxError|segmentation fault|core dumped|command not found' "${TASK_ROOT}/stderr.log"; then
    BLOCK_REASON="task ${task_id} stderr contains an implementation failure"; exit 2
  fi
  find "${TASK_ROOT}/train" -type f -name training_log.csv -print -quit | grep -q .
  find "${TASK_ROOT}/train" -type f -name model.best_actor -print -quit | grep -q .
  find "${TASK_ROOT}/smoke_eval" -type f -name '*.csv' -print -quit | grep -q .
done

[[ -f "${STAGE_ROOT}/trainjob${JOB_ID}/task0/checkpoint/model.best_actor" ]] || {
  BLOCK_REASON="corrected policy model.best was not staged"; exit 2;
}
[[ -f "${STAGE_ROOT}/trainjob${JOB_ID}/task1/checkpoint/model.best_actor" ]] || {
  BLOCK_REASON="hierarchical policy model.best was not staged"; exit 2;
}

echo "POLICY_ROUTES=PASS_BOTH"
echo "TRAINING_LOGS=PASS_2_OF_2"
echo "SCHEDULED_EVALUATIONS=PASS_2_OF_2"
echo "MODEL_BEST_CREATED_AND_RELOADED=PASS_2_OF_2"
echo "SHIFTED_TANH_V1_IDENTITY=PASS"
echo "CORRECTED_EV_ACTION_BOUNDS=PASS"
echo "CORRECTED_NON_EV_EXACT_ZERO=PASS"
echo "HIERARCHICAL_ACTION_CONTRACTS=PASS"
echo "PACKAGE_CREATION_AND_VALIDATION=PASS_2_OF_2"
echo "SMOKE_REDUCER=PASS"
echo "STDERR_IMPLEMENTATION_FAILURES=NONE"
echo "SCIENTIFIC_CLAIM_GENERATED=NO"

FINAL_STATUS="PASS"
BLOCK_REASON="NONE"
