#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${EV_GNN_TRANSFORMER_CONSTRAINT_LOCAL_PYTHON:-python}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/transformer_constraint_40cell_workflow.py"
GATE_SCRIPT="${REPO_ROOT}/scripts/transformer_constraint_comparator_gate.py"

cd "${REPO_ROOT}"

echo "LOCAL_TRANSFORMER_CONSTRAINT_40CELL_DRY_RUN=START"
"${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" --print-matrix >/tmp/evgnn_transformer_constraint_matrix.$$.txt
matrix_count="$(wc -l </tmp/evgnn_transformer_constraint_matrix.$$.txt | tr -d '[:space:]')"
rm -f /tmp/evgnn_transformer_constraint_matrix.$$.txt
summary="$("${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" dry-run-summary)"
printf '%s\n' "${summary}"

[[ "${matrix_count}" == "40" ]] || { echo "DRY_RUN_RESULT=FAIL"; exit 2; }
grep -q "FORMAL_MATRIX_CELL_COUNT=40" <<<"${summary}" || { echo "DRY_RUN_RESULT=FAIL"; exit 2; }
grep -q "FORMAL_MATRIX_UNIQUE_CELL_COUNT=40" <<<"${summary}" || { echo "DRY_RUN_RESULT=FAIL"; exit 2; }
grep -q "FORMAL_ALGORITHM_SET=hierarchical_transformer_constraint" <<<"${summary}" || { echo "DRY_RUN_RESULT=FAIL"; exit 2; }
grep -q "HISTORICAL_FORMAL75K_OUTPUT_WRITE_ALLOWED=NO" <<<"${summary}" || { echo "DRY_RUN_RESULT=FAIL"; exit 2; }
grep -q "DRY_RUN_NO_JOBS_SUBMITTED=YES" <<<"${summary}" || { echo "DRY_RUN_RESULT=FAIL"; exit 2; }

for config in \
  config_files/PublicPST_25cp.yaml \
  config_files/PublicPST_100.yaml \
  config_files/PublicPST_500.yaml \
  config_files/PublicPST_1000.yaml
do
  [[ -s "${config}" ]] || { echo "MISSING_CONFIG=${config}"; echo "DRY_RUN_RESULT=FAIL"; exit 2; }
done

if [[ "${EV_GNN_TRANSFORMER_CONSTRAINT_SKIP_COMPARATOR_GATE:-0}" != "1" ]]; then
  "${PYTHON_BIN}" "${GATE_SCRIPT}"
fi

echo "MATRIX_RESULT=PASS"
echo "PATH_SAFETY_RESULT=PASS"
echo "JOB_SUBMISSION_RESULT=PASS"
echo "DRY_RUN_RESULT=PASS"
