#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${EV_GNN_TRANSFORMER_CONSTRAINT_LOCAL_PYTHON:-python}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/transformer_constraint_40cell_workflow.py"
OUTPUT_ROOT="${EV_GNN_TRANSFORMER_CONSTRAINT_RUNTIME_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/evgnn_transformer_constraint_runtime.XXXXXX")}"
EXECUTE="${EV_GNN_TRANSFORMER_CONSTRAINT_RUNTIME_EXECUTE:-0}"
ALGORITHM="hierarchical_transformer_constraint"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib"

cd "${REPO_ROOT}"
echo "LOCAL_TRANSFORMER_CONSTRAINT_RUNTIME_GATE=START"
echo "ALGORITHM=${ALGORITHM}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "FORMAL_BATCH_SIZE=64"

for task_id in 0 1 2 3; do
  mapping="$("${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" smoke-task-mapping --task-id "${task_id}")"
  scale="$(awk '{for (i=1;i<=NF;i++) if ($i ~ /^scale=/) {split($i,a,"="); print a[2]}}' <<<"${mapping}")"
  run_name="local_transformer_constraint_runtime_${scale}"
  command_text="$("${PYTHON_BIN}" "${WORKFLOW_SCRIPT}" runtime-gate-command --task-id "${task_id}" --save-dir "${OUTPUT_ROOT}/task${task_id}" --run-name "${run_name}")"
  run_command_text="${command_text/#python /${PYTHON_BIN} }"
  echo "RUNTIME_GATE_COMMAND_TASK_${task_id}=${command_text}"
  if [[ "${EXECUTE}" == "1" ]]; then
    start_epoch="$(date +%s)"
    eval "${run_command_text}"
    end_epoch="$(date +%s)"
    wall_clock_seconds=$((end_epoch-start_epoch))
    [[ "${wall_clock_seconds}" -gt 0 ]] || wall_clock_seconds=1
    echo "scale=${scale} measured_training_steps=4 wall_clock_seconds=${wall_clock_seconds} steps_per_second=$("${PYTHON_BIN}" -c "print(4/${wall_clock_seconds})") peak_memory_if_reliably_available="
  fi
done

if [[ "${EXECUTE}" == "1" ]]; then
  echo "RUNTIME_GATE_FUNCTIONAL_RESULT=PASS"
else
  echo "RUNTIME_GATE_DRY_RUN_RESULT=PASS"
fi
