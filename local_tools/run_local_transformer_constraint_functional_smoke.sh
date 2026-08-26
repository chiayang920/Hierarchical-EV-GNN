#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${EV_GNN_TRANSFORMER_CONSTRAINT_LOCAL_PYTHON:-python}"
OUTPUT_ROOT="${EV_GNN_TRANSFORMER_CONSTRAINT_LOCAL_SMOKE_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/evgnn_transformer_constraint_smoke.XXXXXX")}"
RUN_NAME="local_transformer_constraint_functional_smoke"
TRAIN_DIR="${OUTPUT_ROOT}/train"
EVAL_CSV="${OUTPUT_ROOT}/eval/eval30_smoke.csv"
PROJECTION_DIR="${OUTPUT_ROOT}/projection_diagnostics"

cd "${REPO_ROOT}"
mkdir -p "${TRAIN_DIR}" "$(dirname "${EVAL_CSV}")" "${PROJECTION_DIR}"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib"

echo "LOCAL_TRANSFORMER_CONSTRAINT_FUNCTIONAL_SMOKE=START"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
"${PYTHON_BIN}" train_td3_gnn.py \
  --algorithm hierarchical_transformer_constraint \
  --config config_files/PublicPST_25cp.yaml \
  --seed 0 \
  --device cpu \
  --run_name "${RUN_NAME}" \
  --max_timesteps 4 \
  --start_timesteps 1 \
  --eval_freq 4 \
  --eval_episodes 1 \
  --batch_size 64 \
  --replay_buffer_size 5000 \
  --save_dir "${TRAIN_DIR}" \
  --log_to_wandb false

"${PYTHON_BIN}" evaluate_td3_gnn.py \
  --algorithm hierarchical_transformer_constraint \
  --config config_files/PublicPST_25cp.yaml \
  --seed 0 \
  --eval_episodes 1 \
  --checkpoint "${TRAIN_DIR}/${RUN_NAME}/model.best" \
  --device cpu \
  --output_csv "${EVAL_CSV}" \
  --run_name "${RUN_NAME}_eval" \
  --max_episode_steps 112 \
  --deterministic true \
  --eval_expl_noise 0.0 \
  --eval_seed_offset 710000

"${PYTHON_BIN}" evaluate_transformer_constraint_projection_diagnostics.py \
  --algorithm hierarchical_transformer_constraint \
  --scale 25cp \
  --config config_files/PublicPST_25cp.yaml \
  --seed 0 \
  --eval_episodes 1 \
  --checkpoint "${TRAIN_DIR}/${RUN_NAME}/model.best" \
  --output_dir "${PROJECTION_DIR}" \
  --run_name "${RUN_NAME}_projection" \
  --device cpu \
  --max_episode_steps 112 \
  --deterministic true \
  --eval_expl_noise 0.0 \
  --eval_seed_offset 710000 \
  --matrix_job_id local

projection_rows="$("${PYTHON_BIN}" -c 'import csv,sys; print(max(0, sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="")))))' "${PROJECTION_DIR}/projection_transformer_step_rows.csv")"
[[ "${projection_rows}" -gt 0 ]] || { echo "PROJECTION_TRANSFORMER_STEP_ROWS=${projection_rows}"; echo "FUNCTIONAL_SMOKE_RESULT=FAIL"; exit 2; }
violations="$("${PYTHON_BIN}" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], newline=""))); print(sum(int(r["transformer_feasibility_violation_count"]) for r in rows))' "${PROJECTION_DIR}/projection_episode_summary.csv")"
[[ "${violations}" == "0" ]] || { echo "TRANSFORMER_FEASIBILITY_VIOLATION_COUNT=${violations}"; echo "FUNCTIONAL_SMOKE_RESULT=FAIL"; exit 2; }
echo "PROJECTION_TRANSFORMER_STEP_ROWS=${projection_rows}"
echo "TRANSFORMER_FEASIBILITY_VIOLATION_COUNT=0"
echo "FUNCTIONAL_SMOKE_RESULT=PASS"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
