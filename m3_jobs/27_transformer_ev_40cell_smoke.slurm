#!/bin/bash
#SBATCH --job-name=evgnn_transformer_ev_smoke
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-3
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err
set -euo pipefail
REPO_ROOT="${EV_GNN_TRANSFORMER_EV_REPO_ROOT:-}"
[[ -n "${REPO_ROOT}" ]] || { echo "ERROR: EV_GNN_TRANSFORMER_EV_REPO_ROOT is required" >&2; exit 2; }
RUNTIME_LIB="${REPO_ROOT}/m3_jobs/lib_transformer_ev_runtime.sh"
[[ -s "${RUNTIME_LIB}" ]] || { echo "ERROR: transformer-EV runtime library is missing: ${RUNTIME_LIB}" >&2; exit 2; }
# shellcheck source=lib_transformer_ev_runtime.sh
source "${RUNTIME_LIB}"
transformer_ev_require_python311
RUN_ROOT="${EV_GNN_TRANSFORMER_EV_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/transformer_ev_40cell_smoke}"
OUTPUT_ROOT="${EV_GNN_TRANSFORMER_EV_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs/transformer_ev_40cell}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/transformer_ev_40cell_workflow.py"
PROFILE="${EV_GNN_TRANSFORMER_EV_RESOURCE_PROFILE:-}"
SOURCE_IDENTITY="${EV_GNN_TRANSFORMER_EV_SOURCE_IDENTITY:-}"
SOURCE_BUNDLE_IDENTITY="${EV_GNN_TRANSFORMER_EV_SOURCE_BUNDLE_IDENTITY:-}"
transformer_ev_validate_resource_profile "${WORKFLOW_SCRIPT}" "${PROFILE}" smoke
[[ -n "${SOURCE_IDENTITY}" && -n "${SOURCE_BUNDLE_IDENTITY}" ]] || transformer_ev_die "source and source-bundle identities are required"
transformer_ev_require_source_identity "${REPO_ROOT}" "${SOURCE_IDENTITY}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
[[ "${TASK_ID}" =~ ^[0-3]$ ]] || transformer_ev_die "SLURM_ARRAY_TASK_ID must be 0..3"
MAPPING="$(${PYTHON_BIN} "${WORKFLOW_SCRIPT}" smoke-task-mapping --task-id "${TASK_ID}")"
transformer_ev_parse_mapping "${MAPPING}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
transformer_ev_require_numeric_id "smoke array job ID" "${ARRAY_JOB_ID}"
RUN_NAME="transformer_ev_smoke_${scale}_seed${seed}_job${ARRAY_JOB_ID}_task${TASK_ID}"
TASK_ROOT="${RUN_ROOT}/job${ARRAY_JOB_ID}/task${TASK_ID}"
TRAIN_DIR="${TASK_ROOT}/train"
MODEL_DIR="${TRAIN_DIR}/${RUN_NAME}"
CONFIG_REL="config/${scale}_${algorithm}_seed${seed}_config.yaml"
PACKAGE_PATH="${OUTPUT_ROOT}/m3_transformer_ev_smoke_${scale}_${algorithm}_seed${seed}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
if [[ "${EV_GNN_TRANSFORMER_EV_SMOKE_DRY_RUN:-0}" == 1 ]]; then
  echo "PYTHON_VERSION=${PYTHON_VERSION}"; echo "${MAPPING}"; echo "package_path=${PACKAGE_PATH}"; echo "DRY_RUN_NO_TRAINING_EVALUATION_OR_PACKAGING"; exit 0
fi
[[ ! -e "${TASK_ROOT}" && ! -e "${PACKAGE_PATH}" ]] || transformer_ev_die "stale smoke output exists"
mkdir -p "${TRAIN_DIR}" "${TASK_ROOT}/config" "${TASK_ROOT}/runtime_metadata" "${TASK_ROOT}/smoke_eval" "${OUTPUT_ROOT}"
cp "${REPO_ROOT}/${config}" "${TASK_ROOT}/${CONFIG_REL}"
START_EPOCH="$(date +%s)"
: >"${TASK_ROOT}/stdout.log"; : >"${TASK_ROOT}/stderr.log"
set +e
(
  cd "${REPO_ROOT}" && "${PYTHON_BIN}" train_td3_gnn.py --algorithm "${algorithm}" --config "${config}" --seed "${seed}" --device cpu --run_name "${RUN_NAME}" --max_timesteps 512 --start_timesteps 64 --eval_freq 256 --eval_episodes 1 --batch_size 32 --replay_buffer_size 5000 --save_dir "${TRAIN_DIR}" --log_to_wandb false
) >>"${TASK_ROOT}/stdout.log" 2>>"${TASK_ROOT}/stderr.log"
TRAIN_STATUS=$?
if [[ "${TRAIN_STATUS}" -eq 0 ]]; then
  (
    cd "${REPO_ROOT}" && "${PYTHON_BIN}" evaluate_td3_gnn.py --algorithm "${algorithm}" --config "${TASK_ROOT}/${CONFIG_REL}" --seed "${seed}" --eval_episodes 3 --checkpoint "${MODEL_DIR}/model.best" --device cpu --output_csv "${TASK_ROOT}/smoke_eval/${scale}_${algorithm}_seed${seed}_smoke_eval3.csv" --run_name "${RUN_NAME}_eval3" --max_episode_steps 112 --deterministic true --eval_expl_noise 0.0 --eval_seed_offset 710000
  ) >>"${TASK_ROOT}/stdout.log" 2>>"${TASK_ROOT}/stderr.log"
  EVAL_STATUS=$?
else EVAL_STATUS=99; fi
set -e
END_EPOCH="$(date +%s)"
cat >"${TASK_ROOT}/runtime_metadata/task_runtime_metadata.env" <<EOF
protocol_version=transformer_ev_40cell_smoke_v1
task_id=${TASK_ID}
slurm_array_task_id=${TASK_ID}
slurm_array_job_id=${ARRAY_JOB_ID}
scale=${scale}
algorithm=${algorithm}
seed=${seed}
training_steps=512
evaluation_cadence=256
evaluation_episodes=1
expected_scheduled_evaluations=2
actor_output_transform=transformer_ev_hierarchy_v1
checkpoint_selection_rule=model.best selected by strict improvement of scheduled internal eval mean reward at intervals recorded in eval_frequency within the configured training budget; model.last is saved at the configured final step but is not used for canonical eval30 or diagnostics.
source_identity=${SOURCE_IDENTITY}
source_bundle_identity=${SOURCE_BUNDLE_IDENTITY}
run_name=${RUN_NAME}
model_dir_relative=train/${RUN_NAME}
config_copy_relative=${CONFIG_REL}
fresh_run=true
training_exit_status=${TRAIN_STATUS}
reload_eval_exit_status=${EVAL_STATUS}
task_elapsed_seconds=$((END_EPOCH-START_EPOCH))
requested_cpu=${SLURM_CPUS_PER_TASK:-${SMOKE_CPUS_PER_TASK:-}}
requested_memory=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-${SMOKE_MEM:-}}}
requested_walltime=${SLURM_TIMELIMIT:-${SMOKE_TIME:-}}
EOF
COPYFILE_DISABLE=1 tar -czf "${PACKAGE_PATH}" -C "${TASK_ROOT}" runtime_metadata config train smoke_eval stdout.log stderr.log
tar -tzf "${PACKAGE_PATH}" >/dev/null
[[ "${TRAIN_STATUS}" -eq 0 && "${EVAL_STATUS}" -eq 0 ]] || exit 1
echo "TRANSFORMER_EV_SMOKE_TASK_COMPLETED"
