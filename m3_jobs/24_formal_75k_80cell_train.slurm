#!/bin/bash
#SBATCH --job-name=evgnn_formal75k_train
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-79
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"; source "${SCRIPT_DIR}/lib_formal_75k_runtime.sh"; formal75k_require_python311
REPO_ROOT="${EV_GNN_FORMAL_75K_REPO_ROOT:-/projects/fr57/cche0357/EV-GNN}"; RUN_ROOT="${EV_GNN_FORMAL_75K_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/formal_75k_80cell}"; OUTPUT_ROOT="${EV_GNN_FORMAL_75K_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"; WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
formal75k_validate_resource_profile "${WORKFLOW_SCRIPT}" "${EV_GNN_FORMAL_75K_RESOURCE_PROFILE:-}" formal
SOURCE_IDENTITY="${EV_GNN_FORMAL_75K_SOURCE_IDENTITY:-}"; SOURCE_BUNDLE_IDENTITY="${EV_GNN_FORMAL_75K_SOURCE_BUNDLE_IDENTITY:-}"; [[ -n "${SOURCE_IDENTITY}" && -n "${SOURCE_BUNDLE_IDENTITY}" ]] || formal75k_die "source identities are required"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"; [[ "${TASK_ID}" =~ ^[0-9]+$ ]] && ((TASK_ID<=79)) || formal75k_die "task ID must be 0-79"
MAPPING="$(${PYTHON_BIN} "${WORKFLOW_SCRIPT}" formal-task-mapping --task-id "${TASK_ID}")"; formal75k_parse_mapping "${MAPPING}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"; formal75k_require_numeric_id "training array job ID" "${ARRAY_JOB_ID}"
RUN_NAME="formal75k_${scale}_${algorithm}_seed${seed}"; TASK_ROOT="${RUN_ROOT}/job${ARRAY_JOB_ID}/${scale}/${algorithm}/seed${seed}"; TRAIN_DIR="${TASK_ROOT}/train"; CONFIG_REL="config/${scale}_${algorithm}_seed${seed}_config.yaml"; PACKAGE_PATH="${OUTPUT_ROOT}/m3_formal75k_80cell_${scale}_${algorithm}_seed${seed}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
if [[ "${EV_GNN_FORMAL_75K_TRAIN_DRY_RUN:-0}" == 1 ]]; then echo "PYTHON_VERSION=${PYTHON_VERSION}"; echo "${MAPPING}"; echo "package_path=${PACKAGE_PATH}"; echo "DRY_RUN_NO_TRAINING_OR_PACKAGING"; exit 0; fi
[[ ! -e "${TASK_ROOT}" && ! -e "${PACKAGE_PATH}" ]] || formal75k_die "stale training output exists"
mkdir -p "${TRAIN_DIR}" "${TASK_ROOT}/config" "${TASK_ROOT}/runtime_metadata" "${OUTPUT_ROOT}"; cp "${REPO_ROOT}/${config}" "${TASK_ROOT}/${CONFIG_REL}"; : >"${TASK_ROOT}/stdout.log"; : >"${TASK_ROOT}/stderr.log"
START_EPOCH="$(date +%s)"; set +e; (cd "${REPO_ROOT}" && "${PYTHON_BIN}" train_td3_gnn.py --algorithm "${algorithm}" --config "${config}" --seed "${seed}" --device cpu --run_name "${RUN_NAME}" --max_timesteps 75000 --start_timesteps 1000 --eval_freq 5000 --eval_episodes 5 --batch_size 64 --replay_buffer_size 100000 --save_dir "${TRAIN_DIR}" --log_to_wandb false) >>"${TASK_ROOT}/stdout.log" 2>>"${TASK_ROOT}/stderr.log"; TRAIN_STATUS=$?; set -e; END_EPOCH="$(date +%s)"
ACTOR_TRANSFORM="hierarchical_nonnegative_action_contract_v1"; [[ "${algorithm}" == actiongnn_nonnegative ]] && ACTOR_TRANSFORM="shifted_tanh_v1"
cat >"${TASK_ROOT}/runtime_metadata/task_runtime_metadata.env" <<EOF
protocol_version=formal_75k_80cell_v2
task_id=${TASK_ID}
slurm_array_task_id=${TASK_ID}
slurm_array_job_id=${ARRAY_JOB_ID}
scale=${scale}
algorithm=${algorithm}
seed=${seed}
training_steps=75000
evaluation_cadence=5000
evaluation_episodes=5
expected_scheduled_evaluations=15
actor_output_transform=${ACTOR_TRANSFORM}
checkpoint_selection_rule=model.best selected by strict improvement of scheduled internal eval mean reward at 5k-step intervals within the configured training budget; model.last is saved at the configured final step but is not used for canonical eval30 or diagnostics.
source_identity=${SOURCE_IDENTITY}
source_bundle_identity=${SOURCE_BUNDLE_IDENTITY}
run_name=${RUN_NAME}
model_dir_relative=train/${RUN_NAME}
config_copy_relative=${CONFIG_REL}
fresh_run=true
training_exit_status=${TRAIN_STATUS}
task_elapsed_seconds=$((END_EPOCH-START_EPOCH))
requested_cpu=${SLURM_CPUS_PER_TASK:-${TRAIN_CPUS_PER_TASK:-}}
requested_memory=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-${TRAIN_MEM:-}}}
requested_walltime=${SLURM_TIMELIMIT:-${TRAIN_TIME:-}}
EOF
tar -czf "${PACKAGE_PATH}" -C "${TASK_ROOT}" runtime_metadata config train stdout.log stderr.log; tar -tzf "${PACKAGE_PATH}" >/dev/null; [[ "${TRAIN_STATUS}" -eq 0 ]] || exit 1; echo "FORMAL_75K_TRAIN_TASK_COMPLETED"
