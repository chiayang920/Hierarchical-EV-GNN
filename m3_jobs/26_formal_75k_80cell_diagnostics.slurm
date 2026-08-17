#!/bin/bash
#SBATCH --job-name=evgnn_formal75k_diag
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-79
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/%x_%A_%a.err
set -euo pipefail
REPO_ROOT="${EV_GNN_FORMAL_75K_REPO_ROOT:-}"
[[ -n "${REPO_ROOT}" ]] || { echo "ERROR: EV_GNN_FORMAL_75K_REPO_ROOT is required" >&2; exit 2; }
RUNTIME_LIB="${REPO_ROOT}/m3_jobs/lib_formal_75k_runtime.sh"
[[ -s "${RUNTIME_LIB}" ]] || { echo "ERROR: formal 75k runtime library is missing: ${RUNTIME_LIB}" >&2; exit 2; }
# shellcheck source=lib_formal_75k_runtime.sh
source "${REPO_ROOT}/m3_jobs/lib_formal_75k_runtime.sh"
formal75k_require_python311
OUTPUT_ROOT="${EV_GNN_FORMAL_75K_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"; RUN_ROOT="${EV_GNN_FORMAL_75K_DIAGNOSTIC_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/formal_75k_80cell_diagnostics}"; STAGE_ROOT="${EV_GNN_FORMAL_75K_STAGE_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/formal_75k_80cell_stage}"; WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/formal_75k_80cell_workflow.py"
formal75k_validate_resource_profile "${WORKFLOW_SCRIPT}" "${EV_GNN_FORMAL_75K_RESOURCE_PROFILE:-}" formal
TRAIN_JOB_ID="${EV_GNN_FORMAL_75K_ARRAY_JOB_ID:-}"; formal75k_require_numeric_id "training array job ID" "${TRAIN_JOB_ID}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"; [[ "${TASK_ID}" =~ ^[0-9]+$ ]] && ((TASK_ID<=79)) || formal75k_die "task ID must be 0-79"
MAPPING="$(${PYTHON_BIN} "${WORKFLOW_SCRIPT}" formal-task-mapping --task-id "${TASK_ID}")"; formal75k_parse_mapping "${MAPPING}"
DIAGNOSTIC_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"; formal75k_require_numeric_id "diagnostic array job ID" "${DIAGNOSTIC_JOB_ID}"
STAGED_ROOT="${STAGE_ROOT}/trainjob${TRAIN_JOB_ID}/task${TASK_ID}"; CHECKPOINT_PREFIX="${STAGED_ROOT}/checkpoint/model.best"; STAGED_META="${STAGED_ROOT}/staged_identity.json"; CONFIG_PATH="$(formal75k_require_unique_file "${STAGED_ROOT}/config" '*.yaml' 'staged configuration')"; [[ -f "${CHECKPOINT_PREFIX}_actor" && -f "${STAGED_META}" ]] || formal75k_die "validated staged model.best is missing"
readarray -t IDENTITIES < <("${PYTHON_BIN}" -c 'import json,sys,yaml; m=json.load(open(sys.argv[1])); c=yaml.safe_load(open(sys.argv[2])); print(m["source_identity"]); print(m["source_bundle_identity"]); print(m["package_sha256"]); print(m["checkpoint_identity_sha256"]); print(m["config_identity_sha256"]); print(int(c["simulation_length"]))' "${STAGED_META}" "${CONFIG_PATH}")
SOURCE_IDENTITY="${IDENTITIES[0]}"; SOURCE_BUNDLE_IDENTITY="${IDENTITIES[1]}"; TRAINING_PACKAGE_SHA="${IDENTITIES[2]}"; CHECKPOINT_SHA="${IDENTITIES[3]}"; CONFIG_SHA="${IDENTITIES[4]}"; MAX_EPISODE_STEPS="${IDENTITIES[5]}"
TASK_ROOT="${RUN_ROOT}/trainjob${TRAIN_JOB_ID}/job${DIAGNOSTIC_JOB_ID}/task${TASK_ID}"; DIAGNOSTIC_DIR="${TASK_ROOT}/diagnostics"; PACKAGE_PATH="${OUTPUT_ROOT}/m3_formal75k_diagnostics_${scale}_${algorithm}_seed${seed}_trainjob${TRAIN_JOB_ID}_job${DIAGNOSTIC_JOB_ID}_task${TASK_ID}.tar.gz"; RUN_NAME="formal75k_diag_${scale}_${algorithm}_seed${seed}_trainjob${TRAIN_JOB_ID}_job${DIAGNOSTIC_JOB_ID}_task${TASK_ID}"
case "${scale}" in 25cp) BASE=710000;;100cp) BASE=720000;;500cp) BASE=730000;;1000cp) BASE=740000;;*) formal75k_die "unsupported scale";; esac; EVAL_SEED_OFFSET=$((BASE+999*seed))
if [[ "${EV_GNN_FORMAL_75K_DIAGNOSTIC_DRY_RUN:-0}" == 1 ]]; then echo "PYTHON_VERSION=${PYTHON_VERSION}"; echo "${MAPPING}"; echo "--scale ${scale}"; echo "package_path=${PACKAGE_PATH}"; echo "DRY_RUN_NO_DIAGNOSTICS_OR_PACKAGING"; exit 0; fi
[[ ! -e "${TASK_ROOT}" && ! -e "${PACKAGE_PATH}" ]] || formal75k_die "stale diagnostic output exists"; mkdir -p "${DIAGNOSTIC_DIR}" "${TASK_ROOT}/validation" "${TASK_ROOT}/runtime_metadata" "${OUTPUT_ROOT}"; : >"${TASK_ROOT}/stdout.log"; : >"${TASK_ROOT}/stderr.log"
START_EPOCH="$(date +%s)"; set +e; (cd "${REPO_ROOT}" && "${PYTHON_BIN}" evaluate_td3_gnn_infrastructure_diagnostics.py --algorithm "${algorithm}" --scale "${scale}" --config "${CONFIG_PATH}" --seed "${seed}" --eval_episodes 30 --checkpoint "${CHECKPOINT_PREFIX}" --output_dir "${DIAGNOSTIC_DIR}" --run_name "${RUN_NAME}" --device cpu --max_episode_steps "${MAX_EPISODE_STEPS}" --deterministic true --eval_expl_noise 0.0 --eval_seed_offset "${EVAL_SEED_OFFSET}" --matrix_job_id "${DIAGNOSTIC_JOB_ID}") >>"${TASK_ROOT}/stdout.log" 2>>"${TASK_ROOT}/stderr.log"; EVAL_STATUS=$?; if [[ "${EVAL_STATUS}" -eq 0 ]]; then (cd "${REPO_ROOT}" && "${PYTHON_BIN}" scripts/formal_75k_80cell_diagnostic_validation.py --diagnostic-dir "${DIAGNOSTIC_DIR}" --validation-dir "${TASK_ROOT}/validation" --runtime-metadata-dir "${TASK_ROOT}/runtime_metadata" --task-id "${TASK_ID}" --source-identity "${SOURCE_IDENTITY}") >>"${TASK_ROOT}/stdout.log" 2>>"${TASK_ROOT}/stderr.log"; VALIDATION_STATUS=$?; else VALIDATION_STATUS=99; fi; set -e; END_EPOCH="$(date +%s)"
cat >"${TASK_ROOT}/runtime_metadata/task_runtime_metadata.env" <<EOF
protocol_version=formal_75k_diagnostic_v2
training_array_job_id=${TRAIN_JOB_ID}
diagnostic_array_job_id=${DIAGNOSTIC_JOB_ID}
task_id=${TASK_ID}
slurm_array_task_id=${TASK_ID}
scale=${scale}
algorithm=${algorithm}
seed=${seed}
source_identity=${SOURCE_IDENTITY}
source_bundle_identity=${SOURCE_BUNDLE_IDENTITY}
source_training_package_sha256=${TRAINING_PACKAGE_SHA}
checkpoint_role=model.best
checkpoint_identity_sha256=${CHECKPOINT_SHA}
config_identity_sha256=${CONFIG_SHA}
diagnostic_schema_version=3
reconciliation_contract_version=2
evaluation_episodes=30
evaluation_exit_status=${EVAL_STATUS}
validation_exit_status=${VALIDATION_STATUS}
task_elapsed_seconds=$((END_EPOCH-START_EPOCH))
requested_cpu=${SLURM_CPUS_PER_TASK:-${DIAGNOSTIC_CPUS_PER_TASK:-}}
requested_memory=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-${DIAGNOSTIC_MEM:-}}}
requested_walltime=${SLURM_TIMELIMIT:-${DIAGNOSTIC_TIME:-}}
EOF
tar -czf "${PACKAGE_PATH}" -C "${TASK_ROOT}" runtime_metadata diagnostics validation stdout.log stderr.log; tar -tzf "${PACKAGE_PATH}" >/dev/null; [[ "${EVAL_STATUS}" -eq 0 && "${VALIDATION_STATUS}" -eq 0 ]] || exit 1; echo "FORMAL_75K_DIAGNOSTIC_TASK_COMPLETED"
