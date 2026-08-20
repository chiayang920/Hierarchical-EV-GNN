#!/bin/bash
#SBATCH --job-name=evgnn_transformer_ev_eval30
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --array=0-39
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
OUTPUT_ROOT="${EV_GNN_TRANSFORMER_EV_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs/transformer_ev_40cell}"
RUN_ROOT="${EV_GNN_TRANSFORMER_EV_EVAL_RUN_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/transformer_ev_40cell_eval30}"
STAGE_ROOT="${EV_GNN_TRANSFORMER_EV_STAGE_ROOT:-/scratch2/fr57/cche0357/EV-GNN_runs/transformer_ev_40cell_stage}"
WORKFLOW_SCRIPT="${REPO_ROOT}/scripts/transformer_ev_40cell_workflow.py"
transformer_ev_validate_resource_profile "${WORKFLOW_SCRIPT}" "${EV_GNN_TRANSFORMER_EV_RESOURCE_PROFILE:-}" eval
TRAIN_JOB_ID="${EV_GNN_TRANSFORMER_EV_ARRAY_JOB_ID:-}"
TRAINING_SET_ID="${EV_GNN_TRANSFORMER_EV_TRAINING_SET_ID:-}"
if [[ -n "${TRAINING_SET_ID}" ]]; then
  transformer_ev_require_safe_id "training set ID" "${TRAINING_SET_ID}"
  [[ -z "${TRAIN_JOB_ID}" || "${TRAIN_JOB_ID}" =~ ^[0-9]+$ ]] || transformer_ev_die "training array job ID must contain decimal digits only"
else
  transformer_ev_require_numeric_id "training array job ID" "${TRAIN_JOB_ID}"
  TRAINING_SET_ID="trainjob${TRAIN_JOB_ID}"
fi
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
[[ "${TASK_ID}" =~ ^[0-9]+$ ]] && ((TASK_ID<=39)) || transformer_ev_die "task ID must be 0-39"
MAPPING="$(${PYTHON_BIN} "${WORKFLOW_SCRIPT}" formal-task-mapping --task-id "${TASK_ID}")"
transformer_ev_parse_mapping "${MAPPING}"
EVAL_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
transformer_ev_require_numeric_id "eval array job ID" "${EVAL_JOB_ID}"
STAGED_ROOT="${STAGE_ROOT}/${TRAINING_SET_ID}/task${TASK_ID}"
CHECKPOINT_PREFIX="${STAGED_ROOT}/checkpoint/model.best"
STAGED_META="${STAGED_ROOT}/staged_identity.json"
CONFIG_PATH="$(transformer_ev_require_unique_file "${STAGED_ROOT}/config" '*.yaml' 'staged configuration')"
[[ -f "${CHECKPOINT_PREFIX}_actor" && -f "${STAGED_META}" ]] || transformer_ev_die "validated staged model.best is missing"
readarray -t IDENTITIES < <("${PYTHON_BIN}" -c 'import hashlib,json,sys,yaml
from pathlib import Path
meta_path, config_path, checkpoint_prefix = map(Path, sys.argv[1:4])
m=json.load(open(meta_path))
c=yaml.safe_load(open(config_path))
checkpoint_members=[
    checkpoint_prefix.with_name("model.best_actor"),
    checkpoint_prefix.with_name("model.best_actor_optimizer"),
    checkpoint_prefix.with_name("model.best_critic"),
    checkpoint_prefix.with_name("model.best_critic_optimizer"),
    checkpoint_prefix.with_name("kwargs.yaml"),
    checkpoint_prefix.with_name("model.best.metadata.yaml"),
]
checkpoint_digest=hashlib.sha256(b"\0".join(path.read_bytes() for path in checkpoint_members)).hexdigest()
config_digest=hashlib.sha256(config_path.read_bytes()).hexdigest()
if checkpoint_digest != m["checkpoint_identity_sha256"]:
    raise SystemExit("checkpoint_identity_sha256 mismatch")
if config_digest != m["config_identity_sha256"]:
    raise SystemExit("config_identity_sha256 mismatch")
print(m["source_identity"])
print(m["source_bundle_identity"])
print(m["package_sha256"])
print(checkpoint_digest)
print(config_digest)
print(int(c["simulation_length"]))' "${STAGED_META}" "${CONFIG_PATH}" "${CHECKPOINT_PREFIX}")
SOURCE_IDENTITY="${IDENTITIES[0]}"; SOURCE_BUNDLE_IDENTITY="${IDENTITIES[1]}"; TRAINING_PACKAGE_SHA="${IDENTITIES[2]}"; CHECKPOINT_SHA="${IDENTITIES[3]}"; CONFIG_SHA="${IDENTITIES[4]}"; MAX_EPISODE_STEPS="${IDENTITIES[5]}"
transformer_ev_require_source_identity "${REPO_ROOT}" "${SOURCE_IDENTITY}"
TASK_ROOT="${RUN_ROOT}/${TRAINING_SET_ID}/job${EVAL_JOB_ID}/task${TASK_ID}"
OUTPUT_CSV="${TASK_ROOT}/eval/${scale}_${algorithm}_seed${seed}_eval30.csv"
PACKAGE_PATH="${OUTPUT_ROOT}/m3_transformer_ev_eval30_${scale}_${algorithm}_seed${seed}_${TRAINING_SET_ID}_job${EVAL_JOB_ID}_task${TASK_ID}.tar.gz"
RUN_NAME="transformer_ev_eval30_${scale}_${algorithm}_seed${seed}_${TRAINING_SET_ID}_job${EVAL_JOB_ID}_task${TASK_ID}"
case "${scale}" in 25cp) BASE=710000;;100cp) BASE=720000;;500cp) BASE=730000;;1000cp) BASE=740000;;*) transformer_ev_die "unsupported scale";; esac
EVAL_SEED_OFFSET=$((BASE+999*seed))
if [[ "${EV_GNN_TRANSFORMER_EV_EVAL_DRY_RUN:-0}" == 1 ]]; then
  echo "PYTHON_VERSION=${PYTHON_VERSION}"; echo "${MAPPING}"; echo "checkpoint_prefix=${CHECKPOINT_PREFIX}"; echo "package_path=${PACKAGE_PATH}"; echo "DRY_RUN_NO_EVALUATION_OR_PACKAGING"; exit 0
fi
[[ ! -e "${TASK_ROOT}" && ! -e "${PACKAGE_PATH}" ]] || transformer_ev_die "stale eval output exists"
mkdir -p "${TASK_ROOT}/eval" "${TASK_ROOT}/runtime_metadata" "${OUTPUT_ROOT}"
: >"${TASK_ROOT}/stdout.log"; : >"${TASK_ROOT}/stderr.log"
START_EPOCH="$(date +%s)"
set +e
(
  cd "${REPO_ROOT}" && "${PYTHON_BIN}" evaluate_td3_gnn.py --algorithm "${algorithm}" --config "${CONFIG_PATH}" --seed "${seed}" --eval_episodes 30 --checkpoint "${CHECKPOINT_PREFIX}" --device cpu --output_csv "${OUTPUT_CSV}" --run_name "${RUN_NAME}" --max_episode_steps "${MAX_EPISODE_STEPS}" --deterministic true --eval_expl_noise 0.0 --eval_seed_offset "${EVAL_SEED_OFFSET}"
) >>"${TASK_ROOT}/stdout.log" 2>>"${TASK_ROOT}/stderr.log"
STATUS=$?
set -e
END_EPOCH="$(date +%s)"
cat >"${TASK_ROOT}/runtime_metadata/task_runtime_metadata.env" <<EOF
protocol_version=transformer_ev_40cell_eval30_v1
training_array_job_id=${TRAIN_JOB_ID}
training_set_id=${TRAINING_SET_ID}
eval_array_job_id=${EVAL_JOB_ID}
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
evaluation_episodes=30
evaluation_exit_status=${STATUS}
task_elapsed_seconds=$((END_EPOCH-START_EPOCH))
requested_cpu=${SLURM_CPUS_PER_TASK:-${EVAL_CPUS_PER_TASK:-}}
requested_memory=${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-${EVAL_MEM:-}}}
requested_walltime=${SLURM_TIMELIMIT:-${EVAL_TIME:-}}
EOF
COPYFILE_DISABLE=1 tar -czf "${PACKAGE_PATH}" -C "${TASK_ROOT}" runtime_metadata eval stdout.log stderr.log
tar -tzf "${PACKAGE_PATH}" >/dev/null
[[ "${STATUS}" -eq 0 ]] || exit 1
echo "TRANSFORMER_EV_EVAL30_TASK_COMPLETED"
