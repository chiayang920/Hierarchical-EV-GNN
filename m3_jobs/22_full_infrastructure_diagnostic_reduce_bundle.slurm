#!/bin/bash
#SBATCH --job-name=evgnn_infra_eval30_reduce
#SBATCH --partition=comp
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/fr57/cche0357/EV-GNN_outputs/full_infra_eval30_reduce_%j.out
#SBATCH --error=/projects/fr57/cche0357/EV-GNN_outputs/full_infra_eval30_reduce_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="${EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
EXPECTED_SOURCE_COMMIT="${EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT:-}"
ARRAY_JOB_ID="${EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID:-}"
TASK_PACKAGE_ROOT="${EV_GNN_FULL_DIAGNOSTIC_TASK_PACKAGE_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
OUTPUT_ROOT="${EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT:-/projects/fr57/cche0357/EV-GNN_outputs}"
SOURCE_COMMIT_FILE="${REPO_ROOT}/SOURCE_COMMIT_SHA.txt"
ACCOUNTING_MODULE="${REPO_ROOT}/scripts/full_infrastructure_diagnostic_eval30_accounting.py"
PACKAGE_VALIDATOR="${REPO_ROOT}/scripts/full_infrastructure_diagnostic_eval30_packages.py"
SACCT_FIELDS="JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU"

: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
REDUCER_JOB_ID="${SLURM_JOB_ID}"

if ! [[ "${ARRAY_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID must be numeric" >&2
  exit 2
fi
if ! [[ "${REDUCER_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SLURM_JOB_ID must be numeric" >&2
  exit 2
fi

FINAL_BASENAME="full_infrastructure_diagnostics_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"
FINAL_BUNDLE="${OUTPUT_ROOT}/${FINAL_BASENAME}"
FINAL_SIDECAR="${FINAL_BUNDLE}.sha256"
TEMP_BUNDLE="${OUTPUT_ROOT}/.${FINAL_BASENAME}.tmp.${SLURM_JOB_ID}"
TEMP_SIDECAR="${TEMP_BUNDLE}.sha256"
REDUCER_ROOT="${EV_GNN_FULL_DIAGNOSTIC_REDUCER_ROOT:-${TMPDIR:-/tmp}/evgnn_full_infra_eval30_reduce_job${REDUCER_JOB_ID}}"
STAGING_ROOT="${REDUCER_ROOT}/complete_staging"
SACCT_RAW="${REDUCER_ROOT}/sacct_raw.txt"

if [[ "${EV_GNN_FULL_DIAGNOSTIC_REDUCER_DRY_RUN:-0}" == "1" ]]; then
  echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_REDUCER_DRY_RUN"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "reducer_job_id=${REDUCER_JOB_ID}"
  echo "task_package_root=${TASK_PACKAGE_ROOT}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "expected_task_packages=8"
  echo "expected_checkpoint_groups=40"
  echo "expected_episode_count=1200"
  for task_id in 0 1 2 3 4 5 6 7; do
    echo "task_package_${task_id}=${TASK_PACKAGE_ROOT}/full_infrastructure_diagnostic_eval30_task${task_id}_job${ARRAY_JOB_ID}.tar.gz"
  done
  echo "SACCT_COMMAND=sacct -j ${ARRAY_JOB_ID} --parsable2 --noheader --format=${SACCT_FIELDS}"
  echo "final_bundle=${FINAL_BUNDLE}"
  echo "final_bundle_sha256=${FINAL_SIDECAR}"
  echo "DRY_RUN_NO_ACCOUNTING_OR_PACKAGING"
  exit 0
fi

require_regular_file() {
  local path="$1"
  if [[ ! -f "${path}" ]] || [[ -L "${path}" ]] || [[ ! -s "${path}" ]]; then
    echo "ERROR: required non-empty regular file missing: ${path}" >&2
    exit 3
  fi
}

if [[ -z "${EXPECTED_SOURCE_COMMIT}" ]] || ! [[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT must be a 40-character lowercase SHA" >&2
  exit 2
fi
require_regular_file "${SOURCE_COMMIT_FILE}"
require_regular_file "${ACCOUNTING_MODULE}"
require_regular_file "${PACKAGE_VALIDATOR}"

SOURCE_COMMIT_SHA="$(tr -d '[:space:]' < "${SOURCE_COMMIT_FILE}")"
if [[ "${SOURCE_COMMIT_SHA}" != "${EXPECTED_SOURCE_COMMIT}" ]]; then
  echo "ERROR: source commit mismatch: ${SOURCE_COMMIT_SHA} != ${EXPECTED_SOURCE_COMMIT}" >&2
  exit 3
fi
if [[ -e "${FINAL_BUNDLE}" ]] || [[ -e "${FINAL_SIDECAR}" ]]; then
  echo "ERROR: refusing to overwrite existing complete evidence: ${FINAL_BUNDLE}" >&2
  exit 3
fi
if [[ -e "${REDUCER_ROOT}" ]]; then
  echo "ERROR: refusing stale reducer root: ${REDUCER_ROOT}" >&2
  exit 3
fi

mkdir -p "${REDUCER_ROOT}" "${STAGING_ROOT}" "${OUTPUT_ROOT}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu
cd "${REPO_ROOT}"

ACCOUNTING_OK=0
for attempt in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sacct -j "${ARRAY_JOB_ID}" \
    --parsable2 \
    --noheader \
    --format="${SACCT_FIELDS}" \
    > "${SACCT_RAW}.tmp"
  mv "${SACCT_RAW}.tmp" "${SACCT_RAW}"
  if python - "${SACCT_RAW}" "${ARRAY_JOB_ID}" <<'PY'
import sys
from pathlib import Path
from scripts.full_infrastructure_diagnostic_eval30_accounting import (
    parse_stage_d_sacct,
)
parse_stage_d_sacct(Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2])
PY
  then
    ACCOUNTING_OK=1
    break
  fi
  if [[ "${attempt}" -lt 12 ]]; then
    sleep 10
  fi
done
if [[ "${ACCOUNTING_OK}" != "1" ]]; then
  echo "ERROR: complete exact accounting was not visible for array ${ARRAY_JOB_ID}" >&2
  exit 4
fi

TASK_PACKAGES=()
for task_id in 0 1 2 3 4 5 6 7; do
  package="${TASK_PACKAGE_ROOT}/full_infrastructure_diagnostic_eval30_task${task_id}_job${ARRAY_JOB_ID}.tar.gz"
  sidecar="${package}.sha256"
  require_regular_file "${package}"
  require_regular_file "${sidecar}"
  (
    cd "${TASK_PACKAGE_ROOT}"
    sha256sum -c "$(basename "${sidecar}")"
  )
  python - "${package}" "${task_id}" "${ARRAY_JOB_ID}" "${SOURCE_COMMIT_SHA}" <<'PY'
import sys
from pathlib import Path
from scripts.full_infrastructure_diagnostic_eval30_packages import (
    validate_stage_d_task_package,
)
result = validate_stage_d_task_package(Path(sys.argv[1]), int(sys.argv[2]))
if result["array_job_id"] != sys.argv[3]:
    raise SystemExit("task package array job ID mismatch")
if result["source_commit_sha"] != sys.argv[4]:
    raise SystemExit("task package source commit mismatch")
PY
  TASK_PACKAGES+=("${package}")
done

python - \
  "${STAGING_ROOT}" "${SOURCE_COMMIT_SHA}" "${ARRAY_JOB_ID}" \
  "${REDUCER_JOB_ID}" "${SACCT_RAW}" "${TASK_PACKAGES[@]}" <<'PY'
import csv
import hashlib
import io
import json
import statistics
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

from scripts.full_infrastructure_diagnostic_eval30_accounting import (
    parse_stage_d_sacct,
)
from scripts.full_infrastructure_diagnostic_eval30_packages import (
    validate_safe_tar_members,
    validate_stage_d_task_package,
)
from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    SCHEMA_VERSION,
    TRAINING_SEEDS,
    validate_episode_inventory,
)

root = Path(sys.argv[1])
source_sha = sys.argv[2]
array_job_id = sys.argv[3]
reducer_job_id = sys.argv[4]
sacct_raw_path = Path(sys.argv[5])
package_paths = [Path(value) for value in sys.argv[6:]]
if len(package_paths) != 8:
    raise SystemExit(f"expected 8 task package paths, got {len(package_paths)}")

for directory in ("task_packages", "summaries", "validation", "provenance", "checksums"):
    (root / directory).mkdir(parents=True, exist_ok=True)


def read_tar_payloads(path):
    with tarfile.open(path, "r:gz") as tar:
        validate_safe_tar_members(tar.getmembers())
        payloads = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise SystemExit(f"unable to read {path}:{member.name}")
            payloads[member.name] = handle.read()
    return payloads


def read_csv_bytes(payload, label):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if reader.fieldnames is None:
        raise SystemExit(f"{label} has no header")
    return list(reader.fieldnames), list(reader)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


task_inventory = []
checkpoint_rows = []
episode_inventory_rows = []
canonical_rows = []
mapping_rows = []
service_rows = []
metric_rows = defaultdict(list)
task_ids = set()

for package_path in package_paths:
    payload = package_path.read_bytes()
    package_sha = hashlib.sha256(payload).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        validate_safe_tar_members(tar.getmembers())
        task_env_member = tar.extractfile("task_metadata/task.env")
        if task_env_member is None:
            raise SystemExit(f"missing task metadata in {package_path}")
        task_env = {}
        for line in task_env_member.read().decode("utf-8").splitlines():
            key, value = line.split("=", 1)
            task_env[key] = value
        task_id = int(task_env["task_id"])
    if task_id in task_ids:
        raise SystemExit(f"duplicate task package: {task_id}")
    task_ids.add(task_id)
    result = validate_stage_d_task_package(package_path, task_id)
    if result["array_job_id"] != array_job_id or result["source_commit_sha"] != source_sha:
        raise SystemExit(f"task package provenance mismatch: {package_path}")
    destination = root / "task_packages" / package_path.name
    destination.write_bytes(payload)
    task_inventory.append(
        {
            "task_id": str(task_id),
            "package_name": package_path.name,
            "package_sha256": package_sha,
        }
    )

    members = read_tar_payloads(package_path)
    _, task_checkpoints = read_csv_bytes(
        members["summaries/checkpoint_inventory.csv"],
        f"task {task_id} checkpoint inventory",
    )
    checkpoint_rows.extend(task_checkpoints)
    _, task_episodes = read_csv_bytes(
        members["summaries/episode_inventory.csv"],
        f"task {task_id} episode inventory",
    )
    episode_inventory_rows.extend(task_episodes)

    scale = result["scale"]
    algorithm = result["algorithm"]
    for training_seed in TRAINING_SEEDS:
        prefix = f"seed{training_seed}"
        _, seed_canonical = read_csv_bytes(
            members[f"{prefix}/validation/canonical_reconciliation.csv"],
            f"task {task_id} seed {training_seed} canonical reconciliation",
        )
        for row in seed_canonical:
            canonical_rows.append(
                {
                    "task_id": str(task_id),
                    "scale": scale,
                    "algorithm": algorithm,
                    "training_seed": str(training_seed),
                    **row,
                }
            )
        mapping = json.loads(
            members[f"{prefix}/validation/mapping_validation.json"].decode("utf-8")
        )
        if mapping.get("status") != "ok":
            raise SystemExit(f"mapping validation failed: task {task_id} seed {training_seed}")
        mapping_rows.append(
            {
                "task_id": str(task_id),
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": str(training_seed),
                "status": "pass",
                "episodes": str(mapping.get("episodes", "")),
                "diagnostic_schema_version": str(
                    mapping.get("diagnostic_schema_version", "")
                ),
            }
        )
        _, seed_service = read_csv_bytes(
            members[f"{prefix}/validation/service_reconciliation.csv"],
            f"task {task_id} seed {training_seed} service reconciliation",
        )
        for row in seed_service:
            statuses = [
                row.get("served_count_reconciliation_status"),
                row.get("satisfaction_sum_reconciliation_status"),
                row.get("charged_energy_reconciliation_status"),
                row.get("discharged_energy_reconciliation_status"),
            ]
            if any(status != "pass" for status in statuses):
                raise SystemExit(
                    f"service reconciliation failed: task {task_id} seed {training_seed}"
                )
            service_rows.append(
                {
                    "task_id": str(task_id),
                    "scale": scale,
                    "algorithm": algorithm,
                    "training_seed": str(training_seed),
                    "status": "pass",
                    **row,
                }
            )
        _, diagnostics = read_csv_bytes(
            members[f"{prefix}/diagnostics/episode_diagnostics.csv"],
            f"task {task_id} seed {training_seed} episode diagnostics",
        )
        metric_rows[(scale, algorithm, training_seed)].extend(diagnostics)

if task_ids != set(range(8)):
    raise SystemExit(f"task IDs must equal 0..7, got {sorted(task_ids)}")
if len(checkpoint_rows) != 40:
    raise SystemExit(f"expected 40 checkpoint rows, got {len(checkpoint_rows)}")
if len(episode_inventory_rows) != 1200:
    raise SystemExit(f"expected 1,200 episode rows, got {len(episode_inventory_rows)}")
validate_episode_inventory(episode_inventory_rows)

runtime_rows = parse_stage_d_sacct(
    sacct_raw_path.read_text(encoding="utf-8"),
    array_job_id,
)

metric_fields = (
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_transformer_overload",
    "total_energy_charged",
    "average_user_satisfaction",
    "global_action_fraction_at_max_active",
)
paired_rows = []
for scale in ("25cp", "100cp", "500cp", "1000cp"):
    for training_seed in TRAINING_SEEDS:
        action_rows = metric_rows[(scale, "actiongnn", training_seed)]
        hierarchical_rows = metric_rows[(scale, "hierarchical", training_seed)]
        if len(action_rows) != 30 or len(hierarchical_rows) != 30:
            raise SystemExit(
                f"paired metric rows incomplete: {scale} seed {training_seed}"
            )
        output = {
            "scale": scale,
            "training_seed": str(training_seed),
            "status": "ok",
            "episodes_per_algorithm": "30",
        }
        for field in metric_fields:
            action_mean = statistics.fmean(float(row[field]) for row in action_rows)
            hierarchical_mean = statistics.fmean(
                float(row[field]) for row in hierarchical_rows
            )
            output[f"actiongnn_{field}_mean"] = repr(action_mean)
            output[f"hierarchical_{field}_mean"] = repr(hierarchical_mean)
            output[f"hierarchical_minus_actiongnn_{field}"] = repr(
                hierarchical_mean - action_mean
            )
        paired_rows.append(output)

write_csv(
    root / "summaries/task_inventory.csv",
    ["task_id", "package_name", "package_sha256"],
    sorted(task_inventory, key=lambda row: int(row["task_id"])),
)
checkpoint_fields = [
    "task_id",
    "scale",
    "algorithm",
    "training_seed",
    "formal_task_id",
    "status",
]
write_csv(
    root / "summaries/checkpoint_inventory.csv",
    checkpoint_fields,
    sorted(
        checkpoint_rows,
        key=lambda row: (int(row["task_id"]), int(row["training_seed"])),
    ),
)
episode_fields = [
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "diagnostic_schema_version",
]
write_csv(
    root / "summaries/episode_inventory.csv",
    episode_fields,
    sorted(
        episode_inventory_rows,
        key=lambda row: (
            ("25cp", "100cp", "500cp", "1000cp").index(row["scale"]),
            ("actiongnn", "hierarchical").index(row["algorithm"]),
            int(row["training_seed"]),
            int(row["episode_index"]),
        ),
    ),
)
runtime_fields = [
    "task_id",
    "job_id_raw",
    "job_id",
    "job_name",
    "state",
    "exit_code",
    "elapsed_raw",
    "alloc_cpus",
    "max_rss",
    "total_cpu",
    "maxrss_source",
    "totalcpu_source",
]
write_csv(root / "summaries/runtime_summary.csv", runtime_fields, runtime_rows)

canonical_fields = list(canonical_rows[0])
write_csv(
    root / "summaries/canonical_reconciliation_summary.csv",
    canonical_fields,
    canonical_rows,
)
mapping_fields = list(mapping_rows[0])
write_csv(
    root / "summaries/mapping_validation_summary.csv",
    mapping_fields,
    mapping_rows,
)
service_fields = list(service_rows[0])
write_csv(
    root / "summaries/service_reconciliation_summary.csv",
    service_fields,
    service_rows,
)
write_csv(
    root / "summaries/schema_inventory.csv",
    ["diagnostic_schema_version", "episode_count"],
    [{"diagnostic_schema_version": SCHEMA_VERSION, "episode_count": "1200"}],
)
write_csv(root / "summaries/missing_field_inventory.csv", ["field", "count"], [])
paired_fields = list(paired_rows[0])
write_csv(
    root / "summaries/seed_level_paired_summary.csv",
    paired_fields,
    paired_rows,
)

(root / "validation/failure_manifest.txt").write_bytes(b"")
(root / "validation/warning_inventory.txt").write_bytes(b"")
(root / "provenance/source_commit_sha.txt").write_text(
    source_sha + "\n", encoding="utf-8"
)
(root / "provenance/array_job_id.txt").write_text(
    array_job_id + "\n", encoding="utf-8"
)
(root / "provenance/reducer_job_id.txt").write_text(
    reducer_job_id + "\n", encoding="utf-8"
)
(root / "provenance/sacct_raw.txt").write_bytes(sacct_raw_path.read_bytes())

manifest = root / "checksums/complete_file_checksums.sha256"
files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
with manifest.open("w", encoding="utf-8") as handle:
    for path in files:
        handle.write(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
        )
PY

rm -f "${TEMP_BUNDLE}" "${TEMP_SIDECAR}"
tar -czf "${TEMP_BUNDLE}" -C "${STAGING_ROOT}" \
  task_packages summaries validation provenance checksums
python - "${TEMP_BUNDLE}" <<'PY'
import json
import sys
from pathlib import Path
from scripts.full_infrastructure_diagnostic_eval30_packages import (
    validate_stage_d_complete_bundle,
)
result = validate_stage_d_complete_bundle(Path(sys.argv[1]))
print(json.dumps(result, sort_keys=True))
PY
sha256sum "${TEMP_BUNDLE}" | sed "s#${TEMP_BUNDLE}#${FINAL_BASENAME}#" \
  > "${TEMP_SIDECAR}"
mv "${TEMP_BUNDLE}" "${FINAL_BUNDLE}"
mv "${TEMP_SIDECAR}" "${FINAL_SIDECAR}"

echo "FULL_INFRASTRUCTURE_DIAGNOSTIC_COMPLETE_BUNDLE_OK"
echo "array_job_id=${ARRAY_JOB_ID}"
echo "reducer_job_id=${REDUCER_JOB_ID}"
echo "task_packages=8"
echo "checkpoint_groups=40"
echo "episodes=1200"
echo "final_bundle=${FINAL_BUNDLE}"
echo "final_bundle_sha256=${FINAL_SIDECAR}"
