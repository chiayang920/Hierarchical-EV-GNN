# Full Per-Infrastructure Diagnostics Eval30 Protocol

## 1. Purpose

This protocol governs the complete deterministic per-infrastructure diagnostic re-evaluation of the existing EV-GNN formal checkpoints.

No model retraining is permitted. The workflow reuses the 40 checkpoints produced by formal M3 job `58513929`; it does not modify the actor, critic, replay buffer, reward, simulator, checkpoint files, or canonical formal outputs.

The scientific workload is:

```text
4 scales × 2 algorithms × 5 training seeds × 30 deterministic episodes
= 40 checkpoints × 30 episodes
= 1,200 episodes
```

The training seed remains the statistical inference unit. The 30 deterministic episodes are within-seed replicates and must not be treated as 30 independent samples.

## 2. Evidence lineage

The predecessor workflow is smoke job `58622672`, reduced by job `58622673`.

That smoke established:

- eight scale-algorithm tasks could execute;
- schema v3 diagnostics could be generated;
- formal checkpoint loading and infrastructure mapping worked at 25CP, 100CP, 500CP, and 1000CP;
- canonical and service reconciliation worked for one selected checkpoint and one deterministic episode per scale-algorithm cell;
- exact job accounting and final evidence packaging worked.

The smoke is not the complete diagnostic evidence. This workflow supersedes the smoke only after the acceptance and smoke-retirement gates in this protocol pass.

## 3. Approved task topology

Eight top-level Slurm tasks are used:

| Array task | Scale | Algorithm | Formal task IDs |
|---:|---|---|---|
| 0 | 25CP | ActionGNN | 0–4 |
| 1 | 25CP | Hierarchical | 5–9 |
| 2 | 100CP | ActionGNN | 10–14 |
| 3 | 100CP | Hierarchical | 15–19 |
| 4 | 500CP | ActionGNN | 20–24 |
| 5 | 500CP | Hierarchical | 25–29 |
| 6 | 1000CP | ActionGNN | 30–34 |
| 7 | 1000CP | Hierarchical | 35–39 |

Each array task evaluates five checkpoints sequentially. Each checkpoint receives 30 deterministic episodes, so every top-level task produces 150 episodes.

Deterministic scale offsets are paired across algorithms:

| Scale | Offset |
|---|---:|
| 25CP | 710000 |
| 100CP | 720000 |
| 500CP | 730000 |
| 1000CP | 740000 |

Episode identity is:

```text
episode_seed = scale_offset + training_seed + episode_index
```

where `episode_index` is `0–29`.

## 4. Operator files

The implementation is divided into the following operator and validation files:

```text
m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh
m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm
m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm
m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh
scripts/validate_full_infrastructure_diagnostic_eval30.py
scripts/full_infrastructure_diagnostic_eval30_packages.py
scripts/full_infrastructure_diagnostic_eval30_accounting.py
```

The established smoke validator is reused only for already-tested formal-package, configuration, reconciliation, and archive primitives:

```text
scripts/validate_infrastructure_diagnostic_smoke.py
```

## 5. Pre-execution repository gate

Real M3 execution is prohibited until all of the following are true:

1. the implementation pull request has been reviewed and merged;
2. local `main` equals the reviewed merge commit;
3. the local working tree is clean, including untracked files;
4. focused Stage D tests pass;
5. the complete repository test suite passes;
6. Python compilation passes;
7. all four shell scripts pass `bash -n`;
8. all eight array dry-runs pass;
9. reducer dry-run passes;
10. submit-helper dry-run passes and prints `DRY_RUN_NO_SBATCH_CALLED`;
11. a dedicated immutable source bundle is created and independently checksum-validated.

No source archive may be created from an implementation branch. The source-bundle creator requires clean `main`.

## 6. Create the immutable source bundle locally

Run from the physical repository root after the implementation PR is merged:

```bash
cd /Users/jameschen/Desktop/MAIN_HOME/Monash/monash_course/2026S1/Research/Research_project/EV-GNN

SOURCE_SHA="$(git rev-parse HEAD)"
SOURCE_OUTPUT="$HOME/Downloads/EVGNN_Formal_Evidence"

EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXPECTED_HEAD_SHA="$SOURCE_SHA" \
EV_GNN_FULL_DIAGNOSTIC_SOURCE_OUTPUT_ROOT="$SOURCE_OUTPUT" \
bash m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh
```

Required success marker:

```text
SOURCE_BUNDLE_OK
```

Expected archive identity:

```text
EV-GNN-full-infrastructure-diagnostics-eval30-<SOURCE_SHA>.tar.gz
```

The allowlisted archive contains only execution source files and `SOURCE_COMMIT_SHA.txt`. It must not contain checkpoints, `saved_models`, `.git`, task packages, caches, bytecode, raw evidence, symbolic links, hard links, or device entries.

Verify the local sidecar automatically. The operator does not need to read the SHA-256 characters manually:

```bash
cd "$SOURCE_OUTPUT"
shasum -a 256 -c \
  "EV-GNN-full-infrastructure-diagnostics-eval30-${SOURCE_SHA}.tar.gz.sha256"
```

Required result:

```text
EV-GNN-full-infrastructure-diagnostics-eval30-<SOURCE_SHA>.tar.gz: OK
```

## 7. Transfer and extract on M3

Transfer the archive and sidecar printed by the source-bundle creator:

```bash
scp \
  "$SOURCE_OUTPUT/EV-GNN-full-infrastructure-diagnostics-eval30-${SOURCE_SHA}.tar.gz" \
  "$SOURCE_OUTPUT/EV-GNN-full-infrastructure-diagnostics-eval30-${SOURCE_SHA}.tar.gz.sha256" \
  cche0357@m3.massive.org.au:/projects/fr57/cche0357/EV-GNN_sources/
```

On M3:

```bash
SOURCE_SHA="<exact merged implementation SHA>"
TOP="EV-GNN-full-infrastructure-diagnostics-eval30-${SOURCE_SHA}"
ARCHIVE_ROOT="/projects/fr57/cche0357/EV-GNN_sources"
EXTRACT_PARENT="/scratch2/fr57/cche0357/EV-GNN_sources"
REPO_ROOT="${EXTRACT_PARENT}/${TOP}"

cd "$ARCHIVE_ROOT"
sha256sum -c "${TOP}.tar.gz.sha256"

test ! -e "$REPO_ROOT" || {
  echo "ERROR: extracted Stage D source already exists: $REPO_ROOT" >&2
  exit 1
}

mkdir -p "$EXTRACT_PARENT"
tar -xzf "${ARCHIVE_ROOT}/${TOP}.tar.gz" -C "$EXTRACT_PARENT"

cd "$REPO_ROOT"
test "$(tr -d '[:space:]' < SOURCE_COMMIT_SHA.txt)" = "$SOURCE_SHA"
```

Do not edit the extracted source tree.

## 8. Activate and verify the M3 environment

```bash
module load miniforge3/24.3.0-0
conda activate /scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu

python --version
python - <<'PY'
import torch
import torch_geometric
import yaml
print("torch=" + torch.__version__)
print("torch_geometric=" + torch_geometric.__version__)
print("pyyaml=" + yaml.__version__)
PY

bash -n \
  m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm \
  m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm \
  m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh
```

The M3 environment may contain a CUDA-enabled PyTorch build. This is not a GPU experiment: the array runner requests no GPU and passes `--device cpu`.

## 9. M3 submission dry-run

The helper validates source identity, all eight task mappings, all 40 formal packages, their checksums, configs, checkpoints, and canonical eval30 identities before printing submission commands.

```bash
OUTPUT_ROOT="/projects/fr57/cche0357/EV-GNN_outputs"
RUN_ROOT="/scratch2/fr57/cche0357/EV-GNN_runs/full_infrastructure_diagnostics_eval30"

EV_GNN_FULL_DIAGNOSTIC_SUBMIT_DRY_RUN=1 \
EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT="$SOURCE_SHA" \
EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID="58513929" \
EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT="$OUTPUT_ROOT" \
EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE="${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz" \
EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT="$OUTPUT_ROOT" \
EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT="$RUN_ROOT" \
bash m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh \
  | tee "${OUTPUT_ROOT}/full_infrastructure_diagnostics_eval30_submit_dry_run_${SOURCE_SHA}.log"
```

Required markers:

```text
FORMAL_PREFLIGHT_OK
validated_formal_checkpoints=40
FULL_INFRASTRUCTURE_DIAGNOSTIC_SUBMIT_DRY_RUN
DRY_RUN_NO_SBATCH_CALLED
```

Do not proceed if any formal package is missing, ambiguous, checksum-invalid, mapped to the wrong scale/algorithm/seed, or contains an invalid canonical episode inventory.

## 10. Real submission

Run the helper exactly once after the dry-run and all gates pass:

```bash
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUBMISSION_LOG="${OUTPUT_ROOT}/full_infrastructure_diagnostics_eval30_submission_${SOURCE_SHA}_${STAMP}.log"

set -o pipefail

EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT="$SOURCE_SHA" \
EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID="58513929" \
EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT="$OUTPUT_ROOT" \
EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE="${OUTPUT_ROOT}/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz" \
EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT="$OUTPUT_ROOT" \
EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT="$RUN_ROOT" \
bash m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh \
  2>&1 | tee "$SUBMISSION_LOG"

HELPER_STATUS="${PIPESTATUS[0]}"
if [[ "$HELPER_STATUS" -ne 0 ]]; then
  echo "ERROR: helper exited with status ${HELPER_STATUS}" >&2
  echo "Do not rerun automatically. Preserve: ${SUBMISSION_LOG}" >&2
  exit "$HELPER_STATUS"
fi
```

Required success marker:

```text
FULL_INFRASTRUCTURE_DIAGNOSTIC_WORKFLOW_SUBMITTED
array_job_id=<digits>
reducer_job_id=<digits>
```

The reducer dependency is bound to the exact submitted array identity:

```bash
--dependency=afterok:${ARRAY_JOB_ID}
```

## 11. Capture exact job identities

```bash
test "$(grep -Ec '^array_job_id=[0-9]+$' "$SUBMISSION_LOG")" -eq 1
test "$(grep -Ec '^reducer_job_id=[0-9]+$' "$SUBMISSION_LOG")" -eq 1

ARRAY_JOB_ID="$(awk -F= '$1 == "array_job_id" {print $2}' "$SUBMISSION_LOG")"
REDUCER_JOB_ID="$(awk -F= '$1 == "reducer_job_id" {print $2}' "$SUBMISSION_LOG")"

[[ "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]
[[ "$REDUCER_JOB_ID" =~ ^[0-9]+$ ]]

JOB_ID_FILE="${OUTPUT_ROOT}/full_infrastructure_diagnostics_eval30_job_ids_${SOURCE_SHA}_${STAMP}.env"
{
  echo "SOURCE_COMMIT=${SOURCE_SHA}"
  echo "ARRAY_JOB_ID=${ARRAY_JOB_ID}"
  echo "REDUCER_JOB_ID=${REDUCER_JOB_ID}"
  echo "SUBMISSION_LOG=${SUBMISSION_LOG}"
} > "$JOB_ID_FILE"
chmod 600 "$JOB_ID_FILE"
```

Do not discover the latest job. Do not substitute a job found by username, job name, queue order, filesystem modification time, or shell history.

## 12. Exact monitoring

```bash
squeue -j ${ARRAY_JOB_ID},${REDUCER_JOB_ID}
```

Array accounting must use exactly:

```bash
sacct -j "$ARRAY_JOB_ID" \
  --parsable2 \
  --noheader \
  --format=JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU
```

Acceptance requires parent, `.batch`, and `.extern` identities for all eight tasks, all `COMPLETED`, all exit codes `0:0`, and correct `JobIDRaw` parent-child linkage.

Reducer accounting:

```bash
sacct -j "$REDUCER_JOB_ID" \
  --parsable2 \
  --noheader \
  --format=JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU
```

## 13. Reducer failure and recovery

The normal reducer is submitted with `afterok` and must use the exact original array job ID.

When the array succeeded but reducer submission failed after the array `sbatch`, preserve the printed array ID. Do not rerun automatically. Investigate the submission log before any action.

A reducer-only recovery may be performed only when:

1. the exact original array job ID is known from the preserved submission log or job-ID file;
2. exact accounting proves 8/8 array tasks completed with `0:0`;
3. all eight task packages and sidecars exist for that same array ID;
4. every task package validates independently;
5. no complete bundle already exists for that array ID;
6. the recovery command exports that exact original array job ID.

Never combine task packages from different array identities.

## 14. Final M3 evidence validation

Expected final files:

```bash
FINAL_BUNDLE="${OUTPUT_ROOT}/full_infrastructure_diagnostics_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"
FINAL_SIDECAR="${FINAL_BUNDLE}.sha256"
```

Required reducer marker:

```text
FULL_INFRASTRUCTURE_DIAGNOSTIC_COMPLETE_BUNDLE_OK
```

Validate on M3:

```bash
test -s "$FINAL_BUNDLE"
test -s "$FINAL_SIDECAR"

cd "$OUTPUT_ROOT"
sha256sum -c "$(basename "$FINAL_SIDECAR")"

cd "$REPO_ROOT"
python - "$FINAL_BUNDLE" <<'PY'
import json
import sys
from pathlib import Path
from scripts.full_infrastructure_diagnostic_eval30_packages import (
    validate_stage_d_complete_bundle,
)
print(json.dumps(validate_stage_d_complete_bundle(Path(sys.argv[1])), sort_keys=True))
PY
```

Expected validator scope:

```text
8 task packages
40 checkpoint groups
1,200 episodes
schema v3
```

## 15. Download and local verification

```bash
cd ~/Downloads

scp \
  cche0357@m3.massive.org.au:"$FINAL_BUNDLE" \
  cche0357@m3.massive.org.au:"$FINAL_SIDECAR" \
  .

shasum -a 256 -c \
  "full_infrastructure_diagnostics_complete_evidence_job${ARRAY_JOB_ID}.tar.gz.sha256"
```

The final archive must then undergo an independent local audit. The audit must inspect the nested task packages, not merely trust the outer sidecar.

## 16. External evidence directory

Stage D evidence must be stored separately from the smoke evidence.

Canonical semantic name:

```text
EVGNN_per_infrastructure_diagnostics_40checkpoints_eval30_job<ARRAY_JOB_ID>_<YYYYMMDD>
```

Required structure:

```text
README.md
SHA256SUMS
FILE_ORIGIN_MAP.tsv
final_evidence/
execution_provenance/
source_provenance/
verification_reports/
```

Raw archives, Slurm logs, job-ID files, and machine-specific provenance remain outside Git. Git stores only concise result notes and compact derived summaries.

## 17. Acceptance gates

The workflow passes only when all of the following are independently verified:

### Operational

- 8/8 tasks completed;
- 8/8 parent task exit codes are `0:0`;
- all required parent, `.batch`, and `.extern` accounting identities are present;
- reducer parent, `.batch`, and `.extern` are `COMPLETED`, `0:0`;
- exact source commit and exact array/reducer IDs are recorded.

### Package and provenance

- 8/8 task packages are readable;
- 8/8 task-package sidecars validate;
- 8/8 task-package internal manifests validate;
- 40/40 checkpoint groups are present and uniquely mapped;
- formal task IDs are exactly `0–39`;
- all task packages use one source commit and one array job ID;
- the complete bundle and portable sidecar validate.

### Scientific data

- 1,200/1,200 episode keys are present;
- each checkpoint has exactly 30 episode indices `0–29`;
- no duplicate or unexpected episode key exists;
- episode-seed formulas are exact;
- schema v3 is present everywhere;
- every canonical reconciliation status passes;
- every mapping validation status passes;
- every service reconciliation status passes;
- required metrics contain no blank, NaN, positive infinity, or negative infinity;
- the missing-field inventory is empty;
- the failure manifest is empty;
- the warning inventory is empty;
- published stderr logs are empty.

### Feasibility

- 1000CP tasks complete within the six-hour request;
- MaxRSS remains within the 32 GB request;
- final output size is manageable;
- reducer processing and filesystem use remain acceptable.

## 18. Statistical analysis boundary

The reducer creates descriptive seed-level means and paired algorithm differences. Inferential analysis must aggregate the 30 episodes within each training seed first.

Valid inferential sample size per scale-algorithm comparison is five paired training seeds, not 150 episodes.

The Stage D results support mechanism interpretation, infrastructure alignment analysis, and the subsequent Arm C specification. They do not constitute new training evidence.

## 19. Smoke-retirement gate

The active smoke directory may be deleted only after:

1. Stage D passes every gate above;
2. the final Stage D bundle passes independent local audit;
3. Stage D evidence exists in two independent storage locations;
4. repository documentation records smoke job `58622672` as predecessor evidence;
5. the Stage D README records the smoke lineage and supersession;
6. no active thesis note points only to the smoke archive.

After those conditions pass, the smoke may be removed from the active research workspace. A compressed cold-backup copy should remain until thesis submission is complete.

## 20. Claim boundary

Before completion, use:

> The schema-v3 diagnostic workflow passed an eight-task smoke and the complete 40-checkpoint eval30 re-evaluation is pending.

After all Stage D gates pass, use:

> The per-infrastructure diagnostic evidence covers 40 existing formal checkpoints and 1,200 deterministic evaluation episodes, with training seed retained as the statistical inference unit.

Do not describe this workflow as model training, policy retraining, or a new algorithm experiment.
