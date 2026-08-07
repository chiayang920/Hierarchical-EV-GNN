# Formal 75k 80-cell M3 Run Card

## Current gate

```text
NEW_FORMAL_RESULTS=NONE
M3_SMOKE_READY=NO
M3_ACTION_PERFORMED=NO
NEXT_REQUIRED_GATE=USER_RUN_LOCAL_REPAIR_VERIFICATION_AND_LOCAL_FUNCTIONAL_SMOKE
```

This card describes the repaired workflow contract. It does not authorise an M3 submission. No smoke or formal job may be submitted until both user-run local scripts finish with `FINAL_STATUS=PASS`, the resulting evidence is reviewed, and a separate M3 resource profile is approved.

## Source identity

```text
BRANCH=exp/formal-75k-nonnegative-comparison-v1
SOURCE_HEAD=b5f0c2fd1b6895f71699f134aa04b1b9492ded4e
BASE_SCOPE_A_HEAD=e41585c83c8840d5c8aae7819f340c7d4d2619b5
```

The local repair package is not a Git branch or commit. Before a later write operation in the user's live repository, the user must independently verify current branch, HEAD, ancestry and worktree cleanliness.

## Historical Stage D boundary

```text
HISTORICAL_STAGE_D_JOB=58745233
HISTORICAL_STAGE_D_REDUCER=58746039
HISTORICAL_STAGE_D_STATUS=COMPLETE
HISTORICAL_STAGE_D_REPAIR_REQUIRED=NO
HISTORICAL_STAGE_D_RERUN_REQUIRED=NO
```

The repaired `m3_jobs/26_formal_75k_80cell_diagnostics.slurm` belongs only to the new Formal-75k workflow. It does not modify or reopen the completed historical Stage D workflow.

## Scientific matrix

```text
Scales=25cp,100cp,500cp,1000cp
Algorithms=actiongnn_nonnegative,hierarchical
Seeds=0,1,2,3,4,5,6,7,8,9
Formal cells=80
Training steps=75000
Scheduled evaluation cadence=5000
Training evaluation episodes=5
Scheduled evaluations=15
model.best evaluation episodes=30
Expected canonical evaluation rows=2400
Primary inference unit=paired training seed
Primary efficacy outcome=episode_reward
Key mechanism outcome=upper_bound_active_ev_action_fraction
```

Print the matrix locally with the approved interpreter:

```bash
EV_GNN_FORMAL_75K_PYTHON=/opt/anaconda3/envs/evgnn_core/bin/python3.11 \
  /opt/anaconda3/envs/evgnn_core/bin/python3.11 \
  scripts/formal_75k_80cell_workflow.py --print-matrix
```

## Dependency stages

1. Stage S: two-cell smoke.
2. Smoke gate: exact two packages, successful training, scheduled evaluations, `model.best`, successful reload eval, package provenance and no implementation failure.
3. Stage A: 80 fresh training cells.
4. Stage B: exact-job package validation and atomic checkpoint/config staging.
5. Stage C: 80 `model.best` eval30 tasks.
6. Stage D: 80 infrastructure-diagnostic tasks using the same staged checkpoints; Stage C and D remain separate.
7. Stage E: independent eval30 and diagnostic completeness/reconciliation gates.
8. Stage F: final paired-seed reducer, service guardrail and claim assessment.

Every stage that reads M3 outputs requires exact manually supplied job IDs. No output-directory inference is permitted.

## Python bootstrap

Every launcher requires:

```bash
export EV_GNN_FORMAL_75K_PYTHON=/scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu/bin/python
```

The helper verifies Python 3.11 before the first project Python call. There is no bare-`python` fallback.

## Resource governance

Both smoke and formal submission require an explicit reviewed profile. The template at `docs/formal_75k_resource_profile_template.env` intentionally contains `RESOURCE_PROFILE_APPROVED=NO` and blank values. Do not copy the local functional-smoke profile to M3; it is not an M3 resource approval.

## Dry run only

```bash
export EV_GNN_FORMAL_75K_PYTHON=/opt/anaconda3/envs/evgnn_core/bin/python3.11
bash m3_jobs/submit_formal_75k_80cell_workflow.sh --dry-run --print-matrix
```

Expected terminal marker:

```text
DRY_RUN_NO_JOBS_SUBMITTED
```

## Claim boundary

The reducer uses reward as the sole primary efficacy outcome. Tracking error is descriptive. Holm adjustment is applied separately across four scale-specific reward tests and four scale-specific boundary-behaviour tests. Exact sign-flip randomisation uses all `2^10` paired sign assignments. Service status is data-derived as `PASS`, `FAIL` or `UNKNOWN`; `UNKNOWN` forces `STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED=NO`. No service non-inferiority claim is permitted without a pre-approved margin.
