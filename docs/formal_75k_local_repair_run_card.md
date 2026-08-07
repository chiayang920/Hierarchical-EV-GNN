# Formal-75k Local Repair v2 Run Card

## Purpose

This v2 source snapshot repairs the confirmed `training_log.csv` producer defect found by the first local functional smoke. The failed v1 smoke evidence remains authoritative debugging evidence and must not be overwritten.

Confirmed v1 defect:

```text
CSV header columns: 8
Evaluation row columns: 43
eval/mean_reward through csv.DictReader: None
```

The strict package validator is unchanged. The producer now expands the named CSV schema when evaluation metrics first appear and preserves uniform row widths.

## Canonical local storage

Create a separate v2 root; do not replace the v1 directory:

```text
/Users/jameschen/Desktop/MAIN_HOME/Monash/monash_course/2026S1/Research/Research_project/EVGNN_Research_Artefacts/80_audit_and_handover/corrected_nonnegative_flat_actiongnn_design/formal_75k_80cell_workflow/local_repair_20260807_v2_csv_schema
```

Recommended layout:

```text
local_repair_20260807_v2_csv_schema/
├── source_packages/
├── scripts/
├── documentation/
├── extracted/
└── outputs/
    ├── verification/
    └── functional_smoke/
```

Do not unzip over either live Git worktree or the v1 extracted snapshot.

## 1. Full local verification

Required interpreter:

```text
/opt/anaconda3/envs/evgnn_core/bin/python3.11
```

The v2 verification script explicitly includes:

```text
tests/test_train_td3_gnn_logging.py
train_td3_gnn.py Python compile check
```

Execute:

```bash
"$REPAIR_ROOT_V2/scripts/run_local_formal75k_v2_repair_verification.sh" \
  "$REPAIRED_REPO_V2" \
  "$FORMAL_EVIDENCE" \
  "$DIAGNOSTIC_EVIDENCE" \
  "$REPAIR_ROOT_V2/outputs/verification"
```

Required final fields:

```text
PHASE=FORMAL_75K_WORKFLOW_LOCAL_REPAIR_VERIFICATION
FINAL_STATUS=PASS
BLOCK_REASON=NONE
M3_ACTION_PERFORMED=NO
GIT_ACTION_PERFORMED=NO
M3_SUBMISSION_COUNT=0
CONSOLIDATED_LOG=<path>
```

Return the complete terminal output and consolidated log.

## 2. Local functional smoke

Execute only after v2 verification returns `FINAL_STATUS=PASS`:

```bash
SMOKE_EVIDENCE_ROOT="$REPAIR_ROOT_V2/outputs/functional_smoke/$(date +%Y%m%d_%H%M%S)"

"$REPAIR_ROOT_V2/scripts/run_local_formal75k_v2_functional_smoke.sh" \
  "$REPAIRED_REPO_V2" \
  "$SMOKE_EVIDENCE_ROOT"
```

It runs exactly:

```text
25CP actiongnn_nonnegative seed0
25CP hierarchical seed0
```

Required final fields:

```text
PHASE=FORMAL_75K_LOCAL_FUNCTIONAL_SMOKE
FINAL_STATUS=PASS
BLOCK_REASON=NONE
SMOKE_CELL_COUNT=2
SCIENTIFIC_CLAIM_GENERATED=NO
M3_ACTION_PERFORMED=NO
GIT_ACTION_PERFORMED=NO
EVIDENCE_ROOT=<path>
CONSOLIDATED_LOG=<path>
```

The new smoke must also show that both `training_log.csv` files have named `eval/mean_reward` fields and no unnamed extra columns. Return the complete terminal output, consolidated smoke log, `smoke_gate_output.txt`, and both smoke package filenames.
