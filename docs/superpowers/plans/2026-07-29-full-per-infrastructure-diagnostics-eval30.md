# Full Per-Infrastructure Diagnostics Eval30 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provenance-controlled M3 workflow that re-evaluates all 40 existing formal job `58513929` checkpoints for 30 deterministic schema-v3 infrastructure-diagnostic episodes each, using eight scale-by-algorithm Slurm tasks and one exact-ID reducer.

**Architecture:** Reuse the existing diagnostic evaluator and smoke contracts. Each array task owns one scale-algorithm cell, validates five formal checkpoint packages, runs eval30 sequentially for seeds `0–4`, validates each seed output, and publishes one atomic task package. The reducer accepts only the exact array job ID, validates `8/8` packages, `40/40` checkpoint groups, and `1,200/1,200` episode keys, then publishes one complete evidence bundle.

**Tech Stack:** Python 3.11, PyTorch 2.2.2, PyTorch Geometric 2.5.3, PyYAML 6.0.3, pytest, Bash 3.2-compatible shell, Slurm, tar, SHA-256.

## Global constraints

- Base commit: `1f71b1af4a28ae4503c6675e3f9244737326bf18`.
- Formal source job: `58513929`.
- No retraining and no actor, critic, replay-buffer, reward, simulator, checkpoint, or canonical-formal-output changes.
- Scales: `25cp`, `100cp`, `500cp`, `1000cp`.
- Algorithms: `actiongnn`, `hierarchical`.
- Training seeds: `0–4`.
- Episodes: `30` per checkpoint; `1,200` total.
- Top-level tasks: `8`; checkpoint groups: `40`.
- Diagnostic schema: exact version `3`.
- CPU only; initial task request: `4` CPUs, `32G`, `06:00:00`.
- Reducer dependency: `afterok:<exact_array_job_id>`.
- Accounting schema: `JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU`.
- Prohibit latest-job discovery, job-name discovery, unscoped `sacct`, stale-package reuse, and mixed array identities.
- Raw evidence stays outside Git.
- Publication is temporary archive → validation → atomic rename.
- No M3 submission during implementation.

## Files

Create:

- `scripts/validate_full_infrastructure_diagnostic_eval30.py`
- `tests/test_full_infrastructure_diagnostic_eval30_workflow.py`
- `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm`
- `m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm`
- `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh`
- `m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh`
- `docs/full_per_infrastructure_diagnostics_eval30_protocol.md`

Reuse unchanged unless a failing test proves a defect:

- `evaluate_td3_gnn_infrastructure_diagnostics.py`
- `utils/infrastructure_diagnostics.py`
- `scripts/validate_infrastructure_diagnostic_smoke.py`

---

### Task 1: Mapping and deterministic episode identity

**Produces:** `STAGE_D_TASKS`, `TRAINING_SEEDS`, `EVAL_EPISODES`, `SCALE_SEED_OFFSETS`, `stage_d_task()`, `formal_task_id()`, `episode_seed()`, and CLI `task-mapping`.

- [ ] Write parametrised tests for exact mappings:

```python
EXPECTED = {
    0: ("25cp", "actiongnn", range(0, 5)),
    1: ("25cp", "hierarchical", range(5, 10)),
    2: ("100cp", "actiongnn", range(10, 15)),
    3: ("100cp", "hierarchical", range(15, 20)),
    4: ("500cp", "actiongnn", range(20, 25)),
    5: ("500cp", "hierarchical", range(25, 30)),
    6: ("1000cp", "actiongnn", range(30, 35)),
    7: ("1000cp", "hierarchical", range(35, 40)),
}
```

- [ ] Assert `TRAINING_SEEDS == (0,1,2,3,4)` and `EVAL_EPISODES == 30`.
- [ ] Assert seed formula `offset + training_seed + episode_index`, with offsets `710000`, `720000`, `730000`, `740000`.
- [ ] Run red test:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py -k "mapping or episode_identity" -q
```

- [ ] Implement the minimal mapping core and exact CLI output.
- [ ] Run all eight CLI mappings and focused tests.
- [ ] Commit:

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Add full diagnostic task mapping contract"
```

### Task 2: Seed-output and 1,200-episode inventory validation

**Produces:** `expected_episode_keys()`, `validate_episode_inventory()`, `validate_seed_output_directory()`, CLI `episode-inventory` and `validate-seed-output`.

- [ ] Write tests for exactly `1,200` unique `(scale, algorithm, training_seed, episode_index)` keys.
- [ ] Add rejection tests for duplicate/missing/unexpected keys, wrong episode seed, schema not `3`, 29/31 episodes, wrong scale/algorithm/seed, missing CSV, missing required column, blank required metric, NaN/Inf, failed canonical reconciliation, failed mapping validation, failed service reconciliation, and non-empty stderr.
- [ ] Run red tests.
- [ ] Implement strict CSV/header/value validation.
- [ ] Require per seed:

```text
episode_diagnostics.csv: 30 episode rows
seed_summary_diagnostics.csv: 1 summary row
transformer_diagnostics.csv: schema v3
charger_diagnostics.csv: schema v3
canonical_reconciliation.csv: all pass
service_reconciliation.csv: all pass
mapping_validation.json: status ok
stderr.log: empty
```

- [ ] Run focused tests and commit:

```bash
git commit -am "Validate full diagnostic episode inventories"
```

### Task 3: Task-package and complete-bundle contracts

**Produces:** `validate_safe_tar_members()`, `validate_stage_d_task_package()`, `validate_stage_d_complete_bundle()`, CLI `validate-task-package` and `validate-complete-bundle`.

- [ ] Write real-tar fixtures and failing tests for unsafe absolute/traversal paths, duplicates, symlinks, hard links, device entries, checksum mismatch, four/six seed groups, duplicate seed, wrong task identity, checkpoint leakage, seven/nine task packages, missing checkpoint groups, and incomplete episode inventory.
- [ ] A task package must contain exactly five seed groups, five formal task IDs, 150 episode keys, task metadata, inventories, runtime summary, validation JSON, checksum manifest, and empty stderr.
- [ ] A complete bundle must contain eight task packages, 40 checkpoint groups, 1,200 episode keys, schema/reconciliation/missing-field inventories, seed-level paired summary, failure/warning inventories, exact source/array/reducer provenance, and complete checksums.
- [ ] Run focused package tests and commit:

```bash
git commit -am "Add full diagnostic package validation"
```

### Task 4: Eight-task eval30 Slurm runner

**Creates:** `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm`.

- [ ] Write tests for `--array=0-7`, `--cpus-per-task=4`, `--mem=32G`, `--time=06:00:00`, CPU device, eval30, `model.best`, job-scoped run root, and atomic task publication.
- [ ] Implement one task as one scale-algorithm cell. Loop seeds `0–4`; resolve the exact formal task ID; validate package; safely extract; stage `model.best` and immutable config; run evaluator; run mapping/canonical/service validation; validate seed output.
- [ ] Evaluator command must set explicit scale/config/seed, `--eval_episodes 30`, deterministic true, zero exploration noise, validated max steps, scale offset, and current array job ID.
- [ ] Package only after all five seeds pass.
- [ ] Dry-run must print five command templates and `expected_episode_count=150` without evaluation or packaging.
- [ ] Run `bash -n`, eight dry-runs, focused tests, and commit:

```bash
git commit -am "Add full diagnostic eval30 array runner"
```

### Task 5: Reducer and exact accounting

**Creates:** `m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm`.

- [ ] Write exact nine-column parser tests for header rejection, malformed row, wrong array ID, missing/duplicate parent/batch/extern rows, non-COMPLETED state, non-zero exit code, incomplete task IDs, and resource fallback errors.
- [ ] Implement `parse_stage_d_sacct()` requiring task IDs `0–7`, parent/batch/extern identity consistency, `COMPLETED`, and `0:0`.
- [ ] Reducer must validate eight current-array packages, build 40-checkpoint and 1,200-episode inventories, validate schema/reconciliation/missing fields, aggregate seed-level pairs, create complete checksums, validate a temporary final archive, then atomically publish archive and sidecar.
- [ ] Reducer resources: `2` CPUs, `16G`, `01:00:00`.
- [ ] Dry-run must print exact `sacct`, expected counts, and no accounting/packaging marker.
- [ ] Run focused tests, `bash -n`, and commit:

```bash
git commit -am "Add full diagnostic reducer and accounting gates"
```

### Task 6: Guarded submit helper

**Creates:** `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh`.

- [ ] Write failure tests for wrong physical root, missing/mismatched source SHA, missing scripts, missing/ambiguous/invalid formal packages, duplicate formal IDs, and non-numeric `sbatch` output.
- [ ] Before the first `sbatch`, validate all 40 formal task packages and all eight mappings.
- [ ] Submit one array and one `afterok:<array_job_id>` reducer.
- [ ] Print authoritative numeric IDs, final paths, exact `squeue`, exact nine-column `sacct`, M3 validator/checksum commands, and two-file SCP command.
- [ ] Dry-run must print all mappings and `DRY_RUN_NO_SBATCH_CALLED`.
- [ ] Run focused tests, `bash -n`, non-submitting dry-run, and commit:

```bash
git commit -am "Add guarded full diagnostic submission workflow"
```

### Task 7: Immutable Stage D source bundle

**Creates:** `m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh`.

- [ ] Write tests for clean worktree, exact expected HEAD, regular-file allowlist, `SOURCE_COMMIT_SHA.txt`, archive root, no `.git`, bytecode, saved models, or evidence.
- [ ] Dedicated archive name:

```text
EV-GNN-full-infrastructure-diagnostics-eval30-<commit>.tar.gz
```

- [ ] Required success marker: `SOURCE_BUNDLE_OK`.
- [ ] Create a local source bundle and verify its portable sidecar.
- [ ] Run focused tests, `bash -n`, and commit:

```bash
git commit -am "Add immutable full diagnostic source bundle"
```

### Task 8: Protocol and pre-M3 completion gate

**Creates:** `docs/full_per_infrastructure_diagnostics_eval30_protocol.md`.

- [ ] Document source creation, M3 transfer, extraction, environment, eight mappings, 40 checkpoint mappings, dry-run, real submission, exact ID capture, monitoring, same-ID reducer recovery, M3 validation, local download, evidence directory, acceptance gates, failure handling, claim boundaries, and smoke retirement.
- [ ] Final external evidence name:

```text
EVGNN_per_infrastructure_diagnostics_40checkpoints_eval30_job<ARRAY_JOB_ID>_<YYYYMMDD>
```

- [ ] Add protocol consistency tests for `8/8`, `40/40`, `1,200/1,200`, schema v3, `afterok`, exact accounting fields, no retraining, no latest-job discovery, and smoke job `58622672` lineage.
- [ ] Run focused Stage D tests:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py -q
```

- [ ] Run full repository tests:

```bash
python -m pytest tests -q -p no:cacheprovider
```

- [ ] Run compile and shell syntax checks.
- [ ] Run eight array dry-runs, reducer dry-run, submit-helper dry-run, and source-bundle checksum.
- [ ] Verify clean scope and no raw evidence-like files.
- [ ] Commit protocol and create implementation PR titled `Add full per-infrastructure deterministic eval30 workflow`.
- [ ] Do not merge or submit M3 until the complete diff and verification evidence are reviewed.

## Self-review

The plan covers eight tasks, 40 checkpoints, 1,200 episodes, deterministic pairing, schema v3, canonical/mapping/service reconciliation, exact accounting, immutable source provenance, evidence naming, smoke lineage, and the no-training boundary. No unresolved placeholders remain.
