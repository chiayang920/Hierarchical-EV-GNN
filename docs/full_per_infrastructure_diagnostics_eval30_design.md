# Full Per-Infrastructure Diagnostics Eval30 — Stage D Design

Date: 2026-07-29
Status: Design freeze candidate
Approved execution topology: Option 1 — eight scale-by-algorithm tasks

## 1. Purpose

This stage performs the complete per-infrastructure diagnostic re-evaluation of the existing controlled multiscale checkpoints from M3 formal job `58513929`.

It does not retrain any model. It reuses the existing 40 formal checkpoints and extends the one-episode smoke into the full deterministic eval30 mechanism analysis.

The stage tests whether the hierarchical actor's lower global action saturation reflects physically meaningful allocation across transformers and chargers, or primarily reflects lower charging intensity and reduced service margin.

## 2. Evidence lineage

Predecessor smoke:

- array job: `58622672`
- reducer job: `58622673`
- source commit: `6e5fd7f4c6d1f8cb5647263b37b3db1ba02f015c`
- scope: 4 scales × 2 algorithms × 1 selected checkpoint × 1 deterministic episode
- result: infrastructure-diagnostic schema v3 and end-to-end M3 workflow passed

The smoke is a workflow and mechanism sanity check. It is not the final diagnostic evidence.

## 3. Input matrix

Existing formal job `58513929` contains:

- four scales: `25cp`, `100cp`, `500cp`, `1000cp`;
- two algorithms: `actiongnn`, `hierarchical`;
- five training seeds: `0`, `1`, `2`, `3`, `4`;
- 40 checkpoint packages in total;
- no new training.

Formal task-ID mapping:

| Scale | ActionGNN task IDs | Hierarchical task IDs |
|---|---|---|
| 25CP | 0–4 | 5–9 |
| 100CP | 10–14 | 15–19 |
| 500CP | 20–24 | 25–29 |
| 1000CP | 30–34 | 35–39 |

## 4. Approved Slurm topology

Eight top-level array tasks:

| Stage D task | Scale | Algorithm | Formal checkpoint task IDs |
|---:|---|---|---|
| 0 | 25CP | ActionGNN | 0–4 |
| 1 | 25CP | Hierarchical | 5–9 |
| 2 | 100CP | ActionGNN | 10–14 |
| 3 | 100CP | Hierarchical | 15–19 |
| 4 | 500CP | ActionGNN | 20–24 |
| 5 | 500CP | Hierarchical | 25–29 |
| 6 | 1000CP | ActionGNN | 30–34 |
| 7 | 1000CP | Hierarchical | 35–39 |

Each top-level task evaluates five checkpoints sequentially.

Each checkpoint receives 30 deterministic episodes.

Total workload:

```text
8 tasks × 5 checkpoints × 30 episodes = 1,200 episodes
```

This topology preserves the established `8/8 task` operational gate while validating all `40/40 checkpoints`.

## 5. Deterministic episode protocol

Scale-specific offsets remain paired across algorithms:

| Scale | Eval seed offset |
|---|---:|
| 25CP | 710000 |
| 100CP | 720000 |
| 500CP | 730000 |
| 1000CP | 740000 |

Episode identity:

```text
episode_seed = eval_seed_offset + training_seed + episode_index
```

where `episode_index` is `0–29`.

Training seed remains the statistical inference unit. The 30 deterministic episodes are within-seed replicates and must not be treated as 30 independent samples.

## 6. Per-task execution

For each of the five formal checkpoints assigned to a task:

1. resolve exactly one formal task package from job `58513929`;
2. validate package identity, checksum, scale, algorithm, training seed, config, checkpoint, eval evidence, and source provenance;
3. stage `model.best`, immutable config, and required metadata into a job-scoped scratch directory;
4. invoke `evaluate_td3_gnn_infrastructure_diagnostics.py` with:
   - explicit scale;
   - explicit config;
   - matching algorithm;
   - matching training seed;
   - `eval_episodes=30`;
   - deterministic mode;
   - zero exploration noise;
   - validated maximum episode length;
   - scale-specific eval seed offset;
5. validate all generated schema-v3 CSVs;
6. reconcile diagnostic episode rows against canonical eval30 rows;
7. reconcile charger and transformer service/energy totals;
8. package the seed output only after validation passes.

A top-level task succeeds only when all five seed/checkpoint groups pass.

## 7. Output structure

Job-scoped run root:

```text
<run_root>/job<array_job_id>/task<task_id>/
```

Per seed:

```text
seed<training_seed>/
├── formal_input/
├── diagnostics/
│   ├── episode_diagnostics.csv
│   ├── transformer_diagnostics.csv
│   ├── charger_diagnostics.csv
│   └── seed_summary_diagnostics.csv
├── validation/
├── runtime_metadata/
└── logs/
```

Each top-level task publishes one validated task package containing exactly five seed groups.

Final reducer output contains:

- eight task packages;
- task inventory;
- 40-checkpoint inventory;
- 1,200-episode inventory;
- runtime accounting;
- canonical reconciliation summary;
- service reconciliation summary;
- schema inventory;
- missing-field inventory;
- failure and warning inventories;
- seed-level paired summary;
- complete-file checksum manifest;
- final evidence archive and portable sidecar.

## 8. Acceptance gates

Operational gates:

- 8/8 top-level tasks `COMPLETED`;
- 8/8 task exit codes `0:0`;
- reducer parent, batch, and extern records `COMPLETED`, `0:0`;
- 8/8 task packages readable and checksum-valid;
- current array job ID used explicitly throughout;
- no latest-job, job-name, or unscoped accounting discovery.

Scientific-data gates:

- 40/40 checkpoint groups present;
- 40/40 formal mappings valid;
- 1,200/1,200 unique episode keys present;
- no duplicate `(scale, algorithm, training_seed, episode_index)` key;
- 30/30 episodes for every checkpoint;
- diagnostic schema v3 everywhere;
- every canonical reconciliation row passes;
- every mapping-validation row passes;
- every service reconciliation row passes;
- no missing required field;
- no silent blank, NaN, or Inf in required metrics;
- no unexpected non-empty stderr;
- no failure-manifest entry;
- output archive and sidecar validate locally and on M3.

Feasibility gates:

- 1000CP completes within the approved wall-time;
- MaxRSS remains within allocation;
- output size is manageable for project storage and local audit;
- no evidence of pathological filesystem or reducer scaling.

## 9. Initial M3 resource envelope

The smoke used less than 1 GB peak task memory and approximately 29–39 seconds for one checkpoint and one episode, including setup.

The initial Stage D request should therefore be conservative but lower than the smoke's 64 GB allocation:

```text
CPUs per task: 4
Memory per task: 32 GB
Wall-time: 6 hours
Array concurrency: determined by project quota
Device: CPU
```

The reducer should request a separate small allocation because it validates and packages evidence but performs no model evaluation.

These values are pre-submission design values. Dry-run and source-package verification remain mandatory.

## 10. Failure and recovery policy

- Any seed/checkpoint failure fails its top-level task.
- Partial task output remains job-scoped and is never published as a valid task package.
- Final package publication uses temporary archive → validation → atomic rename.
- Reducer submission uses `afterok:<exact_array_job_id>`.
- Reducer recovery must reuse the exact original array job ID.
- Rerunning one top-level task requires a new explicitly recorded array identity or a documented recovery protocol; stale task packages cannot be mixed.
- Failed evidence remains separate from successful evidence.

## 11. Evidence-directory policy

The Stage D evidence must not be added to the smoke directory.

The final external evidence directory should be:

```text
EVGNN_per_infrastructure_diagnostics_40checkpoints_eval30_job<ARRAY_JOB_ID>_<YYYYMMDD>
```

This name is semantic, records the complete scope, and does not rely on internal stage labels such as `C3.2` or `Stage D`.

Required subdirectories:

```text
final_evidence/
execution_provenance/
source_provenance/
verification_reports/
```

The README must record that it supersedes smoke job `58622672`.

## 12. Smoke-retirement gate

The active-workspace smoke directory may be deleted only after:

1. Stage D final bundle and sidecar pass independent audit;
2. 8/8 tasks and 40/40 checkpoints pass;
3. all 1,200 episodes are accounted for;
4. Stage D evidence exists in two independent storage locations;
5. the repository results note records smoke job `58622672` as predecessor evidence;
6. the Stage D README records the smoke lineage;
7. no thesis or review document still points only to the smoke archive.

The smoke may then be removed from the active `Research_project` directory. A compact compressed copy may remain in cold backup until thesis submission.

## 13. Repository implementation scope

Expected new files:

```text
m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm
m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm
m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh
scripts/validate_full_infrastructure_diagnostic_eval30.py
tests/test_full_infrastructure_diagnostic_eval30_workflow.py
docs/full_per_infrastructure_diagnostics_eval30_protocol.md
```

Existing evaluator and schema-v3 diagnostic utilities should be reused unless a failing test proves a necessary defect fix.

No actor, critic, replay buffer, reward, simulator, checkpoint, training logic, or formal job output may be modified.

## 14. TDD and verification strategy

Implementation begins with failing tests for:

- exact eight-task mapping;
- exact five-seed mapping per task;
- exact formal task IDs;
- 30-episode completeness;
- duplicate and missing episode keys;
- wrong scale, algorithm, seed, or config;
- missing or ambiguous formal package;
- checkpoint and checksum mismatch;
- schema-version mismatch;
- canonical and service reconciliation failures;
- missing required fields;
- NaN and Inf handling;
- stale package isolation;
- unsafe archive paths;
- reducer package membership;
- explicit array-ID accounting;
- reducer recovery identity;
- source commit mismatch;
- atomic final publication.

Required verification before M3:

- focused red/green test evidence;
- full repository test suite;
- Python compile checks;
- Bash syntax checks;
- all eight task dry-runs;
- reducer dry-run;
- submit-helper dry-run;
- canonical source-package verification;
- clean repository;
- documentation review;
- pull-request review and merge.

## 15. Arm C boundary

Stage D remains a measurement stage and does not implement Research Arm C.

After Stage D seed-level analysis, the approved Arm C architecture family is:

```text
Graph / CPO → Transformer allocation → EV allocation
```

This removes the charger-budget layer from the full hierarchical actor.

The detailed Arm C implementation contract and training plan must be frozen only after Stage D identifies whether charger-level hierarchy provides incremental infrastructure alignment.
