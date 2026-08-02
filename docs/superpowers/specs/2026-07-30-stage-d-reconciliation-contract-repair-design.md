# Stage D Canonical-Reconciliation Contract Repair Design

Date: 2026-07-30
Status: design freeze for a future repair implementation
Base commit inspected: `12ae8768e72bd9fd5f94980a4d0538e06835f861`
Target branch: `design/stage-d-reconciliation-contract-repair`

## Status and scope

This document specifies the repair for the Stage D canonical-reconciliation contract. It is design documentation only. It does not authorize implementation, M3 resubmission, GitHub push, pull-request creation, model changes, retraining, failed-task-only reruns, or changes to scientific claims.

The inspected design surface is:

- `evaluate_td3_gnn.py`
- `evaluate_td3_gnn_infrastructure_diagnostics.py`
- `utils/infrastructure_diagnostics.py`
- `scripts/validate_full_infrastructure_diagnostic_eval30.py`
- `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm`
- `m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm`
- `tests/test_infrastructure_diagnostics.py`
- `tests/test_full_infrastructure_diagnostic_eval30_workflow.py`
- `docs/full_per_infrastructure_diagnostics_eval30_design.md`
- `docs/superpowers/plans/2026-07-29-full-per-infrastructure-diagnostics-eval30.md`

The repair must preserve diagnostic schema v3 episode, charger, transformer, and seed-summary CSV columns. New reconciliation semantics are versioned separately with the exact field and value `reconciliation_contract_version=2`.

## Evidence and root-cause statement

The formal training evidence remains valid and untouched:

- formal array job: `58513929`
- formal reducer job: `58513930`
- checkpoint count: `40` `model.best` checkpoints
- matrix: `4` scales x `2` algorithms x `5` training seeds
- training horizon: `50,000` timesteps
- historical canonical eval30 evidence: `1,200` episodes

The failed Stage D engineering attempt is calibration and failure evidence only:

- array job: `58656380`
- reducer job: `58656381`
- source commit: `12ae8768e72bd9fd5f94980a4d0538e06835f861`

Job `58656380` must never be used as scientific Stage D evidence and must not be made to pass by post-hoc tolerance relaxation.

The observed mismatch audit was:

- `identity_mismatch=no`
- `trajectory_mismatch=yes`
- `action_summary_only_mismatch=no`

Global mismatch counts were:

- `energy_tracking_error=5`
- `episode_reward=15`
- `global_action_fraction_at_max_all_slots=91`
- `power_tracker_violation=9`
- `total_energy_charged=4`
- `total_transformer_overload=1`
- `tracking_error=15`

Root cause: Stage D currently compares a fresh diagnostic policy re-execution against the historical formal canonical CSV as a hard floating-point gate. That asks two independent executions to be near-bitwise equivalent. The most brittle field is `global_action_fraction_at_max_all_slots`, compared against canonical `action_fraction_at_max` with a fixed absolute tolerance of `1e-4`. This metric is a thresholded count divided by the total action-decision denominator, so a small near-threshold action drift can flip a discrete count before division. The fixed fraction tolerance is scale-dependent: one count is `1 / (episode_steps * action_dim)`, so the same count difference has different fractional size at 25CP, 100CP, 500CP, and 1000CP, and the tolerance silently permits different numbers of count changes per scale.

At 1000CP, independent re-execution also produced very small numerical drift in reward, tracking, and energy metrics while identity, seed, episode length, done, and served-EV fields remained consistent. That is historical cross-run drift, not proof that the fresh Stage D diagnostic trajectory is internally inconsistent.

## Current data flow

The current Stage D array runner performs this flow per seed:

```text
historical formal package from job 58513929
    -> copy the scale, algorithm, and training-seed-specific
       historical eval30 CSV under eval/ to seed*/canonical/complete_eval30.csv
    -> copy model.best checkpoint members and immutable config
    -> run evaluate_td3_gnn_infrastructure_diagnostics.py
       for a fresh 30-episode diagnostic policy re-execution
    -> run prepare-seed-validation with:
       historical canonical CSV + fresh diagnostic CSVs
    -> hard floating-point reconciliation
    -> seed/package publication only if validation passes
```

`prepare_seed_validation_files()` currently builds canonical reconciliation rows, checks for any failed status, raises `ValueError("canonical reconciliation failed")`, and only then would write `validation/canonical_reconciliation.csv`. The result is:

```text
detect mismatch
    -> raise generic ValueError
    -> lose detailed reconciliation CSV evidence
```

The reducer later assumes all seed packages contain passing canonical and service reconciliation files, then aggregates them into complete-bundle summaries.

## Rejected alternatives

Increasing the historical tolerance is rejected. It would only tune the failed job `58656380` around observed drift and would keep historical cross-run comparison as the hard metric gate.

Treating job `58656380` as scientific evidence is rejected. It is engineering calibration and failure evidence only.

Replacing fresh Stage D diagnostic values with historical formal canonical values is rejected. Historical values are provenance and drift-audit inputs; they are never the authoritative Stage D metric source.

Changing actor, critic, replay buffer, reward, simulator, state, action interface, checkpoint files, training algorithm, seed schedule, task mapping, 30-episode protocol, or reducer accounting totals is rejected.

Changing schema-v3 episode, charger, transformer, or seed-summary CSV columns is rejected unless a later implementation review proves an unavoidable defect outside this repair.

A failed-task-only scientific rerun is rejected. Recovery after implementation is a clean full rerun with a new exact array job ID and a reducer submitted after all eight tasks succeed.

## Proposed data flow

The repaired flow separates same-pass hard validation from historical drift audit:

```text
historical formal canonical CSV from job 58513929
    -> immutable provenance input
    -> exact historical identity gate
    -> historical floating drift audit only

fresh Stage D diagnostic policy re-execution
    -> one environment trajectory
    -> one reward stream
    -> one stats object stream
    -> one mapped-action stream
        -> schema-v3 infrastructure diagnostics
        -> same-pass canonical-compatible eval30 rows
            -> hard same-pass metric reconciliation

all reconciliation rows and summary JSON
    -> atomically written before hard-gate error is raised
```

The diagnostic evaluator must emit the exact new file:

```text
diagnostics/same_pass_canonical_eval30.csv
```

The validator must emit these exact new validation files per seed:

```text
validation/same_pass_canonical_reconciliation.csv
validation/historical_canonical_drift.csv
runtime_metadata/reconciliation_summary.json
```

`validation/service_reconciliation.csv` and `validation/mapping_validation.json` remain hard validation evidence. Under contract version 2, task-package and reducer validation must require the new reconciliation evidence in addition to the existing service and mapping evidence.

## File-level component responsibilities

`evaluate_td3_gnn.py` remains the canonical source of pure canonical evaluation helpers. The repair should reuse `action_diagnostics_from_actions()` and `build_csv_rows()` so canonical-compatible action aggregation keeps the existing float32 action semantics and CSV row construction.

`evaluate_td3_gnn_infrastructure_diagnostics.py` remains the fresh diagnostic evaluator. During each episode it must retain the same mapped-action stream, reward stream, stats object, episode identity, and done/step data used for infrastructure diagnostics. After the 30 episodes complete, it must write the existing schema-v3 diagnostic CSVs unchanged and also write `diagnostics/same_pass_canonical_eval30.csv` from the same in-memory episode records.

`utils/infrastructure_diagnostics.py` remains responsible for schema-v3 diagnostic aggregation, infrastructure-level action summaries, service summaries, and same-execution episode/charger/transformer consistency. Its existing schema-v3 column constants must not be changed for this repair.

`scripts/validate_full_infrastructure_diagnostic_eval30.py` owns the Stage D validation contract. It must load historical canonical rows, same-pass canonical rows, diagnostic rows, service rows, mapping metadata, package provenance, and runtime metadata. It must build all reconciliation rows first, atomically write evidence files, then evaluate hard gates and raise a precise error only after evidence exists.

`m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm` remains the array runner. Its future implementation change is limited to using the repaired evaluator/validator evidence contract and packaging the new files after all five seeds pass.

`m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm` remains the exact-ID reducer wrapper. Its future implementation change is limited to requiring and summarising the new reconciliation evidence.

`tests/test_infrastructure_diagnostics.py` and `tests/test_full_infrastructure_diagnostic_eval30_workflow.py` must receive the failing tests listed below before implementation code changes.

## Exact validation contracts

Fresh Stage D diagnostic CSVs are the authoritative Stage D metric source. Same-pass canonical-compatible rows are a validation view over that same execution. Historical formal canonical rows are immutable provenance and drift-audit inputs.

Hard gates:

- historical identity gate: exact
- same-pass metric reconciliation gate: hard fail on disagreement
- service reconciliation gate: existing hard gate remains
- mapping validation gate: existing hard gate remains
- package membership and reducer accounting gates: existing hard gates remain

Audit-only gates:

- historical floating drift
- historical saturation count drift

Historical floating drift must be written and classified, but it is not a scientific result and does not replace fresh Stage D metrics.

### Historical identity hard gate

The following must match exactly against the historical formal record and Stage D package provenance:

- `algorithm`
- `training_seed`
- `episode_index`
- `episode_seed`
- `episode_steps`
- `done`
- `total_ev_served`
- checkpoint provenance and checksums for the staged `model.best` checkpoint members
- configuration provenance and checksum for the staged formal config
- `formal_job_id=58513929`
- exact Stage D source commit recorded by the immutable source bundle, task package, and reducer expectation

Any identity mismatch fails the seed and prevents task package publication. Historical identity failures must appear in `runtime_metadata/reconciliation_summary.json` under `failure_categories`.

### Same-pass metric hard gate

The same-pass canonical-compatible episode rows and the schema-v3 episode diagnostic rows must reconcile for:

- `episode_reward`
- `tracking_error`
- `energy_tracking_error`
- `power_tracker_violation`
- `total_energy_charged`
- `total_energy_discharged`
- `average_user_satisfaction`
- `energy_user_satisfaction`
- `total_transformer_overload`
- `action_mean`
- `action_fraction_at_max`
- active and nonzero action count summaries

The comparison is same-execution only. The implementation must not introduce a larger historical tolerance to pass this gate. Values that overlap between the same-pass canonical view and the diagnostic view must be derived from the same in-memory episode record or the same pure helper output before CSV serialization. Disagreement means an implementation or aggregation defect and must fail with category `same_pass_metric_mismatch`.

## Same-pass canonical-compatible output contract

Exact path:

```text
diagnostics/same_pass_canonical_eval30.csv
```

Purpose: validation-only canonical-compatible output from the same policy execution used for infrastructure diagnostics.

Rows:

- exactly 30 `row_type=episode` rows
- exactly one `row_type=summary` row
- `episode_index` values `0` through `29`
- `episode_seed` values from the existing Stage D seed schedule

Column contract:

- generated through `evaluate_td3_gnn.build_csv_rows()`
- includes the existing canonical `REQUIRED_COLUMNS`
- includes `row_type`
- includes scalar stats keys discovered through `evaluate_td3_gnn.scalar_stats()`
- uses canonical helper action fields `action_mean`, `action_std`, `action_min`, `action_max`, `action_fraction_zero`, `action_fraction_at_max`, and `active_action_count_mean`

Action aggregation contract:

- the mapped-action stream is converted using the existing canonical float32 action aggregation semantics
- the same `max_action` and action tolerance used for diagnostics are passed through only where the current helpers already expose that parameter
- formulas are not duplicated in the diagnostic evaluator

Hard reconciliation output:

```text
validation/same_pass_canonical_reconciliation.csv
```

Exact columns:

```text
reconciliation_contract_version
episode_index
field
comparison_type
same_pass_canonical_value
diagnostic_value
absolute_difference
relative_difference
same_pass_count
diagnostic_count
total_action_decision_denominator
status
failure_category
```

Allowed `status` values are `pass` and `fail`. A failed row uses `failure_category=same_pass_metric_mismatch`.

## Historical drift-audit contract

Exact path:

```text
validation/historical_canonical_drift.csv
```

This file must always be written for a seed validation attempt once the historical canonical, same-pass canonical, and diagnostic seed inputs are readable. It must still be written when another hard gate fails.

Exact columns:

```text
reconciliation_contract_version
scale
algorithm
training_seed
formal_task_id
episode_index
episode_seed
field
historical_source_label
stage_d_source_label
historical_value
stage_d_same_pass_value
absolute_difference
relative_difference
classification
historical_at_max_count
stage_d_at_max_count
total_action_decision_denominator
count_difference
fraction_difference
```

Required source labels:

- `historical_source_label=formal_job_58513929_canonical_eval30`
- `stage_d_source_label=stage_d_same_pass_canonical_eval30`

Allowed `classification` values:

- `exact_match`
- `historical_float_drift`
- `historical_saturation_count_drift`
- `historical_identity_match`
- `historical_identity_mismatch`
- `not_comparable`

Historical floating differences must be reported with field, episode index, historical value, Stage D same-pass value, absolute difference, relative difference, classification, and source labels. These rows are audit evidence only. They do not determine the authoritative Stage D metrics and do not make job `58656380` pass.

## Count-based saturation audit

For `action_fraction_at_max`, `validation/historical_canonical_drift.csv` must populate:

- `historical_at_max_count`
- `stage_d_at_max_count`
- `total_action_decision_denominator`
- `count_difference`
- `fraction_difference`

The denominator is the total number of mapped action decisions in the episode:

```text
total_action_decision_denominator = episode_steps * mapped_action_dimension
```

The mapped action dimension comes from the same-pass execution. Because `episode_steps` is an exact historical identity field, any step mismatch fails before the count audit is interpreted.

The historical count is reconstructed from the historical fraction and denominator for audit reporting. The Stage D count comes from the same-pass mapped-action threshold count. The stored fraction remains available for research reporting; the count audit supplements it and does not silently redefine or replace the metric.

The design explicitly rejects a fixed `1e-4` fraction tolerance as the saturation repair. A fraction tolerance is brittle because the smallest possible fraction step is `1 / denominator`, and the denominator changes by scale and by episode length. It also hides the fact that saturation is a discrete threshold count before it is a floating fraction.

## Failure observability and atomic write order

The repaired validation order is:

```text
read inputs
    -> build same-pass reconciliation rows
    -> build historical drift rows
    -> build service reconciliation rows without raising
    -> build mapping payload
    -> build reconciliation summary payload
    -> atomically write all reconciliation CSVs and JSON files
    -> evaluate hard gates
    -> raise precise error only after evidence exists
```

Atomic writes use temporary files in the destination directory followed by `os.replace()`. The validator must not leave a final filename containing partial content.

Exact summary path:

```text
runtime_metadata/reconciliation_summary.json
```

Exact JSON fields:

```text
reconciliation_contract_version
status
hard_gate_status
historical_identity_status
same_pass_metric_status
service_reconciliation_status
mapping_validation_status
historical_drift_audit_status
evidence_write_status
failure_categories
hard_failure_count
historical_identity_failure_count
same_pass_metric_failure_count
service_reconciliation_failure_count
mapping_validation_failure_count
same_pass_canonical_rows
same_pass_reconciliation_rows
historical_drift_rows
historical_float_drift_count
historical_saturation_count_drift_count
service_reconciliation_rows
mapping_validation_present
source_labels
```

Allowed summary values:

- `reconciliation_contract_version`: JSON number `2`
- `status`: `ok` or `failed`
- `hard_gate_status`: `pass` or `fail`
- `historical_identity_status`: `pass` or `fail`
- `same_pass_metric_status`: `pass` or `fail`
- `service_reconciliation_status`: `pass` or `fail`
- `mapping_validation_status`: `pass` or `fail`
- `historical_drift_audit_status`: `written`
- `evidence_write_status`: `complete`
- `failure_categories`: array containing zero or more of `historical_identity_mismatch`, `same_pass_metric_mismatch`, `service_reconciliation_mismatch`, `mapping_validation_mismatch`, `schema_contract_mismatch`, `input_contract_mismatch`

The raised error message after a failed hard gate must include the failing categories, for example:

```text
Stage D reconciliation contract failed: historical_identity_mismatch,same_pass_metric_mismatch
```

## Schema/versioning decision

Diagnostic schema v3 remains stable. The following existing CSV headers must not change during this repair:

- `diagnostics/episode_diagnostics.csv`
- `diagnostics/charger_diagnostics.csv`
- `diagnostics/transformer_diagnostics.csv`
- `diagnostics/seed_summary_diagnostics.csv`

The new validation semantics are versioned separately:

```text
reconciliation_contract_version=2
```

This field must appear in:

- `validation/same_pass_canonical_reconciliation.csv`
- `validation/historical_canonical_drift.csv`
- `runtime_metadata/reconciliation_summary.json`
- task-level validation metadata that summarizes seed validation evidence
- complete-bundle validation metadata that summarizes reducer evidence

## Package and reducer contract impact

Each seed package must include the new evidence files:

```text
seed0/diagnostics/same_pass_canonical_eval30.csv through seed4/diagnostics/same_pass_canonical_eval30.csv
seed0/validation/same_pass_canonical_reconciliation.csv through seed4/validation/same_pass_canonical_reconciliation.csv
seed0/validation/historical_canonical_drift.csv through seed4/validation/historical_canonical_drift.csv
seed0/runtime_metadata/reconciliation_summary.json through seed4/runtime_metadata/reconciliation_summary.json
```

Task-package validation must require:

- the existing schema-v3 diagnostic CSVs
- `diagnostics/same_pass_canonical_eval30.csv`
- `validation/same_pass_canonical_reconciliation.csv` with all hard statuses passing
- `validation/historical_canonical_drift.csv` present and readable
- `runtime_metadata/reconciliation_summary.json` with `status=ok`, `hard_gate_status=pass`, and `reconciliation_contract_version=2`
- `validation/service_reconciliation.csv` with all existing service statuses passing
- `validation/mapping_validation.json` with existing mapping status passing
- no checkpoint files in final packages
- empty seed and task stderr logs

Reducer validation must require all 40 seed reconciliation summaries. The complete evidence bundle must include reducer-level summaries:

```text
summaries/same_pass_canonical_reconciliation_summary.csv
summaries/historical_canonical_drift_summary.csv
summaries/reconciliation_summary_inventory.csv
```

The reducer must preserve existing `8/8` task, `40/40` checkpoint, and `1,200/1,200` episode accounting totals. Reducer accounting remains exact-ID based and must not use latest-job discovery.

## Test strategy

Implementation must start by adding failing tests before production changes. The required failing tests are:

1. A one-count historical saturation difference does not become a same-pass trajectory failure.
2. Same-pass action-summary mismatch hard-fails.
3. Same-pass reward/tracking mismatch hard-fails.
4. Historical identity mismatch hard-fails.
5. Historical floating drift is written and classified.
6. Reconciliation evidence exists even when a hard gate fails.
7. Historical saturation count and denominator reconstruct the stored fraction.
8. Existing schema-v3 CSV headers remain unchanged.
9. Task-package validation requires the new validation evidence.
10. Reducer includes all required reconciliation summaries.
11. Existing formal task mapping and seed schedule remain unchanged.
12. Existing smoke/full-workflow regression tests remain valid or are explicitly migrated.
13. No checkpoint files leak into final evidence packages.
14. No M3 command is invoked by local tests.

Suggested placement:

- same-pass canonical helper and evaluator tests in `tests/test_infrastructure_diagnostics.py`
- Stage D validation, package, reducer, and no-M3 tests in `tests/test_full_infrastructure_diagnostic_eval30_workflow.py`

Focused local tests should use synthetic CSV/package fixtures only. They must not run heavy ML evaluation, retraining, `sbatch`, `sacct`, or network-dependent commands.

## Migration and clean-rerun strategy

After a future implementation is reviewed and merged:

1. create a new immutable source bundle from the reviewed source commit;
2. run local documentation, syntax, package, and dry-run gates;
3. run M3 dry-run gates without `sbatch`;
4. submit one new complete 8-task Stage D array;
5. record the new exact array job ID;
6. wait for all eight tasks to succeed;
7. run a new reducer against that exact array job ID;
8. validate the complete bundle and sidecar locally and on M3.

Do not reuse job `58656380` task packages in the scientific complete bundle. Do not perform a failed-task-only scientific rerun. Failed evidence remains failure evidence.

## Scientific claim boundaries

The authoritative Stage D scientific metrics come from fresh schema-v3 diagnostic rows generated by the clean full rerun after the repair is implemented and reviewed.

The historical canonical CSV from job `58513929` remains immutable formal provenance and drift-audit input. Historical floating drift is mandatory audit evidence, not a scientific result.

The repair must not change:

- actor architecture
- critic
- replay buffer
- reward
- simulator
- state
- action interface
- checkpoint files
- training algorithm
- formal job `58513929`
- seed schedule
- 30-episode protocol
- task mapping
- reducer accounting totals

No retraining is allowed.

## Explicit non-goals

This design does not implement the repair.

This design does not resubmit Stage D.

This design does not make job `58656380` pass.

This design does not modify scientific models, checkpoints, training data, reward definitions, or simulator behavior.

This design does not change schema-v3 diagnostic CSV headers.

This design does not introduce a tolerance chosen to fit the observed failed run.

This design does not create a pull request or push a branch.

## Acceptance criteria

The design phase is acceptable when:

- this spec is the only changed file before commit;
- no source, test, Slurm, model, checkpoint, or evidence files are modified;
- the spec has no unfinished markers;
- the spec freezes the authoritative metric source as fresh Stage D diagnostics;
- hard-fail versus audit-only comparisons are unambiguous;
- `diagnostics/same_pass_canonical_eval30.csv` is the exact same-pass canonical-compatible output name;
- `validation/historical_canonical_drift.csv` is the exact historical drift-audit output name;
- `runtime_metadata/reconciliation_summary.json` is the exact summary output name;
- `reconciliation_contract_version=2` is the exact new validation contract marker;
- the recovery path requires a clean full rerun with a new exact array job ID;
- no M3 action, GitHub write, implementation, or training is performed in this phase.
