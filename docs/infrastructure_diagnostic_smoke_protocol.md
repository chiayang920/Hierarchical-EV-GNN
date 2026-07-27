# Infrastructure Diagnostic Smoke Protocol

Stage C3.2A defines a deterministic infrastructure-diagnostic smoke workflow for the M3 formal evidence chain. The purpose is to prove that the schema-v3 diagnostic evaluator, formal checkpoint reuse, package provenance, Slurm runtime capture, and evidence bundling path work across the four PublicPST scales and both algorithms. This workflow is an infrastructure diagnostic only. It does not make training-quality, performance-improvement, or statistical generalisation claims.

## Claim Boundary

The smoke workflow runs one deterministic diagnostic episode per selected trained checkpoint. It checks row identity, schema version, topology row counts, action-domain invariants, service reconciliation, canonical episode-0 reconciliation, source provenance, package checksums, Slurm accounting, and final bundle integrity.

It does not retrain models, tune hyperparameters, compare learning curves, or replace the existing controlled multiscale formal evaluation.

## Eight-Task Mapping

| Task | Scale | Algorithm | Training seed | Formal task | Episode seed |
| --- | --- | --- | --- | --- | --- |
| 0 | 25cp | actiongnn | 0 | 0 | 710000 |
| 1 | 25cp | hierarchical | 0 | 5 | 710000 |
| 2 | 100cp | actiongnn | 0 | 10 | 720000 |
| 3 | 100cp | hierarchical | 0 | 15 | 720000 |
| 4 | 500cp | actiongnn | 0 | 20 | 730000 |
| 5 | 500cp | hierarchical | 0 | 25 | 730000 |
| 6 | 1000cp | actiongnn | 0 | 30 | 740000 |
| 7 | 1000cp | hierarchical | 0 | 35 | 740000 |

All model inputs come from formal job `58513929` `model.best` checkpoints.

## Immutable Source Provenance

Create the source bundle locally from the physical repository root on `main` with a clean tracked worktree and the recorded commit SHA. The helper writes a versioned top-level source directory, `SOURCE_COMMIT_SHA.txt`, and a sibling SHA-256 file. It refuses existing archive outputs and scans for prohibited members.

The M3 extracted source root must contain `SOURCE_COMMIT_SHA.txt`. The array, reducer, and submit helper require the expected source commit through `EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT` in real mode.

No Git commands are required or permitted on M3 for this workflow.

## Formal Package Modes

The preferred source is the individual job `58513929` task package:

`m3_controlled_multiscale_formal_<scale>_<algorithm>_seed0_job58513929_task<formal_task>.tar.gz`

If the individual package is absent, the resolver falls back to exactly one nested task package inside the complete formal evidence bundle. Zero or multiple fallback matches fail. Each final task package records whether it used `individual_task_package` or `complete_bundle_nested_task_package`.

## M3 Paths

Default run root:

`/scratch2/fr57/cche0357/EV-GNN_runs/infrastructure_diagnostic_smoke`

Default output root:

`/projects/fr57/cche0357/EV-GNN_outputs`

Default source archive target:

`/projects/fr57/cche0357/EV-GNN_sources`

Override roots with the documented `EV_GNN_DIAGNOSTIC_SMOKE_*` environment variables when needed.

## Source Bundle Creation

Local dry-run:

```bash
EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_DRY_RUN=1 \
bash m3_jobs/create_infrastructure_diagnostic_smoke_source_bundle.sh
```

Real creation must be done from `main` at the recorded commit:

```bash
EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_EXPECTED_HEAD_SHA=<sha> \
bash m3_jobs/create_infrastructure_diagnostic_smoke_source_bundle.sh
```

The helper prints the exact transfer, checksum, and extraction commands.

## Transfer And Extraction

Transfer the source archive and SHA-256 file to M3, verify with `sha256sum -c`, then extract into `/scratch2/fr57/cche0357/EV-GNN_sources`. The extraction command printed by the helper refuses to overwrite an existing source directory.

## Submit Helper Dry-Run

From the extracted source root:

```bash
EV_GNN_DIAGNOSTIC_SMOKE_SUBMIT_DRY_RUN=1 \
EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT=<sha> \
bash m3_jobs/submit_infrastructure_diagnostic_smoke_workflow.sh
```

Dry-run does not require M3 formal package files and does not call `sbatch`. It prints the exact task mapping, array `sbatch` command, reducer `sbatch --dependency=afterok:<array_job_id>` command, monitoring commands, tar validation command, and local download command.

## Real Submission

Real mode requires:

- physical extracted source root
- `SOURCE_COMMIT_SHA.txt`
- `EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT`
- equality between the source file and expected SHA
- runtime scripts present
- all eight formal packages resolvable and valid

Then the helper submits the array job and the reducer job with `afterok:<array_job_id>`.

## Monitoring

Use the printed `squeue` command while jobs are active. Use the printed `sacct` command after completion to inspect `JobIDRaw`, `State`, `ExitCode`, `ElapsedRaw`, `AllocCPUS`, `MaxRSS`, and `TotalCPU`.

The reducer retries boundedly because accounting records may be delayed.

## Reducer Gates

The reducer requires:

- exactly 8 task packages
- exactly 8 Slurm stdout logs
- exactly 8 Slurm stderr logs
- task IDs 0 through 7 exactly once
- 32 diagnostic CSV files inside the packages
- 8 `task_validation.json` files
- 8 `canonical_reconciliation.csv` files
- schema version 3 throughout
- all canonical reconciliation rows passing
- all service and action contracts passing
- all package and complete-bundle checksums passing
- Slurm parent array tasks completed with `ExitCode=0:0`
- `MaxRSS` and `TotalCPU` from the parent row or matching `.batch` step
- no checkpoint/model/optimizer bytes in task or complete evidence packages

Fatal log signatures fail the reducer. The known EV2Gym/pkg_resources deprecation warning is allowed. Other warnings are recorded in `summaries/warning_inventory.csv`.

## Final Evidence

The final bundle is:

`infrastructure_diagnostic_smoke_complete_evidence_job<array_job_id>.tar.gz`

It contains the eight validated task packages, Slurm logs, reducer summaries, runtime metadata, complete file list, and complete SHA-256 manifest. It does not duplicate the extracted diagnostic CSVs outside the task packages.

## Local Verification

After downloading the final bundle, verify it locally:

```bash
python scripts/validate_infrastructure_diagnostic_smoke.py \
  validate-complete-bundle \
  --bundle infrastructure_diagnostic_smoke_complete_evidence_job<array_job_id>.tar.gz
```

Also keep the M3 checksum result and the local SHA-256 result with the evidence notes.

## Failure Recovery

If an array task fails, inspect the task Slurm stderr, packaged stderr, and evaluator timing log. Fix only infrastructure issues in a new source commit. Do not reuse a partially produced task directory unless the explicit safe clean override is set for a rerun.

If the reducer fails due to missing accounting, wait for sacct records and rerun the reducer against the same array job ID. If it fails due to package integrity, source provenance, schema, canonical reconciliation, service reconciliation, action contracts, or checkpoint leakage, treat the evidence as invalid and rerun from a corrected source bundle.

Training is explicitly prohibited in this workflow.
