# 500CP Formal Training Protocol

## Purpose

Run controlled 500CP 50k training for both canonical algorithms after the
500CP 10k pilots completed successfully:

- controlled baseline: `actiongnn`
- hierarchical actor: `hierarchical`

The Slurm script is:

```bash
m3_jobs/14_500cp_formal_train.slurm
```

It uses the canonical training entrypoint and the 500CP config:

```bash
python train_td3_gnn.py --algorithm actiongnn --config ./config_files/PublicPST_500.yaml
python train_td3_gnn.py --algorithm hierarchical --config ./config_files/PublicPST_500.yaml
```

This protocol prepares formal training artefacts only. Formal behavioural
claims still require controlled eval30 after training completes.

## Pilot Evidence Basis

The 500CP 10k runtime pilot array completed for both canonical algorithms:

| Array task | Algorithm | Training wall-time | Slurm elapsed | MaxRSS | CPU efficiency |
| --- | --- | ---: | ---: | ---: | ---: |
| `58415289_0` | `actiongnn` | `4380s` | `4398s` | `8.66G` | about `93.1%` |
| `58415289_1` | `hierarchical` | `5811s` | `5829s` | `6.24G` | about `92.4%` |

Linear 50k projections from the measured training wall-times are:

- baseline `actiongnn`: `4380s * 5 = 21900s`, about `06:05:00`
- `hierarchical`: `5811s * 5 = 29055s`, about `08:04:15`

These are runtime feasibility estimates, not behavioural evidence.

## Controlled Setup

- Entrypoint: `train_td3_gnn.py`
- Config: `./config_files/PublicPST_500.yaml`
- Device: `cpu`
- CPU count: `4`
- Memory: `64G`
- Wall-time request: `11:00:00`
- Training steps: `50000`
- Start timesteps: `1000`
- Eval frequency during training: `5000`
- Eval episodes during training: `5`
- Batch size: `64`
- Replay buffer size: `100000`
- WandB logging: disabled

The script records `EV_GNN_SOURCE_PROVENANCE` and `EV_GNN_SOURCE_MANIFEST` when
provided. It does not run `git pull` or require any M3 git command as a source
update mechanism.

## Array Mapping

The script supports 10 matched-seed tasks:

| Task ID | Algorithm | Seed |
| ---: | --- | ---: |
| `0` | `actiongnn` | `0` |
| `1` | `actiongnn` | `1` |
| `2` | `actiongnn` | `2` |
| `3` | `actiongnn` | `3` |
| `4` | `actiongnn` | `4` |
| `5` | `hierarchical` | `0` |
| `6` | `hierarchical` | `1` |
| `7` | `hierarchical` | `2` |
| `8` | `hierarchical` | `3` |
| `9` | `hierarchical` | `4` |

Equivalent mapping rule:

- tasks `0-4`: `actiongnn`, seed equals task ID
- tasks `5-9`: `hierarchical`, seed equals task ID minus `5`

## Seed0-First Strategy

Run only the seed0 pair first:

```bash
sbatch --array=0,5 m3_jobs/14_500cp_formal_train.slurm
```

If both seed0 runs complete and verify their required artefacts, launch the
remaining matched seeds:

```bash
sbatch --array=1-4,6-9 m3_jobs/14_500cp_formal_train.slurm
```

Do not launch the remaining seeds until the seed0 baseline and seed0
hierarchical runs are both confirmed complete.

## Resource Request Rationale

The first 500CP 50k seed0 pair should use conservative resources because 50k
memory growth has not yet been confirmed. The script therefore requests one
array-wide allocation of:

- CPU: `4`
- memory: `64G`
- wall-time: `11:00:00`

The 10k pilots suggest the baseline 50k runtime should be about `06:05:00` and
the hierarchical 50k runtime should be about `08:04:15`. The `11:00:00`
request leaves buffer for startup, training-time evaluation, checkpoint writes,
filesystem packaging, and non-linear 50k effects.

After seed0 confirmation, future baseline-only jobs may use a shorter
wall-time request. Keep `64G` for the seed0 50k pair because the pilot MaxRSS
values do not prove peak memory at 50k.

## Expected Outputs

Each task writes Slurm logs under:

```bash
/projects/fr57/cche0357/EV-GNN_outputs/evgnn_500cp_formal50k_<array_job>_<task>.out
/projects/fr57/cche0357/EV-GNN_outputs/evgnn_500cp_formal50k_<array_job>_<task>.err
```

Each task writes its run directory under:

```bash
/scratch2/fr57/cche0357/EV-GNN_runs/500cp_formal_train_50k/<baseline|hierarchical>/seed<seed>/<run_name>/
```

Each task packages the run directory to:

```bash
/projects/fr57/cche0357/EV-GNN_outputs/m3_500cp_formal_train_<label>_<algorithm>_cpu4_seed<seed>_50000steps_job<array_job>_task<task>.tar.gz
```

The package should contain:

- `stdout.log`
- `stderr.log`
- `runtime_metadata/runtime_start.txt`
- `runtime_metadata/runtime_final.txt`
- `runtime_metadata/python_environment.txt`
- `runtime_metadata/source_provenance.txt`
- `runtime_metadata/training_command.txt`
- `runtime_metadata/packaged_files.txt`
- `saved_models/<run_name>/config.yaml`
- `saved_models/<run_name>/kwargs.yaml`
- `saved_models/<run_name>/run_args.yaml`
- `saved_models/<run_name>/training_log.csv`
- `saved_models/<run_name>/model.best_actor`
- `saved_models/<run_name>/model.best_critic`
- `saved_models/<run_name>/model.last_actor`
- `saved_models/<run_name>/model.last_critic`

The `m3_500cp_formal_train` package prefix and the
`500cp_formal_train_50k` run root intentionally separate these outputs from
25CP/100CP formal outputs and from the `500cp_runtime_pilot_10k` outputs.

## Claim Boundaries

- Train-time rewards and 5-episode train evaluations are monitoring evidence
  only, not formal eval30 evidence.
- A successful seed0 50k pair supports launching the remaining formal training
  seeds; it does not establish matched-seed behavioural superiority.
- Checkpoints become formal behavioural evidence only after controlled eval30
  and aggregation are completed.
- Runtime feasibility, MaxRSS, and CPU efficiency should be reported separately
  from reward or operational performance claims.
- If a package is missing required artefacts, treat that task as incomplete
  even if Slurm exits successfully.

## Post-Run Decision Rule

After the seed0 pair:

- if seed0 baseline and seed0 hierarchical 50k both complete and verify all
  required artefacts, launch seeds `1-4` for both algorithms;
- if either seed0 run fails, times out, exceeds memory, or misses required
  artefacts, diagnose before launching the remaining seeds.

Only after all intended 50k training runs complete should controlled 500CP
eval30 be launched.
