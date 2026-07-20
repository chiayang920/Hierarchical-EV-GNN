# 500CP Runtime Pilot Protocol

## Purpose

Run a controlled M3 CPU4 10k runtime pilot for both canonical 500CP training
paths before deciding whether 500CP formal training is feasible.

The Slurm script is:

```bash
m3_jobs/13_500cp_runtime_pilot.slurm
```

It calls the canonical training entrypoint, not the profiler:

```bash
python train_td3_gnn.py --algorithm actiongnn --config ./config_files/PublicPST_500.yaml
python train_td3_gnn.py --algorithm hierarchical --config ./config_files/PublicPST_500.yaml
```

## Controlled Setup

- Array task `0`: `actiongnn` controlled baseline
- Array task `1`: `hierarchical`
- Config: `./config_files/PublicPST_500.yaml`
- Seed: `0`
- Device/resource: CPU4
- Memory/time request: `64G`, `04:00:00`
- Default length: `10000` training steps
- Start timesteps: `1000`
- Eval frequency: `5000`
- Eval episodes: `5`
- Batch size: `64`
- Replay buffer size: `100000`
- WandB logging: disabled

The output prefix is `m3_500cp_runtime_pilot`, separated from the 25CP and
100CP formal output prefixes.

## Source Provenance

Do not use M3 git commands as a required source update mechanism for this
pilot. The source state should be staged before submission and validated with a
source manifest.

Recommended launch metadata:

```bash
export EV_GNN_SOURCE_PROVENANCE="branch_exp_500cp_runtime_pilots_commit_<shortsha>"
export EV_GNN_SOURCE_MANIFEST="/projects/fr57/cche0357/EV-GNN_outputs/source_manifests/<manifest>.sha256"
```

The job records both variables when present and copies the manifest into
`runtime_metadata`.

## Outputs To Inspect

Each array task writes a separated run directory under:

```bash
/scratch2/fr57/cche0357/EV-GNN_runs/500cp_runtime_pilot_10k/<baseline|hierarchical>/
```

Each task packages that run directory to:

```bash
/projects/fr57/cche0357/EV-GNN_outputs/m3_500cp_runtime_pilot_<label>_<algorithm>_cpu4_seed0_<steps>steps_job<array_job>_<task>.tar.gz
```

Inside each package, inspect:

- `stdout.log` and `stderr.log`
- `runtime_metadata/runtime_start.txt`
- `runtime_metadata/runtime_final.txt`
- `runtime_metadata/python_environment.txt`
- `runtime_metadata/source_provenance.txt`
- `saved_models/<run_name>/config.yaml`
- `saved_models/<run_name>/kwargs.yaml`
- `saved_models/<run_name>/run_args.yaml`
- `saved_models/<run_name>/training_log.csv`
- `saved_models/<run_name>/model.best_actor`
- `saved_models/<run_name>/model.best_critic`
- `saved_models/<run_name>/model.last_actor`
- `saved_models/<run_name>/model.last_critic`

MaxRSS is not knowable from inside the running job. After completion, record it
from `sacct` or M3 jobstats alongside the packaged runtime metadata.

## Interpretation Boundary

This pilot answers whether both 500CP algorithms can complete the same 10k
end-to-end training path under comparable CPU4 settings. Rewards and checkpoint
quality from this run are smoke information only, not formal behavioural
evidence and not a baseline-vs-hierarchical performance claim.
