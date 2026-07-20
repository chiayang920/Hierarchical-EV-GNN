# 100CP Post-Optimisation Hierarchical Runtime Pilot

## Purpose

This pilot checks whether the PR #4 hierarchical action-composition optimisation
translates from component-level speed-up into lower 100CP hierarchical training
wall-time on M3.

The M3 script is:

```bash
m3_jobs/12_100cp_postopt_hierarchical_runtime_pilot.slurm
```

It calls the existing controlled training entrypoint:

```bash
python train_td3_gnn.py --algorithm hierarchical --config ./config_files/PublicPST_100.yaml
```

The script reuses this entrypoint because it already owns the TD3 training
loop, `TD3_HierarchicalActionGNN` selection, EV2Gym stepping, replay-buffer
contract, checkpoint writing, copied config, `run_args.yaml`, and
`training_log.csv`. Duplicating those responsibilities in Slurm would risk
creating a second, non-canonical training path.

## Pilot Setup

- Config: `./config_files/PublicPST_100.yaml`
- Algorithm: `hierarchical`
- Seed: `0`
- Device/resource: CPU4 only
- Default length: `10000` training steps
- Environment: `/scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu`
- Repository path: `/projects/fr57/cche0357/EV-GNN`
- Output package root: `/projects/fr57/cche0357/EV-GNN_outputs`

The output prefix is `postopt_100cp_hierarchical_runtime_pilot`, separated from
the formal `phase2d_100cp_formal_train` outputs.

## Not Formal Thesis Evidence

This is runtime instrumentation around one short hierarchical-only training
run. It is not matched-seed behavioural evidence, not an ActionGNN-vs-
hierarchical evaluation comparison, and not a replacement for the formal 100CP
50k seed array or controlled eval30 protocol.

Interpret rewards or checkpoint quality only as incidental smoke information.
The useful result is the wall-time of the unchanged training entrypoint under
the controlled 100CP CPU4 pilot setup.

## Interpreting Runtime

Use `runtime_metadata/runtime_final.txt` inside the packaged output for the
training command wall-time. That wall-time includes the existing
`train_td3_gnn.py` training run and its built-in train-time evaluations, so it
is the practical end-to-end cost of this 10k pilot rather than an isolated actor
microbenchmark.

Compare the 10k wall-time and throughput against pre-optimisation 100CP
hierarchical training/profiling records only when the resource class, seed,
config, and training arguments match closely enough to make the comparison
meaningful.

## Completed Runtime Evidence

PR #4 was merged into `main` at merge commit `47c8577`. The post-optimisation
runtime pilot branch records the following M3 CPU4 evidence.

### 10k Pilot

- Job ID: `58406334`
- State: `COMPLETED 0:0`
- Node: `m3j004`
- CPU: `4`
- MaxRSS: `1.38G`
- Training wall-time: `00:35:13` = `2113s`
- Total training steps: `10000`
- Direct 10k speed-up vs pre-opt seed0 elapsed `7110.65s`: `3.37x`
- Direct 10k speed-up vs pre-opt five-seed mean elapsed `7585.39s`: `3.59x`

### 50k Confirmation

- Job ID: `58410107`
- State: `COMPLETED 0:0`
- Node: `m3v115`
- CPU: `4`
- MaxRSS: `2.42G`
- Training wall-time: `01:58:03` = `7083s`
- Slurm elapsed: `01:58:26` = `7106s`
- Total training steps: `50000`
- Final mean reward: `-1833155.701`
- Best mean reward: `-888323.844`
- Speed-up vs pre-opt seed0 final elapsed `40380.43s`: `5.70x`
- Speed-up vs pre-opt five-seed mean final elapsed `41744.80s`: `5.89x`

### Provenance

- Source provenance:
  `branch_perf_100cp_post_optimisation_runtime_pilot_commit_baf696d_50k_confirmation`
- Expanded source manifest:
  `/projects/fr57/cche0357/EV-GNN_outputs/source_manifests/postopt_100cp_runtime_pilot_baf696d_expanded_source.sha256`
- Package:
  `/projects/fr57/cche0357/EV-GNN_outputs/postopt_100cp_hierarchical_runtime_pilot_cpu4_seed0_50000steps_job58410107.tar.gz`
- Package SHA256:
  `2573498bc804cf74e644d8930ff4c5038658271bb02c6d306960cd8030beb17f`

### Interpretation

This is runtime evidence, not formal behavioural evidence. It supports that the
PR #4 component-level speed-up translated into end-to-end 100CP hierarchical
training runtime improvement.

Approach A is sufficient for the current thesis engineering path. Approach B is
not justified unless future 500CP production training becomes necessary.

### Known Limitations

- The post-optimisation confirmation is a single seed0 run.
- Reward should not be compared against formal eval30 results.
- The pre-optimisation `run_args.yaml` lacks an explicit `algorithm` field,
  though run naming, config, and action dimension indicate hierarchical 100CP.
- Package-internal stdout is truncated before the final post-package success
  marker because packaging occurs before final echo lines; `runtime_final.txt`,
  `sacct`, and terminal output confirm completion.
