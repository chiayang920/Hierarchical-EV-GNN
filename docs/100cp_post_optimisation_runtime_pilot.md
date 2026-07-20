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

## Follow-Up Condition

A 50k confirmation run is justified if the 10k pilot completes cleanly, writes
all metadata and training artefacts, and shows a material wall-time reduction
consistent with the PR #4 component-level improvement without introducing new
training failures.

Approach B is justified instead if the 10k pilot remains close to the previous
hierarchical 100CP wall-time profile, or if the saved runtime metadata and
training log suggest the bottleneck moved outside the optimised composition path
and still blocks practical 100CP hierarchical training.
