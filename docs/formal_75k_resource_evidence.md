# Formal 75k Resource Evidence

This file records local evidence available before the corrected non-negative
75k formal workflow is submitted. It does not approve a formal resource
profile.

## Evidence Found

| Source | Scope | Evidence | Use for 75k workflow |
| --- | --- | --- | --- |
| `README.md` | historical 100CP 50k signed ActionGNN vs hierarchical | signed ActionGNN runtime `02:13:09`, CPU efficiency `93.3%`, peak memory `2.8GB / 32GB`; hierarchical runtime `11:08:52`, CPU efficiency `39.1%`, peak memory `1.7GB / 32GB` | Historical context only. The baseline was signed, not corrected non-negative, and the horizon was 50k. |
| `docs/100cp_post_optimisation_runtime_pilot.md` | 100CP hierarchical post-optimisation | CPU `4`; 10k MaxRSS `1.38G`, wall-time `2113s`; 50k MaxRSS `2.42G`, training wall-time `7083s`, Slurm elapsed `7106s` | Hierarchical 100CP runtime feasibility only. Single-seed and not corrected-baseline evidence. |
| `docs/500cp_formal_training_protocol.md` | 500CP 10k runtime pilot | CPU `4`; actiongnn 10k wall-time `4380s`, Slurm elapsed `4398s`, MaxRSS `8.66G`, CPU efficiency about `93.1%`; hierarchical 10k wall-time `5811s`, Slurm elapsed `5829s`, MaxRSS `6.24G`, CPU efficiency about `92.4%` | 500CP 50k planning context only. It does not prove 75k corrected non-negative resource demand. |
| `docs/full_per_infrastructure_diagnostics_eval30_design.md` | diagnostic eval30 design | smoke used less than `1GB` peak task memory and about `29-39s` for one checkpoint and one episode; design request was CPU `4`, memory `32GB`, wall-time `6h` | Diagnostic-stage context only. It is not training memory evidence. |
| `m3_jobs/17_controlled_multiscale_formal_train_eval.slurm` | historical controlled multiscale 50k array | requested CPU `4`, memory `96G`, wall-time `48:00:00` | Existing successful envelope, not a measured requirement and not a 75k corrected-baseline default. |

## Threading Evidence

Local code inspection found no `DataLoader` workers, multiprocessing pool, or
multi-environment process use in the current `train_td3_gnn.py` path. The
ActionGNN replay buffer batches graphs manually in `utils/replay_buffer_actiongnn.py`.
Historical M3 scripts set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from
`SLURM_CPUS_PER_TASK`, so PyTorch/BLAS thread use may matter, but the smallest
evidence-supported CPU request must be reviewed after the two-cell smoke.

## Missing Evidence

`TRAINING_MAXRSS_EVIDENCE=INSUFFICIENT`

No completed corrected non-negative 75k training run was found. No completed
1000CP 75k training accounting was found. Historical eval or diagnostic memory
must not be used as proof of training memory demand.

## Planning Rule

Historical 50k runtime may be scaled by `1.5` only as a linear
scheduler-planning extrapolation, not a guaranteed runtime. This report does
not produce a single exact runtime request.

## Resource Profile Status

`RESOURCE_PROFILE_STATUS=UNAPPROVED_TEMPLATE_ONLY`

The formal submission helper requires an explicit reviewed profile with
`RESOURCE_PROFILE_APPROVED=YES`. Until smoke accounting is reviewed, the formal
training, eval30, and diagnostic resource fields remain unset.
