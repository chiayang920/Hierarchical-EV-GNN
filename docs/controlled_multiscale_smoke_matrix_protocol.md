# Controlled Multiscale Smoke Matrix Protocol

## Purpose

This protocol defines Stage 1 smoke validation for the controlled TD3-GNN pipeline only. It checks whether the existing controlled training and deterministic evaluation path can reset, train briefly, evaluate, validate CSV shape, and package artefacts across 25CP, 100CP, 500CP, and 1000CP.

This is not formal evidence. Passing this smoke matrix does not support any performance claim, SOTA claim, statistical comparison, or thesis result by itself.

## Matrix

| Slurm task | Scale | Algorithm | Seed | Config | Eval seed offset |
| --- | --- | --- | --- | --- | --- |
| 0 | 25CP | actiongnn | 0 | `config_files/PublicPST_25cp.yaml` | 610000 |
| 1 | 25CP | hierarchical | 0 | `config_files/PublicPST_25cp.yaml` | 610000 |
| 2 | 100CP | actiongnn | 0 | `config_files/PublicPST_100.yaml` | 620000 |
| 3 | 100CP | hierarchical | 0 | `config_files/PublicPST_100.yaml` | 620000 |
| 4 | 500CP | actiongnn | 0 | `config_files/PublicPST_500.yaml` | 630000 |
| 5 | 500CP | hierarchical | 0 | `config_files/PublicPST_500.yaml` | 630000 |
| 6 | 1000CP | actiongnn | 0 | `config_files/PublicPST_1000.yaml` | 640000 |
| 7 | 1000CP | hierarchical | 0 | `config_files/PublicPST_1000.yaml` | 640000 |

## Scope

The matrix is controlled TD3-GNN only:

- `actiongnn`
- `hierarchical`

It explicitly excludes:

- SB3 baselines
- SOTA claims
- Formal statistical comparison
- Full five-seed experiments

No SB3 adapters are introduced or exercised by this protocol.

## Smoke Commands

Each task runs one seed only: seed 0.

Training uses short controlled settings:

- `max_timesteps=512`
- `start_timesteps=64`
- `eval_freq=256`
- `eval_episodes=1`
- `batch_size=32`
- `replay_buffer_size=5000`
- `device=cpu`
- `log_to_wandb=false`

Immediately after training, each task runs deterministic controlled evaluation:

- `eval_episodes=3`
- `deterministic=true`
- `eval_expl_noise=0.0`
- `max_episode_steps` resolved from the config file's `simulation_length`
- Scale-specific reproducible eval seed offset from the matrix above

The smoke evaluation CSV must contain 3 episode rows and 1 summary row.

## Smoke Success Criteria

A task passes only if all of the following are true:

- The expected config file exists.
- `simulation_length` resolves to a positive integer from the config.
- The train command exits 0.
- Required checkpoint files exist: `model.best_actor` and `model.best_critic`.
- Required training artefacts exist: `config.yaml`, `kwargs.yaml`, `run_args.yaml`, and `training_log.csv`.
- The eval command exits 0.
- The smoke CSV has exactly 3 episode rows and 1 summary row.
- The required metric columns exist, including `episode_reward`, `tracking_error`, `energy_tracking_error`, `power_tracker_violation`, `total_transformer_overload`, and `action_fraction_at_max`.
- Episode indexes are exactly 0, 1, and 2.
- Episode steps equal the resolved `simulation_length`.
- `done` is true for every episode row.
- The package tarball is readable with `tar -tzf`.
- `stderr` is empty or contains only documented non-critical warnings, such as the source manifest warning when `EV_GNN_SOURCE_MANIFEST` is intentionally not provided.

## 1000CP Boundary

1000CP remains non-formal until this smoke confirms reset, train, eval, and package feasibility. A present config file is not enough to treat 1000CP as ready for formal evidence generation.

## After-Smoke Decision Rules

If all 8 tasks pass, proceed to Stage 2 pilot matrix design.

If 25CP, 100CP, and 500CP pass but 1000CP fails, keep 1000CP as a feasibility gap and do not block the 25CP, 100CP, and 500CP pilot matrix.

If either controlled algorithm fails at a scale, do not launch formal jobs at that scale until the failure is diagnosed.

## Non-Goals

- No formal performance claims
- No p-values
- No new architecture
- No SB3 adapters
- No M3 job submission by this PR
