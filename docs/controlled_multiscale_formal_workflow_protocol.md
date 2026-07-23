# Controlled Multiscale Formal Workflow Protocol

## Purpose

Define a formal controlled ActionGNN-vs-hierarchical workflow for future multiscale evidence generation. The protocol prepares the aggregation and evidence boundary only; it does not launch jobs or support a performance claim by itself.

## Scope

Covers:

- Scales: 25CP, 100CP, 500CP, and 1000CP.
- Algorithms: `actiongnn` and `hierarchical`.
- Training seeds: `0, 1, 2, 3, 4`.
- Deterministic eval30 evaluation, with 30 paired episode seeds per training seed.
- Canonical CSV naming: `<scale>_<algorithm>_seed<seed>_eval30.csv`, for example `25cp_actiongnn_seed0_eval30.csv`.

Excludes:

- SB3 baselines.
- External SOTA comparison.
- Architecture, actor, critic, trainer, evaluator, or config changes.
- Formal claims before all jobs finish and aggregation passes.

## Eval Seed Protocol

For training seed `s`, expected episode seeds are `base + s * 1000 + 0` through `base + s * 1000 + 29`.

| Scale | Eval seed base |
| --- | ---: |
| 25CP | 710000 |
| 100CP | 720000 |
| 500CP | 730000 |
| 1000CP | 740000 |

## Statistical Unit

The training seed is the inference unit. Episode rows are deterministic evaluation replicates and must be averaged to seed-level means before paired p-values or confidence intervals are computed.

Do not treat 30 episodes per seed as 30 independent samples for statistical testing.

## Metrics

Reward:

- `episode_reward`

Tracking and operational:

- `tracking_error`
- `energy_tracking_error`
- `power_tracker_violation`
- `total_ev_served`
- `total_energy_charged`
- `total_energy_discharged`

Satisfaction:

- `average_user_satisfaction`
- `energy_user_satisfaction`

Overload and degradation:

- `total_transformer_overload`
- `battery_degradation`
- `battery_degradation_calendar`
- `battery_degradation_cycling`

Profit:

- `total_profits`

Action diagnostics:

- `action_mean`
- `action_std`
- `action_fraction_zero`
- `action_fraction_at_max`
- `active_action_count_mean`

Resource reporting:

- Runtime.
- Memory.
- Wall-time.

## Evidence Workflow Expectation

A future Slurm workflow should produce one complete evidence package:

1. Train/eval array job for the 4 x 2 x 5 controlled matrix.
2. Dependent reducer or bundler job after all array tasks finish.
3. Aggregation with `scripts/aggregate_controlled_multiscale_eval30.py`.
4. One final complete evidence `tar.gz` containing the validated CSVs, aggregation outputs, logs, and manifests.
5. One final local `scp` of that single `tar.gz` only.

Use an aggregation output directory separate from the exact 40-CSV input directory so generated summaries do not pollute the fixed eval30 evidence set.

This protocol does not create Slurm scripts and does not submit M3 jobs.

## Interpretation Boundaries

Do not claim:

- Hierarchical superiority before aggregation.
- 1000CP superiority from smoke or feasibility alone.
- External SOTA comparison.
- A final thesis result before the full formal workflow and evidence review pass.

Report paired seed-level evidence by scale and metric, and keep context-dependent action diagnostics descriptive unless supported by a predefined interpretation.
