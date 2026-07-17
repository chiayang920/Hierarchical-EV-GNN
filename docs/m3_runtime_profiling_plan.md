# M3 TD3-GNN Runtime Profiling Plan

## Motivation

The hierarchical TD3-GNN actor introduces a physically aligned action pathway
from CPO to transformer, charger, and EV. This structure is scientifically
useful, but the 100CP training runs showed a large runtime gap that must be
understood before any optimisation is attempted.

This PR adds instrumentation only. It does not change actor logic, critic
logic, replay-buffer semantics, EV2Gym action mapping, rewards, or evaluation
metric definitions.

## Known 100CP Computational Issue

The formal 100CP runs showed that the ActionGNN baseline completed training in
about 02:13:09 with about 93.3% CPU efficiency. The hierarchical actor
completed in about 11:08:52 with about 39.1% CPU efficiency.

This suggests that the hierarchical method has a substantial computational
overhead, but the existing formal results do not identify whether the overhead
comes mainly from action selection, environment stepping, replay operations, or
TD3 update calls.

## Measured Phases

`profiling/profile_td3_gnn_runtime.py` runs a short training-like loop using the
current scale-agnostic TD3-GNN entry-point components. For each profiled step it
records wall-clock timings for:

- `env_reset`
- `select_action`
- `env_step`
- `replay_add`
- `replay_sample_or_train_guard`
- `policy_train`
- `total_loop_step`

Replay sampling is called inside `policy.train(...)` in the existing policy
classes. Therefore, `policy_train` should be interpreted as the combined replay
sampling and TD3 update cost. The separate
`replay_sample_or_train_guard` timing only measures the outer training guard.

The profiler writes one step-level CSV and one JSON summary per run. It does
not save checkpoints and does not call wandb.

## 100CP Runtime Profile Job

`m3_jobs/08_100cp_runtime_profile.slurm` compares the canonical `actiongnn` and
`hierarchical` labels under the same 100CP configuration:

- config: `./config_files/PublicPST_100.yaml`
- device: CPU
- profile steps: 2000
- start timesteps: 100
- batch size: 64
- replay buffer size: 100000

This job is intended to answer whether the 100CP hierarchical overhead is
concentrated in action selection, EV2Gym stepping, replay insertion, or the
combined `policy.train(...)` update path.

## 500CP Tiny Smoke Profile Job

`m3_jobs/09_500cp_tiny_smoke_profile.slurm` runs the same profiler on
`./config_files/PublicPST_500.yaml` for both canonical algorithms with only
200 profiled steps and 50 start timesteps.

This is not a formal 500CP experiment. Its purpose is only to verify that the
refactored scale-agnostic pipeline starts correctly at 500CP and produces
profiling CSV/JSON outputs for both algorithms.

## Interpreting Outputs

The CSV output contains one row per timed phase, including the run metadata,
step index, episode step, reward, done flag, replay size, action dimension, and
active graph sizes. This supports direct inspection of per-step timing spikes
and graph-size-dependent behaviour.

The JSON summary aggregates total and mean timings for the main phases. Useful
comparisons include:

- `total_select_action_time_seconds` between `actiongnn` and `hierarchical`
- `total_env_step_time_seconds` to check whether environment stepping dominates
- `total_policy_train_time_seconds` and `mean_policy_train_time_seconds` to
  assess combined replay sampling and TD3 update cost
- `mean_total_loop_step_time_seconds` as an end-to-end per-step diagnostic
- `number_of_policy_train_calls` to confirm matched update counts

The JSON summary also records Python, PyTorch, CUDA availability, device, and
final replay size to make M3 runs easier to audit.

## Decisions Supported

The profiling results should support the next engineering decision:

- If `select_action` dominates, inspect hierarchical actor composition and
  graph-level allocation logic.
- If `env_step` dominates similarly for both algorithms, focus on EV2Gym scale
  behaviour rather than actor optimisation.
- If `policy_train` dominates, inspect replay batching, critic update cost, and
  target actor calls before changing actor semantics.
- If 500CP tiny smoke fails to start, prioritise scale-agnostic contract fixes
  before running larger experiments.

No optimisation conclusion should be drawn until the M3 profiling outputs have
been inspected. These profiling jobs diagnose runtime location; they do not
claim that hierarchical training has been optimised or that formal 500CP
results exist.
