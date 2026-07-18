# Hierarchical Actor Runtime Breakdown Plan

## Motivation

The M3 runtime profiles show that the hierarchical actor is much slower than
the ActionGNN baseline at 100CP and especially in the 500CP tiny smoke. Those
profiles localise the dominant cost to the hierarchical training path, but they
do not separate graph batching, actor forward propagation, hierarchical action
composition, EV2Gym action mapping, and actor backward propagation.

This task adds diagnostic-only instrumentation. It does not optimise,
vectorise, or change the production actor, critic, replay buffer, graph-state,
reward, environment-action, training, or evaluation contracts.

## Controlled Methodology

`profiling/profile_hierarchical_actor_breakdown.py` runs both canonical
algorithms in one process for a single PublicPST-compatible config. The M3 job
runs the 100CP and 500CP configs sequentially in the same Slurm allocation.

For each config and batch size:

- the profiler sets the requested seed;
- it constructs real `PublicPST_GNN` graph states by stepping EV2Gym with zero
  mapped actions;
- it stores active-EV graph transitions in the existing
  `ActionGNN_ReplayBuffer`;
- it treats the replay threshold as the largest requested batch size, ensuring
  the fixed replay batches can be constructed;
- after the replay threshold is reached, it keeps stepping the same episode
  until that episode terminates;
- it selects the batch-size-one representative state deterministically by
  highest active-EV count, then highest total-node count, then earliest
  collection order across all active states observed from collection start
  through the completed episode in which the replay threshold is reached;
- it samples one fixed replay batch per requested batch size;
- it reuses the same representative online state for batch size `1`;
- it reuses the same fixed sampled replay graph batch for batch size `64`;
- it measures ActionGNN and hierarchical actors sequentially in the same
  process, alternating order by schedule index;
- it records configurable warm-up repetitions and measured repetitions. The M3
  job uses 3 warm-up repetitions and 20 measured repetitions.

Replay-buffer sampling is timed separately from the fixed actor inputs. The
sample phase repeatedly samples from the same fixed replay buffer using matched
seeds for both algorithm labels. Actor phases then reuse the fixed state or
fixed replay batch so actor timing is not confounded by different sampled
graphs.

Collection uses zero-based episode indexes internally and in JSON metadata.
Collection steps are recorded one-based because the profiler logs and user
output describe steps as `1, 2, 3, ...`. `max_collection_steps` is a hard safety
limit: if the replay threshold is reached but the threshold episode does not
terminate before the limit, the profiler fails instead of silently claiming a
complete-episode representative search.

Algorithm order alternates deterministically: even schedule indexes run
`actiongnn` then `hierarchical`, and odd schedule indexes run `hierarchical`
then `actiongnn`. The CSV records both `schedule_index` and
`algorithm_order_position`, which makes order effects auditable without
introducing randomisation.

These controls reduce measurement bias in four ways. The representative
batch-size-one state avoids timing a low-load first active graph that may not
reflect scale-sensitive online action selection. Continuing through the end of
the threshold episode lets post-threshold active states compete for
representative selection. The fixed actor inputs keep ActionGNN and
hierarchical measurements on identical graph data. Alternating algorithm order
reduces systematic warm-cache or call-order bias between the two algorithms.

This improves batch-one scale sensitivity, but it is not a claim that the
selected state is the absolute maximum-load graph across every possible
episode, seed, or full configuration.

## Timed Phases

The profiler records repetition-level CSV rows for:

- `replay_buffer_sample`;
- `actiongnn_actor_forward`;
- `actiongnn_actor_forward_backward`;
- `hierarchical_actor_forward`;
- `hierarchical_prepare_features`;
- `hierarchical_prepare_compose_inputs`;
- `hierarchical_encode_nodes`;
- `hierarchical_compose_full_node_action`;
- `hierarchical_map_to_ev2gym_action` for batch-size-one states;
- `hierarchical_actor_forward_backward`.

The hierarchical breakdown invokes the existing private actor methods directly
for diagnostics. It does not duplicate `_compose_full_node_action()` semantics.
After timing the decomposed path, the profiler checks outside timed regions
that the decomposed output has the same shape, dtype, device, and values as the
complete hierarchical actor forward output using `torch.testing.assert_close`
with `atol=1e-6` and `rtol=1e-5`.

Forward-only actor timings run under `torch.no_grad()` to isolate forward
execution. Forward-plus-backward timings zero actor gradients, run the actor,
build `sum(actor_output)` as a deterministic scalar diagnostic loss, call
`backward()`, and verify that all expected actor parameters receive finite
gradients. No optimiser step is performed. This is not the TD3 actor objective.

## Outputs

Each run writes:

- `<run_name>_hierarchical_actor_breakdown_steps.csv`;
- `<run_name>_hierarchical_actor_breakdown_summary.json`.

The CSV contains config, scale, algorithm, batch size, schedule index,
algorithm order position, repetition status, phase, elapsed seconds, graph
counts, CPU thread count, and device.

The JSON summary groups measured rows by algorithm, batch size, and phase. For
each measured phase it reports count, total, mean, median, standard deviation,
minimum, maximum, and p95 using only the Python standard library.

Runtime metadata includes Python, PyTorch, PyG, device, thread counts, action
dimension, max action, seed, batch sizes, representative single-state counts,
the representative-state selection rule, and the diagnostic-backward note.
It also records `collection_steps_completed`,
`collected_active_transition_count`, `completed_collection_episodes`,
`replay_threshold_reached_at_step`, `replay_threshold_episode_index`,
`representative_state_collection_step`, `representative_state_episode_index`,
and `collection_termination_reason`. Successful runs use
`completed_threshold_episode`, which makes it auditable that the profiler
searched through the completed threshold episode.

## Interpretation

The results are intended to choose the next engineering target:

- high `replay_buffer_sample` means replay graph batching should be inspected;
- high `hierarchical_prepare_features` means graph feature preparation is a
  plausible bottleneck;
- high `hierarchical_encode_nodes` means GNN node encoding should be inspected;
- high `hierarchical_compose_full_node_action` means hierarchical allocation
  composition is the likely target;
- high `hierarchical_map_to_ev2gym_action` means environment-action mapping may
  matter for online action selection;
- high actor forward/backward phases show whether propagation or gradient
  computation dominates.

These measurements are diagnostic evidence only. They do not claim a training
speedup, policy quality change, or formal 500CP performance result.
