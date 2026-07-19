# Vectorised Hierarchical Composition Plan

## 1. Motivation

The historical hierarchical actor breakdown identified
`TD3_HierarchicalActionGNN.Actor._compose_full_node_action()` as the dominant
runtime cost. The slow path is Python-heavy: it loops over chargers,
transformers, and EVs, builds dictionaries from tensor IDs, extracts scalar
IDs with `.item()`, and writes one EV action at a time.

This task implements Approach A only: hybrid per-graph vectorisation. The
objective is to preserve the actor, critic, replay-buffer, graph-state, reward,
training, evaluation, and EV2Gym action contracts while removing the
per-EV/per-charger/per-transformer Python work from the production composition
path.

## 2. Approved Approach A Architecture

Approach A keeps one Python loop over graphs because batched replay states are
represented as concatenated graph components with `state.sample_node_length`.
Inside each graph, ID matching, transformer allocation, charger allocation,
and EV action writes are tensor operations.

Approach A does not implement global cross-graph vectorisation, topology
caching, replay-buffer restructuring, critic optimisation, GNN architecture
changes, EV2Gym action-mapping optimisation, CPU-resource scaling experiments,
formal training, or Approach B.

## 3. Reference and Production Dispatch

The original action-composition algorithm remains available as
`_compose_full_node_action_reference()`. The historical compatibility method
`_compose_full_node_action()` calls that reference method unchanged so the
existing breakdown profiler still measures the historical reference
composition path.

The production path is `_compose_full_node_action_vectorised()`. It returns
only `full_node_action` and does not construct hierarchy-detail lists.

Public actor dispatch is explicit:

- `forward(state, return_details=True)` calls `_forward_reference()` and returns
  the reference output plus hierarchy details.
- `forward(state, return_details=False)` calls `_forward_vectorised()` and
  returns only the vectorised full-node action.

## 4. Graph-Local ID Contract

Transformer IDs, charger IDs, and EV mapping IDs are interpreted within each
graph slice. Repeated local IDs in separate batched graphs must not interact.

The vectorised path maps query IDs to graph-local positions through a tensor
lookup helper based on `torch.sort` and `torch.searchsorted`. It returns `-1`
and `matched=False` for missing IDs, handles empty query/reference tensors,
and maps sorted matches back to the original reference ordering.

## 5. Vectorised Tensor Algorithm

Before the graph loop, the actor computes transformer scores, charger scores,
and EV sigmoid gates once from node embeddings. For each graph, it locates
active EV, charger, and transformer positions using masks and `torch.nonzero`.

Transformer weights are graph-local softmax values. EV charger IDs are mapped
to graph-local charger positions with the tensor lookup helper. Charger
transformer IDs are inferred without a charger loop by reducing EV transformer
IDs per charger position with `scatter_reduce(..., reduce="amin")`, matching
the reference `torch.unique(...)[0]` semantics.

Charger weights use `torch_geometric.utils.softmax` as a grouped segment
softmax over graph-local transformer positions, so only sibling chargers
assigned to the same transformer compete. Invalid/unmatched chargers are left
at zero through one indexed tensor update into the graph-local charger-weight
vector.

The production implementation is linear-memory within each graph: it stores
only EV, charger, and transformer vectors and never allocates dense
EV-by-charger, EV-by-transformer, or transformer-by-charger selector matrices.
EV transformer and charger weights are gathered directly using safe
graph-local positions, then unmatched EVs are masked back to zero. EV actions
are composed in one graph-local tensor expression:

```text
active-EV count
* graph-local transformer weight
* graph-local charger weight
* EV sigmoid gate
```

Unmatched EVs remain zero. Final EV values are clamped to
`[0.0, self.max_action]` and written back to full-node action rows with a
single indexed tensor write per graph. Non-EV rows remain zero. Under the
current contract, valid pre-clamp actions are non-negative because graph
budget, transformer softmax weights, grouped charger softmax weights, and EV
sigmoid gates are all non-negative; the lower clamp is therefore defensive.

## 6. Correctness Test Inventory

The new test file covers dispatch isolation, tensor lookup semantics, forward
numerical parity, output contracts, hierarchy allocation invariants, diagnostic
sum-loss gradient parity, critic-coupled TD3 actor-loss gradient parity, and
Adam optimiser-step parity.

The tests compare reference and vectorised actors from identical weights using
`torch.testing.assert_close(..., atol=1e-6, rtol=1e-5)`. They include synthetic
single-graph cases, batched repeated-local-ID cases, replay-buffer-produced
batch-size-64 states, no-active-EV states, unmatched IDs, clamp cases, and
deterministic repeated forwards.

Source-structure tests also guard the intended implementation shape: exactly
one graph-level loop is permitted, per-EV/per-charger/per-transformer loops are
absent, grouped charger softmax must be used, and dense selector/matmul
fragments such as EV-by-charger one-hot matrices are rejected.

## 7. Real-State Validation Matrix

The profiler validates real PublicPST states before timing. For each requested
scale and seed, it collects active states using zero mapped EV2Gym actions
through the completed replay-threshold episode. It records and validates:

- `low_active_ev_state`
- `medium_active_ev_state`
- `high_active_ev_state`

The online benchmark workload is the deterministic representative high-load
state selected by greatest active-EV count, then greatest total-node count,
then earliest collection order.

Forward parity uses `atol=1e-6, rtol=1e-5`. Real-state gradient parity is
reported with an explicit profiler tolerance of `atol=5e-5, rtol=1e-4` to
account for CPU reduction-order differences in repeated EV allocation
gradients.

For every validated workload, the JSON summary records a
`numerical_equivalence_audit` entry identifying scale, seed, workload kind,
load level, and batch size. Each entry reports:

- forward maximum absolute and relative output differences;
- sum-loss maximum absolute and relative actor-gradient differences, including
  the parameter names with the largest absolute and relative differences;
- critic-coupled actor-loss maximum absolute and relative actor-gradient
  differences, the associated parameter names, reference/vectorised losses,
  and absolute/relative loss differences.

The profiler still fails immediately if configured output or gradient
tolerances are exceeded; these audit fields quantify successful checks rather
than relaxing them.

## 8. Performance Measurement Matrix

The harness supports the required scales:

- `25CP` using `config_files/PublicPST_25cp.yaml`
- `100CP` using `config_files/PublicPST_100.yaml`
- `500CP` using `config_files/PublicPST_500.yaml`

It supports seeds `0, 1, 2, 3, 4`, online workload
`online_representative_state` with batch size `1`, and replay workload
`replay_batch` with batch sizes `1, 8, 16, 32, 64`.

Measured phases are:

- `composition_forward`
- `complete_actor_forward`
- `complete_actor_forward_backward_sum_loss`
- `complete_actor_forward_backward_critic_loss`

For `composition_forward`, feature preparation and GNN encoding are performed
outside the timed callable so the phase isolates action composition.

Warm-ups default to `5`; measured repetitions default to `30`. Execution order
alternates by schedule index: even schedules run reference then vectorised, and
odd schedules run vectorised then reference.

## 9. Statistical Reporting

The CSV is repetition-level evidence. The JSON groups measured rows by
replication, scale, seed, workload kind, load level, batch size,
implementation, and phase. Each group reports count, total, mean, median,
standard deviation, minimum, maximum, and p95.

For paired reference/vectorised groups, the summary reports median speedup,
mean speedup, paired speedup values, paired speedup mean, paired speedup
median, paired speedup standard deviation, and
`within_run_paired_timing_speedup_95_percent_confidence_interval`. The
interval uses a normal approximation over paired timing repetitions within one
profiler run:

```text
mean +/- 1.96 * sample_standard_deviation / sqrt(n)
```

This interval characterises within-run timing variation only. It does not
represent uncertainty across seeds, M3 allocations, nodes, or training runs.
The profiler reports measured results only and does not claim statistical
significance.

## 10. M3 Replication Design

The Slurm harness runs as an array with tasks `0-2`; each task is one
independent replication. It is CPU-only by default, uses `SLURM_CPUS_PER_TASK`
to set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS`, and
activates:

```text
/scratch2/fr57/cche0357/conda/envs/evgnn_m3_cpu
```

Each replication runs all three scales, seeds `0-4`, online and replay
workloads, `5` warm-ups, and `30` measured repetitions. Generated CSV and JSON
outputs are written under:

```text
/scratch2/fr57/cche0357/EV-GNN_runs/vectorised_hierarchical_composition/
```

The script packages only generated CSV and JSON outputs into:

```text
/projects/fr57/cche0357/EV-GNN_outputs/
```

It prints dependency versions, requires branch-independent source provenance
through `EV_GNN_SOURCE_PROVENANCE` or `EV_GNN_SOURCE_MANIFEST`, checks outputs
are present and non-empty, and prints a unique completion marker. It never runs
formal training.

The Slurm script sets `REPO_DIR=/projects/fr57/cche0357/EV-GNN`, changes into
that directory before invoking Python, verifies the actor and profiler source
files are present and non-empty, and prints date, hostname, Slurm IDs, CPU
allocation, working directory, and source provenance. The profiler persists the
same execution provenance in JSON metadata through `hostname`, `slurm_job_id`,
`slurm_array_job_id`, `slurm_array_task_id`, `source_provenance`, and
`source_manifest_path`; when `EV_GNN_SOURCE_MANIFEST` is set, it also records
`source_manifest_sha256`.

## 11. Relationship to Approach B

Approach A correct and end-to-end performance sufficient
→ stop; do not implement Approach B

Approach A correct but residual profiling shows graph-level Python iteration remains dominant
→ consider Approach B in a separate branch and PR

Approach A incorrect
→ fix Approach A; do not skip directly to Approach B

A fixed 3x or 5x speed-up is not an automatic success criterion.

## 12. Limitations

This implementation prepares local tests, a real-state validation profiler,
and an M3 replication harness. It does not submit M3 jobs, transfer evidence,
run formal training, or prove end-to-end training speed-up.

The component ratios are approximate diagnostics because separately timed
phases are not strictly additive. CPU resource-scaling remains a later phase.
The critic-coupled backward phase validates actor-gradient parity for a TD3
actor-loss shape but is not a full TD3 training update.

## 13. Decision Rules

Final acceptance requires all of:

- complete correctness validation;
- paired component-level timing evidence;
- scale coverage at 25CP, 100CP, and 500CP;
- batch-size coverage at 1, 8, 16, 32, and 64;
- seed coverage at 0-4;
- three independent M3 replications;
- CPU resource-scaling evidence in a later phase;
- 100CP end-to-end training evidence in a later phase;
- residual bottleneck analysis;
- no material online-inference regression.

This implementation task prepares the harness but does not run the M3 or
end-to-end phases.
