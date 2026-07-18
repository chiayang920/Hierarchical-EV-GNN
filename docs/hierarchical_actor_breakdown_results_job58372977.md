# Hierarchical Actor Breakdown Results: M3 Job 58372977

## 1. Purpose

This diagnostic run was designed to localise the hierarchical actor runtime
bottleneck before any production optimisation. It is a controlled diagnostic
microbenchmark of actor-facing runtime phases, not a formal training run, not a
policy-quality experiment, and not evidence of an achieved speed-up.

The measurements are intended to support an engineering decision about the next
optimisation target while preserving the existing actor, critic, replay-buffer,
state, reward, training, and evaluation semantics.

## 2. Provenance

| Field | Value |
| ----- | ----- |
| Repository | `chiayang920/Hierarchical-EV-GNN` |
| Branch | `perf/hierarchical-actor-breakdown` |
| Source commit | `820e6b7dddadd4d4f41799eebf741ddae13e2532` |
| M3 job ID | `58372977` |
| M3 node | `m3e108` |
| Date | 18 July 2026 |
| Seed | `0` |
| Device | CPU |
| Python | `3.11.15` |
| PyTorch | `2.2.2+cu121` |
| PyTorch Geometric | `2.5.3` |
| Allocated CPU cores | `4` |
| Allocated memory | 16 GB |
| Warm-up repetitions | `3` |
| Measured repetitions | `20` |
| Batch sizes | `1` and `64` |

Raw evidence package:

```text
m3_hierarchical_actor_breakdown_seed0_58372977.tar.gz
```

SHA-256:

```text
c9ea3faf1fe61db338ff731c3659041604eb54c08b3baf89ba0c95d08f52ac2c
```

The raw CSV, JSON, stdout, stderr, tarball, jobstats, and sacct evidence is
archived outside Git and is not committed to the repository.

## 3. Job Integrity

| Field | Value |
| ----- | ----- |
| Slurm state | `COMPLETED` |
| Exit code | `0:0` |
| Elapsed time | `00:02:35` |
| Total CPU | `04:03.013` |
| Allocated CPUs | `4` |
| Batch-step MaxRSS | `1494332K` |
| Jobstats CPU utilisation | 37.9% |
| Jobstats reported memory use | 923.2 MB of 16 GB |
| stderr | empty |
| Blocking error patterns | none |

The job produced one 100CP repetition-level CSV, one 100CP summary JSON, one
500CP repetition-level CSV, and one 500CP summary JSON. Each CSV contained 483
rows, and every measured phase contained 20 observations.

## 4. Controlled Methodology

The profiler runs both algorithms in the same process and in the same Slurm
allocation. For each scale and batch size, identical fixed graph states are
reused for both ActionGNN and the hierarchical actor. Batch size `1` uses a
deterministic representative state, and batch size `64` uses one fixed replay
batch.

Representative-state collection continues through the end of the episode in
which the replay-size threshold is reached. This allows post-threshold active
states from that episode to compete for batch-one representative selection.

Algorithm order alternates by schedule index: even schedules run ActionGNN then
hierarchical, while odd schedules run hierarchical then ActionGNN. Complete
hierarchical forward output is checked against the decomposed forward path
outside timed regions. Actor parameters are not updated; no optimiser step or
checkpoint is produced.

## 5. State and Collection Metadata

| Scale | Collection steps | Threshold step | Active transitions | Representative step | Active EVs | Chargers | Transformers | Total nodes |
| ----- | ---------------: | -------------: | -----------------: | ------------------: | ---------: | -------: | -----------: | ----------: |
| 100CP |              112 |             67 |                108 |                  28 |         99 |       99 |            7 |         206 |
| 500CP |              112 |             67 |                108 |                  27 |        483 |      483 |           35 |        1002 |

Both runs ended with:

```text
collection_termination_reason = completed_threshold_episode
```

## 6. Primary Timing Results

All values in this section are milliseconds.

| Scale | Batch | ActionGNN forward median | ActionGNN forward p95 | Hierarchical forward median | Hierarchical forward p95 | Forward slowdown |
| ----- | ----: | -----------------------: | --------------------: | --------------------------: | -----------------------: | ---------------: |
| 100CP |     1 |                    1.027 |                 1.105 |                       5.051 |                    5.241 |            4.92× |
| 100CP |    64 |                    3.832 |                 6.002 |                     164.394 |                  170.649 |           42.90× |
| 500CP |     1 |                    1.566 |                 1.712 |                      22.053 |                   22.989 |           14.08× |
| 500CP |    64 |                   17.939 |                20.000 |                     830.027 |                  960.225 |           46.27× |

Forward-plus-backward diagnostic timings:

| Scale | Batch | ActionGNN median | Hierarchical median | Slowdown |
| ----- | ----: | ---------------: | ------------------: | -------: |
| 100CP |     1 |            1.684 |              11.533 |    6.85× |
| 100CP |    64 |            7.821 |             410.756 |   52.52× |
| 500CP |     1 |            2.555 |              50.670 |   19.83× |
| 500CP |    64 |           39.946 |            2411.967 |   60.38× |

## 7. Hierarchical Phase Breakdown

| Scale | Batch | Prepare features median | Encode nodes median | Compose action median | Compose p95 | Approximate compose/forward share |
| ----- | ----: | ----------------------: | ------------------: | --------------------: | ----------: | --------------------------------: |
| 100CP |     1 |                   0.179 |               0.602 |                 4.219 |       4.303 |                             83.5% |
| 100CP |    64 |                   0.398 |               2.830 |               158.925 |     170.340 |                             96.7% |
| 500CP |     1 |                   0.246 |               0.971 |                20.494 |      21.162 |                             92.9% |
| 500CP |    64 |                   1.205 |              14.254 |               812.564 |     940.814 |                             97.9% |

The percentage is an approximate diagnostic ratio:

```text
compose median / complete hierarchical-forward median
```

The component timings should not be interpreted as strictly additive accounting.
They are separate diagnostic calls and may include different cache and wrapper
effects.

## 8. Bottleneck Diagnosis

Measured evidence indicates that
`TD3_HierarchicalActionGNN.Actor._compose_full_node_action()` is the dominant
runtime bottleneck.

The production implementation contains per-graph iteration, per-charger
iteration, repeated boolean masking, repeated `torch.any()` and `torch.unique()`
calls, per-transformer iteration, Python dictionaries built from tensor IDs,
repeated `.item()` scalar extraction, per-EV Python iteration, and scalar
indexed writes into the final action tensor.

The observed 100CP-to-500CP scaling is approximately proportional to graph
size, rather than clear evidence of severe super-linear complexity:

```text
Batch-64 active EV growth: approximately 4.81×
Batch-64 composition median growth: approximately 5.11×
```

The interpretation is therefore that the current bottleneck is primarily a
very large Python-side constant factor in an approximately linear workload.

## 9. Secondary Findings

Replay-buffer sampling is not the primary bottleneck. Feature preparation is
not the primary bottleneck. GNN encoding is not the primary bottleneck. EV2Gym
mapping is a secondary online-inference cost, particularly at 500CP batch size
`1`.

Allocating more CPU cores is not an evidence-based immediate fix. CPU4
previously outperformed CPU8, while CPU1 and CPU2 have not yet been
benchmarked. CPU1/CPU2/CPU4 scaling should be tested only after vectorisation
changes the workload characteristics.

## 10. Relationship to Formal Thesis Evidence

Existing 100CP formal training is complete. The ActionGNN baseline array job
was `58183947`, the hierarchical array job was `58183949`, the paired
deterministic eval30 job was `58201109`, and the aggregation job was `58204227`.
Together these completed 5 paired seeds and 300 evaluation episodes. These
results remain the final behavioural evidence for the pre-optimisation
implementation.

Strict numerical and algorithmic equivalence may allow the existing behavioural
evidence to remain valid while adding an efficiency comparison. Any
optimisation that changes action computation, numerical behaviour, gradients,
feature inputs, parameters, or training behaviour requires new formal training
and paired eval30 evidence.

## 11. Engineering Decision

The evidence-driven next target is:

```text
Vectorise _compose_full_node_action() first.
```

The intended optimisation should investigate vectorised graph membership,
grouped or segment softmax, indexed gather/scatter operations, removal of
per-EV `.item()` calls, removal of scalar per-EV assignments, and avoiding
unnecessary hierarchy-detail construction when details are not requested.

Action mapping should remain a separate secondary optimisation. Formal 500CP
training remains blocked until:

1. correctness equivalence is proven;
2. controlled microbenchmark speed-up is demonstrated;
3. a 100CP end-to-end training pilot confirms that the speed-up transfers to
   training.

## 12. Limitations

This evidence is limited to one seed and one M3 node. The collection trajectory
uses controlled zero actions, and fixed states rather than a trained-policy
state distribution. The forward-plus-backward phase uses a diagnostic scalar
loss rather than the complete TD3 objective.

Complete-forward and decomposed-phase timings are separate calls. Component
timings may be affected by cache state and wrapper overhead. The run identifies
a bottleneck but does not demonstrate an optimisation speed-up. CPU efficiency
from this microbenchmark must not be treated as a direct replacement for
full-training resource accounting.

## 13. Final Decision

```text
Diagnostic implementation: accepted
M3 job 58372977: valid
Primary bottleneck: hierarchical action composition
Immediate optimisation target: vectorised composition
Action mapping: secondary target
CPU scaling experiment: after vectorisation
100CP formal evidence: complete for pre-optimisation implementation
500CP formal training: blocked
```
