# Thesis Evidence Index

This document maps the current evidence base and remaining experiment plan for
the EV-GNN hierarchical actor thesis. It is a thesis-level planning document,
not a new experimental result.

## 1. Research Objective

The research objective is to test whether a physically aligned hierarchical
actor improves EV charging coordination compared with the baseline EV-GNN
ActionGNN actor.

The proposed method keeps the EV2Gym PublicPST simulator, graph-state family,
reward setting, final EV2Gym action vector, and TD3 critic/replay-buffer
interfaces aligned with the baseline. The research change is on the actor side:
the policy composes charging actions through the physical allocation hierarchy:

```text
CPO → Transformer → Charger → EV
```

This differs from the baseline flat EV-GNN ActionGNN actor. The baseline already
uses a graph representation containing CPO, transformer, charger, and EV nodes,
but its actor predicts EV-node actions through a flat final ActionGNN output
path. The hierarchical actor instead makes the action-generation path mirror
the physical control hierarchy before returning the same external interfaces:

```text
mapped_action_numpy:
  EV2Gym-compatible action vector for env.step(...)

full_node_action:
  node-aligned action tensor for replay buffer and TD3 critic
```

The thesis claim should therefore be framed as a physically aligned
control-architecture contribution first, with behavioural quality, operational
metrics, interpretability, and scalability assessed through controlled
experiments.

## 2. Completed Evidence

| Evidence category | Status | What it supports | Important limits |
| --- | --- | --- | --- |
| 25CP formal behavioural comparison | Completed. Five paired seeds, 50k training steps, 30 controlled evaluation episodes per seed. | Supports behavioural improvement of hierarchical over controlled ActionGNN at 25CP. README reports mean reward improving from `-103,412.13` to `-76,084.21` with paired `p=0.0219`, and action saturation reducing from `0.5237` to `0.3025` with paired `p=0.0104`. | Applies to the 25CP PublicPST setting. It does not by itself prove large-scale superiority. |
| 100CP pre-optimisation hierarchical runtime bottleneck | Completed. Formal 100CP training exposed a large runtime gap: controlled ActionGNN about `02:13:09`, hierarchical about `11:08:52`. | Establishes that the original hierarchical actor was computationally expensive despite completing 100CP formal training. | This is primarily runtime evidence. Pre-optimisation 100CP behavioural results were favourable on mean values but not statistically robust across five seeds. |
| PR #4 component-level profiling and Approach A optimisation evidence | Completed through the vectorised hierarchical composition work merged in PR #4. The bottleneck diagnosis identified `_compose_full_node_action()` as the dominant cost, especially in replay batches. Approach A introduced hybrid per-graph vectorisation while preserving the actor, critic, replay-buffer, graph-state, reward, training, evaluation, and EV2Gym contracts. | Supports the engineering decision that Approach A is the right current optimisation path. It preserves thesis semantics while removing the per-EV/per-charger/per-transformer Python-heavy production composition path. | Component-level profiling and equivalence evidence are technical validation, not formal behavioural evidence. Approach A does not prove 500CP training success by itself. |
| 100CP post-optimisation 10k runtime pilot | Completed in PR #5 documentation. Job `58406334` ran hierarchical 100CP for 10k steps on M3 CPU4 in `00:35:13` (`2113s`). | Shows the PR #4 optimisation transferred to shorter end-to-end 100CP hierarchical training runtime. Reported speed-up was `3.37x` vs pre-opt seed0 10k elapsed and `3.59x` vs the pre-opt five-seed mean. | Runtime pilot only. Not matched-seed behavioural evidence and not an ActionGNN-vs-hierarchical evaluation. |
| 100CP post-optimisation 50k runtime confirmation | Completed in PR #5 documentation. Job `58410107` ran hierarchical 100CP for 50k steps on M3 CPU4 in `01:58:03` training wall-time and `01:58:26` Slurm elapsed. | Confirms 100CP hierarchical 50k training is runtime-feasible after PR #4. Reported speed-up was `5.70x` vs pre-opt seed0 final elapsed and `5.89x` vs the pre-opt five-seed mean. | Single seed0 runtime confirmation. Final/best rewards are incidental smoke information and must not be treated as formal behavioural evidence. |

## 3. Current Valid Claims

- The 25CP controlled evidence supports a behavioural improvement of the
  hierarchical actor over the controlled ActionGNN baseline.
- 100CP hierarchical training is now runtime-feasible on the documented M3 CPU4
  path after the Approach A vectorised composition optimisation.
- Approach A is sufficient for the current engineering path because it removed
  the dominant Python-heavy composition bottleneck while preserving the
  established actor/critic/replay/env contracts.
- Approach B is deferred. It is not justified unless future 500CP evidence shows
  that residual graph-level iteration or another post-Approach-A bottleneck
  blocks the thesis experiments.

## 4. Current Invalid Or Incomplete Claims

- Do not claim full 100CP behavioural superiority unless the thesis uses formal
  matched-seed evaluation evidence that supports that claim. Existing 100CP
  formal results are favourable on mean reward and saturation but not
  statistically robust across five seeds.
- Do not claim 500CP behavioural superiority yet. No formal 500CP baseline-vs-
  hierarchical evidence is currently established.
- Do not treat 100CP runtime-pilot reward or the post-optimisation 50k seed0
  reward as formal behavioural evidence. Those jobs answer runtime feasibility.
- Do not claim 1000CP feasibility before smoke/runtime tests show that the
  pipeline starts, trains, and fits within a defensible resource envelope.
- Do not present runtime optimisation as the full thesis contribution. Runtime
  feasibility is an enabling result for the behavioural and operational
  comparison.

## 5. Remaining Experiment Matrix

| Scale | Comparison | Current status | Required thesis use |
| --- | --- | --- | --- |
| 25CP | Baseline ActionGNN vs hierarchical actor | Completed. Formal behavioural comparison supports hierarchical improvement. | Use as the small-scale controlled evidence anchor. |
| 100CP | Baseline ActionGNN vs hierarchical actor | Needs post-optimisation behavioural consolidation or eval check. Pre-optimisation formal evidence exists but is not statistically robust; post-optimisation 10k/50k jobs are runtime evidence only. | Decide whether existing formal 100CP evidence is sufficient with careful caveats, or run/aggregate post-optimisation matched-seed evaluation if the thesis needs a stronger 100CP claim. |
| 500CP | Baseline ActionGNN vs hierarchical actor | Required next. Formal baseline-vs-hierarchical evidence is not yet established. | Minimum next large-scale benchmark. First determine whether both algorithms are feasible under matched resource and evaluation settings, then consolidate behavioural and operational metrics. |
| 1000CP | Feasibility/stretch target | Optional. No feasibility claim yet. | Consider only after 500CP baseline and hierarchical feasibility are understood. Start with smoke/runtime tests, not formal 1000CP training. |

## 6. Figure And Table Plan

The paper should exceed the minimum visual standard of the baseline EV-GNN paper
by showing not only reward but also hierarchy, operational behaviour,
scalability, and allocation structure.

| Planned item | Purpose | Suggested content |
| --- | --- | --- |
| Architecture figure | Make the contribution visible. | Side-by-side baseline flat ActionGNN actor and hierarchical CPO → Transformer → Charger → EV actor, while showing the shared EV2Gym graph state and final EV2Gym action interface. |
| Multi-scale reward distribution plot | Show behavioural evidence across scale. | Seed-level or episode-level reward distributions for 25CP, 100CP, and 500CP when available. Mark which scales are formal and which are pending. |
| Training curve plot | Show learning stability and convergence. | Mean evaluation reward over training steps for baseline and hierarchical runs at each formal scale. Include confidence bands or seed traces. |
| Operational metric comparison table | Tie reward to grid-relevant behaviour. | Tracking error, action saturation, service/charging delivery, overload/violation metrics, and any PublicPST-specific operational metrics already produced by the evaluator. |
| Runtime scalability plot | Separate feasibility from quality. | Wall-time or throughput vs CP scale for baseline and hierarchical actors, with pre- and post-optimisation hierarchical points clearly distinguished. |
| Action-distribution / saturation / hierarchy-allocation figure | Make actor behaviour interpretable. | Action histograms, fraction-at-max comparisons, and hierarchy allocation summaries across transformers/chargers/EV gates. |
| Optional generalisation plot | Test transfer if evidence exists. | Train on one scale or seed family and evaluate on another scenario/scale only if the protocol is well controlled. |
| Final algorithm comparison table | Summarise thesis claims. | Baseline EV-GNN ActionGNN vs hierarchical actor across architecture, reward evidence, operational metrics, runtime, interpretability, and scale coverage. |

## 7. M3 Resource Planning

- The 100CP post-optimisation 50k confirmation used about `01:58:26` of a
  requested 6-hour allocation, or roughly `33%` of the requested time.
- Future repeated 100CP 50k post-optimisation jobs should request about `3h`,
  leaving practical headroom without asking for the old 6-hour allocation.
- New 500CP pilots may start conservatively because 500CP end-to-end training
  evidence is not yet established.
- Once 500CP runtime evidence exists, reduce future time requests to match the
  observed wall-time plus a defensible buffer.
- CPU, memory, and wall-time requests should be updated from measured evidence,
  not from pre-optimisation hierarchical runtime assumptions.

## 8. Next Recommended Action

Prepare a 500CP pilot/formal experiment plan before launching jobs. The plan
should first determine whether both the controlled ActionGNN baseline and the
hierarchical actor can train and evaluate at 500CP under comparable settings.

Do not jump directly to 1000CP formal training. The current thesis path needs
500CP baseline and hierarchical feasibility first, followed by controlled
behavioural and operational evidence if the feasibility check is successful.

The recommended order is:

1. Define the 500CP resource, seed, timestep, evaluation, logging, and artifact
   protocol.
2. Run small 500CP smoke/runtime pilots for both baseline and hierarchical
   paths.
3. Use measured runtime to set formal 500CP Slurm allocations.
4. Launch matched 500CP formal training/evaluation only after both paths are
   feasible.
5. Treat 1000CP as an optional stretch target after 500CP evidence is stable.
