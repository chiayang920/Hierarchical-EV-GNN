# 500CP Experiment Plan

This document defines the next controlled 500CP experiment path for the thesis.
It is a planning document only. It does not report new 500CP experimental
results.

## Objective

Establish whether 500CP training and evaluation are feasible for both the
controlled ActionGNN baseline and the hierarchical actor, then prepare the
formal baseline-vs-hierarchical comparison.

The experiment path must keep feasibility/runtime evidence separate from
behavioural evidence:

- Feasibility/runtime evidence answers whether a run starts, trains, finishes,
  fits within memory, and has a defensible wall-time allocation.
- Behavioural evidence answers whether one trained algorithm performs better
  than the other under matched seeds, matched 50k training length, and
  controlled eval30.

## Current Evidence Basis

| Scale | Evidence status | Interpretation boundary |
| --- | --- | --- |
| 25CP | Formal behavioural comparison completed. Five paired seeds, 50k training, and controlled eval30 support the existing small-scale thesis evidence. | Valid behavioural evidence for 25CP only. It does not establish 500CP superiority. |
| 100CP | Post-optimisation hierarchical 50k runtime confirmation completed in `7083s` training wall-time. | Runtime feasibility evidence for hierarchical 100CP after optimisation. It is not matched baseline-vs-hierarchical behavioural evidence. |
| 500CP | Formal comparison is still missing. | No 500CP superiority, feasibility, or formal behavioural claim should be made yet. |

## Required Algorithms

The 500CP path must include both canonical algorithms:

- Controlled ActionGNN baseline, using the controlled baseline actor path.
- Hierarchical actor, using the physically aligned actor path.

Both algorithms must keep the existing EV2Gym action interface, TD3
critic/replay-buffer contract, reward setting, and evaluation protocol aligned.

## Required Stages

### Stage 1: 500CP Source/Config Preflight

Purpose: verify that the intended 500CP source state and configuration are
ready before any Slurm job is launched.

Checks to record:

- branch or commit provenance;
- 500CP config path and immutable copy strategy;
- algorithm labels: `actiongnn` and `hierarchical`;
- training length for later formal runs: `50000` steps;
- pilot length for smoke/runtime runs: `10000` steps;
- seed plan, with matched seeds for the formal comparison;
- output directories, run names, logs, checkpoints, copied config, and
  `run_args.yaml`;
- expected evaluation protocol: eval30 after successful training;
- no source-code, Slurm-script, or config mutation as part of this plan.

Exit criterion: both algorithms have a clearly specified command path and
artifact path for 500CP pilots.

### Stage 2: 500CP Baseline 10k Smoke/Runtime Pilot

Purpose: establish whether the controlled ActionGNN baseline can start and
complete a short 500CP training run.

Record:

- training wall-time;
- MaxRSS;
- throughput in steps per second;
- final success or failure status;
- any failure mode, including timeout, memory pressure, environment failure,
  checkpoint failure, or logging failure.

Reward during this pilot is smoke information only. It must not be compared as
a formal behavioural result.

### Stage 3: 500CP Hierarchical 10k Smoke/Runtime Pilot

Purpose: establish whether the hierarchical actor can start and complete a
short 500CP training run under the same pilot length.

Record the same runtime, memory, throughput, artifact, and failure/success
fields as the baseline pilot.

Reward during this pilot is smoke information only. It must not be used to
claim behavioural superiority or infer final 50k quality.

### Stage 4: Runtime-Based Slurm Allocation Decision

Purpose: set the formal 500CP allocation from measured 10k evidence instead of
blindly reusing an old allocation.

Decision process:

- compare baseline and hierarchical 10k wall-time and throughput;
- estimate each 50k runtime from measured pilot throughput;
- include startup, evaluation, packaging, filesystem, and scheduler buffer;
- set wall-time separately for baseline and hierarchical if their measured
  runtimes differ materially;
- document CPU, memory, wall-time, and observed MaxRSS before launching formal
  training.

Exit criterion: the formal 50k allocation is justified by observed 500CP
runtime plus buffer.

### Stage 5: 500CP Formal 50k Training

Purpose: run formal 500CP training only if both pilot paths are feasible.

Requirements:

- run both algorithms;
- use matched seeds;
- use the same 500CP scenario family and reward setting;
- train each run for `50000` steps;
- preserve per-run artifacts, copied configs, checkpoints, training logs, and
  runtime metadata;
- record success/failure per seed and per algorithm.

If either algorithm cannot complete the pilot stage within a defensible resource
envelope, do not proceed to formal behavioural comparison until the blocker is
resolved or explicitly documented as a feasibility limitation.

### Stage 6: 500CP Eval30

Purpose: evaluate trained 500CP policies only after successful formal training.

Requirements:

- run controlled eval30 for each successful 50k checkpoint;
- keep evaluation seeds/protocol matched across algorithms;
- aggregate per-episode and per-seed results;
- separate failed or missing checkpoints from successful evaluation rows.

Exit criterion: paired 500CP eval30 data exists for the formal comparison, or
the missing data is documented as a feasibility limitation.

## Resource Planning

Do not reuse a 6-hour allocation blindly. The 100CP post-optimisation 50k
confirmation finished in `7083s`, but 500CP end-to-end training evidence has
not yet been established.

For the first 500CP smoke/runtime pilots:

- start conservatively because both 500CP end-to-end paths are still
  unconfirmed;
- request enough wall-time to observe completion or a meaningful failure mode;
- record elapsed time, training wall-time, MaxRSS, throughput, and status;
- avoid interpreting a timeout alone as a behavioural result.

After the smoke/runtime pilots:

- reduce future time requests based on observed runtime plus buffer;
- use the slower measured path to plan matched formal schedule feasibility;
- avoid carrying forward excessive buffer once measured 500CP runtime exists;
- document the buffer used for formal 50k runs.

## Metrics To Collect

Collect the following for both algorithms and for each relevant pilot, training,
and evaluation stage when available:

- reward;
- energy tracking error;
- power tracker violation;
- total EV served;
- action_fraction_at_max;
- training wall-time;
- MaxRSS;
- throughput;
- failure/success status.

## Expected Later Outputs

The 500CP evidence should feed the thesis outputs below after formal training
and eval30 are complete:

- multi-scale reward distribution;
- 25/100/500CP algorithm comparison table;
- runtime scalability plot;
- operational metric comparison;
- action saturation/action distribution plot;
- training curves.

## Claim Boundaries

- Do not claim 500CP superiority before formal eval evidence exists.
- Do not compare runtime-pilot reward as a formal behavioural result.
- Do not treat 500CP feasibility as behavioural superiority.
- Do not present single-seed runtime confirmation as matched-seed evidence.
- Do not move to 1000CP until 500CP feasibility is established.

## Ready-To-Proceed Criteria

The thesis path is ready for formal 500CP baseline-vs-hierarchical comparison
only when:

- source/config preflight is recorded;
- baseline 10k smoke/runtime pilot completes or has a documented,
  resolvable failure mode;
- hierarchical 10k smoke/runtime pilot completes or has a documented,
  resolvable failure mode;
- formal Slurm wall-time and memory requests are based on measured 500CP
  runtime plus buffer;
- both algorithms are expected to complete 50k training under defensible,
  comparable resource assumptions.

