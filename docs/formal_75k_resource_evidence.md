# Formal 75k Resource Evidence

## Status

```text
RESOURCE_DEFAULTS_GUESSED=NO
RESOURCE_PROFILE_STATUS=UNAPPROVED_TEMPLATE_ONLY
TRAINING_MAXRSS_EVIDENCE=INSUFFICIENT_FOR_NEW_75K_MATRIX
M3_SMOKE_READY=NO
```

No corrected non-negative 75k training run exists. Historical resource evidence informs planning but does not approve a new request.

## Historical formal evidence available

The supplied historical formal evidence for job `58513929` contains 40 canonical `model.best` eval30 files, 1,200 episode rows, task runtime metadata and source manifests. It covers the historical signed ActionGNN versus hierarchical comparison at 50,000 steps and five seeds. Its runtime envelope is not a measured requirement for the corrected non-negative 75k matrix.

Historical planning records include 100CP and 500CP pilot/runtime evidence and prior successful Slurm requests. These records may support a labelled scheduler-planning range, but they do not establish the corrected baseline's 75k `MaxRSS`, CPU utilisation or exact wall time.

## Historical diagnostic evidence available

The supplied completed Stage D bundle records:

```text
Array job=58745233
Reducer job=58746039
Task packages=8
Checkpoints=40
Episodes=1200
Diagnostic schema=3
Reconciliation contract=2
Source commit=cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad
```

Its runtime summary reports four allocated CPUs per historical diagnostic task and approximately 1.0–1.2 GiB `MaxRSS`. This is diagnostic evidence only; it is not training memory evidence and does not prove that the new Formal-75k diagnostic request must use four CPUs.

## Threading assessment

The training path does not use multiple environment processes or `DataLoader` workers. Historical jobs controlled PyTorch/BLAS threading through Slurm-related environment variables. The smallest defensible CPU request must therefore be selected after the local functional smoke and, where required, a narrowly scoped M3 smoke accounting review.

## Runtime planning rule

A historical 50k elapsed time may be multiplied by `1.5` only as a linear scheduler-planning extrapolation for a 75k horizon. It is not a guaranteed runtime and must be reported as a range by scale and algorithm when enough evidence exists.

## Approval sequence

1. User runs `run_local_formal75k_repair_verification.sh`.
2. User runs `run_local_formal75k_functional_smoke.sh`.
3. Returned logs and evidence are independently reviewed.
4. A separate smoke resource profile is approved for M3.
5. Only the two-cell M3 smoke may then be considered.
6. Formal Stage A resources are approved only after smoke accounting is reviewed.
