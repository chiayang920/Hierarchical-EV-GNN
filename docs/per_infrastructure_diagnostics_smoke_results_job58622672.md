# Per-Infrastructure Diagnostics Smoke Results — M3 Job 58622672

Date: 2026-07-29

## Purpose

This smoke validated the complete schema-v3 per-infrastructure diagnostic workflow before the full 40-checkpoint deterministic eval30 re-evaluation.

It tested whether existing formal checkpoints could be loaded, evaluated, reconciled, accounted for, and packaged across 25CP, 100CP, 500CP, and 1000CP.

It did not retrain any model.

## Provenance

| Field | Value |
|---|---|
| Repository | `chiayang920/Hierarchical-EV-GNN` |
| Source commit | `6e5fd7f4c6d1f8cb5647263b37b3db1ba02f015c` |
| Formal checkpoint source job | `58513929` |
| Diagnostic array job | `58622672` |
| Reducer job | `58622673` |
| Diagnostic schema | `3` |
| Final archive SHA-256 | `7e7371d5e8a85b377bce9d488cec592dc827dde39b289f8eee712a473097cf7b` |

Raw archives, logs, task packages, and source provenance remain outside Git.

## Scope

Eight tasks were executed:

- four scales;
- two algorithms;
- one selected checkpoint per scale-algorithm pair;
- one deterministic diagnostic episode per task.

This is eight diagnostic episodes in total.

## Integrity result

- 8/8 array tasks completed with exit code `0:0`;
- reducer completed with exit code `0:0`;
- 8/8 task packages validated;
- 40/40 complete-file checksums matched;
- 152/152 canonical reconciliation rows passed;
- 8/8 service reconciliation rows passed;
- schema v3 was present in every task;
- failure and warning inventories were empty;
- stderr logs were empty;
- no unsafe archive path, duplicate member, link, or device entry was detected.

## Descriptive paired result

| Scale | Active max-action saturation: ActionGNN → Hierarchical | Charged-energy change | Average satisfaction change | Transformer overload: ActionGNN → Hierarchical |
|---|---:|---:|---:|---:|
| 25CP | 100.00% → 49.91% | −2.99% | −0.345 percentage points | 0.00 → 0.00 |
| 100CP | 100.00% → 80.27% | −4.21% | −0.723 percentage points | 26.46 → 4.14 |
| 500CP | 99.73% → 86.23% | −5.51% | −1.129 percentage points | 225.83 → 97.35 |
| 1000CP | 100.00% → 60.81% | −0.20% | −0.020 percentage points | 267.87 → 268.18 |

Both algorithms served the same number of EVs at each scale. The hierarchical actor reduced max-action saturation in every smoke task, but infrastructure-level benefit was scale-dependent.

## Claim boundary

This smoke is a mechanism and workflow check. It is not the complete diagnostic evidence and does not support seed-level statistical inference.

The next stage must evaluate:

```text
4 scales × 2 algorithms × 5 checkpoints × 30 deterministic episodes
= 1,200 episodes
```

Training seed remains the inference unit. Episode-level p-values are invalid.

## Next stage

Proceed to the full per-infrastructure deterministic eval30 re-evaluation using eight scale-algorithm tasks, with five formal checkpoints and 30 episodes handled inside each task.

The final external evidence directory should use the semantic name:

```text
EVGNN_per_infrastructure_diagnostics_40checkpoints_eval30_job<ARRAY_JOB_ID>_<YYYYMMDD>
```
