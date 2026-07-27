# Per-Infrastructure Saturation Diagnostics Plan

## Purpose

This note records the diagnostic plan for testing whether the hierarchical actor's lower global maximum-action saturation reflects infrastructure-aware allocation rather than simply lower charging intensity.

## Context

The controlled multiscale formal result for M3 job 58513929 showed that the strongest cross-scale empirical signal is lower global `action_fraction_at_max` for the hierarchical actor across 25cp, 100cp, 500cp, and 1000cp. This does not establish universal reward superiority. The diagnostic layer therefore focuses on mechanism validation.

## Diagnostic distinction

### All-slot saturation

All-slot saturation uses the full EV2Gym action vector denominator. It is retained as a compatibility metric because it reconciles with the canonical eval30 `action_fraction_at_max`.

### Active-only saturation

Active-only saturation uses only currently active EV action slots from `state.action_mapper`. It is the primary diagnostic metric because it isolates actor decisions over active EVs and avoids inactive slots dominating the denominator.

### Active slot count versus canonical non-zero action count

The canonical evaluator's historical `active_action_count_mean` is a non-zero mapped-action count over the full action vector, not the number of active EV decision slots. The diagnostics therefore use:

- `active_slot_count_mean`: mean pre-step active EV slot count from `state.action_mapper`
- `nonzero_action_count_mean_all_slots`: canonical-compatible mean all-slot non-zero mapped-action count
- `global_action_nonzero_fraction_active`: pooled fraction of active EV decisions whose absolute mapped action is above tolerance
- `inactive_nonzero_action_fraction_all_slots`: pooled fraction of inactive-slot action entries whose absolute mapped action is above tolerance; this should be zero for deterministic mapped actions

### Concentration metrics

Schema v2 treats zero-pressure allocation steps explicitly. A zero-pressure step is a step where the positive active-action pressure allocated across the relevant infrastructure level is at or below the diagnostic tolerance. Such steps are reported through `*_allocation_zero_pressure_step_fraction` and `*_allocation_valid_step_count`, but they are excluded from HHI/Gini means so lower charging intensity cannot appear as better allocation balance. If an episode has no positive-pressure steps for that allocation level, the corresponding mean HHI/Gini field is blank rather than a false zero.

Per-transformer charger concentration follows the same rule and reports `charger_allocation_zero_pressure_step_fraction` plus `charger_allocation_valid_step_count` in `transformer_diagnostics.csv`.

### Macro versus micro action saturation

The global active saturation metric remains pooled across active EV decision slots and is therefore a micro decision-level metric. Infrastructure-level macro means are named explicitly in schema v2:

- `transformer_action_fraction_at_max_active_macro_mean`
- `charger_action_fraction_at_max_active_macro_mean`

The same macro convention is used for infrastructure-level active non-zero fractions.

## Implemented diagnostic tooling

The diagnostic implementation adds:

- `evaluate_td3_gnn_infrastructure_diagnostics.py`
- `utils/infrastructure_diagnostics.py`
- focused tests in `tests/test_infrastructure_diagnostics.py`

The diagnostic evaluator writes separate outputs and does not modify the canonical eval30 CSV schema.

## Diagnostic schema versions

### Version 1

Initial diagnostic schema for separate episode, transformer, charger, and seed-summary outputs.

### Version 2

Schema v2 hardens metric semantics before the full multiscale diagnostic run:

- excludes zero-pressure steps from transformer/charger HHI and Gini means
- writes zero-pressure fractions and valid-step counts for global allocation concentration summaries
- writes per-transformer charger zero-pressure fractions and valid-step counts
- adds active non-zero action fractions for global, transformer, and charger diagnostics
- adds `inactive_nonzero_action_fraction_all_slots` to validate the inactive-slot action contract
- renames infrastructure action-at-max seed/episode macro fields to include `_macro_mean`

## Real-checkpoint smoke validation

A one-episode real-checkpoint smoke validation was run for:

- 25cp ActionGNN seed0, episode_seed 710000
- 25cp hierarchical seed0, episode_seed 710000

Both runs produced:

- `episode_diagnostics.csv`
- `transformer_diagnostics.csv`
- `charger_diagnostics.csv`
- `seed_summary_diagnostics.csv`

The diagnostic episode 0 outputs reconciled with the corresponding canonical eval30 episode 0 rows within tolerance. After schema clarification, `nonzero_action_count_mean_all_slots` reconciles with the canonical evaluator's `active_action_count_mean`.

## Full diagnostic run plan

The next formal diagnostic run should re-evaluate the existing job 58513929 checkpoints without retraining:

- 4 scales: 25cp, 100cp, 500cp, 1000cp
- 2 algorithms: actiongnn, hierarchical
- 5 training seeds
- 30 deterministic evaluation episodes per seed

Outputs should remain outside the Git repository and be bundled separately for audit.

## Inference rules

- Training seed remains the inference unit.
- Evaluation episodes are within-seed deterministic replicates, not independent samples.
- Transformer and charger rows are nested diagnostics and must not replace seed-level paired inference.
- No episode-level p-values should be reported.
- Primary mechanism inference is a within-scale paired ActionGNN versus hierarchical comparison at seed level.
- Raw HHI magnitudes should not be directly compared across 25cp, 100cp, 500cp, and 1000cp scales without a rigorously justified normalisation for infrastructure count and active-infrastructure semantics.

## Claim boundaries

These diagnostics can support or reject the mechanism claim that lower global saturation reflects more infrastructure-aware allocation. They do not by themselves establish:

- external SOTA superiority
- SB3 comparison superiority
- universal reward superiority
- deployment readiness

If lower saturation is accompanied by lower service, lower energy delivery, or worse user satisfaction, the interpretation must state that directly.

## Next steps

1. Run the full diagnostic matrix against existing checkpoints.
2. Validate all diagnostic CSV schemas and canonical reconciliation.
3. Aggregate diagnostics to seed level.
4. Compare hierarchical minus ActionGNN at seed level.
5. Create a separate docs-only diagnostic results note after audit.
