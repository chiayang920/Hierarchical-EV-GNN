# EV Charging Infrastructure Control Analysis Design

## Purpose

Create a small, maintainable analysis module that converts the authoritative full per-infrastructure diagnostic evidence bundle into analysis-ready datasets and statistically compares controlled ActionGNN and Hierarchical controllers.

The module exists to answer two questions:

1. Does the hierarchical controller improve Power Setpoint Tracking control quality?
2. When behaviour differs, is the difference consistent with infrastructure-aware allocation rather than merely lower charging intensity or reduced service?

The implementation must remain independent of temporary workflow labels such as `Stage D`, `R5L`, Git job IDs, and commit hashes. These identifiers remain provenance values only.

## Scope

### Included

- Read the immutable diagnostic evidence bundle without modifying it.
- Verify bundle identity and required schema before analysis.
- Extract four semantic datasets:
  - `episode_metrics.csv`
  - `seed_metrics.csv`
  - `transformer_metrics.csv`
  - `charger_metrics.csv`
- Pair ActionGNN and Hierarchical observations by `scale` and `training_seed`.
- Produce scale-specific statistical comparison tables.
- Preserve exact provenance in `provenance.json`.
- Document metric definitions, aggregation rules, directionality, and interpretation limits.
- Add focused automated tests for extraction, schema validation, pairing, aggregation, and statistical calculations.

### Excluded

- Retraining or diagnostic re-evaluation.
- M3 or Slurm workflows.
- Changes to actors, critics, reward, simulator, evaluator, checkpoint handling, or reconciliation logic.
- Automatic analysis of every available column.
- Economic and battery-degradation claims in the primary analysis.
- Arm C implementation or training.
- Renaming or rewriting the authoritative raw evidence bundle.

## Repository structure

```text
analysis/ev_charging_infrastructure_control/
├── README.md
├── metric_definitions.py
├── extract_diagnostic_datasets.py
└── compare_control_architectures.py

tests/analysis/
└── test_ev_charging_infrastructure_control.py
```

### Responsibilities

#### `metric_definitions.py`

Single source of truth for:

- semantic metric name;
- source column;
- source level;
- analysis tier;
- expected direction;
- aggregation rule;
- interpretation warning.

It must contain data declarations and small validation helpers only. It must not read files, run statistics, or generate outputs.

#### `extract_diagnostic_datasets.py`

Responsibilities:

- accept explicit `--bundle` and `--output-dir` arguments;
- validate the bundle and required members;
- read nested task packages directly from archives;
- validate required columns, key uniqueness, row counts, schema version, and provenance;
- write deterministic CSV datasets and `provenance.json`;
- fail before writing partial outputs when validation fails.

It must not run inferential statistics.

#### `compare_control_architectures.py`

Responsibilities:

- read analysis-ready datasets;
- aggregate episodes and infrastructure rows to the training-seed level;
- pair algorithms strictly by `scale` and `training_seed`;
- calculate descriptive and paired statistics;
- write semantic result tables;
- fail on missing, duplicate, or unbalanced pairs.

It must not read the raw evidence bundle directly.

#### `README.md`

Document:

- scientific purpose;
- required inputs;
- exact commands;
- generated outputs;
- metric tiers;
- statistical unit;
- interpretation limitations;
- provenance policy.

## Naming policy

Human-facing filenames and directories must describe their scientific purpose.

Allowed examples:

```text
ev_charging_infrastructure_control_analysis/
episode_metrics.csv
paired_control_comparisons.csv
provenance.json
```

Disallowed examples:

```text
stage_d/
r5l/
job58745233/
cbf4b5fe_results.csv
```

Job IDs, commit SHAs, and checksums must remain inside provenance fields. Existing authoritative raw filenames remain unchanged because their names and checksums are already part of the audit trail.

Variable names must be semantic. Examples:

```text
diagnostic_array_job_id
formal_training_job_id
source_commit_sha
evidence_bundle_sha256
paired_seed_metrics
```

Avoid generic names such as `data`, `df2`, `result1`, or temporary workflow abbreviations.

## Input and output contract

### Input

One immutable complete diagnostic evidence bundle validated as:

- 8 task packages;
- 40 checkpoint groups;
- 1,200 episodes;
- diagnostic schema version 3;
- reconciliation contract version 2.

The module must not assume a specific job-numbered filename. Identity is read from bundle metadata.

### Output directory

The caller supplies an explicit output directory. Generated analysis outputs must not be committed by default.

Recommended external location:

```text
~/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis/
```

Expected structure:

```text
ev_charging_infrastructure_control_analysis/
├── datasets/
│   ├── episode_metrics.csv
│   ├── seed_metrics.csv
│   ├── transformer_metrics.csv
│   └── charger_metrics.csv
├── results/
│   ├── paired_control_comparisons.csv
│   ├── scale_level_summary.csv
│   └── metric_interpretation.md
└── provenance.json
```

The implementation must use temporary files and atomic replacement so failed runs do not leave apparently complete outputs.

## Metric design

### Tier 1: Primary control outcomes

- `episode_reward`
- `tracking_error`
- `energy_tracking_error`
- `power_tracker_violation`

These directly evaluate the Power Setpoint Tracking objective and are the only primary inferential outcomes.

### Tier 2: Core mechanism outcomes

- `global_action_fraction_at_max_active`
- `global_action_nonzero_fraction_active`
- `transformer_action_fraction_at_max_active_macro_mean`
- `charger_action_fraction_at_max_active_macro_mean`
- `transformer_positive_charge_action_hhi_mean`
- `charger_positive_charge_action_hhi_mean`
- `transformer_allocation_zero_pressure_step_fraction`
- `charger_allocation_zero_pressure_step_fraction`

These are exploratory behavioural measures. Lower saturation or concentration must not automatically be interpreted as better control.

### Tier 3: Physical safety outcomes

- `total_transformer_overload`
- seed-level mean transformer overload frequency;
- seed-level total transformer overload magnitude;
- seed-level maximum transformer overload magnitude.

The three transformer-row aggregations must be fixed before observing comparative results:

1. aggregate transformer rows within each episode;
2. aggregate the 30 episode values within each training seed;
3. pair controllers by scale and training seed.

### Tier 4: Service guardrails

- `total_ev_served`
- `total_energy_charged`
- `average_user_satisfaction`
- `energy_user_satisfaction`

These determine whether reduced saturation reflects improved coordination or simply lower service intensity.

### Robustness-only metrics

- `transformer_positive_charge_action_gini_mean`
- `charger_positive_charge_action_gini_mean`

Gini results support HHI interpretation but do not constitute separate primary claims.

### Deferred metrics

- total profit;
- battery degradation;
- negative-action and discharged-energy metrics while V2G is disabled;
- charging-station power maxima and means;
- action-bound integrity fields as scientific outcomes.

Integrity fields remain validation gates where applicable.

## Statistical design

### Unit of inference

The training seed is the independent statistical unit.

Correct hierarchy:

```text
infrastructure rows
→ episode-level aggregation
episode rows
→ 30-episode mean within each training seed
seed means
→ ActionGNN/Hierarchical pair by scale and training_seed
```

Episodes, transformers, and chargers must not be treated as independent replicates.

### Per-scale paired outputs

For every approved metric, report:

- ActionGNN mean;
- Hierarchical mean;
- paired absolute difference, defined as `Hierarchical - ActionGNN`;
- paired relative difference, with explicit zero-denominator handling;
- 95% confidence interval for the paired mean difference;
- paired t-test;
- Wilcoxon signed-rank test where mathematically defined;
- paired standardised effect size;
- number of valid paired seeds.

### Multiplicity

- Apply Holm correction across the four Tier 1 metrics within each scale.
- Tier 2–4 and robustness metrics remain explicitly exploratory.
- Do not apply a broad correction across all exploratory metrics and then present corrected significance as a primary claim.

### Small-sample interpretation

Each scale has five paired training seeds. Results must emphasise direction, magnitude, confidence interval, and consistency. P-values are supporting evidence rather than the sole decision criterion.

## Validation gates

Extraction must fail on:

- checksum or archive-integrity failure;
- unsafe archive members;
- unexpected schema or reconciliation version;
- missing required files or columns;
- duplicate episode, transformer, charger, or seed keys;
- row-count mismatch;
- missing scale, algorithm, or training-seed cells;
- non-finite values in required analysis metrics, except explicitly documented unavailable metrics;
- provenance disagreement between outer and nested evidence.

Comparison must fail on:

- missing algorithm pairs;
- duplicate seed pairs;
- unequal seed sets;
- fewer or more than five paired seeds per scale for this formal experiment;
- unsupported metric definitions;
- silent denominator substitution in relative-change calculations.

## Testing strategy

Use one focused test file with small synthetic archive fixtures and isolated helper tests.

Required coverage:

1. valid extraction produces the four named datasets and provenance file;
2. invalid checksum or unsafe archive member is rejected;
3. missing required columns or duplicate keys are rejected;
4. exact scale/algorithm/seed pairing is enforced;
5. episode-to-seed and infrastructure-to-seed aggregation is correct;
6. paired differences follow `Hierarchical - ActionGNN`;
7. relative differences handle zero denominators explicitly;
8. paired t-test, Wilcoxon, confidence interval, Holm correction, and effect size are tested against fixed oracle values;
9. human-facing outputs do not contain job IDs or commit hashes in filenames;
10. an optional read-only smoke test can validate the authoritative local bundle when its path is explicitly supplied.

Do not add large binary fixtures, duplicate production validators, or temporary CI workflows.

## Risks and prevention

### Pseudo-replication

Risk: treating 1,200 episodes or infrastructure rows as independent samples.

Prevention: aggregation to one value per scale, algorithm, training seed before inference.

### Pairing errors

Risk: comparing unmatched seeds or scales.

Prevention: strict composite-key validation and exact five-pair assertions.

### Metric proliferation

Risk: selecting favourable metrics after inspecting results.

Prevention: approved tiered metric registry committed before statistical execution.

### Ambiguous directionality

Risk: interpreting lower concentration, lower energy, or lower saturation as inherently better.

Prevention: encode interpretation warnings and report control, mechanism, safety, and service outcomes together.

### Provenance loss

Risk: removing unreadable IDs from filenames also removes traceability.

Prevention: semantic filenames plus complete provenance fields in `provenance.json`.

### Repository pollution

Risk: committing generated datasets or evidence bundles.

Prevention: explicit external output directory, documentation, and existing or targeted ignore rules only when necessary.

### Over-engineering

Risk: adding configuration frameworks, class hierarchies, wrappers, or M3 automation not required for this analysis.

Prevention: four implementation files, one focused test file, direct command-line interfaces, and no new abstraction without a demonstrated second use case.

## Acceptance criteria

The implementation is complete only when:

- the authoritative bundle remains byte-identical;
- extraction deterministically produces four validated datasets and provenance;
- comparison produces complete paired results for four scales and five seeds;
- no inference uses episode, transformer, or charger rows as independent replicates;
- primary, exploratory, robustness, and deferred metrics are clearly distinguished;
- all focused and full repository tests pass;
- Python compilation and diff checks pass;
- generated filenames are semantic and contain no hashes or job IDs;
- no training, evaluation, reducer, Slurm, or M3 action occurs.
