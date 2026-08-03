# EV Charging Infrastructure Control Analysis

This module converts the immutable full infrastructure diagnostic evidence
bundle into analysis-ready datasets, then compares ActionGNN and Hierarchical
controllers using paired training-seed statistics.

The training seed is the independent unit. Episodes, transformers, and chargers
are validation and aggregation rows only; they are not independent statistical
samples.

## Required Input

The input is the audited complete diagnostic evidence bundle:

```bash
$HOME/Downloads/EVGNN_Formal_Evidence/full_infrastructure_diagnostic_eval30_complete_evidence_job58745233.tar.gz
```

The job-numbered input basename remains unchanged because it is the immutable
audited source. Generated filenames are semantic and do not include job IDs,
commit hashes, checksums, `stage_d`, or `r5l`.

## Commands

```bash
conda activate evgnn_core

python analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  --bundle "$HOME/Downloads/EVGNN_Formal_Evidence/full_infrastructure_diagnostic_eval30_complete_evidence_job58745233.tar.gz" \
  --output-dir "$HOME/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis"

python analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  --analysis-dir "$HOME/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis"
```

The extraction command refuses to overwrite an existing output directory. The
comparison command refuses to overwrite an existing `results/` directory.

## Generated Outputs

```text
ev_charging_infrastructure_control_analysis/
├── datasets/
│   ├── charger_metrics.csv
│   ├── episode_metrics.csv
│   ├── seed_metrics.csv
│   └── transformer_metrics.csv
├── provenance.json
└── results/
    ├── metric_interpretation.md
    ├── paired_control_comparisons.csv
    └── scale_level_summary.csv
```

Expected production row counts:

```text
episode_metrics.csv=1200
seed_metrics.csv=40
transformer_metrics.csv=34500
charger_metrics.csv=487500
paired_control_comparisons.csv=88
scale_level_summary.csv=4
```

## Metrics

The registry contains exactly 22 approved metrics:

- Primary control outcomes: `episode_reward`, `tracking_error`,
  `energy_tracking_error`, `power_tracker_violation`.
- Mechanism outcomes: saturation, nonzero action, HHI, and zero-pressure
  allocation metrics.
- Physical safety outcomes: transformer overload level, mean overload
  frequency, mean total overload magnitude, and maximum overload magnitude.
- Service guardrails: EVs served, energy charged, and satisfaction metrics.
- Robustness checks: transformer and charger Gini concentration metrics.

Holm correction is applied only to the four primary paired t-test p-values
within each scale. Mechanism, physical safety, service, and robustness metrics
remain exploratory or guardrail evidence.

## Aggregation And Pairing

The required hierarchy is:

```text
infrastructure rows
→ episode-level aggregation
→ training-seed aggregation
→ ActionGNN/Hierarchical pairing by scale + training_seed
```

Paired differences are always `Hierarchical - ActionGNN`. Relative differences
are `100 * paired mean difference / abs(ActionGNN seed mean)` and use explicit
status fields when undefined. Cohen's dz also uses explicit status fields for
zero-variance cases.

Transformer overload aggregation is fixed as:

- mean overload frequency across transformers within each episode, then mean
  across episodes in the training seed;
- sum overload magnitude across transformers within each episode, then mean
  across episodes in the training seed;
- maximum overload magnitude across all transformer rows in the training seed.

## Provenance

`provenance.json` preserves the diagnostic array job ID, reducer job ID, formal
training job ID, source commit SHA, evidence bundle SHA-256, diagnostic schema
version, reconciliation contract version, task-package count, checkpoint-group
count, episode count, and dataset row counts.

This workflow does not train, evaluate, rerun the reducer, connect to M3, or
modify the authoritative evidence bundle.

## Interpretation Boundary

The result files are statistical evidence tables only. They do not make causal
claims or change metric selection after observing results. Scientific
interpretation belongs in a later reviewed phase.
