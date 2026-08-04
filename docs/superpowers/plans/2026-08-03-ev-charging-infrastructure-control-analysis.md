# EV Charging Infrastructure Control Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a maintainable local analysis module that converts the immutable infrastructure-diagnostic evidence bundle into validated semantic datasets and compares ActionGNN with the hierarchical controller using training-seed-level paired statistics.

**Architecture:** One extraction command validates the outer archive and eight nested task packages, then atomically publishes four CSV datasets and provenance. One comparison command reads only those generated datasets, aggregates to one value per scale, algorithm, training seed, enforces exact pairing, and publishes deterministic statistical tables.

**Tech Stack:** Python 3.11, Python standard library, NumPy, pytest. Do not add pandas, SciPy, statsmodels, or another dependency file. Adapt only the required, already-used statistical algorithms from `scripts/aggregate_controlled_multiscale_eval30.py` and verify them with fixed oracle tests.

## Global Constraints

- Work only on branch `analysis/ev-charging-infrastructure-control`.
- Base scientific source remains merged commit `cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad`.
- Do not modify model, simulator, reward, evaluator, diagnostic producer, Stage D validator, M3 scripts, checkpoints, or evidence files.
- Do not run M3, Slurm, training, evaluation, or reducer jobs.
- The authoritative archive remains outside Git and read-only.
- Human-facing paths must be semantic; do not use `stage_d`, `R5L`, job IDs, commit hashes, or checksums in generated names.
- Job IDs and hashes remain only as fields in `provenance.json`.
- Training seed is the only inferential unit. Do not use episodes, transformers, or chargers as independent samples for p-values.
- Approved metrics are fixed before result inspection. Do not auto-discover additional metrics.
- Holm correction applies only to the four primary metrics within each scale.
- Keep implementation to four analysis files and one focused test file.
- Generated outputs must remain outside the repository.

## Files

Create:

```text
analysis/ev_charging_infrastructure_control/
├── README.md
├── metric_definitions.py
├── extract_diagnostic_datasets.py
└── compare_control_architectures.py

tests/analysis/
└── test_ev_charging_infrastructure_control.py
```

Do not modify `.gitignore`, dependency files, existing aggregation scripts, or existing diagnostic validators unless a failing test proves a narrowly scoped change is necessary.

---

### Task 1: Implement the approved metric registry

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/metric_definitions.py`
- Create: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces `MetricDefinition`, `METRIC_DEFINITIONS`, `metric_definition`, `metric_names_for_tier`, and `required_source_columns`.

- [ ] **Step 1: Write failing registry tests**

Test exact metric membership, uniqueness, tiers, directionality, aggregation rules, and interpretation warnings.

```python
EXPECTED_METRICS = {
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "global_action_fraction_at_max_active",
    "global_action_nonzero_fraction_active",
    "transformer_action_fraction_at_max_active_macro_mean",
    "charger_action_fraction_at_max_active_macro_mean",
    "transformer_positive_charge_action_hhi_mean",
    "charger_positive_charge_action_hhi_mean",
    "transformer_allocation_zero_pressure_step_fraction",
    "charger_allocation_zero_pressure_step_fraction",
    "total_transformer_overload",
    "mean_transformer_overload_frequency_fraction",
    "mean_total_transformer_overload_magnitude",
    "maximum_transformer_overload_magnitude",
    "total_ev_served",
    "total_energy_charged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "transformer_positive_charge_action_gini_mean",
    "charger_positive_charge_action_gini_mean",
}


def test_registry_contains_only_approved_metrics():
    definitions = metric_module.METRIC_DEFINITIONS
    assert {definition.name for definition in definitions} == EXPECTED_METRICS
    assert len(definitions) == len(EXPECTED_METRICS)


def test_registry_preserves_scientific_boundaries():
    reward = metric_module.metric_definition("episode_reward")
    saturation = metric_module.metric_definition(
        "global_action_fraction_at_max_active"
    )
    assert reward.tier == "primary"
    assert reward.preferred_direction == "higher"
    assert saturation.tier == "mechanism"
    assert saturation.preferred_direction == "context_dependent"
    assert "not automatically" in saturation.interpretation_warning.lower()
```

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k registry
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the immutable registry**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    source_level: str
    source_column: str
    seed_summary_column: str | None
    tier: str
    preferred_direction: str
    aggregation_rule: str
    interpretation_warning: str
```

Define exactly 22 entries:

| Metric | Source | Seed-summary column | Tier | Direction | Aggregation |
|---|---|---|---|---|---|
| `episode_reward` | episode | none | primary | higher | mean 30 episodes |
| `tracking_error` | episode | `tracking_error_mean` | primary | lower | mean 30 episodes |
| `energy_tracking_error` | episode | `energy_tracking_error_mean` | primary | lower | mean 30 episodes |
| `power_tracker_violation` | episode | `power_tracker_violation_mean` | primary | lower | mean 30 episodes |
| `global_action_fraction_at_max_active` | episode | `global_action_fraction_at_max_active_mean` | mechanism | context-dependent | mean 30 episodes |
| `global_action_nonzero_fraction_active` | episode | `global_action_nonzero_fraction_active_mean` | mechanism | context-dependent | mean 30 episodes |
| `transformer_action_fraction_at_max_active_macro_mean` | episode | same name | mechanism | context-dependent | mean 30 episodes |
| `charger_action_fraction_at_max_active_macro_mean` | episode | same name | mechanism | context-dependent | mean 30 episodes |
| `transformer_positive_charge_action_hhi_mean` | episode | same name | mechanism | context-dependent | mean 30 episodes |
| `charger_positive_charge_action_hhi_mean` | episode | same name | mechanism | context-dependent | mean 30 episodes |
| `transformer_allocation_zero_pressure_step_fraction` | episode | `transformer_allocation_zero_pressure_step_fraction_mean` | mechanism | context-dependent | mean 30 episodes |
| `charger_allocation_zero_pressure_step_fraction` | episode | `charger_allocation_zero_pressure_step_fraction_mean` | mechanism | context-dependent | mean 30 episodes |
| `total_transformer_overload` | episode | `total_transformer_overload_mean` | physical_safety | lower | mean 30 episodes |
| `mean_transformer_overload_frequency_fraction` | transformer | none | physical_safety | lower | mean transformers per episode, then mean episodes |
| `mean_total_transformer_overload_magnitude` | transformer | none | physical_safety | lower | sum transformers per episode, then mean episodes |
| `maximum_transformer_overload_magnitude` | transformer | none | physical_safety | lower | maximum over all transformer rows in seed |
| `total_ev_served` | episode | `total_ev_served_mean` | service_guardrail | context-dependent | mean 30 episodes |
| `total_energy_charged` | episode | `total_energy_charged_mean` | service_guardrail | context-dependent | mean 30 episodes |
| `average_user_satisfaction` | episode | `average_user_satisfaction_mean` | service_guardrail | higher | mean 30 episodes |
| `energy_user_satisfaction` | episode | `energy_user_satisfaction_mean` | service_guardrail | higher | mean 30 episodes |
| `transformer_positive_charge_action_gini_mean` | episode | same name | robustness | context-dependent | mean 30 episodes |
| `charger_positive_charge_action_gini_mean` | episode | same name | robustness | context-dependent | mean 30 episodes |

`episode_reward` has no seed-summary counterpart because the supplied seed summary does not contain reward; it is calculated from episode rows only.

Every context-dependent metric must include a warning that lower or higher is not automatically favourable. Gini entries must state that they are robustness checks for HHI, not independent primary discoveries.

- [ ] **Step 4: Verify GREEN**

Run the focused registry tests.

- [ ] **Step 5: Commit**

```bash
git add analysis/ev_charging_infrastructure_control/metric_definitions.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Add EV infrastructure analysis metric registry"
```

---

### Task 2: Validate the complete and nested evidence archives

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces `Provenance`, `TaskPackage`, `CompleteBundleEvidence`, `validate_safe_archive_members`, `read_checksum_manifest`, `validate_manifest_coverage`, and `read_complete_bundle`.

- [ ] **Step 1: Build a small synthetic bundle fixture**

The fixture must create eight task packages for:

```python
SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
TRAINING_SEEDS = tuple(range(5))
```

Each nested package contains five seed directories with:

```text
seedN/diagnostics/episode_diagnostics.csv
seedN/diagnostics/seed_summary_diagnostics.csv
seedN/diagnostics/transformer_diagnostics.csv
seedN/diagnostics/charger_diagnostics.csv
checksums/package_file_checksums.sha256
```

The outer archive contains:

```text
task_packages/<package_name>
summaries/task_inventory.csv
runtime_metadata/diagnostic_array_job_id.txt
runtime_metadata/reducer_job_id.txt
runtime_metadata/formal_job_id.txt
runtime_metadata/source_commit_sha.txt
validation/complete_workflow_validation.json
runtime_metadata/complete_file_checksums.sha256
```

Synthetic CSVs use one episode per seed. Internal functions accept an expected episode count parameter for tests; the production CLI always fixes it to 30.

- [ ] **Step 2: Write failing archive tests**

```python
def test_complete_bundle_rejects_unsafe_member(tmp_path):
    bundle = build_synthetic_complete_bundle(
        tmp_path, extra_outer_member=("../escape.txt", b"unsafe")
    )
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extraction_module.read_complete_bundle(bundle)


def test_complete_bundle_rejects_package_sha_mismatch(tmp_path):
    bundle = build_synthetic_complete_bundle(
        tmp_path, corrupt_task_package_sha=True
    )
    with pytest.raises(RuntimeError, match="task package SHA-256 mismatch"):
        extraction_module.read_complete_bundle(bundle)


def test_complete_bundle_requires_exact_task_matrix(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, omitted_task_id=7)
    with pytest.raises(RuntimeError, match="exactly 8 task packages"):
        extraction_module.read_complete_bundle(bundle)
```

Also test outer checksum mismatch, nested checksum mismatch, duplicate manifest entry, uncovered regular file, symlink, and wrong workflow identity.

- [ ] **Step 3: Verify RED**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k bundle
```

- [ ] **Step 4: Implement safe-member and checksum rules**

```python
def validate_safe_archive_members(members):
    for member in members:
        path = PurePosixPath(member.name)
        if (
            not member.name
            or "\\" in member.name
            or path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise RuntimeError(f"unsafe archive member: {member.name!r}")
```

Manifest lines must match exactly `<64 lowercase hexadecimal characters><two spaces><member path>`. Reject malformed lines, duplicate names, missing members, manifest self-inclusion, and any regular file not covered by the manifest.

- [ ] **Step 5: Implement immutable evidence records**

```python
@dataclass(frozen=True)
class Provenance:
    diagnostic_array_job_id: str
    reducer_job_id: str
    formal_training_job_id: str
    source_commit_sha: str
    evidence_bundle_sha256: str
    diagnostic_schema_version: str
    reconciliation_contract_version: int


@dataclass(frozen=True)
class TaskPackage:
    task_id: int
    scale: str
    algorithm: str
    package_name: str
    package_sha256: str
    archive_bytes: bytes


@dataclass(frozen=True)
class CompleteBundleEvidence:
    provenance: Provenance
    task_packages: tuple[TaskPackage, ...]
```

Read task identity only from `summaries/task_inventory.csv`, whose required columns are:

```text
task_id,scale,algorithm,package_name,package_sha256,status,checkpoint_groups,episode_count
```

For each inventory row:

- require `status=ok`;
- require `checkpoint_groups=5`;
- require `episode_count=150` in production;
- locate outer member `task_packages/<package_name>`;
- calculate SHA-256 over the nested archive bytes and require equality with `package_sha256`;
- require task IDs 0–7 exactly once;
- require all 4 scales × 2 algorithms exactly once.

Validate outer workflow JSON against outer metadata:

```python
{
    "status": "ok",
    "array_job_id": diagnostic_array_job_id,
    "task_package_count": 8,
    "checkpoint_group_count": 40,
    "episode_count": 1200,
    "formal_job_id": formal_training_job_id,
    "schema_version": "3",
    "reconciliation_contract_version": 2,
}
```

Normalise schema version to string before comparison; do not normalise any ID.

- [ ] **Step 6: Verify GREEN and commit**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k bundle
git add analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Validate EV diagnostic evidence archives"
```

---

### Task 3: Extract four semantic datasets and provenance

**Files:**
- Modify: `analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- CLI: `extract_diagnostic_datasets.py --bundle <archive> --output-dir <directory>`.
- Outputs: four CSVs under `datasets/` and `provenance.json`.

- [ ] **Step 1: Write failing extraction tests**

```python
def test_extraction_publishes_semantic_outputs(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)
    output_dir = tmp_path / "ev_charging_infrastructure_control_analysis"
    extraction_module.extract_diagnostic_datasets(
        bundle, output_dir, expected_episodes_per_seed=1
    )
    assert sorted(path.name for path in (output_dir / "datasets").iterdir()) == [
        "charger_metrics.csv",
        "episode_metrics.csv",
        "seed_metrics.csv",
        "transformer_metrics.csv",
    ]
    assert (output_dir / "provenance.json").is_file()


def test_extraction_rejects_duplicate_episode_key(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, duplicate_episode_key=True)
    with pytest.raises(RuntimeError, match="duplicate episode key"):
        extraction_module.extract_diagnostic_datasets(
            bundle, tmp_path / "analysis-output", expected_episodes_per_seed=1
        )
```

Also test missing required column, non-finite approved value, wrong schema, mismatched matrix job ID, mismatched episode-seed set between algorithms, and pre-existing output directory.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k extraction
```

- [ ] **Step 3: Define exact dataset contracts**

`episode_metrics.csv`:

- key: `scale,algorithm,training_seed,episode_index,episode_seed`;
- production rows: 1,200;
- include semantic identity plus all 68 schema-v3 episode diagnostic columns except raw path fields `config`, `checkpoint_prefix`, and `run_name`;
- retain `matrix_job_id` as a provenance column inside rows, not in filenames.

`seed_metrics.csv`:

- key: `scale,algorithm,training_seed`;
- production rows: 40;
- include all 53 seed-summary columns;
- require `n_eval_episodes=30`.

`transformer_metrics.csv`:

- key: `scale,algorithm,training_seed,episode_index,episode_seed,transformer_id`;
- production rows: 34,500;
- include all 44 transformer columns.

`charger_metrics.csv`:

- key: `scale,algorithm,training_seed,episode_index,episode_seed,transformer_id,charger_id`;
- production rows: 487,500;
- include all 38 charger columns.

All datasets require:

- algorithms exactly `actiongnn` and `hierarchical`;
- training seeds exactly 0–4;
- exact scale set;
- schema version `3` on every row;
- exact episode indices 0–29 per scale/algorithm/seed in production;
- matching episode-seed sets between algorithms by scale and training seed;
- no duplicate composite keys;
- approved metric values finite;
- empty unavailable non-approved fields remain empty, never converted to zero.

- [ ] **Step 4: Stream nested CSVs without extracting to disk**

Use `tarfile.open(fileobj=io.BytesIO(task.archive_bytes), mode="r:gz")`, validate its manifest first, then read members using `csv.DictReader(io.TextIOWrapper(...))`. Write validated rows directly to temporary CSV files and retain only counters and key sets in memory.

- [ ] **Step 5: Publish atomically**

- fail when `--output-dir` already exists;
- create sibling `.ev_charging_infrastructure_control_analysis.tmp.<pid>`;
- write all files there;
- re-open outputs and verify headers and exact counts;
- write sorted, newline-terminated `provenance.json`;
- rename temporary directory with `os.replace`;
- delete only the temporary directory on failure.

`provenance.json` must contain:

```json
{
  "diagnostic_array_job_id": "...",
  "reducer_job_id": "...",
  "formal_training_job_id": "...",
  "source_commit_sha": "...",
  "evidence_bundle_sha256": "...",
  "diagnostic_schema_version": "3",
  "reconciliation_contract_version": 2,
  "task_package_count": 8,
  "checkpoint_group_count": 40,
  "episode_count": 1200,
  "dataset_rows": {
    "episode_metrics": 1200,
    "seed_metrics": 40,
    "transformer_metrics": 34500,
    "charger_metrics": 487500
  }
}
```

Successful CLI markers:

```text
EV_CHARGING_INFRASTRUCTURE_DATASET_EXTRACTION_START
BUNDLE_VALIDATION=PASS
TASK_PACKAGE_COUNT=8
EPISODE_METRICS_ROWS=1200
SEED_METRICS_ROWS=40
TRANSFORMER_METRICS_ROWS=34500
CHARGER_METRICS_ROWS=487500
OUTPUT_PUBLICATION=PASS
EV_CHARGING_INFRASTRUCTURE_DATASET_EXTRACTION_COMPLETED
```

- [ ] **Step 6: Verify GREEN and commit**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k extraction
git add analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Extract validated EV infrastructure diagnostic datasets"
```

---

### Task 4: Aggregate to seed level and enforce exact pairing

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/compare_control_architectures.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces `SeedMetricObservation`, `build_seed_level_metrics`, and `validate_exact_algorithm_pairs`.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_episode_metric_is_averaged_within_seed():
    rows = [
        episode_row("25cp", "actiongnn", 0, 0, episode_reward=2.0),
        episode_row("25cp", "actiongnn", 0, 1, episode_reward=4.0),
    ]
    values = comparison_module.aggregate_episode_metric(rows, "episode_reward")
    assert values[("25cp", "actiongnn", 0)] == pytest.approx(3.0)


def test_transformer_overload_rules_are_fixed():
    rows = [
        transformer_row("25cp", "actiongnn", 0, 0, 0, 0.1, 2.0, 3.0),
        transformer_row("25cp", "actiongnn", 0, 0, 1, 0.3, 5.0, 7.0),
        transformer_row("25cp", "actiongnn", 0, 1, 0, 0.2, 11.0, 13.0),
        transformer_row("25cp", "actiongnn", 0, 1, 1, 0.4, 17.0, 19.0),
    ]
    result = comparison_module.aggregate_transformer_overload_metrics(rows)
    key = ("25cp", "actiongnn", 0)
    assert result[key]["mean_transformer_overload_frequency_fraction"] == pytest.approx(0.25)
    assert result[key]["mean_total_transformer_overload_magnitude"] == pytest.approx(17.5)
    assert result[key]["maximum_transformer_overload_magnitude"] == pytest.approx(19.0)
```

Add tests for unequal seed sets, missing algorithm cell, duplicate seed observation, and more or fewer than five seeds.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k 'aggregate or pairing'
```

- [ ] **Step 3: Implement records and data loading**

```python
@dataclass(frozen=True)
class SeedMetricObservation:
    scale: str
    algorithm: str
    training_seed: int
    metric_name: str
    value: float
```

Read only `provenance.json` and generated datasets. Never reopen the raw bundle.

- [ ] **Step 4: Implement aggregation rules**

Episode-source metrics: arithmetic mean over exact 30 episodes in one scale/algorithm/seed.

For every episode-source metric that has a `seed_summary_column`, compare the calculated mean with `seed_metrics.csv` using:

```python
math.isclose(calculated, reported, rel_tol=1e-12, abs_tol=1e-12)
```

Fail with scale, algorithm, seed, and metric on mismatch. Do not perform this cross-check for `episode_reward`, whose seed summary has no reward column.

Transformer-derived metrics:

```text
mean_transformer_overload_frequency_fraction:
  mean overload_frequency_fraction across transformers in each episode,
  then mean across the 30 episodes.

mean_total_transformer_overload_magnitude:
  sum overload_magnitude_sum across transformers in each episode,
  then mean across the 30 episodes.

maximum_transformer_overload_magnitude:
  maximum overload_magnitude_max across every transformer row in all 30 episodes.
```

The charger dataset is validated but does not generate additional approved seed metrics in this implementation.

- [ ] **Step 5: Enforce exact pairing**

For each scale and each approved metric, require five ActionGNN and five hierarchical observations with identical seed sets `{0,1,2,3,4}`. Pair by `(scale, training_seed)`, never by row order.

- [ ] **Step 6: Verify GREEN and commit**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k 'aggregate or pairing'
git add analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Aggregate EV control diagnostics by training seed"
```

---

### Task 5: Compute paired statistics and publish semantic results

**Files:**
- Modify: `analysis/ev_charging_infrastructure_control/compare_control_architectures.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- CLI: `compare_control_architectures.py --analysis-dir <directory>`.
- Outputs: three files under `results/`.

- [ ] **Step 1: Write fixed statistical oracle tests**

```python
def test_paired_statistics_fixed_oracle():
    result = comparison_module.paired_statistics(
        actiongnn_values=[0, 0, 0, 0, 0],
        hierarchical_values=[1, 2, 3, 4, 5],
    )
    assert result.mean_difference == pytest.approx(3.0)
    assert result.sample_standard_deviation == pytest.approx(1.5811388300841898)
    assert result.t_statistic == pytest.approx(4.242640687119285)
    assert result.paired_t_p_value == pytest.approx(0.0132355995636827)
    assert result.confidence_interval_95_low == pytest.approx(1.036756838522439)
    assert result.confidence_interval_95_high == pytest.approx(4.963243161477561)
    assert result.cohens_dz == pytest.approx(1.8973665961010275)


def test_wilcoxon_and_holm_fixed_oracles():
    statistic, p_value, method, nonzero_count = (
        comparison_module.wilcoxon_signed_rank([1, 2, 3, 4, 5])
    )
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(0.0625)
    assert method == "exact"
    assert nonzero_count == 5
    assert comparison_module.holm_adjust([0.01, 0.04, 0.03, 0.20]) == pytest.approx(
        [0.04, 0.09, 0.09, 0.20]
    )
```

Also test all-zero differences, tied Wilcoxon ranks, zero ActionGNN mean, zero difference variance, non-finite input, and order-preserving Holm output.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py -k 'paired_statistics or wilcoxon or holm'
```

- [ ] **Step 3: Adapt proven numerical functions**

Adapt, without importing the job-specific script:

```text
regularized_beta_continued_fraction
regularized_incomplete_beta
student_t_cdf
student_t_ppf
paired_t_test
average_ranks_for_absolute_values
wilcoxon_signed_rank
```

Keep existing numerical tolerances unless a fixed oracle demonstrates a defect. Do not add SciPy.

Add:

```python
@dataclass(frozen=True)
class PairedStatistics:
    n_pairs: int
    actiongnn_mean: float
    hierarchical_mean: float
    mean_difference: float
    relative_difference_percent_of_actiongnn_mean: float | None
    relative_difference_status: str
    sample_standard_deviation: float
    cohens_dz: float | None
    effect_size_status: str
    t_statistic: float
    paired_t_p_value: float
    confidence_interval_95_low: float
    confidence_interval_95_high: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    wilcoxon_method: str
    wilcoxon_nonzero_count: int
```

Definitions:

```python
differences = hierarchical - actiongnn
relative_percent = 100 * mean_difference / abs(actiongnn_mean)
cohens_dz = mean_difference / sample_std(differences)
```

If ActionGNN mean is zero, leave relative percentage empty and set `undefined_zero_actiongnn_mean`. If difference standard deviation is zero and all differences are zero, use `cohens_dz=0` and `all_differences_zero`. If variance is zero with non-zero mean, leave effect size empty and use `undefined_zero_variance_nonzero_mean`.

- [ ] **Step 4: Implement Holm correction**

Use monotonic step-down Holm adjustment. Apply it only to four primary paired-t p-values within each scale. Non-primary rows have empty adjusted p-value and `holm_reject_0_05=false`.

- [ ] **Step 5: Define exact outputs**

`results/paired_control_comparisons.csv`:

- 4 scales × 22 metrics = 88 rows;
- one unique `(scale, metric_name)` key;
- columns:

```text
scale,metric_name,tier,preferred_direction,aggregation_rule,
interpretation_warning,n_paired_seeds,actiongnn_seed_mean,
hierarchical_seed_mean,paired_difference_hierarchical_minus_actiongnn,
relative_difference_percent_of_actiongnn_mean,relative_difference_status,
paired_difference_sample_standard_deviation,cohens_dz,effect_size_status,
paired_t_statistic,paired_t_p_value_two_sided,paired_difference_95ci_low,
paired_difference_95ci_high,wilcoxon_statistic,wilcoxon_p_value_two_sided,
wilcoxon_method,wilcoxon_nonzero_count,holm_adjusted_p_value,
holm_reject_0_05,observed_favourable_direction
```

`observed_favourable_direction` is one of `hierarchical`, `actiongnn`, `no_difference`, or `context_dependent`.

`results/scale_level_summary.csv`:

- four rows;
- columns:

```text
scale,paired_seed_count,primary_metric_count,
primary_metrics_favouring_hierarchical,primary_metrics_favouring_actiongnn,
primary_metrics_no_difference,primary_raw_p_below_0_05_count,
primary_holm_p_below_0_05_count,mechanism_metric_count,
physical_safety_metric_count,service_guardrail_metric_count,
robustness_metric_count
```

`results/metric_interpretation.md` contains a deterministic table of metric, tier, direction, aggregation, warning, and observed direction by scale. It must not generate free-form causal claims.

- [ ] **Step 6: Publish results atomically**

Fail if `results/` exists. Write to `.results.tmp.<pid>`, validate 88 comparison rows and four summary rows, then rename to `results/`.

Successful markers:

```text
EV_CHARGING_INFRASTRUCTURE_CONTROL_COMPARISON_START
PROVENANCE_VALIDATION=PASS
SEED_LEVEL_AGGREGATION=PASS
PAIRING_VALIDATION=PASS
PAIRED_COMPARISON_ROWS=88
SCALE_SUMMARY_ROWS=4
RESULT_PUBLICATION=PASS
EV_CHARGING_INFRASTRUCTURE_CONTROL_COMPARISON_COMPLETED
```

- [ ] **Step 7: Verify GREEN and commit**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py
git add analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Compare EV control architectures with paired statistics"
```

---

### Task 6: Document and run authoritative local verification

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/README.md`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

- [ ] **Step 1: Add CLI and naming tests**

Run both CLIs against the synthetic bundle. Require zero exit, completion markers, exact semantic outputs, and no generated basename containing `stage_d`, `r5l`, a job ID, or a commit hash. Apply this rule only to generated paths, not provenance fields.

- [ ] **Step 2: Write README commands**

```bash
conda activate evgnn_core

python analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  --bundle "$HOME/Downloads/EVGNN_Formal_Evidence/full_infrastructure_diagnostic_eval30_complete_evidence_job58745233.tar.gz" \
  --output-dir "$HOME/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis"

python analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  --analysis-dir "$HOME/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis"
```

Explain that the job-numbered input basename remains unchanged only because it is the audited immutable source. Generated names are semantic.

Document row counts, inference unit, Holm scope, interpretation boundaries, provenance policy, and rerun behaviour.

- [ ] **Step 3: Run focused and full verification**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py
pytest -q
python -m py_compile \
  analysis/ev_charging_infrastructure_control/metric_definitions.py \
  analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git diff --check
```

- [ ] **Step 4: Verify source checksum locally**

```bash
cd "$HOME/Downloads/EVGNN_Formal_Evidence"
shasum -a 256 -c \
  full_infrastructure_diagnostic_eval30_complete_evidence_job58745233.tar.gz.sha256
```

Expected: `OK`.

- [ ] **Step 5: Run authoritative extraction and comparison**

Run the README commands. Require:

```text
EPISODE_METRICS_ROWS=1200
SEED_METRICS_ROWS=40
TRANSFORMER_METRICS_ROWS=34500
CHARGER_METRICS_ROWS=487500
PAIRED_COMPARISON_ROWS=88
SCALE_SUMMARY_ROWS=4
```

Expected external output files:

```text
datasets/charger_metrics.csv
datasets/episode_metrics.csv
datasets/seed_metrics.csv
datasets/transformer_metrics.csv
provenance.json
results/metric_interpretation.md
results/paired_control_comparisons.csv
results/scale_level_summary.csv
```

Recheck the source checksum after analysis; it must remain unchanged.

- [ ] **Step 6: Validate results without changing the registry**

Require 22 metrics per scale, five pairs per comparison, Holm values on primary rows only, no duplicate `(scale, metric)` keys, no undefined numeric fields without explicit status, and no generated output committed to Git. Do not change metric selection or aggregation after viewing results.

- [ ] **Step 7: Commit documentation**

```bash
git add analysis/ev_charging_infrastructure_control/README.md \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Document EV infrastructure control analysis workflow"
```

- [ ] **Step 8: Record final branch evidence**

```bash
git status --short
git log --oneline --decorate -10
git diff --stat cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad...HEAD
git diff --check cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad...HEAD
```

Required state:

```text
WORKTREE_CLEAN=YES
M3_ACTION_PERFORMED=NO
TRAINING_PERFORMED=NO
EVALUATION_PERFORMED=NO
RAW_BUNDLE_MODIFIED=NO
GENERATED_OUTPUTS_COMMITTED=NO
```

Do not merge or open a pull request until independent review finds no important issues.
