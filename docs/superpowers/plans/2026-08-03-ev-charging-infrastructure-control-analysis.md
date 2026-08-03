# EV Charging Infrastructure Control Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, reproducible local analysis module that extracts validated EV charging infrastructure diagnostics from the authoritative evidence bundle and compares ActionGNN with the hierarchical controller using training-seed-level paired statistics.

**Architecture:** The extraction command reads the immutable outer archive and its eight nested task packages directly, validates identity, checksums, schema, row counts, and composite keys, then atomically writes four semantic CSV datasets plus provenance. The comparison command reads only those datasets, aggregates episode and infrastructure observations to one value per scale, algorithm, and training seed, performs exact five-seed pairing, and writes deterministic statistical tables and an interpretation registry.

**Tech Stack:** Python 3.11, Python standard library, NumPy, pytest. Do not add pandas, SciPy, statsmodels, or a new dependency file. Reuse the repository's already-tested standard-library-plus-NumPy statistical algorithms from `scripts/aggregate_controlled_multiscale_eval30.py` by adapting only the required functions into the new comparison module and testing them against fixed oracle values.

## Global Constraints

- Work only on branch `analysis/ev-charging-infrastructure-control`, based on merged main commit `cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad` plus the approved design commit.
- Do not modify actors, critics, replay, reward, simulator, evaluator, diagnostics producer, Stage D validator, M3 scripts, checkpoints, or the authoritative evidence bundle.
- Do not submit Slurm jobs, run M3 commands, retrain, re-evaluate, or rerun the reducer.
- The raw evidence bundle is read-only and remains outside the repository.
- Human-facing filenames and directories must not contain `stage_d`, `R5L`, job IDs, reducer IDs, commit hashes, or checksums.
- Job IDs, source commit SHA, and archive checksum are provenance values inside `provenance.json` only.
- The training seed is the only inferential unit. Episodes, transformers, and chargers are not independent samples for p-values.
- Approved metrics are fixed before results are inspected. Do not auto-discover or add favourable metrics.
- Tier 1 has four primary metrics. Holm correction applies only across those four metrics within each scale.
- Tier 2, Tier 3, Tier 4, and Gini results are exploratory or guardrail evidence.
- Generated datasets and results go to an explicit external output directory and are not committed.
- Use semantic variable names. Do not introduce names such as `df`, `df2`, `result1`, `stage_d_data`, or hash-derived identifiers.
- Keep the implementation to four analysis files and one focused test file. Do not add frameworks, class hierarchies, plugins, workflow wrappers, or large binary fixtures.

---

## File Map

**Create:**

- `analysis/ev_charging_infrastructure_control/README.md` — scientific purpose, commands, outputs, metric tiers, inference unit, limitations, provenance policy.
- `analysis/ev_charging_infrastructure_control/metric_definitions.py` — immutable metric registry and small validation helpers only.
- `analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py` — read-only archive validation and deterministic dataset extraction.
- `analysis/ev_charging_infrastructure_control/compare_control_architectures.py` — seed-level aggregation, strict pairing, paired statistics, Holm correction, and result writing.
- `tests/analysis/test_ev_charging_infrastructure_control.py` — synthetic archive fixture, extraction gates, aggregation, pairing, statistical oracle tests, CLI tests, and naming checks.

**Do not modify unless a real test proves it necessary:**

- `.gitignore`
- dependency files
- existing aggregation scripts
- existing Stage D tests or validators

---

### Task 1: Add the semantic metric registry

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/metric_definitions.py`
- Create: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces: `MetricDefinition`, `METRIC_DEFINITIONS`, `metric_definition(name: str) -> MetricDefinition`, `metric_names_for_tier(tier: str) -> tuple[str, ...]`, `required_source_columns(level: str) -> tuple[str, ...]`.
- Consumes: no earlier task.

- [ ] **Step 1: Write the failing registry tests**

Add imports using `importlib.util.spec_from_file_location` or the repository's established direct-file test loading pattern so no package initialiser is required.

```python
EXPECTED_PRIMARY_METRICS = {
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
}

EXPECTED_MECHANISM_METRICS = {
    "global_action_fraction_at_max_active",
    "global_action_nonzero_fraction_active",
    "transformer_action_fraction_at_max_active_macro_mean",
    "charger_action_fraction_at_max_active_macro_mean",
    "transformer_positive_charge_action_hhi_mean",
    "charger_positive_charge_action_hhi_mean",
    "transformer_allocation_zero_pressure_step_fraction",
    "charger_allocation_zero_pressure_step_fraction",
}

EXPECTED_PHYSICAL_SAFETY_METRICS = {
    "total_transformer_overload",
    "mean_transformer_overload_frequency_fraction",
    "mean_total_transformer_overload_magnitude",
    "maximum_transformer_overload_magnitude",
}

EXPECTED_SERVICE_GUARDRAILS = {
    "total_ev_served",
    "total_energy_charged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
}

EXPECTED_ROBUSTNESS_METRICS = {
    "transformer_positive_charge_action_gini_mean",
    "charger_positive_charge_action_gini_mean",
}


def test_metric_registry_contains_only_approved_metrics():
    definitions = metric_definitions_module.METRIC_DEFINITIONS
    observed = {definition.name for definition in definitions}
    expected = (
        EXPECTED_PRIMARY_METRICS
        | EXPECTED_MECHANISM_METRICS
        | EXPECTED_PHYSICAL_SAFETY_METRICS
        | EXPECTED_SERVICE_GUARDRAILS
        | EXPECTED_ROBUSTNESS_METRICS
    )
    assert observed == expected
    assert len(observed) == len(definitions)


def test_metric_tiers_and_directions_are_explicit():
    reward = metric_definitions_module.metric_definition("episode_reward")
    assert reward.tier == "primary"
    assert reward.preferred_direction == "higher"
    saturation = metric_definitions_module.metric_definition(
        "global_action_fraction_at_max_active"
    )
    assert saturation.tier == "mechanism"
    assert saturation.preferred_direction == "context_dependent"
    assert "not automatically" in saturation.interpretation_warning.lower()
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py \
  -k 'metric_registry or metric_tiers'
```

Expected: collection or import failure because `metric_definitions.py` does not exist.

- [ ] **Step 3: Implement the minimal immutable registry**

Use this exact public shape:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    source_level: str
    source_column: str
    tier: str
    preferred_direction: str
    aggregation_rule: str
    interpretation_warning: str


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "episode_reward",
        "episode",
        "episode_reward",
        "primary",
        "higher",
        "mean_across_30_episodes_within_training_seed",
        "Interpret with the other PST metrics; reward alone does not identify mechanism.",
    ),
    MetricDefinition(
        "tracking_error",
        "episode",
        "tracking_error",
        "primary",
        "lower",
        "mean_across_30_episodes_within_training_seed",
        "Lower is better for Power Setpoint Tracking.",
    ),
    MetricDefinition(
        "energy_tracking_error",
        "episode",
        "energy_tracking_error",
        "primary",
        "lower",
        "mean_across_30_episodes_within_training_seed",
        "Lower is better for cumulative energy tracking.",
    ),
    MetricDefinition(
        "power_tracker_violation",
        "episode",
        "power_tracker_violation",
        "primary",
        "lower",
        "mean_across_30_episodes_within_training_seed",
        "Lower is better; report with tracking error and service outcomes.",
    ),
    MetricDefinition(
        "global_action_fraction_at_max_active",
        "episode",
        "global_action_fraction_at_max_active",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Lower saturation is not automatically better control.",
    ),
    MetricDefinition(
        "global_action_nonzero_fraction_active",
        "episode",
        "global_action_nonzero_fraction_active",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "A lower value can indicate conservative charging rather than coordination.",
    ),
    MetricDefinition(
        "transformer_action_fraction_at_max_active_macro_mean",
        "episode",
        "transformer_action_fraction_at_max_active_macro_mean",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Macro saturation must be interpreted with overload and service.",
    ),
    MetricDefinition(
        "charger_action_fraction_at_max_active_macro_mean",
        "episode",
        "charger_action_fraction_at_max_active_macro_mean",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Macro saturation must be interpreted with energy and service.",
    ),
    MetricDefinition(
        "transformer_positive_charge_action_hhi_mean",
        "episode",
        "transformer_positive_charge_action_hhi_mean",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Lower concentration is more even, not automatically more physically correct.",
    ),
    MetricDefinition(
        "charger_positive_charge_action_hhi_mean",
        "episode",
        "charger_positive_charge_action_hhi_mean",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Lower concentration is more even, not automatically more physically correct.",
    ),
    MetricDefinition(
        "transformer_allocation_zero_pressure_step_fraction",
        "episode",
        "transformer_allocation_zero_pressure_step_fraction",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Exploratory diagnostic; do not assign a favourable direction without context.",
    ),
    MetricDefinition(
        "charger_allocation_zero_pressure_step_fraction",
        "episode",
        "charger_allocation_zero_pressure_step_fraction",
        "mechanism",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Exploratory diagnostic; do not assign a favourable direction without context.",
    ),
    MetricDefinition(
        "total_transformer_overload",
        "episode",
        "total_transformer_overload",
        "physical_safety",
        "lower",
        "mean_across_30_episodes_within_training_seed",
        "Aggregate simulator overload metric; compare with transformer-row diagnostics.",
    ),
    MetricDefinition(
        "mean_transformer_overload_frequency_fraction",
        "transformer",
        "overload_frequency_fraction",
        "physical_safety",
        "lower",
        "mean_across_transformers_within_episode_then_mean_across_episodes",
        "Lower indicates fewer overloaded transformer steps on average.",
    ),
    MetricDefinition(
        "mean_total_transformer_overload_magnitude",
        "transformer",
        "overload_magnitude_sum",
        "physical_safety",
        "lower",
        "sum_across_transformers_within_episode_then_mean_across_episodes",
        "Lower indicates less cumulative transformer overload magnitude per episode.",
    ),
    MetricDefinition(
        "maximum_transformer_overload_magnitude",
        "transformer",
        "overload_magnitude_max",
        "physical_safety",
        "lower",
        "maximum_across_transformers_and_episodes_within_training_seed",
        "Worst-case guardrail; it is not averaged across episodes.",
    ),
    MetricDefinition(
        "total_ev_served",
        "episode",
        "total_ev_served",
        "service_guardrail",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Lower service may explain reduced saturation.",
    ),
    MetricDefinition(
        "total_energy_charged",
        "episode",
        "total_energy_charged",
        "service_guardrail",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Lower energy may be efficient or may indicate under-service; use tracking outcomes.",
    ),
    MetricDefinition(
        "average_user_satisfaction",
        "episode",
        "average_user_satisfaction",
        "service_guardrail",
        "higher",
        "mean_across_30_episodes_within_training_seed",
        "Interpret with energy delivery and served-EV count.",
    ),
    MetricDefinition(
        "energy_user_satisfaction",
        "episode",
        "energy_user_satisfaction",
        "service_guardrail",
        "higher",
        "mean_across_30_episodes_within_training_seed",
        "Interpret as an energy-delivery service guardrail.",
    ),
    MetricDefinition(
        "transformer_positive_charge_action_gini_mean",
        "episode",
        "transformer_positive_charge_action_gini_mean",
        "robustness",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Robustness check for HHI; do not present as a separate primary discovery.",
    ),
    MetricDefinition(
        "charger_positive_charge_action_gini_mean",
        "episode",
        "charger_positive_charge_action_gini_mean",
        "robustness",
        "context_dependent",
        "mean_across_30_episodes_within_training_seed",
        "Robustness check for HHI; do not present as a separate primary discovery.",
    ),
)
```

Implement lookup helpers with duplicate-name validation at import time. `required_source_columns("episode")` must return identity columns plus only the approved episode source columns. Transformer and charger dataset schemas remain broader than the metric registry because the extracted datasets are audit-ready, but statistical selection remains registry-controlled.

- [ ] **Step 4: Run the registry tests and verify GREEN**

Run the same focused command. Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  analysis/ev_charging_infrastructure_control/metric_definitions.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Add EV infrastructure analysis metric registry"
```

---

### Task 2: Validate and read the immutable evidence archives

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces: `validate_safe_archive_members`, `read_checksum_manifest`, `validate_manifest_coverage`, `read_complete_bundle(bundle_path: Path) -> CompleteBundleEvidence`, `iter_nested_csv_rows(...)`.
- Consumes: `required_source_columns` and metric registry from Task 1.

- [ ] **Step 1: Add a small synthetic complete-bundle fixture**

In the single test file, create a helper that writes one outer `.tar.gz` containing eight nested `.tar.gz` task packages. Each nested package contains five seed directories and these exact members:

```text
seed0/diagnostics/episode_diagnostics.csv
seed0/diagnostics/seed_summary_diagnostics.csv
seed0/diagnostics/transformer_diagnostics.csv
seed0/diagnostics/charger_diagnostics.csv
...
seed4/diagnostics/...
checksums/package_file_checksums.sha256
```

The outer archive must contain:

```text
task_packages/<semantic-task-package-name>.tar.gz   # eight files
summaries/task_inventory.csv
runtime_metadata/diagnostic_array_job_id.txt
runtime_metadata/reducer_job_id.txt
runtime_metadata/formal_job_id.txt
runtime_metadata/source_commit_sha.txt
validation/complete_workflow_validation.json
runtime_metadata/complete_file_checksums.sha256
```

Keep each synthetic CSV to one episode per seed so tests remain small. Parameterise expected episode count in internal helpers instead of weakening the production CLI's exact 30-episode gate.

Use explicit semantic fixture values:

```python
SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
SEEDS = tuple(range(5))
TASKS = tuple(
    (scale, algorithm)
    for scale in SCALES
    for algorithm in ALGORITHMS
)
```

- [ ] **Step 2: Add failing archive-safety and checksum tests**

```python
def test_read_complete_bundle_rejects_unsafe_member(tmp_path):
    bundle = build_synthetic_complete_bundle(
        tmp_path,
        extra_outer_member=("../escape.txt", b"unsafe"),
    )
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extraction_module.read_complete_bundle(bundle)


def test_read_complete_bundle_rejects_checksum_mismatch(tmp_path):
    bundle = build_synthetic_complete_bundle(
        tmp_path,
        corrupt_outer_member="validation/complete_workflow_validation.json",
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        extraction_module.read_complete_bundle(bundle)


def test_read_complete_bundle_requires_exact_eight_task_packages(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, omitted_task_id=7)
    with pytest.raises(RuntimeError, match="exactly 8 task packages"):
        extraction_module.read_complete_bundle(bundle)
```

- [ ] **Step 3: Run the new tests and verify RED**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py \
  -k 'complete_bundle or unsafe_member or checksum'
```

Expected: import or missing-function failures.

- [ ] **Step 4: Implement safe archive and manifest validation**

Implement these exact rules:

```python
def validate_safe_archive_members(members: Sequence[tarfile.TarInfo]) -> None:
    for member in members:
        path = PurePosixPath(member.name)
        unsafe = (
            not member.name
            or "\\" in member.name
            or path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
        )
        if unsafe:
            raise RuntimeError(f"unsafe archive member: {member.name!r}")
```

`read_checksum_manifest` must require lines shaped exactly as `<64 lowercase hex><two spaces><member name>`. Reject duplicates, missing members, manifest self-inclusion, and uncovered regular files. Validate every outer and nested member checksum before parsing CSV content.

Create immutable evidence records:

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
    archive_name: str
    archive_bytes: bytes


@dataclass(frozen=True)
class CompleteBundleEvidence:
    provenance: Provenance
    task_packages: tuple[TaskPackage, ...]
```

Read task identities from `summaries/task_inventory.csv`; do not parse scale or algorithm from hash/job-numbered archive filenames. Require exact keys for all 4 scales × 2 algorithms and require task IDs 0–7 exactly once.

Require outer workflow metadata:

```python
{
    "status": "ok",
    "array_job_id": provenance.diagnostic_array_job_id,
    "task_package_count": 8,
    "checkpoint_group_count": 40,
    "episode_count": 1200,
    "formal_job_id": provenance.formal_training_job_id,
    "schema_version": 3,
    "reconciliation_contract_version": 2,
}
```

Accept schema version `3` as either JSON number or string only after normalising to string. All other identities must match exactly.

- [ ] **Step 5: Run the archive tests and verify GREEN**

Run the focused archive tests. Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Validate EV diagnostic evidence archives"
```

---

### Task 3: Extract four deterministic datasets and provenance atomically

**Files:**
- Modify: `analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces CLI: `python analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py --bundle <path> --output-dir <path>`.
- Produces files: `datasets/episode_metrics.csv`, `datasets/seed_metrics.csv`, `datasets/transformer_metrics.csv`, `datasets/charger_metrics.csv`, `provenance.json`.
- Consumes: archive records from Task 2 and metric definitions from Task 1.

- [ ] **Step 1: Add failing extraction and key-validation tests**

```python
def test_extract_writes_semantic_datasets_and_provenance(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)
    output_dir = tmp_path / "ev_charging_infrastructure_control_analysis"
    extraction_module.extract_diagnostic_datasets(
        bundle_path=bundle,
        output_dir=output_dir,
        expected_episodes_per_seed=1,
    )
    assert sorted(path.name for path in (output_dir / "datasets").iterdir()) == [
        "charger_metrics.csv",
        "episode_metrics.csv",
        "seed_metrics.csv",
        "transformer_metrics.csv",
    ]
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert provenance["diagnostic_array_job_id"] == "synthetic-array"
    assert provenance["formal_training_job_id"] == "synthetic-formal"
    assert not any(
        token in path.name
        for path in output_dir.rglob("*")
        for token in ("synthetic-array", "deadbeef", "stage_d", "r5l")
    )


def test_extract_rejects_duplicate_episode_key(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, duplicate_episode_key=True)
    with pytest.raises(RuntimeError, match="duplicate episode key"):
        extraction_module.extract_diagnostic_datasets(
            bundle,
            tmp_path / "analysis-output",
            expected_episodes_per_seed=1,
        )
```

Also test missing required columns, non-finite approved values, wrong schema version, unequal seed cells, and pre-existing output directory.

- [ ] **Step 2: Run extraction tests and verify RED**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py \
  -k 'extract_ or duplicate_episode or required_column'
```

Expected: missing extraction entry point.

- [ ] **Step 3: Define exact output schemas**

Write only semantic identity plus required diagnostic columns. Keep the order deterministic.

`episode_metrics.csv` key:

```text
scale, algorithm, training_seed, episode_index, episode_seed
```

Required row count in production: 1,200.

`seed_metrics.csv` key:

```text
scale, algorithm, training_seed
```

Required row count: 40. It is the normalised copy of each nested `seed_summary_diagnostics.csv`, not a statistical result table.

`transformer_metrics.csv` key:

```text
scale, algorithm, training_seed, episode_index, episode_seed, transformer_id
```

Required production row count: 34,500.

Include all transformer diagnostic columns from schema v3 so later mechanism and service analyses do not require re-reading the raw bundle.

`charger_metrics.csv` key:

```text
scale, algorithm, training_seed, episode_index, episode_seed, transformer_id, charger_id
```

Required production row count: 487,500.

Include all charger diagnostic columns from schema v3.

For all datasets:

- normalise `scale`, `algorithm`, and integer identifiers;
- require `algorithm in {actiongnn, hierarchical}`;
- require `training_seed in {0,1,2,3,4}`;
- require exact 30 episode indices `0..29` per scale/algorithm/training seed in production;
- require episode-seed sets to match between algorithms for each scale/training seed;
- require `diagnostic_schema_version == "3"` for every row;
- reject duplicate keys;
- reject missing or non-finite values for approved metrics;
- preserve documented unavailable non-approved fields as empty strings rather than converting them to zero.

- [ ] **Step 4: Implement streaming nested CSV extraction**

Do not extract archives to disk. For each nested task package:

```python
with tarfile.open(fileobj=io.BytesIO(task_package.archive_bytes), mode="r:gz") as archive:
    validate_safe_archive_members(archive.getmembers())
    validate_manifest_coverage(archive, "checksums/package_file_checksums.sha256")
    for training_seed in range(5):
        read_csv_member(
            archive,
            f"seed{training_seed}/diagnostics/episode_diagnostics.csv",
        )
```

Use `csv.DictReader(io.TextIOWrapper(member_file, encoding="utf-8", newline=""))`. Write rows to temporary CSV files as they are validated; maintain only key sets and counters in memory.

- [ ] **Step 5: Implement atomic publication**

Rules:

1. Resolve bundle and output paths.
2. Fail if output directory already exists.
3. Create a sibling temporary directory named with process ID only, for example `.ev_charging_infrastructure_control_analysis.tmp.12345`.
4. Write all datasets and provenance to the temporary directory.
5. Re-open generated CSVs and validate exact row counts and headers.
6. Rename the temporary directory to the requested output directory with `os.replace`.
7. On any exception, delete only the temporary directory and leave the raw bundle untouched.

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

Sort JSON keys and end the file with one newline.

- [ ] **Step 6: Add CLI and deterministic completion markers**

CLI arguments:

```text
--bundle      required existing .tar.gz
--output-dir  required path that must not exist
```

Production invocation fixes `expected_episodes_per_seed=30`; the smaller value exists only as an internal test parameter and is not exposed by CLI.

Successful stdout:

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

- [ ] **Step 7: Run extraction tests and verify GREEN**

Run all extraction-focused tests and then the complete focused test file.

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Extract validated EV infrastructure diagnostic datasets"
```

---

### Task 4: Aggregate to the training-seed level and enforce exact pairing

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/compare_control_architectures.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces: `load_analysis_datasets`, `build_seed_level_metrics`, `validate_exact_algorithm_pairs`, `SeedMetricObservation`.
- Consumes: four CSV datasets and `provenance.json` from Task 3; registry from Task 1.

- [ ] **Step 1: Write failing aggregation tests**

Use small in-memory CSV fixtures with two algorithms, five seeds, and two episodes per seed.

```python
def test_episode_metrics_are_averaged_within_training_seed():
    rows = [
        episode_row("25cp", "actiongnn", 0, 0, episode_reward=2.0),
        episode_row("25cp", "actiongnn", 0, 1, episode_reward=4.0),
    ]
    aggregated = comparison_module.aggregate_episode_metric(
        rows,
        source_column="episode_reward",
    )
    assert aggregated[("25cp", "actiongnn", 0)] == pytest.approx(3.0)


def test_transformer_overload_aggregations_follow_frozen_rules():
    rows = [
        transformer_row("25cp", "actiongnn", 0, 0, 0, 0.1, 2.0, 3.0),
        transformer_row("25cp", "actiongnn", 0, 0, 1, 0.3, 5.0, 7.0),
        transformer_row("25cp", "actiongnn", 0, 1, 0, 0.2, 11.0, 13.0),
        transformer_row("25cp", "actiongnn", 0, 1, 1, 0.4, 17.0, 19.0),
    ]
    metrics = comparison_module.aggregate_transformer_overload_metrics(rows)
    key = ("25cp", "actiongnn", 0)
    assert metrics[key]["mean_transformer_overload_frequency_fraction"] == pytest.approx(0.25)
    assert metrics[key]["mean_total_transformer_overload_magnitude"] == pytest.approx(17.5)
    assert metrics[key]["maximum_transformer_overload_magnitude"] == pytest.approx(19.0)


def test_pairing_requires_exact_five_seed_sets():
    observations = complete_seed_observations()
    observations.pop(("25cp", "hierarchical", 4, "episode_reward"))
    with pytest.raises(RuntimeError, match="unequal seed sets"):
        comparison_module.validate_exact_algorithm_pairs(observations)
```

- [ ] **Step 2: Run aggregation tests and verify RED**

Expected: module or function missing.

- [ ] **Step 3: Implement semantic records and CSV loading**

```python
@dataclass(frozen=True)
class SeedMetricObservation:
    scale: str
    algorithm: str
    training_seed: int
    metric_name: str
    value: float
```

Read CSVs with `csv.DictReader`; reject unexpected headers, duplicate keys, non-finite selected values, and provenance disagreement. `compare_control_architectures.py` must never open the raw evidence bundle.

- [ ] **Step 4: Implement frozen aggregation rules**

Episode-source metrics:

```python
seed_value = mean(
    metric value for the exact 30 episodes in one scale/algorithm/training seed
)
```

Cross-check every episode-source seed value against the corresponding column in `seed_metrics.csv` using:

```python
math.isclose(observed, reported, rel_tol=1e-12, abs_tol=1e-12)
```

Fail with metric, scale, algorithm, and seed in the error message if they differ.

Transformer-derived metrics:

```text
mean_transformer_overload_frequency_fraction:
  mean transformer overload_frequency_fraction within each episode,
  then mean the 30 episode values within the training seed.

mean_total_transformer_overload_magnitude:
  sum transformer overload_magnitude_sum within each episode,
  then mean the 30 episode totals within the training seed.

maximum_transformer_overload_magnitude:
  maximum overload_magnitude_max across all transformers and all 30 episodes
  within the training seed.
```

The charger dataset is validated and retained for later analyses but does not create additional approved seed metrics in this implementation.

- [ ] **Step 5: Enforce exact experiment cells and pairs**

Require exactly:

```python
SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
TRAINING_SEEDS = tuple(range(5))
```

For every approved metric and scale, require exactly five ActionGNN and five hierarchical observations with identical seed sets. Pair only by `(scale, training_seed)`. Never pair by row order.

- [ ] **Step 6: Run aggregation and pairing tests and verify GREEN**

Run the focused tests and the complete test file.

- [ ] **Step 7: Commit Task 4**

```bash
git add \
  analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Aggregate EV control diagnostics by training seed"
```

---

### Task 5: Add paired statistics, Holm correction, and semantic result outputs

**Files:**
- Modify: `analysis/ev_charging_infrastructure_control/compare_control_architectures.py`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces CLI: `python analysis/ev_charging_infrastructure_control/compare_control_architectures.py --analysis-dir <path>`.
- Produces: `results/paired_control_comparisons.csv`, `results/scale_level_summary.csv`, `results/metric_interpretation.md`.
- Consumes: exact seed-level observations from Task 4.

- [ ] **Step 1: Add fixed statistical oracle tests**

Use fixed difference values with independently known expectations. Reuse the same numerical conventions as `scripts/aggregate_controlled_multiscale_eval30.py`.

```python
def test_paired_t_and_confidence_interval_fixed_oracle():
    result = comparison_module.paired_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.mean_difference == pytest.approx(3.0)
    assert result.sample_standard_deviation == pytest.approx(1.5811388300841898)
    assert result.t_statistic == pytest.approx(4.242640687119285)
    assert result.paired_t_p_value == pytest.approx(0.0132355995636827)
    assert result.confidence_interval_95_low == pytest.approx(1.036756838522439)
    assert result.confidence_interval_95_high == pytest.approx(4.963243161477561)
    assert result.cohens_dz == pytest.approx(1.8973665961010275)


def test_exact_wilcoxon_fixed_oracle():
    statistic, p_value, method, nonzero_count = (
        comparison_module.wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])
    )
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(0.0625)
    assert method == "exact"
    assert nonzero_count == 5


def test_holm_adjustment_fixed_oracle():
    adjusted = comparison_module.holm_adjust([0.01, 0.04, 0.03, 0.20])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.20])
```

Add edge-case tests:

- all paired differences zero;
- zero ActionGNN mean for relative difference;
- tied absolute Wilcoxon ranks;
- one missing seed;
- NaN or infinity rejected;
- Holm output preserves original metric order.

- [ ] **Step 2: Run statistical tests and verify RED**

Expected: missing functions or record types.

- [ ] **Step 3: Adapt the repository's proven statistical functions**

Adapt these functions from `scripts/aggregate_controlled_multiscale_eval30.py` into the new comparison module without importing that job-specific script:

```text
regularized_beta_continued_fraction
regularized_incomplete_beta
student_t_cdf
student_t_ppf
paired_t_test
average_ranks_for_absolute_values
wilcoxon_signed_rank
```

Keep the algorithms and numerical tolerances unchanged unless a new fixed-oracle test demonstrates a defect. Do not add SciPy.

Add:

```python
@dataclass(frozen=True)
class PairedStatistics:
    n_pairs: int
    actiongnn_mean: float
    hierarchical_mean: float
    mean_difference: float
    sample_standard_deviation: float
    t_statistic: float
    paired_t_p_value: float
    confidence_interval_95_low: float
    confidence_interval_95_high: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    wilcoxon_method: str
    wilcoxon_nonzero_count: int
    cohens_dz: float | None
    effect_size_status: str
    relative_difference_percent_of_actiongnn_mean: float | None
    relative_difference_status: str
```

Definitions:

```python
paired_difference = hierarchical_value - actiongnn_value
relative_difference_percent = (
    100.0 * mean_difference / abs(actiongnn_mean)
)
cohens_dz = mean_difference / sample_standard_deviation
```

If `actiongnn_mean == 0`, write an empty CSV value and status `undefined_zero_actiongnn_mean`. If difference standard deviation is zero and all differences are zero, write `cohens_dz=0` and status `all_differences_zero`. If standard deviation is zero with a non-zero mean, write an empty effect size and status `undefined_zero_variance_nonzero_mean`; do not write infinity.

- [ ] **Step 4: Implement Holm correction only for Tier 1 within each scale**

```python
def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted_sorted = []
    running_max = 0.0
    count = len(indexed)
    for rank, (original_index, p_value) in enumerate(indexed):
        candidate = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted_sorted.append((original_index, running_max))
    adjusted = [0.0] * count
    for original_index, value in adjusted_sorted:
        adjusted[original_index] = value
    return adjusted
```

For non-primary rows, output empty `holm_adjusted_p_value` and `holm_reject_0_05=false`.

- [ ] **Step 5: Define exact result schemas**

`paired_control_comparisons.csv` has 4 scales × 22 approved metrics = 88 rows and these columns:

```text
scale
metric_name
tier
preferred_direction
aggregation_rule
interpretation_warning
n_paired_seeds
actiongnn_seed_mean
hierarchical_seed_mean
paired_difference_hierarchical_minus_actiongnn
relative_difference_percent_of_actiongnn_mean
relative_difference_status
paired_difference_sample_standard_deviation
cohens_dz
effect_size_status
paired_t_statistic
paired_t_p_value_two_sided
paired_difference_95ci_low
paired_difference_95ci_high
wilcoxon_statistic
wilcoxon_p_value_two_sided
wilcoxon_method
wilcoxon_nonzero_count
holm_adjusted_p_value
holm_reject_0_05
observed_favourable_direction
```

`observed_favourable_direction` values:

```text
hierarchical
actiongnn
no_difference
context_dependent
```

`scale_level_summary.csv` has four rows and these columns:

```text
scale
paired_seed_count
primary_metric_count
primary_metrics_favouring_hierarchical
primary_metrics_favouring_actiongnn
primary_metrics_no_difference
primary_raw_p_below_0_05_count
primary_holm_p_below_0_05_count
mechanism_metric_count
physical_safety_metric_count
service_guardrail_metric_count
robustness_metric_count
```

`metric_interpretation.md` is deterministic and contains one table row per approved metric with tier, preferred direction, aggregation rule, interpretation warning, and the observed direction for each scale. It must not generate free-form causal claims or declare significance.

- [ ] **Step 6: Implement atomic result publication**

Require an existing extraction output directory with valid `provenance.json` and `datasets/`. Fail if `results/` already exists. Write to `.results.tmp.<pid>`, validate 88 comparison rows and 4 scale rows, then atomically rename to `results/`.

Successful stdout:

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

- [ ] **Step 7: Run statistics and output tests and verify GREEN**

Run all comparison tests and the complete focused test file.

- [ ] **Step 8: Commit Task 5**

```bash
git add \
  analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git commit -m "Compare EV control architectures with paired statistics"
```

---

### Task 6: Document operation and verify the authoritative local bundle

**Files:**
- Create: `analysis/ev_charging_infrastructure_control/README.md`
- Modify only if required by an observed repository-pollution test: `.gitignore`
- Modify: `tests/analysis/test_ev_charging_infrastructure_control.py`

**Interfaces:**
- Produces complete operator documentation and final verification evidence.
- Consumes all previous tasks.

- [ ] **Step 1: Add CLI and naming tests**

Test both commands through `subprocess.run` against the synthetic bundle. Assert zero exit status, completion markers, exact semantic files, and no output basename containing:

```python
DISALLOWED_FILENAME_TOKENS = (
    "stage_d",
    "r5l",
    "job",
    "cbf4b5fe",
    "58745233",
    "58746039",
)
```

Do not ban the word `job` inside provenance JSON field names; apply the rule only to generated human-facing path basenames.

- [ ] **Step 2: Write the README**

Document these exact local commands:

```bash
conda activate evgnn_core

python analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  --bundle "$HOME/Downloads/EVGNN_Formal_Evidence/full_infrastructure_diagnostic_eval30_complete_evidence_job58745233.tar.gz" \
  --output-dir "$HOME/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis"

python analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  --analysis-dir "$HOME/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis"
```

Explain that the job-numbered input filename is preserved only because it is the immutable audited raw artefact; generated outputs use semantic names.

Document:

- four dataset row counts;
- 88 comparison rows and four scale rows;
- the training-seed inference unit;
- Holm correction scope;
- mechanism and guardrail interpretation limits;
- no M3, training, evaluation, or reducer action;
- raw bundle remains byte-identical;
- rerunning requires a new output directory or deliberate removal after archiving the prior output.

- [ ] **Step 3: Run focused verification**

```bash
pytest -q tests/analysis/test_ev_charging_infrastructure_control.py
```

Expected: all focused tests pass.

- [ ] **Step 4: Run full repository verification**

```bash
pytest -q
python -m py_compile \
  analysis/ev_charging_infrastructure_control/metric_definitions.py \
  analysis/ev_charging_infrastructure_control/extract_diagnostic_datasets.py \
  analysis/ev_charging_infrastructure_control/compare_control_architectures.py \
  tests/analysis/test_ev_charging_infrastructure_control.py
git diff --check
```

Expected: full repository tests pass with only the previously known warning, compilation passes, and diff check is clean.

- [ ] **Step 5: Run the authoritative local extraction**

Before execution, verify the raw bundle remains authoritative:

```bash
cd "$HOME/Downloads/EVGNN_Formal_Evidence"
shasum -a 256 -c \
  full_infrastructure_diagnostic_eval30_complete_evidence_job58745233.tar.gz.sha256
```

Expected:

```text
full_infrastructure_diagnostic_eval30_complete_evidence_job58745233.tar.gz: OK
```

Then run the README extraction command. Verify:

```text
EPISODE_METRICS_ROWS=1200
SEED_METRICS_ROWS=40
TRANSFORMER_METRICS_ROWS=34500
CHARGER_METRICS_ROWS=487500
```

- [ ] **Step 6: Run the authoritative local comparison**

Run the README comparison command. Verify:

```text
PAIRED_COMPARISON_ROWS=88
SCALE_SUMMARY_ROWS=4
RESULT_PUBLICATION=PASS
```

- [ ] **Step 7: Verify output provenance and semantic naming**

```bash
find "$HOME/Downloads/EVGNN_Formal_Evidence/ev_charging_infrastructure_control_analysis" \
  -type f -maxdepth 3 -print | sort
```

Expected files only:

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

Verify the original bundle checksum again after analysis. It must remain unchanged.

- [ ] **Step 8: Review generated results without changing metric selection**

Inspect only for validation and obvious calculation defects:

- exactly 88 rows;
- 22 metrics per scale;
- exactly five paired seeds per row;
- Holm values only on four primary metrics per scale;
- no NaN or infinity in defined statistics;
- explicit status for undefined relative difference or effect size;
- no duplicate scale/metric rows.

Do not change approved metrics, aggregation rules, or test direction after seeing the results. Any scientific interpretation occurs in a separate reviewed phase.

- [ ] **Step 9: Commit documentation and final focused tests**

```bash
git add \
  analysis/ev_charging_infrastructure_control/README.md \
  tests/analysis/test_ev_charging_infrastructure_control.py

git commit -m "Document EV infrastructure control analysis workflow"
```

- [ ] **Step 10: Final branch evidence**

Record:

```bash
git status --short
git log --oneline --decorate -8
git diff --stat cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad...HEAD
git diff --check cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad...HEAD
```

Required final state:

```text
WORKTREE_CLEAN=YES
M3_ACTION_PERFORMED=NO
TRAINING_PERFORMED=NO
EVALUATION_PERFORMED=NO
RAW_BUNDLE_MODIFIED=NO
GENERATED_OUTPUTS_COMMITTED=NO
```

Do not merge or open a pull request until an independent code review confirms no important findings.
