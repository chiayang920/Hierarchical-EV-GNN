# Stage D Reconciliation Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the Stage D reconciliation contract so fresh schema-v3 diagnostics are the authoritative Stage D metrics, same-pass canonical rows are generated from the same policy execution, historical formal eval30 remains immutable provenance and audit input, and no failed job `58656380` output is reused as scientific evidence.

**Architecture:** The diagnostic evaluator will retain the mapped-action stream from each episode and write a canonical-compatible same-pass CSV through the existing canonical helpers in `evaluate_td3_gnn.py`. The validator will split reconciliation into exact same-pass hard gates, exact historical identity hard gates, and historical floating/saturation drift audit rows, then atomically write all evidence before raising any post-read hard-gate failure. Package, reducer, Slurm, source-bundle, and submit-helper contracts will be migrated to require reconciliation contract version `2` while preserving diagnostic schema version `3` and the exact `8/40/1200` workload.

**Tech Stack:** Python 3.11, pytest, NumPy, PyYAML, CSV/JSON standard libraries, tarfile, hashlib, Bash 3.2-compatible shell, Slurm dry-run scripts, existing TD3 EV-GNN evaluator helpers.

## Global Constraints

- Base design commit: `a75b758d684b884a85dc6eeb040aabad7b4e8982`.
- Base parent and `origin/main`: `12ae8768e72bd9fd5f94980a4d0538e06835f861`.
- Formal array job: `58513929`.
- Formal reducer job: `58513930`.
- Failed Stage D array job `58656380` and reducer `58656381` are engineering calibration/failure evidence only.
- Never use job `58656380` as scientific Stage D evidence, never use its task-5 package in a complete bundle, never resume its reducer, and never run failed-task-only scientific recovery.
- Recovery after implementation requires one clean full 8-task rerun under a new exact array job ID.
- Preserve 40 existing `model.best` checkpoints, 4 scales x 2 algorithms x 5 training seeds, 50,000 training timesteps, 30 deterministic evaluation episodes per checkpoint, and 1,200 historical canonical evaluation episodes.
- Do not change actor architecture, critic, replay buffer, reward, simulator, state, action interface, checkpoints, training algorithm, formal job `58513929`, seed schedule, task mapping, 30-episode protocol, diagnostic schema-v3 CSV headers, or reducer totals of 8 tasks, 40 checkpoints, and 1,200 episodes.
- No retraining is allowed.
- Keep `diagnostic_schema_version=3`.
- Add exact `reconciliation_contract_version=2`.
- Do not tune tolerances to job `58656380`; the repair removes the incorrect historical floating hard gate.
- Same-pass metric disagreement is a hard failure with `failure_category=same_pass_metric_mismatch`.
- Historical floating and historical saturation differences are mandatory audit evidence only.
- Missing or malformed required inputs fail before complete evidence fabrication with `input_contract_mismatch`; the all-evidence-write guarantee begins only after historical canonical, same-pass canonical, and diagnostic inputs are readable and schema-readable.
- Use synthetic CSV/package fixtures for local tests. No local test may invoke `sbatch`, `sacct` without a fixture file, network access, retraining, or heavy ML evaluation.

---

## Current File Map

- `evaluate_td3_gnn.py:19-43` defines canonical action and required CSV columns. `evaluate_td3_gnn.py:140-163` defines `action_diagnostics_from_actions()`, `evaluate_td3_gnn.py:209-214` defines `scalar_stats()`, and `evaluate_td3_gnn.py:217-272` defines `build_csv_rows()`. These are the only canonical formulas the diagnostic evaluator may use.
- `evaluate_td3_gnn_infrastructure_diagnostics.py:8-36` imports evaluator and diagnostic helpers. `evaluate_td3_gnn_infrastructure_diagnostics.py:83-148` runs one diagnostic episode and currently discards the mapped-action stream after building `action_summary`. `evaluate_td3_gnn_infrastructure_diagnostics.py:243-342` runs the eval loop and writes the four schema-v3 diagnostic CSVs.
- `utils/infrastructure_diagnostics.py:6-222` freezes `DIAGNOSTIC_SCHEMA_VERSION = "3"` and all schema-v3 diagnostic headers. `utils/infrastructure_diagnostics.py:485-583` aggregates per-infrastructure action summaries. `utils/infrastructure_diagnostics.py:585-705` builds each schema-v3 episode row. `utils/infrastructure_diagnostics.py:1229-1364` computes global action summaries.
- `scripts/validate_full_infrastructure_diagnostic_eval30.py:29-83` defines Stage D seed schedule and task mapping. `scripts/validate_full_infrastructure_diagnostic_eval30.py:157-180` currently defines the incorrect historical canonical-vs-diagnostic floating hard gate. `scripts/validate_full_infrastructure_diagnostic_eval30.py:437-584` validates seed output directories. `scripts/validate_full_infrastructure_diagnostic_eval30.py:605-614` defines seed package members. `scripts/validate_full_infrastructure_diagnostic_eval30.py:1443-1553` builds service reconciliation rows but raises before callers can write evidence. `scripts/validate_full_infrastructure_diagnostic_eval30.py:1619-1778` prepares seed validation and currently raises before writing `validation/canonical_reconciliation.csv` on canonical mismatch. `scripts/validate_full_infrastructure_diagnostic_eval30.py:2301-2343` extracts task package details for reduction. `scripts/validate_full_infrastructure_diagnostic_eval30.py:2470-2724` builds the complete bundle.
- `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:126-158` is the array dry-run. `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:242-313` validates formal inputs, runs the evaluator, and calls `prepare-seed-validation`.
- `m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm:65-84` is the reducer dry-run. `m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm:121-134` calls `validate-complete-workflow`.
- `m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh:17-34` is the source allowlist. `m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh:115-128` is its dry-run. `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:24-43` is the required source-bundle member list, `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:240-286` is preflight, and `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:290-310` is the non-submitting dry-run.
- `tests/test_infrastructure_diagnostics.py:16-118` contains lightweight diagnostic CLI and fake env helpers. `tests/test_infrastructure_diagnostics.py:1203-1439` covers schema and seed-summary behavior. `tests/test_infrastructure_diagnostics.py:1483-1584` covers diagnostic evaluator episode validation. `tests/test_infrastructure_diagnostics.py:1926-1968` covers schema-v3 guard tests.
- `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:28-150` covers task mapping and seed schedule. `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:188-254` defines synthetic canonical eval30 CSV helpers. `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:256-530` builds synthetic seed outputs. `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:737-769` currently expects the old historical canonical float hard gate to fail. `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:811-959` builds task package fixtures. `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1284-1567` runs the array script with a synthetic evaluator. `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1880-1954` builds reducer fixtures.

## New Interfaces To Introduce

All names in this section are exact and must be used consistently.

- In `evaluate_td3_gnn_infrastructure_diagnostics.py`:
  - `mapped_action_count_stats(mapped_actions_by_step: list[np.ndarray], max_action: float, tolerance: float) -> dict[str, int]`
  - `build_same_pass_canonical_episode_record(episode_record: dict[str, object], mapped_actions_by_step: list[np.ndarray], max_action: float, max_action_tolerance: float) -> dict[str, object]`
  - `write_same_pass_canonical_eval30(output_dir: Path, metadata: dict[str, object], episode_records: list[dict[str, object]]) -> Path`
  - `evaluate_diagnostic_episode(...) -> dict[str, object]` will keep its current parameters and add these return keys: `mapped_actions_by_step`, `mapped_action_dimension`, and `same_pass_canonical_episode_record`.
- In `scripts/validate_full_infrastructure_diagnostic_eval30.py`:
  - `RECONCILIATION_CONTRACT_VERSION: Final[int] = 2`
  - `SAME_PASS_CANONICAL_FILENAME: Final[str] = "same_pass_canonical_eval30.csv"`
  - `SAME_PASS_RECONCILIATION_FILENAME: Final[str] = "same_pass_canonical_reconciliation.csv"`
  - `HISTORICAL_DRIFT_FILENAME: Final[str] = "historical_canonical_drift.csv"`
  - `RECONCILIATION_SUMMARY_FILENAME: Final[str] = "reconciliation_summary.json"`
  - `SAME_PASS_RECONCILIATION_COLUMNS: Final[tuple[str, ...]]`
  - `HISTORICAL_DRIFT_COLUMNS: Final[tuple[str, ...]]`
  - `SAME_PASS_FIELD_MAP: Final[tuple[tuple[str, str, str], ...]]`
  - `HISTORICAL_IDENTITY_FIELDS: Final[tuple[tuple[str, str, str, str], ...]]`
  - `HISTORICAL_FLOAT_FIELDS: Final[tuple[tuple[str, str], ...]]`
  - `SATURATION_RECONSTRUCTION_RESIDUAL_LIMIT: Final[Decimal] = Decimal("1e-9")`
  - `SeedReconciliationInputs` dataclass with fields `diagnostic_rows`, `transformer_rows`, `charger_rows`, `historical_canonical_rows`, `same_pass_canonical_rows`, `formal_validation`, `scale`, `algorithm`, `task_id`, `training_seed`, `stage_d_source_commit_sha`, `config_path`, and `checkpoint_prefix`.
  - `load_seed_reconciliation_inputs(...) -> SeedReconciliationInputs`
  - `build_same_pass_reconciliation_rows(inputs: SeedReconciliationInputs) -> list[dict[str, object]]`
  - `build_historical_canonical_drift_rows(inputs: SeedReconciliationInputs) -> list[dict[str, object]]`
  - `build_service_reconciliation_rows(episode_rows: list[dict[str, str]], transformer_rows: list[dict[str, str]], charger_rows: list[dict[str, str]]) -> list[dict[str, object]]`
  - `evaluate_service_reconciliation_rows(rows: list[dict[str, object]]) -> int`
  - `build_mapping_validation_payload(inputs: SeedReconciliationInputs, transformer_rows: list[dict[str, str]], charger_rows: list[dict[str, str]]) -> dict[str, object]`
  - `build_reconciliation_summary_payload(...) -> dict[str, object]`
  - `atomic_write_text(path: Path, text: str) -> None`
  - `atomic_write_csv_rows(path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict[str, object]]) -> None`
  - `atomic_write_json(path: Path, payload: dict[str, object]) -> None`
  - `raise_reconciliation_contract_error(failure_categories: list[str]) -> None`
  - `reconstruct_saturation_count(fraction_text: str, denominator: int) -> tuple[int | None, Decimal, bool]`

## Frozen Same-Pass Field Map

`SAME_PASS_FIELD_MAP` must contain exactly these pairs and comparison types:

```python
SAME_PASS_FIELD_MAP: Final[tuple[tuple[str, str, str], ...]] = (
    ("episode_reward", "episode_reward", "float_exact"),
    ("tracking_error", "tracking_error", "float_exact"),
    ("energy_tracking_error", "energy_tracking_error", "float_exact"),
    ("power_tracker_violation", "power_tracker_violation", "float_exact"),
    ("total_energy_charged", "total_energy_charged", "float_exact"),
    ("total_energy_discharged", "total_energy_discharged", "float_exact"),
    ("average_user_satisfaction", "average_user_satisfaction", "float_exact"),
    ("energy_user_satisfaction", "energy_user_satisfaction", "float_exact"),
    ("total_transformer_overload", "total_transformer_overload", "float_exact"),
    ("action_mean", "global_action_mean_all_slots", "float_exact"),
    ("action_fraction_at_max", "global_action_fraction_at_max_all_slots", "fraction_count_exact"),
    ("active_action_count_mean", "nonzero_action_count_mean_all_slots", "float_exact"),
)
```

Same-pass `float_exact` means both values parse as finite Python floats and `canonical_float == diagnostic_float`. This is deterministic, not configurable, and is valid because the values are generated from one in-memory execution. Same-pass `fraction_count_exact` also requires exact reconstructed count equality using `same_pass_at_max_count`, `diagnostic_count`, and `total_action_decision_denominator`; disagreement sets `failure_category=same_pass_metric_mismatch`.

## Frozen Historical Contracts

`HISTORICAL_IDENTITY_FIELDS` must hard-gate exact identity for `algorithm`, `seed` to `training_seed`, `episode_index`, `episode_seed`, `episode_steps`, `done`, and `total_ev_served`. Checkpoint/config provenance must also hard-gate exact SHA-256 equality between the staged config/checkpoint files and `runtime_metadata/formal_package_validation.json`. The formal job ID must be exactly `"58513929"`. The Stage D source commit must be exact 40 lowercase hex and must match the source commit recorded in seed reconciliation summary, task package metadata, and reducer expectation.

`HISTORICAL_FLOAT_FIELDS` must audit, never hard-gate, these fields:

```python
HISTORICAL_FLOAT_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("episode_reward", "episode_reward"),
    ("tracking_error", "tracking_error"),
    ("energy_tracking_error", "energy_tracking_error"),
    ("power_tracker_violation", "power_tracker_violation"),
    ("total_energy_charged", "total_energy_charged"),
    ("total_energy_discharged", "total_energy_discharged"),
    ("average_user_satisfaction", "average_user_satisfaction"),
    ("energy_user_satisfaction", "energy_user_satisfaction"),
    ("total_transformer_overload", "total_transformer_overload"),
    ("action_mean", "action_mean"),
    ("action_fraction_at_max", "action_fraction_at_max"),
    ("active_action_count_mean", "active_action_count_mean"),
)
```

Allowed drift classifications are exactly `exact_match`, `historical_float_drift`, `historical_saturation_count_drift`, `historical_identity_match`, `historical_identity_mismatch`, and `not_comparable`.

## Task 1: Same-Pass Canonical Episode Record And CSV Generation

**Files:**
- Modify: `evaluate_td3_gnn_infrastructure_diagnostics.py:8-36`, `evaluate_td3_gnn_infrastructure_diagnostics.py:83-148`, `evaluate_td3_gnn_infrastructure_diagnostics.py:188-194`, `evaluate_td3_gnn_infrastructure_diagnostics.py:243-342`
- Test: `tests/test_infrastructure_diagnostics.py:16-118`, `tests/test_infrastructure_diagnostics.py:1483-1584`, `tests/test_infrastructure_diagnostics.py:2206-2218`

**Interfaces:**
- Consumes: `evaluate_td3_gnn.action_diagnostics_from_actions(mapped_actions, max_action, tolerance=1e-6)` from `evaluate_td3_gnn.py:140-163`, `evaluate_td3_gnn.scalar_stats(stats)` from `evaluate_td3_gnn.py:209-214`, and `evaluate_td3_gnn.build_csv_rows(metadata, episode_records)` from `evaluate_td3_gnn.py:217-272`.
- Produces: `mapped_action_count_stats()`, `build_same_pass_canonical_episode_record()`, `write_same_pass_canonical_eval30()`, and new `evaluate_diagnostic_episode()` return keys `mapped_actions_by_step`, `mapped_action_dimension`, `same_pass_canonical_episode_record`.

- [ ] **Step 1: Write the failing same-pass episode-record tests**

Add these tests after `test_diagnostic_evaluator_rejects_invalid_active_slots_before_env_step()` at `tests/test_infrastructure_diagnostics.py:1517-1568`:

```python
class TwoStepDiagnosticEnv:
    def __init__(self):
        self.charging_stations = [fake_charger(0, 2, 0)]
        self.action_space = SimpleNamespace(
            low=np.array([0.0, 0.0], dtype=float),
            high=np.array([1.0, 1.0], dtype=float),
        )
        self.states = [
            fake_state([0, 1], [[0.5, 0.0, 1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            fake_state([0, 1], [[0.5, 0.0, 1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        ]
        self.step_index = 0

    def reset(self, seed=None):
        self.step_index = 0
        return self.states[0], {}

    def step(self, mapped_action):
        self.step_index += 1
        done = self.step_index == 2
        stats = {
            "tracking_error": 4.0,
            "energy_tracking_error": 5.0,
            "power_tracker_violation": 6.0,
            "total_energy_charged": 7.0,
            "total_energy_discharged": 0.0,
            "average_user_satisfaction": 0.9,
            "energy_user_satisfaction": 99.0,
            "total_transformer_overload": 0.0,
            "total_ev_served": 2,
        }
        next_state = self.states[min(self.step_index, 1)]
        return next_state, 1.5, done, stats


class SequenceActionPolicy:
    def __init__(self, mapped_actions):
        self.mapped_actions = [np.asarray(action, dtype=np.float32) for action in mapped_actions]
        self.index = 0

    def select_action(self, state, expl_noise=0.0, return_mapped_action=False):
        assert return_mapped_action is True
        action = self.mapped_actions[self.index]
        self.index += 1
        return action


def test_diagnostic_episode_returns_same_pass_canonical_record_from_one_mapped_action_stream():
    from evaluate_td3_gnn import action_diagnostics_from_actions
    from evaluate_td3_gnn_infrastructure_diagnostics import evaluate_diagnostic_episode

    mapped_actions = [
        np.array([1.0, 0.5], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    episode = evaluate_diagnostic_episode(
        policy=SequenceActionPolicy(mapped_actions),
        env=TwoStepDiagnosticEnv(),
        seed=710000,
        max_action=1.0,
        max_action_tolerance=1e-6,
    )

    assert [action.tolist() for action in episode["mapped_actions_by_step"]] == [[1.0, 0.5], [0.0, 1.0]]
    assert episode["mapped_action_dimension"] == 2
    same_pass = episode["same_pass_canonical_episode_record"]
    assert same_pass["episode_reward"] == pytest.approx(3.0)
    assert same_pass["episode_steps"] == 2
    assert same_pass["done"] is True
    assert same_pass["stats"]["tracking_error"] == pytest.approx(4.0)
    assert same_pass["stats"]["mapped_action_dimension"] == 2
    assert same_pass["stats"]["same_pass_at_max_count"] == 2
    assert same_pass["stats"]["total_action_decision_denominator"] == 4
    assert same_pass["action_fraction_at_max"] == pytest.approx(
        action_diagnostics_from_actions(mapped_actions, max_action=1.0, tolerance=1e-6)["action_fraction_at_max"]
    )


def test_same_pass_canonical_eval30_csv_uses_canonical_required_columns_and_summary(tmp_path):
    import csv
    from evaluate_td3_gnn import REQUIRED_COLUMNS
    from evaluate_td3_gnn_infrastructure_diagnostics import write_same_pass_canonical_eval30

    episode_records = []
    for episode_index in range(30):
        episode_records.append(
            {
                "episode_index": episode_index,
                "episode_seed": 710000 + episode_index,
                "episode_reward": float(episode_index),
                "episode_steps": 112,
                "done": True,
                "stats": {
                    "tracking_error": float(episode_index) + 0.5,
                    "mapped_action_dimension": 25,
                    "same_pass_at_max_count": episode_index,
                    "total_action_decision_denominator": 2800,
                },
                "action_mean": 0.5,
                "action_std": 0.0,
                "action_min": 0.0,
                "action_max": 1.0,
                "action_fraction_zero": 0.5,
                "action_fraction_at_max": 0.5,
                "active_action_count_mean": 1.0,
            }
        )
    output_path = write_same_pass_canonical_eval30(
        tmp_path,
        {
            "run_name": "same_pass",
            "algorithm": "actiongnn",
            "config": "config_files/PublicPST_25cp.yaml",
            "seed": 0,
            "checkpoint": "checkpoint/model.best",
        },
        episode_records,
    )

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert output_path == tmp_path / "same_pass_canonical_eval30.csv"
    assert reader.fieldnames[: len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS
    assert "row_type" in reader.fieldnames
    assert "tracking_error" in reader.fieldnames
    assert "mapped_action_dimension" in reader.fieldnames
    assert "same_pass_at_max_count" in reader.fieldnames
    assert "total_action_decision_denominator" in reader.fieldnames
    assert sum(row["row_type"] == "episode" for row in rows) == 30
    assert sum(row["row_type"] == "summary" for row in rows) == 1
    assert [int(row["episode_seed"]) for row in rows if row["row_type"] == "episode"] == list(range(710000, 710030))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_infrastructure_diagnostics.py::test_diagnostic_episode_returns_same_pass_canonical_record_from_one_mapped_action_stream tests/test_infrastructure_diagnostics.py::test_same_pass_canonical_eval30_csv_uses_canonical_required_columns_and_summary -q
```

Expected: FAIL with `KeyError: 'mapped_actions_by_step'` or `ImportError: cannot import name 'write_same_pass_canonical_eval30'`.

- [ ] **Step 3: Implement the minimal same-pass evaluator data flow**

Patch `evaluate_td3_gnn_infrastructure_diagnostics.py:8-18` to import canonical helpers:

```python
from evaluate_td3_gnn import (
    ALGORITHM_CHOICES,
    action_diagnostics_from_actions,
    build_csv_rows as build_canonical_csv_rows,
    create_policy,
    load_checkpoint_kwargs,
    load_policy_checkpoint,
    make_env,
    normalise_algorithm_label,
    normalise_checkpoint_prefix,
    normalise_step_result,
    reset_env_state,
)
```

Add these functions after `select_mapped_action()` at `evaluate_td3_gnn_infrastructure_diagnostics.py:71-80`:

```python
def mapped_action_count_stats(mapped_actions_by_step, max_action, tolerance):
    stacked_actions = np.asarray(mapped_actions_by_step, dtype=np.float32)
    if stacked_actions.size == 0:
        return {
            "mapped_action_dimension": 0,
            "same_pass_at_max_count": 0,
            "total_action_decision_denominator": 0,
        }
    if stacked_actions.ndim == 1:
        stacked_actions = stacked_actions.reshape(1, -1)
    at_max_mask = stacked_actions >= (float(max_action) - float(tolerance))
    return {
        "mapped_action_dimension": int(stacked_actions.shape[1]),
        "same_pass_at_max_count": int(np.count_nonzero(at_max_mask)),
        "total_action_decision_denominator": int(stacked_actions.shape[0] * stacked_actions.shape[1]),
    }


def build_same_pass_canonical_episode_record(
    episode_record,
    mapped_actions_by_step,
    max_action,
    max_action_tolerance,
):
    canonical_stats = dict(episode_record.get("stats", {}))
    canonical_stats.update(
        mapped_action_count_stats(
            mapped_actions_by_step,
            max_action=max_action,
            tolerance=max_action_tolerance,
        )
    )
    canonical_episode = {
        "episode_reward": episode_record["episode_reward"],
        "episode_steps": episode_record["episode_steps"],
        "done": episode_record["done"],
        "stats": canonical_stats,
        "reset_info": episode_record.get("reset_info", {}),
    }
    canonical_episode.update(
        action_diagnostics_from_actions(
            mapped_actions_by_step,
            max_action=max_action,
            tolerance=max_action_tolerance,
        )
    )
    return canonical_episode


def write_same_pass_canonical_eval30(output_dir, metadata, episode_records):
    rows, fieldnames = build_canonical_csv_rows(metadata, episode_records)
    output_path = Path(output_dir) / "same_pass_canonical_eval30.csv"
    write_csv(output_path, fieldnames, rows)
    return output_path
```

Patch the return value at `evaluate_td3_gnn_infrastructure_diagnostics.py:141-148`:

```python
    episode_record = {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "done": done,
        "stats": stats,
        "reset_info": reset_info,
        "action_summary": action_summary,
        "mapped_actions_by_step": mapped_actions_by_step,
        "mapped_action_dimension": int(slot_to_charger_id.size),
    }
    episode_record["same_pass_canonical_episode_record"] = build_same_pass_canonical_episode_record(
        episode_record,
        mapped_actions_by_step=mapped_actions_by_step,
        max_action=max_action,
        max_action_tolerance=max_action_tolerance,
    )
    return episode_record
```

Patch the eval loop at `evaluate_td3_gnn_infrastructure_diagnostics.py:286-342` to collect canonical records:

```python
    same_pass_canonical_records = []
    for episode_index in range(args.eval_episodes):
        ...
        same_pass_record = dict(episode_record["same_pass_canonical_episode_record"])
        same_pass_record["episode_index"] = episode_index
        same_pass_record["episode_seed"] = episode_seed
        same_pass_canonical_records.append(same_pass_record)
```

After the four existing diagnostic CSV writes at `evaluate_td3_gnn_infrastructure_diagnostics.py:339-342`, add:

```python
    write_same_pass_canonical_eval30(
        output_dir,
        {
            "run_name": args.run_name,
            "algorithm": canonical_algorithm,
            "config": args.config,
            "seed": args.seed,
            "checkpoint": str(checkpoint_prefix),
        },
        same_pass_canonical_records,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_infrastructure_diagnostics.py::test_diagnostic_episode_returns_same_pass_canonical_record_from_one_mapped_action_stream tests/test_infrastructure_diagnostics.py::test_same_pass_canonical_eval30_csv_uses_canonical_required_columns_and_summary -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add evaluate_td3_gnn_infrastructure_diagnostics.py tests/test_infrastructure_diagnostics.py
git commit -m "Add same-pass canonical diagnostic output"
```

## Task 2: Same-Pass Hard Reconciliation

**Files:**
- Modify: `scripts/validate_full_infrastructure_diagnostic_eval30.py:157-180`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1556-1617`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1619-1778`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:2788-2797`
- Test: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:188-254`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:256-530`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:737-769`

**Interfaces:**
- Consumes: same-pass CSV at `diagnostics/same_pass_canonical_eval30.csv` from Task 1 and schema-v3 episode rows at `diagnostics/episode_diagnostics.csv`.
- Produces: `SAME_PASS_FIELD_MAP`, `SAME_PASS_RECONCILIATION_COLUMNS`, `build_same_pass_reconciliation_rows()`, and `validation/same_pass_canonical_reconciliation.csv` rows with `failure_category=same_pass_metric_mismatch` on disagreement.

- [ ] **Step 1: Write failing same-pass reconciliation tests**

Replace `test_prepare_seed_validation_rejects_canonical_float_mismatch()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:737-769` with these tests:

```python
def write_same_pass_canonical_eval30(path, *, scale="25cp", algorithm="actiongnn", training_seed=0, mutator=None):
    write_canonical_eval30(path, scale=scale, algorithm=algorithm, training_seed=training_seed, mutator=mutator)


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_same_pass_reward_tracking_mismatch_hard_fails(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_same_pass_reconciliation_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    same_pass_csv = seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"

    def mutate(rows):
        rows[0]["tracking_error"] = "999.0"

    write_same_pass_canonical_eval30(same_pass_csv, mutator=mutate)
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=tmp_path / "unused_historical.csv",
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        require_historical=False,
    )
    rows = build_same_pass_reconciliation_rows(inputs)

    failed = [row for row in rows if row["status"] == "fail"]
    assert failed
    assert failed[0]["field"] == "tracking_error"
    assert failed[0]["failure_category"] == "same_pass_metric_mismatch"


def test_same_pass_action_summary_mismatch_hard_fails_with_count_fields(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_same_pass_reconciliation_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    same_pass_csv = seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"

    def mutate(rows):
        rows[0]["action_fraction_at_max"] = "0.25"
        rows[0]["same_pass_at_max_count"] = "1"
        rows[0]["total_action_decision_denominator"] = "4"

    write_same_pass_canonical_eval30(same_pass_csv, mutator=mutate)
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=tmp_path / "unused_historical.csv",
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        require_historical=False,
    )
    rows = build_same_pass_reconciliation_rows(inputs)

    row = next(item for item in rows if item["field"] == "action_fraction_at_max")
    assert row["status"] == "fail"
    assert row["same_pass_count"] == 1
    assert row["diagnostic_count"] == 2
    assert row["total_action_decision_denominator"] == 4
    assert row["failure_category"] == "same_pass_metric_mismatch"
```

Also update `build_seed_output()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:333-530` to write a passing same-pass canonical CSV by default:

```python
    write_same_pass_canonical_eval30(diagnostics / "same_pass_canonical_eval30.csv", scale=scale, algorithm=algorithm, training_seed=training_seed)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_same_pass_reward_tracking_mismatch_hard_fails tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_same_pass_action_summary_mismatch_hard_fails_with_count_fields -q
```

Expected: FAIL with `ImportError: cannot import name 'build_same_pass_reconciliation_rows'` or `TypeError: load_seed_reconciliation_inputs() got an unexpected keyword argument`.

- [ ] **Step 3: Implement same-pass builders and deterministic comparison**

Patch `scripts/validate_full_infrastructure_diagnostic_eval30.py:114-119` to add:

```python
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
```

Patch constants around `scripts/validate_full_infrastructure_diagnostic_eval30.py:130-180`:

```python
RECONCILIATION_CONTRACT_VERSION: Final[int] = 2
SAME_PASS_CANONICAL_FILENAME: Final[str] = "same_pass_canonical_eval30.csv"
SAME_PASS_RECONCILIATION_FILENAME: Final[str] = "same_pass_canonical_reconciliation.csv"
HISTORICAL_DRIFT_FILENAME: Final[str] = "historical_canonical_drift.csv"
RECONCILIATION_SUMMARY_FILENAME: Final[str] = "reconciliation_summary.json"

SAME_PASS_RECONCILIATION_COLUMNS: Final[tuple[str, ...]] = (
    "reconciliation_contract_version",
    "episode_index",
    "field",
    "comparison_type",
    "same_pass_canonical_value",
    "diagnostic_value",
    "absolute_difference",
    "relative_difference",
    "same_pass_count",
    "diagnostic_count",
    "total_action_decision_denominator",
    "status",
    "failure_category",
)

SAME_PASS_FIELD_MAP: Final[tuple[tuple[str, str, str], ...]] = (
    ("episode_reward", "episode_reward", "float_exact"),
    ("tracking_error", "tracking_error", "float_exact"),
    ("energy_tracking_error", "energy_tracking_error", "float_exact"),
    ("power_tracker_violation", "power_tracker_violation", "float_exact"),
    ("total_energy_charged", "total_energy_charged", "float_exact"),
    ("total_energy_discharged", "total_energy_discharged", "float_exact"),
    ("average_user_satisfaction", "average_user_satisfaction", "float_exact"),
    ("energy_user_satisfaction", "energy_user_satisfaction", "float_exact"),
    ("total_transformer_overload", "total_transformer_overload", "float_exact"),
    ("action_mean", "global_action_mean_all_slots", "float_exact"),
    ("action_fraction_at_max", "global_action_fraction_at_max_all_slots", "fraction_count_exact"),
    ("active_action_count_mean", "nonzero_action_count_mean_all_slots", "float_exact"),
)
```

Add this dataclass after `EpisodeKey` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:121`:

```python
@dataclass(frozen=True)
class SeedReconciliationInputs:
    diagnostic_rows: list[dict[str, str]]
    transformer_rows: list[dict[str, str]]
    charger_rows: list[dict[str, str]]
    historical_canonical_rows: list[dict[str, str]]
    same_pass_canonical_rows: list[dict[str, str]]
    formal_validation: dict[str, object]
    scale: str
    algorithm: str
    task_id: int
    training_seed: int
    stage_d_source_commit_sha: str
    config_path: Path | None
    checkpoint_prefix: Path | None
```

Add these helpers before `_canonical_exact_values()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:1556`:

```python
def _episode_rows_by_index(rows: list[dict[str, str]], label: str) -> dict[int, dict[str, str]]:
    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("row_type", "episode") != "episode":
            continue
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        if episode_index in by_index:
            raise ValueError(f"{label} duplicate episode index: {episode_index}")
        by_index[episode_index] = row
    missing = sorted(set(range(EVAL_EPISODES)) - set(by_index))
    if missing:
        raise ValueError(f"{label} missing episode index(es): {missing}")
    return by_index


def load_seed_reconciliation_inputs(
    *,
    diagnostic_dir: Path,
    historical_canonical_csv: Path,
    validation_dir: Path,
    task_id: int,
    training_seed: int,
    require_historical: bool = True,
    formal_validation_json: Path | None = None,
    stage_d_source_commit_sha: str = "",
    config_path: Path | None = None,
    checkpoint_prefix: Path | None = None,
) -> SeedReconciliationInputs:
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])
    diagnostic_root = Path(diagnostic_dir)
    _, diagnostic_rows = _read_csv(diagnostic_root / "episode_diagnostics.csv")
    _, transformer_rows = _read_csv(diagnostic_root / "transformer_diagnostics.csv")
    _, charger_rows = _read_csv(diagnostic_root / "charger_diagnostics.csv")
    _, same_pass_rows = _read_csv(diagnostic_root / SAME_PASS_CANONICAL_FILENAME)
    historical_rows: list[dict[str, str]] = []
    if require_historical:
        _, historical_rows = _read_csv(Path(historical_canonical_csv))
    formal_validation = (
        _read_json_object(Path(formal_validation_json), "formal package validation")
        if formal_validation_json is not None
        else {}
    )
    return SeedReconciliationInputs(
        diagnostic_rows=diagnostic_rows,
        transformer_rows=transformer_rows,
        charger_rows=charger_rows,
        historical_canonical_rows=historical_rows,
        same_pass_canonical_rows=same_pass_rows,
        formal_validation=formal_validation,
        scale=scale,
        algorithm=algorithm,
        task_id=task_id,
        training_seed=training_seed,
        stage_d_source_commit_sha=stage_d_source_commit_sha,
        config_path=Path(config_path) if config_path is not None else None,
        checkpoint_prefix=Path(checkpoint_prefix) if checkpoint_prefix is not None else None,
    )


def _same_pass_float_row(episode_index, canonical_field, diagnostic_field, comparison_type, canonical_value, diagnostic_value):
    expected = _finite_number(canonical_value, canonical_field)
    observed = _finite_number(diagnostic_value, diagnostic_field)
    absolute_difference = abs(observed - expected)
    relative_difference = absolute_difference / max(abs(expected), 1e-12)
    status = "pass" if observed == expected else "fail"
    return {
        "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
        "episode_index": episode_index,
        "field": canonical_field,
        "comparison_type": comparison_type,
        "same_pass_canonical_value": expected,
        "diagnostic_value": observed,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "same_pass_count": "",
        "diagnostic_count": "",
        "total_action_decision_denominator": "",
        "status": status,
        "failure_category": "" if status == "pass" else "same_pass_metric_mismatch",
    }


def _count_from_fraction(value: float, denominator: int) -> int:
    return int(Decimal(str(value * denominator)).to_integral_value(rounding=ROUND_HALF_UP))


def build_same_pass_reconciliation_rows(inputs: SeedReconciliationInputs) -> list[dict[str, object]]:
    diagnostic_by_index = _episode_rows_by_index(inputs.diagnostic_rows, "diagnostic rows")
    same_pass_by_index = _episode_rows_by_index(inputs.same_pass_canonical_rows, "same-pass canonical rows")
    rows: list[dict[str, object]] = []
    for episode_index in range(EVAL_EPISODES):
        diagnostic = diagnostic_by_index[episode_index]
        same_pass = same_pass_by_index[episode_index]
        for canonical_field, diagnostic_field, comparison_type in SAME_PASS_FIELD_MAP:
            if comparison_type == "fraction_count_exact":
                expected = _finite_number(same_pass.get(canonical_field), canonical_field)
                observed = _finite_number(diagnostic.get(diagnostic_field), diagnostic_field)
                denominator = _exact_integer(
                    same_pass.get("total_action_decision_denominator"),
                    "total_action_decision_denominator",
                )
                same_pass_count = _exact_integer(same_pass.get("same_pass_at_max_count"), "same_pass_at_max_count")
                diagnostic_count = _count_from_fraction(observed, denominator)
                absolute_difference = abs(observed - expected)
                relative_difference = absolute_difference / max(abs(expected), 1e-12)
                status = (
                    "pass"
                    if expected == observed and same_pass_count == diagnostic_count
                    else "fail"
                )
                rows.append({
                    "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
                    "episode_index": episode_index,
                    "field": canonical_field,
                    "comparison_type": comparison_type,
                    "same_pass_canonical_value": expected,
                    "diagnostic_value": observed,
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                    "same_pass_count": same_pass_count,
                    "diagnostic_count": diagnostic_count,
                    "total_action_decision_denominator": denominator,
                    "status": status,
                    "failure_category": "" if status == "pass" else "same_pass_metric_mismatch",
                })
            else:
                rows.append(
                    _same_pass_float_row(
                        episode_index,
                        canonical_field,
                        diagnostic_field,
                        comparison_type,
                        same_pass.get(canonical_field),
                        diagnostic.get(diagnostic_field),
                    )
                )
    return rows
```

Do not use `CANONICAL_FLOAT_RECONCILIATION` for same-pass validation. It is the old historical gate and will be removed in Task 3.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_same_pass_reward_tracking_mismatch_hard_fails tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_same_pass_action_summary_mismatch_hard_fails_with_count_fields -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Add same-pass reconciliation hard gate"
```

## Task 3: Historical Identity Gate And Floating Drift Audit

**Files:**
- Modify: `scripts/validate_full_infrastructure_diagnostic_eval30.py:600-614`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1217-1290`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1556-1778`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:2788-2797`
- Test: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:188-254`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:737-769`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1659-1830`

**Interfaces:**
- Consumes: historical canonical eval30 CSV from formal job `58513929`, same-pass canonical rows, diagnostic rows, staged config/checkpoint paths, `runtime_metadata/formal_package_validation.json`, and Stage D source commit.
- Produces: `HISTORICAL_DRIFT_COLUMNS`, `HISTORICAL_IDENTITY_FIELDS`, `HISTORICAL_FLOAT_FIELDS`, `build_historical_canonical_drift_rows()`, and audit CSV rows for `validation/historical_canonical_drift.csv`.

- [ ] **Step 1: Write failing historical identity and drift tests**

Add these tests after the same-pass reconciliation tests from Task 2:

```python
def test_historical_identity_mismatch_hard_fails_without_using_float_drift_as_gate(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    write_same_pass_canonical_eval30(seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv")
    historical_csv = tmp_path / "historical_eval30.csv"

    def mutate(rows):
        rows[0]["episode_seed"] = "999999"

    write_canonical_eval30(historical_csv, mutator=mutate)
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=None,
        stage_d_source_commit_sha="a" * 40,
    )
    rows = build_historical_canonical_drift_rows(inputs)

    mismatch = [row for row in rows if row["classification"] == "historical_identity_mismatch"]
    assert mismatch
    assert mismatch[0]["field"] == "episode_seed"


def test_historical_floating_drift_is_audit_only_and_classified(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        build_same_pass_reconciliation_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    write_same_pass_canonical_eval30(seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv")
    historical_csv = tmp_path / "historical_eval30.csv"

    def mutate(rows):
        rows[0]["tracking_error"] = "1.0000001"

    write_canonical_eval30(historical_csv, mutator=mutate)
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=None,
        stage_d_source_commit_sha="a" * 40,
    )

    same_pass_rows = build_same_pass_reconciliation_rows(inputs)
    drift_rows = build_historical_canonical_drift_rows(inputs)

    assert all(row["status"] == "pass" for row in same_pass_rows)
    drift = next(
        row
        for row in drift_rows
        if row["episode_index"] == 0 and row["field"] == "tracking_error"
    )
    assert drift["classification"] == "historical_float_drift"
    assert drift["historical_source_label"] == "formal_job_58513929_canonical_eval30"
    assert drift["stage_d_source_label"] == "stage_d_same_pass_canonical_eval30"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_identity_mismatch_hard_fails_without_using_float_drift_as_gate tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_floating_drift_is_audit_only_and_classified -q
```

Expected: FAIL with `ImportError: cannot import name 'build_historical_canonical_drift_rows'`.

- [ ] **Step 3: Implement historical identity and float audit builders**

Add constants near `SAME_PASS_FIELD_MAP` in `scripts/validate_full_infrastructure_diagnostic_eval30.py:130-180`:

```python
HISTORICAL_DRIFT_COLUMNS: Final[tuple[str, ...]] = (
    "reconciliation_contract_version",
    "scale",
    "algorithm",
    "training_seed",
    "formal_task_id",
    "episode_index",
    "episode_seed",
    "field",
    "historical_source_label",
    "stage_d_source_label",
    "historical_value",
    "stage_d_same_pass_value",
    "absolute_difference",
    "relative_difference",
    "classification",
    "historical_at_max_count",
    "stage_d_at_max_count",
    "total_action_decision_denominator",
    "count_difference",
    "fraction_difference",
)

HISTORICAL_IDENTITY_FIELDS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("algorithm", "algorithm", "algorithm", "string"),
    ("training_seed", "seed", "seed", "integer"),
    ("episode_index", "episode_index", "episode_index", "integer"),
    ("episode_seed", "episode_seed", "episode_seed", "integer"),
    ("episode_steps", "episode_steps", "episode_steps", "integer"),
    ("done", "done", "done", "boolean"),
    ("total_ev_served", "total_ev_served", "total_ev_served", "integer"),
)

HISTORICAL_FLOAT_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("episode_reward", "episode_reward"),
    ("tracking_error", "tracking_error"),
    ("energy_tracking_error", "energy_tracking_error"),
    ("power_tracker_violation", "power_tracker_violation"),
    ("total_energy_charged", "total_energy_charged"),
    ("total_energy_discharged", "total_energy_discharged"),
    ("average_user_satisfaction", "average_user_satisfaction"),
    ("energy_user_satisfaction", "energy_user_satisfaction"),
    ("total_transformer_overload", "total_transformer_overload"),
    ("action_mean", "action_mean"),
    ("action_fraction_at_max", "action_fraction_at_max"),
    ("active_action_count_mean", "active_action_count_mean"),
)
```

Add helper builders after `build_same_pass_reconciliation_rows()`:

```python
def _historical_base_row(inputs: SeedReconciliationInputs, episode_index: int, episode_seed_value: object, field: str) -> dict[str, object]:
    return {
        "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
        "scale": inputs.scale,
        "algorithm": inputs.algorithm,
        "training_seed": inputs.training_seed,
        "formal_task_id": formal_task_id(inputs.task_id, inputs.training_seed),
        "episode_index": episode_index,
        "episode_seed": episode_seed_value,
        "field": field,
        "historical_source_label": "formal_job_58513929_canonical_eval30",
        "stage_d_source_label": "stage_d_same_pass_canonical_eval30",
        "historical_value": "",
        "stage_d_same_pass_value": "",
        "absolute_difference": "",
        "relative_difference": "",
        "classification": "",
        "historical_at_max_count": "",
        "stage_d_at_max_count": "",
        "total_action_decision_denominator": "",
        "count_difference": "",
        "fraction_difference": "",
    }


def _identity_classification(historical, stage_d, value_type, field):
    try:
        if value_type == "integer":
            historical_value = _exact_integer(historical, field)
            stage_d_value = _exact_integer(stage_d, field)
        elif value_type == "boolean":
            historical_value = _parse_bool(historical, field)
            stage_d_value = _parse_bool(stage_d, field)
        else:
            historical_value = str(historical)
            stage_d_value = str(stage_d)
    except ValueError:
        return str(historical), str(stage_d), "not_comparable"
    classification = (
        "historical_identity_match"
        if historical_value == stage_d_value
        else "historical_identity_mismatch"
    )
    return str(historical_value), str(stage_d_value), classification


def build_historical_canonical_drift_rows(inputs: SeedReconciliationInputs) -> list[dict[str, object]]:
    historical_by_index = _episode_rows_by_index(inputs.historical_canonical_rows, "historical canonical rows")
    same_pass_by_index = _episode_rows_by_index(inputs.same_pass_canonical_rows, "same-pass canonical rows")
    rows: list[dict[str, object]] = []
    for episode_index in range(EVAL_EPISODES):
        historical = historical_by_index[episode_index]
        stage_d = same_pass_by_index[episode_index]
        episode_seed_value = stage_d.get("episode_seed", "")
        for field, historical_field, stage_d_field, value_type in HISTORICAL_IDENTITY_FIELDS:
            historical_value, stage_d_value, classification = _identity_classification(
                historical.get(historical_field, ""),
                stage_d.get(stage_d_field, ""),
                value_type,
                field,
            )
            row = _historical_base_row(inputs, episode_index, episode_seed_value, field)
            row.update({
                "historical_value": historical_value,
                "stage_d_same_pass_value": stage_d_value,
                "classification": classification,
            })
            rows.append(row)
        for historical_field, stage_d_field in HISTORICAL_FLOAT_FIELDS:
            row = _historical_base_row(inputs, episode_index, episode_seed_value, historical_field)
            try:
                historical_value = _finite_number(historical.get(historical_field), historical_field)
                stage_d_value = _finite_number(stage_d.get(stage_d_field), stage_d_field)
            except ValueError:
                row.update({
                    "historical_value": historical.get(historical_field, ""),
                    "stage_d_same_pass_value": stage_d.get(stage_d_field, ""),
                    "classification": "not_comparable",
                })
                rows.append(row)
                continue
            absolute_difference = abs(stage_d_value - historical_value)
            relative_difference = absolute_difference / max(abs(historical_value), 1e-12)
            row.update({
                "historical_value": historical_value,
                "stage_d_same_pass_value": stage_d_value,
                "absolute_difference": absolute_difference,
                "relative_difference": relative_difference,
                "classification": "exact_match" if stage_d_value == historical_value else "historical_float_drift",
            })
            rows.append(row)
    return rows
```

Patch `validate_seed_formal_package()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:1217-1290` to include config and checkpoint SHA-256 provenance in the returned JSON:

```python
        config_path = root / config_member(task_id, training_seed)
        checkpoint_members = {
            basename: root / formal_train_member(task_id, training_seed, basename)
            for basename in (
                "model.best_actor",
                "model.best_actor_optimizer",
                "model.best_critic",
                "model.best_critic_optimizer",
                "kwargs.yaml",
            )
        }
```

Then add these returned fields:

```python
        "config_sha256": sha256_file(config_path),
        "checkpoint_member_sha256": {
            basename: sha256_file(path)
            for basename, path in checkpoint_members.items()
        },
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_identity_mismatch_hard_fails_without_using_float_drift_as_gate tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_floating_drift_is_audit_only_and_classified -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Add historical drift audit contract"
```

## Task 4: Count-Based Saturation Audit

**Files:**
- Modify: `scripts/validate_full_infrastructure_diagnostic_eval30.py:130-180`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1556-1778`
- Test: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:737-769`

**Interfaces:**
- Consumes: historical `action_fraction_at_max`, same-pass `action_fraction_at_max`, same-pass `same_pass_at_max_count`, `mapped_action_dimension`, and `total_action_decision_denominator`.
- Produces: `reconstruct_saturation_count(fraction_text: str, denominator: int) -> tuple[int | None, Decimal, bool]` and populated `historical_at_max_count`, `stage_d_at_max_count`, `total_action_decision_denominator`, `count_difference`, and `fraction_difference` in `validation/historical_canonical_drift.csv`.

- [ ] **Step 1: Write failing saturation reconstruction tests**

Add these tests after `test_historical_floating_drift_is_audit_only_and_classified()`:

```python
@pytest.mark.parametrize(
    ("scale", "mapped_action_dimension"),
    [("25cp", 25), ("100cp", 100), ("500cp", 500), ("1000cp", 1000)],
)
def test_one_count_historical_saturation_drift_is_audit_only(scale, mapped_action_dimension, tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        build_same_pass_reconciliation_rows,
        load_seed_reconciliation_inputs,
    )

    task_id = {"25cp": 0, "100cp": 2, "500cp": 4, "1000cp": 6}[scale]
    seed_dir = build_seed_output(tmp_path / "seed0", task_id=task_id)
    same_pass_csv = seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"
    denominator = 112 * mapped_action_dimension
    stage_d_count = 2
    historical_count = 1

    def same_pass_mutate(rows):
        for row in rows:
            if row["row_type"] == "episode":
                row["action_fraction_at_max"] = str(stage_d_count / denominator)
                row["same_pass_at_max_count"] = str(stage_d_count)
                row["mapped_action_dimension"] = str(mapped_action_dimension)
                row["total_action_decision_denominator"] = str(denominator)

    write_same_pass_canonical_eval30(same_pass_csv, scale=scale, mutator=same_pass_mutate)
    diagnostics_path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def diagnostic_mutate(fieldnames, rows):
        for row in rows:
            row["global_action_fraction_at_max_all_slots"] = str(stage_d_count / denominator)

    mutate_csv(diagnostics_path, diagnostic_mutate)
    historical_csv = tmp_path / f"historical_{scale}.csv"

    def historical_mutate(rows):
        for row in rows:
            row["action_fraction_at_max"] = str(historical_count / denominator)

    write_canonical_eval30(historical_csv, scale=scale, mutator=historical_mutate)
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=task_id,
        training_seed=0,
        stage_d_source_commit_sha="a" * 40,
    )

    assert all(row["status"] == "pass" for row in build_same_pass_reconciliation_rows(inputs))
    drift_rows = build_historical_canonical_drift_rows(inputs)
    saturation = next(row for row in drift_rows if row["field"] == "action_fraction_at_max" and row["episode_index"] == 0)
    assert saturation["classification"] == "historical_saturation_count_drift"
    assert saturation["historical_at_max_count"] == historical_count
    assert saturation["stage_d_at_max_count"] == stage_d_count
    assert saturation["total_action_decision_denominator"] == denominator
    assert saturation["count_difference"] == 1
    assert saturation["fraction_difference"] == pytest.approx(1 / denominator)


def test_historical_saturation_count_and_denominator_reconstruct_stored_fraction():
    from scripts.validate_full_infrastructure_diagnostic_eval30 import reconstruct_saturation_count

    reconstructed, residual, comparable = reconstruct_saturation_count(str(17 / 2800), 2800)

    assert reconstructed == 17
    assert residual < 1e-9
    assert comparable is True


def test_historical_saturation_reconstruction_not_comparable_when_residual_is_too_large():
    from scripts.validate_full_infrastructure_diagnostic_eval30 import reconstruct_saturation_count

    reconstructed, residual, comparable = reconstruct_saturation_count("0.333", 2800)

    assert reconstructed is None
    assert residual >= 1e-9
    assert comparable is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_one_count_historical_saturation_drift_is_audit_only tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_saturation_count_and_denominator_reconstruct_stored_fraction tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_saturation_reconstruction_not_comparable_when_residual_is_too_large -q
```

Expected: FAIL with `ImportError: cannot import name 'reconstruct_saturation_count'` or classification `historical_float_drift` instead of `historical_saturation_count_drift`.

- [ ] **Step 3: Implement deterministic count reconstruction**

Add this constant beside the reconciliation constants:

```python
SATURATION_RECONSTRUCTION_RESIDUAL_LIMIT: Final[Decimal] = Decimal("1e-9")
```

Add this helper before `build_historical_canonical_drift_rows()`:

```python
def reconstruct_saturation_count(fraction_text: str, denominator: int) -> tuple[int | None, Decimal, bool]:
    if denominator <= 0:
        return None, Decimal("Infinity"), False
    try:
        fraction = Decimal(str(fraction_text))
    except InvalidOperation:
        return None, Decimal("Infinity"), False
    product = fraction * Decimal(int(denominator))
    nearest_count = int(product.to_integral_value(rounding=ROUND_HALF_UP))
    residual = abs(product - Decimal(nearest_count))
    if nearest_count < 0 or nearest_count > denominator:
        return None, residual, False
    if residual >= SATURATION_RECONSTRUCTION_RESIDUAL_LIMIT:
        return None, residual, False
    return nearest_count, residual, True
```

Patch the `action_fraction_at_max` branch inside `build_historical_canonical_drift_rows()`:

```python
            if historical_field == "action_fraction_at_max":
                try:
                    denominator = _exact_integer(
                        stage_d.get("total_action_decision_denominator"),
                        "total_action_decision_denominator",
                    )
                    stage_d_count = _exact_integer(stage_d.get("same_pass_at_max_count"), "same_pass_at_max_count")
                except ValueError:
                    row.update({
                        "historical_value": historical.get(historical_field, ""),
                        "stage_d_same_pass_value": stage_d.get(stage_d_field, ""),
                        "classification": "not_comparable",
                    })
                    rows.append(row)
                    continue
                historical_count, _residual, comparable = reconstruct_saturation_count(
                    str(historical.get(historical_field, "")),
                    denominator,
                )
                if not comparable or historical_count is None:
                    row.update({
                        "historical_value": historical.get(historical_field, ""),
                        "stage_d_same_pass_value": stage_d.get(stage_d_field, ""),
                        "classification": "not_comparable",
                        "stage_d_at_max_count": stage_d_count,
                        "total_action_decision_denominator": denominator,
                    })
                    rows.append(row)
                    continue
                historical_value = _finite_number(historical.get(historical_field), historical_field)
                stage_d_value = _finite_number(stage_d.get(stage_d_field), stage_d_field)
                count_difference = int(stage_d_count - historical_count)
                fraction_difference = float(stage_d_value - historical_value)
                row.update({
                    "historical_value": historical_value,
                    "stage_d_same_pass_value": stage_d_value,
                    "absolute_difference": abs(stage_d_value - historical_value),
                    "relative_difference": abs(stage_d_value - historical_value) / max(abs(historical_value), 1e-12),
                    "classification": (
                        "exact_match"
                        if count_difference == 0 and stage_d_value == historical_value
                        else "historical_saturation_count_drift"
                    ),
                    "historical_at_max_count": historical_count,
                    "stage_d_at_max_count": stage_d_count,
                    "total_action_decision_denominator": denominator,
                    "count_difference": count_difference,
                    "fraction_difference": fraction_difference,
                })
                rows.append(row)
                continue
```

This reconstruction is audit-only. Do not use `SATURATION_RECONSTRUCTION_RESIDUAL_LIMIT` as a scientific pass/fail tolerance; it only decides whether a historical decimal string can be represented as an integer count for the denominator read from the same-pass execution.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_one_count_historical_saturation_drift_is_audit_only tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_saturation_count_and_denominator_reconstruct_stored_fraction tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_historical_saturation_reconstruction_not_comparable_when_residual_is_too_large -q
```

Expected: PASS, `6 passed` because the first test is parametrized across four scales.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Add historical saturation count audit"
```

## Task 5: Atomic Failure-Evidence Writing And Precise Errors

**Files:**
- Modify: `scripts/validate_full_infrastructure_diagnostic_eval30.py:1439-1553`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1619-1795`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:2897-2906`
- Test: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:737-769`

**Interfaces:**
- Consumes: builders from Tasks 2-4.
- Produces: atomic writers, `runtime_metadata/reconciliation_summary.json`, post-read hard-gate error message `Stage D reconciliation contract failed: <comma-separated categories>`, and input-contract error message containing `input_contract_mismatch`.

- [ ] **Step 1: Write failing evidence persistence and input-contract tests**

Add these tests after the saturation tests:

```python
def test_reconciliation_evidence_remains_present_after_hard_gate_failure(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    same_pass_csv = seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"

    def mutate(rows):
        rows[0]["tracking_error"] = "999.0"

    write_same_pass_canonical_eval30(same_pass_csv, mutator=mutate)
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "prepare-seed-validation",
            "--diagnostic-dir",
            str(seed_dir / "diagnostics"),
            "--historical-canonical-csv",
            str(historical_csv),
            "--validation-dir",
            str(seed_dir / "validation"),
            "--runtime-metadata-dir",
            str(seed_dir / "runtime_metadata"),
            "--task-id",
            "0",
            "--training-seed",
            "0",
            "--stage-d-source-commit-sha",
            "a" * 40,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "same_pass_metric_mismatch" in result.stderr
    assert (seed_dir / "validation" / "same_pass_canonical_reconciliation.csv").is_file()
    assert (seed_dir / "validation" / "historical_canonical_drift.csv").is_file()
    assert (seed_dir / "validation" / "service_reconciliation.csv").is_file()
    summary = json.loads((seed_dir / "runtime_metadata" / "reconciliation_summary.json").read_text(encoding="utf-8"))
    assert summary["reconciliation_contract_version"] == 2
    assert summary["status"] == "failed"
    assert summary["hard_gate_status"] == "fail"
    assert "same_pass_metric_mismatch" in summary["failure_categories"]


def test_missing_required_input_fails_input_contract_without_complete_evidence_set(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv").unlink()
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "prepare-seed-validation",
            "--diagnostic-dir",
            str(seed_dir / "diagnostics"),
            "--historical-canonical-csv",
            str(historical_csv),
            "--validation-dir",
            str(seed_dir / "validation"),
            "--runtime-metadata-dir",
            str(seed_dir / "runtime_metadata"),
            "--task-id",
            "0",
            "--training-seed",
            "0",
            "--stage-d-source-commit-sha",
            "a" * 40,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "input_contract_mismatch" in result.stderr
    complete_evidence_paths = [
        seed_dir / "validation" / "same_pass_canonical_reconciliation.csv",
        seed_dir / "validation" / "historical_canonical_drift.csv",
        seed_dir / "validation" / "service_reconciliation.csv",
        seed_dir / "runtime_metadata" / "reconciliation_summary.json",
    ]
    assert not all(path.exists() for path in complete_evidence_paths)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_reconciliation_evidence_remains_present_after_hard_gate_failure tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_missing_required_input_fails_input_contract_without_complete_evidence_set -q
```

Expected: FAIL because the CLI does not accept `--historical-canonical-csv` or `--runtime-metadata-dir`, or because evidence files are not written before raising.

- [ ] **Step 3: Split pure builders from writers and gate evaluation**

Rename the current `service_reconciliation_rows()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:1443-1553` to `build_service_reconciliation_rows()` and remove the final raise. Add:

```python
def evaluate_service_reconciliation_rows(rows: list[dict[str, object]]) -> int:
    return sum(
        1
        for row in rows
        if any(row[field] != "pass" for field in SERVICE_STATUS_FIELDS)
    )
```

Replace `write_csv_rows()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:1789-1795` with atomic writers:

```python
def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_csv_rows(path, fieldnames, rows) -> None:
    atomic_write_text(Path(path), csv_text(rows, fieldnames))


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write_text(Path(path), json.dumps(payload, sort_keys=True) + "\n")


def write_csv_rows(path, fieldnames, rows) -> None:
    atomic_write_csv_rows(path, fieldnames, rows)
```

Add summary and error helpers:

```python
def build_reconciliation_summary_payload(
    *,
    same_pass_rows,
    historical_rows,
    service_rows,
    mapping_payload,
) -> dict[str, object]:
    same_pass_failures = sum(row["status"] != "pass" for row in same_pass_rows)
    identity_failures = sum(row["classification"] == "historical_identity_mismatch" for row in historical_rows)
    service_failures = evaluate_service_reconciliation_rows(service_rows)
    mapping_failures = 0 if mapping_payload.get("status") == "ok" else 1
    failure_categories = []
    if identity_failures:
        failure_categories.append("historical_identity_mismatch")
    if same_pass_failures:
        failure_categories.append("same_pass_metric_mismatch")
    if service_failures:
        failure_categories.append("service_reconciliation_mismatch")
    if mapping_failures:
        failure_categories.append("mapping_validation_mismatch")
    hard_failure_count = identity_failures + same_pass_failures + service_failures + mapping_failures
    return {
        "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
        "status": "ok" if hard_failure_count == 0 else "failed",
        "hard_gate_status": "pass" if hard_failure_count == 0 else "fail",
        "historical_identity_status": "pass" if identity_failures == 0 else "fail",
        "same_pass_metric_status": "pass" if same_pass_failures == 0 else "fail",
        "service_reconciliation_status": "pass" if service_failures == 0 else "fail",
        "mapping_validation_status": "pass" if mapping_failures == 0 else "fail",
        "historical_drift_audit_status": "written",
        "evidence_write_status": "complete",
        "failure_categories": failure_categories,
        "hard_failure_count": hard_failure_count,
        "historical_identity_failure_count": identity_failures,
        "same_pass_metric_failure_count": same_pass_failures,
        "service_reconciliation_failure_count": service_failures,
        "mapping_validation_failure_count": mapping_failures,
        "same_pass_canonical_rows": EVAL_EPISODES + 1,
        "same_pass_reconciliation_rows": len(same_pass_rows),
        "historical_drift_rows": len(historical_rows),
        "historical_float_drift_count": sum(row["classification"] == "historical_float_drift" for row in historical_rows),
        "historical_saturation_count_drift_count": sum(row["classification"] == "historical_saturation_count_drift" for row in historical_rows),
        "service_reconciliation_rows": len(service_rows),
        "mapping_validation_present": True,
        "source_labels": {
            "historical_source_label": "formal_job_58513929_canonical_eval30",
            "stage_d_source_label": "stage_d_same_pass_canonical_eval30",
        },
    }


def raise_reconciliation_contract_error(failure_categories: list[str]) -> None:
    if failure_categories:
        raise ValueError(
            "Stage D reconciliation contract failed: "
            + ",".join(failure_categories)
        )
```

Patch `prepare_seed_validation_files()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:1619-1778` to:

1. Accept parameters `historical_canonical_csv`, `runtime_metadata_dir`, `formal_validation_json=None`, `stage_d_source_commit_sha=""`, `config_path=None`, `checkpoint_prefix=None`.
2. Wrap `load_seed_reconciliation_inputs()` in `try/except ValueError as exc` and raise `ValueError(f"input_contract_mismatch: {exc}")` before writing complete evidence.
3. After inputs are readable, build in this order: `same_pass_rows`, `historical_rows`, `service_rows`, `mapping_payload`, `summary_payload`.
4. Write atomically in this order: `validation/same_pass_canonical_reconciliation.csv`, `validation/historical_canonical_drift.csv`, `validation/service_reconciliation.csv`, `validation/mapping_validation.json`, `runtime_metadata/reconciliation_summary.json`.
5. Call `raise_reconciliation_contract_error(summary_payload["failure_categories"])`.
6. Return counts and `reconciliation_contract_version`.

Patch CLI at `scripts/validate_full_infrastructure_diagnostic_eval30.py:2788-2797`:

```python
    seed_validation_parser.add_argument("--historical-canonical-csv", required=True, type=Path)
    seed_validation_parser.add_argument("--canonical-csv", dest="historical_canonical_csv", type=Path)
    seed_validation_parser.add_argument("--runtime-metadata-dir", required=True, type=Path)
    seed_validation_parser.add_argument("--formal-validation-json", type=Path)
    seed_validation_parser.add_argument("--stage-d-source-commit-sha", required=True)
    seed_validation_parser.add_argument("--config-path", type=Path)
    seed_validation_parser.add_argument("--checkpoint-prefix", type=Path)
```

Keep `--canonical-csv` as a backward-compatible alias during the implementation commit, but all Slurm and tests must use `--historical-canonical-csv`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_reconciliation_evidence_remains_present_after_hard_gate_failure tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_missing_required_input_fails_input_contract_without_complete_evidence_set -q
```

Expected: PASS, `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Write reconciliation evidence before hard gate failures"
```

## Task 6: Seed Output And Task Package Contract Migration

**Files:**
- Modify: `scripts/validate_full_infrastructure_diagnostic_eval30.py:437-584`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:605-614`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:760-781`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:880-897`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:927-989`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1798-1951`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:1989-2035`
- Test: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:333-530`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:577-735`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:811-1154`

**Interfaces:**
- Consumes: new per-seed files `diagnostics/same_pass_canonical_eval30.csv`, `validation/same_pass_canonical_reconciliation.csv`, `validation/historical_canonical_drift.csv`, and `runtime_metadata/reconciliation_summary.json`.
- Produces: migrated `SEED_PACKAGE_FILES`, task metadata with `reconciliation_contract_version=2`, task validation metadata with `reconciliation_contract_version=2`, and seed-output validation requiring all new evidence.

- [ ] **Step 1: Write failing seed/package contract tests**

Patch `build_seed_output()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:333-530` to create `runtime_metadata` and a passing summary:

```python
    runtime_metadata = root / "runtime_metadata"
    runtime_metadata.mkdir()
    write_same_pass_canonical_eval30(diagnostics / "same_pass_canonical_eval30.csv", scale=scale, algorithm=algorithm, training_seed=training_seed)
    write_csv(
        validation / "same_pass_canonical_reconciliation.csv",
        [
            "reconciliation_contract_version",
            "episode_index",
            "field",
            "comparison_type",
            "same_pass_canonical_value",
            "diagnostic_value",
            "absolute_difference",
            "relative_difference",
            "same_pass_count",
            "diagnostic_count",
            "total_action_decision_denominator",
            "status",
            "failure_category",
        ],
        [
            {
                "reconciliation_contract_version": "2",
                "episode_index": str(episode_index),
                "field": "episode_reward",
                "comparison_type": "float_exact",
                "same_pass_canonical_value": "-1.0",
                "diagnostic_value": "-1.0",
                "absolute_difference": "0.0",
                "relative_difference": "0.0",
                "same_pass_count": "",
                "diagnostic_count": "",
                "total_action_decision_denominator": "",
                "status": "pass",
                "failure_category": "",
            }
            for episode_index in range(30)
        ],
    )
    write_csv(
        validation / "historical_canonical_drift.csv",
        [
            "reconciliation_contract_version",
            "scale",
            "algorithm",
            "training_seed",
            "formal_task_id",
            "episode_index",
            "episode_seed",
            "field",
            "historical_source_label",
            "stage_d_source_label",
            "historical_value",
            "stage_d_same_pass_value",
            "absolute_difference",
            "relative_difference",
            "classification",
            "historical_at_max_count",
            "stage_d_at_max_count",
            "total_action_decision_denominator",
            "count_difference",
            "fraction_difference",
        ],
        [
            {
                "reconciliation_contract_version": "2",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": str(training_seed),
                "formal_task_id": str(formal_task_id(task_id, training_seed)),
                "episode_index": str(episode_index),
                "episode_seed": str(episode_seed(scale, training_seed, episode_index)),
                "field": "episode_reward",
                "historical_source_label": "formal_job_58513929_canonical_eval30",
                "stage_d_source_label": "stage_d_same_pass_canonical_eval30",
                "historical_value": "-1.0",
                "stage_d_same_pass_value": "-1.0",
                "absolute_difference": "0.0",
                "relative_difference": "0.0",
                "classification": "exact_match",
                "historical_at_max_count": "",
                "stage_d_at_max_count": "",
                "total_action_decision_denominator": "",
                "count_difference": "",
                "fraction_difference": "",
            }
            for episode_index in range(30)
        ],
    )
    (runtime_metadata / "reconciliation_summary.json").write_text(
        json.dumps(
            {
                "reconciliation_contract_version": 2,
                "status": "ok",
                "hard_gate_status": "pass",
                "historical_identity_status": "pass",
                "same_pass_metric_status": "pass",
                "service_reconciliation_status": "pass",
                "mapping_validation_status": "pass",
                "historical_drift_audit_status": "written",
                "evidence_write_status": "complete",
                "failure_categories": [],
                "hard_failure_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
```

Add these tests after `test_validate_seed_output_rejects_failed_mapping_validation()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:691-699`:

```python
@pytest.mark.parametrize(
    "relative_path",
    [
        "diagnostics/same_pass_canonical_eval30.csv",
        "validation/same_pass_canonical_reconciliation.csv",
        "validation/historical_canonical_drift.csv",
        "runtime_metadata/reconciliation_summary.json",
    ],
)
def test_validate_seed_output_requires_new_reconciliation_evidence(tmp_path, relative_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / relative_path).unlink()

    with pytest.raises(ValueError, match="reconciliation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_reconciliation_contract_version_mismatch(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    summary_path = seed_dir / "runtime_metadata" / "reconciliation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["reconciliation_contract_version"] = 1
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="reconciliation contract"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)
```

Add this package membership test after `test_task3_rejects_missing_seed_group()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1000-1007`:

```python
def test_task_package_validation_requires_all_new_reconciliation_evidence(tmp_path):
    def mutate(root):
        (root / "seed3" / "validation" / "historical_canonical_drift.csv").unlink()

    package = build_task3_package(tmp_path, staging_mutator=mutate)

    with pytest.raises(ValueError, match="file set mismatch"):
        _task3_validate_package(package, task_id=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_validate_seed_output_requires_new_reconciliation_evidence tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_validate_seed_output_rejects_reconciliation_contract_version_mismatch tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task_package_validation_requires_all_new_reconciliation_evidence -q
```

Expected: FAIL because `validate_seed_output_directory()` and `SEED_PACKAGE_FILES` do not require all new evidence.

- [ ] **Step 3: Implement seed and package migration**

Patch `SEED_PACKAGE_FILES` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:605-614`:

```python
SEED_PACKAGE_FILES: Final[tuple[str, ...]] = (
    "diagnostics/episode_diagnostics.csv",
    "diagnostics/seed_summary_diagnostics.csv",
    "diagnostics/transformer_diagnostics.csv",
    "diagnostics/charger_diagnostics.csv",
    "diagnostics/same_pass_canonical_eval30.csv",
    "validation/same_pass_canonical_reconciliation.csv",
    "validation/historical_canonical_drift.csv",
    "validation/service_reconciliation.csv",
    "validation/mapping_validation.json",
    "runtime_metadata/reconciliation_summary.json",
    "logs/stderr.log",
)
```

Patch `validate_seed_output_directory()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:437-584`:

```python
    runtime_metadata = root / "runtime_metadata"
    same_pass_fields, same_pass_rows = _read_csv(diagnostics / SAME_PASS_CANONICAL_FILENAME)
    _require_columns(
        same_pass_fields,
        {"row_type", "episode_index", "episode_seed", "same_pass_at_max_count", "total_action_decision_denominator"},
        "same-pass canonical eval30",
    )
    same_pass_episode_count = sum(row.get("row_type") == "episode" for row in same_pass_rows)
    same_pass_summary_count = sum(row.get("row_type") == "summary" for row in same_pass_rows)
    if same_pass_episode_count != EVAL_EPISODES or same_pass_summary_count != 1:
        raise ValueError("same-pass canonical reconciliation evidence must contain 30 episodes and one summary")
    same_pass_reconciliation_rows = _validate_status_csv(
        validation / SAME_PASS_RECONCILIATION_FILENAME,
        label="same-pass canonical reconciliation",
        required_status_fields=("status",),
    )
    _read_csv(validation / HISTORICAL_DRIFT_FILENAME)
    reconciliation_summary = _read_json_object(
        runtime_metadata / RECONCILIATION_SUMMARY_FILENAME,
        "reconciliation summary",
    )
    if reconciliation_summary.get("reconciliation_contract_version") != RECONCILIATION_CONTRACT_VERSION:
        raise ValueError("reconciliation contract version mismatch")
    for field, expected in {
        "status": "ok",
        "hard_gate_status": "pass",
        "same_pass_metric_status": "pass",
        "historical_identity_status": "pass",
        "evidence_write_status": "complete",
    }.items():
        if reconciliation_summary.get(field) != expected:
            raise ValueError(f"reconciliation summary {field} mismatch")
```

Return these additional fields:

```python
        "same_pass_canonical_rows": len(same_pass_rows),
        "same_pass_reconciliation_rows": same_pass_reconciliation_rows,
        "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
```

Patch `_validate_task_metadata()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:760-781` and `_write_task_package_metadata()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:1820-1829` to require and write:

```python
"reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
```

Patch `_validate_task_validation()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:880-897` and `validation_payload` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:1916-1922` to require and write:

```python
"reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
```

Patch `validate_stage_d_task_package()` return at `scripts/validate_full_infrastructure_diagnostic_eval30.py:979-989` to include:

```python
"reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_validate_seed_output_accepts_exact_seed_directory tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_validate_seed_output_requires_new_reconciliation_evidence tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_validate_seed_output_rejects_reconciliation_contract_version_mismatch tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task3_valid_five_seed_package_passes tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task_package_validation_requires_all_new_reconciliation_evidence -q
```

Expected: PASS, including updated assertions that task-package validation returns `reconciliation_contract_version == 2`.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Require reconciliation v2 seed evidence"
```

## Task 7: Reducer And Complete-Bundle Summary Migration

**Files:**
- Modify: `scripts/validate_full_infrastructure_diagnostic_eval30.py:2062-2083`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:2301-2343`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:2376-2467`, `scripts/validate_full_infrastructure_diagnostic_eval30.py:2470-2724`
- Test: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1880-1954`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2188-2281`

**Interfaces:**
- Consumes: 40 per-seed reconciliation summaries and per-seed same-pass/historical drift rows from all eight task packages.
- Produces: complete-bundle summaries `summaries/same_pass_canonical_reconciliation_summary.csv`, `summaries/historical_canonical_drift_summary.csv`, and `summaries/reconciliation_summary_inventory.csv`, and complete workflow validation metadata with `reconciliation_contract_version=2`.

- [ ] **Step 1: Write failing reducer summary tests**

Add this test after `test_task5_complete_workflow_accepts_synthetic_8_task_packages()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2212-2224`:

```python
def test_reducer_emits_all_three_reconciliation_summaries(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    result = run_task5_reducer(fixture)
    payload = json.loads(result.stdout)
    bundle = Path(payload["bundle_path"])

    with tarfile.open(bundle, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}

    assert "summaries/same_pass_canonical_reconciliation_summary.csv" in names
    assert "summaries/historical_canonical_drift_summary.csv" in names
    assert "summaries/reconciliation_summary_inventory.csv" in names

    extract_dir = tmp_path / "bundle_extract"
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(extract_dir)
    _, same_pass_rows = read_csv_rows(extract_dir / "summaries" / "same_pass_canonical_reconciliation_summary.csv")
    _, drift_rows = read_csv_rows(extract_dir / "summaries" / "historical_canonical_drift_summary.csv")
    _, inventory_rows = read_csv_rows(extract_dir / "summaries" / "reconciliation_summary_inventory.csv")

    assert len(inventory_rows) == 40
    assert {row["reconciliation_contract_version"] for row in inventory_rows} == {"2"}
    assert same_pass_rows
    assert drift_rows
```

Add this validation test:

```python
def test_complete_bundle_validation_requires_reconciliation_summaries(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    result = run_task5_reducer(fixture)
    bundle = Path(json.loads(result.stdout)["bundle_path"])
    broken = tmp_path / "broken_bundle.tar.gz"
    with tarfile.open(bundle, "r:gz") as source, tarfile.open(broken, "w:gz") as target:
        for member in source.getmembers():
            if member.name == "summaries/historical_canonical_drift_summary.csv":
                continue
            payload = source.extractfile(member).read() if member.isfile() else None
            target.addfile(member, None if payload is None else io.BytesIO(payload))

    result = run_full_validator("validate-complete-bundle", "--bundle", broken, check=False)

    assert result.returncode != 0
    assert "historical_canonical_drift_summary" in result.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_reducer_emits_all_three_reconciliation_summaries tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_complete_bundle_validation_requires_reconciliation_summaries -q
```

Expected: FAIL because the complete bundle still emits `canonical_reconciliation_summary.csv` and does not require the three reconciliation-v2 summaries.

- [ ] **Step 3: Implement reducer migration**

Patch `COMPLETE_REQUIRED_FILES` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:2062-2083`:

```python
    "summaries/same_pass_canonical_reconciliation_summary.csv",
    "summaries/historical_canonical_drift_summary.csv",
    "summaries/reconciliation_summary_inventory.csv",
```

Remove the old required `summaries/canonical_reconciliation_summary.csv` entry.

Patch `_read_task_package_details()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:2301-2343` to collect:

```python
    same_pass_rows: list[dict[str, str]] = []
    historical_drift_rows: list[dict[str, str]] = []
    reconciliation_summary_rows: list[dict[str, object]] = []
```

Inside the seed loop, read:

```python
        _, same_pass = _read_csv(seed_root / "validation" / SAME_PASS_RECONCILIATION_FILENAME)
        _, historical_drift = _read_csv(seed_root / "validation" / HISTORICAL_DRIFT_FILENAME)
        reconciliation_summary = _read_json_object(
            seed_root / "runtime_metadata" / RECONCILIATION_SUMMARY_FILENAME,
            "reconciliation summary",
        )
```

Enrich each same-pass and drift row with `task_id`, `scale`, `algorithm`, `training_seed`, and `formal_task_id`. Add one inventory row per seed:

```python
        reconciliation_summary_rows.append({
            "task_id": str(task_id),
            "scale": str(validation["scale"]),
            "algorithm": str(validation["algorithm"]),
            "training_seed": str(seed),
            "formal_task_id": str(formal_task_id(task_id, seed)),
            "reconciliation_contract_version": str(reconciliation_summary["reconciliation_contract_version"]),
            "status": str(reconciliation_summary["status"]),
            "hard_gate_status": str(reconciliation_summary["hard_gate_status"]),
            "historical_identity_status": str(reconciliation_summary["historical_identity_status"]),
            "same_pass_metric_status": str(reconciliation_summary["same_pass_metric_status"]),
            "historical_float_drift_count": str(reconciliation_summary.get("historical_float_drift_count", 0)),
            "historical_saturation_count_drift_count": str(reconciliation_summary.get("historical_saturation_count_drift_count", 0)),
        })
```

Return keys `same_pass_rows`, `historical_drift_rows`, and `reconciliation_summary_rows`.

Patch `reduce_complete_workflow()` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:2529-2565` to aggregate these lists and write:

```python
    same_pass_fields = sorted({field for row in same_pass_rows for field in row})
    historical_drift_fields = sorted({field for row in historical_drift_rows for field in row})
    reconciliation_inventory_fields = [
        "task_id",
        "scale",
        "algorithm",
        "training_seed",
        "formal_task_id",
        "reconciliation_contract_version",
        "status",
        "hard_gate_status",
        "historical_identity_status",
        "same_pass_metric_status",
        "historical_float_drift_count",
        "historical_saturation_count_drift_count",
    ]
```

Write the three required summary CSVs under `summaries/`. Keep `summaries/service_reconciliation_summary.csv`.

Patch `validation_payload` at `scripts/validate_full_infrastructure_diagnostic_eval30.py:2689-2697` and expected validation at `scripts/validate_full_infrastructure_diagnostic_eval30.py:2431-2438` to use:

```python
"reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
```

Patch `validate_complete_bundle_file()` to read `reconciliation_summary_inventory.csv`, require exactly 40 rows, require every row has version `"2"`, `status == "ok"`, and `hard_gate_status == "pass"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task5_complete_workflow_accepts_synthetic_8_task_packages tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_reducer_emits_all_three_reconciliation_summaries tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_complete_bundle_validation_requires_reconciliation_summaries -q
```

Expected: PASS, `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Add reconciliation summaries to complete bundle"
```

## Task 8: Slurm Runner, Source Bundle, And Submit Helper Integration

**Files:**
- Modify: `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:126-158`, `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:242-313`, `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:315-331`
- Modify: `m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm:65-84`
- Modify: `m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh:17-34`, `m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh:115-128`
- Modify: `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:24-43`, `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:240-286`, `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:290-310`
- Test: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1154-1283`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1284-1567`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2188-2210`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2283-2411`

**Interfaces:**
- Consumes: migrated validator CLI `prepare-seed-validation --historical-canonical-csv --runtime-metadata-dir --formal-validation-json --stage-d-source-commit-sha --config-path --checkpoint-prefix`.
- Produces: dry-run output documenting `same_pass_canonical_eval30.csv`, `same_pass_canonical_reconciliation.csv`, `historical_canonical_drift.csv`, `reconciliation_summary.json`, and `reconciliation_contract_version=2`; real runner passes new validator arguments without changing evaluator workload.

- [ ] **Step 1: Write failing Slurm and helper dry-run tests**

Add to `test_task4_runner_dry_run_uses_formal_eval30_contract()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1240-1260`:

```python
        assert "same_pass_canonical_eval30.csv" in result.stdout
        assert "reconciliation_contract_version=2" in result.stdout
```

Add this test after `test_task4_runner_real_mode_orchestrates_synthetic_seed_evaluations()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1284-1567`:

```python
def test_task4_runner_real_mode_packages_reconciliation_v2_evidence(tmp_path):
    source_root = tmp_path / "source"
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "utils").mkdir()
    shutil.copy2(VALIDATOR, source_root / "scripts" / VALIDATOR.name)
    shutil.copy2(PROJECT_ROOT / "utils" / "infrastructure_diagnostics.py", source_root / "utils" / "infrastructure_diagnostics.py")
    source_commit = "a" * 40
    (source_root / "SOURCE_COMMIT_SHA.txt").write_text(source_commit + "\n", encoding="utf-8")
    formal_root = tmp_path / "formal"
    for seed in TRAINING_SEEDS:
        create_full_formal_package(formal_root, task_id=0, training_seed=seed)
    evaluator_stub = source_root / "stub_evaluator.py"
    evaluator_stub.write_text((PROJECT_ROOT / "tests" / "fixtures" / "stage_d_same_pass_evaluator_stub.py").read_text(encoding="utf-8"), encoding="utf-8")
    env = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_ARRAY_JOB_ID": "424243",
        "SLURM_JOB_ID": "424243_0",
        "EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT": str(source_root),
        "EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT": str(tmp_path / "runs"),
        "EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT": str(formal_root),
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE": str(tmp_path / "missing.tar.gz"),
        "EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT": source_commit,
        "EV_GNN_FULL_DIAGNOSTIC_EVALUATOR_SCRIPT": str(evaluator_stub),
    }
    result = subprocess.run(["bash", str(TASK4_RUNNER)], cwd=PROJECT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stderr + "\n" + result.stdout
    package = tmp_path / "outputs" / task5_package_basename(0, "424243")
    with tarfile.open(package, "r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
    assert "seed0/diagnostics/same_pass_canonical_eval30.csv" in names
    assert "seed0/validation/same_pass_canonical_reconciliation.csv" in names
    assert "seed0/validation/historical_canonical_drift.csv" in names
    assert "seed0/runtime_metadata/reconciliation_summary.json" in names
    assert all("checkpoint/" not in name and "model.best" not in name for name in names)
```

Create exact file `tests/fixtures/stage_d_same_pass_evaluator_stub.py` in this task. Start from the existing evaluator stub at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:1304-1529`, and add local CSV-writing code that writes `same_pass_canonical_eval30.csv` with 30 episode rows, one summary row, `same_pass_at_max_count`, `mapped_action_dimension`, and `total_action_decision_denominator`.

Add to `test_task5_reducer_dry_run_prints_exact_counts_and_scoped_sacct()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2188-2210`:

```python
    assert "reconciliation_contract_version=2" in result.stdout
    assert "same_pass_canonical_reconciliation_summary.csv" in result.stdout
    assert "historical_canonical_drift_summary.csv" in result.stdout
    assert "reconciliation_summary_inventory.csv" in result.stdout
```

Add to `test_task6_source_bundle_dry_run_lists_full_eval30_runtime_files()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2283-2303`:

```python
    assert "reconciliation_contract_version=2" in result.stdout
```

Add to `test_task6_submit_default_dry_run_does_not_invoke_sbatch()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2305-2411`:

```python
    assert "reconciliation_contract_version=2" in result.stdout
    assert "DRY_RUN_NO_SBATCH_CALLED" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task4_runner_dry_run_uses_formal_eval30_contract tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task4_runner_real_mode_packages_reconciliation_v2_evidence tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task5_reducer_dry_run_prints_exact_counts_and_scoped_sacct tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task6_source_bundle_dry_run_lists_full_eval30_runtime_files tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task6_submit_default_dry_run_does_not_invoke_sbatch -q
```

Expected: FAIL because dry-runs omit reconciliation-v2 markers and the runner still calls `prepare-seed-validation --canonical-csv` without runtime metadata or source provenance.

- [ ] **Step 3: Implement runner and helper migration without scientific workload changes**

Patch `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:126-158` dry-run to print:

```bash
echo "same_pass_canonical_eval30=diagnostics/same_pass_canonical_eval30.csv"
echo "same_pass_reconciliation=validation/same_pass_canonical_reconciliation.csv"
echo "historical_drift=validation/historical_canonical_drift.csv"
echo "reconciliation_summary=runtime_metadata/reconciliation_summary.json"
echo "reconciliation_contract_version=2"
```

Patch the real validator call at `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:306-313`:

```bash
  python "${VALIDATOR}" \
    prepare-seed-validation \
    --diagnostic-dir "${diagnostic_dir}" \
    --historical-canonical-csv "${canonical_dir}/complete_eval30.csv" \
    --validation-dir "${validation_dir}" \
    --runtime-metadata-dir "${runtime_dir}" \
    --formal-validation-json "${runtime_dir}/formal_package_validation.json" \
    --stage-d-source-commit-sha "${SOURCE_COMMIT_SHA}" \
    --config-path "${config_dir}/formal_config.yaml" \
    --checkpoint-prefix "${checkpoint_prefix}" \
    --task-id "${TASK_ID}" \
    --training-seed "${training_seed}" \
    > "${runtime_dir}/seed_validation_preparation.json"
```

Do not change the evaluator command at `m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm:275-291` except for comments or dry-run text. It must continue to use `--eval_episodes 30`, `--max_episode_steps "${max_episode_steps}"`, `--deterministic true`, `--eval_expl_noise 0.0`, existing seed offsets, existing task IDs, and `model.best`.

Patch reducer dry-run at `m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm:65-84` to print:

```bash
echo "reconciliation_contract_version=2"
echo "same_pass_summary=summaries/same_pass_canonical_reconciliation_summary.csv"
echo "historical_drift_summary=summaries/historical_canonical_drift_summary.csv"
echo "reconciliation_inventory=summaries/reconciliation_summary_inventory.csv"
```

Patch source bundle dry-run at `m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh:115-128` to print `reconciliation_contract_version=2`. Keep allowlisted source files unchanged unless the new test fixture is intentionally included in source-bundle tests; do not include test fixtures in production source bundles.

Patch submit helper dry-run at `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:290-310` to print `reconciliation_contract_version=2`. Patch `run_preflight()` at `m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh:240-286` to grep for the dry-run reconciliation markers in all eight runner outputs and the reducer dry-run output.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task4_runner_dry_run_uses_formal_eval30_contract tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task4_runner_real_mode_packages_reconciliation_v2_evidence tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task5_reducer_dry_run_prints_exact_counts_and_scoped_sacct tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task6_source_bundle_dry_run_lists_full_eval30_runtime_files tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task6_submit_default_dry_run_does_not_invoke_sbatch -q
```

Expected: PASS. No `sbatch` is invoked; the submit-helper test still observes `DRY_RUN_NO_SBATCH_CALLED`.

- [ ] **Step 5: Commit**

```bash
git add m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh tests/test_full_infrastructure_diagnostic_eval30_workflow.py tests/fixtures/stage_d_same_pass_evaluator_stub.py
git commit -m "Wire reconciliation v2 into Stage D runners"
```

## Task 9: Schema, Mapping, Package Safety, And Synthetic End-To-End Regression

**Files:**
- Modify: `tests/test_infrastructure_diagnostics.py:1926-1968`
- Modify: `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:28-150`, `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2188-2281`
- Modify: `scripts/validate_full_infrastructure_diagnostic_eval30.py:2470-2724`

**Interfaces:**
- Consumes: all prior implementation tasks.
- Produces: regression tests proving schema-v3 headers are byte-for-byte unchanged, formal task mapping remains unchanged, seed schedule remains unchanged, no checkpoints leak into final packages, no local test invokes M3 submission, and synthetic package/reducer validation succeeds.

- [ ] **Step 1: Write the failing final regression tests**

Add this test after `test_schema_v3_keeps_existing_schema_v2_action_and_service_columns()` at `tests/test_infrastructure_diagnostics.py:1951-1968`:

```python
def test_schema_v3_headers_remain_byte_for_byte_unchanged():
    import hashlib
    from utils.infrastructure_diagnostics import (
        CHARGER_DIAGNOSTIC_COLUMNS,
        EPISODE_DIAGNOSTIC_COLUMNS,
        SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
        TRANSFORMER_DIAGNOSTIC_COLUMNS,
    )

    def digest(columns):
        return hashlib.sha256(("\n".join(columns) + "\n").encode("utf-8")).hexdigest()

    assert len(EPISODE_DIAGNOSTIC_COLUMNS) == 68
    assert digest(EPISODE_DIAGNOSTIC_COLUMNS) == "13f1276e0e8ad9a6f69c68aadb922c726f0b8024a6dd3471780ea17e952dd978"
    assert len(CHARGER_DIAGNOSTIC_COLUMNS) == 38
    assert digest(CHARGER_DIAGNOSTIC_COLUMNS) == "88f2045630d8d534fa71012fccbe87ce3887b439b4ff1a052db4ae49c85c0677"
    assert len(TRANSFORMER_DIAGNOSTIC_COLUMNS) == 44
    assert digest(TRANSFORMER_DIAGNOSTIC_COLUMNS) == "58298be54e47444cc9397c6621553a6cb3907a062eb4081022a7da10bbab474e"
    assert len(SEED_SUMMARY_DIAGNOSTIC_COLUMNS) == 53
    assert digest(SEED_SUMMARY_DIAGNOSTIC_COLUMNS) == "6b481917df764e802dc4163124942dd707d828c943745f0b49c352c1718f51c4"
```

Add these tests after `test_stage_d_matches_actual_formal_job_seed_schedule()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:62-67`:

```python
def test_formal_task_mapping_remains_unchanged_for_all_8_tasks():
    assert {
        task_id: (
            stage_d_task(task_id)["scale"],
            stage_d_task(task_id)["algorithm"],
            tuple(formal_task_id(task_id, seed) for seed in TRAINING_SEEDS),
        )
        for task_id in range(8)
    } == {
        0: ("25cp", "actiongnn", (0, 1, 2, 3, 4)),
        1: ("25cp", "hierarchical", (5, 6, 7, 8, 9)),
        2: ("100cp", "actiongnn", (10, 11, 12, 13, 14)),
        3: ("100cp", "hierarchical", (15, 16, 17, 18, 19)),
        4: ("500cp", "actiongnn", (20, 21, 22, 23, 24)),
        5: ("500cp", "hierarchical", (25, 26, 27, 28, 29)),
        6: ("1000cp", "actiongnn", (30, 31, 32, 33, 34)),
        7: ("1000cp", "hierarchical", (35, 36, 37, 38, 39)),
    }


def test_seed_schedule_remains_unchanged_for_all_scales_and_training_seeds():
    expected_offsets = {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}
    for scale, base_offset in expected_offsets.items():
        for training_seed in TRAINING_SEEDS:
            assert evaluator_seed_offset(scale, training_seed) == base_offset + (999 * training_seed)
            assert episode_seed(scale, training_seed, 0) == base_offset + (1000 * training_seed)
            assert episode_seed(scale, training_seed, 29) == base_offset + (1000 * training_seed) + 29
```

Add this test after `test_task5_complete_workflow_rejects_nonempty_reducer_stderr()` at `tests/test_full_infrastructure_diagnostic_eval30_workflow.py:2275-2281`:

```python
def test_clean_synthetic_end_to_end_package_and_reducer_validation_succeeds_without_m3(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    result = run_task5_reducer(fixture)
    payload = json.loads(result.stdout)

    assert result.returncode == 0
    assert payload["status"] == "ok"
    assert payload["task_package_count"] == 8
    assert payload["checkpoint_group_count"] == 40
    assert payload["episode_count"] == 1200
    assert payload["formal_job_id"] == "58513929"
    assert payload["schema_version"] == "3"
    assert payload["reconciliation_contract_version"] == 2


def test_complete_bundle_contains_no_checkpoints_or_failed_job_58656380_reuse(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    result = run_task5_reducer(fixture)
    bundle = Path(json.loads(result.stdout)["bundle_path"])
    with tarfile.open(bundle, "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        payloads = {
            member.name: archive.extractfile(member).read().decode("utf-8", errors="replace")
            for member in archive.getmembers()
            if member.isfile() and member.name.endswith((".json", ".csv", ".txt", ".env"))
        }

    assert all("model.best" not in name and "model.last" not in name for name in names)
    assert all("checkpoint/" not in name for name in names)
    assert "58656380" not in "\n".join(payloads.values())
```

- [ ] **Step 2: Run the tests to verify the guard set fails where migration is incomplete**

Run:

```bash
python -m pytest tests/test_infrastructure_diagnostics.py::test_schema_v3_headers_remain_byte_for_byte_unchanged tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_formal_task_mapping_remains_unchanged_for_all_8_tasks tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_seed_schedule_remains_unchanged_for_all_scales_and_training_seeds tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_clean_synthetic_end_to_end_package_and_reducer_validation_succeeds_without_m3 tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_complete_bundle_contains_no_checkpoints_or_failed_job_58656380_reuse -q
```

Expected: FAIL on `payload["reconciliation_contract_version"]` in `test_clean_synthetic_end_to_end_package_and_reducer_validation_succeeds_without_m3` because the reducer return payload has not yet exposed the version, while the header, mapping, and seed-schedule tests pass.

- [ ] **Step 3: Implement only regression fixes that preserve scientific workload**

Patch only the reducer metadata return path introduced by Tasks 6-8:

```python
# scripts/validate_full_infrastructure_diagnostic_eval30.py reducer return payload
return {
    "status": "ok",
    "bundle_path": str(bundle_path),
    "checksum_path": str(sidecar_path),
    "task_package_count": 8,
    "checkpoint_group_count": 40,
    "episode_count": 1200,
    "formal_job_id": FORMAL_JOB_ID,
    "schema_version": SCHEMA_VERSION,
    "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
    **{f"validated_{key}": value for key, value in validation.items() if key != "status"},
}
```

Do not change `STAGE_D_TASKS`, `TRAINING_SEEDS`, `SCALE_SEED_OFFSETS`, `EVAL_EPISODES`, schema-v3 header constants, model paths, checkpoint members, evaluator episode counts, or Slurm resource directives.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_infrastructure_diagnostics.py::test_schema_v3_headers_remain_byte_for_byte_unchanged tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_formal_task_mapping_remains_unchanged_for_all_8_tasks tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_seed_schedule_remains_unchanged_for_all_scales_and_training_seeds tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_clean_synthetic_end_to_end_package_and_reducer_validation_succeeds_without_m3 tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_complete_bundle_contains_no_checkpoints_or_failed_job_58656380_reuse -q
```

Expected: PASS, `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_full_infrastructure_diagnostic_eval30.py tests/test_infrastructure_diagnostics.py tests/test_full_infrastructure_diagnostic_eval30_workflow.py
git commit -m "Harden Stage D reconciliation regression guards"
```

## Final Verification Plan

Do not run these commands during R2. They are the implementation-phase completion gate.

- [ ] Focused same-pass canonical evaluator tests:

```bash
python -m pytest tests/test_infrastructure_diagnostics.py::test_diagnostic_episode_returns_same_pass_canonical_record_from_one_mapped_action_stream tests/test_infrastructure_diagnostics.py::test_same_pass_canonical_eval30_csv_uses_canonical_required_columns_and_summary -q
```

Expected: PASS.

- [ ] Focused reconciliation validator tests:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py -k "same_pass or historical or saturation or reconciliation_evidence or input_contract" -q
```

Expected: PASS. Historical float drift is classified and does not fail same-pass hard gates. Same-pass action/reward/tracking mismatches fail with `same_pass_metric_mismatch`.

- [ ] All Stage D workflow tests:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py -q
```

Expected: PASS.

- [ ] Full repository tests in `evgnn_core`:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evgnn_core
python -m pytest tests -q -p no:cacheprovider
```

Expected: PASS. No local test invokes `sbatch`, live `sacct`, network access, retraining, or heavy ML evaluation.

- [ ] Python source compilation without writing bytecode:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile evaluate_td3_gnn.py evaluate_td3_gnn_infrastructure_diagnostics.py utils/infrastructure_diagnostics.py scripts/validate_full_infrastructure_diagnostic_eval30.py
```

Expected: PASS with no `.pyc` files written because `PYTHONDONTWRITEBYTECODE=1` is set.

- [ ] Bash syntax checks for affected shell/Slurm files:

```bash
bash -n m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm
bash -n m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm
bash -n m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh
bash -n m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh
```

Expected: PASS with no output.

- [ ] All eight runner dry-runs:

```bash
for task_id in 0 1 2 3 4 5 6 7; do EV_GNN_FULL_DIAGNOSTIC_DRY_RUN=1 SLURM_ARRAY_TASK_ID="${task_id}" SLURM_ARRAY_JOB_ID=123456 SLURM_JOB_ID="123456_${task_id}" bash m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm > "/tmp/stage_d_runner_task${task_id}_dry_run.out"; done
```

Expected: each output contains `FULL_INFRASTRUCTURE_DIAGNOSTIC_TASK_DRY_RUN`, exactly five `EVALUATOR_COMMAND_SEED_` lines, `expected_episode_count=150`, `same_pass_canonical_eval30.csv`, `reconciliation_contract_version=2`, and `DRY_RUN_NO_EVALUATION_OR_PACKAGING`.

- [ ] Reducer dry-run:

```bash
EV_GNN_FULL_DIAGNOSTIC_REDUCER_DRY_RUN=1 EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID=123456 SLURM_JOB_ID=789012 bash m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm
```

Expected: output contains `FULL_INFRASTRUCTURE_DIAGNOSTIC_REDUCER_DRY_RUN`, `task_package_count=8`, `checkpoint_group_count=40`, `episode_count=1200`, `reconciliation_contract_version=2`, the three reconciliation summary names, and `DRY_RUN_NO_REDUCTION_OR_PACKAGING`.

- [ ] Submission-helper dry-run:

```bash
bash m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh --dry-run
```

Expected: output contains `FULL_INFRASTRUCTURE_DIAGNOSTIC_SUBMIT_DRY_RUN`, `SBATCH_ARRAY_COMMAND=sbatch --parsable`, `SBATCH_REDUCER_COMMAND=sbatch --parsable --dependency=afterok:<array_job_id>`, `reconciliation_contract_version=2`, and `DRY_RUN_NO_SBATCH_CALLED`. No `sbatch` is invoked.

- [ ] Synthetic package and complete-bundle validation:

```bash
python -m pytest tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task3_valid_five_seed_package_passes tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_task5_complete_workflow_accepts_synthetic_8_task_packages tests/test_full_infrastructure_diagnostic_eval30_workflow.py::test_clean_synthetic_end_to_end_package_and_reducer_validation_succeeds_without_m3 -q
```

Expected: PASS. The complete bundle validates 8 task packages, 40 checkpoint groups, 1,200 episodes, `formal_job_id=58513929`, `schema_version=3`, and `reconciliation_contract_version=2`.

- [ ] Diff whitespace check:

```bash
git diff --check
```

Expected: PASS with no output.

- [ ] Clean worktree confirmation:

```bash
git status --porcelain=v1
```

Expected: no output after final implementation commits.

- [ ] M3 safety confirmation:

```bash
git log --oneline --max-count=1
```

Expected: shows the final local implementation commit. Confirm from shell history and test logs that no `sbatch`, real `sacct`, retraining, heavy evaluation, Git push, or pull request creation occurred during local verification.

## Implementation Self-Review Checklist

- [ ] Every approved design requirement maps to Tasks 1-9.
- [ ] All exact paths are present: `diagnostics/same_pass_canonical_eval30.csv`, `validation/same_pass_canonical_reconciliation.csv`, `validation/historical_canonical_drift.csv`, `runtime_metadata/reconciliation_summary.json`, `summaries/same_pass_canonical_reconciliation_summary.csv`, `summaries/historical_canonical_drift_summary.csv`, and `summaries/reconciliation_summary_inventory.csv`.
- [ ] Every new CSV/JSON carrying reconciliation semantics has `reconciliation_contract_version=2`.
- [ ] Diagnostic schema-v3 headers remain byte-for-byte unchanged.
- [ ] Historical floating drift and historical saturation count drift are audit-only.
- [ ] Historical identity mismatch remains a hard gate.
- [ ] Same-pass metric mismatch remains a hard gate.
- [ ] No task changes actor, critic, replay buffer, reward, simulator, state, action interface, checkpoints, training algorithm, seed schedule, task mapping, or the 30-episode protocol.
- [ ] No task introduces tolerance tuning to job `58656380`.
- [ ] No task reuses failed job `58656380` output or claims scientific Stage D evidence from that job.
- [ ] Every task has a test-first RED/GREEN cycle and ends with exactly one focused commit.
