import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    EVAL_EPISODES,
    TOPOLOGY,
    TRAINING_SEEDS,
    evaluator_seed_offset,
    episode_seed,
    formal_task_id,
    stage_d_task,
)
from utils.infrastructure_diagnostics import (
    CHARGER_DIAGNOSTIC_COLUMNS,
    EPISODE_DIAGNOSTIC_COLUMNS,
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_full_infrastructure_diagnostic_eval30.py"


@pytest.mark.parametrize(
    ("task_id", "scale", "algorithm", "formal_ids"),
    [
        (0, "25cp", "actiongnn", (0, 1, 2, 3, 4)),
        (1, "25cp", "hierarchical", (5, 6, 7, 8, 9)),
        (2, "100cp", "actiongnn", (10, 11, 12, 13, 14)),
        (3, "100cp", "hierarchical", (15, 16, 17, 18, 19)),
        (4, "500cp", "actiongnn", (20, 21, 22, 23, 24)),
        (5, "500cp", "hierarchical", (25, 26, 27, 28, 29)),
        (6, "1000cp", "actiongnn", (30, 31, 32, 33, 34)),
        (7, "1000cp", "hierarchical", (35, 36, 37, 38, 39)),
    ],
)
def test_stage_d_task_mapping(task_id, scale, algorithm, formal_ids):
    task = stage_d_task(task_id)
    assert task["scale"] == scale
    assert task["algorithm"] == algorithm
    assert tuple(formal_task_id(task_id, seed) for seed in TRAINING_SEEDS) == formal_ids


def test_stage_d_task_returns_defensive_copy():
    task = stage_d_task(0)
    task["scale"] = "mutated"
    assert stage_d_task(0)["scale"] == "25cp"


def test_stage_d_episode_contract():
    assert TRAINING_SEEDS == (0, 1, 2, 3, 4)
    assert EVAL_EPISODES == 30
    assert episode_seed("25cp", 0, 0) == 710000
    assert episode_seed("25cp", 4, 29) == 714029
    assert episode_seed("1000cp", 4, 29) == 744029


def test_stage_d_matches_actual_formal_job_seed_schedule():
    assert episode_seed("25cp", 0, 0) == 710000
    assert episode_seed("25cp", 1, 0) == 711000
    assert episode_seed("25cp", 4, 0) == 714000
    assert episode_seed("1000cp", 4, 0) == 744000


@pytest.mark.parametrize("task_id", [-1, 8, 99, True, "0"])
def test_stage_d_task_rejects_invalid_id(task_id):
    with pytest.raises(ValueError, match="task ID"):
        stage_d_task(task_id)


@pytest.mark.parametrize("training_seed", [-1, 5, True, "0"])
def test_formal_task_id_rejects_invalid_training_seed(training_seed):
    with pytest.raises(ValueError, match="training seed"):
        formal_task_id(0, training_seed)


@pytest.mark.parametrize(
    ("scale", "training_seed", "episode_index", "message"),
    [
        ("invalid", 0, 0, "scale"),
        ("25cp", 5, 0, "training seed"),
        ("25cp", 0, -1, "episode index"),
        ("25cp", 0, 30, "episode index"),
        ("25cp", 0, True, "episode index"),
    ],
)
def test_episode_seed_rejects_invalid_input(
    scale,
    training_seed,
    episode_index,
    message,
):
    with pytest.raises(ValueError, match=message):
        episode_seed(scale, training_seed, episode_index)


@pytest.mark.parametrize(
    ("task_id", "expected_lines"),
    [
        (
            0,
            [
                "task_id=0",
                "scale=25cp",
                "algorithm=actiongnn",
                "training_seeds=0,1,2,3,4",
                "formal_task_ids=0,1,2,3,4",
                "eval_episodes=30",
                "scale_base_seed_offset=710000",
                "eval_seed_offsets=710000,710999,711998,712997,713996",
                "first_episode_seeds=710000,711000,712000,713000,714000",
            ],
        ),
        (
            7,
            [
                "task_id=7",
                "scale=1000cp",
                "algorithm=hierarchical",
                "training_seeds=0,1,2,3,4",
                "formal_task_ids=35,36,37,38,39",
                "eval_episodes=30",
                "scale_base_seed_offset=740000",
                "eval_seed_offsets=740000,740999,741998,742997,743996",
                "first_episode_seeds=740000,741000,742000,743000,744000",
            ],
        ),
    ],
)
def test_task_mapping_cli(task_id, expected_lines):
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "task-mapping",
            "--task-id",
            str(task_id),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected_lines


import csv
import json

from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    expected_episode_keys,
    HISTORICAL_DRIFT_COLUMNS,
    HISTORICAL_FLOAT_FIELDS,
    HISTORICAL_IDENTITY_FIELDS,
    PROVENANCE_IDENTITY_FIELDS,
    SAME_PASS_FIELD_MAP,
    SAME_PASS_RECONCILIATION_COLUMNS,
    validate_episode_inventory,
    validate_seed_output_directory,
)


def complete_inventory_rows():
    return [
        {
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "episode_index": str(episode_index),
            "episode_seed": str(episode_seed(scale, seed, episode_index)),
            "diagnostic_schema_version": "3",
        }
        for scale in ["25cp", "100cp", "500cp", "1000cp"]
        for algorithm in ("actiongnn", "hierarchical")
        for seed in range(5)
        for episode_index in range(30)
    ]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


CANONICAL_EVAL30_COLUMNS = [
    "row_type",
    "algorithm",
    "seed",
    "episode_index",
    "episode_seed",
    "episode_steps",
    "done",
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_energy_charged",
    "total_energy_discharged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_transformer_overload",
    "total_ev_served",
    "action_mean",
    "action_fraction_at_max",
    "active_action_count_mean",
]


def write_canonical_eval30(path, *, scale="25cp", algorithm="actiongnn", training_seed=0, mutator=None):
    rows = []
    for episode_index in range(EVAL_EPISODES):
        diagnostic = valid_episode_row(
            episode_index,
            scale=scale,
            algorithm=algorithm,
            training_seed=training_seed,
        )
        rows.append(
            {
                "row_type": "episode",
                "algorithm": algorithm,
                "seed": str(training_seed),
                "episode_index": diagnostic["episode_index"],
                "episode_seed": diagnostic["episode_seed"],
                "episode_steps": diagnostic["episode_steps"],
                "done": diagnostic["done"],
                "episode_reward": diagnostic["episode_reward"],
                "tracking_error": diagnostic["tracking_error"],
                "energy_tracking_error": diagnostic["energy_tracking_error"],
                "power_tracker_violation": diagnostic["power_tracker_violation"],
                "total_energy_charged": diagnostic["total_energy_charged"],
                "total_energy_discharged": diagnostic["total_energy_discharged"],
                "average_user_satisfaction": diagnostic["average_user_satisfaction"],
                "energy_user_satisfaction": diagnostic["energy_user_satisfaction"],
                "total_transformer_overload": diagnostic["total_transformer_overload"],
                "total_ev_served": diagnostic["total_ev_served"],
                "action_mean": diagnostic.get("global_action_mean_all_slots", "0.0"),
                "action_fraction_at_max": diagnostic.get(
                    "global_action_fraction_at_max_all_slots",
                    "0.0",
                ),
                "active_action_count_mean": diagnostic.get(
                    "nonzero_action_count_mean_all_slots",
                    "0.0",
                ),
            }
        )
    if mutator is not None:
        mutator(rows)
    write_csv(path, CANONICAL_EVAL30_COLUMNS, rows)


def write_same_pass_canonical_eval30(path, *, scale="25cp", algorithm="actiongnn", training_seed=0, mutator=None):
    rows = []
    mapped_action_dimension = int(scale.replace("cp", ""))
    denominator = 112 * mapped_action_dimension
    for episode_index in range(EVAL_EPISODES):
        diagnostic = valid_episode_row(
            episode_index,
            scale=scale,
            algorithm=algorithm,
            training_seed=training_seed,
        )
        at_max_fraction = diagnostic.get("global_action_fraction_at_max_all_slots", "0.0")
        same_pass_at_max_count = int(round(float(at_max_fraction) * denominator))
        rows.append(
            {
                "row_type": "episode",
                "algorithm": algorithm,
                "seed": str(training_seed),
                "episode_index": diagnostic["episode_index"],
                "episode_seed": diagnostic["episode_seed"],
                "episode_steps": diagnostic["episode_steps"],
                "done": diagnostic["done"],
                "episode_reward": diagnostic["episode_reward"],
                "tracking_error": diagnostic["tracking_error"],
                "energy_tracking_error": diagnostic["energy_tracking_error"],
                "power_tracker_violation": diagnostic["power_tracker_violation"],
                "total_energy_charged": diagnostic["total_energy_charged"],
                "total_energy_discharged": diagnostic["total_energy_discharged"],
                "average_user_satisfaction": diagnostic["average_user_satisfaction"],
                "energy_user_satisfaction": diagnostic["energy_user_satisfaction"],
                "total_transformer_overload": diagnostic["total_transformer_overload"],
                "total_ev_served": diagnostic["total_ev_served"],
                "action_mean": diagnostic.get("global_action_mean_all_slots", "0.0"),
                "action_fraction_at_max": at_max_fraction,
                "active_action_count_mean": diagnostic.get(
                    "nonzero_action_count_mean_all_slots",
                    "0.0",
                ),
                "mapped_action_dimension": str(mapped_action_dimension),
                "same_pass_at_max_count": str(same_pass_at_max_count),
                "total_action_decision_denominator": str(denominator),
            }
        )
    rows.append({
        "row_type": "summary",
        "algorithm": algorithm,
        "seed": str(training_seed),
        "episode_index": "",
        "episode_seed": "",
        "episode_steps": "112",
        "done": "",
        "episode_reward": "-1.0",
        "tracking_error": "1.0",
        "energy_tracking_error": "1.0",
        "power_tracker_violation": "1.0",
        "total_energy_charged": "1.0",
        "total_energy_discharged": "0.0",
        "average_user_satisfaction": "1.0",
        "energy_user_satisfaction": "100.0",
        "total_transformer_overload": "0.0",
        "total_ev_served": "1",
        "action_mean": "0.5",
        "action_fraction_at_max": "0.5",
        "active_action_count_mean": "1.0",
        "mapped_action_dimension": str(mapped_action_dimension),
        "same_pass_at_max_count": "",
        "total_action_decision_denominator": "",
    })
    if mutator is not None:
        mutator(rows)
    write_csv(
        path,
        [
            *CANONICAL_EVAL30_COLUMNS,
            "mapped_action_dimension",
            "same_pass_at_max_count",
            "total_action_decision_denominator",
        ],
        rows,
    )


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def passing_same_pass_reconciliation_rows():
    rows = []
    for episode_index in range(EVAL_EPISODES):
        for canonical_field, _diagnostic_field, comparison_type in SAME_PASS_FIELD_MAP:
            is_fraction = comparison_type == "fraction_count_exact"
            rows.append({
                "reconciliation_contract_version": "2",
                "episode_index": str(episode_index),
                "field": canonical_field,
                "comparison_type": comparison_type,
                "same_pass_canonical_value": (
                    "0.5" if canonical_field.startswith("action") else "-1.0"
                ),
                "diagnostic_value": (
                    "0.5" if canonical_field.startswith("action") else "-1.0"
                ),
                "absolute_difference": "0.0",
                "relative_difference": "0.0",
                "same_pass_count": "1400" if is_fraction else "",
                "diagnostic_count": "1400" if is_fraction else "",
                "total_action_decision_denominator": "2800" if is_fraction else "",
                "status": "pass",
                "failure_category": "",
            })
    return rows


def passing_historical_drift_rows(*, scale, algorithm, task_id, training_seed):
    rows = []
    for field in PROVENANCE_IDENTITY_FIELDS:
        rows.append({
            "reconciliation_contract_version": "2",
            "scale": "",
            "algorithm": "",
            "training_seed": "",
            "formal_task_id": "",
            "episode_index": "",
            "episode_seed": "",
            "field": field,
            "historical_source_label": "formal_job_58513929_package_validation",
            "stage_d_source_label": "stage_d_runtime_metadata",
            "historical_value": "match",
            "stage_d_same_pass_value": "match",
            "absolute_difference": "",
            "relative_difference": "",
            "classification": "historical_identity_match",
            "historical_at_max_count": "",
            "stage_d_at_max_count": "",
            "total_action_decision_denominator": "",
            "count_difference": "",
            "fraction_difference": "",
        })
    for episode_index in range(EVAL_EPISODES):
        seed_value = str(episode_seed(scale, training_seed, episode_index))
        for field, _historical_field, _stage_d_field, _value_type in HISTORICAL_IDENTITY_FIELDS:
            rows.append({
                "reconciliation_contract_version": "2",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": str(training_seed),
                "formal_task_id": str(formal_task_id(task_id, training_seed)),
                "episode_index": str(episode_index),
                "episode_seed": seed_value,
                "field": field,
                "historical_source_label": "formal_job_58513929_canonical_eval30",
                "stage_d_source_label": "stage_d_same_pass_canonical_eval30",
                "historical_value": "match",
                "stage_d_same_pass_value": "match",
                "absolute_difference": "",
                "relative_difference": "",
                "classification": "historical_identity_match",
                "historical_at_max_count": "",
                "stage_d_at_max_count": "",
                "total_action_decision_denominator": "",
                "count_difference": "",
                "fraction_difference": "",
            })
        for historical_field, _stage_d_field in HISTORICAL_FLOAT_FIELDS:
            rows.append({
                "reconciliation_contract_version": "2",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": str(training_seed),
                "formal_task_id": str(formal_task_id(task_id, training_seed)),
                "episode_index": str(episode_index),
                "episode_seed": seed_value,
                "field": historical_field,
                "historical_source_label": "formal_job_58513929_canonical_eval30",
                "stage_d_source_label": "stage_d_same_pass_canonical_eval30",
                "historical_value": "0.0",
                "stage_d_same_pass_value": "0.0",
                "absolute_difference": "0.0",
                "relative_difference": "0.0",
                "classification": "exact_match",
                "historical_at_max_count": (
                    "1400" if historical_field == "action_fraction_at_max" else ""
                ),
                "stage_d_at_max_count": (
                    "1400" if historical_field == "action_fraction_at_max" else ""
                ),
                "total_action_decision_denominator": (
                    "2800" if historical_field == "action_fraction_at_max" else ""
                ),
                "count_difference": (
                    "0" if historical_field == "action_fraction_at_max" else ""
                ),
                "fraction_difference": (
                    "0.0" if historical_field == "action_fraction_at_max" else ""
                ),
            })
    return rows


def valid_episode_row(
    episode_index,
    *,
    scale="25cp",
    algorithm="actiongnn",
    training_seed=0,
):
    row = {column: "0.0" for column in EPISODE_DIAGNOSTIC_COLUMNS}
    row.update({
        "matrix_job_id": "99999999",
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": str(training_seed),
        "episode_index": str(episode_index),
        "episode_seed": str(episode_seed(scale, training_seed, episode_index)),
        "config": f"config_files/PublicPST_{scale}.yaml",
        "checkpoint_prefix": "/tmp/model.best",
        "run_name": f"synthetic_{scale}_{algorithm}_seed{training_seed}",
        "episode_steps": "112",
        "done": "True",
        "episode_reward": "-1.0",
        "max_action": "1.0",
        "max_action_tolerance": "1e-06",
        "environment_action_low": "-1.0",
        "environment_action_high": "1.0",
        "observed_action_min_active": "0.0",
        "observed_action_max_active": "1.0",
        "action_tolerance": "1e-06",
        "environment_action_domain_support": "signed",
        "v2g_enabled": "False",
        "v2g_enabled_source": "config",
        "global_action_fraction_at_max_all_slots": "0.5",
        "tracking_error": "1.0",
        "energy_tracking_error": "1.0",
        "power_tracker_violation": "1.0",
        "total_energy_charged": "1.0",
        "total_energy_discharged": "0.0",
        "average_user_satisfaction": "1.0",
        "energy_user_satisfaction": "100.0",
        "total_transformer_overload": "0.0",
        "total_ev_served": "1",
        "global_action_fraction_at_max_active": "0.5",
        "global_action_nonzero_fraction_active": "1.0",
        "active_action_decision_count": "1",
        "active_action_below_environment_low_count": "0",
        "active_action_below_environment_low_fraction": "0.0",
        "active_action_above_environment_high_count": "0",
        "active_action_above_environment_high_fraction": "0.0",
        "global_positive_action_fraction_active": "1.0",
        "global_zero_action_fraction_active": "0.0",
        "global_negative_action_fraction_active": "0.0",
        "global_action_fraction_at_positive_max_active": "0.5",
        "global_action_fraction_at_negative_min_active": "0.0",
        "global_positive_action_sum_active": "1.0",
        "global_negative_action_magnitude_sum_active": "0.0",
        "global_action_mean_all_slots": "0.5",
        "global_action_mean_active": "0.5",
        "global_action_sum_active": "1.0",
        "active_slot_count_mean": "1.0",
        "nonzero_action_count_mean_all_slots": "1.0",
        "inactive_slot_fraction_mean": "0.0",
        "inactive_slot_decision_count": "0",
        "inactive_nonzero_action_count": "0",
        "inactive_nonzero_action_fraction_all_slots": "0.0",
        "transformer_action_fraction_at_max_active_macro_mean": "0.5",
        "charger_action_fraction_at_max_active_macro_mean": "0.5",
        "transformer_action_nonzero_fraction_active_macro_mean": "1.0",
        "charger_action_nonzero_fraction_active_macro_mean": "1.0",
        "transformer_allocation_zero_pressure_step_fraction": "0.0",
        "transformer_allocation_valid_step_count": "1.0",
        "charger_allocation_zero_pressure_step_fraction": "0.0",
        "charger_allocation_valid_step_count": "1.0",
        "diagnostic_schema_version": "3",
    })
    return row


def build_seed_output(
    root,
    *,
    task_id=0,
    training_seed=0,
    source_commit_sha="a" * 40,
):
    mapping = {
        0: ("25cp", "actiongnn", 25, 3),
        1: ("25cp", "hierarchical", 25, 3),
        2: ("100cp", "actiongnn", 100, 7),
        3: ("100cp", "hierarchical", 100, 7),
        4: ("500cp", "actiongnn", 500, 35),
        5: ("500cp", "hierarchical", 500, 35),
        6: ("1000cp", "actiongnn", 1000, 70),
        7: ("1000cp", "hierarchical", 1000, 70),
    }
    scale, algorithm, charger_count, transformer_count = mapping[task_id]
    diagnostics = root / "diagnostics"
    validation = root / "validation"
    logs = root / "logs"
    runtime_metadata = root / "runtime_metadata"
    diagnostics.mkdir(parents=True)
    validation.mkdir()
    logs.mkdir()
    runtime_metadata.mkdir()

    episode_rows = [
        valid_episode_row(
            episode_index,
            scale=scale,
            algorithm=algorithm,
            training_seed=training_seed,
        )
        for episode_index in range(30)
    ]
    write_csv(
        diagnostics / "episode_diagnostics.csv",
        list(episode_rows[0]),
        episode_rows,
    )
    write_same_pass_canonical_eval30(
        diagnostics / "same_pass_canonical_eval30.csv",
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
    )

    seed_summary = {column: "0.0" for column in SEED_SUMMARY_DIAGNOSTIC_COLUMNS}
    seed_summary.update({
        "matrix_job_id": "99999999",
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": str(training_seed),
        "n_eval_episodes": "30",
        "global_action_fraction_at_max_active_mean": "0.5",
        "global_action_fraction_at_max_all_slots_mean": "0.5",
        "global_action_nonzero_fraction_active_mean": "1.0",
        "active_action_decision_count_mean": "1.0",
        "active_action_below_environment_low_count": "0",
        "active_action_below_environment_low_fraction": "0.0",
        "active_action_above_environment_high_count": "0",
        "active_action_above_environment_high_fraction": "0.0",
        "global_positive_action_fraction_active_mean": "1.0",
        "global_zero_action_fraction_active_mean": "0.0",
        "global_negative_action_fraction_active_mean": "0.0",
        "global_action_fraction_at_positive_max_active_mean": "0.5",
        "global_action_fraction_at_negative_min_active_mean": "0.0",
        "global_positive_action_sum_active_mean": "1.0",
        "global_negative_action_magnitude_sum_active_mean": "0.0",
        "observed_action_min_active_mean": "0.0",
        "observed_action_max_active_mean": "1.0",
        "environment_action_low": "-1.0",
        "environment_action_high": "1.0",
        "action_tolerance": "1e-06",
        "environment_action_domain_support": "signed",
        "v2g_enabled": "False",
        "v2g_enabled_source": "config",
        "inactive_slot_decision_count_mean": "0.0",
        "inactive_nonzero_action_count_mean": "0.0",
        "inactive_nonzero_action_fraction_all_slots_mean": "0.0",
        "transformer_action_fraction_at_max_active_macro_mean": "0.5",
        "charger_action_fraction_at_max_active_macro_mean": "0.5",
        "transformer_action_nonzero_fraction_active_macro_mean": "1.0",
        "charger_action_nonzero_fraction_active_macro_mean": "1.0",
        "transformer_allocation_zero_pressure_step_fraction_mean": "0.0",
        "transformer_allocation_valid_step_count_mean": "1.0",
        "charger_allocation_zero_pressure_step_fraction_mean": "0.0",
        "charger_allocation_valid_step_count_mean": "1.0",
        "power_tracker_violation_mean": "1.0",
        "tracking_error_mean": "1.0",
        "energy_tracking_error_mean": "1.0",
        "total_ev_served_mean": "1.0",
        "total_energy_charged_mean": "1.0",
        "average_user_satisfaction_mean": "1.0",
        "energy_user_satisfaction_mean": "100.0",
        "diagnostic_schema_version": "3",
    })
    write_csv(
        diagnostics / "seed_summary_diagnostics.csv",
        list(seed_summary),
        [seed_summary],
    )

    def infrastructure_rows(count, *, label):
        columns = (
            TRANSFORMER_DIAGNOSTIC_COLUMNS
            if label == "transformer"
            else CHARGER_DIAGNOSTIC_COLUMNS
        )
        rows = []
        for episode_index in range(30):
            for row_index in range(count):
                row = {column: "0.0" for column in columns}
                row.update({
                "matrix_job_id": "99999999",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": str(training_seed),
                "episode_index": str(episode_index),
                "episode_seed": str(episode_seed(scale, training_seed, episode_index)),
                f"{label}_id": str(row_index),
                "transformer_id": str(row_index % transformer_count),
                "n_chargers_total": str(charger_count),
                "n_active_chargers_seen": "1",
                "n_ports": "1",
                "n_active_ev_decisions": "1" if row_index == 0 else "0",
                "n_all_slot_decisions": "1",
                "action_sum_active": "1.0" if row_index == 0 else "0.0",
                "action_mean_active": "1.0" if row_index == 0 else "0.0",
                "action_max_active": "1.0" if row_index == 0 else "0.0",
                "action_fraction_at_max_active": "1.0" if row_index == 0 else "0.0",
                "action_nonzero_fraction_active": "1.0" if row_index == 0 else "0.0",
                "positive_action_fraction_active": "1.0" if row_index == 0 else "0.0",
                "zero_action_fraction_active": "0.0" if row_index == 0 else "1.0",
                "negative_action_fraction_active": "0.0",
                "action_fraction_at_positive_max_active": "1.0" if row_index == 0 else "0.0",
                "action_fraction_at_negative_min_active": "0.0",
                "positive_action_sum_active": "1.0" if row_index == 0 else "0.0",
                "negative_action_magnitude_sum_active": "0.0",
                "action_sum_all_slots": "1.0" if row_index == 0 else "0.0",
                "action_mean_all_slots": "1.0" if row_index == 0 else "0.0",
                "action_max_all_slots": "1.0" if row_index == 0 else "0.0",
                "action_fraction_at_max_all_slots": "1.0" if row_index == 0 else "0.0",
                "overload_magnitude_sum": "0.0",
                "overload_magnitude_max": "0.0",
                "overload_frequency_steps": "0",
                "overload_frequency_fraction": "0.0",
                "cs_power_sum_kwh": "1.0" if row_index == 0 else "0.0",
                "cs_power_mean_kw": "1.0" if row_index == 0 else "0.0",
                "cs_power_max_kw": "1.0" if row_index == 0 else "0.0",
                "served_ev_count": "1" if row_index == 0 else "0",
                "energy_charged_kwh": "1.0" if row_index == 0 else "0.0",
                "energy_discharged_kwh": "0.0",
                "user_satisfaction_sum": "1.0" if row_index == 0 else "0.0",
                "user_satisfaction_mean": "1.0" if row_index == 0 else "0.0",
                "user_satisfaction_mean_served_ev_weighted": "1.0" if row_index == 0 else "0.0",
                "user_satisfaction_observation_count": "1" if row_index == 0 else "0",
                "user_satisfaction_source": "synthetic",
                "diagnostic_schema_version": "3",
                })
                row = {column: row[column] for column in columns}
                rows.append(row)
        return rows

    transformer_rows = infrastructure_rows(transformer_count, label="transformer")
    charger_rows = infrastructure_rows(charger_count, label="charger")
    write_csv(
        diagnostics / "transformer_diagnostics.csv",
        list(transformer_rows[0]),
        transformer_rows,
    )
    write_csv(
        diagnostics / "charger_diagnostics.csv",
        list(charger_rows[0]),
        charger_rows,
    )

    write_csv(
        validation / "service_reconciliation.csv",
        [
            "served_count_reconciliation_status",
            "satisfaction_sum_reconciliation_status",
            "charged_energy_reconciliation_status",
            "discharged_energy_reconciliation_status",
        ],
        [
            {
                "served_count_reconciliation_status": "pass",
                "satisfaction_sum_reconciliation_status": "pass",
                "charged_energy_reconciliation_status": "pass",
                "discharged_energy_reconciliation_status": "pass",
            }
            for _ in range(30)
        ],
    )
    (validation / "mapping_validation.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )
    same_pass_rows = passing_same_pass_reconciliation_rows()
    historical_rows = passing_historical_drift_rows(
        scale=scale,
        algorithm=algorithm,
        task_id=task_id,
        training_seed=training_seed,
    )
    write_csv(
        validation / "same_pass_canonical_reconciliation.csv",
        SAME_PASS_RECONCILIATION_COLUMNS,
        same_pass_rows,
    )
    write_csv(
        validation / "historical_canonical_drift.csv",
        HISTORICAL_DRIFT_COLUMNS,
        historical_rows,
    )
    (runtime_metadata / "reconciliation_summary.json").write_text(
        json.dumps(
            {
                "reconciliation_contract_version": 2,
                "stage_d_source_commit_sha": source_commit_sha,
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
                "historical_identity_failure_count": 0,
                "same_pass_metric_failure_count": 0,
                "service_reconciliation_failure_count": 0,
                "mapping_validation_failure_count": 0,
                "same_pass_canonical_rows": 31,
                "same_pass_reconciliation_rows": len(same_pass_rows),
                "historical_drift_rows": len(historical_rows),
                "historical_float_drift_count": 0,
                "historical_saturation_count_drift_count": 0,
                "service_reconciliation_rows": 30,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "stderr.log").write_bytes(b"")
    return root


def mutate_csv(path, mutator):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    mutator(fieldnames, rows)
    write_csv(path, fieldnames, rows)


def test_expected_episode_key_count():
    assert len(expected_episode_keys()) == 1200


def test_validate_episode_inventory_accepts_exact_matrix():
    validate_episode_inventory(complete_inventory_rows())


def test_validate_episode_inventory_rejects_duplicate():
    rows = complete_inventory_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate episode key"):
        validate_episode_inventory(rows)


def test_validate_episode_inventory_rejects_missing():
    rows = complete_inventory_rows()[1:]
    with pytest.raises(ValueError, match="missing episode key"):
        validate_episode_inventory(rows)


def test_validate_episode_inventory_rejects_wrong_seed_formula():
    rows = complete_inventory_rows()
    rows[0]["episode_seed"] = "999"
    with pytest.raises(ValueError, match="episode seed mismatch"):
        validate_episode_inventory(rows)


def test_validate_episode_inventory_requires_schema_v3():
    rows = complete_inventory_rows()
    rows[0]["diagnostic_schema_version"] = "2"
    with pytest.raises(ValueError, match="schema"):
        validate_episode_inventory(rows)


def test_validate_seed_output_accepts_exact_seed_directory(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    result = validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)
    assert result["status"] == "ok"
    assert result["episode_count"] == 30
    assert result["charger_row_count"] == 750
    assert result["transformer_row_count"] == 90
    assert result["formal_task_id"] == 0


@pytest.mark.parametrize("episode_count", [29, 31])
def test_validate_seed_output_rejects_wrong_episode_count(tmp_path, episode_count):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def change_count(fieldnames, rows):
        if episode_count == 29:
            rows.pop()
        else:
            rows.append(dict(rows[-1]))

    mutate_csv(path, change_count)
    with pytest.raises(ValueError, match="exactly 30 rows"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_duplicate_episode_index(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def duplicate_index(fieldnames, rows):
        rows[1]["episode_index"] = rows[0]["episode_index"]
        rows[1]["episode_seed"] = rows[0]["episode_seed"]

    mutate_csv(path, duplicate_index)
    with pytest.raises(ValueError, match="duplicate episode index"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scale", "100cp", "identity mismatch"),
        ("algorithm", "hierarchical", "identity mismatch"),
        ("training_seed", "1", "identity mismatch"),
        ("episode_seed", "999", "episode seed mismatch"),
        ("diagnostic_schema_version", "2", "schema"),
        ("episode_reward", "", "blank required metric"),
        ("episode_reward", "NaN", "non-finite"),
        ("episode_reward", "Inf", "non-finite"),
    ],
)
def test_validate_seed_output_rejects_bad_episode_value(
    tmp_path,
    field,
    value,
    message,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def update(fieldnames, rows):
        rows[0][field] = value

    mutate_csv(path, update)
    with pytest.raises(ValueError, match=message):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_missing_required_column(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def remove_column(fieldnames, rows):
        fieldnames.remove("episode_reward")
        for row in rows:
            row.pop("episode_reward")

    mutate_csv(path, remove_column)
    with pytest.raises(ValueError, match="column contract mismatch"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_missing_required_csv(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "diagnostics" / "charger_diagnostics.csv").unlink()
    with pytest.raises(ValueError, match="missing or empty CSV"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_failed_same_pass_reconciliation(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "validation" / "same_pass_canonical_reconciliation.csv"

    def fail(fieldnames, rows):
        rows[0]["status"] = "fail"

    mutate_csv(path, fail)
    with pytest.raises(ValueError, match="same-pass reconciliation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_failed_service_reconciliation(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "validation" / "service_reconciliation.csv"

    def fail(fieldnames, rows):
        rows[0]["charged_energy_reconciliation_status"] = "fail"

    mutate_csv(path, fail)
    with pytest.raises(ValueError, match="service reconciliation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_failed_mapping_validation(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "validation" / "mapping_validation.json").write_text(
        json.dumps({"status": "fail"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mapping validation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


@pytest.mark.parametrize(
    "relative_path",
    [
        "diagnostics/same_pass_canonical_eval30.csv",
        "validation/same_pass_canonical_reconciliation.csv",
        "validation/historical_canonical_drift.csv",
        "runtime_metadata/reconciliation_summary.json",
    ],
)
def test_validate_seed_output_requires_new_reconciliation_evidence(
    tmp_path,
    relative_path,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / relative_path).unlink()

    with pytest.raises(ValueError, match="reconciliation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_reconciliation_contract_version_mismatch(
    tmp_path,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    summary_path = seed_dir / "runtime_metadata" / "reconciliation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["reconciliation_contract_version"] = 1
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reconciliation contract"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_missing_same_pass_reconciliation_row(
    tmp_path,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "validation" / "same_pass_canonical_reconciliation.csv"
    fields, rows = read_csv_rows(path)
    write_csv(path, fields, rows[:-1])

    with pytest.raises(ValueError, match="same-pass reconciliation row count"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_duplicate_historical_drift_key(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "validation" / "historical_canonical_drift.csv"
    fields, rows = read_csv_rows(path)
    rows.append(dict(rows[-1]))
    write_csv(path, fields, rows)

    with pytest.raises(ValueError, match="duplicate historical drift key"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_reconciliation_row_wrong_version(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "validation" / "same_pass_canonical_reconciliation.csv"
    fields, rows = read_csv_rows(path)
    rows[0]["reconciliation_contract_version"] = "1"
    write_csv(path, fields, rows)

    with pytest.raises(ValueError, match="reconciliation contract"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_summary_count_mismatch(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    summary_path = seed_dir / "runtime_metadata" / "reconciliation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["same_pass_reconciliation_rows"] = (
        summary["same_pass_reconciliation_rows"] - 1
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="summary count"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_nonempty_stderr(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "logs" / "stderr.log").write_text(
        "Traceback",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stderr log must be empty"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_cli(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate-seed-output",
            "--directory",
            str(seed_dir),
            "--task-id",
            "0",
            "--training-seed",
            "0",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["episode_count"] == 30


def test_same_pass_reward_tracking_mismatch_hard_fails(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_same_pass_reconciliation_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    same_pass_csv = seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"

    def mutate(rows):
        rows[0]["tracking_error"] = "999.0"
        rows[1]["episode_reward"] = "999.0"

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
    assert {(row["episode_index"], row["field"]) for row in failed} == {
        (0, "tracking_error"),
        (1, "episode_reward"),
    }
    assert {row["failure_category"] for row in failed} == {"same_pass_metric_mismatch"}


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


def test_historical_identity_mismatch_hard_fails_without_using_float_drift_as_gate(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    write_same_pass_canonical_eval30(
        seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"
    )
    historical_csv = tmp_path / "historical_eval30.csv"

    def mutate(rows):
        rows[0]["episode_seed"] = "999999"

    write_canonical_eval30(historical_csv, mutator=mutate)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    formal_validation_json = tmp_path / "formal_package_validation.json"
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(config_path, checkpoint_prefix),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=formal_validation_json,
        stage_d_source_commit_sha="a" * 40,
        config_path=config_path,
        checkpoint_prefix=checkpoint_prefix,
    )
    rows = build_historical_canonical_drift_rows(inputs)

    mismatch = [
        row
        for row in rows
        if row["classification"] == "historical_identity_mismatch"
    ]
    assert mismatch
    assert mismatch[0]["field"] == "episode_seed"


def test_historical_floating_drift_is_audit_only_and_classified(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        build_same_pass_reconciliation_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    write_same_pass_canonical_eval30(
        seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"
    )
    historical_csv = tmp_path / "historical_eval30.csv"

    def mutate(rows):
        rows[0]["tracking_error"] = "1.0000001"

    write_canonical_eval30(historical_csv, mutator=mutate)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    formal_validation_json = tmp_path / "formal_package_validation.json"
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(config_path, checkpoint_prefix),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=formal_validation_json,
        stage_d_source_commit_sha="a" * 40,
        config_path=config_path,
        checkpoint_prefix=checkpoint_prefix,
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


@pytest.mark.parametrize(
    ("scale", "mapped_action_dimension"),
    [("25cp", 25), ("100cp", 100), ("500cp", 500), ("1000cp", 1000)],
)
def test_one_count_historical_saturation_drift_is_audit_only(
    scale,
    mapped_action_dimension,
    tmp_path,
):
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

    write_same_pass_canonical_eval30(
        same_pass_csv,
        scale=scale,
        mutator=same_pass_mutate,
    )
    diagnostics_path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def diagnostic_mutate(fieldnames, rows):
        for row in rows:
            row["global_action_fraction_at_max_all_slots"] = str(
                stage_d_count / denominator
            )

    mutate_csv(diagnostics_path, diagnostic_mutate)
    historical_csv = tmp_path / f"historical_{scale}.csv"

    def historical_mutate(rows):
        for row in rows:
            row["action_fraction_at_max"] = str(historical_count / denominator)

    write_canonical_eval30(historical_csv, scale=scale, mutator=historical_mutate)
    algorithm = str(stage_d_task(task_id)["algorithm"])
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / f"stage_d_files_{scale}",
        scale=scale,
        algorithm=algorithm,
    )
    formal_validation_json = tmp_path / f"formal_package_validation_{scale}.json"
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(
                config_path,
                checkpoint_prefix,
                task_id=task_id,
                scale=scale,
                algorithm=algorithm,
                training_seed=0,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=task_id,
        training_seed=0,
        formal_validation_json=formal_validation_json,
        stage_d_source_commit_sha="a" * 40,
        config_path=config_path,
        checkpoint_prefix=checkpoint_prefix,
    )

    assert all(row["status"] == "pass" for row in build_same_pass_reconciliation_rows(inputs))
    drift_rows = build_historical_canonical_drift_rows(inputs)
    saturation = next(
        row
        for row in drift_rows
        if row["field"] == "action_fraction_at_max" and row["episode_index"] == 0
    )
    assert saturation["classification"] == "historical_saturation_count_drift"
    assert saturation["historical_at_max_count"] == historical_count
    assert saturation["stage_d_at_max_count"] == stage_d_count
    assert saturation["total_action_decision_denominator"] == denominator
    assert saturation["count_difference"] == 1
    assert saturation["fraction_difference"] == pytest.approx(1 / denominator)


def test_historical_saturation_count_and_denominator_reconstruct_stored_fraction():
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        reconstruct_saturation_count,
    )

    reconstructed, residual, comparable = reconstruct_saturation_count(
        str(17 / 2800),
        2800,
    )

    assert reconstructed == 17
    assert residual < 1e-9
    assert comparable is True


def test_historical_saturation_reconstruction_not_comparable_when_residual_is_too_large():
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        reconstruct_saturation_count,
    )

    reconstructed, residual, comparable = reconstruct_saturation_count("0.333", 2800)

    assert reconstructed is None
    assert residual >= 1e-9
    assert comparable is False


@pytest.mark.parametrize(
    (
        "historical_fraction",
        "stage_d_fraction",
        "historical_count",
        "stage_d_count",
        "expected_classification",
    ),
    [
        (
            "0.0010714285714285715",
            "0.0014285714285714286",
            3,
            4,
            "historical_saturation_count_drift",
        ),
        (
            "0.0010714285714285715",
            "0.0010714285714285715",
            3,
            3,
            "exact_match",
        ),
        (
            "0.0010714285714285715",
            "0.0010714285714285716",
            3,
            3,
            "historical_float_drift",
        ),
        ("0.333", "0.0010714285714285715", None, 3, "not_comparable"),
    ],
)
def test_historical_saturation_classification_matrix(
    tmp_path,
    historical_fraction,
    stage_d_fraction,
    historical_count,
    stage_d_count,
    expected_classification,
):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    same_pass_csv = seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"
    denominator = 2800

    def same_pass_mutate(rows):
        for row in rows:
            if row["row_type"] == "episode":
                row["action_fraction_at_max"] = stage_d_fraction
                row["same_pass_at_max_count"] = str(stage_d_count)
                row["mapped_action_dimension"] = "25"
                row["total_action_decision_denominator"] = str(denominator)

    write_same_pass_canonical_eval30(same_pass_csv, mutator=same_pass_mutate)
    historical_csv = tmp_path / "historical_eval30.csv"

    def historical_mutate(rows):
        for row in rows:
            row["action_fraction_at_max"] = historical_fraction

    write_canonical_eval30(historical_csv, mutator=historical_mutate)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    formal_validation_json = tmp_path / "formal_package_validation.json"
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(config_path, checkpoint_prefix),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=formal_validation_json,
        stage_d_source_commit_sha="a" * 40,
        config_path=config_path,
        checkpoint_prefix=checkpoint_prefix,
    )

    drift_rows = build_historical_canonical_drift_rows(inputs)
    saturation = next(
        row
        for row in drift_rows
        if row["field"] == "action_fraction_at_max" and row["episode_index"] == 0
    )
    assert saturation["classification"] == expected_classification
    if historical_count is None:
        assert saturation["historical_at_max_count"] == ""
    else:
        assert saturation["historical_at_max_count"] == historical_count


def test_reconciliation_evidence_remains_present_after_hard_gate_failure(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    same_pass_csv = seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"

    def mutate(rows):
        rows[0]["tracking_error"] = "999.0"

    write_same_pass_canonical_eval30(same_pass_csv, mutator=mutate)
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    formal_validation_json = tmp_path / "formal_package_validation.json"
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(config_path, checkpoint_prefix),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
            "--formal-validation-json",
            str(formal_validation_json),
            "--config-path",
            str(config_path),
            "--checkpoint-prefix",
            str(checkpoint_prefix),
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
    assert (
        seed_dir / "validation" / "same_pass_canonical_reconciliation.csv"
    ).is_file()
    assert (seed_dir / "validation" / "historical_canonical_drift.csv").is_file()
    assert (seed_dir / "validation" / "service_reconciliation.csv").is_file()
    summary = json.loads(
        (seed_dir / "runtime_metadata" / "reconciliation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["reconciliation_contract_version"] == 2
    assert summary["status"] == "failed"
    assert summary["hard_gate_status"] == "fail"
    assert "same_pass_metric_mismatch" in summary["failure_categories"]


def test_missing_required_input_fails_input_contract_without_complete_evidence_set(
    tmp_path,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv").unlink()
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    formal_validation_json = tmp_path / "formal_package_validation.json"
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(config_path, checkpoint_prefix),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

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
            "--formal-validation-json",
            str(formal_validation_json),
            "--config-path",
            str(config_path),
            "--checkpoint-prefix",
            str(checkpoint_prefix),
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


def run_prepare_seed_validation_with_provenance(
    seed_dir,
    historical_csv,
    *,
    source_commit="a" * 40,
    formal_validation_mutator=None,
    extra_args=(),
):
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        seed_dir / "stage_d_files"
    )
    formal_validation_json = seed_dir / "runtime_metadata" / "formal_package_validation.json"
    formal_validation_json.parent.mkdir(parents=True, exist_ok=True)
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(
                config_path,
                checkpoint_prefix,
                mutator=formal_validation_mutator,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "prepare-seed-validation",
            "--diagnostic-dir",
            str(seed_dir / "diagnostics"),
            *extra_args,
            "--validation-dir",
            str(seed_dir / "validation"),
            "--runtime-metadata-dir",
            str(seed_dir / "runtime_metadata"),
            "--formal-validation-json",
            str(formal_validation_json),
            "--config-path",
            str(config_path),
            "--checkpoint-prefix",
            str(checkpoint_prefix),
            "--task-id",
            "0",
            "--training-seed",
            "0",
            "--stage-d-source-commit-sha",
            source_commit,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.mark.parametrize("option_name", ["--historical-canonical-csv", "--canonical-csv"])
def test_prepare_seed_validation_accepts_canonical_csv_aliases(
    tmp_path,
    option_name,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)

    result = run_prepare_seed_validation_with_provenance(
        seed_dir,
        historical_csv,
        extra_args=[option_name, str(historical_csv)],
    )

    assert result.returncode == 0, result.stderr


def test_prepare_seed_validation_rejects_conflicting_canonical_csv_alias_values(
    tmp_path,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    historical_csv = tmp_path / "historical_eval30.csv"
    other_historical_csv = tmp_path / "other_historical_eval30.csv"
    write_canonical_eval30(historical_csv)
    write_canonical_eval30(other_historical_csv)

    result = run_prepare_seed_validation_with_provenance(
        seed_dir,
        historical_csv,
        extra_args=[
            "--historical-canonical-csv",
            str(historical_csv),
            "--canonical-csv",
            str(other_historical_csv),
        ],
    )

    assert result.returncode != 0
    assert "conflicting canonical CSV arguments" in result.stderr


@pytest.mark.parametrize(
    (
        "mutate_historical",
        "mutate_same_pass",
        "source_commit",
        "formal_validation_mutator",
        "field",
    ),
    [
        (
            lambda rows: rows[0].update({"episode_seed": "bad-seed"}),
            None,
            "a" * 40,
            None,
            "episode_seed",
        ),
        (
            lambda rows: rows[0].update({"done": "maybe"}),
            None,
            "a" * 40,
            None,
            "done",
        ),
        (None, None, "NOT_A_SHA", None, "stage_d_source_commit_sha"),
        (
            None,
            None,
            "a" * 40,
            lambda payload: payload.update({"formal_job_id": "not-a-job"}),
            "formal_job_id",
        ),
    ],
)
def test_malformed_identity_values_hard_fail_after_evidence_without_audit_classification(
    tmp_path,
    mutate_historical,
    mutate_same_pass,
    source_commit,
    formal_validation_mutator,
    field,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    if mutate_same_pass is not None:
        mutate_csv(
            seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv",
            lambda _fields, rows: mutate_same_pass(rows),
        )
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv, mutator=mutate_historical)

    result = run_prepare_seed_validation_with_provenance(
        seed_dir,
        historical_csv,
        source_commit=source_commit,
        formal_validation_mutator=formal_validation_mutator,
        extra_args=["--historical-canonical-csv", str(historical_csv)],
    )

    assert result.returncode != 0
    assert "historical_identity_mismatch" in result.stderr
    _, drift_rows = read_csv_rows(
        seed_dir / "validation" / "historical_canonical_drift.csv"
    )
    identity_rows = [row for row in drift_rows if row["field"] == field]
    assert identity_rows
    assert {row["classification"] for row in identity_rows} == {
        "historical_identity_mismatch"
    }
    assert all(row["classification"] != "not_comparable" for row in identity_rows)


def test_mapping_validation_payload_is_built_without_writing_or_raising(tmp_path):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_mapping_validation_payload,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "validation" / "mapping_validation.json").unlink()
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    formal_validation_json = tmp_path / "formal_package_validation.json"
    formal_validation_json.write_text(
        json.dumps(
            formal_validation_for_stage_d_files(config_path, checkpoint_prefix),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=formal_validation_json,
        stage_d_source_commit_sha="a" * 40,
        config_path=config_path,
        checkpoint_prefix=checkpoint_prefix,
    )

    payload = build_mapping_validation_payload(
        inputs,
        inputs.transformer_rows,
        inputs.charger_rows,
    )

    assert payload["reconciliation_contract_version"] == 2
    assert payload["status"] == "ok"
    assert payload["scale"] == "25cp"
    assert payload["expected_charger_count"] == 25
    assert payload["expected_transformer_count"] == 3
    assert payload["episode_count"] == 30
    assert not (seed_dir / "validation" / "mapping_validation.json").exists()


def write_stage_d_provenance_files(root, *, scale="25cp", algorithm="actiongnn"):
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "formal_config.yaml"
    checkpoint_prefix = root / "model.best"
    config_path.write_text(
        f"scale: {scale}\nalgorithm: {algorithm}\n",
        encoding="utf-8",
    )
    for basename, payload in {
        "model.best_actor": "actor\n",
        "model.best_actor_optimizer": "actor-opt\n",
        "model.best_critic": "critic\n",
        "model.best_critic_optimizer": "critic-opt\n",
        "kwargs.yaml": "{}\n",
    }.items():
        (root / basename).write_text(payload, encoding="utf-8")
    return config_path, checkpoint_prefix


def formal_validation_for_stage_d_files(
    config_path,
    checkpoint_prefix,
    *,
    task_id=0,
    scale="25cp",
    algorithm="actiongnn",
    training_seed=0,
    mutator=None,
):
    from scripts.validate_full_infrastructure_diagnostic_eval30 import sha256_file

    payload = {
        "status": "ok",
        "task_id": task_id,
        "scale": scale,
        "algorithm": algorithm,
        "formal_job_id": "58513929",
        "training_seed": training_seed,
        "formal_task_id": formal_task_id(task_id, training_seed),
        "config_sha256": sha256_file(config_path),
        "checkpoint_member_sha256": {
            "model.best_actor": sha256_file(
                checkpoint_prefix.parent / "model.best_actor"
            ),
            "model.best_actor_optimizer": sha256_file(
                checkpoint_prefix.parent / "model.best_actor_optimizer"
            ),
            "model.best_critic": sha256_file(
                checkpoint_prefix.parent / "model.best_critic"
            ),
            "model.best_critic_optimizer": sha256_file(
                checkpoint_prefix.parent / "model.best_critic_optimizer"
            ),
            "kwargs.yaml": sha256_file(checkpoint_prefix.parent / "kwargs.yaml"),
        },
    }
    if mutator is not None:
        mutator(payload)
    return payload


@pytest.mark.parametrize(
    ("mutator", "field"),
    [
        (lambda payload: payload.update({"config_sha256": "0" * 64}), "config_sha256"),
        (
            lambda payload: payload["checkpoint_member_sha256"].update(
                {"model.best_actor": "0" * 64}
            ),
            "checkpoint_member_sha256:model.best_actor",
        ),
        (lambda payload: payload.update({"formal_job_id": "58656380"}), "formal_job_id"),
    ],
)
def test_historical_provenance_mismatch_is_hard_identity_failure(
    tmp_path,
    mutator,
    field,
):
    import dataclasses
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    write_same_pass_canonical_eval30(
        seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"
    )
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=None,
        stage_d_source_commit_sha="a" * 40,
        config_path=config_path,
        checkpoint_prefix=checkpoint_prefix,
    )
    inputs = dataclasses.replace(
        inputs,
        formal_validation=formal_validation_for_stage_d_files(
            config_path,
            checkpoint_prefix,
            mutator=mutator,
        ),
    )

    rows = build_historical_canonical_drift_rows(inputs)
    mismatch = next(row for row in rows if row["field"] == field)
    assert mismatch["episode_index"] == ""
    assert mismatch["episode_seed"] == ""
    assert mismatch["classification"] == "historical_identity_mismatch"


def test_stage_d_source_commit_malformed_is_hard_identity_failure(tmp_path):
    import dataclasses
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        build_historical_canonical_drift_rows,
        load_seed_reconciliation_inputs,
    )

    seed_dir = build_seed_output(tmp_path / "seed0")
    write_same_pass_canonical_eval30(
        seed_dir / "diagnostics" / "same_pass_canonical_eval30.csv"
    )
    historical_csv = tmp_path / "historical_eval30.csv"
    write_canonical_eval30(historical_csv)
    config_path, checkpoint_prefix = write_stage_d_provenance_files(
        tmp_path / "stage_d_files"
    )
    inputs = load_seed_reconciliation_inputs(
        diagnostic_dir=seed_dir / "diagnostics",
        historical_canonical_csv=historical_csv,
        validation_dir=seed_dir / "validation",
        task_id=0,
        training_seed=0,
        formal_validation_json=None,
        stage_d_source_commit_sha="NOT_A_SHA",
        config_path=config_path,
        checkpoint_prefix=checkpoint_prefix,
    )
    inputs = dataclasses.replace(
        inputs,
        formal_validation=formal_validation_for_stage_d_files(
            config_path,
            checkpoint_prefix,
        ),
    )

    rows = build_historical_canonical_drift_rows(inputs)
    source_row = next(row for row in rows if row["field"] == "stage_d_source_commit_sha")
    assert source_row["classification"] == "historical_identity_mismatch"


def test_validate_seed_formal_package_returns_digests_with_temporary_extract_dir(tmp_path):
    import re
    from scripts.validate_full_infrastructure_diagnostic_eval30 import (
        validate_seed_formal_package,
    )

    package = create_full_formal_package(tmp_path / "formal", task_id=0, training_seed=0)
    payload = validate_seed_formal_package(
        package,
        task_id=0,
        training_seed=0,
        extract_dir=None,
    )

    assert re.fullmatch(r"[0-9a-f]{64}", payload["config_sha256"])
    assert set(payload["checkpoint_member_sha256"]) == {
        "model.best_actor",
        "model.best_actor_optimizer",
        "model.best_critic",
        "model.best_critic_optimizer",
        "kwargs.yaml",
    }
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", value)
        for value in payload["checkpoint_member_sha256"].values()
    )

# STAGE_D_TASK3_MINIMAL_PACKAGE_CONTRACT_TESTS
import hashlib
import io
import shutil
import tarfile

import scripts.validate_full_infrastructure_diagnostic_eval30 as task3_validator_module


def _task3_validate_package(package, task_id):
    return task3_validator_module.validate_stage_d_task_package(package, task_id)


def _task3_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _task3_read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _task3_rewrite_csv(path, mutator):
    fieldnames, rows = _task3_read_csv(path)
    mutator(fieldnames, rows)
    write_csv(path, fieldnames, rows)


def _task3_tar_member(name, payload=b"x", *, member_type=None, linkname=""):
    info = tarfile.TarInfo(name)
    if member_type is not None:
        info.type = member_type
        info.linkname = linkname
        info.size = 0
        return info, None
    info.size = len(payload)
    return info, payload


def build_task3_package(
    tmp_path,
    *,
    task_id=0,
    array_job_id="99999999",
    source_commit_sha="f" * 40,
    staging_mutator=None,
    manifest_mutator=None,
    extra_members=(),
):
    root = tmp_path / "task_package_staging"
    root.mkdir(parents=True)
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])

    for seed in TRAINING_SEEDS:
        build_seed_output(
            root / f"seed{seed}",
            task_id=task_id,
            training_seed=seed,
            source_commit_sha=source_commit_sha,
        )

    _task3_write_json(
        root / "task_metadata" / "task.json",
        {
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "formal_job_id": "58513929",
            "training_seeds": list(TRAINING_SEEDS),
            "formal_task_ids": [formal_task_id(task_id, seed) for seed in TRAINING_SEEDS],
            "eval_episodes_per_seed": EVAL_EPISODES,
            "diagnostic_schema_version": "3",
            "reconciliation_contract_version": 2,
        },
    )

    checkpoint_rows = [
        {
            "task_id": str(task_id),
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "formal_task_id": str(formal_task_id(task_id, seed)),
            "diagnostic_schema_version": "3",
        }
        for seed in TRAINING_SEEDS
    ]
    write_csv(
        root / "summaries" / "checkpoint_inventory.csv",
        list(checkpoint_rows[0]),
        checkpoint_rows,
    )

    episode_rows = [
        {
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "episode_index": str(episode_index),
            "episode_seed": str(episode_seed(scale, seed, episode_index)),
            "diagnostic_schema_version": "3",
        }
        for seed in TRAINING_SEEDS
        for episode_index in range(EVAL_EPISODES)
    ]
    write_csv(
        root / "summaries" / "episode_inventory.csv",
        list(episode_rows[0]),
        episode_rows,
    )

    runtime_rows = [
        {
            "task_id": str(task_id),
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "formal_task_id": str(formal_task_id(task_id, seed)),
            "status": "ok",
            "episode_count": str(EVAL_EPISODES),
        }
        for seed in TRAINING_SEEDS
    ]
    write_csv(
        root / "summaries" / "runtime_summary.csv",
        list(runtime_rows[0]),
        runtime_rows,
    )

    _task3_write_json(
        root / "validation" / "task_validation.json",
        {
            "status": "ok",
            "task_id": task_id,
            "checkpoint_groups": 5,
            "episode_count": 150,
            "diagnostic_schema_version": "3",
            "reconciliation_contract_version": 2,
        },
    )
    logs = root / "logs"
    logs.mkdir()
    (logs / "stdout.log").write_text("TASK_OK\n", encoding="utf-8")
    (logs / "stderr.log").write_bytes(b"")
    runtime_metadata = root / "runtime_metadata"
    runtime_metadata.mkdir()
    (runtime_metadata / "source_commit_sha.txt").write_text(
        source_commit_sha + "\n",
        encoding="utf-8",
    )
    (runtime_metadata / "array_job_id.txt").write_text(
        str(array_job_id) + "\n",
        encoding="utf-8",
    )

    if staging_mutator is not None:
        staging_mutator(root)

    file_list_path = root / "runtime_metadata" / "package_file_list.txt"
    manifest_path = root / "checksums" / "package_file_checksums.sha256"
    manifest_path.parent.mkdir()
    payload_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path not in {file_list_path, manifest_path}
    )
    file_names = [path.relative_to(root).as_posix() for path in payload_paths]
    file_names.extend(
        [
            file_list_path.relative_to(root).as_posix(),
            manifest_path.relative_to(root).as_posix(),
        ]
    )
    file_list_path.write_text("\n".join(file_names) + "\n", encoding="utf-8")
    payload_paths = sorted(path for path in root.rglob("*") if path.is_file())
    manifest_path.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
            for path in payload_paths
        ),
        encoding="utf-8",
    )
    if manifest_mutator is not None:
        manifest_mutator(manifest_path)

    package = tmp_path / "task0.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        for info, payload in extra_members:
            archive.addfile(info, None if payload is None else io.BytesIO(payload))
    return package


def test_task3_valid_five_seed_package_passes(tmp_path):
    package = build_task3_package(tmp_path)
    result = _task3_validate_package(package, task_id=0)
    assert result == {
        "status": "ok",
        "task_id": 0,
        "scale": "25cp",
        "algorithm": "actiongnn",
        "checkpoint_groups": 5,
        "episode_count": 150,
        "diagnostic_schema_version": "3",
        "reconciliation_contract_version": 2,
        "source_commit_sha": "f" * 40,
        "array_job_id": "99999999",
    }


def test_task3_package_cli(tmp_path):
    package = build_task3_package(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate-task-package",
            "--package",
            str(package),
            "--task-id",
            "0",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["episode_count"] == 150


def test_task3_rejects_missing_seed_group(tmp_path):
    def mutate(root):
        shutil.rmtree(root / "seed4")

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="file set mismatch"):
        _task3_validate_package(package, task_id=0)


def test_task_package_validation_requires_all_new_reconciliation_evidence(tmp_path):
    def mutate(root):
        (root / "seed3" / "validation" / "historical_canonical_drift.csv").unlink()

    package = build_task3_package(tmp_path, staging_mutator=mutate)

    with pytest.raises(ValueError, match="file set mismatch"):
        _task3_validate_package(package, task_id=0)


def test_task_package_validation_rejects_seed_summary_source_commit_mismatch(tmp_path):
    def mutate(root):
        summary_path = (
            root
            / "seed3"
            / "runtime_metadata"
            / "reconciliation_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["stage_d_source_commit_sha"] = "b" * 40
        summary_path.write_text(
            json.dumps(summary, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    package = build_task3_package(
        tmp_path,
        staging_mutator=mutate,
        source_commit_sha="a" * 40,
    )

    with pytest.raises(ValueError, match="Stage D source commit mismatch"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_duplicate_checkpoint_seed(tmp_path):
    def mutate(root):
        path = root / "summaries" / "checkpoint_inventory.csv"

        def duplicate(fieldnames, rows):
            rows[1]["training_seed"] = rows[0]["training_seed"]
            rows[1]["formal_task_id"] = rows[0]["formal_task_id"]

        _task3_rewrite_csv(path, duplicate)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="checkpoint inventory"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_wrong_formal_task_id(tmp_path):
    def mutate(root):
        path = root / "summaries" / "checkpoint_inventory.csv"

        def wrong_id(fieldnames, rows):
            rows[2]["formal_task_id"] = "999"

        _task3_rewrite_csv(path, wrong_id)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="checkpoint inventory"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_missing_aggregate_episode_key(tmp_path):
    def mutate(root):
        path = root / "summaries" / "episode_inventory.csv"

        def remove(fieldnames, rows):
            rows.pop()

        _task3_rewrite_csv(path, remove)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="episode inventory"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_failed_seed_contract(tmp_path):
    def mutate(root):
        (root / "seed2" / "logs" / "stderr.log").write_text(
            "Traceback", encoding="utf-8"
        )

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="stderr log must be empty"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_nonempty_task_stderr(tmp_path):
    def mutate(root):
        (root / "logs" / "stderr.log").write_text("failure", encoding="utf-8")

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="task stderr"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_manifest_coverage_gap(tmp_path):
    def mutate(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

    package = build_task3_package(tmp_path, manifest_mutator=mutate)
    with pytest.raises(ValueError, match="manifest coverage"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_checksum_mismatch(tmp_path):
    def mutate(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        _, name = lines[0].split("  ", 1)
        lines[0] = f"{'0' * 64}  {name}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    package = build_task3_package(tmp_path, manifest_mutator=mutate)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_unsafe_tar_path(tmp_path):
    package = build_task3_package(
        tmp_path,
        extra_members=[_task3_tar_member("../escape.txt")],
    )
    with pytest.raises(ValueError, match="unsafe tar path"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_duplicate_tar_member(tmp_path):
    package = build_task3_package(
        tmp_path,
        extra_members=[_task3_tar_member("logs/stdout.log", b"duplicate")],
    )
    with pytest.raises(ValueError, match="duplicate tar member"):
        _task3_validate_package(package, task_id=0)


@pytest.mark.parametrize(
    ("member_type", "linkname"),
    [
        (tarfile.SYMTYPE, "logs/stdout.log"),
        (tarfile.LNKTYPE, "logs/stdout.log"),
        (tarfile.CHRTYPE, ""),
    ],
)
def test_task3_rejects_links_and_devices(tmp_path, member_type, linkname):
    package = build_task3_package(
        tmp_path,
        extra_members=[
            _task3_tar_member(
                "unsafe-member",
                member_type=member_type,
                linkname=linkname,
            )
        ],
    )
    with pytest.raises(ValueError, match="link or device"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_checkpoint_leakage(tmp_path):
    package = build_task3_package(
        tmp_path,
        extra_members=[_task3_tar_member("seed0/checkpoint/model.best_actor")],
    )
    with pytest.raises(ValueError, match="checkpoint artefact"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_wrong_task_metadata(tmp_path):
    def mutate(root):
        path = root / "task_metadata" / "task.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["algorithm"] = "hierarchical"
        _task3_write_json(path, payload)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="task metadata"):
        _task3_validate_package(package, task_id=0)
# STAGE_D_TASK4_BATCH1_DRY_RUN_TESTS
import os

TASK4_RUNNER = (
    PROJECT_ROOT / "m3_jobs" / "21_full_infrastructure_diagnostic_eval30.slurm"
)


def run_task4_dry_run(task_id, tmp_path, *, dry_run=True, extra_env=None):
    env = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": str(task_id),
        "SLURM_ARRAY_JOB_ID": "99999999",
        "SLURM_JOB_ID": f"99999999_{task_id}",
        "SLURM_JOB_NAME": "evgnn_full_infra_diag_eval30",
        "EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT": str(PROJECT_ROOT),
        "EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT": str(tmp_path / "runs"),
        "EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT": "/formal",
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE": (
            "/formal/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz"
        ),
        "KMP_INIT_AT_FORK": "FALSE",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if dry_run:
        env["EV_GNN_FULL_DIAGNOSTIC_DRY_RUN"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(TASK4_RUNNER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_task4_runner_has_exact_slurm_resources():
    text = TASK4_RUNNER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=comp" in text
    assert "#SBATCH --array=0-7" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --time=06:00:00" in text


@pytest.mark.parametrize(
    ("task_id", "scale", "algorithm", "formal_ids", "offset"),
    [
        (0, "25cp", "actiongnn", "0,1,2,3,4", "710000,710999,711998,712997,713996"),
        (1, "25cp", "hierarchical", "5,6,7,8,9", "710000,710999,711998,712997,713996"),
        (2, "100cp", "actiongnn", "10,11,12,13,14", "720000,720999,721998,722997,723996"),
        (3, "100cp", "hierarchical", "15,16,17,18,19", "720000,720999,721998,722997,723996"),
        (4, "500cp", "actiongnn", "20,21,22,23,24", "730000,730999,731998,732997,733996"),
        (5, "500cp", "hierarchical", "25,26,27,28,29", "730000,730999,731998,732997,733996"),
        (6, "1000cp", "actiongnn", "30,31,32,33,34", "740000,740999,741998,742997,743996"),
        (7, "1000cp", "hierarchical", "35,36,37,38,39", "740000,740999,741998,742997,743996"),
    ],
)
def test_task4_runner_dry_run_exact_mapping(
    tmp_path,
    task_id,
    scale,
    algorithm,
    formal_ids,
    offset,
):
    result = run_task4_dry_run(task_id, tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"task_id={task_id}" in result.stdout
    assert f"scale={scale}" in result.stdout
    assert f"algorithm={algorithm}" in result.stdout
    assert "training_seeds=0,1,2,3,4" in result.stdout
    assert f"formal_task_ids={formal_ids}" in result.stdout
    assert "eval_episodes=30" in result.stdout
    assert f"eval_seed_offsets={offset}" in result.stdout
    assert "expected_episode_count=150" in result.stdout
    assert result.stdout.count("EVALUATOR_COMMAND_SEED_") == 5
    assert "DRY_RUN_NO_EVALUATION_OR_PACKAGING" in result.stdout


def test_task4_runner_dry_run_uses_formal_eval30_contract(tmp_path):
    result = run_task4_dry_run(0, tmp_path)
    assert result.returncode == 0, result.stderr
    for seed in TRAINING_SEEDS:
        marker = f"EVALUATOR_COMMAND_SEED_{seed}="
        command = next(
            line.removeprefix(marker)
            for line in result.stdout.splitlines()
            if line.startswith(marker)
        )
        assert f"--seed {seed}" in command
        assert "--eval_episodes 30" in command
        assert "--device cpu" in command
        assert "--deterministic true" in command
        assert "--eval_expl_noise 0.0" in command
        expected_offset = 710000 + 999 * seed
        assert f"--eval_seed_offset {expected_offset}" in command
        assert f"# formal_task_id={seed}" in command
        assert "model.best" in command
        assert "model.last" not in command


def test_task4_runner_dry_run_is_side_effect_free(tmp_path):
    result = run_task4_dry_run(0, tmp_path)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("task_id", ["invalid", "-1", "8"])
def test_task4_runner_rejects_invalid_task_id(tmp_path, task_id):
    result = run_task4_dry_run(task_id, tmp_path)
    assert result.returncode != 0
    assert "task ID" in result.stderr


def test_task4_runner_real_mode_requires_expected_source_commit_before_side_effects(tmp_path):
    result = run_task4_dry_run(0, tmp_path, dry_run=False)
    assert result.returncode != 0
    assert "EXPECTED_SOURCE_COMMIT" in result.stderr
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "outputs").exists()


def test_task4_runner_real_mode_orchestrates_synthetic_seed_evaluations(tmp_path):
    source_root = tmp_path / "source"
    (source_root / "scripts").mkdir(parents=True)
    (source_root / "utils").mkdir()
    shutil.copy2(VALIDATOR, source_root / "scripts" / VALIDATOR.name)
    shutil.copy2(
        PROJECT_ROOT / "utils" / "infrastructure_diagnostics.py",
        source_root / "utils" / "infrastructure_diagnostics.py",
    )
    source_commit = "a" * 40
    (source_root / "SOURCE_COMMIT_SHA.txt").write_text(
        source_commit + "\n",
        encoding="utf-8",
    )

    formal_root = tmp_path / "formal"
    for seed in TRAINING_SEEDS:
        create_full_formal_package(formal_root, task_id=0, training_seed=seed)

    evaluator_stub = source_root / "stub_evaluator.py"
    evaluator_stub.write_text(
        """import argparse
import csv
from pathlib import Path
from utils.infrastructure_diagnostics import (
    CHARGER_DIAGNOSTIC_COLUMNS,
    EPISODE_DIAGNOSTIC_COLUMNS,
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
)

OFFSETS = {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}
TOPOLOGY = {"25cp": (25, 3), "100cp": (100, 7), "500cp": (500, 35), "1000cp": (1000, 70)}


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", required=True)
parser.add_argument("--scale", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--eval_episodes", required=True, type=int)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--max_episode_steps", required=True, type=int)
parser.add_argument("--deterministic", required=True)
parser.add_argument("--eval_expl_noise", required=True)
parser.add_argument("--eval_seed_offset", required=True, type=int)
parser.add_argument("--matrix_job_id", required=True)
args = parser.parse_args()

if args.eval_episodes != 30:
    raise SystemExit("expected 30 episodes")
if args.device != "cpu" or args.deterministic != "true" or args.eval_expl_noise != "0.0":
    raise SystemExit("diagnostic execution contract mismatch")
if "model.best" not in args.checkpoint or "model.last" in args.checkpoint:
    raise SystemExit("checkpoint contract mismatch")
if args.max_episode_steps != 112:
    raise SystemExit("simulation length mismatch")

output = Path(args.output_dir)
episode_rows = []
for episode_index in range(args.eval_episodes):
    row = {column: "0.0" for column in EPISODE_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": args.matrix_job_id,
            "scale": args.scale,
            "algorithm": args.algorithm,
            "training_seed": str(args.seed),
            "episode_index": str(episode_index),
            "episode_seed": str(args.eval_seed_offset + args.seed + episode_index),
            "config": args.config,
            "checkpoint_prefix": args.checkpoint,
            "run_name": args.run_name,
            "episode_steps": "112",
            "done": "True",
            "episode_reward": "-1.0",
            "max_action": "1.0",
            "max_action_tolerance": "1e-06",
            "environment_action_low": "-1.0",
            "environment_action_high": "1.0",
            "observed_action_min_active": "0.0",
            "observed_action_max_active": "1.0",
            "action_tolerance": "1e-06",
            "environment_action_domain_support": "signed",
            "v2g_enabled": "False",
            "v2g_enabled_source": "config",
            "global_action_fraction_at_max_all_slots": "0.5",
            "global_action_fraction_at_max_active": "0.5",
            "global_action_nonzero_fraction_active": "1.0",
            "active_action_decision_count": "1",
            "active_action_below_environment_low_count": "0",
            "active_action_below_environment_low_fraction": "0.0",
            "active_action_above_environment_high_count": "0",
            "active_action_above_environment_high_fraction": "0.0",
            "global_positive_action_fraction_active": "1.0",
            "global_zero_action_fraction_active": "0.0",
            "global_negative_action_fraction_active": "0.0",
            "global_action_fraction_at_positive_max_active": "0.5",
            "global_action_fraction_at_negative_min_active": "0.0",
            "global_positive_action_sum_active": "1.0",
            "global_negative_action_magnitude_sum_active": "0.0",
            "global_action_mean_all_slots": "0.5",
            "global_action_mean_active": "0.5",
            "global_action_sum_active": "1.0",
            "active_slot_count_mean": "1.0",
            "nonzero_action_count_mean_all_slots": "1.0",
            "inactive_slot_fraction_mean": "0.0",
            "inactive_slot_decision_count": "0",
            "inactive_nonzero_action_count": "0",
            "inactive_nonzero_action_fraction_all_slots": "0.0",
            "transformer_action_fraction_at_max_active_macro_mean": "0.5",
            "charger_action_fraction_at_max_active_macro_mean": "0.5",
            "transformer_action_nonzero_fraction_active_macro_mean": "1.0",
            "charger_action_nonzero_fraction_active_macro_mean": "1.0",
            "transformer_allocation_zero_pressure_step_fraction": "0.0",
            "transformer_allocation_valid_step_count": "1.0",
            "charger_allocation_zero_pressure_step_fraction": "0.0",
            "charger_allocation_valid_step_count": "1.0",
            "tracking_error": "1.0",
            "energy_tracking_error": "1.0",
            "power_tracker_violation": "1.0",
            "total_energy_charged": "1.0",
            "total_energy_discharged": "0.0",
            "average_user_satisfaction": "1.0",
            "energy_user_satisfaction": "100.0",
            "total_transformer_overload": "0.0",
            "total_ev_served": "1",
            "diagnostic_schema_version": "3",
        }
    )
    episode_rows.append({column: row[column] for column in EPISODE_DIAGNOSTIC_COLUMNS})
write_csv(output / "episode_diagnostics.csv", EPISODE_DIAGNOSTIC_COLUMNS, episode_rows)
summary = {column: "0.0" for column in SEED_SUMMARY_DIAGNOSTIC_COLUMNS}
summary.update(
    {
        "matrix_job_id": args.matrix_job_id,
        "scale": args.scale,
        "algorithm": args.algorithm,
        "training_seed": str(args.seed),
        "n_eval_episodes": "30",
        "global_action_fraction_at_max_active_mean": "0.5",
        "global_action_fraction_at_max_all_slots_mean": "0.5",
        "global_action_nonzero_fraction_active_mean": "1.0",
        "active_action_decision_count_mean": "1.0",
        "active_action_below_environment_low_count": "0",
        "active_action_below_environment_low_fraction": "0.0",
        "active_action_above_environment_high_count": "0",
        "active_action_above_environment_high_fraction": "0.0",
        "global_positive_action_fraction_active_mean": "1.0",
        "global_zero_action_fraction_active_mean": "0.0",
        "global_negative_action_fraction_active_mean": "0.0",
        "environment_action_low": "-1.0",
        "environment_action_high": "1.0",
        "action_tolerance": "1e-06",
        "environment_action_domain_support": "signed",
        "v2g_enabled": "False",
        "v2g_enabled_source": "config",
        "power_tracker_violation_mean": "1.0",
        "tracking_error_mean": "1.0",
        "energy_tracking_error_mean": "1.0",
        "total_ev_served_mean": "1.0",
        "total_energy_charged_mean": "1.0",
        "average_user_satisfaction_mean": "1.0",
        "energy_user_satisfaction_mean": "100.0",
        "diagnostic_schema_version": "3",
    }
)
write_csv(output / "seed_summary_diagnostics.csv", SEED_SUMMARY_DIAGNOSTIC_COLUMNS, [summary])


def infrastructure_rows(count, label):
    columns = TRANSFORMER_DIAGNOSTIC_COLUMNS if label == "transformer" else CHARGER_DIAGNOSTIC_COLUMNS
    rows = []
    for episode_index in range(args.eval_episodes):
        for row_index in range(count):
            row = {column: "0.0" for column in columns}
            row.update(
                {
            "matrix_job_id": args.matrix_job_id,
            "scale": args.scale,
            "algorithm": args.algorithm,
            "training_seed": str(args.seed),
            "episode_index": str(episode_index),
            "episode_seed": str(args.eval_seed_offset + args.seed + episode_index),
            f"{label}_id": str(row_index),
            "transformer_id": str(row_index % transformer_count),
            "n_chargers_total": str(charger_count),
            "n_active_chargers_seen": "1",
            "n_ports": "1",
            "n_active_ev_decisions": "1" if row_index == 0 else "0",
            "n_all_slot_decisions": "1",
            "action_sum_active": "1.0" if row_index == 0 else "0.0",
            "action_mean_active": "1.0" if row_index == 0 else "0.0",
            "action_max_active": "1.0" if row_index == 0 else "0.0",
            "action_fraction_at_max_active": "1.0" if row_index == 0 else "0.0",
            "action_nonzero_fraction_active": "1.0" if row_index == 0 else "0.0",
            "positive_action_fraction_active": "1.0" if row_index == 0 else "0.0",
            "zero_action_fraction_active": "0.0" if row_index == 0 else "1.0",
            "negative_action_fraction_active": "0.0",
            "action_fraction_at_positive_max_active": "1.0" if row_index == 0 else "0.0",
            "action_fraction_at_negative_min_active": "0.0",
            "positive_action_sum_active": "1.0" if row_index == 0 else "0.0",
            "negative_action_magnitude_sum_active": "0.0",
            "action_sum_all_slots": "1.0" if row_index == 0 else "0.0",
            "action_mean_all_slots": "1.0" if row_index == 0 else "0.0",
            "action_max_all_slots": "1.0" if row_index == 0 else "0.0",
            "action_fraction_at_max_all_slots": "1.0" if row_index == 0 else "0.0",
            "overload_magnitude_sum": "0.0",
            "overload_magnitude_max": "0.0",
            "overload_frequency_steps": "0",
            "overload_frequency_fraction": "0.0",
            "cs_power_sum_kwh": "1.0" if row_index == 0 else "0.0",
            "cs_power_mean_kw": "1.0" if row_index == 0 else "0.0",
            "cs_power_max_kw": "1.0" if row_index == 0 else "0.0",
            "served_ev_count": "1" if row_index == 0 else "0",
            "energy_charged_kwh": "1.0" if row_index == 0 else "0.0",
            "energy_discharged_kwh": "0.0",
            "user_satisfaction_sum": "1.0" if row_index == 0 else "0.0",
            "user_satisfaction_mean": "1.0" if row_index == 0 else "0.0",
            "user_satisfaction_mean_served_ev_weighted": "1.0" if row_index == 0 else "0.0",
            "user_satisfaction_observation_count": "1" if row_index == 0 else "0",
            "user_satisfaction_source": "synthetic",
            "diagnostic_schema_version": "3",
                }
            )
            rows.append({column: row[column] for column in columns})
    return rows


charger_count, transformer_count = TOPOLOGY[args.scale]
transformer_rows = infrastructure_rows(transformer_count, "transformer")
charger_rows = infrastructure_rows(charger_count, "charger")
write_csv(output / "transformer_diagnostics.csv", TRANSFORMER_DIAGNOSTIC_COLUMNS, transformer_rows)
write_csv(output / "charger_diagnostics.csv", CHARGER_DIAGNOSTIC_COLUMNS, charger_rows)
""",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_ARRAY_JOB_ID": "424242",
        "SLURM_JOB_ID": "424242_0",
        "EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT": str(source_root),
        "EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT": str(tmp_path / "runs"),
        "EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT": str(formal_root),
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE": str(tmp_path / "missing.tar.gz"),
        "EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT": source_commit,
        "EV_GNN_FULL_DIAGNOSTIC_EVALUATOR_SCRIPT": str(evaluator_stub),
    }
    result = subprocess.run(
        ["bash", str(TASK4_RUNNER)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr + "\n" + result.stdout

    package = tmp_path / "outputs" / task5_package_basename(0, "424242")
    assert package.is_file()
    assert package.with_name(package.name + ".sha256").is_file()
    validation = _task3_validate_package(package, task_id=0)
    assert validation["episode_count"] == 150
    assert validation["source_commit_sha"] == source_commit
    with tarfile.open(package, "r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
    assert all("checkpoint/" not in name for name in names)
    assert all("model.best" not in name for name in names)


def test_task4_runner_has_no_ambiguous_discovery_patterns():
    text = TASK4_RUNNER.read_text(encoding="utf-8")
    prohibited = (
        "squeue -n",
        "sacct -n",
        "find /projects",
        "model.last",
    )
    for token in prohibited:
        assert token not in text


# STAGE_D_TASK4_TO_TASK6_FULL_WORKFLOW_TESTS
TASK5_REDUCER = (
    PROJECT_ROOT / "m3_jobs" / "22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm"
)
TASK6_SOURCE_BUNDLE = (
    PROJECT_ROOT
    / "m3_jobs"
    / "create_full_infrastructure_diagnostic_eval30_source_bundle.sh"
)
TASK6_SUBMIT = (
    PROJECT_ROOT / "m3_jobs" / "submit_full_infrastructure_diagnostic_eval30_workflow.sh"
)
SACCT_FIELDS = (
    "JobIDRaw",
    "JobID",
    "JobName",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocCPUS",
    "MaxRSS",
    "TotalCPU",
)


def run_full_validator(*args, check=True):
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), *map(str, args)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"validator failed with {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def full_formal_package_basename(
    task_id=0,
    training_seed=0,
    *,
    scale=None,
    algorithm=None,
    formal_id=None,
):
    task = stage_d_task(task_id)
    scale = scale or str(task["scale"])
    algorithm = algorithm or str(task["algorithm"])
    formal_id = formal_task_id(task_id, training_seed) if formal_id is None else formal_id
    return (
        f"m3_controlled_multiscale_formal_{scale}_{algorithm}_seed{training_seed}_"
        f"job58513929_task{formal_id}.tar.gz"
    )


def full_formal_config_text(
    scale="25cp",
    *,
    simulation_length=112,
    v2g_enabled=False,
    charger_count=None,
    transformer_count=None,
):
    charger_count = TOPOLOGY[scale][0] if charger_count is None else charger_count
    transformer_count = TOPOLOGY[scale][1] if transformer_count is None else transformer_count
    return (
        f"simulation_length: {simulation_length}\n"
        f"v2g_enabled: {'true' if v2g_enabled else 'false'}\n"
        f"number_of_charging_stations: {charger_count}\n"
        f"number_of_transformers: {transformer_count}\n"
    )


def full_formal_members(
    *,
    task_id=0,
    training_seed=0,
    scale=None,
    algorithm=None,
    config_text=None,
    missing=(),
    use_model_last=False,
):
    task = stage_d_task(task_id)
    scale = scale or str(task["scale"])
    algorithm = algorithm or str(task["algorithm"])
    run_name = f"controlled_multiscale_formal_{scale}_{algorithm}_seed{training_seed}"
    train_dir = f"train/{run_name}"
    checkpoint_prefix = "model.last" if use_model_last else "model.best"
    eval_seed_offset = evaluator_seed_offset(scale, training_seed)
    canonical_dict_rows = []
    for episode_index in range(EVAL_EPISODES):
        diagnostic = valid_episode_row(
            episode_index,
            scale=scale,
            algorithm=algorithm,
            training_seed=training_seed,
        )
        canonical_dict_rows.append(
            {
                "row_type": "episode",
                "algorithm": algorithm,
                "seed": str(training_seed),
                "episode_index": diagnostic["episode_index"],
                "episode_seed": diagnostic["episode_seed"],
                "episode_steps": diagnostic["episode_steps"],
                "done": diagnostic["done"],
                "episode_reward": diagnostic["episode_reward"],
                "tracking_error": diagnostic["tracking_error"],
                "energy_tracking_error": diagnostic["energy_tracking_error"],
                "power_tracker_violation": diagnostic["power_tracker_violation"],
                "total_energy_charged": diagnostic["total_energy_charged"],
                "total_energy_discharged": diagnostic["total_energy_discharged"],
                "average_user_satisfaction": diagnostic["average_user_satisfaction"],
                "energy_user_satisfaction": diagnostic["energy_user_satisfaction"],
                "total_transformer_overload": diagnostic["total_transformer_overload"],
                "total_ev_served": diagnostic["total_ev_served"],
                "action_mean": diagnostic["global_action_mean_all_slots"],
                "action_fraction_at_max": diagnostic[
                    "global_action_fraction_at_max_all_slots"
                ],
                "active_action_count_mean": diagnostic[
                    "nonzero_action_count_mean_all_slots"
                ],
            }
        )
    canonical_rows = [
        ",".join(CANONICAL_EVAL30_COLUMNS) + "\n",
        *[
            ",".join(row[column] for column in CANONICAL_EVAL30_COLUMNS) + "\n"
            for row in canonical_dict_rows
        ],
    ]
    source_manifest_files = [
        "train_td3_gnn.py",
        "evaluate_td3_gnn.py",
        f"config_files/PublicPST_{scale.replace('cp', '')}.yaml",
    ]
    source_manifest = "".join(
        f"{hashlib.sha256(name.encode()).hexdigest()}  {name}\n"
        for name in source_manifest_files
    )
    members = {
        f"config/{scale}_{algorithm}_seed{training_seed}_config.yaml": (
            config_text or full_formal_config_text(scale)
        ),
        f"eval/{scale}_{algorithm}_seed{training_seed}_eval30.csv": "".join(
            canonical_rows
        ),
        f"{train_dir}/{checkpoint_prefix}_actor": "actor\n",
        f"{train_dir}/{checkpoint_prefix}_actor_optimizer": "actor-opt\n",
        f"{train_dir}/{checkpoint_prefix}_critic": "critic\n",
        f"{train_dir}/{checkpoint_prefix}_critic_optimizer": "critic-opt\n",
        f"{train_dir}/kwargs.yaml": "{}\n",
        "runtime_metadata/source_manifest.sha256": source_manifest,
        "runtime_metadata/source_manifest_files.txt": "".join(
            f"{name}\n" for name in source_manifest_files
        ),
        "runtime_metadata/task_runtime_metadata.env": (
            f"task_id={formal_task_id(task_id, training_seed)}\n"
            "slurm_array_job_id=58513929\n"
            f"scale={scale}\n"
            f"algorithm={algorithm}\n"
            f"seed={training_seed}\n"
            f"config_path=config_files/PublicPST_{scale.replace('cp', '')}.yaml\n"
            f"train_command=python train_td3_gnn.py --algorithm {algorithm} --seed {training_seed} --max_timesteps 50000\n"
            f"eval_command=python evaluate_td3_gnn.py --algorithm {algorithm} --seed {training_seed} --eval_episodes 30 --checkpoint /tmp/{run_name}/model.best --max_episode_steps 112 --deterministic true --eval_expl_noise 0.0 --eval_seed_offset {eval_seed_offset}\n"
            "training_exit_status=0\n"
            "evaluation_exit_status=0\n"
        ),
    }
    for name in list(members):
        if name in set(missing) or Path(name).name in set(missing):
            members.pop(name)
    manifest = ""
    for name, payload in sorted(members.items()):
        manifest += f"{hashlib.sha256(str(payload).encode()).hexdigest()}  {name}\n"
    members["runtime_metadata/package_file_checksums.sha256"] = manifest
    if "runtime_metadata/package_file_checksums.sha256" in set(missing):
        members.pop("runtime_metadata/package_file_checksums.sha256")
    return members


def create_full_formal_package(
    directory,
    *,
    task_id=0,
    training_seed=0,
    scale=None,
    algorithm=None,
    formal_id=None,
    missing=(),
    manifest_omit=(),
    corrupt_checksum=False,
    config_text=None,
    use_model_last=False,
    extra_members=(),
    member_mutator=None,
):
    members = full_formal_members(
        task_id=task_id,
        training_seed=training_seed,
        scale=scale,
        algorithm=algorithm,
        config_text=config_text,
        missing=missing,
        use_model_last=use_model_last,
    )
    if member_mutator is not None:
        member_mutator(members)
        members["runtime_metadata/package_file_checksums.sha256"] = "".join(
            f"{hashlib.sha256(str(payload).encode()).hexdigest()}  {name}\n"
            for name, payload in sorted(members.items())
            if name != "runtime_metadata/package_file_checksums.sha256"
        )
    if manifest_omit:
        lines = []
        for line in members["runtime_metadata/package_file_checksums.sha256"].splitlines():
            _digest, name = line.split("  ", 1)
            if name not in set(manifest_omit) and Path(name).name not in set(manifest_omit):
                lines.append(line)
        members["runtime_metadata/package_file_checksums.sha256"] = "\n".join(lines) + "\n"
    if corrupt_checksum:
        lines = members["runtime_metadata/package_file_checksums.sha256"].splitlines()
        digest, name = lines[0].split("  ", 1)
        assert digest != "0" * 64
        lines[0] = f"{'0' * 64}  {name}"
        members["runtime_metadata/package_file_checksums.sha256"] = "\n".join(lines) + "\n"
    package = directory / full_formal_package_basename(
        task_id,
        training_seed,
        scale=scale,
        algorithm=algorithm,
        formal_id=formal_id,
    )
    package.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            data = str(payload).encode("utf-8")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for info, payload in extra_members:
            archive.addfile(info, None if payload is None else io.BytesIO(payload))
    return package


def task5_package_basename(task_id, array_job_id="777777"):
    task = stage_d_task(task_id)
    return (
        "m3_full_infrastructure_diagnostic_eval30_"
        f"{task['scale']}_{task['algorithm']}_seeds0-4_"
        f"job{array_job_id}_task{task_id}.tar.gz"
    )


def write_sidecar(package):
    sidecar = package.with_name(package.name + ".sha256")
    sidecar.write_text(
        f"{hashlib.sha256(package.read_bytes()).hexdigest()}  {package.name}\n",
        encoding="utf-8",
    )
    return sidecar


def full_sacct_text(array_job_id="777777", *, state_overrides=None):
    state_overrides = state_overrides or {}
    rows = []
    for task_id in range(8):
        raw = str(int(array_job_id) + task_id + 10)
        composite = f"{array_job_id}_{task_id}"
        for suffix, job_name, max_rss, total_cpu in [
            ("", "evgnn_full_infra_diag_eval30", "2048K", "00:00:10"),
            (".batch", "batch", "4096K", "00:00:12"),
            (".extern", "extern", "", "00:00:00"),
        ]:
            rows.append(
                "|".join(
                    [
                        raw + suffix,
                        composite + suffix,
                        job_name,
                        state_overrides.get(task_id, "COMPLETED"),
                        "0:0",
                        "12",
                        "4",
                        max_rss,
                        total_cpu,
                    ]
                )
            )
    return "\n".join(rows) + "\n"


def create_task5_reducer_fixture(tmp_path, *, package_overrides=None, sacct_text=None):
    package_overrides = package_overrides or {}
    array_job_id = "777777"
    source_commit = "f" * 40
    package_root = tmp_path / "packages"
    log_root = tmp_path / "logs"
    output_root = tmp_path / "output"
    work_root = tmp_path / "work"
    package_root.mkdir()
    log_root.mkdir()
    output_root.mkdir()
    work_root.mkdir()
    for task_id in range(8):
        source = build_task3_package(
            tmp_path / f"fixture_task{task_id}",
            task_id=task_id,
            array_job_id=array_job_id,
            source_commit_sha=source_commit,
            **package_overrides.get(task_id, {}),
        )
        package = package_root / task5_package_basename(task_id, array_job_id)
        shutil.copy2(source, package)
        write_sidecar(package)
        (log_root / f"evgnn_full_infra_diag_eval30_{array_job_id}_{task_id}.out").write_text(
            f"task {task_id} ok\n",
            encoding="utf-8",
        )
        (log_root / f"evgnn_full_infra_diag_eval30_{array_job_id}_{task_id}.err").write_text(
            "",
            encoding="utf-8",
        )
    sacct_path = tmp_path / "sacct_raw.txt"
    sacct_path.write_text(sacct_text or full_sacct_text(array_job_id), encoding="utf-8")
    reducer_stdout = tmp_path / "reducer_stdout.log"
    reducer_stderr = tmp_path / "reducer_stderr.log"
    reducer_stdout.write_text("reducer stdout\n", encoding="utf-8")
    reducer_stderr.write_text("", encoding="utf-8")
    return {
        "array_job_id": array_job_id,
        "source_commit": source_commit,
        "package_root": package_root,
        "log_root": log_root,
        "output_root": output_root,
        "work_root": work_root,
        "sacct_path": sacct_path,
        "reducer_stdout": reducer_stdout,
        "reducer_stderr": reducer_stderr,
    }


def run_task5_reducer(fixture, *, check=False):
    return run_full_validator(
        "validate-complete-workflow",
        "--array-job-id",
        fixture["array_job_id"],
        "--task-package-root",
        fixture["package_root"],
        "--slurm-log-root",
        fixture["log_root"],
        "--output-root",
        fixture["output_root"],
        "--work-root",
        fixture["work_root"],
        "--source-commit-sha",
        fixture["source_commit"],
        "--reducer-job-id",
        "888888",
        "--sacct-raw-file",
        fixture["sacct_path"],
        "--reducer-stdout-log",
        fixture["reducer_stdout"],
        "--reducer-stderr-log",
        fixture["reducer_stderr"],
        check=check,
    )


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_task4_formal_package_name_cli_is_seed_aware(seed):
    result = run_full_validator(
        "formal-package-name",
        "--task-id",
        1,
        "--training-seed",
        seed,
    )
    assert result.stdout.strip() == full_formal_package_basename(1, seed)


def test_task4_resolve_formal_package_prefers_individual_seed_package(tmp_path):
    individual = create_full_formal_package(tmp_path / "formal", task_id=0, training_seed=3)
    bundle = tmp_path / "unused.tar.gz"
    create_tar = tarfile.open
    with create_tar(bundle, "w:gz"):
        pass

    result = run_full_validator(
        "resolve-formal-package",
        "--task-id",
        0,
        "--training-seed",
        3,
        "--individual-package-root",
        individual.parent,
        "--complete-bundle",
        bundle,
        "--staging-dir",
        tmp_path / "staging",
    )
    payload = json.loads(result.stdout)
    assert payload["source_mode"] == "individual_task_package"
    assert Path(payload["package_path"]) == individual


def test_task4_resolve_formal_package_uses_exact_complete_bundle_member(tmp_path):
    package = create_full_formal_package(tmp_path / "formal", task_id=1, training_seed=4)
    bundle = tmp_path / "complete.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        info = tarfile.TarInfo(f"complete/task_packages/{package.name}")
        data = package.read_bytes()
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    result = run_full_validator(
        "resolve-formal-package",
        "--task-id",
        1,
        "--training-seed",
        4,
        "--individual-package-root",
        tmp_path / "missing",
        "--complete-bundle",
        bundle,
        "--staging-dir",
        tmp_path / "staging",
    )
    payload = json.loads(result.stdout)
    assert payload["source_mode"] == "complete_bundle_nested_task_package"
    assert Path(payload["package_path"]).is_file()
    assert payload["bundle_member"] == f"complete/task_packages/{package.name}"


@pytest.mark.parametrize("match_count", [0, 2])
def test_task4_resolve_formal_package_rejects_zero_or_multiple_bundle_matches(
    tmp_path,
    match_count,
):
    package = create_full_formal_package(tmp_path / "formal", task_id=0, training_seed=0)
    members = {}
    for index in range(match_count):
        members[f"complete/task_packages/copy{index}/{package.name}"] = package.read_bytes()
    bundle = tmp_path / "complete.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    result = run_full_validator(
        "resolve-formal-package",
        "--task-id",
        0,
        "--training-seed",
        0,
        "--individual-package-root",
        tmp_path / "missing",
        "--complete-bundle",
        bundle,
        "--staging-dir",
        tmp_path / "staging",
        check=False,
    )
    assert result.returncode != 0
    assert "exactly one" in result.stderr


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_task4_validate_formal_package_accepts_all_training_seeds(tmp_path, seed):
    package = create_full_formal_package(tmp_path, task_id=2, training_seed=seed)
    result = run_full_validator(
        "validate-formal-package",
        "--task-id",
        2,
        "--training-seed",
        seed,
        "--package",
        package,
        "--extract-dir",
        tmp_path / f"extract{seed}",
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["training_seed"] == seed
    assert payload["formal_task_id"] == formal_task_id(2, seed)
    assert payload["simulation_length"] == 112


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"extra_members": [_task3_tar_member("../escape.txt")]}, "unsafe"),
        ({"extra_members": [_task3_tar_member("./runtime_metadata/source_manifest.sha256")]}, "duplicate"),
        (
            {
                "extra_members": [
                    _task3_tar_member(
                        "unsafe-link",
                        member_type=tarfile.SYMTYPE,
                        linkname="../escape",
                    )
                ]
            },
            "link or device",
        ),
        ({"missing": ["model.best_actor"]}, "missing"),
        ({"use_model_last": True}, "model.best"),
        ({"manifest_omit": ["kwargs.yaml"]}, "coverage"),
        ({"corrupt_checksum": True}, "checksum mismatch"),
        ({"scale": "100cp"}, "package name"),
        ({"algorithm": "hierarchical"}, "package name"),
        ({"formal_id": 99}, "package name"),
        (
            {"config_text": full_formal_config_text(simulation_length=111)},
            "simulation_length",
        ),
        ({"config_text": full_formal_config_text(v2g_enabled=True)}, "v2g"),
        ({"config_text": full_formal_config_text(charger_count=24)}, "charging"),
        ({"config_text": full_formal_config_text(transformer_count=2)}, "transformer"),
    ],
)
def test_task4_validate_formal_package_rejects_single_property_mutations(
    tmp_path,
    kwargs,
    message,
):
    package = create_full_formal_package(tmp_path, task_id=0, training_seed=0, **kwargs)
    result = run_full_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--training-seed",
        0,
        "--package",
        package,
        "--extract-dir",
        tmp_path / "extract",
        check=False,
    )
    assert result.returncode != 0
    assert message.lower() in result.stderr.lower()


def test_task4_validate_formal_package_rejects_wrong_runtime_metadata(tmp_path):
    def mutate(members):
        metadata = members["runtime_metadata/task_runtime_metadata.env"]
        members["runtime_metadata/task_runtime_metadata.env"] = metadata.replace(
            "slurm_array_job_id=58513929",
            "slurm_array_job_id=123456",
        )

    package = create_full_formal_package(
        tmp_path,
        task_id=0,
        training_seed=0,
        member_mutator=mutate,
    )
    result = run_full_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--training-seed",
        0,
        "--package",
        package,
        "--extract-dir",
        tmp_path / "extract",
        check=False,
    )
    assert result.returncode != 0
    assert "formal task runtime metadata" in result.stderr


def test_task4_validate_formal_package_rejects_source_manifest_coverage(tmp_path):
    def mutate(members):
        members["runtime_metadata/source_manifest_files.txt"] = "train_td3_gnn.py\n"

    package = create_full_formal_package(
        tmp_path,
        task_id=0,
        training_seed=0,
        member_mutator=mutate,
    )
    result = run_full_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--training-seed",
        0,
        "--package",
        package,
        "--extract-dir",
        tmp_path / "extract",
        check=False,
    )
    assert result.returncode != 0
    assert "source manifest coverage" in result.stderr


def test_task5_reducer_dry_run_prints_exact_counts_and_scoped_sacct():
    env = {
        **os.environ,
        "EV_GNN_FULL_DIAGNOSTIC_REDUCER_DRY_RUN": "1",
        "EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID": "777777",
        "SLURM_JOB_ID": "888888",
    }
    result = subprocess.run(
        ["bash", str(TASK5_REDUCER)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "task_package_count=8" in result.stdout
    assert "checkpoint_group_count=40" in result.stdout
    assert "episode_count=1200" in result.stdout
    assert "sacct -j 777777 --parsable2 --noheader" in result.stdout
    assert "DRY_RUN_NO_REDUCTION_OR_PACKAGING" in result.stdout


def test_task5_complete_workflow_accepts_synthetic_8_task_packages(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    result = run_task5_reducer(fixture)
    payload = json.loads(result.stdout)
    bundle = Path(payload["bundle_path"])

    assert payload["status"] == "ok"
    assert payload["task_package_count"] == 8
    assert payload["checkpoint_group_count"] == 40
    assert payload["episode_count"] == 1200
    assert bundle.is_file()
    assert bundle.with_name(bundle.name + ".sha256").is_file()


def test_task5_complete_workflow_rejects_wrong_sidecar(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    sidecar = next(fixture["package_root"].glob("*.sha256"))
    sidecar.write_text("0" * 64 + "  wrong.tar.gz\n", encoding="utf-8")

    result = run_task5_reducer(fixture)
    assert result.returncode != 0
    assert "sidecar" in result.stderr.lower()


def test_task5_complete_workflow_rejects_29_episode_checkpoint(tmp_path):
    def mutate(root):
        path = root / "seed2" / "diagnostics" / "episode_diagnostics.csv"

        def remove_one(fieldnames, rows):
            rows.pop()

        _task3_rewrite_csv(path, remove_one)

    fixture = create_task5_reducer_fixture(
        tmp_path,
        package_overrides={3: {"staging_mutator": mutate}},
    )
    result = run_task5_reducer(fixture)
    assert result.returncode != 0
    assert "exactly 30 rows" in result.stderr


def test_task5_complete_workflow_rejects_failed_array_element(tmp_path):
    fixture = create_task5_reducer_fixture(
        tmp_path,
        sacct_text=full_sacct_text("777777", state_overrides={4: "FAILED"}),
    )
    result = run_task5_reducer(fixture)
    assert result.returncode != 0
    assert "COMPLETED" in result.stderr


def test_task5_complete_workflow_rejects_nonempty_slurm_stderr(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    (
        fixture["log_root"]
        / f"evgnn_full_infra_diag_eval30_{fixture['array_job_id']}_3.err"
    ).write_text("warning\n", encoding="utf-8")
    result = run_task5_reducer(fixture)
    assert result.returncode != 0
    assert "stderr log must be empty" in result.stderr


def test_task5_complete_workflow_rejects_nonempty_reducer_stderr(tmp_path):
    fixture = create_task5_reducer_fixture(tmp_path)
    fixture["reducer_stderr"].write_text("warning\n", encoding="utf-8")
    result = run_task5_reducer(fixture)
    assert result.returncode != 0
    assert "reducer stderr snapshot" in result.stderr


def test_task6_source_bundle_dry_run_lists_full_eval30_runtime_files():
    env = {
        **os.environ,
        "EV_GNN_FULL_DIAGNOSTIC_SOURCE_DRY_RUN": "1",
        "EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXPECTED_HEAD_SHA": "f" * 40,
    }
    result = subprocess.run(
        ["bash", str(TASK6_SOURCE_BUNDLE)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm" in result.stdout
    assert "m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm" in result.stdout
    assert "scripts/validate_full_infrastructure_diagnostic_eval30.py" in result.stdout
    assert "DRY_RUN_NO_ARCHIVE_CREATED" in result.stdout


def test_task6_submit_default_dry_run_does_not_invoke_sbatch(tmp_path):
    source_root = tmp_path / "source_root"
    (source_root / "m3_jobs").mkdir(parents=True)
    (source_root / "scripts").mkdir()
    (source_root / "utils").mkdir()
    for script in [TASK4_RUNNER, TASK5_REDUCER, TASK6_SUBMIT]:
        shutil.copy2(script, source_root / "m3_jobs" / script.name)
    shutil.copy2(VALIDATOR, source_root / "scripts" / VALIDATOR.name)
    shutil.copy2(
        PROJECT_ROOT / "utils" / "infrastructure_diagnostics.py",
        source_root / "utils" / "infrastructure_diagnostics.py",
    )
    source_commit = "f" * 40
    (source_root / "SOURCE_COMMIT_SHA.txt").write_text(source_commit + "\n", encoding="utf-8")

    source_bundle_root = tmp_path / "source_bundle_root"
    top_level = source_bundle_root / f"EV-GNN-full-infrastructure-diagnostics-eval30-{source_commit}"
    required_source_members = [
        "evaluate_td3_gnn.py",
        "evaluate_td3_gnn_infrastructure_diagnostics.py",
        "TD3/TD3_ActionGNN_Controlled.py",
        "TD3/TD3_HierarchicalActionGNN.py",
        "config_files/PublicPST_25cp.yaml",
        "config_files/PublicPST_100.yaml",
        "config_files/PublicPST_500.yaml",
        "config_files/PublicPST_1000.yaml",
        "m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm",
        "m3_jobs/22_full_infrastructure_diagnostic_eval30_reduce_bundle.slurm",
        "m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh",
        "m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh",
        "scripts/validate_full_infrastructure_diagnostic_eval30.py",
        "utils/ev2gym_training_utils.py",
        "utils/infrastructure_diagnostics.py",
        "utils/state_public_pst_gnn.py",
    ]
    for relative_name in required_source_members:
        output_path = top_level / relative_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT_ROOT / relative_name, output_path)
    (top_level / "runtime_metadata").mkdir(parents=True, exist_ok=True)
    (top_level / "SOURCE_COMMIT_SHA.txt").write_text(source_commit + "\n", encoding="utf-8")
    source_manifest = top_level / "runtime_metadata" / "source_file_checksums.sha256"
    source_manifest.write_text("", encoding="utf-8")
    manifest_paths = sorted(
        path for path in top_level.rglob("*") if path.is_file() and path != source_manifest
    )
    source_manifest.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(top_level).as_posix()}\n"
            for path in manifest_paths
        ),
        encoding="utf-8",
    )
    source_archive = tmp_path / "source.tar.gz"
    with tarfile.open(source_archive, "w:gz") as archive:
        for path in sorted(top_level.rglob("*")):
            if path.is_file():
                archive.add(
                    path,
                    arcname=path.relative_to(source_bundle_root).as_posix(),
                )
    source_sidecar = source_archive.with_name(source_archive.name + ".sha256")
    source_sidecar.write_text(
        f"{hashlib.sha256(source_archive.read_bytes()).hexdigest()}  {source_archive.name}\n",
        encoding="utf-8",
    )

    formal_root = tmp_path / "formal"
    for task_id in range(8):
        for seed in TRAINING_SEEDS:
            create_full_formal_package(formal_root, task_id=task_id, training_seed=seed)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/sh\n"
        "echo sbatch must not run >&2\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT": source_commit,
        "EV_GNN_FULL_DIAGNOSTIC_SOURCE_ARCHIVE": str(source_archive),
        "EV_GNN_FULL_DIAGNOSTIC_SOURCE_ARCHIVE_SHA256": str(source_sidecar),
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT": str(formal_root),
        "EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE": str(tmp_path / "unused_complete.tar.gz"),
        "EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT": str(tmp_path / "output"),
        "EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT": str(tmp_path / "runs"),
    }
    result = subprocess.run(
        ["bash", str(source_root / "m3_jobs" / TASK6_SUBMIT.name)],
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SBATCH_ARRAY_COMMAND=sbatch --parsable" in result.stdout
    assert "--dependency=afterok:<array_job_id>" in result.stdout
    assert "DRY_RUN_NO_SBATCH_CALLED" in result.stdout
