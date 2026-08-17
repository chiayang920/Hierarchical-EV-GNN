import csv
from pathlib import Path

import pytest

from scripts import formal_75k_80cell_statistics as statistics_module
from scripts import formal_75k_80cell_workflow as workflow


def training_rows():
    rows = []
    for cell in workflow.formal_matrix():
        rows.append({
            "task_id": cell.task_id,
            "scale": cell.scale,
            "algorithm": cell.algorithm,
            "seed": cell.seed,
            "training_steps": 75000,
            "evaluation_cadence": 5000,
            "evaluation_episodes": 5,
            "expected_scheduled_evaluations": 15,
            "scheduled_evaluation_steps": ";".join(str(x) for x in range(5000, 75001, 5000)),
            "model_best_present": True,
            "runtime_args_present": True,
            "package_complete": True,
            "source_identity": "source-head",
            "actor_output_transform": cell.actor_output_transform,
            "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
            "fresh_run": True,
            "required_values_finite": True,
            "runtime_elapsed_seconds": 1000 + cell.task_id,
            "checkpoint_identity_sha256": f"checkpoint-{cell.task_id}",
            "config_identity_sha256": f"config-{cell.task_id}",
        })
    return rows


def curve_rows():
    rows = []
    for cell in workflow.formal_matrix():
        for index, step in enumerate(range(5000, 75001, 5000)):
            algorithm_gain = 20 if cell.algorithm == "hierarchical" else 0
            rows.append({
                "task_id": cell.task_id,
                "scale": cell.scale,
                "algorithm": cell.algorithm,
                "seed": cell.seed,
                "source_identity": "source-head",
                "training_job_id": "12345",
                "timestep": step,
                "eval_mean_reward": -1000 + index * 10 + algorithm_gain + cell.seed,
                "eval_std_reward": 5 + index / 10,
                "elapsed_seconds": index + 1,
            })
    return rows


def eval_rows(*, hierarchy_service_multiplier=1.0):
    rows = []
    for cell in workflow.formal_matrix():
        for episode in range(30):
            hierarchy = cell.algorithm == "hierarchical"
            multiplier = hierarchy_service_multiplier if hierarchy else 1.0
            rows.append({
                "task_id": cell.task_id,
                "scale": cell.scale,
                "algorithm": cell.algorithm,
                "seed": cell.seed,
                "episode_index": episode,
                "checkpoint_role": "model.best",
                "source_identity": "source-head",
                "episode_reward": -1000 + cell.seed + (100 if hierarchy else 0),
                "tracking_error": 1000 - (100 if hierarchy else 0),
                "energy_tracking_error": 20 - (2 if hierarchy else 0),
                "power_tracker_violation": 10 - (1 if hierarchy else 0),
                "energy_delivered": 100 * multiplier,
                "evs_served": 50 * multiplier,
                "average_satisfaction": 0.9 * multiplier,
                "transformer_overload": 2 - (1 if hierarchy else 0),
                "required_values_finite": True,
            })
    return rows


def diagnostic_rows():
    rows = []
    for cell in workflow.formal_matrix():
        for episode in range(30):
            hierarchy = cell.algorithm == "hierarchical"
            rows.append({
                "task_id": cell.task_id,
                "scale": cell.scale,
                "algorithm": cell.algorithm,
                "seed": cell.seed,
                "episode_index": episode,
                "checkpoint_role": "model.best",
                "source_identity": "source-head",
                workflow.KEY_MECHANISM_METRIC: 0.2 if hierarchy else 0.8,
                "charger_upper_bound_action_fraction": 0.3 if hierarchy else 0.7,
                "transformer_positive_pressure_hhi": 0.2,
                "transformer_positive_pressure_gini": 0.3,
                "charger_positive_pressure_hhi": 0.4,
                "charger_positive_pressure_gini": 0.5,
                "tail_satisfaction": 0.8,
                "mapping_validation": "pass",
                "canonical_reconciliation": "pass",
                "service_reconciliation": "pass",
                "energy_reconciliation": "pass",
                "required_values_finite": True,
            })
    return rows


def test_service_guardrail_is_data_derived_pass_fail_or_unknown():
    unknown = statistics_module.derive_service_guardrail_status(eval_rows(), policy=None)
    assert unknown.status == "UNKNOWN"

    policy = {
        "energy_delivered_max_relative_decline": 0.05,
        "evs_served_max_relative_decline": 0.05,
        "average_satisfaction_max_relative_decline": 0.05,
    }
    passed = statistics_module.derive_service_guardrail_status(eval_rows(), policy=policy)
    assert passed.status == "PASS"

    failed = statistics_module.derive_service_guardrail_status(
        eval_rows(hierarchy_service_multiplier=0.8), policy=policy
    )
    assert failed.status == "FAIL"


def test_learning_dynamics_use_actual_fifteen_checkpoint_rows():
    result = statistics_module.analyse_learning_dynamics(training_rows(), curve_rows())
    assert len(result.training_curve_long) == 1200
    assert len(result.per_run_learning_summary) == 80
    assert len(result.checkpoint_selection_summary) == 80
    assert len(result.late_training_stability_summary) == 80
    assert len(result.per_scale_algorithm_learning_summary) == 8
    assert all(row["late_checkpoint_count"] == 4 for row in result.late_training_stability_summary)
    assert all(row["theil_sen_terminal_slope"] == pytest.approx(0.002) for row in result.late_training_stability_summary)
    assert all(row["model_best_step"] == 75000 for row in result.checkpoint_selection_summary)


def test_reducer_writes_real_outputs_and_blocks_strong_claim_when_service_unknown(tmp_path):
    outputs = statistics_module.reduce_formal_results(
        training_rows=training_rows(),
        training_curve_rows=curve_rows(),
        eval_rows=eval_rows(),
        diagnostic_rows=diagnostic_rows(),
        out_dir=tmp_path,
        service_policy=None,
    )
    assert outputs
    claims = (tmp_path / "claim_assessment.env").read_text()
    assert "SERVICE_GUARDRAIL_STATUS=UNKNOWN" in claims
    assert "STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED=NO" in claims

    with (tmp_path / "training_curve_long.csv").open(newline="") as handle:
        actual_curve_rows = list(csv.DictReader(handle))
    assert len(actual_curve_rows) == 1200
    assert "row_status" not in actual_curve_rows[0]
    assert actual_curve_rows[0]["eval_mean_reward"] != ""

    with (tmp_path / "late_training_stability_summary.csv").open(newline="") as handle:
        stability_rows = list(csv.DictReader(handle))
    assert stability_rows[0]["theil_sen_terminal_slope"] != ""
    assert stability_rows[0]["relative_late_loss_change"] != ""

    with (tmp_path / "resource_efficiency_summary.csv").open(newline="") as handle:
        resource_rows = list(csv.DictReader(handle))
    assert len(resource_rows) == 8
    assert all(row["runtime_evidence_status"] == "ELAPSED_ONLY" for row in resource_rows)

    figure_files = sorted((tmp_path / "paper_ready_figures").glob("*.svg"))
    assert [path.name for path in figure_files] == [
        "boundary_effect_by_scale.svg",
        "reward_standardised_effect_by_scale.svg",
    ]
    assert all(path.stat().st_size > 200 for path in figure_files)


def test_reducer_with_approved_service_policy_can_support_strong_claim(tmp_path):
    policy = {
        "energy_delivered_max_relative_decline": 0.05,
        "evs_served_max_relative_decline": 0.05,
        "average_satisfaction_max_relative_decline": 0.05,
    }
    statistics_module.reduce_formal_results(
        training_rows=training_rows(),
        training_curve_rows=curve_rows(),
        eval_rows=eval_rows(),
        diagnostic_rows=diagnostic_rows(),
        out_dir=tmp_path,
        service_policy=policy,
    )
    claims = (tmp_path / "claim_assessment.env").read_text()
    assert "SERVICE_GUARDRAIL_STATUS=PASS" in claims
    assert "STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED=YES" in claims
