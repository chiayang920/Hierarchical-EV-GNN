import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import formal_75k_80cell_workflow as workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPT = PROJECT_ROOT / "scripts" / "formal_75k_80cell_workflow.py"
SUBMIT_SCRIPT = PROJECT_ROOT / "m3_jobs" / "submit_formal_75k_80cell_workflow.sh"


def scheduled_steps():
    return ";".join(str(step) for step in range(5000, 75001, 5000))


def complete_training_rows():
    return [
        {
            "task_id": str(cell.task_id),
            "scale": cell.scale,
            "algorithm": cell.algorithm,
            "seed": str(cell.seed),
            "training_steps": "75000",
            "evaluation_cadence": "5000",
            "evaluation_episodes": "5",
            "expected_scheduled_evaluations": "15",
            "scheduled_evaluation_steps": scheduled_steps(),
            "model_best_present": "true",
            "runtime_args_present": "true",
            "package_complete": "true",
            "source_identity": "source-a",
            "actor_output_transform": cell.actor_output_transform,
            "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
            "fresh_run": "true",
            "required_values_finite": "true",
        }
        for cell in workflow.formal_matrix()
    ]


def complete_eval_rows():
    rows = []
    for cell in workflow.formal_matrix():
        for episode_index in range(30):
            rows.append(
                {
                    "scale": cell.scale,
                    "algorithm": cell.algorithm,
                    "seed": str(cell.seed),
                    "checkpoint_role": "model.best",
                    "episode_index": str(episode_index),
                    "episode_reward": str(
                        -1000 + cell.seed + (100 if cell.algorithm == "hierarchical" else 0)
                    ),
                    "tracking_error": str(50 - (5 if cell.algorithm == "hierarchical" else 0)),
                    "energy_tracking_error": "1.0",
                    "power_tracker_violation": "0.0",
                    "required_values_finite": "true",
                }
            )
    return rows


def complete_diagnostic_rows():
    rows = []
    for cell in workflow.formal_matrix():
        for episode_index in range(30):
            rows.append(
                {
                    "scale": cell.scale,
                    "algorithm": cell.algorithm,
                    "seed": str(cell.seed),
                    "checkpoint_role": "model.best",
                    "episode_index": str(episode_index),
                    "upper_bound_active_ev_action_fraction": (
                        "0.10" if cell.algorithm == "hierarchical" else "0.80"
                    ),
                    "energy_delivered": "10.0",
                    "evs_served": "10.0",
                    "average_satisfaction": "0.9",
                    "tail_satisfaction": "0.8",
                    "mapping_validation": "pass",
                    "canonical_reconciliation": "pass",
                    "service_reconciliation": "pass",
                    "energy_reconciliation": "pass",
                    "required_values_finite": "true",
                }
            )
    return rows


def test_formal_matrix_mapping_is_exact_and_unique():
    cells = workflow.formal_matrix()

    assert len(cells) == 80
    assert len({(cell.scale, cell.algorithm, cell.seed) for cell in cells}) == 80
    assert [cell.task_id for cell in cells] == list(range(80))
    assert [(cell.scale, cell.algorithm, cell.seed) for cell in cells[:12]] == [
        ("25cp", "actiongnn_nonnegative", 0),
        ("25cp", "actiongnn_nonnegative", 1),
        ("25cp", "actiongnn_nonnegative", 2),
        ("25cp", "actiongnn_nonnegative", 3),
        ("25cp", "actiongnn_nonnegative", 4),
        ("25cp", "actiongnn_nonnegative", 5),
        ("25cp", "actiongnn_nonnegative", 6),
        ("25cp", "actiongnn_nonnegative", 7),
        ("25cp", "actiongnn_nonnegative", 8),
        ("25cp", "actiongnn_nonnegative", 9),
        ("25cp", "hierarchical", 0),
        ("25cp", "hierarchical", 1),
    ]
    assert [(cell.task_id, cell.scale, cell.algorithm, cell.seed) for cell in cells[60:80]] == [
        *[(60 + seed, "1000cp", "actiongnn_nonnegative", seed) for seed in range(10)],
        *[(70 + seed, "1000cp", "hierarchical", seed) for seed in range(10)],
    ]


def test_formal_matrix_contract_values_are_propagated():
    cell = workflow.resolve_formal_cell(40)

    assert cell.scale == "500cp"
    assert cell.algorithm == "actiongnn_nonnegative"
    assert cell.seed == 0
    assert cell.training_steps == 75000
    assert cell.evaluation_cadence == 5000
    assert cell.training_evaluation_episodes == 5
    assert cell.expected_scheduled_evaluations == 15
    assert cell.actor_output_transform == "shifted_tanh_v1"
    assert workflow.scheduled_evaluation_steps() == tuple(range(5000, 75001, 5000))


def test_smoke_matrix_is_exact_two_cell_gate():
    smoke = workflow.smoke_matrix()

    assert [(cell.scale, cell.algorithm, cell.seed) for cell in smoke] == [
        ("25cp", "actiongnn_nonnegative", 0),
        ("25cp", "hierarchical", 0),
    ]
    assert all(cell.training_steps == workflow.SMOKE_TRAINING_STEPS for cell in smoke)


def test_training_command_uses_canonical_names_and_formal_protocol():
    command = workflow.training_command(
        workflow.resolve_formal_cell(0),
        save_dir="/scratch/formal/train",
        run_name="formal_25cp_corrected_seed0",
    )

    assert command == [
        "python",
        "train_td3_gnn.py",
        "--algorithm",
        "actiongnn_nonnegative",
        "--config",
        "config_files/PublicPST_25cp.yaml",
        "--seed",
        "0",
        "--device",
        "cpu",
        "--run_name",
        "formal_25cp_corrected_seed0",
        "--max_timesteps",
        "75000",
        "--start_timesteps",
        "1000",
        "--eval_freq",
        "5000",
        "--eval_episodes",
        "5",
        "--batch_size",
        "64",
        "--replay_buffer_size",
        "100000",
        "--save_dir",
        "/scratch/formal/train",
        "--log_to_wandb",
        "false",
    ]


def test_legacy_actiongnn_alias_is_rejected_for_new_formal_matrix():
    with pytest.raises(ValueError, match="legacy actiongnn"):
        workflow.validate_formal_algorithm("actiongnn")


def test_training_gate_requires_complete_75k_matrix():
    result = workflow.validate_training_gate(complete_training_rows())

    assert result.status == "PASS"
    assert result.cell_count == 80
    assert result.missing_cells == []
    assert result.duplicate_cells == []

    incomplete = complete_training_rows()[:-1]
    with pytest.raises(ValueError, match="80/80"):
        workflow.validate_training_gate(incomplete)

    fifty_k = complete_training_rows()
    fifty_k[0]["training_steps"] = "50000"
    with pytest.raises(ValueError, match="50k-only"):
        workflow.validate_training_gate(fifty_k)

    fourteen_checkpoints = complete_training_rows()
    fourteen_checkpoints[0]["scheduled_evaluation_steps"] = ";".join(
        str(step) for step in range(5000, 70001, 5000)
    )
    with pytest.raises(ValueError, match="15 scheduled"):
        workflow.validate_training_gate(fourteen_checkpoints)


def test_eval30_gate_requires_2400_model_best_episode_rows():
    result = workflow.validate_eval30_gate(complete_eval_rows())

    assert result.status == "PASS"
    assert result.cell_count == 80
    assert result.episode_count == 2400

    with pytest.raises(ValueError, match="2,400/2,400"):
        workflow.validate_eval30_gate(complete_eval_rows()[:-1])


def test_diagnostic_gate_requires_full_checkpoint_coverage_and_reconciliation():
    result = workflow.validate_diagnostic_gate(complete_diagnostic_rows())

    assert result.status == "PASS"
    assert result.cell_count == 80
    assert result.episode_count == 2400

    broken = complete_diagnostic_rows()
    broken[0]["mapping_validation"] = "fail"
    with pytest.raises(ValueError, match="mapping validation"):
        workflow.validate_diagnostic_gate(broken)


def test_reducer_uses_paired_seed_inference_and_holm_corrections(tmp_path):
    output_paths = workflow.reduce_formal_results(
        complete_training_rows(),
        complete_eval_rows(),
        complete_diagnostic_rows(),
        tmp_path,
    )

    assert (tmp_path / "claim_assessment.env") in output_paths
    assert (tmp_path / "primary_reward_statistical_tests.csv") in output_paths
    claim_text = (tmp_path / "claim_assessment.env").read_text(encoding="utf-8")
    assert "PRIMARY_INFERENCE_UNIT=paired_training_seed" in claim_text
    assert "STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED=YES" in claim_text

    with (tmp_path / "primary_reward_statistical_tests.csv").open(newline="", encoding="utf-8") as handle:
        reward_rows = list(csv.DictReader(handle))
    assert len(reward_rows) == 4
    assert {row["metric"] for row in reward_rows} == {"episode_reward"}
    assert all(row["n_paired_seeds"] == "10" for row in reward_rows)
    assert all(row["holm_family"] == "primary_reward_by_scale" for row in reward_rows)

    with (tmp_path / "key_mechanism_statistical_tests.csv").open(newline="", encoding="utf-8") as handle:
        mechanism_rows = list(csv.DictReader(handle))
    assert len(mechanism_rows) == 4
    assert {row["metric"] for row in mechanism_rows} == {"upper_bound_active_ev_action_fraction"}
    assert all(row["effect_orientation"] == "corrected_minus_hierarchy" for row in mechanism_rows)


def test_reducer_blocks_on_incomplete_matrices(tmp_path):
    with pytest.raises(ValueError, match="STATUS=BLOCKED"):
        workflow.reduce_formal_results(
            complete_training_rows()[:-1],
            complete_eval_rows(),
            complete_diagnostic_rows(),
            tmp_path,
        )


def test_tracking_error_is_not_an_independent_primary_hypothesis():
    assert workflow.primary_hypothesis_metrics() == ("episode_reward",)
    assert "tracking_error" in workflow.secondary_metrics()


def test_holm_correction_across_four_scale_hypotheses():
    adjusted = workflow.holm_adjust([0.001, 0.02, 0.03, 0.2])

    assert adjusted == pytest.approx([0.004, 0.06, 0.06, 0.2])


def test_claim_boundary_logic_handles_positive_partial_null_and_harmful_cases():
    positive = workflow.assess_claims(
        reward_effects=[
            {
                "scale": scale,
                "mean_benefit": 1.0,
                "ci_low": 0.1,
                "holm_adjusted_pvalue": 0.01,
                "seeds_favouring_hierarchy": 8,
                "leave_one_seed_out_direction_stable": True,
                "statistically_supported_harm": False,
            }
            for scale in workflow.SCALES
        ],
        boundary_effects=[
            {
                "scale": scale,
                "mean_benefit": 1.0,
                "holm_adjusted_pvalue": 0.01,
            }
            for scale in workflow.SCALES
        ],
        service_guardrails={"clear_material_collapse": False},
    )
    assert positive["STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED"] == "YES"

    partial = workflow.assess_claims(
        reward_effects=[
            {
                "scale": scale,
                "mean_benefit": 1.0 if index < 2 else 0.0,
                "ci_low": 0.1 if index < 2 else -0.1,
                "holm_adjusted_pvalue": 0.01 if index < 2 else 1.0,
                "seeds_favouring_hierarchy": 8 if index < 2 else 5,
                "leave_one_seed_out_direction_stable": index < 2,
                "statistically_supported_harm": False,
            }
            for index, scale in enumerate(workflow.SCALES)
        ],
        boundary_effects=[
            {"scale": scale, "mean_benefit": 1.0, "holm_adjusted_pvalue": 0.01}
            for scale in workflow.SCALES
        ],
        service_guardrails={"clear_material_collapse": False},
    )
    assert partial["STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED"] == "NO"
    assert partial["CROSS_SCALE_REWARD_DIRECTION_CONSISTENT"] == "NO"

    harmful = workflow.assess_claims(
        reward_effects=[
            {
                "scale": scale,
                "mean_benefit": -1.0 if index == 0 else 1.0,
                "ci_low": -2.0 if index == 0 else 0.1,
                "holm_adjusted_pvalue": 0.01,
                "seeds_favouring_hierarchy": 2 if index == 0 else 8,
                "leave_one_seed_out_direction_stable": True,
                "statistically_supported_harm": index == 0,
            }
            for index, scale in enumerate(workflow.SCALES)
        ],
        boundary_effects=[
            {"scale": scale, "mean_benefit": 1.0, "holm_adjusted_pvalue": 0.01}
            for scale in workflow.SCALES
        ],
        service_guardrails={"clear_material_collapse": False},
    )
    assert harmful["STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED"] == "NO"


def test_resource_profile_blocks_formal_submission_until_reviewed(tmp_path):
    profile = tmp_path / "formal_75k_resource_profile.env"
    profile.write_text("RESOURCE_PROFILE_APPROVED=NO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="approved resource profile"):
        workflow.load_resource_profile(profile)


def test_cli_print_matrix_outputs_all_80_rows():
    result = subprocess.run(
        [sys.executable, str(WORKFLOW_SCRIPT), "--print-matrix"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 80
    assert lines[0] == (
        "task_id=0 scale=25cp algorithm=actiongnn_nonnegative seed=0 "
        "config=config_files/PublicPST_25cp.yaml training_steps=75000"
    )
    assert lines[-1] == (
        "task_id=79 scale=1000cp algorithm=hierarchical seed=9 "
        "config=config_files/PublicPST_1000.yaml training_steps=75000"
    )


def test_submit_helper_dry_run_prints_dependency_stages_without_submitting():
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Stage S - two-cell smoke" in result.stdout
    assert "Stage A - 80-cell formal training array" in result.stdout
    assert "Stage F - final reducer and scientific analysis" in result.stdout
    assert "DRY_RUN_NO_JOBS_SUBMITTED" in result.stdout
