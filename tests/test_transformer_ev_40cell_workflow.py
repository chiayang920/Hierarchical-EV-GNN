import csv
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPT = PROJECT_ROOT / "scripts" / "transformer_ev_40cell_workflow.py"


def _workflow():
    from scripts import transformer_ev_40cell_workflow as workflow

    return workflow


def _scheduled_steps():
    return ";".join(str(step) for step in range(5000, 75001, 5000))


def _complete_training_rows():
    workflow = _workflow()
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
            "scheduled_evaluation_steps": _scheduled_steps(),
            "model_best_present": "true",
            "runtime_args_present": "true",
            "package_complete": "true",
            "fresh_run": "true",
            "required_values_finite": "true",
            "source_identity": "source-head",
            "source_bundle_identity": "bundle-a",
            "actor_output_transform": "transformer_ev_hierarchy_v1",
            "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
            "checkpoint_identity_sha256": f"checkpoint-{cell.task_id}",
            "config_identity_sha256": f"config-{cell.task_id}",
        }
        for cell in workflow.formal_matrix()
    ]


def _complete_episode_rows(kind):
    workflow = _workflow()
    rows = []
    for cell in workflow.formal_matrix():
        for episode_index in range(30):
            row = {
                "task_id": str(cell.task_id),
                "scale": cell.scale,
                "algorithm": cell.algorithm,
                "seed": str(cell.seed),
                "checkpoint_role": "model.best",
                "episode_index": str(episode_index),
                "episode_reward": str(-1000 + episode_index),
                "tracking_error": "10.0",
                "energy_tracking_error": "1.0",
                "power_tracker_violation": "0.0",
                "required_values_finite": "true",
            }
            if kind == "diagnostic":
                row.update(
                    {
                        "upper_bound_active_ev_action_fraction": "0.2",
                        "mapping_validation": "pass",
                        "canonical_reconciliation": "pass",
                        "service_reconciliation": "pass",
                        "energy_reconciliation": "pass",
                    }
                )
            rows.append(row)
    return rows


def test_formal_matrix_is_exact_40_unique_transformer_ev_cells():
    workflow = _workflow()
    cells = workflow.formal_matrix()

    assert len(cells) == 40
    assert [cell.task_id for cell in cells] == list(range(40))
    assert {cell.algorithm for cell in cells} == {"hierarchical_transformer_ev"}
    assert len({(cell.scale, cell.algorithm, cell.seed) for cell in cells}) == 40
    assert [(cell.task_id, cell.scale, cell.seed) for cell in cells[:10]] == [
        (seed, "25cp", seed) for seed in range(10)
    ]
    assert [(cell.task_id, cell.scale, cell.seed) for cell in cells[10:20]] == [
        (10 + seed, "100cp", seed) for seed in range(10)
    ]
    assert [(cell.task_id, cell.scale, cell.seed) for cell in cells[20:30]] == [
        (20 + seed, "500cp", seed) for seed in range(10)
    ]
    assert [(cell.task_id, cell.scale, cell.seed) for cell in cells[30:40]] == [
        (30 + seed, "1000cp", seed) for seed in range(10)
    ]


def test_formal_protocol_and_configs_are_fixed():
    workflow = _workflow()
    cell = workflow.resolve_formal_cell(39)

    assert cell.scale == "1000cp"
    assert cell.algorithm == "hierarchical_transformer_ev"
    assert cell.seed == 9
    assert cell.config_path == "config_files/PublicPST_1000.yaml"
    assert cell.training_steps == 75000
    assert cell.evaluation_cadence == 5000
    assert cell.training_evaluation_episodes == 5
    assert cell.expected_scheduled_evaluations == 15
    assert cell.actor_output_transform == "transformer_ev_hierarchy_v1"
    assert workflow.scheduled_evaluation_steps() == tuple(range(5000, 75001, 5000))


def test_training_commands_use_only_reduced_hierarchy_protocol():
    workflow = _workflow()
    representative_tasks = (0, 9, 10, 19, 20, 29, 30, 39)
    for task_id in representative_tasks:
        cell = workflow.resolve_formal_cell(task_id)
        command = workflow.training_command(
            cell,
            save_dir=f"/scratch/train/task{task_id}",
            run_name=f"transformer_ev_{cell.scale}_seed{cell.seed}",
        )
        assert "--algorithm" in command
        assert command[command.index("--algorithm") + 1] == "hierarchical_transformer_ev"
        assert command[command.index("--max_timesteps") + 1] == "75000"
        assert command[command.index("--start_timesteps") + 1] == "1000"
        assert command[command.index("--eval_freq") + 1] == "5000"
        assert command[command.index("--eval_episodes") + 1] == "5"
        assert command[command.index("--config") + 1] == cell.config_path
        assert command[command.index("--seed") + 1] == str(cell.seed)
        assert not any(token == "actiongnn_nonnegative" for token in command)
        assert not any(token == "hierarchical" for token in command)


def test_smoke_matrix_is_four_scale_seed0_gate():
    workflow = _workflow()
    smoke = workflow.smoke_matrix()

    assert len(smoke) == 4
    assert [(cell.task_id, cell.scale, cell.seed, cell.training_steps) for cell in smoke] == [
        (0, "25cp", 0, 512),
        (1, "100cp", 0, 512),
        (2, "500cp", 0, 512),
        (3, "1000cp", 0, 512),
    ]
    for cell in smoke:
        command = workflow.training_command(cell, save_dir="/scratch/smoke", run_name="smoke")
        assert command[command.index("--algorithm") + 1] == "hierarchical_transformer_ev"
        assert command[command.index("--max_timesteps") + 1] == "512"
        assert command[command.index("--start_timesteps") + 1] == "64"
        assert command[command.index("--eval_freq") + 1] == "256"
        assert command[command.index("--eval_episodes") + 1] == "1"


def test_eval30_and_diagnostic_commands_are_independent_post_training_consumers():
    workflow = _workflow()
    cell = workflow.resolve_formal_cell(20)
    eval_command = workflow.eval30_command(
        cell,
        checkpoint_prefix="/stage/task20/checkpoint/model.best",
        output_csv="/eval/task20.csv",
        run_name="eval30",
        max_episode_steps=112,
        eval_seed_offset=730000,
    )
    diagnostic_command = workflow.diagnostic_command(
        cell,
        checkpoint_prefix="/stage/task20/checkpoint/model.best",
        output_dir="/diag/task20",
        run_name="diag",
        max_episode_steps=112,
        eval_seed_offset=730000,
        matrix_job_id="34567",
    )

    assert eval_command[eval_command.index("--algorithm") + 1] == "hierarchical_transformer_ev"
    assert eval_command[eval_command.index("--eval_episodes") + 1] == "30"
    assert diagnostic_command[diagnostic_command.index("--algorithm") + 1] == "hierarchical_transformer_ev"
    assert diagnostic_command[diagnostic_command.index("--scale") + 1] == "500cp"
    assert diagnostic_command[diagnostic_command.index("--eval_episodes") + 1] == "30"


def test_gates_require_exact_40_training_cells_and_1200_episode_rows():
    workflow = _workflow()
    training = workflow.validate_training_gate(_complete_training_rows())
    eval30 = workflow.validate_eval30_gate(_complete_episode_rows("eval"))
    diagnostic = workflow.validate_diagnostic_gate(_complete_episode_rows("diagnostic"))

    assert training.status == "PASS"
    assert training.cell_count == 40
    assert eval30.episode_count == 1200
    assert diagnostic.episode_count == 1200

    with pytest.raises(ValueError, match="40/40"):
        workflow.validate_training_gate(_complete_training_rows()[:-1])
    bad_eval = _complete_episode_rows("eval")[:-1]
    with pytest.raises(ValueError, match="1,200/1,200"):
        workflow.validate_eval30_gate(bad_eval)


def test_resource_profile_fails_closed_until_reviewed_values_exist(tmp_path):
    workflow = _workflow()
    profile = tmp_path / "resource.env"
    profile.write_text("RESOURCE_PROFILE_APPROVED=NO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="approved resource profile"):
        workflow.load_resource_profile(profile, scope="all")

    profile.write_text(
        "\n".join(
            [
                "RESOURCE_PROFILE_APPROVED=YES",
                "SMOKE_CPUS_PER_TASK=1",
                "SMOKE_MEM=4G",
                "SMOKE_TIME=00:30:00",
                "TRAIN_SMALL_CPUS_PER_TASK=1",
                "TRAIN_SMALL_MEM=8G",
                "TRAIN_SMALL_TIME=08:00:00",
                "TRAIN_500CP_CPUS_PER_TASK=1",
                "TRAIN_500CP_MEM=16G",
                "TRAIN_500CP_TIME=12:00:00",
                "TRAIN_1000CP_CPUS_PER_TASK=1",
                "TRAIN_1000CP_MEM=24G",
                "TRAIN_1000CP_TIME=18:00:00",
                "EVAL_CPUS_PER_TASK=1",
                "EVAL_MEM=4G",
                "EVAL_TIME=02:00:00",
                "DIAGNOSTIC_CPUS_PER_TASK=1",
                "DIAGNOSTIC_MEM=8G",
                "DIAGNOSTIC_TIME=04:00:00",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert workflow.load_resource_profile(profile, scope="all")["RESOURCE_PROFILE_APPROVED"] == "YES"


def test_cli_print_matrix_outputs_all_40_rows_from_unrelated_cwd(tmp_path):
    result = subprocess.run(
        [sys.executable, str(WORKFLOW_SCRIPT), "--print-matrix"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 40
    assert lines[0] == (
        "task_id=0 scale=25cp algorithm=hierarchical_transformer_ev seed=0 "
        "config=config_files/PublicPST_25cp.yaml training_steps=75000"
    )
    assert lines[-1] == (
        "task_id=39 scale=1000cp algorithm=hierarchical_transformer_ev seed=9 "
        "config=config_files/PublicPST_1000.yaml training_steps=75000"
    )
