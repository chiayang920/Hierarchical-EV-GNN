import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPT = PROJECT_ROOT / "scripts" / "transformer_constraint_40cell_workflow.py"


def _workflow():
    from scripts import transformer_constraint_40cell_workflow as workflow

    return workflow


def test_transformer_constraint_matrix_is_exact_40_unique_new_controller_cells():
    workflow = _workflow()
    cells = workflow.formal_matrix()

    assert len(cells) == 40
    assert len({(cell.scale, cell.algorithm, cell.seed) for cell in cells}) == 40
    assert [cell.task_id for cell in cells] == list(range(40))
    assert {cell.algorithm for cell in cells} == {"hierarchical_transformer_constraint"}
    assert [(cell.task_id, cell.scale, cell.seed) for cell in cells[:10]] == [
        (seed, "25cp", seed) for seed in range(10)
    ]
    assert [(cell.task_id, cell.scale, cell.seed) for cell in cells[30:40]] == [
        (30 + seed, "1000cp", seed) for seed in range(10)
    ]


@pytest.mark.parametrize(
    "algorithm",
    ["actiongnn", "actiongnn_nonnegative", "hierarchical", "hierarchical_transformer_ev"],
)
def test_transformer_constraint_workflow_rejects_non_intervention_algorithms(algorithm):
    workflow = _workflow()

    with pytest.raises(ValueError):
        workflow.validate_algorithm(algorithm)


def test_formal_commands_pin_training_and_eval_seed_contracts():
    workflow = _workflow()
    cell = workflow.resolve_formal_cell(23)

    train_command = workflow.training_command(
        cell,
        save_dir="/scratch/train/task23",
        run_name="constraint_500cp_seed3",
    )
    eval_command = workflow.eval30_command(
        cell,
        checkpoint_prefix="/stage/task23/checkpoint/model.best",
        output_csv="/eval/task23.csv",
        run_name="eval30",
        max_episode_steps=112,
        eval_seed_offset=workflow.eval_seed_offset(cell),
    )
    diagnostic_command = workflow.projection_diagnostic_command(
        cell,
        checkpoint_prefix="/stage/task23/checkpoint/model.best",
        output_dir="/projection/task23",
        run_name="projection",
        max_episode_steps=112,
        eval_seed_offset=workflow.eval_seed_offset(cell),
        matrix_job_id="34567",
    )

    assert train_command[train_command.index("--algorithm") + 1] == "hierarchical_transformer_constraint"
    assert train_command[train_command.index("--max_timesteps") + 1] == "75000"
    assert train_command[train_command.index("--start_timesteps") + 1] == "1000"
    assert eval_command[eval_command.index("--eval_episodes") + 1] == "30"
    assert eval_command[eval_command.index("--eval_seed_offset") + 1] == "732997"
    assert "model.last" not in eval_command
    assert diagnostic_command[0:2] == ["python", "evaluate_transformer_constraint_projection_diagnostics.py"]
    assert diagnostic_command[diagnostic_command.index("--eval_seed_offset") + 1] == "732997"
    assert cell.seed + int(eval_command[eval_command.index("--eval_seed_offset") + 1]) == 733000


def test_eval_and_diagnostic_commands_reject_model_last():
    workflow = _workflow()
    cell = workflow.resolve_formal_cell(0)

    with pytest.raises(ValueError, match="model.best"):
        workflow.eval30_command(
            cell,
            checkpoint_prefix="/stage/task0/checkpoint/model.last",
            output_csv="/eval/task0.csv",
            run_name="eval30",
            max_episode_steps=112,
            eval_seed_offset=710000,
        )
    with pytest.raises(ValueError, match="model.best"):
        workflow.diagnostic_command(
            cell,
            checkpoint_prefix="/stage/task0/checkpoint/model.last",
            output_dir="/diag/task0",
            run_name="diag",
            max_episode_steps=112,
            eval_seed_offset=710000,
            matrix_job_id="34567",
        )


def test_dry_run_summary_reports_matrix_and_historical_path_safety():
    workflow = _workflow()
    lines = workflow.dry_run_summary()

    assert "FORMAL_MATRIX_CELL_COUNT=40" in lines
    assert "FORMAL_MATRIX_UNIQUE_CELL_COUNT=40" in lines
    assert "FORMAL_ALGORITHM_SET=hierarchical_transformer_constraint" in lines
    assert "HISTORICAL_FORMAL75K_OUTPUT_WRITE_ALLOWED=NO" in lines
    assert "DRY_RUN_NO_JOBS_SUBMITTED=YES" in lines


def test_runtime_gate_commands_cover_all_scales_after_start_timesteps():
    workflow = _workflow()
    cells = workflow.runtime_gate_matrix()

    assert [cell.scale for cell in cells] == ["25cp", "100cp", "500cp", "1000cp"]
    for cell in cells:
        command = workflow.runtime_gate_command(cell, save_dir="/tmp/runtime", run_name="runtime")
        assert command[command.index("--algorithm") + 1] == "hierarchical_transformer_constraint"
        assert int(command[command.index("--start_timesteps") + 1]) == 1
        assert int(command[command.index("--max_timesteps") + 1]) > 1
        assert command[command.index("--batch_size") + 1] == "64"


def test_cli_print_matrix_runs_from_unrelated_working_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(WORKFLOW_SCRIPT), "--print-matrix"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 40
    assert "hierarchical_transformer_constraint" in result.stdout
