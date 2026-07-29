import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAY_SCRIPT = PROJECT_ROOT / "m3_jobs" / "21_full_infrastructure_diagnostic_eval30.slurm"


def test_array_slurm_resources_and_topology():
    text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-7" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=32G" in text
    assert "#SBATCH --time=06:00:00" in text
    assert "--device cpu" in text
    assert "--eval_episodes 30" in text
    assert "model.best" in text
    assert "model.last" not in text


def test_array_slurm_uses_job_scoped_run_root():
    text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    assert 'job${SLURM_ARRAY_JOB_ID}/task${SLURM_ARRAY_TASK_ID}' in text


def test_array_slurm_has_atomic_task_publication():
    text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    assert ".tmp.${SLURM_JOB_ID}" in text
    assert "validate_stage_d_task_package" in text
    assert "mv " in text


@pytest.mark.parametrize(
    ("task_id", "scale", "algorithm", "formal_ids", "seed_offset"),
    [
        (0, "25cp", "actiongnn", "0,1,2,3,4", "710000"),
        (1, "25cp", "hierarchical", "5,6,7,8,9", "710000"),
        (2, "100cp", "actiongnn", "10,11,12,13,14", "720000"),
        (3, "100cp", "hierarchical", "15,16,17,18,19", "720000"),
        (4, "500cp", "actiongnn", "20,21,22,23,24", "730000"),
        (5, "500cp", "hierarchical", "25,26,27,28,29", "730000"),
        (6, "1000cp", "actiongnn", "30,31,32,33,34", "740000"),
        (7, "1000cp", "hierarchical", "35,36,37,38,39", "740000"),
    ],
)
def test_array_slurm_dry_run_exact_mapping(
    task_id,
    scale,
    algorithm,
    formal_ids,
    seed_offset,
):
    env = os.environ.copy()
    env.update(
        {
            "SLURM_ARRAY_TASK_ID": str(task_id),
            "SLURM_ARRAY_JOB_ID": "99999999",
            "SLURM_JOB_ID": f"9999999{task_id}",
            "EV_GNN_FULL_DIAGNOSTIC_DRY_RUN": "1",
            "EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT": str(PROJECT_ROOT),
            "EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT": "f" * 40,
            "EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID": "58513929",
            "EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT": "/formal",
            "EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE": "/formal/complete.tar.gz",
            "EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT": "/output",
            "EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT": "/run",
        }
    )
    result = subprocess.run(
        ["bash", str(ARRAY_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert f"task_id={task_id}" in lines
    assert f"scale={scale}" in lines
    assert f"algorithm={algorithm}" in lines
    assert f"formal_task_ids={formal_ids}" in lines
    assert f"eval_seed_offset={seed_offset}" in lines
    assert "training_seeds=0,1,2,3,4" in lines
    assert "eval_episodes=30" in lines
    assert "expected_episode_count=150" in lines
    assert sum(line.startswith("EVALUATOR_COMMAND_SEED_") for line in lines) == 5
    assert "DRY_RUN_NO_EVALUATION_OR_PACKAGING" in lines
