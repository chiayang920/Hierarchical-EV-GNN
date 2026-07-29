import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    EVAL_EPISODES,
    TRAINING_SEEDS,
    episode_seed,
    formal_task_id,
    stage_d_task,
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
    assert episode_seed("25cp", 4, 29) == 710033
    assert episode_seed("1000cp", 4, 29) == 740033


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
                "eval_seed_offset=710000",
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
                "eval_seed_offset=740000",
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
