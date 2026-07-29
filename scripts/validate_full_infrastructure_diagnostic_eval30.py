#!/usr/bin/env python3
"""Validation contracts for the full per-infrastructure diagnostic eval30 workflow."""

from __future__ import annotations

import argparse
from typing import Final


TRAINING_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
EVAL_EPISODES: Final[int] = 30

SCALE_SEED_OFFSETS: Final[dict[str, int]] = {
    "25cp": 710000,
    "100cp": 720000,
    "500cp": 730000,
    "1000cp": 740000,
}

STAGE_D_TASKS: Final[dict[int, dict[str, object]]] = {
    0: {"scale": "25cp", "algorithm": "actiongnn", "formal_start": 0},
    1: {"scale": "25cp", "algorithm": "hierarchical", "formal_start": 5},
    2: {"scale": "100cp", "algorithm": "actiongnn", "formal_start": 10},
    3: {"scale": "100cp", "algorithm": "hierarchical", "formal_start": 15},
    4: {"scale": "500cp", "algorithm": "actiongnn", "formal_start": 20},
    5: {"scale": "500cp", "algorithm": "hierarchical", "formal_start": 25},
    6: {"scale": "1000cp", "algorithm": "actiongnn", "formal_start": 30},
    7: {"scale": "1000cp", "algorithm": "hierarchical", "formal_start": 35},
}


def stage_d_task(task_id: int) -> dict[str, object]:
    """Return a defensive copy of one exact Stage D task mapping."""
    if type(task_id) is not int or task_id not in STAGE_D_TASKS:
        raise ValueError(f"unsupported Stage D task ID: {task_id!r}")
    return dict(STAGE_D_TASKS[task_id])


def formal_task_id(task_id: int, training_seed: int) -> int:
    """Resolve the exact formal-job task ID for one Stage D task and seed."""
    if type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed: {training_seed!r}")
    return int(stage_d_task(task_id)["formal_start"]) + training_seed


def episode_seed(scale: str, training_seed: int, episode_index: int) -> int:
    """Resolve the deterministic episode seed used by canonical eval30."""
    if scale not in SCALE_SEED_OFFSETS:
        raise ValueError(f"unsupported scale: {scale!r}")
    if type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed: {training_seed!r}")
    if type(episode_index) is not int or not 0 <= episode_index < EVAL_EPISODES:
        raise ValueError(f"episode index must be 0..{EVAL_EPISODES - 1}")
    return SCALE_SEED_OFFSETS[scale] + training_seed + episode_index


def task_mapping_lines(task_id: int) -> list[str]:
    """Render the stable shell-readable task-mapping contract."""
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    formal_ids = ",".join(
        str(formal_task_id(task_id, training_seed))
        for training_seed in TRAINING_SEEDS
    )
    return [
        f"task_id={task_id}",
        f"scale={scale}",
        f"algorithm={task['algorithm']}",
        "training_seeds=" + ",".join(map(str, TRAINING_SEEDS)),
        f"formal_task_ids={formal_ids}",
        f"eval_episodes={EVAL_EPISODES}",
        f"eval_seed_offset={SCALE_SEED_OFFSETS[scale]}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate contracts for the full per-infrastructure deterministic "
            "eval30 workflow."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mapping_parser = subparsers.add_parser(
        "task-mapping",
        help="Print one exact scale-algorithm and formal-checkpoint mapping.",
    )
    mapping_parser.add_argument("--task-id", required=True, type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "task-mapping":
        for line in task_mapping_lines(args.task_id):
            print(line)
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
