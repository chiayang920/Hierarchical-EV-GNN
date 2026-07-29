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


import csv
import json
import math
from pathlib import Path
from typing import TypeAlias


EpisodeKey: TypeAlias = tuple[str, str, int, int]

TOPOLOGY: Final[dict[str, tuple[int, int]]] = {
    "25cp": (25, 3),
    "100cp": (100, 7),
    "500cp": (500, 35),
    "1000cp": (1000, 70),
}

ALGORITHMS: Final[tuple[str, ...]] = ("actiongnn", "hierarchical")
SCHEMA_VERSION: Final[str] = "3"

EPISODE_REQUIRED_COLUMNS: Final[set[str]] = {
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
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
    "global_action_fraction_at_max_active",
    "diagnostic_schema_version",
}

EPISODE_REQUIRED_NUMERIC_FIELDS: Final[tuple[str, ...]] = (
    "episode_index",
    "episode_seed",
    "episode_steps",
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
    "global_action_fraction_at_max_active",
)

SEED_SUMMARY_REQUIRED_COLUMNS: Final[set[str]] = {
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "n_eval_episodes",
    "diagnostic_schema_version",
}

INFRASTRUCTURE_REQUIRED_COLUMNS: Final[set[str]] = {
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "diagnostic_schema_version",
}

SERVICE_STATUS_FIELDS: Final[tuple[str, ...]] = (
    "served_count_reconciliation_status",
    "satisfaction_sum_reconciliation_status",
    "charged_energy_reconciliation_status",
    "discharged_energy_reconciliation_status",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _require_columns(
    fieldnames: list[str],
    required_columns: set[str],
    label: str,
) -> None:
    missing = sorted(required_columns - set(fieldnames))
    if missing:
        raise ValueError(
            f"{label} missing required column(s): {', '.join(missing)}"
        )


def _finite_number(value: object, field: str) -> float:
    if value in ("", None):
        raise ValueError(f"blank required metric: {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric value for {field}: {value!r}")
    return parsed


def _exact_integer(value: object, field: str) -> int:
    parsed = _finite_number(value, field)
    if not parsed.is_integer():
        raise ValueError(f"{field} must be integral: {value!r}")
    return int(parsed)


def _validate_identity(
    rows: list[dict[str, str]],
    *,
    scale: str,
    algorithm: str,
    training_seed: int,
    label: str,
) -> None:
    for row_index, row in enumerate(rows):
        observed = (
            row.get("scale"),
            row.get("algorithm"),
            _exact_integer(row.get("training_seed"), "training_seed"),
        )
        expected = (scale, algorithm, training_seed)
        if observed != expected:
            raise ValueError(
                f"{label} row {row_index} identity mismatch: "
                f"expected={expected}, observed={observed}"
            )
        if row.get("diagnostic_schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{label} row {row_index} diagnostic schema must be "
                f"{SCHEMA_VERSION}"
            )


def expected_episode_keys() -> set[EpisodeKey]:
    return {
        (scale, algorithm, training_seed, episode_index)
        for scale in SCALE_SEED_OFFSETS
        for algorithm in ALGORITHMS
        for training_seed in TRAINING_SEEDS
        for episode_index in range(EVAL_EPISODES)
    }


def validate_episode_inventory(rows: list[dict[str, str]]) -> None:
    observed: set[EpisodeKey] = set()

    for row_number, row in enumerate(rows, start=2):
        scale = str(row.get("scale", ""))
        algorithm = str(row.get("algorithm", ""))
        training_seed = _exact_integer(row.get("training_seed"), "training_seed")
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        key = (scale, algorithm, training_seed, episode_index)

        if key in observed:
            raise ValueError(f"duplicate episode key at CSV row {row_number}: {key}")
        observed.add(key)

        expected_seed = episode_seed(scale, training_seed, episode_index)
        observed_seed = _exact_integer(row.get("episode_seed"), "episode_seed")
        if observed_seed != expected_seed:
            raise ValueError(
                f"episode seed mismatch for {key}: "
                f"expected={expected_seed}, observed={observed_seed}"
            )

        if row.get("diagnostic_schema_version") != SCHEMA_VERSION:
            raise ValueError(f"diagnostic schema must be {SCHEMA_VERSION} for {key}")

    expected = expected_episode_keys()
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    if missing:
        raise ValueError(f"missing episode key: {missing[0]}")
    if unexpected:
        raise ValueError(f"unexpected episode key: {unexpected[0]}")


def _validate_per_seed_episode_rows(
    rows: list[dict[str, str]],
    *,
    scale: str,
    algorithm: str,
    training_seed: int,
) -> None:
    if len(rows) != EVAL_EPISODES:
        raise ValueError(
            f"episode diagnostics must contain exactly {EVAL_EPISODES} rows, "
            f"got {len(rows)}"
        )

    _validate_identity(
        rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        label="episode diagnostics",
    )

    observed_indices: set[int] = set()
    for row_index, row in enumerate(rows):
        for field in EPISODE_REQUIRED_NUMERIC_FIELDS:
            _finite_number(row.get(field), field)

        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        if episode_index in observed_indices:
            raise ValueError(f"duplicate episode index: {episode_index}")
        observed_indices.add(episode_index)

        expected_seed_value = episode_seed(scale, training_seed, episode_index)
        observed_seed = _exact_integer(row.get("episode_seed"), "episode_seed")
        if observed_seed != expected_seed_value:
            raise ValueError(
                f"episode seed mismatch for episode index {episode_index}: "
                f"expected={expected_seed_value}, observed={observed_seed}"
            )

    expected_indices = set(range(EVAL_EPISODES))
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)
        unexpected = sorted(observed_indices - expected_indices)
        raise ValueError(
            f"episode index inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_infrastructure_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    scale: str,
    algorithm: str,
    training_seed: int,
    infrastructure_label: str,
    expected_rows_per_episode: int,
) -> None:
    _require_columns(
        fieldnames,
        INFRASTRUCTURE_REQUIRED_COLUMNS,
        infrastructure_label,
    )
    expected_total = expected_rows_per_episode * EVAL_EPISODES
    if len(rows) != expected_total:
        raise ValueError(
            f"{infrastructure_label} must contain exactly {expected_total} rows, "
            f"got {len(rows)}"
        )

    _validate_identity(
        rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        label=infrastructure_label,
    )

    episode_counts = {episode_index: 0 for episode_index in range(EVAL_EPISODES)}
    for row_index, row in enumerate(rows):
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        observed_seed = _exact_integer(row.get("episode_seed"), "episode_seed")
        expected_seed_value = episode_seed(scale, training_seed, episode_index)
        if observed_seed != expected_seed_value:
            raise ValueError(
                f"{infrastructure_label} row {row_index} episode seed mismatch"
            )
        if episode_index not in episode_counts:
            raise ValueError(
                f"{infrastructure_label} row {row_index} invalid episode index: "
                f"{episode_index}"
            )
        episode_counts[episode_index] += 1

    invalid_counts = {
        episode_index: count
        for episode_index, count in episode_counts.items()
        if count != expected_rows_per_episode
    }
    if invalid_counts:
        raise ValueError(
            f"{infrastructure_label} per-episode row count mismatch: "
            f"{invalid_counts}"
        )


def _validate_status_csv(
    path: Path,
    *,
    label: str,
    required_status_fields: tuple[str, ...],
) -> int:
    fieldnames, rows = _read_csv(path)
    _require_columns(fieldnames, set(required_status_fields), label)
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    for row_index, row in enumerate(rows):
        for field in required_status_fields:
            if row.get(field) != "pass":
                raise ValueError(
                    f"{label} row {row_index} failed {field}: {row.get(field)!r}"
                )
    return len(rows)


def validate_seed_output_directory(
    path: Path,
    task_id: int,
    training_seed: int,
) -> dict[str, object]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"seed output directory does not exist: {root}")

    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed: {training_seed!r}")

    diagnostics = root / "diagnostics"
    validation = root / "validation"
    logs = root / "logs"

    episode_path = diagnostics / "episode_diagnostics.csv"
    seed_summary_path = diagnostics / "seed_summary_diagnostics.csv"
    transformer_path = diagnostics / "transformer_diagnostics.csv"
    charger_path = diagnostics / "charger_diagnostics.csv"

    episode_fields, episode_rows = _read_csv(episode_path)
    _require_columns(
        episode_fields,
        EPISODE_REQUIRED_COLUMNS,
        "episode diagnostics",
    )
    _validate_per_seed_episode_rows(
        episode_rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
    )

    seed_fields, seed_rows = _read_csv(seed_summary_path)
    _require_columns(
        seed_fields,
        SEED_SUMMARY_REQUIRED_COLUMNS,
        "seed summary diagnostics",
    )
    if len(seed_rows) != 1:
        raise ValueError(
            f"seed summary diagnostics must contain exactly one row, "
            f"got {len(seed_rows)}"
        )
    _validate_identity(
        seed_rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        label="seed summary diagnostics",
    )
    if _exact_integer(
        seed_rows[0].get("n_eval_episodes"),
        "n_eval_episodes",
    ) != EVAL_EPISODES:
        raise ValueError(
            f"seed summary n_eval_episodes must equal {EVAL_EPISODES}"
        )

    expected_chargers, expected_transformers = TOPOLOGY[scale]
    transformer_fields, transformer_rows = _read_csv(transformer_path)
    charger_fields, charger_rows = _read_csv(charger_path)

    _validate_infrastructure_rows(
        transformer_rows,
        transformer_fields,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        infrastructure_label="transformer diagnostics",
        expected_rows_per_episode=expected_transformers,
    )
    _validate_infrastructure_rows(
        charger_rows,
        charger_fields,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        infrastructure_label="charger diagnostics",
        expected_rows_per_episode=expected_chargers,
    )

    canonical_rows = _validate_status_csv(
        validation / "canonical_reconciliation.csv",
        label="canonical reconciliation",
        required_status_fields=("status",),
    )
    service_rows = _validate_status_csv(
        validation / "service_reconciliation.csv",
        label="service reconciliation",
        required_status_fields=SERVICE_STATUS_FIELDS,
    )

    mapping_path = validation / "mapping_validation.json"
    if not mapping_path.is_file():
        raise ValueError(f"missing mapping validation JSON: {mapping_path}")
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("mapping validation JSON is invalid") from exc
    if not isinstance(mapping, dict) or mapping.get("status") != "ok":
        raise ValueError("mapping validation status must be ok")

    stderr_path = logs / "stderr.log"
    if not stderr_path.is_file():
        raise ValueError(f"missing stderr log: {stderr_path}")
    if stderr_path.read_bytes():
        raise ValueError("stderr log must be empty")

    return {
        "status": "ok",
        "task_id": task_id,
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": training_seed,
        "formal_task_id": formal_task_id(task_id, training_seed),
        "episode_count": len(episode_rows),
        "transformer_row_count": len(transformer_rows),
        "charger_row_count": len(charger_rows),
        "canonical_reconciliation_rows": canonical_rows,
        "service_reconciliation_rows": service_rows,
        "diagnostic_schema_version": SCHEMA_VERSION,
    }


def _read_csv_rows_for_cli(path: Path) -> list[dict[str, str]]:
    _, rows = _read_csv(path)
    return rows


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

    inventory_parser = subparsers.add_parser(
        "episode-inventory",
        help="Validate the exact 1,200-episode matrix in a CSV.",
    )
    inventory_parser.add_argument("--csv", required=True, type=Path)

    seed_parser = subparsers.add_parser(
        "validate-seed-output",
        help="Validate one 30-episode seed output directory.",
    )
    seed_parser.add_argument("--directory", required=True, type=Path)
    seed_parser.add_argument("--task-id", required=True, type=int)
    seed_parser.add_argument("--training-seed", required=True, type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "task-mapping":
        for line in task_mapping_lines(args.task_id):
            print(line)
        return 0

    if args.command == "episode-inventory":
        rows = _read_csv_rows_for_cli(args.csv)
        validate_episode_inventory(rows)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "episodes": len(rows),
                    "checkpoint_groups": 40,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "validate-seed-output":
        result = validate_seed_output_directory(
            args.directory,
            args.task_id,
            args.training_seed,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
