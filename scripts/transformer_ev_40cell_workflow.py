#!/usr/bin/env python3
"""Local contracts for the transformer-EV reduced-hierarchy 40-cell workflow."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))


SCALES = ("25cp", "100cp", "500cp", "1000cp")
SEEDS = tuple(range(10))
ALGORITHM = "hierarchical_transformer_ev"
FORMAL_ALGORITHMS = (ALGORITHM,)
MODEL_NAMES = {
    ALGORITHM: (
        "Transformer-to-EV Hierarchical TD3 EV-GNN with Charger-Level "
        "Decision Allocation Removed"
    ),
}
FORMAL_TRAINING_STEPS = 75000
FORMAL_EVALUATION_CADENCE = 5000
FORMAL_TRAINING_EVALUATION_EPISODES = 5
FORMAL_SCHEDULED_EVALUATIONS = 15
FORMAL_MODEL_BEST_EVAL_EPISODES = 30
EXPECTED_FORMAL_CELLS = 40
EXPECTED_FORMAL_EVAL_ROWS = 1200
SMOKE_TRAINING_STEPS = 512
SMOKE_START_TIMESTEPS = 64
SMOKE_EVALUATION_CADENCE = 256
SMOKE_TRAINING_EVALUATION_EPISODES = 1
START_TIMESTEPS = 1000
BATCH_SIZE = 64
SMOKE_BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
SMOKE_REPLAY_BUFFER_SIZE = 5000
DEVICE = "cpu"
LOG_TO_WANDB = "false"
ACTOR_OUTPUT_TRANSFORM = "transformer_ev_hierarchy_v1"
KEY_MECHANISM_METRIC = "upper_bound_active_ev_action_fraction"
CHECKPOINT_SELECTION_RULE = (
    "model.best selected by strict improvement of scheduled internal eval mean reward "
    "at intervals recorded in eval_frequency within the configured training budget; "
    "model.last is saved at the configured final step but is not used for canonical "
    "eval30 or diagnostics."
)

CONFIG_BY_SCALE = {
    "25cp": "config_files/PublicPST_25cp.yaml",
    "100cp": "config_files/PublicPST_100.yaml",
    "500cp": "config_files/PublicPST_500.yaml",
    "1000cp": "config_files/PublicPST_1000.yaml",
}
EVAL_SEED_BASE_BY_SCALE = {
    "25cp": 710000,
    "100cp": 720000,
    "500cp": 730000,
    "1000cp": 740000,
}
TRAINING_REQUIRED_TRUE_FIELDS = (
    "model_best_present",
    "runtime_args_present",
    "package_complete",
    "fresh_run",
    "required_values_finite",
)
DIAGNOSTIC_REQUIRED_PASS_FIELDS = (
    "mapping_validation",
    "canonical_reconciliation",
    "service_reconciliation",
    "energy_reconciliation",
)


@dataclass(frozen=True)
class TransformerEVCell:
    task_id: int
    scale: str
    algorithm: str
    seed: int
    config_path: str
    training_steps: int = FORMAL_TRAINING_STEPS
    evaluation_cadence: int = FORMAL_EVALUATION_CADENCE
    training_evaluation_episodes: int = FORMAL_TRAINING_EVALUATION_EPISODES
    expected_scheduled_evaluations: int = FORMAL_SCHEDULED_EVALUATIONS
    actor_output_transform: str = ACTOR_OUTPUT_TRANSFORM
    checkpoint_selection_rule: str = CHECKPOINT_SELECTION_RULE


@dataclass(frozen=True)
class GateResult:
    status: str
    cell_count: int
    episode_count: int = 0
    missing_cells: list[tuple[str, str, int]] | None = None
    duplicate_cells: list[tuple[str, str, int]] | None = None


def scheduled_evaluation_steps() -> tuple[int, ...]:
    return tuple(
        range(
            FORMAL_EVALUATION_CADENCE,
            FORMAL_TRAINING_STEPS + 1,
            FORMAL_EVALUATION_CADENCE,
        )
    )


def validate_algorithm(algorithm: str) -> str:
    if algorithm in {"actiongnn", "actiongnn_nonnegative", "hierarchical"}:
        raise ValueError(f"frozen reference algorithm is not trained in this workflow: {algorithm!r}")
    if algorithm != ALGORITHM:
        raise ValueError(f"unsupported transformer-EV workflow algorithm: {algorithm!r}")
    return algorithm


def _make_cell(task_id: int, scale: str, seed: int, training_steps: int) -> TransformerEVCell:
    expected_cadence = FORMAL_EVALUATION_CADENCE if training_steps == FORMAL_TRAINING_STEPS else SMOKE_EVALUATION_CADENCE
    expected_episodes = (
        FORMAL_TRAINING_EVALUATION_EPISODES
        if training_steps == FORMAL_TRAINING_STEPS
        else SMOKE_TRAINING_EVALUATION_EPISODES
    )
    expected_count = (
        FORMAL_SCHEDULED_EVALUATIONS
        if training_steps == FORMAL_TRAINING_STEPS
        else max(1, training_steps // SMOKE_EVALUATION_CADENCE)
    )
    return TransformerEVCell(
        task_id=task_id,
        scale=scale,
        algorithm=ALGORITHM,
        seed=seed,
        config_path=CONFIG_BY_SCALE[scale],
        training_steps=training_steps,
        evaluation_cadence=expected_cadence,
        training_evaluation_episodes=expected_episodes,
        expected_scheduled_evaluations=expected_count,
    )


def formal_matrix() -> tuple[TransformerEVCell, ...]:
    cells: list[TransformerEVCell] = []
    task_id = 0
    for scale in SCALES:
        for seed in SEEDS:
            cells.append(_make_cell(task_id, scale, seed, FORMAL_TRAINING_STEPS))
            task_id += 1
    return tuple(cells)


def smoke_matrix() -> tuple[TransformerEVCell, ...]:
    return tuple(_make_cell(task_id, scale, 0, SMOKE_TRAINING_STEPS) for task_id, scale in enumerate(SCALES))


def resolve_formal_cell(task_id: int) -> TransformerEVCell:
    if type(task_id) is not int or not 0 <= task_id < EXPECTED_FORMAL_CELLS:
        raise ValueError(f"transformer-EV formal task ID must be 0..39; got {task_id!r}")
    return formal_matrix()[task_id]


def resolve_smoke_cell(task_id: int) -> TransformerEVCell:
    if type(task_id) is not int or not 0 <= task_id < len(SCALES):
        raise ValueError(f"transformer-EV smoke task ID must be 0..3; got {task_id!r}")
    return smoke_matrix()[task_id]


def format_cell_line(cell: TransformerEVCell) -> str:
    return (
        f"task_id={cell.task_id} scale={cell.scale} algorithm={cell.algorithm} "
        f"seed={cell.seed} config={cell.config_path} training_steps={cell.training_steps}"
    )


def training_command(cell: TransformerEVCell, save_dir: str, run_name: str) -> list[str]:
    is_formal = cell.training_steps == FORMAL_TRAINING_STEPS
    return [
        "python",
        "train_td3_gnn.py",
        "--algorithm",
        cell.algorithm,
        "--config",
        cell.config_path,
        "--seed",
        str(cell.seed),
        "--device",
        DEVICE,
        "--run_name",
        run_name,
        "--max_timesteps",
        str(cell.training_steps),
        "--start_timesteps",
        str(START_TIMESTEPS if is_formal else SMOKE_START_TIMESTEPS),
        "--eval_freq",
        str(cell.evaluation_cadence),
        "--eval_episodes",
        str(cell.training_evaluation_episodes),
        "--batch_size",
        str(BATCH_SIZE if is_formal else SMOKE_BATCH_SIZE),
        "--replay_buffer_size",
        str(REPLAY_BUFFER_SIZE if is_formal else SMOKE_REPLAY_BUFFER_SIZE),
        "--save_dir",
        save_dir,
        "--log_to_wandb",
        LOG_TO_WANDB,
    ]


def eval_seed_offset(cell: TransformerEVCell) -> int:
    return EVAL_SEED_BASE_BY_SCALE[cell.scale] + 999 * cell.seed


def eval30_command(
    cell: TransformerEVCell,
    checkpoint_prefix: str,
    output_csv: str,
    run_name: str,
    max_episode_steps: int,
    eval_seed_offset: int,
) -> list[str]:
    return [
        "python",
        "evaluate_td3_gnn.py",
        "--algorithm",
        cell.algorithm,
        "--config",
        cell.config_path,
        "--seed",
        str(cell.seed),
        "--eval_episodes",
        str(FORMAL_MODEL_BEST_EVAL_EPISODES),
        "--checkpoint",
        checkpoint_prefix,
        "--device",
        DEVICE,
        "--output_csv",
        output_csv,
        "--run_name",
        run_name,
        "--max_episode_steps",
        str(max_episode_steps),
        "--deterministic",
        "true",
        "--eval_expl_noise",
        "0.0",
        "--eval_seed_offset",
        str(eval_seed_offset),
    ]


def diagnostic_command(
    cell: TransformerEVCell,
    checkpoint_prefix: str,
    output_dir: str,
    run_name: str,
    max_episode_steps: int,
    eval_seed_offset: int,
    matrix_job_id: str,
) -> list[str]:
    return [
        "python",
        "evaluate_td3_gnn_infrastructure_diagnostics.py",
        "--algorithm",
        cell.algorithm,
        "--scale",
        cell.scale,
        "--config",
        cell.config_path,
        "--seed",
        str(cell.seed),
        "--eval_episodes",
        str(FORMAL_MODEL_BEST_EVAL_EPISODES),
        "--checkpoint",
        checkpoint_prefix,
        "--output_dir",
        output_dir,
        "--run_name",
        run_name,
        "--device",
        DEVICE,
        "--max_episode_steps",
        str(max_episode_steps),
        "--deterministic",
        "true",
        "--eval_expl_noise",
        "0.0",
        "--eval_seed_offset",
        str(eval_seed_offset),
        "--matrix_job_id",
        matrix_job_id,
    ]


def _expected_cell_keys() -> set[tuple[str, str, int]]:
    return {(cell.scale, cell.algorithm, cell.seed) for cell in formal_matrix()}


def _parse_int(value: object, field: str) -> int:
    text = "" if value is None else str(value).strip()
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer; got {value!r}") from exc
    if str(number) != text:
        raise ValueError(f"{field} must be an integer; got {value!r}")
    return number


def _parse_float(value: object, field: str) -> float:
    text = "" if value is None else str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return number


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _row_cell_key(row: dict[str, object]) -> tuple[str, str, int]:
    scale = str(row.get("scale", "")).strip().lower()
    algorithm = validate_algorithm(str(row.get("algorithm", "")).strip())
    seed = _parse_int(row.get("seed"), "seed")
    return scale, algorithm, seed


def _duplicates(keys: Iterable[tuple[str, str, int]]) -> tuple[tuple[str, str, int], ...]:
    seen: set[tuple[str, str, int]] = set()
    duplicates: list[tuple[str, str, int]] = []
    for key in keys:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return tuple(duplicates)


def _require_exact_cell_inventory(rows: Sequence[dict[str, object]], label: str) -> GateResult:
    keys = [_row_cell_key(row) for row in rows]
    observed = set(keys)
    expected = _expected_cell_keys()
    missing = tuple(sorted(expected - observed))
    duplicate = _duplicates(keys)
    if missing or duplicate or len(observed) != EXPECTED_FORMAL_CELLS:
        raise ValueError(
            f"{label} gate requires 40/40 unique transformer-EV cells; "
            f"observed={len(observed)} missing={len(missing)} duplicates={len(duplicate)}"
        )
    return GateResult(
        status="PASS",
        cell_count=len(observed),
        missing_cells=list(missing),
        duplicate_cells=list(duplicate),
    )


def _parse_scheduled_steps(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        raise ValueError("15 scheduled evaluation steps are required")
    separator = ";" if ";" in text else ","
    return tuple(_parse_int(part, "scheduled_evaluation_steps") for part in text.split(separator) if part)


def validate_training_gate(rows: Sequence[dict[str, object]]) -> GateResult:
    gate = _require_exact_cell_inventory(rows, "training")
    source_identities = {str(row.get("source_identity", "")).strip() for row in rows}
    bundle_identities = {str(row.get("source_bundle_identity", "")).strip() for row in rows}
    if len(source_identities) != 1 or "" in source_identities:
        raise ValueError("training gate requires one consistent non-empty source identity")
    if len(bundle_identities) != 1 or "" in bundle_identities:
        raise ValueError("training gate requires one consistent non-empty source-bundle identity")
    for row in rows:
        scale, algorithm, seed = _row_cell_key(row)
        context = f"{scale} {algorithm} seed {seed}"
        if _parse_int(row.get("training_steps"), f"{context} training_steps") != FORMAL_TRAINING_STEPS:
            raise ValueError(f"{context}: training_steps must be 75000")
        if _parse_int(row.get("evaluation_cadence"), f"{context} evaluation_cadence") != FORMAL_EVALUATION_CADENCE:
            raise ValueError(f"{context}: evaluation cadence must be 5000")
        if _parse_int(row.get("evaluation_episodes"), f"{context} evaluation_episodes") != FORMAL_TRAINING_EVALUATION_EPISODES:
            raise ValueError(f"{context}: training evaluation episodes must be 5")
        if _parse_int(row.get("expected_scheduled_evaluations"), f"{context} expected_scheduled_evaluations") != FORMAL_SCHEDULED_EVALUATIONS:
            raise ValueError(f"{context}: expected 15 scheduled evaluations")
        if _parse_scheduled_steps(row.get("scheduled_evaluation_steps")) != scheduled_evaluation_steps():
            raise ValueError(f"{context}: 15 scheduled evaluation steps must be 5000..75000")
        for field in TRAINING_REQUIRED_TRUE_FIELDS:
            if not _is_true(row.get(field)):
                raise ValueError(f"{context}: {field} must be true")
        if str(row.get("actor_output_transform", "")).strip() != ACTOR_OUTPUT_TRANSFORM:
            raise ValueError(f"{context}: actor transform identity mismatch")
        if str(row.get("checkpoint_selection_rule", "")).strip() != CHECKPOINT_SELECTION_RULE:
            raise ValueError(f"{context}: checkpoint selection rule mismatch")
    return gate


def _validate_episode_inventory(
    rows: Sequence[dict[str, object]],
    label: str,
    required_episode_count: int,
) -> GateResult:
    expected_episode_indexes = set(range(required_episode_count))
    keys = []
    cells = set()
    for row in rows:
        cell_key = _row_cell_key(row)
        episode_index = _parse_int(row.get("episode_index"), "episode_index")
        if episode_index not in expected_episode_indexes:
            raise ValueError(f"{label} gate episode index out of range: {episode_index}")
        keys.append((*cell_key, episode_index))
        cells.add(cell_key)
    expected_rows = EXPECTED_FORMAL_CELLS * required_episode_count
    if len(keys) != expected_rows or len(set(keys)) != len(keys):
        raise ValueError(
            f"{label} gate requires 1,200/1,200 episode rows; "
            f"observed_rows={len(keys)} unique_rows={len(set(keys))}"
        )
    missing_cells = tuple(sorted(_expected_cell_keys() - cells))
    if missing_cells or len(cells) != EXPECTED_FORMAL_CELLS:
        raise ValueError(f"{label} gate requires 40/40 checkpoints; missing={len(missing_cells)}")
    return GateResult(
        status="PASS",
        cell_count=len(cells),
        episode_count=len(keys),
        missing_cells=list(missing_cells),
        duplicate_cells=[],
    )


def validate_eval30_gate(rows: Sequence[dict[str, object]]) -> GateResult:
    gate = _validate_episode_inventory(rows, "transformer-EV eval30", FORMAL_MODEL_BEST_EVAL_EPISODES)
    for row in rows:
        context = f"{row.get('scale')} {row.get('algorithm')} seed {row.get('seed')} episode {row.get('episode_index')}"
        if str(row.get("checkpoint_role", "")).strip() != "model.best":
            raise ValueError(f"{context}: model.best identity is required")
        for field in ("episode_reward", "tracking_error", "energy_tracking_error", "power_tracker_violation"):
            _parse_float(row.get(field), f"{context} {field}")
        if not _is_true(row.get("required_values_finite")):
            raise ValueError(f"{context}: required finite fields flag failed")
    return gate


def validate_diagnostic_gate(rows: Sequence[dict[str, object]]) -> GateResult:
    gate = _validate_episode_inventory(rows, "transformer-EV diagnostic", FORMAL_MODEL_BEST_EVAL_EPISODES)
    for row in rows:
        context = f"{row.get('scale')} {row.get('algorithm')} seed {row.get('seed')} episode {row.get('episode_index')}"
        if str(row.get("checkpoint_role", "")).strip() != "model.best":
            raise ValueError(f"{context}: model.best identity is required")
        _parse_float(row.get(KEY_MECHANISM_METRIC), f"{context} {KEY_MECHANISM_METRIC}")
        for field in DIAGNOSTIC_REQUIRED_PASS_FIELDS:
            if str(row.get(field, "")).strip().lower() != "pass":
                raise ValueError(f"{context}: {field.replace('_', ' ')} failed")
        if not _is_true(row.get("required_values_finite")):
            raise ValueError(f"{context}: required finite fields flag failed")
    return gate


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path} line {line_number} is not key=value")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _resource_keys_by_scope() -> dict[str, tuple[str, ...]]:
    smoke = ("SMOKE_CPUS_PER_TASK", "SMOKE_MEM", "SMOKE_TIME")
    small = ("TRAIN_SMALL_CPUS_PER_TASK", "TRAIN_SMALL_MEM", "TRAIN_SMALL_TIME")
    medium = ("TRAIN_500CP_CPUS_PER_TASK", "TRAIN_500CP_MEM", "TRAIN_500CP_TIME")
    large = ("TRAIN_1000CP_CPUS_PER_TASK", "TRAIN_1000CP_MEM", "TRAIN_1000CP_TIME")
    eval_keys = ("EVAL_CPUS_PER_TASK", "EVAL_MEM", "EVAL_TIME")
    diagnostic = ("DIAGNOSTIC_CPUS_PER_TASK", "DIAGNOSTIC_MEM", "DIAGNOSTIC_TIME")
    train = (*small, *medium, *large)
    formal = (*train, *eval_keys, *diagnostic)
    return {
        "smoke": smoke,
        "train-small": small,
        "train-500cp": medium,
        "train-1000cp": large,
        "train": train,
        "eval": eval_keys,
        "diagnostic": diagnostic,
        "formal": formal,
        "all": (*smoke, *formal),
    }


def load_resource_profile(path: Path | str, scope: str = "formal") -> dict[str, str]:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ValueError(f"approved resource profile is required: {profile_path}")
    values = _read_env(profile_path)
    if values.get("RESOURCE_PROFILE_APPROVED") != "YES":
        raise ValueError("approved resource profile is required before smoke or formal submission")
    required_by_scope = _resource_keys_by_scope()
    if scope not in required_by_scope:
        raise ValueError(f"unsupported resource profile scope: {scope!r}")
    missing = [key for key in required_by_scope[scope] if not values.get(key)]
    if missing:
        raise ValueError("approved resource profile is missing: " + ", ".join(missing))
    return values


def resource_profile_template() -> str:
    lines = [
        "# Fill only after reviewed four-scale smoke/jobstats evidence. Do not guess.",
        "RESOURCE_PROFILE_APPROVED=NO",
    ]
    for key in _resource_keys_by_scope()["all"]:
        lines.append(f"{key}=")
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def dry_run_summary() -> list[str]:
    formal = formal_matrix()
    smoke = smoke_matrix()
    lines = [
        "PHASE=TRANSFORMER_EV_40CELL_WORKFLOW_DRY_RUN",
        f"FORMAL_MATRIX_COUNT={len(formal)}",
        f"FORMAL_UNIQUE_CELL_COUNT={len({(cell.scale, cell.algorithm, cell.seed) for cell in formal})}",
        "FORMAL_TASK_IDS=0..39",
        "FORMAL_ALGORITHM_SET=hierarchical_transformer_ev",
        "FORMAL_RESOURCE_RANGES=0-19:25cp_100cp,20-29:500cp,30-39:1000cp",
        f"SMOKE_MATRIX_COUNT={len(smoke)}",
        "SMOKE_SCALES=" + ",".join(cell.scale for cell in smoke),
        "SMOKE_SEEDS=0",
    ]
    for task_id in (0, 9, 10, 19, 20, 29, 30, 39):
        cell = resolve_formal_cell(task_id)
        run_name = f"transformer_ev_40cell_{cell.scale}_seed{cell.seed}"
        command = training_command(cell, f"<train_root>/task{task_id}", run_name)
        lines.append(f"FORMAL_COMMAND_TASK_{task_id}={_command_text(command)}")
    for cell in smoke:
        run_name = f"transformer_ev_smoke_{cell.scale}_seed0"
        command = training_command(cell, f"<smoke_root>/task{cell.task_id}", run_name)
        lines.append(f"SMOKE_COMMAND_TASK_{cell.task_id}={_command_text(command)}")
    lines.extend(
        [
            "POST_TRAINING_GATE=40/40_validated_model.best_required",
            "EVAL30_AND_DIAGNOSTICS_PARALLEL_AFTER_GATE=YES",
            "DRY_RUN_SBATCH_COUNT=0",
            "DRY_RUN_NO_JOBS_SUBMITTED",
        ]
    )
    return lines


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--print-smoke", action="store_true")
    parser.add_argument("--write-resource-template", type=Path)
    parser.add_argument("--validate-resource-profile", type=Path)
    parser.add_argument(
        "--resource-profile-scope",
        choices=tuple(_resource_keys_by_scope()),
        default="formal",
    )
    subparsers = parser.add_subparsers(dest="command")
    formal_task = subparsers.add_parser("formal-task-mapping")
    formal_task.add_argument("--task-id", type=int, required=True)
    smoke_task = subparsers.add_parser("smoke-task-mapping")
    smoke_task.add_argument("--task-id", type=int, required=True)
    formal_command = subparsers.add_parser("formal-training-command")
    formal_command.add_argument("--task-id", type=int, required=True)
    formal_command.add_argument("--save-dir", required=True)
    formal_command.add_argument("--run-name", required=True)
    smoke_command = subparsers.add_parser("smoke-training-command")
    smoke_command.add_argument("--task-id", type=int, required=True)
    smoke_command.add_argument("--save-dir", required=True)
    smoke_command.add_argument("--run-name", required=True)
    eval_command = subparsers.add_parser("eval30-command")
    eval_command.add_argument("--task-id", type=int, required=True)
    eval_command.add_argument("--checkpoint-prefix", required=True)
    eval_command.add_argument("--output-csv", required=True)
    eval_command.add_argument("--run-name", required=True)
    eval_command.add_argument("--max-episode-steps", type=int, required=True)
    diagnostic_cmd = subparsers.add_parser("diagnostic-command")
    diagnostic_cmd.add_argument("--task-id", type=int, required=True)
    diagnostic_cmd.add_argument("--checkpoint-prefix", required=True)
    diagnostic_cmd.add_argument("--output-dir", required=True)
    diagnostic_cmd.add_argument("--run-name", required=True)
    diagnostic_cmd.add_argument("--max-episode-steps", type=int, required=True)
    diagnostic_cmd.add_argument("--matrix-job-id", required=True)
    subparsers.add_parser("dry-run-summary")

    smoke_gate = subparsers.add_parser("validate-smoke-packages")
    smoke_gate.add_argument("--package-root", type=Path, required=True)
    smoke_gate.add_argument("--job-id", required=True)
    smoke_gate.add_argument("--stage-root", type=Path, required=True)
    smoke_gate.add_argument("--source-identity")
    smoke_gate.add_argument("--source-bundle-identity")
    training_aggregate = subparsers.add_parser("aggregate-training")
    training_aggregate.add_argument("--package-root", type=Path, required=True)
    training_aggregate.add_argument("--array-job-id")
    training_aggregate.add_argument("--small-job-id")
    training_aggregate.add_argument("--cp500-job-id")
    training_aggregate.add_argument("--cp1000-job-id")
    training_aggregate.add_argument("--stage-root", type=Path, required=True)
    training_aggregate.add_argument("--output-dir", type=Path, required=True)
    training_aggregate.add_argument("--source-identity")
    training_aggregate.add_argument("--source-bundle-identity")
    eval_aggregate = subparsers.add_parser("aggregate-eval30")
    eval_aggregate.add_argument("--package-root", type=Path, required=True)
    eval_aggregate.add_argument("--array-job-id")
    eval_aggregate.add_argument("--training-set-id")
    eval_aggregate.add_argument("--eval-job-id", required=True)
    eval_aggregate.add_argument("--output-dir", type=Path, required=True)
    eval_aggregate.add_argument("--source-identity")
    eval_aggregate.add_argument("--source-bundle-identity")
    diagnostic_aggregate = subparsers.add_parser("aggregate-diagnostics")
    diagnostic_aggregate.add_argument("--package-root", type=Path, required=True)
    diagnostic_aggregate.add_argument("--array-job-id")
    diagnostic_aggregate.add_argument("--training-set-id")
    diagnostic_aggregate.add_argument("--diagnostic-job-id", required=True)
    diagnostic_aggregate.add_argument("--output-dir", type=Path, required=True)
    diagnostic_aggregate.add_argument("--source-identity")
    diagnostic_aggregate.add_argument("--source-bundle-identity")
    training_gate = subparsers.add_parser("validate-training")
    training_gate.add_argument("--csv", type=Path, required=True)
    eval_gate = subparsers.add_parser("validate-eval30")
    eval_gate.add_argument("--csv", type=Path, required=True)
    diagnostic_gate = subparsers.add_parser("validate-diagnostics")
    diagnostic_gate.add_argument("--csv", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.print_matrix:
        for cell in formal_matrix():
            print(format_cell_line(cell))
        return 0
    if args.print_smoke:
        for cell in smoke_matrix():
            print(format_cell_line(cell))
        return 0
    if args.write_resource_template:
        args.write_resource_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_resource_template.write_text(resource_profile_template(), encoding="utf-8")
        print(f"RESOURCE_TEMPLATE={args.write_resource_template}")
        return 0
    if args.validate_resource_profile:
        load_resource_profile(args.validate_resource_profile, scope=args.resource_profile_scope)
        print("RESOURCE_PROFILE_VALIDATED")
        print(f"RESOURCE_PROFILE_SCOPE={args.resource_profile_scope}")
        return 0
    if args.command == "formal-task-mapping":
        print(format_cell_line(resolve_formal_cell(args.task_id)))
        return 0
    if args.command == "smoke-task-mapping":
        print(format_cell_line(resolve_smoke_cell(args.task_id)))
        return 0
    if args.command == "formal-training-command":
        print(_command_text(training_command(resolve_formal_cell(args.task_id), args.save_dir, args.run_name)))
        return 0
    if args.command == "smoke-training-command":
        print(_command_text(training_command(resolve_smoke_cell(args.task_id), args.save_dir, args.run_name)))
        return 0
    if args.command == "eval30-command":
        cell = resolve_formal_cell(args.task_id)
        print(
            _command_text(
                eval30_command(
                    cell,
                    args.checkpoint_prefix,
                    args.output_csv,
                    args.run_name,
                    args.max_episode_steps,
                    eval_seed_offset(cell),
                )
            )
        )
        return 0
    if args.command == "diagnostic-command":
        cell = resolve_formal_cell(args.task_id)
        print(
            _command_text(
                diagnostic_command(
                    cell,
                    args.checkpoint_prefix,
                    args.output_dir,
                    args.run_name,
                    args.max_episode_steps,
                    eval_seed_offset(cell),
                    args.matrix_job_id,
                )
            )
        )
        return 0
    if args.command == "dry-run-summary":
        print("\n".join(dry_run_summary()))
        return 0
    if args.command in {
        "validate-smoke-packages",
        "aggregate-training",
        "aggregate-eval30",
        "aggregate-diagnostics",
    }:
        from scripts import transformer_ev_40cell_artifacts as artifacts

        if args.command == "validate-smoke-packages":
            result = artifacts.validate_smoke_packages(
                args.package_root,
                args.job_id,
                args.stage_root,
                expected_source_identity=args.source_identity,
                expected_source_bundle_identity=args.source_bundle_identity,
            )
            print(f"STATUS={result.status}")
            print(f"SMOKE_CELL_COUNT={result.cell_count}")
            print("SCIENTIFIC_CLAIM_GENERATED=NO")
            return 0
        if args.command == "aggregate-training":
            split_job_ids = {
                key: value
                for key, value in {
                    "small": args.small_job_id,
                    "500cp": args.cp500_job_id,
                    "1000cp": args.cp1000_job_id,
                }.items()
                if value is not None
            }
            if split_job_ids:
                training_kwargs = {"training_job_ids_by_group": split_job_ids}
            else:
                training_kwargs = {"training_job_id": args.array_job_id}
            result = artifacts.aggregate_training_packages(
                args.package_root,
                **training_kwargs,
                stage_root=args.stage_root,
                expected_source_identity=args.source_identity,
                expected_source_bundle_identity=args.source_bundle_identity,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = _write_csv(args.output_dir / "transformer_ev_training_matrix_summary.csv", result.summary_rows)
            curve_path = _write_csv(args.output_dir / "transformer_ev_training_curve_long.csv", result.training_curve_rows)
            print("STATUS=PASS")
            print(f"TRAINING_CELL_COUNT={len(result.summary_rows)}")
            print(f"TRAINING_CURVE_ROWS={len(result.training_curve_rows)}")
            print(f"STAGED_CHECKPOINT_COUNT={len(result.staged_tasks)}")
            print(f"TRAINING_SET_ID={result.training_set_id}")
            print(f"OUTPUT={summary_path}")
            print(f"OUTPUT={curve_path}")
            return 0
        if args.command == "aggregate-eval30":
            rows = artifacts.aggregate_eval30_packages(
                args.package_root,
                training_job_id=args.array_job_id,
                training_set_id=args.training_set_id,
                eval_job_id=args.eval_job_id,
                expected_source_identity=args.source_identity,
                expected_source_bundle_identity=args.source_bundle_identity,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output = _write_csv(args.output_dir / "transformer_ev_eval30_episode_rows.csv", rows)
            print("STATUS=PASS")
            print(f"EVAL30_EPISODE_ROWS={len(rows)}")
            print(f"OUTPUT={output}")
            return 0
        result = artifacts.aggregate_diagnostic_packages(
            args.package_root,
            training_job_id=args.array_job_id,
            training_set_id=args.training_set_id,
            diagnostic_job_id=args.diagnostic_job_id,
            expected_source_identity=args.source_identity,
            expected_source_bundle_identity=args.source_bundle_identity,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_path = _write_csv(args.output_dir / "transformer_ev_diagnostic_episode_rows.csv", result.diagnostic_rows)
        same_pass_path = _write_csv(args.output_dir / "transformer_ev_diagnostic_same_pass_eval30_rows.csv", result.same_pass_eval_rows)
        package_path = _write_csv(args.output_dir / "transformer_ev_diagnostic_package_summary.csv", result.package_summaries)
        print("STATUS=PASS")
        print(f"DIAGNOSTIC_EPISODE_ROWS={len(result.diagnostic_rows)}")
        print(f"SAME_PASS_CANONICAL_EPISODE_ROWS={len(result.same_pass_eval_rows)}")
        print(f"OUTPUT={diagnostic_path}")
        print(f"OUTPUT={same_pass_path}")
        print(f"OUTPUT={package_path}")
        return 0
    if args.command == "validate-training":
        result = validate_training_gate(_read_csv(args.csv))
        print(f"STATUS={result.status}")
        print(f"TRAINING_CELL_COUNT={result.cell_count}")
        return 0
    if args.command == "validate-eval30":
        result = validate_eval30_gate(_read_csv(args.csv))
        print(f"STATUS={result.status}")
        print(f"EVAL30_CELL_COUNT={result.cell_count}")
        print(f"EVAL30_EPISODE_ROWS={result.episode_count}")
        return 0
    if args.command == "validate-diagnostics":
        result = validate_diagnostic_gate(_read_csv(args.csv))
        print(f"STATUS={result.status}")
        print(f"DIAGNOSTIC_CELL_COUNT={result.cell_count}")
        print(f"DIAGNOSTIC_EPISODE_ROWS={result.episode_count}")
        return 0
    parser.print_help()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except ValueError as exc:
        print("STATUS=BLOCKED", file=sys.stderr)
        print(f"BLOCK_REASON={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
