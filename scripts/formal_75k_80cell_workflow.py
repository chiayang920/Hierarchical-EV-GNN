#!/usr/bin/env python3
"""Local contracts for the formal 75k non-negative 80-cell EV-GNN workflow."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scripts.aggregate_controlled_multiscale_eval30 import (
    paired_t_test,
    wilcoxon_signed_rank,
)


SCALES = ("25cp", "100cp", "500cp", "1000cp")
FORMAL_ALGORITHMS = ("actiongnn_nonnegative", "hierarchical")
MODEL_NAMES = {
    "actiongnn_nonnegative": "Corrected non-negative ActionGNN",
    "hierarchical": "Hierarchical non-negative ActionGNN",
    "historical_actiongnn": "Historical controlled signed ActionGNN",
}
SEEDS = tuple(range(10))
FORMAL_TRAINING_STEPS = 75000
FORMAL_EVALUATION_CADENCE = 5000
FORMAL_TRAINING_EVALUATION_EPISODES = 5
FORMAL_SCHEDULED_EVALUATIONS = 15
FORMAL_MODEL_BEST_EVAL_EPISODES = 30
EXPECTED_FORMAL_CELLS = 80
EXPECTED_FORMAL_EVAL_ROWS = 2400
SMOKE_TRAINING_STEPS = 512
SMOKE_START_TIMESTEPS = 64
SMOKE_EVALUATION_CADENCE = 256
SMOKE_TRAINING_EVALUATION_EPISODES = 1
START_TIMESTEPS = 1000
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
DEVICE = "cpu"
LOG_TO_WANDB = "false"
CHECKPOINT_SELECTION_RULE = (
    "model.best selected by strict improvement of scheduled internal eval mean reward "
    "at 5k-step intervals within the configured training budget; model.last is saved "
    "at the configured final step but is not used for canonical eval30 or diagnostics."
)
KEY_MECHANISM_METRIC = "upper_bound_active_ev_action_fraction"
KEY_MECHANISM_PUBLIC_LABEL = "Upper-bound active-EV action fraction (%) ↓"

CONFIG_BY_SCALE = {
    "25cp": "config_files/PublicPST_25cp.yaml",
    "100cp": "config_files/PublicPST_100.yaml",
    "500cp": "config_files/PublicPST_500.yaml",
    "1000cp": "config_files/PublicPST_1000.yaml",
}

ACTOR_OUTPUT_TRANSFORM_BY_ALGORITHM = {
    "actiongnn_nonnegative": "shifted_tanh_v1",
    "hierarchical": "hierarchical_nonnegative_action_contract_v1",
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
class FormalCell:
    task_id: int
    scale: str
    algorithm: str
    seed: int
    config_path: str
    training_steps: int = FORMAL_TRAINING_STEPS
    evaluation_cadence: int = FORMAL_EVALUATION_CADENCE
    training_evaluation_episodes: int = FORMAL_TRAINING_EVALUATION_EPISODES
    expected_scheduled_evaluations: int = FORMAL_SCHEDULED_EVALUATIONS
    actor_output_transform: str = ""
    checkpoint_selection_rule: str = CHECKPOINT_SELECTION_RULE


@dataclass(frozen=True)
class GateResult:
    status: str
    cell_count: int
    episode_count: int = 0
    missing_cells: list[tuple[str, str, int]] | None = None
    duplicate_cells: list[tuple[str, str, int]] | None = None


def scheduled_evaluation_steps() -> tuple[int, ...]:
    return tuple(range(FORMAL_EVALUATION_CADENCE, FORMAL_TRAINING_STEPS + 1, FORMAL_EVALUATION_CADENCE))


def validate_formal_algorithm(algorithm: str) -> str:
    if algorithm == "actiongnn":
        raise ValueError("legacy actiongnn is not a fresh formal 75k algorithm")
    if algorithm not in FORMAL_ALGORITHMS:
        raise ValueError(f"unsupported formal algorithm: {algorithm!r}")
    return algorithm


def _make_cell(task_id: int, scale: str, algorithm: str, seed: int, training_steps: int) -> FormalCell:
    validate_formal_algorithm(algorithm)
    return FormalCell(
        task_id=task_id,
        scale=scale,
        algorithm=algorithm,
        seed=seed,
        config_path=CONFIG_BY_SCALE[scale],
        training_steps=training_steps,
        evaluation_cadence=FORMAL_EVALUATION_CADENCE if training_steps == FORMAL_TRAINING_STEPS else SMOKE_EVALUATION_CADENCE,
        training_evaluation_episodes=(
            FORMAL_TRAINING_EVALUATION_EPISODES
            if training_steps == FORMAL_TRAINING_STEPS
            else SMOKE_TRAINING_EVALUATION_EPISODES
        ),
        expected_scheduled_evaluations=(
            FORMAL_SCHEDULED_EVALUATIONS
            if training_steps == FORMAL_TRAINING_STEPS
            else max(1, training_steps // SMOKE_EVALUATION_CADENCE)
        ),
        actor_output_transform=ACTOR_OUTPUT_TRANSFORM_BY_ALGORITHM[algorithm],
    )


def formal_matrix() -> tuple[FormalCell, ...]:
    cells: list[FormalCell] = []
    task_id = 0
    for scale in SCALES:
        for algorithm in FORMAL_ALGORITHMS:
            for seed in SEEDS:
                cells.append(_make_cell(task_id, scale, algorithm, seed, FORMAL_TRAINING_STEPS))
                task_id += 1
    return tuple(cells)


def smoke_matrix() -> tuple[FormalCell, ...]:
    return (
        _make_cell(0, "25cp", "actiongnn_nonnegative", 0, SMOKE_TRAINING_STEPS),
        _make_cell(1, "25cp", "hierarchical", 0, SMOKE_TRAINING_STEPS),
    )


def resolve_formal_cell(task_id: int) -> FormalCell:
    if type(task_id) is not int or not 0 <= task_id < EXPECTED_FORMAL_CELLS:
        raise ValueError(f"formal task ID must be 0..79; got {task_id!r}")
    return formal_matrix()[task_id]


def resolve_smoke_cell(task_id: int) -> FormalCell:
    if type(task_id) is not int or not 0 <= task_id < 2:
        raise ValueError(f"smoke task ID must be 0..1; got {task_id!r}")
    return smoke_matrix()[task_id]


def format_cell_line(cell: FormalCell) -> str:
    return (
        f"task_id={cell.task_id} scale={cell.scale} algorithm={cell.algorithm} "
        f"seed={cell.seed} config={cell.config_path} training_steps={cell.training_steps}"
    )


def training_command(cell: FormalCell, save_dir: str, run_name: str) -> list[str]:
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
        str(START_TIMESTEPS if cell.training_steps == FORMAL_TRAINING_STEPS else SMOKE_START_TIMESTEPS),
        "--eval_freq",
        str(cell.evaluation_cadence),
        "--eval_episodes",
        str(cell.training_evaluation_episodes),
        "--batch_size",
        str(BATCH_SIZE),
        "--replay_buffer_size",
        str(REPLAY_BUFFER_SIZE),
        "--save_dir",
        save_dir,
        "--log_to_wandb",
        LOG_TO_WANDB,
    ]


def eval30_command(
    cell: FormalCell,
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
    cell: FormalCell,
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


def _row_cell_key(row: dict[str, object]) -> tuple[str, str, int]:
    scale = str(row.get("scale", "")).strip().lower()
    algorithm = str(row.get("algorithm", "")).strip()
    seed = _parse_int(row.get("seed"), "seed")
    validate_formal_algorithm(algorithm)
    return scale, algorithm, seed


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
            f"{label} gate requires 80/80 unique cells; "
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
    if len(source_identities) != 1 or "" in source_identities:
        raise ValueError("training gate requires one consistent non-empty source identity")

    for row in rows:
        scale, algorithm, seed = _row_cell_key(row)
        context = f"{scale} {algorithm} seed {seed}"
        training_steps = _parse_int(row.get("training_steps"), f"{context} training_steps")
        if training_steps == 50000:
            raise ValueError(f"{context}: reject 50k-only runs for the formal 75k matrix")
        if training_steps != FORMAL_TRAINING_STEPS:
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
        expected_transform = ACTOR_OUTPUT_TRANSFORM_BY_ALGORITHM[algorithm]
        if str(row.get("actor_output_transform", "")).strip() != expected_transform:
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
    if len(keys) != EXPECTED_FORMAL_CELLS * required_episode_count or len(set(keys)) != len(keys):
        raise ValueError(
            f"{label} gate requires 2,400/2,400 episode rows; "
            f"observed_rows={len(keys)} unique_rows={len(set(keys))}"
        )
    missing_cells = tuple(sorted(_expected_cell_keys() - cells))
    if missing_cells or len(cells) != EXPECTED_FORMAL_CELLS:
        raise ValueError(f"{label} gate requires 80/80 checkpoints; missing={len(missing_cells)}")
    return GateResult(status="PASS", cell_count=len(cells), episode_count=len(keys), missing_cells=list(missing_cells), duplicate_cells=[])


def validate_eval30_gate(rows: Sequence[dict[str, object]]) -> GateResult:
    gate = _validate_episode_inventory(rows, "formal eval30", FORMAL_MODEL_BEST_EVAL_EPISODES)
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
    gate = _validate_episode_inventory(rows, "diagnostic", FORMAL_MODEL_BEST_EVAL_EPISODES)
    for row in rows:
        context = f"{row.get('scale')} {row.get('algorithm')} seed {row.get('seed')} episode {row.get('episode_index')}"
        if str(row.get("checkpoint_role", "")).strip() != "model.best":
            raise ValueError(f"{context}: model.best identity is required")
        _parse_float(row.get(KEY_MECHANISM_METRIC), f"{context} {KEY_MECHANISM_METRIC}")
        for field in DIAGNOSTIC_REQUIRED_PASS_FIELDS:
            if str(row.get(field, "")).strip().lower() != "pass":
                public_name = field.replace("_", " ")
                raise ValueError(f"{context}: {public_name} failed")
        if not _is_true(row.get("required_values_finite")):
            raise ValueError(f"{context}: required finite fields flag failed")
    return gate


def primary_hypothesis_metrics() -> tuple[str, ...]:
    return ("episode_reward",)


def secondary_metrics() -> tuple[str, ...]:
    return (
        "tracking_error",
        "energy_tracking_error",
        "power_tracker_violation",
        "transformer_overload",
        "charger_level_boundary_behaviour",
        "energy_delivered",
        "evs_served",
        "average_satisfaction",
        "tail_satisfaction",
        "hhi_gini",
        "training_stability",
        "best_checkpoint_timing",
        "runtime_resource_efficiency",
    )


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    count = len(pvalues)
    indexed = sorted((index, float(value)) for index, value in enumerate(pvalues))
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (original_index, pvalue) in enumerate(indexed):
        candidate = min(1.0, (count - rank) * pvalue)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _format_float(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12g}"
    return str(value)


def _sign_flip_pvalue(diff_values: Sequence[float]) -> float:
    positive = sum(1 for value in diff_values if value > 0.0)
    negative = sum(1 for value in diff_values if value < 0.0)
    nonzero = positive + negative
    if nonzero == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2 ** nonzero)
    return min(1.0, 2.0 * probability)


def _episode_metric_means(rows: Sequence[dict[str, object]], metric: str) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = _row_cell_key(row)
        grouped.setdefault(key, []).append(_parse_float(row.get(metric), metric))
    means = {}
    for key, values in grouped.items():
        if len(values) != FORMAL_MODEL_BEST_EVAL_EPISODES:
            raise ValueError(f"{key}: expected 30 episode values for {metric}")
        means[key] = _mean(values)
    return means


def _paired_effect_rows(
    means: dict[tuple[str, str, int], float],
    metric: str,
    effect_orientation: str,
    holm_family: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    pvalues: list[float] = []
    for scale in SCALES:
        corrected_values = []
        hierarchy_values = []
        diff_values = []
        for seed in SEEDS:
            corrected = means[(scale, "actiongnn_nonnegative", seed)]
            hierarchy = means[(scale, "hierarchical", seed)]
            corrected_values.append(corrected)
            hierarchy_values.append(hierarchy)
            if effect_orientation == "hierarchy_minus_corrected":
                diff_values.append(hierarchy - corrected)
            elif effect_orientation == "corrected_minus_hierarchy":
                diff_values.append(corrected - hierarchy)
            else:
                raise ValueError(f"unsupported effect orientation: {effect_orientation}")

        t_statistic, t_pvalue, ci_low, ci_high, diff_std = paired_t_test(diff_values)
        wilcoxon_statistic, wilcoxon_pvalue, wilcoxon_method, wilcoxon_nonzero_n = wilcoxon_signed_rank(diff_values)
        diff_mean = _mean(diff_values)
        raw_row = {
            "scale": scale,
            "metric": metric,
            "model_a": MODEL_NAMES["actiongnn_nonnegative"],
            "model_b": MODEL_NAMES["hierarchical"],
            "effect_orientation": effect_orientation,
            "actiongnn_nonnegative_seed_mean": _mean(corrected_values),
            "actiongnn_nonnegative_seed_sd": _sample_std(corrected_values),
            "hierarchical_seed_mean": _mean(hierarchy_values),
            "hierarchical_seed_sd": _sample_std(hierarchy_values),
            "paired_differences": ";".join(_format_float(value) for value in diff_values),
            "paired_mean_difference": diff_mean,
            "paired_median_difference": _median(diff_values),
            "paired_95ci_low": ci_low,
            "paired_95ci_high": ci_high,
            "paired_cohens_dz": 0.0 if diff_std == 0.0 else diff_mean / diff_std,
            "seeds_favouring_hierarchy": sum(1 for value in diff_values if value > 0.0),
            "n_paired_seeds": len(SEEDS),
            "paired_t_statistic": t_statistic,
            "paired_t_pvalue_two_sided": t_pvalue,
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_pvalue_two_sided": wilcoxon_pvalue,
            "wilcoxon_method": wilcoxon_method,
            "wilcoxon_nonzero_n": wilcoxon_nonzero_n,
            "leave_one_seed_out_paired_means": ";".join(
                _format_float(_mean([value for index, value in enumerate(diff_values) if index != held_out]))
                for held_out in range(len(diff_values))
            ),
            "leave_one_seed_out_direction_stable": all(
                _mean([value for index, value in enumerate(diff_values) if index != held_out]) > 0.0
                for held_out in range(len(diff_values))
            ),
            "sign_flip_pvalue_two_sided": _sign_flip_pvalue(diff_values),
            "holm_family": holm_family,
        }
        raw_rows.append(raw_row)
        pvalues.append(t_pvalue)

    adjusted = holm_adjust(pvalues)
    for raw_row, adjusted_pvalue in zip(raw_rows, adjusted):
        raw_row["holm_adjusted_pvalue"] = adjusted_pvalue
        raw_row["statistically_supported_harm"] = (
            float(raw_row["paired_95ci_high"]) < 0.0 and adjusted_pvalue < 0.05
        )
        rows.append(raw_row)
    return rows


def assess_claims(
    reward_effects: Sequence[dict[str, object]],
    boundary_effects: Sequence[dict[str, object]],
    service_guardrails: dict[str, object],
) -> dict[str, str]:
    claims: dict[str, str] = {}
    scale_supported_count = 0
    harm_supported = False
    for row in reward_effects:
        scale = str(row["scale"]).upper()
        supported = (
            float(row["mean_benefit"]) > 0.0
            and float(row["ci_low"]) > 0.0
            and float(row["holm_adjusted_pvalue"]) < 0.05
            and int(row["seeds_favouring_hierarchy"]) >= 7
            and bool(row["leave_one_seed_out_direction_stable"])
        )
        if supported:
            scale_supported_count += 1
        if bool(row.get("statistically_supported_harm", False)):
            harm_supported = True
        claims[f"{scale}_SCALE_SPECIFIC_REWARD_IMPROVEMENT_SUPPORTED"] = "YES" if supported else "NO"

    all_reward_directions = all(float(row["mean_benefit"]) > 0.0 for row in reward_effects)
    all_boundary_directions = all(float(row["mean_benefit"]) > 0.0 for row in boundary_effects)
    boundary_supported_count = sum(
        1
        for row in boundary_effects
        if float(row["mean_benefit"]) > 0.0 and float(row["holm_adjusted_pvalue"]) < 0.05
    )
    service_collapse = bool(service_guardrails.get("clear_material_collapse", False))

    claims["CROSS_SCALE_REWARD_DIRECTION_CONSISTENT"] = "YES" if all_reward_directions else "NO"
    claims["STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED"] = (
        "YES"
        if all_reward_directions
        and scale_supported_count >= 3
        and not harm_supported
        and all_boundary_directions
        and not service_collapse
        else "NO"
    )
    claims["CROSS_SCALE_BOUNDARY_BEHAVIOUR_CLAIM_SUPPORTED"] = (
        "YES" if all_boundary_directions and boundary_supported_count >= 3 else "NO"
    )
    return claims


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames or (rows[0].keys() if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_float(row.get(field, "")) for field in fieldnames})
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _normalise_effects_for_claims(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scale": row["scale"],
            "mean_benefit": float(row["paired_mean_difference"]),
            "ci_low": float(row["paired_95ci_low"]),
            "holm_adjusted_pvalue": float(row["holm_adjusted_pvalue"]),
            "seeds_favouring_hierarchy": int(row["seeds_favouring_hierarchy"]),
            "leave_one_seed_out_direction_stable": bool(row["leave_one_seed_out_direction_stable"]),
            "statistically_supported_harm": bool(row["statistically_supported_harm"]),
        }
        for row in rows
    ]


def reduce_formal_results(
    training_rows: Sequence[dict[str, object]],
    eval_rows: Sequence[dict[str, object]],
    diagnostic_rows: Sequence[dict[str, object]],
    out_dir: Path | str,
) -> list[Path]:
    try:
        validate_training_gate(training_rows)
        validate_eval30_gate(eval_rows)
        validate_diagnostic_gate(diagnostic_rows)
    except ValueError as exc:
        raise ValueError(f"STATUS=BLOCKED: {exc}") from exc

    out_dir = Path(out_dir)
    output_paths: list[Path] = []

    reward_means = _episode_metric_means(eval_rows, "episode_reward")
    tracking_means = _episode_metric_means(eval_rows, "tracking_error")
    boundary_means = _episode_metric_means(diagnostic_rows, KEY_MECHANISM_METRIC)

    reward_effects = _paired_effect_rows(
        reward_means,
        "episode_reward",
        "hierarchy_minus_corrected",
        "primary_reward_by_scale",
    )
    boundary_effects = _paired_effect_rows(
        boundary_means,
        KEY_MECHANISM_METRIC,
        "corrected_minus_hierarchy",
        "key_mechanism_boundary_by_scale",
    )
    tracking_effects = _paired_effect_rows(
        tracking_means,
        "tracking_error",
        "corrected_minus_hierarchy",
        "secondary_descriptive",
    )

    claims = assess_claims(
        _normalise_effects_for_claims(reward_effects),
        _normalise_effects_for_claims(boundary_effects),
        {"clear_material_collapse": False},
    )
    claims["PRIMARY_INFERENCE_UNIT"] = "paired_training_seed"

    output_paths.append(_write_csv(out_dir / "formal_training_matrix_summary.csv", list(training_rows)))
    output_paths.append(_write_csv(out_dir / "canonical_eval30_episode_rows.csv", list(eval_rows)))
    output_paths.append(_write_csv(out_dir / "per_scale_paired_reward_effects.csv", reward_effects))
    output_paths.append(_write_csv(out_dir / "per_scale_paired_boundary_effects.csv", boundary_effects))
    output_paths.append(_write_csv(out_dir / "primary_reward_statistical_tests.csv", reward_effects))
    output_paths.append(_write_csv(out_dir / "key_mechanism_statistical_tests.csv", boundary_effects))
    output_paths.append(_write_csv(out_dir / "secondary_outcome_statistical_tests.csv", tracking_effects))
    output_paths.append(_write_csv(out_dir / "per_seed_primary_outcomes.csv", _per_seed_primary_rows(reward_means)))
    output_paths.append(_write_csv(out_dir / "cross_scale_direction_summary.csv", _cross_scale_rows(reward_effects, boundary_effects)))
    output_paths.append(_write_csv(out_dir / "service_guardrail_summary.csv", [{"guardrail": "service_preservation", "status": "NO_NON_INFERIORITY_MARGIN_APPROVED"}]))
    output_paths.append(_write_csv(out_dir / "resource_efficiency_summary.csv", [{"status": "RESOURCE_DEFAULTS_GUESSED_NO", "note": "Formal resource efficiency awaits smoke/formal accounting."}]))
    output_paths.append(_write_csv(out_dir / "training_curve_long.csv", _training_curve_rows(training_rows)))
    output_paths.append(_write_csv(out_dir / "per_run_learning_summary.csv", _learning_summary_rows(training_rows)))
    output_paths.append(_write_csv(out_dir / "per_scale_algorithm_learning_summary.csv", _scale_algorithm_learning_rows(training_rows)))
    output_paths.append(_write_csv(out_dir / "checkpoint_selection_summary.csv", _checkpoint_rows(training_rows)))
    output_paths.append(_write_csv(out_dir / "late_training_stability_summary.csv", _late_stability_rows(training_rows)))

    claim_lines = [f"{key}={value}" for key, value in sorted(claims.items())]
    output_paths.append(_write_text(out_dir / "claim_assessment.env", "\n".join(claim_lines) + "\n"))
    output_paths.append(_write_text(out_dir / "formal_scientific_results.md", _scientific_markdown(claims)))
    output_paths.append(_write_text(out_dir / "formal_supervisor_summary.html", _summary_html(claims)))
    output_paths.append(_write_text(out_dir / "paper_ready_tables.md", _paper_tables_markdown(reward_effects, boundary_effects)))
    output_paths.append(_write_text(out_dir / "paper_ready_tables.html", _paper_tables_html(reward_effects, boundary_effects)))
    figures_dir = out_dir / "paper_ready_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_paths.append(_write_text(figures_dir / "README.md", "Figures must use relative or standardised within-scale effects.\n"))
    return output_paths


def _per_seed_primary_rows(reward_means: dict[tuple[str, str, int], float]) -> list[dict[str, object]]:
    rows = []
    for scale in SCALES:
        for seed in SEEDS:
            corrected = reward_means[(scale, "actiongnn_nonnegative", seed)]
            hierarchy = reward_means[(scale, "hierarchical", seed)]
            rows.append(
                {
                    "scale": scale,
                    "seed": seed,
                    "primary_outcome": "episode_reward",
                    "corrected_nonnegative_actiongnn_mean": corrected,
                    "hierarchical_nonnegative_actiongnn_mean": hierarchy,
                    "hierarchy_benefit": hierarchy - corrected,
                }
            )
    return rows


def _cross_scale_rows(
    reward_effects: Sequence[dict[str, object]],
    boundary_effects: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for reward, boundary in zip(reward_effects, boundary_effects):
        rows.append(
            {
                "scale": reward["scale"],
                "reward_direction": "hierarchy" if float(reward["paired_mean_difference"]) > 0 else "corrected_or_tie",
                "boundary_direction": "hierarchy" if float(boundary["paired_mean_difference"]) > 0 else "corrected_or_tie",
                "raw_reward_axis_note": "Do not compare raw reward magnitudes across CP scales on one unqualified axis.",
            }
        )
    return rows


def _training_curve_rows(training_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in training_rows:
        for step in scheduled_evaluation_steps():
            rows.append(
                {
                    "scale": row["scale"],
                    "algorithm": row["algorithm"],
                    "seed": row["seed"],
                    "timestep": step,
                    "row_status": "scheduled_checkpoint_required",
                }
            )
    return rows


def _learning_summary_rows(training_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scale": row["scale"],
            "algorithm": row["algorithm"],
            "seed": row["seed"],
            "scheduled_checkpoints": FORMAL_SCHEDULED_EVALUATIONS,
            "model_best_step": row.get("model_best_step", ""),
            "best_checkpoint_percentage_of_75k": row.get("best_checkpoint_percentage_of_75k", ""),
        }
        for row in training_rows
    ]


def _scale_algorithm_learning_rows(training_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scale": scale,
            "algorithm": algorithm,
            "n_training_seeds": len(SEEDS),
            "scheduled_checkpoints_per_run": FORMAL_SCHEDULED_EVALUATIONS,
        }
        for scale in SCALES
        for algorithm in FORMAL_ALGORITHMS
    ]


def _checkpoint_rows(training_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scale": row["scale"],
            "algorithm": row["algorithm"],
            "seed": row["seed"],
            "checkpoint_role": "model.best",
            "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        }
        for row in training_rows
    ]


def _late_stability_rows(training_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scale": row["scale"],
            "algorithm": row["algorithm"],
            "seed": row["seed"],
            "late_checkpoint_window": "60000;65000;70000;75000",
            "claim_boundary": "fixed-budget adequacy only; no asymptotic convergence claim",
        }
        for row in training_rows
    ]


def _scientific_markdown(claims: dict[str, str]) -> str:
    return "\n".join(
        [
            "# Formal 75k Non-negative Architecture Comparison",
            "",
            f"Primary efficacy outcome: `episode_reward`; inference unit: `paired_training_seed` with n=10 per scale.",
            f"Key mechanism outcome: `{KEY_MECHANISM_PUBLIC_LABEL}`.",
            "",
            "Tracking error is reported descriptively and is not an independent primary hypothesis.",
            "No service non-inferiority claim is made because no margin was approved before the formal run.",
            "",
            "## Claim Assessment",
            *[f"- `{key}={value}`" for key, value in sorted(claims.items())],
            "",
        ]
    )


def _summary_html(claims: dict[str, str]) -> str:
    items = "".join(f"<li><code>{key}={value}</code></li>" for key, value in sorted(claims.items()))
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Formal 75k Summary</title></head>"
        "<body><h1>Formal 75k Non-negative Architecture Comparison</h1>"
        "<p>Primary inference uses n=10 paired training seeds per scale.</p>"
        f"<ul>{items}</ul></body></html>\n"
    )


def _paper_tables_markdown(
    reward_effects: Sequence[dict[str, object]],
    boundary_effects: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Paper-ready Tables",
        "",
        "Direction: positive reward effect means hierarchy better; positive boundary effect means lower upper-bound active-EV action fraction for hierarchy.",
        "",
        "| Scale | Reward Effect | Boundary Effect | n |",
        "| --- | ---: | ---: | ---: |",
    ]
    for reward, boundary in zip(reward_effects, boundary_effects):
        lines.append(
            f"| {reward['scale']} | {_format_float(reward['paired_mean_difference'])} | "
            f"{_format_float(boundary['paired_mean_difference'])} | 10 |"
        )
    return "\n".join(lines) + "\n"


def _paper_tables_html(
    reward_effects: Sequence[dict[str, object]],
    boundary_effects: Sequence[dict[str, object]],
) -> str:
    body_rows = "".join(
        "<tr>"
        f"<td>{reward['scale']}</td>"
        f"<td>{_format_float(reward['paired_mean_difference'])}</td>"
        f"<td>{_format_float(boundary['paired_mean_difference'])}</td>"
        "<td>10</td>"
        "</tr>"
        for reward, boundary in zip(reward_effects, boundary_effects)
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Paper-ready Tables</title></head>"
        "<body><table><thead><tr><th>Scale</th><th>Reward Effect</th><th>Boundary Effect</th><th>n</th></tr></thead>"
        f"<tbody>{body_rows}</tbody></table></body></html>\n"
    )


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


def load_resource_profile(path: Path | str) -> dict[str, str]:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ValueError(f"approved resource profile is required: {profile_path}")
    values = _read_env(profile_path)
    if values.get("RESOURCE_PROFILE_APPROVED") != "YES":
        raise ValueError("approved resource profile is required before formal submission")
    required = (
        "TRAIN_CPUS_PER_TASK",
        "TRAIN_MEM",
        "TRAIN_TIME",
        "EVAL_CPUS_PER_TASK",
        "EVAL_MEM",
        "EVAL_TIME",
        "DIAGNOSTIC_CPUS_PER_TASK",
        "DIAGNOSTIC_MEM",
        "DIAGNOSTIC_TIME",
    )
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError("approved resource profile is missing: " + ", ".join(missing))
    return values


def resource_profile_template() -> str:
    return "\n".join(
        [
            "# Fill only after reviewed smoke/resource evidence. Do not guess.",
            "RESOURCE_PROFILE_APPROVED=NO",
            "TRAIN_CPUS_PER_TASK=",
            "TRAIN_MEM=",
            "TRAIN_TIME=",
            "EVAL_CPUS_PER_TASK=",
            "EVAL_MEM=",
            "EVAL_TIME=",
            "DIAGNOSTIC_CPUS_PER_TASK=",
            "DIAGNOSTIC_MEM=",
            "DIAGNOSTIC_TIME=",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--print-smoke", action="store_true")
    parser.add_argument("--write-resource-template", type=Path)
    parser.add_argument("--validate-resource-profile", type=Path)
    subparsers = parser.add_subparsers(dest="command")
    formal_task = subparsers.add_parser("formal-task-mapping")
    formal_task.add_argument("--task-id", type=int, required=True)
    smoke_task = subparsers.add_parser("smoke-task-mapping")
    smoke_task.add_argument("--task-id", type=int, required=True)
    training_gate = subparsers.add_parser("validate-training")
    training_gate.add_argument("--csv", type=Path, required=True)
    eval_gate = subparsers.add_parser("validate-eval30")
    eval_gate.add_argument("--csv", type=Path, required=True)
    diagnostic_gate = subparsers.add_parser("validate-diagnostics")
    diagnostic_gate.add_argument("--csv", type=Path, required=True)
    reducer = subparsers.add_parser("reduce")
    reducer.add_argument("--training-csv", type=Path, required=True)
    reducer.add_argument("--eval30-csv", type=Path, required=True)
    reducer.add_argument("--diagnostic-csv", type=Path, required=True)
    reducer.add_argument("--out-dir", type=Path, required=True)
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
        args.write_resource_template.write_text(resource_profile_template(), encoding="utf-8")
        return 0
    if args.validate_resource_profile:
        load_resource_profile(args.validate_resource_profile)
        print("RESOURCE_PROFILE_VALIDATED")
        return 0
    if args.command == "formal-task-mapping":
        print(format_cell_line(resolve_formal_cell(args.task_id)))
        return 0
    if args.command == "smoke-task-mapping":
        print(format_cell_line(resolve_smoke_cell(args.task_id)))
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
    if args.command == "reduce":
        outputs = reduce_formal_results(
            _read_csv(args.training_csv),
            _read_csv(args.eval30_csv),
            _read_csv(args.diagnostic_csv),
            args.out_dir,
        )
        print("STATUS=PASS")
        for output_path in outputs:
            print(f"OUTPUT={output_path}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
