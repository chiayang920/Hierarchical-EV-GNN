#!/usr/bin/env python3
"""Local contracts for the formal 75k non-negative 80-cell EV-GNN workflow."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Support direct execution by absolute path from any working directory.
if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

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
    """Return Holm step-down adjusted p-values in the original order."""
    count = len(pvalues)
    indexed: list[tuple[float, int]] = []
    for original_index, value in enumerate(pvalues):
        pvalue = float(value)
        if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise ValueError(f"p-value must be finite and within [0,1]; got {value!r}")
        indexed.append((pvalue, original_index))
    indexed.sort(key=lambda item: (item[0], item[1]))
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (pvalue, original_index) in enumerate(indexed):
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


def exact_sign_flip_pvalue(diff_values: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip randomisation p-value using |mean|."""
    values = [float(value) for value in diff_values]
    if not values:
        raise ValueError("exact sign-flip test requires at least one paired difference")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("exact sign-flip differences must be finite")
    observed = abs(_mean(values))
    total = 1 << len(values)
    extreme = 0
    tolerance = 1e-15
    for mask in range(total):
        signed_sum = 0.0
        for index, value in enumerate(values):
            signed_sum += value if (mask >> index) & 1 else -value
        statistic = abs(signed_sum / len(values))
        if statistic + tolerance >= observed:
            extreme += 1
    return extreme / total


def _sign_flip_pvalue(diff_values: Sequence[float]) -> float:
    return exact_sign_flip_pvalue(diff_values)


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
    service_status = str(service_guardrails.get("status", "UNKNOWN")).strip().upper()
    if service_status not in {"PASS", "FAIL", "UNKNOWN"}:
        raise ValueError(f"unsupported service guardrail status: {service_status!r}")
    claims["SERVICE_GUARDRAIL_STATUS"] = service_status

    claims["CROSS_SCALE_REWARD_DIRECTION_CONSISTENT"] = "YES" if all_reward_directions else "NO"
    claims["STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED"] = (
        "YES"
        if all_reward_directions
        and scale_supported_count >= 3
        and not harm_supported
        and all_boundary_directions
        and service_status == "PASS"
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
    *,
    training_curve_rows: Sequence[dict[str, object]] | None = None,
    service_policy: dict[str, float] | None = None,
) -> list[Path]:
    """Delegate scientific reduction to the focused statistics module.

    Real scheduled-checkpoint evidence is mandatory.  The former synthetic
    placeholder generation is intentionally rejected.
    """
    if training_curve_rows is None:
        raise ValueError(
            "STATUS=BLOCKED: real training_curve_rows are required; "
            "placeholder learning-dynamics outputs are prohibited"
        )
    from scripts.formal_75k_80cell_statistics import reduce_formal_results as reduce_statistics

    return reduce_statistics(
        training_rows=training_rows,
        training_curve_rows=training_curve_rows,
        eval_rows=eval_rows,
        diagnostic_rows=diagnostic_rows,
        out_dir=out_dir,
        service_policy=service_policy,
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


def load_resource_profile(path: Path | str, scope: str = "formal") -> dict[str, str]:
    profile_path = Path(path)
    if not profile_path.is_file():
        raise ValueError(f"approved resource profile is required: {profile_path}")
    values = _read_env(profile_path)
    if values.get("RESOURCE_PROFILE_APPROVED") != "YES":
        raise ValueError("approved resource profile is required before smoke or formal submission")
    required_by_scope = {
        "smoke": (
            "SMOKE_CPUS_PER_TASK",
            "SMOKE_MEM",
            "SMOKE_TIME",
        ),
        "formal": (
            "TRAIN_CPUS_PER_TASK",
            "TRAIN_MEM",
            "TRAIN_TIME",
            "EVAL_CPUS_PER_TASK",
            "EVAL_MEM",
            "EVAL_TIME",
            "DIAGNOSTIC_CPUS_PER_TASK",
            "DIAGNOSTIC_MEM",
            "DIAGNOSTIC_TIME",
        ),
        "all": (
            "SMOKE_CPUS_PER_TASK",
            "SMOKE_MEM",
            "SMOKE_TIME",
            "TRAIN_CPUS_PER_TASK",
            "TRAIN_MEM",
            "TRAIN_TIME",
            "EVAL_CPUS_PER_TASK",
            "EVAL_MEM",
            "EVAL_TIME",
            "DIAGNOSTIC_CPUS_PER_TASK",
            "DIAGNOSTIC_MEM",
            "DIAGNOSTIC_TIME",
        ),
    }
    if scope not in required_by_scope:
        raise ValueError(f"unsupported resource profile scope: {scope!r}")
    missing = [key for key in required_by_scope[scope] if not values.get(key)]
    if missing:
        raise ValueError("approved resource profile is missing: " + ", ".join(missing))
    return values


def resource_profile_template() -> str:
    return "\n".join(
        [
            "# Fill only after reviewed local smoke/resource evidence. Do not guess.",
            "RESOURCE_PROFILE_APPROVED=NO",
            "SMOKE_CPUS_PER_TASK=",
            "SMOKE_MEM=",
            "SMOKE_TIME=",
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


def _load_numeric_policy(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    values = _read_env(path)
    result: dict[str, float] = {}
    for key, value in values.items():
        result[key] = _parse_float(value, key)
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--print-smoke", action="store_true")
    parser.add_argument("--write-resource-template", type=Path)
    parser.add_argument("--validate-resource-profile", type=Path)
    parser.add_argument(
        "--resource-profile-scope",
        choices=("smoke", "formal", "all"),
        default="formal",
    )
    subparsers = parser.add_subparsers(dest="command")

    formal_task = subparsers.add_parser("formal-task-mapping")
    formal_task.add_argument("--task-id", type=int, required=True)
    smoke_task = subparsers.add_parser("smoke-task-mapping")
    smoke_task.add_argument("--task-id", type=int, required=True)

    smoke_gate = subparsers.add_parser("validate-smoke-packages")
    smoke_gate.add_argument("--package-root", type=Path, required=True)
    smoke_gate.add_argument("--job-id", required=True)
    smoke_gate.add_argument("--stage-root", type=Path, required=True)
    smoke_gate.add_argument("--source-identity")
    smoke_gate.add_argument("--source-bundle-identity")

    training_aggregate = subparsers.add_parser("aggregate-training")
    training_aggregate.add_argument("--package-root", type=Path, required=True)
    training_aggregate.add_argument("--array-job-id", required=True)
    training_aggregate.add_argument("--stage-root", type=Path, required=True)
    training_aggregate.add_argument("--output-dir", type=Path, required=True)
    training_aggregate.add_argument("--source-identity")
    training_aggregate.add_argument("--source-bundle-identity")

    eval_aggregate = subparsers.add_parser("aggregate-eval30")
    eval_aggregate.add_argument("--package-root", type=Path, required=True)
    eval_aggregate.add_argument("--array-job-id", required=True)
    eval_aggregate.add_argument("--eval-job-id", required=True)
    eval_aggregate.add_argument("--output-dir", type=Path, required=True)
    eval_aggregate.add_argument("--source-identity")
    eval_aggregate.add_argument("--source-bundle-identity")

    diagnostic_aggregate = subparsers.add_parser("aggregate-diagnostics")
    diagnostic_aggregate.add_argument("--package-root", type=Path, required=True)
    diagnostic_aggregate.add_argument("--array-job-id", required=True)
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

    reducer = subparsers.add_parser("reduce")
    reducer.add_argument("--training-csv", type=Path, required=True)
    reducer.add_argument("--training-curve-csv", type=Path, required=True)
    reducer.add_argument("--eval30-csv", type=Path, required=True)
    reducer.add_argument("--diagnostic-csv", type=Path, required=True)
    reducer.add_argument("--service-policy", type=Path)
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

    if args.command in {
        "validate-smoke-packages",
        "aggregate-training",
        "aggregate-eval30",
        "aggregate-diagnostics",
    }:
        from scripts import formal_75k_80cell_artifacts as artifacts

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
            result = artifacts.aggregate_training_packages(
                args.package_root,
                training_job_id=args.array_job_id,
                stage_root=args.stage_root,
                expected_source_identity=args.source_identity,
                expected_source_bundle_identity=args.source_bundle_identity,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            summary_path = _write_csv(args.output_dir / "formal_training_matrix_summary.csv", result.summary_rows)
            curve_path = _write_csv(args.output_dir / "training_curve_long.csv", result.training_curve_rows)
            print("STATUS=PASS")
            print(f"TRAINING_CELL_COUNT={len(result.summary_rows)}")
            print(f"TRAINING_CURVE_ROWS={len(result.training_curve_rows)}")
            print(f"STAGED_CHECKPOINT_COUNT={len(result.staged_tasks)}")
            print(f"OUTPUT={summary_path}")
            print(f"OUTPUT={curve_path}")
            return 0
        if args.command == "aggregate-eval30":
            rows = artifacts.aggregate_eval30_packages(
                args.package_root,
                training_job_id=args.array_job_id,
                eval_job_id=args.eval_job_id,
                expected_source_identity=args.source_identity,
                expected_source_bundle_identity=args.source_bundle_identity,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output = _write_csv(args.output_dir / "canonical_eval30_episode_rows.csv", rows)
            print("STATUS=PASS")
            print(f"EVAL30_EPISODE_ROWS={len(rows)}")
            print(f"OUTPUT={output}")
            return 0
        result = artifacts.aggregate_diagnostic_packages(
            args.package_root,
            training_job_id=args.array_job_id,
            diagnostic_job_id=args.diagnostic_job_id,
            expected_source_identity=args.source_identity,
            expected_source_bundle_identity=args.source_bundle_identity,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_path = _write_csv(args.output_dir / "diagnostic_episode_rows.csv", result.diagnostic_rows)
        same_pass_path = _write_csv(args.output_dir / "diagnostic_same_pass_canonical_eval30_rows.csv", result.same_pass_eval_rows)
        package_path = _write_csv(args.output_dir / "diagnostic_package_summary.csv", result.package_summaries)
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
    if args.command == "reduce":
        outputs = reduce_formal_results(
            _read_csv(args.training_csv),
            _read_csv(args.eval30_csv),
            _read_csv(args.diagnostic_csv),
            args.out_dir,
            training_curve_rows=_read_csv(args.training_curve_csv),
            service_policy=_load_numeric_policy(args.service_policy),
        )
        print("STATUS=PASS")
        for output_path in outputs:
            print(f"OUTPUT={output_path}")
        return 0
    parser.print_help()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except ValueError as exc:
        print(f"STATUS=BLOCKED", file=sys.stderr)
        print(f"BLOCK_REASON={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
