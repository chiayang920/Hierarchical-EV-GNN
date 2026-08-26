#!/usr/bin/env python3
"""Diagnostic-only aggregation for transformer feasibility projection evidence."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROJECTION_DIAGNOSTIC_TOLERANCE = 1e-7

PROJECTION_TRANSFORMER_STEP_COLUMNS = [
    "algorithm",
    "scale",
    "training_seed",
    "episode_index",
    "episode_seed",
    "environment_step",
    "graph_index",
    "transformer_id",
    "constraint_activated",
    "alpha",
    "raw_estimated_commanded_power",
    "safe_estimated_commanded_power",
    "physical_max_power",
    "usable_safe_capacity",
    "rounding_margin",
    "commanded_power_removed",
    "active_ev_count_under_transformer",
    "corrected_ev_decisions",
    "total_action_correction_magnitude",
]

PROJECTION_EPISODE_SUMMARY_COLUMNS = [
    "algorithm",
    "scale",
    "training_seed",
    "episode_index",
    "episode_seed",
    "environment_steps",
    "activated_environment_steps",
    "transformer_step_observations",
    "activated_transformer_step_observations",
    "any_constraint_activation_step_fraction",
    "transformer_activation_rate",
    "active_alpha_mean",
    "active_alpha_min",
    "active_alpha_sum",
    "commanded_power_removed_kw_step_sum",
    "commanded_power_removed_kw_mean_when_active",
    "estimated_energy_removed_kwh",
    "action_correction_l1_sum",
    "action_correction_l1_mean_per_active_ev_decision",
    "corrected_ev_decision_fraction",
    "corrected_ev_decisions",
    "active_ev_decisions",
    "transformer_feasibility_violation_count",
    "transformer_feasibility_max_excess_kw",
]

PROJECTION_SEED_SUMMARY_COLUMNS = [
    "algorithm",
    "scale",
    "training_seed",
    "episodes",
    "environment_steps",
    "activated_environment_steps",
    "transformer_step_observations",
    "activated_transformer_step_observations",
    "any_constraint_activation_step_fraction",
    "transformer_activation_rate",
    "active_alpha_mean",
    "active_alpha_min",
    "active_alpha_sum",
    "commanded_power_removed_kw_step_sum",
    "commanded_power_removed_kw_mean_when_active",
    "estimated_energy_removed_kwh",
    "action_correction_l1_sum",
    "action_correction_l1_mean_per_active_ev_decision",
    "corrected_ev_decision_fraction",
    "corrected_ev_decisions",
    "active_ev_decisions",
    "transformer_feasibility_violation_count",
    "transformer_feasibility_max_excess_kw",
]


def _as_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def _as_int(value) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return int(value)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "pass"}
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return bool(value)


def _has_value(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def transformer_step_rows_from_projection_details(
    *,
    metadata: Mapping[str, object],
    episode_index: int,
    episode_seed: int,
    environment_step: int,
    details: Mapping[str, object],
    active_ev_count_by_transformer: Mapping[object, int],
    corrected_ev_decisions: int,
) -> list[dict[str, object]]:
    correction = _as_float(details.get("total_action_correction_magnitude", 0.0))
    rows: list[dict[str, object]] = []
    for transformer in details.get("transformers", []):
        row_graph_index = _as_int(transformer.get("graph_index", 0))
        transformer_id = _as_int(transformer["transformer_id"])
        row = {
            **dict(metadata),
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "environment_step": int(environment_step),
            "graph_index": row_graph_index,
            "transformer_id": transformer_id,
            "constraint_activated": _as_bool(transformer["constraint_activated"]),
            "alpha": _as_float(transformer["alpha"]),
            "raw_estimated_commanded_power": _as_float(
                transformer["raw_estimated_commanded_power"]
            ),
            "safe_estimated_commanded_power": _as_float(
                transformer["safe_estimated_commanded_power"]
            ),
            "physical_max_power": _as_float(transformer["physical_max_power"]),
            "usable_safe_capacity": _as_float(transformer["usable_safe_capacity"]),
            "rounding_margin": _as_float(transformer["rounding_margin"]),
            "commanded_power_removed": _as_float(transformer["commanded_power_removed"]),
            "active_ev_count_under_transformer": int(
                active_ev_count_by_transformer.get(
                    (row_graph_index, transformer_id),
                    active_ev_count_by_transformer.get(transformer_id, 0),
                )
            ),
            "corrected_ev_decisions": int(corrected_ev_decisions),
            "total_action_correction_magnitude": correction,
        }
        rows.append(row)
    return rows


def summarise_projection_episode(
    *,
    metadata: Mapping[str, object],
    episode_index: int,
    episode_seed: int,
    environment_steps: int,
    transformer_step_rows: Sequence[Mapping[str, object]],
    timestep_minutes: float,
    corrected_ev_decisions: int,
    active_ev_decisions: int,
) -> dict[str, object]:
    step_count = int(environment_steps)
    if step_count < 0:
        raise ValueError("environment_steps must be non-negative")
    activated_rows = [row for row in transformer_step_rows if _as_bool(row["constraint_activated"])]
    activated_steps = {
        int(row["environment_step"])
        for row in activated_rows
    }
    removed_values = [_as_float(row.get("commanded_power_removed", 0.0)) for row in transformer_step_rows]
    active_alphas = [_as_float(row["alpha"]) for row in activated_rows]
    action_l1_by_step: dict[int, float] = defaultdict(float)
    for row in transformer_step_rows:
        row_step = int(row["environment_step"])
        if row_step < 0 or row_step >= step_count:
            raise ValueError(
                f"transformer row environment_step {row_step} outside executed "
                f"episode length {step_count}"
            )
        action_l1_by_step[row_step] = max(
            action_l1_by_step[row_step],
            _as_float(row.get("total_action_correction_magnitude", 0.0)),
        )

    transformer_observations = len(transformer_step_rows)
    activated_transformer_observations = len(activated_rows)
    removed_sum = float(np.sum(removed_values)) if removed_values else 0.0
    action_l1_sum = float(np.sum(list(action_l1_by_step.values()))) if action_l1_by_step else 0.0
    active_alpha_sum = float(np.sum(active_alphas)) if active_alphas else 0.0
    timestep_hours = float(timestep_minutes) / 60.0

    return {
        **dict(metadata),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "environment_steps": step_count,
        "activated_environment_steps": len(activated_steps),
        "transformer_step_observations": transformer_observations,
        "activated_transformer_step_observations": activated_transformer_observations,
        "any_constraint_activation_step_fraction": (
            len(activated_steps) / step_count if step_count else 0.0
        ),
        "transformer_activation_rate": (
            activated_transformer_observations / transformer_observations
            if transformer_observations
            else 0.0
        ),
        "active_alpha_mean": (
            active_alpha_sum / activated_transformer_observations
            if activated_transformer_observations
            else None
        ),
        "active_alpha_min": float(np.min(active_alphas)) if active_alphas else None,
        "active_alpha_sum": active_alpha_sum,
        "commanded_power_removed_kw_step_sum": removed_sum,
        "commanded_power_removed_kw_mean_when_active": (
            removed_sum / activated_transformer_observations
            if activated_transformer_observations
            else None
        ),
        "estimated_energy_removed_kwh": removed_sum * timestep_hours,
        "action_correction_l1_sum": action_l1_sum,
        "action_correction_l1_mean_per_active_ev_decision": (
            action_l1_sum / int(active_ev_decisions) if int(active_ev_decisions) else 0.0
        ),
        "corrected_ev_decision_fraction": (
            int(corrected_ev_decisions) / int(active_ev_decisions)
            if int(active_ev_decisions)
            else 0.0
        ),
        "corrected_ev_decisions": int(corrected_ev_decisions),
        "active_ev_decisions": int(active_ev_decisions),
        "transformer_feasibility_violation_count": 0,
        "transformer_feasibility_max_excess_kw": 0.0,
    }


def summarise_projection_seed(
    metadata: Mapping[str, object],
    episode_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not episode_rows:
        raise ValueError("projection seed summary requires at least one episode row")

    environment_steps = sum(_as_int(row["environment_steps"]) for row in episode_rows)
    activated_environment_steps = sum(
        _as_int(row["activated_environment_steps"]) for row in episode_rows
    )
    transformer_observations = sum(
        _as_int(row["transformer_step_observations"]) for row in episode_rows
    )
    activated_transformer_observations = sum(
        _as_int(row["activated_transformer_step_observations"]) for row in episode_rows
    )
    corrected = sum(_as_int(row["corrected_ev_decisions"]) for row in episode_rows)
    active = sum(_as_int(row["active_ev_decisions"]) for row in episode_rows)
    violation_count = sum(
        _as_int(row.get("transformer_feasibility_violation_count", 0))
        for row in episode_rows
    )
    removed_sum = sum(_as_float(row["commanded_power_removed_kw_step_sum"]) for row in episode_rows)
    action_l1_sum = sum(_as_float(row["action_correction_l1_sum"]) for row in episode_rows)
    active_alpha_sum = sum(_as_float(row.get("active_alpha_sum", 0.0)) for row in episode_rows)
    active_alpha_mins = [
        _as_float(row["active_alpha_min"])
        for row in episode_rows
        if _has_value(row.get("active_alpha_min"))
    ]

    return {
        **dict(metadata),
        "episodes": len(episode_rows),
        "environment_steps": environment_steps,
        "activated_environment_steps": activated_environment_steps,
        "transformer_step_observations": transformer_observations,
        "activated_transformer_step_observations": activated_transformer_observations,
        "any_constraint_activation_step_fraction": (
            activated_environment_steps / environment_steps if environment_steps else 0.0
        ),
        "transformer_activation_rate": (
            activated_transformer_observations / transformer_observations
            if transformer_observations
            else 0.0
        ),
        "active_alpha_mean": (
            active_alpha_sum / activated_transformer_observations
            if activated_transformer_observations
            else None
        ),
        "active_alpha_min": float(np.min(active_alpha_mins)) if active_alpha_mins else None,
        "active_alpha_sum": active_alpha_sum,
        "commanded_power_removed_kw_step_sum": removed_sum,
        "commanded_power_removed_kw_mean_when_active": (
            removed_sum / activated_transformer_observations
            if activated_transformer_observations
            else None
        ),
        "estimated_energy_removed_kwh": sum(
            _as_float(row["estimated_energy_removed_kwh"]) for row in episode_rows
        ),
        "action_correction_l1_sum": action_l1_sum,
        "action_correction_l1_mean_per_active_ev_decision": (
            action_l1_sum / active if active else 0.0
        ),
        "corrected_ev_decision_fraction": corrected / active if active else 0.0,
        "corrected_ev_decisions": corrected,
        "active_ev_decisions": active,
        "transformer_feasibility_violation_count": violation_count,
        "transformer_feasibility_max_excess_kw": max(
            _as_float(row.get("transformer_feasibility_max_excess_kw", 0.0))
            for row in episode_rows
        ),
    }


def write_csv_rows(
    path: Path | str,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row.get(key) is None else row.get(key, "")) for key in fieldnames})
    return output_path
