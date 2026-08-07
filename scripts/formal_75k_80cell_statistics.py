#!/usr/bin/env python3
"""Pre-declared statistics and evidence-derived scientific outputs for Formal-75k."""

from __future__ import annotations

import csv
import html
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


# Support direct execution by absolute path from any working directory.
if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.aggregate_controlled_multiscale_eval30 import paired_t_test, wilcoxon_signed_rank
from scripts import formal_75k_80cell_workflow as workflow


@dataclass(frozen=True)
class ServiceGuardrailResult:
    status: str
    rows: list[dict[str, object]]
    policy_applied: bool


@dataclass(frozen=True)
class LearningDynamicsResult:
    training_curve_long: list[dict[str, object]]
    per_run_learning_summary: list[dict[str, object]]
    per_scale_algorithm_learning_summary: list[dict[str, object]]
    checkpoint_selection_summary: list[dict[str, object]]
    late_training_stability_summary: list[dict[str, object]]


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    count = len(pvalues)
    indexed: list[tuple[float, int]] = []
    for index, raw in enumerate(pvalues):
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"p-value must be finite and within [0,1]; got {raw!r}")
        indexed.append((value, index))
    indexed.sort(key=lambda item: (item[0], item[1]))
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (value, original_index) in enumerate(indexed):
        running_max = max(running_max, min(1.0, (count - rank) * value))
        adjusted[original_index] = running_max
    return adjusted


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = [float(value) for value in differences]
    if not values:
        raise ValueError("exact sign-flip test requires at least one paired difference")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("exact sign-flip differences must be finite")
    observed = abs(statistics.fmean(values))
    extreme = 0
    total = 1 << len(values)
    for mask in range(total):
        signed_mean = statistics.fmean(
            value if ((mask >> index) & 1) else -value
            for index, value in enumerate(values)
        )
        if abs(signed_mean) + 1e-15 >= observed:
            extreme += 1
    return extreme / total


def _float(value: object, field: str) -> float:
    try:
        result = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric; got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return result


def _int(value: object, field: str) -> int:
    text = str(value).strip()
    try:
        result = int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer; got {value!r}") from exc
    if str(result) != text:
        raise ValueError(f"{field} must be an integer; got {value!r}")
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return float(statistics.fmean(values))


def _sample_std(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    return float(statistics.median(values))


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.12g}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    if not rows:
        raise ValueError(f"refusing to represent an empty/placeholder CSV as complete: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(row.get(key, "")) for key in fieldnames})
    return path


def _write_text(path: Path, text: str) -> Path:
    if not text.strip():
        raise ValueError(f"refusing to write empty placeholder output: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _episode_means(rows: Sequence[Mapping[str, object]], metric: str) -> dict[tuple[str, str, int], float]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = (str(row["scale"]), str(row["algorithm"]), _int(row["seed"], "seed"))
        if metric not in row or str(row.get(metric, "")).strip() == "":
            raise ValueError(f"required metric {metric!r} is missing for {key}")
        grouped.setdefault(key, []).append(_float(row[metric], metric))
    expected = {
        (scale, algorithm, seed)
        for scale in workflow.SCALES
        for algorithm in workflow.FORMAL_ALGORITHMS
        for seed in workflow.SEEDS
    }
    if set(grouped) != expected:
        raise ValueError(f"metric {metric}: expected complete 80-cell inventory")
    for key, values in grouped.items():
        if len(values) != 30:
            raise ValueError(f"metric {metric}: {key} requires 30 deterministic episodes")
    return {key: _mean(values) for key, values in grouped.items()}


def _paired_effects(
    means: Mapping[tuple[str, str, int], float],
    *,
    metric: str,
    orientation: str,
    family: str,
) -> list[dict[str, object]]:
    raw_rows: list[dict[str, object]] = []
    pvalues: list[float] = []
    for scale in workflow.SCALES:
        corrected = [means[(scale, "actiongnn_nonnegative", seed)] for seed in workflow.SEEDS]
        hierarchy = [means[(scale, "hierarchical", seed)] for seed in workflow.SEEDS]
        if orientation == "hierarchy_minus_corrected":
            differences = [b - a for a, b in zip(corrected, hierarchy)]
        elif orientation == "corrected_minus_hierarchy":
            differences = [a - b for a, b in zip(corrected, hierarchy)]
        else:
            raise ValueError(f"unsupported orientation: {orientation}")
        t_stat, t_p, ci_low, ci_high, diff_sd = paired_t_test(differences)
        w_stat, w_p, w_method, w_n = wilcoxon_signed_rank(differences)
        mean_diff = _mean(differences)
        relative_seed_effects = [
            diff / abs(base) if base != 0.0 else math.nan
            for diff, base in zip(differences, corrected)
        ]
        finite_relative = [value for value in relative_seed_effects if math.isfinite(value)]
        leave_one_out = [
            _mean([value for index, value in enumerate(differences) if index != held_out])
            for held_out in range(len(differences))
        ]
        row = {
            "scale": scale,
            "metric": metric,
            "model_a": workflow.MODEL_NAMES["actiongnn_nonnegative"],
            "model_b": workflow.MODEL_NAMES["hierarchical"],
            "effect_orientation": orientation,
            "actiongnn_nonnegative_seed_mean": _mean(corrected),
            "actiongnn_nonnegative_seed_sd": _sample_std(corrected),
            "hierarchical_seed_mean": _mean(hierarchy),
            "hierarchical_seed_sd": _sample_std(hierarchy),
            "paired_differences": ";".join(_format(value) for value in differences),
            "paired_mean_difference": mean_diff,
            "paired_median_difference": _median(differences),
            "paired_relative_effect_mean": _mean(finite_relative) if finite_relative else math.nan,
            "paired_95ci_low": ci_low,
            "paired_95ci_high": ci_high,
            "paired_cohens_dz": (mean_diff / diff_sd) if diff_sd != 0.0 else (math.copysign(math.inf, mean_diff) if mean_diff else 0.0),
            "seeds_favouring_hierarchy": sum(value > 0.0 for value in differences),
            "n_paired_seeds": 10,
            "paired_t_statistic": t_stat,
            "paired_t_pvalue_two_sided": t_p,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue_two_sided": w_p,
            "wilcoxon_method": w_method,
            "wilcoxon_nonzero_n": w_n,
            "leave_one_seed_out_paired_means": ";".join(_format(value) for value in leave_one_out),
            "leave_one_seed_out_direction_stable": all(value > 0.0 for value in leave_one_out),
            "exact_sign_flip_randomisation_pvalue_two_sided": exact_sign_flip_pvalue(differences),
            "holm_family": family,
        }
        raw_rows.append(row)
        pvalues.append(t_p)
    for row, adjusted in zip(raw_rows, holm_adjust(pvalues)):
        row["holm_adjusted_pvalue"] = adjusted
        row["statistically_supported_harm"] = float(row["paired_95ci_high"]) < 0.0 and adjusted < 0.05
    return raw_rows


def _normalise_claim_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
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


def derive_service_guardrail_status(
    eval_rows: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, float] | None,
) -> ServiceGuardrailResult:
    metrics = ("energy_delivered", "evs_served", "average_satisfaction")
    descriptive_rows: list[dict[str, object]] = []
    policy_complete = policy is not None and all(
        f"{metric}_max_relative_decline" in policy for metric in metrics
    )
    any_fail = False
    any_missing = False
    for metric in metrics:
        try:
            means = _episode_means(eval_rows, metric)
        except ValueError:
            any_missing = True
            continue
        threshold = float(policy[f"{metric}_max_relative_decline"]) if policy_complete else math.nan
        if policy_complete and (not math.isfinite(threshold) or threshold < 0.0):
            raise ValueError(f"invalid pre-approved service threshold for {metric}")
        for scale in workflow.SCALES:
            corrected = _mean([means[(scale, "actiongnn_nonnegative", seed)] for seed in workflow.SEEDS])
            hierarchy = _mean([means[(scale, "hierarchical", seed)] for seed in workflow.SEEDS])
            relative_change = (hierarchy - corrected) / abs(corrected) if corrected != 0.0 else math.nan
            row_status = "UNKNOWN"
            if policy_complete and math.isfinite(relative_change):
                row_status = "FAIL" if relative_change < -threshold else "PASS"
                any_fail = any_fail or row_status == "FAIL"
            descriptive_rows.append(
                {
                    "scale": scale,
                    "metric": metric,
                    "corrected_nonnegative_actiongnn_mean": corrected,
                    "hierarchical_nonnegative_actiongnn_mean": hierarchy,
                    "relative_change_hierarchy_vs_corrected": relative_change,
                    "max_relative_decline_policy": threshold if policy_complete else "",
                    "guardrail_status": row_status,
                    "non_inferiority_claim_authorised": False,
                }
            )
    if any_missing or not policy_complete:
        status = "UNKNOWN"
    else:
        status = "FAIL" if any_fail else "PASS"
    return ServiceGuardrailResult(status=status, rows=descriptive_rows, policy_applied=bool(policy_complete))


def _theil_sen(points: Sequence[tuple[int, float]]) -> float:
    slopes = [
        (y2 - y1) / (x2 - x1)
        for index, (x1, y1) in enumerate(points)
        for x2, y2 in points[index + 1 :]
        if x2 != x1
    ]
    return _median(slopes)


def analyse_learning_dynamics(
    training_rows: Sequence[Mapping[str, object]],
    training_curve_rows: Sequence[Mapping[str, object]],
) -> LearningDynamicsResult:
    workflow.validate_training_gate(training_rows)
    summary_by_key = {
        (str(row["scale"]), str(row["algorithm"]), _int(row["seed"], "seed")): row
        for row in training_rows
    }
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    normalised_curve: list[dict[str, object]] = []
    for raw in training_curve_rows:
        key = (str(raw["scale"]), str(raw["algorithm"]), _int(raw["seed"], "seed"))
        row = {
            "task_id": _int(raw["task_id"], "task_id"),
            "scale": key[0],
            "algorithm": key[1],
            "model_name": workflow.MODEL_NAMES[key[1]],
            "seed": key[2],
            "timestep": _int(raw["timestep"], "timestep"),
            "eval_mean_reward": _float(raw["eval_mean_reward"], "eval_mean_reward"),
            "eval_std_reward": _float(raw["eval_std_reward"], "eval_std_reward"),
            "elapsed_seconds": _float(raw["elapsed_seconds"], "elapsed_seconds"),
            "source_identity": str(raw.get("source_identity", "")),
            "training_job_id": str(raw.get("training_job_id", "")),
        }
        grouped.setdefault(key, []).append(row)
        normalised_curve.append(row)
    if set(grouped) != set(summary_by_key):
        raise ValueError("training-curve inventory must match all 80 training cells")

    per_run: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    late_rows: list[dict[str, object]] = []
    expected_steps = list(workflow.scheduled_evaluation_steps())
    for key in sorted(grouped, key=lambda item: (workflow.SCALES.index(item[0]), workflow.FORMAL_ALGORITHMS.index(item[1]), item[2])):
        rows = sorted(grouped[key], key=lambda row: int(row["timestep"]))
        if [row["timestep"] for row in rows] != expected_steps:
            raise ValueError(f"{key}: training curves require the exact 15 scheduled checkpoints")
        best = max(rows, key=lambda row: float(row["eval_mean_reward"]))
        late = [row for row in rows if row["timestep"] in {60000, 65000, 70000, 75000}]
        if len(late) != 4:
            raise ValueError(f"{key}: final-four checkpoint evidence is incomplete")
        first, last = late[0], late[-1]
        loss_first = -float(first["eval_mean_reward"])
        loss_last = -float(last["eval_mean_reward"])
        relative_late_loss_change = (
            (loss_last - loss_first) / abs(loss_first) if loss_first != 0.0 else math.nan
        )
        noise_denominator = math.sqrt(
            float(first["eval_std_reward"]) ** 2 + float(last["eval_std_reward"]) ** 2
        )
        noise_standardised_change = (
            (float(last["eval_mean_reward"]) - float(first["eval_mean_reward"])) / noise_denominator
            if noise_denominator != 0.0
            else math.nan
        )
        terminal_slope = _theil_sen(
            [(int(row["timestep"]), float(row["eval_mean_reward"])) for row in late]
        )
        base = {
            "task_id": rows[0]["task_id"],
            "scale": key[0],
            "algorithm": key[1],
            "model_name": workflow.MODEL_NAMES[key[1]],
            "seed": key[2],
        }
        per_run.append(
            {
                **base,
                "scheduled_checkpoint_count": len(rows),
                "initial_scheduled_reward": rows[0]["eval_mean_reward"],
                "final_scheduled_reward": rows[-1]["eval_mean_reward"],
                "best_scheduled_reward": best["eval_mean_reward"],
                "model_best_step": best["timestep"],
                "best_checkpoint_percentage_of_75k": int(best["timestep"]) / 75000 * 100.0,
                "terminal_theil_sen_reward_slope_per_step": terminal_slope,
            }
        )
        checkpoint_rows.append(
            {
                **base,
                "checkpoint_role": "model.best",
                "model_best_step": best["timestep"],
                "model_best_reward": best["eval_mean_reward"],
                "best_checkpoint_percentage_of_75k": int(best["timestep"]) / 75000 * 100.0,
                "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
                "checkpoint_identity_sha256": summary_by_key[key].get("checkpoint_identity_sha256", ""),
            }
        )
        late_rows.append(
            {
                **base,
                "late_checkpoint_steps": "60000;65000;70000;75000",
                "late_checkpoint_count": 4,
                "relative_late_loss_change": relative_late_loss_change,
                "evaluation_noise_standardised_late_reward_change": noise_standardised_change,
                "theil_sen_terminal_slope": terminal_slope,
                "late_direction": "improving" if terminal_slope > 0 else ("declining" if terminal_slope < 0 else "flat"),
                "claim_boundary": "75k fixed-budget adequacy only; no asymptotic convergence claim",
            }
        )

    aggregate_rows: list[dict[str, object]] = []
    for scale in workflow.SCALES:
        for algorithm in workflow.FORMAL_ALGORITHMS:
            runs = [row for row in per_run if row["scale"] == scale and row["algorithm"] == algorithm]
            stability = [row for row in late_rows if row["scale"] == scale and row["algorithm"] == algorithm]
            aggregate_rows.append(
                {
                    "scale": scale,
                    "algorithm": algorithm,
                    "model_name": workflow.MODEL_NAMES[algorithm],
                    "n_training_seeds": len(runs),
                    "mean_model_best_step": _mean([float(row["model_best_step"]) for row in runs]),
                    "median_model_best_step": _median([float(row["model_best_step"]) for row in runs]),
                    "mean_best_checkpoint_percentage_of_75k": _mean([float(row["best_checkpoint_percentage_of_75k"]) for row in runs]),
                    "mean_terminal_theil_sen_slope": _mean([float(row["theil_sen_terminal_slope"]) for row in stability]),
                    "seeds_with_improving_terminal_slope": sum(float(row["theil_sen_terminal_slope"]) > 0 for row in stability),
                    "fixed_budget_interpretation_only": True,
                }
            )
    return LearningDynamicsResult(
        training_curve_long=normalised_curve,
        per_run_learning_summary=per_run,
        per_scale_algorithm_learning_summary=aggregate_rows,
        checkpoint_selection_summary=checkpoint_rows,
        late_training_stability_summary=late_rows,
    )


def _resource_summary(training_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    accounting_fields = {"total_cpu_seconds", "alloc_cpus", "max_rss_bytes", "req_mem_bytes"}
    for scale in workflow.SCALES:
        for algorithm in workflow.FORMAL_ALGORITHMS:
            group = [row for row in training_rows if str(row["scale"]) == scale and str(row["algorithm"]) == algorithm]
            elapsed = [_float(row["runtime_elapsed_seconds"], "runtime_elapsed_seconds") for row in group]
            complete_accounting = all(all(str(row.get(field, "")).strip() for field in accounting_fields) for row in group)
            output = {
                "scale": scale,
                "algorithm": algorithm,
                "model_name": workflow.MODEL_NAMES[algorithm],
                "n_runs": len(group),
                "elapsed_seconds_min": min(elapsed),
                "elapsed_seconds_median": _median(elapsed),
                "elapsed_seconds_max": max(elapsed),
                "runtime_evidence_status": "ACCOUNTING_COMPLETE" if complete_accounting else "ELAPSED_ONLY",
                "resource_defaults_guessed": False,
            }
            if complete_accounting:
                output.update(
                    {
                        "total_cpu_seconds_median": _median([_float(row["total_cpu_seconds"], "total_cpu_seconds") for row in group]),
                        "alloc_cpus_median": _median([_float(row["alloc_cpus"], "alloc_cpus") for row in group]),
                        "max_rss_bytes_max": max(_float(row["max_rss_bytes"], "max_rss_bytes") for row in group),
                        "req_mem_bytes_median": _median([_float(row["req_mem_bytes"], "req_mem_bytes") for row in group]),
                    }
                )
            rows.append(output)
    return rows


def _per_seed_primary(means: Mapping[tuple[str, str, int], float]) -> list[dict[str, object]]:
    return [
        {
            "scale": scale,
            "seed": seed,
            "primary_outcome": "episode_reward",
            "corrected_nonnegative_actiongnn_mean": means[(scale, "actiongnn_nonnegative", seed)],
            "hierarchical_nonnegative_actiongnn_mean": means[(scale, "hierarchical", seed)],
            "hierarchy_benefit": means[(scale, "hierarchical", seed)] - means[(scale, "actiongnn_nonnegative", seed)],
        }
        for scale in workflow.SCALES
        for seed in workflow.SEEDS
    ]


def _cross_scale_summary(reward: Sequence[Mapping[str, object]], boundary: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scale": reward_row["scale"],
            "reward_hierarchy_benefit": reward_row["paired_mean_difference"],
            "reward_relative_effect": reward_row["paired_relative_effect_mean"],
            "reward_paired_standardised_effect": reward_row["paired_cohens_dz"],
            "reward_seeds_favouring_hierarchy": reward_row["seeds_favouring_hierarchy"],
            "boundary_hierarchy_benefit": boundary_row["paired_mean_difference"],
            "boundary_paired_standardised_effect": boundary_row["paired_cohens_dz"],
            "direction_definition": "positive values favour Hierarchical non-negative ActionGNN",
            "raw_cross_scale_reward_comparison_prohibited": True,
        }
        for reward_row, boundary_row in zip(reward, boundary)
    ]


def _secondary_effects(
    eval_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    specifications = [
        (eval_rows, "tracking_error", "corrected_minus_hierarchy"),
        (eval_rows, "energy_tracking_error", "corrected_minus_hierarchy"),
        (eval_rows, "power_tracker_violation", "corrected_minus_hierarchy"),
        (eval_rows, "transformer_overload", "corrected_minus_hierarchy"),
        (eval_rows, "energy_delivered", "hierarchy_minus_corrected"),
        (eval_rows, "evs_served", "hierarchy_minus_corrected"),
        (eval_rows, "average_satisfaction", "hierarchy_minus_corrected"),
        (diagnostic_rows, "charger_upper_bound_action_fraction", "corrected_minus_hierarchy"),
        (diagnostic_rows, "transformer_positive_pressure_hhi", "corrected_minus_hierarchy"),
        (diagnostic_rows, "transformer_positive_pressure_gini", "corrected_minus_hierarchy"),
        (diagnostic_rows, "charger_positive_pressure_hhi", "corrected_minus_hierarchy"),
        (diagnostic_rows, "charger_positive_pressure_gini", "corrected_minus_hierarchy"),
    ]
    rows: list[dict[str, object]] = []
    for source, metric, orientation in specifications:
        if not source or any(metric not in row or str(row.get(metric, "")).strip() == "" for row in source):
            continue
        for row in _paired_effects(
            _episode_means(source, metric),
            metric=metric,
            orientation=orientation,
            family=f"secondary_{metric}_descriptive",
        ):
            row["outcome_class"] = "secondary"
            rows.append(row)
    if not rows:
        raise ValueError("no complete secondary-outcome evidence was available")
    return rows


def _svg_forest_plot(path: Path, rows: Sequence[Mapping[str, object]], *, value_field: str, title: str, x_label: str) -> Path:
    width, height = 760, 340
    left, right, top, bottom = 150, 40, 55, 55
    values: list[float] = []
    for row in rows:
        value = float(row[value_field])
        if not math.isfinite(value):
            value = math.copysign(8.0, value) if value != 0.0 else 0.0
        values.append(max(-8.0, min(8.0, value)))
    bound = max(1.0, max(abs(value) for value in values) * 1.25)
    x0 = left + (0.0 + bound) / (2 * bound) * (width - left - right)
    def x(value: float) -> float:
        return left + (value + bound) / (2 * bound) * (width - left - right)
    row_gap = (height - top - bottom) / max(1, len(rows))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{x0:.2f}" y1="{top-12}" x2="{x0:.2f}" y2="{height-bottom+5}" stroke="black" stroke-dasharray="4 4"/>',
    ]
    for index, (row, value) in enumerate(zip(rows, values)):
        y = top + (index + 0.5) * row_gap
        elements.extend([
            f'<text x="{left-15}" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="14">{html.escape(str(row["scale"]).upper())}</text>',
            f'<circle cx="{x(value):.2f}" cy="{y:.2f}" r="6" fill="black"/>',
            f'<text x="{x(value)+10:.2f}" y="{y+5:.2f}" font-family="sans-serif" font-size="12">{html.escape(_format(row[value_field]))}</text>',
        ])
    elements.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="13">{html.escape(x_label)}</text>',
        '</svg>\n',
    ])
    return _write_text(path, "\n".join(elements))


def reduce_formal_results(
    *,
    training_rows: Sequence[Mapping[str, object]],
    training_curve_rows: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    out_dir: Path | str,
    service_policy: Mapping[str, float] | None = None,
) -> list[Path]:
    try:
        workflow.validate_training_gate(training_rows)
        workflow.validate_eval30_gate(eval_rows)
        workflow.validate_diagnostic_gate(diagnostic_rows)
        learning = analyse_learning_dynamics(training_rows, training_curve_rows)
    except ValueError as exc:
        raise ValueError(f"STATUS=BLOCKED: {exc}") from exc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reward_means = _episode_means(eval_rows, "episode_reward")
    boundary_means = _episode_means(diagnostic_rows, workflow.KEY_MECHANISM_METRIC)
    reward_effects = _paired_effects(
        reward_means,
        metric="episode_reward",
        orientation="hierarchy_minus_corrected",
        family="primary_reward_by_scale",
    )
    boundary_effects = _paired_effects(
        boundary_means,
        metric=workflow.KEY_MECHANISM_METRIC,
        orientation="corrected_minus_hierarchy",
        family="key_mechanism_boundary_by_scale",
    )
    secondary = _secondary_effects(eval_rows, diagnostic_rows)
    service = derive_service_guardrail_status(eval_rows, policy=service_policy)
    claims = workflow.assess_claims(
        _normalise_claim_rows(reward_effects),
        _normalise_claim_rows(boundary_effects),
        {"status": service.status},
    )
    claims["PRIMARY_INFERENCE_UNIT"] = "paired_training_seed"
    claims["PRIMARY_EFFICACY_OUTCOME"] = "episode_reward"
    claims["KEY_MECHANISM_OUTCOME"] = workflow.KEY_MECHANISM_METRIC

    outputs: list[Path] = []
    outputs.append(_write_csv(out / "formal_training_matrix_summary.csv", training_rows))
    outputs.append(_write_csv(out / "training_curve_long.csv", learning.training_curve_long))
    outputs.append(_write_csv(out / "per_run_learning_summary.csv", learning.per_run_learning_summary))
    outputs.append(_write_csv(out / "per_scale_algorithm_learning_summary.csv", learning.per_scale_algorithm_learning_summary))
    outputs.append(_write_csv(out / "checkpoint_selection_summary.csv", learning.checkpoint_selection_summary))
    outputs.append(_write_csv(out / "late_training_stability_summary.csv", learning.late_training_stability_summary))
    outputs.append(_write_csv(out / "canonical_eval30_episode_rows.csv", eval_rows))
    outputs.append(_write_csv(out / "diagnostic_episode_rows.csv", diagnostic_rows))
    outputs.append(_write_csv(out / "per_seed_primary_outcomes.csv", _per_seed_primary(reward_means)))
    outputs.append(_write_csv(out / "per_scale_paired_reward_effects.csv", reward_effects))
    outputs.append(_write_csv(out / "per_scale_paired_boundary_effects.csv", boundary_effects))
    outputs.append(_write_csv(out / "primary_reward_statistical_tests.csv", reward_effects))
    outputs.append(_write_csv(out / "key_mechanism_statistical_tests.csv", boundary_effects))
    outputs.append(_write_csv(out / "secondary_outcome_statistical_tests.csv", secondary))
    outputs.append(_write_csv(out / "service_guardrail_summary.csv", service.rows))
    outputs.append(_write_csv(out / "cross_scale_direction_summary.csv", _cross_scale_summary(reward_effects, boundary_effects)))
    outputs.append(_write_csv(out / "resource_efficiency_summary.csv", _resource_summary(training_rows)))

    claim_text = "\n".join(f"{key}={value}" for key, value in sorted(claims.items())) + "\n"
    outputs.append(_write_text(out / "claim_assessment.env", claim_text))
    service_sentence = (
        "Service guardrail status is UNKNOWN because no pre-approved quantitative preservation policy was supplied."
        if service.status == "UNKNOWN"
        else f"The pre-approved operational service guardrail status is {service.status}; this is not a service non-inferiority claim."
    )
    scientific_md = "\n".join([
        "# Formal 75k Non-negative Architecture Comparison",
        "",
        "Primary efficacy outcome: `episode_reward`; primary inference unit: ten paired training seeds per scale.",
        f"Key mechanism outcome: `{workflow.KEY_MECHANISM_PUBLIC_LABEL}`.",
        "Tracking error is the equivalent descriptive control-loss view, not a second primary hypothesis.",
        service_sentence,
        "Learning dynamics describe fixed-budget adequacy at 75,000 steps and do not establish asymptotic convergence.",
        "",
        "## Claim assessment",
        *[f"- `{key}={value}`" for key, value in sorted(claims.items())],
        "",
    ])
    outputs.append(_write_text(out / "formal_scientific_results.md", scientific_md))
    summary_items = "".join(f"<li><code>{html.escape(key)}={html.escape(value)}</code></li>" for key, value in sorted(claims.items()))
    outputs.append(_write_text(
        out / "formal_supervisor_summary.html",
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Formal 75k summary</title></head>"
        f"<body><h1>Formal 75k architecture comparison</h1><p>{html.escape(service_sentence)}</p><ul>{summary_items}</ul></body></html>\n",
    ))

    table_lines = [
        "# Paper-ready paired effects",
        "",
        "Positive effects favour Hierarchical non-negative ActionGNN. Intervals are 95% paired t intervals; n=10 paired training seeds per scale.",
        "",
        "| Scale | Reward mean benefit | Reward 95% CI | Reward Holm p | Upper-bound fraction benefit | Mechanism 95% CI | Mechanism Holm p |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for reward, boundary in zip(reward_effects, boundary_effects):
        table_lines.append(
            f"| {str(reward['scale']).upper()} | {_format(reward['paired_mean_difference'])} | "
            f"[{_format(reward['paired_95ci_low'])}, {_format(reward['paired_95ci_high'])}] | {_format(reward['holm_adjusted_pvalue'])} | "
            f"{_format(boundary['paired_mean_difference'])} | [{_format(boundary['paired_95ci_low'])}, {_format(boundary['paired_95ci_high'])}] | "
            f"{_format(boundary['holm_adjusted_pvalue'])} |"
        )
    outputs.append(_write_text(out / "paper_ready_tables.md", "\n".join(table_lines) + "\n"))
    html_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(reward['scale']).upper())}</td>"
        f"<td>{html.escape(_format(reward['paired_mean_difference']))}</td>"
        f"<td>[{html.escape(_format(reward['paired_95ci_low']))}, {html.escape(_format(reward['paired_95ci_high']))}]</td>"
        f"<td>{html.escape(_format(reward['holm_adjusted_pvalue']))}</td>"
        f"<td>{html.escape(_format(boundary['paired_mean_difference']))}</td>"
        f"<td>[{html.escape(_format(boundary['paired_95ci_low']))}, {html.escape(_format(boundary['paired_95ci_high']))}]</td>"
        f"<td>{html.escape(_format(boundary['holm_adjusted_pvalue']))}</td>"
        "</tr>"
        for reward, boundary in zip(reward_effects, boundary_effects)
    )
    outputs.append(_write_text(
        out / "paper_ready_tables.html",
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Paper-ready paired effects</title></head><body>"
        "<p>Positive effects favour Hierarchical non-negative ActionGNN; n=10 paired seeds per scale.</p>"
        "<table><thead><tr><th>Scale</th><th>Reward benefit</th><th>Reward 95% CI</th><th>Reward Holm p</th>"
        "<th>Boundary benefit</th><th>Mechanism 95% CI</th><th>Mechanism Holm p</th></tr></thead>"
        f"<tbody>{html_rows}</tbody></table></body></html>\n",
    ))

    figures = out / "paper_ready_figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs.append(_svg_forest_plot(
        figures / "reward_standardised_effect_by_scale.svg",
        reward_effects,
        value_field="paired_cohens_dz",
        title="Paired standardised reward effect by scale",
        x_label="Paired Cohen's dz; positive favours hierarchy (display capped at ±8)",
    ))
    outputs.append(_svg_forest_plot(
        figures / "boundary_effect_by_scale.svg",
        boundary_effects,
        value_field="paired_mean_difference",
        title="Upper-bound active-EV action fraction benefit by scale",
        x_label="Corrected minus hierarchical fraction; positive favours hierarchy",
    ))
    return outputs
