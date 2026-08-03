"""Compare EV infrastructure control architectures at training-seed level."""

from __future__ import annotations

import math
import csv
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from analysis.ev_charging_infrastructure_control.metric_definitions import (
    METRIC_DEFINITIONS,
    PRIMARY,
)


SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
TRAINING_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class SeedMetricObservation:
    scale: str
    algorithm: str
    training_seed: int
    metric_name: str
    value: float


@dataclass(frozen=True)
class PairedSeedMetric:
    training_seed: int
    actiongnn_value: float
    hierarchical_value: float


@dataclass(frozen=True)
class PairedStatistics:
    n_pairs: int
    actiongnn_mean: float
    hierarchical_mean: float
    mean_difference: float
    relative_difference_percent_of_actiongnn_mean: float | None
    relative_difference_status: str
    sample_standard_deviation: float
    cohens_dz: float | None
    effect_size_status: str
    t_statistic: float
    paired_t_p_value: float
    confidence_interval_95_low: float
    confidence_interval_95_high: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    wilcoxon_method: str
    wilcoxon_nonzero_count: int


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_finite_float(value: object, context: str) -> float:
    text = "" if value is None else str(value).strip()
    if text == "":
        fail(f"{context}: missing numeric value")
    try:
        number = float(text)
    except ValueError as exc:
        raise RuntimeError(f"{context}: invalid numeric value {text!r}") from exc
    if not math.isfinite(number):
        fail(f"{context}: non-finite numeric value {text!r}")
    return number


def parse_int(value: object, context: str) -> int:
    number = parse_finite_float(value, context)
    if not float(number).is_integer():
        fail(f"{context}: expected integer, found {value!r}")
    return int(number)


def mean(values: list[float]) -> float:
    if not values:
        fail("mean requires at least one value")
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        fail("sample standard deviation requires at least two values")
    value_mean = sum(values) / len(values)
    variance = sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def observation_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        str(row["scale"]),
        str(row["algorithm"]),
        parse_int(row["training_seed"], "training_seed"),
    )


def aggregate_episode_metric(
    rows: list[dict[str, str]],
    metric_name: str,
) -> dict[tuple[str, str, int], float]:
    grouped_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = observation_key(row)
        grouped_values[key].append(
            parse_finite_float(row.get(metric_name), f"{key}: {metric_name}")
        )
    return {key: mean(values) for key, values in grouped_values.items()}


def aggregate_transformer_overload_metrics(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, float]]:
    frequency_by_episode: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    magnitude_sum_by_episode: dict[tuple[str, str, int, int], float] = defaultdict(float)
    maximum_by_seed: dict[tuple[str, str, int], float] = {}

    for row in rows:
        scale, algorithm, training_seed = observation_key(row)
        episode_index = parse_int(row["episode_index"], "episode_index")
        seed_key = (scale, algorithm, training_seed)
        episode_key = (scale, algorithm, training_seed, episode_index)
        frequency_by_episode[episode_key].append(
            parse_finite_float(
                row["overload_frequency_fraction"],
                f"{episode_key}: overload_frequency_fraction",
            )
        )
        magnitude_sum_by_episode[episode_key] += parse_finite_float(
            row["overload_magnitude_sum"],
            f"{episode_key}: overload_magnitude_sum",
        )
        overload_magnitude_max = parse_finite_float(
            row["overload_magnitude_max"],
            f"{episode_key}: overload_magnitude_max",
        )
        maximum_by_seed[seed_key] = max(
            maximum_by_seed.get(seed_key, overload_magnitude_max),
            overload_magnitude_max,
        )

    episode_frequency_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    episode_magnitude_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for episode_key, frequency_values in frequency_by_episode.items():
        seed_key = episode_key[:3]
        episode_frequency_values[seed_key].append(mean(frequency_values))
    for episode_key, magnitude_sum in magnitude_sum_by_episode.items():
        seed_key = episode_key[:3]
        episode_magnitude_values[seed_key].append(magnitude_sum)

    return {
        seed_key: {
            "mean_transformer_overload_frequency_fraction": mean(
                episode_frequency_values[seed_key]
            ),
            "mean_total_transformer_overload_magnitude": mean(
                episode_magnitude_values[seed_key]
            ),
            "maximum_transformer_overload_magnitude": maximum_by_seed[seed_key],
        }
        for seed_key in sorted(maximum_by_seed)
    }


def build_seed_level_metrics(
    episode_rows: list[dict[str, str]],
    transformer_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]] | None = None,
) -> list[SeedMetricObservation]:
    observations: list[SeedMetricObservation] = []
    seed_summary_values: dict[tuple[str, str, int, str], float] = {}
    if seed_rows is not None:
        for seed_row in seed_rows:
            seed_key = observation_key(seed_row)
            for definition in METRIC_DEFINITIONS:
                if definition.seed_summary_column is None:
                    continue
                seed_summary_values[
                    (*seed_key, definition.seed_summary_column)
                ] = parse_finite_float(
                    seed_row[definition.seed_summary_column],
                    f"{seed_key}: {definition.seed_summary_column}",
                )

    for definition in METRIC_DEFINITIONS:
        if definition.source_level == "episode":
            aggregated = aggregate_episode_metric(episode_rows, definition.source_column)
            if definition.seed_summary_column is not None:
                for key, calculated_value in aggregated.items():
                    reported_value = seed_summary_values.get(
                        (*key, definition.seed_summary_column)
                    )
                    if reported_value is not None and not math.isclose(
                        calculated_value,
                        reported_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        fail(
                            "seed summary mismatch for "
                            f"{key}/{definition.name}: calculated "
                            f"{calculated_value}, reported {reported_value}"
                        )
            observations.extend(
                SeedMetricObservation(
                    scale=scale,
                    algorithm=algorithm,
                    training_seed=training_seed,
                    metric_name=definition.name,
                    value=value,
                )
                for (scale, algorithm, training_seed), value in aggregated.items()
            )
    transformer_metrics = aggregate_transformer_overload_metrics(transformer_rows)
    for (scale, algorithm, training_seed), metric_values in transformer_metrics.items():
        observations.extend(
            SeedMetricObservation(
                scale=scale,
                algorithm=algorithm,
                training_seed=training_seed,
                metric_name=metric_name,
                value=value,
            )
            for metric_name, value in metric_values.items()
        )
    return observations


def validate_exact_algorithm_pairs(
    observations: list[SeedMetricObservation],
    scale: str,
    metric_name: str,
) -> tuple[PairedSeedMetric, ...]:
    values_by_algorithm_seed: dict[tuple[str, int], float] = {}
    for observation in observations:
        if observation.scale != scale or observation.metric_name != metric_name:
            continue
        key = (observation.algorithm, observation.training_seed)
        if key in values_by_algorithm_seed:
            fail(f"duplicate seed observation for {scale}/{metric_name}: {key}")
        values_by_algorithm_seed[key] = observation.value

    observed_algorithms = {
        algorithm for algorithm, _training_seed in values_by_algorithm_seed
    }
    if observed_algorithms != set(ALGORITHMS):
        fail(f"missing algorithm cell for {scale}/{metric_name}")

    actiongnn_seeds = {
        training_seed
        for algorithm, training_seed in values_by_algorithm_seed
        if algorithm == "actiongnn"
    }
    hierarchical_seeds = {
        training_seed
        for algorithm, training_seed in values_by_algorithm_seed
        if algorithm == "hierarchical"
    }
    if actiongnn_seeds != hierarchical_seeds:
        fail(f"unequal seed sets for {scale}/{metric_name}")
    if actiongnn_seeds != set(TRAINING_SEEDS):
        fail(f"{scale}/{metric_name} requires exactly five paired seeds")

    return tuple(
        PairedSeedMetric(
            training_seed=training_seed,
            actiongnn_value=values_by_algorithm_seed[("actiongnn", training_seed)],
            hierarchical_value=values_by_algorithm_seed[
                ("hierarchical", training_seed)
            ],
        )
        for training_seed in TRAINING_SEEDS
    )


def regularized_beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c_value = 1.0
    d_value = 1.0 - qab * x / qap
    if abs(d_value) < fpmin:
        d_value = fpmin
    d_value = 1.0 / d_value
    h_value = d_value

    for iteration in range(1, max_iterations + 1):
        doubled_iteration = 2 * iteration
        aa_value = (
            iteration
            * (b - iteration)
            * x
            / ((qam + doubled_iteration) * (a + doubled_iteration))
        )
        d_value = 1.0 + aa_value * d_value
        if abs(d_value) < fpmin:
            d_value = fpmin
        c_value = 1.0 + aa_value / c_value
        if abs(c_value) < fpmin:
            c_value = fpmin
        d_value = 1.0 / d_value
        h_value *= d_value * c_value

        aa_value = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled_iteration) * (qap + doubled_iteration))
        )
        d_value = 1.0 + aa_value * d_value
        if abs(d_value) < fpmin:
            d_value = fpmin
        c_value = 1.0 + aa_value / c_value
        if abs(c_value) < fpmin:
            c_value = fpmin
        d_value = 1.0 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1.0) < eps:
            return h_value

    fail("Incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0:
        fail(f"regularized beta x out of range: {x}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_beta_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta_term)

    if x < (a + 1.0) / (a + b + 2.0):
        return beta_term * regularized_beta_continued_fraction(a, b, x) / a
    return 1.0 - beta_term * regularized_beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t_value: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        fail("degrees_of_freedom must be positive")
    if math.isinf(t_value):
        return 1.0 if t_value > 0 else 0.0
    x_value = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    beta_value = regularized_incomplete_beta(x_value, degrees_of_freedom / 2.0, 0.5)
    if t_value >= 0.0:
        return 1.0 - 0.5 * beta_value
    return 0.5 * beta_value


def student_t_ppf(probability: float, degrees_of_freedom: int) -> float:
    if not 0.0 < probability < 1.0:
        fail("probability must be between 0 and 1")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_ppf(1.0 - probability, degrees_of_freedom)

    low = 0.0
    high = 1.0
    while student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
    for _iteration in range(100):
        midpoint = (low + high) / 2.0
        if student_t_cdf(midpoint, degrees_of_freedom) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def paired_t_test(diff_values: Sequence[float]) -> tuple[float, float, float, float, float]:
    sample_count = len(diff_values)
    if sample_count < 2:
        fail("paired t-test requires at least two paired seeds")
    diff_mean = sum(diff_values) / sample_count
    diff_std = sample_std(diff_values)
    if diff_std == 0.0:
        t_statistic = 0.0 if diff_mean == 0.0 else math.copysign(float("inf"), diff_mean)
        p_value = 1.0 if diff_mean == 0.0 else 0.0
        return t_statistic, p_value, diff_mean, diff_mean, diff_std

    standard_error = diff_std / math.sqrt(sample_count)
    t_statistic = diff_mean / standard_error
    p_value = 2.0 * (1.0 - student_t_cdf(abs(t_statistic), sample_count - 1))
    p_value = min(1.0, max(0.0, p_value))
    t_critical = student_t_ppf(0.975, sample_count - 1)
    ci_low = diff_mean - t_critical * standard_error
    ci_high = diff_mean + t_critical * standard_error
    return t_statistic, p_value, ci_low, ci_high, diff_std


def average_ranks_for_absolute_values(values: Sequence[float]) -> list[float]:
    sorted_pairs = sorted(
        enumerate(abs(value) for value in values),
        key=lambda item: item[1],
    )
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while (
            end < len(sorted_pairs)
            and math.isclose(
                sorted_pairs[end][1],
                sorted_pairs[index][1],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for pair_index in range(index, end):
            original_index = sorted_pairs[pair_index][0]
            ranks[original_index] = average_rank
        index = end
    return ranks


def wilcoxon_signed_rank(diff_values: Sequence[float]) -> tuple[float, float, str, int]:
    nonzero_diffs = [
        value
        for value in diff_values
        if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)
    ]
    if not nonzero_diffs:
        return 0.0, 1.0, "exact", 0
    if len(nonzero_diffs) > 20:
        fail(f"exact Wilcoxon signed-rank test is not feasible for n={len(nonzero_diffs)}")

    ranks = average_ranks_for_absolute_values(nonzero_diffs)
    total_rank = sum(ranks)
    w_plus = sum(rank for rank, value in zip(ranks, nonzero_diffs) if value > 0.0)
    w_minus = total_rank - w_plus
    statistic = min(w_plus, w_minus)

    possible_sums = [0.0]
    for rank in ranks:
        possible_sums += [current_sum + rank for current_sum in possible_sums]
    extreme_count = sum(
        1
        for rank_sum in possible_sums
        if min(rank_sum, total_rank - rank_sum) <= statistic + 1e-12
    )
    return statistic, min(1.0, extreme_count / len(possible_sums)), "exact", len(nonzero_diffs)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    previous_adjusted = 0.0
    total = len(p_values)
    for rank, (original_index, p_value) in enumerate(indexed, start=1):
        raw_adjusted = min(1.0, (total - rank + 1) * p_value)
        previous_adjusted = max(previous_adjusted, raw_adjusted)
        adjusted[original_index] = previous_adjusted
    return adjusted


def paired_statistics(
    actiongnn_values: Sequence[float],
    hierarchical_values: Sequence[float],
) -> PairedStatistics:
    if len(actiongnn_values) != len(hierarchical_values):
        fail("paired statistics require equal length inputs")
    if len(actiongnn_values) < 2:
        fail("paired statistics require at least two pairs")
    action_values = [
        parse_finite_float(value, "actiongnn paired value") for value in actiongnn_values
    ]
    hierarchical_seed_values = [
        parse_finite_float(value, "hierarchical paired value")
        for value in hierarchical_values
    ]
    diff_values = [
        hierarchical_value - action_value
        for action_value, hierarchical_value in zip(action_values, hierarchical_seed_values)
    ]
    action_mean = mean(action_values)
    hierarchical_mean = mean(hierarchical_seed_values)
    mean_difference = mean(diff_values)
    if action_mean == 0.0:
        relative_difference = None
        relative_status = "undefined_zero_actiongnn_mean"
    else:
        relative_difference = 100.0 * mean_difference / abs(action_mean)
        relative_status = "defined"

    t_statistic, t_p_value, ci_low, ci_high, diff_std = paired_t_test(diff_values)
    if diff_std == 0.0 and mean_difference == 0.0:
        cohens_dz = 0.0
        effect_status = "all_differences_zero"
    elif diff_std == 0.0:
        cohens_dz = None
        effect_status = "undefined_zero_variance_nonzero_mean"
    else:
        cohens_dz = mean_difference / diff_std
        effect_status = "defined"
    wilcoxon_statistic, wilcoxon_p_value, wilcoxon_method, wilcoxon_nonzero_count = (
        wilcoxon_signed_rank(diff_values)
    )
    return PairedStatistics(
        n_pairs=len(action_values),
        actiongnn_mean=action_mean,
        hierarchical_mean=hierarchical_mean,
        mean_difference=mean_difference,
        relative_difference_percent_of_actiongnn_mean=relative_difference,
        relative_difference_status=relative_status,
        sample_standard_deviation=diff_std,
        cohens_dz=cohens_dz,
        effect_size_status=effect_status,
        t_statistic=t_statistic,
        paired_t_p_value=t_p_value,
        confidence_interval_95_low=ci_low,
        confidence_interval_95_high=ci_high,
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_p_value=wilcoxon_p_value,
        wilcoxon_method=wilcoxon_method,
        wilcoxon_nonzero_count=wilcoxon_nonzero_count,
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return format(value, ".17g")


def observed_favourable_direction(preferred_direction: str, mean_difference: float) -> str:
    if preferred_direction == "context_dependent":
        return "context_dependent"
    if math.isclose(mean_difference, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "no_difference"
    if preferred_direction == "higher":
        return "hierarchical" if mean_difference > 0.0 else "actiongnn"
    if preferred_direction == "lower":
        return "hierarchical" if mean_difference < 0.0 else "actiongnn"
    fail(f"unsupported preferred direction: {preferred_direction}")


PAIRED_COMPARISON_COLUMNS = (
    "scale",
    "metric_name",
    "tier",
    "preferred_direction",
    "aggregation_rule",
    "interpretation_warning",
    "n_paired_seeds",
    "actiongnn_seed_mean",
    "hierarchical_seed_mean",
    "paired_difference_hierarchical_minus_actiongnn",
    "relative_difference_percent_of_actiongnn_mean",
    "relative_difference_status",
    "paired_difference_sample_standard_deviation",
    "cohens_dz",
    "effect_size_status",
    "paired_t_statistic",
    "paired_t_p_value_two_sided",
    "paired_difference_95ci_low",
    "paired_difference_95ci_high",
    "wilcoxon_statistic",
    "wilcoxon_p_value_two_sided",
    "wilcoxon_method",
    "wilcoxon_nonzero_count",
    "holm_adjusted_p_value",
    "holm_reject_0_05",
    "observed_favourable_direction",
)

SCALE_SUMMARY_COLUMNS = (
    "scale",
    "paired_seed_count",
    "primary_metric_count",
    "primary_metrics_favouring_hierarchical",
    "primary_metrics_favouring_actiongnn",
    "primary_metrics_no_difference",
    "primary_raw_p_below_0_05_count",
    "primary_holm_p_below_0_05_count",
    "mechanism_metric_count",
    "physical_safety_metric_count",
    "service_guardrail_metric_count",
    "robustness_metric_count",
)


def comparison_row(
    scale: str,
    definition,
    statistics: PairedStatistics,
) -> dict[str, str]:
    return {
        "scale": scale,
        "metric_name": definition.name,
        "tier": definition.tier,
        "preferred_direction": definition.preferred_direction,
        "aggregation_rule": definition.aggregation_rule,
        "interpretation_warning": definition.interpretation_warning,
        "n_paired_seeds": str(statistics.n_pairs),
        "actiongnn_seed_mean": format_optional_float(statistics.actiongnn_mean),
        "hierarchical_seed_mean": format_optional_float(statistics.hierarchical_mean),
        "paired_difference_hierarchical_minus_actiongnn": format_optional_float(
            statistics.mean_difference
        ),
        "relative_difference_percent_of_actiongnn_mean": format_optional_float(
            statistics.relative_difference_percent_of_actiongnn_mean
        ),
        "relative_difference_status": statistics.relative_difference_status,
        "paired_difference_sample_standard_deviation": format_optional_float(
            statistics.sample_standard_deviation
        ),
        "cohens_dz": format_optional_float(statistics.cohens_dz),
        "effect_size_status": statistics.effect_size_status,
        "paired_t_statistic": format_optional_float(statistics.t_statistic),
        "paired_t_p_value_two_sided": format_optional_float(statistics.paired_t_p_value),
        "paired_difference_95ci_low": format_optional_float(
            statistics.confidence_interval_95_low
        ),
        "paired_difference_95ci_high": format_optional_float(
            statistics.confidence_interval_95_high
        ),
        "wilcoxon_statistic": format_optional_float(statistics.wilcoxon_statistic),
        "wilcoxon_p_value_two_sided": format_optional_float(statistics.wilcoxon_p_value),
        "wilcoxon_method": statistics.wilcoxon_method,
        "wilcoxon_nonzero_count": str(statistics.wilcoxon_nonzero_count),
        "holm_adjusted_p_value": "",
        "holm_reject_0_05": "false",
        "observed_favourable_direction": observed_favourable_direction(
            definition.preferred_direction,
            statistics.mean_difference,
        ),
    }


def apply_holm_to_primary_rows(rows: list[dict[str, str]]) -> None:
    for scale in SCALES:
        primary_rows = [
            row for row in rows if row["scale"] == scale and row["tier"] == PRIMARY
        ]
        if len(primary_rows) != 4:
            fail(f"expected four primary rows for scale {scale}")
        adjusted_values = holm_adjust(
            [float(row["paired_t_p_value_two_sided"]) for row in primary_rows]
        )
        for row, adjusted_value in zip(primary_rows, adjusted_values):
            row["holm_adjusted_p_value"] = format_optional_float(adjusted_value)
            row["holm_reject_0_05"] = "true" if adjusted_value < 0.05 else "false"


def build_comparison_rows(
    observations: list[SeedMetricObservation],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scale in SCALES:
        for definition in METRIC_DEFINITIONS:
            paired_seed_metrics = validate_exact_algorithm_pairs(
                observations, scale, definition.name
            )
            statistics = paired_statistics(
                actiongnn_values=[
                    pair.actiongnn_value for pair in paired_seed_metrics
                ],
                hierarchical_values=[
                    pair.hierarchical_value for pair in paired_seed_metrics
                ],
            )
            rows.append(comparison_row(scale, definition, statistics))
    apply_holm_to_primary_rows(rows)
    return rows


def build_scale_summary_rows(comparison_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    for scale in SCALES:
        scale_rows = [row for row in comparison_rows if row["scale"] == scale]
        primary_rows = [row for row in scale_rows if row["tier"] == PRIMARY]
        summary_rows.append(
            {
                "scale": scale,
                "paired_seed_count": "5",
                "primary_metric_count": str(len(primary_rows)),
                "primary_metrics_favouring_hierarchical": str(
                    sum(
                        1
                        for row in primary_rows
                        if row["observed_favourable_direction"] == "hierarchical"
                    )
                ),
                "primary_metrics_favouring_actiongnn": str(
                    sum(
                        1
                        for row in primary_rows
                        if row["observed_favourable_direction"] == "actiongnn"
                    )
                ),
                "primary_metrics_no_difference": str(
                    sum(
                        1
                        for row in primary_rows
                        if row["observed_favourable_direction"] == "no_difference"
                    )
                ),
                "primary_raw_p_below_0_05_count": str(
                    sum(
                        1
                        for row in primary_rows
                        if float(row["paired_t_p_value_two_sided"]) < 0.05
                    )
                ),
                "primary_holm_p_below_0_05_count": str(
                    sum(
                        1
                        for row in primary_rows
                        if row["holm_adjusted_p_value"] != ""
                        and float(row["holm_adjusted_p_value"]) < 0.05
                    )
                ),
                "mechanism_metric_count": str(
                    sum(1 for row in scale_rows if row["tier"] == "mechanism")
                ),
                "physical_safety_metric_count": str(
                    sum(1 for row in scale_rows if row["tier"] == "physical_safety")
                ),
                "service_guardrail_metric_count": str(
                    sum(1 for row in scale_rows if row["tier"] == "service_guardrail")
                ),
                "robustness_metric_count": str(
                    sum(1 for row in scale_rows if row["tier"] == "robustness")
                ),
            }
        )
    return summary_rows


def write_csv_file(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metric_interpretation(path: Path, comparison_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Metric Interpretation",
        "",
        "| scale | metric | tier | direction | aggregation | warning | observed direction |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {scale} | {metric_name} | {tier} | {preferred_direction} | "
            "{aggregation_rule} | {interpretation_warning} | "
            "{observed_favourable_direction} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish_results(
    analysis_dir: Path,
    comparison_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> None:
    results_dir = analysis_dir / "results"
    if results_dir.exists():
        fail(f"results directory already exists: {results_dir}")
    temporary_results_dir = analysis_dir / f".results.tmp.{os.getpid()}"
    if temporary_results_dir.exists():
        fail(f"temporary results directory already exists: {temporary_results_dir}")
    try:
        temporary_results_dir.mkdir()
        write_csv_file(
            temporary_results_dir / "paired_control_comparisons.csv",
            PAIRED_COMPARISON_COLUMNS,
            comparison_rows,
        )
        write_csv_file(
            temporary_results_dir / "scale_level_summary.csv",
            SCALE_SUMMARY_COLUMNS,
            summary_rows,
        )
        write_metric_interpretation(
            temporary_results_dir / "metric_interpretation.md",
            comparison_rows,
        )
        os.replace(temporary_results_dir, results_dir)
    except Exception:
        shutil.rmtree(temporary_results_dir, ignore_errors=True)
        raise


def compare_control_architectures(
    analysis_dir: str | Path,
    *,
    expected_episodes_per_seed: int = 30,
) -> dict[str, int]:
    resolved_analysis_dir = Path(analysis_dir).expanduser().resolve()
    provenance_path = resolved_analysis_dir / "provenance.json"
    if not provenance_path.is_file():
        fail(f"missing provenance.json: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("diagnostic_schema_version") != "3":
        fail("provenance validation failed: diagnostic_schema_version")
    dataset_dir = resolved_analysis_dir / "datasets"
    episode_rows = read_csv_rows(dataset_dir / "episode_metrics.csv")
    seed_rows = read_csv_rows(dataset_dir / "seed_metrics.csv")
    transformer_rows = read_csv_rows(dataset_dir / "transformer_metrics.csv")
    observations = build_seed_level_metrics(episode_rows, transformer_rows, seed_rows)
    comparison_rows = build_comparison_rows(observations)
    summary_rows = build_scale_summary_rows(comparison_rows)
    if len(comparison_rows) != 88:
        fail(f"expected 88 paired comparison rows, found {len(comparison_rows)}")
    if len(summary_rows) != 4:
        fail(f"expected 4 scale summary rows, found {len(summary_rows)}")
    publish_results(resolved_analysis_dir, comparison_rows, summary_rows)
    return {
        "paired_control_comparisons": len(comparison_rows),
        "scale_level_summary": len(summary_rows),
    }
