#!/usr/bin/env python3
"""Aggregate controlled multiscale eval30 CSVs for ActionGNN vs hierarchical.

This script uses only the Python standard library plus numpy. It treats the
training seed as the inference unit: episode rows are averaged within each
scale/algorithm/seed before paired statistical tests are computed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
SEEDS = tuple(range(5))
EXPECTED_EPISODES_PER_FILE = 30
EXPECTED_CSV_COUNT = len(SCALES) * len(ALGORITHMS) * len(SEEDS)
EPISODE_SEED_STRIDE = 1000
EVAL_SEED_BASE_BY_SCALE = {
    "25cp": 710000,
    "100cp": 720000,
    "500cp": 730000,
    "1000cp": 740000,
}

CSV_NAME_RE = re.compile(
    r"^(?P<scale>25cp|100cp|500cp|1000cp)_"
    r"(?P<algorithm>actiongnn|hierarchical)_"
    r"seed(?P<seed>[0-4])_eval30\.csv$"
)

REQUIRED_METRICS = [
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_transformer_overload",
    "total_ev_served",
    "total_energy_charged",
    "total_energy_discharged",
    "battery_degradation",
    "battery_degradation_calendar",
    "battery_degradation_cycling",
    "total_profits",
    "action_mean",
    "action_std",
    "action_fraction_zero",
    "action_fraction_at_max",
    "active_action_count_mean",
]

REQUIRED_COLUMNS = {
    "row_type",
    "episode_index",
    "episode_seed",
    "episode_steps",
    "done",
    *REQUIRED_METRICS,
}

LOWER_IS_BETTER = {
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_transformer_overload",
    "battery_degradation",
    "battery_degradation_calendar",
    "battery_degradation_cycling",
    "action_fraction_at_max",
}

HIGHER_IS_BETTER = {
    "episode_reward",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_ev_served",
    "total_energy_charged",
    "total_profits",
}

CONTEXT_DEPENDENT = {
    "total_energy_discharged",
    "action_mean",
    "action_std",
    "action_fraction_zero",
    "active_action_count_mean",
}

KEY_MARKDOWN_METRICS = [
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_transformer_overload",
    "action_fraction_at_max",
]


@dataclass(frozen=True)
class MatrixFile:
    scale: str
    algorithm: str
    seed: int
    path: Path


@dataclass(frozen=True)
class MetricDirection:
    preferred_direction: str
    interpretation: str
    positive_diff_interpretation: str


@dataclass(frozen=True)
class ValidationResult:
    episode_rows: List[Dict[str, str]]
    summary_rows: List[Dict[str, str]]
    csv_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate controlled multiscale eval30 CSVs.",
    )
    parser.add_argument(
        "--csv-dir",
        required=True,
        type=Path,
        help="Directory containing the 40 controlled multiscale eval30 CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where aggregation outputs will be written.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def resolve_path(path: Path) -> Path:
    expanded_path = path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path.resolve()
    return (Path.cwd() / expanded_path).resolve()


def parse_float(value: object, context: str) -> float:
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
    number = parse_float(value, context)
    if not float(number).is_integer():
        fail(f"{context}: expected integer, found {value!r}")
    return int(number)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalise_text(value: object) -> str:
    return str(value).strip().lower()


def expected_episode_seed(scale: str, seed: int, episode_index: int) -> int:
    return EVAL_SEED_BASE_BY_SCALE[scale] + seed * EPISODE_SEED_STRIDE + episode_index


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def discover_matrix_files(csv_dir: Path) -> Dict[Tuple[str, str, int], MatrixFile]:
    require(csv_dir.exists(), f"CSV directory does not exist: {csv_dir}")
    require(csv_dir.is_dir(), f"CSV path is not a directory: {csv_dir}")

    csv_paths = sorted(csv_dir.glob("*.csv"))
    require(
        len(csv_paths) == EXPECTED_CSV_COUNT,
        f"Expected exactly {EXPECTED_CSV_COUNT} CSV files, found {len(csv_paths)}",
    )

    matrix_files: Dict[Tuple[str, str, int], MatrixFile] = {}
    unexpected_names: List[str] = []
    for path in csv_paths:
        match = CSV_NAME_RE.match(path.name)
        if match is None:
            unexpected_names.append(path.name)
            continue
        scale = match.group("scale")
        algorithm = match.group("algorithm")
        seed = int(match.group("seed"))
        key = (scale, algorithm, seed)
        require(key not in matrix_files, f"Duplicate matrix CSV for {key}: {path.name}")
        matrix_files[key] = MatrixFile(scale=scale, algorithm=algorithm, seed=seed, path=path)

    require(
        not unexpected_names,
        "Unexpected CSV filename(s); expected <scale>_<algorithm>_seed<seed>_eval30.csv: "
        + ", ".join(unexpected_names),
    )

    expected_keys = {
        (scale, algorithm, seed)
        for scale in SCALES
        for algorithm in ALGORITHMS
        for seed in SEEDS
    }
    observed_keys = set(matrix_files)
    missing_names = [
        f"{scale}_{algorithm}_seed{seed}_eval30.csv"
        for scale, algorithm, seed in sorted(expected_keys.difference(observed_keys))
    ]
    extra_names = [
        f"{scale}_{algorithm}_seed{seed}_eval30.csv"
        for scale, algorithm, seed in sorted(observed_keys.difference(expected_keys))
    ]
    require(not missing_names, "Missing expected CSV file(s): " + ", ".join(missing_names))
    require(not extra_names, "Unexpected matrix CSV file(s): " + ", ".join(extra_names))

    return matrix_files


def validate_optional_identity_columns(
    row: Dict[str, str],
    fieldnames: Sequence[str],
    matrix_file: MatrixFile,
    context: str,
) -> None:
    if "scale" in fieldnames and str(row.get("scale", "")).strip():
        require(
            normalise_text(row.get("scale")) == matrix_file.scale,
            f"{context}: scale column must be {matrix_file.scale}",
        )
    if "algorithm" in fieldnames and str(row.get("algorithm", "")).strip():
        require(
            normalise_text(row.get("algorithm")) == matrix_file.algorithm,
            f"{context}: algorithm column must be {matrix_file.algorithm}",
        )
    if "seed" in fieldnames and str(row.get("seed", "")).strip():
        require(
            parse_int(row.get("seed"), f"{context}: seed") == matrix_file.seed,
            f"{context}: seed column must be {matrix_file.seed}",
        )


def validate_one_file(matrix_file: MatrixFile) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    fieldnames, rows = read_csv_rows(matrix_file.path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(fieldnames))
    require(not missing_columns, f"{matrix_file.path.name}: missing required columns {missing_columns}")
    require(
        len(rows) == EXPECTED_EPISODES_PER_FILE + 1,
        f"{matrix_file.path.name}: expected 31 total rows, found {len(rows)}",
    )

    episode_rows = [
        row for row in rows
        if normalise_text(row.get("row_type")) == "episode"
    ]
    summary_rows = [
        row for row in rows
        if normalise_text(row.get("row_type")) == "summary"
    ]
    require(
        len(episode_rows) == EXPECTED_EPISODES_PER_FILE,
        f"{matrix_file.path.name}: expected {EXPECTED_EPISODES_PER_FILE} episode rows, found {len(episode_rows)}",
    )
    require(
        len(summary_rows) == 1,
        f"{matrix_file.path.name}: expected 1 summary row, found {len(summary_rows)}",
    )

    sorted_episodes = sorted(
        episode_rows,
        key=lambda row: parse_int(row.get("episode_index"), f"{matrix_file.path.name}: episode_index"),
    )
    episode_indices = [
        parse_int(row.get("episode_index"), f"{matrix_file.path.name}: episode_index")
        for row in sorted_episodes
    ]
    require(
        episode_indices == list(range(EXPECTED_EPISODES_PER_FILE)),
        f"{matrix_file.path.name}: episode_index must be 0-29, found {episode_indices}",
    )

    episode_step_values: List[float] = []
    for row in sorted_episodes:
        episode_index = parse_int(row.get("episode_index"), f"{matrix_file.path.name}: episode_index")
        context = f"{matrix_file.path.name}: episode_index={episode_index}"
        validate_optional_identity_columns(row, fieldnames, matrix_file, context)

        observed_episode_seed = parse_int(row.get("episode_seed"), f"{context}: episode_seed")
        required_episode_seed = expected_episode_seed(
            matrix_file.scale,
            matrix_file.seed,
            episode_index,
        )
        require(
            observed_episode_seed == required_episode_seed,
            f"{context}: episode_seed must be {required_episode_seed}, found {observed_episode_seed}",
        )

        episode_steps = parse_float(row.get("episode_steps"), f"{context}: episode_steps")
        episode_step_values.append(episode_steps)
        require(is_true(row.get("done")), f"{context}: done must be true")

        for metric in REQUIRED_METRICS:
            parse_float(row.get(metric), f"{context}: {metric}")

        row["source_scale"] = matrix_file.scale
        row["source_algorithm"] = matrix_file.algorithm
        row["source_seed"] = str(matrix_file.seed)
        row["source_file"] = matrix_file.path.name

    first_episode_steps = episode_step_values[0]
    inconsistent_steps = [
        episode_steps for episode_steps in episode_step_values
        if episode_steps != first_episode_steps
    ]
    require(
        not inconsistent_steps,
        f"{matrix_file.path.name}: episode_steps must be consistent within file, found {episode_step_values}",
    )

    summary_row = summary_rows[0]
    validate_optional_identity_columns(
        summary_row,
        fieldnames,
        matrix_file,
        f"{matrix_file.path.name}: summary row",
    )
    summary_row["source_scale"] = matrix_file.scale
    summary_row["source_algorithm"] = matrix_file.algorithm
    summary_row["source_seed"] = str(matrix_file.seed)
    summary_row["source_file"] = matrix_file.path.name

    return sorted_episodes, summary_row


def validate_pairing(episode_rows: Sequence[Dict[str, str]]) -> None:
    for scale in SCALES:
        for seed in SEEDS:
            seed_sets_by_algorithm: Dict[str, set[int]] = {}
            for algorithm in ALGORITHMS:
                matching_rows = [
                    row for row in episode_rows
                    if row["source_scale"] == scale
                    and row["source_algorithm"] == algorithm
                    and int(row["source_seed"]) == seed
                ]
                seed_sets_by_algorithm[algorithm] = {
                    parse_int(row.get("episode_seed"), f"{row['source_file']}: episode_seed")
                    for row in matching_rows
                }
            require(
                seed_sets_by_algorithm["actiongnn"] == seed_sets_by_algorithm["hierarchical"],
                f"{scale} seed {seed}: actiongnn and hierarchical episode_seed sets differ",
            )


def validate_csv_dir(csv_dir: Path) -> ValidationResult:
    matrix_files = discover_matrix_files(csv_dir)
    episode_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []

    for scale in SCALES:
        for algorithm in ALGORITHMS:
            for seed in SEEDS:
                file_episode_rows, summary_row = validate_one_file(
                    matrix_files[(scale, algorithm, seed)]
                )
                episode_rows.extend(file_episode_rows)
                summary_rows.append(summary_row)

    validate_pairing(episode_rows)
    return ValidationResult(
        episode_rows=episode_rows,
        summary_rows=summary_rows,
        csv_count=len(matrix_files),
    )


def mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.12g}"


def metric_direction(metric: str) -> MetricDirection:
    if metric in HIGHER_IS_BETTER:
        return MetricDirection(
            preferred_direction="higher",
            interpretation="Higher is better.",
            positive_diff_interpretation="Positive hierarchical-minus-ActionGNN diff favours hierarchical.",
        )
    if metric in LOWER_IS_BETTER:
        return MetricDirection(
            preferred_direction="lower",
            interpretation="Lower is better.",
            positive_diff_interpretation="Positive hierarchical-minus-ActionGNN diff favours ActionGNN.",
        )
    require(metric in CONTEXT_DEPENDENT, f"Metric direction is not defined for {metric}")
    return MetricDirection(
        preferred_direction="context_dependent",
        interpretation="Context-dependent diagnostic; report descriptively.",
        positive_diff_interpretation="Positive hierarchical-minus-ActionGNN diff is descriptive, not automatically favourable.",
    )


def observed_favourable_direction(metric: str, diff_mean: float) -> str:
    direction = metric_direction(metric)
    if math.isclose(diff_mean, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "no_difference"
    if direction.preferred_direction == "higher":
        return "hierarchical" if diff_mean > 0.0 else "actiongnn"
    if direction.preferred_direction == "lower":
        return "hierarchical" if diff_mean < 0.0 else "actiongnn"
    return "context_dependent"


def build_per_seed_metric_summary(
    episode_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scale in SCALES:
        for seed in SEEDS:
            for metric in REQUIRED_METRICS:
                values_by_algorithm: Dict[str, List[float]] = {}
                for algorithm in ALGORITHMS:
                    values_by_algorithm[algorithm] = [
                        parse_float(row.get(metric), f"{row['source_file']}: {metric}")
                        for row in episode_rows
                        if row["source_scale"] == scale
                        and row["source_algorithm"] == algorithm
                        and int(row["source_seed"]) == seed
                    ]
                    require(
                        len(values_by_algorithm[algorithm]) == EXPECTED_EPISODES_PER_FILE,
                        f"{scale} {algorithm} seed {seed} metric {metric}: expected "
                        f"{EXPECTED_EPISODES_PER_FILE} episode values",
                    )

                actiongnn_mean = mean(values_by_algorithm["actiongnn"])
                hierarchical_mean = mean(values_by_algorithm["hierarchical"])
                rows.append(
                    {
                        "scale": scale,
                        "seed": seed,
                        "metric": metric,
                        "actiongnn_episode_count": len(values_by_algorithm["actiongnn"]),
                        "hierarchical_episode_count": len(values_by_algorithm["hierarchical"]),
                        "actiongnn_mean": actiongnn_mean,
                        "hierarchical_mean": hierarchical_mean,
                        "paired_diff_hierarchical_minus_actiongnn": hierarchical_mean - actiongnn_mean,
                    }
                )
    return rows


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
    require(0.0 <= x <= 1.0, f"regularized beta x out of range: {x}")
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
    require(degrees_of_freedom > 0, "degrees_of_freedom must be positive")
    if math.isinf(t_value):
        return 1.0 if t_value > 0 else 0.0
    x_value = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    beta_value = regularized_incomplete_beta(x_value, degrees_of_freedom / 2.0, 0.5)
    if t_value >= 0.0:
        return 1.0 - 0.5 * beta_value
    return 0.5 * beta_value


def student_t_ppf(probability: float, degrees_of_freedom: int) -> float:
    require(0.0 < probability < 1.0, "probability must be between 0 and 1")
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


def paired_t_test(diff_values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    sample_count = len(diff_values)
    require(sample_count >= 2, "paired t-test requires at least two paired seeds")
    diff_mean = mean(diff_values)
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


def average_ranks_for_absolute_values(values: Sequence[float]) -> List[float]:
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


def wilcoxon_signed_rank(diff_values: Sequence[float]) -> Tuple[float, float, str, int]:
    nonzero_diffs = [
        value for value in diff_values
        if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)
    ]
    if not nonzero_diffs:
        return 0.0, 1.0, "exact", 0
    require(
        len(nonzero_diffs) <= 20,
        f"exact Wilcoxon signed-rank test is not feasible for n={len(nonzero_diffs)}",
    )

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


def rows_for_scale_metric(
    per_seed_rows: Sequence[Dict[str, object]],
    scale: str,
    metric: str,
) -> List[Dict[str, object]]:
    return [
        row for row in per_seed_rows
        if row["scale"] == scale and row["metric"] == metric
    ]


def build_paired_metric_tests(
    per_seed_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    test_rows: List[Dict[str, object]] = []
    for scale in SCALES:
        for metric in REQUIRED_METRICS:
            seed_rows = sorted(rows_for_scale_metric(per_seed_rows, scale, metric), key=lambda row: int(row["seed"]))
            require(
                len(seed_rows) == len(SEEDS),
                f"{scale} {metric}: expected {len(SEEDS)} paired seed rows, found {len(seed_rows)}",
            )
            actiongnn_values = [float(row["actiongnn_mean"]) for row in seed_rows]
            hierarchical_values = [float(row["hierarchical_mean"]) for row in seed_rows]
            diff_values = [
                float(row["paired_diff_hierarchical_minus_actiongnn"])
                for row in seed_rows
            ]

            t_statistic, t_pvalue, ci_low, ci_high, diff_std = paired_t_test(diff_values)
            wilcoxon_statistic, wilcoxon_pvalue, wilcoxon_method, wilcoxon_nonzero_n = (
                wilcoxon_signed_rank(diff_values)
            )
            direction = metric_direction(metric)
            diff_mean = mean(diff_values)

            test_rows.append(
                {
                    "scale": scale,
                    "metric": metric,
                    "preferred_direction": direction.preferred_direction,
                    "interpretation": direction.interpretation,
                    "actiongnn_seed_mean": mean(actiongnn_values),
                    "hierarchical_seed_mean": mean(hierarchical_values),
                    "paired_diff_mean_hierarchical_minus_actiongnn": diff_mean,
                    "paired_diff_std": diff_std,
                    "paired_t_statistic": t_statistic,
                    "paired_t_pvalue_two_sided": t_pvalue,
                    "paired_95ci_low": ci_low,
                    "paired_95ci_high": ci_high,
                    "wilcoxon_statistic": wilcoxon_statistic,
                    "wilcoxon_pvalue_two_sided": wilcoxon_pvalue,
                    "wilcoxon_method": wilcoxon_method,
                    "wilcoxon_nonzero_n": wilcoxon_nonzero_n,
                    "observed_favourable_direction": observed_favourable_direction(metric, diff_mean),
                    "n_paired_seeds": len(seed_rows),
                }
            )
    return test_rows


def build_metric_direction_summary(
    test_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for test_row in test_rows:
        metric = str(test_row["metric"])
        direction = metric_direction(metric)
        rows.append(
            {
                "scale": test_row["scale"],
                "metric": metric,
                "preferred_direction": direction.preferred_direction,
                "interpretation": direction.interpretation,
                "positive_diff_interpretation": direction.positive_diff_interpretation,
                "observed_diff_mean_hierarchical_minus_actiongnn": test_row[
                    "paired_diff_mean_hierarchical_minus_actiongnn"
                ],
                "observed_favourable_direction": test_row["observed_favourable_direction"],
                "paired_t_pvalue_two_sided": test_row["paired_t_pvalue_two_sided"],
                "wilcoxon_pvalue_two_sided": test_row["wilcoxon_pvalue_two_sided"],
            }
        )
    return rows


def row_by_scale_metric(
    test_rows: Sequence[Dict[str, object]],
    scale: str,
    metric: str,
) -> Dict[str, object]:
    for row in test_rows:
        if row["scale"] == scale and row["metric"] == metric:
            return row
    fail(f"Missing paired test row for {scale} {metric}")


def build_scale_level_summary(
    validation: ValidationResult,
    test_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scale in SCALES:
        scale_test_rows = [row for row in test_rows if row["scale"] == scale]
        lower_rows = [
            row for row in scale_test_rows
            if row["preferred_direction"] == "lower"
        ]
        higher_rows = [
            row for row in scale_test_rows
            if row["preferred_direction"] == "higher"
        ]
        context_rows = [
            row for row in scale_test_rows
            if row["preferred_direction"] == "context_dependent"
        ]

        summary_row: Dict[str, object] = {
            "scale": scale,
            "csv_files": len(ALGORITHMS) * len(SEEDS),
            "episode_rows": sum(1 for row in validation.episode_rows if row["source_scale"] == scale),
            "summary_rows": sum(1 for row in validation.summary_rows if row["source_scale"] == scale),
            "paired_seeds": len(SEEDS),
            "metrics": len(REQUIRED_METRICS),
            "higher_is_better_metrics_favouring_hierarchical": sum(
                1 for row in higher_rows if row["observed_favourable_direction"] == "hierarchical"
            ),
            "higher_is_better_metrics_favouring_actiongnn": sum(
                1 for row in higher_rows if row["observed_favourable_direction"] == "actiongnn"
            ),
            "lower_is_better_metrics_favouring_hierarchical": sum(
                1 for row in lower_rows if row["observed_favourable_direction"] == "hierarchical"
            ),
            "lower_is_better_metrics_favouring_actiongnn": sum(
                1 for row in lower_rows if row["observed_favourable_direction"] == "actiongnn"
            ),
            "context_dependent_metrics": len(context_rows),
            "paired_t_p_lt_0_05_count": sum(
                1 for row in scale_test_rows if float(row["paired_t_pvalue_two_sided"]) < 0.05
            ),
            "wilcoxon_p_lt_0_05_count": sum(
                1 for row in scale_test_rows if float(row["wilcoxon_pvalue_two_sided"]) < 0.05
            ),
        }

        for metric in KEY_MARKDOWN_METRICS:
            metric_row = row_by_scale_metric(test_rows, scale, metric)
            summary_row[f"{metric}_actiongnn_seed_mean"] = metric_row["actiongnn_seed_mean"]
            summary_row[f"{metric}_hierarchical_seed_mean"] = metric_row["hierarchical_seed_mean"]
            summary_row[f"{metric}_diff_hierarchical_minus_actiongnn"] = metric_row[
                "paired_diff_mean_hierarchical_minus_actiongnn"
            ]
        rows.append(summary_row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted_row = {
                key: format_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(formatted_row)


def p_text(value: object) -> str:
    p_value = float(value)
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.6f}"


def write_markdown_summary(
    path: Path,
    validation: ValidationResult,
    test_rows: Sequence[Dict[str, object]],
    output_paths: Sequence[Path],
) -> None:
    lines = [
        "# Controlled Multiscale Formal Aggregation Summary",
        "",
        "## Validation",
        "",
        "- `VALIDATION_OK`",
        f"- CSV files: {validation.csv_count}",
        f"- episode rows: {len(validation.episode_rows)}",
        f"- summary rows: {len(validation.summary_rows)}",
        f"- scales: {', '.join(SCALES)}",
        f"- algorithms: {', '.join(ALGORITHMS)}",
        "- training seeds: 0, 1, 2, 3, 4",
        "- evaluation episodes per CSV: 30",
        "- primary inference unit: training seed",
        "- episode rows are averaged before p-values are computed",
        "",
        "## Eval Seed Protocol",
        "",
        "| scale | base | seed formula |",
        "|---|---:|---|",
    ]
    for scale in SCALES:
        lines.append(
            f"| `{scale}` | {EVAL_SEED_BASE_BY_SCALE[scale]} | "
            f"`base + training_seed * {EPISODE_SEED_STRIDE} + episode_index` |"
        )

    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| scale | metric | direction | ActionGNN mean | hierarchical mean | hierarchical - ActionGNN | paired t p | Wilcoxon p |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scale in SCALES:
        for metric in KEY_MARKDOWN_METRICS:
            row = row_by_scale_metric(test_rows, scale, metric)
            lines.append(
                "| "
                f"`{scale}` | `{metric}` | {row['preferred_direction']} | "
                f"{float(row['actiongnn_seed_mean']):.6f} | "
                f"{float(row['hierarchical_seed_mean']):.6f} | "
                f"{float(row['paired_diff_mean_hierarchical_minus_actiongnn']):.6f} | "
                f"{p_text(row['paired_t_pvalue_two_sided'])} | "
                f"{p_text(row['wilcoxon_pvalue_two_sided'])} |"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for output_path in output_paths:
        lines.append(f"- `{output_path}`")

    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- Paired tests use five training-seed means per scale.",
            "- Episodes are deterministic evaluation replicates, not independent samples for p-values.",
            "- Direction summaries are descriptive and do not by themselves establish thesis claims.",
            "- Do not use this aggregation to claim external SOTA comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    out_dir: Path,
    validation: ValidationResult,
    per_seed_rows: Sequence[Dict[str, object]],
    test_rows: Sequence[Dict[str, object]],
) -> List[Path]:
    direction_rows = build_metric_direction_summary(test_rows)
    scale_rows = build_scale_level_summary(validation, test_rows)

    per_seed_path = out_dir / "multiscale_per_seed_metric_summary.csv"
    paired_tests_path = out_dir / "multiscale_paired_metric_tests.csv"
    direction_path = out_dir / "multiscale_metric_direction_summary.csv"
    scale_summary_path = out_dir / "multiscale_scale_level_summary.csv"
    markdown_path = out_dir / "controlled_multiscale_formal_summary.md"

    write_csv(
        per_seed_path,
        per_seed_rows,
        [
            "scale",
            "seed",
            "metric",
            "actiongnn_episode_count",
            "hierarchical_episode_count",
            "actiongnn_mean",
            "hierarchical_mean",
            "paired_diff_hierarchical_minus_actiongnn",
        ],
    )
    write_csv(
        paired_tests_path,
        test_rows,
        [
            "scale",
            "metric",
            "preferred_direction",
            "interpretation",
            "actiongnn_seed_mean",
            "hierarchical_seed_mean",
            "paired_diff_mean_hierarchical_minus_actiongnn",
            "paired_diff_std",
            "paired_t_statistic",
            "paired_t_pvalue_two_sided",
            "paired_95ci_low",
            "paired_95ci_high",
            "wilcoxon_statistic",
            "wilcoxon_pvalue_two_sided",
            "wilcoxon_method",
            "wilcoxon_nonzero_n",
            "observed_favourable_direction",
            "n_paired_seeds",
        ],
    )
    write_csv(
        direction_path,
        direction_rows,
        [
            "scale",
            "metric",
            "preferred_direction",
            "interpretation",
            "positive_diff_interpretation",
            "observed_diff_mean_hierarchical_minus_actiongnn",
            "observed_favourable_direction",
            "paired_t_pvalue_two_sided",
            "wilcoxon_pvalue_two_sided",
        ],
    )

    scale_fieldnames = [
        "scale",
        "csv_files",
        "episode_rows",
        "summary_rows",
        "paired_seeds",
        "metrics",
        "higher_is_better_metrics_favouring_hierarchical",
        "higher_is_better_metrics_favouring_actiongnn",
        "lower_is_better_metrics_favouring_hierarchical",
        "lower_is_better_metrics_favouring_actiongnn",
        "context_dependent_metrics",
        "paired_t_p_lt_0_05_count",
        "wilcoxon_p_lt_0_05_count",
    ]
    for metric in KEY_MARKDOWN_METRICS:
        scale_fieldnames.extend(
            [
                f"{metric}_actiongnn_seed_mean",
                f"{metric}_hierarchical_seed_mean",
                f"{metric}_diff_hierarchical_minus_actiongnn",
            ]
        )
    write_csv(scale_summary_path, scale_rows, scale_fieldnames)

    output_paths = [
        per_seed_path,
        paired_tests_path,
        direction_path,
        scale_summary_path,
        markdown_path,
    ]
    write_markdown_summary(markdown_path, validation, test_rows, output_paths)
    return output_paths


def main() -> int:
    args = parse_args()
    csv_dir = resolve_path(args.csv_dir)
    out_dir = resolve_path(args.out_dir)
    require(
        out_dir != csv_dir,
        "--out-dir must differ from --csv-dir so aggregation outputs do not pollute the exact 40-CSV input set",
    )

    print("CONTROLLED_MULTISCALE_AGGREGATION_START")
    if csv_dir.exists() and csv_dir.is_dir():
        print(f"CSV_COUNT={len(list(csv_dir.glob('*.csv')))}")

    validation = validate_csv_dir(csv_dir)
    print("VALIDATION_OK")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows = build_per_seed_metric_summary(validation.episode_rows)
    test_rows = build_paired_metric_tests(per_seed_rows)
    output_paths = write_outputs(out_dir, validation, per_seed_rows, test_rows)
    print("AGGREGATION_OK")
    print("output_files:")
    for output_path in output_paths:
        print(f"  {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
