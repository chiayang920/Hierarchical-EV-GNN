#!/usr/bin/env python3
"""Aggregate 500CP formal eval30 canonical CSVs across multiple metrics.

The script intentionally uses only the standard library and numpy so it can run
on lightweight analysis nodes without scipy or pandas.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ALGORITHMS = ("actiongnn", "hierarchical")
SEEDS = tuple(range(5))
EXPECTED_EPISODES_PER_FILE = 30
EXPECTED_EPISODE_STEPS = 112
EPISODE_SEED_BASE = 520000
EPISODE_SEED_STRIDE = 1000

KEY_METRICS = [
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
    "action_min",
    "action_max",
    "action_fraction_zero",
    "action_fraction_at_max",
    "active_action_count_mean",
]

REQUIRED_COLUMNS = {
    "row_type",
    "algorithm",
    "seed",
    "episode_seed",
    "episode_index",
    "episode_reward",
    "episode_steps",
    "done",
    *KEY_METRICS,
}

NON_METRIC_COLUMNS = {
    "run_name",
    "algorithm",
    "config",
    "seed",
    "episode_seed",
    "checkpoint",
    "episode_index",
    "episode_steps",
    "done",
    "row_type",
    "source_file",
    "source_seed",
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
    "mean_reward",
    "total_reward",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_energy_charged",
    "total_ev_served",
    "total_profits",
}

KEY_PRINT_METRICS = [
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_transformer_overload",
    "action_fraction_at_max",
]


@dataclass(frozen=True)
class MetricDirection:
    preferred_direction: str
    interpretation: str
    positive_diff_interpretation: str


@dataclass(frozen=True)
class ValidationResult:
    episode_rows: List[Dict[str, str]]
    summary_rows: List[Dict[str, str]]
    metric_names: List[str]
    configs: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate 500CP formal eval30 canonical CSV metrics.",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        type=Path,
        help="Directory containing the 10 canonical actiongnn/hierarchical 500CP eval30 CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where aggregation outputs will be written.",
    )
    return parser.parse_args()


def expected_file_names() -> List[str]:
    return [
        f"{algorithm}_500cp_seed{seed}_eval30.csv"
        for algorithm in ALGORITHMS
        for seed in SEEDS
    ]


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


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


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return fieldnames, rows


def validate_expected_files(eval_dir: Path) -> Dict[Tuple[str, int], Path]:
    csv_files = sorted(path.name for path in eval_dir.glob("*.csv"))
    expected_names = sorted(expected_file_names())
    require(
        csv_files == expected_names,
        "Expected exactly these canonical CSVs:\n"
        f"  expected={expected_names}\n"
        f"  found={csv_files}",
    )

    files: Dict[Tuple[str, int], Path] = {}
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            files[(algorithm, seed)] = eval_dir / f"{algorithm}_500cp_seed{seed}_eval30.csv"
    return files


def expected_episode_seeds(seed: int) -> List[int]:
    start = EPISODE_SEED_BASE + seed * EPISODE_SEED_STRIDE
    return list(range(start, start + EXPECTED_EPISODES_PER_FILE))


def validate_one_file(
    path: Path,
    algorithm: str,
    seed: int,
) -> Tuple[List[str], List[Dict[str, str]], Dict[str, str]]:
    fieldnames, rows = read_csv_rows(path)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(fieldnames))
    require(not missing_columns, f"{path.name}: missing required columns {missing_columns}")
    require(
        len(rows) == EXPECTED_EPISODES_PER_FILE + 1,
        f"{path.name}: expected 31 total rows, found {len(rows)}",
    )

    episode_rows = [row for row in rows if row.get("row_type") == "episode"]
    summary_rows = [row for row in rows if row.get("row_type") == "summary"]
    require(
        len(episode_rows) == EXPECTED_EPISODES_PER_FILE,
        f"{path.name}: expected {EXPECTED_EPISODES_PER_FILE} episode rows, found {len(episode_rows)}",
    )
    require(len(summary_rows) == 1, f"{path.name}: expected 1 summary row, found {len(summary_rows)}")

    sorted_episodes = sorted(
        episode_rows,
        key=lambda row: parse_int(row["episode_index"], f"{path.name}: episode_index"),
    )
    episode_indices = [
        parse_int(row["episode_index"], f"{path.name}: episode_index")
        for row in sorted_episodes
    ]
    require(
        episode_indices == list(range(EXPECTED_EPISODES_PER_FILE)),
        f"{path.name}: episode_index must be 0-29, found {episode_indices}",
    )

    episode_seed_values = [
        parse_int(row["episode_seed"], f"{path.name}: episode_seed")
        for row in sorted_episodes
    ]
    require(
        episode_seed_values == expected_episode_seeds(seed),
        f"{path.name}: episode seeds must be {expected_episode_seeds(seed)}, found {episode_seed_values}",
    )

    for row in sorted_episodes:
        row_context = f"{path.name}: episode_index={row['episode_index']}"
        require(row.get("row_type") == "episode", f"{row_context}: row_type must be episode")
        require(row.get("algorithm") == algorithm, f"{row_context}: algorithm must be {algorithm}")
        require(parse_int(row.get("seed"), f"{row_context}: seed") == seed, f"{row_context}: seed mismatch")
        require(
            parse_int(row.get("episode_steps"), f"{row_context}: episode_steps") == EXPECTED_EPISODE_STEPS,
            f"{row_context}: episode_steps must be {EXPECTED_EPISODE_STEPS}",
        )
        require(is_true(row.get("done")), f"{row_context}: done must be true")

    summary_row = summary_rows[0]
    require(summary_row.get("row_type") == "summary", f"{path.name}: summary row_type must be summary")
    require(summary_row.get("algorithm") == algorithm, f"{path.name}: summary algorithm must be {algorithm}")
    require(parse_int(summary_row.get("seed"), f"{path.name}: summary seed") == seed, f"{path.name}: summary seed mismatch")

    for row in sorted_episodes:
        row["source_file"] = path.name
        row["source_seed"] = str(seed)
    summary_row["source_file"] = path.name
    summary_row["source_seed"] = str(seed)

    return fieldnames, sorted_episodes, summary_row


def looks_finite_for_all_episode_rows(rows: Sequence[Dict[str, str]], column: str) -> bool:
    for row in rows:
        value = row.get(column)
        if value is None or str(value).strip() == "":
            return False
        try:
            number = float(str(value).strip())
        except ValueError:
            return False
        if not math.isfinite(number):
            return False
    return True


def detect_metric_columns(
    all_fieldnames: Iterable[str],
    episode_rows: Sequence[Dict[str, str]],
) -> List[str]:
    fieldname_order = list(dict.fromkeys(all_fieldnames))
    metric_names = [
        column
        for column in fieldname_order
        if column not in NON_METRIC_COLUMNS and looks_finite_for_all_episode_rows(episode_rows, column)
    ]
    for metric in KEY_METRICS:
        if metric not in metric_names:
            metric_names.append(metric)

    for metric in metric_names:
        for row in episode_rows:
            parse_float(
                row.get(metric),
                f"{row.get('source_file')}: episode_index={row.get('episode_index')}: {metric}",
            )

    return metric_names


def validate_eval_dir(eval_dir: Path) -> ValidationResult:
    require(eval_dir.exists(), f"Eval directory does not exist: {eval_dir}")
    require(eval_dir.is_dir(), f"Eval path is not a directory: {eval_dir}")

    files = validate_expected_files(eval_dir)
    all_fieldnames: List[str] = []
    episode_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []

    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            path = files[(algorithm, seed)]
            fieldnames, file_episode_rows, summary_row = validate_one_file(path, algorithm, seed)
            all_fieldnames.extend(fieldnames)
            episode_rows.extend(file_episode_rows)
            summary_rows.append(summary_row)

    for seed in SEEDS:
        by_algorithm: Dict[str, set[int]] = {}
        for algorithm in ALGORITHMS:
            seed_rows = [
                row for row in episode_rows
                if row["algorithm"] == algorithm and parse_int(row["seed"], f"{row['source_file']}: seed") == seed
            ]
            by_algorithm[algorithm] = {
                parse_int(row["episode_seed"], f"{row['source_file']}: episode_seed")
                for row in seed_rows
            }
        require(
            by_algorithm["actiongnn"] == by_algorithm["hierarchical"],
            f"ActionGNN and hierarchical episode_seed sets differ for seed {seed}",
        )

    metric_names = detect_metric_columns(all_fieldnames, episode_rows)
    configs = sorted({row.get("config", "") for row in episode_rows if row.get("config")})

    return ValidationResult(
        episode_rows=episode_rows,
        summary_rows=summary_rows,
        metric_names=metric_names,
        configs=configs,
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
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def metric_direction(metric: str) -> MetricDirection:
    if metric in HIGHER_IS_BETTER:
        return MetricDirection(
            preferred_direction="higher",
            interpretation="Higher is generally better.",
            positive_diff_interpretation="Positive hierarchical-minus-ActionGNN diff favours hierarchical.",
        )
    if metric in LOWER_IS_BETTER:
        return MetricDirection(
            preferred_direction="lower",
            interpretation="Lower is generally better.",
            positive_diff_interpretation="Positive hierarchical-minus-ActionGNN diff favours ActionGNN.",
        )
    return MetricDirection(
        preferred_direction="ambiguous",
        interpretation="Direction is context-dependent; report without overclaiming.",
        positive_diff_interpretation="Positive hierarchical-minus-ActionGNN diff is descriptive, not automatically favourable.",
    )


def build_per_seed_metric_summary(
    episode_rows: Sequence[Dict[str, str]],
    metric_names: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric in metric_names:
        for seed in SEEDS:
            values_by_algorithm: Dict[str, List[float]] = {}
            for algorithm in ALGORITHMS:
                values_by_algorithm[algorithm] = [
                    parse_float(row[metric], f"{row['source_file']}: {metric}")
                    for row in episode_rows
                    if row["algorithm"] == algorithm and parse_int(row["seed"], f"{row['source_file']}: seed") == seed
                ]
                require(
                    len(values_by_algorithm[algorithm]) == EXPECTED_EPISODES_PER_FILE,
                    f"{algorithm} seed {seed} metric {metric}: expected {EXPECTED_EPISODES_PER_FILE} values",
                )

            actiongnn_mean = mean(values_by_algorithm["actiongnn"])
            hierarchical_mean = mean(values_by_algorithm["hierarchical"])
            rows.append(
                {
                    "metric": metric,
                    "seed": seed,
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
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h

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
    x = degrees_of_freedom / (degrees_of_freedom + t_value * t_value)
    beta_value = regularized_incomplete_beta(x, degrees_of_freedom / 2.0, 0.5)
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
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if student_t_cdf(midpoint, degrees_of_freedom) < probability:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def paired_t_test(diff_values: Sequence[float]) -> Tuple[float, float, float, float, float]:
    n = len(diff_values)
    require(n >= 2, "paired t-test requires at least two paired seeds")
    diff_mean = mean(diff_values)
    diff_std = sample_std(diff_values)
    if diff_std == 0.0:
        t_statistic = 0.0 if diff_mean == 0.0 else math.copysign(float("inf"), diff_mean)
        p_value = 1.0 if diff_mean == 0.0 else 0.0
        return t_statistic, p_value, diff_mean, diff_mean, diff_std

    standard_error = diff_std / math.sqrt(n)
    t_statistic = diff_mean / standard_error
    p_value = 2.0 * (1.0 - student_t_cdf(abs(t_statistic), n - 1))
    t_critical = student_t_ppf(0.975, n - 1)
    ci_low = diff_mean - t_critical * standard_error
    ci_high = diff_mean + t_critical * standard_error
    return t_statistic, p_value, ci_low, ci_high, diff_std


def average_ranks_for_absolute_values(values: Sequence[float]) -> List[float]:
    sorted_pairs = sorted(enumerate(abs(value) for value in values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and math.isclose(sorted_pairs[end][1], sorted_pairs[index][1], rel_tol=0.0, abs_tol=1e-12):
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for pair_index in range(index, end):
            original_index = sorted_pairs[pair_index][0]
            ranks[original_index] = average_rank
        index = end
    return ranks


def normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def wilcoxon_signed_rank(diff_values: Sequence[float]) -> Tuple[float, float, str, int]:
    nonzero_diffs = [value for value in diff_values if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-12)]
    if not nonzero_diffs:
        return 0.0, 1.0, "exact", 0

    ranks = average_ranks_for_absolute_values(nonzero_diffs)
    total_rank = sum(ranks)
    w_plus = sum(rank for rank, value in zip(ranks, nonzero_diffs) if value > 0.0)
    w_minus = total_rank - w_plus
    statistic = min(w_plus, w_minus)

    if len(nonzero_diffs) <= 20:
        possible_sums = [0.0]
        for rank in ranks:
            possible_sums += [current_sum + rank for current_sum in possible_sums]
        extreme_count = sum(
            1
            for rank_sum in possible_sums
            if min(rank_sum, total_rank - rank_sum) <= statistic + 1e-12
        )
        return statistic, min(1.0, extreme_count / len(possible_sums)), "exact", len(nonzero_diffs)

    mean_w = total_rank / 2.0
    variance_w = sum(rank * rank for rank in ranks) / 4.0
    z_value = (statistic - mean_w + 0.5) / math.sqrt(variance_w)
    return statistic, normal_two_sided_p(z_value), "normal_approximation", len(nonzero_diffs)


def build_paired_metric_tests(
    per_seed_rows: Sequence[Dict[str, object]],
    metric_names: Sequence[str],
) -> List[Dict[str, object]]:
    per_seed_by_metric: Dict[str, List[Dict[str, object]]] = {metric: [] for metric in metric_names}
    for row in per_seed_rows:
        per_seed_by_metric[str(row["metric"])].append(row)

    test_rows: List[Dict[str, object]] = []
    for metric in metric_names:
        seed_rows = sorted(per_seed_by_metric[metric], key=lambda row: int(row["seed"]))
        actiongnn_values = [float(row["actiongnn_mean"]) for row in seed_rows]
        hierarchical_values = [float(row["hierarchical_mean"]) for row in seed_rows]
        diff_values = [float(row["paired_diff_hierarchical_minus_actiongnn"]) for row in seed_rows]

        t_statistic, t_pvalue, ci_low, ci_high, diff_std = paired_t_test(diff_values)
        wilcoxon_statistic, wilcoxon_pvalue, wilcoxon_method, wilcoxon_nonzero_n = wilcoxon_signed_rank(diff_values)
        direction = metric_direction(metric)
        diff_mean = mean(diff_values)

        if direction.preferred_direction == "higher":
            observed_favourable_direction = "hierarchical" if diff_mean > 0 else "actiongnn" if diff_mean < 0 else "no_difference"
        elif direction.preferred_direction == "lower":
            observed_favourable_direction = "hierarchical" if diff_mean < 0 else "actiongnn" if diff_mean > 0 else "no_difference"
        else:
            observed_favourable_direction = "ambiguous"

        test_rows.append(
            {
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
                "observed_favourable_direction": observed_favourable_direction,
                "n_paired_seeds": len(seed_rows),
            }
        )
    return test_rows


def build_metric_direction_rows(test_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for test_row in test_rows:
        metric = str(test_row["metric"])
        direction = metric_direction(metric)
        rows.append(
            {
                "metric": metric,
                "preferred_direction": direction.preferred_direction,
                "interpretation": direction.interpretation,
                "positive_diff_interpretation": direction.positive_diff_interpretation,
                "observed_diff_mean_hierarchical_minus_actiongnn": test_row["paired_diff_mean_hierarchical_minus_actiongnn"],
                "observed_favourable_direction": test_row["observed_favourable_direction"],
            }
        )
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


def row_by_metric(rows: Sequence[Dict[str, object]], metric: str) -> Dict[str, object]:
    for row in rows:
        if row["metric"] == metric:
            return row
    fail(f"Metric not found: {metric}")


def p_text(value: object) -> str:
    p_value = float(value)
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.6f}"


def write_markdown_summary(
    path: Path,
    validation: ValidationResult,
    test_rows: Sequence[Dict[str, object]],
) -> None:
    key_metrics = [metric for metric in KEY_PRINT_METRICS if any(row["metric"] == metric for row in test_rows)]
    lines = [
        "# 500CP Multimetric Aggregation Summary",
        "",
        "## Validation",
        "",
        "- `CANONICAL_CSV_VALIDATION_OK`",
        f"- canonical CSV files: {len(expected_file_names())}",
        f"- episode rows: {len(validation.episode_rows)}",
        f"- summary rows: {len(validation.summary_rows)}",
        f"- configs: {', '.join(validation.configs) if validation.configs else 'not recorded'}",
        "- paired episode seeds: validated for seeds 0-4",
        "",
        "## Key Metrics",
        "",
        "| metric | ActionGNN mean | hierarchical mean | hierarchical - ActionGNN | paired t p | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for metric in key_metrics:
        row = row_by_metric(test_rows, metric)
        lines.append(
            "| "
            f"`{metric}` | "
            f"{float(row['actiongnn_seed_mean']):.6f} | "
            f"{float(row['hierarchical_seed_mean']):.6f} | "
            f"{float(row['paired_diff_mean_hierarchical_minus_actiongnn']):.6f} | "
            f"{p_text(row['paired_t_pvalue_two_sided'])} | "
            f"{p_text(row['wilcoxon_pvalue_two_sided'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Positive reward difference means hierarchical has higher / less negative reward.",
            "- For tracking, violation, overload, degradation, and maximum-action saturation metrics, lower is generally better.",
            "- For user satisfaction, total energy charged, total EV served, and profit, higher is generally better.",
            "- Ambiguous diagnostics are labelled context-dependent in `metric_direction_summary.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_key_metric_lines(test_rows: Sequence[Dict[str, object]]) -> None:
    for metric in KEY_PRINT_METRICS:
        if not any(row["metric"] == metric for row in test_rows):
            continue
        row = row_by_metric(test_rows, metric)
        print(
            f"{metric}: "
            f"actiongnn_mean={format_float(float(row['actiongnn_seed_mean']))} "
            f"hierarchical_mean={format_float(float(row['hierarchical_seed_mean']))} "
            f"diff={format_float(float(row['paired_diff_mean_hierarchical_minus_actiongnn']))} "
            f"paired_t_p={format_float(float(row['paired_t_pvalue_two_sided']))} "
            f"wilcoxon_p={format_float(float(row['wilcoxon_pvalue_two_sided']))}"
        )


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_eval_dir(eval_dir)
    per_seed_rows = build_per_seed_metric_summary(validation.episode_rows, validation.metric_names)
    test_rows = build_paired_metric_tests(per_seed_rows, validation.metric_names)
    direction_rows = build_metric_direction_rows(test_rows)

    per_seed_path = out_dir / "per_seed_metric_summary.csv"
    paired_tests_path = out_dir / "paired_metric_tests.csv"
    direction_path = out_dir / "metric_direction_summary.csv"
    markdown_path = out_dir / "500cp_multimetric_aggregation_summary.md"

    write_csv(
        per_seed_path,
        per_seed_rows,
        [
            "metric",
            "seed",
            "actiongnn_mean",
            "hierarchical_mean",
            "paired_diff_hierarchical_minus_actiongnn",
        ],
    )
    write_csv(
        paired_tests_path,
        test_rows,
        [
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
            "metric",
            "preferred_direction",
            "interpretation",
            "positive_diff_interpretation",
            "observed_diff_mean_hierarchical_minus_actiongnn",
            "observed_favourable_direction",
        ],
    )
    write_markdown_summary(markdown_path, validation, test_rows)

    print("CANONICAL_CSV_VALIDATION_OK")
    print(f"episode_rows={len(validation.episode_rows)}")
    print(f"summary_rows={len(validation.summary_rows)}")
    print_key_metric_lines(test_rows)
    print("output_files:")
    for output_path in [per_seed_path, paired_tests_path, direction_path, markdown_path]:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
