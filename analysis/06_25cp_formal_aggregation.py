"""Historical 25CP formal aggregation script.

This file intentionally preserves legacy 25CP result filenames and labels so
completed 25CP evaluations remain reproducible.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List, Optional, Tuple


INPUT_DIR = Path("results/phase2c_formal_25cp_eval")
OUTPUT_DIR = Path("results/phase2c_formal_25cp_analysis")

INPUT_FILES = [
    INPUT_DIR / "baseline_25cp_seed0_eval30.csv",
    INPUT_DIR / "baseline_25cp_seed1_eval30.csv",
    INPUT_DIR / "baseline_25cp_seed2_eval30.csv",
    INPUT_DIR / "baseline_25cp_seed3_eval30.csv",
    INPUT_DIR / "baseline_25cp_seed4_eval30.csv",
    INPUT_DIR / "hierarchical_25cp_seed0_eval30.csv",
    INPUT_DIR / "hierarchical_25cp_seed1_eval30.csv",
    INPUT_DIR / "hierarchical_25cp_seed2_eval30.csv",
    INPUT_DIR / "hierarchical_25cp_seed3_eval30.csv",
    INPUT_DIR / "hierarchical_25cp_seed4_eval30.csv",
]

REQUIRED_COLUMNS = {
    "run_name",
    "algorithm",
    "seed",
    "episode_seed",
    "episode_index",
    "episode_reward",
    "episode_steps",
    "mean_reward",
    "std_reward",
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "action_fraction_zero",
    "action_fraction_at_max",
    "active_action_count_mean",
    "row_type",
}

PRIMARY_METRICS = [
    "mean_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "action_fraction_at_max",
    "action_fraction_zero",
    "active_action_count_mean",
    "total_ev_served",
    "total_transformer_overload",
]

SUPPORTING_METRICS = [
    "episode_reward",
    "episode_steps",
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "average_user_satisfaction",
    "battery_degradation",
    "battery_degradation_calendar",
    "battery_degradation_cycling",
    "energy_user_satisfaction",
    "total_energy_charged",
    "total_energy_discharged",
    "total_profits",
    "total_reward",
]

ALL_METRICS = list(dict.fromkeys(PRIMARY_METRICS + SUPPORTING_METRICS))


def to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text_value = str(value).strip()
    if text_value == "":
        return None
    try:
        parsed_value = float(text_value)
    except ValueError:
        return None
    if math.isnan(parsed_value) or math.isinf(parsed_value):
        return None
    return parsed_value


def to_int(value: object) -> int:
    parsed_value = to_float(value)
    if parsed_value is None:
        raise ValueError(f"Cannot parse integer from value: {value!r}")
    return int(parsed_value)


def safe_mean(values: Iterable[float]) -> str:
    numeric_values = list(values)
    if not numeric_values:
        return ""
    return f"{mean(numeric_values):.12g}"


def safe_std_population(values: Iterable[float]) -> str:
    numeric_values = list(values)
    if not numeric_values:
        return ""
    population_mean = mean(numeric_values)
    variance = mean([(value - population_mean) ** 2 for value in numeric_values])
    return f"{math.sqrt(variance):.12g}"


def safe_std_sample(values: Iterable[float]) -> str:
    numeric_values = list(values)
    if len(numeric_values) < 2:
        return ""
    return f"{stdev(numeric_values):.12g}"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    missing_columns = REQUIRED_COLUMNS.difference(fieldnames)
    if missing_columns:
        raise ValueError(f"{path} missing required columns: {sorted(missing_columns)}")

    for row in rows:
        row["source_file"] = str(path)

    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: List[Dict[str, object]]) -> List[str]:
    ordered_fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in ordered_fieldnames:
                ordered_fieldnames.append(key)
    return ordered_fieldnames


def numeric_column_values(rows: List[Dict[str, str]], column_name: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        parsed_value = to_float(row.get(column_name))
        if parsed_value is not None:
            values.append(parsed_value)
    return values


def group_episode_rows_by_algorithm_seed(
    episode_rows: List[Dict[str, str]],
) -> Dict[Tuple[str, int], List[Dict[str, str]]]:
    grouped_rows: Dict[Tuple[str, int], List[Dict[str, str]]] = {}

    for row in episode_rows:
        algorithm = str(row["algorithm"])
        seed = to_int(row["seed"])
        grouped_rows.setdefault((algorithm, seed), []).append(row)

    return grouped_rows


def build_seed_summary(episode_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped_rows = group_episode_rows_by_algorithm_seed(episode_rows)
    seed_summary_rows: List[Dict[str, object]] = []

    for (algorithm, seed), group_rows in sorted(grouped_rows.items()):
        episode_seed_values = [to_int(row["episode_seed"]) for row in group_rows]
        episode_reward_values = numeric_column_values(group_rows, "episode_reward")

        summary_row: Dict[str, object] = {
            "algorithm": algorithm,
            "seed": seed,
            "n_episodes": len(group_rows),
            "episode_seed_min": min(episode_seed_values),
            "episode_seed_max": max(episode_seed_values),
            "mean_reward": safe_mean(episode_reward_values),
            "std_reward": safe_std_population(episode_reward_values),
        }

        for metric_name in ALL_METRICS:
            if metric_name in {"mean_reward", "episode_reward"}:
                continue
            metric_values = numeric_column_values(group_rows, metric_name)
            if metric_values:
                summary_row[metric_name] = safe_mean(metric_values)

        seed_summary_rows.append(summary_row)

    return seed_summary_rows


def build_crosscheck_rows(seed_summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    preliminary_summary_path = INPUT_DIR / "phase2c_seed_summary.csv"
    if not preliminary_summary_path.exists():
        return []

    with preliminary_summary_path.open(newline="") as csv_file:
        preliminary_rows = list(csv.DictReader(csv_file))

    preliminary_by_key = {
        (str(row["algorithm"]), to_int(row["seed"])): row
        for row in preliminary_rows
    }

    comparable_columns = [
        "mean_reward",
        "std_reward",
        "action_fraction_at_max",
        "action_fraction_zero",
        "active_action_count_mean",
    ]

    crosscheck_rows: List[Dict[str, object]] = []

    for new_row in seed_summary_rows:
        key = (str(new_row["algorithm"]), int(new_row["seed"]))
        preliminary_row = preliminary_by_key.get(key)
        if preliminary_row is None:
            continue

        for column_name in comparable_columns:
            new_value = to_float(new_row.get(column_name))
            old_value = to_float(preliminary_row.get(column_name))
            if new_value is None or old_value is None:
                continue

            absolute_difference = abs(new_value - old_value)
            crosscheck_rows.append(
                {
                    "algorithm": key[0],
                    "seed": key[1],
                    "metric": column_name,
                    "new_value": f"{new_value:.12g}",
                    "preliminary_value": f"{old_value:.12g}",
                    "absolute_difference": f"{absolute_difference:.12g}",
                    "passed_1e-6": absolute_difference <= 1e-6,
                }
            )

    return crosscheck_rows


def build_paired_differences(seed_summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    baseline_by_seed = {
        int(row["seed"]): row
        for row in seed_summary_rows
        if row["algorithm"] == "baseline_25cp"
    }
    hierarchical_by_seed = {
        int(row["seed"]): row
        for row in seed_summary_rows
        if row["algorithm"] == "hierarchical_25cp"
    }

    paired_rows: List[Dict[str, object]] = []

    for seed in sorted(set(baseline_by_seed).intersection(hierarchical_by_seed)):
        baseline_row = baseline_by_seed[seed]
        hierarchical_row = hierarchical_by_seed[seed]
        paired_row: Dict[str, object] = {"seed": seed}

        for metric_name in PRIMARY_METRICS:
            baseline_value = to_float(baseline_row.get(metric_name))
            hierarchical_value = to_float(hierarchical_row.get(metric_name))

            if baseline_value is None or hierarchical_value is None:
                continue

            paired_row[f"{metric_name}_baseline"] = f"{baseline_value:.12g}"
            paired_row[f"{metric_name}_hierarchical"] = f"{hierarchical_value:.12g}"
            paired_row[f"{metric_name}_diff_hier_minus_base"] = f"{hierarchical_value - baseline_value:.12g}"

        paired_rows.append(paired_row)

    return paired_rows


def confidence_interval_95(values: List[float]) -> Tuple[str, str]:
    if len(values) < 2:
        return "", ""

    diff_mean = mean(values)
    sample_std = stdev(values)
    standard_error = sample_std / math.sqrt(len(values))

    if len(values) == 5:
        t_critical = 2.776
    else:
        t_critical = 1.96

    ci_low = diff_mean - t_critical * standard_error
    ci_high = diff_mean + t_critical * standard_error
    return f"{ci_low:.12g}", f"{ci_high:.12g}"


def scipy_paired_tests(values: List[float]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "scipy_available": False,
        "paired_t_statistic": "",
        "paired_t_pvalue_two_sided": "",
        "wilcoxon_statistic": "",
        "wilcoxon_pvalue_two_sided": "",
    }

    try:
        from scipy import stats
    except Exception as scipy_error:
        result["scipy_error"] = str(scipy_error)
        return result

    result["scipy_available"] = True

    if not values:
        return result

    t_test_result = stats.ttest_1samp(values, popmean=0.0)
    result["paired_t_statistic"] = f"{float(t_test_result.statistic):.12g}"
    result["paired_t_pvalue_two_sided"] = f"{float(t_test_result.pvalue):.12g}"

    if any(value != 0 for value in values):
        wilcoxon_result = stats.wilcoxon(values)
        result["wilcoxon_statistic"] = f"{float(wilcoxon_result.statistic):.12g}"
        result["wilcoxon_pvalue_two_sided"] = f"{float(wilcoxon_result.pvalue):.12g}"

    return result


def build_aggregate_statistics(
    seed_summary_rows: List[Dict[str, object]],
    paired_difference_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    aggregate_rows: List[Dict[str, object]] = []

    for metric_name in PRIMARY_METRICS:
        baseline_values = [
            value
            for row in seed_summary_rows
            if row["algorithm"] == "baseline_25cp"
            for value in [to_float(row.get(metric_name))]
            if value is not None
        ]

        hierarchical_values = [
            value
            for row in seed_summary_rows
            if row["algorithm"] == "hierarchical_25cp"
            for value in [to_float(row.get(metric_name))]
            if value is not None
        ]

        diff_column = f"{metric_name}_diff_hier_minus_base"
        diff_values = [
            value
            for row in paired_difference_rows
            for value in [to_float(row.get(diff_column))]
            if value is not None
        ]

        ci_low, ci_high = confidence_interval_95(diff_values)
        test_results = scipy_paired_tests(diff_values)

        aggregate_row: Dict[str, object] = {
            "metric": metric_name,
            "baseline_mean": safe_mean(baseline_values),
            "baseline_std": safe_std_sample(baseline_values),
            "hierarchical_mean": safe_mean(hierarchical_values),
            "hierarchical_std": safe_std_sample(hierarchical_values),
            "paired_diff_mean_hier_minus_base": safe_mean(diff_values),
            "paired_diff_std": safe_std_sample(diff_values),
            "paired_diff_median": f"{median(diff_values):.12g}" if diff_values else "",
            "paired_diff_min": f"{min(diff_values):.12g}" if diff_values else "",
            "paired_diff_max": f"{max(diff_values):.12g}" if diff_values else "",
            "paired_diff_95ci_low": ci_low,
            "paired_diff_95ci_high": ci_high,
            **test_results,
        }

        aggregate_rows.append(aggregate_row)

    return aggregate_rows


def create_figures(
    seed_summary_rows: List[Dict[str, object]],
    paired_difference_rows: List[Dict[str, object]],
) -> Tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as matplotlib_error:
        return False, str(matplotlib_error)

    baseline_rows = sorted(
        [row for row in seed_summary_rows if row["algorithm"] == "baseline_25cp"],
        key=lambda row: int(row["seed"]),
    )
    hierarchical_rows = sorted(
        [row for row in seed_summary_rows if row["algorithm"] == "hierarchical_25cp"],
        key=lambda row: int(row["seed"]),
    )

    def get_values(rows: List[Dict[str, object]], metric_name: str) -> List[float]:
        return [
            value
            for row in rows
            for value in [to_float(row.get(metric_name))]
            if value is not None
        ]

    def save_line_plot(metric_name: str, filename: str, ylabel: str, title: str) -> None:
        seeds = [int(row["seed"]) for row in baseline_rows]
        baseline_values = get_values(baseline_rows, metric_name)
        hierarchical_values = get_values(hierarchical_rows, metric_name)

        if not baseline_values or not hierarchical_values:
            return

        fig, axis = plt.subplots(figsize=(7, 4))
        axis.plot(seeds, baseline_values, marker="o", label="Baseline TD3 ActionGNN")
        axis.plot(seeds, hierarchical_values, marker="o", label="Hierarchical TD3 ActionGNN")
        axis.set_xlabel("Training seed")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="--", alpha=0.5)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close(fig)

    save_line_plot(
        "mean_reward",
        "mean_reward_by_seed.png",
        "Mean reward",
        "25 CP formal evaluation: mean reward by seed",
    )

    save_line_plot(
        "action_fraction_at_max",
        "action_fraction_at_max_by_seed.png",
        "Action fraction at max",
        "25 CP formal evaluation: action saturation by seed",
    )

    save_line_plot(
        "energy_tracking_error",
        "energy_tracking_error_by_seed.png",
        "Energy tracking error",
        "25 CP formal evaluation: energy tracking error by seed",
    )

    reward_diff_values = [
        value
        for row in paired_difference_rows
        for value in [to_float(row.get("mean_reward_diff_hier_minus_base"))]
        if value is not None
    ]
    reward_diff_seeds = [int(row["seed"]) for row in paired_difference_rows]

    if reward_diff_values:
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.axhline(0, linestyle="--", linewidth=1)
        axis.bar(reward_diff_seeds, reward_diff_values)
        axis.set_xlabel("Training seed")
        axis.set_ylabel("Hierarchical − baseline reward")
        axis.set_title("Paired reward difference by seed")
        axis.grid(True, axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "paired_reward_difference.png", dpi=200)
        plt.close(fig)

    return True, ""


def create_interpretation_notes(aggregate_rows: List[Dict[str, object]]) -> None:
    aggregate_by_metric = {
        str(row["metric"]): row
        for row in aggregate_rows
    }

    reward_row = aggregate_by_metric.get("mean_reward", {})
    saturation_row = aggregate_by_metric.get("action_fraction_at_max", {})

    notes = [
        "# Phase 2C Formal 25 CP Analysis Notes",
        "",
        "## Scope",
        "",
        "- Environment: `PublicPST_25cp.yaml`",
        "- Algorithms: `baseline_25cp` vs `hierarchical_25cp`",
        "- Training budget: 50,000 timesteps per seed",
        "- Seeds: 0, 1, 2, 3, 4",
        "- Final evaluation: 30 deterministic episodes per seed",
        "- Primary comparison unit: training seed",
        "",
        "## Main observations",
        "",
        (
            "- Mean reward baseline: "
            f"{reward_row.get('baseline_mean', '')}; hierarchical: "
            f"{reward_row.get('hierarchical_mean', '')}; paired difference: "
            f"{reward_row.get('paired_diff_mean_hier_minus_base', '')}."
        ),
        (
            "- Action fraction at max baseline: "
            f"{saturation_row.get('baseline_mean', '')}; hierarchical: "
            f"{saturation_row.get('hierarchical_mean', '')}; paired difference: "
            f"{saturation_row.get('paired_diff_mean_hier_minus_base', '')}."
        ),
        "- Seed 2 is an anomaly: hierarchical and baseline are nearly identical on reward/tracking.",
        "",
        "## Interpretation boundary",
        "",
        (
            "These results are formal 25 CP short-budget evidence. They should not be phrased "
            "as final thesis-wide superiority because 100 CP / 500 CP evidence is not yet available."
        ),
        "",
        "## Limitations",
        "",
        "- 50k timesteps is still a short-budget formal run.",
        "- The current hierarchy uses fixed total budget `num_active_evs`, not a learned CPO total-budget head.",
        "- No 100 CP or 500 CP transfer evidence is included.",
        "- Hierarchical actor was slower on CPU in earlier pilot/formal runtime observations.",
        "",
        "## Recommended next step",
        "",
        "Prepare M3/HPC scripts for 100 CP or 500 CP scale validation rather than running larger settings locally.",
        "",
    ]

    (OUTPUT_DIR / "interpretation_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading input CSV files...")
    all_rows: List[Dict[str, str]] = []

    for input_file in INPUT_FILES:
        rows = read_csv_rows(input_file)
        all_rows.extend(rows)
        print(f"  OK: {input_file} ({len(rows)} rows)")

    episode_rows = [row for row in all_rows if row.get("row_type") == "episode"]
    summary_rows = [row for row in all_rows if row.get("row_type") == "summary"]

    if len(episode_rows) != 300:
        raise AssertionError(f"Expected 300 episode rows, got {len(episode_rows)}")
    if len(summary_rows) != 10:
        raise AssertionError(f"Expected 10 summary rows, got {len(summary_rows)}")

    write_csv(
        OUTPUT_DIR / "phase2c_episode_summary.csv",
        episode_rows,
        union_fieldnames(episode_rows),
    )

    seed_summary_rows = build_seed_summary(episode_rows)
    write_csv(
        OUTPUT_DIR / "phase2c_seed_summary.csv",
        seed_summary_rows,
        union_fieldnames(seed_summary_rows),
    )

    crosscheck_rows = build_crosscheck_rows(seed_summary_rows)
    if crosscheck_rows:
        write_csv(
            OUTPUT_DIR / "phase2c_seed_summary_crosscheck.csv",
            crosscheck_rows,
            union_fieldnames(crosscheck_rows),
        )

    paired_difference_rows = build_paired_differences(seed_summary_rows)
    write_csv(
        OUTPUT_DIR / "phase2c_paired_differences.csv",
        paired_difference_rows,
        union_fieldnames(paired_difference_rows),
    )

    aggregate_rows = build_aggregate_statistics(seed_summary_rows, paired_difference_rows)
    write_csv(
        OUTPUT_DIR / "phase2c_aggregate_statistics.csv",
        aggregate_rows,
        union_fieldnames(aggregate_rows),
    )

    figures_created, figure_error = create_figures(seed_summary_rows, paired_difference_rows)
    create_interpretation_notes(aggregate_rows)

    required_outputs = [
        OUTPUT_DIR / "phase2c_seed_summary.csv",
        OUTPUT_DIR / "phase2c_episode_summary.csv",
        OUTPUT_DIR / "phase2c_paired_differences.csv",
        OUTPUT_DIR / "phase2c_aggregate_statistics.csv",
        OUTPUT_DIR / "interpretation_notes.md",
    ]

    for required_output in required_outputs:
        if not required_output.exists():
            raise FileNotFoundError(required_output)

    print("")
    print("Created required analysis artefacts:")
    for required_output in required_outputs:
        print(f"  OK: {required_output}")

    if crosscheck_rows:
        failed_crosschecks = [
            row for row in crosscheck_rows
            if str(row.get("passed_1e-6")) != "True"
        ]
        print("")
        print(f"Seed-summary cross-check rows: {len(crosscheck_rows)}")
        print(f"Seed-summary cross-check failures: {len(failed_crosschecks)}")
        if failed_crosschecks:
            for row in failed_crosschecks:
                print("  FAIL:", row)

    print("")
    print(f"Figures created: {figures_created}")
    if figure_error:
        print(f"Figure error: {figure_error}")

    print("")
    print("Seed-level summary:")
    for row in seed_summary_rows:
        print(
            f"  {row['algorithm']} seed={row['seed']} "
            f"mean_reward={row.get('mean_reward', '')} "
            f"tracking_error={row.get('tracking_error', '')} "
            f"energy_tracking_error={row.get('energy_tracking_error', '')} "
            f"power_tracker_violation={row.get('power_tracker_violation', '')} "
            f"action_fraction_at_max={row.get('action_fraction_at_max', '')}"
        )

    print("")
    print("Aggregate statistics:")
    for row in aggregate_rows:
        print(
            f"  {row['metric']}: "
            f"baseline_mean={row.get('baseline_mean', '')}, "
            f"hierarchical_mean={row.get('hierarchical_mean', '')}, "
            f"diff_mean={row.get('paired_diff_mean_hier_minus_base', '')}, "
            f"ci95=[{row.get('paired_diff_95ci_low', '')}, {row.get('paired_diff_95ci_high', '')}], "
            f"p_t={row.get('paired_t_pvalue_two_sided', '')}, "
            f"p_wilcoxon={row.get('wilcoxon_pvalue_two_sided', '')}"
        )

    print("")
    print("Phase 2C.2 aggregation completed.")


if __name__ == "__main__":
    main()
