#!/usr/bin/env python3
"""Phase 2D.4: aggregate 100CP formal eval30 controlled-evaluation CSVs.

Source of truth:
  /scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_eval30/eval/*.csv

This script reads:
  - baseline_100cp_seed{0..4}_eval30.csv
  - hierarchical_100cp_seed{0..4}_eval30.csv

It writes:
  - phase2d_100cp_episode_summary.csv
  - phase2d_100cp_seed_summary.csv
  - phase2d_100cp_paired_differences.csv
  - phase2d_100cp_aggregate_statistics.csv
  - phase2d_100cp_metric_direction_table.csv
  - phase2d_100cp_interpretation_notes.md
  - plots/*.png

Controlled evaluator CSVs are used as formal evidence.
Training logs are intentionally not used because they are not strict single-schema CSVs.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover
    stats = None
    SCIPY_IMPORT_ERROR = repr(exc)
else:
    SCIPY_IMPORT_ERROR = ""

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    MATPLOTLIB_IMPORT_ERROR = repr(exc)
else:
    MATPLOTLIB_IMPORT_ERROR = ""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source_column: str
    higher_is_better: bool | None
    description: str


METRICS: List[MetricSpec] = [
    MetricSpec("mean_reward", "episode_reward", True, "Mean episode reward; higher / less negative is better."),
    MetricSpec("tracking_error", "tracking_error", False, "Mean PST tracking error; lower is better."),
    MetricSpec("energy_tracking_error", "energy_tracking_error", False, "Mean energy tracking error; lower is better."),
    MetricSpec("power_tracker_violation", "power_tracker_violation", False, "Mean power tracker violation; lower is better."),
    MetricSpec("action_fraction_at_max", "action_fraction_at_max", False, "Mean fraction of actions saturated at max; lower is generally better."),
    MetricSpec("action_fraction_zero", "action_fraction_zero", None, "Mean fraction of zero actions; interpretation is context-dependent."),
    MetricSpec("active_action_count_mean", "active_action_count_mean", None, "Mean count of active/non-zero EV actions; interpretation is context-dependent."),
    MetricSpec("total_ev_served", "total_ev_served", True, "Mean number of served EVs; higher is generally better."),
    MetricSpec("total_transformer_overload", "total_transformer_overload", False, "Mean transformer overload; lower is better."),
    MetricSpec("total_energy_charged", "total_energy_charged", None, "Mean energy charged; context-dependent."),
    MetricSpec("total_energy_discharged", "total_energy_discharged", None, "Mean energy discharged; context-dependent."),
    MetricSpec("total_profits", "total_profits", True, "Mean profit proxy; higher is better if comparable."),
    MetricSpec("average_user_satisfaction", "average_user_satisfaction", True, "Mean user satisfaction; higher is better."),
    MetricSpec("battery_degradation", "battery_degradation", False, "Mean battery degradation; lower is better."),
]

KEY_PLOT_METRICS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Phase 2D 100CP formal eval30 results.")
    parser.add_argument(
        "--eval_dir",
        type=Path,
        default=Path("/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_eval30/eval"),
        help="Directory containing baseline_100cp_seed*_eval30.csv and hierarchical_100cp_seed*_eval30.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/scratch2/fr57/cche0357/EV-GNN_runs/phase2d_100cp_formal_aggregation"),
        help="Directory where aggregation artefacts will be written.",
    )
    parser.add_argument("--scenario_name", default="phase2d_100cp", help="Prefix for output artefacts.")
    parser.add_argument("--expected_seeds", type=int, default=5)
    parser.add_argument("--expected_episodes", type=int, default=30)
    return parser.parse_args()


def discover_files(eval_dir: Path) -> Dict[Tuple[str, int], Path]:
    pattern = re.compile(r"^(baseline|hierarchical)_100cp_seed(\d+)_eval30\.csv$")
    files: Dict[Tuple[str, int], Path] = {}
    for path in sorted(eval_dir.glob("*_100cp_seed*_eval30.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        alg_short = match.group(1)
        seed = int(match.group(2))
        files[(alg_short, seed)] = path
    return files


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_and_validate(
    files: Dict[Tuple[str, int], Path],
    expected_seeds: int,
    expected_episodes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expected_keys = {(alg, seed) for alg in ("baseline", "hierarchical") for seed in range(expected_seeds)}
    missing = sorted(expected_keys.difference(files.keys()))
    extra = sorted(set(files.keys()).difference(expected_keys))
    require(not missing, f"Missing eval CSVs for: {missing}")
    require(not extra, f"Unexpected eval CSVs for: {extra}")

    episode_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []

    for (alg_short, seed), path in sorted(files.items()):
        df = pd.read_csv(path)
        required_cols = ["row_type", "episode_reward", "episode_seed", "episode_steps", "done"]
        for col in required_cols:
            require(col in df.columns, f"{path.name} lacks required column: {col}")

        episode_df = df[df["row_type"] == "episode"].copy()
        summary_df = df[df["row_type"] == "summary"].copy()

        require(len(episode_df) == expected_episodes, f"{path.name}: expected {expected_episodes} episode rows, found {len(episode_df)}")
        require(len(summary_df) == 1, f"{path.name}: expected 1 summary row, found {len(summary_df)}")
        require(set(episode_df["episode_steps"].astype(int)) == {112}, f"{path.name}: non-112 episode_steps detected")
        require(episode_df["done"].astype(str).str.lower().isin(["true", "1"]).all(), f"{path.name}: not all episodes done=True")

        episode_df.insert(0, "alg_short", alg_short)
        episode_df.insert(1, "source_seed", seed)
        episode_df.insert(2, "source_file", path.name)
        summary_df.insert(0, "alg_short", alg_short)
        summary_df.insert(1, "source_seed", seed)
        summary_df.insert(2, "source_file", path.name)

        episode_frames.append(episode_df)
        summary_frames.append(summary_df)

    episode_all = pd.concat(episode_frames, ignore_index=True)
    summary_all = pd.concat(summary_frames, ignore_index=True)

    for seed in range(expected_seeds):
        base_episode_seeds = episode_all[
            (episode_all["alg_short"] == "baseline") & (episode_all["source_seed"] == seed)
        ]["episode_seed"].astype(int).tolist()
        hier_episode_seeds = episode_all[
            (episode_all["alg_short"] == "hierarchical") & (episode_all["source_seed"] == seed)
        ]["episode_seed"].astype(int).tolist()
        require(
            base_episode_seeds == hier_episode_seeds,
            f"Episode seed mismatch for seed {seed}: baseline {base_episode_seeds[:3]}..., hierarchical {hier_episode_seeds[:3]}...",
        )

    return episode_all, summary_all


def build_seed_summary(episode_all: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (alg_short, seed), group in episode_all.groupby(["alg_short", "source_seed"], sort=True):
        row = {
            "alg_short": alg_short,
            "algorithm": group["algorithm"].iloc[0] if "algorithm" in group.columns else alg_short,
            "seed": int(seed),
            "n_eval_episodes": int(len(group)),
            "episode_seed_min": int(group["episode_seed"].min()),
            "episode_seed_max": int(group["episode_seed"].max()),
            "all_done": bool(group["done"].astype(str).str.lower().isin(["true", "1"]).all()),
            "episode_steps_mean": float(group["episode_steps"].mean()),
        }

        for metric in METRICS:
            if metric.source_column in group.columns:
                row[metric.name] = float(pd.to_numeric(group[metric.source_column], errors="coerce").mean())

        row["reward_std"] = float(pd.to_numeric(group["episode_reward"], errors="coerce").std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["seed", "alg_short"]).reset_index(drop=True)


def paired_statistics(values_base: np.ndarray, values_hier: np.ndarray, higher_is_better: bool | None) -> dict:
    diff = values_hier - values_base
    n = len(diff)
    diff_mean = float(np.nanmean(diff))
    diff_std = float(np.nanstd(diff, ddof=1)) if n > 1 else float("nan")
    diff_se = diff_std / math.sqrt(n) if n > 1 and np.isfinite(diff_std) else float("nan")

    all_finite = np.all(np.isfinite(diff))
    all_zero_diff = bool(all_finite and np.allclose(diff, 0.0, atol=1e-12, rtol=1e-12))

    if stats is not None and n > 1 and all_finite:
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = diff_mean - t_crit * diff_se
        ci_high = diff_mean + t_crit * diff_se

        if all_zero_diff:
            t_stat = float("nan")
            t_p = float("nan")
            w_stat = 0.0
            w_p = 1.0
        else:
            t_res = stats.ttest_rel(values_hier, values_base, nan_policy="omit")
            t_stat = float(t_res.statistic)
            t_p = float(t_res.pvalue)
            try:
                w_res = stats.wilcoxon(values_hier, values_base, zero_method="wilcox", alternative="two-sided", mode="auto")
                w_stat = float(w_res.statistic)
                w_p = float(w_res.pvalue)
            except Exception:
                w_stat = float("nan")
                w_p = float("nan")
    else:
        ci_low = ci_high = t_stat = t_p = w_stat = w_p = float("nan")

    if higher_is_better is True:
        favourable_diff = diff_mean
        favourable_label = "hierarchical_minus_baseline_positive_is_better"
    elif higher_is_better is False:
        favourable_diff = -diff_mean
        favourable_label = "baseline_minus_hierarchical_positive_is_better"
    else:
        favourable_diff = float("nan")
        favourable_label = "context_dependent"

    base_mean = float(np.nanmean(values_base))
    hier_mean = float(np.nanmean(values_hier))
    pct_change_hier_vs_base = float((hier_mean - base_mean) / abs(base_mean) * 100.0) if abs(base_mean) > 1e-12 else float("nan")

    return {
        "baseline_mean": base_mean,
        "hierarchical_mean": hier_mean,
        "diff_hier_minus_base_mean": diff_mean,
        "diff_hier_minus_base_std": diff_std,
        "diff_hier_minus_base_ci95_low": float(ci_low),
        "diff_hier_minus_base_ci95_high": float(ci_high),
        "paired_t_stat": t_stat,
        "paired_t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "favourable_diff_mean": favourable_diff,
        "favourable_direction": favourable_label,
        "pct_change_hier_vs_base": pct_change_hier_vs_base,
        "n_paired_seeds": n,
    }


def build_paired_outputs(seed_summary: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = seed_summary[seed_summary["alg_short"] == "baseline"].set_index("seed").sort_index()
    hier = seed_summary[seed_summary["alg_short"] == "hierarchical"].set_index("seed").sort_index()
    require(list(base.index) == list(hier.index), "Baseline/hierarchical seed indices do not match")

    paired_rows: List[dict] = []
    for seed in base.index:
        row = {"seed": int(seed)}
        for metric in METRICS:
            if metric.name in base.columns and metric.name in hier.columns:
                row[f"baseline_{metric.name}"] = float(base.loc[seed, metric.name])
                row[f"hierarchical_{metric.name}"] = float(hier.loc[seed, metric.name])
                row[f"diff_hier_minus_base_{metric.name}"] = float(hier.loc[seed, metric.name] - base.loc[seed, metric.name])
        paired_rows.append(row)
    paired_df = pd.DataFrame(paired_rows)

    stat_rows: List[dict] = []
    for metric in METRICS:
        if metric.name not in base.columns or metric.name not in hier.columns:
            continue
        values_base = pd.to_numeric(base[metric.name], errors="coerce").to_numpy(dtype=float)
        values_hier = pd.to_numeric(hier[metric.name], errors="coerce").to_numpy(dtype=float)
        stat = paired_statistics(values_base, values_hier, metric.higher_is_better)
        stat_rows.append({
            "metric": metric.name,
            "source_column": metric.source_column,
            "higher_is_better": metric.higher_is_better,
            "description": metric.description,
            **stat,
        })

    return paired_df, pd.DataFrame(stat_rows)


def write_plots(seed_summary: pd.DataFrame, paired_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if plt is None:
        (plots_dir / "PLOTS_SKIPPED.txt").write_text(f"matplotlib import failed: {MATPLOTLIB_IMPORT_ERROR}\n")
        return

    for metric in KEY_PLOT_METRICS:
        if metric not in seed_summary.columns:
            continue

        pivot = seed_summary.pivot(index="seed", columns="alg_short", values=metric).sort_index()
        if not {"baseline", "hierarchical"}.issubset(set(pivot.columns)):
            continue

        fig = plt.figure(figsize=(7, 4.5))
        ax = fig.gca()
        ax.plot(pivot.index, pivot["baseline"], marker="o", label="baseline")
        ax.plot(pivot.index, pivot["hierarchical"], marker="o", label="hierarchical")
        ax.set_title(f"100CP eval30: {metric} by seed")
        ax.set_xlabel("Seed")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_by_seed.png", dpi=180)
        plt.close(fig)

        diff_col = f"diff_hier_minus_base_{metric}"
        if diff_col in paired_df.columns:
            fig = plt.figure(figsize=(7, 4.5))
            ax = fig.gca()
            ax.axhline(0, linewidth=1)
            ax.plot(paired_df["seed"], paired_df[diff_col], marker="o")
            ax.set_title(f"100CP eval30: hierarchical - baseline ({metric})")
            ax.set_xlabel("Seed")
            ax.set_ylabel(f"Δ {metric}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric}_paired_diff.png", dpi=180)
            plt.close(fig)


def p_to_text(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def write_interpretation_notes(stats_df: pd.DataFrame, output_dir: Path, scenario_name: str) -> None:
    def get_row(metric: str) -> pd.Series:
        matches = stats_df[stats_df["metric"] == metric]
        if matches.empty:
            raise KeyError(metric)
        return matches.iloc[0]

    lines: List[str] = []
    lines.append(f"# {scenario_name} Formal Eval30 Aggregation Notes")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("These notes are generated from the controlled evaluator CSVs only. Training logs are not used as formal performance evidence.")
    lines.append("")
    lines.append("## Key Results")
    lines.append("")

    for metric in [
        "mean_reward",
        "tracking_error",
        "energy_tracking_error",
        "power_tracker_violation",
        "action_fraction_at_max",
        "total_ev_served",
        "total_transformer_overload",
    ]:
        if metric not in set(stats_df["metric"]):
            continue
        r = get_row(metric)
        lines.append(
            f"- `{metric}`: baseline mean `{r['baseline_mean']:.6g}`, hierarchical mean `{r['hierarchical_mean']:.6g}`, "
            f"hierarchical-baseline diff `{r['diff_hier_minus_base_mean']:.6g}`, 95% CI "
            f"[`{r['diff_hier_minus_base_ci95_low']:.6g}`, `{r['diff_hier_minus_base_ci95_high']:.6g}`], "
            f"paired t-test p={p_to_text(float(r['paired_t_p']))}, Wilcoxon p={p_to_text(float(r['wilcoxon_p']))}."
        )

    lines.append("")
    lines.append("## Interpretation Rules")
    lines.append("")
    lines.append("- For `mean_reward`, higher / less negative is better.")
    lines.append("- For tracking/violation/overload/degradation metrics, lower is better.")
    lines.append("- `action_fraction_zero` and `active_action_count_mean` are context-dependent diagnostics, not direct success metrics.")
    lines.append("- With only five paired seeds, p-values should be interpreted cautiously; report direction, effect size, and seed-level consistency rather than relying only on significance thresholds.")
    lines.append("")
    lines.append("## Automatically Generated Caution")
    lines.append("")

    if "mean_reward" in set(stats_df["metric"]):
        r = get_row("mean_reward")
        if float(r["diff_hier_minus_base_mean"]) > 0 and np.isfinite(float(r["paired_t_p"])) and float(r["paired_t_p"]) < 0.05:
            lines.append("The hierarchical actor improves mean reward with statistically significant paired t-test evidence at p < 0.05, subject to the small n=5 seed limitation.")
        elif float(r["diff_hier_minus_base_mean"]) > 0:
            lines.append("The hierarchical actor improves mean reward on average, but the paired evidence is not statistically robust at p < 0.05 with n=5 seeds.")
        else:
            lines.append("The hierarchical actor does not improve mean reward on average in this 100CP eval30 run.")

    lines.append("")
    lines.append("## Generated Artefacts")
    lines.append("")
    lines.append("- `phase2d_100cp_episode_summary.csv`")
    lines.append("- `phase2d_100cp_seed_summary.csv`")
    lines.append("- `phase2d_100cp_paired_differences.csv`")
    lines.append("- `phase2d_100cp_aggregate_statistics.csv`")
    lines.append("- `phase2d_100cp_metric_direction_table.csv`")
    lines.append("- `plots/*.png`")

    (output_dir / f"{scenario_name}_interpretation_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.eval_dir = args.eval_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.eval_dir)
    print(f"Discovered {len(files)} eval CSV files in {args.eval_dir}")
    for key, path in sorted(files.items()):
        print(f"  {key}: {path}")

    episode_all, summary_all = read_and_validate(files, args.expected_seeds, args.expected_episodes)
    seed_summary = build_seed_summary(episode_all)
    paired_df, stats_df = build_paired_outputs(seed_summary)

    prefix = args.scenario_name

    episode_cols_front = [
        "alg_short",
        "source_seed",
        "source_file",
        "episode_index",
        "episode_seed",
        "episode_reward",
        "episode_steps",
        "done",
    ]
    episode_cols = episode_cols_front + [c for c in episode_all.columns if c not in episode_cols_front]

    episode_all[episode_cols].to_csv(args.output_dir / f"{prefix}_episode_summary.csv", index=False)
    summary_all.to_csv(args.output_dir / f"{prefix}_raw_summary_rows.csv", index=False)
    seed_summary.to_csv(args.output_dir / f"{prefix}_seed_summary.csv", index=False)
    paired_df.to_csv(args.output_dir / f"{prefix}_paired_differences.csv", index=False)
    stats_df.to_csv(args.output_dir / f"{prefix}_aggregate_statistics.csv", index=False)

    metric_direction = pd.DataFrame([
        {
            "metric": metric.name,
            "source_column": metric.source_column,
            "higher_is_better": metric.higher_is_better,
            "description": metric.description,
        }
        for metric in METRICS
    ])
    metric_direction.to_csv(args.output_dir / f"{prefix}_metric_direction_table.csv", index=False)

    write_plots(seed_summary, paired_df, args.output_dir)
    write_interpretation_notes(stats_df, args.output_dir, prefix)

    print("\n=== Seed summary ===")
    print(
        seed_summary[
            [
                "alg_short",
                "seed",
                "mean_reward",
                "tracking_error",
                "energy_tracking_error",
                "power_tracker_violation",
                "action_fraction_at_max",
                "total_ev_served",
            ]
        ].to_string(index=False)
    )

    print("\n=== Aggregate statistics ===")
    show_cols = [
        "metric",
        "baseline_mean",
        "hierarchical_mean",
        "diff_hier_minus_base_mean",
        "diff_hier_minus_base_ci95_low",
        "diff_hier_minus_base_ci95_high",
        "paired_t_p",
        "wilcoxon_p",
    ]
    print(stats_df[show_cols].to_string(index=False))

    if SCIPY_IMPORT_ERROR:
        print(f"WARNING: scipy import failed, statistical tests may be NA: {SCIPY_IMPORT_ERROR}")
    if MATPLOTLIB_IMPORT_ERROR:
        print(f"WARNING: matplotlib import failed, plots skipped: {MATPLOTLIB_IMPORT_ERROR}")

    print("\n=== Output files ===")
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
