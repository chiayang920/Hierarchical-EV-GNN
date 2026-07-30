import argparse
import csv
from pathlib import Path

from utils.infrastructure_diagnostics import (
    CHARGER_DIAGNOSTIC_COLUMNS,
    EPISODE_DIAGNOSTIC_COLUMNS,
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
)


OFFSETS = {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}
TOPOLOGY = {
    "25cp": (25, 3),
    "100cp": (100, 7),
    "500cp": (500, 35),
    "1000cp": (1000, 70),
}
CANONICAL_EVAL30_COLUMNS = [
    "row_type",
    "algorithm",
    "seed",
    "episode_index",
    "episode_seed",
    "episode_steps",
    "done",
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_energy_charged",
    "total_energy_discharged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_transformer_overload",
    "total_ev_served",
    "action_mean",
    "action_fraction_at_max",
    "active_action_count_mean",
    "mapped_action_dimension",
    "same_pass_at_max_count",
    "total_action_decision_denominator",
]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def episode_seed(scale, training_seed, episode_index):
    return OFFSETS[scale] + (1000 * training_seed) + episode_index


def episode_row(args, episode_index):
    row = {column: "0.0" for column in EPISODE_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": args.matrix_job_id,
            "scale": args.scale,
            "algorithm": args.algorithm,
            "training_seed": str(args.seed),
            "episode_index": str(episode_index),
            "episode_seed": str(episode_seed(args.scale, args.seed, episode_index)),
            "config": args.config,
            "checkpoint_prefix": args.checkpoint,
            "run_name": args.run_name,
            "episode_steps": str(args.max_episode_steps),
            "done": "True",
            "episode_reward": "-1.0",
            "max_action": "1.0",
            "max_action_tolerance": "1e-06",
            "environment_action_low": "-1.0",
            "environment_action_high": "1.0",
            "observed_action_min_active": "0.0",
            "observed_action_max_active": "1.0",
            "action_tolerance": "1e-06",
            "environment_action_domain_support": "signed",
            "v2g_enabled": "False",
            "v2g_enabled_source": "config",
            "global_action_fraction_at_max_all_slots": "0.5",
            "global_action_fraction_at_max_active": "0.5",
            "global_action_nonzero_fraction_active": "1.0",
            "active_action_decision_count": "1",
            "active_action_below_environment_low_count": "0",
            "active_action_below_environment_low_fraction": "0.0",
            "active_action_above_environment_high_count": "0",
            "active_action_above_environment_high_fraction": "0.0",
            "global_positive_action_fraction_active": "1.0",
            "global_zero_action_fraction_active": "0.0",
            "global_negative_action_fraction_active": "0.0",
            "global_action_fraction_at_positive_max_active": "0.5",
            "global_action_fraction_at_negative_min_active": "0.0",
            "global_positive_action_sum_active": "1.0",
            "global_negative_action_magnitude_sum_active": "0.0",
            "global_action_mean_all_slots": "0.5",
            "global_action_mean_active": "0.5",
            "global_action_sum_active": "1.0",
            "active_slot_count_mean": "1.0",
            "nonzero_action_count_mean_all_slots": "1.0",
            "inactive_slot_fraction_mean": "0.0",
            "inactive_slot_decision_count": "0",
            "inactive_nonzero_action_count": "0",
            "inactive_nonzero_action_fraction_all_slots": "0.0",
            "transformer_action_fraction_at_max_active_macro_mean": "0.5",
            "charger_action_fraction_at_max_active_macro_mean": "0.5",
            "transformer_action_nonzero_fraction_active_macro_mean": "1.0",
            "charger_action_nonzero_fraction_active_macro_mean": "1.0",
            "transformer_allocation_zero_pressure_step_fraction": "0.0",
            "transformer_allocation_valid_step_count": "1.0",
            "charger_allocation_zero_pressure_step_fraction": "0.0",
            "charger_allocation_valid_step_count": "1.0",
            "tracking_error": "1.0",
            "energy_tracking_error": "1.0",
            "power_tracker_violation": "1.0",
            "total_energy_charged": "1.0",
            "total_energy_discharged": "0.0",
            "average_user_satisfaction": "1.0",
            "energy_user_satisfaction": "100.0",
            "total_transformer_overload": "0.0",
            "total_ev_served": "1",
            "diagnostic_schema_version": "3",
        }
    )
    return {column: row[column] for column in EPISODE_DIAGNOSTIC_COLUMNS}


def canonical_rows(args, episode_rows):
    mapped_action_dimension = TOPOLOGY[args.scale][0]
    denominator = args.max_episode_steps * mapped_action_dimension
    rows = []
    for episode in episode_rows:
        rows.append(
            {
                "row_type": "episode",
                "algorithm": args.algorithm,
                "seed": str(args.seed),
                "episode_index": episode["episode_index"],
                "episode_seed": episode["episode_seed"],
                "episode_steps": episode["episode_steps"],
                "done": episode["done"],
                "episode_reward": episode["episode_reward"],
                "tracking_error": episode["tracking_error"],
                "energy_tracking_error": episode["energy_tracking_error"],
                "power_tracker_violation": episode["power_tracker_violation"],
                "total_energy_charged": episode["total_energy_charged"],
                "total_energy_discharged": episode["total_energy_discharged"],
                "average_user_satisfaction": episode["average_user_satisfaction"],
                "energy_user_satisfaction": episode["energy_user_satisfaction"],
                "total_transformer_overload": episode["total_transformer_overload"],
                "total_ev_served": episode["total_ev_served"],
                "action_mean": episode["global_action_mean_all_slots"],
                "action_fraction_at_max": episode[
                    "global_action_fraction_at_max_all_slots"
                ],
                "active_action_count_mean": episode[
                    "nonzero_action_count_mean_all_slots"
                ],
                "mapped_action_dimension": str(mapped_action_dimension),
                "same_pass_at_max_count": str(denominator // 2),
                "total_action_decision_denominator": str(denominator),
            }
        )
    summary = {column: "" for column in CANONICAL_EVAL30_COLUMNS}
    summary.update({"row_type": "summary", "algorithm": args.algorithm, "seed": str(args.seed)})
    rows.append(summary)
    return rows


def seed_summary(args):
    row = {column: "0.0" for column in SEED_SUMMARY_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": args.matrix_job_id,
            "scale": args.scale,
            "algorithm": args.algorithm,
            "training_seed": str(args.seed),
            "n_eval_episodes": "30",
            "global_action_fraction_at_max_active_mean": "0.5",
            "global_action_fraction_at_max_all_slots_mean": "0.5",
            "global_action_nonzero_fraction_active_mean": "1.0",
            "active_action_decision_count_mean": "1.0",
            "active_action_below_environment_low_count": "0",
            "active_action_below_environment_low_fraction": "0.0",
            "active_action_above_environment_high_count": "0",
            "active_action_above_environment_high_fraction": "0.0",
            "global_positive_action_fraction_active_mean": "1.0",
            "global_zero_action_fraction_active_mean": "0.0",
            "global_negative_action_fraction_active_mean": "0.0",
            "environment_action_low": "-1.0",
            "environment_action_high": "1.0",
            "action_tolerance": "1e-06",
            "environment_action_domain_support": "signed",
            "v2g_enabled": "False",
            "v2g_enabled_source": "config",
            "power_tracker_violation_mean": "1.0",
            "tracking_error_mean": "1.0",
            "energy_tracking_error_mean": "1.0",
            "total_ev_served_mean": "1.0",
            "total_energy_charged_mean": "1.0",
            "average_user_satisfaction_mean": "1.0",
            "energy_user_satisfaction_mean": "100.0",
            "diagnostic_schema_version": "3",
        }
    )
    return {column: row[column] for column in SEED_SUMMARY_DIAGNOSTIC_COLUMNS}


def infrastructure_rows(args, count, label):
    columns = (
        TRANSFORMER_DIAGNOSTIC_COLUMNS
        if label == "transformer"
        else CHARGER_DIAGNOSTIC_COLUMNS
    )
    rows = []
    for episode_index in range(args.eval_episodes):
        for row_index in range(count):
            row = {column: "0.0" for column in columns}
            row.update(
                {
                    "matrix_job_id": args.matrix_job_id,
                    "scale": args.scale,
                    "algorithm": args.algorithm,
                    "training_seed": str(args.seed),
                    "episode_index": str(episode_index),
                    "episode_seed": str(
                        episode_seed(args.scale, args.seed, episode_index)
                    ),
                    f"{label}_id": str(row_index),
                    "transformer_id": str(row_index % TOPOLOGY[args.scale][1]),
                    "n_chargers_total": str(TOPOLOGY[args.scale][0]),
                    "n_active_chargers_seen": "1",
                    "n_ports": "1",
                    "n_active_ev_decisions": "1" if row_index == 0 else "0",
                    "n_all_slot_decisions": "1",
                    "action_sum_active": "1.0" if row_index == 0 else "0.0",
                    "action_mean_active": "1.0" if row_index == 0 else "0.0",
                    "action_max_active": "1.0" if row_index == 0 else "0.0",
                    "action_fraction_at_max_active": (
                        "1.0" if row_index == 0 else "0.0"
                    ),
                    "action_nonzero_fraction_active": (
                        "1.0" if row_index == 0 else "0.0"
                    ),
                    "positive_action_fraction_active": (
                        "1.0" if row_index == 0 else "0.0"
                    ),
                    "zero_action_fraction_active": (
                        "0.0" if row_index == 0 else "1.0"
                    ),
                    "negative_action_fraction_active": "0.0",
                    "action_fraction_at_positive_max_active": (
                        "1.0" if row_index == 0 else "0.0"
                    ),
                    "action_fraction_at_negative_min_active": "0.0",
                    "positive_action_sum_active": "1.0" if row_index == 0 else "0.0",
                    "negative_action_magnitude_sum_active": "0.0",
                    "action_sum_all_slots": "1.0" if row_index == 0 else "0.0",
                    "action_mean_all_slots": "1.0" if row_index == 0 else "0.0",
                    "action_max_all_slots": "1.0" if row_index == 0 else "0.0",
                    "action_fraction_at_max_all_slots": (
                        "1.0" if row_index == 0 else "0.0"
                    ),
                    "overload_magnitude_sum": "0.0",
                    "overload_magnitude_max": "0.0",
                    "overload_frequency_steps": "0",
                    "overload_frequency_fraction": "0.0",
                    "cs_power_sum_kwh": "1.0" if row_index == 0 else "0.0",
                    "cs_power_mean_kw": "1.0" if row_index == 0 else "0.0",
                    "cs_power_max_kw": "1.0" if row_index == 0 else "0.0",
                    "served_ev_count": "1" if row_index == 0 else "0",
                    "energy_charged_kwh": "1.0" if row_index == 0 else "0.0",
                    "energy_discharged_kwh": "0.0",
                    "user_satisfaction_sum": "1.0" if row_index == 0 else "0.0",
                    "user_satisfaction_mean": "1.0" if row_index == 0 else "0.0",
                    "user_satisfaction_mean_served_ev_weighted": (
                        "1.0" if row_index == 0 else "0.0"
                    ),
                    "user_satisfaction_observation_count": (
                        "1" if row_index == 0 else "0"
                    ),
                    "user_satisfaction_source": "synthetic",
                    "diagnostic_schema_version": "3",
                }
            )
            rows.append({column: row[column] for column in columns})
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", required=True)
parser.add_argument("--scale", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--seed", required=True, type=int)
parser.add_argument("--eval_episodes", required=True, type=int)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--device", required=True)
parser.add_argument("--max_episode_steps", required=True, type=int)
parser.add_argument("--deterministic", required=True)
parser.add_argument("--eval_expl_noise", required=True)
parser.add_argument("--eval_seed_offset", required=True, type=int)
parser.add_argument("--matrix_job_id", required=True)
args = parser.parse_args()

if args.eval_episodes != 30:
    raise SystemExit("expected 30 episodes")
if args.device != "cpu" or args.deterministic != "true" or args.eval_expl_noise != "0.0":
    raise SystemExit("diagnostic execution contract mismatch")
if "model.best" not in args.checkpoint or "model.last" in args.checkpoint:
    raise SystemExit("checkpoint contract mismatch")
if args.max_episode_steps != 112:
    raise SystemExit("simulation length mismatch")

output = Path(args.output_dir)
episode_rows = [episode_row(args, episode_index) for episode_index in range(30)]
charger_count, transformer_count = TOPOLOGY[args.scale]
write_csv(output / "episode_diagnostics.csv", EPISODE_DIAGNOSTIC_COLUMNS, episode_rows)
write_csv(
    output / "same_pass_canonical_eval30.csv",
    CANONICAL_EVAL30_COLUMNS,
    canonical_rows(args, episode_rows),
)
write_csv(
    output / "seed_summary_diagnostics.csv",
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    [seed_summary(args)],
)
write_csv(
    output / "transformer_diagnostics.csv",
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
    infrastructure_rows(args, transformer_count, "transformer"),
)
write_csv(
    output / "charger_diagnostics.csv",
    CHARGER_DIAGNOSTIC_COLUMNS,
    infrastructure_rows(args, charger_count, "charger"),
)
