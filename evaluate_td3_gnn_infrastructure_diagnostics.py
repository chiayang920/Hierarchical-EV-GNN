import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from evaluate_td3_gnn import (
    ALGORITHM_CHOICES,
    action_diagnostics_from_actions,
    build_csv_rows as build_canonical_csv_rows,
    create_policy,
    load_checkpoint_kwargs,
    load_policy_checkpoint,
    make_env,
    normalise_algorithm_label,
    normalise_checkpoint_prefix,
    normalise_step_result,
    reset_env_state,
    validate_checkpoint_identity,
)
from utils.ev2gym_training_utils import resolve_device, str2bool
from utils.infrastructure_diagnostics import (
    CHARGER_DIAGNOSTIC_COLUMNS,
    EPISODE_DIAGNOSTIC_COLUMNS,
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
    aggregate_infrastructure_actions,
    build_charger_rows,
    build_charger_to_transformer_id,
    build_episode_row,
    build_seed_summary_row,
    build_slot_to_charger_id,
    build_transformer_rows,
    validate_active_infrastructure_mapping,
    validate_diagnostic_action_contract,
    validate_diagnostic_reconciliation,
    validate_environment_action_bounds,
)


SCALE_CHOICES = ("25cp", "100cp", "500cp", "1000cp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only evaluator for per-infrastructure action saturation. "
            "It preserves the canonical eval30 CSV schema by writing separate outputs."
        )
    )
    parser.add_argument("--algorithm", required=True, choices=ALGORITHM_CHOICES)
    parser.add_argument("--scale", required=True, choices=SCALE_CHOICES)
    parser.add_argument(
        "--config",
        required=True,
        help="Explicit EV2Gym config file.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="infrastructure_diagnostics")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--eval_expl_noise", type=float, default=0.0)
    parser.add_argument("--eval_seed_offset", type=int, default=100000)
    parser.add_argument("--matrix_job_id", default="")
    parser.add_argument("--max_action_tolerance", type=float, default=1e-6)
    return parser.parse_args(argv)


def select_mapped_action(policy, state, deterministic=True, eval_expl_noise=0.0):
    exploration_noise = 0.0 if deterministic else float(eval_expl_noise)
    selected_action = policy.select_action(
        state,
        expl_noise=exploration_noise,
        return_mapped_action=True,
    )
    if isinstance(selected_action, tuple):
        return selected_action[0]
    return selected_action


def canonical_overlapping_action_metrics(
    mapped_actions_by_step: list[np.ndarray],
    max_action: float,
    tolerance: float,
) -> dict[str, float | int]:
    stacked_actions = np.asarray(mapped_actions_by_step, dtype=np.float32)
    if stacked_actions.size == 0:
        return {
            **action_diagnostics_from_actions(
                [],
                max_action=max_action,
                tolerance=tolerance,
            ),
            "mapped_action_dimension": 0,
            "same_pass_at_max_count": 0,
            "total_action_decision_denominator": 0,
        }
    if stacked_actions.ndim == 1:
        stacked_actions = stacked_actions.reshape(1, -1)
    at_max_mask = stacked_actions >= (float(max_action) - float(tolerance))
    action_metrics = action_diagnostics_from_actions(
        [row for row in stacked_actions],
        max_action=max_action,
        tolerance=tolerance,
    )
    return {
        **action_metrics,
        "mapped_action_dimension": int(stacked_actions.shape[1]),
        "same_pass_at_max_count": int(np.count_nonzero(at_max_mask)),
        "total_action_decision_denominator": int(
            stacked_actions.shape[0] * stacked_actions.shape[1]
        ),
    }


def inject_canonical_overlapping_action_metrics(
    action_summary,
    overlapping_metrics,
) -> None:
    global_summary = action_summary["global"]
    global_summary["action_mean_all_slots"] = overlapping_metrics["action_mean"]
    global_summary["action_fraction_at_max_all_slots"] = overlapping_metrics[
        "action_fraction_at_max"
    ]
    global_summary["nonzero_action_count_mean_all_slots"] = overlapping_metrics[
        "active_action_count_mean"
    ]


def build_same_pass_canonical_episode_record(
    episode_record,
    overlapping_action_metrics,
):
    canonical_stats = dict(episode_record.get("stats", {}))
    canonical_stats.update(
        {
            "mapped_action_dimension": overlapping_action_metrics[
                "mapped_action_dimension"
            ],
            "same_pass_at_max_count": overlapping_action_metrics[
                "same_pass_at_max_count"
            ],
            "total_action_decision_denominator": overlapping_action_metrics[
                "total_action_decision_denominator"
            ],
        }
    )
    canonical_episode = {
        "episode_reward": episode_record["episode_reward"],
        "episode_steps": episode_record["episode_steps"],
        "done": episode_record["done"],
        "stats": canonical_stats,
        "reset_info": episode_record.get("reset_info", {}),
    }
    canonical_episode.update(
        {
            action_metric_key: overlapping_action_metrics[action_metric_key]
            for action_metric_key in (
                "action_mean",
                "action_std",
                "action_min",
                "action_max",
                "action_fraction_zero",
                "action_fraction_at_max",
                "active_action_count_mean",
            )
        }
    )
    return canonical_episode


def write_same_pass_canonical_eval30(output_dir, metadata, episode_records):
    rows, fieldnames = build_canonical_csv_rows(metadata, episode_records)
    output_path = Path(output_dir) / "same_pass_canonical_eval30.csv"
    write_csv(output_path, fieldnames, rows)
    return output_path


def evaluate_diagnostic_episode(
    policy,
    env,
    seed,
    max_action,
    max_action_tolerance,
    max_episode_steps=None,
    deterministic=True,
    eval_expl_noise=0.0,
):
    state, reset_info = reset_env_state(env, seed=seed)
    slot_to_charger_id = build_slot_to_charger_id(env)
    charger_to_transformer_id = build_charger_to_transformer_id(env)
    done = False
    episode_reward = 0.0
    episode_steps = 0
    stats = {}
    mapped_actions_by_step = []
    active_slots_by_step = []

    while not done:
        active_metadata = validate_active_infrastructure_mapping(
            state=state,
            slot_to_charger_id=slot_to_charger_id,
            charger_to_transformer_id=charger_to_transformer_id,
        )
        mapped_action = select_mapped_action(
            policy=policy,
            state=state,
            deterministic=deterministic,
            eval_expl_noise=eval_expl_noise,
        )
        mapped_action_numpy = np.asarray(mapped_action, dtype=np.float32).reshape(-1)
        active_slots = validate_diagnostic_action_contract(
            mapped_action=mapped_action_numpy,
            active_slots=active_metadata["active_slots"],
            action_dim=slot_to_charger_id.size,
            tolerance=max_action_tolerance,
        )
        active_slots_by_step.append(active_slots)
        mapped_actions_by_step.append(mapped_action_numpy)
        state, reward, done, stats = normalise_step_result(env.step(mapped_action_numpy))
        episode_reward += float(reward)
        episode_steps += 1

        if max_episode_steps is not None and episode_steps >= max_episode_steps:
            done = True

    action_summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=mapped_actions_by_step,
        active_slots_by_step=active_slots_by_step,
        slot_to_charger_id=slot_to_charger_id,
        charger_to_transformer_id=charger_to_transformer_id,
        max_action=max_action,
        tolerance=max_action_tolerance,
        env=env,
    )
    overlapping_action_metrics = canonical_overlapping_action_metrics(
        mapped_actions_by_step,
        max_action=max_action,
        tolerance=max_action_tolerance,
    )
    inject_canonical_overlapping_action_metrics(
        action_summary,
        overlapping_action_metrics,
    )

    episode_record = {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "done": done,
        "stats": stats,
        "reset_info": reset_info,
        "action_summary": action_summary,
        "mapped_actions_by_step": mapped_actions_by_step,
        "mapped_action_dimension": int(slot_to_charger_id.size),
        "same_pass_overlapping_action_metrics": overlapping_action_metrics,
    }
    episode_record["same_pass_canonical_episode_record"] = (
        build_same_pass_canonical_episode_record(
            episode_record,
            overlapping_action_metrics=overlapping_action_metrics,
        )
    )
    return episode_record


def validate_config_scale_contract(config_path, scale):
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"config path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"config path must be a readable regular file: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"config must contain valid YAML: {path}") from exc
    except OSError as exc:
        raise ValueError(f"config path must be a readable regular file: {path}") from exc

    if not isinstance(config, dict):
        raise ValueError("config top-level YAML value must be a mapping")
    if "number_of_charging_stations" not in config:
        raise ValueError("config is missing number_of_charging_stations")

    station_count = config["number_of_charging_stations"]
    if type(station_count) is not int:
        raise ValueError("number_of_charging_stations must be an exact integer")
    if station_count <= 0:
        raise ValueError("number_of_charging_stations must be positive")
    if scale not in SCALE_CHOICES:
        raise ValueError(f"unsupported explicit scale: {scale}")

    expected_count = int(scale.removesuffix("cp"))
    if station_count != expected_count:
        raise ValueError(
            "number_of_charging_stations mismatch: "
            f"config={station_count}, explicit scale={scale}, expected={expected_count}"
        )
    return config


def write_csv(output_path, fieldnames, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_v2g_metadata(env, config_path):
    if env is not None and hasattr(env, "v2g_enabled"):
        v2g_value = _normalise_bool(getattr(env, "v2g_enabled"))
        if v2g_value is not None:
            return {
                "v2g_enabled": v2g_value,
                "v2g_enabled_source": "env.v2g_enabled",
            }

    config_value = _load_config_v2g_enabled(config_path)
    if config_value is not None:
        return {
            "v2g_enabled": config_value,
            "v2g_enabled_source": "config:v2g_enabled",
        }

    return {
        "v2g_enabled": "",
        "v2g_enabled_source": "unavailable",
    }


def _load_config_v2g_enabled(config_path):
    try:
        with Path(config_path).open("r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(config, dict) or "v2g_enabled" not in config:
        return None
    return _normalise_bool(config["v2g_enabled"])


def _normalise_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "yes", "1"}:
            return True
        if cleaned in {"false", "no", "0"}:
            return False
    return None


def main(argv=None):
    args = parse_args(argv)
    validate_config_scale_contract(args.config, args.scale)
    canonical_algorithm = normalise_algorithm_label(args.algorithm)
    device = resolve_device(args.device)
    checkpoint_prefix = normalise_checkpoint_prefix(args.checkpoint)
    checkpoint_kwargs = load_checkpoint_kwargs(checkpoint_prefix)

    probe_env = make_env(args.config, seed=args.seed)
    action_dim = probe_env.action_space.shape[0]
    action_bounds = validate_environment_action_bounds(
        probe_env.action_space,
        tolerance=args.max_action_tolerance,
    )
    max_action = float(action_bounds["environment_action_high"])
    v2g_metadata = resolve_v2g_metadata(probe_env, args.config)

    validate_checkpoint_identity(checkpoint_prefix, canonical_algorithm)
    policy = create_policy(
        algorithm=canonical_algorithm,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        checkpoint_kwargs=checkpoint_kwargs,
    )
    load_policy_checkpoint(policy, checkpoint_prefix, canonical_algorithm)

    metadata = {
        "matrix_job_id": args.matrix_job_id,
        "scale": args.scale,
        "algorithm": canonical_algorithm,
        "training_seed": args.seed,
        "config": args.config,
        "checkpoint_prefix": str(checkpoint_prefix),
        "run_name": args.run_name,
        **v2g_metadata,
    }
    summary_metadata = {
        "matrix_job_id": args.matrix_job_id,
        "scale": args.scale,
        "algorithm": canonical_algorithm,
        "training_seed": args.seed,
    }

    episode_rows = []
    charger_rows = []
    transformer_rows = []
    same_pass_canonical_records = []
    for episode_index in range(args.eval_episodes):
        episode_seed = args.seed + args.eval_seed_offset + episode_index
        env = make_env(args.config, seed=episode_seed)
        episode_record = evaluate_diagnostic_episode(
            policy=policy,
            env=env,
            seed=episode_seed,
            max_action=max_action,
            max_action_tolerance=args.max_action_tolerance,
            max_episode_steps=args.max_episode_steps,
            deterministic=args.deterministic,
            eval_expl_noise=args.eval_expl_noise,
        )
        same_pass_record = dict(episode_record["same_pass_canonical_episode_record"])
        same_pass_record["episode_index"] = episode_index
        same_pass_record["episode_seed"] = episode_seed
        same_pass_canonical_records.append(same_pass_record)
        action_summary = episode_record["action_summary"]
        episode_row = build_episode_row(
            metadata=metadata,
            episode_index=episode_index,
            episode_seed=episode_seed,
            episode_record=episode_record,
            action_summary=action_summary,
            stats=episode_record.get("stats", {}),
            max_action=max_action,
            tolerance=args.max_action_tolerance,
        )
        episode_charger_rows = build_charger_rows(
            metadata,
            episode_index,
            episode_seed,
            action_summary,
        )
        episode_transformer_rows = build_transformer_rows(
            metadata,
            episode_index,
            episode_seed,
            action_summary,
        )
        validate_diagnostic_reconciliation(
            episode_row=episode_row,
            charger_rows=episode_charger_rows,
            transformer_rows=episode_transformer_rows,
            tolerance=args.max_action_tolerance,
        )
        episode_rows.append(episode_row)
        charger_rows.extend(episode_charger_rows)
        transformer_rows.extend(episode_transformer_rows)

    seed_summary_rows = [
        build_seed_summary_row(summary_metadata, episode_rows)
    ]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "episode_diagnostics.csv", EPISODE_DIAGNOSTIC_COLUMNS, episode_rows)
    write_csv(output_dir / "transformer_diagnostics.csv", TRANSFORMER_DIAGNOSTIC_COLUMNS, transformer_rows)
    write_csv(output_dir / "charger_diagnostics.csv", CHARGER_DIAGNOSTIC_COLUMNS, charger_rows)
    write_csv(output_dir / "seed_summary_diagnostics.csv", SEED_SUMMARY_DIAGNOSTIC_COLUMNS, seed_summary_rows)
    write_same_pass_canonical_eval30(
        output_dir,
        {
            "run_name": args.run_name,
            "algorithm": canonical_algorithm,
            "config": args.config,
            "seed": args.seed,
            "checkpoint": str(checkpoint_prefix),
        },
        same_pass_canonical_records,
    )

    print("---------------------------------------")
    print(f"Run name: {args.run_name}")
    print(f"Algorithm: {canonical_algorithm}")
    print(f"Checkpoint: {checkpoint_prefix}")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Output dir: {output_dir}")
    print("---------------------------------------")

if __name__ == "__main__":
    main()
