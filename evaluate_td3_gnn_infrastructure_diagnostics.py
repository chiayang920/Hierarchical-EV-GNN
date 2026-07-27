import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from evaluate_td3_gnn import (
    ALGORITHM_CHOICES,
    create_policy,
    load_checkpoint_kwargs,
    load_policy_checkpoint,
    make_env,
    normalise_algorithm_label,
    normalise_checkpoint_prefix,
    normalise_step_result,
    reset_env_state,
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
    extract_active_action_slots,
    validate_active_infrastructure_mapping,
    validate_environment_action_bounds,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only evaluator for per-infrastructure action saturation. "
            "It preserves the canonical eval30 CSV schema by writing separate outputs."
        )
    )
    parser.add_argument("--algorithm", required=True, choices=ALGORITHM_CHOICES)
    parser.add_argument(
        "--config",
        default="./config_files/PublicPST_25cp.yaml",
        help="EV2Gym config file; CP scale is inferred from this path for diagnostics metadata.",
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
        validate_active_infrastructure_mapping(
            state=state,
            slot_to_charger_id=slot_to_charger_id,
            charger_to_transformer_id=charger_to_transformer_id,
        )
        active_slots_by_step.append(extract_active_action_slots(state))
        mapped_action = select_mapped_action(
            policy=policy,
            state=state,
            deterministic=deterministic,
            eval_expl_noise=eval_expl_noise,
        )
        mapped_action_numpy = np.asarray(mapped_action, dtype=np.float32).reshape(-1)
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

    return {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "done": done,
        "stats": stats,
        "reset_info": reset_info,
        "action_summary": action_summary,
    }


def infer_scale_label(config_path):
    config_name = Path(config_path).stem.lower()
    if "1000" in config_name:
        return "1000cp"
    if "500" in config_name:
        return "500cp"
    if "100" in config_name:
        return "100cp"
    if "25cp" in config_name or config_name.endswith("25"):
        return "25cp"
    return config_name


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

    policy = create_policy(
        algorithm=canonical_algorithm,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        checkpoint_kwargs=checkpoint_kwargs,
    )
    load_policy_checkpoint(policy, checkpoint_prefix)

    metadata = {
        "matrix_job_id": args.matrix_job_id,
        "scale": infer_scale_label(args.config),
        "algorithm": canonical_algorithm,
        "training_seed": args.seed,
        "config": args.config,
        "checkpoint_prefix": str(checkpoint_prefix),
        "run_name": args.run_name,
        **v2g_metadata,
    }
    summary_metadata = {
        "matrix_job_id": args.matrix_job_id,
        "scale": infer_scale_label(args.config),
        "algorithm": canonical_algorithm,
        "training_seed": args.seed,
    }

    episode_rows = []
    charger_rows = []
    transformer_rows = []
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
        episode_rows.append(episode_row)
        charger_rows.extend(build_charger_rows(metadata, episode_index, episode_seed, action_summary))
        transformer_rows.extend(build_transformer_rows(metadata, episode_index, episode_seed, action_summary))

    seed_summary_rows = [
        build_seed_summary_row(summary_metadata, episode_rows)
    ]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "episode_diagnostics.csv", EPISODE_DIAGNOSTIC_COLUMNS, episode_rows)
    write_csv(output_dir / "transformer_diagnostics.csv", TRANSFORMER_DIAGNOSTIC_COLUMNS, transformer_rows)
    write_csv(output_dir / "charger_diagnostics.csv", CHARGER_DIAGNOSTIC_COLUMNS, charger_rows)
    write_csv(output_dir / "seed_summary_diagnostics.csv", SEED_SUMMARY_DIAGNOSTIC_COLUMNS, seed_summary_rows)

    print("---------------------------------------")
    print(f"Run name: {args.run_name}")
    print(f"Algorithm: {canonical_algorithm}")
    print(f"Checkpoint: {checkpoint_prefix}")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Output dir: {output_dir}")
    print("---------------------------------------")

if __name__ == "__main__":
    main()
