import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

from TD3.TD3_ActionGNN_25cp import TD3_ActionGNN
from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN
from train_RL_GNN_25cp import make_env, normalise_step_result, reset_env, resolve_device
from utils.state_25cp import PublicPST_GNN


ALGORITHM_CLASSES = {
    "baseline_25cp": TD3_ActionGNN,
    "hierarchical_25cp": TD3_HierarchicalActionGNN,
}

ACTION_DIAGNOSTIC_COLUMNS = [
    "action_mean",
    "action_std",
    "action_min",
    "action_max",
    "action_fraction_zero",
    "action_fraction_at_max",
    "active_action_count_mean",
]

REQUIRED_COLUMNS = [
    "run_name",
    "algorithm",
    "config",
    "seed",
    "episode_seed",
    "checkpoint",
    "episode_index",
    "episode_reward",
    "episode_steps",
    "done",
    "mean_reward",
    "std_reward",
    *ACTION_DIAGNOSTIC_COLUMNS,
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalised_value = value.lower()
    if normalised_value in {"true", "1", "yes", "y"}:
        return True
    if normalised_value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def normalise_checkpoint_prefix(checkpoint):
    checkpoint_text = str(checkpoint)
    for checkpoint_suffix in ["_actor", "_actor_optimizer", "_critic", "_critic_optimizer"]:
        if checkpoint_text.endswith(checkpoint_suffix):
            checkpoint_text = checkpoint_text[: -len(checkpoint_suffix)]
            break
    return Path(checkpoint_text)


def load_checkpoint_kwargs(checkpoint_prefix):
    kwargs_path = checkpoint_prefix.parent / "kwargs.yaml"
    if not kwargs_path.exists():
        return {}
    with kwargs_path.open("r") as kwargs_file:
        checkpoint_kwargs = yaml.load(kwargs_file, Loader=yaml.FullLoader)
    return checkpoint_kwargs or {}


def create_policy(
    algorithm,
    action_dim,
    max_action,
    device="cpu",
    checkpoint_kwargs=None,
):
    if algorithm not in ALGORITHM_CLASSES:
        supported_algorithms = ", ".join(sorted(ALGORITHM_CLASSES))
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose one of: {supported_algorithms}.")

    policy_kwargs = dict(checkpoint_kwargs or {})
    policy_kwargs.update({
        "action_dim": action_dim,
        "max_action": max_action,
        "fx_node_sizes": PublicPST_GNN.node_sizes,
        "device": device,
    })
    return ALGORITHM_CLASSES[algorithm](**policy_kwargs)


def load_policy_checkpoint(policy, checkpoint_prefix):
    try:
        policy.load(str(checkpoint_prefix))
    except Exception as checkpoint_error:
        raise RuntimeError(
            f"Failed to load checkpoint prefix '{checkpoint_prefix}': {checkpoint_error}"
        ) from checkpoint_error


def reset_env_state(env, seed):
    reset_result = reset_env(env, seed=seed)
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


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


def action_diagnostics_from_actions(mapped_actions, max_action, tolerance=1e-6):
    if not mapped_actions:
        return {
            action_diagnostic_key: 0.0
            for action_diagnostic_key in ACTION_DIAGNOSTIC_COLUMNS
        }

    stacked_mapped_actions = np.asarray(mapped_actions, dtype=np.float32)
    if stacked_mapped_actions.ndim == 1:
        stacked_mapped_actions = stacked_mapped_actions.reshape(1, -1)

    zero_action_mask = np.abs(stacked_mapped_actions) <= tolerance
    at_max_action_mask = stacked_mapped_actions >= (float(max_action) - tolerance)
    active_action_counts = np.count_nonzero(~zero_action_mask, axis=1)

    return {
        "action_mean": float(np.mean(stacked_mapped_actions)),
        "action_std": float(np.std(stacked_mapped_actions)),
        "action_min": float(np.min(stacked_mapped_actions)),
        "action_max": float(np.max(stacked_mapped_actions)),
        "action_fraction_zero": float(np.mean(zero_action_mask)),
        "action_fraction_at_max": float(np.mean(at_max_action_mask)),
        "active_action_count_mean": float(np.mean(active_action_counts)),
    }


def evaluate_episode(
    policy,
    env,
    seed,
    max_action,
    max_episode_steps=None,
    deterministic=True,
    eval_expl_noise=0.0,
):
    state, reset_info = reset_env_state(env, seed=seed)
    done = False
    episode_reward = 0.0
    episode_steps = 0
    stats = {}
    mapped_actions = []

    while not done:
        mapped_action = select_mapped_action(
            policy=policy,
            state=state,
            deterministic=deterministic,
            eval_expl_noise=eval_expl_noise,
        )
        mapped_action_numpy = np.asarray(mapped_action, dtype=np.float32)
        mapped_actions.append(mapped_action_numpy)
        state, reward, done, stats = normalise_step_result(env.step(mapped_action_numpy))
        episode_reward += float(reward)
        episode_steps += 1

        if max_episode_steps is not None and episode_steps >= max_episode_steps:
            done = True

    episode_record = {
        "episode_reward": episode_reward,
        "episode_steps": episode_steps,
        "done": done,
        "stats": stats,
        "reset_info": reset_info,
    }
    episode_record.update(action_diagnostics_from_actions(mapped_actions, max_action=max_action))
    return episode_record


def scalar_stats(stats):
    return {
        stat_key: stat_value
        for stat_key, stat_value in stats.items()
        if np.isscalar(stat_value)
    }


def build_csv_rows(metadata, episode_records):
    rewards = [float(episode_record["episode_reward"]) for episode_record in episode_records]
    mean_reward = float(np.mean(rewards)) if rewards else float("nan")
    std_reward = float(np.std(rewards)) if rewards else float("nan")

    scalar_stat_keys = sorted({
        stat_key
        for episode_record in episode_records
        for stat_key in scalar_stats(episode_record.get("stats", {}))
    })
    fieldnames = REQUIRED_COLUMNS + [
        "row_type",
        *[stat_key for stat_key in scalar_stat_keys if stat_key not in REQUIRED_COLUMNS],
    ]

    rows = []
    for episode_record in episode_records:
        row = dict(metadata)
        row.update({
            "row_type": "episode",
            "episode_index": episode_record["episode_index"],
            "episode_seed": episode_record["episode_seed"],
            "episode_reward": float(episode_record["episode_reward"]),
            "episode_steps": int(episode_record["episode_steps"]),
            "done": bool(episode_record["done"]),
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        })
        for action_diagnostic_key in ACTION_DIAGNOSTIC_COLUMNS:
            row[action_diagnostic_key] = episode_record.get(action_diagnostic_key, "")
        row.update(scalar_stats(episode_record.get("stats", {})))
        rows.append(row)

    summary_row = dict(metadata)
    summary_row.update({
        "row_type": "summary",
        "episode_index": "summary",
        "episode_seed": "",
        "episode_reward": "",
        "episode_steps": sum(int(episode_record["episode_steps"]) for episode_record in episode_records),
        "done": all(bool(episode_record["done"]) for episode_record in episode_records),
        "mean_reward": mean_reward,
        "std_reward": std_reward,
    })
    for action_diagnostic_key in ACTION_DIAGNOSTIC_COLUMNS:
        action_diagnostic_values = [
            float(episode_record[action_diagnostic_key])
            for episode_record in episode_records
            if action_diagnostic_key in episode_record
        ]
        summary_row[action_diagnostic_key] = (
            float(np.mean(action_diagnostic_values)) if action_diagnostic_values else ""
        )
    rows.append(summary_row)

    return rows, fieldnames


def write_csv(output_csv, rows, fieldnames):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Controlled evaluator for baseline and hierarchical 25 CP TD3 ActionGNN."
    )
    parser.add_argument("--algorithm", required=True, choices=sorted(ALGORITHM_CLASSES))
    parser.add_argument("--config", default="./config_files/PublicPST_25cp.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--run_name", default="controlled_eval")
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--deterministic", type=str2bool, default=True)
    parser.add_argument("--eval_expl_noise", type=float, default=0.0)
    parser.add_argument("--eval_seed_offset", type=int, default=100000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_prefix = normalise_checkpoint_prefix(args.checkpoint)
    checkpoint_kwargs = load_checkpoint_kwargs(checkpoint_prefix)

    probe_env = make_env(args.config, seed=args.seed)
    action_dim = probe_env.action_space.shape[0]
    max_action = float(probe_env.action_space.high[0])

    policy = create_policy(
        algorithm=args.algorithm,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        checkpoint_kwargs=checkpoint_kwargs,
    )
    load_policy_checkpoint(policy, checkpoint_prefix)

    episode_records = []
    for episode_index in range(args.eval_episodes):
        episode_seed = args.seed + args.eval_seed_offset + episode_index
        env = make_env(args.config, seed=episode_seed)
        episode_record = evaluate_episode(
            policy=policy,
            env=env,
            seed=episode_seed,
            max_action=max_action,
            max_episode_steps=args.max_episode_steps,
            deterministic=args.deterministic,
            eval_expl_noise=args.eval_expl_noise,
        )
        episode_record["episode_index"] = episode_index
        episode_record["episode_seed"] = episode_seed
        episode_records.append(episode_record)

    metadata = {
        "run_name": args.run_name,
        "algorithm": args.algorithm,
        "config": args.config,
        "seed": args.seed,
        "checkpoint": str(checkpoint_prefix),
    }
    rows, fieldnames = build_csv_rows(metadata, episode_records)
    output_path = write_csv(args.output_csv, rows, fieldnames)

    rewards = [episode_record["episode_reward"] for episode_record in episode_records]
    print("---------------------------------------")
    print(f"Run name: {args.run_name}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Checkpoint: {checkpoint_prefix}")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Mean reward: {float(np.mean(rewards)):.3f}")
    print(f"Std reward: {float(np.std(rewards)):.3f}")
    print(f"Output CSV: {output_path}")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
