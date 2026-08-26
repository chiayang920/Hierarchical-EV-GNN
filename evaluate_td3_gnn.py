import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import yaml

from utils.ev2gym_training_utils import make_env, normalise_step_result, reset_env, resolve_device, str2bool


ALGORITHM_CHOICES = (
    "actiongnn",
    "actiongnn_nonnegative",
    "hierarchical",
    "hierarchical_transformer_constraint",
    "hierarchical_transformer_ev",
)

LEGACY_ALGORITHM_ALIASES = {
    "baseline_25cp": "actiongnn",
    "hierarchical_25cp": "hierarchical",
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


def normalise_algorithm_label(algorithm):
    if algorithm in LEGACY_ALGORITHM_ALIASES:
        canonical_algorithm = LEGACY_ALGORITHM_ALIASES[algorithm]
        warnings.warn(
            f"Algorithm label '{algorithm}' is deprecated; use '{canonical_algorithm}' for new runs.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical_algorithm
    return algorithm


def get_policy_class(algorithm):
    if algorithm == "actiongnn":
        from TD3.TD3_ActionGNN_Controlled import TD3_ActionGNN

        return TD3_ActionGNN
    if algorithm == "actiongnn_nonnegative":
        from TD3.TD3_ActionGNN_NonNegative import TD3_ActionGNN_NonNegative

        return TD3_ActionGNN_NonNegative
    if algorithm == "hierarchical":
        from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN

        return TD3_HierarchicalActionGNN
    if algorithm == "hierarchical_transformer_constraint":
        from TD3.TD3_HierarchicalActionGNN_TransformerConstraint import (
            TD3_HierarchicalActionGNN_TransformerConstraint,
        )

        return TD3_HierarchicalActionGNN_TransformerConstraint
    if algorithm == "hierarchical_transformer_ev":
        from TD3.TD3_TransformerEVActionGNN import TD3_TransformerEVActionGNN

        return TD3_TransformerEVActionGNN
    raise ValueError(f"Unsupported algorithm: {algorithm}")


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


def _load_yaml_mapping(yaml_path):
    if not yaml_path.exists():
        return None
    with yaml_path.open("r") as yaml_file:
        yaml_payload = yaml.load(yaml_file, Loader=yaml.FullLoader)
    if yaml_payload is None:
        return {}
    if not isinstance(yaml_payload, dict):
        raise ValueError(f"Expected YAML mapping at {yaml_path}; got {type(yaml_payload).__name__}.")
    return yaml_payload


def _expected_corrected_metadata(checkpoint_role):
    from TD3.TD3_ActionGNN_NonNegative import (
        ACTION_DOMAIN_CONTRACT,
        ACTOR_OUTPUT_TRANSFORM,
        ACTOR_OUTPUT_TRANSFORM_FORMULA,
        CANONICAL_ALGORITHM_LABEL,
        CHECKPOINT_METADATA_SCHEMA,
        CHECKPOINT_SELECTION_RULE,
        NON_EV_ACTION,
        SUPPORTED_TRAINING_PROTOCOLS,
    )

    expected = {
        "metadata_schema": CHECKPOINT_METADATA_SCHEMA,
        "algorithm": CANONICAL_ALGORITHM_LABEL,
        "action_domain_contract": ACTION_DOMAIN_CONTRACT,
        "actor_output_transform": ACTOR_OUTPUT_TRANSFORM,
        "actor_output_transform_formula": ACTOR_OUTPUT_TRANSFORM_FORMULA,
        "non_ev_action": NON_EV_ACTION,
        "discrete_actions": 1,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "checkpoint_role": checkpoint_role,
    }
    expected["supported_training_protocols"] = SUPPORTED_TRAINING_PROTOCOLS
    return expected


def _checkpoint_role_from_prefix(checkpoint_prefix):
    checkpoint_name = Path(checkpoint_prefix).name
    if checkpoint_name == "model.best":
        return "best"
    if checkpoint_name == "model.last":
        return "last"
    raise ValueError(
        "corrected checkpoint metadata requires checkpoint prefix named model.best or model.last"
    )


def _validate_corrected_metadata(checkpoint_prefix, metadata):
    expected_checkpoint_role = _checkpoint_role_from_prefix(checkpoint_prefix)
    expected_metadata = _expected_corrected_metadata(expected_checkpoint_role)
    supported_training_protocols = expected_metadata.pop("supported_training_protocols")
    for metadata_key, expected_value in expected_metadata.items():
        actual_value = metadata.get(metadata_key)
        if actual_value != expected_value:
            raise ValueError(
                f"corrected checkpoint metadata {metadata_key} mismatch: "
                f"expected {expected_value!r}; got {actual_value!r}"
            )
    training_budget = metadata.get("training_budget")
    if training_budget not in supported_training_protocols:
        raise ValueError(
            "corrected checkpoint metadata training_budget mismatch: "
            f"expected one of {tuple(supported_training_protocols)!r}; got {training_budget!r}"
        )
    for metadata_key, expected_value in supported_training_protocols[training_budget].items():
        actual_value = metadata.get(metadata_key)
        if actual_value != expected_value:
            raise ValueError(
                f"corrected checkpoint metadata {metadata_key} mismatch: "
                f"expected {expected_value!r}; got {actual_value!r}"
            )


def _expected_transformer_ev_metadata(checkpoint_role):
    from TD3.TD3_TransformerEVActionGNN import (
        ACTION_DOMAIN_CONTRACT,
        ACTOR_OUTPUT_TRANSFORM,
        ACTOR_OUTPUT_TRANSFORM_FORMULA,
        CANONICAL_ALGORITHM_LABEL,
        CHECKPOINT_METADATA_SCHEMA,
        CHECKPOINT_SELECTION_RULE,
        NON_EV_ACTION,
    )

    return {
        "metadata_schema": CHECKPOINT_METADATA_SCHEMA,
        "algorithm": CANONICAL_ALGORITHM_LABEL,
        "action_domain_contract": ACTION_DOMAIN_CONTRACT,
        "actor_output_transform": ACTOR_OUTPUT_TRANSFORM,
        "actor_output_transform_formula": ACTOR_OUTPUT_TRANSFORM_FORMULA,
        "non_ev_action": NON_EV_ACTION,
        "discrete_actions": 1,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "checkpoint_role": checkpoint_role,
    }


def _validate_transformer_ev_metadata(checkpoint_prefix, metadata, run_args=None):
    expected_checkpoint_role = _checkpoint_role_from_prefix(checkpoint_prefix)
    expected_metadata = _expected_transformer_ev_metadata(expected_checkpoint_role)
    for metadata_key, expected_value in expected_metadata.items():
        actual_value = metadata.get(metadata_key)
        if actual_value != expected_value:
            raise ValueError(
                f"hierarchical_transformer_ev checkpoint metadata {metadata_key} mismatch: "
                f"expected {expected_value!r}; got {actual_value!r}"
            )

    for metadata_key in (
        "training_budget",
        "start_timesteps",
        "eval_frequency",
        "internal_eval_episodes",
    ):
        actual_value = metadata.get(metadata_key)
        if not isinstance(actual_value, int) or actual_value <= 0:
            raise ValueError(
                f"hierarchical_transformer_ev checkpoint metadata {metadata_key} "
                f"must be a positive integer; got {actual_value!r}"
            )

    if run_args is not None:
        run_arg_to_metadata_key = {
            "max_timesteps": "training_budget",
            "start_timesteps": "start_timesteps",
            "eval_freq": "eval_frequency",
            "eval_episodes": "internal_eval_episodes",
        }
        for run_args_key, metadata_key in run_arg_to_metadata_key.items():
            if run_args_key not in run_args:
                continue
            expected_value = int(run_args[run_args_key])
            actual_value = metadata.get(metadata_key)
            if actual_value != expected_value:
                raise ValueError(
                    f"hierarchical_transformer_ev checkpoint metadata {metadata_key} mismatch: "
                    f"expected run_args.yaml {run_args_key}={expected_value!r}; got {actual_value!r}"
                )


def _expected_transformer_constraint_metadata(checkpoint_role):
    from TD3.TD3_HierarchicalActionGNN_TransformerConstraint import (
        ACTION_DOMAIN_CONTRACT,
        ACTOR_OUTPUT_TRANSFORM,
        ACTOR_OUTPUT_TRANSFORM_FORMULA,
        CANONICAL_ALGORITHM_LABEL,
        CHECKPOINT_METADATA_SCHEMA,
        CHECKPOINT_SELECTION_RULE,
        NON_EV_ACTION,
    )

    return {
        "metadata_schema": CHECKPOINT_METADATA_SCHEMA,
        "algorithm": CANONICAL_ALGORITHM_LABEL,
        "action_domain_contract": ACTION_DOMAIN_CONTRACT,
        "actor_output_transform": ACTOR_OUTPUT_TRANSFORM,
        "actor_output_transform_formula": ACTOR_OUTPUT_TRANSFORM_FORMULA,
        "non_ev_action": NON_EV_ACTION,
        "discrete_actions": 1,
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "checkpoint_role": checkpoint_role,
    }


def _validate_transformer_constraint_metadata(checkpoint_prefix, metadata, run_args=None):
    expected_checkpoint_role = _checkpoint_role_from_prefix(checkpoint_prefix)
    expected_metadata = _expected_transformer_constraint_metadata(expected_checkpoint_role)
    for metadata_key, expected_value in expected_metadata.items():
        actual_value = metadata.get(metadata_key)
        if actual_value != expected_value:
            raise ValueError(
                f"hierarchical_transformer_constraint checkpoint metadata {metadata_key} mismatch: "
                f"expected {expected_value!r}; got {actual_value!r}"
            )

    for metadata_key in (
        "training_budget",
        "start_timesteps",
        "eval_frequency",
        "internal_eval_episodes",
    ):
        actual_value = metadata.get(metadata_key)
        if not isinstance(actual_value, int) or actual_value <= 0:
            raise ValueError(
                f"hierarchical_transformer_constraint checkpoint metadata {metadata_key} "
                f"must be a positive integer; got {actual_value!r}"
            )

    if run_args is not None:
        run_arg_to_metadata_key = {
            "max_timesteps": "training_budget",
            "start_timesteps": "start_timesteps",
            "eval_freq": "eval_frequency",
            "eval_episodes": "internal_eval_episodes",
        }
        for run_args_key, metadata_key in run_arg_to_metadata_key.items():
            if run_args_key not in run_args:
                continue
            expected_value = int(run_args[run_args_key])
            actual_value = metadata.get(metadata_key)
            if actual_value != expected_value:
                raise ValueError(
                    f"hierarchical_transformer_constraint checkpoint metadata {metadata_key} mismatch: "
                    f"expected run_args.yaml {run_args_key}={expected_value!r}; got {actual_value!r}"
                )


def validate_checkpoint_identity(checkpoint_prefix, canonical_algorithm):
    from TD3.TD3_ActionGNN_NonNegative import checkpoint_metadata_path

    canonical_algorithm = normalise_algorithm_label(canonical_algorithm)
    checkpoint_prefix = Path(checkpoint_prefix)
    metadata_path = checkpoint_metadata_path(checkpoint_prefix)
    corrected_metadata = _load_yaml_mapping(metadata_path)
    run_args = _load_yaml_mapping(checkpoint_prefix.parent / "run_args.yaml")

    if run_args is not None and "algorithm" in run_args:
        run_args_algorithm = normalise_algorithm_label(run_args["algorithm"])
        if run_args_algorithm != canonical_algorithm:
            raise ValueError(
                f"run_args.yaml algorithm {run_args_algorithm!r} does not match "
                f"requested canonical algorithm {canonical_algorithm!r}"
            )

    if corrected_metadata is not None:
        metadata_algorithm = corrected_metadata.get("algorithm")
        if metadata_algorithm != canonical_algorithm:
            raise ValueError(
                f"corrected checkpoint metadata algorithm {metadata_algorithm!r} does not match "
                f"requested canonical algorithm {canonical_algorithm!r}"
            )

    if canonical_algorithm == "actiongnn_nonnegative":
        if corrected_metadata is None:
            raise ValueError(
                f"actiongnn_nonnegative checkpoint metadata is required at {metadata_path}"
            )
        _validate_corrected_metadata(checkpoint_prefix, corrected_metadata)
    if canonical_algorithm == "hierarchical_transformer_ev":
        if corrected_metadata is None:
            raise ValueError(
                f"hierarchical_transformer_ev checkpoint metadata is required at {metadata_path}"
            )
        _validate_transformer_ev_metadata(
            checkpoint_prefix,
            corrected_metadata,
            run_args=run_args,
        )
    if canonical_algorithm == "hierarchical_transformer_constraint":
        if corrected_metadata is None:
            raise ValueError(
                f"hierarchical_transformer_constraint checkpoint metadata is required at {metadata_path}"
            )
        _validate_transformer_constraint_metadata(
            checkpoint_prefix,
            corrected_metadata,
            run_args=run_args,
        )


def create_policy(
    algorithm,
    action_dim,
    max_action,
    device="cpu",
    checkpoint_kwargs=None,
):
    algorithm = normalise_algorithm_label(algorithm)
    if algorithm not in ALGORITHM_CHOICES:
        supported_algorithms = ", ".join(sorted(ALGORITHM_CHOICES))
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose one of: {supported_algorithms}.")

    from utils.state_public_pst_gnn import PublicPST_GNN

    policy_kwargs = dict(checkpoint_kwargs or {})
    policy_kwargs.update({
        "action_dim": action_dim,
        "max_action": max_action,
        "fx_node_sizes": PublicPST_GNN.node_sizes,
        "device": device,
    })
    return get_policy_class(algorithm)(**policy_kwargs)


def load_policy_checkpoint(policy, checkpoint_prefix, canonical_algorithm):
    validate_checkpoint_identity(checkpoint_prefix, canonical_algorithm)
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
        description="Controlled evaluator for TD3 EV-GNN policies. CP scale is selected by the EV2Gym config file."
    )
    parser.add_argument("--algorithm", required=True, choices=sorted(set(ALGORITHM_CHOICES) | set(LEGACY_ALGORITHM_ALIASES)))
    parser.add_argument(
        "--config",
        default="./config_files/PublicPST_25cp.yaml",
        help="EV2Gym config file; CP scale is config-driven.",
    )
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
    canonical_algorithm = normalise_algorithm_label(args.algorithm)
    device = resolve_device(args.device)
    checkpoint_prefix = normalise_checkpoint_prefix(args.checkpoint)
    checkpoint_kwargs = load_checkpoint_kwargs(checkpoint_prefix)
    validate_checkpoint_identity(checkpoint_prefix, canonical_algorithm)

    probe_env = make_env(args.config, seed=args.seed)
    action_dim = probe_env.action_space.shape[0]
    max_action = float(probe_env.action_space.high[0])

    policy = create_policy(
        algorithm=canonical_algorithm,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        checkpoint_kwargs=checkpoint_kwargs,
    )
    load_policy_checkpoint(policy, checkpoint_prefix, canonical_algorithm)

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
        "algorithm": canonical_algorithm,
        "config": args.config,
        "seed": args.seed,
        "checkpoint": str(checkpoint_prefix),
    }
    rows, fieldnames = build_csv_rows(metadata, episode_records)
    output_path = write_csv(args.output_csv, rows, fieldnames)

    rewards = [episode_record["episode_reward"] for episode_record in episode_records]
    print("---------------------------------------")
    print(f"Run name: {args.run_name}")
    print(f"Algorithm: {canonical_algorithm}")
    print(f"Checkpoint: {checkpoint_prefix}")
    print(f"Episodes: {args.eval_episodes}")
    print(f"Mean reward: {float(np.mean(rewards)):.3f}")
    print(f"Std reward: {float(np.std(rewards)):.3f}")
    print(f"Output CSV: {output_path}")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
