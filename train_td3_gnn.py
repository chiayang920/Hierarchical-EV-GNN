import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from utils.ev2gym_training_utils import (
    make_env,
    normalise_step_result,
    reset_env,
    resolve_device,
    set_global_seed,
    str2bool,
)


ALGORITHM_CHOICES = ("actiongnn", "hierarchical")


def get_policy_class(algorithm):
    if algorithm == "actiongnn":
        from TD3.TD3_ActionGNN_Controlled import TD3_ActionGNN

        return TD3_ActionGNN
    if algorithm == "hierarchical":
        from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN

        return TD3_HierarchicalActionGNN
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def evaluate_policy(policy, args, config_file, eval_episodes):
    rewards = []
    stats_list = []
    total_steps = 0
    start_time = time.time()

    for episode in tqdm(range(eval_episodes), desc="evaluation", leave=False):
        env = make_env(config_file, seed=args.seed + 100_000 + episode)
        state, _ = reset_env(env, seed=args.seed + 100_000 + episode)
        done = False
        episode_reward = 0.0

        while not done:
            mapped_action, _ = policy.select_action(state, expl_noise=0, return_mapped_action=True)
            state, reward, done, stats = normalise_step_result(env.step(mapped_action))
            episode_reward += reward
            total_steps += 1

        rewards.append(episode_reward)
        stats_list.append(stats)

    eval_stats = {
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/std_reward": float(np.std(rewards)),
        "eval/min_reward": float(np.min(rewards)),
        "eval/max_reward": float(np.max(rewards)),
        "eval/time_seconds": float(time.time() - start_time),
        "eval/steps": int(total_steps),
    }

    if stats_list:
        for key in stats_list[0].keys():
            values = [x[key] for x in stats_list if key in x]
            if values and np.isscalar(values[0]):
                eval_stats[f"eval_metrics/{key}_mean"] = float(np.mean(values))
                eval_stats[f"eval_metrics/{key}_std"] = float(np.std(values))

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {eval_stats['eval/mean_reward']:.3f}")
    print("---------------------------------------")
    return eval_stats


def write_log_row(log_path, row):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="TD3 EV-GNN training. CP scale is selected by the EV2Gym config file.")
    parser.add_argument("--algorithm", default="actiongnn", choices=sorted(ALGORITHM_CHOICES))
    parser.add_argument(
        "--config",
        default="./config_files/PublicPST_25cp.yaml",
        help="EV2Gym config file; CP scale is config-driven.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--run_name", default=None)

    parser.add_argument("--max_timesteps", type=int, default=2_240_000)
    parser.add_argument("--eval_freq", type=int, default=33_600)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--start_timesteps", type=int, default=2_500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--replay_buffer_size", type=int, default=1_000_000)

    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expl_noise", type=float, default=0.1)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--fx_dim", type=int, default=32)
    parser.add_argument("--fx_GNN_hidden_dim", type=int, default=64)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--actor_num_gcn_layers", type=int, default=3)
    parser.add_argument("--critic_num_gcn_layers", type=int, default=3)
    parser.add_argument("--discrete_actions", type=int, default=1)

    parser.add_argument("--save_dir", default="./artifacts/experiments")
    parser.add_argument("--log_to_wandb", type=str2bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    set_global_seed(args.seed)

    config_file = args.config
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    env = make_env(config_file, seed=args.seed)
    state, _ = reset_env(env, seed=args.seed)

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    simulation_length = int(config["simulation_length"])
    from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer
    from utils.state_public_pst_gnn import PublicPST_GNN

    policy_class = get_policy_class(args.algorithm)

    run_name = args.run_name or f"td3_gnn_{args.algorithm}_seed{args.seed}"
    args.run_name = run_name

    save_path = Path(args.save_dir) / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    print("---------------------------------------")
    print("TD3 EV-GNN training")
    print(f"Algorithm: {args.algorithm}")
    print(f"Config: {config_file}")
    print(f"Detected action dimension: {action_dim}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Max timesteps: {args.max_timesteps}")
    print(f"Eval frequency: {args.eval_freq}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Save path: {save_path}")
    print("---------------------------------------")

    kwargs = {
        "action_dim": action_dim,
        "max_action": max_action,
        "fx_node_sizes": PublicPST_GNN.node_sizes,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "fx_dim": args.fx_dim,
        "fx_GNN_hidden_dim": args.fx_GNN_hidden_dim,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "lr": args.lr,
        "discrete_actions": args.discrete_actions,
        "actor_num_gcn_layers": args.actor_num_gcn_layers,
        "critic_num_gcn_layers": args.critic_num_gcn_layers,
        "device": args.device,
    }

    with (save_path / "config.yaml").open("w") as file:
        yaml.dump(config, file)
    with (save_path / "kwargs.yaml").open("w") as file:
        yaml.dump(kwargs, file)
    with (save_path / "run_args.yaml").open("w") as file:
        yaml.dump(vars(args), file)

    policy = policy_class(**kwargs)
    replay_buffer = ActionGNN_ReplayBuffer(
        action_dim=action_dim,
        max_size=args.replay_buffer_size,
        device=args.device,
    )

    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Simulation length: {simulation_length}")
    print(f"Save path: {save_path}")

    best_reward = -np.inf
    episode_num = 0
    episode_timesteps = 0
    episode_reward = 0.0
    start_time = time.time()
    ep_start_time = time.time()
    log_path = save_path / "training_log.csv"

    for t in range(args.max_timesteps):
        episode_timesteps += 1

        mapped_action, node_action = policy.select_action(state, expl_noise=args.expl_noise)
        next_state, reward, done, stats = normalise_step_result(env.step(mapped_action))

        replay_buffer.add(state, node_action, next_state, reward, done)
        state = next_state
        episode_reward += reward

        critic_loss = None
        actor_loss = None
        if t >= args.start_timesteps:
            critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)

        if done:
            elapsed_episode = time.time() - ep_start_time
            print(
                f"Total T: {t + 1} | Episode: {episode_num} | "
                f"Episode T: {episode_timesteps} | Reward: {episode_reward:.3f} | "
                f"Time: {elapsed_episode:.3f}s"
            )
            write_log_row(log_path, {
                "type": "train_episode",
                "timestep": t + 1,
                "episode": episode_num,
                "episode_timesteps": episode_timesteps,
                "episode_reward": episode_reward,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "elapsed_seconds": time.time() - start_time,
            })

            episode_num += 1
            episode_timesteps = 0
            episode_reward = 0.0
            state, _ = reset_env(env, seed=args.seed + episode_num)
            ep_start_time = time.time()

        if (t + 1) % args.eval_freq == 0:
            eval_stats = evaluate_policy(policy, args, config_file, args.eval_episodes)
            mean_reward = eval_stats["eval/mean_reward"]

            if mean_reward > best_reward:
                best_reward = mean_reward
                policy.save(str(save_path / "model.best"))
                print(f"Saved new best model: {best_reward:.3f}")

            row = {
                "type": "evaluation",
                "timestep": t + 1,
                "episode": episode_num,
                "episode_timesteps": episode_timesteps,
                "episode_reward": episode_reward,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "elapsed_seconds": time.time() - start_time,
            }
            row.update(eval_stats)
            row["eval/best_reward"] = best_reward
            write_log_row(log_path, row)

            steps_per_second = (t + 1) / max(time.time() - start_time, 1e-9)
            remaining_steps = args.max_timesteps - (t + 1)
            eta_hours = remaining_steps / max(steps_per_second, 1e-9) / 3600
            print(f"Approx. training throughput: {steps_per_second:.2f} steps/s | ETA: {eta_hours:.2f} h")

    policy.save(str(save_path / "model.last"))
    final_stats = evaluate_policy(policy, args, config_file, args.eval_episodes)
    final_row = {
        "type": "final_evaluation",
        "timestep": args.max_timesteps,
        "episode": episode_num,
        "episode_timesteps": episode_timesteps,
        "episode_reward": episode_reward,
        "critic_loss": None,
        "actor_loss": None,
        "elapsed_seconds": time.time() - start_time,
    }
    final_row.update(final_stats)
    final_row["eval/best_reward"] = best_reward
    write_log_row(log_path, final_row)
    print(f"Final mean reward: {final_stats['eval/mean_reward']:.3f}")
    print(f"Best mean reward: {best_reward:.3f}")
    print(f"Saved final model under: {save_path}")


if __name__ == "__main__":
    main()
