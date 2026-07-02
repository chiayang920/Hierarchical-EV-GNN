import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from gymnasium import Space
from torch_geometric.data import Data
from tqdm import tqdm

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SimpleReward

from TD3.TD3_ActionGNN_25cp import TD3_ActionGNN
from utils.state_25cp import PublicPST_GNN


class PyGDataSpace(Space):
    def __init__(self):
        super().__init__((), None)

    def sample(self):
        return Data()

    def contains(self, x):
        return isinstance(x, Data)


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_env(env, seed=None):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def normalise_step_result(step_result):
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, stats = step_result
        return next_state, reward, bool(terminated or truncated), stats
    if len(step_result) == 4:
        next_state, reward, done, stats = step_result
        return next_state, reward, bool(done), stats
    raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")


def make_env(config_file, seed):
    set_global_seed(seed)
    env = EV2Gym(
        config_file=config_file,
        generate_rnd_game=True,
        reward_function=SimpleReward,
        state_function=PublicPST_GNN,
    )
    env.observation_space = PyGDataSpace()
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 25 CP TD3 EV-GNN model.")
    parser.add_argument("--config", default="./config_files/PublicPST_25cp.yaml")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="./results/eval_25cp_td3_actiongnn.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    kwargs_path = model_dir / "kwargs.yaml"
    if not kwargs_path.exists():
        raise FileNotFoundError(f"Missing kwargs file: {kwargs_path}")

    with kwargs_path.open("r") as file:
        kwargs = yaml.load(file, Loader=yaml.FullLoader)

    policy = TD3_ActionGNN(**kwargs)
    checkpoint_prefix = model_dir / f"model.{args.checkpoint}"
    policy.load(str(checkpoint_prefix))

    rows = []
    total_start = time.time()

    for episode in tqdm(range(args.episodes), desc="evaluating"):
        env = make_env(args.config, args.seed + episode)
        state, _ = reset_env(env, seed=args.seed + episode)
        done = False
        episode_reward = 0.0
        episode_start = time.time()
        stats = {}

        while not done:
            mapped_action, _ = policy.select_action(state, expl_noise=0, return_mapped_action=True)
            state, reward, done, stats = normalise_step_result(env.step(mapped_action))
            episode_reward += reward

        row = {
            "episode": episode,
            "total_reward": episode_reward,
            "time_seconds": time.time() - episode_start,
        }
        for key, value in stats.items():
            if np.isscalar(value):
                row[key] = value
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rewards = [row["total_reward"] for row in rows]
    print("---------------------------------------")
    print(f"Model: {model_dir}")
    print(f"Checkpoint: model.{args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Std reward: {np.std(rewards):.3f}")
    print(f"Output CSV: {output_path}")
    print(f"Total evaluation time: {time.time() - total_start:.2f}s")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
