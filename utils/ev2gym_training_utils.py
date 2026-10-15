import argparse
import random

import numpy as np
from gymnasium import Space


class PyGDataSpace(Space):
    def __init__(self):
        super().__init__((), None)

    def sample(self):
        from torch_geometric.data import Data

        return Data()

    def contains(self, value):
        from torch_geometric.data import Data

        return isinstance(value, Data)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_device(device_arg):
    import torch

    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return "cpu"
    return device_arg


def set_global_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_env(env, seed=None):
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def make_env(config_file, seed=None):
    from ev2gym.models.ev2gym_env import EV2Gym
    from ev2gym.rl_agent.reward import SimpleReward
    from utils.state_public_pst_gnn import PublicPST_GNN

    if seed is not None:
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


def normalise_step_result(step_result):
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, stats = step_result
        done = bool(terminated or truncated)
        return next_state, reward, done, stats
    if len(step_result) == 4:
        next_state, reward, done, stats = step_result
        return next_state, reward, bool(done), stats
    raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")
