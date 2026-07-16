from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from gymnasium import Space
from torch_geometric.data import Data

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SimpleReward

from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN
from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer
from utils.state_public_pst_gnn import PublicPST_GNN


class PyGDataSpace(Space):
    def __init__(self):
        super().__init__((), None)

    def sample(self):
        return Data()

    def contains(self, value):
        return isinstance(value, Data)


def reset_env(env, seed=None):
    try:
        reset_result = env.reset(seed=seed)
    except TypeError:
        reset_result = env.reset()

    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


def normalise_step_result(step_result):
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, stats = step_result
        done = bool(terminated or truncated)
        return next_state, reward, done, stats
    if len(step_result) == 4:
        next_state, reward, done, stats = step_result
        return next_state, reward, bool(done), stats
    raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")


def first_active_ev_state(env, seed=0):
    state, reset_info = reset_env(env, seed=seed)
    zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)

    for step_index in range(env.simulation_length):
        if len(state.ev_indexes) > 0:
            return state, reset_info
        state, reward, done, stats = normalise_step_result(env.step(zero_action))
        if done:
            break

    raise AssertionError("PublicPST test config did not produce an active-EV graph state.")


def make_public_pst_test_env(seed=0):
    config_file = PROJECT_ROOT / "config_files" / "PublicPST_25cp.yaml"
    env = EV2Gym(
        config_file=str(config_file),
        generate_rnd_game=True,
        reward_function=SimpleReward,
        state_function=PublicPST_GNN,
    )
    env.observation_space = PyGDataSpace()
    env.action_space.seed(seed)
    return env


def make_hierarchical_policy(action_dim=25, max_action=1.0):
    return TD3_HierarchicalActionGNN(
        action_dim=action_dim,
        max_action=max_action,
        fx_node_sizes=PublicPST_GNN.node_sizes,
        fx_dim=16,
        fx_GNN_hidden_dim=32,
        mlp_hidden_dim=64,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        batch_size=4,
        device="cpu",
    )


def tensor_index(values):
    return torch.as_tensor(values, dtype=torch.long)


def assert_non_ev_rows_are_zero(full_node_action, state):
    total_nodes = int(sum(state.sample_node_length))
    active_ev_node_indexes = tensor_index(state.ev_indexes)
    non_ev_mask = torch.ones(total_nodes, dtype=torch.bool)
    non_ev_mask[active_ev_node_indexes] = False
    assert torch.allclose(
        full_node_action.reshape(-1, 1)[non_ev_mask],
        torch.zeros((int(non_ev_mask.sum().item()), 1), dtype=full_node_action.dtype),
    )


def test_select_action_preserves_ev2gym_and_full_node_contracts():
    torch.manual_seed(0)
    np.random.seed(0)

    env = make_public_pst_test_env(seed=0)
    state, reset_info = first_active_ev_state(env, seed=0)
    assert isinstance(reset_info, dict)

    policy = make_hierarchical_policy(
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
    )

    mapped_action_numpy, full_node_action = policy.select_action(
        state,
        expl_noise=0,
        return_mapped_action=True,
    )

    total_nodes = int(sum(state.sample_node_length))
    assert mapped_action_numpy.shape == (env.action_space.shape[0],)
    assert mapped_action_numpy.dtype == np.float32
    assert full_node_action.shape == (total_nodes, 1)
    assert_non_ev_rows_are_zero(full_node_action, state)

    active_ev_node_indexes = tensor_index(state.ev_indexes)
    for active_ev_position, action_slot in enumerate(state.action_mapper):
        ev_node_index = active_ev_node_indexes[active_ev_position].item()
        assert np.isclose(
            mapped_action_numpy[action_slot],
            full_node_action[ev_node_index, 0].item(),
            atol=1e-6,
        )

    replay_buffer = ActionGNN_ReplayBuffer(
        action_dim=env.action_space.shape[0],
        max_size=4,
        device="cpu",
    )
    replay_buffer.add(state, full_node_action, state, reward=0.0, done=False)
    sampled_state, sampled_action, next_state, reward, not_done = replay_buffer.sample(1)

    assert sampled_action.shape == (int(sum(sampled_state.sample_node_length)), 1)
    critic_q1, critic_q2 = policy.critic(sampled_state, sampled_action)
    assert critic_q1.shape == (1, 1)
    assert critic_q2.shape == (1, 1)
    assert torch.isfinite(critic_q1).all()
    assert torch.isfinite(critic_q2).all()
    assert torch.isfinite(reward).all()
    assert torch.isfinite(not_done).all()

    policy.actor_optimizer.zero_grad()
    actor_action = policy.actor(next_state)
    actor_loss = -policy.critic.Q1(next_state, actor_action).mean()
    actor_loss.backward()

    hierarchy_head_names = [
        "transformer_score_head",
        "charger_score_head",
        "ev_gate_head",
    ]
    for hierarchy_head_name in hierarchy_head_names:
        matching_parameters = [
            parameter
            for parameter_name, parameter in policy.actor.named_parameters()
            if hierarchy_head_name in parameter_name
        ]
        assert matching_parameters
        assert any(parameter.grad is not None for parameter in matching_parameters)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in matching_parameters
        )


def build_no_active_ev_state():
    return Data(
        ev_features=np.empty((0, 6), dtype=float),
        cs_features=np.empty((0, 4), dtype=float),
        tr_features=np.empty((0, 2), dtype=float),
        env_features=np.array([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float),
        edge_index=np.empty((2, 0), dtype=np.int64),
        node_types=np.array([0], dtype=int),
        sample_node_length=[1],
        action_mapper=[],
        ev_indexes=np.array([], dtype=int),
        cs_indexes=np.array([], dtype=int),
        tr_indexes=np.array([], dtype=int),
        env_indexes=np.array([0], dtype=int),
    )


def test_select_action_handles_no_active_ev_state():
    action_dim = 25
    policy = make_hierarchical_policy(action_dim=action_dim, max_action=1.0)
    state = build_no_active_ev_state()

    mapped_action_numpy, full_node_action = policy.select_action(
        state,
        expl_noise=0.1,
        return_mapped_action=True,
    )

    assert mapped_action_numpy.shape == (action_dim,)
    assert mapped_action_numpy.dtype == np.float32
    assert np.allclose(mapped_action_numpy, 0.0)
    assert full_node_action.shape == (1, 1)
    assert torch.allclose(full_node_action, torch.zeros((1, 1)))

    training_action = policy.actor(state)
    assert training_action.shape == (1, 1)
    assert torch.allclose(training_action.detach().cpu(), torch.zeros((1, 1)))
