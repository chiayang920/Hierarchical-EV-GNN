from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch_geometric.data import Data

from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN
from utils.replay_buffer_25cp import ActionGNN_ReplayBuffer
from utils.state_25cp import PublicPST_GNN


def make_hierarchical_policy():
    return TD3_HierarchicalActionGNN(
        action_dim=25,
        max_action=1.0,
        fx_node_sizes=PublicPST_GNN.node_sizes,
        fx_dim=16,
        fx_GNN_hidden_dim=32,
        mlp_hidden_dim=64,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        device="cpu",
    )


def build_repeated_id_graph(ev_action_values):
    state = Data(
        ev_features=np.array(
            [
                [0.5, 4.0, 1.0, 0.0, 0.0, 0.0],
                [0.5, 6.0, 2.0, 1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        cs_features=np.array(
            [
                [0.0, 32.0, 1.0, 0.0],
                [0.0, 32.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        tr_features=np.array(
            [
                [100.0, 0.0],
                [100.0, 1.0],
            ],
            dtype=float,
        ),
        env_features=np.array([[0.0, 0.0, 1.0, 10.0, 0.0]], dtype=float),
        edge_index=np.array(
            [
                [0, 1, 1, 2, 2, 3, 0, 4, 4, 5, 5, 6],
                [1, 0, 2, 1, 3, 2, 4, 0, 5, 4, 6, 5],
            ],
            dtype=np.int64,
        ),
        node_types=np.array([0, 1, 2, 3, 1, 2, 3], dtype=int),
        sample_node_length=[7],
        action_mapper=[0, 1],
        ev_indexes=np.array([3, 6], dtype=int),
        cs_indexes=np.array([2, 5], dtype=int),
        tr_indexes=np.array([1, 4], dtype=int),
        env_indexes=np.array([0], dtype=int),
    )
    full_node_action = torch.zeros((7, 1), dtype=torch.float32)
    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    full_node_action[active_ev_node_indexes, 0] = torch.as_tensor(
        ev_action_values,
        dtype=torch.float32,
    )
    return state, full_node_action


def build_sibling_charger_graph():
    return Data(
        ev_features=np.array(
            [
                [0.5, 4.0, 1.0, 0.0, 0.0, 0.0],
                [0.5, 6.0, 2.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cs_features=np.array(
            [
                [0.0, 32.0, 1.0, 0.0],
                [0.0, 32.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        tr_features=np.array([[100.0, 0.0]], dtype=float),
        env_features=np.array([[0.0, 0.0, 1.0, 10.0, 0.0]], dtype=float),
        edge_index=np.array(
            [
                [0, 1, 1, 2, 2, 3, 1, 4, 4, 5],
                [1, 0, 2, 1, 3, 2, 4, 1, 5, 4],
            ],
            dtype=np.int64,
        ),
        node_types=np.array([0, 1, 2, 3, 2, 3], dtype=int),
        sample_node_length=[6],
        action_mapper=[0, 1],
        ev_indexes=np.array([3, 5], dtype=int),
        cs_indexes=np.array([2, 4], dtype=int),
        tr_indexes=np.array([1], dtype=int),
        env_indexes=np.array([0], dtype=int),
    )


def assert_non_ev_rows_are_zero(full_node_action, state):
    total_nodes = int(sum(state.sample_node_length))
    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    non_ev_mask = torch.ones(total_nodes, dtype=torch.bool)
    non_ev_mask[active_ev_node_indexes] = False
    assert torch.allclose(
        full_node_action.reshape(-1, 1)[non_ev_mask],
        torch.zeros((int(non_ev_mask.sum().item()), 1), dtype=full_node_action.dtype),
    )


def test_batched_replay_forward_uses_per_graph_hierarchy_without_action_mapper():
    torch.manual_seed(3)
    np.random.seed(3)

    first_state, first_action = build_repeated_id_graph([0.25, 0.75])
    second_state, second_action = build_repeated_id_graph([0.5, 0.125])

    replay_buffer = ActionGNN_ReplayBuffer(action_dim=25, max_size=4, device="cpu")
    replay_buffer.add(first_state, first_action, first_state, reward=1.0, done=False)
    replay_buffer.add(second_state, second_action, second_state, reward=2.0, done=False)

    sampled_state, sampled_action, next_state, reward, not_done = replay_buffer.sample(2)

    assert sampled_action.shape == (int(sum(sampled_state.sample_node_length)), 1)
    assert not hasattr(sampled_state, "action_mapper")
    assert not hasattr(next_state, "action_mapper")
    assert torch.isfinite(reward).all()
    assert torch.isfinite(not_done).all()
    assert_non_ev_rows_are_zero(sampled_action, sampled_state)

    policy = make_hierarchical_policy()
    full_node_action, hierarchy_details = policy.actor(sampled_state, return_details=True)

    assert full_node_action.shape == (int(sum(sampled_state.sample_node_length)), 1)
    assert_non_ev_rows_are_zero(full_node_action.detach().cpu(), sampled_state)
    assert torch.isfinite(full_node_action).all()

    per_graph_transformer_weights = hierarchy_details["per_graph_transformer_weights"]
    per_graph_charger_weights = hierarchy_details["per_graph_charger_weights"]
    per_graph_charger_to_transformer_id = hierarchy_details["per_graph_charger_to_transformer_id"]
    per_graph_transformer_ids = hierarchy_details["per_graph_transformer_ids"]

    assert len(per_graph_transformer_weights) == 2
    assert len(per_graph_charger_weights) == 2

    for graph_transformer_weights in per_graph_transformer_weights:
        assert graph_transformer_weights.shape == (2,)
        assert torch.isclose(
            graph_transformer_weights.sum(),
            torch.tensor(1.0, device=graph_transformer_weights.device),
            atol=1e-6,
        )

    for graph_charger_weights, charger_to_transformer_id, graph_transformer_ids in zip(
        per_graph_charger_weights,
        per_graph_charger_to_transformer_id,
        per_graph_transformer_ids,
    ):
        assert graph_charger_weights.shape == (2,)
        for transformer_id in graph_transformer_ids:
            sibling_charger_mask = charger_to_transformer_id == transformer_id
            assert torch.isclose(
                graph_charger_weights[sibling_charger_mask].sum(),
                torch.tensor(1.0, device=graph_charger_weights.device),
                atol=1e-6,
            )

    critic_q1, critic_q2 = policy.critic(sampled_state, full_node_action)
    assert critic_q1.shape == (2, 1)
    assert critic_q2.shape == (2, 1)
    assert torch.isfinite(critic_q1).all()
    assert torch.isfinite(critic_q2).all()


def test_charger_weights_sum_within_single_transformer_siblings():
    torch.manual_seed(5)
    state = build_sibling_charger_graph()
    policy = make_hierarchical_policy()

    full_node_action, hierarchy_details = policy.actor(state, return_details=True)

    graph_charger_weights = hierarchy_details["per_graph_charger_weights"][0]
    charger_to_transformer_id = hierarchy_details["per_graph_charger_to_transformer_id"][0]
    graph_transformer_id = hierarchy_details["per_graph_transformer_ids"][0][0]
    sibling_charger_mask = charger_to_transformer_id == graph_transformer_id

    assert full_node_action.shape == (int(sum(state.sample_node_length)), 1)
    assert_non_ev_rows_are_zero(full_node_action.detach().cpu(), state)
    assert graph_charger_weights.shape == (2,)
    assert torch.isclose(
        graph_charger_weights[sibling_charger_mask].sum(),
        torch.tensor(1.0, device=graph_charger_weights.device),
        atol=1e-6,
    )
