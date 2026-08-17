import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer


FX_NODE_SIZES = {"ev": 6, "cs": 4, "tr": 2, "env": 5}


def transformer_ev_module():
    import TD3.TD3_TransformerEVActionGNN as module

    return module


def tensor_index(values):
    return torch.as_tensor(values, dtype=torch.long).reshape(-1)


def make_actor(seed=0, max_action=10.0):
    torch.manual_seed(seed)
    module = transformer_ev_module()
    return module.Actor(
        max_action=max_action,
        fx_node_sizes=FX_NODE_SIZES,
        feature_dim=8,
        GNN_hidden_dim=16,
        num_gcn_layers=3,
        discrete_actions=1,
        device=torch.device("cpu"),
    )


def make_policy(action_dim=8, max_action=10.0):
    module = transformer_ev_module()
    return module.TD3_TransformerEVActionGNN(
        action_dim=action_dim,
        max_action=max_action,
        fx_node_sizes=FX_NODE_SIZES,
        fx_dim=8,
        fx_GNN_hidden_dim=16,
        mlp_hidden_dim=32,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        discrete_actions=1,
        device="cpu",
    )


def build_graph(
    *,
    transformer_ids=(0,),
    charger_specs=((0, 0),),
    ev_specs=((0, 0, 0),),
    action_mapper=None,
    charger_feature_shift=0.0,
):
    node_types = [0]
    node_counter = 1
    env_indexes = [0]
    tr_indexes = []
    cs_indexes = []
    ev_indexes = []
    tr_features = []
    cs_features = []
    ev_features = []
    edge_from = []
    edge_to = []
    transformer_node_by_id = {}
    charger_node_by_id = {}

    for transformer_id in transformer_ids:
        transformer_node_by_id[int(transformer_id)] = node_counter
        tr_indexes.append(node_counter)
        tr_features.append([100.0 + float(transformer_id), float(transformer_id)])
        node_types.append(1)
        edge_from.extend([0, node_counter])
        edge_to.extend([node_counter, 0])
        node_counter += 1

    for charger_id, transformer_id in charger_specs:
        charger_id = int(charger_id)
        transformer_id = int(transformer_id)
        charger_node_by_id[charger_id] = node_counter
        cs_indexes.append(node_counter)
        cs_features.append(
            [
                charger_feature_shift,
                32.0 + charger_feature_shift + float(charger_id),
                2.0,
                float(charger_id),
            ]
        )
        node_types.append(2)
        transformer_node = transformer_node_by_id[transformer_id]
        edge_from.extend([transformer_node, node_counter])
        edge_to.extend([node_counter, transformer_node])
        node_counter += 1

    for ev_id, charger_id, transformer_id in ev_specs:
        ev_indexes.append(node_counter)
        ev_features.append(
            [
                0.5,
                2.0 + float(ev_id),
                1.0 + float(ev_id),
                float(ev_id),
                float(charger_id),
                float(transformer_id),
            ]
        )
        node_types.append(3)
        charger_node = charger_node_by_id[int(charger_id)]
        edge_from.extend([charger_node, node_counter])
        edge_to.extend([node_counter, charger_node])
        node_counter += 1

    if action_mapper is None:
        action_mapper = [int(ev_id) for ev_id, _charger_id, _transformer_id in ev_specs]

    return Data(
        ev_features=np.asarray(ev_features, dtype=float).reshape(-1, 6),
        cs_features=np.asarray(cs_features, dtype=float).reshape(-1, 4),
        tr_features=np.asarray(tr_features, dtype=float).reshape(-1, 2),
        env_features=np.asarray([[0.0, 0.0, 1.0, 4.0, 0.0]], dtype=float),
        edge_index=np.asarray([edge_from, edge_to], dtype=np.int64),
        node_types=np.asarray(node_types, dtype=int),
        sample_node_length=[len(node_types)],
        action_mapper=list(action_mapper),
        ev_indexes=np.asarray(ev_indexes, dtype=int),
        cs_indexes=np.asarray(cs_indexes, dtype=int),
        tr_indexes=np.asarray(tr_indexes, dtype=int),
        env_indexes=np.asarray(env_indexes, dtype=int),
    )


def build_no_active_ev_state():
    return Data(
        ev_features=np.empty((0, 6), dtype=float),
        cs_features=np.empty((0, 4), dtype=float),
        tr_features=np.empty((0, 2), dtype=float),
        env_features=np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float),
        edge_index=np.empty((2, 0), dtype=np.int64),
        node_types=np.asarray([0], dtype=int),
        sample_node_length=[1],
        action_mapper=[],
        ev_indexes=np.asarray([], dtype=int),
        cs_indexes=np.asarray([], dtype=int),
        tr_indexes=np.asarray([], dtype=int),
        env_indexes=np.asarray([0], dtype=int),
    )


def batch_graphs(graphs):
    edge_parts = []
    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = []
    sample_node_length = []
    action_mapper = []
    node_offset = 0

    for graph in graphs:
        edge_index = np.asarray(graph.edge_index, dtype=np.int64)
        if edge_index.size:
            edge_parts.append(edge_index + node_offset)
        ev_indexes.append(np.asarray(graph.ev_indexes, dtype=np.int64) + node_offset)
        cs_indexes.append(np.asarray(graph.cs_indexes, dtype=np.int64) + node_offset)
        tr_indexes.append(np.asarray(graph.tr_indexes, dtype=np.int64) + node_offset)
        env_indexes.append(np.asarray(graph.env_indexes, dtype=np.int64) + node_offset)
        sample_node_length.append(int(len(graph.node_types)))
        action_mapper.extend(list(graph.action_mapper))
        node_offset += int(len(graph.node_types))

    return Data(
        ev_features=np.concatenate([graph.ev_features for graph in graphs], axis=0),
        cs_features=np.concatenate([graph.cs_features for graph in graphs], axis=0),
        tr_features=np.concatenate([graph.tr_features for graph in graphs], axis=0),
        env_features=np.concatenate([graph.env_features for graph in graphs], axis=0),
        edge_index=np.concatenate(edge_parts, axis=1) if edge_parts else np.empty((2, 0), dtype=np.int64),
        node_types=np.concatenate([graph.node_types for graph in graphs], axis=0),
        sample_node_length=sample_node_length,
        action_mapper=action_mapper,
        ev_indexes=np.concatenate(ev_indexes).astype(int),
        cs_indexes=np.concatenate(cs_indexes).astype(int),
        tr_indexes=np.concatenate(tr_indexes).astype(int),
        env_indexes=np.concatenate(env_indexes).astype(int),
    )


def graph_slice(state, graph_index):
    start = sum(state.sample_node_length[:graph_index])
    end = start + state.sample_node_length[graph_index]
    return slice(start, end)


def assert_non_ev_rows_are_zero(full_node_action, state):
    non_ev_mask = torch.ones(full_node_action.shape[0], dtype=torch.bool)
    non_ev_mask[tensor_index(state.ev_indexes)] = False
    assert torch.equal(
        full_node_action[non_ev_mask],
        torch.zeros_like(full_node_action[non_ev_mask]),
    )


def test_transformer_ev_actor_has_expected_heads_and_no_charger_head():
    actor = make_actor(seed=1)

    assert hasattr(actor, "transformer_score_head")
    assert hasattr(actor, "ev_allocation_score_head")
    assert hasattr(actor, "ev_gate_head")
    assert not hasattr(actor, "charger_score_head")


def test_transformer_weights_sum_to_one_per_graph():
    actor = make_actor(seed=2)
    state = batch_graphs(
        [
            build_graph(
                transformer_ids=(0, 1),
                charger_specs=((0, 0), (1, 1)),
                ev_specs=((0, 0, 0), (1, 1, 1)),
            ),
            build_graph(
                transformer_ids=(0, 1),
                charger_specs=((0, 0), (1, 1)),
                ev_specs=((2, 0, 0), (3, 1, 1)),
            ),
        ]
    )

    _full_node_action, details = actor(state, return_details=True)

    assert len(details["per_graph_transformer_weights"]) == 2
    for graph_transformer_weights in details["per_graph_transformer_weights"]:
        assert torch.isclose(graph_transformer_weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_ev_allocation_weights_sum_to_one_within_transformer():
    actor = make_actor(seed=3)
    state = build_graph(
        transformer_ids=(0, 1),
        charger_specs=((0, 0), (1, 0), (2, 1)),
        ev_specs=((0, 0, 0), (1, 1, 0), (2, 1, 0), (3, 2, 1)),
    )

    _full_node_action, details = actor(state, return_details=True)

    graph_ev_weights = details["per_graph_ev_allocation_weights"][0]
    graph_ev_transformer_ids = details["per_graph_ev_to_transformer_id"][0]
    for transformer_id in torch.unique(graph_ev_transformer_ids):
        sibling_weights = graph_ev_weights[graph_ev_transformer_ids == transformer_id]
        assert torch.isclose(sibling_weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_no_charger_softmax_or_per_graph_charger_weights():
    actor = make_actor(seed=4)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )

    _full_node_action, details = actor(state, return_details=True)

    assert not hasattr(actor, "charger_score_head")
    assert "per_graph_charger_weights" not in details
    assert all("charger" not in detail_key for detail_key in details)
    assert all("charger_score_head" not in name for name, _parameter in actor.named_parameters())


def test_output_shape_non_ev_rows_zero_and_ev_domain_nonnegative():
    max_action = 2.5
    actor = make_actor(seed=5, max_action=max_action)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0), (2, 1, 0)),
    )

    full_node_action = actor(state)
    active_ev_node_indexes = tensor_index(state.ev_indexes)

    assert full_node_action.shape == (len(state.node_types), 1)
    assert_non_ev_rows_are_zero(full_node_action, state)
    assert torch.all(full_node_action[active_ev_node_indexes] >= 0.0)
    assert torch.all(full_node_action[active_ev_node_indexes] <= max_action)


def test_action_mapper_reconstruction_matches_full_node_action():
    policy = make_policy(action_dim=6, max_action=5.0)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
        action_mapper=[4, 1],
    )

    mapped_action, full_node_action = policy.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )

    active_ev_node_indexes = tensor_index(state.ev_indexes)
    assert mapped_action.shape == (6,)
    assert mapped_action.dtype == np.float32
    assert mapped_action[4] == pytest.approx(full_node_action[active_ev_node_indexes[0], 0].item())
    assert mapped_action[1] == pytest.approx(full_node_action[active_ev_node_indexes[1], 0].item())
    assert mapped_action[0] == pytest.approx(0.0)
    assert mapped_action[2] == pytest.approx(0.0)
    assert mapped_action[3] == pytest.approx(0.0)
    assert mapped_action[5] == pytest.approx(0.0)


def test_no_active_ev_state_returns_exact_zeros():
    policy = make_policy(action_dim=4, max_action=5.0)
    state = build_no_active_ev_state()

    mapped_action, full_node_action = policy.select_action(
        state,
        expl_noise=0.25,
        return_mapped_action=True,
    )

    assert np.array_equal(mapped_action, np.zeros(4, dtype=np.float32))
    assert torch.equal(full_node_action, torch.zeros((1, 1), dtype=full_node_action.dtype))


def test_gradients_reach_transformer_ev_allocation_and_gate_heads():
    policy = make_policy(action_dim=8, max_action=100.0)
    state = build_graph(
        transformer_ids=(0, 1),
        charger_specs=((0, 0), (1, 0), (2, 1)),
        ev_specs=((0, 0, 0), (1, 1, 0), (2, 2, 1), (3, 2, 1)),
    )

    policy.actor_optimizer.zero_grad()
    full_node_action = policy.actor(state)
    active_ev_node_indexes = tensor_index(state.ev_indexes)
    coefficients = torch.arange(
        1,
        active_ev_node_indexes.numel() + 1,
        dtype=full_node_action.dtype,
    ).reshape(-1, 1)
    actor_loss = (full_node_action[active_ev_node_indexes] * coefficients).sum()
    actor_loss.backward()

    for head_name in (
        "transformer_score_head",
        "ev_allocation_score_head",
        "ev_gate_head",
    ):
        matching_parameters = [
            parameter
            for parameter_name, parameter in policy.actor.named_parameters()
            if head_name in parameter_name
        ]
        assert matching_parameters
        assert any(parameter.grad is not None for parameter in matching_parameters)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in matching_parameters
        )


def test_batched_graphs_do_not_share_transformer_or_ev_allocation_mass():
    actor = make_actor(seed=6)
    first_graph = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0), (1, 0, 0)),
    )
    second_graph = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((2, 0, 0), (3, 0, 0), (4, 0, 0)),
    )
    batched_state = batch_graphs([first_graph, second_graph])

    standalone_output = actor(first_graph)
    batched_output, details = actor(batched_state, return_details=True)

    assert torch.allclose(batched_output[graph_slice(batched_state, 0)], standalone_output, atol=1e-6)
    assert torch.isclose(
        sum(weights.sum() for weights in details["per_graph_transformer_weights"]),
        torch.tensor(2.0),
        atol=1e-6,
    )
    assert torch.isclose(
        sum(weights.sum() for weights in details["per_graph_ev_allocation_weights"]),
        torch.tensor(2.0),
        atol=1e-6,
    )


def test_repeated_graph_local_transformer_ids_are_isolated():
    actor = make_actor(seed=7)
    state = batch_graphs(
        [
            build_graph(
                transformer_ids=(0, 1),
                charger_specs=((0, 0), (1, 1)),
                ev_specs=((0, 0, 0), (1, 1, 1)),
            ),
            build_graph(
                transformer_ids=(0, 1),
                charger_specs=((0, 0), (1, 1)),
                ev_specs=((2, 0, 0), (3, 1, 1)),
            ),
        ]
    )

    _full_node_action, details = actor(state, return_details=True)

    assert len(details["per_graph_transformer_ids"]) == 2
    assert [ids.tolist() for ids in details["per_graph_transformer_ids"]] == [[0, 1], [0, 1]]
    for graph_transformer_weights in details["per_graph_transformer_weights"]:
        assert torch.isclose(graph_transformer_weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_charger_nodes_remain_in_encoder_representation_but_not_action_allocation():
    actor = make_actor(seed=8, max_action=100.0)
    baseline_state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0), (1, 0, 0)),
        charger_feature_shift=0.0,
    )
    changed_charger_state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0), (1, 0, 0)),
        charger_feature_shift=100.0,
    )

    baseline_action, details = actor(baseline_state, return_details=True)
    changed_action = actor(changed_charger_state)

    assert hasattr(actor, "cs_embedding")
    assert not hasattr(actor, "charger_score_head")
    assert "per_graph_charger_weights" not in details
    assert not torch.allclose(baseline_action, changed_action)


def test_policy_critic_and_replay_accept_full_node_actions():
    policy = make_policy(action_dim=6, max_action=5.0)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
        action_mapper=[4, 1],
    )
    _mapped_action, full_node_action = policy.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=6, max_size=4, device="cpu")
    replay_buffer.add(state, full_node_action, state, reward=0.0, done=False)

    sampled_state, sampled_action, next_state, reward, not_done = replay_buffer.sample(1)
    critic_q1, critic_q2 = policy.critic(sampled_state, sampled_action)

    assert sampled_action.shape == (int(sum(sampled_state.sample_node_length)), 1)
    assert critic_q1.shape == (1, 1)
    assert critic_q2.shape == (1, 1)
    assert torch.isfinite(critic_q1).all()
    assert torch.isfinite(critic_q2).all()
    assert torch.isfinite(reward).all()
    assert torch.isfinite(not_done).all()
