from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


FX_NODE_SIZES = {
    "ev": 6,
    "cs": 4,
    "tr": 2,
    "env": 5,
}


def nonnegative_module():
    import TD3.TD3_ActionGNN_NonNegative as module

    return module


def build_active_ev_state():
    return Data(
        ev_features=np.array(
            [
                [0.5, 4.0, 1.0, 0.0, 0.0, 0.0],
                [0.25, 6.0, 2.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cs_features=np.array([[0.0, 32.0, 1.0, 0.0]], dtype=float),
        tr_features=np.array([[100.0, 0.0]], dtype=float),
        env_features=np.array([[0.0, 0.0, 1.0, 10.0, 0.0]], dtype=float),
        edge_index=np.array(
            [
                [0, 1, 1, 2, 2, 3, 2, 4],
                [1, 0, 2, 1, 3, 2, 4, 2],
            ],
            dtype=np.int64,
        ),
        node_types=np.array([0, 1, 2, 3, 3], dtype=int),
        sample_node_length=[5],
        action_mapper=[1, 3],
        ev_indexes=np.array([3, 4], dtype=int),
        cs_indexes=np.array([2], dtype=int),
        tr_indexes=np.array([1], dtype=int),
        env_indexes=np.array([0], dtype=int),
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


def make_actor(max_action=1.0, device="cpu", discrete_actions=1):
    module = nonnegative_module()
    resolved_device = torch.device(device)
    return module.Actor(
        max_action=max_action,
        fx_node_sizes=FX_NODE_SIZES,
        feature_dim=8,
        GNN_hidden_dim=16,
        num_gcn_layers=3,
        discrete_actions=discrete_actions,
        device=resolved_device,
    ).to(resolved_device)


def make_policy(max_action=1.0, discrete_actions=1):
    module = nonnegative_module()
    return module.TD3_ActionGNN_NonNegative(
        action_dim=4,
        max_action=max_action,
        fx_node_sizes=FX_NODE_SIZES,
        fx_dim=8,
        fx_GNN_hidden_dim=16,
        mlp_hidden_dim=32,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        discrete_actions=discrete_actions,
        device="cpu",
    )


def non_ev_mask_for(state):
    total_nodes = int(sum(state.sample_node_length))
    mask = torch.ones(total_nodes, dtype=torch.bool)
    mask[torch.as_tensor(state.ev_indexes, dtype=torch.long)] = False
    return mask


def test_shifted_tanh_known_logits_fixed_oracle():
    module = nonnegative_module()
    raw_logits = torch.tensor([[-2.0], [0.0], [2.0]], dtype=torch.float32)

    max_one_action = module.shifted_tanh_nonnegative(raw_logits, max_action=1.0)
    assert torch.allclose(
        max_one_action.reshape(-1),
        torch.tensor(
            [0.11920292202211755, 0.5, 0.8807970779778824],
            dtype=torch.float32,
        ),
        atol=1e-7,
    )

    max_two_action = module.shifted_tanh_nonnegative(raw_logits, max_action=2.0)
    assert torch.allclose(
        max_two_action.reshape(-1),
        torch.tensor(
            [0.2384058440442351, 1.0, 1.7615941559557649],
            dtype=torch.float32,
        ),
        atol=1e-7,
    )


def test_shifted_tanh_gradients_are_finite_near_zero():
    module = nonnegative_module()
    raw_logits = torch.tensor([[-0.001], [0.0], [0.001]], requires_grad=True)

    module.shifted_tanh_nonnegative(raw_logits, max_action=1.0).sum().backward()

    assert raw_logits.grad is not None
    assert torch.isfinite(raw_logits.grad).all()
    assert torch.all(raw_logits.grad > 0.0)


def test_actor_ev_rows_are_within_nonnegative_bounds():
    torch.manual_seed(11)
    actor = make_actor(max_action=2.0)
    action = actor(build_active_ev_state()).detach().cpu()
    active_ev_rows = action[torch.as_tensor([3, 4], dtype=torch.long)]

    assert torch.all(active_ev_rows >= 0.0)
    assert torch.all(active_ev_rows <= 2.0)


def test_actor_non_ev_rows_are_exact_zero():
    torch.manual_seed(12)
    state = build_active_ev_state()
    actor = make_actor(max_action=1.0)
    action = actor(state).detach().cpu()

    assert torch.equal(
        action[non_ev_mask_for(state)],
        torch.zeros((3, 1), dtype=torch.float32),
    )


def test_actor_no_active_ev_returns_all_zero_tensor():
    torch.manual_seed(13)
    state = build_no_active_ev_state()
    actor = make_actor(max_action=1.0)
    full_node_action = actor(state).detach().cpu()

    policy = make_policy(max_action=1.0)
    mapped_action, selected_full_node_action = policy.select_action(
        state,
        expl_noise=0.1,
        return_mapped_action=True,
    )

    assert torch.equal(full_node_action, torch.zeros((1, 1), dtype=torch.float32))
    assert mapped_action.shape == (4,)
    assert mapped_action.dtype == np.float32
    assert np.array_equal(mapped_action, np.zeros(4, dtype=np.float32))
    assert torch.equal(
        selected_full_node_action,
        torch.zeros((1, 1), dtype=torch.float32),
    )


def test_actor_output_shape_is_total_nodes_by_one():
    actor = make_actor(max_action=1.0)
    action = actor(build_active_ev_state())

    assert action.shape == (5, 1)


def test_actor_output_dtype_is_torch_float32():
    actor = make_actor(max_action=1.0)
    action = actor(build_active_ev_state())

    assert action.dtype == torch.float32


def test_actor_preserves_requested_device():
    actor = make_actor(max_action=1.0, device="cpu")
    action = actor(build_active_ev_state())

    assert action.device == torch.device("cpu")


def test_discrete_actions_two_fails_before_actor_or_critic_construction(monkeypatch):
    module = nonnegative_module()
    constructor_calls = []

    with pytest.raises(
        ValueError,
        match="actiongnn_nonnegative supports only discrete_actions=1",
    ):
        module.Actor(
            max_action=1.0,
            fx_node_sizes=FX_NODE_SIZES,
            discrete_actions=2,
            device=torch.device("cpu"),
        )

    class ActorMustNotConstruct:
        def __init__(self, *args, **kwargs):
            constructor_calls.append("actor")
            raise AssertionError("Actor construction should be guarded first")

    class CriticMustNotConstruct:
        def __init__(self, *args, **kwargs):
            constructor_calls.append("critic")
            raise AssertionError("Critic construction should be guarded first")

    monkeypatch.setattr(module, "Actor", ActorMustNotConstruct)
    monkeypatch.setattr(module, "Critic", CriticMustNotConstruct)

    with pytest.raises(
        ValueError,
        match="actiongnn_nonnegative supports only discrete_actions=1",
    ):
        module.TD3_ActionGNN_NonNegative(
            action_dim=4,
            max_action=1.0,
            fx_node_sizes=FX_NODE_SIZES,
            discrete_actions=2,
            device="cpu",
        )

    assert constructor_calls == []


def test_discrete_actions_one_constructs_normally():
    make_actor(discrete_actions=1)
    make_policy(discrete_actions=1)
