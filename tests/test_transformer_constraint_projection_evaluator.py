import importlib

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from utils.transformer_feasibility_projection import project_transformer_feasible_actions


def build_graph(*, transformer_capacity=100.0, action_mapper=(0, 1)):
    return Data(
        ev_features=np.asarray(
            [
                [0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 1.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cs_features=np.asarray(
            [[0.0, 56.0, 1.0, 0.0], [0.0, 56.0, 1.0, 1.0]],
            dtype=float,
        ),
        tr_features=np.asarray([[transformer_capacity, 0.0]], dtype=float),
        env_features=np.asarray([[0.0, 0.0, 1.0, 10.0, 0.0]], dtype=float),
        edge_index=np.asarray(
            [[0, 1, 1, 2, 2, 3, 1, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 1, 5, 4]],
            dtype=np.int64,
        ),
        node_types=np.asarray([0, 1, 2, 3, 2, 3], dtype=int),
        sample_node_length=[6],
        action_mapper=list(action_mapper),
        ev_indexes=np.asarray([3, 5], dtype=int),
        cs_indexes=np.asarray([2, 4], dtype=int),
        tr_indexes=np.asarray([1], dtype=int),
        env_indexes=np.asarray([0], dtype=int),
    )


def unsafe_full_action(state, values=(1.0, 0.8)):
    action = torch.zeros((int(sum(state.sample_node_length)), 1), dtype=torch.float32)
    action[torch.as_tensor(state.ev_indexes, dtype=torch.long), 0] = torch.tensor(values)
    return action


def _policy():
    from evaluate_td3_gnn import create_policy

    return create_policy(
        algorithm="hierarchical_transformer_constraint",
        action_dim=4,
        max_action=1.0,
        device="cpu",
        checkpoint_kwargs={
            "transformer_voltage": 230.0,
            "transformer_phases": 3.0,
            "fx_dim": 8,
            "fx_GNN_hidden_dim": 16,
            "mlp_hidden_dim": 32,
            "actor_num_gcn_layers": 3,
            "critic_num_gcn_layers": 3,
            "discrete_actions": 1,
        },
    )


def test_projection_diagnostic_action_matches_normal_deterministic_select_action(monkeypatch):
    evaluator = importlib.import_module("evaluate_transformer_constraint_projection_diagnostics")
    state = build_graph(transformer_capacity=5.0, action_mapper=(2, 0))
    unsafe_actor_action = unsafe_full_action(state, (1.0, 1.0))
    policy = _policy()
    monkeypatch.setattr(policy.actor, "forward", lambda _state: unsafe_actor_action.clone())

    diagnostic = evaluator.select_action_with_projection_diagnostics(policy, state)
    normal_mapped, normal_safe = policy.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )
    expected_safe = project_transformer_feasible_actions(
        state=state,
        full_node_action=unsafe_actor_action,
        voltage=230.0,
        phases=3.0,
        max_action=1.0,
    )

    assert np.allclose(diagnostic.mapped_action, normal_mapped)
    assert diagnostic.safe_action.equal(normal_safe)
    assert diagnostic.safe_action.equal(expected_safe)
    assert any(row["constraint_activated"] for row in diagnostic.transformer_rows)


def test_projection_diagnostic_rejects_unconstrained_policy():
    evaluator = importlib.import_module("evaluate_transformer_constraint_projection_diagnostics")

    class NoProjectionPolicy:
        pass

    with pytest.raises(ValueError, match="hierarchical_transformer_constraint"):
        evaluator.select_action_with_projection_diagnostics(NoProjectionPolicy(), build_graph())
