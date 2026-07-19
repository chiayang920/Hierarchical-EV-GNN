from pathlib import Path
import copy
import importlib
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from TD3.TD3_HierarchicalActionGNN import Actor, Critic
from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer, _batch_graphs
from utils.state_public_pst_gnn import PublicPST_GNN


profiler = importlib.import_module("profiling.profile_vectorised_hierarchical_composition")

ATOL = 1e-6
RTOL = 1e-5


def make_actor(seed=0, max_action=10.0, feature_dim=8, hidden_dim=16):
    torch.manual_seed(seed)
    return Actor(
        max_action=max_action,
        fx_node_sizes=PublicPST_GNN.node_sizes,
        feature_dim=feature_dim,
        GNN_hidden_dim=hidden_dim,
        num_gcn_layers=3,
        discrete_actions=1,
        device=torch.device("cpu"),
    )


def make_critic(seed=0, feature_dim=8, hidden_dim=16, mlp_hidden_dim=32):
    torch.manual_seed(seed)
    return Critic(
        fx_node_sizes=PublicPST_GNN.node_sizes,
        feature_dim=feature_dim,
        GNN_hidden_dim=hidden_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        discrete_actions=1,
        num_gcn_layers=3,
        device=torch.device("cpu"),
    )


class SumLossMismatchActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def _full_node_output(self, state, active_ev_value):
        total_nodes = profiler.graph_counts(state)["total_nodes"]
        active_ev_node_indexes = torch.as_tensor(
            state.ev_indexes,
            dtype=torch.long,
            device=active_ev_value.device,
        ).reshape(-1)
        full_node_output = active_ev_value.new_zeros((total_nodes, 1))
        return full_node_output.index_copy(
            0,
            active_ev_node_indexes[:1],
            active_ev_value.reshape(1, 1),
        )

    def _forward_reference(self, state, return_details=False):
        return self._full_node_output(state, self.weight)

    def _forward_vectorised(self, state):
        base_output = self.weight.reshape(1, 1)
        active_ev_value = 2.0 * base_output - base_output.detach()
        return self._full_node_output(state, active_ev_value)


class CriticOnlyMismatchActor(SumLossMismatchActor):
    def _forward_vectorised(self, state):
        base_output = self.weight.reshape(1, 1)
        if isinstance(getattr(state, "ev_features", None), torch.Tensor):
            active_ev_value = 2.0 * base_output - base_output.detach()
        else:
            active_ev_value = base_output
        return self._full_node_output(state, active_ev_value)


class ForwardMismatchActor(SumLossMismatchActor):
    def _forward_vectorised(self, state):
        return self._full_node_output(state, self.weight + 1.0)


class ZeroActionGradientCritic(torch.nn.Module):
    def Q1(self, state, action):
        return (action * 0.0).sum().reshape(1)


class LinearActionGradientCritic(torch.nn.Module):
    def Q1(self, state, action):
        return action.sum().reshape(1)


def build_graph(
    transformer_ids=(0,),
    charger_specs=((0, 0),),
    ev_specs=((0, 0, 0),),
):
    tr_features = [[100.0 + float(transformer_id), float(transformer_id)] for transformer_id in transformer_ids]
    cs_features = [
        [0.0, 32.0 + float(charger_id % 5), 1.0, float(charger_id)]
        for charger_id, _ in charger_specs
    ]
    ev_features = [
        [
            0.5,
            1.0 + float(ev_id),
            float(ev_id % 4),
            float(ev_id),
            float(charger_id),
            float(transformer_id),
        ]
        for ev_id, charger_id, transformer_id in ev_specs
    ]

    transformer_node_indexes = list(range(1, 1 + len(transformer_ids)))
    charger_node_indexes = list(
        range(
            1 + len(transformer_ids),
            1 + len(transformer_ids) + len(charger_specs),
        )
    )
    ev_node_indexes = list(
        range(
            1 + len(transformer_ids) + len(charger_specs),
            1 + len(transformer_ids) + len(charger_specs) + len(ev_specs),
        )
    )
    total_nodes = 1 + len(transformer_ids) + len(charger_specs) + len(ev_specs)

    transformer_node_by_id = {
        int(transformer_id): int(node_index)
        for transformer_id, node_index in zip(transformer_ids, transformer_node_indexes)
    }
    charger_node_by_id = {
        int(charger_id): int(node_index)
        for (charger_id, _), node_index in zip(charger_specs, charger_node_indexes)
    }

    edge_pairs = []
    for transformer_id in transformer_ids:
        transformer_node_index = transformer_node_by_id[int(transformer_id)]
        edge_pairs.extend([(0, transformer_node_index), (transformer_node_index, 0)])
    for (charger_id, transformer_id), charger_node_index in zip(charger_specs, charger_node_indexes):
        parent_transformer_node_index = transformer_node_by_id.get(int(transformer_id))
        if parent_transformer_node_index is not None:
            edge_pairs.extend(
                [
                    (parent_transformer_node_index, charger_node_index),
                    (charger_node_index, parent_transformer_node_index),
                ]
            )
    for (_, charger_id, _), ev_node_index in zip(ev_specs, ev_node_indexes):
        charger_node_index = charger_node_by_id.get(int(charger_id))
        if charger_node_index is not None:
            edge_pairs.extend([(charger_node_index, ev_node_index), (ev_node_index, charger_node_index)])

    if edge_pairs:
        edge_index = np.asarray(edge_pairs, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return Data(
        ev_features=np.asarray(ev_features, dtype=float).reshape(-1, 6),
        cs_features=np.asarray(cs_features, dtype=float).reshape(-1, 4),
        tr_features=np.asarray(tr_features, dtype=float).reshape(-1, 2),
        env_features=np.asarray([[0.0, 0.0, 1.0, 10.0, 0.0]], dtype=float),
        edge_index=edge_index,
        node_types=np.asarray(
            [0] + [1] * len(transformer_ids) + [2] * len(charger_specs) + [3] * len(ev_specs),
            dtype=int,
        ),
        sample_node_length=[total_nodes],
        action_mapper=list(range(len(ev_specs))),
        ev_indexes=np.asarray(ev_node_indexes, dtype=int),
        cs_indexes=np.asarray(charger_node_indexes, dtype=int),
        tr_indexes=np.asarray(transformer_node_indexes, dtype=int),
        env_indexes=np.asarray([0], dtype=int),
    )


def build_no_active_ev_state(total_nodes=1):
    return Data(
        ev_features=np.empty((0, 6), dtype=float),
        cs_features=np.empty((0, 4), dtype=float),
        tr_features=np.empty((0, 2), dtype=float),
        env_features=np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float),
        edge_index=np.empty((2, 0), dtype=np.int64),
        node_types=np.asarray([0] * total_nodes, dtype=int),
        sample_node_length=[total_nodes],
        action_mapper=[],
        ev_indexes=np.asarray([], dtype=int),
        cs_indexes=np.asarray([], dtype=int),
        tr_indexes=np.asarray([], dtype=int),
        env_indexes=np.asarray([0], dtype=int),
    )


def batch_graphs(graphs):
    return _batch_graphs(graphs, device=torch.device("cpu"))


def zero_full_node_action(state):
    return torch.zeros((int(sum(state.sample_node_length)), 1), dtype=torch.float32)


def build_replay_batch(batch_size=64):
    np.random.seed(123)
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=25, max_size=batch_size, device="cpu")
    for graph_index in range(batch_size):
        state = build_graph(
            transformer_ids=(0, 1),
            charger_specs=((0, 0), (1, 1)),
            ev_specs=((0, 0, 0), (1, 1, 1), (2, 0, 0 if graph_index % 2 == 0 else 1)),
        )
        replay_buffer.add(
            state,
            zero_full_node_action(state),
            copy.deepcopy(state),
            reward=float(graph_index % 3),
            done=False,
        )
    np.random.seed(321)
    sampled_state, _, _, _, _ = replay_buffer.sample(batch_size)
    return sampled_state


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def vectorised_compose_source():
    source = (PROJECT_ROOT / "TD3" / "TD3_HierarchicalActionGNN.py").read_text()
    method_start = source.index("    def _compose_full_node_action_vectorised(")
    method_end = source.index("    def _forward_reference(", method_start)
    return source[method_start:method_end]


def assert_reference_vectorised_parity(state, actor=None):
    actor = actor or make_actor()
    reference_output = actor._forward_reference(state, return_details=False)
    vectorised_output = actor._forward_vectorised(state)
    assert_close(vectorised_output, reference_output)
    return reference_output, vectorised_output


def prepared_compose_inputs(actor, state):
    embedded_node_features, edge_index = actor._prepare_features(state)
    return {
        "state": state,
        "node_embeddings": actor._encode_nodes(embedded_node_features, edge_index),
        "ev_features": actor._tensor(state.ev_features, torch.float32),
        "cs_features": actor._tensor(state.cs_features, torch.float32),
        "tr_features": actor._tensor(state.tr_features, torch.float32),
        "active_ev_node_indexes": actor._index_tensor(state.ev_indexes),
        "charger_node_indexes": actor._index_tensor(state.cs_indexes),
        "transformer_node_indexes": actor._index_tensor(state.tr_indexes),
    }


def force_equal_scores_and_open_gate(actor):
    with torch.no_grad():
        actor.transformer_score_head.weight.zero_()
        actor.transformer_score_head.bias.zero_()
        actor.charger_score_head.weight.zero_()
        actor.charger_score_head.bias.zero_()
        actor.ev_gate_head.weight.zero_()
        actor.ev_gate_head.bias.fill_(20.0)


def test_vectorised_source_uses_only_one_graph_level_loop():
    source = vectorised_compose_source()
    for_lines = [
        line.strip()
        for line in source.splitlines()
        if line.lstrip().startswith("for ")
    ]

    assert for_lines == ["for sample_node_length in self._sample_node_lengths(state):"]
    assert ".item(" not in source
    assert "dict(" not in source
    assert ".append(" not in source
    assert "full_node_action[active_ev_node_indexes[graph_ev_positions], 0]" in source
    assert "full_node_action[active_ev_node_indexes[graph_ev_position]" not in source


def test_vectorised_source_does_not_use_dense_selectors_or_matmul():
    source = vectorised_compose_source()

    for forbidden_fragment in (
        "ev_transformer_selector",
        "ev_charger_selector",
        "grouped_charger_scores",
        ".matmul(",
        "torch.matmul",
    ):
        assert forbidden_fragment not in source


def test_vectorised_source_uses_grouped_softmax_for_charger_weights():
    source = vectorised_compose_source()

    assert "pyg_group_softmax(" in source
    assert "index=valid_charger_groups" in source
    assert "num_nodes=graph_transformer_positions.numel()" in source


def test_profiler_sum_loss_gradient_violation_is_diagnostic_only():
    audit_record = profiler.validate_workload(
        actor=SumLossMismatchActor(),
        critic=ZeroActionGradientCritic(),
        state=build_graph(),
        max_action=10.0,
        device="cpu",
        scale="unit",
        seed=0,
        workload_kind="unit",
        load_level="unit",
        batch_size=1,
    )

    sum_loss_audit = audit_record["sum_loss_gradient_audit"]
    assert sum_loss_audit["audit_mode"] == "diagnostic_only"
    assert sum_loss_audit["strict_tolerance_passed"] is False
    assert sum_loss_audit["violating_parameter_names"] == ["weight"]
    assert sum_loss_audit["violating_element_count_by_parameter"] == {"weight": 1}
    assert sum_loss_audit["total_violating_element_count"] == 1


def test_profiler_critic_coupled_gradient_mismatch_still_raises():
    with pytest.raises(AssertionError, match="Gradient mismatch for weight"):
        profiler.validate_workload(
            actor=CriticOnlyMismatchActor(),
            critic=LinearActionGradientCritic(),
            state=build_graph(),
            max_action=10.0,
            device="cpu",
            scale="unit",
            seed=0,
            workload_kind="unit",
            load_level="unit",
            batch_size=1,
        )


def test_profiler_forward_output_contract_mismatch_still_raises():
    with pytest.raises(AssertionError):
        profiler.validate_workload(
            actor=ForwardMismatchActor(),
            critic=ZeroActionGradientCritic(),
            state=build_graph(),
            max_action=10.0,
            device="cpu",
            scale="unit",
            seed=0,
            workload_kind="unit",
            load_level="unit",
            batch_size=1,
        )


def test_profiler_json_summary_exposes_validation_policy():
    summary = profiler.build_json_summary(
        records=[],
        metadata={"run_name": "unit"},
        numerical_equivalence_audit=[],
    )

    assert summary["validation_policy"] == {
        "forward_output_contract": {
            "mode": "hard_fail",
            "description": "Forward parity, output shape/dtype/device, finite values, non-EV zero rows, and action bounds are required.",
        },
        "sum_loss_gradient_audit": {
            "mode": "diagnostic_only",
            "description": "Artificial sum(actor_output) actor-gradient parity is recorded with strict tolerance pass/fail and violation counts but does not stop profiling.",
        },
        "critic_coupled_gradient_audit": {
            "mode": "hard_fail",
            "description": "Critic-coupled TD3 actor-loss gradient parity is the behavioural training-relevant gradient gate.",
        },
    }


def snapshot_parameters(module):
    return {
        parameter_name: parameter.detach().clone()
        for parameter_name, parameter in module.named_parameters()
    }


def assert_module_parameters_unchanged(module, snapshot):
    current_parameters = dict(module.named_parameters())
    assert set(current_parameters) == set(snapshot)
    for parameter_name, expected_parameter in snapshot.items():
        assert_close(current_parameters[parameter_name].detach(), expected_parameter)


def assert_actor_parameter_grads_close(reference_actor, vectorised_actor):
    reference_parameters = dict(reference_actor.named_parameters())
    vectorised_parameters = dict(vectorised_actor.named_parameters())
    assert set(reference_parameters) == set(vectorised_parameters)

    exact_required_names = {
        "transformer_score_head.weight",
        "transformer_score_head.bias",
        "charger_score_head.weight",
        "charger_score_head.bias",
        "ev_gate_head.weight",
        "ev_gate_head.bias",
        "ev_embedding.weight",
        "ev_embedding.bias",
        "cs_embedding.weight",
        "cs_embedding.bias",
        "tr_embedding.weight",
        "tr_embedding.bias",
        "env_embedding.weight",
        "env_embedding.bias",
    }
    assert exact_required_names.issubset(reference_parameters)
    assert any(parameter_name.startswith("gcn_conv.") for parameter_name in reference_parameters)
    assert any(parameter_name.startswith("gcn_layers.") for parameter_name in reference_parameters)

    for parameter_name, reference_parameter in reference_parameters.items():
        vectorised_parameter = vectorised_parameters[parameter_name]
        assert reference_parameter.grad is not None, f"reference gradient missing for {parameter_name}"
        assert vectorised_parameter.grad is not None, f"vectorised gradient missing for {parameter_name}"
        assert torch.isfinite(reference_parameter.grad).all(), parameter_name
        assert torch.isfinite(vectorised_parameter.grad).all(), parameter_name
        assert_close(vectorised_parameter.grad, reference_parameter.grad)


def actor_pair(seed=0):
    reference_actor = make_actor(seed=seed)
    vectorised_actor = make_actor(seed=seed + 1)
    vectorised_actor.load_state_dict(copy.deepcopy(reference_actor.state_dict()))
    return reference_actor, vectorised_actor


def critic_pair(seed=0):
    reference_critic = make_critic(seed=seed)
    vectorised_critic = make_critic(seed=seed + 1)
    vectorised_critic.load_state_dict(copy.deepcopy(reference_critic.state_dict()))
    return reference_critic, vectorised_critic


def run_sum_loss_backward(actor, state, implementation):
    actor.zero_grad(set_to_none=True)
    if implementation == "reference":
        actor_output = actor._forward_reference(state, return_details=False)
    else:
        actor_output = actor._forward_vectorised(state)
    loss = actor_output.sum()
    loss.backward()
    return actor_output.detach(), loss.detach()


def run_critic_loss_backward(actor, critic, state, implementation):
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    if implementation == "reference":
        actor_output = actor._forward_reference(state, return_details=False)
    else:
        actor_output = actor._forward_vectorised(state)
    actor_loss = -critic.Q1(state, actor_output).mean()
    actor_loss.backward()
    return actor_output.detach(), actor_loss.detach()


def assert_sum_loss_gradient_parity(state):
    reference_actor, vectorised_actor = actor_pair(seed=11)
    reference_output, reference_loss = run_sum_loss_backward(reference_actor, state, "reference")
    vectorised_output, vectorised_loss = run_sum_loss_backward(vectorised_actor, state, "vectorised")
    assert_close(vectorised_output, reference_output)
    assert_close(vectorised_loss, reference_loss)
    assert_actor_parameter_grads_close(reference_actor, vectorised_actor)


def assert_critic_loss_gradient_parity(state):
    reference_actor, vectorised_actor = actor_pair(seed=17)
    reference_critic, vectorised_critic = critic_pair(seed=19)
    reference_output, reference_loss = run_critic_loss_backward(
        reference_actor,
        reference_critic,
        state,
        "reference",
    )
    vectorised_output, vectorised_loss = run_critic_loss_backward(
        vectorised_actor,
        vectorised_critic,
        state,
        "vectorised",
    )
    assert_close(vectorised_output, reference_output)
    assert_close(vectorised_loss, reference_loss)
    assert_actor_parameter_grads_close(reference_actor, vectorised_actor)


def graph_slice(state, graph_index):
    graph_start = int(sum(state.sample_node_length[:graph_index]))
    graph_end = graph_start + int(state.sample_node_length[graph_index])
    return slice(graph_start, graph_end)


# 4.1 Dispatch and reference isolation


def test_return_details_false_invokes_vectorised_forward_path():
    actor = make_actor(seed=1)
    state = object()
    sentinel = torch.full((1, 1), 7.0)
    calls = []

    def fake_vectorised(received_state):
        calls.append(received_state)
        return sentinel

    actor._forward_vectorised = fake_vectorised

    result = actor.forward(state, return_details=False)

    assert result is sentinel
    assert calls == [state]


def test_return_details_false_does_not_invoke_reference_composition_path():
    actor = make_actor(seed=2)
    sentinel = torch.full((1, 1), 3.0)

    def fail_reference_composition(*args, **kwargs):
        raise AssertionError("reference composition should not run")

    actor._compose_full_node_action_reference = fail_reference_composition
    actor._forward_vectorised = lambda state: sentinel

    assert actor.forward(object(), return_details=False) is sentinel


def test_return_details_true_invokes_reference_forward_path():
    actor = make_actor(seed=3)
    state = object()
    sentinel = torch.full((1, 1), 5.0)
    details = {"ok": True}
    calls = []

    def fake_reference(received_state, return_details=False):
        calls.append((received_state, return_details))
        return sentinel, details

    actor._forward_reference = fake_reference

    result = actor.forward(state, return_details=True)

    assert result == (sentinel, details)
    assert calls == [(state, True)]


def test_return_details_true_does_not_invoke_vectorised_composition_path():
    actor = make_actor(seed=4)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0),),
    )

    def fail_vectorised_composition(*args, **kwargs):
        raise AssertionError("vectorised composition should not run")

    actor._compose_full_node_action_vectorised = fail_vectorised_composition

    full_node_action, hierarchy_details = actor.forward(state, return_details=True)

    assert full_node_action.shape == (int(sum(state.sample_node_length)), 1)
    assert list(hierarchy_details.keys()) == [
        "per_graph_transformer_ids",
        "per_graph_transformer_weights",
        "per_graph_charger_weights",
        "per_graph_charger_to_transformer_id",
    ]


def test_vectorised_path_does_not_call_empty_details():
    actor = make_actor(seed=5)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0),),
    )

    def fail_empty_details():
        raise AssertionError("_empty_details should not run in vectorised production path")

    actor._empty_details = fail_empty_details

    full_node_action = actor.forward(state, return_details=False)

    assert full_node_action.shape == (int(sum(state.sample_node_length)), 1)


def test_compose_full_node_action_remains_compatibility_wrapper_around_reference_path():
    actor = make_actor(seed=6)
    state = build_graph()
    compose_inputs = prepared_compose_inputs(actor, state)
    sentinel = (
        torch.full((int(sum(state.sample_node_length)), 1), 2.0),
        {"reference": True},
    )
    calls = []

    def fake_reference(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    actor._compose_full_node_action_reference = fake_reference

    result = actor._compose_full_node_action(**compose_inputs)

    assert result is sentinel
    assert len(calls) == 1


def test_reference_hierarchy_detail_keys_remain_exact():
    actor = make_actor(seed=7)
    state = build_graph(
        transformer_ids=(0, 1),
        charger_specs=((0, 0), (1, 1)),
        ev_specs=((0, 0, 0), (1, 1, 1)),
    )

    _, hierarchy_details = actor.forward(state, return_details=True)

    assert list(hierarchy_details.keys()) == [
        "per_graph_transformer_ids",
        "per_graph_transformer_weights",
        "per_graph_charger_weights",
        "per_graph_charger_to_transformer_id",
    ]


# 4.2 Tensor lookup helper


def test_tensor_lookup_helper_contiguous_ids():
    actor = make_actor(seed=8)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([0, 1, 2]),
        torch.tensor([2, 0]),
    )

    torch.testing.assert_close(positions, torch.tensor([2, 0]))
    torch.testing.assert_close(matched, torch.tensor([True, True]))


def test_tensor_lookup_helper_non_contiguous_ids():
    actor = make_actor(seed=9)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([10, 30, 20]),
        torch.tensor([30, 10]),
    )

    torch.testing.assert_close(positions, torch.tensor([1, 0]))
    torch.testing.assert_close(matched, torch.tensor([True, True]))


def test_tensor_lookup_helper_shuffled_reference_ids():
    actor = make_actor(seed=10)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([5, 1, 9]),
        torch.tensor([9, 5, 1]),
    )

    torch.testing.assert_close(positions, torch.tensor([2, 0, 1]))
    torch.testing.assert_close(matched, torch.tensor([True, True, True]))


def test_tensor_lookup_helper_unmatched_query_ids():
    actor = make_actor(seed=11)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([5, 1, 9]),
        torch.tensor([7, 1]),
    )

    torch.testing.assert_close(positions, torch.tensor([-1, 1]))
    torch.testing.assert_close(matched, torch.tensor([False, True]))


def test_tensor_lookup_helper_empty_query_ids():
    actor = make_actor(seed=12)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([5, 1, 9]),
        torch.empty((0,), dtype=torch.long),
    )

    assert positions.shape == (0,)
    assert matched.shape == (0,)


def test_tensor_lookup_helper_empty_reference_ids():
    actor = make_actor(seed=13)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.empty((0,), dtype=torch.long),
        torch.tensor([1, 2]),
    )

    torch.testing.assert_close(positions, torch.tensor([-1, -1]))
    torch.testing.assert_close(matched, torch.tensor([False, False]))


def test_tensor_lookup_helper_repeated_query_ids():
    actor = make_actor(seed=14)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([1, 3, 5]),
        torch.tensor([3, 3, 1]),
    )

    torch.testing.assert_close(positions, torch.tensor([1, 1, 0]))
    torch.testing.assert_close(matched, torch.tensor([True, True, True]))


def test_tensor_lookup_helper_output_dtype():
    actor = make_actor(seed=15)
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([1, 3, 5], dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
    )

    assert positions.dtype == torch.long
    assert matched.dtype == torch.bool


def test_tensor_lookup_helper_output_device():
    actor = make_actor(seed=16)
    query_ids = torch.tensor([3], dtype=torch.long, device=torch.device("cpu"))
    positions, matched = actor._lookup_tensor_id_positions(
        torch.tensor([1, 3, 5], dtype=torch.long, device=torch.device("cpu")),
        query_ids,
    )

    assert positions.device == query_ids.device
    assert matched.device == query_ids.device


# 4.3 Forward numerical parity


def test_forward_parity_one_transformer_one_charger_one_ev():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0),),
            ev_specs=((0, 0, 0),),
        ),
        make_actor(seed=17),
    )


def test_forward_parity_one_transformer_with_multiple_sibling_chargers():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0), (1, 0), (2, 0)),
            ev_specs=((0, 0, 0), (1, 1, 0), (2, 2, 0)),
        ),
        make_actor(seed=18),
    )


def test_forward_parity_multiple_transformers():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0, 1, 2),
            charger_specs=((0, 0), (1, 1), (2, 2)),
            ev_specs=((0, 0, 0), (1, 1, 1), (2, 2, 2)),
        ),
        make_actor(seed=19),
    )


def test_forward_parity_multiple_evs_connected_to_one_charger():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0),),
            ev_specs=((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)),
        ),
        make_actor(seed=20),
    )


def test_forward_parity_multiple_chargers_with_different_ev_counts():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0), (1, 0), (2, 0)),
            ev_specs=((0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 2, 0), (5, 2, 0)),
        ),
        make_actor(seed=21),
    )


def test_forward_parity_two_batched_graphs():
    state = batch_graphs(
        [
            build_graph(transformer_ids=(0,), charger_specs=((0, 0),), ev_specs=((0, 0, 0),)),
            build_graph(transformer_ids=(2,), charger_specs=((3, 2),), ev_specs=((1, 3, 2),)),
        ]
    )
    assert_reference_vectorised_parity(state, make_actor(seed=22))


def test_forward_parity_batched_graphs_with_repeated_local_transformer_ids():
    state = batch_graphs(
        [
            build_graph(transformer_ids=(0, 1), charger_specs=((0, 0), (1, 1)), ev_specs=((0, 0, 0), (1, 1, 1))),
            build_graph(transformer_ids=(0, 1), charger_specs=((0, 0), (1, 1)), ev_specs=((2, 0, 0), (3, 1, 1))),
        ]
    )
    assert_reference_vectorised_parity(state, make_actor(seed=23))


def test_forward_parity_preserves_active_ev_count_cases():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0), (1, 0)),
            ev_specs=((0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0)),
        ),
        make_actor(seed=24),
    )


def test_forward_parity_batched_graphs_with_repeated_local_charger_ids():
    state = batch_graphs(
        [
            build_graph(transformer_ids=(0,), charger_specs=((0, 0), (1, 0)), ev_specs=((0, 0, 0), (1, 1, 0))),
            build_graph(transformer_ids=(0,), charger_specs=((0, 0), (1, 0)), ev_specs=((2, 0, 0), (3, 1, 0))),
        ]
    )
    assert_reference_vectorised_parity(state, make_actor(seed=25))


def test_forward_parity_graph_local_non_contiguous_transformer_ids():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(10, 30),
            charger_specs=((0, 10), (1, 30)),
            ev_specs=((0, 0, 10), (1, 1, 30)),
        ),
        make_actor(seed=26),
    )


def test_forward_parity_graph_local_non_contiguous_charger_ids():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((10, 0), (30, 0)),
            ev_specs=((0, 10, 0), (1, 30, 0)),
        ),
        make_actor(seed=27),
    )


def test_forward_parity_shuffled_transformer_feature_order():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(30, 10, 20),
            charger_specs=((0, 30), (1, 10), (2, 20)),
            ev_specs=((0, 0, 30), (1, 1, 10), (2, 2, 20)),
        ),
        make_actor(seed=28),
    )


def test_forward_parity_shuffled_charger_feature_order():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((20, 0), (10, 0), (30, 0)),
            ev_specs=((0, 20, 0), (1, 10, 0), (2, 30, 0)),
        ),
        make_actor(seed=29),
    )


def test_forward_parity_shuffled_ev_feature_order_contract_consistent():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0, 1),
            charger_specs=((0, 0), (1, 1)),
            ev_specs=((2, 1, 1), (0, 0, 0), (1, 1, 1), (3, 0, 0)),
        ),
        make_actor(seed=30),
    )


def test_forward_parity_batch_size_64_produced_through_replay_buffer():
    assert_reference_vectorised_parity(build_replay_batch(batch_size=64), make_actor(seed=31))


def test_forward_parity_no_active_ev_state():
    assert_reference_vectorised_parity(build_no_active_ev_state(), make_actor(seed=32))


def test_forward_parity_missing_charger_mapping():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0),),
            ev_specs=((0, 99, 0),),
        ),
        make_actor(seed=33),
    )


def test_forward_parity_missing_transformer_mapping():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0),),
            ev_specs=((0, 0, 99),),
        ),
        make_actor(seed=34),
    )


def test_forward_parity_charger_with_no_matching_ev():
    assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0), (1, 0)),
            ev_specs=((0, 0, 0),),
        ),
        make_actor(seed=35),
    )


def test_forward_parity_unmatched_ev_action_remains_zero():
    actor = make_actor(seed=36, max_action=1.0)
    reference_output, vectorised_output = assert_reference_vectorised_parity(
        build_graph(
            transformer_ids=(0,),
            charger_specs=((0, 0),),
            ev_specs=((0, 99, 0),),
        ),
        actor,
    )

    assert torch.equal(reference_output, torch.zeros_like(reference_output))
    assert torch.equal(vectorised_output, torch.zeros_like(vectorised_output))


def test_forward_parity_values_above_upper_clamp_boundary():
    actor = make_actor(seed=37, max_action=0.5)
    force_equal_scores_and_open_gate(actor)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=tuple((ev_id, 0, 0) for ev_id in range(4)),
    )

    reference_output, vectorised_output = assert_reference_vectorised_parity(state, actor)

    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    assert torch.all(reference_output[active_ev_node_indexes] <= actor.max_action)
    assert torch.all(vectorised_output[active_ev_node_indexes] <= actor.max_action)


def test_forward_parity_mixed_clamped_and_unclamped_ev_values():
    actor = make_actor(seed=38, max_action=1.0)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    node_embeddings = torch.zeros((int(sum(state.sample_node_length)), actor.feature_dim))
    node_embeddings[int(state.cs_indexes[0]), 0] = 10.0
    node_embeddings[int(state.cs_indexes[1]), 0] = -10.0
    with torch.no_grad():
        actor.transformer_score_head.weight.zero_()
        actor.transformer_score_head.bias.zero_()
        actor.charger_score_head.weight.zero_()
        actor.charger_score_head.weight[0, 0] = 1.0
        actor.charger_score_head.bias.zero_()
        actor.ev_gate_head.weight.zero_()
        actor.ev_gate_head.bias.fill_(20.0)

    compose_inputs = {
        **prepared_compose_inputs(actor, state),
        "node_embeddings": node_embeddings,
    }
    reference_output, _ = actor._compose_full_node_action_reference(**compose_inputs)
    vectorised_output = actor._compose_full_node_action_vectorised(**compose_inputs)

    assert_close(vectorised_output, reference_output)
    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    assert torch.isclose(reference_output[active_ev_node_indexes[0], 0], torch.tensor(actor.max_action))
    assert reference_output[active_ev_node_indexes[1], 0] < 0.01


def test_forward_parity_deterministic_repeated_forward_calls():
    actor = make_actor(seed=39)
    state = build_graph(
        transformer_ids=(0, 1),
        charger_specs=((0, 0), (1, 1)),
        ev_specs=((0, 0, 0), (1, 1, 1), (2, 0, 0)),
    )

    first_output = actor._forward_vectorised(state)
    second_output = actor._forward_vectorised(state)

    assert_close(second_output, first_output)


# 4.4 Output contract


def test_output_contract_shape_equality():
    reference_output, vectorised_output = assert_reference_vectorised_parity(build_graph(), make_actor(seed=40))
    assert vectorised_output.shape == reference_output.shape


def test_output_contract_dtype_equality():
    reference_output, vectorised_output = assert_reference_vectorised_parity(build_graph(), make_actor(seed=41))
    assert vectorised_output.dtype == reference_output.dtype


def test_output_contract_device_equality():
    reference_output, vectorised_output = assert_reference_vectorised_parity(build_graph(), make_actor(seed=42))
    assert vectorised_output.device == reference_output.device


def test_output_contract_finite_output():
    _, vectorised_output = assert_reference_vectorised_parity(build_graph(), make_actor(seed=43))
    assert torch.isfinite(vectorised_output).all()


def test_output_contract_non_ev_rows_are_exactly_zero():
    state = build_graph(
        transformer_ids=(0, 1),
        charger_specs=((0, 0), (1, 1)),
        ev_specs=((0, 0, 0), (1, 1, 1)),
    )
    _, vectorised_output = assert_reference_vectorised_parity(state, make_actor(seed=44))
    non_ev_mask = torch.ones(int(sum(state.sample_node_length)), dtype=torch.bool)
    non_ev_mask[torch.as_tensor(state.ev_indexes, dtype=torch.long)] = False
    assert torch.equal(vectorised_output[non_ev_mask], torch.zeros_like(vectorised_output[non_ev_mask]))


def test_output_contract_all_ev_actions_greater_than_or_equal_zero():
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    _, vectorised_output = assert_reference_vectorised_parity(state, make_actor(seed=45))
    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    assert torch.all(vectorised_output[active_ev_node_indexes] >= 0.0)


def test_output_contract_all_ev_actions_less_than_or_equal_max_action():
    actor = make_actor(seed=46, max_action=0.25)
    force_equal_scores_and_open_gate(actor)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=tuple((ev_id, 0, 0) for ev_id in range(3)),
    )
    _, vectorised_output = assert_reference_vectorised_parity(state, actor)
    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    assert torch.all(vectorised_output[active_ev_node_indexes] <= actor.max_action)


def test_output_contract_empty_state_output_has_total_nodes_by_one_shape():
    state = build_no_active_ev_state(total_nodes=1)
    _, vectorised_output = assert_reference_vectorised_parity(state, make_actor(seed=47))
    assert vectorised_output.shape == (int(sum(state.sample_node_length)), 1)


# 4.5 Hierarchical allocation invariants


def test_invariant_transformer_weights_sum_to_one_within_each_graph():
    actor = make_actor(seed=48)
    state = build_graph(
        transformer_ids=(0, 1, 2),
        charger_specs=((0, 0), (1, 1), (2, 2)),
        ev_specs=((0, 0, 0), (1, 1, 1), (2, 2, 2)),
    )
    _, details = actor._forward_reference(state, return_details=True)
    for graph_transformer_weights in details["per_graph_transformer_weights"]:
        assert_close(graph_transformer_weights.sum(), torch.tensor(1.0))


def test_invariant_transformer_weights_do_not_normalise_across_graphs():
    actor = make_actor(seed=49)
    state = batch_graphs(
        [
            build_graph(transformer_ids=(0, 1), charger_specs=((0, 0), (1, 1)), ev_specs=((0, 0, 0), (1, 1, 1))),
            build_graph(transformer_ids=(0, 1), charger_specs=((0, 0), (1, 1)), ev_specs=((2, 0, 0), (3, 1, 1))),
        ]
    )
    _, details = actor._forward_reference(state, return_details=True)
    total_transformer_weight = sum(weights.sum() for weights in details["per_graph_transformer_weights"])
    assert_close(total_transformer_weight, torch.tensor(2.0))


def test_invariant_charger_weights_sum_to_one_within_each_transformer_sibling_group():
    actor = make_actor(seed=50)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0), (2, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0), (2, 2, 0)),
    )
    _, details = actor._forward_reference(state, return_details=True)
    graph_charger_weights = details["per_graph_charger_weights"][0]
    charger_to_transformer_id = details["per_graph_charger_to_transformer_id"][0]
    transformer_id = details["per_graph_transformer_ids"][0][0]
    assert_close(graph_charger_weights[charger_to_transformer_id == transformer_id].sum(), torch.tensor(1.0))


def test_invariant_charger_weights_do_not_normalise_across_transformers():
    actor = make_actor(seed=51)
    state = build_graph(
        transformer_ids=(0, 1),
        charger_specs=((0, 0), (1, 0), (2, 1), (3, 1)),
        ev_specs=((0, 0, 0), (1, 1, 0), (2, 2, 1), (3, 3, 1)),
    )
    _, details = actor._forward_reference(state, return_details=True)
    graph_charger_weights = details["per_graph_charger_weights"][0]
    charger_to_transformer_id = details["per_graph_charger_to_transformer_id"][0]
    group_sums = [
        graph_charger_weights[charger_to_transformer_id == transformer_id].sum()
        for transformer_id in details["per_graph_transformer_ids"][0]
    ]
    assert_close(sum(group_sums), torch.tensor(2.0))


def test_invariant_charger_weights_do_not_normalise_across_graphs():
    actor = make_actor(seed=52)
    state = batch_graphs(
        [
            build_graph(transformer_ids=(0,), charger_specs=((0, 0), (1, 0)), ev_specs=((0, 0, 0), (1, 1, 0))),
            build_graph(transformer_ids=(0,), charger_specs=((0, 0), (1, 0)), ev_specs=((2, 0, 0), (3, 1, 0))),
        ]
    )
    _, details = actor._forward_reference(state, return_details=True)
    total_charger_weight = sum(weights.sum() for weights in details["per_graph_charger_weights"])
    assert_close(total_charger_weight, torch.tensor(2.0))


def test_invariant_graph_budget_equals_that_graph_active_ev_count():
    actor = make_actor(seed=53, max_action=100.0)
    force_equal_scores_and_open_gate(actor)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0), (1, 0, 0), (2, 0, 0)),
    )

    vectorised_output = actor._forward_vectorised(state)

    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    expected_budget = torch.full(
        (len(state.ev_indexes), 1),
        float(len(state.ev_indexes)),
        dtype=vectorised_output.dtype,
    )
    assert_close(vectorised_output[active_ev_node_indexes], expected_budget)


def test_invariant_ev_uses_only_its_graph_local_transformer_allocation():
    actor = make_actor(seed=54)
    first_graph = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0),),
    )
    batched_state = batch_graphs(
        [
            first_graph,
            build_graph(
                transformer_ids=(0, 1, 2),
                charger_specs=((0, 0), (1, 1), (2, 2)),
                ev_specs=((1, 0, 0), (2, 1, 1), (3, 2, 2)),
            ),
        ]
    )

    standalone_output = actor._forward_vectorised(first_graph)
    batched_output = actor._forward_vectorised(batched_state)

    assert_close(batched_output[graph_slice(batched_state, 0)], standalone_output)


def test_invariant_ev_uses_only_its_graph_local_charger_allocation():
    actor = make_actor(seed=55)
    first_graph = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0), (1, 0)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    batched_state = batch_graphs(
        [
            first_graph,
            build_graph(
                transformer_ids=(0,),
                charger_specs=((0, 0), (1, 0), (2, 0), (3, 0)),
                ev_specs=((2, 0, 0), (3, 1, 0), (4, 2, 0), (5, 3, 0)),
            ),
        ]
    )

    standalone_output = actor._forward_vectorised(first_graph)
    batched_output = actor._forward_vectorised(batched_state)

    assert_close(batched_output[graph_slice(batched_state, 0)], standalone_output)


def test_invariant_repeated_local_ids_in_separate_graphs_do_not_leak():
    actor = make_actor(seed=56)
    first_graph = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 0, 0),),
    )
    state = batch_graphs(
        [
            first_graph,
            build_graph(transformer_ids=(0,), charger_specs=((0, 0),), ev_specs=((1, 0, 0), (2, 0, 0))),
        ]
    )

    standalone_output = actor._forward_vectorised(first_graph)
    batched_output = actor._forward_vectorised(state)

    assert_close(batched_output[graph_slice(state, 0)], standalone_output)


def test_invariant_unmatched_ids_retain_zero_action():
    actor = make_actor(seed=57)
    state = build_graph(
        transformer_ids=(0,),
        charger_specs=((0, 0),),
        ev_specs=((0, 99, 0), (1, 0, 99)),
    )

    vectorised_output = actor._forward_vectorised(state)

    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    assert torch.equal(vectorised_output[active_ev_node_indexes], torch.zeros_like(vectorised_output[active_ev_node_indexes]))


def test_invariant_one_graph_output_unchanged_when_unrelated_second_graph_appended():
    actor = make_actor(seed=58)
    first_graph = build_graph(
        transformer_ids=(10, 20),
        charger_specs=((30, 10), (40, 20)),
        ev_specs=((0, 30, 10), (1, 40, 20)),
    )
    batched_state = batch_graphs(
        [
            first_graph,
            build_graph(
                transformer_ids=(0, 1, 2),
                charger_specs=((0, 0), (1, 1), (2, 2)),
                ev_specs=((2, 0, 0), (3, 1, 1), (4, 2, 2)),
            ),
        ]
    )

    standalone_output = actor._forward_vectorised(first_graph)
    batched_output = actor._forward_vectorised(batched_state)

    assert_close(batched_output[graph_slice(batched_state, 0)], standalone_output)


# 4.6 Gradient parity using diagnostic sum loss


def test_gradient_parity_sum_loss_single_graph():
    state = build_graph(
        transformer_ids=(0, 1),
        charger_specs=((0, 0), (1, 1)),
        ev_specs=((0, 0, 0), (1, 1, 1), (2, 0, 0)),
    )
    assert_sum_loss_gradient_parity(state)


def test_gradient_parity_sum_loss_two_batched_graphs_with_repeated_local_ids():
    state = batch_graphs(
        [
            build_graph(transformer_ids=(0, 1), charger_specs=((0, 0), (1, 1)), ev_specs=((0, 0, 0), (1, 1, 1))),
            build_graph(transformer_ids=(0, 1), charger_specs=((0, 0), (1, 1)), ev_specs=((2, 0, 0), (3, 1, 1))),
        ]
    )
    assert_sum_loss_gradient_parity(state)


def test_gradient_parity_sum_loss_batch_size_64_replay_graph():
    assert_sum_loss_gradient_parity(build_replay_batch(batch_size=64))


# 4.7 Gradient parity using critic-coupled TD3 actor loss


def test_gradient_parity_critic_coupled_actor_loss_single_graph():
    state = batch_graphs(
        [
            build_graph(
                transformer_ids=(0, 1),
                charger_specs=((0, 0), (1, 1)),
                ev_specs=((0, 0, 0), (1, 1, 1), (2, 0, 0)),
            )
        ]
    )
    assert_critic_loss_gradient_parity(state)


def test_gradient_parity_critic_coupled_actor_loss_batch_size_64_replay_graph():
    assert_critic_loss_gradient_parity(build_replay_batch(batch_size=64))


# 4.8 Optimiser-step parity


def test_optimizer_step_parity_batch_size_64_replay_graph_with_repeated_local_ids():
    state = build_replay_batch(batch_size=64)
    reference_actor, vectorised_actor = actor_pair(seed=61)
    reference_critic, vectorised_critic = critic_pair(seed=62)
    reference_optimizer = torch.optim.Adam(reference_actor.parameters(), lr=3e-4)
    vectorised_optimizer = torch.optim.Adam(vectorised_actor.parameters(), lr=3e-4)
    reference_critic_snapshot = snapshot_parameters(reference_critic)
    vectorised_critic_snapshot = snapshot_parameters(vectorised_critic)

    reference_actor.zero_grad(set_to_none=True)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_output = reference_actor._forward_reference(state, return_details=False)
    reference_loss = -reference_critic.Q1(state, reference_output).mean()
    reference_loss.backward()
    reference_optimizer.step()

    vectorised_actor.zero_grad(set_to_none=True)
    vectorised_optimizer.zero_grad(set_to_none=True)
    vectorised_output = vectorised_actor._forward_vectorised(state)
    vectorised_loss = -vectorised_critic.Q1(state, vectorised_output).mean()
    vectorised_loss.backward()
    vectorised_optimizer.step()

    assert_close(vectorised_output.detach(), reference_output.detach())
    assert_close(vectorised_loss.detach(), reference_loss.detach())

    for (reference_name, reference_parameter), (vectorised_name, vectorised_parameter) in zip(
        reference_actor.named_parameters(),
        vectorised_actor.named_parameters(),
    ):
        assert reference_name == vectorised_name
        assert_close(vectorised_parameter.detach(), reference_parameter.detach())

        reference_state = reference_optimizer.state[reference_parameter]
        vectorised_state = vectorised_optimizer.state[vectorised_parameter]
        assert set(reference_state) == set(vectorised_state)
        assert_close(vectorised_state["step"], reference_state["step"])
        assert_close(vectorised_state["exp_avg"], reference_state["exp_avg"])
        assert_close(vectorised_state["exp_avg_sq"], reference_state["exp_avg_sq"])

    assert_module_parameters_unchanged(reference_critic, reference_critic_snapshot)
    assert_module_parameters_unchanged(vectorised_critic, vectorised_critic_snapshot)
