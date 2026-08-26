from pathlib import Path
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from utils.transformer_feasibility_projection import (
    project_transformer_feasible_actions,
)


VOLTAGE = 230.0
PHASES = 3.0
MAX_CURRENT = 56.0


def charger_power_kw(current=MAX_CURRENT):
    return current * VOLTAGE * math.sqrt(PHASES) / 1000.0


def build_graph(
    *,
    transformer_specs=((0, 100.0),),
    charger_specs=((0, 0, MAX_CURRENT),),
    ev_specs=((0, 0, 0),),
    sample_node_length=None,
):
    node_types = [0]
    node_counter = 1
    edge_from = []
    edge_to = []
    tr_indexes = []
    cs_indexes = []
    ev_indexes = []
    tr_features = []
    cs_features = []
    ev_features = []
    transformer_node_by_id = {}
    charger_node_by_id = {}

    for transformer_id, transformer_capacity in transformer_specs:
        transformer_node_by_id[int(transformer_id)] = node_counter
        tr_indexes.append(node_counter)
        tr_features.append([float(transformer_capacity), float(transformer_id)])
        node_types.append(1)
        edge_from.extend([0, node_counter])
        edge_to.extend([node_counter, 0])
        node_counter += 1

    for charger_id, transformer_id, max_current in charger_specs:
        charger_node_by_id[int(charger_id)] = node_counter
        cs_indexes.append(node_counter)
        cs_features.append([0.0, float(max_current), 1.0, float(charger_id)])
        node_types.append(2)
        transformer_node = transformer_node_by_id[int(transformer_id)]
        edge_from.extend([transformer_node, node_counter])
        edge_to.extend([node_counter, transformer_node])
        node_counter += 1

    for ev_id, charger_id, transformer_id in ev_specs:
        ev_indexes.append(node_counter)
        ev_features.append(
            [
                0.5,
                0.0,
                1.0,
                float(ev_id),
                float(charger_id),
                float(transformer_id),
            ]
        )
        node_types.append(3)
        charger_node = charger_node_by_id.get(int(charger_id), max(charger_node_by_id.values(), default=0))
        if charger_node:
            edge_from.extend([charger_node, node_counter])
            edge_to.extend([node_counter, charger_node])
        node_counter += 1

    return Data(
        ev_features=np.asarray(ev_features, dtype=float).reshape(-1, 6),
        cs_features=np.asarray(cs_features, dtype=float).reshape(-1, 4),
        tr_features=np.asarray(tr_features, dtype=float).reshape(-1, 2),
        env_features=np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]], dtype=float),
        edge_index=np.asarray([edge_from, edge_to], dtype=np.int64),
        node_types=np.asarray(node_types, dtype=int),
        sample_node_length=sample_node_length or [len(node_types)],
        action_mapper=list(range(len(ev_indexes))),
        ev_indexes=np.asarray(ev_indexes, dtype=int),
        cs_indexes=np.asarray(cs_indexes, dtype=int),
        tr_indexes=np.asarray(tr_indexes, dtype=int),
        env_indexes=np.asarray([0], dtype=int),
    )


def batch_graphs(graphs):
    edge_parts = []
    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = []
    sample_node_length = []
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
        node_offset += int(len(graph.node_types))

    return Data(
        ev_features=np.concatenate([graph.ev_features for graph in graphs], axis=0),
        cs_features=np.concatenate([graph.cs_features for graph in graphs], axis=0),
        tr_features=np.concatenate([graph.tr_features for graph in graphs], axis=0),
        env_features=np.concatenate([graph.env_features for graph in graphs], axis=0),
        edge_index=np.concatenate(edge_parts, axis=1) if edge_parts else np.empty((2, 0), dtype=np.int64),
        node_types=np.concatenate([graph.node_types for graph in graphs], axis=0),
        sample_node_length=sample_node_length,
        action_mapper=[],
        ev_indexes=np.concatenate(ev_indexes).astype(int),
        cs_indexes=np.concatenate(cs_indexes).astype(int),
        tr_indexes=np.concatenate(tr_indexes).astype(int),
        env_indexes=np.concatenate(env_indexes).astype(int),
    )


def candidate_action(state, values, requires_grad=False):
    full_node_action = torch.zeros((int(sum(state.sample_node_length)), 1), dtype=torch.float32)
    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    ev_values = torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)
    if requires_grad:
        ev_values = ev_values.clone().detach().requires_grad_(True)
    full_node_action[active_ev_node_indexes] = ev_values
    return full_node_action, ev_values


def project(state, full_node_action, return_details=True):
    return project_transformer_feasible_actions(
        state=state,
        full_node_action=full_node_action,
        voltage=VOLTAGE,
        phases=PHASES,
        max_action=1.0,
        return_details=return_details,
    )


def transformer_detail(details, graph_index, transformer_id):
    for detail in details["transformers"]:
        if detail["graph_index"] == graph_index and detail["transformer_id"] == transformer_id:
            return detail
    raise AssertionError(f"missing graph {graph_index} transformer {transformer_id}")


def active_values(state, full_node_action):
    return full_node_action[torch.as_tensor(state.ev_indexes, dtype=torch.long), 0]


def assert_non_ev_rows_zero(state, full_node_action):
    non_ev_mask = torch.ones(full_node_action.shape[0], dtype=torch.bool)
    non_ev_mask[torch.as_tensor(state.ev_indexes, dtype=torch.long)] = False
    assert torch.equal(full_node_action[non_ev_mask], torch.zeros_like(full_node_action[non_ev_mask]))


def test_no_active_ev_returns_valid_zero_action():
    state = Data(
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

    safe_action, details = project(state, torch.zeros((1, 1), dtype=torch.float32))

    assert torch.equal(safe_action, torch.zeros((1, 1), dtype=torch.float32))
    assert details["total_action_correction_magnitude"].item() == pytest.approx(0.0)


def test_safe_transformer_leaves_actions_unchanged():
    state = build_graph(
        transformer_specs=((0, 100.0),),
        charger_specs=((0, 0, MAX_CURRENT), (1, 0, MAX_CURRENT)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    full_node_action, _ev_values = candidate_action(state, [0.2, 0.3])

    safe_action, details = project(state, full_node_action)

    assert torch.allclose(safe_action, full_node_action)
    assert transformer_detail(details, 0, 0)["alpha"].item() == pytest.approx(1.0)


def test_overloaded_transformer_scales_all_child_evs_by_one_alpha():
    state = build_graph(
        transformer_specs=((0, 10.0),),
        charger_specs=((0, 0, MAX_CURRENT), (1, 0, MAX_CURRENT)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    full_node_action, _ev_values = candidate_action(state, [1.0, 0.5])

    safe_action, details = project(state, full_node_action)
    detail = transformer_detail(details, 0, 0)

    assert 0.0 < detail["alpha"].item() < 1.0
    assert active_values(state, safe_action)[0] == pytest.approx(detail["alpha"].item())
    assert active_values(state, safe_action)[1] == pytest.approx(0.5 * detail["alpha"].item())
    assert detail["safe_estimated_commanded_power"].item() <= detail["usable_safe_capacity"].item() + 1e-6


def test_within_transformer_action_ratio_is_preserved_when_constrained():
    state = build_graph(
        transformer_specs=((0, 5.0),),
        charger_specs=((0, 0, MAX_CURRENT), (1, 0, MAX_CURRENT)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    full_node_action, _ev_values = candidate_action(state, [0.8, 0.2])

    safe_action, _details = project(state, full_node_action)

    raw_ratio = 0.8 / 0.2
    safe_ratio = (active_values(state, safe_action)[0] / active_values(state, safe_action)[1]).item()
    assert safe_ratio == pytest.approx(raw_ratio, rel=1e-6)


def test_multiple_transformers_only_changes_unsafe_transformer():
    state = build_graph(
        transformer_specs=((0, 5.0), (1, 100.0)),
        charger_specs=((0, 0, MAX_CURRENT), (1, 1, MAX_CURRENT)),
        ev_specs=((0, 0, 0), (1, 1, 1)),
    )
    full_node_action, _ev_values = candidate_action(state, [1.0, 0.4])

    safe_action, details = project(state, full_node_action)

    assert active_values(state, safe_action)[0] < active_values(state, full_node_action)[0]
    assert active_values(state, safe_action)[1] == pytest.approx(active_values(state, full_node_action)[1].item())
    assert transformer_detail(details, 0, 0)["constraint_activated"] is True
    assert transformer_detail(details, 0, 1)["constraint_activated"] is False


def test_repeated_transformer_ids_in_batched_graphs_are_processed_independently():
    first_graph = build_graph(
        transformer_specs=((0, 5.0),),
        charger_specs=((0, 0, MAX_CURRENT),),
        ev_specs=((0, 0, 0),),
    )
    second_graph = build_graph(
        transformer_specs=((0, 100.0),),
        charger_specs=((0, 0, MAX_CURRENT),),
        ev_specs=((0, 0, 0),),
    )
    state = batch_graphs([first_graph, second_graph])
    full_node_action, _ev_values = candidate_action(state, [1.0, 1.0])

    safe_action, details = project(state, full_node_action)

    safe_ev_values = active_values(state, safe_action)
    assert safe_ev_values[0] < 1.0
    assert safe_ev_values[1] == pytest.approx(1.0)
    assert transformer_detail(details, 0, 0)["constraint_activated"] is True
    assert transformer_detail(details, 1, 0)["constraint_activated"] is False


def test_zero_raw_requested_power_uses_alpha_one_without_divide_by_zero():
    state = build_graph()
    full_node_action, _ev_values = candidate_action(state, [0.0])

    safe_action, details = project(state, full_node_action)

    assert torch.equal(safe_action, full_node_action)
    assert transformer_detail(details, 0, 0)["alpha"].item() == pytest.approx(1.0)


def test_capacity_lower_than_rounding_margin_zeroes_child_ev_actions():
    state = build_graph(transformer_specs=((0, 1e-8),))
    full_node_action, _ev_values = candidate_action(state, [1.0])

    safe_action, details = project(state, full_node_action)

    assert active_values(state, safe_action)[0].item() == pytest.approx(0.0)
    assert transformer_detail(details, 0, 0)["usable_safe_capacity"].item() == pytest.approx(0.0)


def test_action_bounds_and_non_ev_rows_are_preserved():
    state = build_graph(
        transformer_specs=((0, 100.0),),
        charger_specs=((0, 0, MAX_CURRENT), (1, 0, MAX_CURRENT)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    full_node_action, _ev_values = candidate_action(state, [1.5, -0.5])

    safe_action, _details = project(state, full_node_action)

    assert torch.all(active_values(state, safe_action) >= 0.0)
    assert torch.all(active_values(state, safe_action) <= 1.0)
    assert_non_ev_rows_zero(state, safe_action)


def test_invalid_ev_to_charger_mapping_fails_clearly():
    state = build_graph(charger_specs=((0, 0, MAX_CURRENT),), ev_specs=((0, 99, 0),))
    full_node_action, _ev_values = candidate_action(state, [0.5])

    with pytest.raises(ValueError, match="charger"):
        project(state, full_node_action)


def test_invalid_ev_to_transformer_mapping_fails_clearly():
    state = build_graph(transformer_specs=((0, 100.0),), ev_specs=((0, 0, 99),))
    full_node_action, _ev_values = candidate_action(state, [0.5])

    with pytest.raises(ValueError, match="transformer"):
        project(state, full_node_action)


def test_five_decimal_rounding_still_respects_physical_transformer_max_power():
    state = build_graph(
        transformer_specs=((0, 10.0),),
        charger_specs=((0, 0, MAX_CURRENT), (1, 0, MAX_CURRENT)),
        ev_specs=((0, 0, 0), (1, 1, 0)),
    )
    full_node_action, _ev_values = candidate_action(state, [1.0, 0.5])

    safe_action, details = project(state, full_node_action)
    rounded_actions = torch.round(active_values(state, safe_action) * 100000.0) / 100000.0
    rounded_power = (rounded_actions * charger_power_kw()).sum().item()

    assert rounded_power <= transformer_detail(details, 0, 0)["physical_max_power"].item() + 1e-9


def test_autograd_unconstrained_projection_has_finite_connected_gradients():
    state = build_graph(transformer_specs=((0, 100.0),), ev_specs=((0, 0, 0), (1, 0, 0)))
    full_node_action, ev_values = candidate_action(state, [0.2, 0.3], requires_grad=True)

    safe_action = project(state, full_node_action, return_details=False)
    loss = (active_values(state, safe_action) * torch.tensor([1.0, 3.0])).sum()
    loss.backward()

    assert ev_values.grad is not None
    assert torch.isfinite(ev_values.grad).all()


def test_autograd_constrained_projection_has_finite_connected_gradients():
    state = build_graph(transformer_specs=((0, 5.0),), ev_specs=((0, 0, 0), (1, 0, 0)))
    full_node_action, ev_values = candidate_action(state, [1.0, 0.5], requires_grad=True)

    safe_action = project(state, full_node_action, return_details=False)
    loss = (active_values(state, safe_action) * torch.tensor([1.0, 3.0])).sum()
    loss.backward()

    assert ev_values.grad is not None
    assert torch.isfinite(ev_values.grad).all()
