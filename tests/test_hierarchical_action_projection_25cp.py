"""
The program is designed for testing the hierarchical action projection
module used by the 25 CP EV-GNN prototype.

These tests do not train TD3, do not run EV2Gym, and do not test GNN node
embeddings. They only verify that hierarchical raw actor outputs can be
converted into:

1. differentiable active EV-node actions; and
2. a fixed 25-dimensional EV2Gym-compatible action vector.

arg:

return:
"""

from pathlib import Path
import sys

# Add the EV-GNN project root to Python's module search path.
# This makes the test executable directly via:
#     python tests/test_hierarchical_action_projection_25cp.py
#
# Without this, Python may only search inside EV-GNN/tests/, which means
# project-level imports such as "utils.hierarchical_action_projection"
# cannot be resolved.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch_geometric.data import Data

from utils.hierarchical_action_projection import (
    HierarchicalActionProjection,
    HierarchicalRawAction,
)


def build_mock_public_pst_state() -> Data:
    """
    The program is designed for creating a small mock PublicPST_GNN-style
    graph state.

    The mock graph contains:
        - 2 transformers;
        - 4 chargers;
        - 4 active EVs;
        - a 25-dimensional EV2Gym action space;
        - active EVs mapped into action slots [0, 5, 9, 12].

    This is sufficient to test whether the projection layer can map
    hierarchical actor outputs back into EV2Gym's flat action vector.

    arg:

    return:
        PyTorch Geometric Data object.
    """

    state = Data()

    # In the original PublicPST_GNN-style state, action_mapper tells us
    # which EV2Gym action slot corresponds to each active EV node.
    state.action_mapper = [0, 5, 9, 12]

    # EV feature layout:
    # [soc_flag, energy_exchanged, time_since_arrival, ev_id, cs_id, tr_id]
    #
    # EV0 -> charger 0  -> transformer 0
    # EV1 -> charger 5  -> transformer 0
    # EV2 -> charger 9  -> transformer 1
    # EV3 -> charger 12 -> transformer 1
    state.ev_features = np.array(
        [
            [0.5, 10.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 12.0, 2.0, 0.0, 5.0, 0.0],
            [0.5, 8.0, 1.0, 0.0, 9.0, 1.0],
            [0.5, 9.0, 3.0, 0.0, 12.0, 1.0],
        ],
        dtype=float,
    )

    # Charger feature layout:
    # [min_charge_current, max_charge_current, n_ports, cs_id]
    state.cs_features = np.array(
        [
            [0.0, 32.0, 1.0, 0.0],
            [0.0, 32.0, 1.0, 5.0],
            [0.0, 32.0, 1.0, 9.0],
            [0.0, 32.0, 1.0, 12.0],
        ],
        dtype=float,
    )

    # Transformer feature layout:
    # [max_power, tr_id]
    state.tr_features = np.array(
        [
            [100.0, 0.0],
            [100.0, 1.0],
        ],
        dtype=float,
    )

    # These fields are not directly used by the projection layer, but
    # they mirror the Data object structure returned by the real state function.
    state.ev_indexes = np.array([4, 5, 6, 7])
    state.cs_indexes = np.array([2, 3, 8, 10])
    state.tr_indexes = np.array([1, 11])
    state.env_indexes = np.array([0])

    return state


def test_project_for_training_preserves_shapes_and_gradients() -> None:
    """
    The program is designed for checking whether project_for_training(...)
    returns differentiable tensors with correct shapes.

    arg:

    return:
        None. Raises AssertionError if the projection is invalid.
    """

    state = build_mock_public_pst_state()

    projection = HierarchicalActionProjection(
        action_dim=25,
        min_action=0.0,
        max_action=1.0,
    )

    # Raw actor outputs. In the real hierarchical actor, these tensors will
    # come from GNN node embeddings. Here they are mock tensors.
    transformer_scores = torch.tensor([1.0, 0.0], requires_grad=True)
    charger_scores = torch.tensor([0.5, 0.2, 0.1, 0.3], requires_grad=True)
    ev_ratios = torch.tensor([[0.1], [0.2], [0.3], [0.4]], requires_grad=True)

    # Keep the total budget modest in this test to avoid clipping saturation,
    # so gradients remain visible.
    total_budget = torch.tensor(1.0, requires_grad=True)

    raw_action = HierarchicalRawAction(
        transformer_scores=transformer_scores,
        charger_scores=charger_scores,
        ev_ratios=ev_ratios,
        total_budget=total_budget,
    )

    result = projection.project_for_training(
        state=state,
        raw_action=raw_action,
    )

    assert result.node_action.shape == (4, 1)
    assert result.mapped_action_tensor.shape == (25,)
    assert result.mapped_action_numpy is None

    # All final action values must be inside the valid EV2Gym continuous
    # action range.
    assert torch.all(result.node_action >= 0.0)
    assert torch.all(result.node_action <= 1.0)
    assert torch.all(result.mapped_action_tensor >= 0.0)
    assert torch.all(result.mapped_action_tensor <= 1.0)

    # Only action slots [0, 5, 9, 12] should be non-zero.
    active_slots = [0, 5, 9, 12]
    inactive_slots = [index for index in range(25) if index not in active_slots]

    assert torch.all(result.mapped_action_tensor[active_slots] > 0.0)
    assert torch.allclose(
        result.mapped_action_tensor[inactive_slots],
        torch.zeros(len(inactive_slots)),
    )

    # Transformer weights should sum to 1 globally.
    assert torch.isclose(
        result.transformer_weights.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )

    # Charger weights should sum to 1 within each transformer group.
    # In this mock setup:
    #   transformer 0 owns charger 0 and charger 5
    #   transformer 1 owns charger 9 and charger 12
    assert torch.isclose(
        result.charger_weights[0:2].sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )
    assert torch.isclose(
        result.charger_weights[2:4].sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )

    # Differentiability check:
    # If the projected action participates in a loss, gradients should flow
    # back to raw actor outputs. This confirms that the training path does
    # not accidentally detach the actor output.
    loss = result.mapped_action_tensor.sum()
    loss.backward()

    assert transformer_scores.grad is not None
    assert charger_scores.grad is not None
    assert ev_ratios.grad is not None
    assert total_budget.grad is not None


def test_project_for_env_returns_numpy_action_vector() -> None:
    """
    The program is designed for checking whether project_for_env(...)
    returns the NumPy vector format required by EV2Gym env.step(...).

    arg:

    return:
        None. Raises AssertionError if the projection is invalid.
    """

    state = build_mock_public_pst_state()

    projection = HierarchicalActionProjection(
        action_dim=25,
        min_action=0.0,
        max_action=1.0,
    )

    raw_action = HierarchicalRawAction(
        transformer_scores=torch.tensor([1.0, 0.0]),
        charger_scores=torch.tensor([0.5, 0.2, 0.1, 0.3]),
        ev_ratios=torch.tensor([[0.1], [0.2], [0.3], [0.4]]),
        total_budget=torch.tensor(1.0),
    )

    result = projection.project_for_env(
        state=state,
        raw_action=raw_action,
    )

    assert isinstance(result.mapped_action_numpy, np.ndarray)
    assert result.mapped_action_numpy.shape == (25,)
    assert result.mapped_action_numpy.dtype == np.float32

    assert np.all(result.mapped_action_numpy >= 0.0)
    assert np.all(result.mapped_action_numpy <= 1.0)

    # The EV2Gym vector should contain non-zero actions only at active
    # action slots.
    active_slots = [0, 5, 9, 12]
    inactive_slots = [index for index in range(25) if index not in active_slots]

    assert np.all(result.mapped_action_numpy[active_slots] > 0.0)
    assert np.allclose(result.mapped_action_numpy[inactive_slots], 0.0)


def run_all_tests() -> None:
    """
    The program is designed for running all tests without requiring pytest.

    arg:

    return:
        None.
    """

    test_project_for_training_preserves_shapes_and_gradients()
    test_project_for_env_returns_numpy_action_vector()

    print("All hierarchical action projection tests passed.")


if __name__ == "__main__":
    run_all_tests()