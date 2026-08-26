from pathlib import Path
from types import MethodType
import copy
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import Data

import train_td3_gnn
from TD3.TD3_HierarchicalActionGNN import TD3_HierarchicalActionGNN
from TD3.TD3_HierarchicalActionGNN_TransformerConstraint import (
    TD3_HierarchicalActionGNN_TransformerConstraint,
)
from utils.ev2gym_training_utils import make_env, normalise_step_result, reset_env
from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer
from utils.state_public_pst_gnn import PublicPST_GNN
from utils.transformer_feasibility_projection import (
    project_transformer_feasible_actions,
    validate_transformer_constraint_config,
)


FX_NODE_SIZES = {"ev": 6, "cs": 4, "tr": 2, "env": 5}
CONFIG_FILES = (
    PROJECT_ROOT / "config_files" / "PublicPST_25cp.yaml",
    PROJECT_ROOT / "config_files" / "PublicPST_100.yaml",
    PROJECT_ROOT / "config_files" / "PublicPST_500.yaml",
    PROJECT_ROOT / "config_files" / "PublicPST_1000.yaml",
)


def build_graph(
    *,
    transformer_capacity=100.0,
    action_mapper=(0, 1),
    charger_max_current=56.0,
):
    return Data(
        ev_features=np.asarray(
            [
                [0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 1.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cs_features=np.asarray(
            [
                [0.0, charger_max_current, 1.0, 0.0],
                [0.0, charger_max_current, 1.0, 1.0],
            ],
            dtype=float,
        ),
        tr_features=np.asarray([[transformer_capacity, 0.0]], dtype=float),
        env_features=np.asarray([[0.0, 0.0, 1.0, 10.0, 0.0]], dtype=float),
        edge_index=np.asarray(
            [
                [0, 1, 1, 2, 2, 3, 1, 4, 4, 5],
                [1, 0, 2, 1, 3, 2, 4, 1, 5, 4],
            ],
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


def make_policy(action_dim=4, transformer_capacity=100.0, policy_freq=2, policy_noise=0.0):
    return TD3_HierarchicalActionGNN_TransformerConstraint(
        action_dim=action_dim,
        max_action=1.0,
        fx_node_sizes=FX_NODE_SIZES,
        fx_dim=8,
        fx_GNN_hidden_dim=16,
        mlp_hidden_dim=32,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        discrete_actions=1,
        device="cpu",
        transformer_voltage=230.0,
        transformer_phases=3.0,
        policy_freq=policy_freq,
        policy_noise=policy_noise,
        noise_clip=policy_noise,
    )


def unsafe_full_action(state, values=(1.0, 0.8)):
    full_node_action = torch.zeros((int(sum(state.sample_node_length)), 1), dtype=torch.float32)
    full_node_action[torch.as_tensor(state.ev_indexes, dtype=torch.long), 0] = torch.tensor(values)
    return full_node_action


def active_values(state, full_node_action):
    return full_node_action[torch.as_tensor(state.ev_indexes, dtype=torch.long), 0]


def test_inactive_constraint_matches_original_full_hierarchical_action():
    torch.manual_seed(11)
    state = build_graph(transformer_capacity=1000.0, action_mapper=(2, 3))
    original = TD3_HierarchicalActionGNN(
        action_dim=4,
        max_action=1.0,
        fx_node_sizes=FX_NODE_SIZES,
        fx_dim=8,
        fx_GNN_hidden_dim=16,
        mlp_hidden_dim=32,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        discrete_actions=1,
        device="cpu",
    )
    constrained = make_policy(action_dim=4)
    constrained.actor.load_state_dict(original.actor.state_dict())

    original_mapped, original_node_action = original.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )
    constrained_mapped, constrained_node_action = constrained.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )

    assert np.allclose(constrained_mapped, original_mapped, atol=1e-6)
    assert torch.allclose(constrained_node_action, original_node_action, atol=1e-6)


def test_select_action_returns_same_safe_action_used_for_ev2gym_mapping():
    torch.manual_seed(12)
    state = build_graph(transformer_capacity=5.0, action_mapper=(2, 0))
    policy = make_policy(action_dim=4)

    mapped_action, safe_node_action = policy.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )
    expected_safe_action = project_transformer_feasible_actions(
        state=state,
        full_node_action=policy.actor(state),
        voltage=230.0,
        phases=3.0,
        max_action=1.0,
    )

    assert torch.allclose(safe_node_action, expected_safe_action.detach().cpu(), atol=1e-6)
    assert mapped_action[2] == pytest.approx(safe_node_action[3, 0].item(), abs=1e-6)
    assert mapped_action[0] == pytest.approx(safe_node_action[5, 0].item(), abs=1e-6)


def test_replay_transition_stores_safe_executed_action():
    state = build_graph(transformer_capacity=5.0)
    policy = make_policy(action_dim=4)
    _mapped_action, safe_node_action = policy.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=4, max_size=2, device="cpu")

    replay_buffer.add(state, safe_node_action, state, reward=0.0, done=False)
    sampled_state, sampled_action, _next_state, _reward, _not_done = replay_buffer.sample(1)

    expected_safe_action = project_transformer_feasible_actions(
        state=sampled_state,
        full_node_action=sampled_action,
        voltage=230.0,
        phases=3.0,
        max_action=1.0,
    )
    assert torch.allclose(sampled_action, expected_safe_action, atol=1e-6)


def test_target_critic_receives_noisy_then_projected_safe_target_action(monkeypatch):
    state = build_graph(transformer_capacity=5.0)
    policy = make_policy(action_dim=4, policy_freq=100)
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=4, max_size=2, device="cpu")
    replay_buffer.add(state, unsafe_full_action(state, (0.1, 0.2)), state, reward=0.0, done=False)
    forced_noisy_action = unsafe_full_action(state, (1.0, 1.0))
    captured = {}

    def fake_target_actor(_state):
        return torch.zeros_like(forced_noisy_action)

    def fake_add_ev_noise(_state, _full_node_action, _noise_scale, _noise_clip=None):
        return forced_noisy_action.clone()

    def recording_target_forward(self, next_state, next_action):
        captured["target_action"] = next_action.detach().clone()
        return torch.zeros((1, 1)), torch.zeros((1, 1))

    monkeypatch.setattr(policy.actor_target, "forward", fake_target_actor)
    monkeypatch.setattr(policy, "_add_ev_noise", fake_add_ev_noise)
    monkeypatch.setattr(policy.critic_target, "forward", MethodType(recording_target_forward, policy.critic_target))

    policy.train(replay_buffer, batch_size=1)
    expected_safe_target = project_transformer_feasible_actions(
        state=state,
        full_node_action=forced_noisy_action,
        voltage=230.0,
        phases=3.0,
        max_action=1.0,
    )

    assert torch.allclose(captured["target_action"], expected_safe_target, atol=1e-6)


def test_actor_loss_uses_projected_action_and_keeps_actor_gradients(monkeypatch):
    torch.manual_seed(13)
    state = build_graph(transformer_capacity=5.0)
    policy = make_policy(action_dim=4, policy_freq=1)
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=4, max_size=2, device="cpu")
    replay_buffer.add(state, unsafe_full_action(state, (0.2, 0.4)), state, reward=0.0, done=False)
    captured = {}

    def recording_q1(self, sampled_state, actor_action):
        captured["actor_action"] = actor_action
        return actor_action[torch.as_tensor(sampled_state.ev_indexes, dtype=torch.long)].sum().reshape(1, 1)

    monkeypatch.setattr(policy.critic, "Q1", MethodType(recording_q1, policy.critic))

    expected_safe_actor_action = project_transformer_feasible_actions(
        state=state,
        full_node_action=policy.actor(state),
        voltage=230.0,
        phases=3.0,
        max_action=1.0,
    )
    _critic_loss, actor_loss = policy.train(replay_buffer, batch_size=1)

    assert actor_loss is not None
    assert torch.allclose(captured["actor_action"], expected_safe_actor_action, atol=1e-6)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.actor.parameters()
    )


def test_deterministic_evaluation_keeps_constraint_enabled():
    torch.manual_seed(14)
    state = build_graph(transformer_capacity=5.0)
    policy = make_policy(action_dim=4)

    raw_actor_action = policy.actor(state).detach()
    _mapped_action, safe_node_action = policy.select_action(
        state,
        expl_noise=0.0,
        return_mapped_action=True,
    )

    assert torch.all(active_values(state, safe_node_action) <= active_values(state, raw_actor_action) + 1e-6)
    assert torch.any(active_values(state, safe_node_action) < active_values(state, raw_actor_action) - 1e-6)


def test_train_td3_gnn_registers_new_algorithm_and_preserves_existing_labels():
    assert train_td3_gnn.get_policy_class("hierarchical_transformer_constraint").__name__ == (
        "TD3_HierarchicalActionGNN_TransformerConstraint"
    )
    assert train_td3_gnn.get_policy_class("actiongnn").__name__ == "TD3_ActionGNN"
    assert train_td3_gnn.get_policy_class("actiongnn_nonnegative").__name__ == "TD3_ActionGNN_NonNegative"
    assert train_td3_gnn.get_policy_class("hierarchical").__name__ == "TD3_HierarchicalActionGNN"
    assert train_td3_gnn.get_policy_class("hierarchical_transformer_ev").__name__ == "TD3_TransformerEVActionGNN"


def test_standalone_evaluator_registers_constraint_policy_with_checkpoint_kwargs():
    evaluator_module = importlib.import_module("evaluate_td3_gnn")

    assert "hierarchical_transformer_constraint" in evaluator_module.ALGORITHM_CHOICES
    policy = evaluator_module.create_policy(
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

    assert policy.__class__.__name__ == "TD3_HierarchicalActionGNN_TransformerConstraint"


def test_standalone_evaluator_deterministic_selection_executes_projected_safe_action(monkeypatch):
    evaluator_module = importlib.import_module("evaluate_td3_gnn")
    state = build_graph(transformer_capacity=5.0, action_mapper=(2, 0))
    unsafe_actor_action = unsafe_full_action(state, (1.0, 1.0))
    policy = evaluator_module.create_policy(
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
    monkeypatch.setattr(policy.actor, "forward", lambda _state: unsafe_actor_action.clone())

    mapped_action = evaluator_module.select_mapped_action(
        policy=policy,
        state=state,
        deterministic=True,
        eval_expl_noise=0.0,
    )
    safe_action = project_transformer_feasible_actions(
        state=state,
        full_node_action=unsafe_actor_action,
        voltage=230.0,
        phases=3.0,
        max_action=1.0,
    )

    assert mapped_action[2] == pytest.approx(safe_action[3, 0].item(), abs=1e-6)
    assert mapped_action[0] == pytest.approx(safe_action[5, 0].item(), abs=1e-6)
    assert np.any(mapped_action < 1.0)


def test_diagnostic_evaluator_accepts_constraint_label_and_uses_canonical_factory(monkeypatch):
    diagnostic_module = importlib.import_module("evaluate_td3_gnn_infrastructure_diagnostics")
    state = build_graph(transformer_capacity=5.0, action_mapper=(2, 0))
    unsafe_actor_action = unsafe_full_action(state, (1.0, 1.0))

    args = diagnostic_module.parse_args(
        [
            "--algorithm",
            "hierarchical_transformer_constraint",
            "--scale",
            "25cp",
            "--config",
            "config_files/PublicPST_25cp.yaml",
            "--checkpoint",
            "model.best",
            "--output_dir",
            "diagnostics",
        ]
    )
    policy = diagnostic_module.create_policy(
        algorithm=args.algorithm,
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
    monkeypatch.setattr(policy.actor, "forward", lambda _state: unsafe_actor_action.clone())

    mapped_action = diagnostic_module.select_mapped_action(
        policy=policy,
        state=state,
        deterministic=True,
        eval_expl_noise=0.0,
    )

    assert args.algorithm == "hierarchical_transformer_constraint"
    assert policy.__class__.__name__ == "TD3_HierarchicalActionGNN_TransformerConstraint"
    assert np.any(mapped_action < 1.0)


def test_formal_public_pst_configs_pass_transformer_constraint_validation():
    for config_file in CONFIG_FILES:
        with config_file.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        electrical = validate_transformer_constraint_config(config)
        assert electrical["transformer_voltage"] == pytest.approx(230.0)
        assert electrical["transformer_phases"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "patch",
    [
        {"v2g_enabled": True},
        {"inflexible_loads": {"include": True}},
        {"solar_power": {"include": True}},
        {"demand_response": {"include": True}},
        {"charging_network_topology": "custom.json"},
        {"charging_station": {"voltage": 0}},
        {"charging_station": {"phases": 0}},
        {"charging_station": {"max_charge_current": 0}},
    ],
)
def test_unsupported_config_features_are_rejected_for_new_algorithm_only(patch):
    with CONFIG_FILES[0].open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    unsupported = copy.deepcopy(config)
    for key, value in patch.items():
        if isinstance(value, dict):
            unsupported[key].update(value)
        else:
            unsupported[key] = value

    with pytest.raises(ValueError):
        validate_transformer_constraint_config(unsupported)

    args = type("Args", (), {"algorithm": "hierarchical", "max_timesteps": 1})()
    assert train_td3_gnn.validate_transformer_constraint_training_contract(args, unsupported) == {}


def test_real_ev2gym_transformer_invariant_holds_for_executed_steps():
    torch.manual_seed(15)
    np.random.seed(15)
    env = make_env(str(CONFIG_FILES[0]), seed=15)
    state, _reset_info = reset_env(env, seed=15)
    with CONFIG_FILES[0].open("r", encoding="utf-8") as handle:
        electrical = validate_transformer_constraint_config(yaml.safe_load(handle))
    policy = TD3_HierarchicalActionGNN_TransformerConstraint(
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        fx_node_sizes=PublicPST_GNN.node_sizes,
        fx_dim=8,
        fx_GNN_hidden_dim=16,
        mlp_hidden_dim=32,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        discrete_actions=1,
        device="cpu",
        **electrical,
    )

    inspected_steps = 0
    for _step in range(16):
        mapped_action, _safe_node_action = policy.select_action(
            state,
            expl_noise=0.5,
            return_mapped_action=True,
        )
        state, _reward, done, _stats = normalise_step_result(env.step(mapped_action))
        inspected_steps += 1
        inspected_step_index = env.current_step - 1
        for transformer in env.transformers:
            assert transformer.current_power <= transformer.max_power[inspected_step_index] + 1e-9
        if done:
            break

    assert inspected_steps > 0


def test_real_ev2gym_forced_activation_projects_before_env_step():
    env = make_env(str(CONFIG_FILES[0]), seed=0)
    state, _reset_info = reset_env(env, seed=0)
    with CONFIG_FILES[0].open("r", encoding="utf-8") as handle:
        electrical = validate_transformer_constraint_config(yaml.safe_load(handle))
    policy = TD3_HierarchicalActionGNN_TransformerConstraint(
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        fx_node_sizes=PublicPST_GNN.node_sizes,
        fx_dim=8,
        fx_GNN_hidden_dim=16,
        mlp_hidden_dim=32,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        discrete_actions=1,
        device="cpu",
        **electrical,
    )
    zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)

    for _step in range(env.simulation_length):
        if len(state.ev_indexes) > 0:
            candidate_action = torch.zeros((int(sum(state.sample_node_length)), 1), dtype=torch.float32)
            active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
            candidate_action[active_ev_node_indexes, 0] = 1.0
            safe_action, details = project_transformer_feasible_actions(
                state=state,
                full_node_action=candidate_action,
                voltage=electrical["transformer_voltage"],
                phases=electrical["transformer_phases"],
                max_action=1.0,
                return_details=True,
            )
            activated_details = [
                detail for detail in details["transformers"] if detail["constraint_activated"]
            ]
            if activated_details:
                break
        state, _reward, done, _stats = normalise_step_result(env.step(zero_action.copy()))
        if done:
            pytest.fail("PublicPST seed 0 ended before a forced-activation state was reached.")
    else:
        pytest.fail("PublicPST seed 0 did not reach a forced-activation state.")

    activated_transformer_ids = {
        detail["transformer_id"] for detail in activated_details
    }
    assert any(
        detail["raw_estimated_commanded_power"].item()
        > detail["usable_safe_capacity"].item()
        for detail in activated_details
    )
    assert torch.any(safe_action[active_ev_node_indexes, 0] < candidate_action[active_ev_node_indexes, 0])

    mapped_action = policy._map_to_ev2gym_action(state.to(policy.device), safe_action)
    for active_ev_position, action_slot in enumerate(state.action_mapper):
        ev_node_index = active_ev_node_indexes[active_ev_position]
        assert mapped_action[action_slot] == pytest.approx(
            safe_action[ev_node_index, 0].item(),
            abs=1e-6,
        )

    _next_state, _reward, _done, _stats = normalise_step_result(env.step(mapped_action))
    inspected_step_index = env.current_step - 1
    for transformer in env.transformers:
        assert transformer.current_power <= transformer.max_power[inspected_step_index] + 1e-9
        if transformer.id in activated_transformer_ids:
            assert transformer.current_power <= transformer.max_power[inspected_step_index] + 1e-9
