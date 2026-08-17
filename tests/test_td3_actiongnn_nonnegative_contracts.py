import builtins
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from TD3.TD3_ActionGNN_Controlled import Actor as SignedActionGNNActor
from TD3.TD3_ActionGNN_Controlled import TD3_ActionGNN
from utils.replay_buffer_actiongnn import ActionGNN_ReplayBuffer

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


def build_second_active_ev_state():
    state = build_active_ev_state()
    state.action_mapper = [0, 2]
    state.ev_features = np.array(
        [
            [0.75, 3.0, 1.0, 0.0, 0.0, 0.0],
            [0.10, 8.0, 2.0, 1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    return state


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


def make_signed_actor(max_action=1.0):
    return SignedActionGNNActor(
        max_action=max_action,
        fx_node_sizes=FX_NODE_SIZES,
        feature_dim=8,
        GNN_hidden_dim=16,
        num_gcn_layers=3,
        discrete_actions=1,
        device=torch.device("cpu"),
    )


def make_signed_policy(max_action=1.0):
    return TD3_ActionGNN(
        action_dim=4,
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


def non_ev_mask_for(state):
    total_nodes = int(sum(state.sample_node_length))
    mask = torch.ones(total_nodes, dtype=torch.bool)
    mask[torch.as_tensor(state.ev_indexes, dtype=torch.long)] = False
    return mask


def full_node_action_for(state, ev_action_values):
    total_nodes = int(sum(state.sample_node_length))
    full_node_action = torch.zeros((total_nodes, 1), dtype=torch.float32)
    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    full_node_action[active_ev_node_indexes, 0] = torch.as_tensor(
        ev_action_values,
        dtype=torch.float32,
    )
    return full_node_action


def patch_ev_noise(monkeypatch, module, noise_values):
    noise_tensor = torch.as_tensor(noise_values, dtype=torch.float32).reshape(-1, 1)

    def fixed_randn(*shape, dtype=None, device=None, **kwargs):
        requested_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        assert tuple(requested_shape) == tuple(noise_tensor.shape)
        return noise_tensor.to(dtype=dtype or torch.float32, device=device)

    monkeypatch.setattr(module.torch, "randn", fixed_randn)


def test_shifted_tanh_known_logits_fixed_oracle():
    module = nonnegative_module()
    raw_logits = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float32)

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


def test_exploration_noise_is_added_only_to_ev_rows(monkeypatch):
    module = nonnegative_module()
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    base_action = full_node_action_for(state, [0.2, 0.4])
    patch_ev_noise(monkeypatch, module, [0.5, -0.5])

    noisy_action = policy._add_ev_noise(state, base_action, noise_scale=0.2)

    assert torch.allclose(noisy_action[torch.as_tensor(state.ev_indexes), 0], torch.tensor([0.3, 0.3]))
    assert torch.equal(
        noisy_action[non_ev_mask_for(state)],
        torch.zeros((3, 1), dtype=torch.float32),
    )


def test_exploration_noise_post_clamp_bounds_and_non_ev_zero(monkeypatch):
    module = nonnegative_module()
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    base_action = full_node_action_for(state, [0.9, 0.1])
    patch_ev_noise(monkeypatch, module, [10.0, -10.0])

    noisy_action = policy._add_ev_noise(state, base_action, noise_scale=1.0)

    assert torch.allclose(noisy_action[torch.as_tensor(state.ev_indexes), 0], torch.tensor([1.0, 0.0]))
    assert torch.all(noisy_action >= 0.0)
    assert torch.all(noisy_action <= 1.0)
    assert torch.equal(
        noisy_action[non_ev_mask_for(state)],
        torch.zeros((3, 1), dtype=torch.float32),
    )


def test_mapped_action_dtype_shape_bounds_and_action_mapper_values():
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    full_node_action = full_node_action_for(state, [0.25, 0.75])

    mapped_action = policy._map_to_ev2gym_action(state, full_node_action)

    assert mapped_action.shape == (4,)
    assert mapped_action.dtype == np.float32
    assert np.array_equal(mapped_action, np.array([0.0, 0.25, 0.0, 0.75], dtype=np.float32))
    assert np.all(mapped_action >= 0.0)
    assert np.all(mapped_action <= 1.0)


def test_replay_receives_same_corrected_full_node_action_used_for_mapping():
    torch.manual_seed(21)
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    mapped_action, full_node_action = policy.select_action(
        state,
        expl_noise=0,
        return_mapped_action=True,
    )

    assert np.isclose(mapped_action[1], full_node_action[3, 0].item(), atol=1e-6)
    assert np.isclose(mapped_action[3], full_node_action[4, 0].item(), atol=1e-6)

    replay_buffer = ActionGNN_ReplayBuffer(action_dim=4, max_size=2, device="cpu")
    replay_buffer.add(state, full_node_action, state, reward=0.0, done=False)
    sampled_state, sampled_action, next_state, reward, not_done = replay_buffer.sample(1)

    assert sampled_action.shape == (int(sum(sampled_state.sample_node_length)), 1)
    assert torch.allclose(sampled_action.cpu(), full_node_action)
    assert next_state.sample_node_length == sampled_state.sample_node_length
    assert torch.isfinite(reward).all()
    assert torch.isfinite(not_done).all()


def test_current_critic_accepts_corrected_replay_domain_action():
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=4, max_size=2, device="cpu")
    replay_buffer.add(state, full_node_action_for(state, [0.2, 0.8]), state, reward=1.0, done=False)

    sampled_state, sampled_action, next_state, reward, not_done = replay_buffer.sample(1)
    critic_q1, critic_q2 = policy.critic(sampled_state, sampled_action)

    assert critic_q1.shape == (1, 1)
    assert critic_q2.shape == (1, 1)
    assert torch.isfinite(critic_q1).all()
    assert torch.isfinite(critic_q2).all()
    assert next_state.sample_node_length == sampled_state.sample_node_length
    assert torch.isfinite(reward).all()
    assert torch.isfinite(not_done).all()


def test_target_policy_noise_is_added_only_to_ev_rows(monkeypatch):
    module = nonnegative_module()
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    base_action = full_node_action_for(state, [0.5, 0.5])
    patch_ev_noise(monkeypatch, module, [1.0, -1.0])

    target_action = policy._add_ev_noise(
        state,
        base_action,
        noise_scale=0.1,
        noise_clip=0.5,
    )

    assert torch.allclose(target_action[torch.as_tensor(state.ev_indexes), 0], torch.tensor([0.6, 0.4]))
    assert torch.equal(
        target_action[non_ev_mask_for(state)],
        torch.zeros((3, 1), dtype=torch.float32),
    )


def test_target_policy_noise_is_clipped_before_action_clamp(monkeypatch):
    module = nonnegative_module()
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    base_action = full_node_action_for(state, [0.5, 0.5])
    patch_ev_noise(monkeypatch, module, [1.0, -1.0])

    target_action = policy._add_ev_noise(
        state,
        base_action,
        noise_scale=1.0,
        noise_clip=0.1,
    )

    assert torch.allclose(target_action[torch.as_tensor(state.ev_indexes), 0], torch.tensor([0.6, 0.4]))
    assert torch.all(target_action >= 0.0)
    assert torch.all(target_action <= 1.0)


def test_target_critic_receives_corrected_target_domain_action(monkeypatch):
    module = nonnegative_module()
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=4, max_size=2, device="cpu")
    replay_buffer.add(state, full_node_action_for(state, [0.3, 0.7]), state, reward=1.0, done=False)
    sampled_state, sampled_action, next_state, reward, not_done = replay_buffer.sample(1)
    patch_ev_noise(monkeypatch, module, [0.25, -0.25])

    with torch.no_grad():
        target_action = policy.actor_target(next_state)
        target_action = policy._add_ev_noise(
            next_state,
            target_action,
            policy.policy_noise,
            policy.noise_clip,
        )
        target_q1, target_q2 = policy.critic_target(next_state, target_action)

    assert sampled_action.shape == (int(sum(sampled_state.sample_node_length)), 1)
    assert target_action.shape == (int(sum(next_state.sample_node_length)), 1)
    assert torch.equal(
        target_action[non_ev_mask_for(next_state)].cpu(),
        torch.zeros((3, 1), dtype=torch.float32),
    )
    assert target_q1.shape == (1, 1)
    assert target_q2.shape == (1, 1)
    assert torch.isfinite(target_q1).all()
    assert torch.isfinite(target_q2).all()
    assert torch.isfinite(reward).all()
    assert torch.isfinite(not_done).all()


def test_batched_replay_keeps_full_node_shape_without_action_mapper():
    first_state = build_active_ev_state()
    second_state = build_second_active_ev_state()
    replay_buffer = ActionGNN_ReplayBuffer(action_dim=4, max_size=4, device="cpu")
    replay_buffer.add(first_state, full_node_action_for(first_state, [0.2, 0.8]), first_state, reward=1.0, done=False)
    replay_buffer.add(second_state, full_node_action_for(second_state, [0.1, 0.7]), second_state, reward=2.0, done=True)

    sampled_state, sampled_action, next_state, reward, not_done = replay_buffer.sample(2)
    policy = make_policy(max_action=1.0)
    critic_q1, critic_q2 = policy.critic(sampled_state, sampled_action)

    assert sampled_action.shape == (int(sum(sampled_state.sample_node_length)), 1)
    assert not hasattr(sampled_state, "action_mapper")
    assert not hasattr(next_state, "action_mapper")
    assert torch.equal(
        sampled_action[non_ev_mask_for(sampled_state)].cpu(),
        torch.zeros((6, 1), dtype=torch.float32),
    )
    assert critic_q1.shape == (2, 1)
    assert critic_q2.shape == (2, 1)
    assert torch.isfinite(reward).all()
    assert torch.isfinite(not_done).all()


def test_corrected_policy_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(31)
    state = build_active_ev_state()
    policy = make_policy(max_action=1.0)
    before_save_action = policy.actor(state).detach().cpu()
    checkpoint_prefix = tmp_path / "model.best"

    policy.save(str(checkpoint_prefix))

    loaded_policy = make_policy(max_action=1.0)
    loaded_policy.load(str(checkpoint_prefix))
    after_load_action = loaded_policy.actor(state).detach().cpu()

    assert torch.allclose(after_load_action, before_save_action)


def test_legacy_signed_actiongnn_still_emits_signed_values():
    torch.manual_seed(41)
    state = build_active_ev_state()
    signed_actor = make_signed_actor(max_action=1.0)
    with torch.no_grad():
        for signed_parameter in signed_actor.parameters():
            signed_parameter.zero_()
        signed_actor.gcn_last.bias.fill_(-10.0)

    signed_action = signed_actor(state).detach().cpu()

    assert signed_action.shape == (5,)
    assert torch.any(signed_action[torch.as_tensor(state.ev_indexes)] < 0.0)


def test_legacy_signed_actiongnn_exploration_remains_signed(monkeypatch):
    torch.manual_seed(42)
    state = build_active_ev_state()
    signed_policy = make_signed_policy(max_action=1.0)

    def negative_noise_like(action):
        return torch.full_like(action, -10.0)

    monkeypatch.setattr(torch, "randn_like", negative_noise_like)
    mapped_action, full_node_action = signed_policy.select_action(
        state,
        expl_noise=0.2,
        return_mapped_action=True,
    )

    active_ev_node_indexes = torch.as_tensor(state.ev_indexes, dtype=torch.long)
    assert np.any(mapped_action < 0.0)
    assert torch.any(full_node_action[active_ev_node_indexes] < 0.0)


def corrected_protocol_args(**overrides):
    args = SimpleNamespace(
        algorithm="actiongnn_nonnegative",
        config="./config_files/PublicPST_25cp.yaml",
        seed=0,
        device="cpu",
        run_name="corrected_protocol_contract",
        max_timesteps=50000,
        eval_freq=5000,
        eval_episodes=5,
        start_timesteps=1000,
        batch_size=1,
        replay_buffer_size=4,
        discount=0.99,
        tau=0.005,
        expl_noise=0.0,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        lr=3e-4,
        fx_dim=8,
        fx_GNN_hidden_dim=16,
        mlp_hidden_dim=32,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        discrete_actions=1,
        save_dir="./artifacts/experiments",
        log_to_wandb=False,
    )
    for field_name, field_value in overrides.items():
        setattr(args, field_name, field_value)
    return args


def test_training_policy_factory_supports_actiongnn_nonnegative():
    import train_td3_gnn

    policy_class = train_td3_gnn.get_policy_class("actiongnn_nonnegative")

    assert policy_class.__name__ == "TD3_ActionGNN_NonNegative"


def test_corrected_training_protocol_accepts_frozen_contract(tmp_path, monkeypatch):
    import train_td3_gnn
    import utils.replay_buffer_actiongnn as replay_module

    config_path = tmp_path / "formal_config.yaml"
    config_path.write_text("simulation_length: 1\n", encoding="utf-8")
    args = corrected_protocol_args(
        config=str(config_path),
        save_dir=str(tmp_path),
    )
    train_td3_gnn.validate_corrected_training_protocol(args)

    captured = {"return_mapped_action_values": [], "replay_actions": []}
    state = build_no_active_ev_state()

    class OneStepEnv:
        action_space = SimpleNamespace(
            shape=(4,),
            high=np.ones(4, dtype=np.float32),
        )

        def reset(self, seed=None):
            return state, {}

        def step(self, mapped_action):
            return state, 0.0, True, {}

    class RecordingPolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def select_action(self, state, expl_noise=0, return_mapped_action=False, **kwargs):
            captured["return_mapped_action_values"].append(return_mapped_action)
            if return_mapped_action:
                return (
                    np.zeros(4, dtype=np.float32),
                    torch.zeros((1, 1), dtype=torch.float32),
                )
            return np.zeros(4, dtype=np.float32)

        def save(self, checkpoint_prefix):
            captured["saved_checkpoint_prefix"] = checkpoint_prefix

    class RecordingReplayBuffer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def add(self, state, action, next_state, reward, done):
            captured["replay_actions"].append(action)

    monkeypatch.setattr(train_td3_gnn, "parse_args", lambda: args)
    monkeypatch.setattr(train_td3_gnn, "resolve_device", lambda _value: "cpu")
    monkeypatch.setattr(train_td3_gnn, "set_global_seed", lambda _seed: None)
    monkeypatch.setattr(train_td3_gnn, "make_env", lambda *_args, **_kwargs: OneStepEnv())
    monkeypatch.setattr(train_td3_gnn, "get_policy_class", lambda _algorithm: RecordingPolicy)
    monkeypatch.setattr(replay_module, "ActionGNN_ReplayBuffer", RecordingReplayBuffer)
    monkeypatch.setattr(train_td3_gnn, "evaluate_policy", lambda *_args, **_kwargs: {
        "eval/mean_reward": 0.0,
        "eval/std_reward": 0.0,
    })
    monkeypatch.setattr(train_td3_gnn, "range", lambda _stop: builtins.range(1), raising=False)

    train_td3_gnn.main()

    assert captured["return_mapped_action_values"] == [True]
    assert len(captured["replay_actions"]) == 1
    assert captured["replay_actions"][0].shape == (1, 1)


def test_corrected_training_protocol_accepts_formal_75k_budget():
    import train_td3_gnn

    args = corrected_protocol_args(max_timesteps=75000)

    train_td3_gnn.validate_corrected_training_protocol(args)


def test_corrected_training_protocol_accepts_short_smoke_contract():
    import train_td3_gnn

    args = corrected_protocol_args(
        max_timesteps=512,
        eval_freq=256,
        eval_episodes=1,
        start_timesteps=64,
    )

    train_td3_gnn.validate_corrected_training_protocol(args)


def test_corrected_training_protocol_rejects_mismatch_before_environment_creation(monkeypatch):
    import train_td3_gnn

    args = corrected_protocol_args(max_timesteps=49999)
    args.device = "cpu"

    def forbidden_runtime_side_effect(*args, **kwargs):
        raise AssertionError("runtime side effect should not happen before protocol validation")

    monkeypatch.setattr(train_td3_gnn, "parse_args", lambda: args)
    monkeypatch.setattr(train_td3_gnn, "resolve_device", forbidden_runtime_side_effect)
    monkeypatch.setattr(train_td3_gnn, "make_env", forbidden_runtime_side_effect)

    with pytest.raises(ValueError, match="max_timesteps"):
        train_td3_gnn.main()


def test_checkpoint_metadata_uses_shifted_tanh_v1():
    module = nonnegative_module()
    metadata = module.build_checkpoint_metadata(
        corrected_protocol_args(),
        checkpoint_role="best",
    )

    assert metadata["metadata_schema"] == "actiongnn_nonnegative_checkpoint_v1"
    assert metadata["algorithm"] == "actiongnn_nonnegative"
    assert metadata["actor_output_transform"] == "shifted_tanh_v1"
    assert metadata["actor_output_transform_formula"] == "0.5 * max_action * (tanh(z) + 1.0)"
    assert metadata["discrete_actions"] == 1
    assert metadata["training_budget"] == 50000
    assert metadata["start_timesteps"] == 1000
    assert metadata["eval_frequency"] == 5000
    assert metadata["internal_eval_episodes"] == 5
    assert metadata["checkpoint_role"] == "best"
