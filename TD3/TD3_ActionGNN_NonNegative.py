import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from TD3.TD3_ActionGNN_Controlled import Actor as ControlledActionGNNActor
from TD3.TD3_ActionGNN_Controlled import Critic, resolve_device


CANONICAL_ALGORITHM_LABEL = "actiongnn_nonnegative"
SELECTED_ACTOR_TRANSFORM = "shifted_tanh"
ACTION_DOMAIN_CONTRACT = "nonnegative_ev_rows_exact_zero_nonev_v1"
ACTOR_OUTPUT_TRANSFORM = "shifted_tanh_v1"
ACTOR_OUTPUT_TRANSFORM_FORMULA = "0.5 * max_action * (tanh(z) + 1.0)"
NON_EV_ACTION = 0.0
CHECKPOINT_METADATA_SCHEMA = "actiongnn_nonnegative_checkpoint_v1"
CHECKPOINT_SELECTION_RULE = (
    "model.best selected by strict improvement of scheduled internal eval mean reward "
    "at 5k-step intervals within the 50k training budget; model.last is saved at "
    "50k but is not used for canonical eval30 or diagnostics."
)


def shifted_tanh_nonnegative(raw_logits, max_action):
    return 0.5 * float(max_action) * (torch.tanh(raw_logits) + 1.0)


def checkpoint_metadata_path(checkpoint_prefix):
    return Path(str(checkpoint_prefix) + ".metadata.yaml")


def build_checkpoint_metadata(args, checkpoint_role):
    metadata = {
        "metadata_schema": CHECKPOINT_METADATA_SCHEMA,
        "algorithm": CANONICAL_ALGORITHM_LABEL,
        "action_domain_contract": ACTION_DOMAIN_CONTRACT,
        "actor_output_transform": ACTOR_OUTPUT_TRANSFORM,
        "actor_output_transform_formula": ACTOR_OUTPUT_TRANSFORM_FORMULA,
        "non_ev_action": NON_EV_ACTION,
        "discrete_actions": int(args.discrete_actions),
        "training_budget": int(args.max_timesteps),
        "start_timesteps": int(args.start_timesteps),
        "eval_frequency": int(args.eval_freq),
        "internal_eval_episodes": int(args.eval_episodes),
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "checkpoint_role": checkpoint_role,
    }
    return metadata


def write_checkpoint_metadata(checkpoint_prefix, args, checkpoint_role):
    metadata_path = checkpoint_metadata_path(checkpoint_prefix)
    with metadata_path.open("w") as metadata_file:
        yaml.safe_dump(
            build_checkpoint_metadata(args, checkpoint_role),
            metadata_file,
            sort_keys=False,
        )
    return metadata_path


class Actor(ControlledActionGNNActor):
    def __init__(
        self,
        max_action,
        fx_node_sizes,
        feature_dim=32,
        GNN_hidden_dim=64,
        num_gcn_layers=3,
        discrete_actions=1,
        device=torch.device("cpu"),
    ):
        if discrete_actions != 1:
            raise ValueError(
                "actiongnn_nonnegative supports only discrete_actions=1 scalar PublicPST actions"
            )

        super().__init__(
            max_action,
            fx_node_sizes,
            feature_dim,
            GNN_hidden_dim,
            num_gcn_layers,
            discrete_actions,
            device,
        )
        self.max_action = float(max_action)

    def _active_ev_indexes(self, state):
        return torch.as_tensor(
            state.ev_indexes,
            dtype=torch.long,
            device=self.device,
        ).reshape(-1)

    def forward(self, state):
        embedded_x, edge_index = self._prepare_features(state)

        node_embeddings = F.relu(self.gcn_conv(embedded_x, edge_index))
        for gcn_layer in self.gcn_layers:
            node_embeddings = F.relu(gcn_layer(node_embeddings, edge_index))

        raw_logits = self.gcn_last(node_embeddings, edge_index).reshape(-1, 1)
        shifted_action = shifted_tanh_nonnegative(raw_logits, self.max_action)
        full_node_action = torch.full_like(shifted_action, NON_EV_ACTION)

        active_ev_node_indexes = self._active_ev_indexes(state)
        if active_ev_node_indexes.numel() > 0:
            full_node_action[active_ev_node_indexes] = shifted_action[active_ev_node_indexes]
        return full_node_action


class TD3_ActionGNN_NonNegative(object):
    def __init__(
        self,
        action_dim,
        max_action,
        fx_node_sizes,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        fx_dim=32,
        fx_GNN_hidden_dim=64,
        mlp_hidden_dim=512,
        lr=3e-4,
        discrete_actions=1,
        actor_num_gcn_layers=3,
        critic_num_gcn_layers=3,
        device=None,
        **kwargs,
    ):
        if discrete_actions != 1:
            raise ValueError(
                "actiongnn_nonnegative supports only discrete_actions=1 scalar PublicPST actions"
            )

        self.device = resolve_device(device)
        self.discrete_actions = discrete_actions

        self.actor = Actor(
            max_action,
            fx_node_sizes,
            fx_dim,
            fx_GNN_hidden_dim,
            actor_num_gcn_layers,
            discrete_actions,
            self.device,
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(
            fx_node_sizes,
            fx_dim,
            fx_GNN_hidden_dim,
            mlp_hidden_dim,
            discrete_actions,
            critic_num_gcn_layers,
            self.device,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def _index_tensor(self, values):
        return torch.as_tensor(values, dtype=torch.long, device=self.device).reshape(-1)

    def _zero_non_ev_rows(self, state, full_node_action):
        corrected_action = full_node_action.reshape(-1, self.discrete_actions).clone()
        active_ev_node_indexes = self._index_tensor(state.ev_indexes)
        zeroed_action = torch.zeros_like(corrected_action)
        if active_ev_node_indexes.numel() > 0:
            zeroed_action[active_ev_node_indexes] = corrected_action[active_ev_node_indexes]
        return zeroed_action

    def _add_ev_noise(self, state, full_node_action, noise_scale, noise_clip=None):
        corrected_action = self._zero_non_ev_rows(state, full_node_action)
        active_ev_node_indexes = self._index_tensor(state.ev_indexes)
        if noise_scale == 0 or active_ev_node_indexes.numel() == 0:
            return corrected_action

        ev_noise = torch.randn(
            active_ev_node_indexes.numel(),
            1,
            dtype=corrected_action.dtype,
            device=self.device,
        ) * noise_scale
        if noise_clip is not None:
            ev_noise = ev_noise.clamp(-noise_clip, noise_clip)

        corrected_action[active_ev_node_indexes] = (
            corrected_action[active_ev_node_indexes] + ev_noise
        ).clamp(0.0, self.max_action)
        return self._zero_non_ev_rows(state, corrected_action)

    def _map_to_ev2gym_action(self, state, full_node_action):
        mapped_action_numpy = np.zeros(self.action_dim, dtype=np.float32)
        active_ev_node_indexes = self._index_tensor(state.ev_indexes)
        if active_ev_node_indexes.numel() == 0:
            return mapped_action_numpy

        action_mapper = np.asarray(state.action_mapper, dtype=np.int64)
        if action_mapper.shape[0] != active_ev_node_indexes.numel():
            raise ValueError("state.action_mapper length must match state.ev_indexes length.")

        for active_ev_position, action_index in enumerate(action_mapper):
            mapped_action_numpy[action_index] = (
                full_node_action[active_ev_node_indexes[active_ev_position], 0]
                .detach()
                .cpu()
                .item()
            )
        return mapped_action_numpy

    def select_action(self, state, expl_noise=0, return_mapped_action=False, **kwargs):
        state = state.to(self.device)
        with torch.no_grad():
            full_node_action = self.actor(state)
            full_node_action = self._add_ev_noise(state, full_node_action, expl_noise)

        mapped_action_numpy = self._map_to_ev2gym_action(state, full_node_action)
        if expl_noise != 0 or return_mapped_action:
            return mapped_action_numpy, full_node_action.detach().cpu()
        return mapped_action_numpy

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            next_action = self._add_ev_noise(
                next_state,
                next_action,
                self.policy_noise,
                self.noise_clip,
            )

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for critic_parameter, critic_target_parameter in zip(
                self.critic.parameters(),
                self.critic_target.parameters(),
            ):
                critic_target_parameter.data.copy_(
                    self.tau * critic_parameter.data
                    + (1 - self.tau) * critic_target_parameter.data
                )
            for actor_parameter, actor_target_parameter in zip(
                self.actor.parameters(),
                self.actor_target.parameters(),
            ):
                actor_target_parameter.data.copy_(
                    self.tau * actor_parameter.data
                    + (1 - self.tau) * actor_target_parameter.data
                )

            return critic_loss.item(), actor_loss.item()

        return critic_loss.item(), None

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=self.device)
        )
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", map_location=self.device)
        )
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
