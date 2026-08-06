import copy

import numpy as np
import torch
import torch.nn.functional as F

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

        mapped_action_numpy = self._map_to_ev2gym_action(state, full_node_action)
        if expl_noise != 0 or return_mapped_action:
            return mapped_action_numpy, full_node_action.detach().cpu()
        return mapped_action_numpy
