import copy

import numpy as np
import torch
import torch.nn.functional as F

from TD3.TD3_ActionGNN_Controlled import Actor as ActionGNNActor, Critic, resolve_device


class Actor(ActionGNNActor):
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
            raise ValueError("TD3_HierarchicalActionGNN currently supports discrete_actions=1 only.")

        self.max_action = float(max_action)
        super().__init__(
            max_action,
            fx_node_sizes,
            feature_dim,
            GNN_hidden_dim,
            num_gcn_layers,
            discrete_actions,
            device,
        )
        del self.gcn_last

        self.transformer_score_head = torch.nn.Linear(feature_dim, 1)
        self.charger_score_head = torch.nn.Linear(feature_dim, 1)
        self.ev_gate_head = torch.nn.Linear(feature_dim, 1)

    def _tensor(self, value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _index_tensor(self, value):
        return self._tensor(value, torch.long).reshape(-1)

    def _sample_node_lengths(self, state):
        return [int(sample_node_length) for sample_node_length in state.sample_node_length]

    def _encode_nodes(self, embedded_node_features, edge_index):
        node_embeddings = F.relu(self.gcn_conv(embedded_node_features, edge_index))
        for gcn_layer in self.gcn_layers:
            node_embeddings = F.relu(gcn_layer(node_embeddings, edge_index))
        return node_embeddings

    def _infer_charger_to_transformer_id(
        self,
        graph_charger_ids,
        graph_ev_to_charger_id,
        graph_ev_to_transformer_id,
    ):
        charger_transformer_ids = []
        for charger_id in graph_charger_ids:
            matching_ev_mask = graph_ev_to_charger_id == charger_id
            if not torch.any(matching_ev_mask):
                charger_transformer_ids.append(torch.full_like(charger_id, -1))
                continue
            transformer_ids_for_charger = graph_ev_to_transformer_id[matching_ev_mask]
            charger_transformer_ids.append(torch.unique(transformer_ids_for_charger)[0])
        return torch.stack(charger_transformer_ids).long()

    def _empty_details(self):
        return {
            "per_graph_transformer_ids": [],
            "per_graph_transformer_weights": [],
            "per_graph_charger_weights": [],
            "per_graph_charger_to_transformer_id": [],
        }

    def _compose_full_node_action(
        self,
        state,
        node_embeddings,
        ev_features,
        cs_features,
        tr_features,
        active_ev_node_indexes,
        charger_node_indexes,
        transformer_node_indexes,
    ):
        total_nodes = node_embeddings.shape[0]
        full_node_action = torch.zeros((total_nodes, 1), dtype=torch.float32, device=self.device)
        hierarchy_details = self._empty_details()

        if active_ev_node_indexes.numel() == 0:
            return full_node_action, hierarchy_details

        transformer_scores = self.transformer_score_head(
            node_embeddings[transformer_node_indexes]
        ).reshape(-1)
        charger_scores = self.charger_score_head(node_embeddings[charger_node_indexes]).reshape(-1)
        ev_action_gate = torch.sigmoid(
            self.ev_gate_head(node_embeddings[active_ev_node_indexes]).reshape(-1)
        )

        graph_node_start = 0
        for sample_node_length in self._sample_node_lengths(state):
            graph_node_end = graph_node_start + sample_node_length

            graph_ev_mask = (
                (active_ev_node_indexes >= graph_node_start)
                & (active_ev_node_indexes < graph_node_end)
            )
            graph_charger_mask = (
                (charger_node_indexes >= graph_node_start)
                & (charger_node_indexes < graph_node_end)
            )
            graph_transformer_mask = (
                (transformer_node_indexes >= graph_node_start)
                & (transformer_node_indexes < graph_node_end)
            )

            graph_ev_positions = torch.nonzero(graph_ev_mask, as_tuple=False).reshape(-1)
            graph_charger_positions = torch.nonzero(graph_charger_mask, as_tuple=False).reshape(-1)
            graph_transformer_positions = torch.nonzero(graph_transformer_mask, as_tuple=False).reshape(-1)

            if (
                graph_ev_positions.numel() == 0
                or graph_charger_positions.numel() == 0
                or graph_transformer_positions.numel() == 0
            ):
                graph_node_start = graph_node_end
                continue

            graph_transformer_ids = tr_features[graph_transformer_positions, 1].round().long()
            graph_charger_ids = cs_features[graph_charger_positions, 3].round().long()
            graph_ev_to_charger_id = ev_features[graph_ev_positions, 4].round().long()
            graph_ev_to_transformer_id = ev_features[graph_ev_positions, 5].round().long()

            graph_transformer_weights = torch.softmax(
                transformer_scores[graph_transformer_positions],
                dim=0,
            )
            graph_charger_to_transformer_id = self._infer_charger_to_transformer_id(
                graph_charger_ids=graph_charger_ids,
                graph_ev_to_charger_id=graph_ev_to_charger_id,
                graph_ev_to_transformer_id=graph_ev_to_transformer_id,
            )
            graph_charger_weights = torch.zeros(
                graph_charger_positions.numel(),
                dtype=torch.float32,
                device=self.device,
            )

            for transformer_id in graph_transformer_ids:
                sibling_charger_mask = graph_charger_to_transformer_id == transformer_id
                if not torch.any(sibling_charger_mask):
                    continue
                sibling_charger_positions = graph_charger_positions[sibling_charger_mask]
                graph_charger_weights[sibling_charger_mask] = torch.softmax(
                    charger_scores[sibling_charger_positions],
                    dim=0,
                )

            transformer_position_by_id = {
                int(transformer_id.item()): graph_position
                for graph_position, transformer_id in enumerate(graph_transformer_ids)
            }
            charger_position_by_id = {
                int(charger_id.item()): graph_position
                for graph_position, charger_id in enumerate(graph_charger_ids)
            }
            graph_total_budget = torch.tensor(
                float(graph_ev_positions.numel()),
                dtype=torch.float32,
                device=self.device,
            )

            for graph_ev_position, ev_feature_position in enumerate(graph_ev_positions):
                transformer_id = int(graph_ev_to_transformer_id[graph_ev_position].item())
                charger_id = int(graph_ev_to_charger_id[graph_ev_position].item())
                if transformer_id not in transformer_position_by_id or charger_id not in charger_position_by_id:
                    continue

                transformer_position = transformer_position_by_id[transformer_id]
                charger_position = charger_position_by_id[charger_id]
                ev_node_index = active_ev_node_indexes[ev_feature_position]
                action_value = (
                    graph_total_budget
                    * graph_transformer_weights[transformer_position]
                    * graph_charger_weights[charger_position]
                    * ev_action_gate[ev_feature_position]
                )
                full_node_action[ev_node_index, 0] = torch.clamp(
                    action_value,
                    min=0.0,
                    max=self.max_action,
                )

            hierarchy_details["per_graph_transformer_ids"].append(graph_transformer_ids)
            hierarchy_details["per_graph_transformer_weights"].append(graph_transformer_weights)
            hierarchy_details["per_graph_charger_weights"].append(graph_charger_weights)
            hierarchy_details["per_graph_charger_to_transformer_id"].append(
                graph_charger_to_transformer_id
            )

            graph_node_start = graph_node_end

        return full_node_action, hierarchy_details

    def forward(self, state, return_details=False):
        embedded_node_features, edge_index = self._prepare_features(state)
        ev_features = self._tensor(state.ev_features, torch.float32)
        cs_features = self._tensor(state.cs_features, torch.float32)
        tr_features = self._tensor(state.tr_features, torch.float32)
        active_ev_node_indexes = self._index_tensor(state.ev_indexes)
        charger_node_indexes = self._index_tensor(state.cs_indexes)
        transformer_node_indexes = self._index_tensor(state.tr_indexes)
        node_embeddings = self._encode_nodes(embedded_node_features, edge_index)
        full_node_action, hierarchy_details = self._compose_full_node_action(
            state=state,
            node_embeddings=node_embeddings,
            ev_features=ev_features,
            cs_features=cs_features,
            tr_features=tr_features,
            active_ev_node_indexes=active_ev_node_indexes,
            charger_node_indexes=charger_node_indexes,
            transformer_node_indexes=transformer_node_indexes,
        )
        if return_details:
            return full_node_action, hierarchy_details
        return full_node_action


class TD3_HierarchicalActionGNN(object):
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
            raise ValueError("TD3_HierarchicalActionGNN currently supports discrete_actions=1 only.")

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

    def _add_ev_noise(self, state, full_node_action, noise_scale, noise_clip=None):
        active_ev_node_indexes = self._index_tensor(state.ev_indexes)
        if noise_scale == 0 or active_ev_node_indexes.numel() == 0:
            return full_node_action

        noisy_full_node_action = full_node_action.clone()
        ev_noise = torch.randn(
            active_ev_node_indexes.numel(),
            1,
            dtype=full_node_action.dtype,
            device=self.device,
        ) * noise_scale
        if noise_clip is not None:
            ev_noise = ev_noise.clamp(-noise_clip, noise_clip)
        noisy_full_node_action[active_ev_node_indexes] = (
            noisy_full_node_action[active_ev_node_indexes] + ev_noise
        ).clamp(0.0, self.max_action)
        return noisy_full_node_action

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
