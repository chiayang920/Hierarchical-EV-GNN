import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


def resolve_device(device=None):
    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        # PyTorch Geometric sparse operations are often safer on CPU on Apple Silicon.
        return requested
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Actor(nn.Module):
    def __init__(self,
                 max_action,
                 fx_node_sizes,
                 feature_dim=32,
                 GNN_hidden_dim=64,
                 num_gcn_layers=3,
                 discrete_actions=1,
                 device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.discrete_actions = discrete_actions
        self.num_gcn_layers = num_gcn_layers
        self.max_action = max_action

        self.ev_embedding = nn.Linear(fx_node_sizes["ev"], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes["cs"], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes["tr"], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes["env"], feature_dim)

        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)

        if num_gcn_layers == 3:
            self.gcn_layers = nn.ModuleList([GCNConv(GNN_hidden_dim, feature_dim)])
        elif num_gcn_layers == 4:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2 * GNN_hidden_dim),
                GCNConv(2 * GNN_hidden_dim, feature_dim),
            ])
        elif num_gcn_layers == 5:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2 * GNN_hidden_dim),
                GCNConv(2 * GNN_hidden_dim, GNN_hidden_dim),
                GCNConv(GNN_hidden_dim, feature_dim),
            ])
        elif num_gcn_layers == 6:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2 * GNN_hidden_dim),
                GCNConv(2 * GNN_hidden_dim, 3 * GNN_hidden_dim),
                GCNConv(3 * GNN_hidden_dim, 2 * GNN_hidden_dim),
                GCNConv(2 * GNN_hidden_dim, feature_dim),
            ])
        else:
            raise ValueError("Actor GCN layers must be 3, 4, 5, or 6.")

        self.gcn_last = GCNConv(feature_dim, discrete_actions)

    def _prepare_features(self, state):
        if isinstance(state.env_features, np.ndarray):
            ev_features = torch.from_numpy(state.ev_features).float().to(self.device)
            cs_features = torch.from_numpy(state.cs_features).float().to(self.device)
            tr_features = torch.from_numpy(state.tr_features).float().to(self.device)
            env_features = torch.from_numpy(state.env_features).float().to(self.device)
            edge_index = torch.from_numpy(state.edge_index).long().to(self.device)
        else:
            ev_features = state.ev_features.float().to(self.device)
            cs_features = state.cs_features.float().to(self.device)
            tr_features = state.tr_features.float().to(self.device)
            env_features = state.env_features.float().to(self.device)
            edge_index = state.edge_index.long().to(self.device)

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + tr_features.shape[0] + env_features.shape[0]
        embedded_x = torch.zeros(total_nodes, self.feature_dim, device=self.device).float()

        if len(state.ev_indexes) != 0:
            embedded_x[state.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[state.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[state.tr_indexes] = self.tr_embedding(tr_features)
        embedded_x[state.env_indexes] = self.env_embedding(env_features)

        return F.relu(embedded_x.reshape(-1, self.feature_dim)), edge_index

    def forward(self, state, return_mapper=False):
        embedded_x, edge_index = self._prepare_features(state)

        x = F.relu(self.gcn_conv(embedded_x, edge_index))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, edge_index))

        x = self.gcn_last(x, edge_index)
        x = self.max_action * torch.tanh(x)

        if self.discrete_actions > 1:
            x = torch.softmax(x, dim=1)

        x = x.reshape(-1)
        if return_mapper:
            return x, None, state.ev_indexes
        return x


class Critic_GNN(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 feature_dim=32,
                 GNN_hidden_dim=64,
                 mlp_hidden_dim=512,
                 discrete_actions=1,
                 num_gcn_layers=3,
                 device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.discrete_actions = discrete_actions

        self.ev_embedding = nn.Linear(fx_node_sizes["ev"], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes["cs"], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes["tr"], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes["env"], feature_dim)

        self.gcn_conv = GCNConv(feature_dim + discrete_actions, GNN_hidden_dim)

        if num_gcn_layers == 3:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2 * GNN_hidden_dim),
                GCNConv(2 * GNN_hidden_dim, 3 * GNN_hidden_dim),
            ])
            mlp_layer_features = 3 * GNN_hidden_dim
        elif num_gcn_layers == 4:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2 * GNN_hidden_dim),
                GCNConv(2 * GNN_hidden_dim, 3 * GNN_hidden_dim),
                GCNConv(3 * GNN_hidden_dim, 2 * GNN_hidden_dim),
            ])
            mlp_layer_features = 2 * GNN_hidden_dim
        elif num_gcn_layers == 5:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2 * GNN_hidden_dim),
                GCNConv(2 * GNN_hidden_dim, 3 * GNN_hidden_dim),
                GCNConv(3 * GNN_hidden_dim, 4 * GNN_hidden_dim),
                GCNConv(4 * GNN_hidden_dim, 3 * GNN_hidden_dim),
            ])
            mlp_layer_features = 3 * GNN_hidden_dim
        else:
            raise ValueError("Critic GCN layers must be 3, 4, or 5.")

        self.l1 = nn.Linear(mlp_layer_features, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

    def _prepare_features(self, state):
        ev_features = state.ev_features.float().to(self.device)
        cs_features = state.cs_features.float().to(self.device)
        tr_features = state.tr_features.float().to(self.device)
        env_features = state.env_features.float().to(self.device)
        edge_index = state.edge_index.long().to(self.device)

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + tr_features.shape[0] + env_features.shape[0]
        embedded_x = torch.zeros(total_nodes, self.feature_dim, device=self.device).float()

        if len(state.ev_indexes) != 0:
            embedded_x[state.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[state.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[state.tr_indexes] = self.tr_embedding(tr_features)
        embedded_x[state.env_indexes] = self.env_embedding(env_features)

        return F.relu(embedded_x.reshape(-1, self.feature_dim)), edge_index

    def forward(self, state, action):
        embedded_x, edge_index = self._prepare_features(state)
        state_action = torch.cat([embedded_x, action.reshape(-1, self.discrete_actions)], dim=1)

        x = F.relu(self.gcn_conv(state_action, edge_index))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, edge_index))

        batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        counter = 0
        for graph_index, node_count in enumerate(state.sample_node_length):
            batch[counter: counter + node_count] = graph_index
            counter += node_count

        pooled_embedding = global_mean_pool(x, batch=batch)
        x = F.relu(self.l1(pooled_embedding))
        x = F.relu(self.l2(x))
        return self.l3(x)


class Critic(nn.Module):
    def __init__(self,
                 fx_node_sizes,
                 feature_dim=32,
                 GNN_hidden_dim=64,
                 mlp_hidden_dim=512,
                 discrete_actions=1,
                 num_gcn_layers=3,
                 device=torch.device("cpu")):
        super().__init__()
        self.q1 = Critic_GNN(fx_node_sizes, feature_dim, GNN_hidden_dim, mlp_hidden_dim,
                             discrete_actions, num_gcn_layers, device)
        self.q2 = Critic_GNN(fx_node_sizes, feature_dim, GNN_hidden_dim, mlp_hidden_dim,
                             discrete_actions, num_gcn_layers, device)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def Q1(self, state, action):
        return self.q1(state, action)


class TD3_ActionGNN(object):
    def __init__(self,
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
                 **kwargs):
        self.device = resolve_device(device)
        self.discrete_actions = discrete_actions

        self.actor = Actor(max_action, fx_node_sizes, fx_dim, fx_GNN_hidden_dim,
                           actor_num_gcn_layers, discrete_actions, self.device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(fx_node_sizes, fx_dim, fx_GNN_hidden_dim, mlp_hidden_dim,
                             discrete_actions, critic_num_gcn_layers, self.device).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, expl_noise=0, return_mapped_action=False, **kwargs):
        state = state.to(self.device)
        with torch.no_grad():
            action, _, ev_indexes = self.actor(state, return_mapper=True)

        if expl_noise != 0:
            noise = torch.randn_like(action) * expl_noise
            action = (action + noise).clamp(-self.max_action, self.max_action)

        mapped_action = np.zeros(self.action_dim, dtype=np.float32)
        if self.discrete_actions == 1:
            for index, action_index in enumerate(state.action_mapper):
                mapped_action[action_index] = action[ev_indexes[index]].detach().cpu().item()
        else:
            temp_action = torch.argmax(action.reshape(-1, self.discrete_actions), dim=1)
            for index, action_index in enumerate(state.action_mapper):
                mapped_action[action_index] = temp_action[ev_indexes[index]].detach().cpu().item()

        if expl_noise != 0 or return_mapped_action:
            return mapped_action, action.detach().cpu()
        return mapped_action

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, _, _ = self.actor_target(next_state, return_mapper=True)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

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

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return critic_loss.item(), actor_loss.item()

        return critic_loss.item(), None

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
