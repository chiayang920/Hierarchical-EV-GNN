import numpy as np
import torch
from torch_geometric.data import Data


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_numpy_index(values):
    return np.asarray(values, dtype=np.int64)


def _batch_graphs(graphs, device):
    """
    Batch EV-GNN graph Data objects manually.

    Important correction relative to the original replay_buffer.py:
    edge_index must be offset by the cumulative number of nodes, not by the
    maximum previous edge index. Using max(edge_index) can connect different
    sampled graphs incorrectly when batching.
    """
    edge_index_parts = []
    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = []

    ev_features = np.concatenate([g.ev_features for g in graphs], axis=0)
    cs_features = np.concatenate([g.cs_features for g in graphs], axis=0)
    tr_features = np.concatenate([g.tr_features for g in graphs], axis=0)
    env_features = np.concatenate([g.env_features for g in graphs], axis=0)
    node_types = np.concatenate([g.node_types for g in graphs], axis=0)
    sample_node_length = [len(g.node_types) for g in graphs]

    node_counter = 0
    for g in graphs:
        edge_index = np.asarray(g.edge_index, dtype=np.int64)
        if edge_index.size > 0:
            edge_index_parts.append(edge_index + node_counter)

        ev_indexes.append(_to_numpy_index(g.ev_indexes) + node_counter)
        cs_indexes.append(_to_numpy_index(g.cs_indexes) + node_counter)
        tr_indexes.append(_to_numpy_index(g.tr_indexes) + node_counter)
        env_indexes.append(_to_numpy_index(g.env_indexes) + node_counter)
        node_counter += len(g.node_types)

    if edge_index_parts:
        edge_index = np.concatenate(edge_index_parts, axis=1)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    return Data(
        edge_index=torch.from_numpy(edge_index).long().to(device),
        ev_features=torch.from_numpy(ev_features).float().to(device),
        cs_features=torch.from_numpy(cs_features).float().to(device),
        tr_features=torch.from_numpy(tr_features).float().to(device),
        env_features=torch.from_numpy(env_features).float().to(device),
        node_types=torch.from_numpy(node_types).long().to(device),
        sample_node_length=sample_node_length,
        ev_indexes=np.concatenate(ev_indexes).astype(np.int64) if ev_indexes else np.array([], dtype=np.int64),
        cs_indexes=np.concatenate(cs_indexes).astype(np.int64) if cs_indexes else np.array([], dtype=np.int64),
        tr_indexes=np.concatenate(tr_indexes).astype(np.int64) if tr_indexes else np.array([], dtype=np.int64),
        env_indexes=np.concatenate(env_indexes).astype(np.int64) if env_indexes else np.array([], dtype=np.int64),
    )


class ActionGNN_ReplayBuffer(object):
    """Scale-agnostic replay buffer for ActionGNN node-level actions."""

    def __init__(self, action_dim, max_size=int(1e6), device=None):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.action_dim = action_dim
        self.device = resolve_device(device)

        self.state = [None for _ in range(self.max_size)]
        self.action = [None for _ in range(self.max_size)]
        self.next_state = [None for _ in range(self.max_size)]
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        if isinstance(action, torch.Tensor):
            self.action[self.ptr] = action.detach().cpu().float()
        else:
            self.action[self.ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indices = np.random.randint(0, self.size, size=batch_size)
        states = [self.state[i] for i in indices]
        next_states = [self.next_state[i] for i in indices]

        state_batch = _batch_graphs(states, self.device)
        next_state_batch = _batch_graphs(next_states, self.device)
        action_batch = torch.cat([self.action[i] for i in indices], dim=0).to(self.device)

        return (
            state_batch,
            action_batch,
            next_state_batch,
            torch.from_numpy(self.reward[indices]).float().to(self.device),
            torch.from_numpy(self.not_done[indices]).float().to(self.device),
        )
