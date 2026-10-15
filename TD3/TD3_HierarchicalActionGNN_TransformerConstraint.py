import copy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from TD3.TD3_HierarchicalActionGNN import Actor, TD3_HierarchicalActionGNN
from TD3.TD3_ActionGNN_Controlled import Critic, resolve_device
from utils.transformer_feasibility_projection import (
    project_transformer_feasible_actions,
)


CANONICAL_ALGORITHM_LABEL = "hierarchical_transformer_constraint"
ACTION_DOMAIN_CONTRACT = "nonnegative_ev_rows_exact_zero_nonev_hard_transformer_feasible_v1"
ACTOR_OUTPUT_TRANSFORM = "full_hierarchy_hard_transformer_constraint_v1"
ACTOR_OUTPUT_TRANSFORM_FORMULA = (
    "full_hierarchical_action -> training_noise_if_any -> "
    "hard_per_transformer_feasibility_projection"
)
NON_EV_ACTION = 0.0
CHECKPOINT_METADATA_SCHEMA = "hierarchical_transformer_constraint_checkpoint_v1"
CHECKPOINT_SELECTION_RULE = (
    "model.best selected by strict improvement of scheduled internal eval mean reward "
    "at intervals recorded in eval_frequency within the configured training budget; "
    "model.last is saved at the configured final step but is not used for canonical "
    "eval30 or diagnostics."
)


def checkpoint_metadata_path(checkpoint_prefix):
    return Path(str(checkpoint_prefix) + ".metadata.yaml")


def build_checkpoint_metadata(args, checkpoint_role):
    return {
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


def write_checkpoint_metadata(checkpoint_prefix, args, checkpoint_role):
    metadata_path = checkpoint_metadata_path(checkpoint_prefix)
    with metadata_path.open("w") as metadata_file:
        yaml.safe_dump(
            build_checkpoint_metadata(args, checkpoint_role),
            metadata_file,
            sort_keys=False,
        )
    return metadata_path


class TD3_HierarchicalActionGNN_TransformerConstraint(TD3_HierarchicalActionGNN):
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
        transformer_voltage=None,
        transformer_phases=None,
        **kwargs,
    ):
        if discrete_actions != 1:
            raise ValueError(
                "TD3_HierarchicalActionGNN_TransformerConstraint supports discrete_actions=1 only."
            )
        if transformer_voltage is None or transformer_phases is None:
            raise ValueError(
                "TD3_HierarchicalActionGNN_TransformerConstraint requires "
                "transformer_voltage and transformer_phases."
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
        self.transformer_voltage = float(transformer_voltage)
        self.transformer_phases = float(transformer_phases)

    def _project_transformer_feasible_action(self, state, full_node_action):
        return project_transformer_feasible_actions(
            state=state,
            full_node_action=full_node_action,
            voltage=self.transformer_voltage,
            phases=self.transformer_phases,
            max_action=self.max_action,
        )

    def select_action(self, state, expl_noise=0, return_mapped_action=False, **kwargs):
        state = state.to(self.device)
        with torch.no_grad():
            full_node_action = self.actor(state)
            full_node_action = self._add_ev_noise(state, full_node_action, expl_noise)
            full_node_action = self._project_transformer_feasible_action(
                state,
                full_node_action,
            )

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
            next_action = self._project_transformer_feasible_action(
                next_state,
                next_action,
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
            actor_action = self._project_transformer_feasible_action(
                state,
                self.actor(state),
            )
            actor_loss = -self.critic.Q1(state, actor_action).mean()

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
