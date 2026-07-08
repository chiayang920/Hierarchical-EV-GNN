from __future__ import annotations


from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class HierarchicalRawAction:
    """
    The program is designed for storing raw outputs produced by the
    hierarchical actor.

    These values are not sent directly to EV2Gym. They are intermediate
    decision signals produced by the actor and then projected into valid
    charging actions.

    arg:
        transformer_scores:
            Raw actor scores for transformer-level allocation.
            Shape: (num_transformers,).

        charger_scores:
            Raw actor scores for charger-level allocation.
            Shape: (num_chargers,).

        ev_ratios:
            Raw actor scores for active EV-level local charging intensity.
            Shape: (num_active_evs,) or (num_active_evs, 1).

        total_budget:
            Optional scalar controlling the total normalised charging mass.
            This is not physical kW. It is a normalised action-budget scale
            used before clipping actions into [0, 1]. If None, the number
            of active EVs is used.

    return:
        HierarchicalRawAction object.
    """
    # default torch.FlaotTensor
    transformer_scores: torch.Tensor
    charger_scores: torch.Tensor
    ev_ratios: torch.Tensor
    total_budget: Optional[torch.Tensor] = None


@dataclass
class ProjectionMetadata:
    """
    The program is designed for storing graph-to-action mapping metadata
    required by the projection layer.

    This metadata is extracted from the PyTorch Geometric Data object
    produced by the state function.

    arg:
        action_dim:
            EV2Gym action dimension. For the current 25 CP setting, this is 25.

        action_mapper:
            Tensor mapping each active EV node to its EV2Gym flat action slot.
            Shape: (num_active_evs,).

        active_ev_to_charger_id:
            Charger ID for each active EV.
            Shape: (num_active_evs,).

        active_ev_to_transformer_id:
            Transformer ID for each active EV.
            Shape: (num_active_evs,).

        charger_ids:
            Charger IDs present in the pruned graph.
            Shape: (num_chargers,).

        charger_to_transformer_id:
            Transformer ID for each charger in charger_ids.
            Shape: (num_chargers,).

        transformer_ids:
            Transformer IDs present in the pruned graph.
            Shape: (num_transformers,).

    return:
        ProjectionMetadata object.
    """

    action_dim: int
    action_mapper: torch.Tensor
    active_ev_to_charger_id: torch.Tensor
    active_ev_to_transformer_id: torch.Tensor
    charger_ids: torch.Tensor
    charger_to_transformer_id: torch.Tensor
    transformer_ids: torch.Tensor


@dataclass
class ProjectionResult:
    """
    The program is designed for storing outputs from the hierarchical
    projection layer.

    arg:
        node_action:
            Differentiable active EV-node action tensor.
            Shape: (num_active_evs, 1).
            This representation is suitable for the graph critic and replay
            buffer path.

        mapped_action_tensor:
            Differentiable fixed-size action tensor.
            Shape: (action_dim,).
            This remains in torch form and can be used for debugging or
            differentiability tests.

        mapped_action_numpy:
            Detached NumPy action vector.
            Shape: (action_dim,).
            This is the format sent to EV2Gym via env.step(...). It is None
            when projection is requested only for the training path.

        transformer_weights:
            Normalised transformer allocation weights.
            Shape: (num_transformers,).

        charger_weights:
            Normalised charger allocation weights within each transformer.
            Shape: (num_chargers,).

        ev_local_gates:
            Local EV charging gates after sigmoid.
            Shape: (num_active_evs, 1).

    return:
        ProjectionResult object.
    """

    node_action: torch.Tensor
    mapped_action_tensor: torch.Tensor
    mapped_action_numpy: Optional[np.ndarray]
    transformer_weights: torch.Tensor
    charger_weights: torch.Tensor
    ev_local_gates: torch.Tensor


class CPOTransformerBudgetProjector:
    """
    The program is designed for converting CPO-level transformer scores
    into transformer-level allocation weights.

    This class does not learn. It applies a differentiable softmax so that
    the actor's raw transformer scores become valid allocation weights.

    arg:

    return:
        CPOTransformerBudgetProjector object.
    """

    def project(self, transformer_scores: torch.Tensor) -> torch.Tensor:
        """
        The program is designed for producing transformer allocation weights.

        arg:
            transformer_scores:
                Raw transformer-level scores from the actor.
                Shape: (num_transformers,).

        return:
            transformer_weights:
                Softmax-normalised allocation weights.
                Shape: (num_transformers,).
        """

        transformer_scores = transformer_scores.reshape(-1).float()

        if transformer_scores.numel() == 0:
            raise ValueError("transformer_scores cannot be empty.")

        # Softmax makes transformer-level allocation differentiable and
        # ensures the weights sum to one.
        return torch.softmax(transformer_scores, dim=0)


class TransformerChargerBudgetProjector:
    """
    The program is designed for converting charger-level scores into
    charger allocation weights within each transformer group.

    Each transformer distributes its own allocation among its chargers.
    This preserves the CPO -> transformer -> charger hierarchy.

    arg:

    return:
        TransformerChargerBudgetProjector object.
    """

    def project(
        self,
        charger_scores: torch.Tensor,
        charger_to_transformer_id: torch.Tensor,
        transformer_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        The program is designed for producing charger allocation weights
        grouped by transformer.

        arg:
            charger_scores:
                Raw charger-level scores from the actor.
                Shape: (num_chargers,).

            charger_to_transformer_id:
                Transformer ID for each charger.
                Shape: (num_chargers,).

            transformer_ids:
                Transformer IDs in the current graph.
                Shape: (num_transformers,).

        return:
            charger_weights:
                Charger allocation weights. For chargers under the same
                transformer, weights sum to one.
                Shape: (num_chargers,).
        """

        charger_scores = charger_scores.reshape(-1).float()

        if charger_scores.numel() != charger_to_transformer_id.numel():
            raise ValueError(
                "charger_scores length must match charger_to_transformer_id length."
            )

        charger_weights = torch.zeros_like(charger_scores)

        # Apply softmax separately inside each transformer group.
        # This means chargers only compete with sibling chargers under
        # the same transformer, not with all chargers globally.
        for transformer_id in transformer_ids:
            group_mask = charger_to_transformer_id == transformer_id
            group_scores = charger_scores[group_mask]

            if group_scores.numel() == 0:
                continue

            charger_weights[group_mask] = torch.softmax(group_scores, dim=0)

        return charger_weights


class ChargerEVActionProjector:
    """
    The program is designed for converting hierarchical transformer/charger
    allocations into active EV-level charging actions.

    This class combines:
    1. transformer allocation;
    2. charger allocation; and
    3. EV local charging gate.

    arg:
        min_action:
            Minimum continuous action value.

        max_action:
            Maximum continuous action value.

    return:
        ChargerEVActionProjector object.
    """

    def __init__(self, min_action: float = 0.0, max_action: float = 1.0) -> None:
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        if self.min_action > self.max_action:
            raise ValueError("min_action cannot be greater than max_action.")

    def project(
        self,
        raw_action: HierarchicalRawAction,
        metadata: ProjectionMetadata,
        transformer_weights: torch.Tensor,
        charger_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        The program is designed for producing active EV-node actions.

        arg:
            raw_action:
                Hierarchical raw action object from the actor.

            metadata:
                Projection metadata extracted from the graph state.

            transformer_weights:
                Transformer allocation weights.
                Shape: (num_transformers,).

            charger_weights:
                Charger allocation weights.
                Shape: (num_chargers,).

        return:
            node_action:
                Active EV-node action tensor clipped to [min_action, max_action].
                Shape: (num_active_evs, 1).
        """

        ev_ratios = raw_action.ev_ratios.float().reshape(-1, 1)
        num_active_evs = metadata.action_mapper.numel()

        if ev_ratios.shape[0] != num_active_evs:
            raise ValueError(
                "ev_ratios length must match the number of active EVs. "
                f"Got {ev_ratios.shape[0]} and expected {num_active_evs}."
            )

        # Sigmoid converts unconstrained raw EV-level actor outputs into
        # local gates in [0, 1] while keeping the path differentiable.
        ev_local_gates = torch.sigmoid(ev_ratios)

        if raw_action.total_budget is None:
            # Default budget scale: one normalised unit per active EV.
            # This is not physical kW. It is only an action-mass scale.
            total_budget = torch.tensor(
                float(num_active_evs),
                dtype=ev_ratios.dtype,
                device=ev_ratios.device,
            )
        else:
            total_budget = raw_action.total_budget.to(
                dtype=ev_ratios.dtype,
                device=ev_ratios.device,
            ).reshape(())

        # Build lookup tables from IDs to local tensor positions.
        transformer_id_to_position = {
            int(transformer_id.item()): index
            for index, transformer_id in enumerate(metadata.transformer_ids)
        }

        charger_id_to_position = {
            int(charger_id.item()): index
            for index, charger_id in enumerate(metadata.charger_ids)
        }

        node_actions = []

        for ev_index in range(num_active_evs):
            transformer_id = int(metadata.active_ev_to_transformer_id[ev_index].item())
            charger_id = int(metadata.active_ev_to_charger_id[ev_index].item())

            transformer_position = transformer_id_to_position[transformer_id]
            charger_position = charger_id_to_position[charger_id]

            # Hierarchical composition:
            # total budget -> transformer share -> charger share -> EV local gate.
            action_value = (
                total_budget
                * transformer_weights[transformer_position]
                * charger_weights[charger_position]
                * ev_local_gates[ev_index, 0]
            )

            node_actions.append(action_value)

        node_action = torch.stack(node_actions).reshape(-1, 1)

        # Final clipping ensures EV2Gym-compatible continuous action bounds.
        return torch.clamp(
            node_action,
            min=self.min_action,
            max=self.max_action,
        )


class EV2GymActionMapper:
    """
    The program is designed for mapping active EV-node actions into the
    fixed EV2Gym action vector.

    EV2Gym does not consume graph nodes or hierarchical budgets. It only
    consumes a flat action vector with one value per charging port.

    arg:
        action_dim:
            Fixed EV2Gym action dimension.

    return:
        EV2GymActionMapper object.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = int(action_dim)

        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")

    def map_to_tensor(
        self,
        node_action: torch.Tensor,
        action_mapper: torch.Tensor,
    ) -> torch.Tensor:
        """
        The program is designed for producing a differentiable fixed-size
        EV2Gym action tensor.

        arg:
            node_action:
                Active EV-node action tensor.
                Shape: (num_active_evs, 1).

            action_mapper:
                Mapping from active EV node order to EV2Gym action slots.
                Shape: (num_active_evs,).

        return:
            mapped_action_tensor:
                Fixed-size torch action tensor.
                Shape: (action_dim,).
        """

        node_action = node_action.reshape(-1)
        action_mapper = action_mapper.long().reshape(-1)

        if node_action.numel() != action_mapper.numel():
            raise ValueError(
                "node_action length must match action_mapper length."
            )

        if torch.any(action_mapper < 0) or torch.any(action_mapper >= self.action_dim):
            raise IndexError(
                "action_mapper contains indices outside the EV2Gym action range."
            )

        if torch.unique(action_mapper).numel() != action_mapper.numel():
            raise ValueError(
                "action_mapper contains duplicate EV2Gym action slots."
            )

        # Create a flat action vector.
        # Empty charging ports remain zero.
        mapped_action = torch.zeros(
            self.action_dim,
            dtype=node_action.dtype,
            device=node_action.device,
        )

        # index_copy keeps the value path differentiable with respect to
        # node_action. This is important for training-path tests.
        mapped_action = mapped_action.index_copy(
            dim=0,
            index=action_mapper,
            source=node_action,
        )

        return mapped_action

    def map_to_numpy(
        self,
        node_action: torch.Tensor,
        action_mapper: torch.Tensor,
    ) -> np.ndarray:
        """
        The program is designed for producing the detached NumPy action
        vector required by EV2Gym env.step(...).

        arg:
            node_action:
                Active EV-node action tensor.
                Shape: (num_active_evs, 1).

            action_mapper:
                Mapping from active EV node order to EV2Gym action slots.
                Shape: (num_active_evs,).

        return:
            NumPy action vector with shape (action_dim,).
        """

        mapped_action = self.map_to_tensor(node_action, action_mapper)

        # EV2Gym is a simulator interface. It does not use gradients.
        return mapped_action.detach().cpu().numpy().astype(np.float32)


class HierarchicalActionProjection:
    """
    The program is designed for coordinating hierarchical action projection.

    This is the only class that TD3_HierarchicalActionGNN should call.
    It coordinates the full deterministic path:

        actor raw outputs
        -> transformer allocation
        -> charger allocation
        -> active EV-node action
        -> EV2Gym 25-dimensional action vector

    It does not implement TD3. It only converts action format.

    arg:
        action_dim:
            Fixed EV2Gym action dimension. For 25 CP, use 25.

        min_action:
            Minimum action value.

        max_action:
            Maximum action value.

    return:
        HierarchicalActionProjection object.
    """

    def __init__(
        self,
        action_dim: int,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ) -> None:
        self.action_dim = int(action_dim)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        self.cpo_transformer_projector = CPOTransformerBudgetProjector()
        self.transformer_charger_projector = TransformerChargerBudgetProjector()
        self.charger_ev_projector = ChargerEVActionProjector(
            min_action=self.min_action,
            max_action=self.max_action,
        )
        self.ev2gym_mapper = EV2GymActionMapper(action_dim=self.action_dim)

    def build_metadata_from_state(self, state) -> ProjectionMetadata:
        """
        The program is designed for extracting projection metadata from
        a PyTorch Geometric Data object produced by the state function.

        This method expects the PublicPST_GNN-style feature layout:
            ev_features columns:
                [soc_flag, energy_exchanged, time_since_arrival, ev_id, cs_id, tr_id]

            cs_features columns:
                [min_charge_current, max_charge_current, n_ports, cs_id]

            tr_features columns:
                [max_power, tr_id]

        arg:
            state:
                PyTorch Geometric Data object.

        return:
            ProjectionMetadata object.
        """

        self._validate_state(state)

        device = self._infer_device_from_raw_state(state)

        action_mapper = self._to_tensor(state.action_mapper, device=device).long()
        ev_features = self._to_tensor(state.ev_features, device=device).float()
        cs_features = self._to_tensor(state.cs_features, device=device).float()
        tr_features = self._to_tensor(state.tr_features, device=device).float()

        if ev_features.ndim != 2 or ev_features.shape[1] < 6:
            raise ValueError(
                "ev_features must have at least 6 columns for PublicPST_GNN metadata."
            )

        if cs_features.ndim != 2 or cs_features.shape[1] < 4:
            raise ValueError(
                "cs_features must have at least 4 columns for PublicPST_GNN metadata."
            )

        if tr_features.ndim != 2 or tr_features.shape[1] < 2:
            raise ValueError(
                "tr_features must have at least 2 columns for PublicPST_GNN metadata."
            )

        active_ev_to_charger_id = ev_features[:, 4].round().long()
        active_ev_to_transformer_id = ev_features[:, 5].round().long()

        charger_ids = cs_features[:, 3].round().long()
        transformer_ids = tr_features[:, 1].round().long()

        charger_to_transformer_id = self._infer_charger_to_transformer_id(
            charger_ids=charger_ids,
            active_ev_to_charger_id=active_ev_to_charger_id,
            active_ev_to_transformer_id=active_ev_to_transformer_id,
        )

        return ProjectionMetadata(
            action_dim=self.action_dim,
            action_mapper=action_mapper,
            active_ev_to_charger_id=active_ev_to_charger_id,
            active_ev_to_transformer_id=active_ev_to_transformer_id,
            charger_ids=charger_ids,
            charger_to_transformer_id=charger_to_transformer_id,
            transformer_ids=transformer_ids,
        )

    def project_for_training(
        self,
        state,
        raw_action: HierarchicalRawAction,
    ) -> ProjectionResult:
        """
        The program is designed for projecting hierarchical actor outputs
        while preserving differentiability.

        This method should be used inside the learning path because it does
        not detach tensors.

        arg:
            state:
                PyTorch Geometric Data object.

            raw_action:
                Hierarchical raw action object from actor output.

        return:
            ProjectionResult with differentiable torch tensors.
        """

        metadata = self.build_metadata_from_state(state)

        raw_action = self._move_raw_action_to_metadata_device(raw_action, metadata)

        self._validate_raw_action_shapes(raw_action, metadata)

        transformer_weights = self.cpo_transformer_projector.project(
            raw_action.transformer_scores
        )

        charger_weights = self.transformer_charger_projector.project(
            charger_scores=raw_action.charger_scores,
            charger_to_transformer_id=metadata.charger_to_transformer_id,
            transformer_ids=metadata.transformer_ids,
        )

        node_action = self.charger_ev_projector.project(
            raw_action=raw_action,
            metadata=metadata,
            transformer_weights=transformer_weights,
            charger_weights=charger_weights,
        )

        mapped_action_tensor = self.ev2gym_mapper.map_to_tensor(
            node_action=node_action,
            action_mapper=metadata.action_mapper,
        )

        ev_local_gates = torch.sigmoid(raw_action.ev_ratios.reshape(-1, 1).float())

        return ProjectionResult(
            node_action=node_action,
            mapped_action_tensor=mapped_action_tensor,
            mapped_action_numpy=None,
            transformer_weights=transformer_weights,
            charger_weights=charger_weights,
            ev_local_gates=ev_local_gates,
        )

    def project_for_env(
        self,
        state,
        raw_action: HierarchicalRawAction,
    ) -> ProjectionResult:
        """
        The program is designed for projecting hierarchical actor outputs
        into the NumPy action vector required by EV2Gym.

        This method is used for environment interaction:
            env.step(mapped_action_numpy)

        arg:
            state:
                PyTorch Geometric Data object.

            raw_action:
                Hierarchical raw action object from actor output.

        return:
            ProjectionResult with mapped_action_numpy included.
        """

        result = self.project_for_training(state=state, raw_action=raw_action)

        mapped_action_numpy = (
            result.mapped_action_tensor
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return ProjectionResult(
            node_action=result.node_action,
            mapped_action_tensor=result.mapped_action_tensor,
            mapped_action_numpy=mapped_action_numpy,
            transformer_weights=result.transformer_weights,
            charger_weights=result.charger_weights,
            ev_local_gates=result.ev_local_gates,
        )

    def _validate_state(self, state) -> None:
        """
        The program is designed for checking whether state contains the
        minimum metadata required for projection.

        arg:
            state:
                PyTorch Geometric Data object.

        return:
            None.
        """

        required_fields = [
            "action_mapper",
            "ev_features",
            "cs_features",
            "tr_features",
        ]

        for field in required_fields:
            if not hasattr(state, field):
                raise AttributeError(f"state is missing required field: {field}")

    def _validate_raw_action_shapes(
        self,
        raw_action: HierarchicalRawAction,
        metadata: ProjectionMetadata,
    ) -> None:
        """
        The program is designed for validating actor raw output dimensions.

        arg:
            raw_action:
                Hierarchical raw action object.

            metadata:
                Projection metadata.

        return:
            None.
        """

        if raw_action.transformer_scores.reshape(-1).numel() != metadata.transformer_ids.numel():
            raise ValueError(
                "transformer_scores length must match number of transformers. "
                f"Got {raw_action.transformer_scores.reshape(-1).numel()} and "
                f"{metadata.transformer_ids.numel()}."
            )

        if raw_action.charger_scores.reshape(-1).numel() != metadata.charger_ids.numel():
            raise ValueError(
                "charger_scores length must match number of chargers. "
                f"Got {raw_action.charger_scores.reshape(-1).numel()} and "
                f"{metadata.charger_ids.numel()}."
            )

        if raw_action.ev_ratios.reshape(-1).numel() != metadata.action_mapper.numel():
            raise ValueError(
                "ev_ratios length must match number of active EVs. "
                f"Got {raw_action.ev_ratios.reshape(-1).numel()} and "
                f"{metadata.action_mapper.numel()}."
            )

    def _infer_charger_to_transformer_id(
        self,
        charger_ids: torch.Tensor,
        active_ev_to_charger_id: torch.Tensor,
        active_ev_to_transformer_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        The program is designed for inferring each charger's transformer ID
        from active EV metadata.

        In the pruned PublicPST_GNN graph, chargers are included only when
        they have active EVs. Therefore each charger ID should appear in
        active_ev_to_charger_id.

        arg:
            charger_ids:
                Charger IDs in the graph.

            active_ev_to_charger_id:
                Charger ID for each active EV.

            active_ev_to_transformer_id:
                Transformer ID for each active EV.

        return:
            charger_to_transformer_id tensor.
        """

        charger_transformer_ids = []

        for charger_id in charger_ids:
            matching_ev_mask = active_ev_to_charger_id == charger_id

            if not torch.any(matching_ev_mask):
                raise ValueError(
                    "Cannot infer transformer ID for charger "
                    f"{int(charger_id.item())}. No active EV is mapped to it."
                )

            transformer_ids_for_charger = active_ev_to_transformer_id[matching_ev_mask]
            unique_transformer_ids = torch.unique(transformer_ids_for_charger)

            if unique_transformer_ids.numel() != 1:
                raise ValueError(
                    "A charger appears to be associated with multiple transformers, "
                    f"charger_id={int(charger_id.item())}."
                )

            charger_transformer_ids.append(unique_transformer_ids[0])

        return torch.stack(charger_transformer_ids).long()

    def _move_raw_action_to_metadata_device(
        self,
        raw_action: HierarchicalRawAction,
        metadata: ProjectionMetadata,
    ) -> HierarchicalRawAction:
        """
        The program is designed for moving raw actor outputs onto the same
        device as projection metadata.

        arg:
            raw_action:
                Hierarchical raw action object.

            metadata:
                Projection metadata.

        return:
            HierarchicalRawAction on the correct device.
        """

        device = metadata.action_mapper.device

        total_budget = raw_action.total_budget
        if total_budget is not None:
            total_budget = total_budget.to(device)

        return HierarchicalRawAction(
            transformer_scores=raw_action.transformer_scores.to(device),
            charger_scores=raw_action.charger_scores.to(device),
            ev_ratios=raw_action.ev_ratios.to(device),
            total_budget=total_budget,
        )

    def _to_tensor(self, value, device: torch.device) -> torch.Tensor:
        """
        The program is designed for converting NumPy arrays, lists, or tensors
        into torch tensors on the required device.

        arg:
            value:
                Input object.

            device:
                Target torch device.

        return:
            Torch tensor.
        """

        if isinstance(value, torch.Tensor):
            return value.to(device)

        return torch.as_tensor(value, device=device)

    def _infer_device_from_raw_state(self, state) -> torch.device:
        """
        The program is designed for selecting the device used by the input
        graph state.

        arg:
            state:
                PyTorch Geometric Data object.

        return:
            Torch device.
        """

        for field in ["ev_features", "cs_features", "tr_features"]:
            value = getattr(state, field)
            if isinstance(value, torch.Tensor):
                return value.device

        return torch.device("cpu")