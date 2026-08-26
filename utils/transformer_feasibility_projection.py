import math

import torch


ROUNDING_ERROR_PER_ACTION = 5e-6
NUMERICAL_POWER_MARGIN_KW = 1e-4


def _as_tensor(value, dtype, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _index_tensor(value, device):
    return _as_tensor(value, torch.long, device).reshape(-1)


def _unique_id_positions(ids, label):
    positions = {}
    for position, identifier in enumerate(ids.detach().cpu().tolist()):
        identifier = int(identifier)
        if identifier in positions:
            raise ValueError(f"Ambiguous duplicate {label} ID {identifier}.")
        positions[identifier] = position
    return positions


def _require_positive(value, label):
    value = float(value)
    if value <= 0:
        raise ValueError(f"{label} must be positive; got {value!r}.")
    return value


def validate_transformer_constraint_config(config):
    if config.get("v2g_enabled") is True:
        raise ValueError("hierarchical_transformer_constraint requires v2g_enabled=False.")

    for section_name in ("inflexible_loads", "solar_power", "demand_response"):
        if config.get(section_name, {}).get("include") is True:
            raise ValueError(
                f"hierarchical_transformer_constraint requires {section_name}.include=False."
            )

    topology = config.get("charging_network_topology")
    if topology not in (None, "None"):
        raise ValueError(
            "hierarchical_transformer_constraint requires charging_network_topology=None."
        )

    charging_station = config.get("charging_station", {})
    voltage = _require_positive(charging_station.get("voltage"), "charging_station.voltage")
    phases = _require_positive(charging_station.get("phases"), "charging_station.phases")
    _require_positive(
        charging_station.get("max_charge_current"),
        "charging_station.max_charge_current",
    )
    return {
        "transformer_voltage": voltage,
        "transformer_phases": phases,
    }


def _empty_details(device, dtype):
    return {
        "transformers": [],
        "total_action_correction_magnitude": torch.tensor(0.0, dtype=dtype, device=device),
    }


def project_transformer_feasible_actions(
    *,
    state,
    full_node_action,
    voltage,
    phases,
    max_action=1.0,
    return_details=False,
):
    voltage = _require_positive(voltage, "voltage")
    phases = _require_positive(phases, "phases")
    max_action = _require_positive(max_action, "max_action")

    candidate_action = full_node_action.reshape(-1, 1)
    device = candidate_action.device
    dtype = candidate_action.dtype

    safe_action = torch.zeros_like(candidate_action)
    details = _empty_details(device, dtype)

    active_ev_node_indexes = _index_tensor(state.ev_indexes, device)
    if active_ev_node_indexes.numel() == 0:
        if return_details:
            return safe_action, details
        return safe_action

    ev_features = _as_tensor(state.ev_features, torch.float32, device)
    cs_features = _as_tensor(state.cs_features, torch.float32, device)
    tr_features = _as_tensor(state.tr_features, torch.float32, device)
    charger_node_indexes = _index_tensor(state.cs_indexes, device)
    transformer_node_indexes = _index_tensor(state.tr_indexes, device)

    bounded_action = candidate_action.clamp(0.0, max_action)
    ev_action_values = bounded_action[active_ev_node_indexes, 0]
    safe_ev_action_values = torch.zeros_like(ev_action_values)

    graph_node_start = 0
    transformer_power_factor = voltage * math.sqrt(phases) / 1000.0
    sample_node_lengths = [int(sample_node_length) for sample_node_length in state.sample_node_length]

    for graph_index, sample_node_length in enumerate(sample_node_lengths):
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
        graph_transformer_positions = torch.nonzero(
            graph_transformer_mask,
            as_tuple=False,
        ).reshape(-1)

        if graph_ev_positions.numel() == 0:
            graph_node_start = graph_node_end
            continue
        if graph_charger_positions.numel() == 0:
            raise ValueError(f"Graph {graph_index} has active EVs but no charger nodes.")
        if graph_transformer_positions.numel() == 0:
            raise ValueError(f"Graph {graph_index} has active EVs but no transformer nodes.")

        graph_charger_ids = cs_features[graph_charger_positions, 3].round().long()
        graph_transformer_ids = tr_features[graph_transformer_positions, 1].round().long()
        charger_position_by_id = _unique_id_positions(graph_charger_ids, "charger")
        transformer_position_by_id = _unique_id_positions(graph_transformer_ids, "transformer")

        graph_ev_to_charger_id = ev_features[graph_ev_positions, 4].round().long()
        graph_ev_to_transformer_id = ev_features[graph_ev_positions, 5].round().long()

        ev_charger_positions = []
        ev_transformer_positions = []
        charger_to_transformer_id = {}
        for ev_position, (charger_id_tensor, transformer_id_tensor) in enumerate(
            zip(graph_ev_to_charger_id, graph_ev_to_transformer_id)
        ):
            charger_id = int(charger_id_tensor.detach().cpu().item())
            transformer_id = int(transformer_id_tensor.detach().cpu().item())
            if charger_id not in charger_position_by_id:
                raise ValueError(
                    f"Active EV at graph {graph_index} position {ev_position} "
                    f"references missing charger ID {charger_id}."
                )
            if transformer_id not in transformer_position_by_id:
                raise ValueError(
                    f"Active EV at graph {graph_index} position {ev_position} "
                    f"references missing transformer ID {transformer_id}."
                )
            existing_transformer_id = charger_to_transformer_id.setdefault(
                charger_id,
                transformer_id,
            )
            if existing_transformer_id != transformer_id:
                raise ValueError(
                    f"Charger ID {charger_id} maps to multiple transformer IDs "
                    f"within graph {graph_index}."
                )
            ev_charger_positions.append(charger_position_by_id[charger_id])
            ev_transformer_positions.append(transformer_position_by_id[transformer_id])

        ev_charger_positions = torch.as_tensor(
            ev_charger_positions,
            dtype=torch.long,
            device=device,
        )
        ev_transformer_positions = torch.as_tensor(
            ev_transformer_positions,
            dtype=torch.long,
            device=device,
        )

        charger_max_current = cs_features[graph_charger_positions, 1].to(dtype=dtype)
        if torch.any(charger_max_current <= 0):
            raise ValueError("Charger maximum charging current must be positive.")
        ev_power_upper_bound_kw = (
            charger_max_current[ev_charger_positions] * transformer_power_factor
        ).to(dtype=dtype)
        graph_ev_action_values = ev_action_values[graph_ev_positions]

        for transformer_position, transformer_id_tensor in enumerate(graph_transformer_ids):
            transformer_ev_mask = ev_transformer_positions == transformer_position
            transformer_id = int(transformer_id_tensor.detach().cpu().item())
            physical_max_power = tr_features[
                graph_transformer_positions[transformer_position],
                0,
            ].to(dtype=dtype)

            if not torch.any(transformer_ev_mask):
                raw_power = torch.tensor(0.0, dtype=dtype, device=device)
                rounding_margin = torch.tensor(
                    NUMERICAL_POWER_MARGIN_KW,
                    dtype=dtype,
                    device=device,
                )
                usable_capacity = torch.clamp(physical_max_power - rounding_margin, min=0.0)
                alpha = torch.tensor(1.0, dtype=dtype, device=device)
                safe_power = raw_power
            else:
                transformer_actions = graph_ev_action_values[transformer_ev_mask]
                transformer_power_bounds = ev_power_upper_bound_kw[transformer_ev_mask]
                raw_power = (transformer_actions * transformer_power_bounds).sum()
                rounding_margin = (
                    ROUNDING_ERROR_PER_ACTION * transformer_power_bounds.sum()
                    + NUMERICAL_POWER_MARGIN_KW
                )
                usable_capacity = torch.clamp(physical_max_power - rounding_margin, min=0.0)
                alpha = torch.where(
                    (raw_power <= usable_capacity) | (raw_power <= 0.0),
                    torch.ones_like(raw_power),
                    usable_capacity / raw_power.clamp_min(torch.finfo(dtype).eps),
                ).clamp(0.0, 1.0)
                graph_safe_values = graph_ev_action_values[transformer_ev_mask] * alpha
                safe_ev_action_values[graph_ev_positions[transformer_ev_mask]] = graph_safe_values
                safe_power = (graph_safe_values * transformer_power_bounds).sum()

            if return_details:
                activated = bool(
                    ((raw_power > usable_capacity) & (raw_power > 0.0)).detach().cpu().item()
                )
                details["transformers"].append(
                    {
                        "graph_index": graph_index,
                        "transformer_id": transformer_id,
                        "raw_estimated_commanded_power": raw_power,
                        "safe_estimated_commanded_power": safe_power,
                        "physical_max_power": physical_max_power,
                        "usable_safe_capacity": usable_capacity,
                        "rounding_margin": rounding_margin,
                        "alpha": alpha,
                        "constraint_activated": activated,
                        "commanded_power_removed": raw_power - safe_power,
                    }
                )

        graph_node_start = graph_node_end

    safe_action[active_ev_node_indexes, 0] = safe_ev_action_values.clamp(0.0, max_action)
    if return_details:
        details["total_action_correction_magnitude"] = (
            ev_action_values - safe_action[active_ev_node_indexes, 0]
        ).abs().sum()
        return safe_action, details
    return safe_action
