from collections import defaultdict

import numpy as np


DIAGNOSTIC_SCHEMA_VERSION = "1"
UNAVAILABLE = ""

EPISODE_DIAGNOSTIC_COLUMNS = [
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "config",
    "checkpoint_prefix",
    "run_name",
    "episode_steps",
    "done",
    "episode_reward",
    "max_action",
    "max_action_tolerance",
    "global_action_fraction_at_max_all_slots",
    "global_action_fraction_at_max_active",
    "global_action_mean_all_slots",
    "global_action_mean_active",
    "global_action_sum_active",
    "active_slot_count_mean",
    "nonzero_action_count_mean_all_slots",
    "inactive_slot_fraction_mean",
    "transformer_action_hhi_mean",
    "transformer_action_gini_mean",
    "charger_action_hhi_mean",
    "charger_action_gini_mean",
    "total_transformer_overload",
    "power_tracker_violation",
    "tracking_error",
    "energy_tracking_error",
    "total_ev_served",
    "total_energy_charged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "diagnostic_schema_version",
]

TRANSFORMER_DIAGNOSTIC_COLUMNS = [
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "transformer_id",
    "n_chargers_total",
    "n_active_chargers_seen",
    "n_active_ev_decisions",
    "n_all_slot_decisions",
    "action_sum_active",
    "action_mean_active",
    "action_max_active",
    "action_fraction_at_max_active",
    "action_sum_all_slots",
    "action_mean_all_slots",
    "action_max_all_slots",
    "action_fraction_at_max_all_slots",
    "charger_action_hhi_mean",
    "charger_action_gini_mean",
    "overload_magnitude_sum",
    "overload_magnitude_max",
    "overload_frequency_steps",
    "overload_frequency_fraction",
    "served_ev_count",
    "energy_charged_kwh",
    "user_satisfaction_mean",
    "diagnostic_schema_version",
]

CHARGER_DIAGNOSTIC_COLUMNS = [
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "charger_id",
    "transformer_id",
    "n_ports",
    "n_active_ev_decisions",
    "n_all_slot_decisions",
    "action_sum_active",
    "action_mean_active",
    "action_max_active",
    "action_fraction_at_max_active",
    "action_sum_all_slots",
    "action_mean_all_slots",
    "action_max_all_slots",
    "action_fraction_at_max_all_slots",
    "cs_power_sum_kwh",
    "cs_power_mean_kw",
    "cs_power_max_kw",
    "served_ev_count",
    "energy_charged_kwh",
    "user_satisfaction_mean",
    "diagnostic_schema_version",
]

SEED_SUMMARY_DIAGNOSTIC_COLUMNS = [
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "n_eval_episodes",
    "global_action_fraction_at_max_active_mean",
    "global_action_fraction_at_max_all_slots_mean",
    "transformer_action_fraction_at_max_active_mean",
    "charger_action_fraction_at_max_active_mean",
    "transformer_action_hhi_mean",
    "transformer_action_gini_mean",
    "charger_action_hhi_mean",
    "charger_action_gini_mean",
    "total_transformer_overload_mean",
    "power_tracker_violation_mean",
    "tracking_error_mean",
    "energy_tracking_error_mean",
    "total_ev_served_mean",
    "total_energy_charged_mean",
    "average_user_satisfaction_mean",
    "energy_user_satisfaction_mean",
    "diagnostic_schema_version",
]


def build_slot_to_charger_id(env):
    slot_to_charger_id = []
    for charging_station in env.charging_stations:
        slot_to_charger_id.extend([int(charging_station.id)] * int(charging_station.n_ports))
    return np.asarray(slot_to_charger_id, dtype=int)


def build_charger_to_transformer_id(env):
    return {
        int(charging_station.id): int(charging_station.connected_transformer)
        for charging_station in env.charging_stations
    }


def extract_active_action_slots(state):
    return np.asarray(getattr(state, "action_mapper", []), dtype=int).reshape(-1)


def extract_active_ev_infrastructure(state):
    ev_features = np.asarray(getattr(state, "ev_features", []), dtype=float)
    if ev_features.size == 0:
        ev_features = ev_features.reshape(0, 6)
    if ev_features.ndim != 2 or ev_features.shape[1] < 6:
        raise ValueError("state.ev_features must have at least 6 columns.")

    active_slots = extract_active_action_slots(state)
    if active_slots.shape[0] != ev_features.shape[0]:
        raise ValueError("state.action_mapper length must match active EV feature rows.")

    return {
        "active_slots": active_slots,
        "charger_ids": ev_features[:, 4].round().astype(int),
        "transformer_ids": ev_features[:, 5].round().astype(int),
    }


def validate_active_infrastructure_mapping(state, slot_to_charger_id, charger_to_transformer_id):
    active_metadata = extract_active_ev_infrastructure(state)
    active_slots = active_metadata["active_slots"]
    if active_slots.size == 0:
        return active_metadata

    slot_to_charger_id = np.asarray(slot_to_charger_id, dtype=int).reshape(-1)
    if np.any(active_slots < 0) or np.any(active_slots >= slot_to_charger_id.size):
        raise ValueError("Active action slot is outside the EV2Gym action space.")

    mapped_charger_ids = slot_to_charger_id[active_slots]
    if not np.array_equal(mapped_charger_ids, active_metadata["charger_ids"]):
        raise ValueError("Active slot charger mapping mismatch.")

    mapped_transformer_ids = []
    for charger_id in mapped_charger_ids:
        if int(charger_id) not in charger_to_transformer_id:
            raise ValueError("Active slot transformer mapping mismatch.")
        mapped_transformer_ids.append(int(charger_to_transformer_id[int(charger_id)]))

    if not np.array_equal(
        np.asarray(mapped_transformer_ids, dtype=int),
        active_metadata["transformer_ids"],
    ):
        raise ValueError("Active slot transformer mapping mismatch.")

    return active_metadata


def action_diagnostics(action_values, max_action, tolerance=1e-6):
    values = np.asarray(action_values, dtype=float).reshape(-1)
    if values.size == 0:
        return {
            "count": 0,
            "action_sum": 0.0,
            "action_mean": 0.0,
            "action_max": 0.0,
            "action_fraction_at_max": 0.0,
            "action_nonzero_fraction": 0.0,
        }

    at_max_threshold = float(max_action) - float(tolerance)
    return {
        "count": int(values.size),
        "action_sum": float(np.sum(values)),
        "action_mean": float(np.mean(values)),
        "action_max": float(np.max(values)),
        "action_fraction_at_max": float(np.mean(values >= at_max_threshold)),
        "action_nonzero_fraction": float(np.mean(np.abs(values) > float(tolerance))),
    }


def allocation_concentration(action_values, tolerance=1e-6):
    values = np.asarray(action_values, dtype=float).reshape(-1)
    if values.size == 0:
        return {"hhi": 0.0, "gini": 0.0, "zero_pressure": True}

    values = np.maximum(values, 0.0)
    total_pressure = float(np.sum(values))
    if total_pressure <= float(tolerance):
        return {"hhi": 0.0, "gini": 0.0, "zero_pressure": True}

    shares = values / total_pressure
    sorted_values = np.sort(values)
    value_count = sorted_values.size
    weighted_sum = np.sum((2 * np.arange(1, value_count + 1) - value_count - 1) * sorted_values)
    gini = weighted_sum / (value_count * total_pressure)
    return {
        "hhi": float(np.sum(shares ** 2)),
        "gini": float(gini),
        "zero_pressure": False,
    }


def aggregate_infrastructure_actions(
    mapped_actions_by_step,
    active_slots_by_step,
    slot_to_charger_id,
    charger_to_transformer_id,
    max_action,
    tolerance=1e-6,
    env=None,
):
    slot_to_charger_id = np.asarray(slot_to_charger_id, dtype=int).reshape(-1)
    action_dim = int(slot_to_charger_id.size)
    mapped_actions = [
        np.asarray(mapped_action, dtype=float).reshape(-1)
        for mapped_action in mapped_actions_by_step
    ]
    active_slots = [
        _valid_active_slots(active_slots_for_step, action_dim)
        for active_slots_for_step in active_slots_by_step
    ]

    for mapped_action in mapped_actions:
        if mapped_action.size != action_dim:
            raise ValueError("Each mapped action must match slot_to_charger_id length.")

    if len(mapped_actions) != len(active_slots):
        raise ValueError("mapped_actions_by_step and active_slots_by_step must have the same length.")

    charger_ids = _ordered_charger_ids(slot_to_charger_id, charger_to_transformer_id, env)
    transformer_ids = sorted({int(transformer_id) for transformer_id in charger_to_transformer_id.values()})
    charger_lookup = _charger_lookup(env)
    chargers_by_transformer = _chargers_by_transformer(charger_ids, charger_to_transformer_id)

    stacked_actions = (
        np.vstack(mapped_actions)
        if mapped_actions
        else np.zeros((0, action_dim), dtype=float)
    )
    active_values_by_step = [
        mapped_action[slots] if slots.size else np.asarray([], dtype=float)
        for mapped_action, slots in zip(mapped_actions, active_slots)
    ]
    all_values = stacked_actions.reshape(-1)
    active_values = _concatenate_or_empty(active_values_by_step)

    global_summary = _global_action_summary(
        mapped_actions=mapped_actions,
        active_slots_by_step=active_slots,
        slot_to_charger_id=slot_to_charger_id,
        charger_to_transformer_id=charger_to_transformer_id,
        charger_ids=charger_ids,
        transformer_ids=transformer_ids,
        max_action=max_action,
        tolerance=tolerance,
        all_values=all_values,
        active_values=active_values,
    )

    charger_rows = {
        charger_id: _charger_action_summary(
            charger_id=charger_id,
            mapped_actions=mapped_actions,
            active_slots_by_step=active_slots,
            slot_to_charger_id=slot_to_charger_id,
            charger_to_transformer_id=charger_to_transformer_id,
            max_action=max_action,
            tolerance=tolerance,
            env=env,
            charger=charger_lookup.get(charger_id),
        )
        for charger_id in charger_ids
    }

    transformer_rows = {
        transformer_id: _transformer_action_summary(
            transformer_id=transformer_id,
            mapped_actions=mapped_actions,
            active_slots_by_step=active_slots,
            slot_to_charger_id=slot_to_charger_id,
            charger_to_transformer_id=charger_to_transformer_id,
            charger_ids=chargers_by_transformer.get(transformer_id, []),
            charger_rows=charger_rows,
            max_action=max_action,
            tolerance=tolerance,
            env=env,
        )
        for transformer_id in transformer_ids
    }

    return {
        "global": global_summary,
        "chargers": charger_rows,
        "transformers": transformer_rows,
    }


def build_episode_row(
    metadata,
    episode_index,
    episode_seed,
    episode_record,
    action_summary,
    stats,
    max_action,
    tolerance,
):
    global_summary = action_summary["global"]
    row = {
        **metadata,
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "episode_steps": int(episode_record["episode_steps"]),
        "done": bool(episode_record["done"]),
        "episode_reward": float(episode_record["episode_reward"]),
        "max_action": float(max_action),
        "max_action_tolerance": float(tolerance),
        "global_action_fraction_at_max_all_slots": global_summary["action_fraction_at_max_all_slots"],
        "global_action_fraction_at_max_active": global_summary["action_fraction_at_max_active"],
        "global_action_mean_all_slots": global_summary["action_mean_all_slots"],
        "global_action_mean_active": global_summary["action_mean_active"],
        "global_action_sum_active": global_summary["action_sum_active"],
        "active_slot_count_mean": global_summary["active_slot_count_mean"],
        "nonzero_action_count_mean_all_slots": global_summary["nonzero_action_count_mean_all_slots"],
        "inactive_slot_fraction_mean": global_summary["inactive_slot_fraction_mean"],
        "transformer_action_hhi_mean": global_summary["transformer_action_hhi_mean"],
        "transformer_action_gini_mean": global_summary["transformer_action_gini_mean"],
        "charger_action_hhi_mean": global_summary["charger_action_hhi_mean"],
        "charger_action_gini_mean": global_summary["charger_action_gini_mean"],
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
    }
    for stat_key in [
        "total_transformer_overload",
        "power_tracker_violation",
        "tracking_error",
        "energy_tracking_error",
        "total_ev_served",
        "total_energy_charged",
        "average_user_satisfaction",
        "energy_user_satisfaction",
    ]:
        row[stat_key] = _float_stat(stats, stat_key)
    return row


def build_charger_rows(metadata, episode_index, episode_seed, action_summary):
    rows = []
    for charger_id, charger_summary in sorted(action_summary["chargers"].items()):
        rows.append({
            **metadata,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "charger_id": charger_id,
            "transformer_id": charger_summary["transformer_id"],
            "n_ports": charger_summary["n_ports"],
            "n_active_ev_decisions": charger_summary["n_active_ev_decisions"],
            "n_all_slot_decisions": charger_summary["n_all_slot_decisions"],
            "action_sum_active": charger_summary["action_sum_active"],
            "action_mean_active": charger_summary["action_mean_active"],
            "action_max_active": charger_summary["action_max_active"],
            "action_fraction_at_max_active": charger_summary["action_fraction_at_max_active"],
            "action_sum_all_slots": charger_summary["action_sum_all_slots"],
            "action_mean_all_slots": charger_summary["action_mean_all_slots"],
            "action_max_all_slots": charger_summary["action_max_all_slots"],
            "action_fraction_at_max_all_slots": charger_summary["action_fraction_at_max_all_slots"],
            "cs_power_sum_kwh": charger_summary["cs_power_sum_kwh"],
            "cs_power_mean_kw": charger_summary["cs_power_mean_kw"],
            "cs_power_max_kw": charger_summary["cs_power_max_kw"],
            "served_ev_count": charger_summary["served_ev_count"],
            "energy_charged_kwh": charger_summary["energy_charged_kwh"],
            "user_satisfaction_mean": charger_summary["user_satisfaction_mean"],
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        })
    return rows


def build_transformer_rows(metadata, episode_index, episode_seed, action_summary):
    rows = []
    for transformer_id, transformer_summary in sorted(action_summary["transformers"].items()):
        rows.append({
            **metadata,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "transformer_id": transformer_id,
            "n_chargers_total": transformer_summary["n_chargers_total"],
            "n_active_chargers_seen": transformer_summary["n_active_chargers_seen"],
            "n_active_ev_decisions": transformer_summary["n_active_ev_decisions"],
            "n_all_slot_decisions": transformer_summary["n_all_slot_decisions"],
            "action_sum_active": transformer_summary["action_sum_active"],
            "action_mean_active": transformer_summary["action_mean_active"],
            "action_max_active": transformer_summary["action_max_active"],
            "action_fraction_at_max_active": transformer_summary["action_fraction_at_max_active"],
            "action_sum_all_slots": transformer_summary["action_sum_all_slots"],
            "action_mean_all_slots": transformer_summary["action_mean_all_slots"],
            "action_max_all_slots": transformer_summary["action_max_all_slots"],
            "action_fraction_at_max_all_slots": transformer_summary["action_fraction_at_max_all_slots"],
            "charger_action_hhi_mean": transformer_summary["charger_action_hhi_mean"],
            "charger_action_gini_mean": transformer_summary["charger_action_gini_mean"],
            "overload_magnitude_sum": transformer_summary["overload_magnitude_sum"],
            "overload_magnitude_max": transformer_summary["overload_magnitude_max"],
            "overload_frequency_steps": transformer_summary["overload_frequency_steps"],
            "overload_frequency_fraction": transformer_summary["overload_frequency_fraction"],
            "served_ev_count": transformer_summary["served_ev_count"],
            "energy_charged_kwh": transformer_summary["energy_charged_kwh"],
            "user_satisfaction_mean": transformer_summary["user_satisfaction_mean"],
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        })
    return rows


def build_seed_summary_row(metadata, episode_rows):
    return {
        **metadata,
        "n_eval_episodes": len(episode_rows),
        "global_action_fraction_at_max_active_mean": _mean_existing(
            episode_rows, "global_action_fraction_at_max_active"
        ),
        "global_action_fraction_at_max_all_slots_mean": _mean_existing(
            episode_rows, "global_action_fraction_at_max_all_slots"
        ),
        "transformer_action_fraction_at_max_active_mean": _mean_existing(
            episode_rows, "transformer_action_fraction_at_max_active"
        ),
        "charger_action_fraction_at_max_active_mean": _mean_existing(
            episode_rows, "charger_action_fraction_at_max_active"
        ),
        "transformer_action_hhi_mean": _mean_existing(episode_rows, "transformer_action_hhi_mean"),
        "transformer_action_gini_mean": _mean_existing(episode_rows, "transformer_action_gini_mean"),
        "charger_action_hhi_mean": _mean_existing(episode_rows, "charger_action_hhi_mean"),
        "charger_action_gini_mean": _mean_existing(episode_rows, "charger_action_gini_mean"),
        "total_transformer_overload_mean": _mean_existing(episode_rows, "total_transformer_overload"),
        "power_tracker_violation_mean": _mean_existing(episode_rows, "power_tracker_violation"),
        "tracking_error_mean": _mean_existing(episode_rows, "tracking_error"),
        "energy_tracking_error_mean": _mean_existing(episode_rows, "energy_tracking_error"),
        "total_ev_served_mean": _mean_existing(episode_rows, "total_ev_served"),
        "total_energy_charged_mean": _mean_existing(episode_rows, "total_energy_charged"),
        "average_user_satisfaction_mean": _mean_existing(episode_rows, "average_user_satisfaction"),
        "energy_user_satisfaction_mean": _mean_existing(episode_rows, "energy_user_satisfaction"),
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
    }


def _valid_active_slots(active_slots_for_step, action_dim):
    active_slots = np.asarray(active_slots_for_step, dtype=int).reshape(-1)
    return active_slots[(active_slots >= 0) & (active_slots < action_dim)]


def _ordered_charger_ids(slot_to_charger_id, charger_to_transformer_id, env):
    charger_ids = {int(charger_id) for charger_id in np.unique(slot_to_charger_id)}
    charger_ids.update(int(charger_id) for charger_id in charger_to_transformer_id)
    if env is not None:
        charger_ids.update(int(charging_station.id) for charging_station in env.charging_stations)
    return sorted(charger_ids)


def _charger_lookup(env):
    if env is None:
        return {}
    return {int(charging_station.id): charging_station for charging_station in env.charging_stations}


def _chargers_by_transformer(charger_ids, charger_to_transformer_id):
    chargers_by_transformer = defaultdict(list)
    for charger_id in charger_ids:
        transformer_id = charger_to_transformer_id.get(int(charger_id))
        if transformer_id is not None:
            chargers_by_transformer[int(transformer_id)].append(int(charger_id))
    return chargers_by_transformer


def _concatenate_or_empty(value_arrays):
    non_empty_arrays = [
        np.asarray(values, dtype=float).reshape(-1)
        for values in value_arrays
        if np.asarray(values).size > 0
    ]
    if not non_empty_arrays:
        return np.asarray([], dtype=float)
    return np.concatenate(non_empty_arrays)


def _global_action_summary(
    mapped_actions,
    active_slots_by_step,
    slot_to_charger_id,
    charger_to_transformer_id,
    charger_ids,
    transformer_ids,
    max_action,
    tolerance,
    all_values,
    active_values,
):
    all_slot_diagnostics = action_diagnostics(all_values, max_action=max_action, tolerance=tolerance)
    active_diagnostics = action_diagnostics(active_values, max_action=max_action, tolerance=tolerance)
    action_dim = int(slot_to_charger_id.size)
    active_slot_counts = [int(active_slots.size) for active_slots in active_slots_by_step]
    nonzero_action_counts = [
        int(np.count_nonzero(np.abs(mapped_action) > float(tolerance)))
        for mapped_action in mapped_actions
    ]
    inactive_slot_fractions = [
        (action_dim - active_slot_count) / action_dim
        for active_slot_count in active_slot_counts
        if action_dim > 0
    ]

    charger_concentrations = []
    transformer_concentrations = []
    for mapped_action, active_slots in zip(mapped_actions, active_slots_by_step):
        active_slot_mask = np.zeros(action_dim, dtype=bool)
        active_slot_mask[active_slots] = True
        charger_concentrations.append(allocation_concentration([
            _positive_action_sum(mapped_action, active_slot_mask & (slot_to_charger_id == charger_id))
            for charger_id in charger_ids
        ], tolerance=tolerance))
        transformer_concentrations.append(allocation_concentration([
            _positive_action_sum(
                mapped_action,
                active_slot_mask & _transformer_slot_mask(
                    slot_to_charger_id, charger_to_transformer_id, transformer_id
                ),
            )
            for transformer_id in transformer_ids
        ], tolerance=tolerance))

    return {
        "action_fraction_at_max_all_slots": all_slot_diagnostics["action_fraction_at_max"],
        "action_fraction_at_max_active": active_diagnostics["action_fraction_at_max"],
        "action_mean_all_slots": all_slot_diagnostics["action_mean"],
        "action_mean_active": active_diagnostics["action_mean"],
        "action_sum_active": active_diagnostics["action_sum"],
        "active_slot_count_mean": _mean_values(active_slot_counts),
        "nonzero_action_count_mean_all_slots": _mean_values(nonzero_action_counts),
        "inactive_slot_fraction_mean": _mean_values(inactive_slot_fractions),
        "transformer_action_hhi_mean": _mean_concentration(transformer_concentrations, "hhi"),
        "transformer_action_gini_mean": _mean_concentration(transformer_concentrations, "gini"),
        "charger_action_hhi_mean": _mean_concentration(charger_concentrations, "hhi"),
        "charger_action_gini_mean": _mean_concentration(charger_concentrations, "gini"),
    }


def _charger_action_summary(
    charger_id,
    mapped_actions,
    active_slots_by_step,
    slot_to_charger_id,
    charger_to_transformer_id,
    max_action,
    tolerance,
    env,
    charger,
):
    charger_slot_mask = slot_to_charger_id == int(charger_id)
    all_values_by_step = [mapped_action[charger_slot_mask] for mapped_action in mapped_actions]
    active_values_by_step = []
    for mapped_action, active_slots in zip(mapped_actions, active_slots_by_step):
        active_slot_mask = np.zeros(slot_to_charger_id.size, dtype=bool)
        active_slot_mask[active_slots] = True
        active_values_by_step.append(mapped_action[active_slot_mask & charger_slot_mask])

    all_diagnostics = action_diagnostics(
        _concatenate_or_empty(all_values_by_step),
        max_action=max_action,
        tolerance=tolerance,
    )
    active_diagnostics = action_diagnostics(
        _concatenate_or_empty(active_values_by_step),
        max_action=max_action,
        tolerance=tolerance,
    )
    power_summary = _charger_power_summary(env, int(charger_id))
    service_summary = _charger_service_summary(charger)

    return {
        "transformer_id": int(charger_to_transformer_id.get(int(charger_id), -1)),
        "n_ports": int(np.count_nonzero(charger_slot_mask)),
        "n_active_ev_decisions": active_diagnostics["count"],
        "n_all_slot_decisions": all_diagnostics["count"],
        "action_sum_active": active_diagnostics["action_sum"],
        "action_mean_active": active_diagnostics["action_mean"],
        "action_max_active": active_diagnostics["action_max"],
        "action_fraction_at_max_active": active_diagnostics["action_fraction_at_max"],
        "action_nonzero_fraction_active": active_diagnostics["action_nonzero_fraction"],
        "action_sum_all_slots": all_diagnostics["action_sum"],
        "action_mean_all_slots": all_diagnostics["action_mean"],
        "action_max_all_slots": all_diagnostics["action_max"],
        "action_fraction_at_max_all_slots": all_diagnostics["action_fraction_at_max"],
        "cs_power_sum_kwh": power_summary["cs_power_sum_kwh"],
        "cs_power_mean_kw": power_summary["cs_power_mean_kw"],
        "cs_power_max_kw": power_summary["cs_power_max_kw"],
        "served_ev_count": service_summary["served_ev_count"],
        "energy_charged_kwh": service_summary["energy_charged_kwh"],
        "user_satisfaction_mean": service_summary["user_satisfaction_mean"],
    }


def _transformer_action_summary(
    transformer_id,
    mapped_actions,
    active_slots_by_step,
    slot_to_charger_id,
    charger_to_transformer_id,
    charger_ids,
    charger_rows,
    max_action,
    tolerance,
    env,
):
    transformer_slot_mask = _transformer_slot_mask(
        slot_to_charger_id, charger_to_transformer_id, transformer_id
    )
    all_values_by_step = [mapped_action[transformer_slot_mask] for mapped_action in mapped_actions]
    active_values_by_step = []
    active_charger_ids = set()
    charger_concentrations = []

    for mapped_action, active_slots in zip(mapped_actions, active_slots_by_step):
        active_slot_mask = np.zeros(slot_to_charger_id.size, dtype=bool)
        active_slot_mask[active_slots] = True
        active_transformer_mask = active_slot_mask & transformer_slot_mask
        active_values_by_step.append(mapped_action[active_transformer_mask])
        active_charger_ids.update(int(charger_id) for charger_id in np.unique(slot_to_charger_id[active_transformer_mask]))
        charger_concentrations.append(allocation_concentration([
            _positive_action_sum(mapped_action, active_slot_mask & (slot_to_charger_id == charger_id))
            for charger_id in charger_ids
        ], tolerance=tolerance))

    all_diagnostics = action_diagnostics(
        _concatenate_or_empty(all_values_by_step),
        max_action=max_action,
        tolerance=tolerance,
    )
    active_diagnostics = action_diagnostics(
        _concatenate_or_empty(active_values_by_step),
        max_action=max_action,
        tolerance=tolerance,
    )
    overload_summary = _transformer_overload_summary(env, int(transformer_id), tolerance)
    service_summary = _transformer_service_summary(charger_ids, charger_rows)

    return {
        "n_chargers_total": len(charger_ids),
        "n_active_chargers_seen": len(active_charger_ids),
        "n_active_ev_decisions": active_diagnostics["count"],
        "n_all_slot_decisions": all_diagnostics["count"],
        "action_sum_active": active_diagnostics["action_sum"],
        "action_mean_active": active_diagnostics["action_mean"],
        "action_max_active": active_diagnostics["action_max"],
        "action_fraction_at_max_active": active_diagnostics["action_fraction_at_max"],
        "action_nonzero_fraction_active": active_diagnostics["action_nonzero_fraction"],
        "action_sum_all_slots": all_diagnostics["action_sum"],
        "action_mean_all_slots": all_diagnostics["action_mean"],
        "action_max_all_slots": all_diagnostics["action_max"],
        "action_fraction_at_max_all_slots": all_diagnostics["action_fraction_at_max"],
        "charger_action_hhi_mean": _mean_concentration(charger_concentrations, "hhi"),
        "charger_action_gini_mean": _mean_concentration(charger_concentrations, "gini"),
        "overload_magnitude_sum": overload_summary["overload_magnitude_sum"],
        "overload_magnitude_max": overload_summary["overload_magnitude_max"],
        "overload_frequency_steps": overload_summary["overload_frequency_steps"],
        "overload_frequency_fraction": overload_summary["overload_frequency_fraction"],
        "served_ev_count": service_summary["served_ev_count"],
        "energy_charged_kwh": service_summary["energy_charged_kwh"],
        "user_satisfaction_mean": service_summary["user_satisfaction_mean"],
    }


def _transformer_slot_mask(slot_to_charger_id, charger_to_transformer_id, transformer_id):
    transformer_charger_ids = {
        int(charger_id)
        for charger_id, mapped_transformer_id in charger_to_transformer_id.items()
        if int(mapped_transformer_id) == int(transformer_id)
    }
    return np.asarray([
        int(charger_id) in transformer_charger_ids
        for charger_id in slot_to_charger_id
    ], dtype=bool)


def _positive_action_sum(mapped_action, slot_mask):
    if not np.any(slot_mask):
        return 0.0
    return float(np.sum(np.maximum(mapped_action[slot_mask], 0.0)))


def _charger_power_summary(env, charger_id):
    if env is None or not hasattr(env, "cs_power"):
        return {
            "cs_power_sum_kwh": UNAVAILABLE,
            "cs_power_mean_kw": UNAVAILABLE,
            "cs_power_max_kw": UNAVAILABLE,
        }

    cs_power = np.asarray(env.cs_power, dtype=float)
    if cs_power.ndim < 2 or charger_id < 0 or charger_id >= cs_power.shape[0]:
        return {
            "cs_power_sum_kwh": UNAVAILABLE,
            "cs_power_mean_kw": UNAVAILABLE,
            "cs_power_max_kw": UNAVAILABLE,
        }

    charger_power = cs_power[charger_id]
    if charger_power.size == 0:
        return {
            "cs_power_sum_kwh": UNAVAILABLE,
            "cs_power_mean_kw": UNAVAILABLE,
            "cs_power_max_kw": UNAVAILABLE,
        }
    timescale_hours = float(getattr(env, "timescale", 0.0)) / 60.0
    return {
        "cs_power_sum_kwh": float(np.sum(charger_power) * timescale_hours),
        "cs_power_mean_kw": float(np.mean(charger_power)),
        "cs_power_max_kw": float(np.max(charger_power)),
    }


def _charger_service_summary(charger):
    if charger is None:
        return {
            "served_ev_count": UNAVAILABLE,
            "energy_charged_kwh": UNAVAILABLE,
            "user_satisfaction_mean": UNAVAILABLE,
        }

    satisfaction_source = getattr(charger, "all_user_satisfaction", None)
    satisfaction_values = (
        np.asarray(satisfaction_source, dtype=float)
        if satisfaction_source is not None
        else np.asarray([], dtype=float)
    )
    return {
        "served_ev_count": (
            int(getattr(charger, "total_evs_served"))
            if hasattr(charger, "total_evs_served")
            else UNAVAILABLE
        ),
        "energy_charged_kwh": (
            float(getattr(charger, "total_energy_charged"))
            if hasattr(charger, "total_energy_charged")
            else UNAVAILABLE
        ),
        "user_satisfaction_mean": (
            float(np.mean(satisfaction_values))
            if satisfaction_values.size
            else (0.0 if satisfaction_source is not None else UNAVAILABLE)
        ),
    }


def _transformer_overload_summary(env, transformer_id, tolerance):
    if env is None or not hasattr(env, "tr_overload"):
        return {
            "overload_magnitude_sum": UNAVAILABLE,
            "overload_magnitude_max": UNAVAILABLE,
            "overload_frequency_steps": UNAVAILABLE,
            "overload_frequency_fraction": UNAVAILABLE,
        }

    tr_overload = np.asarray(env.tr_overload, dtype=float)
    if tr_overload.ndim < 2 or transformer_id < 0 or transformer_id >= tr_overload.shape[0]:
        return {
            "overload_magnitude_sum": UNAVAILABLE,
            "overload_magnitude_max": UNAVAILABLE,
            "overload_frequency_steps": UNAVAILABLE,
            "overload_frequency_fraction": UNAVAILABLE,
        }

    transformer_overload = tr_overload[transformer_id]
    if transformer_overload.size == 0:
        return {
            "overload_magnitude_sum": UNAVAILABLE,
            "overload_magnitude_max": UNAVAILABLE,
            "overload_frequency_steps": UNAVAILABLE,
            "overload_frequency_fraction": UNAVAILABLE,
        }
    frequency_steps = int(np.count_nonzero(transformer_overload > float(tolerance)))
    return {
        "overload_magnitude_sum": float(np.sum(transformer_overload)),
        "overload_magnitude_max": float(np.max(transformer_overload)),
        "overload_frequency_steps": frequency_steps,
        "overload_frequency_fraction": float(frequency_steps / transformer_overload.size),
    }


def _transformer_service_summary(charger_ids, charger_rows):
    served_ev_count = 0
    energy_charged_kwh = 0.0
    satisfaction_means = []
    satisfaction_weights = []
    served_seen = False
    energy_seen = False

    for charger_id in charger_ids:
        charger_row = charger_rows.get(charger_id, {})
        served_value = charger_row.get("served_ev_count", UNAVAILABLE)
        if _is_available_number(served_value):
            served_seen = True
            served = int(served_value)
            served_ev_count += served
        else:
            served = 0

        energy_charged_value = charger_row.get("energy_charged_kwh", UNAVAILABLE)
        if _is_available_number(energy_charged_value):
            energy_seen = True
            energy_charged_kwh += float(energy_charged_value)

        satisfaction_value = charger_row.get("user_satisfaction_mean", UNAVAILABLE)
        if served > 0 and _is_available_number(satisfaction_value):
            satisfaction_means.append(float(satisfaction_value))
            satisfaction_weights.append(served)

    user_satisfaction_mean = (
        float(np.average(satisfaction_means, weights=satisfaction_weights))
        if satisfaction_weights
        else (0.0 if served_seen else UNAVAILABLE)
    )
    return {
        "served_ev_count": served_ev_count if served_seen else UNAVAILABLE,
        "energy_charged_kwh": energy_charged_kwh if energy_seen else UNAVAILABLE,
        "user_satisfaction_mean": user_satisfaction_mean,
    }


def _mean_concentration(concentration_rows, key):
    if not concentration_rows:
        return 0.0
    return _mean_values([concentration[key] for concentration in concentration_rows])


def _mean_values(values):
    numeric_values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(numeric_values)) if numeric_values else 0.0


def _float_stat(stats, stat_key):
    value = stats.get(stat_key, 0.0) if stats else 0.0
    if np.isscalar(value):
        return float(value)
    return 0.0


def _is_available_number(value):
    if value in (UNAVAILABLE, None):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _mean_existing(rows, key):
    values = []
    for row in rows:
        if key not in row:
            continue
        value = row[key]
        if value in ("", None):
            continue
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            values.append(numeric_value)
    return float(np.mean(values)) if values else 0.0
