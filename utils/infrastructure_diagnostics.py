from collections import defaultdict

import numpy as np


DIAGNOSTIC_SCHEMA_VERSION = "3"
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
    "environment_action_low",
    "environment_action_high",
    "observed_action_min_active",
    "observed_action_max_active",
    "action_tolerance",
    "environment_action_domain_support",
    "v2g_enabled",
    "v2g_enabled_source",
    "global_action_fraction_at_max_all_slots",
    "global_action_fraction_at_max_active",
    "global_action_nonzero_fraction_active",
    "active_action_decision_count",
    "active_action_below_environment_low_count",
    "active_action_below_environment_low_fraction",
    "active_action_above_environment_high_count",
    "active_action_above_environment_high_fraction",
    "global_positive_action_fraction_active",
    "global_zero_action_fraction_active",
    "global_negative_action_fraction_active",
    "global_action_fraction_at_positive_max_active",
    "global_action_fraction_at_negative_min_active",
    "global_positive_action_sum_active",
    "global_negative_action_magnitude_sum_active",
    "global_action_mean_all_slots",
    "global_action_mean_active",
    "global_action_sum_active",
    "active_slot_count_mean",
    "nonzero_action_count_mean_all_slots",
    "inactive_slot_fraction_mean",
    "inactive_slot_decision_count",
    "inactive_nonzero_action_count",
    "inactive_nonzero_action_fraction_all_slots",
    "transformer_action_fraction_at_max_active_macro_mean",
    "charger_action_fraction_at_max_active_macro_mean",
    "transformer_action_nonzero_fraction_active_macro_mean",
    "charger_action_nonzero_fraction_active_macro_mean",
    "transformer_positive_charge_action_hhi_mean",
    "transformer_positive_charge_action_gini_mean",
    "transformer_allocation_zero_pressure_step_fraction",
    "transformer_allocation_valid_step_count",
    "charger_positive_charge_action_hhi_mean",
    "charger_positive_charge_action_gini_mean",
    "charger_allocation_zero_pressure_step_fraction",
    "charger_allocation_valid_step_count",
    "total_transformer_overload",
    "power_tracker_violation",
    "tracking_error",
    "energy_tracking_error",
    "total_ev_served",
    "total_energy_charged",
    "total_energy_discharged",
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
    "action_nonzero_fraction_active",
    "positive_action_fraction_active",
    "zero_action_fraction_active",
    "negative_action_fraction_active",
    "action_fraction_at_positive_max_active",
    "action_fraction_at_negative_min_active",
    "positive_action_sum_active",
    "negative_action_magnitude_sum_active",
    "action_sum_all_slots",
    "action_mean_all_slots",
    "action_max_all_slots",
    "action_fraction_at_max_all_slots",
    "charger_positive_charge_action_hhi_mean",
    "charger_positive_charge_action_gini_mean",
    "charger_allocation_zero_pressure_step_fraction",
    "charger_allocation_valid_step_count",
    "overload_magnitude_sum",
    "overload_magnitude_max",
    "overload_frequency_steps",
    "overload_frequency_fraction",
    "served_ev_count",
    "energy_charged_kwh",
    "energy_discharged_kwh",
    "user_satisfaction_sum",
    "user_satisfaction_mean",
    "user_satisfaction_mean_served_ev_weighted",
    "user_satisfaction_observation_count",
    "user_satisfaction_source",
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
    "action_nonzero_fraction_active",
    "positive_action_fraction_active",
    "zero_action_fraction_active",
    "negative_action_fraction_active",
    "action_fraction_at_positive_max_active",
    "action_fraction_at_negative_min_active",
    "positive_action_sum_active",
    "negative_action_magnitude_sum_active",
    "action_sum_all_slots",
    "action_mean_all_slots",
    "action_max_all_slots",
    "action_fraction_at_max_all_slots",
    "cs_power_sum_kwh",
    "cs_power_mean_kw",
    "cs_power_max_kw",
    "served_ev_count",
    "energy_charged_kwh",
    "energy_discharged_kwh",
    "user_satisfaction_sum",
    "user_satisfaction_mean",
    "user_satisfaction_observation_count",
    "user_satisfaction_source",
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
    "global_action_nonzero_fraction_active_mean",
    "active_action_decision_count_mean",
    "active_action_below_environment_low_count",
    "active_action_below_environment_low_fraction",
    "active_action_above_environment_high_count",
    "active_action_above_environment_high_fraction",
    "global_positive_action_fraction_active_mean",
    "global_zero_action_fraction_active_mean",
    "global_negative_action_fraction_active_mean",
    "global_action_fraction_at_positive_max_active_mean",
    "global_action_fraction_at_negative_min_active_mean",
    "global_positive_action_sum_active_mean",
    "global_negative_action_magnitude_sum_active_mean",
    "observed_action_min_active_mean",
    "observed_action_max_active_mean",
    "environment_action_low",
    "environment_action_high",
    "action_tolerance",
    "environment_action_domain_support",
    "v2g_enabled",
    "v2g_enabled_source",
    "inactive_slot_decision_count_mean",
    "inactive_nonzero_action_count_mean",
    "inactive_nonzero_action_fraction_all_slots_mean",
    "transformer_action_fraction_at_max_active_macro_mean",
    "charger_action_fraction_at_max_active_macro_mean",
    "transformer_action_nonzero_fraction_active_macro_mean",
    "charger_action_nonzero_fraction_active_macro_mean",
    "transformer_positive_charge_action_hhi_mean",
    "transformer_positive_charge_action_gini_mean",
    "transformer_allocation_zero_pressure_step_fraction_mean",
    "transformer_allocation_valid_step_count_mean",
    "charger_positive_charge_action_hhi_mean",
    "charger_positive_charge_action_gini_mean",
    "charger_allocation_zero_pressure_step_fraction_mean",
    "charger_allocation_valid_step_count_mean",
    "total_transformer_overload_mean",
    "power_tracker_violation_mean",
    "tracking_error_mean",
    "energy_tracking_error_mean",
    "total_ev_served_mean",
    "total_energy_charged_mean",
    "total_energy_discharged_mean",
    "average_user_satisfaction_mean",
    "energy_user_satisfaction_mean",
    "diagnostic_schema_version",
]


def build_slot_to_charger_id(env):
    slot_to_charger_id = []
    seen_charger_ids = set()
    for charging_station in env.charging_stations:
        charger_id = _validated_non_negative_integral_scalar(
            getattr(charging_station, "id", None),
            "charging station ID",
        )
        if charger_id in seen_charger_ids:
            raise ValueError("Duplicate charging station IDs are unsupported.")
        seen_charger_ids.add(charger_id)
        slot_to_charger_id.extend([charger_id] * int(charging_station.n_ports))
    return np.asarray(slot_to_charger_id, dtype=int)


def build_charger_to_transformer_id(env):
    charger_to_transformer_id = {}
    for charging_station in env.charging_stations:
        charger_id = _validated_non_negative_integral_scalar(
            getattr(charging_station, "id", None),
            "charging station ID",
        )
        if charger_id in charger_to_transformer_id:
            raise ValueError("Duplicate charging station IDs are unsupported.")
        transformer_id = _validated_non_negative_integral_scalar(
            getattr(charging_station, "connected_transformer", None),
            "connected transformer ID",
        )
        charger_to_transformer_id[charger_id] = transformer_id
    return charger_to_transformer_id


def extract_active_action_slots(state):
    try:
        return np.asarray(getattr(state, "action_mapper", []), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("state.action_mapper must contain numeric action slot IDs.") from exc


def validate_diagnostic_action_contract(
    mapped_action,
    active_slots,
    action_dim,
    tolerance=1e-6,
):
    values = np.asarray(mapped_action, dtype=float).reshape(-1)
    if values.size != int(action_dim):
        raise ValueError("Mapped action length must match EV2Gym action dimension.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Mapped action contains non-finite values.")
    return _validated_active_slots(active_slots, action_dim, tolerance=tolerance)


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
        "charger_ids": _validated_non_negative_integral_values(
            ev_features[:, 4],
            "EV feature charger IDs",
        ),
        "transformer_ids": _validated_non_negative_integral_values(
            ev_features[:, 5],
            "EV feature transformer IDs",
        ),
    }


def validate_active_infrastructure_mapping(state, slot_to_charger_id, charger_to_transformer_id):
    active_metadata = extract_active_ev_infrastructure(state)
    slot_to_charger_id = np.asarray(slot_to_charger_id, dtype=int).reshape(-1)
    active_slots = _validated_active_slots(
        active_metadata["active_slots"],
        slot_to_charger_id.size,
        tolerance=0.0,
    )
    active_metadata["active_slots"] = active_slots
    if active_slots.size == 0:
        return active_metadata

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


def validate_environment_action_bounds(action_space, tolerance=1e-6):
    if action_space is None or not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise ValueError("Environment action_space must expose low and high bounds.")

    low_values = np.asarray(action_space.low, dtype=float).reshape(-1)
    high_values = np.asarray(action_space.high, dtype=float).reshape(-1)
    if low_values.size == 0 or high_values.size == 0:
        raise ValueError("Environment action-space bounds must be non-empty.")
    if low_values.size != high_values.size:
        raise ValueError("Environment action-space low/high bounds must have matching sizes.")
    if not np.all(np.isfinite(low_values)) or not np.all(np.isfinite(high_values)):
        raise ValueError("Environment action-space bounds must be finite.")
    if np.any(low_values > high_values):
        raise ValueError("Environment action-space low bounds must not exceed high bounds.")

    low = float(low_values[0])
    high = float(high_values[0])
    tolerance = float(tolerance)
    if not np.all(np.abs(low_values - low) <= tolerance):
        raise ValueError("heterogeneous action-space low bounds are unsupported by diagnostic schema v3.")
    if not np.all(np.abs(high_values - high) <= tolerance):
        raise ValueError("heterogeneous action-space high bounds are unsupported by diagnostic schema v3.")

    return {
        "environment_action_low": low,
        "environment_action_high": high,
        "environment_action_domain_support": _environment_action_domain_support(low, high, tolerance),
    }


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


def signed_action_diagnostics(
    action_values,
    environment_action_low=UNAVAILABLE,
    environment_action_high=UNAVAILABLE,
    tolerance=1e-6,
):
    values = np.asarray(action_values, dtype=float).reshape(-1)
    tolerance = float(tolerance)
    if values.size == 0:
        return {
            "active_action_decision_count": 0,
            "observed_action_min_active": UNAVAILABLE,
            "observed_action_max_active": UNAVAILABLE,
            "positive_action_fraction_active": UNAVAILABLE,
            "zero_action_fraction_active": UNAVAILABLE,
            "negative_action_fraction_active": UNAVAILABLE,
            "action_fraction_at_positive_max_active": UNAVAILABLE,
            "action_fraction_at_negative_min_active": UNAVAILABLE,
            "positive_action_sum_active": 0.0,
            "negative_action_magnitude_sum_active": 0.0,
        }

    positive_mask = values > tolerance
    zero_mask = np.abs(values) <= tolerance
    negative_mask = values < -tolerance

    positive_max_fraction = UNAVAILABLE
    if _is_available_number(environment_action_high):
        positive_max_fraction = float(
            np.mean(values >= float(environment_action_high) - tolerance)
        )

    negative_min_fraction = UNAVAILABLE
    if (
        _is_available_number(environment_action_low)
        and float(environment_action_low) < -tolerance
    ):
        negative_min_fraction = float(
            np.mean(values <= float(environment_action_low) + tolerance)
        )

    diagnostics = {
        "active_action_decision_count": int(values.size),
        "observed_action_min_active": float(np.min(values)),
        "observed_action_max_active": float(np.max(values)),
        "positive_action_fraction_active": float(np.mean(positive_mask)),
        "zero_action_fraction_active": float(np.mean(zero_mask)),
        "negative_action_fraction_active": float(np.mean(negative_mask)),
        "action_fraction_at_positive_max_active": positive_max_fraction,
        "action_fraction_at_negative_min_active": negative_min_fraction,
        "positive_action_sum_active": float(np.sum(values[positive_mask])),
        "negative_action_magnitude_sum_active": float(np.sum(np.abs(values[negative_mask]))),
    }
    validate_signed_fraction_invariant(diagnostics, tolerance=tolerance)
    return diagnostics


def validate_signed_fraction_invariant(signed_diagnostics, tolerance=1e-6):
    active_count = int(signed_diagnostics.get("active_action_decision_count", 0))
    if active_count <= 0:
        return

    fraction_keys = [
        "positive_action_fraction_active",
        "zero_action_fraction_active",
        "negative_action_fraction_active",
    ]
    fractions = []
    for key in fraction_keys:
        value = signed_diagnostics.get(key, UNAVAILABLE)
        if not _is_available_number(value):
            raise ValueError("Signed action fractions must be finite when active decisions exist.")
        fractions.append(float(value))
    if abs(sum(fractions) - 1.0) > float(tolerance):
        raise ValueError("Signed action fractions must sum to one.")


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

    if len(mapped_actions) != len(active_slots_by_step):
        raise ValueError("mapped_actions_by_step and active_slots_by_step must have the same length.")
    active_slots = [
        validate_diagnostic_action_contract(
            mapped_action=mapped_action,
            active_slots=active_slots_for_step,
            action_dim=action_dim,
            tolerance=tolerance,
        )
        for mapped_action, active_slots_for_step in zip(mapped_actions, active_slots_by_step)
    ]

    charger_ids = _ordered_charger_ids(slot_to_charger_id, charger_to_transformer_id, env)
    transformer_ids = sorted({int(transformer_id) for transformer_id in charger_to_transformer_id.values()})
    charger_lookup = _charger_lookup(env)
    chargers_by_transformer = _chargers_by_transformer(charger_ids, charger_to_transformer_id)
    action_bounds = _environment_action_bounds(env, max_action, tolerance)

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
        action_bounds=action_bounds,
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
            action_bounds=action_bounds,
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
            action_bounds=action_bounds,
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
        "environment_action_low": global_summary.get("environment_action_low", UNAVAILABLE),
        "environment_action_high": global_summary.get("environment_action_high", UNAVAILABLE),
        "observed_action_min_active": global_summary["observed_action_min_active"],
        "observed_action_max_active": global_summary["observed_action_max_active"],
        "action_tolerance": float(tolerance),
        "environment_action_domain_support": global_summary.get(
            "environment_action_domain_support", UNAVAILABLE
        ),
        "v2g_enabled": metadata.get("v2g_enabled", UNAVAILABLE),
        "v2g_enabled_source": metadata.get("v2g_enabled_source", UNAVAILABLE),
        "global_action_fraction_at_max_all_slots": global_summary["action_fraction_at_max_all_slots"],
        "global_action_fraction_at_max_active": global_summary["action_fraction_at_max_active"],
        "global_action_nonzero_fraction_active": global_summary["action_nonzero_fraction_active"],
        "active_action_decision_count": global_summary["active_action_decision_count"],
        "active_action_below_environment_low_count": global_summary[
            "active_action_below_environment_low_count"
        ],
        "active_action_below_environment_low_fraction": global_summary[
            "active_action_below_environment_low_fraction"
        ],
        "active_action_above_environment_high_count": global_summary[
            "active_action_above_environment_high_count"
        ],
        "active_action_above_environment_high_fraction": global_summary[
            "active_action_above_environment_high_fraction"
        ],
        "global_positive_action_fraction_active": global_summary["positive_action_fraction_active"],
        "global_zero_action_fraction_active": global_summary["zero_action_fraction_active"],
        "global_negative_action_fraction_active": global_summary["negative_action_fraction_active"],
        "global_action_fraction_at_positive_max_active": global_summary[
            "action_fraction_at_positive_max_active"
        ],
        "global_action_fraction_at_negative_min_active": global_summary[
            "action_fraction_at_negative_min_active"
        ],
        "global_positive_action_sum_active": global_summary["positive_action_sum_active"],
        "global_negative_action_magnitude_sum_active": global_summary[
            "negative_action_magnitude_sum_active"
        ],
        "global_action_mean_all_slots": global_summary["action_mean_all_slots"],
        "global_action_mean_active": global_summary["action_mean_active"],
        "global_action_sum_active": global_summary["action_sum_active"],
        "active_slot_count_mean": global_summary["active_slot_count_mean"],
        "nonzero_action_count_mean_all_slots": global_summary["nonzero_action_count_mean_all_slots"],
        "inactive_slot_fraction_mean": global_summary["inactive_slot_fraction_mean"],
        "inactive_slot_decision_count": global_summary["inactive_slot_decision_count"],
        "inactive_nonzero_action_count": global_summary["inactive_nonzero_action_count"],
        "inactive_nonzero_action_fraction_all_slots": global_summary[
            "inactive_nonzero_action_fraction_all_slots"
        ],
        "transformer_action_fraction_at_max_active_macro_mean": _mean_infrastructure_metric(
            action_summary["transformers"], "action_fraction_at_max_active"
        ),
        "charger_action_fraction_at_max_active_macro_mean": _mean_infrastructure_metric(
            action_summary["chargers"], "action_fraction_at_max_active"
        ),
        "transformer_action_nonzero_fraction_active_macro_mean": _mean_infrastructure_metric(
            action_summary["transformers"], "action_nonzero_fraction_active"
        ),
        "charger_action_nonzero_fraction_active_macro_mean": _mean_infrastructure_metric(
            action_summary["chargers"], "action_nonzero_fraction_active"
        ),
        "transformer_positive_charge_action_hhi_mean": global_summary[
            "transformer_positive_charge_action_hhi_mean"
        ],
        "transformer_positive_charge_action_gini_mean": global_summary[
            "transformer_positive_charge_action_gini_mean"
        ],
        "transformer_allocation_zero_pressure_step_fraction": global_summary[
            "transformer_allocation_zero_pressure_step_fraction"
        ],
        "transformer_allocation_valid_step_count": global_summary[
            "transformer_allocation_valid_step_count"
        ],
        "charger_positive_charge_action_hhi_mean": global_summary[
            "charger_positive_charge_action_hhi_mean"
        ],
        "charger_positive_charge_action_gini_mean": global_summary[
            "charger_positive_charge_action_gini_mean"
        ],
        "charger_allocation_zero_pressure_step_fraction": global_summary[
            "charger_allocation_zero_pressure_step_fraction"
        ],
        "charger_allocation_valid_step_count": global_summary[
            "charger_allocation_valid_step_count"
        ],
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
    }
    for stat_key in [
        "total_transformer_overload",
        "power_tracker_violation",
        "tracking_error",
        "energy_tracking_error",
        "total_ev_served",
        "total_energy_charged",
        "total_energy_discharged",
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
            "action_nonzero_fraction_active": charger_summary["action_nonzero_fraction_active"],
            "positive_action_fraction_active": charger_summary["positive_action_fraction_active"],
            "zero_action_fraction_active": charger_summary["zero_action_fraction_active"],
            "negative_action_fraction_active": charger_summary["negative_action_fraction_active"],
            "action_fraction_at_positive_max_active": charger_summary[
                "action_fraction_at_positive_max_active"
            ],
            "action_fraction_at_negative_min_active": charger_summary[
                "action_fraction_at_negative_min_active"
            ],
            "positive_action_sum_active": charger_summary["positive_action_sum_active"],
            "negative_action_magnitude_sum_active": charger_summary[
                "negative_action_magnitude_sum_active"
            ],
            "action_sum_all_slots": charger_summary["action_sum_all_slots"],
            "action_mean_all_slots": charger_summary["action_mean_all_slots"],
            "action_max_all_slots": charger_summary["action_max_all_slots"],
            "action_fraction_at_max_all_slots": charger_summary["action_fraction_at_max_all_slots"],
            "cs_power_sum_kwh": charger_summary["cs_power_sum_kwh"],
            "cs_power_mean_kw": charger_summary["cs_power_mean_kw"],
            "cs_power_max_kw": charger_summary["cs_power_max_kw"],
            "served_ev_count": charger_summary["served_ev_count"],
            "energy_charged_kwh": charger_summary["energy_charged_kwh"],
            "energy_discharged_kwh": charger_summary["energy_discharged_kwh"],
            "user_satisfaction_sum": charger_summary["user_satisfaction_sum"],
            "user_satisfaction_mean": charger_summary["user_satisfaction_mean"],
            "user_satisfaction_observation_count": charger_summary[
                "user_satisfaction_observation_count"
            ],
            "user_satisfaction_source": charger_summary["user_satisfaction_source"],
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
            "action_nonzero_fraction_active": transformer_summary["action_nonzero_fraction_active"],
            "positive_action_fraction_active": transformer_summary["positive_action_fraction_active"],
            "zero_action_fraction_active": transformer_summary["zero_action_fraction_active"],
            "negative_action_fraction_active": transformer_summary["negative_action_fraction_active"],
            "action_fraction_at_positive_max_active": transformer_summary[
                "action_fraction_at_positive_max_active"
            ],
            "action_fraction_at_negative_min_active": transformer_summary[
                "action_fraction_at_negative_min_active"
            ],
            "positive_action_sum_active": transformer_summary["positive_action_sum_active"],
            "negative_action_magnitude_sum_active": transformer_summary[
                "negative_action_magnitude_sum_active"
            ],
            "action_sum_all_slots": transformer_summary["action_sum_all_slots"],
            "action_mean_all_slots": transformer_summary["action_mean_all_slots"],
            "action_max_all_slots": transformer_summary["action_max_all_slots"],
            "action_fraction_at_max_all_slots": transformer_summary["action_fraction_at_max_all_slots"],
            "charger_positive_charge_action_hhi_mean": transformer_summary[
                "charger_positive_charge_action_hhi_mean"
            ],
            "charger_positive_charge_action_gini_mean": transformer_summary[
                "charger_positive_charge_action_gini_mean"
            ],
            "charger_allocation_zero_pressure_step_fraction": transformer_summary[
                "charger_allocation_zero_pressure_step_fraction"
            ],
            "charger_allocation_valid_step_count": transformer_summary[
                "charger_allocation_valid_step_count"
            ],
            "overload_magnitude_sum": transformer_summary["overload_magnitude_sum"],
            "overload_magnitude_max": transformer_summary["overload_magnitude_max"],
            "overload_frequency_steps": transformer_summary["overload_frequency_steps"],
            "overload_frequency_fraction": transformer_summary["overload_frequency_fraction"],
            "served_ev_count": transformer_summary["served_ev_count"],
            "energy_charged_kwh": transformer_summary["energy_charged_kwh"],
            "energy_discharged_kwh": transformer_summary["energy_discharged_kwh"],
            "user_satisfaction_sum": transformer_summary["user_satisfaction_sum"],
            "user_satisfaction_mean": transformer_summary["user_satisfaction_mean"],
            "user_satisfaction_mean_served_ev_weighted": transformer_summary[
                "user_satisfaction_mean_served_ev_weighted"
            ],
            "user_satisfaction_observation_count": transformer_summary[
                "user_satisfaction_observation_count"
            ],
            "user_satisfaction_source": transformer_summary["user_satisfaction_source"],
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
        "global_action_nonzero_fraction_active_mean": _mean_existing(
            episode_rows, "global_action_nonzero_fraction_active"
        ),
        "active_action_decision_count_mean": _mean_existing(
            episode_rows, "active_action_decision_count"
        ),
        "active_action_below_environment_low_count": _sum_existing(
            episode_rows, "active_action_below_environment_low_count"
        ),
        "active_action_below_environment_low_fraction": _fraction_from_count_sums(
            episode_rows,
            "active_action_below_environment_low_count",
            "active_action_decision_count",
        ),
        "active_action_above_environment_high_count": _sum_existing(
            episode_rows, "active_action_above_environment_high_count"
        ),
        "active_action_above_environment_high_fraction": _fraction_from_count_sums(
            episode_rows,
            "active_action_above_environment_high_count",
            "active_action_decision_count",
        ),
        "global_positive_action_fraction_active_mean": _mean_existing(
            episode_rows, "global_positive_action_fraction_active", default=UNAVAILABLE
        ),
        "global_zero_action_fraction_active_mean": _mean_existing(
            episode_rows, "global_zero_action_fraction_active", default=UNAVAILABLE
        ),
        "global_negative_action_fraction_active_mean": _mean_existing(
            episode_rows, "global_negative_action_fraction_active", default=UNAVAILABLE
        ),
        "global_action_fraction_at_positive_max_active_mean": _mean_existing(
            episode_rows, "global_action_fraction_at_positive_max_active", default=UNAVAILABLE
        ),
        "global_action_fraction_at_negative_min_active_mean": _mean_existing(
            episode_rows, "global_action_fraction_at_negative_min_active", default=UNAVAILABLE
        ),
        "global_positive_action_sum_active_mean": _mean_existing(
            episode_rows, "global_positive_action_sum_active"
        ),
        "global_negative_action_magnitude_sum_active_mean": _mean_existing(
            episode_rows, "global_negative_action_magnitude_sum_active"
        ),
        "observed_action_min_active_mean": _mean_existing(
            episode_rows, "observed_action_min_active", default=UNAVAILABLE
        ),
        "observed_action_max_active_mean": _mean_existing(
            episode_rows, "observed_action_max_active", default=UNAVAILABLE
        ),
        "environment_action_low": _first_existing(episode_rows, "environment_action_low"),
        "environment_action_high": _first_existing(episode_rows, "environment_action_high"),
        "action_tolerance": _first_existing(episode_rows, "action_tolerance"),
        "environment_action_domain_support": _first_existing(
            episode_rows, "environment_action_domain_support"
        ),
        "v2g_enabled": _first_existing(episode_rows, "v2g_enabled"),
        "v2g_enabled_source": _first_existing(episode_rows, "v2g_enabled_source"),
        "inactive_slot_decision_count_mean": _mean_existing(
            episode_rows, "inactive_slot_decision_count"
        ),
        "inactive_nonzero_action_count_mean": _mean_existing(
            episode_rows, "inactive_nonzero_action_count"
        ),
        "inactive_nonzero_action_fraction_all_slots_mean": _mean_existing(
            episode_rows, "inactive_nonzero_action_fraction_all_slots", default=UNAVAILABLE
        ),
        "transformer_action_fraction_at_max_active_macro_mean": _mean_existing(
            episode_rows, "transformer_action_fraction_at_max_active_macro_mean"
        ),
        "charger_action_fraction_at_max_active_macro_mean": _mean_existing(
            episode_rows, "charger_action_fraction_at_max_active_macro_mean"
        ),
        "transformer_action_nonzero_fraction_active_macro_mean": _mean_existing(
            episode_rows, "transformer_action_nonzero_fraction_active_macro_mean"
        ),
        "charger_action_nonzero_fraction_active_macro_mean": _mean_existing(
            episode_rows, "charger_action_nonzero_fraction_active_macro_mean"
        ),
        "transformer_positive_charge_action_hhi_mean": _mean_existing(
            episode_rows, "transformer_positive_charge_action_hhi_mean", default=UNAVAILABLE
        ),
        "transformer_positive_charge_action_gini_mean": _mean_existing(
            episode_rows, "transformer_positive_charge_action_gini_mean", default=UNAVAILABLE
        ),
        "transformer_allocation_zero_pressure_step_fraction_mean": _mean_existing(
            episode_rows, "transformer_allocation_zero_pressure_step_fraction"
        ),
        "transformer_allocation_valid_step_count_mean": _mean_existing(
            episode_rows, "transformer_allocation_valid_step_count"
        ),
        "charger_positive_charge_action_hhi_mean": _mean_existing(
            episode_rows, "charger_positive_charge_action_hhi_mean", default=UNAVAILABLE
        ),
        "charger_positive_charge_action_gini_mean": _mean_existing(
            episode_rows, "charger_positive_charge_action_gini_mean", default=UNAVAILABLE
        ),
        "charger_allocation_zero_pressure_step_fraction_mean": _mean_existing(
            episode_rows, "charger_allocation_zero_pressure_step_fraction"
        ),
        "charger_allocation_valid_step_count_mean": _mean_existing(
            episode_rows, "charger_allocation_valid_step_count"
        ),
        "total_transformer_overload_mean": _mean_existing(
            episode_rows, "total_transformer_overload", default=UNAVAILABLE
        ),
        "power_tracker_violation_mean": _mean_existing(
            episode_rows, "power_tracker_violation", default=UNAVAILABLE
        ),
        "tracking_error_mean": _mean_existing(
            episode_rows, "tracking_error", default=UNAVAILABLE
        ),
        "energy_tracking_error_mean": _mean_existing(
            episode_rows, "energy_tracking_error", default=UNAVAILABLE
        ),
        "total_ev_served_mean": _mean_existing(
            episode_rows, "total_ev_served", default=UNAVAILABLE
        ),
        "total_energy_charged_mean": _mean_existing(
            episode_rows, "total_energy_charged", default=UNAVAILABLE
        ),
        "total_energy_discharged_mean": _mean_existing(
            episode_rows, "total_energy_discharged", default=UNAVAILABLE
        ),
        "average_user_satisfaction_mean": _mean_existing(
            episode_rows, "average_user_satisfaction", default=UNAVAILABLE
        ),
        "energy_user_satisfaction_mean": _mean_existing(
            episode_rows, "energy_user_satisfaction", default=UNAVAILABLE
        ),
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
    }


def validate_diagnostic_reconciliation(
    episode_row,
    charger_rows,
    transformer_rows,
    tolerance=1e-6,
    relative_tolerance=1e-12,
):
    _validate_exact_reconciliation(
        episode_row=episode_row,
        infrastructure_rows=charger_rows,
        episode_key="total_ev_served",
        infrastructure_key="served_ev_count",
        label="charger served_ev_count",
        tolerance=tolerance,
    )
    _validate_exact_reconciliation(
        episode_row=episode_row,
        infrastructure_rows=transformer_rows,
        episode_key="total_ev_served",
        infrastructure_key="served_ev_count",
        label="transformer served_ev_count",
        tolerance=tolerance,
    )
    _validate_tolerant_reconciliation(
        episode_row=episode_row,
        infrastructure_rows=charger_rows,
        episode_key="total_energy_charged",
        infrastructure_key="energy_charged_kwh",
        label="charger energy_charged_kwh",
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )
    _validate_tolerant_reconciliation(
        episode_row=episode_row,
        infrastructure_rows=transformer_rows,
        episode_key="total_energy_charged",
        infrastructure_key="energy_charged_kwh",
        label="transformer energy_charged_kwh",
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )
    _validate_tolerant_reconciliation(
        episode_row=episode_row,
        infrastructure_rows=charger_rows,
        episode_key="total_energy_discharged",
        infrastructure_key="energy_discharged_kwh",
        label="charger energy_discharged_kwh",
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )
    _validate_tolerant_reconciliation(
        episode_row=episode_row,
        infrastructure_rows=transformer_rows,
        episode_key="total_energy_discharged",
        infrastructure_key="energy_discharged_kwh",
        label="transformer energy_discharged_kwh",
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )


def _validate_exact_reconciliation(
    episode_row,
    infrastructure_rows,
    episode_key,
    infrastructure_key,
    label,
    tolerance,
):
    expected = episode_row.get(episode_key, UNAVAILABLE)
    observed = _row_sum_if_all_available(infrastructure_rows, infrastructure_key)
    if not _is_available_number(expected) or not _is_available_number(observed):
        return
    if not _is_integral_number(expected) or not _is_integral_number(observed):
        raise ValueError(
            f"Diagnostic reconciliation mismatch for {label}: "
            f"{observed} != episode {episode_key} {expected}."
        )
    if int(float(expected)) != int(float(observed)):
        raise ValueError(
            f"Diagnostic reconciliation mismatch for {label}: "
            f"{observed} != episode {episode_key} {expected}."
        )


def _validate_tolerant_reconciliation(
    episode_row,
    infrastructure_rows,
    episode_key,
    infrastructure_key,
    label,
    tolerance,
    relative_tolerance,
):
    expected = episode_row.get(episode_key, UNAVAILABLE)
    observed = _row_sum_if_all_available(infrastructure_rows, infrastructure_key)
    if not _is_available_number(expected) or not _is_available_number(observed):
        return
    if not _within_reconciliation_tolerance(
        observed=float(observed),
        expected=float(expected),
        absolute_tolerance=float(tolerance),
        relative_tolerance=float(relative_tolerance),
    ):
        raise ValueError(
            f"Diagnostic reconciliation mismatch for {label}: "
            f"{observed} != episode {episode_key} {expected}."
        )


def _row_sum_if_all_available(rows, key):
    total = 0.0
    for row in rows:
        value = row.get(key, UNAVAILABLE)
        if not _is_available_number(value):
            return UNAVAILABLE
        total += float(value)
    return total


def _within_reconciliation_tolerance(
    observed,
    expected,
    absolute_tolerance,
    relative_tolerance,
):
    difference = abs(float(observed) - float(expected))
    return (
        difference <= float(absolute_tolerance)
        or difference <= float(relative_tolerance) * max(abs(float(expected)), 1.0)
    )


def _validated_active_slots(active_slots_for_step, action_dim, tolerance=1e-6):
    try:
        active_slots = np.asarray(active_slots_for_step, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("active slots must contain numeric EV2Gym action IDs.") from exc

    if active_slots.size == 0:
        return np.asarray([], dtype=int)
    if not np.all(np.isfinite(active_slots)):
        raise ValueError("active slots must be finite EV2Gym action IDs.")
    if not np.all(active_slots == np.rint(active_slots)):
        raise ValueError("active slots must be integral EV2Gym action IDs.")

    active_slot_ids = np.rint(active_slots).astype(int)
    if np.any(active_slot_ids < 0):
        raise ValueError("active slots contain negative EV2Gym action IDs.")
    if np.any(active_slot_ids >= int(action_dim)):
        raise ValueError("active slots outside the EV2Gym action range.")
    if np.unique(active_slot_ids).size != active_slot_ids.size:
        raise ValueError("active slots contain duplicate EV2Gym action IDs.")
    return active_slot_ids


def _ordered_charger_ids(slot_to_charger_id, charger_to_transformer_id, env):
    charger_ids = {int(charger_id) for charger_id in np.unique(slot_to_charger_id)}
    charger_ids.update(int(charger_id) for charger_id in charger_to_transformer_id)
    if env is not None:
        charger_ids.update(
            _validated_non_negative_integral_scalar(
                getattr(charging_station, "id", None),
                "charging station ID",
            )
            for charging_station in env.charging_stations
        )
    return sorted(charger_ids)


def _charger_lookup(env):
    if env is None:
        return {}
    charger_lookup = {}
    for charging_station in env.charging_stations:
        charger_id = _validated_non_negative_integral_scalar(
            getattr(charging_station, "id", None),
            "charging station ID",
        )
        if charger_id in charger_lookup:
            raise ValueError("Duplicate charging station IDs are unsupported.")
        charger_lookup[charger_id] = charging_station
    return charger_lookup


def _chargers_by_transformer(charger_ids, charger_to_transformer_id):
    chargers_by_transformer = defaultdict(list)
    for charger_id in charger_ids:
        transformer_id = charger_to_transformer_id.get(int(charger_id))
        if transformer_id is not None:
            chargers_by_transformer[int(transformer_id)].append(int(charger_id))
    return chargers_by_transformer


def _environment_action_bounds(env, max_action, tolerance):
    if env is not None and hasattr(env, "action_space"):
        return validate_environment_action_bounds(env.action_space, tolerance=tolerance)

    return {
        "environment_action_low": UNAVAILABLE,
        "environment_action_high": UNAVAILABLE,
        "environment_action_domain_support": UNAVAILABLE,
    }


def _environment_action_domain_support(low, high, tolerance):
    low = float(low)
    high = float(high)
    tolerance = float(tolerance)
    if low < -tolerance and high > tolerance:
        return "signed"
    if low >= -tolerance and high > tolerance:
        return "non_negative"
    if low < -tolerance and high <= tolerance:
        return "non_positive"
    return "zero_only"


def _environment_bound_violation_diagnostics(active_values, action_bounds, tolerance):
    if not (
        _is_available_number(action_bounds.get("environment_action_low", UNAVAILABLE))
        and _is_available_number(action_bounds.get("environment_action_high", UNAVAILABLE))
    ):
        return {
            "active_action_below_environment_low_count": UNAVAILABLE,
            "active_action_below_environment_low_fraction": UNAVAILABLE,
            "active_action_above_environment_high_count": UNAVAILABLE,
            "active_action_above_environment_high_fraction": UNAVAILABLE,
        }

    values = np.asarray(active_values, dtype=float).reshape(-1)
    low = float(action_bounds["environment_action_low"])
    high = float(action_bounds["environment_action_high"])
    below_count = int(np.count_nonzero(values < low - float(tolerance)))
    above_count = int(np.count_nonzero(values > high + float(tolerance)))
    if values.size == 0:
        below_fraction = UNAVAILABLE
        above_fraction = UNAVAILABLE
    else:
        below_fraction = float(below_count / values.size)
        above_fraction = float(above_count / values.size)
    return {
        "active_action_below_environment_low_count": below_count,
        "active_action_below_environment_low_fraction": below_fraction,
        "active_action_above_environment_high_count": above_count,
        "active_action_above_environment_high_fraction": above_fraction,
    }


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
    action_bounds,
    all_values,
    active_values,
):
    all_slot_diagnostics = action_diagnostics(all_values, max_action=max_action, tolerance=tolerance)
    active_diagnostics = action_diagnostics(active_values, max_action=max_action, tolerance=tolerance)
    signed_diagnostics = signed_action_diagnostics(
        active_values,
        environment_action_low=action_bounds.get("environment_action_low", UNAVAILABLE),
        environment_action_high=action_bounds.get("environment_action_high", max_action),
        tolerance=tolerance,
    )
    environment_bound_diagnostics = _environment_bound_violation_diagnostics(
        active_values,
        action_bounds,
        tolerance,
    )
    action_dim = int(slot_to_charger_id.size)
    active_slot_counts = [int(active_slots.size) for active_slots in active_slots_by_step]
    nonzero_action_counts = [
        int(np.count_nonzero(np.abs(mapped_action) > float(tolerance)))
        for mapped_action in mapped_actions
    ]
    inactive_contract = _inactive_nonzero_action_contract(
        mapped_actions=mapped_actions,
        active_slots_by_step=active_slots_by_step,
        action_dim=action_dim,
        tolerance=tolerance,
    )
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
        "environment_action_low": action_bounds.get("environment_action_low", UNAVAILABLE),
        "environment_action_high": action_bounds.get("environment_action_high", UNAVAILABLE),
        "environment_action_domain_support": action_bounds.get(
            "environment_action_domain_support", UNAVAILABLE
        ),
        "action_tolerance": float(tolerance),
        "active_action_decision_count": signed_diagnostics["active_action_decision_count"],
        "active_action_below_environment_low_count": environment_bound_diagnostics[
            "active_action_below_environment_low_count"
        ],
        "active_action_below_environment_low_fraction": environment_bound_diagnostics[
            "active_action_below_environment_low_fraction"
        ],
        "active_action_above_environment_high_count": environment_bound_diagnostics[
            "active_action_above_environment_high_count"
        ],
        "active_action_above_environment_high_fraction": environment_bound_diagnostics[
            "active_action_above_environment_high_fraction"
        ],
        "observed_action_min_active": signed_diagnostics["observed_action_min_active"],
        "observed_action_max_active": signed_diagnostics["observed_action_max_active"],
        "action_fraction_at_max_all_slots": all_slot_diagnostics["action_fraction_at_max"],
        "action_fraction_at_max_active": active_diagnostics["action_fraction_at_max"],
        "action_nonzero_fraction_active": active_diagnostics["action_nonzero_fraction"],
        "positive_action_fraction_active": signed_diagnostics["positive_action_fraction_active"],
        "zero_action_fraction_active": signed_diagnostics["zero_action_fraction_active"],
        "negative_action_fraction_active": signed_diagnostics["negative_action_fraction_active"],
        "action_fraction_at_positive_max_active": signed_diagnostics[
            "action_fraction_at_positive_max_active"
        ],
        "action_fraction_at_negative_min_active": signed_diagnostics[
            "action_fraction_at_negative_min_active"
        ],
        "positive_action_sum_active": signed_diagnostics["positive_action_sum_active"],
        "negative_action_magnitude_sum_active": signed_diagnostics[
            "negative_action_magnitude_sum_active"
        ],
        "action_mean_all_slots": all_slot_diagnostics["action_mean"],
        "action_mean_active": active_diagnostics["action_mean"],
        "action_sum_active": active_diagnostics["action_sum"],
        "active_slot_count_mean": _mean_values(active_slot_counts),
        "nonzero_action_count_mean_all_slots": _mean_values(nonzero_action_counts),
        "inactive_slot_fraction_mean": _mean_values(inactive_slot_fractions),
        "inactive_slot_decision_count": inactive_contract["inactive_slot_decision_count"],
        "inactive_nonzero_action_count": inactive_contract["inactive_nonzero_action_count"],
        "inactive_nonzero_action_fraction_all_slots": inactive_contract[
            "inactive_nonzero_action_fraction_all_slots"
        ],
        "transformer_positive_charge_action_hhi_mean": _mean_concentration(
            transformer_concentrations, "hhi"
        ),
        "transformer_positive_charge_action_gini_mean": _mean_concentration(
            transformer_concentrations, "gini"
        ),
        "transformer_allocation_zero_pressure_step_fraction": _zero_pressure_step_fraction(
            transformer_concentrations
        ),
        "transformer_allocation_valid_step_count": _valid_concentration_count(
            transformer_concentrations
        ),
        "charger_positive_charge_action_hhi_mean": _mean_concentration(
            charger_concentrations, "hhi"
        ),
        "charger_positive_charge_action_gini_mean": _mean_concentration(
            charger_concentrations, "gini"
        ),
        "charger_allocation_zero_pressure_step_fraction": _zero_pressure_step_fraction(
            charger_concentrations
        ),
        "charger_allocation_valid_step_count": _valid_concentration_count(charger_concentrations),
    }


def _charger_action_summary(
    charger_id,
    mapped_actions,
    active_slots_by_step,
    slot_to_charger_id,
    charger_to_transformer_id,
    max_action,
    tolerance,
    action_bounds,
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
    active_values = _concatenate_or_empty(active_values_by_step)
    signed_diagnostics = signed_action_diagnostics(
        active_values,
        environment_action_low=action_bounds.get("environment_action_low", UNAVAILABLE),
        environment_action_high=action_bounds.get("environment_action_high", max_action),
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
        "positive_action_fraction_active": signed_diagnostics["positive_action_fraction_active"],
        "zero_action_fraction_active": signed_diagnostics["zero_action_fraction_active"],
        "negative_action_fraction_active": signed_diagnostics["negative_action_fraction_active"],
        "action_fraction_at_positive_max_active": signed_diagnostics[
            "action_fraction_at_positive_max_active"
        ],
        "action_fraction_at_negative_min_active": signed_diagnostics[
            "action_fraction_at_negative_min_active"
        ],
        "positive_action_sum_active": signed_diagnostics["positive_action_sum_active"],
        "negative_action_magnitude_sum_active": signed_diagnostics[
            "negative_action_magnitude_sum_active"
        ],
        "action_sum_all_slots": all_diagnostics["action_sum"],
        "action_mean_all_slots": all_diagnostics["action_mean"],
        "action_max_all_slots": all_diagnostics["action_max"],
        "action_fraction_at_max_all_slots": all_diagnostics["action_fraction_at_max"],
        "cs_power_sum_kwh": power_summary["cs_power_sum_kwh"],
        "cs_power_mean_kw": power_summary["cs_power_mean_kw"],
        "cs_power_max_kw": power_summary["cs_power_max_kw"],
        "served_ev_count": service_summary["served_ev_count"],
        "energy_charged_kwh": service_summary["energy_charged_kwh"],
        "energy_discharged_kwh": service_summary["energy_discharged_kwh"],
        "user_satisfaction_sum": service_summary["user_satisfaction_sum"],
        "user_satisfaction_mean": service_summary["user_satisfaction_mean"],
        "user_satisfaction_observation_count": service_summary[
            "user_satisfaction_observation_count"
        ],
        "user_satisfaction_source": service_summary["user_satisfaction_source"],
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
    action_bounds,
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
    active_values = _concatenate_or_empty(active_values_by_step)
    signed_diagnostics = signed_action_diagnostics(
        active_values,
        environment_action_low=action_bounds.get("environment_action_low", UNAVAILABLE),
        environment_action_high=action_bounds.get("environment_action_high", max_action),
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
        "positive_action_fraction_active": signed_diagnostics["positive_action_fraction_active"],
        "zero_action_fraction_active": signed_diagnostics["zero_action_fraction_active"],
        "negative_action_fraction_active": signed_diagnostics["negative_action_fraction_active"],
        "action_fraction_at_positive_max_active": signed_diagnostics[
            "action_fraction_at_positive_max_active"
        ],
        "action_fraction_at_negative_min_active": signed_diagnostics[
            "action_fraction_at_negative_min_active"
        ],
        "positive_action_sum_active": signed_diagnostics["positive_action_sum_active"],
        "negative_action_magnitude_sum_active": signed_diagnostics[
            "negative_action_magnitude_sum_active"
        ],
        "action_sum_all_slots": all_diagnostics["action_sum"],
        "action_mean_all_slots": all_diagnostics["action_mean"],
        "action_max_all_slots": all_diagnostics["action_max"],
        "action_fraction_at_max_all_slots": all_diagnostics["action_fraction_at_max"],
        "charger_positive_charge_action_hhi_mean": _mean_concentration(
            charger_concentrations, "hhi"
        ),
        "charger_positive_charge_action_gini_mean": _mean_concentration(
            charger_concentrations, "gini"
        ),
        "charger_allocation_zero_pressure_step_fraction": _zero_pressure_step_fraction(
            charger_concentrations
        ),
        "charger_allocation_valid_step_count": _valid_concentration_count(charger_concentrations),
        "overload_magnitude_sum": overload_summary["overload_magnitude_sum"],
        "overload_magnitude_max": overload_summary["overload_magnitude_max"],
        "overload_frequency_steps": overload_summary["overload_frequency_steps"],
        "overload_frequency_fraction": overload_summary["overload_frequency_fraction"],
        "served_ev_count": service_summary["served_ev_count"],
        "energy_charged_kwh": service_summary["energy_charged_kwh"],
        "energy_discharged_kwh": service_summary["energy_discharged_kwh"],
        "user_satisfaction_sum": service_summary["user_satisfaction_sum"],
        "user_satisfaction_mean": service_summary["user_satisfaction_mean"],
        "user_satisfaction_mean_served_ev_weighted": service_summary[
            "user_satisfaction_mean_served_ev_weighted"
        ],
        "user_satisfaction_observation_count": service_summary[
            "user_satisfaction_observation_count"
        ],
        "user_satisfaction_source": service_summary["user_satisfaction_source"],
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
            "energy_discharged_kwh": UNAVAILABLE,
            "user_satisfaction_sum": UNAVAILABLE,
            "user_satisfaction_mean": UNAVAILABLE,
            "user_satisfaction_observation_count": 0,
            "user_satisfaction_source": UNAVAILABLE,
        }

    served_ev_count = _count_attr(charger, "total_evs_served")
    energy_charged_kwh = _numeric_attr(charger, "total_energy_charged")
    energy_discharged_kwh = _numeric_attr(charger, "total_energy_discharged")

    satisfaction_sum = UNAVAILABLE
    satisfaction_mean = UNAVAILABLE
    satisfaction_observation_count = 0
    satisfaction_source = UNAVAILABLE

    if hasattr(charger, "all_user_satisfaction"):
        satisfaction_values = _finite_satisfaction_values(
            getattr(charger, "all_user_satisfaction")
        )
        if satisfaction_values.size:
            satisfaction_sum = float(np.sum(satisfaction_values))
            satisfaction_observation_count = int(satisfaction_values.size)
            satisfaction_mean = float(satisfaction_sum / satisfaction_observation_count)
            satisfaction_source = "charger_all_user_satisfaction"

    if satisfaction_source == UNAVAILABLE:
        total_satisfaction = _numeric_attr(charger, "total_user_satisfaction")
        if _is_available_number(total_satisfaction) and _is_available_number(served_ev_count):
            satisfaction_sum = float(total_satisfaction)
            satisfaction_observation_count = served_ev_count
            satisfaction_mean = (
                float(satisfaction_sum / satisfaction_observation_count)
                if satisfaction_observation_count > 0
                else UNAVAILABLE
            )
            satisfaction_source = "charger_total_user_satisfaction"

    return {
        "served_ev_count": served_ev_count if _is_available_number(served_ev_count) else UNAVAILABLE,
        "energy_charged_kwh": (
            float(energy_charged_kwh)
            if _is_available_number(energy_charged_kwh)
            else UNAVAILABLE
        ),
        "energy_discharged_kwh": (
            float(energy_discharged_kwh)
            if _is_available_number(energy_discharged_kwh)
            else UNAVAILABLE
        ),
        "user_satisfaction_sum": satisfaction_sum,
        "user_satisfaction_mean": satisfaction_mean,
        "user_satisfaction_observation_count": satisfaction_observation_count,
        "user_satisfaction_source": satisfaction_source,
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
    energy_discharged_kwh = 0.0
    user_satisfaction_sum = 0.0
    satisfaction_observation_count = 0
    served_seen = False
    charged_energy_seen = False
    discharged_energy_seen = False
    satisfaction_seen = False

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
            charged_energy_seen = True
            energy_charged_kwh += float(energy_charged_value)

        energy_discharged_value = charger_row.get("energy_discharged_kwh", UNAVAILABLE)
        if _is_available_number(energy_discharged_value):
            discharged_energy_seen = True
            energy_discharged_kwh += float(energy_discharged_value)

        satisfaction_sum_value = charger_row.get("user_satisfaction_sum", UNAVAILABLE)
        satisfaction_count = charger_row.get("user_satisfaction_observation_count", 0)
        if (
            _is_available_number(satisfaction_sum_value)
            and _is_available_number(satisfaction_count)
        ):
            satisfaction_seen = True
            user_satisfaction_sum += float(satisfaction_sum_value)
            satisfaction_observation_count += int(satisfaction_count)

    user_satisfaction_mean_served_ev_weighted = (
        float(user_satisfaction_sum / satisfaction_observation_count)
        if satisfaction_observation_count
        else UNAVAILABLE
    )
    user_satisfaction_source = (
        "charger_satisfaction_sum_count"
        if satisfaction_seen
        else UNAVAILABLE
    )
    return {
        "served_ev_count": served_ev_count if served_seen else UNAVAILABLE,
        "energy_charged_kwh": energy_charged_kwh if charged_energy_seen else UNAVAILABLE,
        "energy_discharged_kwh": (
            energy_discharged_kwh
            if discharged_energy_seen
            else UNAVAILABLE
        ),
        "user_satisfaction_sum": user_satisfaction_sum if satisfaction_seen else UNAVAILABLE,
        "user_satisfaction_mean": user_satisfaction_mean_served_ev_weighted,
        "user_satisfaction_mean_served_ev_weighted": user_satisfaction_mean_served_ev_weighted,
        "user_satisfaction_observation_count": satisfaction_observation_count,
        "user_satisfaction_source": user_satisfaction_source,
    }


def _inactive_nonzero_action_contract(mapped_actions, active_slots_by_step, action_dim, tolerance):
    inactive_value_count = 0
    inactive_nonzero_count = 0
    for mapped_action, active_slots in zip(mapped_actions, active_slots_by_step):
        inactive_slot_mask = np.ones(action_dim, dtype=bool)
        inactive_slot_mask[active_slots] = False
        inactive_values = mapped_action[inactive_slot_mask]
        inactive_value_count += int(inactive_values.size)
        inactive_nonzero_count += int(np.count_nonzero(np.abs(inactive_values) > float(tolerance)))

    if inactive_value_count == 0:
        fraction = UNAVAILABLE
    else:
        fraction = float(inactive_nonzero_count / inactive_value_count)
    return {
        "inactive_slot_decision_count": inactive_value_count,
        "inactive_nonzero_action_count": inactive_nonzero_count,
        "inactive_nonzero_action_fraction_all_slots": fraction,
    }


def _valid_concentration_rows(concentration_rows):
    return [
        concentration
        for concentration in concentration_rows
        if not bool(concentration.get("zero_pressure", False))
    ]


def _mean_concentration(concentration_rows, key):
    valid_concentrations = _valid_concentration_rows(concentration_rows)
    if not valid_concentrations:
        return UNAVAILABLE
    return _mean_values([concentration[key] for concentration in valid_concentrations])


def _zero_pressure_step_fraction(concentration_rows):
    if not concentration_rows:
        return 0.0
    zero_pressure_count = sum(
        1
        for concentration in concentration_rows
        if bool(concentration.get("zero_pressure", False))
    )
    return float(zero_pressure_count / len(concentration_rows))


def _valid_concentration_count(concentration_rows):
    return len(_valid_concentration_rows(concentration_rows))


def _mean_infrastructure_metric(infrastructure_rows, metric_key):
    values = [
        float(row[metric_key])
        for row in infrastructure_rows.values()
        if row.get("n_active_ev_decisions", 0) > 0
    ]
    return float(np.mean(values)) if values else 0.0


def _mean_values(values):
    numeric_values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(numeric_values)) if numeric_values else 0.0


def _float_stat(stats, stat_key):
    if not stats or stat_key not in stats:
        return UNAVAILABLE
    value = stats[stat_key]
    if not np.isscalar(value):
        return UNAVAILABLE
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if not np.isfinite(numeric_value):
        return UNAVAILABLE
    return numeric_value


def _numeric_attr(obj, attr_name):
    if not hasattr(obj, attr_name):
        return UNAVAILABLE
    value = getattr(obj, attr_name)
    if not _is_available_number(value):
        return UNAVAILABLE
    return float(value)


def _count_attr(obj, attr_name):
    if not hasattr(obj, attr_name):
        return UNAVAILABLE
    return _validated_non_negative_integral_scalar(getattr(obj, attr_name), attr_name)


def _finite_satisfaction_values(satisfaction_source):
    if satisfaction_source is None:
        return np.asarray([], dtype=float)
    try:
        values = np.asarray(satisfaction_source, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return values[np.isfinite(values)]


def _is_available_number(value):
    if value in (UNAVAILABLE, None):
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _is_integral_number(value):
    if not _is_available_number(value):
        return False
    numeric_value = float(value)
    return numeric_value == round(numeric_value)


def _validated_non_negative_integral_scalar(value, label):
    values = _validated_non_negative_integral_values([value], label)
    return int(values[0])


def _validated_non_negative_integral_values(values, label):
    try:
        numeric_values = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError(f"{label} must be finite.")
    if np.any(numeric_values < 0):
        raise ValueError(f"{label} must be non-negative.")
    if not np.all(numeric_values == np.rint(numeric_values)):
        raise ValueError(f"{label} must be integral.")
    return np.rint(numeric_values).astype(int)


def _mean_existing(rows, key, default=0.0):
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
    return float(np.mean(values)) if values else default


def _sum_existing(rows, key, default=UNAVAILABLE):
    values = []
    for row in rows:
        if key not in row:
            continue
        value = row[key]
        if value in (UNAVAILABLE, None):
            continue
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            values.append(numeric_value)
    return float(np.sum(values)) if values else default


def _fraction_from_count_sums(rows, numerator_key, denominator_key):
    numerator = _sum_existing(rows, numerator_key, default=UNAVAILABLE)
    denominator = _sum_existing(rows, denominator_key, default=UNAVAILABLE)
    if not _is_available_number(numerator) or not _is_available_number(denominator):
        return UNAVAILABLE
    denominator = float(denominator)
    if denominator <= 0.0:
        return UNAVAILABLE
    return float(float(numerator) / denominator)


def _first_existing(rows, key, default=UNAVAILABLE):
    for row in rows:
        if key not in row:
            continue
        value = row[key]
        if value in (UNAVAILABLE, None):
            continue
        return value
    return default
