import importlib
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

metric_module = importlib.import_module(
    "analysis.ev_charging_infrastructure_control.metric_definitions"
)


EXPECTED_METRICS = {
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "global_action_fraction_at_max_active",
    "global_action_nonzero_fraction_active",
    "transformer_action_fraction_at_max_active_macro_mean",
    "charger_action_fraction_at_max_active_macro_mean",
    "transformer_positive_charge_action_hhi_mean",
    "charger_positive_charge_action_hhi_mean",
    "transformer_allocation_zero_pressure_step_fraction",
    "charger_allocation_zero_pressure_step_fraction",
    "total_transformer_overload",
    "mean_transformer_overload_frequency_fraction",
    "mean_total_transformer_overload_magnitude",
    "maximum_transformer_overload_magnitude",
    "total_ev_served",
    "total_energy_charged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "transformer_positive_charge_action_gini_mean",
    "charger_positive_charge_action_gini_mean",
}


def test_registry_contains_only_approved_metrics():
    definitions = metric_module.METRIC_DEFINITIONS

    assert {definition.name for definition in definitions} == EXPECTED_METRICS
    assert len(definitions) == len(EXPECTED_METRICS)
    assert len({definition.name for definition in definitions}) == len(definitions)


def test_registry_preserves_scientific_boundaries():
    reward = metric_module.metric_definition("episode_reward")
    saturation = metric_module.metric_definition(
        "global_action_fraction_at_max_active"
    )
    gini = metric_module.metric_definition(
        "transformer_positive_charge_action_gini_mean"
    )

    assert reward.tier == "primary"
    assert reward.preferred_direction == "higher"
    assert reward.aggregation_rule == "mean 30 episodes"
    assert reward.seed_summary_column is None
    assert saturation.tier == "mechanism"
    assert saturation.preferred_direction == "context_dependent"
    assert "not automatically" in saturation.interpretation_warning.lower()
    assert gini.tier == "robustness"
    assert "robustness" in gini.interpretation_warning.lower()
    assert "not independent primary" in gini.interpretation_warning.lower()


def test_registry_tier_lookup_and_required_columns_are_semantic():
    primary_names = metric_module.metric_names_for_tier("primary")
    required_episode_columns = metric_module.required_source_columns("episode")
    required_transformer_columns = metric_module.required_source_columns("transformer")

    assert primary_names == (
        "episode_reward",
        "tracking_error",
        "energy_tracking_error",
        "power_tracker_violation",
    )
    assert "episode_reward" in required_episode_columns
    assert "overload_frequency_fraction" in required_transformer_columns
    assert "battery_degradation" not in required_episode_columns


def test_registry_rejects_unknown_metric_and_tier():
    with pytest.raises(KeyError, match="unknown metric"):
        metric_module.metric_definition("battery_degradation")
    with pytest.raises(KeyError, match="unknown tier"):
        metric_module.metric_names_for_tier("economic")
