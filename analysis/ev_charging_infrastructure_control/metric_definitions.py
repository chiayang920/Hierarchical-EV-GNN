"""Approved EV charging infrastructure control analysis metric registry."""

from __future__ import annotations

from dataclasses import dataclass


PRIMARY = "primary"
MECHANISM = "mechanism"
PHYSICAL_SAFETY = "physical_safety"
SERVICE_GUARDRAIL = "service_guardrail"
ROBUSTNESS = "robustness"

HIGHER = "higher"
LOWER = "lower"
CONTEXT_DEPENDENT = "context_dependent"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    source_level: str
    source_column: str
    seed_summary_column: str | None
    tier: str
    preferred_direction: str
    aggregation_rule: str
    interpretation_warning: str


CONTROL_WARNING = "Primary control outcome for paired seed-level inference."
SAFETY_WARNING = "Physical safety guardrail; lower values are preferred."
SERVICE_WARNING = (
    "Service guardrail; changes must be interpreted alongside control outcomes."
)
MECHANISM_WARNING = (
    "Mechanism evidence; lower or higher values are not automatically favourable."
)
GINI_WARNING = (
    "Robustness check for HHI concentration patterns, not independent primary "
    "discoveries; lower or higher values are not automatically favourable."
)


METRIC_DEFINITIONS = (
    MetricDefinition(
        "episode_reward",
        "episode",
        "episode_reward",
        None,
        PRIMARY,
        HIGHER,
        "mean 30 episodes",
        CONTROL_WARNING,
    ),
    MetricDefinition(
        "tracking_error",
        "episode",
        "tracking_error",
        "tracking_error_mean",
        PRIMARY,
        LOWER,
        "mean 30 episodes",
        CONTROL_WARNING,
    ),
    MetricDefinition(
        "energy_tracking_error",
        "episode",
        "energy_tracking_error",
        "energy_tracking_error_mean",
        PRIMARY,
        LOWER,
        "mean 30 episodes",
        CONTROL_WARNING,
    ),
    MetricDefinition(
        "power_tracker_violation",
        "episode",
        "power_tracker_violation",
        "power_tracker_violation_mean",
        PRIMARY,
        LOWER,
        "mean 30 episodes",
        CONTROL_WARNING,
    ),
    MetricDefinition(
        "global_action_fraction_at_max_active",
        "episode",
        "global_action_fraction_at_max_active",
        "global_action_fraction_at_max_active_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "global_action_nonzero_fraction_active",
        "episode",
        "global_action_nonzero_fraction_active",
        "global_action_nonzero_fraction_active_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "transformer_action_fraction_at_max_active_macro_mean",
        "episode",
        "transformer_action_fraction_at_max_active_macro_mean",
        "transformer_action_fraction_at_max_active_macro_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "charger_action_fraction_at_max_active_macro_mean",
        "episode",
        "charger_action_fraction_at_max_active_macro_mean",
        "charger_action_fraction_at_max_active_macro_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "transformer_positive_charge_action_hhi_mean",
        "episode",
        "transformer_positive_charge_action_hhi_mean",
        "transformer_positive_charge_action_hhi_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "charger_positive_charge_action_hhi_mean",
        "episode",
        "charger_positive_charge_action_hhi_mean",
        "charger_positive_charge_action_hhi_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "transformer_allocation_zero_pressure_step_fraction",
        "episode",
        "transformer_allocation_zero_pressure_step_fraction",
        "transformer_allocation_zero_pressure_step_fraction_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "charger_allocation_zero_pressure_step_fraction",
        "episode",
        "charger_allocation_zero_pressure_step_fraction",
        "charger_allocation_zero_pressure_step_fraction_mean",
        MECHANISM,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        MECHANISM_WARNING,
    ),
    MetricDefinition(
        "total_transformer_overload",
        "episode",
        "total_transformer_overload",
        "total_transformer_overload_mean",
        PHYSICAL_SAFETY,
        LOWER,
        "mean 30 episodes",
        SAFETY_WARNING,
    ),
    MetricDefinition(
        "mean_transformer_overload_frequency_fraction",
        "transformer",
        "overload_frequency_fraction",
        None,
        PHYSICAL_SAFETY,
        LOWER,
        "mean transformers per episode, then mean episodes",
        SAFETY_WARNING,
    ),
    MetricDefinition(
        "mean_total_transformer_overload_magnitude",
        "transformer",
        "overload_magnitude_sum",
        None,
        PHYSICAL_SAFETY,
        LOWER,
        "sum transformers per episode, then mean episodes",
        SAFETY_WARNING,
    ),
    MetricDefinition(
        "maximum_transformer_overload_magnitude",
        "transformer",
        "overload_magnitude_max",
        None,
        PHYSICAL_SAFETY,
        LOWER,
        "maximum over all transformer rows in seed",
        SAFETY_WARNING,
    ),
    MetricDefinition(
        "total_ev_served",
        "episode",
        "total_ev_served",
        "total_ev_served_mean",
        SERVICE_GUARDRAIL,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        SERVICE_WARNING,
    ),
    MetricDefinition(
        "total_energy_charged",
        "episode",
        "total_energy_charged",
        "total_energy_charged_mean",
        SERVICE_GUARDRAIL,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        SERVICE_WARNING,
    ),
    MetricDefinition(
        "average_user_satisfaction",
        "episode",
        "average_user_satisfaction",
        "average_user_satisfaction_mean",
        SERVICE_GUARDRAIL,
        HIGHER,
        "mean 30 episodes",
        SERVICE_WARNING,
    ),
    MetricDefinition(
        "energy_user_satisfaction",
        "episode",
        "energy_user_satisfaction",
        "energy_user_satisfaction_mean",
        SERVICE_GUARDRAIL,
        HIGHER,
        "mean 30 episodes",
        SERVICE_WARNING,
    ),
    MetricDefinition(
        "transformer_positive_charge_action_gini_mean",
        "episode",
        "transformer_positive_charge_action_gini_mean",
        "transformer_positive_charge_action_gini_mean",
        ROBUSTNESS,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        GINI_WARNING,
    ),
    MetricDefinition(
        "charger_positive_charge_action_gini_mean",
        "episode",
        "charger_positive_charge_action_gini_mean",
        "charger_positive_charge_action_gini_mean",
        ROBUSTNESS,
        CONTEXT_DEPENDENT,
        "mean 30 episodes",
        GINI_WARNING,
    ),
)


def validate_metric_definitions(
    definitions: tuple[MetricDefinition, ...],
) -> tuple[MetricDefinition, ...]:
    observed_names: set[str] = set()
    duplicate_names: list[str] = []
    for definition in definitions:
        if definition.name in observed_names:
            duplicate_names.append(definition.name)
        observed_names.add(definition.name)
    if duplicate_names:
        raise RuntimeError(
            "duplicate metric definition name(s): " + ", ".join(duplicate_names)
        )
    return definitions


METRIC_DEFINITIONS = validate_metric_definitions(METRIC_DEFINITIONS)
_METRICS_BY_NAME = {definition.name: definition for definition in METRIC_DEFINITIONS}
_KNOWN_TIERS = {
    PRIMARY,
    MECHANISM,
    PHYSICAL_SAFETY,
    SERVICE_GUARDRAIL,
    ROBUSTNESS,
}


def metric_definition(metric_name: str) -> MetricDefinition:
    try:
        return _METRICS_BY_NAME[metric_name]
    except KeyError as exc:
        raise KeyError(f"unknown metric: {metric_name}") from exc


def metric_names_for_tier(tier: str) -> tuple[str, ...]:
    if tier not in _KNOWN_TIERS:
        raise KeyError(f"unknown tier: {tier}")
    return tuple(
        definition.name for definition in METRIC_DEFINITIONS if definition.tier == tier
    )


def required_source_columns(source_level: str) -> tuple[str, ...]:
    columns = {
        definition.source_column
        for definition in METRIC_DEFINITIONS
        if definition.source_level == source_level
    }
    return tuple(sorted(columns))
