"""Compare EV infrastructure control architectures at training-seed level."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from analysis.ev_charging_infrastructure_control.metric_definitions import (
    METRIC_DEFINITIONS,
)


SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
TRAINING_SEEDS = tuple(range(5))


@dataclass(frozen=True)
class SeedMetricObservation:
    scale: str
    algorithm: str
    training_seed: int
    metric_name: str
    value: float


@dataclass(frozen=True)
class PairedSeedMetric:
    training_seed: int
    actiongnn_value: float
    hierarchical_value: float


def fail(message: str) -> None:
    raise RuntimeError(message)


def parse_finite_float(value: object, context: str) -> float:
    text = "" if value is None else str(value).strip()
    if text == "":
        fail(f"{context}: missing numeric value")
    try:
        number = float(text)
    except ValueError as exc:
        raise RuntimeError(f"{context}: invalid numeric value {text!r}") from exc
    if not math.isfinite(number):
        fail(f"{context}: non-finite numeric value {text!r}")
    return number


def parse_int(value: object, context: str) -> int:
    number = parse_finite_float(value, context)
    if not float(number).is_integer():
        fail(f"{context}: expected integer, found {value!r}")
    return int(number)


def mean(values: list[float]) -> float:
    if not values:
        fail("mean requires at least one value")
    return sum(values) / len(values)


def observation_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (
        str(row["scale"]),
        str(row["algorithm"]),
        parse_int(row["training_seed"], "training_seed"),
    )


def aggregate_episode_metric(
    rows: list[dict[str, str]],
    metric_name: str,
) -> dict[tuple[str, str, int], float]:
    grouped_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        key = observation_key(row)
        grouped_values[key].append(
            parse_finite_float(row.get(metric_name), f"{key}: {metric_name}")
        )
    return {key: mean(values) for key, values in grouped_values.items()}


def aggregate_transformer_overload_metrics(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, int], dict[str, float]]:
    frequency_by_episode: dict[tuple[str, str, int, int], list[float]] = defaultdict(list)
    magnitude_sum_by_episode: dict[tuple[str, str, int, int], float] = defaultdict(float)
    maximum_by_seed: dict[tuple[str, str, int], float] = {}

    for row in rows:
        scale, algorithm, training_seed = observation_key(row)
        episode_index = parse_int(row["episode_index"], "episode_index")
        seed_key = (scale, algorithm, training_seed)
        episode_key = (scale, algorithm, training_seed, episode_index)
        frequency_by_episode[episode_key].append(
            parse_finite_float(
                row["overload_frequency_fraction"],
                f"{episode_key}: overload_frequency_fraction",
            )
        )
        magnitude_sum_by_episode[episode_key] += parse_finite_float(
            row["overload_magnitude_sum"],
            f"{episode_key}: overload_magnitude_sum",
        )
        overload_magnitude_max = parse_finite_float(
            row["overload_magnitude_max"],
            f"{episode_key}: overload_magnitude_max",
        )
        maximum_by_seed[seed_key] = max(
            maximum_by_seed.get(seed_key, overload_magnitude_max),
            overload_magnitude_max,
        )

    episode_frequency_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    episode_magnitude_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for episode_key, frequency_values in frequency_by_episode.items():
        seed_key = episode_key[:3]
        episode_frequency_values[seed_key].append(mean(frequency_values))
    for episode_key, magnitude_sum in magnitude_sum_by_episode.items():
        seed_key = episode_key[:3]
        episode_magnitude_values[seed_key].append(magnitude_sum)

    return {
        seed_key: {
            "mean_transformer_overload_frequency_fraction": mean(
                episode_frequency_values[seed_key]
            ),
            "mean_total_transformer_overload_magnitude": mean(
                episode_magnitude_values[seed_key]
            ),
            "maximum_transformer_overload_magnitude": maximum_by_seed[seed_key],
        }
        for seed_key in sorted(maximum_by_seed)
    }


def build_seed_level_metrics(
    episode_rows: list[dict[str, str]],
    transformer_rows: list[dict[str, str]],
) -> list[SeedMetricObservation]:
    observations: list[SeedMetricObservation] = []
    for definition in METRIC_DEFINITIONS:
        if definition.source_level == "episode":
            aggregated = aggregate_episode_metric(episode_rows, definition.source_column)
            observations.extend(
                SeedMetricObservation(
                    scale=scale,
                    algorithm=algorithm,
                    training_seed=training_seed,
                    metric_name=definition.name,
                    value=value,
                )
                for (scale, algorithm, training_seed), value in aggregated.items()
            )
    transformer_metrics = aggregate_transformer_overload_metrics(transformer_rows)
    for (scale, algorithm, training_seed), metric_values in transformer_metrics.items():
        observations.extend(
            SeedMetricObservation(
                scale=scale,
                algorithm=algorithm,
                training_seed=training_seed,
                metric_name=metric_name,
                value=value,
            )
            for metric_name, value in metric_values.items()
        )
    return observations


def validate_exact_algorithm_pairs(
    observations: list[SeedMetricObservation],
    scale: str,
    metric_name: str,
) -> tuple[PairedSeedMetric, ...]:
    values_by_algorithm_seed: dict[tuple[str, int], float] = {}
    for observation in observations:
        if observation.scale != scale or observation.metric_name != metric_name:
            continue
        key = (observation.algorithm, observation.training_seed)
        if key in values_by_algorithm_seed:
            fail(f"duplicate seed observation for {scale}/{metric_name}: {key}")
        values_by_algorithm_seed[key] = observation.value

    observed_algorithms = {
        algorithm for algorithm, _training_seed in values_by_algorithm_seed
    }
    if observed_algorithms != set(ALGORITHMS):
        fail(f"missing algorithm cell for {scale}/{metric_name}")

    actiongnn_seeds = {
        training_seed
        for algorithm, training_seed in values_by_algorithm_seed
        if algorithm == "actiongnn"
    }
    hierarchical_seeds = {
        training_seed
        for algorithm, training_seed in values_by_algorithm_seed
        if algorithm == "hierarchical"
    }
    if actiongnn_seeds != hierarchical_seeds:
        fail(f"unequal seed sets for {scale}/{metric_name}")
    if actiongnn_seeds != set(TRAINING_SEEDS):
        fail(f"{scale}/{metric_name} requires exactly five paired seeds")

    return tuple(
        PairedSeedMetric(
            training_seed=training_seed,
            actiongnn_value=values_by_algorithm_seed[("actiongnn", training_seed)],
            hierarchical_value=values_by_algorithm_seed[
                ("hierarchical", training_seed)
            ],
        )
        for training_seed in TRAINING_SEEDS
    )
