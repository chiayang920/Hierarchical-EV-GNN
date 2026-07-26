from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def fake_charger(
    charger_id,
    n_ports,
    transformer_id,
    total_energy_charged=0.0,
    total_evs_served=0,
    all_user_satisfaction=None,
):
    return SimpleNamespace(
        id=charger_id,
        n_ports=n_ports,
        connected_transformer=transformer_id,
        total_energy_charged=total_energy_charged,
        total_evs_served=total_evs_served,
        all_user_satisfaction=list(all_user_satisfaction or []),
    )


def fake_env(chargers, timescale=5, tr_overload=None, cs_power=None):
    env = SimpleNamespace(
        charging_stations=chargers,
        timescale=timescale,
    )
    if tr_overload is not None:
        env.tr_overload = np.asarray(tr_overload, dtype=float)
    if cs_power is not None:
        env.cs_power = np.asarray(cs_power, dtype=float)
    return env


def fake_state(action_mapper, ev_features):
    return SimpleNamespace(
        action_mapper=action_mapper,
        ev_features=np.asarray(ev_features, dtype=float),
    )


def test_slot_to_charger_id_uses_ev2gym_port_order_for_non_uniform_ports():
    from utils.infrastructure_diagnostics import build_slot_to_charger_id

    env = fake_env([
        fake_charger(10, 2, 7),
        fake_charger(11, 1, 7),
        fake_charger(12, 3, 8),
    ])

    assert build_slot_to_charger_id(env).tolist() == [10, 10, 11, 12, 12, 12]


def test_charger_to_transformer_id_uses_realized_env_topology():
    from utils.infrastructure_diagnostics import build_charger_to_transformer_id

    env = fake_env([
        fake_charger(0, 1, 3),
        fake_charger(1, 1, 3),
        fake_charger(2, 1, 4),
    ])

    assert build_charger_to_transformer_id(env) == {0: 3, 1: 3, 2: 4}


def test_extract_active_ev_infrastructure_uses_state_action_mapper_and_features():
    from utils.infrastructure_diagnostics import extract_active_ev_infrastructure

    state = fake_state(
        action_mapper=[0, 3],
        ev_features=[
            [0.5, 0.0, 2.0, 0.0, 10.0, 7.0],
            [0.5, 0.0, 4.0, 0.0, 12.0, 8.0],
        ],
    )

    metadata = extract_active_ev_infrastructure(state)

    assert metadata["active_slots"].tolist() == [0, 3]
    assert metadata["charger_ids"].tolist() == [10, 12]
    assert metadata["transformer_ids"].tolist() == [7, 8]


def test_active_only_denominator_excludes_inactive_slots():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0), fake_charger(1, 2, 1)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 1.0, 0.0, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 3], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 1, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["action_fraction_at_max_active"] == pytest.approx(0.5)
    assert summary["global"]["active_slot_count_mean"] == pytest.approx(2.0)
    assert summary["global"]["nonzero_action_count_mean_all_slots"] == pytest.approx(3.0)
    assert summary["global"]["inactive_slot_fraction_mean"] == pytest.approx(0.5)


def test_all_slot_denominator_includes_every_action_slot():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0), fake_charger(1, 2, 1)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 1.0, 0.0, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 3], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 1, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["action_fraction_at_max_all_slots"] == pytest.approx(0.5)
    assert summary["global"]["action_mean_all_slots"] == pytest.approx(0.625)
    assert summary["global"]["nonzero_action_count_mean_all_slots"] == pytest.approx(3.0)


def test_active_slot_count_can_differ_from_all_slot_nonzero_action_count():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0), fake_charger(1, 2, 1)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 0.0, 0.0, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 3], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 1, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["active_slot_count_mean"] == pytest.approx(3.0)
    assert summary["global"]["nonzero_action_count_mean_all_slots"] == pytest.approx(2.0)


def test_action_fraction_at_max_uses_max_action_minus_tolerance_threshold():
    from utils.infrastructure_diagnostics import action_diagnostics

    diagnostics = action_diagnostics(
        np.array([1.0, 0.999999, 0.999], dtype=float),
        max_action=1.0,
        tolerance=1e-6,
    )

    assert diagnostics["action_fraction_at_max"] == pytest.approx(2.0 / 3.0)


def test_zero_active_slots_return_safe_values_without_crashing():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.2, 1.0], dtype=float)],
        active_slots_by_step=[np.array([], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["action_fraction_at_max_active"] == 0.0
    assert summary["global"]["action_mean_active"] == 0.0
    assert summary["global"]["active_slot_count_mean"] == 0.0
    assert summary["global"]["nonzero_action_count_mean_all_slots"] == pytest.approx(2.0)


def test_missing_env_power_and_overload_arrays_are_blank_not_zero():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 1, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["chargers"][0]["cs_power_sum_kwh"] == ""
    assert summary["chargers"][0]["cs_power_mean_kw"] == ""
    assert summary["chargers"][0]["cs_power_max_kw"] == ""
    assert summary["transformers"][0]["overload_magnitude_sum"] == ""
    assert summary["transformers"][0]["overload_magnitude_max"] == ""
    assert summary["transformers"][0]["overload_frequency_steps"] == ""
    assert summary["transformers"][0]["overload_frequency_fraction"] == ""


def test_active_infrastructure_validation_rejects_slot_charger_mismatch():
    from utils.infrastructure_diagnostics import validate_active_infrastructure_mapping

    state = fake_state(
        action_mapper=[1],
        ev_features=[[0.5, 0.0, 2.0, 0.0, 99.0, 0.0]],
    )

    with pytest.raises(ValueError, match="Active slot charger mapping mismatch"):
        validate_active_infrastructure_mapping(
            state=state,
            slot_to_charger_id=np.array([10, 10], dtype=int),
            charger_to_transformer_id={10: 0},
        )


def test_active_infrastructure_validation_rejects_charger_transformer_mismatch():
    from utils.infrastructure_diagnostics import validate_active_infrastructure_mapping

    state = fake_state(
        action_mapper=[0],
        ev_features=[[0.5, 0.0, 2.0, 0.0, 10.0, 7.0]],
    )

    with pytest.raises(ValueError, match="Active slot transformer mapping mismatch"):
        validate_active_infrastructure_mapping(
            state=state,
            slot_to_charger_id=np.array([10], dtype=int),
            charger_to_transformer_id={10: 8},
        )


def test_allocation_concentration_handles_uniform_concentrated_and_zero_pressure():
    from utils.infrastructure_diagnostics import allocation_concentration

    uniform = allocation_concentration([2.0, 2.0])
    concentrated = allocation_concentration([4.0, 0.0])
    zero_pressure = allocation_concentration([0.0, 0.0])

    assert uniform["hhi"] == pytest.approx(0.5)
    assert uniform["gini"] == pytest.approx(0.0)
    assert uniform["zero_pressure"] is False
    assert concentrated["hhi"] == pytest.approx(1.0)
    assert concentrated["gini"] == pytest.approx(0.5)
    assert concentrated["zero_pressure"] is False
    assert zero_pressure["hhi"] == 0.0
    assert zero_pressure["gini"] == 0.0
    assert zero_pressure["zero_pressure"] is True


def test_per_charger_aggregation_matches_hand_computed_toy_example():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 2, 0),
        fake_charger(1, 1, 0),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[
            np.array([1.0, 0.5, 0.0], dtype=float),
            np.array([0.0, 1.0, 1.0], dtype=float),
        ],
        active_slots_by_step=[
            np.array([0, 1], dtype=int),
            np.array([2], dtype=int),
        ],
        slot_to_charger_id=np.array([0, 0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    charger_zero = summary["chargers"][0]
    charger_one = summary["chargers"][1]
    assert charger_zero["action_sum_active"] == pytest.approx(1.5)
    assert charger_zero["action_mean_active"] == pytest.approx(0.75)
    assert charger_zero["action_max_active"] == pytest.approx(1.0)
    assert charger_zero["action_fraction_at_max_active"] == pytest.approx(0.5)
    assert charger_zero["action_sum_all_slots"] == pytest.approx(2.5)
    assert charger_zero["action_mean_all_slots"] == pytest.approx(0.625)
    assert charger_zero["action_fraction_at_max_all_slots"] == pytest.approx(0.5)
    assert charger_one["action_sum_active"] == pytest.approx(1.0)
    assert charger_one["action_fraction_at_max_active"] == pytest.approx(1.0)


def test_per_transformer_aggregation_matches_hand_computed_toy_example():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0),
        fake_charger(1, 1, 0),
        fake_charger(2, 1, 1),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 0.5, 1.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2], dtype=int)],
        slot_to_charger_id=np.array([0, 1, 2], dtype=int),
        charger_to_transformer_id={0: 0, 1: 0, 2: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    transformer_zero = summary["transformers"][0]
    transformer_one = summary["transformers"][1]
    assert transformer_zero["n_chargers_total"] == 2
    assert transformer_zero["n_active_chargers_seen"] == 2
    assert transformer_zero["action_sum_active"] == pytest.approx(1.5)
    assert transformer_zero["action_mean_active"] == pytest.approx(0.75)
    assert transformer_zero["action_fraction_at_max_active"] == pytest.approx(0.5)
    assert transformer_one["n_chargers_total"] == 1
    assert transformer_one["action_sum_active"] == pytest.approx(1.0)
    assert transformer_one["action_fraction_at_max_active"] == pytest.approx(1.0)


def test_seed_summary_averages_episode_diagnostics():
    from utils.infrastructure_diagnostics import build_seed_summary_row

    seed_row = build_seed_summary_row(
        metadata={
            "matrix_job_id": "58513929",
            "scale": "25cp",
            "algorithm": "hierarchical",
            "training_seed": 3,
        },
        episode_rows=[
            {
                "global_action_fraction_at_max_active": 0.2,
                "global_action_fraction_at_max_all_slots": 0.1,
                "transformer_action_hhi_mean": 0.5,
                "transformer_action_gini_mean": 0.0,
                "charger_action_hhi_mean": 0.6,
                "charger_action_gini_mean": 0.1,
                "total_transformer_overload": 4.0,
                "power_tracker_violation": 8.0,
                "tracking_error": 12.0,
                "energy_tracking_error": 16.0,
                "total_ev_served": 20.0,
                "total_energy_charged": 24.0,
                "average_user_satisfaction": 0.9,
                "energy_user_satisfaction": 95.0,
            },
            {
                "global_action_fraction_at_max_active": 0.4,
                "global_action_fraction_at_max_all_slots": 0.3,
                "transformer_action_hhi_mean": 0.7,
                "transformer_action_gini_mean": 0.2,
                "charger_action_hhi_mean": 0.8,
                "charger_action_gini_mean": 0.3,
                "total_transformer_overload": 6.0,
                "power_tracker_violation": 10.0,
                "tracking_error": 14.0,
                "energy_tracking_error": 18.0,
                "total_ev_served": 22.0,
                "total_energy_charged": 26.0,
                "average_user_satisfaction": 1.0,
                "energy_user_satisfaction": 97.0,
            },
        ],
    )

    assert seed_row["n_eval_episodes"] == 2
    assert seed_row["global_action_fraction_at_max_active_mean"] == pytest.approx(0.3)
    assert seed_row["total_transformer_overload_mean"] == pytest.approx(5.0)
    assert seed_row["total_energy_charged_mean"] == pytest.approx(25.0)
    assert seed_row["energy_user_satisfaction_mean"] == pytest.approx(96.0)


def test_diagnostic_evaluator_help_works():
    command_result = subprocess.run(
        [sys.executable, "evaluate_td3_gnn_infrastructure_diagnostics.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert command_result.returncode == 0
    assert "--output_dir" in command_result.stdout
    assert "--matrix_job_id" in command_result.stdout
