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
    total_energy_discharged=0.0,
    total_evs_served=0,
    total_user_satisfaction=None,
    all_user_satisfaction=None,
):
    if total_user_satisfaction is None:
        total_user_satisfaction = (
            0.0
            if all_user_satisfaction is None
            else float(np.sum(np.asarray(all_user_satisfaction, dtype=float)))
        )
    return SimpleNamespace(
        id=charger_id,
        n_ports=n_ports,
        connected_transformer=transformer_id,
        total_energy_charged=total_energy_charged,
        total_energy_discharged=total_energy_discharged,
        total_evs_served=total_evs_served,
        total_user_satisfaction=total_user_satisfaction,
        all_user_satisfaction=(
            None
            if all_user_satisfaction is None
            else list(all_user_satisfaction)
        ),
    )


def fake_env(
    chargers,
    timescale=5,
    tr_overload=None,
    cs_power=None,
    action_low=-1.0,
    action_high=1.0,
    v2g_enabled=None,
):
    action_dim = sum(int(charger.n_ports) for charger in chargers)
    env = SimpleNamespace(
        charging_stations=chargers,
        timescale=timescale,
        action_space=SimpleNamespace(
            low=np.full(action_dim, action_low, dtype=float),
            high=np.full(action_dim, action_high, dtype=float),
        ),
    )
    if tr_overload is not None:
        env.tr_overload = np.asarray(tr_overload, dtype=float)
    if cs_power is not None:
        env.cs_power = np.asarray(cs_power, dtype=float)
    if v2g_enabled is not None:
        env.v2g_enabled = v2g_enabled
    return env


def fake_state(action_mapper, ev_features):
    return SimpleNamespace(
        action_mapper=action_mapper,
        ev_features=np.asarray(ev_features, dtype=float),
    )


class FailingStepEnv:
    def __init__(self, state):
        self.state = state
        self.step_called = False
        self.charging_stations = [fake_charger(0, 1, 0)]

    def reset(self, seed=None):
        return self.state, {}

    def step(self, mapped_action):
        self.step_called = True
        raise AssertionError("env.step should not be called for invalid diagnostic actions")


class ConstantActionPolicy:
    def __init__(self, mapped_action):
        self.mapped_action = np.asarray(mapped_action, dtype=np.float32)

    def select_action(self, state, expl_noise=0.0, return_mapped_action=False):
        assert return_mapped_action is True
        return self.mapped_action


def test_slot_to_charger_id_uses_ev2gym_port_order_for_non_uniform_ports():
    from utils.infrastructure_diagnostics import build_slot_to_charger_id

    env = fake_env([
        fake_charger(10, 2, 7),
        fake_charger(11, 1, 7),
        fake_charger(12, 3, 8),
    ])

    assert build_slot_to_charger_id(env).tolist() == [10, 10, 11, 12, 12, 12]


def test_slot_to_charger_id_rejects_duplicate_charger_ids():
    from utils.infrastructure_diagnostics import build_slot_to_charger_id

    env = fake_env([
        fake_charger(0, 1, 0),
        fake_charger(0, 1, 1),
    ])

    with pytest.raises(ValueError, match="Duplicate charging station IDs"):
        build_slot_to_charger_id(env)


@pytest.mark.parametrize(
    ("charger_id", "expected_message"),
    [
        (0.5, "charging station ID must be integral"),
        (-1, "charging station ID must be non-negative"),
        (np.nan, "charging station ID must be finite"),
        (np.inf, "charging station ID must be finite"),
    ],
)
def test_slot_to_charger_id_rejects_invalid_charger_ids(charger_id, expected_message):
    from utils.infrastructure_diagnostics import build_slot_to_charger_id

    env = fake_env([fake_charger(charger_id, 1, 0)])

    with pytest.raises(ValueError, match=expected_message):
        build_slot_to_charger_id(env)


def test_charger_to_transformer_id_uses_realized_env_topology():
    from utils.infrastructure_diagnostics import build_charger_to_transformer_id

    env = fake_env([
        fake_charger(0, 1, 3),
        fake_charger(1, 1, 3),
        fake_charger(2, 1, 4),
    ])

    assert build_charger_to_transformer_id(env) == {0: 3, 1: 3, 2: 4}


@pytest.mark.parametrize(
    ("transformer_id", "expected_message"),
    [
        (0.5, "connected transformer ID must be integral"),
        (-1, "connected transformer ID must be non-negative"),
        (np.nan, "connected transformer ID must be finite"),
        (np.inf, "connected transformer ID must be finite"),
    ],
)
def test_charger_to_transformer_id_rejects_invalid_transformer_ids(
    transformer_id,
    expected_message,
):
    from utils.infrastructure_diagnostics import build_charger_to_transformer_id

    env = fake_env([fake_charger(0, 1, transformer_id)])

    with pytest.raises(ValueError, match=expected_message):
        build_charger_to_transformer_id(env)


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


@pytest.mark.parametrize(
    ("charger_id", "expected_message"),
    [
        (10.4, "EV feature charger IDs must be integral"),
        (np.nan, "EV feature charger IDs must be finite"),
        (np.inf, "EV feature charger IDs must be finite"),
        (-1.0, "EV feature charger IDs must be non-negative"),
    ],
)
def test_extract_active_ev_infrastructure_rejects_invalid_charger_ids(
    charger_id,
    expected_message,
):
    from utils.infrastructure_diagnostics import extract_active_ev_infrastructure

    state = fake_state(
        action_mapper=[0],
        ev_features=[[0.5, 0.0, 1.0, 0.0, charger_id, 0.0]],
    )

    with pytest.raises(ValueError, match=expected_message):
        extract_active_ev_infrastructure(state)


@pytest.mark.parametrize(
    ("transformer_id", "expected_message"),
    [
        (7.4, "EV feature transformer IDs must be integral"),
        (np.nan, "EV feature transformer IDs must be finite"),
        (np.inf, "EV feature transformer IDs must be finite"),
        (-1.0, "EV feature transformer IDs must be non-negative"),
    ],
)
def test_extract_active_ev_infrastructure_rejects_invalid_transformer_ids(
    transformer_id,
    expected_message,
):
    from utils.infrastructure_diagnostics import extract_active_ev_infrastructure

    state = fake_state(
        action_mapper=[0],
        ev_features=[[0.5, 0.0, 1.0, 0.0, 0.0, transformer_id]],
    )

    with pytest.raises(ValueError, match=expected_message):
        extract_active_ev_infrastructure(state)


def test_extract_active_ev_infrastructure_accepts_float_encoded_integer_ids():
    from utils.infrastructure_diagnostics import extract_active_ev_infrastructure

    state = fake_state(
        action_mapper=[0],
        ev_features=[[0.5, 0.0, 1.0, 0.0, 10.0, 7.0]],
    )

    metadata = extract_active_ev_infrastructure(state)

    assert metadata["charger_ids"].tolist() == [10]
    assert metadata["transformer_ids"].tolist() == [7]


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


def test_active_nonzero_fraction_distinguishes_max_moderate_and_zero_actions():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0),
        fake_charger(1, 1, 1),
        fake_charger(2, 1, 2),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 0.5, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2], dtype=int)],
        slot_to_charger_id=np.array([0, 1, 2], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1, 2: 2},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["action_fraction_at_max_active"] == pytest.approx(1.0 / 3.0)
    assert summary["global"]["action_nonzero_fraction_active"] == pytest.approx(2.0 / 3.0)
    assert summary["chargers"][0]["action_nonzero_fraction_active"] == pytest.approx(1.0)
    assert summary["chargers"][1]["action_nonzero_fraction_active"] == pytest.approx(1.0)
    assert summary["chargers"][2]["action_nonzero_fraction_active"] == pytest.approx(0.0)


def test_inactive_nonzero_action_fraction_detects_nonzero_inactive_slot():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 3, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5, 0.2, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["inactive_nonzero_action_fraction_all_slots"] == pytest.approx(0.5)


def test_inactive_nonzero_action_fraction_is_zero_for_valid_mapped_actions():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 3, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5, 0.0, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["inactive_nonzero_action_fraction_all_slots"] == pytest.approx(0.0)


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


def test_homogeneous_action_bounds_are_accepted():
    from utils.infrastructure_diagnostics import validate_environment_action_bounds

    action_space = SimpleNamespace(
        low=np.array([-1.0, -1.0, -1.0], dtype=float),
        high=np.array([1.0, 1.0, 1.0], dtype=float),
    )

    bounds = validate_environment_action_bounds(action_space, tolerance=1e-6)

    assert bounds["environment_action_low"] == pytest.approx(-1.0)
    assert bounds["environment_action_high"] == pytest.approx(1.0)
    assert bounds["environment_action_domain_support"] == "signed"


def test_heterogeneous_action_bounds_fail_clearly():
    from utils.infrastructure_diagnostics import validate_environment_action_bounds

    action_space = SimpleNamespace(
        low=np.array([-1.0, 0.0], dtype=float),
        high=np.array([1.0, 1.0], dtype=float),
    )

    with pytest.raises(ValueError, match="heterogeneous action-space low bounds"):
        validate_environment_action_bounds(action_space, tolerance=1e-6)


def test_heterogeneous_action_high_bounds_fail_clearly():
    from utils.infrastructure_diagnostics import validate_environment_action_bounds

    action_space = SimpleNamespace(
        low=np.array([0.0, 0.0], dtype=float),
        high=np.array([1.0, 0.5], dtype=float),
    )

    with pytest.raises(ValueError, match="heterogeneous action-space high bounds"):
        validate_environment_action_bounds(action_space, tolerance=1e-6)


def test_missing_environment_action_space_leaves_bound_metadata_blank():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=None,
    )

    global_summary = summary["global"]
    assert global_summary["environment_action_low"] == ""
    assert global_summary["environment_action_high"] == ""
    assert global_summary["environment_action_domain_support"] == ""
    assert global_summary["action_fraction_at_positive_max_active"] == ""
    assert global_summary["action_fraction_at_negative_min_active"] == ""


def test_positive_zero_negative_active_fractions_sum_to_one():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 2, 0),
        fake_charger(1, 2, 1),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 0.0, -0.5, 0.2], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2, 3], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 1, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    global_summary = summary["global"]
    assert global_summary["active_action_decision_count"] == 4
    assert global_summary["positive_action_fraction_active"] == pytest.approx(0.5)
    assert global_summary["zero_action_fraction_active"] == pytest.approx(0.25)
    assert global_summary["negative_action_fraction_active"] == pytest.approx(0.25)
    assert (
        global_summary["positive_action_fraction_active"]
        + global_summary["zero_action_fraction_active"]
        + global_summary["negative_action_fraction_active"]
    ) == pytest.approx(1.0)


def test_zero_active_observations_produce_blank_signed_fractions():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.2, -0.3], dtype=float)],
        active_slots_by_step=[np.array([], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    global_summary = summary["global"]
    assert global_summary["active_action_decision_count"] == 0
    assert global_summary["positive_action_fraction_active"] == ""
    assert global_summary["zero_action_fraction_active"] == ""
    assert global_summary["negative_action_fraction_active"] == ""
    assert global_summary["action_fraction_at_positive_max_active"] == ""
    assert global_summary["action_fraction_at_negative_min_active"] == ""
    assert global_summary["observed_action_min_active"] == ""
    assert global_summary["observed_action_max_active"] == ""
    assert global_summary["positive_action_sum_active"] == 0.0
    assert global_summary["negative_action_magnitude_sum_active"] == 0.0


def test_positive_max_threshold_uses_environment_high_minus_tolerance():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 3, 0)], action_low=-1.0, action_high=1.0)
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 0.999999, 0.9], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["action_fraction_at_positive_max_active"] == pytest.approx(2.0 / 3.0)


def test_negative_min_threshold_uses_environment_low_plus_tolerance():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 3, 0)], action_low=-1.0, action_high=1.0)
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([-1.0, -0.999999, -0.9], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["action_fraction_at_negative_min_active"] == pytest.approx(2.0 / 3.0)


def test_negative_min_metric_is_blank_for_non_negative_environment_domain():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0)], action_low=0.0, action_high=1.0)
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.0, 1.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["environment_action_domain_support"] == "non_negative"
    assert summary["global"]["action_fraction_at_negative_min_active"] == ""


def test_observed_action_min_and_max_are_distinct_from_action_space_bounds():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0)], action_low=-1.0, action_high=1.0)
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([-0.2, 0.6], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    global_summary = summary["global"]
    assert global_summary["environment_action_low"] == pytest.approx(-1.0)
    assert global_summary["environment_action_high"] == pytest.approx(1.0)
    assert global_summary["observed_action_min_active"] == pytest.approx(-0.2)
    assert global_summary["observed_action_max_active"] == pytest.approx(0.6)


def test_positive_and_negative_action_sums_do_not_cancel():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.7, -0.4], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["action_sum_active"] == pytest.approx(0.3)
    assert summary["global"]["positive_action_sum_active"] == pytest.approx(0.7)
    assert summary["global"]["negative_action_magnitude_sum_active"] == pytest.approx(0.4)


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


def test_zero_pressure_steps_do_not_reduce_positive_pressure_concentration_means():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 1, 0), fake_charger(1, 1, 1)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[
            np.array([1.0, 0.0], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        ],
        active_slots_by_step=[
            np.array([0, 1], dtype=int),
            np.array([0, 1], dtype=int),
        ],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["transformer_positive_charge_action_hhi_mean"] == pytest.approx(1.0)
    assert summary["global"]["transformer_positive_charge_action_gini_mean"] == pytest.approx(0.5)
    assert summary["global"]["transformer_allocation_valid_step_count"] == 1
    assert summary["global"]["transformer_allocation_zero_pressure_step_fraction"] == pytest.approx(0.5)
    assert summary["global"]["charger_positive_charge_action_hhi_mean"] == pytest.approx(1.0)
    assert summary["global"]["charger_positive_charge_action_gini_mean"] == pytest.approx(0.5)
    assert summary["global"]["charger_allocation_valid_step_count"] == 1
    assert summary["global"]["charger_allocation_zero_pressure_step_fraction"] == pytest.approx(0.5)


def test_all_zero_pressure_episodes_have_unavailable_concentration_means_and_zero_valid_steps():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 1, 0), fake_charger(1, 1, 1)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.0, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["transformer_positive_charge_action_hhi_mean"] == ""
    assert summary["global"]["transformer_positive_charge_action_gini_mean"] == ""
    assert summary["global"]["transformer_allocation_valid_step_count"] == 0
    assert summary["global"]["transformer_allocation_zero_pressure_step_fraction"] == pytest.approx(1.0)
    assert summary["global"]["charger_positive_charge_action_hhi_mean"] == ""
    assert summary["global"]["charger_positive_charge_action_gini_mean"] == ""
    assert summary["global"]["charger_allocation_valid_step_count"] == 0
    assert summary["global"]["charger_allocation_zero_pressure_step_fraction"] == pytest.approx(1.0)


def test_zero_pressure_fraction_counts_only_zero_pressure_steps():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 1, 0), fake_charger(1, 1, 1)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[
            np.array([0.5, 0.5], dtype=float),
            np.array([0.0, 0.0], dtype=float),
            np.array([0.1, 0.0], dtype=float),
        ],
        active_slots_by_step=[
            np.array([0, 1], dtype=int),
            np.array([0, 1], dtype=int),
            np.array([0, 1], dtype=int),
        ],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["transformer_allocation_zero_pressure_step_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["global"]["transformer_allocation_valid_step_count"] == 2
    assert summary["global"]["charger_allocation_zero_pressure_step_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["global"]["charger_allocation_valid_step_count"] == 2


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


def test_per_charger_signed_metrics_use_active_decision_denominator():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 2, 0),
        fake_charger(1, 1, 1),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, -0.5, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    charger_zero = summary["chargers"][0]
    charger_one = summary["chargers"][1]
    assert charger_zero["positive_action_fraction_active"] == pytest.approx(0.5)
    assert charger_zero["zero_action_fraction_active"] == pytest.approx(0.0)
    assert charger_zero["negative_action_fraction_active"] == pytest.approx(0.5)
    assert charger_zero["action_fraction_at_positive_max_active"] == pytest.approx(0.5)
    assert charger_zero["action_fraction_at_negative_min_active"] == pytest.approx(0.0)
    assert charger_zero["positive_action_sum_active"] == pytest.approx(1.0)
    assert charger_zero["negative_action_magnitude_sum_active"] == pytest.approx(0.5)
    assert charger_one["positive_action_fraction_active"] == pytest.approx(0.0)
    assert charger_one["zero_action_fraction_active"] == pytest.approx(1.0)
    assert charger_one["negative_action_fraction_active"] == pytest.approx(0.0)


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


def test_per_transformer_signed_metrics_use_active_decision_denominator():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0),
        fake_charger(1, 1, 0),
        fake_charger(2, 1, 1),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, -1.0, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2], dtype=int)],
        slot_to_charger_id=np.array([0, 1, 2], dtype=int),
        charger_to_transformer_id={0: 0, 1: 0, 2: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    transformer_zero = summary["transformers"][0]
    transformer_one = summary["transformers"][1]
    assert transformer_zero["positive_action_fraction_active"] == pytest.approx(0.5)
    assert transformer_zero["zero_action_fraction_active"] == pytest.approx(0.0)
    assert transformer_zero["negative_action_fraction_active"] == pytest.approx(0.5)
    assert transformer_zero["action_fraction_at_positive_max_active"] == pytest.approx(0.5)
    assert transformer_zero["action_fraction_at_negative_min_active"] == pytest.approx(0.5)
    assert transformer_zero["positive_action_sum_active"] == pytest.approx(1.0)
    assert transformer_zero["negative_action_magnitude_sum_active"] == pytest.approx(1.0)
    assert transformer_one["positive_action_fraction_active"] == pytest.approx(0.0)
    assert transformer_one["zero_action_fraction_active"] == pytest.approx(1.0)
    assert transformer_one["negative_action_fraction_active"] == pytest.approx(0.0)


def test_per_transformer_charger_concentration_reports_zero_pressure_fraction_and_valid_count():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0),
        fake_charger(1, 1, 0),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[
            np.array([1.0, 0.0], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        ],
        active_slots_by_step=[
            np.array([0, 1], dtype=int),
            np.array([0, 1], dtype=int),
        ],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    transformer_zero = summary["transformers"][0]
    assert transformer_zero["charger_positive_charge_action_hhi_mean"] == pytest.approx(1.0)
    assert transformer_zero["charger_positive_charge_action_gini_mean"] == pytest.approx(0.5)
    assert transformer_zero["charger_allocation_zero_pressure_step_fraction"] == pytest.approx(0.5)
    assert transformer_zero["charger_allocation_valid_step_count"] == 1


def test_positive_charge_concentration_ignores_negative_commands_explicitly():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0),
        fake_charger(1, 1, 1),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([-1.0, 1.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["transformer_positive_charge_action_hhi_mean"] == pytest.approx(1.0)
    assert summary["global"]["transformer_positive_charge_action_gini_mean"] == pytest.approx(0.5)
    assert summary["global"]["charger_positive_charge_action_hhi_mean"] == pytest.approx(1.0)
    assert summary["global"]["charger_positive_charge_action_gini_mean"] == pytest.approx(0.5)


def test_episode_row_distinguishes_macro_infrastructure_mean_from_global_pooled_active_fraction():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions, build_episode_row

    env = fake_env([
        fake_charger(0, 3, 0),
        fake_charger(1, 1, 1),
    ])
    action_summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([1.0, 1.0, 1.0, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2, 3], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 1},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )
    episode_row = build_episode_row(
        metadata={
            "matrix_job_id": "58513929",
            "scale": "25cp",
            "algorithm": "hierarchical",
            "training_seed": 0,
            "config": "config_files/PublicPST_25cp.yaml",
            "checkpoint_prefix": "models/hierarchical_seed0",
            "run_name": "diagnostic",
        },
        episode_index=0,
        episode_seed=710000,
        episode_record={"episode_steps": 1, "done": True, "episode_reward": 0.0},
        action_summary=action_summary,
        stats={},
        max_action=1.0,
        tolerance=1e-6,
    )

    assert episode_row["global_action_fraction_at_max_active"] == pytest.approx(0.75)
    assert episode_row["transformer_action_fraction_at_max_active_macro_mean"] == pytest.approx(0.5)
    assert episode_row["charger_action_fraction_at_max_active_macro_mean"] == pytest.approx(0.5)


def test_episode_row_records_total_energy_discharged_without_inferring_from_commands():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions, build_episode_row

    env = fake_env([fake_charger(0, 2, 0)])
    action_summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([-1.0, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )
    episode_row = build_episode_row(
        metadata={"matrix_job_id": "", "scale": "25cp", "algorithm": "actiongnn", "training_seed": 0},
        episode_index=0,
        episode_seed=710000,
        episode_record={"episode_steps": 1, "done": True, "episode_reward": 0.0},
        action_summary=action_summary,
        stats={"total_energy_discharged": 0.0},
        max_action=1.0,
        tolerance=1e-6,
    )

    assert episode_row["global_negative_action_magnitude_sum_active"] == pytest.approx(1.0)
    assert episode_row["total_energy_discharged"] == pytest.approx(0.0)


def test_missing_scalar_stats_are_blank_not_false_zero():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions, build_episode_row

    env = fake_env([fake_charger(0, 1, 0)])
    action_summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )
    episode_row = build_episode_row(
        metadata={"matrix_job_id": "", "scale": "25cp", "algorithm": "actiongnn", "training_seed": 0},
        episode_index=0,
        episode_seed=710000,
        episode_record={"episode_steps": 1, "done": True, "episode_reward": 0.0},
        action_summary=action_summary,
        stats={},
        max_action=1.0,
        tolerance=1e-6,
    )

    assert episode_row["tracking_error"] == ""
    assert episode_row["total_energy_discharged"] == ""


def test_scalar_zero_stats_remain_zero():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions, build_episode_row

    env = fake_env([fake_charger(0, 1, 0)])
    action_summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )
    episode_row = build_episode_row(
        metadata={"matrix_job_id": "", "scale": "25cp", "algorithm": "actiongnn", "training_seed": 0},
        episode_index=0,
        episode_seed=710000,
        episode_record={"episode_steps": 1, "done": True, "episode_reward": 0.0},
        action_summary=action_summary,
        stats={"tracking_error": 0.0, "total_energy_discharged": 0.0},
        max_action=1.0,
        tolerance=1e-6,
    )

    assert episode_row["tracking_error"] == pytest.approx(0.0)
    assert episode_row["total_energy_discharged"] == pytest.approx(0.0)


def test_nan_and_inf_stats_are_blank():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions, build_episode_row

    env = fake_env([fake_charger(0, 1, 0)])
    action_summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )
    episode_row = build_episode_row(
        metadata={"matrix_job_id": "", "scale": "25cp", "algorithm": "actiongnn", "training_seed": 0},
        episode_index=0,
        episode_seed=710000,
        episode_record={"episode_steps": 1, "done": True, "episode_reward": 0.0},
        action_summary=action_summary,
        stats={"tracking_error": np.nan, "energy_tracking_error": np.inf},
        max_action=1.0,
        tolerance=1e-6,
    )

    assert episode_row["tracking_error"] == ""
    assert episode_row["energy_tracking_error"] == ""


def test_empty_satisfaction_has_blank_mean_and_zero_observation_count():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 1, 0, all_user_satisfaction=[])])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["chargers"][0]["user_satisfaction_mean"] == ""
    assert summary["chargers"][0]["user_satisfaction_observation_count"] == 0
    assert summary["transformers"][0]["user_satisfaction_mean"] == ""
    assert summary["transformers"][0]["user_satisfaction_observation_count"] == 0


def test_satisfaction_count_and_weighted_transformer_aggregation():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0, all_user_satisfaction=[0.0, 1.0]),
        fake_charger(1, 1, 0, all_user_satisfaction=[1.0]),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["chargers"][0]["user_satisfaction_mean"] == pytest.approx(0.5)
    assert summary["chargers"][0]["user_satisfaction_observation_count"] == 2
    assert summary["chargers"][1]["user_satisfaction_mean"] == pytest.approx(1.0)
    assert summary["chargers"][1]["user_satisfaction_observation_count"] == 1
    assert summary["transformers"][0]["user_satisfaction_mean"] == pytest.approx(2.0 / 3.0)
    assert summary["transformers"][0]["user_satisfaction_observation_count"] == 3


def test_zero_inactive_denominator_produces_blank_fraction():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["inactive_slot_decision_count"] == 0
    assert summary["global"]["inactive_nonzero_action_count"] == 0
    assert summary["global"]["inactive_nonzero_action_fraction_all_slots"] == ""


def test_inactive_violation_counts_and_fraction():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 3, 0)])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5, -0.2, 0.0], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["global"]["inactive_slot_decision_count"] == 2
    assert summary["global"]["inactive_nonzero_action_count"] == 1
    assert summary["global"]["inactive_nonzero_action_fraction_all_slots"] == pytest.approx(0.5)


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
                "global_action_nonzero_fraction_active": 0.6,
                "active_action_decision_count": 10,
                "global_positive_action_fraction_active": 0.4,
                "global_zero_action_fraction_active": 0.5,
                "global_negative_action_fraction_active": 0.1,
                "global_action_fraction_at_positive_max_active": 0.2,
                "global_action_fraction_at_negative_min_active": 0.05,
                "global_positive_action_sum_active": 3.0,
                "global_negative_action_magnitude_sum_active": 1.0,
                "inactive_slot_decision_count": 2,
                "inactive_nonzero_action_count": 0,
                "inactive_nonzero_action_fraction_all_slots": 0.0,
                "observed_action_min_active": -0.5,
                "observed_action_max_active": 1.0,
                "environment_action_low": -1.0,
                "environment_action_high": 1.0,
                "action_tolerance": 1e-6,
                "environment_action_domain_support": "signed",
                "v2g_enabled": False,
                "v2g_enabled_source": "config:v2g_enabled",
                "transformer_action_fraction_at_max_active_macro_mean": 0.3,
                "charger_action_fraction_at_max_active_macro_mean": 0.4,
                "transformer_action_nonzero_fraction_active_macro_mean": 0.5,
                "charger_action_nonzero_fraction_active_macro_mean": 0.7,
                "transformer_positive_charge_action_hhi_mean": 0.5,
                "transformer_positive_charge_action_gini_mean": 0.0,
                "transformer_allocation_zero_pressure_step_fraction": 0.25,
                "transformer_allocation_valid_step_count": 3,
                "charger_positive_charge_action_hhi_mean": 0.6,
                "charger_positive_charge_action_gini_mean": 0.1,
                "charger_allocation_zero_pressure_step_fraction": 0.5,
                "charger_allocation_valid_step_count": 2,
                "total_transformer_overload": 4.0,
                "power_tracker_violation": 8.0,
                "tracking_error": 12.0,
                "energy_tracking_error": 16.0,
                "total_ev_served": 20.0,
                "total_energy_charged": 24.0,
                "total_energy_discharged": 0.0,
                "average_user_satisfaction": 0.9,
                "energy_user_satisfaction": 95.0,
            },
            {
                "global_action_fraction_at_max_active": 0.4,
                "global_action_fraction_at_max_all_slots": 0.3,
                "global_action_nonzero_fraction_active": 0.8,
                "active_action_decision_count": 20,
                "global_positive_action_fraction_active": 0.5,
                "global_zero_action_fraction_active": 0.25,
                "global_negative_action_fraction_active": 0.25,
                "global_action_fraction_at_positive_max_active": 0.4,
                "global_action_fraction_at_negative_min_active": 0.15,
                "global_positive_action_sum_active": 5.0,
                "global_negative_action_magnitude_sum_active": 3.0,
                "inactive_slot_decision_count": 4,
                "inactive_nonzero_action_count": 1,
                "inactive_nonzero_action_fraction_all_slots": 0.1,
                "observed_action_min_active": -1.0,
                "observed_action_max_active": 0.8,
                "environment_action_low": -1.0,
                "environment_action_high": 1.0,
                "action_tolerance": 1e-6,
                "environment_action_domain_support": "signed",
                "v2g_enabled": False,
                "v2g_enabled_source": "config:v2g_enabled",
                "transformer_action_fraction_at_max_active_macro_mean": 0.5,
                "charger_action_fraction_at_max_active_macro_mean": 0.6,
                "transformer_action_nonzero_fraction_active_macro_mean": 0.7,
                "charger_action_nonzero_fraction_active_macro_mean": 0.9,
                "transformer_positive_charge_action_hhi_mean": 0.7,
                "transformer_positive_charge_action_gini_mean": 0.2,
                "transformer_allocation_zero_pressure_step_fraction": 0.75,
                "transformer_allocation_valid_step_count": 1,
                "charger_positive_charge_action_hhi_mean": 0.8,
                "charger_positive_charge_action_gini_mean": 0.3,
                "charger_allocation_zero_pressure_step_fraction": 0.0,
                "charger_allocation_valid_step_count": 4,
                "total_transformer_overload": 6.0,
                "power_tracker_violation": 10.0,
                "tracking_error": 14.0,
                "energy_tracking_error": 18.0,
                "total_ev_served": 22.0,
                "total_energy_charged": 26.0,
                "total_energy_discharged": 0.0,
                "average_user_satisfaction": 1.0,
                "energy_user_satisfaction": 97.0,
            },
        ],
    )

    assert seed_row["n_eval_episodes"] == 2
    assert seed_row["global_action_fraction_at_max_active_mean"] == pytest.approx(0.3)
    assert seed_row["global_action_nonzero_fraction_active_mean"] == pytest.approx(0.7)
    assert seed_row["active_action_decision_count_mean"] == pytest.approx(15.0)
    assert seed_row["global_positive_action_fraction_active_mean"] == pytest.approx(0.45)
    assert seed_row["global_zero_action_fraction_active_mean"] == pytest.approx(0.375)
    assert seed_row["global_negative_action_fraction_active_mean"] == pytest.approx(0.175)
    assert seed_row["global_action_fraction_at_positive_max_active_mean"] == pytest.approx(0.3)
    assert seed_row["global_action_fraction_at_negative_min_active_mean"] == pytest.approx(0.1)
    assert seed_row["global_positive_action_sum_active_mean"] == pytest.approx(4.0)
    assert seed_row["global_negative_action_magnitude_sum_active_mean"] == pytest.approx(2.0)
    assert seed_row["inactive_slot_decision_count_mean"] == pytest.approx(3.0)
    assert seed_row["inactive_nonzero_action_count_mean"] == pytest.approx(0.5)
    assert seed_row["inactive_nonzero_action_fraction_all_slots_mean"] == pytest.approx(0.05)
    assert seed_row["observed_action_min_active_mean"] == pytest.approx(-0.75)
    assert seed_row["observed_action_max_active_mean"] == pytest.approx(0.9)
    assert seed_row["environment_action_low"] == pytest.approx(-1.0)
    assert seed_row["environment_action_high"] == pytest.approx(1.0)
    assert seed_row["action_tolerance"] == pytest.approx(1e-6)
    assert seed_row["environment_action_domain_support"] == "signed"
    assert seed_row["v2g_enabled"] is False
    assert seed_row["v2g_enabled_source"] == "config:v2g_enabled"
    assert seed_row["transformer_action_fraction_at_max_active_macro_mean"] == pytest.approx(0.4)
    assert seed_row["charger_action_fraction_at_max_active_macro_mean"] == pytest.approx(0.5)
    assert seed_row["transformer_action_nonzero_fraction_active_macro_mean"] == pytest.approx(0.6)
    assert seed_row["charger_action_nonzero_fraction_active_macro_mean"] == pytest.approx(0.8)
    assert seed_row["transformer_allocation_zero_pressure_step_fraction_mean"] == pytest.approx(0.5)
    assert seed_row["transformer_allocation_valid_step_count_mean"] == pytest.approx(2.0)
    assert seed_row["charger_allocation_zero_pressure_step_fraction_mean"] == pytest.approx(0.25)
    assert seed_row["charger_allocation_valid_step_count_mean"] == pytest.approx(3.0)
    assert seed_row["total_transformer_overload_mean"] == pytest.approx(5.0)
    assert seed_row["total_energy_charged_mean"] == pytest.approx(25.0)
    assert seed_row["total_energy_discharged_mean"] == pytest.approx(0.0)
    assert seed_row["energy_user_satisfaction_mean"] == pytest.approx(96.0)


def test_schema_columns_use_explicit_macro_and_action_contract_names():
    from utils.infrastructure_diagnostics import (
        CHARGER_DIAGNOSTIC_COLUMNS,
        EPISODE_DIAGNOSTIC_COLUMNS,
        SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
        TRANSFORMER_DIAGNOSTIC_COLUMNS,
    )

    assert "global_action_nonzero_fraction_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "environment_action_low" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "environment_action_high" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "observed_action_min_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "observed_action_max_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "action_tolerance" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "environment_action_domain_support" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "active_action_decision_count" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_positive_action_fraction_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_zero_action_fraction_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_negative_action_fraction_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_action_fraction_at_positive_max_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_action_fraction_at_negative_min_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_positive_action_sum_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_negative_action_magnitude_sum_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "inactive_slot_decision_count" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "inactive_nonzero_action_count" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "inactive_nonzero_action_fraction_all_slots" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "transformer_positive_charge_action_hhi_mean" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "transformer_positive_charge_action_gini_mean" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "charger_positive_charge_action_hhi_mean" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "charger_positive_charge_action_gini_mean" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "total_energy_discharged" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "v2g_enabled" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "v2g_enabled_source" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "transformer_action_fraction_at_max_active_macro_mean" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "charger_action_fraction_at_max_active_macro_mean" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "transformer_action_fraction_at_max_active_mean" not in EPISODE_DIAGNOSTIC_COLUMNS
    assert "charger_action_fraction_at_max_active_mean" not in EPISODE_DIAGNOSTIC_COLUMNS
    assert "transformer_action_hhi_mean" not in EPISODE_DIAGNOSTIC_COLUMNS
    assert "transformer_action_gini_mean" not in EPISODE_DIAGNOSTIC_COLUMNS
    assert "charger_action_hhi_mean" not in EPISODE_DIAGNOSTIC_COLUMNS
    assert "charger_action_gini_mean" not in EPISODE_DIAGNOSTIC_COLUMNS
    assert "action_nonzero_fraction_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "positive_action_fraction_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "zero_action_fraction_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "negative_action_fraction_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "action_fraction_at_positive_max_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "action_fraction_at_negative_min_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "positive_action_sum_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "negative_action_magnitude_sum_active" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_observation_count" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "charger_positive_charge_action_hhi_mean" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "charger_positive_charge_action_gini_mean" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "charger_action_hhi_mean" not in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "charger_action_gini_mean" not in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "action_nonzero_fraction_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "positive_action_fraction_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "zero_action_fraction_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "negative_action_fraction_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "action_fraction_at_positive_max_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "action_fraction_at_negative_min_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "positive_action_sum_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "negative_action_magnitude_sum_active" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_observation_count" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "charger_allocation_zero_pressure_step_fraction" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "charger_allocation_valid_step_count" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "active_action_decision_count_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "global_positive_action_fraction_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "global_zero_action_fraction_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "global_negative_action_fraction_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "global_action_fraction_at_positive_max_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "global_action_fraction_at_negative_min_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "global_positive_action_sum_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "global_negative_action_magnitude_sum_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "inactive_slot_decision_count_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "inactive_nonzero_action_count_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "observed_action_min_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "observed_action_max_active_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "environment_action_low" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "environment_action_high" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "action_tolerance" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "environment_action_domain_support" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "v2g_enabled" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "v2g_enabled_source" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "transformer_positive_charge_action_hhi_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "transformer_positive_charge_action_gini_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "charger_positive_charge_action_hhi_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "charger_positive_charge_action_gini_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "total_energy_discharged_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "transformer_action_fraction_at_max_active_macro_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "charger_action_fraction_at_max_active_macro_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "transformer_action_fraction_at_max_active_mean" not in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "charger_action_fraction_at_max_active_mean" not in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "transformer_action_hhi_mean" not in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "transformer_action_gini_mean" not in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "charger_action_hhi_mean" not in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "charger_action_gini_mean" not in SEED_SUMMARY_DIAGNOSTIC_COLUMNS


def test_v2g_metadata_prefers_runtime_attribute_over_config(tmp_path):
    from evaluate_td3_gnn_infrastructure_diagnostics import resolve_v2g_metadata

    config_path = tmp_path / "config.yaml"
    config_path.write_text("v2g_enabled: False\n")
    metadata = resolve_v2g_metadata(
        SimpleNamespace(v2g_enabled=True),
        config_path,
    )

    assert metadata == {
        "v2g_enabled": True,
        "v2g_enabled_source": "env.v2g_enabled",
    }


def test_v2g_metadata_parses_config_when_runtime_attribute_unavailable(tmp_path):
    from evaluate_td3_gnn_infrastructure_diagnostics import resolve_v2g_metadata

    config_path = tmp_path / "config.yaml"
    config_path.write_text("v2g_enabled: False\n")
    metadata = resolve_v2g_metadata(SimpleNamespace(), config_path)

    assert metadata == {
        "v2g_enabled": False,
        "v2g_enabled_source": "config:v2g_enabled",
    }


def test_v2g_metadata_records_unavailable_when_runtime_and_config_do_not_expose_it(tmp_path):
    from evaluate_td3_gnn_infrastructure_diagnostics import resolve_v2g_metadata

    config_path = tmp_path / "config.yaml"
    config_path.write_text("number_of_charging_stations: 25\n")
    metadata = resolve_v2g_metadata(SimpleNamespace(), config_path)

    assert metadata == {
        "v2g_enabled": "",
        "v2g_enabled_source": "unavailable",
    }


@pytest.mark.parametrize(
    ("mapped_action", "expected_message"),
    [
        ([np.nan], "Mapped action contains non-finite values"),
        ([np.inf], "Mapped action contains non-finite values"),
        ([-np.inf], "Mapped action contains non-finite values"),
        ([], "Mapped action length must match EV2Gym action dimension"),
        ([0.1, 0.2], "Mapped action length must match EV2Gym action dimension"),
    ],
)
def test_diagnostic_evaluator_rejects_invalid_mapped_action_before_env_step(
    mapped_action,
    expected_message,
):
    from evaluate_td3_gnn_infrastructure_diagnostics import evaluate_diagnostic_episode

    state = fake_state(
        action_mapper=[0],
        ev_features=[[0.5, 0.0, 1.0, 0.0, 0.0, 0.0]],
    )
    env = FailingStepEnv(state)

    with pytest.raises(ValueError, match=expected_message):
        evaluate_diagnostic_episode(
            policy=ConstantActionPolicy(mapped_action),
            env=env,
            seed=710000,
            max_action=1.0,
            max_action_tolerance=1e-6,
        )

    assert env.step_called is False


@pytest.mark.parametrize(
    ("active_slots", "expected_message"),
    [
        ([0, 0], "active slots contain duplicate"),
        ([-1], "active slots contain negative"),
        ([2], "active slots outside the EV2Gym action range"),
        ([0.5], "active slots must be integral"),
        ([1.0000005], "active slots must be integral"),
    ],
)
def test_diagnostic_action_contract_rejects_invalid_active_slots(active_slots, expected_message):
    from utils.infrastructure_diagnostics import validate_diagnostic_action_contract

    with pytest.raises(ValueError, match=expected_message):
        validate_diagnostic_action_contract(
            mapped_action=np.array([0.0, 0.0], dtype=float),
            active_slots=np.asarray(active_slots, dtype=float),
            action_dim=2,
            tolerance=1e-6,
        )


@pytest.mark.parametrize(
    ("active_slots", "ev_features", "expected_message"),
    [
        ([0, 0], [[0.5, 0.0, 1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0, 0.0, 0.0]], "active slots contain duplicate"),
        ([-1], [[0.5, 0.0, 1.0, 0.0, 0.0, 0.0]], "active slots contain negative"),
        ([1], [[0.5, 0.0, 1.0, 0.0, 0.0, 0.0]], "active slots outside the EV2Gym action range"),
        ([0.5], [[0.5, 0.0, 1.0, 0.0, 0.0, 0.0]], "active slots must be integral"),
    ],
)
def test_diagnostic_evaluator_rejects_invalid_active_slots_before_env_step(
    active_slots,
    ev_features,
    expected_message,
):
    from evaluate_td3_gnn_infrastructure_diagnostics import evaluate_diagnostic_episode

    state = fake_state(action_mapper=active_slots, ev_features=ev_features)
    env = FailingStepEnv(state)

    with pytest.raises(ValueError, match=expected_message):
        evaluate_diagnostic_episode(
            policy=ConstantActionPolicy([0.0]),
            env=env,
            seed=710000,
            max_action=1.0,
            max_action_tolerance=1e-6,
        )

    assert env.step_called is False


def test_standalone_aggregation_rejects_invalid_active_slots_without_filtering():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 2, 0)])
    with pytest.raises(ValueError, match="active slots outside the EV2Gym action range"):
        aggregate_infrastructure_actions(
            mapped_actions_by_step=[np.array([0.5, 0.0], dtype=float)],
            active_slots_by_step=[np.array([0, 2], dtype=float)],
            slot_to_charger_id=np.array([0, 0], dtype=int),
            charger_to_transformer_id={0: 0},
            max_action=1.0,
            tolerance=1e-6,
            env=env,
        )


def test_signed_fraction_invariant_passes_on_valid_fractions():
    from utils.infrastructure_diagnostics import validate_signed_fraction_invariant

    validate_signed_fraction_invariant(
        {
            "active_action_decision_count": 4,
            "positive_action_fraction_active": 0.5,
            "zero_action_fraction_active": 0.25,
            "negative_action_fraction_active": 0.25,
        },
        tolerance=1e-6,
    )


def test_signed_fraction_invariant_raises_on_invalid_fractions():
    from utils.infrastructure_diagnostics import validate_signed_fraction_invariant

    with pytest.raises(ValueError, match="Signed action fractions must sum to one"):
        validate_signed_fraction_invariant(
            {
                "active_action_decision_count": 4,
                "positive_action_fraction_active": 0.5,
                "zero_action_fraction_active": 0.5,
                "negative_action_fraction_active": 0.5,
            },
            tolerance=1e-6,
        )


def test_environment_low_and_high_violations_are_recorded_not_failed():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([fake_charger(0, 3, 0)], action_low=0.0, action_high=1.0)
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([-0.2, 1.2, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 1, 2], dtype=int)],
        slot_to_charger_id=np.array([0, 0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    global_summary = summary["global"]
    assert global_summary["active_action_below_environment_low_count"] == 1
    assert global_summary["active_action_below_environment_low_fraction"] == pytest.approx(1.0 / 3.0)
    assert global_summary["active_action_above_environment_high_count"] == 1
    assert global_summary["active_action_above_environment_high_fraction"] == pytest.approx(1.0 / 3.0)


def test_environment_bound_violation_fractions_are_blank_when_bounds_unavailable():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([-0.2, 1.2], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=None,
    )

    global_summary = summary["global"]
    assert global_summary["active_action_below_environment_low_count"] == ""
    assert global_summary["active_action_below_environment_low_fraction"] == ""
    assert global_summary["active_action_above_environment_high_count"] == ""
    assert global_summary["active_action_above_environment_high_fraction"] == ""


def test_aggregate_charger_satisfaction_fallback_uses_total_sum_and_served_count():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(
            0,
            1,
            0,
            total_evs_served=4,
            total_user_satisfaction=3.0,
            all_user_satisfaction=None,
        )
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    charger_summary = summary["chargers"][0]
    assert charger_summary["user_satisfaction_sum"] == pytest.approx(3.0)
    assert charger_summary["user_satisfaction_mean"] == pytest.approx(0.75)
    assert charger_summary["user_satisfaction_observation_count"] == 4
    assert charger_summary["user_satisfaction_source"] == "charger_total_user_satisfaction"


@pytest.mark.parametrize(
    ("served_count", "expected_message"),
    [
        (2.6, "total_evs_served must be integral"),
        (-1, "total_evs_served must be non-negative"),
        (np.nan, "total_evs_served must be finite"),
        (np.inf, "total_evs_served must be finite"),
    ],
)
def test_charger_service_summary_rejects_invalid_total_evs_served(
    served_count,
    expected_message,
):
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(
            0,
            1,
            0,
            total_evs_served=served_count,
            total_user_satisfaction=1.0,
            all_user_satisfaction=None,
        )
    ])

    with pytest.raises(ValueError, match=expected_message):
        aggregate_infrastructure_actions(
            mapped_actions_by_step=[np.array([0.5], dtype=float)],
            active_slots_by_step=[np.array([0], dtype=int)],
            slot_to_charger_id=np.array([0], dtype=int),
            charger_to_transformer_id={0: 0},
            max_action=1.0,
            tolerance=1e-6,
            env=env,
        )


@pytest.mark.parametrize("served_count", [2, 2.0])
def test_charger_service_summary_accepts_integral_total_evs_served(served_count):
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(
            0,
            1,
            0,
            total_evs_served=served_count,
            total_user_satisfaction=1.5,
            all_user_satisfaction=None,
        )
    ])

    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    charger_summary = summary["chargers"][0]
    assert charger_summary["served_ev_count"] == 2
    assert charger_summary["user_satisfaction_observation_count"] == 2
    assert charger_summary["user_satisfaction_mean"] == pytest.approx(0.75)


def test_zero_served_charger_has_blank_satisfaction_mean_with_source_metadata():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(
            0,
            1,
            0,
            total_evs_served=0,
            total_user_satisfaction=0.0,
            all_user_satisfaction=None,
        )
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.0], dtype=float)],
        active_slots_by_step=[np.array([0], dtype=int)],
        slot_to_charger_id=np.array([0], dtype=int),
        charger_to_transformer_id={0: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    charger_summary = summary["chargers"][0]
    assert charger_summary["user_satisfaction_sum"] == pytest.approx(0.0)
    assert charger_summary["user_satisfaction_mean"] == ""
    assert charger_summary["user_satisfaction_observation_count"] == 0
    assert charger_summary["user_satisfaction_source"] == "charger_total_user_satisfaction"


def test_transformer_satisfaction_is_served_ev_weighted_from_sums_and_counts():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0, total_evs_served=2, total_user_satisfaction=1.0, all_user_satisfaction=None),
        fake_charger(1, 1, 0, total_evs_served=1, total_user_satisfaction=1.0, all_user_satisfaction=None),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([0.5, 0.5], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    transformer_summary = summary["transformers"][0]
    assert transformer_summary["user_satisfaction_sum"] == pytest.approx(2.0)
    assert transformer_summary["user_satisfaction_observation_count"] == 3
    assert transformer_summary["user_satisfaction_mean_served_ev_weighted"] == pytest.approx(2.0 / 3.0)
    assert transformer_summary["user_satisfaction_source"] == "charger_satisfaction_sum_count"


def test_charger_and_transformer_discharge_are_realised_runtime_totals():
    from utils.infrastructure_diagnostics import aggregate_infrastructure_actions

    env = fake_env([
        fake_charger(0, 1, 0, total_energy_charged=2.0, total_energy_discharged=0.5),
        fake_charger(1, 1, 0, total_energy_charged=3.0, total_energy_discharged=0.75),
    ])
    summary = aggregate_infrastructure_actions(
        mapped_actions_by_step=[np.array([-1.0, -1.0], dtype=float)],
        active_slots_by_step=[np.array([0, 1], dtype=int)],
        slot_to_charger_id=np.array([0, 1], dtype=int),
        charger_to_transformer_id={0: 0, 1: 0},
        max_action=1.0,
        tolerance=1e-6,
        env=env,
    )

    assert summary["chargers"][0]["energy_discharged_kwh"] == pytest.approx(0.5)
    assert summary["chargers"][1]["energy_discharged_kwh"] == pytest.approx(0.75)
    assert summary["transformers"][0]["energy_discharged_kwh"] == pytest.approx(1.25)


def test_diagnostic_reconciliation_accepts_matching_counts_and_energy():
    from utils.infrastructure_diagnostics import validate_diagnostic_reconciliation

    episode_row = {
        "total_ev_served": 3.0,
        "total_energy_charged": 5.0,
        "total_energy_discharged": 1.25,
    }
    charger_rows = [
        {"served_ev_count": 2, "energy_charged_kwh": 2.0, "energy_discharged_kwh": 0.5},
        {"served_ev_count": 1, "energy_charged_kwh": 3.0, "energy_discharged_kwh": 0.75},
    ]
    transformer_rows = [
        {"served_ev_count": 3, "energy_charged_kwh": 5.0, "energy_discharged_kwh": 1.25},
    ]

    validate_diagnostic_reconciliation(episode_row, charger_rows, transformer_rows, tolerance=1e-6)


def test_diagnostic_reconciliation_raises_on_served_count_mismatch():
    from utils.infrastructure_diagnostics import validate_diagnostic_reconciliation

    episode_row = {"total_ev_served": 3.0}
    charger_rows = [{"served_ev_count": 2}]
    transformer_rows = [{"served_ev_count": 3}]

    with pytest.raises(ValueError, match="charger served_ev_count"):
        validate_diagnostic_reconciliation(episode_row, charger_rows, transformer_rows, tolerance=1e-6)


def test_diagnostic_reconciliation_raises_on_fractional_served_count_sum():
    from utils.infrastructure_diagnostics import validate_diagnostic_reconciliation

    episode_row = {"total_ev_served": 3.0}
    charger_rows = [{"served_ev_count": 2.6}]
    transformer_rows = [{"served_ev_count": 3.0}]

    with pytest.raises(ValueError, match="charger served_ev_count"):
        validate_diagnostic_reconciliation(episode_row, charger_rows, transformer_rows, tolerance=1e-6)


def test_diagnostic_reconciliation_raises_on_charged_energy_mismatch():
    from utils.infrastructure_diagnostics import validate_diagnostic_reconciliation

    episode_row = {"total_energy_charged": 5.0}
    charger_rows = [{"energy_charged_kwh": 4.9}]
    transformer_rows = [{"energy_charged_kwh": 5.0}]

    with pytest.raises(ValueError, match="charger energy_charged_kwh"):
        validate_diagnostic_reconciliation(episode_row, charger_rows, transformer_rows, tolerance=1e-6)


def test_diagnostic_reconciliation_raises_on_discharged_energy_mismatch():
    from utils.infrastructure_diagnostics import validate_diagnostic_reconciliation

    episode_row = {"total_energy_discharged": 1.25}
    charger_rows = [{"energy_discharged_kwh": 1.25}]
    transformer_rows = [{"energy_discharged_kwh": 1.0}]

    with pytest.raises(ValueError, match="transformer energy_discharged_kwh"):
        validate_diagnostic_reconciliation(episode_row, charger_rows, transformer_rows, tolerance=1e-6)


@pytest.mark.parametrize(
    ("low_values", "high_values"),
    [
        ([np.nan, 0.0], [1.0, 1.0]),
        ([0.0, 0.0], [np.inf, 1.0]),
        ([0.0, 0.0], [-np.inf, 1.0]),
    ],
)
def test_non_finite_environment_action_bounds_fail_clearly(low_values, high_values):
    from utils.infrastructure_diagnostics import validate_environment_action_bounds

    action_space = SimpleNamespace(
        low=np.asarray(low_values, dtype=float),
        high=np.asarray(high_values, dtype=float),
    )

    with pytest.raises(ValueError, match="Environment action-space bounds must be finite"):
        validate_environment_action_bounds(action_space, tolerance=1e-6)


def test_environment_action_bounds_reject_low_greater_than_high():
    from utils.infrastructure_diagnostics import validate_environment_action_bounds

    action_space = SimpleNamespace(
        low=np.array([0.0, 2.0], dtype=float),
        high=np.array([1.0, 1.0], dtype=float),
    )

    with pytest.raises(ValueError, match="Environment action-space low bounds must not exceed high bounds"):
        validate_environment_action_bounds(action_space, tolerance=1e-6)


def test_schema_v3_columns_include_service_discharge_and_environment_bound_metrics():
    from utils.infrastructure_diagnostics import (
        CHARGER_DIAGNOSTIC_COLUMNS,
        DIAGNOSTIC_SCHEMA_VERSION,
        EPISODE_DIAGNOSTIC_COLUMNS,
        SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
        TRANSFORMER_DIAGNOSTIC_COLUMNS,
    )

    assert DIAGNOSTIC_SCHEMA_VERSION == "3"
    assert "active_action_below_environment_low_count" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "active_action_below_environment_low_fraction" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "active_action_above_environment_high_count" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "active_action_above_environment_high_fraction" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "active_action_below_environment_low_count" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "active_action_above_environment_high_fraction" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_sum" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_source" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "energy_discharged_kwh" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_sum" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_mean_served_ev_weighted" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_source" in TRANSFORMER_DIAGNOSTIC_COLUMNS
    assert "energy_discharged_kwh" in TRANSFORMER_DIAGNOSTIC_COLUMNS


def test_schema_v3_keeps_existing_schema_v2_action_and_service_columns():
    from utils.infrastructure_diagnostics import (
        CHARGER_DIAGNOSTIC_COLUMNS,
        EPISODE_DIAGNOSTIC_COLUMNS,
        SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
        TRANSFORMER_DIAGNOSTIC_COLUMNS,
    )

    assert "global_positive_action_fraction_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_zero_action_fraction_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "global_negative_action_fraction_active" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "inactive_nonzero_action_fraction_all_slots" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "total_energy_discharged" in EPISODE_DIAGNOSTIC_COLUMNS
    assert "average_user_satisfaction_mean" in SEED_SUMMARY_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_mean" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_observation_count" in CHARGER_DIAGNOSTIC_COLUMNS
    assert "user_satisfaction_observation_count" in TRANSFORMER_DIAGNOSTIC_COLUMNS


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
