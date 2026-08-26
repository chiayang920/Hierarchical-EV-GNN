import csv
import io

import pytest


def _diag():
    from scripts import transformer_constraint_projection_diagnostics as diagnostics

    return diagnostics


def test_projection_episode_summary_denominators_and_energy_conversion():
    diagnostics = _diag()
    rows = [
        {
            "environment_step": 0,
            "transformer_id": 0,
            "constraint_activated": False,
            "alpha": 1.0,
            "commanded_power_removed": 0.0,
        },
        {
            "environment_step": 0,
            "transformer_id": 1,
            "constraint_activated": True,
            "alpha": 0.5,
            "commanded_power_removed": 12.0,
        },
        {
            "environment_step": 1,
            "transformer_id": 0,
            "constraint_activated": True,
            "alpha": 0.25,
            "commanded_power_removed": 8.0,
        },
        {
            "environment_step": 1,
            "transformer_id": 1,
            "constraint_activated": False,
            "alpha": 1.0,
            "commanded_power_removed": 0.0,
        },
    ]

    summary = diagnostics.summarise_projection_episode(
        metadata={"algorithm": "hierarchical_transformer_constraint"},
        episode_index=3,
        episode_seed=710003,
        environment_steps=2,
        transformer_step_rows=rows,
        timestep_minutes=15,
        corrected_ev_decisions=3,
        active_ev_decisions=12,
    )

    assert summary["any_constraint_activation_step_fraction"] == pytest.approx(1.0)
    assert summary["transformer_activation_rate"] == pytest.approx(0.5)
    assert summary["active_alpha_mean"] == pytest.approx(0.375)
    assert summary["active_alpha_min"] == pytest.approx(0.25)
    assert summary["commanded_power_removed_kw_step_sum"] == pytest.approx(20.0)
    assert summary["commanded_power_removed_kw_mean_when_active"] == pytest.approx(10.0)
    assert summary["estimated_energy_removed_kwh"] == pytest.approx(5.0)
    assert summary["corrected_ev_decision_fraction"] == pytest.approx(0.25)


def test_projection_episode_summary_uses_explicit_environment_step_denominator():
    diagnostics = _diag()
    rows = [
        {
            "environment_step": 1,
            "transformer_id": 0,
            "constraint_activated": True,
            "alpha": 0.5,
            "commanded_power_removed": 3.0,
        },
        {
            "environment_step": 3,
            "transformer_id": 0,
            "constraint_activated": False,
            "alpha": 1.0,
            "commanded_power_removed": 0.0,
        },
    ]

    summary = diagnostics.summarise_projection_episode(
        metadata={},
        episode_index=0,
        episode_seed=710000,
        environment_steps=4,
        transformer_step_rows=rows,
        timestep_minutes=15,
        corrected_ev_decisions=1,
        active_ev_decisions=4,
    )

    assert summary["environment_steps"] == 4
    assert summary["activated_environment_steps"] == 1
    assert summary["transformer_step_observations"] == 2
    assert summary["activated_transformer_step_observations"] == 1
    assert summary["any_constraint_activation_step_fraction"] == pytest.approx(0.25)
    assert summary["transformer_activation_rate"] == pytest.approx(0.5)


def test_projection_seed_summary_pools_unequal_denominators_and_active_means():
    diagnostics = _diag()
    episode_rows = [
        {
            "environment_steps": 10,
            "activated_environment_steps": 10,
            "transformer_step_observations": 100,
            "activated_transformer_step_observations": 100,
            "any_constraint_activation_step_fraction": 1.0,
            "transformer_activation_rate": 1.0,
            "active_alpha_mean": 0.5,
            "active_alpha_min": 0.25,
            "active_alpha_sum": 50.0,
            "commanded_power_removed_kw_step_sum": 100.0,
            "commanded_power_removed_kw_mean_when_active": 1.0,
            "estimated_energy_removed_kwh": 25.0,
            "action_correction_l1_sum": 10.0,
            "action_correction_l1_mean_per_active_ev_decision": 0.5,
            "corrected_ev_decision_fraction": 0.5,
            "corrected_ev_decisions": 10,
            "active_ev_decisions": 20,
            "transformer_feasibility_violation_count": 0,
            "transformer_feasibility_max_excess_kw": 0.0,
        },
        {
            "environment_steps": 1,
            "activated_environment_steps": 0,
            "transformer_step_observations": 1,
            "activated_transformer_step_observations": 0,
            "any_constraint_activation_step_fraction": 0.0,
            "transformer_activation_rate": 0.0,
            "active_alpha_mean": None,
            "active_alpha_min": None,
            "active_alpha_sum": 0.0,
            "commanded_power_removed_kw_step_sum": 0.0,
            "commanded_power_removed_kw_mean_when_active": None,
            "estimated_energy_removed_kwh": 0.0,
            "action_correction_l1_sum": 0.0,
            "action_correction_l1_mean_per_active_ev_decision": 0.0,
            "corrected_ev_decision_fraction": 0.0,
            "corrected_ev_decisions": 0,
            "active_ev_decisions": 1,
            "transformer_feasibility_violation_count": 0,
            "transformer_feasibility_max_excess_kw": 0.0,
        },
    ]

    summary = diagnostics.summarise_projection_seed({}, episode_rows)

    assert summary["any_constraint_activation_step_fraction"] == pytest.approx(10 / 11)
    assert summary["transformer_activation_rate"] == pytest.approx(100 / 101)
    assert summary["active_alpha_mean"] == pytest.approx(0.5)
    assert summary["active_alpha_min"] == pytest.approx(0.25)
    assert summary["commanded_power_removed_kw_mean_when_active"] == pytest.approx(1.0)
    assert summary["corrected_ev_decision_fraction"] == pytest.approx(10 / 21)


def test_projection_episode_summary_represents_zero_activation_explicitly():
    diagnostics = _diag()
    rows = [
        {
            "environment_step": 0,
            "transformer_id": 0,
            "constraint_activated": False,
            "alpha": 1.0,
            "commanded_power_removed": 0.0,
        }
    ]

    summary = diagnostics.summarise_projection_episode(
        metadata={},
        episode_index=0,
        episode_seed=710000,
        environment_steps=1,
        transformer_step_rows=rows,
        timestep_minutes=15,
        corrected_ev_decisions=0,
        active_ev_decisions=4,
    )

    assert summary["any_constraint_activation_step_fraction"] == 0.0
    assert summary["transformer_activation_rate"] == 0.0
    assert summary["active_alpha_mean"] is None
    assert summary["active_alpha_min"] is None
    assert summary["commanded_power_removed_kw_mean_when_active"] is None
    assert summary["corrected_ev_decision_fraction"] == 0.0


def test_projection_transformer_rows_expose_required_audit_fields():
    diagnostics = _diag()
    details = {
        "transformers": [
            {
                "graph_index": 0,
                "transformer_id": 7,
                "constraint_activated": True,
                "alpha": 0.75,
                "raw_estimated_commanded_power": 12.0,
                "safe_estimated_commanded_power": 9.0,
                "physical_max_power": 10.0,
                "usable_safe_capacity": 9.0,
                "rounding_margin": 1.0,
                "commanded_power_removed": 3.0,
            }
        ],
        "total_action_correction_magnitude": 0.25,
    }

    rows = diagnostics.transformer_step_rows_from_projection_details(
        metadata={
            "algorithm": "hierarchical_transformer_constraint",
            "scale": "25cp",
            "training_seed": 0,
        },
        episode_index=2,
        episode_seed=710002,
        environment_step=5,
        details=details,
        active_ev_count_by_transformer={7: 3},
        corrected_ev_decisions=2,
    )

    assert rows == [
        {
            "algorithm": "hierarchical_transformer_constraint",
            "scale": "25cp",
            "training_seed": 0,
            "episode_index": 2,
            "episode_seed": 710002,
            "environment_step": 5,
            "graph_index": 0,
            "transformer_id": 7,
            "constraint_activated": True,
            "alpha": 0.75,
            "raw_estimated_commanded_power": 12.0,
            "safe_estimated_commanded_power": 9.0,
            "physical_max_power": 10.0,
            "usable_safe_capacity": 9.0,
            "rounding_margin": 1.0,
            "commanded_power_removed": 3.0,
            "active_ev_count_under_transformer": 3,
            "corrected_ev_decisions": 2,
            "total_action_correction_magnitude": 0.25,
        }
    ]


def test_projection_csv_writer_uses_stable_columns_and_blank_none(tmp_path):
    diagnostics = _diag()
    path = tmp_path / "projection_episode_summary.csv"

    diagnostics.write_csv_rows(
        path,
        diagnostics.PROJECTION_EPISODE_SUMMARY_COLUMNS,
        [
            {
                "algorithm": "hierarchical_transformer_constraint",
                "episode_index": 0,
                "active_alpha_min": None,
            }
        ],
    )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == diagnostics.PROJECTION_EPISODE_SUMMARY_COLUMNS
        rows = list(reader)
    assert rows[0]["active_alpha_min"] == ""
