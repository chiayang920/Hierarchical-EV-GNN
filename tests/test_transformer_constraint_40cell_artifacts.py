import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml


def _workflow():
    from scripts import transformer_constraint_40cell_workflow as workflow

    return workflow


def _artifacts():
    from scripts import transformer_constraint_40cell_artifacts as artifacts

    return artifacts


def _canonical_episode_seed(cell, episode_index):
    return {
        "25cp": 710000,
        "100cp": 720000,
        "500cp": 730000,
        "1000cp": 740000,
    }[cell.scale] + 1000 * cell.seed + episode_index


def _csv_bytes(fieldnames, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _write_tar(path: Path, members: dict[str, bytes]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _training_log_bytes():
    rows = [
        {
            "type": "evaluation",
            "timestep": 5000 * (index + 1),
            "eval/mean_reward": -1000 + index,
            "eval/std_reward": 1,
            "elapsed_seconds": index + 1,
        }
        for index in range(15)
    ]
    return _csv_bytes(
        ["type", "timestep", "eval/mean_reward", "eval/std_reward", "elapsed_seconds"],
        rows,
    )


def _checkpoint_metadata(cell):
    workflow = _workflow()
    return {
        "metadata_schema": "hierarchical_transformer_constraint_checkpoint_v1",
        "algorithm": cell.algorithm,
        "action_domain_contract": "nonnegative_ev_rows_exact_zero_nonev_hard_transformer_feasible_v1",
        "actor_output_transform": workflow.ACTOR_OUTPUT_TRANSFORM,
        "actor_output_transform_formula": (
            "full_hierarchical_action -> training_noise_if_any -> "
            "hard_per_transformer_feasibility_projection"
        ),
        "non_ev_action": 0.0,
        "discrete_actions": 1,
        "training_budget": 75000,
        "start_timesteps": 1000,
        "eval_frequency": 5000,
        "internal_eval_episodes": 5,
        "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
        "checkpoint_role": "best",
    }


def _config(cell):
    return {
        "number_of_charging_stations": int(cell.scale.removesuffix("cp")),
        "simulation_length": 112,
        "v2g_enabled": False,
    }


def make_training_package(root, cell, *, job_id="12345", source="source-head", bundle="bundle-a"):
    workflow = _workflow()
    run_name = f"transformer_constraint_40cell_{cell.scale}_seed{cell.seed}"
    model_root = f"train/{run_name}"
    config_member = f"config/{cell.scale}_{cell.algorithm}_seed{cell.seed}_config.yaml"
    metadata = "\n".join(
        [
            "protocol_version=transformer_constraint_40cell_train_v1",
            f"task_id={cell.task_id}",
            f"slurm_array_task_id={cell.task_id}",
            f"slurm_array_job_id={job_id}",
            f"scale={cell.scale}",
            f"algorithm={cell.algorithm}",
            f"seed={cell.seed}",
            "training_steps=75000",
            "evaluation_cadence=5000",
            "evaluation_episodes=5",
            "expected_scheduled_evaluations=15",
            f"actor_output_transform={workflow.ACTOR_OUTPUT_TRANSFORM}",
            f"checkpoint_selection_rule={workflow.CHECKPOINT_SELECTION_RULE}",
            f"source_identity={source}",
            f"source_bundle_identity={bundle}",
            f"run_name={run_name}",
            f"model_dir_relative={model_root}",
            f"config_copy_relative={config_member}",
            "fresh_run=true",
            "training_exit_status=0",
            "task_elapsed_seconds=100",
            "requested_cpu=1",
            "requested_memory=8G",
            "requested_walltime=08:00:00",
        ]
    ) + "\n"
    kwargs = {
        "discrete_actions": 1,
        "transformer_voltage": 230.0,
        "transformer_phases": 3.0,
    }
    run_args = {
        "algorithm": cell.algorithm,
        "config": cell.config_path,
        "seed": cell.seed,
        "max_timesteps": 75000,
        "start_timesteps": 1000,
        "eval_freq": 5000,
        "eval_episodes": 5,
        "discrete_actions": 1,
    }
    package = root / f"m3_transformer_constraint_40cell_{cell.scale}_{cell.algorithm}_seed{cell.seed}_job{job_id}_task{cell.task_id}.tar.gz"
    members = {
        "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
        config_member: yaml.safe_dump(_config(cell)).encode(),
        f"{model_root}/run_args.yaml": yaml.safe_dump(run_args).encode(),
        f"{model_root}/kwargs.yaml": yaml.safe_dump(kwargs).encode(),
        f"{model_root}/config.yaml": yaml.safe_dump(_config(cell)).encode(),
        f"{model_root}/training_log.csv": _training_log_bytes(),
        f"{model_root}/model.best_actor": b"actor",
        f"{model_root}/model.best_actor_optimizer": b"actor-opt",
        f"{model_root}/model.best_critic": b"critic",
        f"{model_root}/model.best_critic_optimizer": b"critic-opt",
        f"{model_root}/model.best.metadata.yaml": yaml.safe_dump(_checkpoint_metadata(cell)).encode(),
        f"{model_root}/model.last_actor": b"last-actor",
        f"{model_root}/model.last_critic": b"last-critic",
        "stdout.log": b"TRANSFORMER_CONSTRAINT_TRAIN_TASK_COMPLETED\n",
        "stderr.log": b"",
    }
    _write_tar(package, members)
    return package


def _eval_rows(cell, *, wrong_episode_seed=None, checkpoint="/stage/checkpoint/model.best"):
    fieldnames = [
        "run_name",
        "algorithm",
        "config",
        "seed",
        "episode_seed",
        "checkpoint",
        "episode_index",
        "episode_reward",
        "episode_steps",
        "done",
        "row_type",
        "tracking_error",
        "energy_tracking_error",
        "power_tracker_violation",
        "average_user_satisfaction",
        "total_energy_charged",
        "total_ev_served",
        "total_transformer_overload",
    ]
    rows = []
    for episode_index in range(30):
        episode_seed = _canonical_episode_seed(cell, episode_index)
        if wrong_episode_seed == episode_index:
            episode_seed += 1
        rows.append(
            {
                "run_name": "eval30",
                "algorithm": cell.algorithm,
                "config": cell.config_path,
                "seed": cell.seed,
                "episode_seed": episode_seed,
                "checkpoint": checkpoint,
                "episode_index": episode_index,
                "episode_reward": -1000 + episode_index,
                "episode_steps": 112,
                "done": "True",
                "row_type": "episode",
                "tracking_error": 10,
                "energy_tracking_error": 1,
                "power_tracker_violation": 0,
                "average_user_satisfaction": 0.9,
                "total_energy_charged": 10,
                "total_ev_served": 5,
                "total_transformer_overload": 0,
            }
        )
    rows.append({**rows[-1], "row_type": "summary", "episode_index": ""})
    return fieldnames, rows


def make_eval_package(
    root,
    cell,
    *,
    train_job="12345",
    eval_job="23456",
    source="source-head",
    bundle="bundle-a",
    wrong_episode_seed=None,
    checkpoint="/stage/checkpoint/model.best",
):
    fieldnames, rows = _eval_rows(
        cell,
        wrong_episode_seed=wrong_episode_seed,
        checkpoint=checkpoint,
    )
    metadata = "\n".join(
        [
            "protocol_version=transformer_constraint_40cell_eval30_v1",
            f"training_array_job_id={train_job}",
            f"training_set_id=trainjob{train_job}",
            f"eval_array_job_id={eval_job}",
            f"task_id={cell.task_id}",
            f"slurm_array_task_id={cell.task_id}",
            f"scale={cell.scale}",
            f"algorithm={cell.algorithm}",
            f"seed={cell.seed}",
            f"source_identity={source}",
            f"source_bundle_identity={bundle}",
            "source_training_package_sha256=training-package-sha",
            "checkpoint_role=model.best",
            "checkpoint_identity_sha256=checkpoint-sha",
            "config_identity_sha256=config-sha",
            "evaluation_episodes=30",
            "evaluation_exit_status=0",
        ]
    ) + "\n"
    package = root / f"m3_transformer_constraint_eval30_{cell.scale}_{cell.algorithm}_seed{cell.seed}_trainjob{train_job}_job{eval_job}_task{cell.task_id}.tar.gz"
    _write_tar(
        package,
        {
            "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
            f"eval/{cell.scale}_{cell.algorithm}_seed{cell.seed}_eval30.csv": _csv_bytes(fieldnames, rows),
            "stderr.log": b"",
        },
    )
    return package


def make_all_eval_packages(root, *, bad_task=None, **bad_kwargs):
    for cell in _workflow().formal_matrix():
        kwargs = bad_kwargs if cell.task_id == bad_task else {}
        make_eval_package(root, cell, **kwargs)


def _diagnostic_rows(cell, *, wrong_episode_seed=None):
    fields = [
        "matrix_job_id",
        "scale",
        "algorithm",
        "training_seed",
        "episode_index",
        "episode_seed",
        "episode_reward",
        "global_action_mean_active",
        "global_action_nonzero_fraction_active",
        "transformer_action_fraction_at_max_active_macro_mean",
        "charger_action_fraction_at_max_active_macro_mean",
        "transformer_positive_charge_action_hhi_mean",
        "transformer_positive_charge_action_gini_mean",
        "charger_positive_charge_action_hhi_mean",
        "charger_positive_charge_action_gini_mean",
        "transformer_allocation_zero_pressure_step_fraction",
        "charger_allocation_zero_pressure_step_fraction",
        "total_transformer_overload",
        "power_tracker_violation",
        "tracking_error",
        "energy_tracking_error",
        "total_ev_served",
        "total_energy_charged",
        "average_user_satisfaction",
        "diagnostic_schema_version",
    ]
    rows = []
    for episode_index in range(30):
        episode_seed = _canonical_episode_seed(cell, episode_index)
        if wrong_episode_seed == episode_index:
            episode_seed += 1
        rows.append(
            {
                "matrix_job_id": "34567",
                "scale": cell.scale,
                "algorithm": cell.algorithm,
                "training_seed": cell.seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "episode_reward": -1000 + episode_index,
                "global_action_mean_active": 0.5,
                "global_action_nonzero_fraction_active": 0.9,
                "transformer_action_fraction_at_max_active_macro_mean": 0.2,
                "charger_action_fraction_at_max_active_macro_mean": 0.3,
                "transformer_positive_charge_action_hhi_mean": 0.1,
                "transformer_positive_charge_action_gini_mean": 0.2,
                "charger_positive_charge_action_hhi_mean": 0.3,
                "charger_positive_charge_action_gini_mean": 0.4,
                "transformer_allocation_zero_pressure_step_fraction": 0.01,
                "charger_allocation_zero_pressure_step_fraction": 0.02,
                "total_transformer_overload": 0,
                "power_tracker_violation": 0,
                "tracking_error": 10,
                "energy_tracking_error": 1,
                "total_ev_served": 5,
                "total_energy_charged": 10,
                "average_user_satisfaction": 0.9,
                "diagnostic_schema_version": 3,
            }
        )
    return fields, rows


def _projection_episode_rows(cell, *, wrong_episode_seed=None):
    fields = [
        "algorithm",
        "scale",
        "training_seed",
        "episode_index",
        "episode_seed",
        "environment_steps",
        "activated_environment_steps",
        "transformer_step_observations",
        "activated_transformer_step_observations",
        "any_constraint_activation_step_fraction",
        "transformer_activation_rate",
        "active_alpha_mean",
        "active_alpha_min",
        "active_alpha_sum",
        "commanded_power_removed_kw_step_sum",
        "commanded_power_removed_kw_mean_when_active",
        "estimated_energy_removed_kwh",
        "action_correction_l1_sum",
        "action_correction_l1_mean_per_active_ev_decision",
        "corrected_ev_decision_fraction",
        "corrected_ev_decisions",
        "active_ev_decisions",
        "transformer_feasibility_violation_count",
        "transformer_feasibility_max_excess_kw",
    ]
    rows = []
    for episode_index in range(30):
        episode_seed = _canonical_episode_seed(cell, episode_index)
        if wrong_episode_seed == episode_index:
            episode_seed += 1
        rows.append(
            {
                "algorithm": cell.algorithm,
                "scale": cell.scale,
                "training_seed": cell.seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "environment_steps": 112,
                "activated_environment_steps": 1,
                "transformer_step_observations": 2,
                "activated_transformer_step_observations": 1,
                "any_constraint_activation_step_fraction": 1 / 112,
                "transformer_activation_rate": 0.5,
                "active_alpha_mean": 0.5,
                "active_alpha_min": 0.5,
                "active_alpha_sum": 0.5,
                "commanded_power_removed_kw_step_sum": 1.0,
                "commanded_power_removed_kw_mean_when_active": 1.0,
                "estimated_energy_removed_kwh": 0.25,
                "action_correction_l1_sum": 0.1,
                "action_correction_l1_mean_per_active_ev_decision": 0.01,
                "corrected_ev_decision_fraction": 0.1,
                "corrected_ev_decisions": 1,
                "active_ev_decisions": 10,
                "transformer_feasibility_violation_count": 0,
                "transformer_feasibility_max_excess_kw": 0.0,
            }
        )
    return fields, rows


def _projection_seed_summary(cell, *, patch=None):
    row = {
        "algorithm": cell.algorithm,
        "scale": cell.scale,
        "training_seed": cell.seed,
        "episodes": 30,
        "environment_steps": 3360,
        "activated_environment_steps": 30,
        "transformer_step_observations": 60,
        "activated_transformer_step_observations": 30,
        "any_constraint_activation_step_fraction": 30 / 3360,
        "transformer_activation_rate": 0.5,
        "active_alpha_mean": 0.5,
        "active_alpha_min": 0.5,
        "active_alpha_sum": 15.0,
        "commanded_power_removed_kw_step_sum": 30.0,
        "commanded_power_removed_kw_mean_when_active": 1.0,
        "estimated_energy_removed_kwh": 7.5,
        "action_correction_l1_sum": 3.0,
        "action_correction_l1_mean_per_active_ev_decision": 0.01,
        "corrected_ev_decision_fraction": 0.1,
        "corrected_ev_decisions": 30,
        "active_ev_decisions": 300,
        "transformer_feasibility_violation_count": 0,
        "transformer_feasibility_max_excess_kw": 0.0,
    }
    if patch:
        row.update(patch)
    return list(row), [row]


def _projection_transformer_rows(cell):
    fields = [
        "algorithm",
        "scale",
        "training_seed",
        "episode_index",
        "episode_seed",
        "environment_step",
        "graph_index",
        "transformer_id",
        "constraint_activated",
        "alpha",
        "raw_estimated_commanded_power",
        "safe_estimated_commanded_power",
        "physical_max_power",
        "usable_safe_capacity",
        "rounding_margin",
        "commanded_power_removed",
        "active_ev_count_under_transformer",
        "corrected_ev_decisions",
        "total_action_correction_magnitude",
    ]
    return fields, [
        {
            "algorithm": cell.algorithm,
            "scale": cell.scale,
            "training_seed": cell.seed,
            "episode_index": 0,
            "episode_seed": _canonical_episode_seed(cell, 0),
            "environment_step": 0,
            "graph_index": 0,
            "transformer_id": 0,
            "constraint_activated": True,
            "alpha": 0.5,
            "raw_estimated_commanded_power": 2.0,
            "safe_estimated_commanded_power": 1.0,
            "physical_max_power": 1.1,
            "usable_safe_capacity": 1.0,
            "rounding_margin": 0.1,
            "commanded_power_removed": 1.0,
            "active_ev_count_under_transformer": 1,
            "corrected_ev_decisions": 1,
            "total_action_correction_magnitude": 0.1,
        }
    ]


def _reconciliation_rows():
    fields = ["reconciliation_contract_version", "episode_index", "field", "status", "failure_category"]
    rows = [
        {
            "reconciliation_contract_version": 2,
            "episode_index": episode_index,
            "field": "episode_reward",
            "status": "pass",
            "failure_category": "",
        }
        for episode_index in range(30)
    ]
    return fields, rows


def _service_rows():
    fields = [
        "episode_index",
        "served_count_reconciliation_status",
        "satisfaction_sum_reconciliation_status",
        "charged_energy_reconciliation_status",
        "discharged_energy_reconciliation_status",
    ]
    rows = [
        {
            "episode_index": episode_index,
            "served_count_reconciliation_status": "pass",
            "satisfaction_sum_reconciliation_status": "pass",
            "charged_energy_reconciliation_status": "pass",
            "discharged_energy_reconciliation_status": "pass",
        }
        for episode_index in range(30)
    ]
    return fields, rows


def make_diagnostic_package(
    root,
    cell,
    *,
    wrong_projection_episode_seed=None,
    wrong_diagnostic_episode_seed=None,
    wrong_same_pass_episode_seed=None,
    projection_seed_summary_patch=None,
    checkpoint_role="model.best",
    source="source-head",
    bundle="bundle-a",
):
    diagnostic_fields, diagnostic_rows = _diagnostic_rows(
        cell,
        wrong_episode_seed=wrong_diagnostic_episode_seed,
    )
    same_pass_fields, same_pass_rows = _eval_rows(
        cell,
        wrong_episode_seed=wrong_same_pass_episode_seed,
    )
    projection_episode_fields, projection_episode_rows = _projection_episode_rows(
        cell,
        wrong_episode_seed=wrong_projection_episode_seed,
    )
    projection_seed_fields, projection_seed_rows = _projection_seed_summary(
        cell,
        patch=projection_seed_summary_patch,
    )
    projection_transformer_fields, projection_transformer_rows = _projection_transformer_rows(cell)
    reconciliation_fields, reconciliation_rows = _reconciliation_rows()
    service_fields, service_rows = _service_rows()
    metadata = "\n".join(
        [
            "protocol_version=transformer_constraint_40cell_diagnostic_v1",
            "training_array_job_id=12345",
            "training_set_id=trainjob12345",
            "diagnostic_array_job_id=34567",
            f"task_id={cell.task_id}",
            f"slurm_array_task_id={cell.task_id}",
            f"scale={cell.scale}",
            f"algorithm={cell.algorithm}",
            f"seed={cell.seed}",
            f"source_identity={source}",
            f"source_bundle_identity={bundle}",
            "source_training_package_sha256=training-package-sha",
            f"checkpoint_role={checkpoint_role}",
            "checkpoint_identity_sha256=checkpoint-sha",
            "config_identity_sha256=config-sha",
            "diagnostic_schema_version=3",
            "reconciliation_contract_version=2",
            "evaluation_episodes=30",
            "evaluation_exit_status=0",
            "projection_diagnostic_exit_status=0",
            "validation_exit_status=0",
        ]
    ) + "\n"
    mapping = {
        "status": "ok",
        "scale": cell.scale,
        "algorithm": cell.algorithm,
        "training_seed": cell.seed,
        "episode_count": 30,
        "checks": {"episode_count": True, "schema_version": True},
    }
    summary = {
        "status": "ok",
        "hard_gate_status": "pass",
        "mapping_validation_status": "pass",
        "same_pass_metric_status": "pass",
        "service_reconciliation_status": "pass",
        "mapping_validation_failure_count": 0,
        "same_pass_metric_failure_count": 0,
        "service_reconciliation_failure_count": 0,
        "diagnostic_schema_version": 3,
        "reconciliation_contract_version": 2,
    }
    package = root / f"m3_transformer_constraint_diagnostics_{cell.scale}_{cell.algorithm}_seed{cell.seed}_trainjob12345_job34567_task{cell.task_id}.tar.gz"
    _write_tar(
        package,
        {
            "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
            "diagnostics/episode_diagnostics.csv": _csv_bytes(diagnostic_fields, diagnostic_rows),
            "diagnostics/same_pass_canonical_eval30.csv": _csv_bytes(same_pass_fields, same_pass_rows),
            "projection_diagnostics/projection_transformer_step_rows.csv": _csv_bytes(projection_transformer_fields, projection_transformer_rows),
            "projection_diagnostics/projection_episode_summary.csv": _csv_bytes(projection_episode_fields, projection_episode_rows),
            "projection_diagnostics/projection_seed_summary.csv": _csv_bytes(projection_seed_fields, projection_seed_rows),
            "validation/mapping_validation.json": json.dumps(mapping).encode(),
            "validation/same_pass_canonical_reconciliation.csv": _csv_bytes(reconciliation_fields, reconciliation_rows),
            "validation/service_reconciliation.csv": _csv_bytes(service_fields, service_rows),
            "runtime_metadata/reconciliation_summary.json": json.dumps(summary).encode(),
            "stderr.log": b"",
        },
    )
    return package


def test_canonical_episode_seed_helper_matches_approved_contract():
    workflow = _workflow()
    cell = workflow.resolve_formal_cell(23)

    assert workflow.canonical_episode_seed(cell, 0) == 733000
    assert workflow.canonical_episode_seed(cell, 29) == 733029


def test_valid_constrained_training_package_is_accepted(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_training_package(tmp_path, cell)

    record = artifacts.validate_training_package(
        package,
        cell,
        "12345",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )

    assert record.summary["algorithm"] == "hierarchical_transformer_constraint"
    assert record.summary["checkpoint_selection_rule"] == _workflow().CHECKPOINT_SELECTION_RULE


def test_valid_constrained_eval30_packages_are_accepted(tmp_path):
    artifacts = _artifacts()
    make_all_eval_packages(tmp_path)

    rows = artifacts.aggregate_eval30_packages(
        tmp_path,
        training_job_id="12345",
        eval_job_id="23456",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )

    assert len(rows) == 1200
    assert all(row["algorithm"] == "hierarchical_transformer_constraint" for row in rows)
    assert all(row["checkpoint_role"] == "model.best" for row in rows)


def test_valid_constrained_diagnostic_projection_package_is_accepted(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_diagnostic_package(tmp_path, cell)

    record = artifacts.validate_diagnostic_package(
        package,
        cell,
        training_job_id="12345",
        diagnostic_job_id="34567",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )

    assert len(record.diagnostic_rows) == 30
    assert len(record.same_pass_eval_rows) == 30
    assert record.summary["projection_episode_count"] == 30
    assert record.summary["transformer_feasibility_violation_count"] == 0


def test_wrong_independent_eval30_episode_seed_is_rejected(tmp_path):
    artifacts = _artifacts()
    make_all_eval_packages(tmp_path, bad_task=0, wrong_episode_seed=7)

    with pytest.raises(ValueError, match="episode_seed"):
        artifacts.aggregate_eval30_packages(
            tmp_path,
            training_job_id="12345",
            eval_job_id="23456",
        )


def test_wrong_projection_episode_seed_is_rejected(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_diagnostic_package(tmp_path, cell, wrong_projection_episode_seed=7)

    with pytest.raises(ValueError, match="projection.*episode_seed|episode_seed"):
        artifacts.validate_diagnostic_package(package, cell, training_job_id="12345", diagnostic_job_id="34567")


def test_wrong_diagnostic_episode_seed_is_rejected(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_diagnostic_package(tmp_path, cell, wrong_diagnostic_episode_seed=7)

    with pytest.raises(ValueError, match="diagnostic.*episode_seed|episode_seed"):
        artifacts.validate_diagnostic_package(package, cell, training_job_id="12345", diagnostic_job_id="34567")


def test_wrong_same_pass_episode_seed_is_rejected(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_diagnostic_package(tmp_path, cell, wrong_same_pass_episode_seed=7)

    with pytest.raises(ValueError, match="same-pass.*episode_seed|episode_seed"):
        artifacts.validate_diagnostic_package(package, cell, training_job_id="12345", diagnostic_job_id="34567")


@pytest.mark.parametrize(
    "patch",
    [
        {"scale": "100cp"},
        {"algorithm": "hierarchical"},
        {"training_seed": 9},
    ],
)
def test_wrong_projection_seed_summary_identity_is_rejected(tmp_path, patch):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_diagnostic_package(
        tmp_path,
        cell,
        projection_seed_summary_patch=patch,
    )

    with pytest.raises(ValueError, match="projection seed summary"):
        artifacts.validate_diagnostic_package(package, cell, training_job_id="12345", diagnostic_job_id="34567")


def test_projection_seed_summary_nonzero_transformer_violation_is_rejected(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_diagnostic_package(
        tmp_path,
        cell,
        projection_seed_summary_patch={
            "transformer_feasibility_violation_count": 1,
            "transformer_feasibility_max_excess_kw": 0.01,
        },
    )

    with pytest.raises(ValueError, match="transformer feasibility invariant"):
        artifacts.validate_diagnostic_package(package, cell, training_job_id="12345", diagnostic_job_id="34567")


def test_model_last_cannot_substitute_for_canonical_model_best(tmp_path):
    artifacts = _artifacts()
    make_all_eval_packages(tmp_path, bad_task=0, checkpoint="/stage/checkpoint/model.last")

    with pytest.raises(ValueError, match="model.best"):
        artifacts.aggregate_eval30_packages(
            tmp_path,
            training_job_id="12345",
            eval_job_id="23456",
        )


def test_provenance_mismatch_remains_rejected(tmp_path):
    artifacts = _artifacts()
    make_all_eval_packages(tmp_path)

    with pytest.raises(ValueError, match="source_identity"):
        artifacts.aggregate_eval30_packages(
            tmp_path,
            training_job_id="12345",
            eval_job_id="23456",
            expected_source_identity="different-source",
        )
