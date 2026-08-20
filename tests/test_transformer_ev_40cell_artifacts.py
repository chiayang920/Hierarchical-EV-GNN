import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml


def _workflow():
    from scripts import transformer_ev_40cell_workflow as workflow

    return workflow


def _artifacts():
    from scripts import transformer_ev_40cell_artifacts as artifacts

    return artifacts


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


def _training_log_bytes(*, steps=75000, cadence=5000, count=15):
    rows = [
        {
            "type": "evaluation",
            "timestep": cadence * (index + 1),
            "eval/mean_reward": -1000 + index,
            "eval/std_reward": 10 + index,
            "elapsed_seconds": index + 1,
        }
        for index in range(count)
    ]
    rows.append(
        {
            "type": "final_evaluation",
            "timestep": steps,
            "eval/mean_reward": -900,
            "eval/std_reward": 9,
            "elapsed_seconds": count + 1,
        }
    )
    return _csv_bytes(["type", "timestep", "eval/mean_reward", "eval/std_reward", "elapsed_seconds"], rows)


def _checkpoint_metadata(cell, *, steps=75000, cadence=5000, episodes=5, algorithm=None):
    return {
        "metadata_schema": "hierarchical_transformer_ev_checkpoint_v1",
        "algorithm": algorithm or cell.algorithm,
        "action_domain_contract": "nonnegative_ev_rows_exact_zero_nonev_transformer_ev_v1",
        "actor_output_transform": "transformer_ev_hierarchy_v1",
        "actor_output_transform_formula": "clip(B_G * w_T(t_i) * w_EV(i | t_i) * g_i, 0, max_action)",
        "non_ev_action": 0.0,
        "discrete_actions": 1,
        "training_budget": steps,
        "start_timesteps": 64 if steps == 512 else 1000,
        "eval_frequency": cadence,
        "internal_eval_episodes": episodes,
        "checkpoint_selection_rule": _workflow().CHECKPOINT_SELECTION_RULE,
        "checkpoint_role": "best",
    }


def make_training_package(root, cell, *, job_id="12345", source="source-head", bundle="bundle-a", smoke=False, algorithm=None, suffix=""):
    workflow = _workflow()
    effective_algorithm = algorithm or cell.algorithm
    steps = workflow.SMOKE_TRAINING_STEPS if smoke else workflow.FORMAL_TRAINING_STEPS
    cadence = workflow.SMOKE_EVALUATION_CADENCE if smoke else workflow.FORMAL_EVALUATION_CADENCE
    episodes = workflow.SMOKE_TRAINING_EVALUATION_EPISODES if smoke else workflow.FORMAL_TRAINING_EVALUATION_EPISODES
    expected = 2 if smoke else 15
    prefix = "m3_transformer_ev_smoke" if smoke else "m3_transformer_ev_40cell"
    filename = f"{prefix}_{cell.scale}_{cell.algorithm}_seed{cell.seed}_job{job_id}_task{cell.task_id}{suffix}.tar.gz"
    path = root / filename
    run_name = (
        f"transformer_ev_smoke_{cell.scale}_seed{cell.seed}_job{job_id}_task{cell.task_id}"
        if smoke
        else f"transformer_ev_40cell_{cell.scale}_seed{cell.seed}"
    )
    model_root = f"train/{run_name}"
    metadata = "\n".join(
        [
            f"protocol_version={'transformer_ev_40cell_smoke_v1' if smoke else 'transformer_ev_40cell_train_v1'}",
            f"task_id={cell.task_id}",
            f"slurm_array_task_id={cell.task_id}",
            f"slurm_array_job_id={job_id}",
            f"scale={cell.scale}",
            f"algorithm={effective_algorithm}",
            f"seed={cell.seed}",
            f"training_steps={steps}",
            f"evaluation_cadence={cadence}",
            f"evaluation_episodes={episodes}",
            f"expected_scheduled_evaluations={expected}",
            "actor_output_transform=transformer_ev_hierarchy_v1",
            f"checkpoint_selection_rule={workflow.CHECKPOINT_SELECTION_RULE}",
            f"source_identity={source}",
            f"source_bundle_identity={bundle}",
            f"run_name={run_name}",
            f"model_dir_relative={model_root}",
            f"config_copy_relative=config/{cell.scale}_{cell.algorithm}_seed{cell.seed}_config.yaml",
            "fresh_run=true",
            "training_exit_status=0",
            *(["reload_eval_exit_status=0"] if smoke else []),
            "task_elapsed_seconds=100",
            "requested_cpu=1",
            "requested_memory=8G",
            "requested_walltime=08:00:00",
        ]
    ) + "\n"
    run_args = {
        "algorithm": effective_algorithm,
        "config": cell.config_path,
        "seed": cell.seed,
        "max_timesteps": steps,
        "start_timesteps": 64 if smoke else 1000,
        "eval_freq": cadence,
        "eval_episodes": episodes,
        "run_name": run_name,
        "discrete_actions": 1,
    }
    config = {
        "number_of_charging_stations": int(cell.scale.removesuffix("cp")),
        "simulation_length": 112,
        "v2g_enabled": False,
    }
    members = {
        "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
        f"config/{cell.scale}_{cell.algorithm}_seed{cell.seed}_config.yaml": yaml.safe_dump(config).encode(),
        f"{model_root}/run_args.yaml": yaml.safe_dump(run_args).encode(),
        f"{model_root}/kwargs.yaml": yaml.safe_dump({"discrete_actions": 1}).encode(),
        f"{model_root}/config.yaml": yaml.safe_dump(config).encode(),
        f"{model_root}/training_log.csv": _training_log_bytes(steps=steps, cadence=cadence, count=expected),
        f"{model_root}/model.best_actor": b"actor",
        f"{model_root}/model.best_actor_optimizer": b"actor-opt",
        f"{model_root}/model.best_critic": b"critic",
        f"{model_root}/model.best_critic_optimizer": b"critic-opt",
        f"{model_root}/model.best.metadata.yaml": yaml.safe_dump(
            _checkpoint_metadata(cell, steps=steps, cadence=cadence, episodes=episodes, algorithm=effective_algorithm)
        ).encode(),
        f"{model_root}/model.last_actor": b"last-actor",
        f"{model_root}/model.last_critic": b"last-critic",
        "stdout.log": b"TRANSFORMER_EV_TASK_COMPLETED\n",
        "stderr.log": b"",
    }
    if smoke:
        rows = []
        fields = [
            "run_name", "algorithm", "config", "seed", "episode_seed", "checkpoint",
            "episode_index", "episode_reward", "episode_steps", "done", "row_type",
            "tracking_error", "energy_tracking_error", "power_tracker_violation",
            "average_user_satisfaction", "total_energy_charged", "total_ev_served",
            "total_transformer_overload",
        ]
        for episode in range(3):
            rows.append({
                "run_name": f"{run_name}_reload_eval",
                "algorithm": effective_algorithm,
                "config": cell.config_path,
                "seed": cell.seed,
                "episode_seed": 710000 + episode,
                "checkpoint": f"{model_root}/model.best",
                "episode_index": episode,
                "episode_reward": -100 + episode,
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
            })
        rows.append({**rows[-1], "row_type": "summary", "episode_index": ""})
        members[f"smoke_eval/{cell.scale}_{cell.algorithm}_seed{cell.seed}_eval3.csv"] = _csv_bytes(fields, rows)
    _write_tar(path, members)
    return path


def test_package_status_distinguishes_missing_incomplete_mismatch_and_valid(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)

    assert artifacts.classify_training_package(tmp_path / "missing.tar.gz", cell, "12345").status == "MISSING"

    incomplete = make_training_package(tmp_path, cell, suffix="_incomplete")
    with tarfile.open(incomplete, "r:gz") as archive:
        members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    del members[next(name for name in members if name.endswith("model.best_critic"))]
    _write_tar(incomplete, members)
    assert artifacts.classify_training_package(incomplete, cell, "12345").status == "INCOMPLETE"

    mismatch = make_training_package(tmp_path, cell, suffix="_mismatch", algorithm="hierarchical")
    assert artifacts.classify_training_package(mismatch, cell, "12345").status == "IDENTITY_MISMATCH"

    valid = make_training_package(tmp_path, cell, suffix="_valid")
    assert artifacts.classify_training_package(valid, cell, "12345").status == "VALID"


def test_training_package_validation_rejects_wrong_algorithm_identity(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_training_package(tmp_path, cell, algorithm="actiongnn_nonnegative")

    with pytest.raises(ValueError, match="algorithm"):
        artifacts.validate_training_package(package, cell, "12345")


def test_exact_job_discovery_and_single_job_formal_aggregation_fails_closed(tmp_path):
    artifacts = _artifacts()
    workflow = _workflow()
    packages = tmp_path / "packages"
    expected_first = make_training_package(packages, workflow.resolve_formal_cell(0), job_id="12345")
    make_training_package(packages, workflow.resolve_formal_cell(0), job_id="99999")

    found = artifacts.discover_training_packages(packages, "12345", cells=[workflow.resolve_formal_cell(0)])
    assert found == {0: expected_first}

    for cell in workflow.formal_matrix()[1:]:
        make_training_package(packages, cell, job_id="12345")
    with pytest.raises(ValueError, match="split training job IDs"):
        artifacts.aggregate_training_packages(
            packages,
            training_job_id="12345",
            stage_root=tmp_path / "stage",
            expected_source_identity="source-head",
            expected_source_bundle_identity="bundle-a",
        )


def test_split_resource_training_jobs_aggregate_into_one_logical_40_cell_gate(tmp_path):
    artifacts = _artifacts()
    workflow = _workflow()
    packages = tmp_path / "packages"
    job_by_task = {}
    for cell in workflow.formal_matrix():
        if cell.task_id < 20:
            job_id = "11111"
        elif cell.task_id < 30:
            job_id = "22222"
        else:
            job_id = "33333"
        job_by_task[cell.task_id] = job_id
        make_training_package(packages, cell, job_id=job_id)

    result = artifacts.aggregate_training_packages(
        packages,
        training_job_ids_by_group={
            "small": "11111",
            "500cp": "22222",
            "1000cp": "33333",
        },
        stage_root=tmp_path / "stage",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )

    assert len(result.summary_rows) == 40
    assert {row["training_job_id"] for row in result.summary_rows} == {"11111", "22222", "33333"}
    assert {row["training_set_id"] for row in result.summary_rows} == {"trainset_11111_22222_33333"}
    assert len(result.staged_tasks) == 40
    assert all("/trainset_11111_22222_33333/" in str(staged.task_root) for staged in result.staged_tasks)
    assert all(staged.config_path.is_file() for staged in result.staged_tasks)
    assert all(staged.metadata_path.is_file() for staged in result.staged_tasks)


def test_checkpoint_identity_includes_load_affecting_kwargs_and_metadata(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_training_package(tmp_path, cell, suffix="_a")
    record_a = artifacts.validate_training_package(package, cell, "12345")

    package_b = make_training_package(tmp_path, cell, suffix="_b")
    with tarfile.open(package_b, "r:gz") as archive:
        members = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    kwargs_member = next(name for name in members if name.endswith("/kwargs.yaml"))
    members[kwargs_member] = yaml.safe_dump({"discrete_actions": 1, "feature_dim": 64}).encode()
    _write_tar(package_b, members)
    record_b = artifacts.validate_training_package(package_b, cell, "12345")

    assert record_a.summary["checkpoint_identity_sha256"] != record_b.summary["checkpoint_identity_sha256"]


def test_training_package_validation_rejects_incompatible_actor_kwargs(tmp_path):
    artifacts = _artifacts()
    cell = _workflow().resolve_formal_cell(0)
    package = make_training_package(tmp_path, cell)
    with tarfile.open(package, "r:gz") as archive:
        members = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    kwargs_member = next(name for name in members if name.endswith("/kwargs.yaml"))
    members[kwargs_member] = yaml.safe_dump({"discrete_actions": 0}).encode()
    _write_tar(package, members)

    with pytest.raises(ValueError, match="kwargs.discrete_actions"):
        artifacts.validate_training_package(package, cell, "12345")


def test_smoke_gate_requires_four_scale_seed0_packages(tmp_path):
    artifacts = _artifacts()
    workflow = _workflow()
    packages = tmp_path / "packages"
    for cell in workflow.smoke_matrix():
        make_training_package(packages, cell, job_id="888", smoke=True)

    result = artifacts.validate_smoke_packages(
        packages,
        "888",
        tmp_path / "smoke_stage",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )

    assert result.status == "PASS"
    assert result.cell_count == 4
    assert result.scientific_claim_generated is False


def _eval_csv_bytes(cell):
    fields = [
        "run_name", "algorithm", "config", "seed", "episode_seed", "checkpoint",
        "episode_index", "episode_reward", "episode_steps", "done", "row_type",
        "tracking_error", "energy_tracking_error", "power_tracker_violation",
        "average_user_satisfaction", "total_energy_charged", "total_ev_served",
        "total_transformer_overload",
    ]
    rows = []
    for episode in range(30):
        rows.append({
            "run_name": "eval30",
            "algorithm": cell.algorithm,
            "config": cell.config_path,
            "seed": cell.seed,
            "episode_seed": 710000 + episode,
            "checkpoint": "/stage/model.best",
            "episode_index": episode,
            "episode_reward": -1000 + episode,
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
        })
    rows.append({**rows[-1], "row_type": "summary", "episode_index": ""})
    return _csv_bytes(fields, rows)


def make_eval_package(root, cell, train_job="12345", eval_job="23456", bundle="bundle-a"):
    path = root / f"m3_transformer_ev_eval30_{cell.scale}_{cell.algorithm}_seed{cell.seed}_trainjob{train_job}_job{eval_job}_task{cell.task_id}.tar.gz"
    metadata = "\n".join([
        "protocol_version=transformer_ev_40cell_eval30_v1",
        f"training_array_job_id={train_job}",
        f"eval_array_job_id={eval_job}",
        f"task_id={cell.task_id}",
        f"slurm_array_task_id={cell.task_id}",
        f"scale={cell.scale}",
        f"algorithm={cell.algorithm}",
        f"seed={cell.seed}",
        "source_identity=source-head",
        f"source_bundle_identity={bundle}",
        "source_training_package_sha256=training-package-sha",
        "checkpoint_role=model.best",
        "checkpoint_identity_sha256=checkpoint-sha",
        "config_identity_sha256=config-sha",
        "evaluation_episodes=30",
        "evaluation_exit_status=0",
    ]) + "\n"
    _write_tar(path, {
        "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
        f"eval/{cell.scale}_{cell.algorithm}_seed{cell.seed}_eval30.csv": _eval_csv_bytes(cell),
        "stderr.log": b"",
    })
    return path


def test_eval30_aggregation_produces_1200_model_best_rows(tmp_path):
    artifacts = _artifacts()
    for cell in _workflow().formal_matrix():
        make_eval_package(tmp_path, cell)

    rows = artifacts.aggregate_eval30_packages(
        tmp_path,
        training_job_id="12345",
        eval_job_id="23456",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )

    assert len(rows) == 1200
    assert len({(row["scale"], row["algorithm"], row["seed"], row["episode_index"]) for row in rows}) == 1200
    assert all(row["algorithm"] == "hierarchical_transformer_ev" for row in rows)
    assert all(row["checkpoint_role"] == "model.best" for row in rows)


def test_eval30_aggregation_rejects_mixed_source_bundle_identities(tmp_path):
    artifacts = _artifacts()
    for cell in _workflow().formal_matrix():
        make_eval_package(tmp_path, cell, bundle="bundle-b" if cell.task_id == 0 else "bundle-a")

    with pytest.raises(ValueError, match="source-bundle"):
        artifacts.aggregate_eval30_packages(
            tmp_path,
            training_job_id="12345",
            eval_job_id="23456",
            expected_source_identity="source-head",
        )


def _diagnostic_csv_bytes(cell):
    fields = [
        "matrix_job_id", "scale", "algorithm", "training_seed", "episode_index",
        "episode_seed", "episode_reward", "global_action_fraction_at_max_active",
        "global_action_mean_active", "global_action_nonzero_fraction_active",
        "transformer_action_fraction_at_max_active_macro_mean",
        "charger_action_fraction_at_max_active_macro_mean",
        "transformer_positive_charge_action_hhi_mean",
        "transformer_positive_charge_action_gini_mean",
        "charger_positive_charge_action_hhi_mean",
        "charger_positive_charge_action_gini_mean",
        "transformer_allocation_zero_pressure_step_fraction",
        "charger_allocation_zero_pressure_step_fraction",
        "total_transformer_overload", "power_tracker_violation", "tracking_error",
        "energy_tracking_error", "total_ev_served", "total_energy_charged",
        "average_user_satisfaction", "diagnostic_schema_version",
    ]
    rows = []
    for episode in range(30):
        rows.append({
            "matrix_job_id": "34567",
            "scale": cell.scale,
            "algorithm": cell.algorithm,
            "training_seed": cell.seed,
            "episode_index": episode,
            "episode_seed": 740000 + episode,
            "episode_reward": -1000 + episode,
            "global_action_fraction_at_max_active": 0.2,
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
        })
    return _csv_bytes(fields, rows)


def _same_pass_reconciliation_bytes():
    fields = ["reconciliation_contract_version", "episode_index", "field", "status", "failure_category"]
    rows = [
        {
            "reconciliation_contract_version": 2,
            "episode_index": episode,
            "field": "episode_reward",
            "status": "pass",
            "failure_category": "",
        }
        for episode in range(30)
    ]
    return _csv_bytes(fields, rows)


def _service_reconciliation_bytes():
    fields = [
        "episode_index", "served_count_reconciliation_status",
        "satisfaction_sum_reconciliation_status", "charged_energy_reconciliation_status",
        "discharged_energy_reconciliation_status",
    ]
    rows = [
        {
            "episode_index": episode,
            "served_count_reconciliation_status": "pass",
            "satisfaction_sum_reconciliation_status": "pass",
            "charged_energy_reconciliation_status": "pass",
            "discharged_energy_reconciliation_status": "pass",
        }
        for episode in range(30)
    ]
    return _csv_bytes(fields, rows)


def make_diagnostic_package(root, cell, train_job="12345", diagnostic_job="34567", bundle="bundle-a"):
    path = root / f"m3_transformer_ev_diagnostics_{cell.scale}_{cell.algorithm}_seed{cell.seed}_trainjob{train_job}_job{diagnostic_job}_task{cell.task_id}.tar.gz"
    metadata = "\n".join([
        "protocol_version=transformer_ev_40cell_diagnostic_v1",
        f"training_array_job_id={train_job}",
        f"diagnostic_array_job_id={diagnostic_job}",
        f"task_id={cell.task_id}",
        f"slurm_array_task_id={cell.task_id}",
        f"scale={cell.scale}",
        f"algorithm={cell.algorithm}",
        f"seed={cell.seed}",
        "source_identity=source-head",
        f"source_bundle_identity={bundle}",
        "source_training_package_sha256=training-package-sha",
        "checkpoint_role=model.best",
        "checkpoint_identity_sha256=checkpoint-sha",
        "config_identity_sha256=config-sha",
        "diagnostic_schema_version=3",
        "reconciliation_contract_version=2",
        "evaluation_episodes=30",
        "evaluation_exit_status=0",
        "validation_exit_status=0",
    ]) + "\n"
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
    _write_tar(path, {
        "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
        "diagnostics/episode_diagnostics.csv": _diagnostic_csv_bytes(cell),
        "diagnostics/same_pass_canonical_eval30.csv": _eval_csv_bytes(cell),
        "validation/mapping_validation.json": json.dumps(mapping).encode(),
        "validation/same_pass_canonical_reconciliation.csv": _same_pass_reconciliation_bytes(),
        "validation/service_reconciliation.csv": _service_reconciliation_bytes(),
        "runtime_metadata/reconciliation_summary.json": json.dumps(summary).encode(),
        "stderr.log": b"",
    })
    return path


def test_diagnostic_aggregation_produces_1200_reconciled_rows(tmp_path):
    artifacts = _artifacts()
    for cell in _workflow().formal_matrix():
        make_diagnostic_package(tmp_path, cell)

    result = artifacts.aggregate_diagnostic_packages(
        tmp_path,
        training_job_id="12345",
        diagnostic_job_id="34567",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )

    assert len(result.diagnostic_rows) == 1200
    assert len(result.same_pass_eval_rows) == 1200
    assert all(row["mapping_validation"] == "pass" for row in result.diagnostic_rows)
    assert all(row["algorithm"] == "hierarchical_transformer_ev" for row in result.diagnostic_rows)


def test_diagnostic_aggregation_rejects_mixed_source_bundle_identities(tmp_path):
    artifacts = _artifacts()
    for cell in _workflow().formal_matrix():
        make_diagnostic_package(tmp_path, cell, bundle="bundle-b" if cell.task_id == 0 else "bundle-a")

    with pytest.raises(ValueError, match="source-bundle"):
        artifacts.aggregate_diagnostic_packages(
            tmp_path,
            training_job_id="12345",
            diagnostic_job_id="34567",
            expected_source_identity="source-head",
        )
