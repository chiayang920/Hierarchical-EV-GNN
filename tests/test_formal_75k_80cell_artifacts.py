import csv
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from scripts import formal_75k_80cell_artifacts as artifacts
from scripts import formal_75k_80cell_workflow as workflow


def _csv_bytes(fieldnames, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _training_log_bytes(*, steps=75000, cadence=5000, checkpoint_count=15):
    fieldnames = [
        "type",
        "timestep",
        "eval/mean_reward",
        "eval/std_reward",
        "elapsed_seconds",
    ]
    rows = []
    for index in range(checkpoint_count):
        step = cadence * (index + 1)
        rows.append(
            {
                "type": "evaluation",
                "timestep": step,
                "eval/mean_reward": -1000 + index,
                "eval/std_reward": 10 + index,
                "elapsed_seconds": index + 1,
            }
        )
    rows.append(
        {
            "type": "final_evaluation",
            "timestep": steps,
            "eval/mean_reward": -900,
            "eval/std_reward": 9,
            "elapsed_seconds": checkpoint_count + 1,
        }
    )
    return _csv_bytes(fieldnames, rows)


def _write_tar(path: Path, members: dict[str, bytes]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def make_training_package(
    root: Path,
    cell,
    *,
    job_id="12345",
    source_identity="source-head",
    steps=75000,
    checkpoint_count=15,
    smoke=False,
    suffix="",
):
    prefix = "m3_formal75k_smoke" if smoke else "m3_formal75k_80cell"
    filename = (
        f"{prefix}_{cell.scale}_{cell.algorithm}_seed{cell.seed}_"
        f"job{job_id}_task{cell.task_id}{suffix}.tar.gz"
    )
    path = root / filename
    run_name = (
        f"formal75k_smoke_{cell.scale}_{cell.algorithm}_seed{cell.seed}_job{job_id}_task{cell.task_id}"
        if smoke
        else f"formal75k_{cell.scale}_{cell.algorithm}_seed{cell.seed}"
    )
    model_root = f"train/{run_name}"
    cadence = 256 if smoke else 5000
    eval_episodes = 1 if smoke else 5
    expected = 2 if smoke else 15
    metadata = "\n".join(
        [
            f"protocol_version={'formal_75k_smoke_v2' if smoke else 'formal_75k_80cell_v2'}",
            f"task_id={cell.task_id}",
            f"slurm_array_task_id={cell.task_id}",
            f"slurm_array_job_id={job_id}",
            f"scale={cell.scale}",
            f"algorithm={cell.algorithm}",
            f"seed={cell.seed}",
            f"training_steps={512 if smoke else steps}",
            f"evaluation_cadence={cadence}",
            f"evaluation_episodes={eval_episodes}",
            f"expected_scheduled_evaluations={expected}",
            f"actor_output_transform={cell.actor_output_transform}",
            f"checkpoint_selection_rule={workflow.CHECKPOINT_SELECTION_RULE}",
            f"source_identity={source_identity}",
            "source_bundle_identity=bundle-a",
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
        "algorithm": cell.algorithm,
        "config": cell.config_path,
        "seed": cell.seed,
        "max_timesteps": 512 if smoke else steps,
        "eval_freq": cadence,
        "eval_episodes": eval_episodes,
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
        f"{model_root}/training_log.csv": _training_log_bytes(
            steps=512 if smoke else steps,
            cadence=cadence,
            checkpoint_count=2 if smoke else checkpoint_count,
        ),
        f"{model_root}/model.best_actor": b"actor",
        f"{model_root}/model.best_actor_optimizer": b"actor-opt",
        f"{model_root}/model.best_critic": b"critic",
        f"{model_root}/model.best_critic_optimizer": b"critic-opt",
        f"{model_root}/model.last_actor": b"last-actor",
        f"{model_root}/model.last_critic": b"last-critic",
        "stdout.log": b"FORMAL_TASK_COMPLETED\n",
        "stderr.log": b"",
    }
    if smoke:
        smoke_eval_fields = [
            "run_name", "algorithm", "config", "seed", "episode_seed", "checkpoint",
            "episode_index", "episode_reward", "episode_steps", "done", "row_type",
            "tracking_error", "energy_tracking_error", "power_tracker_violation",
            "average_user_satisfaction", "total_energy_charged", "total_ev_served",
            "total_transformer_overload",
        ]
        smoke_rows = []
        for episode in range(3):
            smoke_rows.append({
                "run_name": f"{run_name}_reload_eval",
                "algorithm": cell.algorithm,
                "config": cell.config_path,
                "seed": cell.seed,
                "episode_seed": 760000 + episode,
                "checkpoint": f"{model_root}/model.best",
                "episode_index": episode,
                "episode_reward": -100 + episode,
                "episode_steps": 112,
                "done": "True",
                "row_type": "episode",
                "tracking_error": 100 - episode,
                "energy_tracking_error": 1,
                "power_tracker_violation": 0,
                "average_user_satisfaction": 0.9,
                "total_energy_charged": 10,
                "total_ev_served": 5,
                "total_transformer_overload": 0,
            })
        smoke_rows.append({**smoke_rows[-1], "row_type": "summary", "episode_index": ""})
        members[f"smoke_eval/{cell.scale}_{cell.algorithm}_seed{cell.seed}_eval.csv"] = _csv_bytes(
            smoke_eval_fields, smoke_rows
        )
    if cell.algorithm == "actiongnn_nonnegative":
        members[f"{model_root}/model.best.metadata.yaml"] = yaml.safe_dump(
            {
                "algorithm": "actiongnn_nonnegative",
                "actor_output_transform": "shifted_tanh_v1",
                "checkpoint_role": "best",
                "training_budget": 512 if smoke else steps,
                "eval_frequency": cadence,
                "internal_eval_episodes": eval_episodes,
                "discrete_actions": 1,
            }
        ).encode()
    _write_tar(path, members)
    return path


def test_exact_job_discovery_ignores_other_jobs_and_rejects_duplicates(tmp_path):
    cell = workflow.resolve_formal_cell(0)
    expected = make_training_package(tmp_path, cell, job_id="12345")
    make_training_package(tmp_path, cell, job_id="99999")

    found = artifacts.discover_training_packages(
        tmp_path, "12345", cells=[cell], smoke=False
    )
    assert found == {0: expected}

    make_training_package(tmp_path, cell, job_id="12345", suffix="_duplicate")
    with pytest.raises(ValueError, match="ambiguous|duplicate"):
        artifacts.discover_training_packages(tmp_path, "12345", cells=[cell], smoke=False)


def test_training_package_validation_rejects_50k_and_fourteen_checkpoints(tmp_path):
    cell = workflow.resolve_formal_cell(0)
    fifty = make_training_package(tmp_path, cell, steps=50000, suffix="_50k")
    with pytest.raises(ValueError, match="50k|75000"):
        artifacts.validate_training_package(fifty, cell, "12345", smoke=False)

    fourteen = make_training_package(
        tmp_path, cell, checkpoint_count=14, suffix="_14checkpoints"
    )
    with pytest.raises(ValueError, match="15 scheduled"):
        artifacts.validate_training_package(fourteen, cell, "12345", smoke=False)


def test_training_package_requires_protocol_and_slurm_task_identity(tmp_path):
    cell = workflow.resolve_formal_cell(0)
    package = make_training_package(tmp_path, cell)
    with tarfile.open(package, "r:gz") as archive:
        members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    metadata_name = "runtime_metadata/task_runtime_metadata.env"
    members[metadata_name] = members[metadata_name].replace(
        b"protocol_version=formal_75k_80cell_v2",
        b"protocol_version=wrong_protocol",
    )
    _write_tar(package, members)
    with pytest.raises(ValueError, match="protocol_version"):
        artifacts.validate_training_package(package, cell, "12345", smoke=False)

    package = make_training_package(tmp_path, cell, suffix="_task_identity")
    with tarfile.open(package, "r:gz") as archive:
        members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    members[metadata_name] = members[metadata_name].replace(
        b"slurm_array_task_id=0", b"slurm_array_task_id=79"
    )
    _write_tar(package, members)
    with pytest.raises(ValueError, match="slurm_array_task_id"):
        artifacts.validate_training_package(package, cell, "12345", smoke=False)


def test_training_package_validation_and_atomic_staging_preserve_identity(tmp_path):
    cell = workflow.resolve_formal_cell(0)
    package = make_training_package(tmp_path / "packages", cell)

    record = artifacts.validate_training_package(
        package,
        cell,
        "12345",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
        smoke=False,
    )
    assert record.summary["source_identity"] == "source-head"
    assert record.summary["model_best_present"] is True
    assert len(record.training_curve_rows) == 15
    assert record.summary["requested_cpu"] == "1"
    assert record.summary["requested_memory"] == "8G"
    assert record.summary["requested_walltime"] == "08:00:00"

    with pytest.raises(ValueError, match="source_bundle_identity"):
        artifacts.validate_training_package(
            package,
            cell,
            "12345",
            expected_source_identity="source-head",
            expected_source_bundle_identity="wrong-bundle",
            smoke=False,
        )

    staged = artifacts.stage_training_package(record, tmp_path / "stage")
    assert staged.checkpoint_prefix.name == "model.best"
    assert (staged.checkpoint_prefix.parent / "model.best_actor").read_bytes() == b"actor"
    assert staged.config_path.is_file()
    assert staged.metadata_path.is_file()

    with pytest.raises(ValueError, match="already exists|stale"):
        artifacts.stage_training_package(record, tmp_path / "stage")


def test_smoke_gate_requires_exact_two_cells_and_no_stderr_failure(tmp_path):
    packages = tmp_path / "packages"
    for cell in workflow.smoke_matrix():
        make_training_package(packages, cell, job_id="888", smoke=True)

    result = artifacts.validate_smoke_packages(
        packages, "888", tmp_path / "smoke_stage", expected_source_identity="source-head"
    )
    assert result.status == "PASS"
    assert result.cell_count == 2
    assert result.scientific_claim_generated is False

    bad = artifacts.discover_training_packages(
        packages, "888", cells=[workflow.resolve_smoke_cell(0)], smoke=True
    )[0]
    with tarfile.open(bad, "r:gz") as archive:
        bad_members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    bad_members["stderr.log"] = b"Traceback (most recent call last): implementation failure\n"
    _write_tar(bad, bad_members)
    with pytest.raises(ValueError, match="stderr.*implementation failure"):
        artifacts.validate_smoke_packages(
            packages, "888", tmp_path / "bad_smoke_stage", expected_source_identity="source-head"
        )


def test_smoke_gate_requires_successful_model_best_reload_and_eval_rows(tmp_path):
    packages = tmp_path / "packages"
    for cell in workflow.smoke_matrix():
        make_training_package(packages, cell, job_id="889", smoke=True)

    target = artifacts.discover_training_packages(
        packages, "889", cells=[workflow.resolve_smoke_cell(0)], smoke=True
    )[0]
    with tarfile.open(target, "r:gz") as archive:
        members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    metadata_name = "runtime_metadata/task_runtime_metadata.env"
    members[metadata_name] = members[metadata_name].replace(
        b"reload_eval_exit_status=0", b"reload_eval_exit_status=7"
    )
    _write_tar(target, members)
    with pytest.raises(ValueError, match="reload_eval_exit_status"):
        artifacts.validate_smoke_packages(
            packages, "889", tmp_path / "bad_reload", expected_source_identity="source-head"
        )

    make_training_package(packages, workflow.resolve_smoke_cell(0), job_id="889", smoke=True)
    with tarfile.open(target, "r:gz") as archive:
        members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    for name in list(members):
        if name.startswith("smoke_eval/"):
            del members[name]
    _write_tar(target, members)
    with pytest.raises(ValueError, match="smoke eval"):
        artifacts.validate_smoke_packages(
            packages, "889", tmp_path / "missing_eval", expected_source_identity="source-head"
        )


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
            "run_name": "eval", "algorithm": cell.algorithm, "config": cell.config_path,
            "seed": cell.seed, "episode_seed": 710000 + episode,
            "checkpoint": "/stage/model.best", "episode_index": episode,
            "episode_reward": -1000 + episode, "episode_steps": 112, "done": "True",
            "row_type": "episode", "tracking_error": 1000 - episode,
            "energy_tracking_error": 10, "power_tracker_violation": 1,
            "average_user_satisfaction": 0.9, "total_energy_charged": 100,
            "total_ev_served": 50, "total_transformer_overload": 0,
        })
    rows.append({**rows[-1], "row_type": "summary", "episode_index": ""})
    return _csv_bytes(fields, rows)


def make_eval_package(root, cell, train_job="12345", eval_job="23456"):
    path = root / (
        f"m3_formal75k_eval30_{cell.scale}_{cell.algorithm}_seed{cell.seed}_"
        f"trainjob{train_job}_job{eval_job}_task{cell.task_id}.tar.gz"
    )
    metadata = "\n".join([
        "protocol_version=formal_75k_eval30_v2",
        f"training_array_job_id={train_job}", f"eval_array_job_id={eval_job}",
        f"task_id={cell.task_id}", f"slurm_array_task_id={cell.task_id}",
        f"scale={cell.scale}", f"algorithm={cell.algorithm}",
        f"seed={cell.seed}", "source_identity=source-head",
        "source_bundle_identity=bundle-a",
        "source_training_package_sha256=training-package-sha",
        "checkpoint_role=model.best",
        "checkpoint_identity_sha256=checkpoint-sha", "config_identity_sha256=config-sha",
        "evaluation_episodes=30", "evaluation_exit_status=0",
    ]) + "\n"
    _write_tar(path, {
        "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
        f"eval/{cell.scale}_{cell.algorithm}_seed{cell.seed}_eval30.csv": _eval_csv_bytes(cell),
        "stderr.log": b"",
    })
    return path


def test_raw_eval_packages_produce_exact_canonical_rows(tmp_path):
    package_root = tmp_path / "eval_packages"
    for cell in workflow.formal_matrix():
        make_eval_package(package_root, cell)

    rows = artifacts.aggregate_eval30_packages(
        package_root,
        training_job_id="12345",
        eval_job_id="23456",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )
    assert len(rows) == 2400
    assert len({(r["scale"], r["algorithm"], r["seed"], r["episode_index"]) for r in rows}) == 2400
    assert all(r["checkpoint_role"] == "model.best" for r in rows)
    assert all(r["source_identity"] == "source-head" for r in rows)
    assert all(r["source_bundle_identity"] == "bundle-a" for r in rows)
    assert all(r["source_training_package_identity"] == "training-package-sha" for r in rows)


def test_eval_package_rejects_incomplete_training_provenance(tmp_path):
    cell = workflow.resolve_formal_cell(0)
    package = make_eval_package(tmp_path, cell)
    with tarfile.open(package, "r:gz") as archive:
        members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    metadata_name = "runtime_metadata/task_runtime_metadata.env"
    members[metadata_name] = members[metadata_name].replace(
        b"source_training_package_sha256=training-package-sha\n", b""
    )
    _write_tar(package, members)
    for other_cell in workflow.formal_matrix()[1:]:
        make_eval_package(tmp_path, other_cell)
    with pytest.raises(ValueError, match="provenance|source_training_package"):
        artifacts.aggregate_eval30_packages(
            tmp_path,
            training_job_id="12345",
            eval_job_id="23456",
            expected_source_identity="source-head",
            expected_source_bundle_identity="bundle-a",
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
            "matrix_job_id": "34567", "scale": cell.scale,
            "algorithm": cell.algorithm, "training_seed": cell.seed,
            "episode_index": episode, "episode_seed": 740000 + episode,
            "episode_reward": -1000 + episode,
            "global_action_fraction_at_max_active": 0.2,
            "global_action_mean_active": 0.5,
            "global_action_nonzero_fraction_active": 0.99,
            "transformer_action_fraction_at_max_active_macro_mean": 0.21,
            "charger_action_fraction_at_max_active_macro_mean": 0.22,
            "transformer_positive_charge_action_hhi_mean": 0.1,
            "transformer_positive_charge_action_gini_mean": 0.2,
            "charger_positive_charge_action_hhi_mean": 0.3,
            "charger_positive_charge_action_gini_mean": 0.4,
            "transformer_allocation_zero_pressure_step_fraction": 0.05,
            "charger_allocation_zero_pressure_step_fraction": 0.06,
            "total_transformer_overload": 0,
            "power_tracker_violation": 1,
            "tracking_error": 1000 - episode,
            "energy_tracking_error": 10,
            "total_ev_served": 50,
            "total_energy_charged": 100,
            "average_user_satisfaction": 0.9,
            "diagnostic_schema_version": 3,
        })
    return _csv_bytes(fields, rows)


def _same_pass_reconciliation_bytes():
    fields = ["reconciliation_contract_version", "episode_index", "field", "status", "failure_category"]
    rows = []
    for episode in range(30):
        for field in ("episode_reward", "tracking_error", "same_pass_at_max_count"):
            rows.append({
                "reconciliation_contract_version": 2,
                "episode_index": episode,
                "field": field,
                "status": "pass",
                "failure_category": "",
            })
    return _csv_bytes(fields, rows)


def _service_reconciliation_bytes():
    fields = [
        "episode_index", "served_count_reconciliation_status",
        "satisfaction_sum_reconciliation_status", "charged_energy_reconciliation_status",
        "discharged_energy_reconciliation_status",
    ]
    return _csv_bytes(fields, [
        {
            "episode_index": episode,
            "served_count_reconciliation_status": "pass",
            "satisfaction_sum_reconciliation_status": "pass",
            "charged_energy_reconciliation_status": "pass",
            "discharged_energy_reconciliation_status": "pass",
        }
        for episode in range(30)
    ])


def make_diagnostic_package(root, cell, train_job="12345", diagnostic_job="34567"):
    path = root / (
        f"m3_formal75k_diagnostics_{cell.scale}_{cell.algorithm}_seed{cell.seed}_"
        f"trainjob{train_job}_job{diagnostic_job}_task{cell.task_id}.tar.gz"
    )
    metadata = "\n".join([
        "protocol_version=formal_75k_diagnostic_v2",
        f"training_array_job_id={train_job}", f"diagnostic_array_job_id={diagnostic_job}",
        f"task_id={cell.task_id}", f"slurm_array_task_id={cell.task_id}",
        f"scale={cell.scale}", f"algorithm={cell.algorithm}",
        f"seed={cell.seed}", "source_identity=source-head",
        "source_bundle_identity=bundle-a",
        "source_training_package_sha256=training-package-sha",
        "checkpoint_role=model.best",
        "checkpoint_identity_sha256=checkpoint-sha", "config_identity_sha256=config-sha",
        "diagnostic_schema_version=3", "reconciliation_contract_version=2",
        "evaluation_episodes=30", "evaluation_exit_status=0", "validation_exit_status=0",
    ]) + "\n"
    mapping = {
        "status": "ok", "scale": cell.scale, "algorithm": cell.algorithm,
        "training_seed": cell.seed, "episode_count": 30,
        "expected_episode_count": 30, "checks": {"episode_count": True, "schema_version": True},
    }
    summary = {
        "status": "ok", "hard_gate_status": "pass", "mapping_validation_status": "pass",
        "same_pass_metric_status": "pass", "service_reconciliation_status": "pass",
        "mapping_validation_failure_count": 0, "same_pass_metric_failure_count": 0,
        "service_reconciliation_failure_count": 0,
        "diagnostic_schema_version": 3, "reconciliation_contract_version": 2,
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


def test_raw_diagnostic_packages_produce_full_reconciled_rows(tmp_path):
    package_root = tmp_path / "diagnostic_packages"
    for cell in workflow.formal_matrix():
        make_diagnostic_package(package_root, cell)

    result = artifacts.aggregate_diagnostic_packages(
        package_root,
        training_job_id="12345",
        diagnostic_job_id="34567",
        expected_source_identity="source-head",
        expected_source_bundle_identity="bundle-a",
    )
    assert len(result.diagnostic_rows) == 2400
    assert len(result.same_pass_eval_rows) == 2400
    assert len({
        (r["scale"], r["algorithm"], r["seed"], r["episode_index"])
        for r in result.diagnostic_rows
    }) == 2400
    assert all(r["mapping_validation"] == "pass" for r in result.diagnostic_rows)
    assert all(r["canonical_reconciliation"] == "pass" for r in result.diagnostic_rows)
    assert all(r["service_reconciliation"] == "pass" for r in result.diagnostic_rows)
    assert all(r["energy_reconciliation"] == "pass" for r in result.diagnostic_rows)
    assert all(r[workflow.KEY_MECHANISM_METRIC] == pytest.approx(0.2) for r in result.diagnostic_rows)
    assert all(r["source_bundle_identity"] == "bundle-a" for r in result.diagnostic_rows)
    assert all(r["source_training_package_identity"] == "training-package-sha" for r in result.diagnostic_rows)


def test_diagnostic_package_rejects_failed_reconciliation(tmp_path):
    cell = workflow.resolve_formal_cell(0)
    package = make_diagnostic_package(tmp_path, cell)
    with tarfile.open(package, "r:gz") as archive:
        members = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    members["validation/service_reconciliation.csv"] = _csv_bytes(
        [
            "episode_index", "served_count_reconciliation_status",
            "satisfaction_sum_reconciliation_status", "charged_energy_reconciliation_status",
            "discharged_energy_reconciliation_status",
        ],
        [{
            "episode_index": episode,
            "served_count_reconciliation_status": "fail" if episode == 0 else "pass",
            "satisfaction_sum_reconciliation_status": "pass",
            "charged_energy_reconciliation_status": "pass",
            "discharged_energy_reconciliation_status": "pass",
        } for episode in range(30)],
    )
    _write_tar(package, members)

    with pytest.raises(ValueError, match="service reconciliation"):
        artifacts.validate_diagnostic_package(
            package, cell, training_job_id="12345", diagnostic_job_id="34567"
        )


def test_full_training_aggregation_requires_80_packages_and_stages_all_cells(tmp_path):
    package_root = tmp_path / "training_packages"
    for cell in workflow.formal_matrix():
        make_training_package(package_root, cell)

    result = artifacts.aggregate_training_packages(
        package_root,
        training_job_id="12345",
        stage_root=tmp_path / "stage",
        expected_source_identity="source-head",
    )
    assert len(result.summary_rows) == 80
    assert len(result.training_curve_rows) == 1200
    assert len(result.staged_tasks) == 80
    assert all(staged.checkpoint_prefix.with_name("model.best_actor").is_file() for staged in result.staged_tasks)
