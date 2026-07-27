import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_infrastructure_diagnostic_smoke.py"
SOURCE_BUNDLE_SCRIPT = (
    PROJECT_ROOT / "m3_jobs" / "create_infrastructure_diagnostic_smoke_source_bundle.sh"
)
ARRAY_SCRIPT = PROJECT_ROOT / "m3_jobs" / "19_infrastructure_diagnostic_smoke_eval.slurm"

TASKS = [
    (0, "25cp", "actiongnn", 0, 710000, 25, 3),
    (1, "25cp", "hierarchical", 5, 710000, 25, 3),
    (2, "100cp", "actiongnn", 10, 720000, 100, 7),
    (3, "100cp", "hierarchical", 15, 720000, 100, 7),
    (4, "500cp", "actiongnn", 20, 730000, 500, 35),
    (5, "500cp", "hierarchical", 25, 730000, 500, 35),
    (6, "1000cp", "actiongnn", 30, 740000, 1000, 70),
    (7, "1000cp", "hierarchical", 35, 740000, 1000, 70),
]


def run_validator(*args, check=True):
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), *map(str, args)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"validator failed with {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def add_bytes(tar, name, payload=b"x"):
    data = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def create_tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            if isinstance(payload, tarfile.TarInfo):
                tar.addfile(payload)
            else:
                add_bytes(tar, name, payload)
    return path


def package_basename(scale="25cp", algorithm="actiongnn", formal_task_id=0):
    return (
        f"m3_controlled_multiscale_formal_{scale}_{algorithm}_seed0_"
        f"job58513929_task{formal_task_id}.tar.gz"
    )


def formal_members(scale="25cp", algorithm="actiongnn", missing=()):
    run_name = f"controlled_multiscale_formal_{scale}_{algorithm}_seed0"
    train_dir = f"train/{run_name}"
    members = {
        f"config/{scale}_{algorithm}_seed0_config.yaml": "simulation_length: 112\n",
        f"eval/{scale}_{algorithm}_seed0_eval30.csv": canonical_csv_text(
            scale=scale,
            algorithm=algorithm,
        ),
        f"{train_dir}/model.best_actor": "actor",
        f"{train_dir}/model.best_actor_optimizer": "actor-opt",
        f"{train_dir}/model.best_critic": "critic",
        f"{train_dir}/model.best_critic_optimizer": "critic-opt",
        f"{train_dir}/kwargs.yaml": "{}\n",
        "runtime_metadata/source_manifest.sha256": "abc  source.py\n",
        "runtime_metadata/task_runtime_metadata.env": "task_id=0\n",
    }
    for name in list(members):
        if name in set(missing) or Path(name).name in set(missing):
            members.pop(name)
    manifest_lines = []
    for name, payload in sorted(members.items()):
        digest = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
        manifest_lines.append(f"{digest}  {name}\n")
    members["runtime_metadata/package_file_checksums.sha256"] = "".join(manifest_lines)
    if "runtime_metadata/package_file_checksums.sha256" in set(missing):
        members.pop("runtime_metadata/package_file_checksums.sha256")
    return members


def create_formal_package(
    directory,
    scale="25cp",
    algorithm="actiongnn",
    formal_task_id=0,
    missing=(),
    corrupt_checksum=False,
):
    members = formal_members(scale=scale, algorithm=algorithm, missing=missing)
    if corrupt_checksum:
        members["runtime_metadata/package_file_checksums.sha256"] = (
            "0" * 64 + f"  config/{scale}_{algorithm}_seed0_config.yaml\n"
        )
    return create_tar(directory / package_basename(scale, algorithm, formal_task_id), members)


def canonical_csv_text(scale="25cp", algorithm="actiongnn", episode_reward=-10.0):
    output = io.StringIO()
    fieldnames = [
        "row_type",
        "algorithm",
        "seed",
        "episode_index",
        "episode_seed",
        "episode_steps",
        "done",
        "episode_reward",
        "tracking_error",
        "energy_tracking_error",
        "power_tracker_violation",
        "total_energy_charged",
        "total_energy_discharged",
        "average_user_satisfaction",
        "energy_user_satisfaction",
        "total_transformer_overload",
        "total_ev_served",
        "action_mean",
        "action_fraction_at_max",
        "active_action_count_mean",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for episode_index in range(30):
        writer.writerow(
            {
                "row_type": "episode",
                "algorithm": algorithm,
                "seed": "0",
                "episode_index": str(episode_index),
                "episode_seed": str({"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}[scale] + episode_index),
                "episode_steps": "112",
                "done": "True",
                "episode_reward": str(episode_reward),
                "tracking_error": str(abs(episode_reward)),
                "energy_tracking_error": "5.0",
                "power_tracker_violation": "6.0",
                "total_energy_charged": "25.0",
                "total_energy_discharged": "0.0",
                "average_user_satisfaction": "1.0",
                "energy_user_satisfaction": "100.0",
                "total_transformer_overload": "0.0",
                "total_ev_served": "25",
                "action_mean": "0.5",
                "action_fraction_at_max": "0.25",
                "active_action_count_mean": "12.0",
            }
        )
    writer.writerow({"row_type": "summary", "algorithm": algorithm, "seed": "0"})
    return output.getvalue()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_diagnostics(
    directory,
    scale="25cp",
    algorithm="actiongnn",
    charger_count=25,
    transformer_count=3,
    schema_version="3",
    inactive_nonzero=0,
    negative_fraction=0.0,
    signed_sum=1.0,
    service_delta=0.0,
):
    diagnostic_dir = directory / "diagnostics"
    episode_row = {
        "matrix_job_id": "999",
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": "0",
        "episode_index": "0",
        "episode_seed": str({"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}[scale]),
        "config": "config.yaml",
        "checkpoint_prefix": "model.best",
        "run_name": "run",
        "episode_steps": "112",
        "done": "True",
        "episode_reward": "-10.0",
        "global_action_mean_all_slots": "0.5",
        "global_action_fraction_at_max_all_slots": "0.25",
        "nonzero_action_count_mean_all_slots": "12.0",
        "active_action_decision_count": "10",
        "global_positive_action_fraction_active": str(signed_sum - negative_fraction),
        "global_zero_action_fraction_active": "0.0",
        "global_negative_action_fraction_active": str(negative_fraction),
        "inactive_nonzero_action_count": str(inactive_nonzero),
        "v2g_enabled": "False",
        "total_ev_served": str(25 + service_delta),
        "total_energy_charged": "25.0",
        "total_energy_discharged": "0.0",
        "average_user_satisfaction": "1.0",
        "energy_user_satisfaction": "100.0",
        "tracking_error": "10.0",
        "energy_tracking_error": "5.0",
        "power_tracker_violation": "6.0",
        "total_transformer_overload": "0.0",
        "diagnostic_schema_version": schema_version,
    }
    seed_row = {
        "matrix_job_id": "999",
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": "0",
        "n_eval_episodes": "1",
        "global_negative_action_fraction_active_mean": str(negative_fraction),
        "inactive_nonzero_action_count_mean": str(inactive_nonzero),
        "v2g_enabled": "False",
        "diagnostic_schema_version": schema_version,
    }
    charger_rows = []
    for charger_id in range(charger_count):
        charger_rows.append(
            {
                "matrix_job_id": "999",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": "0",
                "episode_index": "0",
                "episode_seed": episode_row["episode_seed"],
                "charger_id": str(charger_id),
                "transformer_id": str(charger_id % transformer_count),
                "n_active_ev_decisions": "1",
                "positive_action_fraction_active": str(signed_sum - negative_fraction),
                "zero_action_fraction_active": "0.0",
                "negative_action_fraction_active": str(negative_fraction),
                "served_ev_count": "1" if charger_id < 25 else "0",
                "energy_charged_kwh": "1.0" if charger_id < 25 else "0.0",
                "energy_discharged_kwh": "0.0",
                "user_satisfaction_sum": "1.0" if charger_id < 25 else "0.0",
                "user_satisfaction_mean": "1.0" if charger_id < 25 else "",
                "user_satisfaction_observation_count": "1" if charger_id < 25 else "0",
                "user_satisfaction_source": "fixture" if charger_id < 25 else "",
                "diagnostic_schema_version": schema_version,
            }
        )
    transformer_rows = []
    for transformer_id in range(transformer_count):
        chargers = [row for row in charger_rows if int(row["transformer_id"]) == transformer_id]
        served = sum(int(row["served_ev_count"]) for row in chargers)
        charged = sum(float(row["energy_charged_kwh"]) for row in chargers)
        satisfaction = sum(float(row["user_satisfaction_sum"]) for row in chargers)
        transformer_rows.append(
            {
                "matrix_job_id": "999",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": "0",
                "episode_index": "0",
                "episode_seed": episode_row["episode_seed"],
                "transformer_id": str(transformer_id),
                "n_active_ev_decisions": str(max(served, 1)),
                "positive_action_fraction_active": str(signed_sum - negative_fraction),
                "zero_action_fraction_active": "0.0",
                "negative_action_fraction_active": str(negative_fraction),
                "served_ev_count": str(served),
                "energy_charged_kwh": str(charged),
                "energy_discharged_kwh": "0.0",
                "user_satisfaction_sum": str(satisfaction),
                "user_satisfaction_mean": "1.0" if served else "",
                "user_satisfaction_mean_served_ev_weighted": "1.0" if served else "",
                "user_satisfaction_observation_count": str(served),
                "user_satisfaction_source": "fixture" if served else "",
                "diagnostic_schema_version": schema_version,
            }
        )
    write_csv(diagnostic_dir / "episode_diagnostics.csv", list(episode_row), [episode_row])
    write_csv(diagnostic_dir / "seed_summary_diagnostics.csv", list(seed_row), [seed_row])
    write_csv(diagnostic_dir / "charger_diagnostics.csv", list(charger_rows[0]), charger_rows)
    write_csv(
        diagnostic_dir / "transformer_diagnostics.csv",
        list(transformer_rows[0]),
        transformer_rows,
    )
    return diagnostic_dir


def create_task_package(path, include_checkpoint=False):
    members = {
        "stdout.log": "",
        "stderr.log": "",
        "diagnostics/episode_diagnostics.csv": "x",
        "diagnostics/seed_summary_diagnostics.csv": "x",
        "diagnostics/transformer_diagnostics.csv": "x",
        "diagnostics/charger_diagnostics.csv": "x",
        "canonical/complete_eval30.csv": "x",
        "canonical/canonical_episode0.csv": "x",
        "config/formal_config.yaml": "x",
        "validation/task_validation.json": "{}",
        "validation/canonical_reconciliation.csv": "field,pass\nx,True\n",
        "runtime_metadata/source_commit_sha.txt": "c72d7f5\n",
        "runtime_metadata/source_formal_job.env": "formal_job_id=58513929\n",
        "runtime_metadata/source_package.env": "x=1\n",
        "runtime_metadata/source_package.sha256": "x",
        "runtime_metadata/checkpoint_member_hashes.sha256": "x",
        "runtime_metadata/original_source_manifest.sha256": "x",
        "runtime_metadata/original_task_runtime_metadata.env": "x",
        "runtime_metadata/diagnostic_command.txt": "x",
        "runtime_metadata/evaluator_time_verbose.txt": "x",
        "runtime_metadata/task_runtime_metadata.env": "x",
    }
    if include_checkpoint:
        members["checkpoint_staging/model.best_actor"] = "leak"
    manifest = ""
    for name, payload in sorted(members.items()):
        digest = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
        manifest += f"{digest}  {name}\n"
    members["runtime_metadata/package_file_checksums.sha256"] = manifest
    members["runtime_metadata/package_file_list.txt"] = "\n".join(sorted(members)) + "\n"
    return create_tar(path, members)


@pytest.mark.parametrize(
    "task_id,scale,algorithm,formal_task_id,episode_seed,charger_count,transformer_count",
    TASKS,
)
def test_exact_eight_task_mapping(
    task_id, scale, algorithm, formal_task_id, episode_seed, charger_count, transformer_count
):
    result = run_validator("task-mapping", "--task-id", task_id)
    mapping = json.loads(result.stdout)

    assert mapping["task_id"] == task_id
    assert mapping["scale"] == scale
    assert mapping["algorithm"] == algorithm
    assert mapping["formal_task_id"] == formal_task_id
    assert mapping["episode_seed"] == episode_seed
    assert mapping["expected_charger_rows"] == charger_count
    assert mapping["expected_transformer_rows"] == transformer_count


def test_preferred_package_resolution_uses_individual_package(tmp_path):
    individual_root = tmp_path / "packages"
    individual_package = create_formal_package(individual_root)
    fallback_bundle = create_tar(
        tmp_path / "complete.tar.gz",
        {f"bundle/task_packages/{individual_package.name}": individual_package.read_bytes()},
    )

    result = run_validator(
        "resolve-package",
        "--task-id",
        0,
        "--individual-package-root",
        individual_root,
        "--complete-bundle",
        fallback_bundle,
        "--staging-dir",
        tmp_path / "staging",
    )
    payload = json.loads(result.stdout)

    assert payload["source_mode"] == "individual"
    assert Path(payload["package_path"]) == individual_package


def test_complete_bundle_fallback_extracts_exactly_one_nested_package(tmp_path):
    formal_package = create_formal_package(tmp_path / "source")
    complete_bundle = create_tar(
        tmp_path / "complete.tar.gz",
        {f"complete/task_packages/{formal_package.name}": formal_package.read_bytes()},
    )

    result = run_validator(
        "resolve-package",
        "--task-id",
        0,
        "--individual-package-root",
        tmp_path / "empty",
        "--complete-bundle",
        complete_bundle,
        "--staging-dir",
        tmp_path / "staging",
    )
    payload = json.loads(result.stdout)

    assert payload["source_mode"] == "complete_bundle"
    assert Path(payload["package_path"]).is_file()


@pytest.mark.parametrize("bundle_member_count", [0, 2])
def test_package_resolution_fails_on_zero_or_multiple_fallback_matches(tmp_path, bundle_member_count):
    members = {}
    for index in range(bundle_member_count):
        formal_package = create_formal_package(tmp_path / f"source{index}")
        members[f"complete/task_packages/{index}/{formal_package.name}"] = formal_package.read_bytes()
    complete_bundle = create_tar(tmp_path / "complete.tar.gz", members)

    result = run_validator(
        "resolve-package",
        "--task-id",
        0,
        "--individual-package-root",
        tmp_path / "empty",
        "--complete-bundle",
        complete_bundle,
        "--staging-dir",
        tmp_path / "staging",
        check=False,
    )

    assert result.returncode != 0


def test_formal_package_validation_rejects_unsafe_tar_path(tmp_path):
    package_path = create_tar(tmp_path / package_basename(), {"../evil": "bad"})

    result = run_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--package",
        package_path,
        "--extract-dir",
        tmp_path / "extract",
        check=False,
    )

    assert result.returncode != 0
    assert "unsafe" in result.stderr.lower()


def test_formal_package_validation_rejects_unsafe_tar_link(tmp_path):
    link = tarfile.TarInfo("safe-link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../outside"
    package_path = create_tar(tmp_path / package_basename(), {"safe-link": link})

    result = run_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--package",
        package_path,
        "--extract-dir",
        tmp_path / "extract",
        check=False,
    )

    assert result.returncode != 0
    assert "link" in result.stderr.lower()


def test_formal_package_validation_rejects_checksum_failure(tmp_path):
    package_path = create_formal_package(tmp_path, corrupt_checksum=True)

    result = run_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--package",
        package_path,
        "--extract-dir",
        tmp_path / "extract",
        check=False,
    )

    assert result.returncode != 0
    assert "checksum" in result.stderr.lower()


@pytest.mark.parametrize(
    "missing",
    [
        "model.best_actor",
        "model.best_actor_optimizer",
        "model.best_critic",
        "model.best_critic_optimizer",
        "kwargs.yaml",
        "runtime_metadata/package_file_checksums.sha256",
        "runtime_metadata/source_manifest.sha256",
        "runtime_metadata/task_runtime_metadata.env",
        "eval/25cp_actiongnn_seed0_eval30.csv",
        "config/25cp_actiongnn_seed0_config.yaml",
    ],
)
def test_formal_package_validation_rejects_missing_required_files(tmp_path, missing):
    package_path = create_formal_package(tmp_path, missing=[missing])

    result = run_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--package",
        package_path,
        "--extract-dir",
        tmp_path / "extract",
        check=False,
    )

    assert result.returncode != 0
    assert "missing" in result.stderr.lower()


@pytest.mark.parametrize(
    "task_id,scale,algorithm,formal_task_id,episode_seed,charger_count,transformer_count",
    TASKS,
)
def test_diagnostic_validation_accepts_expected_topology_rows(
    tmp_path, task_id, scale, algorithm, formal_task_id, episode_seed, charger_count, transformer_count
):
    diagnostic_dir = create_diagnostics(
        tmp_path,
        scale=scale,
        algorithm=algorithm,
        charger_count=charger_count,
        transformer_count=transformer_count,
    )

    run_validator(
        "validate-diagnostics",
        "--task-id",
        task_id,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
    )

    assert (tmp_path / "validation" / "task_validation.json").is_file()


def test_diagnostic_validation_rejects_schema_mismatch(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path, schema_version="2")

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        check=False,
    )

    assert result.returncode != 0
    assert "schema" in result.stderr.lower()


def test_diagnostic_validation_rejects_signed_fraction_invariant_failure(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path, signed_sum=1.2)

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        check=False,
    )

    assert result.returncode != 0
    assert "fraction" in result.stderr.lower()


def test_diagnostic_validation_rejects_inactive_action_failure(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path, inactive_nonzero=1)

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        check=False,
    )

    assert result.returncode != 0
    assert "inactive" in result.stderr.lower()


def test_diagnostic_validation_rejects_hierarchical_negative_actions(tmp_path):
    diagnostic_dir = create_diagnostics(
        tmp_path,
        scale="25cp",
        algorithm="hierarchical",
        negative_fraction=0.1,
    )

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        1,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        check=False,
    )

    assert result.returncode != 0
    assert "negative" in result.stderr.lower()


def test_diagnostic_validation_rejects_service_reconciliation_failure(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path, service_delta=1.0)

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        check=False,
    )

    assert result.returncode != 0
    assert "reconciliation" in result.stderr.lower()


def test_canonical_reconciliation_accepts_values_inside_tolerance(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    canonical_path = tmp_path / "canonical.csv"
    canonical_path.write_text(canonical_csv_text(), encoding="utf-8")

    run_validator(
        "reconcile-canonical",
        "--task-id",
        0,
        "--episode-diagnostics",
        diagnostic_dir / "episode_diagnostics.csv",
        "--canonical-csv",
        canonical_path,
        "--validation-dir",
        tmp_path / "validation",
    )

    rows = list(csv.DictReader((tmp_path / "validation" / "canonical_reconciliation.csv").open()))
    assert rows
    assert all(row["pass"] == "True" for row in rows)


def test_canonical_reconciliation_rejects_values_outside_tolerance(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    canonical_path = tmp_path / "canonical.csv"
    canonical_path.write_text(canonical_csv_text(episode_reward=-100.0), encoding="utf-8")

    result = run_validator(
        "reconcile-canonical",
        "--task-id",
        0,
        "--episode-diagnostics",
        diagnostic_dir / "episode_diagnostics.csv",
        "--canonical-csv",
        canonical_path,
        "--validation-dir",
        tmp_path / "validation",
        check=False,
    )

    assert result.returncode != 0
    rows = list(csv.DictReader((tmp_path / "validation" / "canonical_reconciliation.csv").open()))
    assert any(row["field"] == "episode_reward" and row["pass"] == "False" for row in rows)


def test_task_package_validation_rejects_checkpoint_leak(tmp_path):
    task_package = create_task_package(tmp_path / "task.tar.gz", include_checkpoint=True)

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()


def test_task_package_validation_accepts_checkpoint_free_package(tmp_path):
    task_package = create_task_package(tmp_path / "task.tar.gz")

    run_validator("validate-task-package", "--task-id", 0, "--package", task_package)


def test_source_bundle_prohibited_path_rejection():
    env = {
        **os.environ,
        "EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_DRY_RUN": "1",
        "EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_EXTRA_PATH": "saved_models/bad",
    }
    result = subprocess.run(
        ["bash", str(SOURCE_BUNDLE_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert result.returncode != 0
    assert "prohibited" in result.stderr.lower()


def test_source_bundle_dry_run():
    env = {**os.environ, "EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_DRY_RUN": "1"}
    result = subprocess.run(
        ["bash", str(SOURCE_BUNDLE_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "DRY_RUN_NO_ARCHIVE_CREATED" in result.stdout


@pytest.mark.parametrize("task_id", range(8))
def test_array_script_dry_run_maps_all_tasks(task_id):
    env = {
        **os.environ,
        "EV_GNN_DIAGNOSTIC_SMOKE_DRY_RUN": "1",
        "SLURM_ARRAY_TASK_ID": str(task_id),
    }
    result = subprocess.run(
        ["bash", str(ARRAY_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert f"task_id={task_id}" in result.stdout
    assert "DRY_RUN_NO_EVALUATION_OR_PACKAGING" in result.stdout


def test_m3_array_script_contains_no_git_or_training_command():
    script_text = ARRAY_SCRIPT.read_text(encoding="utf-8")

    assert "git " not in script_text
    assert "train_td3_gnn.py" not in script_text


def test_bash_syntax_for_batch_a_scripts():
    result = subprocess.run(
        ["bash", "-n", str(SOURCE_BUNDLE_SCRIPT), str(ARRAY_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
