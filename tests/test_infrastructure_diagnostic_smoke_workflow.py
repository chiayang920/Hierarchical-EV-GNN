import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
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
REDUCER_SCRIPT = PROJECT_ROOT / "m3_jobs" / "20_infrastructure_diagnostic_smoke_reduce_bundle.slurm"
SUBMIT_SCRIPT = PROJECT_ROOT / "m3_jobs" / "submit_infrastructure_diagnostic_smoke_workflow.sh"

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

BASE_SHA = "c72d7f5da6738da6561904b0840c2faf540b8262"
DYNAMIC_SHA = "f" * 40
TOPOLOGY = {
    "25cp": (25, 3),
    "100cp": (100, 7),
    "500cp": (500, 35),
    "1000cp": (1000, 70),
}


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


def create_duplicate_tar(path, name="dup.txt"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        add_bytes(tar, name, "first")
        add_bytes(tar, name, "second")
    return path


def package_basename(scale="25cp", algorithm="actiongnn", formal_task_id=0):
    return (
        f"m3_controlled_multiscale_formal_{scale}_{algorithm}_seed0_"
        f"job58513929_task{formal_task_id}.tar.gz"
    )


def formal_config_text(
    scale="25cp",
    simulation_length=112,
    v2g_enabled=False,
    charger_count=None,
    transformer_count=None,
):
    expected_chargers, expected_transformers = TOPOLOGY[scale]
    charger_count = expected_chargers if charger_count is None else charger_count
    transformer_count = expected_transformers if transformer_count is None else transformer_count
    v2g_text = "True" if v2g_enabled else "False"
    return (
        f"simulation_length: {simulation_length}\n"
        f"v2g_enabled: {v2g_text}\n"
        f"number_of_charging_stations: {charger_count}\n"
        f"number_of_transformers: {transformer_count}\n"
    )


def formal_members(
    scale="25cp",
    algorithm="actiongnn",
    missing=(),
    manifest_omit=(),
    config_text=None,
):
    run_name = f"controlled_multiscale_formal_{scale}_{algorithm}_seed0"
    train_dir = f"train/{run_name}"
    members = {
        f"config/{scale}_{algorithm}_seed0_config.yaml": config_text or formal_config_text(scale),
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
        if name in set(manifest_omit) or Path(name).name in set(manifest_omit):
            continue
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
    manifest_omit=(),
    config_text=None,
):
    members = formal_members(
        scale=scale,
        algorithm=algorithm,
        missing=missing,
        manifest_omit=manifest_omit,
        config_text=config_text,
    )
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


def read_csv_dicts(path):
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader.fieldnames or []), list(reader)


def rewrite_csv(path, fieldnames, rows):
    write_csv(Path(path), fieldnames, rows)


def mutate_csv(path, row_index=0, updates=None, delete_columns=()):
    fieldnames, rows = read_csv_dicts(path)
    for column in delete_columns:
        fieldnames = [field for field in fieldnames if field != column]
        for row in rows:
            row.pop(column, None)
    if updates:
        rows[row_index].update(updates)
        for field in updates:
            if field not in fieldnames:
                fieldnames.append(field)
    rewrite_csv(path, fieldnames, rows)


def task_package_basename(task_id, array_job_id="123456"):
    _, scale, algorithm, _, _, _, _ = TASKS[task_id]
    return (
        f"m3_infrastructure_diagnostic_smoke_{scale}_{algorithm}_seed0_"
        f"job{array_job_id}_task{task_id}.tar.gz"
    )


def create_diagnostics(
    directory,
    scale="25cp",
    algorithm="actiongnn",
    charger_count=25,
    transformer_count=3,
    schema_version="3",
    matrix_job_id="999",
    inactive_nonzero=0,
    negative_fraction=0.0,
    signed_sum=1.0,
    service_delta=0.0,
):
    diagnostic_dir = directory / "diagnostics"
    episode_row = {
        "matrix_job_id": matrix_job_id,
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
        "active_action_below_environment_low_count": "0",
        "active_action_below_environment_low_fraction": "0.0",
        "active_action_above_environment_high_count": "0",
        "active_action_above_environment_high_fraction": "0.0",
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
        "matrix_job_id": matrix_job_id,
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": "0",
        "n_eval_episodes": "1",
        "active_action_decision_count_mean": "10",
        "active_action_below_environment_low_count": "0",
        "active_action_below_environment_low_fraction": "0.0",
        "active_action_above_environment_high_count": "0",
        "active_action_above_environment_high_fraction": "0.0",
        "global_positive_action_fraction_active_mean": str(signed_sum - negative_fraction),
        "global_zero_action_fraction_active_mean": "0.0",
        "global_negative_action_fraction_active_mean": str(negative_fraction),
        "inactive_nonzero_action_count_mean": str(inactive_nonzero),
        "v2g_enabled": "False",
        "diagnostic_schema_version": schema_version,
    }
    charger_rows = []
    for charger_id in range(charger_count):
        charger_rows.append(
            {
                "matrix_job_id": matrix_job_id,
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": "0",
                "episode_index": "0",
                "episode_seed": episode_row["episode_seed"],
                "charger_id": str(charger_id),
                "transformer_id": str(charger_id % transformer_count),
                "n_ports": "1",
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
                "matrix_job_id": matrix_job_id,
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": "0",
                "episode_index": "0",
                "episode_seed": episode_row["episode_seed"],
                "transformer_id": str(transformer_id),
                "n_chargers_total": str(len(chargers)),
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


def canonical_episode0_csv_text(scale="25cp", algorithm="actiongnn"):
    source = canonical_csv_text(scale=scale, algorithm=algorithm)
    reader = csv.DictReader(io.StringIO(source))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        if row["row_type"] == "episode" and row["episode_index"] == "0":
            writer.writerow(row)
            break
    return output.getvalue()


def reconciliation_csv_text(
    algorithm="actiongnn",
    episode_seed=710000,
    failed=False,
    missing_field=None,
    duplicate_field=None,
    extra_field=None,
    wrong_type_field=None,
    pass_value_override=None,
):
    fieldnames = [
        "field",
        "comparison_type",
        "canonical_value",
        "diagnostic_value",
        "absolute_difference",
        "relative_difference",
        "absolute_tolerance",
        "relative_tolerance",
        "pass",
    ]
    rows = []
    for field, value in [
        ("algorithm", algorithm),
        ("training seed", "0"),
        ("episode index", "0"),
        ("episode seed", str(episode_seed)),
        ("episode steps", "112"),
        ("done", "True"),
        ("total_ev_served", "25"),
    ]:
        rows.append(
            {
                "field": field,
                "comparison_type": "exact",
                "canonical_value": value,
                "diagnostic_value": value,
                "absolute_difference": "0",
                "relative_difference": "0",
                "absolute_tolerance": "0",
                "relative_tolerance": "0",
                "pass": "True",
            }
        )
    for field in [
        "episode_reward",
        "tracking_error",
        "energy_tracking_error",
        "power_tracker_violation",
        "total_energy_charged",
        "total_energy_discharged",
        "average_user_satisfaction",
        "energy_user_satisfaction",
        "total_transformer_overload",
        "global_action_mean_all_slots",
        "global_action_fraction_at_max_all_slots",
        "nonzero_action_count_mean_all_slots",
    ]:
        rows.append(
            {
                "field": field,
                "comparison_type": "floating",
                "canonical_value": "0",
                "diagnostic_value": "0",
                "absolute_difference": "0",
                "relative_difference": "0",
                "absolute_tolerance": "1",
                "relative_tolerance": "0",
                "pass": "True",
            }
        )
    if failed:
        rows[-1]["pass"] = "False"
        rows[-1]["absolute_difference"] = "2"
    if missing_field:
        rows = [row for row in rows if row["field"] != missing_field]
    if duplicate_field:
        duplicate = next(row for row in rows if row["field"] == duplicate_field).copy()
        rows.append(duplicate)
    if extra_field:
        extra = rows[-1].copy()
        extra["field"] = extra_field
        rows.append(extra)
    if wrong_type_field:
        for row in rows:
            if row["field"] == wrong_type_field:
                row["comparison_type"] = "floating" if row["comparison_type"] == "exact" else "exact"
                break
    if pass_value_override:
        field, pass_value = pass_value_override
        for row in rows:
            if row["field"] == field:
                row["pass"] = pass_value
                break
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def create_task_package(
    path,
    task_id=0,
    array_job_id="999",
    source_commit_sha=BASE_SHA,
    include_checkpoint=False,
    task_validation=None,
    failed_reconciliation=False,
    reconciliation_kwargs=None,
    source_resolution=None,
    stdout_log="",
    stderr_log="",
    evaluator_stdout="",
    evaluator_time_verbose="Maximum resident set size (kbytes): 1\n",
    manifest_omit=(),
    file_list_omit=(),
    omit_members=(),
):
    _, scale, algorithm, formal_task_id, episode_seed, charger_count, transformer_count = TASKS[task_id]
    fixture_root = path.parent / f"{path.stem}_fixture"
    diagnostic_dir = create_diagnostics(
        fixture_root,
        scale=scale,
        algorithm=algorithm,
        charger_count=charger_count,
        transformer_count=transformer_count,
        matrix_job_id=str(array_job_id),
    )
    task_validation = (
        {
            "status": "ok",
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "formal_task_id": formal_task_id,
            "episode_seed": episode_seed,
            "schema_version": "3",
            "episode_rows": 1,
            "seed_summary_rows": 1,
            "charger_rows": charger_count,
            "transformer_rows": transformer_count,
            "matrix_job_id": str(array_job_id),
        }
        if task_validation is None
        else task_validation
    )
    source_resolution = (
        {
            "source_mode": "individual_task_package",
            "package_path": f"/evidence/{package_basename(scale, algorithm, formal_task_id)}",
            "expected_package_name": package_basename(scale, algorithm, formal_task_id),
        }
        if source_resolution is None
        else source_resolution
    )
    source_package_env = (
        f"source_mode={source_resolution.get('source_mode', '')}\n"
        f"package_path={source_resolution.get('package_path', '')}\n"
        f"expected_package_name={source_resolution.get('expected_package_name', '')}\n"
    )
    if source_resolution.get("bundle_member"):
        source_package_env += f"bundle_member={source_resolution['bundle_member']}\n"
    members = {
        "stdout.log": stdout_log,
        "stderr.log": stderr_log,
        "diagnostics/episode_diagnostics.csv": (diagnostic_dir / "episode_diagnostics.csv").read_text(encoding="utf-8"),
        "diagnostics/seed_summary_diagnostics.csv": (diagnostic_dir / "seed_summary_diagnostics.csv").read_text(encoding="utf-8"),
        "diagnostics/transformer_diagnostics.csv": (diagnostic_dir / "transformer_diagnostics.csv").read_text(encoding="utf-8"),
        "diagnostics/charger_diagnostics.csv": (diagnostic_dir / "charger_diagnostics.csv").read_text(encoding="utf-8"),
        "canonical/complete_eval30.csv": canonical_csv_text(scale=scale, algorithm=algorithm),
        "canonical/canonical_episode0.csv": canonical_episode0_csv_text(scale=scale, algorithm=algorithm),
        "config/formal_config.yaml": formal_config_text(scale=scale),
        "validation/task_validation.json": json.dumps(task_validation, sort_keys=True) + "\n",
        "validation/canonical_reconciliation.csv": reconciliation_csv_text(
            algorithm=algorithm,
            episode_seed=episode_seed,
            failed=failed_reconciliation,
            **(reconciliation_kwargs or {}),
        ),
        "runtime_metadata/source_commit_sha.txt": source_commit_sha + "\n",
        "runtime_metadata/source_formal_job.env": f"formal_job_id=58513929\nformal_task_id={formal_task_id}\n",
        "runtime_metadata/source_package.env": source_package_env,
        "runtime_metadata/source_package_resolution.json": json.dumps(source_resolution, sort_keys=True) + "\n",
        "runtime_metadata/source_package.sha256": "0" * 64 + "  package.tar.gz\n",
        "runtime_metadata/checkpoint_member_hashes.sha256": "0" * 64 + "  train/model.best_actor\n",
        "runtime_metadata/original_source_manifest.sha256": "0" * 64 + "  source.py\n",
        "runtime_metadata/original_task_runtime_metadata.env": "task_id=0\n",
        "runtime_metadata/diagnostic_command.txt": "python evaluate_td3_gnn_infrastructure_diagnostics.py\n",
        "runtime_metadata/evaluator_stdout.txt": evaluator_stdout,
        "runtime_metadata/evaluator_time_verbose.txt": evaluator_time_verbose,
        "runtime_metadata/task_runtime_metadata.env": f"task_id={task_id}\n",
    }
    if include_checkpoint:
        members["checkpoint_staging/model.best_actor"] = "leak"
    for name in omit_members:
        members.pop(name, None)
    manifest = ""
    for name, payload in sorted(members.items()):
        if name in set(manifest_omit):
            continue
        digest = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
        manifest += f"{digest}  {name}\n"
    members["runtime_metadata/package_file_checksums.sha256"] = manifest
    listed_members = [
        name
        for name in sorted([*members, "runtime_metadata/package_file_list.txt"])
        if name not in set(file_list_omit)
    ]
    members["runtime_metadata/package_file_list.txt"] = "\n".join(listed_members) + "\n"
    if "runtime_metadata/package_file_list.txt" not in set(manifest_omit):
        digest = hashlib.sha256(members["runtime_metadata/package_file_list.txt"].encode("utf-8")).hexdigest()
        members["runtime_metadata/package_file_checksums.sha256"] += (
            f"{digest}  runtime_metadata/package_file_list.txt\n"
        )
    return create_tar(path, members)


def make_fake_git(tmp_path, head_sha=DYNAMIC_SHA, branch="main", dirty=False):
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    fake_git = bin_dir / "git"
    fake_git.write_text(
        f"""#!{sys.executable}
import os
import sys
import tarfile

args = sys.argv[1:]
if args[:1] == ["-C"]:
    args = args[2:]

head_sha = os.environ.get("FAKE_GIT_HEAD_SHA", "{head_sha}")
branch = os.environ.get("FAKE_GIT_BRANCH", "{branch}")
dirty = os.environ.get("FAKE_GIT_DIRTY", "{'1' if dirty else '0'}") == "1"

if args == ["branch", "--show-current"]:
    print(branch)
    raise SystemExit(0)
if args == ["rev-parse", "HEAD"]:
    print(head_sha)
    raise SystemExit(0)
if args == ["diff-index", "--quiet", "HEAD", "--"]:
    raise SystemExit(1 if dirty else 0)
if args and args[0] == "archive":
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w") as archive:
        pass
    raise SystemExit(0)

print("unexpected fake git args: " + repr(args), file=sys.stderr)
raise SystemExit(99)
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return bin_dir


def run_source_bundle(tmp_path, extra_env=None, check=False):
    fake_git_bin = make_fake_git(tmp_path)
    env = {
        **os.environ,
        "PATH": f"{fake_git_bin}{os.pathsep}{os.environ['PATH']}",
        "EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_OUTPUT_ROOT": str(tmp_path / "out"),
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(SOURCE_BUNDLE_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=check,
    )


def run_array_real_guard(tmp_path, extra_env=None):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "SOURCE_COMMIT_SHA.txt").write_text(DYNAMIC_SHA + "\n", encoding="utf-8")
    env = {
        **os.environ,
        "SLURM_ARRAY_TASK_ID": "0",
        "EV_GNN_DIAGNOSTIC_SMOKE_REPO_ROOT": str(repo_root),
        "EV_GNN_DIAGNOSTIC_SMOKE_RUN_ROOT": str(tmp_path / "runs"),
        "EV_GNN_DIAGNOSTIC_SMOKE_OUTPUT_ROOT": str(tmp_path / "out"),
        "EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT": DYNAMIC_SHA,
        **(extra_env or {}),
    }
    return subprocess.run(
        ["bash", str(ARRAY_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )


def default_slurm_task_ids(array_job_id="123456"):
    return {
        0: "123463",
        1: "123469",
        2: "123475",
        3: "123479",
        4: "123483",
        5: "123488",
        6: "123500",
        7: str(array_job_id),
    }


def legacy_sacct_raw_text(array_job_id="123456"):
    rows = []
    for task_id in range(8):
        composite_id = f"{array_job_id}_{task_id}"
        rows.append(
            "|".join(
                [
                    composite_id,
                    "COMPLETED",
                    "0:0",
                    "12",
                    "4",
                    "2048K",
                    "00:00:10",
                ]
            )
        )
    return "\n".join(rows) + "\n"


def sacct_raw_text(
    array_job_id="123456",
    missing_task_ids=(),
    missing_batch_ids=(),
    missing_extern_ids=(),
    state_overrides=None,
    exit_overrides=None,
    batch_state_overrides=None,
    batch_exit_overrides=None,
    extern_state_overrides=None,
    extern_exit_overrides=None,
    parent_resource_blanks=(),
    parent_maxrss_blanks=(),
    parent_totalcpu_blanks=(),
    parent_elapsed_blanks=(),
    parent_alloc_blanks=(),
    batch_maxrss_blanks=(),
    batch_totalcpu_blanks=(),
    duplicate_parent_rows=(),
    duplicate_batch_rows=(),
    duplicate_extern_rows=(),
    extern_maxrss_overrides=None,
    extern_totalcpu_overrides=None,
):
    state_overrides = state_overrides or {}
    exit_overrides = exit_overrides or {}
    batch_state_overrides = batch_state_overrides or {}
    batch_exit_overrides = batch_exit_overrides or {}
    extern_state_overrides = extern_state_overrides or {}
    extern_exit_overrides = extern_exit_overrides or {}
    extern_maxrss_overrides = extern_maxrss_overrides or {}
    extern_totalcpu_overrides = extern_totalcpu_overrides or {}

    missing_task_ids = set(missing_task_ids)
    missing_batch_ids = set(missing_batch_ids)
    missing_extern_ids = set(missing_extern_ids)
    parent_resource_blanks = set(parent_resource_blanks)
    parent_maxrss_blanks = set(parent_maxrss_blanks)
    parent_totalcpu_blanks = set(parent_totalcpu_blanks)
    parent_elapsed_blanks = set(parent_elapsed_blanks)
    parent_alloc_blanks = set(parent_alloc_blanks)
    batch_maxrss_blanks = set(batch_maxrss_blanks)
    batch_totalcpu_blanks = set(batch_totalcpu_blanks)
    duplicate_parent_rows = set(duplicate_parent_rows)
    duplicate_batch_rows = set(duplicate_batch_rows)
    duplicate_extern_rows = set(duplicate_extern_rows)

    rows = []
    numeric_ids = default_slurm_task_ids(array_job_id)
    for task_id in range(8):
        if task_id in missing_task_ids:
            continue
        job_id_raw = numeric_ids[task_id]
        job_id = f"{array_job_id}_{task_id}"
        parent_maxrss = (
            ""
            if task_id in parent_resource_blanks or task_id in parent_maxrss_blanks
            else "2048K"
        )
        parent_total_cpu = (
            ""
            if task_id in parent_resource_blanks or task_id in parent_totalcpu_blanks
            else "00:00:10"
        )
        parent_row = "|".join(
            [
                job_id_raw,
                job_id,
                "evgnn_infra_diag_smoke",
                state_overrides.get(task_id, "COMPLETED"),
                exit_overrides.get(task_id, "0:0"),
                "" if task_id in parent_elapsed_blanks else "12",
                "" if task_id in parent_alloc_blanks else "4",
                parent_maxrss,
                parent_total_cpu,
            ]
        )
        rows.append(parent_row)
        if task_id in duplicate_parent_rows:
            rows.append(parent_row)

        if task_id not in missing_batch_ids:
            batch_row = "|".join(
                [
                    f"{job_id_raw}.batch",
                    f"{job_id}.batch",
                    "batch",
                    batch_state_overrides.get(task_id, "COMPLETED"),
                    batch_exit_overrides.get(task_id, "0:0"),
                    "12",
                    "4",
                    "" if task_id in batch_maxrss_blanks else "4096K",
                    "" if task_id in batch_totalcpu_blanks else "00:00:12",
                ]
            )
            rows.append(batch_row)
            if task_id in duplicate_batch_rows:
                rows.append(batch_row)

        if task_id not in missing_extern_ids:
            extern_row = "|".join(
                [
                    f"{job_id_raw}.extern",
                    f"{job_id}.extern",
                    "extern",
                    extern_state_overrides.get(task_id, "COMPLETED"),
                    extern_exit_overrides.get(task_id, "0:0"),
                    "12",
                    "4",
                    extern_maxrss_overrides.get(task_id, ""),
                    extern_totalcpu_overrides.get(task_id, "00:00:00"),
                ]
            )
            rows.append(extern_row)
            if task_id in duplicate_extern_rows:
                rows.append(extern_row)
    return "\n".join(rows) + "\n"


M3_SACCT_FIELDS = (
    "JobIDRaw",
    "JobID",
    "JobName",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocCPUS",
    "MaxRSS",
    "TotalCPU",
)

RUNTIME_SUMMARY_FIELDS = (
    "task_id",
    "job_id_raw",
    "job_id",
    "state",
    "exit_code",
    "elapsed_raw",
    "alloc_cpus",
    "max_rss",
    "total_cpu",
    "maxrss_source",
    "totalcpu_source",
)

FAILED_JOB_58579309_SACCT = """\
58579316|58579309_0|evgnn_infra_diag_smoke|FAILED|1:0|41|4||00:21.129
58579316.batch|58579309_0.batch|batch|FAILED|1:0|41|4|927160K|00:21.129
58579316.extern|58579309_0.extern|extern|COMPLETED|0:0|41|4||00:00:00
58579322|58579309_1|evgnn_infra_diag_smoke|FAILED|1:0|29|4||00:48.102
58579322.batch|58579309_1.batch|batch|FAILED|1:0|29|4|451612K|00:48.102
58579322.extern|58579309_1.extern|extern|COMPLETED|0:0|29|4||00:00:00
58579328|58579309_2|evgnn_infra_diag_smoke|FAILED|1:0|23|4||00:34.384
58579328.batch|58579309_2.batch|batch|FAILED|1:0|23|4|450928K|00:34.384
58579328.extern|58579309_2.extern|extern|COMPLETED|0:0|23|4||00:00:00
58579332|58579309_3|evgnn_infra_diag_smoke|FAILED|1:0|24|4||00:38.894
58579332.batch|58579309_3.batch|batch|FAILED|1:0|24|4|449420K|00:38.894
58579332.extern|58579309_3.extern|extern|COMPLETED|0:0|24|4||00:00:00
58579336|58579309_4|evgnn_infra_diag_smoke|FAILED|1:0|50|4||01:29.707
58579336.batch|58579309_4.batch|batch|FAILED|1:0|50|4|452748K|01:29.707
58579336.extern|58579309_4.extern|extern|COMPLETED|0:0|50|4||00:00:00
58579341|58579309_5|evgnn_infra_diag_smoke|FAILED|1:0|32|4||00:49.294
58579341.batch|58579309_5.batch|batch|FAILED|1:0|32|4|448620K|00:49.294
58579341.extern|58579309_5.extern|extern|COMPLETED|0:0|32|4||00:00:00
58579353|58579309_6|evgnn_infra_diag_smoke|FAILED|1:0|68|4||02:17.780
58579353.batch|58579309_6.batch|batch|FAILED|1:0|68|4|461888K|02:17.780
58579353.extern|58579309_6.extern|extern|COMPLETED|0:0|68|4||00:00:00
58579309|58579309_7|evgnn_infra_diag_smoke|FAILED|1:0|38|4||00:55.791
58579309.batch|58579309_7.batch|batch|FAILED|1:0|38|4|454664K|00:55.791
58579309.extern|58579309_7.extern|extern|COMPLETED|0:0|39|4||00:00:00
"""


def m3_sacct_row(
    job_id_raw,
    job_id,
    job_name,
    state="COMPLETED",
    exit_code="0:0",
    elapsed_raw="12",
    alloc_cpus="4",
    max_rss="2048K",
    total_cpu="00:00:10",
):
    return "|".join(
        [
            str(job_id_raw),
            str(job_id),
            str(job_name),
            str(state),
            str(exit_code),
            str(elapsed_raw),
            str(alloc_cpus),
            str(max_rss),
            str(total_cpu),
        ]
    )


def completed_m3_sacct_text(array_job_id="58579309", include_aggregate=False):
    numeric_ids = {
        0: "58579316",
        1: "58579322",
        2: "58579328",
        3: "58579332",
        4: "58579336",
        5: "58579341",
        6: "58579353",
        7: str(array_job_id),
    }
    rows = []
    if include_aggregate:
        rows.append(
            m3_sacct_row(
                array_job_id,
                array_job_id,
                "evgnn_infra_diag_smoke",
                max_rss="",
            )
        )
    for task_id in range(8):
        raw_id = numeric_ids[task_id]
        composite_id = f"{array_job_id}_{task_id}"
        rows.extend(
            [
                m3_sacct_row(raw_id, composite_id, "evgnn_infra_diag_smoke"),
                m3_sacct_row(
                    f"{raw_id}.batch",
                    f"{composite_id}.batch",
                    "batch",
                    max_rss="4096K",
                    total_cpu="00:00:12",
                ),
                m3_sacct_row(
                    f"{raw_id}.extern",
                    f"{composite_id}.extern",
                    "extern",
                    max_rss="",
                    total_cpu="00:00:00",
                ),
            ]
        )
    return "\n".join(rows) + "\n"

def create_reducer_fixture(
    tmp_path,
    array_job_id="123456",
    source_commit_sha=DYNAMIC_SHA,
    package_overrides=None,
    slurm_stderr_overrides=None,
    missing_stdout_ids=(),
    missing_stderr_ids=(),
    sacct_text=None,
):
    package_overrides = package_overrides or {}
    slurm_stderr_overrides = slurm_stderr_overrides or {}
    package_root = tmp_path / "task_packages"
    slurm_log_root = tmp_path / "slurm_logs"
    output_root = tmp_path / "output"
    work_root = tmp_path / "work"
    package_root.mkdir()
    slurm_log_root.mkdir()
    output_root.mkdir()
    work_root.mkdir()
    for task_id in range(8):
        create_task_package(
            package_root / task_package_basename(task_id, array_job_id),
            task_id=task_id,
            array_job_id=array_job_id,
            source_commit_sha=source_commit_sha,
            **package_overrides.get(task_id, {}),
        )
        if task_id not in set(missing_stdout_ids):
            (slurm_log_root / f"evgnn_infra_diag_smoke_{array_job_id}_{task_id}.out").write_text(
                f"task {task_id} completed\n",
                encoding="utf-8",
            )
        if task_id not in set(missing_stderr_ids):
            (slurm_log_root / f"evgnn_infra_diag_smoke_{array_job_id}_{task_id}.err").write_text(
                slurm_stderr_overrides.get(task_id, ""),
                encoding="utf-8",
            )
    sacct_path = tmp_path / "sacct_raw.txt"
    sacct_path.write_text(sacct_text or sacct_raw_text(array_job_id), encoding="utf-8")
    reducer_stdout = tmp_path / "reducer_stdout.log"
    reducer_stderr = tmp_path / "reducer_stderr.log"
    reducer_stdout.write_text("reducer stdout snapshot\n", encoding="utf-8")
    reducer_stderr.write_text("reducer stderr snapshot\n", encoding="utf-8")
    return {
        "array_job_id": array_job_id,
        "source_commit_sha": source_commit_sha,
        "package_root": package_root,
        "slurm_log_root": slurm_log_root,
        "output_root": output_root,
        "work_root": work_root,
        "sacct_path": sacct_path,
        "reducer_stdout": reducer_stdout,
        "reducer_stderr": reducer_stderr,
    }


def run_reducer_validator(fixture, extra_args=(), check=False):
    return run_validator(
        "reduce-bundle",
        "--array-job-id",
        fixture["array_job_id"],
        "--task-package-root",
        fixture["package_root"],
        "--slurm-log-root",
        fixture["slurm_log_root"],
        "--output-root",
        fixture["output_root"],
        "--work-root",
        fixture["work_root"],
        "--source-commit-sha",
        fixture["source_commit_sha"],
        "--reducer-job-id",
        "reducer123",
        "--sacct-raw-file",
        fixture["sacct_path"],
        "--sacct-attempts",
        "1",
        "--sacct-delay-seconds",
        "0",
        "--reducer-stdout-log",
        fixture["reducer_stdout"],
        "--reducer-stderr-log",
        fixture["reducer_stderr"],
        *extra_args,
        check=check,
    )


def read_bundle_member(bundle_path, member_name):
    with tarfile.open(bundle_path, "r:gz") as bundle:
        member = bundle.extractfile(member_name)
        assert member is not None
        return member.read().decode("utf-8")


def complete_bundle_path(output_root, array_job_id="123456"):
    return output_root / f"infrastructure_diagnostic_smoke_complete_evidence_job{array_job_id}.tar.gz"


def complete_bundle_checksum_path(output_root, array_job_id="123456"):
    return output_root / f"infrastructure_diagnostic_smoke_complete_evidence_job{array_job_id}.tar.gz.sha256"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("infra_diag_smoke_validator_under_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_bundle_member_bytes(bundle_path, member_name):
    with tarfile.open(bundle_path, "r:gz") as bundle:
        member = bundle.extractfile(member_name)
        assert member is not None
        return member.read()


def refresh_complete_manifest_for_test(staging_root):
    staging_root = Path(staging_root)
    file_list_path = staging_root / "runtime_metadata/complete_file_list.txt"
    checksum_path = staging_root / "runtime_metadata/complete_file_checksums.sha256"
    file_list_path.unlink(missing_ok=True)
    checksum_path.unlink(missing_ok=True)
    file_names = sorted(
        path.relative_to(staging_root).as_posix()
        for path in staging_root.rglob("*")
        if path.is_file()
    )
    file_names.extend(
        [
            "runtime_metadata/complete_file_checksums.sha256",
            "runtime_metadata/complete_file_list.txt",
        ]
    )
    file_list_path.write_text("\n".join(sorted(file_names)) + "\n", encoding="utf-8")
    manifest_names = sorted(name for name in sorted(file_names) if name != "runtime_metadata/complete_file_checksums.sha256")
    with checksum_path.open("w", encoding="utf-8") as manifest:
        for name in manifest_names:
            path = staging_root / name
            manifest.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n")


def create_complete_bundle_from_staging(staging_root, bundle_path):
    with tarfile.open(bundle_path, "w:gz") as bundle:
        for path in sorted(Path(staging_root).rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(staging_root).as_posix())
    return bundle_path


def mutate_complete_bundle(tmp_path, bundle_path, mutator, name="mutated_complete.tar.gz"):
    staging_root = tmp_path / f"{Path(name).stem}_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    with tarfile.open(bundle_path, "r:gz") as bundle:
        bundle.extractall(staging_root)
    mutator(staging_root)
    refresh_complete_manifest_for_test(staging_root)
    return create_complete_bundle_from_staging(staging_root, tmp_path / name)


def mutate_complete_bundle_text(tmp_path, bundle_path, member_name, updater, name="mutated_complete.tar.gz"):
    def mutate(staging_root):
        path = staging_root / member_name
        path.write_text(updater(path.read_text(encoding="utf-8")), encoding="utf-8")

    return mutate_complete_bundle(tmp_path, bundle_path, mutate, name=name)


def mutate_complete_bundle_csv(tmp_path, bundle_path, member_name, updater, name="mutated_complete.tar.gz"):
    def mutate(staging_root):
        path = staging_root / member_name
        fieldnames, rows = read_csv_dicts(path)
        new_fieldnames, new_rows = updater(fieldnames, rows)
        write_csv(path, new_fieldnames, new_rows)

    return mutate_complete_bundle(tmp_path, bundle_path, mutate, name=name)


def make_fake_sbatch(tmp_path, outputs):
    bin_dir = tmp_path / "fake-sbatch-bin"
    bin_dir.mkdir()
    counter_path = tmp_path / "fake_sbatch_counter.txt"
    calls_path = tmp_path / "fake_sbatch_calls.txt"
    counter_path.write_text("0", encoding="utf-8")
    calls_path.write_text("", encoding="utf-8")
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(
        f"""#!{sys.executable}
import sys
from pathlib import Path

outputs = {list(outputs)!r}
counter_path = Path({str(counter_path)!r})
calls_path = Path({str(calls_path)!r})
count = int(counter_path.read_text(encoding="utf-8"))
counter_path.write_text(str(count + 1), encoding="utf-8")
with calls_path.open("a", encoding="utf-8") as handle:
    handle.write(" ".join(sys.argv[1:]) + "\\n")
if count >= len(outputs):
    print("999999")
else:
    print(outputs[count])
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    return bin_dir, calls_path


def run_submit_real_with_fake_sbatch(tmp_path, outputs):
    source_root = tmp_path / "source_root"
    (source_root / "m3_jobs").mkdir(parents=True)
    (source_root / "scripts").mkdir()
    for script in [ARRAY_SCRIPT, REDUCER_SCRIPT, SUBMIT_SCRIPT]:
        shutil.copy2(script, source_root / "m3_jobs" / script.name)
    shutil.copy2(VALIDATOR, source_root / "scripts" / VALIDATOR.name)
    (source_root / "SOURCE_COMMIT_SHA.txt").write_text(DYNAMIC_SHA + "\n", encoding="utf-8")
    submit_script = source_root / "m3_jobs" / SUBMIT_SCRIPT.name

    formal_root = tmp_path / "formal_packages"
    for task_id, scale, algorithm, formal_task_id, *_ in TASKS:
        create_formal_package(formal_root, scale=scale, algorithm=algorithm, formal_task_id=formal_task_id)
    fake_bin, calls_path = make_fake_sbatch(tmp_path, outputs)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT": DYNAMIC_SHA,
        "EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_PACKAGE_ROOT": str(formal_root),
        "EV_GNN_DIAGNOSTIC_SMOKE_FORMAL_COMPLETE_BUNDLE": str(tmp_path / "unused_complete_bundle.tar.gz"),
        "EV_GNN_DIAGNOSTIC_SMOKE_OUTPUT_ROOT": str(tmp_path / "output"),
        "EV_GNN_DIAGNOSTIC_SMOKE_RUN_ROOT": str(tmp_path / "runs"),
    }
    result = subprocess.run(
        ["bash", str(submit_script)],
        cwd=source_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    return result, calls_path.read_text(encoding="utf-8")


def extract_bash_function(script_text, function_name):
    marker = f"{function_name}() {{"
    start = script_text.index(marker)
    index = start
    depth = 0
    while index < len(script_text):
        character = script_text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return script_text[start:index + 1]
        index += 1
    raise AssertionError(f"could not extract bash function: {function_name}")


def make_delayed_fake_sacct(tmp_path, array_job_id="123456"):
    fake_sacct = tmp_path / "fake_sacct.py"
    counter_path = tmp_path / "sacct_counter.txt"
    counter_path.write_text("0", encoding="utf-8")
    fake_sacct.write_text(
        f"""#!{sys.executable}
from pathlib import Path

counter_path = Path({str(counter_path)!r})
count = int(counter_path.read_text(encoding="utf-8"))
counter_path.write_text(str(count + 1), encoding="utf-8")
if count == 0:
    print("")
else:
    print({sacct_raw_text(array_job_id)!r}, end="")
""",
        encoding="utf-8",
    )
    fake_sacct.chmod(0o755)
    return fake_sacct


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

    assert payload["source_mode"] == "individual_task_package"
    assert Path(payload["package_path"]) == individual_package


def test_preferred_package_resolution_reports_precise_source_mode(tmp_path):
    individual_root = tmp_path / "packages"
    individual_package = create_formal_package(individual_root)
    fallback_bundle = create_tar(tmp_path / "complete.tar.gz", {})

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

    assert json.loads(result.stdout)["source_mode"] == "individual_task_package"


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

    assert payload["source_mode"] == "complete_bundle_nested_task_package"
    assert Path(payload["package_path"]).is_file()


def test_complete_bundle_fallback_reports_precise_source_mode(tmp_path):
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

    assert json.loads(result.stdout)["source_mode"] == "complete_bundle_nested_task_package"


def test_complete_bundle_fallback_rejects_matching_basename_outside_task_packages(tmp_path):
    formal_package = create_formal_package(tmp_path / "source")
    complete_bundle = create_tar(
        tmp_path / "complete.tar.gz",
        {f"complete/unrelated/{formal_package.name}": formal_package.read_bytes()},
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
        check=False,
    )

    assert result.returncode != 0
    assert "task_packages" in result.stderr


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


def test_formal_package_validation_accepts_benign_root_directory_member(tmp_path):
    root_member = tarfile.TarInfo(".")
    root_member.type = tarfile.DIRTYPE
    package_path = create_tar(
        tmp_path / package_basename(),
        {".": root_member, **formal_members()},
    )

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

    assert result.returncode == 0, result.stderr


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


def test_formal_package_validation_rejects_duplicate_normalized_tar_members(tmp_path):
    package_path = create_duplicate_tar(tmp_path / package_basename(), "./safe.txt")

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
    assert "duplicate" in result.stderr.lower()


@pytest.mark.parametrize("member_type", [tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_formal_package_validation_rejects_special_tar_members(tmp_path, member_type):
    special = tarfile.TarInfo("special")
    special.type = member_type
    package_path = create_tar(tmp_path / package_basename(), {"special": special})

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
    assert "unsupported" in result.stderr.lower() or "special" in result.stderr.lower()


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


def test_formal_package_validation_rejects_required_member_absent_from_checksum_manifest(tmp_path):
    package_path = create_formal_package(
        tmp_path,
        manifest_omit=["train/controlled_multiscale_formal_25cp_actiongnn_seed0/model.best_actor"],
    )

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
    assert "coverage" in result.stderr.lower()


def test_formal_package_validation_rejects_invalid_checksum_digest_syntax(tmp_path):
    package_path = create_formal_package(tmp_path)
    with tarfile.open(package_path, "r:gz") as tar:
        members = {
            member.name: tar.extractfile(member).read() if member.isfile() else b""
            for member in tar.getmembers()
            if member.isfile()
        }
    members["runtime_metadata/package_file_checksums.sha256"] = (
        "not-a-sha  config/25cp_actiongnn_seed0_config.yaml\n"
    )
    package_path = create_tar(tmp_path / "bad_digest.tar.gz", members)

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
    assert "digest" in result.stderr.lower()


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


def test_formal_package_validation_returns_validated_config_values(tmp_path):
    package_path = create_formal_package(tmp_path)

    result = run_validator(
        "validate-formal-package",
        "--task-id",
        0,
        "--package",
        package_path,
        "--extract-dir",
        tmp_path / "extract",
    )
    payload = json.loads(result.stdout)

    assert payload["simulation_length"] == 112
    assert payload["v2g_enabled"] is False
    assert payload["number_of_charging_stations"] == 25
    assert payload["number_of_transformers"] == 3


@pytest.mark.parametrize(
    "config_text,error_text",
    [
        (formal_config_text(simulation_length=111), "simulation_length"),
        (formal_config_text(v2g_enabled=True), "v2g_enabled"),
        (formal_config_text(charger_count=24), "charging"),
        (formal_config_text(transformer_count=2), "transformer"),
    ],
)
def test_formal_package_validation_rejects_config_mismatch(tmp_path, config_text, error_text):
    package_path = create_formal_package(tmp_path, config_text=config_text)

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
    assert error_text in result.stderr


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


def test_diagnostic_validation_rejects_missing_signed_fraction_when_active(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(
        diagnostic_dir / "episode_diagnostics.csv",
        delete_columns=["global_zero_action_fraction_active"],
    )

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
    assert "global_zero_action_fraction_active" in result.stderr


@pytest.mark.parametrize("bad_value", ["nan", "inf"])
def test_diagnostic_validation_rejects_non_finite_signed_fraction(tmp_path, bad_value):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(
        diagnostic_dir / "episode_diagnostics.csv",
        updates={"global_positive_action_fraction_active": bad_value},
    )

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
    assert "finite" in result.stderr.lower()


def test_diagnostic_validation_rejects_fraction_outside_unit_interval(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(
        diagnostic_dir / "charger_diagnostics.csv",
        updates={"positive_action_fraction_active": "1.1", "zero_action_fraction_active": "-0.1"},
    )

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
    assert "range" in result.stderr.lower()


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


def test_diagnostic_validation_rejects_missing_inactive_action_count(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "episode_diagnostics.csv", delete_columns=["inactive_nonzero_action_count"])

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
    assert "inactive_nonzero_action_count" in result.stderr


def test_diagnostic_validation_rejects_missing_episode_v2g(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "episode_diagnostics.csv", delete_columns=["v2g_enabled"])

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
    assert "v2g_enabled" in result.stderr


def test_diagnostic_validation_rejects_seed_summary_episode_count_not_one(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "seed_summary_diagnostics.csv", updates={"n_eval_episodes": "2"})

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
    assert "n_eval_episodes" in result.stderr


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


def test_diagnostic_validation_rejects_hierarchical_below_environment_low_actions(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path, scale="25cp", algorithm="hierarchical")
    mutate_csv(
        diagnostic_dir / "episode_diagnostics.csv",
        updates={"active_action_below_environment_low_count": "1"},
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
    assert "below" in result.stderr.lower()


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


def test_diagnostic_validation_rejects_satisfaction_count_zero_with_served_evs(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(
        diagnostic_dir / "charger_diagnostics.csv",
        updates={"user_satisfaction_observation_count": "0"},
    )

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
    assert "satisfaction" in result.stderr.lower()


def test_diagnostic_validation_rejects_per_transformer_energy_mismatch(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(
        diagnostic_dir / "transformer_diagnostics.csv",
        updates={"energy_charged_kwh": "999.0"},
    )

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
    assert "energy" in result.stderr.lower()


def test_diagnostic_validation_rejects_duplicate_charger_ids(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "charger_diagnostics.csv", row_index=1, updates={"charger_id": "0"})

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
    assert "unique" in result.stderr.lower()


def test_diagnostic_validation_rejects_unknown_transformer_reference(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "charger_diagnostics.csv", updates={"transformer_id": "99"})

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
    assert "transformer_id" in result.stderr


def test_diagnostic_validation_rejects_matrix_job_id_mismatch_when_supplied(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        "--matrix-job-id",
        "different",
        check=False,
    )

    assert result.returncode != 0
    assert "matrix_job_id" in result.stderr


@pytest.mark.parametrize("csv_name", ["charger_diagnostics.csv", "transformer_diagnostics.csv"])
@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("scale", "wrong-scale"),
        ("algorithm", "wrong-algorithm"),
        ("training_seed", "1"),
        ("matrix_job_id", "wrong-job"),
        ("episode_index", "1"),
        ("episode_seed", "1"),
    ],
)
def test_diagnostic_validation_rejects_infrastructure_row_identity_corruption(
    tmp_path, csv_name, field, bad_value
):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / csv_name, updates={field: bad_value})

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        "--matrix-job-id",
        "999",
        check=False,
    )

    assert result.returncode != 0
    assert field in result.stderr


@pytest.mark.parametrize("csv_name", ["episode_diagnostics.csv", "seed_summary_diagnostics.csv"])
@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("scale", "wrong-scale"),
        ("algorithm", "wrong-algorithm"),
        ("training_seed", "1"),
        ("matrix_job_id", "wrong-job"),
    ],
)
def test_diagnostic_validation_rejects_episode_and_seed_identity_corruption(
    tmp_path, csv_name, field, bad_value
):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / csv_name, updates={field: bad_value})

    result = run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
        "--matrix-job-id",
        "999",
        check=False,
    )

    assert result.returncode != 0
    assert field in result.stderr


@pytest.mark.parametrize(
    "updates,error_text",
    [
        ({"active_action_decision_count": "nan"}, "active_action_decision_count"),
        ({"active_action_below_environment_low_count": "nan"}, "below_environment_low_count"),
        ({"active_action_below_environment_low_fraction": "inf"}, "below_environment_low_fraction"),
        ({"active_action_below_environment_low_count": "11"}, "active_action_decision_count"),
        (
            {
                "active_action_below_environment_low_count": "1",
                "active_action_below_environment_low_fraction": "0.2",
            },
            "below_environment_low_fraction",
        ),
        ({"active_action_above_environment_high_count": "nan"}, "above_environment_high_count"),
        ({"active_action_above_environment_high_fraction": "inf"}, "above_environment_high_fraction"),
    ],
)
def test_diagnostic_validation_rejects_episode_environment_bound_metric_failures(
    tmp_path, updates, error_text
):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "episode_diagnostics.csv", updates=updates)

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
    assert error_text in result.stderr


def test_diagnostic_validation_accepts_blank_environment_fractions_when_active_count_zero(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(
        diagnostic_dir / "episode_diagnostics.csv",
        updates={
            "active_action_decision_count": "0",
            "active_action_below_environment_low_count": "0",
            "active_action_below_environment_low_fraction": "",
            "active_action_above_environment_high_count": "0",
            "active_action_above_environment_high_fraction": "",
        },
    )
    mutate_csv(
        diagnostic_dir / "seed_summary_diagnostics.csv",
        updates={
            "active_action_decision_count_mean": "0",
            "active_action_below_environment_low_count": "0",
            "active_action_below_environment_low_fraction": "",
            "active_action_above_environment_high_count": "0",
            "active_action_above_environment_high_fraction": "",
        },
    )

    run_validator(
        "validate-diagnostics",
        "--task-id",
        0,
        "--diagnostic-dir",
        diagnostic_dir,
        "--validation-dir",
        tmp_path / "validation",
    )


@pytest.mark.parametrize(
    "updates,error_text",
    [
        ({"active_action_below_environment_low_count": "nan"}, "below_environment_low_count"),
        ({"active_action_below_environment_low_fraction": "inf"}, "below_environment_low_fraction"),
        ({"global_negative_action_fraction_active_mean": "nan"}, "negative_action_fraction"),
        (
            {
                "global_positive_action_fraction_active_mean": "0.7",
                "global_zero_action_fraction_active_mean": "0.1",
                "global_negative_action_fraction_active_mean": "0.1",
            },
            "signed",
        ),
        ({"active_action_decision_count_mean": "9"}, "seed summary"),
        ({"global_positive_action_fraction_active_mean": "0.9"}, "seed summary"),
    ],
)
def test_diagnostic_validation_rejects_seed_summary_action_domain_failures(
    tmp_path, updates, error_text
):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "seed_summary_diagnostics.csv", updates=updates)

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
    assert error_text.lower() in result.stderr.lower()


def test_diagnostic_validation_rejects_hierarchical_below_low_fraction_without_count(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path, scale="25cp", algorithm="hierarchical")
    mutate_csv(
        diagnostic_dir / "episode_diagnostics.csv",
        updates={"active_action_below_environment_low_fraction": "0.1"},
    )
    mutate_csv(
        diagnostic_dir / "seed_summary_diagnostics.csv",
        updates={"active_action_below_environment_low_fraction": "0.1"},
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
    assert "below" in result.stderr.lower()


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
    assert set(rows[0]) == {
        "field",
        "comparison_type",
        "canonical_value",
        "diagnostic_value",
        "absolute_difference",
        "relative_difference",
        "absolute_tolerance",
        "relative_tolerance",
        "pass",
    }


def test_canonical_reconciliation_includes_exact_fields(tmp_path):
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
    exact_fields = {row["field"] for row in rows if row["comparison_type"] == "exact"}

    assert exact_fields == {
        "algorithm",
        "training seed",
        "episode index",
        "episode seed",
        "episode steps",
        "done",
        "total_ev_served",
    }


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


def test_canonical_reconciliation_rejects_fractional_exact_integral_field(tmp_path):
    diagnostic_dir = create_diagnostics(tmp_path)
    mutate_csv(diagnostic_dir / "episode_diagnostics.csv", updates={"total_ev_served": "25.5"})
    canonical_path = tmp_path / "canonical.csv"
    canonical_path.write_text(canonical_csv_text(), encoding="utf-8")

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
    assert any(row["field"] == "total_ev_served" and row["pass"] == "False" for row in rows)


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


def test_task_package_validation_rejects_duplicate_normalized_member(tmp_path):
    task_package = create_duplicate_tar(tmp_path / "task.tar.gz", "./stdout.log")

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()


def test_task_package_validation_rejects_empty_task_validation_json(tmp_path):
    task_package = create_task_package(tmp_path / "task.tar.gz", task_validation={})

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "task_validation" in result.stderr


def test_task_package_validation_rejects_failed_canonical_row(tmp_path):
    task_package = create_task_package(tmp_path / "task.tar.gz", failed_reconciliation=True)

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "canonical" in result.stderr.lower()


@pytest.mark.parametrize(
    "reconciliation_kwargs,error_text",
    [
        ({"missing_field": "episode_reward"}, "missing"),
        ({"duplicate_field": "episode_reward"}, "duplicate"),
        ({"extra_field": "unexpected_metric"}, "unexpected"),
        ({"wrong_type_field": "algorithm"}, "comparison_type"),
        ({"pass_value_override": ("episode_reward", "true")}, "pass"),
    ],
)
def test_task_package_validation_rejects_noncanonical_reconciliation_row_set(
    tmp_path, reconciliation_kwargs, error_text
):
    task_package = create_task_package(
        tmp_path / "task.tar.gz",
        reconciliation_kwargs=reconciliation_kwargs,
    )

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert error_text in result.stderr.lower()


def test_task_package_validation_requires_source_package_resolution_json(tmp_path):
    task_package = create_task_package(
        tmp_path / "task.tar.gz",
        omit_members=["runtime_metadata/source_package_resolution.json"],
    )

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "source_package_resolution.json" in result.stderr


def test_task_package_validation_rejects_invalid_source_package_mode(tmp_path):
    task_package = create_task_package(
        tmp_path / "task.tar.gz",
        source_resolution={
            "source_mode": "unknown",
            "package_path": "/evidence/package.tar.gz",
            "expected_package_name": package_basename(),
        },
    )

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "source_mode" in result.stderr


def test_task_package_validation_preserves_nested_bundle_source_provenance(tmp_path):
    task_package = create_task_package(
        tmp_path / "task.tar.gz",
        source_resolution={
            "source_mode": "complete_bundle_nested_task_package",
            "package_path": "/scratch/job/package.tar.gz",
            "expected_package_name": package_basename(),
            "bundle_member": "complete/task_packages/" + package_basename(),
        },
    )

    run_validator("validate-task-package", "--task-id", 0, "--package", task_package)


def test_task_package_validation_rejects_incomplete_manifest_coverage(tmp_path):
    task_package = create_task_package(
        tmp_path / "task.tar.gz",
        manifest_omit=["diagnostics/episode_diagnostics.csv"],
    )

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "coverage" in result.stderr.lower()


def test_task_package_validation_rejects_file_list_mismatch(tmp_path):
    task_package = create_task_package(
        tmp_path / "task.tar.gz",
        file_list_omit=["diagnostics/episode_diagnostics.csv"],
    )

    result = run_validator(
        "validate-task-package",
        "--task-id",
        0,
        "--package",
        task_package,
        check=False,
    )

    assert result.returncode != 0
    assert "package_file_list" in result.stderr


def test_task_package_validation_accepts_checkpoint_free_package(tmp_path):
    task_package = create_task_package(tmp_path / "task.tar.gz")

    run_validator("validate-task-package", "--task-id", 0, "--package", task_package)


def test_reducer_accepts_exactly_eight_task_packages_and_creates_complete_bundle(tmp_path):
    fixture = create_reducer_fixture(tmp_path)

    result = run_reducer_validator(fixture)
    payload = json.loads(result.stdout)
    bundle_path = Path(payload["bundle_path"])

    assert bundle_path == complete_bundle_path(fixture["output_root"], fixture["array_job_id"])
    assert bundle_path.is_file()
    markers = read_bundle_member(bundle_path, "runtime_metadata/reducer_markers.env")
    assert "TASK_PACKAGE_COUNT=8" in markers
    assert "COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_OK=1" in markers


def test_reducer_writes_portable_final_bundle_checksum_sidecar(tmp_path):
    fixture = create_reducer_fixture(tmp_path)

    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])
    sidecar_path = bundle_path.with_name(bundle_path.name + ".sha256")
    checksum_line = sidecar_path.read_text(encoding="utf-8").strip()

    expected_digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    assert sidecar_path.is_file()
    assert checksum_line == f"{expected_digest}  {bundle_path.name}"


def test_complete_bundle_rejects_adversarial_eight_task0_packages_with_distinct_names(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])
    task0_bytes = read_bundle_member_bytes(
        bundle_path,
        f"task_packages/{task_package_basename(0, fixture['array_job_id'])}",
    )

    def mutate(staging_root):
        task_dir = staging_root / "task_packages"
        for package in task_dir.glob("*.tar.gz"):
            package.unlink()
        for copy_index in range(8):
            (task_dir / f"adversarial_copy{copy_index}_task0.tar.gz").write_bytes(task0_bytes)

    mutated = mutate_complete_bundle(tmp_path, bundle_path, mutate, name="adversarial_task0_complete.tar.gz")

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "task package" in validation.stderr.lower()


def test_complete_bundle_rejects_non_sha_top_level_source_commit(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])
    mutated = mutate_complete_bundle_text(
        tmp_path,
        bundle_path,
        "runtime_metadata/source_commit_sha.txt",
        lambda _text: "not-a-sha\n",
        name="bad_source_commit_complete.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "source commit" in validation.stderr.lower()


def test_complete_bundle_rejects_source_provenance_summary_disagreement(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        rows[0]["source_mode"] = "complete_bundle_nested_task_package"
        rows[0]["bundle_member"] = "unexpected/member.tar.gz"
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/source_provenance_summary.csv",
        updater,
        name="bad_source_provenance_summary.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "source provenance" in validation.stderr.lower()


def test_complete_bundle_rejects_runtime_summary_array_job_mismatch(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        rows[0]["job_id_raw"] = "999999_0"
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/runtime_summary.csv",
        updater,
        name="bad_runtime_summary.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "runtime_summary" in validation.stderr


def test_complete_bundle_rejects_runtime_summary_resource_disagreement(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        rows[0]["max_rss"] = "999K"
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/runtime_summary.csv",
        updater,
        name="bad_runtime_summary_resource.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "runtime_summary" in validation.stderr


def test_complete_bundle_rejects_runtime_summary_fallback_source_disagreement(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        rows[0]["maxrss_source"] = "batch"
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/runtime_summary.csv",
        updater,
        name="bad_runtime_summary_source.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "runtime_summary" in validation.stderr


def test_runtime_summary_has_exact_schema_and_eight_rows(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text("123456", parent_resource_blanks={0}),
    )
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    reader = csv.DictReader(
        io.StringIO(
            read_bundle_member(bundle_path, "summaries/runtime_summary.csv")
        )
    )
    rows = list(reader)

    assert tuple(reader.fieldnames or ()) == RUNTIME_SUMMARY_FIELDS
    assert len(rows) == 8
    assert [row["task_id"] for row in rows] == [str(task_id) for task_id in range(8)]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("job_id_raw", "999999"),
        ("job_id", "123456_7"),
        ("state", "FAILED"),
        ("exit_code", "1:0"),
        ("elapsed_raw", "999"),
        ("alloc_cpus", "99"),
        ("max_rss", "1K"),
        ("total_cpu", "99:00:00"),
        ("maxrss_source", "parent"),
        ("totalcpu_source", "extern"),
    ],
)
def test_complete_bundle_rejects_runtime_summary_field_tampering(
    tmp_path,
    field,
    replacement,
):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text("123456", parent_resource_blanks={0}),
    )
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        assert rows[0]["maxrss_source"] == "batch"
        assert rows[0]["totalcpu_source"] == "batch"
        rows[0][field] = replacement
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/runtime_summary.csv",
        updater,
        name=f"tampered_runtime_summary_{field}.tar.gz",
    )

    validation = run_validator(
        "validate-complete-bundle",
        "--bundle",
        mutated,
        check=False,
    )

    assert validation.returncode != 0
    assert "runtime_summary" in validation.stderr
    assert field in validation.stderr


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_complete_bundle_rejects_noncanonical_runtime_summary_header(
    tmp_path,
    mutation,
):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        fieldnames = list(fieldnames)
        if mutation == "missing":
            fieldnames.remove("job_id")
            for row in rows:
                row.pop("job_id", None)
        elif mutation == "extra":
            fieldnames.append("unexpected")
            for row in rows:
                row["unexpected"] = "value"
        elif mutation == "reordered":
            fieldnames[0], fieldnames[1] = fieldnames[1], fieldnames[0]
        else:
            raise AssertionError(f"unsupported mutation: {mutation}")
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/runtime_summary.csv",
        updater,
        name=f"noncanonical_runtime_summary_header_{mutation}.tar.gz",
    )

    validation = run_validator(
        "validate-complete-bundle",
        "--bundle",
        mutated,
        check=False,
    )

    assert validation.returncode != 0
    assert "runtime_summary.csv header mismatch" in validation.stderr


def test_complete_bundle_rejects_task_inventory_duplicate_identity(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        rows[1]["task_id"] = rows[0]["task_id"]
        rows[1]["package_name"] = rows[0]["package_name"]
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/task_inventory.csv",
        updater,
        name="bad_task_inventory.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "task_inventory" in validation.stderr


def test_complete_bundle_rejects_canonical_summary_row_count_mismatch(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        return fieldnames, rows[:-1]

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/canonical_reconciliation_summary.csv",
        updater,
        name="bad_canonical_summary.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "canonical" in validation.stderr.lower()


def test_complete_bundle_rejects_service_summary_task_count_mismatch(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        return fieldnames, rows[:-1]

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/service_reconciliation_summary.csv",
        updater,
        name="bad_service_summary_count.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "service" in validation.stderr.lower()


def test_complete_bundle_rejects_service_summary_value_mismatch(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])

    def updater(fieldnames, rows):
        if "charger_served_sum" not in fieldnames:
            fieldnames = [*fieldnames, "charger_served_sum"]
        rows[0]["charger_served_sum"] = "24"
        return fieldnames, rows

    mutated = mutate_complete_bundle_csv(
        tmp_path,
        bundle_path,
        "summaries/service_reconciliation_summary.csv",
        updater,
        name="bad_service_summary_value.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert "service" in validation.stderr.lower()


@pytest.mark.parametrize(
    "mutator,error_text",
    [
        (lambda text: "", "marker"),
        (
            lambda text: text.replace("ALL_SCHEMA_VERSION_3=1\n", ""),
            "ALL_SCHEMA_VERSION_3",
        ),
        (
            lambda text: text + "ALL_SCHEMA_VERSION_3=1\n",
            "duplicate",
        ),
        (
            lambda text: text + "MALFORMED_MARKER_LINE\n",
            "malformed",
        ),
        (
            lambda text: text.replace("TASK_PACKAGE_COUNT=8\n", "TASK_PACKAGE_COUNT=7\n"),
            "TASK_PACKAGE_COUNT",
        ),
        (
            lambda text: text.replace(
                "COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_PATH=infrastructure_diagnostic_smoke_complete_evidence_job123456.tar.gz\n",
                "COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_PATH=wrong_bundle.tar.gz\n",
            ),
            "COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_PATH",
        ),
    ],
)
def test_complete_bundle_rejects_missing_duplicate_malformed_or_contradictory_markers(
    tmp_path, mutator, error_text
):
    fixture = create_reducer_fixture(tmp_path)
    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])
    mutated = mutate_complete_bundle_text(
        tmp_path,
        bundle_path,
        "runtime_metadata/reducer_markers.env",
        mutator,
        name=f"bad_marker_{re.sub(r'[^a-z0-9]+', '_', error_text.lower())}.tar.gz",
    )

    validation = run_validator("validate-complete-bundle", "--bundle", mutated, check=False)

    assert validation.returncode != 0
    assert error_text.lower() in validation.stderr.lower()


def test_reducer_ignores_task_packages_from_other_array_jobs(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    other_array_job_id = "654321"
    create_task_package(
        fixture["package_root"] / task_package_basename(0, other_array_job_id),
        task_id=0,
        array_job_id=other_array_job_id,
        source_commit_sha=fixture["source_commit_sha"],
    )

    result = run_reducer_validator(fixture)

    assert result.returncode == 0, result.stderr


def test_reducer_rejects_missing_package_count(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    (fixture["package_root"] / task_package_basename(7, fixture["array_job_id"])).unlink()

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "exactly 8 task packages" in result.stderr


def test_reducer_rejects_missing_expected_package_name(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    expected = fixture["package_root"] / task_package_basename(3, fixture["array_job_id"])
    expected.rename(fixture["package_root"] / "m3_infrastructure_diagnostic_smoke_unexpected_job123456_task3.tar.gz")

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "missing expected task package" in result.stderr


def test_reducer_rejects_duplicate_task_identity_inside_packages(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    task_one_package = fixture["package_root"] / task_package_basename(1, fixture["array_job_id"])
    task_validation = {
        "status": "ok",
        "task_id": 0,
        "scale": "25cp",
        "algorithm": "actiongnn",
        "formal_task_id": 0,
        "episode_seed": 710000,
        "schema_version": "3",
        "episode_rows": 1,
        "seed_summary_rows": 1,
        "charger_rows": 25,
        "transformer_rows": 3,
        "matrix_job_id": fixture["array_job_id"],
    }
    create_task_package(
        task_one_package,
        task_id=1,
        array_job_id=fixture["array_job_id"],
        source_commit_sha=fixture["source_commit_sha"],
        task_validation=task_validation,
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "task_validation" in result.stderr


def test_reducer_rejects_wrong_source_commit(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        source_commit_sha="e" * 40,
    )

    result = run_reducer_validator(
        fixture,
        extra_args=["--source-commit-sha", DYNAMIC_SHA],
    )

    assert result.returncode != 0
    assert "source commit" in result.stderr.lower()


def test_reducer_invokes_task_package_validator(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        package_overrides={0: {"omit_members": ["runtime_metadata/source_package_resolution.json"]}},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "source_package_resolution.json" in result.stderr


def test_reducer_counts_32_internal_diagnostic_csvs(tmp_path):
    fixture = create_reducer_fixture(tmp_path)

    result = run_reducer_validator(fixture)
    payload = json.loads(result.stdout)
    markers = read_bundle_member(Path(payload["bundle_path"]), "runtime_metadata/reducer_markers.env")

    assert "DIAGNOSTIC_CSV_COUNT=32" in markers


def test_reducer_requires_eight_stdout_and_eight_stderr_logs(tmp_path):
    fixture = create_reducer_fixture(tmp_path, missing_stdout_ids={4}, missing_stderr_ids={5})

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "Slurm stdout" in result.stderr or "Slurm stderr" in result.stderr


def test_reducer_rejects_serious_stderr_signature(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        package_overrides={2: {"stderr_log": "RuntimeError: failed\n"}},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "RuntimeError" in result.stderr


def test_reducer_log_scan_allows_benign_oom_substrings(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        slurm_stderr_overrides={0: "the waiting room can bloom without warning\n"},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode == 0, result.stderr


def test_reducer_log_scan_rejects_bounded_oom_signature(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        slurm_stderr_overrides={0: "slurmstepd: error: Detected 1 OOM event\n"},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "OOM" in result.stderr


def test_reducer_rejects_serious_evaluator_stdout_signature(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        package_overrides={0: {"evaluator_stdout": "Traceback (most recent call last):\n"}},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "Traceback" in result.stderr


def test_reducer_records_evaluator_stdout_unknown_warning_inventory(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        package_overrides={0: {"evaluator_stdout": "UserWarning: stdout warning inventory item\n"}},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode == 0, result.stderr
    warning_inventory = read_bundle_member(
        Path(json.loads(result.stdout)["bundle_path"]),
        "summaries/warning_inventory.csv",
    )
    assert "stdout warning inventory item" in warning_inventory


def test_reducer_allows_known_pkg_resources_warning(tmp_path):
    known_warning = (
        "/opt/env/site-packages/ev2gym/utilities/loaders.py:9: "
        "UserWarning: pkg_resources is deprecated as an API\n"
    )
    fixture = create_reducer_fixture(tmp_path, slurm_stderr_overrides={0: known_warning})

    result = run_reducer_validator(fixture)

    assert result.returncode == 0, result.stderr
    warning_inventory = read_bundle_member(
        Path(json.loads(result.stdout)["bundle_path"]),
        "summaries/warning_inventory.csv",
    )
    assert "pkg_resources is deprecated" not in warning_inventory


def test_reducer_records_unknown_warning_inventory(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        slurm_stderr_overrides={0: "UserWarning: unexpected calibration warning\n"},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode == 0, result.stderr
    warning_inventory = read_bundle_member(
        Path(json.loads(result.stdout)["bundle_path"]),
        "summaries/warning_inventory.csv",
    )
    assert "unexpected calibration warning" in warning_inventory


def test_reducer_retries_until_sacct_data_is_available(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    fake_sacct = make_delayed_fake_sacct(tmp_path, fixture["array_job_id"])

    result = run_validator(
        "reduce-bundle",
        "--array-job-id",
        fixture["array_job_id"],
        "--task-package-root",
        fixture["package_root"],
        "--slurm-log-root",
        fixture["slurm_log_root"],
        "--output-root",
        fixture["output_root"],
        "--work-root",
        fixture["work_root"],
        "--source-commit-sha",
        fixture["source_commit_sha"],
        "--reducer-job-id",
        "reducer123",
        "--sacct-command",
        fake_sacct,
        "--sacct-attempts",
        "2",
        "--sacct-delay-seconds",
        "0",
        "--reducer-stdout-log",
        fixture["reducer_stdout"],
        "--reducer-stderr-log",
        fixture["reducer_stderr"],
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_reducer_rejects_missing_task_accounting(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text("123456", missing_task_ids={6}),
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "accounting" in result.stderr.lower()


def test_reducer_rejects_non_completed_state(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text("123456", state_overrides={2: "FAILED"}),
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "COMPLETED" in result.stderr


def test_reducer_rejects_nonzero_exit_code(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text("123456", exit_overrides={2: "1:0"}),
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "ExitCode" in result.stderr


def test_reducer_uses_batch_step_for_missing_parent_maxrss(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text("123456", parent_resource_blanks={0}),
    )

    result = run_reducer_validator(fixture)
    runtime_summary = read_bundle_member(
        Path(json.loads(result.stdout)["bundle_path"]),
        "summaries/runtime_summary.csv",
    )

    assert "123463,123456_0,COMPLETED,0:0,12,4,4096K,00:00:12,batch,batch" in runtime_summary


def test_reducer_rejects_failed_batch_step_when_used_for_resource_fallback(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text(
            "123456",
            parent_resource_blanks={0},
            batch_state_overrides={0: "FAILED"},
        ),
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert ".batch" in result.stderr or "batch" in result.stderr.lower()


def test_reducer_rejects_nonzero_batch_exit_when_used_for_resource_fallback(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text(
            "123456",
            parent_resource_blanks={0},
            batch_exit_overrides={0: "1:0"},
        ),
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert ".batch" in result.stderr or "batch" in result.stderr.lower()


def test_reducer_rejects_contradictory_accounting_rows(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        sacct_text=sacct_raw_text("123456", duplicate_parent_rows={0}),
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower() or "contradictory" in result.stderr.lower()



def test_strict_sacct_exact_failed_job_rows_reach_state_gate():
    validator = load_validator_module()

    with pytest.raises(
        validator.ValidationError,
        match=r"must be COMPLETED, got FAILED",
    ):
        validator.parse_sacct_raw(FAILED_JOB_58579309_SACCT, "58579309")


def test_strict_sacct_task7_numeric_raw_id_collision_is_unambiguous():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(completed_m3_sacct_text(), "58579309")
    by_task = {row["task_id"]: row for row in rows}

    assert sorted(by_task) == list(range(8))
    assert by_task[7]["job_id_raw"] == "58579309"
    assert by_task[7]["job_id"] == "58579309_7"


def test_strict_sacct_rejects_legacy_seven_column_rows():
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=r"exactly 9 fields"):
        validator.parse_sacct_raw(legacy_sacct_raw_text("58579309"), "58579309")


def test_strict_sacct_rejects_header_rows():
    validator = load_validator_module()
    raw = "|".join(M3_SACCT_FIELDS) + "\n" + completed_m3_sacct_text()

    with pytest.raises(validator.ValidationError, match=r"header rows are not allowed"):
        validator.parse_sacct_raw(raw, "58579309")


@pytest.mark.parametrize(
    "raw",
    [
        "1|58579309_0|name|COMPLETED|0:0|12|4|2048K\n",
        "1|58579309_0|name|COMPLETED|0:0|12|4|2048K|00:00:10|extra\n",
    ],
)
def test_strict_sacct_requires_exactly_nine_cells(raw):
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=r"exactly 9 fields"):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_rejects_wrong_array_identity():
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace("58579309_0", "999999_0", 1)

    with pytest.raises(validator.ValidationError, match=r"unexpected accounting JobID"):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_rejects_out_of_range_task_identity():
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace("58579309_0", "58579309_8", 1)

    with pytest.raises(validator.ValidationError, match=r"unexpected accounting task ID"):
        validator.parse_sacct_raw(raw, "58579309")


@pytest.mark.parametrize("suffix", ["0", "interactive"])
def test_strict_sacct_rejects_unknown_slurm_steps(suffix):
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace(
        "58579309_0|evgnn_infra_diag_smoke",
        f"58579309_0.{suffix}|evgnn_infra_diag_smoke",
        1,
    )

    with pytest.raises(validator.ValidationError, match=r"unexpected accounting JobID"):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_rejects_jobidraw_jobid_suffix_contradiction():
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace(
        "58579316|58579309_0|",
        "58579316.batch|58579309_0|",
        1,
    )

    with pytest.raises(validator.ValidationError, match=r"JobIDRaw suffix"):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_rejects_parent_batch_raw_base_mismatch():
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace(
        "58579316.batch|58579309_0.batch|",
        "999999.batch|58579309_0.batch|",
        1,
    )

    with pytest.raises(validator.ValidationError, match=r"raw ID base mismatch"):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_rejects_parent_extern_raw_base_mismatch():
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace(
        "58579316.extern|58579309_0.extern|",
        "999999.extern|58579309_0.extern|",
        1,
    )

    with pytest.raises(validator.ValidationError, match=r"raw ID base mismatch"):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_rejects_duplicate_numeric_parent_raw_id_across_tasks():
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace(
        "58579322|58579309_1|",
        "58579316|58579309_1|",
        1,
    )

    with pytest.raises(
        validator.ValidationError,
        match=r"duplicate numeric parent JobIDRaw",
    ):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_rejects_nonnumeric_parent_jobidraw():
    validator = load_validator_module()
    raw = completed_m3_sacct_text().replace(
        "58579316|58579309_0|",
        "not_numeric|58579309_0|",
        1,
    )

    with pytest.raises(
        validator.ValidationError,
        match=r"parent JobIDRaw must contain digits only",
    ):
        validator.parse_sacct_raw(raw, "58579309")


def test_strict_sacct_accepts_explicit_array_aggregate_without_confusing_task7():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(
        completed_m3_sacct_text(include_aggregate=True),
        "58579309",
    )
    by_task = {row["task_id"]: row for row in rows}

    assert len(rows) == 8
    assert by_task[7]["job_id"] == "58579309_7"


def test_strict_sacct_rejects_missing_parent_only_for_missing_task():
    validator = load_validator_module()

    with pytest.raises(
        validator.ValidationError,
        match=r"required accounting unavailable for task\(s\): \[3\]",
    ):
        validator.parse_sacct_raw(
            sacct_raw_text("123456", missing_task_ids={3}),
            "123456",
        )


def test_strict_sacct_rejects_duplicate_parent():
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=r"duplicate parent"):
        validator.parse_sacct_raw(
            sacct_raw_text("123456", duplicate_parent_rows={2}),
            "123456",
        )


def test_strict_sacct_rejects_duplicate_batch():
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=r"duplicate batch"):
        validator.parse_sacct_raw(
            sacct_raw_text("123456", duplicate_batch_rows={2}),
            "123456",
        )


def test_strict_sacct_rejects_duplicate_extern():
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=r"duplicate extern"):
        validator.parse_sacct_raw(
            sacct_raw_text("123456", duplicate_extern_rows={2}),
            "123456",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"state_overrides": {2: "FAILED"}}, r"123456_2 must be COMPLETED, got FAILED"),
        ({"exit_overrides": {2: "1:0"}}, r"123456_2 ExitCode must be 0:0, got 1:0"),
        (
            {"batch_state_overrides": {2: "FAILED"}},
            r"123456_2\.batch must be COMPLETED, got FAILED",
        ),
        (
            {"batch_exit_overrides": {2: "1:0"}},
            r"123456_2\.batch ExitCode must be 0:0, got 1:0",
        ),
        (
            {"extern_state_overrides": {2: "FAILED"}},
            r"123456_2\.extern must be COMPLETED, got FAILED",
        ),
        (
            {"extern_exit_overrides": {2: "1:0"}},
            r"123456_2\.extern ExitCode must be 0:0, got 1:0",
        ),
    ],
)
def test_strict_sacct_validates_every_present_row_state_and_exit(kwargs, message):
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=message):
        validator.parse_sacct_raw(sacct_raw_text("123456", **kwargs), "123456")


def test_strict_sacct_accepts_missing_batch_when_parent_resources_are_complete():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(
        sacct_raw_text("123456", missing_batch_ids={0}),
        "123456",
    )

    assert rows[0]["maxrss_source"] == "parent"
    assert rows[0]["totalcpu_source"] == "parent"


def test_strict_sacct_accepts_missing_extern():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(
        sacct_raw_text("123456", missing_extern_ids={0}),
        "123456",
    )

    assert len(rows) == 8


def test_strict_sacct_uses_parent_resources_when_available():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(sacct_raw_text("123456"), "123456")
    row = {item["task_id"]: item for item in rows}[0]

    assert row["max_rss"] == "2048K"
    assert row["total_cpu"] == "00:00:10"
    assert row["maxrss_source"] == "parent"
    assert row["totalcpu_source"] == "parent"


def test_strict_sacct_falls_back_only_for_missing_parent_maxrss():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(
        sacct_raw_text("123456", parent_maxrss_blanks={0}),
        "123456",
    )
    row = {item["task_id"]: item for item in rows}[0]

    assert row["max_rss"] == "4096K"
    assert row["total_cpu"] == "00:00:10"
    assert row["maxrss_source"] == "batch"
    assert row["totalcpu_source"] == "parent"


def test_strict_sacct_falls_back_only_for_missing_parent_totalcpu():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(
        sacct_raw_text("123456", parent_totalcpu_blanks={0}),
        "123456",
    )
    row = {item["task_id"]: item for item in rows}[0]

    assert row["max_rss"] == "2048K"
    assert row["total_cpu"] == "00:00:12"
    assert row["maxrss_source"] == "parent"
    assert row["totalcpu_source"] == "batch"


def test_strict_sacct_falls_back_for_both_missing_parent_resources():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(
        sacct_raw_text("123456", parent_resource_blanks={0}),
        "123456",
    )
    row = {item["task_id"]: item for item in rows}[0]

    assert row["max_rss"] == "4096K"
    assert row["total_cpu"] == "00:00:12"
    assert row["maxrss_source"] == "batch"
    assert row["totalcpu_source"] == "batch"


def test_strict_sacct_rejects_missing_batch_when_fallback_is_required():
    validator = load_validator_module()

    with pytest.raises(
        validator.ValidationError,
        match=r"required \.batch accounting unavailable for Slurm task 123456_0",
    ):
        validator.parse_sacct_raw(
            sacct_raw_text(
                "123456",
                parent_resource_blanks={0},
                missing_batch_ids={0},
            ),
            "123456",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"parent_maxrss_blanks": {0}, "batch_maxrss_blanks": {0}},
            r"MaxRSS unavailable for Slurm task 123456_0",
        ),
        (
            {"parent_totalcpu_blanks": {0}, "batch_totalcpu_blanks": {0}},
            r"TotalCPU unavailable for Slurm task 123456_0",
        ),
    ],
)
def test_strict_sacct_rejects_unavailable_required_batch_resource(kwargs, message):
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=message):
        validator.parse_sacct_raw(sacct_raw_text("123456", **kwargs), "123456")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parent_elapsed_blanks": {0}}, r"ElapsedRaw unavailable for Slurm task 123456_0"),
        ({"parent_alloc_blanks": {0}}, r"AllocCPUS unavailable for Slurm task 123456_0"),
    ],
)
def test_strict_sacct_requires_parent_elapsed_and_alloc_cpus(kwargs, message):
    validator = load_validator_module()

    with pytest.raises(validator.ValidationError, match=message):
        validator.parse_sacct_raw(sacct_raw_text("123456", **kwargs), "123456")


def test_strict_sacct_never_uses_extern_resources():
    validator = load_validator_module()

    rows = validator.parse_sacct_raw(
        sacct_raw_text(
            "123456",
            extern_maxrss_overrides={0: "999999K"},
            extern_totalcpu_overrides={0: "99:59:59"},
        ),
        "123456",
    )
    row = {item["task_id"]: item for item in rows}[0]

    assert row["max_rss"] == "2048K"
    assert row["total_cpu"] == "00:00:10"
    assert row["maxrss_source"] == "parent"
    assert row["totalcpu_source"] == "parent"


def test_strict_sacct_collect_command_is_scoped_to_explicit_array_id(tmp_path):
    validator = load_validator_module()
    capture_path = tmp_path / "argv.json"
    fake_sacct = tmp_path / "fake_sacct.py"
    expected_raw = completed_m3_sacct_text(array_job_id="123456")
    fake_sacct.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        f"pathlib.Path({str(capture_path)!r}).write_text("
        "json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        f"print({expected_raw!r}, end='')\n",
        encoding="utf-8",
    )
    fake_sacct.chmod(0o755)
    args = argparse.Namespace(
        sacct_raw_file=None,
        sacct_command=str(fake_sacct),
        sacct_attempts=1,
        sacct_delay_seconds=0,
        array_job_id="123456",
    )

    observed_raw = validator.collect_sacct_raw(args)

    assert observed_raw == expected_raw
    assert json.loads(capture_path.read_text(encoding="utf-8")) == [
        "-j",
        "123456",
        "--parsable2",
        "--noheader",
        "--format=JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU",
    ]


def test_strict_array_job_id_accepts_numeric_value():
    validator = load_validator_module()

    assert validator.validate_array_job_id("58579309") == "58579309"


@pytest.mark.parametrize(
    "value",
    ["", "123_0", "latest", "123;cluster", "../123", " 123 "],
)
def test_strict_array_job_id_rejects_non_numeric_values(value):
    validator = load_validator_module()

    with pytest.raises(
        validator.ValidationError,
        match=r"array job ID must contain digits only",
    ):
        validator.validate_array_job_id(value)


def test_strict_array_job_id_fails_before_output_creation():
    validator = load_validator_module()
    output_root = Path("must_not_be_created_for_invalid_array_id")
    if output_root.exists():
        shutil.rmtree(output_root)
    args = argparse.Namespace(
        array_job_id="latest",
        source_commit_sha=DYNAMIC_SHA,
        output_root=str(output_root),
        task_package_root="unused",
        slurm_log_root="unused",
        work_root="unused",
        reducer_job_id="unused",
        sacct_raw_file=None,
        sacct_command="sacct",
        sacct_attempts=1,
        sacct_delay_seconds=0,
        reducer_stdout_log="unused",
        reducer_stderr_log="unused",
    )

    try:
        with pytest.raises(
            validator.ValidationError,
            match=r"array job ID must contain digits only",
        ):
            validator.reduce_bundle(args)
        assert not output_root.exists()
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)

def test_reducer_refuses_to_overwrite_final_bundle(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    final_bundle = complete_bundle_path(fixture["output_root"], fixture["array_job_id"])
    final_bundle.write_text("existing", encoding="utf-8")

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "already exists" in result.stderr


def test_reducer_refuses_to_overwrite_final_checksum_sidecar(tmp_path):
    fixture = create_reducer_fixture(tmp_path)
    checksum_path = complete_bundle_checksum_path(fixture["output_root"], fixture["array_job_id"])
    checksum_path.write_text("old", encoding="utf-8")

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "checksum" in result.stderr.lower() or "already exists" in result.stderr.lower()


def test_reducer_publication_is_atomic_on_final_validation_failure(tmp_path, monkeypatch):
    fixture = create_reducer_fixture(tmp_path)
    validator_module = load_validator_module()

    def fail_validation(_bundle_path):
        raise validator_module.ValidationError("injected final validation failure")

    monkeypatch.setattr(validator_module, "validate_complete_bundle_file", fail_validation)
    args = argparse.Namespace(
        array_job_id=fixture["array_job_id"],
        task_package_root=fixture["package_root"],
        slurm_log_root=fixture["slurm_log_root"],
        output_root=fixture["output_root"],
        work_root=fixture["work_root"],
        source_commit_sha=fixture["source_commit_sha"],
        reducer_job_id="reducer123",
        sacct_raw_file=fixture["sacct_path"],
        sacct_command="sacct",
        sacct_attempts=1,
        sacct_delay_seconds=0,
        reducer_stdout_log=fixture["reducer_stdout"],
        reducer_stderr_log=fixture["reducer_stderr"],
    )

    with pytest.raises(validator_module.ValidationError, match="injected"):
        validator_module.reduce_bundle(args)

    assert not complete_bundle_path(fixture["output_root"], fixture["array_job_id"]).exists()
    assert not complete_bundle_checksum_path(fixture["output_root"], fixture["array_job_id"]).exists()


def test_reducer_publication_cleans_final_artifacts_on_sidecar_failure(tmp_path, monkeypatch):
    fixture = create_reducer_fixture(tmp_path)
    validator_module = load_validator_module()

    def fail_after_sidecar_created(archive_path, sidecar_path):
        Path(sidecar_path).write_text("0" * 64 + f"  {Path(archive_path).name}\n", encoding="utf-8")
        raise validator_module.ValidationError("injected sidecar failure")

    monkeypatch.setattr(validator_module, "write_validated_sha256_sidecar", fail_after_sidecar_created)
    args = argparse.Namespace(
        array_job_id=fixture["array_job_id"],
        task_package_root=fixture["package_root"],
        slurm_log_root=fixture["slurm_log_root"],
        output_root=fixture["output_root"],
        work_root=fixture["work_root"],
        source_commit_sha=fixture["source_commit_sha"],
        reducer_job_id="reducer123",
        sacct_raw_file=fixture["sacct_path"],
        sacct_command="sacct",
        sacct_attempts=1,
        sacct_delay_seconds=0,
        reducer_stdout_log=fixture["reducer_stdout"],
        reducer_stderr_log=fixture["reducer_stderr"],
    )

    with pytest.raises(validator_module.ValidationError, match="sidecar"):
        validator_module.reduce_bundle(args)

    assert not complete_bundle_path(fixture["output_root"], fixture["array_job_id"]).exists()
    assert not complete_bundle_checksum_path(fixture["output_root"], fixture["array_job_id"]).exists()


def test_reducer_rejects_checkpoint_leak_from_task_package(tmp_path):
    fixture = create_reducer_fixture(
        tmp_path,
        package_overrides={0: {"include_checkpoint": True}},
    )

    result = run_reducer_validator(fixture)

    assert result.returncode != 0
    assert "checkpoint" in result.stderr.lower()


def test_reducer_validates_final_manifest_and_file_list(tmp_path):
    fixture = create_reducer_fixture(tmp_path)

    result = run_reducer_validator(fixture)
    bundle_path = Path(json.loads(result.stdout)["bundle_path"])
    validation = run_validator("validate-complete-bundle", "--bundle", bundle_path)

    assert json.loads(validation.stdout)["status"] == "ok"


def test_reducer_expands_service_reconciliation_summary_from_packaged_csvs(tmp_path):
    fixture = create_reducer_fixture(tmp_path)

    result = run_reducer_validator(fixture)
    service_summary = read_bundle_member(
        Path(json.loads(result.stdout)["bundle_path"]),
        "summaries/service_reconciliation_summary.csv",
    )
    rows = list(csv.DictReader(io.StringIO(service_summary)))

    assert len(rows) == 8
    task0 = next(row for row in rows if row["task_id"] == "0")
    assert task0["episode_total_ev_served"] == "25"
    assert task0["charger_served_sum"] == "25"
    assert task0["transformer_served_sum"] == "25"
    assert task0["charger_satisfaction_count_sum"] == "25"
    assert task0["transformer_satisfaction_count_sum"] == "25"
    assert task0["episode_total_energy_charged"] == "25.0"
    assert task0["charger_energy_charged_sum"] == "25.0"
    assert task0["transformer_energy_charged_sum"] == "25.0"
    assert task0["episode_total_energy_discharged"] == "0.0"
    assert task0["charger_energy_discharged_sum"] == "0.0"
    assert task0["transformer_energy_discharged_sum"] == "0.0"
    assert task0["served_reconciliation_pass"] == "True"
    assert task0["satisfaction_reconciliation_pass"] == "True"
    assert task0["charged_energy_reconciliation_pass"] == "True"
    assert task0["discharged_energy_reconciliation_pass"] == "True"


def test_reducer_script_dry_run(tmp_path):
    env = {
        **os.environ,
        "EV_GNN_DIAGNOSTIC_SMOKE_REDUCER_DRY_RUN": "1",
        "EV_GNN_DIAGNOSTIC_SMOKE_ARRAY_JOB_ID": "123456",
        "SLURM_JOB_ID": "777",
    }
    result = subprocess.run(
        ["bash", str(REDUCER_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "DRY_RUN_NO_REDUCTION_OR_PACKAGING" in result.stdout
    assert "infrastructure_diagnostic_smoke_complete_evidence_job123456.tar.gz" in result.stdout
    assert "reducer_job777" in result.stdout


def test_submit_helper_dry_run_prints_mapping_and_sbatch_commands():
    env = {
        **os.environ,
        "EV_GNN_DIAGNOSTIC_SMOKE_SUBMIT_DRY_RUN": "1",
        "EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT": DYNAMIC_SHA,
    }
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "task_id=0 scale=25cp algorithm=actiongnn formal_task_id=0 episode_seed=710000" in result.stdout
    assert "SBATCH_ARRAY_COMMAND=sbatch --parsable" in result.stdout
    assert "SBATCH_REDUCER_COMMAND=sbatch --parsable" in result.stdout


def test_submit_helper_uses_afterok_dependency_in_dry_run():
    env = {
        **os.environ,
        "EV_GNN_DIAGNOSTIC_SMOKE_SUBMIT_DRY_RUN": "1",
        "EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT": DYNAMIC_SHA,
    }
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "--dependency=afterok:<array_job_id>" in result.stdout


@pytest.mark.parametrize(
    "array_output,reducer_output",
    [
        ("123456", "789012"),
        ("123456;cluster-name", "789012;cluster-name"),
    ],
)
def test_submit_helper_normalises_sbatch_parsable_output(tmp_path, array_output, reducer_output):
    result, calls = run_submit_real_with_fake_sbatch(tmp_path, [array_output, reducer_output])

    assert result.returncode == 0, result.stderr
    assert "array_job_id=123456" in result.stdout
    assert "reducer_job_id=789012" in result.stdout
    assert "--dependency=afterok:123456" in calls
    assert "job123456.tar.gz" in result.stdout
    assert "job123456.tar.gz.sha256" in result.stdout


def test_submit_helper_rejects_non_numeric_sbatch_job_id(tmp_path):
    result, _calls = run_submit_real_with_fake_sbatch(tmp_path, ["abc;cluster-name"])

    assert result.returncode != 0
    assert "sbatch" in result.stderr.lower()


def test_task8_submit_helper_dry_run_prints_exact_scoped_sacct_command():
    env = {
        **os.environ,
        "EV_GNN_DIAGNOSTIC_SMOKE_SUBMIT_DRY_RUN": "1",
        "EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT": DYNAMIC_SHA,
    }
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    expected = (
        "SACCT_COMMAND=sacct -j <array_job_id> --parsable2 --noheader "
        f"--format={','.join(M3_SACCT_FIELDS)}"
    )
    sacct_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("SACCT_COMMAND=")
    ]

    assert result.returncode == 0, result.stderr
    assert sacct_lines == [expected]


def test_task8_submit_helper_real_prints_exact_current_array_sacct_command(tmp_path):
    result, _calls = run_submit_real_with_fake_sbatch(
        tmp_path,
        ["123456;cluster-name", "789012;cluster-name"],
    )

    expected = (
        "SACCT_COMMAND=sacct -j 123456 --parsable2 --noheader "
        f"--format={','.join(M3_SACCT_FIELDS)}"
    )
    sacct_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("SACCT_COMMAND=")
    ]

    assert result.returncode == 0, result.stderr
    assert sacct_lines == [expected]


def test_task8_protocol_documents_explicit_array_id_and_nine_column_accounting():
    text = (
        PROJECT_ROOT / "docs/infrastructure_diagnostic_smoke_protocol.md"
    ).read_text(encoding="utf-8")
    lower = text.lower()

    assert ",".join(M3_SACCT_FIELDS) in text
    assert "explicit array" in lower
    assert "same array job id" in lower
    assert "latest job" in lower
    assert "do not use" in lower or "must not use" in lower


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


def test_source_bundle_dry_run_defaults_output_outside_repository():
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

    expected_root = Path.home() / "Downloads" / "EVGNN_Formal_Evidence"
    assert result.returncode == 0
    assert f"SOURCE_ARCHIVE={expected_root}/" in result.stdout


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


def test_source_bundle_allowlist_includes_batch_b_runtime_files():
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
    assert "m3_jobs/20_infrastructure_diagnostic_smoke_reduce_bundle.slurm" in result.stdout
    assert "m3_jobs/submit_infrastructure_diagnostic_smoke_workflow.sh" in result.stdout
    assert "docs/infrastructure_diagnostic_smoke_protocol.md" in result.stdout


def test_source_bundle_real_mode_writes_basename_checksum_and_dynamic_head(tmp_path):
    result = run_source_bundle(tmp_path)

    assert result.returncode == 0, result.stderr
    archive_match = re.search(r"^SOURCE_ARCHIVE=(.+)$", result.stdout, re.MULTILINE)
    checksum_match = re.search(r"^SOURCE_ARCHIVE_SHA256=(.+)$", result.stdout, re.MULTILINE)
    assert archive_match and checksum_match
    archive_path = Path(archive_match.group(1))
    checksum_path = Path(checksum_match.group(1))
    checksum_line = checksum_path.read_text(encoding="utf-8").strip()

    assert re.fullmatch(r"[0-9a-f]{64}  " + re.escape(archive_path.name), checksum_line)
    assert "/Users/" not in checksum_line
    assert f"sha256sum -c {archive_path.name}.sha256" in result.stdout
    assert "test ! -e /scratch2/fr57/cche0357/EV-GNN_sources/" in result.stdout
    assert DYNAMIC_SHA in archive_path.name
    with tarfile.open(archive_path, "r:gz") as archive:
        source_member = f"EV-GNN-infrastructure-diagnostic-smoke-{DYNAMIC_SHA}/SOURCE_COMMIT_SHA.txt"
        source_sha = archive.extractfile(source_member).read().decode("utf-8").strip()
    assert source_sha == DYNAMIC_SHA


def test_source_bundle_real_mode_refuses_existing_archive_outputs(tmp_path):
    first = run_source_bundle(tmp_path)
    assert first.returncode == 0, first.stderr

    second = run_source_bundle(tmp_path)

    assert second.returncode != 0
    assert "already exists" in second.stderr


def test_source_bundle_real_mode_rejects_explicit_head_mismatch(tmp_path):
    result = run_source_bundle(
        tmp_path,
        extra_env={"EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_EXPECTED_HEAD_SHA": BASE_SHA},
    )

    assert result.returncode != 0
    assert "does not match" in result.stderr


@pytest.mark.parametrize(
    ("task_id", "expected_scale", "expected_algorithm"),
    [(task_id, scale, algorithm) for task_id, scale, algorithm, *_ in TASKS],
)
def test_array_script_dry_run_maps_all_tasks(
    task_id,
    expected_scale,
    expected_algorithm,
):
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
    assert f"scale={expected_scale}" in result.stdout
    assert f"algorithm={expected_algorithm}" in result.stdout

    command_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("DIAGNOSTIC_COMMAND=")
    ]
    assert len(command_lines) == 1
    command = command_lines[0]
    assert command.count("--algorithm ") == 1
    assert f"--algorithm {expected_algorithm}" in command
    assert command.count("--scale ") == 1
    assert f"--scale {expected_scale}" in command
    assert command.count("--config ") == 1
    assert "--config <extracted_formal_config>" in command
    assert "DRY_RUN_NO_EVALUATION_OR_PACKAGING" in result.stdout


def test_tracked_evaluator_runtime_invocation_supplies_explicit_scale_and_config():
    script_text = ARRAY_SCRIPT.read_text(encoding="utf-8")
    command_blocks = re.findall(
        r"(?ms)^DIAGNOSTIC_COMMAND=\(\n(.*?)^\)\n",
        script_text,
    )

    assert len(command_blocks) == 1
    command_block = command_blocks[0]
    assert command_block.count(
        "python evaluate_td3_gnn_infrastructure_diagnostics.py"
    ) == 1
    assert command_block.count('--algorithm "${ALGORITHM}"') == 1
    assert command_block.count('--scale "${SCALE}"') == 1
    assert command_block.count('--config "${CONFIG_DIR}/formal_config.yaml"') == 1
    assert command_block.index('--algorithm "${ALGORITHM}"') < command_block.index(
        '--scale "${SCALE}"'
    )
    assert command_block.index('--scale "${SCALE}"') < command_block.index(
        '--config "${CONFIG_DIR}/formal_config.yaml"'
    )


def test_array_script_requires_explicit_expected_source_commit_before_real_work(tmp_path):
    result = run_array_real_guard(
        tmp_path,
        extra_env={"EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT": ""},
    )

    assert result.returncode != 0
    assert "EV_GNN_DIAGNOSTIC_SMOKE_EXPECTED_SOURCE_COMMIT" in result.stderr


def test_array_script_rejects_stale_task_directory_before_real_work(tmp_path):
    stale_task_dir = tmp_path / "runs" / "jobmanual" / "25cp" / "actiongnn" / "seed0"
    stale_task_dir.mkdir(parents=True)

    result = run_array_real_guard(tmp_path)

    assert result.returncode != 0
    assert "task directory already exists" in result.stderr


def test_array_script_rejects_existing_output_package_before_real_work(tmp_path):
    output_root = tmp_path / "out"
    output_root.mkdir()
    package_path = output_root / "m3_infrastructure_diagnostic_smoke_25cp_actiongnn_seed0_jobmanual_task0.tar.gz"
    package_path.write_text("old", encoding="utf-8")

    result = run_array_real_guard(tmp_path)

    assert result.returncode != 0
    assert "package already exists" in result.stderr


def test_array_task_package_publication_is_atomic_on_validation_failure(tmp_path):
    package_task_function = extract_bash_function(
        ARRAY_SCRIPT.read_text(encoding="utf-8"),
        "package_task",
    )
    output_root = tmp_path / "out"
    staging_root = tmp_path / "package_staging"
    repo_root = tmp_path / "repo"
    output_root.mkdir()
    for member in [
        "stdout.log",
        "stderr.log",
        "diagnostics/episode_diagnostics.csv",
        "canonical/complete_eval30.csv",
        "config/formal_config.yaml",
        "validation/task_validation.json",
        "runtime_metadata/package_file_list.txt",
    ]:
        path = staging_root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    validator = repo_root / "scripts/validate_infrastructure_diagnostic_smoke.py"
    validator.parent.mkdir(parents=True)
    validator.write_text("raise SystemExit(1)\n", encoding="utf-8")
    final_package = output_root / "m3_infrastructure_diagnostic_smoke_25cp_actiongnn_seed0_jobmanual_task0.tar.gz"
    runner = tmp_path / "run_package_task.sh"
    runner.write_text(
        f"""#!/bin/bash
set -euo pipefail
OUTPUT_ROOT={str(output_root)!r}
PACKAGE_STAGING_DIR={str(staging_root)!r}
PACKAGE_PATH={str(final_package)!r}
PACKAGE_BASENAME="$(basename "${{PACKAGE_PATH}}")"
REPO_ROOT={str(repo_root)!r}
TASK_ID=0
prepare_package_staging() {{ :; }}
generate_package_manifest() {{ :; }}
{package_task_function}
if package_task; then
  echo "package_task unexpectedly succeeded" >&2
  exit 10
fi
if [[ -e "${{PACKAGE_PATH}}" ]]; then
  echo "final package appeared after validation failure" >&2
  exit 11
fi
if compgen -G "${{OUTPUT_ROOT}}/.${{PACKAGE_BASENAME}}.tmp.*" >/dev/null; then
  echo "temporary package leaked after validation failure" >&2
  exit 12
fi
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)

    result = subprocess.run(
        ["bash", str(runner)],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script_path", [ARRAY_SCRIPT, REDUCER_SCRIPT, SUBMIT_SCRIPT])
def test_m3_runtime_scripts_contain_no_git_or_training_command(script_path):
    script_text = script_path.read_text(encoding="utf-8")

    assert "git " not in script_text
    assert "train_td3_gnn.py" not in script_text
    assert "train_td3_gnn" not in script_text


def test_bash_syntax_for_batch_a_and_b_scripts():
    result = subprocess.run(
        [
            "bash",
            "-n",
            str(SOURCE_BUNDLE_SCRIPT),
            str(ARRAY_SCRIPT),
            str(REDUCER_SCRIPT),
            str(SUBMIT_SCRIPT),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
