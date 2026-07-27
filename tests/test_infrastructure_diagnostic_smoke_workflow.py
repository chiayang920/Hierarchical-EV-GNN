import csv
import hashlib
import io
import json
import os
import re
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
        "active_action_below_environment_low_count": "0",
        "active_action_below_environment_low_fraction": "0.0",
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
        "active_action_below_environment_low_count": "0",
        "active_action_below_environment_low_fraction": "0.0",
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
                "matrix_job_id": "999",
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


def reconciliation_csv_text(failed=False):
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
        ("algorithm", "actiongnn"),
        ("training seed", "0"),
        ("episode index", "0"),
        ("episode seed", "710000"),
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
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def create_task_package(
    path,
    include_checkpoint=False,
    task_validation=None,
    failed_reconciliation=False,
    manifest_omit=(),
    file_list_omit=(),
):
    fixture_root = path.parent / f"{path.stem}_fixture"
    diagnostic_dir = create_diagnostics(fixture_root)
    task_validation = (
        {
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
            "matrix_job_id": "999",
        }
        if task_validation is None
        else task_validation
    )
    members = {
        "stdout.log": "",
        "stderr.log": "",
        "diagnostics/episode_diagnostics.csv": (diagnostic_dir / "episode_diagnostics.csv").read_text(encoding="utf-8"),
        "diagnostics/seed_summary_diagnostics.csv": (diagnostic_dir / "seed_summary_diagnostics.csv").read_text(encoding="utf-8"),
        "diagnostics/transformer_diagnostics.csv": (diagnostic_dir / "transformer_diagnostics.csv").read_text(encoding="utf-8"),
        "diagnostics/charger_diagnostics.csv": (diagnostic_dir / "charger_diagnostics.csv").read_text(encoding="utf-8"),
        "canonical/complete_eval30.csv": canonical_csv_text(),
        "canonical/canonical_episode0.csv": canonical_episode0_csv_text(),
        "config/formal_config.yaml": formal_config_text(),
        "validation/task_validation.json": json.dumps(task_validation, sort_keys=True) + "\n",
        "validation/canonical_reconciliation.csv": reconciliation_csv_text(failed=failed_reconciliation),
        "runtime_metadata/source_commit_sha.txt": BASE_SHA + "\n",
        "runtime_metadata/source_formal_job.env": "formal_job_id=58513929\nformal_task_id=0\n",
        "runtime_metadata/source_package.env": "source_mode=individual_task_package\n",
        "runtime_metadata/source_package.sha256": "0" * 64 + "  package.tar.gz\n",
        "runtime_metadata/checkpoint_member_hashes.sha256": "0" * 64 + "  train/model.best_actor\n",
        "runtime_metadata/original_source_manifest.sha256": "0" * 64 + "  source.py\n",
        "runtime_metadata/original_task_runtime_metadata.env": "task_id=0\n",
        "runtime_metadata/diagnostic_command.txt": "python evaluate_td3_gnn_infrastructure_diagnostics.py\n",
        "runtime_metadata/evaluator_time_verbose.txt": "Maximum resident set size (kbytes): 1\n",
        "runtime_metadata/task_runtime_metadata.env": "task_id=0\n",
    }
    if include_checkpoint:
        members["checkpoint_staging/model.best_actor"] = "leak"
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
    bin_dir.mkdir()
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


def test_source_bundle_real_mode_rejects_explicit_head_mismatch(tmp_path):
    result = run_source_bundle(
        tmp_path,
        extra_env={"EV_GNN_DIAGNOSTIC_SMOKE_SOURCE_EXPECTED_HEAD_SHA": BASE_SHA},
    )

    assert result.returncode != 0
    assert "does not match" in result.stderr


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
