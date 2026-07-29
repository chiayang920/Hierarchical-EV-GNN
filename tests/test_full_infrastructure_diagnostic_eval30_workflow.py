import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    EVAL_EPISODES,
    TRAINING_SEEDS,
    episode_seed,
    formal_task_id,
    stage_d_task,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = PROJECT_ROOT / "scripts" / "validate_full_infrastructure_diagnostic_eval30.py"


@pytest.mark.parametrize(
    ("task_id", "scale", "algorithm", "formal_ids"),
    [
        (0, "25cp", "actiongnn", (0, 1, 2, 3, 4)),
        (1, "25cp", "hierarchical", (5, 6, 7, 8, 9)),
        (2, "100cp", "actiongnn", (10, 11, 12, 13, 14)),
        (3, "100cp", "hierarchical", (15, 16, 17, 18, 19)),
        (4, "500cp", "actiongnn", (20, 21, 22, 23, 24)),
        (5, "500cp", "hierarchical", (25, 26, 27, 28, 29)),
        (6, "1000cp", "actiongnn", (30, 31, 32, 33, 34)),
        (7, "1000cp", "hierarchical", (35, 36, 37, 38, 39)),
    ],
)
def test_stage_d_task_mapping(task_id, scale, algorithm, formal_ids):
    task = stage_d_task(task_id)
    assert task["scale"] == scale
    assert task["algorithm"] == algorithm
    assert tuple(formal_task_id(task_id, seed) for seed in TRAINING_SEEDS) == formal_ids


def test_stage_d_task_returns_defensive_copy():
    task = stage_d_task(0)
    task["scale"] = "mutated"
    assert stage_d_task(0)["scale"] == "25cp"


def test_stage_d_episode_contract():
    assert TRAINING_SEEDS == (0, 1, 2, 3, 4)
    assert EVAL_EPISODES == 30
    assert episode_seed("25cp", 0, 0) == 710000
    assert episode_seed("25cp", 4, 29) == 710033
    assert episode_seed("1000cp", 4, 29) == 740033


@pytest.mark.parametrize("task_id", [-1, 8, 99, True, "0"])
def test_stage_d_task_rejects_invalid_id(task_id):
    with pytest.raises(ValueError, match="task ID"):
        stage_d_task(task_id)


@pytest.mark.parametrize("training_seed", [-1, 5, True, "0"])
def test_formal_task_id_rejects_invalid_training_seed(training_seed):
    with pytest.raises(ValueError, match="training seed"):
        formal_task_id(0, training_seed)


@pytest.mark.parametrize(
    ("scale", "training_seed", "episode_index", "message"),
    [
        ("invalid", 0, 0, "scale"),
        ("25cp", 5, 0, "training seed"),
        ("25cp", 0, -1, "episode index"),
        ("25cp", 0, 30, "episode index"),
        ("25cp", 0, True, "episode index"),
    ],
)
def test_episode_seed_rejects_invalid_input(
    scale,
    training_seed,
    episode_index,
    message,
):
    with pytest.raises(ValueError, match=message):
        episode_seed(scale, training_seed, episode_index)


@pytest.mark.parametrize(
    ("task_id", "expected_lines"),
    [
        (
            0,
            [
                "task_id=0",
                "scale=25cp",
                "algorithm=actiongnn",
                "training_seeds=0,1,2,3,4",
                "formal_task_ids=0,1,2,3,4",
                "eval_episodes=30",
                "eval_seed_offset=710000",
            ],
        ),
        (
            7,
            [
                "task_id=7",
                "scale=1000cp",
                "algorithm=hierarchical",
                "training_seeds=0,1,2,3,4",
                "formal_task_ids=35,36,37,38,39",
                "eval_episodes=30",
                "eval_seed_offset=740000",
            ],
        ),
    ],
)
def test_task_mapping_cli(task_id, expected_lines):
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "task-mapping",
            "--task-id",
            str(task_id),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == expected_lines


import csv
import json

from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    expected_episode_keys,
    validate_episode_inventory,
    validate_seed_output_directory,
)


def complete_inventory_rows():
    return [
        {
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "episode_index": str(episode_index),
            "episode_seed": str(offset + seed + episode_index),
            "diagnostic_schema_version": "3",
        }
        for scale, offset in [
            ("25cp", 710000),
            ("100cp", 720000),
            ("500cp", 730000),
            ("1000cp", 740000),
        ]
        for algorithm in ("actiongnn", "hierarchical")
        for seed in range(5)
        for episode_index in range(30)
    ]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def valid_episode_row(
    episode_index,
    *,
    scale="25cp",
    algorithm="actiongnn",
    training_seed=0,
):
    return {
        "matrix_job_id": "99999999",
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": str(training_seed),
        "episode_index": str(episode_index),
        "episode_seed": str(
            {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}[scale]
            + training_seed
            + episode_index
        ),
        "episode_steps": "112",
        "done": "True",
        "episode_reward": "-1.0",
        "tracking_error": "1.0",
        "energy_tracking_error": "1.0",
        "power_tracker_violation": "1.0",
        "total_energy_charged": "1.0",
        "total_energy_discharged": "0.0",
        "average_user_satisfaction": "1.0",
        "energy_user_satisfaction": "100.0",
        "total_transformer_overload": "0.0",
        "total_ev_served": "1",
        "global_action_fraction_at_max_active": "0.5",
        "diagnostic_schema_version": "3",
    }


def build_seed_output(
    root,
    *,
    task_id=0,
    training_seed=0,
):
    mapping = {
        0: ("25cp", "actiongnn", 25, 3),
        1: ("25cp", "hierarchical", 25, 3),
        2: ("100cp", "actiongnn", 100, 7),
        3: ("100cp", "hierarchical", 100, 7),
        4: ("500cp", "actiongnn", 500, 35),
        5: ("500cp", "hierarchical", 500, 35),
        6: ("1000cp", "actiongnn", 1000, 70),
        7: ("1000cp", "hierarchical", 1000, 70),
    }
    scale, algorithm, charger_count, transformer_count = mapping[task_id]
    diagnostics = root / "diagnostics"
    validation = root / "validation"
    logs = root / "logs"
    diagnostics.mkdir(parents=True)
    validation.mkdir()
    logs.mkdir()

    episode_rows = [
        valid_episode_row(
            episode_index,
            scale=scale,
            algorithm=algorithm,
            training_seed=training_seed,
        )
        for episode_index in range(30)
    ]
    write_csv(
        diagnostics / "episode_diagnostics.csv",
        list(episode_rows[0]),
        episode_rows,
    )

    seed_summary = {
        "matrix_job_id": "99999999",
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": str(training_seed),
        "n_eval_episodes": "30",
        "diagnostic_schema_version": "3",
    }
    write_csv(
        diagnostics / "seed_summary_diagnostics.csv",
        list(seed_summary),
        [seed_summary],
    )

    def infrastructure_rows(count):
        return [
            {
                "matrix_job_id": "99999999",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": str(training_seed),
                "episode_index": str(episode_index),
                "episode_seed": str(
                    {
                        "25cp": 710000,
                        "100cp": 720000,
                        "500cp": 730000,
                        "1000cp": 740000,
                    }[scale]
                    + training_seed
                    + episode_index
                ),
                "diagnostic_schema_version": "3",
            }
            for episode_index in range(30)
            for _ in range(count)
        ]

    transformer_rows = infrastructure_rows(transformer_count)
    charger_rows = infrastructure_rows(charger_count)
    write_csv(
        diagnostics / "transformer_diagnostics.csv",
        list(transformer_rows[0]),
        transformer_rows,
    )
    write_csv(
        diagnostics / "charger_diagnostics.csv",
        list(charger_rows[0]),
        charger_rows,
    )

    write_csv(
        validation / "canonical_reconciliation.csv",
        ["status"],
        [{"status": "pass"} for _ in range(30)],
    )
    write_csv(
        validation / "service_reconciliation.csv",
        [
            "served_count_reconciliation_status",
            "satisfaction_sum_reconciliation_status",
            "charged_energy_reconciliation_status",
            "discharged_energy_reconciliation_status",
        ],
        [
            {
                "served_count_reconciliation_status": "pass",
                "satisfaction_sum_reconciliation_status": "pass",
                "charged_energy_reconciliation_status": "pass",
                "discharged_energy_reconciliation_status": "pass",
            }
            for _ in range(30)
        ],
    )
    (validation / "mapping_validation.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )
    (logs / "stderr.log").write_bytes(b"")
    return root


def mutate_csv(path, mutator):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    mutator(fieldnames, rows)
    write_csv(path, fieldnames, rows)


def test_expected_episode_key_count():
    assert len(expected_episode_keys()) == 1200


def test_validate_episode_inventory_accepts_exact_matrix():
    validate_episode_inventory(complete_inventory_rows())


def test_validate_episode_inventory_rejects_duplicate():
    rows = complete_inventory_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate episode key"):
        validate_episode_inventory(rows)


def test_validate_episode_inventory_rejects_missing():
    rows = complete_inventory_rows()[1:]
    with pytest.raises(ValueError, match="missing episode key"):
        validate_episode_inventory(rows)


def test_validate_episode_inventory_rejects_wrong_seed_formula():
    rows = complete_inventory_rows()
    rows[0]["episode_seed"] = "999"
    with pytest.raises(ValueError, match="episode seed mismatch"):
        validate_episode_inventory(rows)


def test_validate_episode_inventory_requires_schema_v3():
    rows = complete_inventory_rows()
    rows[0]["diagnostic_schema_version"] = "2"
    with pytest.raises(ValueError, match="schema"):
        validate_episode_inventory(rows)


def test_validate_seed_output_accepts_exact_seed_directory(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    result = validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)
    assert result["status"] == "ok"
    assert result["episode_count"] == 30
    assert result["charger_row_count"] == 750
    assert result["transformer_row_count"] == 90
    assert result["formal_task_id"] == 0


@pytest.mark.parametrize("episode_count", [29, 31])
def test_validate_seed_output_rejects_wrong_episode_count(tmp_path, episode_count):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def change_count(fieldnames, rows):
        if episode_count == 29:
            rows.pop()
        else:
            rows.append(dict(rows[-1]))

    mutate_csv(path, change_count)
    with pytest.raises(ValueError, match="exactly 30 rows"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_duplicate_episode_index(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def duplicate_index(fieldnames, rows):
        rows[1]["episode_index"] = rows[0]["episode_index"]
        rows[1]["episode_seed"] = rows[0]["episode_seed"]

    mutate_csv(path, duplicate_index)
    with pytest.raises(ValueError, match="duplicate episode index"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scale", "100cp", "identity mismatch"),
        ("algorithm", "hierarchical", "identity mismatch"),
        ("training_seed", "1", "identity mismatch"),
        ("episode_seed", "999", "episode seed mismatch"),
        ("diagnostic_schema_version", "2", "schema"),
        ("episode_reward", "", "blank required metric"),
        ("episode_reward", "NaN", "non-finite"),
        ("episode_reward", "Inf", "non-finite"),
    ],
)
def test_validate_seed_output_rejects_bad_episode_value(
    tmp_path,
    field,
    value,
    message,
):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def update(fieldnames, rows):
        rows[0][field] = value

    mutate_csv(path, update)
    with pytest.raises(ValueError, match=message):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_missing_required_column(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "diagnostics" / "episode_diagnostics.csv"

    def remove_column(fieldnames, rows):
        fieldnames.remove("episode_reward")
        for row in rows:
            row.pop("episode_reward")

    mutate_csv(path, remove_column)
    with pytest.raises(ValueError, match="missing required column"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_missing_required_csv(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "diagnostics" / "charger_diagnostics.csv").unlink()
    with pytest.raises(ValueError, match="missing or empty CSV"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_failed_canonical_reconciliation(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "validation" / "canonical_reconciliation.csv"

    def fail(fieldnames, rows):
        rows[0]["status"] = "fail"

    mutate_csv(path, fail)
    with pytest.raises(ValueError, match="canonical reconciliation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_failed_service_reconciliation(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    path = seed_dir / "validation" / "service_reconciliation.csv"

    def fail(fieldnames, rows):
        rows[0]["charged_energy_reconciliation_status"] = "fail"

    mutate_csv(path, fail)
    with pytest.raises(ValueError, match="service reconciliation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_failed_mapping_validation(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "validation" / "mapping_validation.json").write_text(
        json.dumps({"status": "fail"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mapping validation"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_rejects_nonempty_stderr(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    (seed_dir / "logs" / "stderr.log").write_text(
        "Traceback",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stderr log must be empty"):
        validate_seed_output_directory(seed_dir, task_id=0, training_seed=0)


def test_validate_seed_output_cli(tmp_path):
    seed_dir = build_seed_output(tmp_path / "seed0")
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate-seed-output",
            "--directory",
            str(seed_dir),
            "--task-id",
            "0",
            "--training-seed",
            "0",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["episode_count"] == 30

# STAGE_D_TASK3_MINIMAL_PACKAGE_CONTRACT_TESTS
import hashlib
import io
import shutil
import tarfile

import scripts.validate_full_infrastructure_diagnostic_eval30 as task3_validator_module


def _task3_validate_package(package, task_id):
    return task3_validator_module.validate_stage_d_task_package(package, task_id)


def _task3_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _task3_read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _task3_rewrite_csv(path, mutator):
    fieldnames, rows = _task3_read_csv(path)
    mutator(fieldnames, rows)
    write_csv(path, fieldnames, rows)


def _task3_tar_member(name, payload=b"x", *, member_type=None, linkname=""):
    info = tarfile.TarInfo(name)
    if member_type is not None:
        info.type = member_type
        info.linkname = linkname
        info.size = 0
        return info, None
    info.size = len(payload)
    return info, payload


def build_task3_package(
    tmp_path,
    *,
    task_id=0,
    staging_mutator=None,
    manifest_mutator=None,
    extra_members=(),
):
    root = tmp_path / "task_package_staging"
    root.mkdir()
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])

    for seed in TRAINING_SEEDS:
        build_seed_output(root / f"seed{seed}", task_id=task_id, training_seed=seed)

    _task3_write_json(
        root / "task_metadata" / "task.json",
        {
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "formal_job_id": "58513929",
            "training_seeds": list(TRAINING_SEEDS),
            "formal_task_ids": [formal_task_id(task_id, seed) for seed in TRAINING_SEEDS],
            "eval_episodes_per_seed": EVAL_EPISODES,
            "diagnostic_schema_version": "3",
        },
    )

    checkpoint_rows = [
        {
            "task_id": str(task_id),
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "formal_task_id": str(formal_task_id(task_id, seed)),
            "diagnostic_schema_version": "3",
        }
        for seed in TRAINING_SEEDS
    ]
    write_csv(
        root / "summaries" / "checkpoint_inventory.csv",
        list(checkpoint_rows[0]),
        checkpoint_rows,
    )

    episode_rows = [
        {
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "episode_index": str(episode_index),
            "episode_seed": str(episode_seed(scale, seed, episode_index)),
            "diagnostic_schema_version": "3",
        }
        for seed in TRAINING_SEEDS
        for episode_index in range(EVAL_EPISODES)
    ]
    write_csv(
        root / "summaries" / "episode_inventory.csv",
        list(episode_rows[0]),
        episode_rows,
    )

    runtime_rows = [
        {
            "task_id": str(task_id),
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(seed),
            "formal_task_id": str(formal_task_id(task_id, seed)),
            "status": "ok",
            "episode_count": str(EVAL_EPISODES),
        }
        for seed in TRAINING_SEEDS
    ]
    write_csv(
        root / "summaries" / "runtime_summary.csv",
        list(runtime_rows[0]),
        runtime_rows,
    )

    _task3_write_json(
        root / "validation" / "task_validation.json",
        {
            "status": "ok",
            "task_id": task_id,
            "checkpoint_groups": 5,
            "episode_count": 150,
            "diagnostic_schema_version": "3",
        },
    )
    logs = root / "logs"
    logs.mkdir()
    (logs / "stdout.log").write_text("TASK_OK\n", encoding="utf-8")
    (logs / "stderr.log").write_bytes(b"")

    if staging_mutator is not None:
        staging_mutator(root)

    manifest_path = root / "checksums" / "package_file_checksums.sha256"
    manifest_path.parent.mkdir()
    payload_paths = sorted(path for path in root.rglob("*") if path.is_file())
    manifest_path.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(root).as_posix()}\n"
            for path in payload_paths
        ),
        encoding="utf-8",
    )
    if manifest_mutator is not None:
        manifest_mutator(manifest_path)

    package = tmp_path / "task0.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        for info, payload in extra_members:
            archive.addfile(info, None if payload is None else io.BytesIO(payload))
    return package


def test_task3_valid_five_seed_package_passes(tmp_path):
    package = build_task3_package(tmp_path)
    result = _task3_validate_package(package, task_id=0)
    assert result == {
        "status": "ok",
        "task_id": 0,
        "scale": "25cp",
        "algorithm": "actiongnn",
        "checkpoint_groups": 5,
        "episode_count": 150,
        "diagnostic_schema_version": "3",
    }


def test_task3_package_cli(tmp_path):
    package = build_task3_package(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate-task-package",
            "--package",
            str(package),
            "--task-id",
            "0",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["episode_count"] == 150


def test_task3_rejects_missing_seed_group(tmp_path):
    def mutate(root):
        shutil.rmtree(root / "seed4")

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="file set mismatch"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_duplicate_checkpoint_seed(tmp_path):
    def mutate(root):
        path = root / "summaries" / "checkpoint_inventory.csv"

        def duplicate(fieldnames, rows):
            rows[1]["training_seed"] = rows[0]["training_seed"]
            rows[1]["formal_task_id"] = rows[0]["formal_task_id"]

        _task3_rewrite_csv(path, duplicate)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="checkpoint inventory"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_wrong_formal_task_id(tmp_path):
    def mutate(root):
        path = root / "summaries" / "checkpoint_inventory.csv"

        def wrong_id(fieldnames, rows):
            rows[2]["formal_task_id"] = "999"

        _task3_rewrite_csv(path, wrong_id)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="checkpoint inventory"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_missing_aggregate_episode_key(tmp_path):
    def mutate(root):
        path = root / "summaries" / "episode_inventory.csv"

        def remove(fieldnames, rows):
            rows.pop()

        _task3_rewrite_csv(path, remove)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="episode inventory"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_failed_seed_contract(tmp_path):
    def mutate(root):
        (root / "seed2" / "logs" / "stderr.log").write_text(
            "Traceback", encoding="utf-8"
        )

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="stderr log must be empty"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_nonempty_task_stderr(tmp_path):
    def mutate(root):
        (root / "logs" / "stderr.log").write_text("failure", encoding="utf-8")

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="task stderr"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_manifest_coverage_gap(tmp_path):
    def mutate(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

    package = build_task3_package(tmp_path, manifest_mutator=mutate)
    with pytest.raises(ValueError, match="manifest coverage"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_checksum_mismatch(tmp_path):
    def mutate(path):
        lines = path.read_text(encoding="utf-8").splitlines()
        _, name = lines[0].split("  ", 1)
        lines[0] = f"{'0' * 64}  {name}"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    package = build_task3_package(tmp_path, manifest_mutator=mutate)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_unsafe_tar_path(tmp_path):
    package = build_task3_package(
        tmp_path,
        extra_members=[_task3_tar_member("../escape.txt")],
    )
    with pytest.raises(ValueError, match="unsafe tar path"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_duplicate_tar_member(tmp_path):
    package = build_task3_package(
        tmp_path,
        extra_members=[_task3_tar_member("logs/stdout.log", b"duplicate")],
    )
    with pytest.raises(ValueError, match="duplicate tar member"):
        _task3_validate_package(package, task_id=0)


@pytest.mark.parametrize(
    ("member_type", "linkname"),
    [
        (tarfile.SYMTYPE, "logs/stdout.log"),
        (tarfile.LNKTYPE, "logs/stdout.log"),
        (tarfile.CHRTYPE, ""),
    ],
)
def test_task3_rejects_links_and_devices(tmp_path, member_type, linkname):
    package = build_task3_package(
        tmp_path,
        extra_members=[
            _task3_tar_member(
                "unsafe-member",
                member_type=member_type,
                linkname=linkname,
            )
        ],
    )
    with pytest.raises(ValueError, match="link or device"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_checkpoint_leakage(tmp_path):
    package = build_task3_package(
        tmp_path,
        extra_members=[_task3_tar_member("seed0/checkpoint/model.best_actor")],
    )
    with pytest.raises(ValueError, match="checkpoint artefact"):
        _task3_validate_package(package, task_id=0)


def test_task3_rejects_wrong_task_metadata(tmp_path):
    def mutate(root):
        path = root / "task_metadata" / "task.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["algorithm"] = "hierarchical"
        _task3_write_json(path, payload)

    package = build_task3_package(tmp_path, staging_mutator=mutate)
    with pytest.raises(ValueError, match="task metadata"):
        _task3_validate_package(package, task_id=0)
