import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.full_infrastructure_diagnostic_eval30_packages import (
    validate_safe_tar_members,
    validate_stage_d_complete_bundle,
    validate_stage_d_task_package,
)

TASKS = {
    0: ("25cp", "actiongnn", 0),
    1: ("25cp", "hierarchical", 5),
    2: ("100cp", "actiongnn", 10),
    3: ("100cp", "hierarchical", 15),
    4: ("500cp", "actiongnn", 20),
    5: ("500cp", "hierarchical", 25),
    6: ("1000cp", "actiongnn", 30),
    7: ("1000cp", "hierarchical", 35),
}
OFFSETS = {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}


def csv_bytes(fieldnames, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def add_bytes(tar, name, payload=b""):
    if isinstance(payload, str):
        payload = payload.encode()
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def checksum_manifest(members, exclude):
    lines = []
    for name, payload in sorted(members.items()):
        if name == exclude:
            continue
        if isinstance(payload, str):
            payload = payload.encode()
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {name}\n")
    return "".join(lines).encode()


def task_members(task_id=0, seeds=(0, 1, 2, 3, 4), array_job_id="99999999"):
    scale, algorithm, formal_start = TASKS[task_id]
    members = {}
    members["task_metadata/task.env"] = (
        f"task_id={task_id}\nscale={scale}\nalgorithm={algorithm}\n"
        f"array_job_id={array_job_id}\n"
    )
    members["task_metadata/source_commit_sha.txt"] = "f" * 40 + "\n"

    checkpoint_rows = []
    episode_rows = []
    for seed in seeds:
        formal_id = formal_start + seed
        checkpoint_rows.append(
            {
                "task_id": str(task_id),
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": str(seed),
                "formal_task_id": str(formal_id),
                "status": "ok",
            }
        )
        marker = {
            "status": "ok",
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": seed,
            "formal_task_id": formal_id,
            "episode_count": 30,
            "diagnostic_schema_version": "3",
        }
        members[f"seed{seed}/validation/seed_validation.json"] = json.dumps(
            marker, sort_keys=True
        )
        for episode_index in range(30):
            episode_rows.append(
                {
                    "scale": scale,
                    "algorithm": algorithm,
                    "training_seed": str(seed),
                    "episode_index": str(episode_index),
                    "episode_seed": str(OFFSETS[scale] + seed + episode_index),
                    "diagnostic_schema_version": "3",
                }
            )

    checkpoint_fields = [
        "task_id", "scale", "algorithm", "training_seed", "formal_task_id", "status"
    ]
    members["task_metadata/formal_checkpoint_inventory.csv"] = csv_bytes(
        checkpoint_fields, checkpoint_rows
    )
    members["summaries/checkpoint_inventory.csv"] = csv_bytes(
        checkpoint_fields, checkpoint_rows
    )
    episode_fields = [
        "scale", "algorithm", "training_seed", "episode_index",
        "episode_seed", "diagnostic_schema_version"
    ]
    members["summaries/episode_inventory.csv"] = csv_bytes(
        episode_fields, episode_rows
    )
    runtime_rows = [
        {"training_seed": str(seed), "state": "COMPLETED", "exit_code": "0:0"}
        for seed in seeds
    ]
    members["summaries/runtime_summary.csv"] = csv_bytes(
        ["training_seed", "state", "exit_code"], runtime_rows
    )
    members["validation/task_validation.json"] = json.dumps(
        {"status": "ok", "task_id": task_id}, sort_keys=True
    )
    members["logs/stdout.log"] = "ok\n"
    members["logs/stderr.log"] = b""
    members["checksums/task_file_checksums.sha256"] = checksum_manifest(
        members, "checksums/task_file_checksums.sha256"
    )
    return members


def create_tar(path, members, duplicate_name=None, symlink_name=None, unsafe_name=None):
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members.items():
            add_bytes(tar, name, payload)
        if duplicate_name is not None:
            add_bytes(tar, duplicate_name, b"duplicate")
        if symlink_name is not None:
            info = tarfile.TarInfo(symlink_name)
            info.type = tarfile.SYMTYPE
            info.linkname = "target"
            tar.addfile(info)
        if unsafe_name is not None:
            add_bytes(tar, unsafe_name, b"unsafe")
    return path


def build_task_package(tmp_path, task_id=0, seeds=(0, 1, 2, 3, 4)):
    return create_tar(
        tmp_path / f"full_infrastructure_diagnostic_eval30_task{task_id}_job99999999.tar.gz",
        task_members(task_id=task_id, seeds=seeds),
    )


def complete_members(tmp_path, task_ids=range(8), array_job_id="99999999"):
    members = {}
    task_inventory = []
    checkpoint_rows = []
    episode_rows = []
    runtime_rows = []

    for task_id in task_ids:
        task_package = build_task_package(tmp_path, task_id=task_id)
        payload = task_package.read_bytes()
        package_name = task_package.name
        members[f"task_packages/{package_name}"] = payload
        task_inventory.append(
            {
                "task_id": str(task_id),
                "package_name": package_name,
                "package_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        scale, algorithm, formal_start = TASKS[task_id]
        runtime_rows.append(
            {
                "task_id": str(task_id),
                "state": "COMPLETED",
                "exit_code": "0:0",
            }
        )
        for seed in range(5):
            checkpoint_rows.append(
                {
                    "task_id": str(task_id),
                    "scale": scale,
                    "algorithm": algorithm,
                    "training_seed": str(seed),
                    "formal_task_id": str(formal_start + seed),
                    "status": "ok",
                }
            )
            for episode_index in range(30):
                episode_rows.append(
                    {
                        "scale": scale,
                        "algorithm": algorithm,
                        "training_seed": str(seed),
                        "episode_index": str(episode_index),
                        "episode_seed": str(OFFSETS[scale] + seed + episode_index),
                        "diagnostic_schema_version": "3",
                    }
                )

    members["summaries/task_inventory.csv"] = csv_bytes(
        ["task_id", "package_name", "package_sha256"], task_inventory
    )
    members["summaries/checkpoint_inventory.csv"] = csv_bytes(
        ["task_id", "scale", "algorithm", "training_seed", "formal_task_id", "status"],
        checkpoint_rows,
    )
    members["summaries/episode_inventory.csv"] = csv_bytes(
        ["scale", "algorithm", "training_seed", "episode_index", "episode_seed",
         "diagnostic_schema_version"],
        episode_rows,
    )
    members["summaries/runtime_summary.csv"] = csv_bytes(
        ["task_id", "state", "exit_code"], runtime_rows
    )
    pass_row = [{"status": "pass"}]
    for name in (
        "canonical_reconciliation_summary.csv",
        "mapping_validation_summary.csv",
        "service_reconciliation_summary.csv",
    ):
        members[f"summaries/{name}"] = csv_bytes(["status"], pass_row)
    members["summaries/schema_inventory.csv"] = csv_bytes(
        ["diagnostic_schema_version", "episode_count"],
        [{"diagnostic_schema_version": "3", "episode_count": str(len(episode_rows))}],
    )
    members["summaries/missing_field_inventory.csv"] = csv_bytes(
        ["field", "count"], []
    )
    members["summaries/seed_level_paired_summary.csv"] = csv_bytes(
        ["scale", "training_seed", "status"],
        [
            {"scale": scale, "training_seed": str(seed), "status": "ok"}
            for scale in OFFSETS for seed in range(5)
        ],
    )
    members["validation/failure_manifest.txt"] = b""
    members["validation/warning_inventory.txt"] = b""
    members["provenance/source_commit_sha.txt"] = "f" * 40 + "\n"
    members["provenance/array_job_id.txt"] = array_job_id + "\n"
    members["provenance/reducer_job_id.txt"] = "99999998\n"
    members["checksums/complete_file_checksums.sha256"] = checksum_manifest(
        members, "checksums/complete_file_checksums.sha256"
    )
    return members


def build_complete_bundle(tmp_path, task_ids=range(8)):
    return create_tar(
        tmp_path / "full_infrastructure_diagnostics_complete_evidence_job99999999.tar.gz",
        complete_members(tmp_path, task_ids=task_ids),
    )


def test_validate_safe_tar_members_rejects_duplicate(tmp_path):
    bundle = create_tar(
        tmp_path / "duplicate.tar.gz",
        {"safe.txt": b"a"},
        duplicate_name="safe.txt",
    )
    with tarfile.open(bundle, "r:gz") as tar:
        with pytest.raises(ValueError, match="duplicate"):
            validate_safe_tar_members(tar.getmembers())


@pytest.mark.parametrize("unsafe_name", ["/absolute", "../traversal", "safe/../../escape"])
def test_archive_rejects_unsafe_path(tmp_path, unsafe_name):
    bundle = create_tar(
        tmp_path / "unsafe.tar.gz",
        {"safe.txt": b"a"},
        unsafe_name=unsafe_name,
    )
    with pytest.raises(ValueError, match="unsafe"):
        validate_stage_d_complete_bundle(bundle)


def test_archive_rejects_symbolic_link(tmp_path):
    bundle = create_tar(
        tmp_path / "link.tar.gz",
        {"safe.txt": b"a"},
        symlink_name="unsafe_link",
    )
    with pytest.raises(ValueError, match="link"):
        validate_stage_d_complete_bundle(bundle)


def test_task_package_accepts_five_seed_groups(tmp_path):
    bundle = build_task_package(tmp_path, task_id=0)
    result = validate_stage_d_task_package(bundle, task_id=0)
    assert result["status"] == "ok"
    assert result["seed_groups"] == 5
    assert result["episode_count"] == 150


def test_task_package_requires_five_seed_groups(tmp_path):
    bundle = build_task_package(tmp_path, task_id=0, seeds=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="five seed groups"):
        validate_stage_d_task_package(bundle, task_id=0)


def test_task_package_rejects_duplicate_seed_inventory(tmp_path):
    members = task_members(task_id=0)
    marker = json.loads(members["seed4/validation/seed_validation.json"])
    marker["training_seed"] = 3
    members["seed4/validation/seed_validation.json"] = json.dumps(marker)
    members["checksums/task_file_checksums.sha256"] = checksum_manifest(
        members, "checksums/task_file_checksums.sha256"
    )
    bundle = create_tar(tmp_path / "duplicate-seed.tar.gz", members)
    with pytest.raises(ValueError, match="duplicate training seed"):
        validate_stage_d_task_package(bundle, task_id=0)


def test_task_package_rejects_checksum_mismatch(tmp_path):
    members = task_members(task_id=0)
    members["logs/stdout.log"] = "mutated after manifest\n"
    bundle = create_tar(tmp_path / "bad-checksum.tar.gz", members)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_stage_d_task_package(bundle, task_id=0)


def test_task_package_rejects_checkpoint_leak(tmp_path):
    members = task_members(task_id=0)
    members["seed0/model.best_actor"] = b"checkpoint"
    members["checksums/task_file_checksums.sha256"] = checksum_manifest(
        members, "checksums/task_file_checksums.sha256"
    )
    bundle = create_tar(tmp_path / "checkpoint-leak.tar.gz", members)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_stage_d_task_package(bundle, task_id=0)


def test_complete_bundle_accepts_eight_task_packages(tmp_path):
    bundle = build_complete_bundle(tmp_path)
    result = validate_stage_d_complete_bundle(bundle)
    assert result["status"] == "ok"
    assert result["task_packages"] == 8
    assert result["checkpoint_groups"] == 40
    assert result["episodes"] == 1200


def test_complete_bundle_requires_eight_task_packages(tmp_path):
    bundle = build_complete_bundle(tmp_path, task_ids=range(7))
    with pytest.raises(ValueError, match="eight task packages"):
        validate_stage_d_complete_bundle(bundle)


def test_complete_bundle_rejects_task_package_hash_mismatch(tmp_path):
    members = complete_members(tmp_path)
    rows = list(csv.DictReader(io.StringIO(
        members["summaries/task_inventory.csv"].decode()
    )))
    rows[0]["package_sha256"] = "0" * 64
    members["summaries/task_inventory.csv"] = csv_bytes(
        ["task_id", "package_name", "package_sha256"], rows
    )
    members["checksums/complete_file_checksums.sha256"] = checksum_manifest(
        members, "checksums/complete_file_checksums.sha256"
    )
    bundle = create_tar(tmp_path / "bad-package-hash.tar.gz", members)
    with pytest.raises(ValueError, match="task package hash mismatch"):
        validate_stage_d_complete_bundle(bundle)


def test_complete_bundle_rejects_nonempty_failure_manifest(tmp_path):
    members = complete_members(tmp_path)
    members["validation/failure_manifest.txt"] = b"failure\n"
    members["checksums/complete_file_checksums.sha256"] = checksum_manifest(
        members, "checksums/complete_file_checksums.sha256"
    )
    bundle = create_tar(tmp_path / "failure-manifest.tar.gz", members)
    with pytest.raises(ValueError, match="failure manifest"):
        validate_stage_d_complete_bundle(bundle)
