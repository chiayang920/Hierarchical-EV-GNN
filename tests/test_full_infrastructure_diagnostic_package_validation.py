import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts.validate_full_infrastructure_diagnostic_packages import (
    PackageValidationError,
    expected_episode_keys,
    validate_safe_tar_members,
    validate_stage_d_complete_bundle,
    validate_stage_d_task_package,
)


def csv_bytes(fieldnames, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def tar_bytes(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members:
            if isinstance(payload, tarfile.TarInfo):
                archive.addfile(payload)
                continue
            data = payload if isinstance(payload, bytes) else str(payload).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def add_manifest(members, manifest_name):
    lines = []
    for name, payload in members:
        if isinstance(payload, tarfile.TarInfo):
            continue
        data = payload if isinstance(payload, bytes) else str(payload).encode()
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
    return [*members, (manifest_name, "".join(lines))]


def checkpoint_rows(task_id):
    starts = [0, 5, 10, 15, 20, 25, 30, 35]
    task_algorithms = ["actiongnn", "hierarchical"] * 4
    scales = ["25cp", "25cp", "100cp", "100cp", "500cp", "500cp", "1000cp", "1000cp"]
    return [
        {
            "task_id": task_id,
            "scale": scales[task_id],
            "algorithm": task_algorithms[task_id],
            "training_seed": seed,
            "formal_task_id": starts[task_id] + seed,
            "status": "ok",
        }
        for seed in range(5)
    ]


def episode_rows(task_id):
    scales = ["25cp", "25cp", "100cp", "100cp", "500cp", "500cp", "1000cp", "1000cp"]
    algorithms = ["actiongnn", "hierarchical"] * 4
    offsets = {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}
    scale = scales[task_id]
    return [
        {
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithms[task_id],
            "training_seed": seed,
            "episode_index": episode,
            "episode_seed": offsets[scale] + seed + episode,
            "diagnostic_schema_version": 3,
        }
        for seed in range(5)
        for episode in range(30)
    ]


def task_package_bytes(task_id=0, checkpoint_override=None, episode_override=None, stderr=b""):
    checkpoints = checkpoint_rows(task_id) if checkpoint_override is None else checkpoint_override
    episodes = episode_rows(task_id) if episode_override is None else episode_override
    members = [
        (
            "runtime_metadata/task_metadata.json",
            json.dumps({"task_id": task_id, "status": "ok", "diagnostic_schema_version": 3}),
        ),
        (
            "summaries/checkpoint_inventory.csv",
            csv_bytes(
                ["task_id", "scale", "algorithm", "training_seed", "formal_task_id", "status"],
                checkpoints,
            ),
        ),
        (
            "summaries/episode_inventory.csv",
            csv_bytes(
                ["task_id", "scale", "algorithm", "training_seed", "episode_index", "episode_seed", "diagnostic_schema_version"],
                episodes,
            ),
        ),
        ("summaries/runtime_summary.csv", b"task_id,status\n%d,ok\n" % task_id),
        ("summaries/validation_summary.json", json.dumps({"status": "ok"})),
        ("logs/stderr.log", stderr),
    ]
    return tar_bytes(add_manifest(members, "runtime_metadata/package_file_checksums.sha256"))


def complete_bundle_bytes(task_ids=range(8), checkpoint_override=None, episode_override=None):
    checkpoints = [row for task_id in range(8) for row in checkpoint_rows(task_id)]
    episodes = [row for task_id in range(8) for row in episode_rows(task_id)]
    if checkpoint_override is not None:
        checkpoints = checkpoint_override
    if episode_override is not None:
        episodes = episode_override
    members = [(f"task_packages/task{task_id}.tar.gz", task_package_bytes(task_id)) for task_id in task_ids]
    members.extend(
        [
            (
                "summaries/checkpoint_inventory.csv",
                csv_bytes(
                    ["task_id", "scale", "algorithm", "training_seed", "formal_task_id", "status"],
                    checkpoints,
                ),
            ),
            (
                "summaries/episode_inventory.csv",
                csv_bytes(
                    ["task_id", "scale", "algorithm", "training_seed", "episode_index", "episode_seed", "diagnostic_schema_version"],
                    episodes,
                ),
            ),
            ("summaries/schema_inventory.csv", b"diagnostic_schema_version,status\n3,ok\n"),
            ("summaries/reconciliation_inventory.csv", b"check,status\nall,ok\n"),
            ("summaries/missing_field_inventory.csv", b"field,status\nnone,ok\n"),
            ("summaries/seed_level_paired_summary.csv", b"scale,training_seed,status\n25cp,0,ok\n"),
            ("summaries/failure_inventory.csv", b"failure\n"),
            ("summaries/warning_inventory.csv", b"warning\n"),
            ("runtime_metadata/source_commit_sha.txt", "f" * 40 + "\n"),
            ("runtime_metadata/array_job_id.txt", "12345\n"),
            ("runtime_metadata/reducer_job_id.txt", "12346\n"),
        ]
    )
    return tar_bytes(add_manifest(members, "runtime_metadata/complete_file_checksums.sha256"))


def write_bytes(tmp_path, name, payload):
    path = tmp_path / name
    path.write_bytes(payload)
    return path


@pytest.mark.parametrize("name", ["/absolute", "../traversal", "safe/../../escape"])
def test_safe_tar_rejects_unsafe_paths(name):
    payload = tar_bytes([(name, b"x")])
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        with pytest.raises(PackageValidationError, match="unsafe tar path"):
            validate_safe_tar_members(archive)


def test_safe_tar_rejects_duplicate_member():
    payload = tar_bytes([("dup", b"a"), ("dup", b"b")])
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        with pytest.raises(PackageValidationError, match="duplicate tar member"):
            validate_safe_tar_members(archive)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "device"])
def test_safe_tar_rejects_special_entries(kind):
    info = tarfile.TarInfo("unsafe")
    if kind == "symlink":
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
    elif kind == "hardlink":
        info.type = tarfile.LNKTYPE
        info.linkname = "target"
    else:
        info.type = tarfile.CHRTYPE
    payload = tar_bytes([("unsafe", info)])
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        with pytest.raises(PackageValidationError, match="unsupported tar member type"):
            validate_safe_tar_members(archive)


def test_task_package_accepts_exact_contract(tmp_path):
    result = validate_stage_d_task_package(write_bytes(tmp_path, "task0.tar.gz", task_package_bytes(0)), expected_task_id=0)
    assert result == {"task_id": 0, "checkpoint_groups": 5, "episode_keys": 150, "status": "ok"}


@pytest.mark.parametrize("count", [4, 6])
def test_task_package_rejects_wrong_seed_group_count(tmp_path, count):
    checkpoints = checkpoint_rows(0)[:count]
    if count == 6:
        checkpoints.append({**checkpoint_rows(0)[-1], "training_seed": 5, "formal_task_id": 5})
    with pytest.raises(PackageValidationError, match="exactly five checkpoint groups"):
        validate_stage_d_task_package(write_bytes(tmp_path, "task.tar.gz", task_package_bytes(0, checkpoint_override=checkpoints)), expected_task_id=0)


def test_task_package_rejects_duplicate_seed(tmp_path):
    checkpoints = checkpoint_rows(0)
    checkpoints[-1] = dict(checkpoints[0])
    with pytest.raises(PackageValidationError, match="duplicate training seed"):
        validate_stage_d_task_package(write_bytes(tmp_path, "task.tar.gz", task_package_bytes(0, checkpoint_override=checkpoints)), expected_task_id=0)


def test_task_package_rejects_wrong_task_identity(tmp_path):
    with pytest.raises(PackageValidationError, match="task identity mismatch"):
        validate_stage_d_task_package(write_bytes(tmp_path, "task.tar.gz", task_package_bytes(0)), expected_task_id=1)


def test_task_package_rejects_incomplete_episode_inventory(tmp_path):
    with pytest.raises(PackageValidationError, match="exactly 150 episode keys"):
        validate_stage_d_task_package(write_bytes(tmp_path, "task.tar.gz", task_package_bytes(0, episode_override=episode_rows(0)[:-1])), expected_task_id=0)


def test_task_package_rejects_checkpoint_leakage(tmp_path):
    payload = task_package_bytes(0)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as source:
        members = [(member.name, source.extractfile(member).read()) for member in source.getmembers() if member.isfile() and member.name != "runtime_metadata/package_file_checksums.sha256"]
    members.append(("leaked/model.best_actor", b"checkpoint"))
    payload = tar_bytes(add_manifest(members, "runtime_metadata/package_file_checksums.sha256"))
    with pytest.raises(PackageValidationError, match="checkpoint leakage"):
        validate_stage_d_task_package(write_bytes(tmp_path, "task.tar.gz", payload), expected_task_id=0)


def test_task_package_rejects_checksum_mismatch(tmp_path):
    payload = task_package_bytes(0)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as source:
        members = [(member.name, source.extractfile(member).read()) for member in source.getmembers() if member.isfile()]
    members[0] = (members[0][0], b"corrupted")
    with pytest.raises(PackageValidationError, match="checksum mismatch"):
        validate_stage_d_task_package(write_bytes(tmp_path, "task.tar.gz", tar_bytes(members)), expected_task_id=0)


def test_complete_bundle_accepts_exact_contract(tmp_path):
    result = validate_stage_d_complete_bundle(write_bytes(tmp_path, "complete.tar.gz", complete_bundle_bytes()))
    assert result["task_packages"] == 8
    assert result["checkpoint_groups"] == 40
    assert result["episode_keys"] == 1200
    assert result["status"] == "ok"


@pytest.mark.parametrize("task_ids", [range(7), range(9)])
def test_complete_bundle_rejects_wrong_task_package_count(tmp_path, task_ids):
    with pytest.raises(PackageValidationError, match="exactly eight task packages"):
        validate_stage_d_complete_bundle(write_bytes(tmp_path, "complete.tar.gz", complete_bundle_bytes(task_ids=task_ids)))


def test_complete_bundle_rejects_missing_checkpoint_group(tmp_path):
    checkpoints = [row for task_id in range(8) for row in checkpoint_rows(task_id)][:-1]
    with pytest.raises(PackageValidationError, match="exactly 40 checkpoint groups"):
        validate_stage_d_complete_bundle(write_bytes(tmp_path, "complete.tar.gz", complete_bundle_bytes(checkpoint_override=checkpoints)))


def test_complete_bundle_rejects_incomplete_episode_inventory(tmp_path):
    episodes = [row for task_id in range(8) for row in episode_rows(task_id)][:-1]
    with pytest.raises(PackageValidationError, match="exactly 1,200 episode keys"):
        validate_stage_d_complete_bundle(write_bytes(tmp_path, "complete.tar.gz", complete_bundle_bytes(episode_override=episodes)))


def test_expected_episode_keys_is_exact():
    keys = expected_episode_keys()
    assert len(keys) == 1200
    assert len(set(keys)) == 1200
