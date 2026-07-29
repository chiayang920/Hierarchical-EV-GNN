#!/usr/bin/env python3
"""Archive contracts for the full per-infrastructure diagnostic eval30 workflow."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    EVAL_EPISODES,
    SCHEMA_VERSION,
    TRAINING_SEEDS,
    episode_seed,
    formal_task_id,
    stage_d_task,
    validate_episode_inventory,
)

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TASK_PACKAGE_PATTERN = re.compile(
    r"^full_infrastructure_diagnostic_eval30_task(?P<task_id>[0-7])_"
    r"job(?P<array_job_id>[0-9]+)\.tar\.gz$"
)

TASK_REQUIRED_FILES = {
    "task_metadata/task.env",
    "task_metadata/source_commit_sha.txt",
    "task_metadata/formal_checkpoint_inventory.csv",
    "summaries/checkpoint_inventory.csv",
    "summaries/episode_inventory.csv",
    "summaries/runtime_summary.csv",
    "validation/task_validation.json",
    "checksums/task_file_checksums.sha256",
    "logs/stdout.log",
    "logs/stderr.log",
}

COMPLETE_REQUIRED_FILES = {
    "summaries/task_inventory.csv",
    "summaries/checkpoint_inventory.csv",
    "summaries/episode_inventory.csv",
    "summaries/runtime_summary.csv",
    "summaries/canonical_reconciliation_summary.csv",
    "summaries/mapping_validation_summary.csv",
    "summaries/service_reconciliation_summary.csv",
    "summaries/schema_inventory.csv",
    "summaries/missing_field_inventory.csv",
    "summaries/seed_level_paired_summary.csv",
    "validation/failure_manifest.txt",
    "validation/warning_inventory.txt",
    "provenance/source_commit_sha.txt",
    "provenance/array_job_id.txt",
    "provenance/reducer_job_id.txt",
    "checksums/complete_file_checksums.sha256",
}


def validate_safe_tar_members(members: Iterable[tarfile.TarInfo]) -> None:
    seen: set[str] = set()
    for member in members:
        name = member.name
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or name.startswith("/"):
            raise ValueError(f"unsafe archive path: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate archive member: {name}")
        seen.add(name)
        if member.issym() or member.islnk():
            raise ValueError(f"archive link member is prohibited: {name}")
        if member.isdev():
            raise ValueError(f"archive device member is prohibited: {name}")


def _regular_member_payloads(tar: tarfile.TarFile) -> dict[str, bytes]:
    validate_safe_tar_members(tar.getmembers())
    payloads: dict[str, bytes] = {}
    for member in tar.getmembers():
        if not member.isfile():
            continue
        handle = tar.extractfile(member)
        if handle is None:
            raise ValueError(f"unable to read archive member: {member.name}")
        payloads[member.name] = handle.read()
    return payloads


def _parse_csv(payload: bytes, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(f"{label} has no CSV header")
    return list(reader.fieldnames), list(reader)


def _parse_key_values(payload: bytes, label: str) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        if "=" not in raw_line:
            raise ValueError(f"{label} line {line_number} is not key=value")
        key, value = raw_line.split("=", 1)
        if not key or key in values:
            raise ValueError(f"{label} has duplicate or blank key: {key!r}")
        values[key] = value
    return values


def _parse_json(payload: bytes, label: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc


def _parse_int(value: object, label: str) -> int:
    text = str(value)
    if not re.fullmatch(r"-?[0-9]+", text):
        raise ValueError(f"{label} must be an integer: {value!r}")
    return int(text)


def _parse_sha_manifest(payload: bytes, label: str) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"{label} line {line_number} is invalid")
        digest, name = match.groups()
        if name in rows:
            raise ValueError(f"{label} contains duplicate path: {name}")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{label} contains unsafe path: {name}")
        rows[name] = digest
    return rows


def _validate_manifest(payloads: dict[str, bytes], manifest_name: str, *, label: str) -> None:
    if manifest_name not in payloads:
        raise ValueError(f"missing checksum manifest: {manifest_name}")
    manifest = _parse_sha_manifest(payloads[manifest_name], label)
    expected_paths = set(payloads) - {manifest_name}
    observed_paths = set(manifest)
    missing = sorted(expected_paths - observed_paths)
    unexpected = sorted(observed_paths - expected_paths)
    if missing or unexpected:
        raise ValueError(
            f"{label} coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    for name, expected_digest in manifest.items():
        observed_digest = hashlib.sha256(payloads[name]).hexdigest()
        if observed_digest != expected_digest:
            raise ValueError(
                f"checksum mismatch for {name}: "
                f"expected={expected_digest}, observed={observed_digest}"
            )


def _require_files(payloads: dict[str, bytes], required: set[str], label: str) -> None:
    missing = sorted(required - set(payloads))
    if missing:
        raise ValueError(f"{label} missing required file(s): {', '.join(missing)}")


def _validate_no_checkpoint_leak(payloads: dict[str, bytes]) -> None:
    for name in payloads:
        basename = PurePosixPath(name).name
        if basename.startswith("model.best") or basename.startswith("model.last"):
            raise ValueError(f"checkpoint file leaked into evidence package: {name}")


def _validate_checkpoint_rows(
    rows: list[dict[str, str]], *, task_id: int | None = None
) -> set[tuple[int, str, str, int, int]]:
    expected_count = len(TRAINING_SEEDS) if task_id is not None else 40
    if len(rows) != expected_count:
        if task_id is not None:
            raise ValueError(
                "task package must contain five seed groups; "
                f"checkpoint inventory has {len(rows)} row(s)"
            )
        raise ValueError(
            f"checkpoint inventory must contain exactly {expected_count} rows, "
            f"got {len(rows)}"
        )

    observed: set[tuple[int, str, str, int, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        row_task_id = _parse_int(row.get("task_id"), "task_id")
        training_seed = _parse_int(row.get("training_seed"), "training_seed")
        formal_id = _parse_int(row.get("formal_task_id"), "formal_task_id")
        mapping = stage_d_task(row_task_id)
        scale = str(mapping["scale"])
        algorithm = str(mapping["algorithm"])
        key = (row_task_id, scale, algorithm, training_seed, formal_id)
        if task_id is not None and row_task_id != task_id:
            raise ValueError(
                f"checkpoint inventory row {row_number} task mismatch: "
                f"{row_task_id} != {task_id}"
            )
        if row.get("scale") != scale or row.get("algorithm") != algorithm:
            raise ValueError(f"checkpoint inventory row {row_number} identity mismatch")
        if training_seed not in TRAINING_SEEDS:
            raise ValueError(f"unsupported training seed: {training_seed}")
        if formal_id != formal_task_id(row_task_id, training_seed):
            raise ValueError(f"formal task ID mismatch at row {row_number}")
        if row.get("status") != "ok":
            raise ValueError(f"checkpoint inventory row {row_number} status must be ok")
        if key in observed:
            raise ValueError(f"duplicate checkpoint group: {key}")
        observed.add(key)
    return observed


def _validate_task_episode_rows(rows: list[dict[str, str]], *, task_id: int) -> None:
    mapping = stage_d_task(task_id)
    scale = str(mapping["scale"])
    algorithm = str(mapping["algorithm"])
    if len(rows) != len(TRAINING_SEEDS) * EVAL_EPISODES:
        raise ValueError("task episode inventory must contain exactly 150 rows")
    seen: set[tuple[int, int]] = set()
    for row_number, row in enumerate(rows, start=2):
        seed = _parse_int(row.get("training_seed"), "training_seed")
        episode_index = _parse_int(row.get("episode_index"), "episode_index")
        key = (seed, episode_index)
        if key in seen:
            raise ValueError(f"duplicate task episode key: {key}")
        seen.add(key)
        if seed not in TRAINING_SEEDS or not 0 <= episode_index < EVAL_EPISODES:
            raise ValueError(f"unexpected task episode key: {key}")
        if row.get("scale") != scale or row.get("algorithm") != algorithm:
            raise ValueError(f"task episode row {row_number} identity mismatch")
        expected_seed = episode_seed(scale, seed, episode_index)
        if _parse_int(row.get("episode_seed"), "episode_seed") != expected_seed:
            raise ValueError(f"task episode seed mismatch at row {row_number}")
        if row.get("diagnostic_schema_version") != SCHEMA_VERSION:
            raise ValueError(f"task episode schema must be {SCHEMA_VERSION}")
    expected = {
        (seed, episode_index)
        for seed in TRAINING_SEEDS
        for episode_index in range(EVAL_EPISODES)
    }
    if seen != expected:
        raise ValueError("task episode inventory is incomplete")


def _validate_runtime_rows(
    rows: list[dict[str, str]], *, expected_ids: set[int], id_field: str
) -> None:
    if len(rows) != len(expected_ids):
        raise ValueError(f"runtime summary must contain exactly {len(expected_ids)} rows")
    observed: set[int] = set()
    for row in rows:
        identity = _parse_int(row.get(id_field), id_field)
        if identity in observed:
            raise ValueError(f"duplicate runtime identity: {identity}")
        observed.add(identity)
        if row.get("state") != "COMPLETED" or row.get("exit_code") != "0:0":
            raise ValueError(f"runtime row {identity} did not complete successfully")
    if observed != expected_ids:
        raise ValueError(
            f"runtime identity mismatch: expected={sorted(expected_ids)}, "
            f"observed={sorted(observed)}"
        )


def validate_stage_d_task_package(bundle: Path, task_id: int) -> dict[str, object]:
    bundle = Path(bundle)
    if not bundle.is_file() or bundle.stat().st_size == 0:
        raise ValueError(f"missing or empty task package: {bundle}")
    stage_d_task(task_id)

    with tarfile.open(bundle, "r:gz") as tar:
        payloads = _regular_member_payloads(tar)

    _require_files(payloads, TASK_REQUIRED_FILES, "task package")
    _validate_no_checkpoint_leak(payloads)
    _validate_manifest(
        payloads,
        "checksums/task_file_checksums.sha256",
        label="task checksum manifest",
    )

    task_env = _parse_key_values(payloads["task_metadata/task.env"], "task metadata")
    mapping = stage_d_task(task_id)
    if _parse_int(task_env.get("task_id"), "task_id") != task_id:
        raise ValueError("task metadata task ID mismatch")
    if task_env.get("scale") != mapping["scale"]:
        raise ValueError("task metadata scale mismatch")
    if task_env.get("algorithm") != mapping["algorithm"]:
        raise ValueError("task metadata algorithm mismatch")
    array_job_id = task_env.get("array_job_id", "")
    if not array_job_id.isdigit():
        raise ValueError("task metadata array job ID must be numeric")

    source_sha = payloads["task_metadata/source_commit_sha.txt"].decode().strip()
    if not SHA1_RE.fullmatch(source_sha):
        raise ValueError("source commit SHA must contain 40 lowercase hex characters")

    _, formal_rows = _parse_csv(
        payloads["task_metadata/formal_checkpoint_inventory.csv"],
        "formal checkpoint inventory",
    )
    _, checkpoint_rows = _parse_csv(
        payloads["summaries/checkpoint_inventory.csv"],
        "checkpoint inventory",
    )
    formal_groups = _validate_checkpoint_rows(formal_rows, task_id=task_id)
    summary_groups = _validate_checkpoint_rows(checkpoint_rows, task_id=task_id)
    if formal_groups != summary_groups:
        raise ValueError("formal and summary checkpoint inventories disagree")

    marker_seeds: set[int] = set()
    for seed in TRAINING_SEEDS:
        marker_name = f"seed{seed}/validation/seed_validation.json"
        if marker_name not in payloads:
            continue
        marker = _parse_json(payloads[marker_name], marker_name)
        if not isinstance(marker, dict) or marker.get("status") != "ok":
            raise ValueError(f"seed validation marker status must be ok: {marker_name}")
        marker_seed = _parse_int(marker.get("training_seed"), "training_seed")
        if marker_seed in marker_seeds:
            raise ValueError(f"duplicate training seed: {marker_seed}")
        marker_seeds.add(marker_seed)
        if marker_seed != seed:
            raise ValueError(
                f"seed marker path and training seed disagree: {seed} != {marker_seed}"
            )
        expected_marker = {
            "task_id": task_id,
            "scale": mapping["scale"],
            "algorithm": mapping["algorithm"],
            "formal_task_id": formal_task_id(task_id, seed),
            "episode_count": EVAL_EPISODES,
            "diagnostic_schema_version": SCHEMA_VERSION,
        }
        for field, expected in expected_marker.items():
            if marker.get(field) != expected:
                raise ValueError(
                    f"seed validation marker mismatch for {field}: "
                    f"{marker.get(field)!r} != {expected!r}"
                )

    if marker_seeds != set(TRAINING_SEEDS):
        raise ValueError(
            f"task package must contain five seed groups; "
            f"observed={sorted(marker_seeds)}"
        )

    _, episode_rows = _parse_csv(
        payloads["summaries/episode_inventory.csv"], "task episode inventory"
    )
    _validate_task_episode_rows(episode_rows, task_id=task_id)

    _, runtime_rows = _parse_csv(
        payloads["summaries/runtime_summary.csv"], "task runtime summary"
    )
    _validate_runtime_rows(
        runtime_rows,
        expected_ids=set(TRAINING_SEEDS),
        id_field="training_seed",
    )

    task_validation = _parse_json(
        payloads["validation/task_validation.json"], "task validation"
    )
    if (
        not isinstance(task_validation, dict)
        or task_validation.get("status") != "ok"
        or _parse_int(task_validation.get("task_id"), "task_id") != task_id
    ):
        raise ValueError("task validation status or task ID mismatch")

    if payloads["logs/stderr.log"]:
        raise ValueError("task stderr log must be empty")

    return {
        "status": "ok",
        "task_id": task_id,
        "scale": mapping["scale"],
        "algorithm": mapping["algorithm"],
        "array_job_id": array_job_id,
        "seed_groups": len(marker_seeds),
        "checkpoint_groups": len(summary_groups),
        "episode_count": len(episode_rows),
        "diagnostic_schema_version": SCHEMA_VERSION,
        "source_commit_sha": source_sha,
    }


def _validate_pass_summary(payload: bytes, label: str) -> int:
    _, rows = _parse_csv(payload, label)
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    for row_number, row in enumerate(rows, start=2):
        if row.get("status") != "pass":
            raise ValueError(f"{label} row {row_number} status must be pass")
    return len(rows)


def validate_stage_d_complete_bundle(bundle: Path) -> dict[str, object]:
    bundle = Path(bundle)
    if not bundle.is_file() or bundle.stat().st_size == 0:
        raise ValueError(f"missing or empty complete bundle: {bundle}")

    with tarfile.open(bundle, "r:gz") as tar:
        payloads = _regular_member_payloads(tar)

    _require_files(payloads, COMPLETE_REQUIRED_FILES, "complete bundle")
    _validate_no_checkpoint_leak(payloads)
    _validate_manifest(
        payloads,
        "checksums/complete_file_checksums.sha256",
        label="complete checksum manifest",
    )

    source_sha = payloads["provenance/source_commit_sha.txt"].decode().strip()
    if not SHA1_RE.fullmatch(source_sha):
        raise ValueError("complete source commit SHA must contain 40 lowercase hex characters")
    array_job_id = payloads["provenance/array_job_id.txt"].decode().strip()
    reducer_job_id = payloads["provenance/reducer_job_id.txt"].decode().strip()
    if not array_job_id.isdigit() or not reducer_job_id.isdigit():
        raise ValueError("array and reducer job IDs must be numeric")

    _, task_rows = _parse_csv(payloads["summaries/task_inventory.csv"], "task inventory")
    if len(task_rows) != 8:
        raise ValueError(
            f"complete bundle must contain eight task packages, got {len(task_rows)}"
        )

    observed_task_ids: set[int] = set()
    task_results: list[dict[str, object]] = []
    for row_number, row in enumerate(task_rows, start=2):
        task_id = _parse_int(row.get("task_id"), "task_id")
        if task_id in observed_task_ids:
            raise ValueError(f"duplicate task package inventory row: {task_id}")
        observed_task_ids.add(task_id)
        if task_id not in range(8):
            raise ValueError(f"unexpected task ID in complete bundle: {task_id}")
        package_name = row.get("package_name", "")
        package_path = f"task_packages/{package_name}"
        match = TASK_PACKAGE_PATTERN.fullmatch(package_name)
        if match is None:
            raise ValueError(f"invalid task package name: {package_name}")
        if int(match.group("task_id")) != task_id:
            raise ValueError(f"task package filename task ID mismatch: {package_name}")
        if match.group("array_job_id") != array_job_id:
            raise ValueError(f"task package array job ID mismatch: {package_name}")
        if package_path not in payloads:
            raise ValueError(f"missing task package payload: {package_path}")
        expected_hash = row.get("package_sha256", "")
        if not SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"invalid task package hash in inventory row {row_number}")
        observed_hash = hashlib.sha256(payloads[package_path]).hexdigest()
        if observed_hash != expected_hash:
            raise ValueError(
                f"task package hash mismatch for {package_name}: "
                f"expected={expected_hash}, observed={observed_hash}"
            )
        with tempfile.TemporaryDirectory(prefix=f"stage_d_task{task_id}_") as tmp:
            task_path = Path(tmp) / package_name
            task_path.write_bytes(payloads[package_path])
            task_result = validate_stage_d_task_package(task_path, task_id)
        if task_result["array_job_id"] != array_job_id:
            raise ValueError(f"task package metadata array job ID mismatch: {package_name}")
        if task_result["source_commit_sha"] != source_sha:
            raise ValueError(f"task package source SHA mismatch: {package_name}")
        task_results.append(task_result)

    if observed_task_ids != set(range(8)):
        raise ValueError(
            f"complete bundle must contain task IDs 0..7, got {sorted(observed_task_ids)}"
        )

    _, checkpoint_rows = _parse_csv(
        payloads["summaries/checkpoint_inventory.csv"], "complete checkpoint inventory"
    )
    checkpoint_groups = _validate_checkpoint_rows(checkpoint_rows)

    _, episode_rows = _parse_csv(
        payloads["summaries/episode_inventory.csv"], "complete episode inventory"
    )
    validate_episode_inventory(episode_rows)

    _, runtime_rows = _parse_csv(
        payloads["summaries/runtime_summary.csv"], "complete runtime summary"
    )
    _validate_runtime_rows(runtime_rows, expected_ids=set(range(8)), id_field="task_id")

    _validate_pass_summary(
        payloads["summaries/canonical_reconciliation_summary.csv"],
        "canonical reconciliation summary",
    )
    _validate_pass_summary(
        payloads["summaries/mapping_validation_summary.csv"],
        "mapping validation summary",
    )
    _validate_pass_summary(
        payloads["summaries/service_reconciliation_summary.csv"],
        "service reconciliation summary",
    )

    _, schema_rows = _parse_csv(payloads["summaries/schema_inventory.csv"], "schema inventory")
    if len(schema_rows) != 1:
        raise ValueError("schema inventory must contain exactly one row")
    if schema_rows[0].get("diagnostic_schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema inventory must report version {SCHEMA_VERSION}")
    if _parse_int(schema_rows[0].get("episode_count"), "episode_count") != 1200:
        raise ValueError("schema inventory must report 1,200 episodes")

    _, missing_field_rows = _parse_csv(
        payloads["summaries/missing_field_inventory.csv"], "missing-field inventory"
    )
    if missing_field_rows:
        raise ValueError("missing-field inventory must be empty")

    _, paired_rows = _parse_csv(
        payloads["summaries/seed_level_paired_summary.csv"], "seed-level paired summary"
    )
    expected_pairs = {
        (scale, seed)
        for scale in ("25cp", "100cp", "500cp", "1000cp")
        for seed in TRAINING_SEEDS
    }
    observed_pairs: set[tuple[str, int]] = set()
    for row in paired_rows:
        pair = (
            row.get("scale", ""),
            _parse_int(row.get("training_seed"), "training_seed"),
        )
        if pair in observed_pairs:
            raise ValueError(f"duplicate seed-level paired summary row: {pair}")
        observed_pairs.add(pair)
        if row.get("status") != "ok":
            raise ValueError(f"seed-level paired summary status must be ok: {pair}")
    if observed_pairs != expected_pairs:
        raise ValueError("seed-level paired summary must contain 20 exact scale-seed rows")

    if payloads["validation/failure_manifest.txt"]:
        raise ValueError("failure manifest must be empty")
    if payloads["validation/warning_inventory.txt"]:
        raise ValueError("warning inventory must be empty")

    return {
        "status": "ok",
        "bundle": str(bundle),
        "array_job_id": array_job_id,
        "reducer_job_id": reducer_job_id,
        "source_commit_sha": source_sha,
        "task_packages": len(task_results),
        "checkpoint_groups": len(checkpoint_groups),
        "episodes": len(episode_rows),
        "diagnostic_schema_version": SCHEMA_VERSION,
    }
