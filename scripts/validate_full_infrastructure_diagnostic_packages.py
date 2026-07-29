#!/usr/bin/env python3
"""Strict package validation for the full per-infrastructure diagnostic eval30 workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


TRAINING_SEEDS = (0, 1, 2, 3, 4)
EVAL_EPISODES = 30
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
SEED_OFFSETS = {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}
CHECKPOINT_MARKERS = ("model.best", "_actor", "_critic", "optimizer", ".pt", ".pth")
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
INTEGER_RE = re.compile(r"^[0-9]+$")

TASK_REQUIRED_FILES = {
    "runtime_metadata/task_metadata.json",
    "runtime_metadata/package_file_checksums.sha256",
    "summaries/checkpoint_inventory.csv",
    "summaries/episode_inventory.csv",
    "summaries/runtime_summary.csv",
    "summaries/validation_summary.json",
    "logs/stderr.log",
}
COMPLETE_REQUIRED_FILES = {
    "summaries/checkpoint_inventory.csv",
    "summaries/episode_inventory.csv",
    "summaries/schema_inventory.csv",
    "summaries/reconciliation_inventory.csv",
    "summaries/missing_field_inventory.csv",
    "summaries/seed_level_paired_summary.csv",
    "summaries/failure_inventory.csv",
    "summaries/warning_inventory.csv",
    "runtime_metadata/source_commit_sha.txt",
    "runtime_metadata/array_job_id.txt",
    "runtime_metadata/reducer_job_id.txt",
    "runtime_metadata/complete_file_checksums.sha256",
}


class PackageValidationError(ValueError):
    """Raised when a Stage D evidence package violates its integrity contract."""


def _integer(value: object, label: str) -> int:
    text = str(value).strip()
    if not INTEGER_RE.fullmatch(text):
        raise PackageValidationError(f"invalid integer for {label}: {value!r}")
    return int(text)


def _task(task_id: int) -> tuple[str, str, int]:
    if type(task_id) is not int or task_id not in TASKS:
        raise PackageValidationError(f"unsupported task ID: {task_id!r}")
    return TASKS[task_id]


def expected_episode_keys(task_id: int | None = None) -> tuple[tuple[str, str, int, int], ...]:
    task_ids = range(8) if task_id is None else (task_id,)
    keys = []
    for current_task_id in task_ids:
        scale, algorithm, _ = _task(current_task_id)
        for training_seed in TRAINING_SEEDS:
            for episode_index in range(EVAL_EPISODES):
                keys.append((scale, algorithm, training_seed, episode_index))
    return tuple(keys)


def validate_safe_tar_members(archive: tarfile.TarFile) -> tuple[tarfile.TarInfo, ...]:
    seen = set()
    validated = []
    for member in archive.getmembers():
        name = member.name
        path = PurePosixPath(name)
        if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise PackageValidationError(f"unsafe tar path: {name!r}")
        if name in seen:
            raise PackageValidationError(f"duplicate tar member: {name}")
        seen.add(name)
        if not (member.isfile() or member.isdir()):
            raise PackageValidationError(f"unsupported tar member type: {name}")
        validated.append(member)
    return tuple(validated)


def _regular_files(archive: tarfile.TarFile) -> dict[str, bytes]:
    files = {}
    for member in validate_safe_tar_members(archive):
        if not member.isfile():
            continue
        extracted = archive.extractfile(member)
        if extracted is None:
            raise PackageValidationError(f"unable to read tar member: {member.name}")
        files[member.name] = extracted.read()
    return files


def _verify_manifest(files: dict[str, bytes], manifest_name: str) -> None:
    if manifest_name not in files:
        raise PackageValidationError(f"missing checksum manifest: {manifest_name}")
    expected_names = set(files) - {manifest_name}
    observed_names = set()
    for line_number, raw_line in enumerate(files[manifest_name].decode("utf-8").splitlines(), start=1):
        parts = raw_line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise PackageValidationError(f"malformed checksum manifest line {line_number}")
        digest, name = parts
        if name in observed_names:
            raise PackageValidationError(f"duplicate checksum entry: {name}")
        observed_names.add(name)
        if name not in files:
            raise PackageValidationError(f"checksum references missing member: {name}")
        actual = hashlib.sha256(files[name]).hexdigest()
        if actual != digest:
            raise PackageValidationError(f"checksum mismatch for {name}")
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise PackageValidationError(f"checksum coverage mismatch: missing={missing} extra={extra}")


def _csv_rows(payload: bytes, name: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackageValidationError(f"invalid UTF-8 CSV: {name}") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise PackageValidationError(f"CSV has no header: {name}")
    return list(reader.fieldnames), list(reader)


def _json_object(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError(f"invalid JSON: {name}") from exc
    if not isinstance(value, dict):
        raise PackageValidationError(f"JSON must be an object: {name}")
    return value


def _require_files(files: dict[str, bytes], required: set[str]) -> None:
    missing = sorted(required - set(files))
    if missing:
        raise PackageValidationError(f"missing required package member(s): {', '.join(missing)}")


def _validate_checkpoint_rows(rows: list[dict[str, str]], task_id: int | None) -> set[tuple[int, int]]:
    expected_count = 40 if task_id is None else 5
    count_phrase = "exactly 40 checkpoint groups" if task_id is None else "exactly five checkpoint groups"
    if len(rows) != expected_count:
        raise PackageValidationError(f"{count_phrase}; observed {len(rows)}")
    seen_pairs = set()
    seeds_by_task: dict[int, set[int]] = {}
    for index, row in enumerate(rows):
        observed_task_id = _integer(row.get("task_id", ""), "task_id")
        scale, algorithm, formal_start = _task(observed_task_id)
        if task_id is not None and observed_task_id != task_id:
            raise PackageValidationError(f"task identity mismatch at checkpoint row {index}")
        if row.get("scale") != scale or row.get("algorithm") != algorithm:
            raise PackageValidationError(f"task identity mismatch at checkpoint row {index}")
        seed = _integer(row.get("training_seed", ""), "training_seed")
        if seed not in TRAINING_SEEDS:
            raise PackageValidationError(f"unexpected training seed: {seed}")
        task_seeds = seeds_by_task.setdefault(observed_task_id, set())
        if seed in task_seeds:
            raise PackageValidationError(f"duplicate training seed for task {observed_task_id}: {seed}")
        task_seeds.add(seed)
        formal_task_id = _integer(row.get("formal_task_id", ""), "formal_task_id")
        if formal_task_id != formal_start + seed:
            raise PackageValidationError(f"formal task ID mismatch for task {observed_task_id} seed {seed}")
        if row.get("status") != "ok":
            raise PackageValidationError(f"checkpoint group status is not ok at row {index}")
        pair = (observed_task_id, seed)
        if pair in seen_pairs:
            raise PackageValidationError(f"duplicate checkpoint group: {pair}")
        seen_pairs.add(pair)
    expected_tasks = {task_id} if task_id is not None else set(range(8))
    if set(seeds_by_task) != expected_tasks or any(seeds != set(TRAINING_SEEDS) for seeds in seeds_by_task.values()):
        raise PackageValidationError("checkpoint seed coverage mismatch")
    return seen_pairs


def _validate_episode_rows(rows: list[dict[str, str]], task_id: int | None) -> set[tuple[str, str, int, int]]:
    expected_count = 1200 if task_id is None else 150
    count_phrase = "exactly 1,200 episode keys" if task_id is None else "exactly 150 episode keys"
    if len(rows) != expected_count:
        raise PackageValidationError(f"{count_phrase}; observed {len(rows)}")
    observed = set()
    for index, row in enumerate(rows):
        observed_task_id = _integer(row.get("task_id", ""), "task_id")
        scale, algorithm, _ = _task(observed_task_id)
        if task_id is not None and observed_task_id != task_id:
            raise PackageValidationError(f"task identity mismatch at episode row {index}")
        if row.get("scale") != scale or row.get("algorithm") != algorithm:
            raise PackageValidationError(f"task identity mismatch at episode row {index}")
        seed = _integer(row.get("training_seed", ""), "training_seed")
        episode_index = _integer(row.get("episode_index", ""), "episode_index")
        episode_seed = _integer(row.get("episode_seed", ""), "episode_seed")
        schema = _integer(row.get("diagnostic_schema_version", ""), "diagnostic_schema_version")
        if seed not in TRAINING_SEEDS or not 0 <= episode_index < EVAL_EPISODES:
            raise PackageValidationError(f"unexpected episode identity at row {index}")
        if episode_seed != SEED_OFFSETS[scale] + seed + episode_index:
            raise PackageValidationError(f"episode seed mismatch at row {index}")
        if schema != 3:
            raise PackageValidationError(f"diagnostic schema version must equal 3 at row {index}")
        key = (scale, algorithm, seed, episode_index)
        if key in observed:
            raise PackageValidationError(f"duplicate episode key: {key}")
        observed.add(key)
    expected = set(expected_episode_keys(task_id))
    if observed != expected:
        raise PackageValidationError("episode inventory identity mismatch")
    return observed


def validate_stage_d_task_package(path: str | Path, expected_task_id: int | None = None) -> dict[str, object]:
    package_path = Path(path)
    if expected_task_id is not None:
        _task(expected_task_id)
    try:
        with tarfile.open(package_path, "r:gz") as archive:
            files = _regular_files(archive)
    except (OSError, tarfile.TarError) as exc:
        raise PackageValidationError(f"unable to read task package: {package_path}") from exc
    _require_files(files, TASK_REQUIRED_FILES)
    _verify_manifest(files, "runtime_metadata/package_file_checksums.sha256")
    leaked = sorted(name for name in files if any(marker in name.lower() for marker in CHECKPOINT_MARKERS))
    if leaked:
        raise PackageValidationError(f"checkpoint leakage detected: {', '.join(leaked)}")
    metadata = _json_object(files["runtime_metadata/task_metadata.json"], "task metadata")
    task_id = _integer(metadata.get("task_id", ""), "task metadata task_id")
    _task(task_id)
    if expected_task_id is not None and task_id != expected_task_id:
        raise PackageValidationError(f"task identity mismatch: {task_id} != {expected_task_id}")
    if metadata.get("status") != "ok" or _integer(metadata.get("diagnostic_schema_version", ""), "task schema") != 3:
        raise PackageValidationError("task metadata status/schema mismatch")
    checkpoint_rows = _csv_rows(files["summaries/checkpoint_inventory.csv"], "checkpoint inventory")[1]
    episode_rows = _csv_rows(files["summaries/episode_inventory.csv"], "episode inventory")[1]
    _validate_checkpoint_rows(checkpoint_rows, task_id)
    episodes = _validate_episode_rows(episode_rows, task_id)
    validation = _json_object(files["summaries/validation_summary.json"], "validation summary")
    if validation.get("status") != "ok":
        raise PackageValidationError("validation summary status is not ok")
    if files["logs/stderr.log"] != b"":
        raise PackageValidationError("task stderr must be empty")
    return {"task_id": task_id, "checkpoint_groups": 5, "episode_keys": len(episodes), "status": "ok"}


def validate_stage_d_complete_bundle(path: str | Path) -> dict[str, object]:
    bundle_path = Path(path)
    try:
        with tarfile.open(bundle_path, "r:gz") as archive:
            files = _regular_files(archive)
    except (OSError, tarfile.TarError) as exc:
        raise PackageValidationError(f"unable to read complete bundle: {bundle_path}") from exc
    _require_files(files, COMPLETE_REQUIRED_FILES)
    _verify_manifest(files, "runtime_metadata/complete_file_checksums.sha256")
    package_names = sorted(name for name in files if re.fullmatch(r"task_packages/task[0-9]+\.tar\.gz", name))
    if len(package_names) != 8:
        raise PackageValidationError(f"complete bundle must contain exactly eight task packages; observed {len(package_names)}")
    expected_names = {f"task_packages/task{task_id}.tar.gz" for task_id in range(8)}
    if set(package_names) != expected_names:
        raise PackageValidationError("complete bundle task-package identity mismatch")
    for task_id in range(8):
        with tarfile.open(fileobj=io.BytesIO(files[f"task_packages/task{task_id}.tar.gz"]), mode="r:gz") as nested:
            nested_files = _regular_files(nested)
        temporary_members = []
        for name, payload in nested_files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            temporary_members.append((info, payload))
        temp_stream = io.BytesIO()
        with tarfile.open(fileobj=temp_stream, mode="w:gz") as rebuilt:
            for info, payload in temporary_members:
                rebuilt.addfile(info, io.BytesIO(payload))
        temp_stream.seek(0)
        with tarfile.open(fileobj=temp_stream, mode="r:gz") as nested_rebuilt:
            nested_files = _regular_files(nested_rebuilt)
        _require_files(nested_files, TASK_REQUIRED_FILES)
        _verify_manifest(nested_files, "runtime_metadata/package_file_checksums.sha256")
        metadata = _json_object(nested_files["runtime_metadata/task_metadata.json"], "nested task metadata")
        if _integer(metadata.get("task_id", ""), "nested task_id") != task_id:
            raise PackageValidationError(f"nested task identity mismatch for task {task_id}")
        _validate_checkpoint_rows(_csv_rows(nested_files["summaries/checkpoint_inventory.csv"], "nested checkpoint inventory")[1], task_id)
        _validate_episode_rows(_csv_rows(nested_files["summaries/episode_inventory.csv"], "nested episode inventory")[1], task_id)
        if nested_files["logs/stderr.log"] != b"":
            raise PackageValidationError(f"nested task stderr must be empty for task {task_id}")
    checkpoints = _validate_checkpoint_rows(_csv_rows(files["summaries/checkpoint_inventory.csv"], "complete checkpoint inventory")[1], None)
    episodes = _validate_episode_rows(_csv_rows(files["summaries/episode_inventory.csv"], "complete episode inventory")[1], None)
    if not SHA1_RE.fullmatch(files["runtime_metadata/source_commit_sha.txt"].decode().strip()):
        raise PackageValidationError("invalid source commit SHA")
    array_job_id = files["runtime_metadata/array_job_id.txt"].decode().strip()
    reducer_job_id = files["runtime_metadata/reducer_job_id.txt"].decode().strip()
    if not INTEGER_RE.fullmatch(array_job_id) or not INTEGER_RE.fullmatch(reducer_job_id):
        raise PackageValidationError("array/reducer job IDs must be numeric")
    return {
        "task_packages": 8,
        "checkpoint_groups": len(checkpoints),
        "episode_keys": len(episodes),
        "array_job_id": array_job_id,
        "reducer_job_id": reducer_job_id,
        "status": "ok",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    task_parser = subparsers.add_parser("validate-task-package")
    task_parser.add_argument("--bundle", required=True)
    task_parser.add_argument("--task-id", required=True, type=int)
    complete_parser = subparsers.add_parser("validate-complete-bundle")
    complete_parser.add_argument("--bundle", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-task-package":
            result = validate_stage_d_task_package(args.bundle, expected_task_id=args.task_id)
        else:
            result = validate_stage_d_complete_bundle(args.bundle)
    except PackageValidationError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
