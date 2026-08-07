#!/usr/bin/env python3
"""Read-only schema validation for supplied historical EV-GNN evidence bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import tarfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


# Support direct execution by absolute path from any working directory.
if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import formal_75k_80cell_artifacts as artifacts


HISTORICAL_FORMAL_JOB = "58513929"
HISTORICAL_FORMAL_REDUCER = "58513930"
HISTORICAL_DIAGNOSTIC_JOB = "58745233"
HISTORICAL_DIAGNOSTIC_REDUCER = "58746039"
HISTORICAL_DIAGNOSTIC_SOURCE = "cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad"
HISTORICAL_DIAGNOSTIC_SHA256 = "be68f1aae52f06c90dbe79b2fbf25a2e5601c501464ff6fbfd17692ba7d7ccd7"
SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
SEEDS = tuple(range(5))


@dataclass(frozen=True)
class HistoricalEvidenceResult:
    status: str
    formal_checkpoint_count: int
    formal_episode_count: int
    diagnostic_task_package_count: int
    diagnostic_checkpoint_count: int
    diagnostic_episode_count: int
    diagnostic_schema_version: int
    reconciliation_contract_version: int
    formal_job_id: str
    diagnostic_job_id: str
    diagnostic_sha256: str
    historical_stage_d_unchanged: bool


def _parse_csv(payload: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"{label} has no header")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{label} has no rows")
    return rows


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _require(members: Mapping[str, bytes], name: str) -> bytes:
    try:
        return members[name]
    except KeyError as exc:
        raise ValueError(f"historical evidence member missing: {name}") from exc


def _as_int(value: object, label: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"-?[0-9]+", text):
        raise ValueError(f"{label} must be an integer; got {value!r}")
    return int(text)


def _as_float(value: object, label: str) -> float:
    try:
        result = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite; got {value!r}")
    return result


def _eq(actual: object, expected: object, label: str) -> None:
    if str(actual).strip() != str(expected):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _expected_historical_cells() -> set[tuple[str, str, int]]:
    return {(scale, algorithm, seed) for scale in SCALES for algorithm in ALGORITHMS for seed in SEEDS}


def _validate_formal_bundle(path: Path) -> tuple[int, int]:
    members = artifacts._read_safe_tar(path)
    reducer = artifacts._parse_env(
        _require(members, "reducer_metadata/reducer_runtime_metadata.env"),
        "reducer_metadata/reducer_runtime_metadata.env",
    )
    _eq(reducer.get("array_job_id", ""), HISTORICAL_FORMAL_JOB, "historical formal array job")
    _eq(reducer.get("reducer_job_id", ""), HISTORICAL_FORMAL_REDUCER, "historical formal reducer job")
    _eq(reducer.get("package_count", ""), 40, "historical formal package_count")
    _eq(reducer.get("canonical_csv_count", ""), 40, "historical formal canonical_csv_count")
    _eq(reducer.get("aggregation_exit_status", ""), 0, "historical formal aggregation_exit_status")
    _require(members, "aggregation_outputs/controlled_multiscale_formal_summary.md")

    pattern = re.compile(
        r"^canonical_eval30_csv/(25cp|100cp|500cp|1000cp)_(actiongnn|hierarchical)_seed([0-4])_eval30\.csv$"
    )
    observed: set[tuple[str, str, int]] = set()
    episode_total = 0
    for name, payload in members.items():
        match = pattern.match(name)
        if not match:
            continue
        scale, algorithm, seed_text = match.groups()
        seed = int(seed_text)
        key = (scale, algorithm, seed)
        if key in observed:
            raise ValueError(f"duplicate historical canonical cell: {key}")
        rows = _parse_csv(payload, name)
        episodes = [row for row in rows if str(row.get("row_type", "")).strip() == "episode"]
        summaries = [row for row in rows if str(row.get("row_type", "")).strip() == "summary"]
        if len(episodes) != 30 or len(summaries) != 1 or len(rows) != 31:
            raise ValueError(f"{name}: expected 30 episode rows and one summary row")
        indexes = []
        for row in episodes:
            _eq(row.get("algorithm", ""), algorithm, f"{name} algorithm")
            _eq(row.get("seed", ""), seed, f"{name} seed")
            scale_token = scale.removesuffix("cp")
            config_text = str(row.get("config", ""))
            if f"PublicPST_{scale_token}" not in config_text:
                raise ValueError(f"{name}: config does not preserve scale identity")
            indexes.append(_as_int(row.get("episode_index", ""), f"{name} episode_index"))
            _as_float(row.get("episode_reward", ""), f"{name} episode_reward")
        if indexes != list(range(30)):
            raise ValueError(f"{name}: episode indexes must be exactly 0..29")
        stem = f"{scale}_{algorithm}_seed{seed}"
        _require(members, f"source_manifests/{stem}_source_manifest.sha256")
        validation = artifacts._parse_env(
            _require(members, f"task_metadata/{stem}_eval30_csv_validation.env"),
            f"task_metadata/{stem}_eval30_csv_validation.env",
        )
        _eq(validation.get("data_rows", ""), 31, f"{stem} data_rows")
        _eq(validation.get("episode_rows", ""), 30, f"{stem} episode_rows")
        _eq(validation.get("summary_rows", ""), 1, f"{stem} summary_rows")
        runtime = artifacts._parse_env(
            _require(members, f"task_metadata/{stem}_task_runtime_metadata.env"),
            f"task_metadata/{stem}_task_runtime_metadata.env",
        )
        _eq(runtime.get("slurm_array_job_id", ""), HISTORICAL_FORMAL_JOB, f"{stem} array job")
        _eq(runtime.get("scale", ""), scale, f"{stem} scale")
        _eq(runtime.get("algorithm", ""), algorithm, f"{stem} algorithm")
        _eq(runtime.get("seed", ""), seed, f"{stem} seed")
        _eq(runtime.get("training_exit_status", ""), 0, f"{stem} training_exit_status")
        _eq(runtime.get("evaluation_exit_status", ""), 0, f"{stem} evaluation_exit_status")
        observed.add(key)
        episode_total += len(episodes)
    if observed != _expected_historical_cells():
        missing = sorted(_expected_historical_cells() - observed)
        extra = sorted(observed - _expected_historical_cells())
        raise ValueError(f"historical formal inventory mismatch: missing={missing}, extra={extra}")
    if episode_total != 1200:
        raise ValueError(f"historical formal episode count must be 1200; observed {episode_total}")
    return len(observed), episode_total


def _validate_nested_diagnostic_package(payload: bytes, *, task_id: int, scale: str, algorithm: str) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            members: dict[str, bytes] = {}
            for member in archive.getmembers():
                name = artifacts._safe_member_name(member.name)
                if name in members:
                    raise ValueError(f"duplicate nested diagnostic member: {name}")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ValueError(f"unsafe nested diagnostic member: {name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported nested diagnostic member: {name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unreadable nested diagnostic member: {name}")
                members[name] = handle.read()
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"unreadable historical diagnostic task package {task_id}: {exc}") from exc
    source = _require(members, "runtime_metadata/source_commit_sha.txt").decode("utf-8").strip()
    _eq(source, HISTORICAL_DIAGNOSTIC_SOURCE, f"diagnostic task {task_id} source commit")
    task = _parse_json(_require(members, "task_metadata/task.json"), "task_metadata/task.json")
    _eq(task.get("task_id", ""), task_id, "nested task_id")
    _eq(task.get("scale", ""), scale, "nested scale")
    _eq(task.get("algorithm", ""), algorithm, "nested algorithm")
    task_validation = _parse_json(
        _require(members, "validation/task_validation.json"), "validation/task_validation.json"
    )
    _eq(task_validation.get("status", ""), "ok", "nested task validation status")
    _eq(task_validation.get("checkpoint_groups", ""), 5, "nested checkpoint groups")
    _eq(task_validation.get("episode_count", ""), 150, "nested episode count")
    for seed in SEEDS:
        required = (
            f"seed{seed}/diagnostics/episode_diagnostics.csv",
            f"seed{seed}/diagnostics/same_pass_canonical_eval30.csv",
            f"seed{seed}/validation/mapping_validation.json",
            f"seed{seed}/validation/same_pass_canonical_reconciliation.csv",
            f"seed{seed}/validation/service_reconciliation.csv",
            f"seed{seed}/runtime_metadata/reconciliation_summary.json",
        )
        for name in required:
            _require(members, name)
        summary = _parse_json(
            members[f"seed{seed}/runtime_metadata/reconciliation_summary.json"],
            f"seed{seed}/runtime_metadata/reconciliation_summary.json",
        )
        _eq(summary.get("status", ""), "ok", f"task {task_id} seed {seed} status")
        _eq(summary.get("hard_gate_status", ""), "pass", f"task {task_id} seed {seed} hard gate")
        _eq(summary.get("reconciliation_contract_version", ""), 2, f"task {task_id} seed {seed} contract")
        _eq(summary.get("mapping_validation_status", ""), "pass", f"task {task_id} seed {seed} mapping")
        _eq(summary.get("same_pass_metric_status", ""), "pass", f"task {task_id} seed {seed} same-pass")
        _eq(summary.get("service_reconciliation_status", ""), "pass", f"task {task_id} seed {seed} service")


def _validate_diagnostic_bundle(path: Path) -> tuple[int, int, int, int, int, str]:
    diagnostic_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    members = artifacts._read_safe_tar(path)
    complete = _parse_json(
        _require(members, "validation/complete_workflow_validation.json"),
        "validation/complete_workflow_validation.json",
    )
    expectations = {
        "array_job_id": HISTORICAL_DIAGNOSTIC_JOB,
        "formal_job_id": HISTORICAL_FORMAL_JOB,
        "task_package_count": 8,
        "checkpoint_group_count": 40,
        "episode_count": 1200,
        "schema_version": 3,
        "reconciliation_contract_version": 2,
        "status": "ok",
    }
    for field, expected in expectations.items():
        _eq(complete.get(field, ""), expected, f"complete diagnostic {field}")

    tasks = _parse_csv(_require(members, "summaries/task_inventory.csv"), "summaries/task_inventory.csv")
    checkpoints = _parse_csv(
        _require(members, "summaries/checkpoint_inventory.csv"), "summaries/checkpoint_inventory.csv"
    )
    episodes = _parse_csv(_require(members, "summaries/episode_inventory.csv"), "summaries/episode_inventory.csv")
    if len(tasks) != 8 or len(checkpoints) != 40 or len(episodes) != 1200:
        raise ValueError(
            f"historical diagnostic inventory mismatch: tasks={len(tasks)}, checkpoints={len(checkpoints)}, episodes={len(episodes)}"
        )
    expected_tasks = [(task_id, scale, algorithm) for task_id, (scale, algorithm) in enumerate(
        ( (scale, algorithm) for scale in SCALES for algorithm in ALGORITHMS )
    )]
    observed_tasks: list[tuple[int, str, str]] = []
    for row in tasks:
        task_id = _as_int(row.get("task_id", ""), "diagnostic task_id")
        scale = str(row.get("scale", "")).strip()
        algorithm = str(row.get("algorithm", "")).strip()
        observed_tasks.append((task_id, scale, algorithm))
        _eq(row.get("status", ""), "ok", f"diagnostic task {task_id} status")
        _eq(row.get("checkpoint_groups", ""), 5, f"diagnostic task {task_id} checkpoint_groups")
        _eq(row.get("episode_count", ""), 150, f"diagnostic task {task_id} episode_count")
        pattern = re.compile(
            rf"^task_packages/m3_full_infrastructure_diagnostic_eval30_{scale}_{algorithm}_seeds0-4_job{HISTORICAL_DIAGNOSTIC_JOB}_task{task_id}\.tar\.gz$"
        )
        package_matches = [name for name in members if pattern.match(name)]
        if len(package_matches) != 1:
            raise ValueError(f"diagnostic task {task_id} requires exactly one task package")
        _validate_nested_diagnostic_package(
            members[package_matches[0]], task_id=task_id, scale=scale, algorithm=algorithm
        )
    if observed_tasks != expected_tasks:
        raise ValueError(f"historical diagnostic task mapping mismatch: {observed_tasks}")

    expected_checkpoint_keys = {
        (scale, algorithm, seed) for scale in SCALES for algorithm in ALGORITHMS for seed in SEEDS
    }
    checkpoint_keys: set[tuple[str, str, int]] = set()
    for row in checkpoints:
        key = (
            str(row.get("scale", "")).strip(),
            str(row.get("algorithm", "")).strip(),
            _as_int(row.get("training_seed", ""), "training_seed"),
        )
        if key in checkpoint_keys:
            raise ValueError(f"duplicate historical diagnostic checkpoint: {key}")
        _eq(row.get("diagnostic_schema_version", ""), 3, f"checkpoint {key} schema")
        checkpoint_keys.add(key)
    if checkpoint_keys != expected_checkpoint_keys:
        raise ValueError("historical diagnostic checkpoint inventory is incomplete")

    episode_keys: set[tuple[str, str, int, int]] = set()
    for row in episodes:
        scale = str(row.get("scale", "")).strip()
        algorithm = str(row.get("algorithm", "")).strip()
        seed = _as_int(row.get("training_seed", ""), "episode training_seed")
        episode = _as_int(row.get("episode_index", ""), "episode_index")
        key = (scale, algorithm, seed, episode)
        if key in episode_keys:
            raise ValueError(f"duplicate historical diagnostic episode: {key}")
        if episode not in range(30):
            raise ValueError(f"historical diagnostic episode out of range: {key}")
        _eq(row.get("diagnostic_schema_version", ""), 3, f"episode {key} schema")
        episode_keys.add(key)
    expected_episode_keys = {
        (scale, algorithm, seed, episode)
        for scale in SCALES for algorithm in ALGORITHMS for seed in SEEDS for episode in range(30)
    }
    if episode_keys != expected_episode_keys:
        raise ValueError("historical diagnostic episode inventory is incomplete")
    return len(tasks), len(checkpoints), len(episodes), 3, 2, diagnostic_sha


def validate_historical_evidence(
    formal_bundle: Path | str,
    diagnostic_bundle: Path | str,
) -> HistoricalEvidenceResult:
    formal_path = Path(formal_bundle)
    diagnostic_path = Path(diagnostic_bundle)
    formal_checkpoints, formal_episodes = _validate_formal_bundle(formal_path)
    diagnostic_tasks, diagnostic_checkpoints, diagnostic_episodes, schema, contract, diagnostic_sha = (
        _validate_diagnostic_bundle(diagnostic_path)
    )
    return HistoricalEvidenceResult(
        status="PASS",
        formal_checkpoint_count=formal_checkpoints,
        formal_episode_count=formal_episodes,
        diagnostic_task_package_count=diagnostic_tasks,
        diagnostic_checkpoint_count=diagnostic_checkpoints,
        diagnostic_episode_count=diagnostic_episodes,
        diagnostic_schema_version=schema,
        reconciliation_contract_version=contract,
        formal_job_id=HISTORICAL_FORMAL_JOB,
        diagnostic_job_id=HISTORICAL_DIAGNOSTIC_JOB,
        diagnostic_sha256=diagnostic_sha,
        historical_stage_d_unchanged=True,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-bundle", type=Path, required=True)
    parser.add_argument("--diagnostic-bundle", type=Path, required=True)
    parser.add_argument("--require-authoritative-diagnostic-sha", action="store_true")
    args = parser.parse_args(argv)
    result = validate_historical_evidence(args.formal_bundle, args.diagnostic_bundle)
    if args.require_authoritative_diagnostic_sha and result.diagnostic_sha256 != HISTORICAL_DIAGNOSTIC_SHA256:
        raise ValueError(
            "historical Stage D SHA-256 mismatch: "
            f"expected {HISTORICAL_DIAGNOSTIC_SHA256}, got {result.diagnostic_sha256}"
        )
    print(f"STATUS={result.status}")
    print(f"HISTORICAL_FORMAL_JOB={result.formal_job_id}")
    print(f"HISTORICAL_FORMAL_CHECKPOINTS={result.formal_checkpoint_count}")
    print(f"HISTORICAL_FORMAL_EPISODES={result.formal_episode_count}")
    print(f"HISTORICAL_STAGE_D_JOB={result.diagnostic_job_id}")
    print(f"HISTORICAL_STAGE_D_REDUCER={HISTORICAL_DIAGNOSTIC_REDUCER}")
    print(f"HISTORICAL_STAGE_D_TASK_PACKAGES={result.diagnostic_task_package_count}")
    print(f"HISTORICAL_STAGE_D_CHECKPOINTS={result.diagnostic_checkpoint_count}")
    print(f"HISTORICAL_STAGE_D_EPISODES={result.diagnostic_episode_count}")
    print(f"HISTORICAL_STAGE_D_SCHEMA={result.diagnostic_schema_version}")
    print(f"HISTORICAL_STAGE_D_RECONCILIATION_CONTRACT={result.reconciliation_contract_version}")
    print(f"HISTORICAL_STAGE_D_SHA256={result.diagnostic_sha256}")
    print("HISTORICAL_STAGE_D_UNCHANGED=YES")
    print("HISTORICAL_STAGE_D_REPAIR_REQUIRED=NO")
    print("HISTORICAL_STAGE_D_RERUN_REQUIRED=NO")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except ValueError as exc:
        print("STATUS=BLOCKED")
        print(f"BLOCK_REASON={exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
