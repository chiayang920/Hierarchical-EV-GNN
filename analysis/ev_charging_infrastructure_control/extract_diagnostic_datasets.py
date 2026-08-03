"""Extract validated EV infrastructure diagnostic datasets from evidence archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from analysis.ev_charging_infrastructure_control.metric_definitions import (
    METRIC_DEFINITIONS,
)
from utils.infrastructure_diagnostics import (
    CHARGER_DIAGNOSTIC_COLUMNS,
    EPISODE_DIAGNOSTIC_COLUMNS,
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
)


SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
TRAINING_SEEDS = tuple(range(5))
DIAGNOSTIC_SCHEMA_VERSION = "3"
RECONCILIATION_CONTRACT_VERSION = 2
EXPECTED_TASK_PACKAGE_COUNT = 8
EXPECTED_CHECKPOINT_GROUP_COUNT = 40
EXPECTED_TOTAL_EPISODE_COUNT = 1200
EXPECTED_PACKAGE_EPISODE_COUNT = 150
CHECKSUM_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([^\n\r]+)$")
TOPOLOGY = {
    "25cp": (3, 25),
    "100cp": (7, 100),
    "500cp": (35, 500),
    "1000cp": (70, 1000),
}
EPISODE_OUTPUT_COLUMNS = tuple(
    column
    for column in EPISODE_DIAGNOSTIC_COLUMNS
    if column not in {"config", "checkpoint_prefix", "run_name"}
)
APPROVED_EPISODE_METRIC_COLUMNS = tuple(
    definition.source_column
    for definition in METRIC_DEFINITIONS
    if definition.source_level == "episode"
)
APPROVED_TRANSFORMER_METRIC_COLUMNS = tuple(
    definition.source_column
    for definition in METRIC_DEFINITIONS
    if definition.source_level == "transformer"
)
APPROVED_SEED_SUMMARY_COLUMNS = tuple(
    definition.seed_summary_column
    for definition in METRIC_DEFINITIONS
    if definition.seed_summary_column is not None
)


@dataclass(frozen=True)
class Provenance:
    diagnostic_array_job_id: str
    reducer_job_id: str
    formal_training_job_id: str
    source_commit_sha: str
    evidence_bundle_sha256: str
    diagnostic_schema_version: str
    reconciliation_contract_version: int


@dataclass(frozen=True)
class TaskPackage:
    task_id: int
    scale: str
    algorithm: str
    package_name: str
    package_sha256: str
    archive_bytes: bytes


@dataclass(frozen=True)
class CompleteBundleEvidence:
    provenance: Provenance
    task_packages: tuple[TaskPackage, ...]


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_safe_archive_members(members: list[tarfile.TarInfo]) -> None:
    for member in members:
        path = PurePosixPath(member.name)
        if (
            not member.name
            or "\\" in member.name
            or path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            fail(f"unsafe archive member: {member.name!r}")


def read_checksum_manifest(manifest_bytes: bytes) -> dict[str, str]:
    manifest: dict[str, str] = {}
    text = manifest_bytes.decode("utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        match = CHECKSUM_LINE_RE.match(raw_line)
        if match is None:
            fail(f"malformed checksum manifest line {line_number}")
        digest, member_name = match.groups()
        if member_name in manifest:
            fail(f"duplicate checksum manifest entry: {member_name}")
        manifest[member_name] = digest
    return manifest


def validate_manifest_coverage(
    regular_files: dict[str, bytes],
    manifest_name: str,
    checksum_label: str,
) -> dict[str, str]:
    require(
        manifest_name in regular_files,
        f"missing checksum manifest: {manifest_name}",
    )
    manifest = read_checksum_manifest(regular_files[manifest_name])
    if manifest_name in manifest:
        fail("checksum manifest must not include itself")

    expected_member_names = set(regular_files) - {manifest_name}
    manifest_member_names = set(manifest)
    missing = sorted(expected_member_names - manifest_member_names)
    extra = sorted(manifest_member_names - expected_member_names)
    if missing:
        fail(
            "regular file not covered by checksum manifest: "
            + ", ".join(missing)
        )
    if extra:
        fail("checksum manifest references missing member: " + ", ".join(extra))

    for member_name in sorted(expected_member_names):
        observed_digest = sha256_bytes(regular_files[member_name])
        expected_digest = manifest[member_name]
        if observed_digest != expected_digest:
            fail(
                f"{checksum_label} SHA-256 mismatch for {member_name}: "
                f"expected {expected_digest}, observed {observed_digest}"
            )
    return manifest


def read_tar_regular_files(archive_bytes: bytes, archive_label: str) -> dict[str, bytes]:
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = archive.getmembers()
            validate_safe_archive_members(members)
            regular_files: dict[str, bytes] = {}
            for member in members:
                if member.isdir():
                    continue
                require(member.isfile(), f"unsupported archive member type: {member.name}")
                extracted = archive.extractfile(member)
                require(
                    extracted is not None,
                    f"unable to read archive member: {member.name}",
                )
                regular_files[member.name] = extracted.read()
    except tarfile.TarError as exc:
        raise RuntimeError(f"{archive_label} archive integrity failure") from exc
    return regular_files


def read_required_text(regular_files: dict[str, bytes], member_name: str) -> str:
    require(member_name in regular_files, f"missing required member: {member_name}")
    return regular_files[member_name].decode("utf-8").strip()


def read_required_json(regular_files: dict[str, bytes], member_name: str) -> dict[str, object]:
    require(member_name in regular_files, f"missing required member: {member_name}")
    parsed = json.loads(regular_files[member_name].decode("utf-8"))
    require(isinstance(parsed, dict), f"expected JSON object in {member_name}")
    return parsed


def read_task_inventory(regular_files: dict[str, bytes]) -> list[dict[str, str]]:
    member_name = "summaries/task_inventory.csv"
    require(member_name in regular_files, f"missing required member: {member_name}")
    text_stream = io.StringIO(regular_files[member_name].decode("utf-8"))
    reader = csv.DictReader(text_stream)
    required_columns = {
        "task_id",
        "scale",
        "algorithm",
        "package_name",
        "package_sha256",
        "status",
        "checkpoint_groups",
        "episode_count",
    }
    fieldnames = set(reader.fieldnames or ())
    missing_columns = sorted(required_columns - fieldnames)
    require(
        not missing_columns,
        "task inventory missing required column(s): " + ", ".join(missing_columns),
    )
    return list(reader)


def parse_int(text: str, context: str) -> int:
    try:
        return int(str(text).strip())
    except ValueError as exc:
        raise RuntimeError(f"{context}: expected integer, found {text!r}") from exc


def parse_finite_float(text: object, context: str) -> float:
    raw_text = "" if text is None else str(text).strip()
    if raw_text == "":
        fail(f"{context}: missing numeric value")
    try:
        value = float(raw_text)
    except ValueError as exc:
        raise RuntimeError(f"{context}: invalid numeric value {raw_text!r}") from exc
    if not math.isfinite(value):
        fail(f"{context}: non-finite approved metric")
    return value


def validate_workflow_identity(
    workflow: dict[str, object],
    *,
    diagnostic_array_job_id: str,
    formal_training_job_id: str,
) -> tuple[str, int]:
    expected = {
        "status": "ok",
        "array_job_id": diagnostic_array_job_id,
        "task_package_count": EXPECTED_TASK_PACKAGE_COUNT,
        "checkpoint_group_count": EXPECTED_CHECKPOINT_GROUP_COUNT,
        "episode_count": EXPECTED_TOTAL_EPISODE_COUNT,
        "formal_job_id": formal_training_job_id,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
    }
    observed = dict(workflow)
    if "schema_version" in observed:
        observed["schema_version"] = str(observed["schema_version"])
    for field_name, expected_value in expected.items():
        if observed.get(field_name) != expected_value:
            fail(
                f"workflow identity mismatch for {field_name}: "
                f"expected {expected_value!r}, observed {observed.get(field_name)!r}"
            )
    return str(observed["schema_version"]), int(
        observed["reconciliation_contract_version"]
    )


def read_complete_bundle(bundle_path: str | Path) -> CompleteBundleEvidence:
    resolved_bundle_path = Path(bundle_path).expanduser().resolve()
    bundle_bytes = resolved_bundle_path.read_bytes()
    evidence_bundle_sha256 = sha256_bytes(bundle_bytes)
    outer_files = read_tar_regular_files(bundle_bytes, "complete evidence bundle")
    validate_manifest_coverage(
        outer_files,
        "runtime_metadata/complete_file_checksums.sha256",
        "outer",
    )

    diagnostic_array_job_id = read_required_text(
        outer_files, "runtime_metadata/diagnostic_array_job_id.txt"
    )
    reducer_job_id = read_required_text(
        outer_files, "runtime_metadata/reducer_job_id.txt"
    )
    formal_training_job_id = read_required_text(
        outer_files, "runtime_metadata/formal_job_id.txt"
    )
    source_commit_sha = read_required_text(
        outer_files, "runtime_metadata/source_commit_sha.txt"
    )
    workflow = read_required_json(
        outer_files, "validation/complete_workflow_validation.json"
    )
    diagnostic_schema_version, reconciliation_contract_version = (
        validate_workflow_identity(
            workflow,
            diagnostic_array_job_id=diagnostic_array_job_id,
            formal_training_job_id=formal_training_job_id,
        )
    )

    inventory_rows = read_task_inventory(outer_files)
    require(
        len(inventory_rows) == EXPECTED_TASK_PACKAGE_COUNT,
        f"expected exactly 8 task packages, found {len(inventory_rows)}",
    )

    task_packages: list[TaskPackage] = []
    observed_task_ids: set[int] = set()
    observed_scale_algorithms: set[tuple[str, str]] = set()
    for inventory_row in inventory_rows:
        task_id = parse_int(inventory_row["task_id"], "task_id")
        require(task_id not in observed_task_ids, f"duplicate task_id: {task_id}")
        observed_task_ids.add(task_id)

        scale = inventory_row["scale"]
        algorithm = inventory_row["algorithm"]
        require(scale in SCALES, f"unsupported scale in task inventory: {scale}")
        require(
            algorithm in ALGORITHMS,
            f"unsupported algorithm in task inventory: {algorithm}",
        )
        observed_scale_algorithms.add((scale, algorithm))
        require(inventory_row["status"] == "ok", f"task {task_id} status is not ok")
        require(
            parse_int(inventory_row["checkpoint_groups"], "checkpoint_groups") == 5,
            f"task {task_id} must contain 5 checkpoint groups",
        )
        require(
            parse_int(inventory_row["episode_count"], "episode_count")
            == EXPECTED_PACKAGE_EPISODE_COUNT,
            f"task {task_id} must contain 150 episodes",
        )

        package_name = inventory_row["package_name"]
        package_member_name = f"task_packages/{package_name}"
        require(
            package_member_name in outer_files,
            f"missing task package member: {package_member_name}",
        )
        archive_bytes = outer_files[package_member_name]
        observed_package_sha256 = sha256_bytes(archive_bytes)
        expected_package_sha256 = inventory_row["package_sha256"]
        if observed_package_sha256 != expected_package_sha256:
            fail(
                f"task package SHA-256 mismatch for {package_name}: "
                f"expected {expected_package_sha256}, observed {observed_package_sha256}"
            )

        nested_files = read_tar_regular_files(archive_bytes, package_name)
        validate_manifest_coverage(
            nested_files,
            "checksums/package_file_checksums.sha256",
            "nested",
        )
        task_packages.append(
            TaskPackage(
                task_id=task_id,
                scale=scale,
                algorithm=algorithm,
                package_name=package_name,
                package_sha256=expected_package_sha256,
                archive_bytes=archive_bytes,
            )
        )

    require(
        observed_task_ids == set(range(EXPECTED_TASK_PACKAGE_COUNT)),
        "task inventory must contain task IDs 0-7 exactly once",
    )
    expected_scale_algorithms = {
        (scale, algorithm) for scale in SCALES for algorithm in ALGORITHMS
    }
    require(
        observed_scale_algorithms == expected_scale_algorithms,
        "task inventory must contain all 4 scales x 2 algorithms exactly once",
    )

    provenance = Provenance(
        diagnostic_array_job_id=diagnostic_array_job_id,
        reducer_job_id=reducer_job_id,
        formal_training_job_id=formal_training_job_id,
        source_commit_sha=source_commit_sha,
        evidence_bundle_sha256=evidence_bundle_sha256,
        diagnostic_schema_version=diagnostic_schema_version,
        reconciliation_contract_version=reconciliation_contract_version,
    )
    return CompleteBundleEvidence(
        provenance=provenance,
        task_packages=tuple(sorted(task_packages, key=lambda task: task.task_id)),
    )


def csv_rows_from_member(
    regular_files: dict[str, bytes],
    member_name: str,
    required_columns: tuple[str, ...] | list[str],
) -> list[dict[str, str]]:
    require(member_name in regular_files, f"missing required member: {member_name}")
    text_stream = io.StringIO(regular_files[member_name].decode("utf-8"))
    reader = csv.DictReader(text_stream)
    fieldnames = tuple(reader.fieldnames or ())
    missing_columns = [column for column in required_columns if column not in fieldnames]
    if missing_columns:
        fail(
            f"{member_name}: missing required column(s): "
            + ", ".join(missing_columns)
        )
    return list(reader)


def validate_row_identity(
    row: dict[str, str],
    *,
    task_package: TaskPackage,
    provenance: Provenance,
    training_seed: int,
    member_name: str,
) -> None:
    if row.get("scale") != task_package.scale:
        fail(f"{member_name}: scale mismatch")
    if row.get("algorithm") != task_package.algorithm:
        fail(f"{member_name}: algorithm mismatch")
    if parse_int(row.get("training_seed", ""), f"{member_name}: training_seed") != training_seed:
        fail(f"{member_name}: training_seed mismatch")
    if row.get("matrix_job_id") != provenance.diagnostic_array_job_id:
        fail(f"{member_name}: matrix_job_id mismatch")
    if str(row.get("diagnostic_schema_version", "")) != DIAGNOSTIC_SCHEMA_VERSION:
        fail(f"{member_name}: diagnostic schema version mismatch")


def validate_finite_columns(
    row: dict[str, str],
    columns: tuple[str, ...],
    member_name: str,
) -> None:
    for column in columns:
        parse_finite_float(row.get(column), f"{member_name}: {column}")


def output_row(row: dict[str, str], fieldnames: tuple[str, ...]) -> dict[str, str]:
    return {fieldname: row.get(fieldname, "") for fieldname in fieldnames}


def expected_dataset_counts(expected_episodes_per_seed: int) -> dict[str, int]:
    scale_count = len(SCALES)
    algorithm_count = len(ALGORITHMS)
    seed_count = len(TRAINING_SEEDS)
    transformer_count = sum(counts[0] for counts in TOPOLOGY.values())
    charger_count = sum(counts[1] for counts in TOPOLOGY.values())
    return {
        "episode_metrics": scale_count
        * algorithm_count
        * seed_count
        * expected_episodes_per_seed,
        "seed_metrics": scale_count * algorithm_count * seed_count,
        "transformer_metrics": algorithm_count
        * seed_count
        * expected_episodes_per_seed
        * transformer_count,
        "charger_metrics": algorithm_count
        * seed_count
        * expected_episodes_per_seed
        * charger_count,
    }


def validate_complete_matrix(
    episode_keys: set[tuple[str, str, int, int, int]],
    expected_episodes_per_seed: int,
) -> None:
    for scale in SCALES:
        for algorithm in ALGORITHMS:
            for training_seed in TRAINING_SEEDS:
                observed_episode_indices = {
                    episode_index
                    for (
                        observed_scale,
                        observed_algorithm,
                        observed_training_seed,
                        episode_index,
                        _episode_seed,
                    ) in episode_keys
                    if (
                        observed_scale,
                        observed_algorithm,
                        observed_training_seed,
                    )
                    == (scale, algorithm, training_seed)
                }
                expected_episode_indices = set(range(expected_episodes_per_seed))
                if observed_episode_indices != expected_episode_indices:
                    fail(
                        "episode indices mismatch for "
                        f"{scale}/{algorithm}/seed{training_seed}"
                    )

    for scale in SCALES:
        actiongnn_pairs = {
            (training_seed, episode_index, episode_seed)
            for (
                observed_scale,
                observed_algorithm,
                training_seed,
                episode_index,
                episode_seed,
            ) in episode_keys
            if observed_scale == scale and observed_algorithm == "actiongnn"
        }
        hierarchical_pairs = {
            (training_seed, episode_index, episode_seed)
            for (
                observed_scale,
                observed_algorithm,
                training_seed,
                episode_index,
                episode_seed,
            ) in episode_keys
            if observed_scale == scale and observed_algorithm == "hierarchical"
        }
        if actiongnn_pairs != hierarchical_pairs:
            fail(f"episode seed sets mismatch for scale {scale}")


def collect_validated_dataset_rows(
    evidence: CompleteBundleEvidence,
    expected_episodes_per_seed: int,
) -> dict[str, list[dict[str, str]]]:
    episode_rows_out: list[dict[str, str]] = []
    seed_rows_out: list[dict[str, str]] = []
    transformer_rows_out: list[dict[str, str]] = []
    charger_rows_out: list[dict[str, str]] = []

    episode_keys: set[tuple[str, str, int, int, int]] = set()
    seed_keys: set[tuple[str, str, int]] = set()
    transformer_keys: set[tuple[str, str, int, int, int, int]] = set()
    charger_keys: set[tuple[str, str, int, int, int, int, int]] = set()

    for task_package in evidence.task_packages:
        nested_files = read_tar_regular_files(
            task_package.archive_bytes, task_package.package_name
        )
        validate_manifest_coverage(
            nested_files,
            "checksums/package_file_checksums.sha256",
            "nested",
        )
        for training_seed in TRAINING_SEEDS:
            seed_prefix = f"seed{training_seed}/diagnostics"
            episode_member = f"{seed_prefix}/episode_diagnostics.csv"
            seed_member = f"{seed_prefix}/seed_summary_diagnostics.csv"
            transformer_member = f"{seed_prefix}/transformer_diagnostics.csv"
            charger_member = f"{seed_prefix}/charger_diagnostics.csv"

            episode_rows = csv_rows_from_member(
                nested_files, episode_member, EPISODE_DIAGNOSTIC_COLUMNS
            )
            for row in episode_rows:
                validate_row_identity(
                    row,
                    task_package=task_package,
                    provenance=evidence.provenance,
                    training_seed=training_seed,
                    member_name=episode_member,
                )
                validate_finite_columns(row, APPROVED_EPISODE_METRIC_COLUMNS, episode_member)
                episode_index = parse_int(row["episode_index"], f"{episode_member}: episode_index")
                episode_seed = parse_int(row["episode_seed"], f"{episode_member}: episode_seed")
                key = (
                    task_package.scale,
                    task_package.algorithm,
                    training_seed,
                    episode_index,
                    episode_seed,
                )
                if key in episode_keys:
                    fail(f"duplicate episode key: {key}")
                episode_keys.add(key)
                episode_rows_out.append(output_row(row, EPISODE_OUTPUT_COLUMNS))
            if len(episode_rows) != expected_episodes_per_seed:
                fail(
                    f"{episode_member}: expected {expected_episodes_per_seed} episodes, "
                    f"found {len(episode_rows)}"
                )

            seed_rows = csv_rows_from_member(
                nested_files, seed_member, SEED_SUMMARY_DIAGNOSTIC_COLUMNS
            )
            if len(seed_rows) != 1:
                fail(f"{seed_member}: expected exactly one seed summary row")
            seed_row = seed_rows[0]
            validate_row_identity(
                seed_row,
                task_package=task_package,
                provenance=evidence.provenance,
                training_seed=training_seed,
                member_name=seed_member,
            )
            if parse_int(seed_row["n_eval_episodes"], f"{seed_member}: n_eval_episodes") != expected_episodes_per_seed:
                fail(f"{seed_member}: n_eval_episodes mismatch")
            validate_finite_columns(seed_row, APPROVED_SEED_SUMMARY_COLUMNS, seed_member)
            seed_key = (task_package.scale, task_package.algorithm, training_seed)
            if seed_key in seed_keys:
                fail(f"duplicate seed key: {seed_key}")
            seed_keys.add(seed_key)
            seed_rows_out.append(output_row(seed_row, tuple(SEED_SUMMARY_DIAGNOSTIC_COLUMNS)))

            transformer_rows = csv_rows_from_member(
                nested_files, transformer_member, TRANSFORMER_DIAGNOSTIC_COLUMNS
            )
            for row in transformer_rows:
                validate_row_identity(
                    row,
                    task_package=task_package,
                    provenance=evidence.provenance,
                    training_seed=training_seed,
                    member_name=transformer_member,
                )
                validate_finite_columns(
                    row, APPROVED_TRANSFORMER_METRIC_COLUMNS, transformer_member
                )
                episode_index = parse_int(
                    row["episode_index"], f"{transformer_member}: episode_index"
                )
                episode_seed = parse_int(
                    row["episode_seed"], f"{transformer_member}: episode_seed"
                )
                transformer_id = parse_int(
                    row["transformer_id"], f"{transformer_member}: transformer_id"
                )
                key = (
                    task_package.scale,
                    task_package.algorithm,
                    training_seed,
                    episode_index,
                    episode_seed,
                    transformer_id,
                )
                if key in transformer_keys:
                    fail(f"duplicate transformer key: {key}")
                transformer_keys.add(key)
                transformer_rows_out.append(
                    output_row(row, tuple(TRANSFORMER_DIAGNOSTIC_COLUMNS))
                )

            charger_rows = csv_rows_from_member(
                nested_files, charger_member, CHARGER_DIAGNOSTIC_COLUMNS
            )
            for row in charger_rows:
                validate_row_identity(
                    row,
                    task_package=task_package,
                    provenance=evidence.provenance,
                    training_seed=training_seed,
                    member_name=charger_member,
                )
                episode_index = parse_int(
                    row["episode_index"], f"{charger_member}: episode_index"
                )
                episode_seed = parse_int(
                    row["episode_seed"], f"{charger_member}: episode_seed"
                )
                transformer_id = parse_int(
                    row["transformer_id"], f"{charger_member}: transformer_id"
                )
                charger_id = parse_int(row["charger_id"], f"{charger_member}: charger_id")
                key = (
                    task_package.scale,
                    task_package.algorithm,
                    training_seed,
                    episode_index,
                    episode_seed,
                    transformer_id,
                    charger_id,
                )
                if key in charger_keys:
                    fail(f"duplicate charger key: {key}")
                charger_keys.add(key)
                charger_rows_out.append(output_row(row, tuple(CHARGER_DIAGNOSTIC_COLUMNS)))

    validate_complete_matrix(episode_keys, expected_episodes_per_seed)
    rows_by_dataset = {
        "episode_metrics": episode_rows_out,
        "seed_metrics": seed_rows_out,
        "transformer_metrics": transformer_rows_out,
        "charger_metrics": charger_rows_out,
    }
    expected_counts = expected_dataset_counts(expected_episodes_per_seed)
    for dataset_name, rows in rows_by_dataset.items():
        if len(rows) != expected_counts[dataset_name]:
            fail(
                f"{dataset_name}: expected {expected_counts[dataset_name]} rows, "
                f"found {len(rows)}"
            )
    return rows_by_dataset


def write_csv_file(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_provenance_file(
    path: Path,
    provenance: Provenance,
    dataset_rows: dict[str, int],
    expected_episodes_per_seed: int,
) -> None:
    payload = {
        "diagnostic_array_job_id": provenance.diagnostic_array_job_id,
        "reducer_job_id": provenance.reducer_job_id,
        "formal_training_job_id": provenance.formal_training_job_id,
        "source_commit_sha": provenance.source_commit_sha,
        "evidence_bundle_sha256": provenance.evidence_bundle_sha256,
        "diagnostic_schema_version": provenance.diagnostic_schema_version,
        "reconciliation_contract_version": provenance.reconciliation_contract_version,
        "task_package_count": EXPECTED_TASK_PACKAGE_COUNT,
        "checkpoint_group_count": EXPECTED_CHECKPOINT_GROUP_COUNT,
        "episode_count": len(SCALES)
        * len(ALGORITHMS)
        * len(TRAINING_SEEDS)
        * expected_episodes_per_seed,
        "dataset_rows": dataset_rows,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def publish_dataset_outputs(
    output_dir: Path,
    evidence: CompleteBundleEvidence,
    rows_by_dataset: dict[str, list[dict[str, str]]],
    expected_episodes_per_seed: int,
) -> None:
    if output_dir.exists():
        fail(f"output directory already exists: {output_dir}")
    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_parent / f".ev_charging_infrastructure_control_analysis.tmp.{os.getpid()}"
    if temporary_dir.exists():
        fail(f"temporary output directory already exists: {temporary_dir}")
    dataset_dir = temporary_dir / "datasets"
    try:
        dataset_dir.mkdir(parents=True)
        write_csv_file(
            dataset_dir / "episode_metrics.csv",
            EPISODE_OUTPUT_COLUMNS,
            rows_by_dataset["episode_metrics"],
        )
        write_csv_file(
            dataset_dir / "seed_metrics.csv",
            tuple(SEED_SUMMARY_DIAGNOSTIC_COLUMNS),
            rows_by_dataset["seed_metrics"],
        )
        write_csv_file(
            dataset_dir / "transformer_metrics.csv",
            tuple(TRANSFORMER_DIAGNOSTIC_COLUMNS),
            rows_by_dataset["transformer_metrics"],
        )
        write_csv_file(
            dataset_dir / "charger_metrics.csv",
            tuple(CHARGER_DIAGNOSTIC_COLUMNS),
            rows_by_dataset["charger_metrics"],
        )
        dataset_counts = {
            dataset_name: len(rows) for dataset_name, rows in rows_by_dataset.items()
        }
        write_provenance_file(
            temporary_dir / "provenance.json",
            evidence.provenance,
            dataset_counts,
            expected_episodes_per_seed,
        )
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def extract_diagnostic_datasets(
    bundle_path: str | Path,
    output_dir: str | Path,
    *,
    expected_episodes_per_seed: int = 30,
) -> dict[str, int]:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if resolved_output_dir.exists():
        fail(f"output directory already exists: {resolved_output_dir}")
    evidence = read_complete_bundle(bundle_path)
    rows_by_dataset = collect_validated_dataset_rows(
        evidence, expected_episodes_per_seed
    )
    publish_dataset_outputs(
        resolved_output_dir,
        evidence,
        rows_by_dataset,
        expected_episodes_per_seed,
    )
    return {dataset_name: len(rows) for dataset_name, rows in rows_by_dataset.items()}
