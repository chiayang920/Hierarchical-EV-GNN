"""Extract validated EV infrastructure diagnostic datasets from evidence archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


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
