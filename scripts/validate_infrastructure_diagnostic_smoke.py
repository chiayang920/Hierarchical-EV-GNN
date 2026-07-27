#!/usr/bin/env python3
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from argparse import Namespace
from pathlib import Path, PurePosixPath

import yaml


FORMAL_JOB_ID = "58513929"
SCHEMA_VERSION = "3"
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")

TASKS = {
    0: {
        "task_id": 0,
        "scale": "25cp",
        "algorithm": "actiongnn",
        "formal_task_id": 0,
        "episode_seed": 710000,
        "config_name": "PublicPST_25cp.yaml",
        "expected_charger_rows": 25,
        "expected_transformer_rows": 3,
    },
    1: {
        "task_id": 1,
        "scale": "25cp",
        "algorithm": "hierarchical",
        "formal_task_id": 5,
        "episode_seed": 710000,
        "config_name": "PublicPST_25cp.yaml",
        "expected_charger_rows": 25,
        "expected_transformer_rows": 3,
    },
    2: {
        "task_id": 2,
        "scale": "100cp",
        "algorithm": "actiongnn",
        "formal_task_id": 10,
        "episode_seed": 720000,
        "config_name": "PublicPST_100.yaml",
        "expected_charger_rows": 100,
        "expected_transformer_rows": 7,
    },
    3: {
        "task_id": 3,
        "scale": "100cp",
        "algorithm": "hierarchical",
        "formal_task_id": 15,
        "episode_seed": 720000,
        "config_name": "PublicPST_100.yaml",
        "expected_charger_rows": 100,
        "expected_transformer_rows": 7,
    },
    4: {
        "task_id": 4,
        "scale": "500cp",
        "algorithm": "actiongnn",
        "formal_task_id": 20,
        "episode_seed": 730000,
        "config_name": "PublicPST_500.yaml",
        "expected_charger_rows": 500,
        "expected_transformer_rows": 35,
    },
    5: {
        "task_id": 5,
        "scale": "500cp",
        "algorithm": "hierarchical",
        "formal_task_id": 25,
        "episode_seed": 730000,
        "config_name": "PublicPST_500.yaml",
        "expected_charger_rows": 500,
        "expected_transformer_rows": 35,
    },
    6: {
        "task_id": 6,
        "scale": "1000cp",
        "algorithm": "actiongnn",
        "formal_task_id": 30,
        "episode_seed": 740000,
        "config_name": "PublicPST_1000.yaml",
        "expected_charger_rows": 1000,
        "expected_transformer_rows": 70,
    },
    7: {
        "task_id": 7,
        "scale": "1000cp",
        "algorithm": "hierarchical",
        "formal_task_id": 35,
        "episode_seed": 740000,
        "config_name": "PublicPST_1000.yaml",
        "expected_charger_rows": 1000,
        "expected_transformer_rows": 70,
    },
}

FLOAT_RECONCILIATION = [
    ("episode_reward", "episode_reward", 1.0, 1e-7),
    ("tracking_error", "tracking_error", 1.0, 1e-7),
    ("energy_tracking_error", "energy_tracking_error", 1e-2, 1e-7),
    ("power_tracker_violation", "power_tracker_violation", 1e-2, 1e-7),
    ("total_energy_charged", "total_energy_charged", 1e-2, 1e-7),
    ("total_energy_discharged", "total_energy_discharged", 1e-6, 1e-7),
    ("average_user_satisfaction", "average_user_satisfaction", 1e-6, 1e-7),
    ("energy_user_satisfaction", "energy_user_satisfaction", 1e-6, 1e-7),
    ("total_transformer_overload", "total_transformer_overload", 1e-3, 1e-7),
    ("global_action_mean_all_slots", "action_mean", 1e-6, 0.0),
    ("global_action_fraction_at_max_all_slots", "action_fraction_at_max", 1e-4, 0.0),
    ("nonzero_action_count_mean_all_slots", "active_action_count_mean", 1e-6, 0.0),
]

EXACT_RECONCILIATION = [
    ("algorithm", "algorithm", "algorithm", "string"),
    ("training seed", "seed", "training_seed", "integer"),
    ("episode index", "episode_index", "episode_index", "integer"),
    ("episode seed", "episode_seed", "episode_seed", "integer"),
    ("episode steps", "episode_steps", "episode_steps", "integer"),
    ("done", "done", "done", "boolean"),
    ("total_ev_served", "total_ev_served", "total_ev_served", "integer"),
]

EPISODE_REQUIRED_COLUMNS = {
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "episode_steps",
    "done",
    "episode_reward",
    "v2g_enabled",
    "active_action_decision_count",
    "active_action_below_environment_low_count",
    "active_action_below_environment_low_fraction",
    "active_action_above_environment_high_count",
    "active_action_above_environment_high_fraction",
    "global_positive_action_fraction_active",
    "global_zero_action_fraction_active",
    "global_negative_action_fraction_active",
    "inactive_nonzero_action_count",
    "global_action_mean_all_slots",
    "global_action_fraction_at_max_all_slots",
    "nonzero_action_count_mean_all_slots",
    "total_transformer_overload",
    "power_tracker_violation",
    "tracking_error",
    "energy_tracking_error",
    "total_ev_served",
    "total_energy_charged",
    "total_energy_discharged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "diagnostic_schema_version",
}

SEED_SUMMARY_REQUIRED_COLUMNS = {
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "n_eval_episodes",
    "active_action_decision_count_mean",
    "active_action_below_environment_low_count",
    "active_action_below_environment_low_fraction",
    "active_action_above_environment_high_count",
    "active_action_above_environment_high_fraction",
    "global_positive_action_fraction_active_mean",
    "global_zero_action_fraction_active_mean",
    "global_negative_action_fraction_active_mean",
    "inactive_nonzero_action_count_mean",
    "v2g_enabled",
    "diagnostic_schema_version",
}

CHARGER_REQUIRED_COLUMNS = {
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "charger_id",
    "transformer_id",
    "n_active_ev_decisions",
    "positive_action_fraction_active",
    "zero_action_fraction_active",
    "negative_action_fraction_active",
    "served_ev_count",
    "energy_charged_kwh",
    "energy_discharged_kwh",
    "user_satisfaction_sum",
    "user_satisfaction_observation_count",
    "user_satisfaction_source",
    "diagnostic_schema_version",
}

TRANSFORMER_REQUIRED_COLUMNS = {
    "matrix_job_id",
    "scale",
    "algorithm",
    "training_seed",
    "episode_index",
    "episode_seed",
    "transformer_id",
    "n_active_ev_decisions",
    "positive_action_fraction_active",
    "zero_action_fraction_active",
    "negative_action_fraction_active",
    "served_ev_count",
    "energy_charged_kwh",
    "energy_discharged_kwh",
    "user_satisfaction_sum",
    "user_satisfaction_observation_count",
    "user_satisfaction_source",
    "diagnostic_schema_version",
}

DIAGNOSTIC_PACKAGE_CSVS = [
    "diagnostics/episode_diagnostics.csv",
    "diagnostics/seed_summary_diagnostics.csv",
    "diagnostics/transformer_diagnostics.csv",
    "diagnostics/charger_diagnostics.csv",
]

SERIOUS_LOG_PATTERNS = [
    ("Traceback", re.compile(r"\bTraceback\b", re.IGNORECASE)),
    ("RuntimeError", re.compile(r"\bRuntimeError\b", re.IGNORECASE)),
    ("ValueError", re.compile(r"\bValueError\b", re.IGNORECASE)),
    ("Killed", re.compile(r"\bKilled\b", re.IGNORECASE)),
    ("Out Of Memory", re.compile(r"\bOut\s+Of\s+Memory\b", re.IGNORECASE)),
    ("OOM", re.compile(r"(?<![A-Za-z0-9_])OOM(?![A-Za-z0-9_])", re.IGNORECASE)),
    ("Segmentation fault", re.compile(r"\bSegmentation\s+fault\b", re.IGNORECASE)),
    ("Bus error", re.compile(r"\bBus\s+error\b", re.IGNORECASE)),
]

KNOWN_WARNING_SUBSTRINGS = [
    "pkg_resources is deprecated as an API",
]

COMPLETE_BUNDLE_REQUIRED_FILES = [
    "summaries/task_inventory.csv",
    "summaries/runtime_summary.csv",
    "summaries/canonical_reconciliation_summary.csv",
    "summaries/service_reconciliation_summary.csv",
    "summaries/source_provenance_summary.csv",
    "summaries/warning_inventory.csv",
    "summaries/failure_manifest.csv",
    "runtime_metadata/source_commit_sha.txt",
    "runtime_metadata/array_job_id.txt",
    "runtime_metadata/reducer_job_id.txt",
    "runtime_metadata/sacct_raw.txt",
    "runtime_metadata/reducer_runtime_metadata.env",
    "runtime_metadata/reducer_markers.env",
    "runtime_metadata/reducer_stdout_snapshot.log",
    "runtime_metadata/reducer_stderr_snapshot.log",
    "runtime_metadata/complete_file_list.txt",
    "runtime_metadata/complete_file_checksums.sha256",
]

EXPECTED_REDUCER_MARKERS = {
    "TASK_PACKAGE_COUNT": "8",
    "STDOUT_LOG_COUNT": "8",
    "STDERR_LOG_COUNT": "8",
    "DIAGNOSTIC_CSV_COUNT": "32",
    "ALL_TASK_CHECKSUMS_OK": "1",
    "ALL_SCHEMA_VERSION_3": "1",
    "ALL_CANONICAL_RECONCILIATIONS_OK": "1",
    "ALL_SERVICE_RECONCILIATIONS_OK": "1",
    "ALL_ACTION_CONTRACTS_OK": "1",
    "ALL_RUNTIME_METADATA_PRESENT": "1",
    "ALL_SLURM_TASKS_COMPLETED": "1",
    "COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_OK": "1",
    "INFRASTRUCTURE_DIAGNOSTIC_SMOKE_REDUCER_COMPLETED": "1",
}

SERVICE_SUMMARY_FIELDS = [
    "task_id",
    "scale",
    "algorithm",
    "episode_total_ev_served",
    "charger_served_sum",
    "transformer_served_sum",
    "charger_satisfaction_count_sum",
    "transformer_satisfaction_count_sum",
    "episode_total_energy_charged",
    "charger_energy_charged_sum",
    "transformer_energy_charged_sum",
    "episode_total_energy_discharged",
    "charger_energy_discharged_sum",
    "transformer_energy_discharged_sum",
    "served_reconciliation_pass",
    "satisfaction_reconciliation_pass",
    "charged_energy_reconciliation_pass",
    "discharged_energy_reconciliation_pass",
]


class ValidationError(Exception):
    pass


def task_mapping(task_id):
    try:
        return dict(TASKS[int(task_id)])
    except (KeyError, ValueError) as exc:
        raise ValidationError(f"invalid task id: {task_id}") from exc


def formal_package_name(mapping):
    return (
        f"m3_controlled_multiscale_formal_{mapping['scale']}_{mapping['algorithm']}_seed0_"
        f"job{FORMAL_JOB_ID}_task{mapping['formal_task_id']}.tar.gz"
    )


def smoke_task_package_name(mapping, array_job_id):
    return (
        f"m3_infrastructure_diagnostic_smoke_{mapping['scale']}_{mapping['algorithm']}_seed0_"
        f"job{array_job_id}_task{mapping['task_id']}.tar.gz"
    )


def smoke_slurm_log_name(array_job_id, task_id, suffix):
    return f"evgnn_infra_diag_smoke_{array_job_id}_{task_id}.{suffix}"


def run_name(mapping):
    return f"controlled_multiscale_formal_{mapping['scale']}_{mapping['algorithm']}_seed0"


def config_member(mapping):
    return f"config/{mapping['scale']}_{mapping['algorithm']}_seed0_config.yaml"


def canonical_member(mapping):
    return f"eval/{mapping['scale']}_{mapping['algorithm']}_seed0_eval30.csv"


def train_member(mapping, basename):
    return f"train/{run_name(mapping)}/{basename}"


def required_formal_members(mapping):
    return [
        train_member(mapping, "model.best_actor"),
        train_member(mapping, "model.best_actor_optimizer"),
        train_member(mapping, "model.best_critic"),
        train_member(mapping, "model.best_critic_optimizer"),
        train_member(mapping, "kwargs.yaml"),
        config_member(mapping),
        canonical_member(mapping),
        "runtime_metadata/source_manifest.sha256",
        "runtime_metadata/task_runtime_metadata.env",
        "runtime_metadata/package_file_checksums.sha256",
    ]


def normalise_member_name(name):
    raw = str(name)
    path = PurePosixPath(raw)
    if raw.startswith("/") or path.is_absolute():
        raise ValidationError(f"unsafe absolute tar member path: {name}")
    if any(part == ".." for part in path.parts):
        raise ValidationError(f"unsafe tar member path contains '..': {name}")
    clean_parts = [part for part in path.parts if part not in {"", "."}]
    if not clean_parts:
        raise ValidationError(f"unsafe empty tar member path: {name}")
    return PurePosixPath(*clean_parts).as_posix()


def reject_unsafe_tar_member(member):
    normalise_member_name(member.name)
    if member.issym() or member.islnk():
        raise ValidationError(f"tar links are not allowed in smoke evidence: {member.name}")
    if not (member.isfile() or member.isdir()):
        raise ValidationError(f"unsupported special tar member: {member.name}")


def open_tar(path):
    try:
        return tarfile.open(path, "r:gz")
    except tarfile.TarError as exc:
        raise ValidationError(f"unreadable tar package: {path}") from exc


def safe_tar_members(tar):
    members = tar.getmembers()
    seen = set()
    for member in members:
        reject_unsafe_tar_member(member)
        normalised = normalise_member_name(member.name)
        if normalised in seen:
            raise ValidationError(f"duplicate normalized tar member path: {normalised}")
        seen.add(normalised)
    return members


def safe_extract_tar(package_path, extract_dir):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with open_tar(package_path) as tar:
        members = safe_tar_members(tar)
        for member in members:
            normalised = normalise_member_name(member.name)
            target = (root / normalised).resolve()
            if root not in [target, *target.parents]:
                raise ValidationError(f"unsafe tar extraction target: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            payload = tar.extractfile(member)
            if payload is None:
                raise ValidationError(f"cannot read package member: {normalised}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(payload, output)
    return root


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest_text(text):
    entries = []
    seen_paths = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            digest, relative = stripped.split(None, 1)
        except ValueError as exc:
            raise ValidationError(f"invalid checksum manifest line: {line!r}") from exc
        if not HEX_SHA256.fullmatch(digest):
            raise ValidationError(f"invalid checksum digest syntax: {digest!r}")
        normalised = normalise_member_name(relative.strip())
        if normalised in seen_paths:
            raise ValidationError(f"duplicate checksum manifest path: {normalised}")
        seen_paths.add(normalised)
        entries.append((digest.lower(), normalised))
    return entries


def verify_manifest_coverage(covered_paths, required_paths, context):
    missing = sorted(set(required_paths) - set(covered_paths))
    if missing:
        raise ValidationError(
            f"checksum coverage missing required {context} file(s): {', '.join(missing)}"
        )


def verify_extracted_manifest(root, manifest_path, required_coverage=()):
    manifest = Path(root) / manifest_path
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise ValidationError(f"missing required checksum manifest: {manifest_path}")
    entries = parse_manifest_text(manifest.read_text(encoding="utf-8"))
    covered = set()
    for expected_digest, normalised in entries:
        covered.add(normalised)
        path = Path(root) / normalised
        if normalised == manifest_path:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise ValidationError(f"checksum manifest references missing file: {normalised}")
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValidationError(
                f"checksum mismatch for {normalised}: {actual_digest} != {expected_digest}"
            )
    verify_manifest_coverage(covered, required_coverage, "formal package")
    return covered


def verify_tar_manifest(package_path, manifest_member, required_coverage=(), require_all_files=False):
    with open_tar(package_path) as tar:
        members = safe_tar_members(tar)
        member_by_name = {normalise_member_name(member.name): member for member in members if member.isfile()}
        if manifest_member not in member_by_name:
            raise ValidationError(f"missing required checksum manifest: {manifest_member}")
        manifest_file = tar.extractfile(member_by_name[manifest_member])
        if manifest_file is None:
            raise ValidationError(f"cannot read checksum manifest: {manifest_member}")
        entries = parse_manifest_text(manifest_file.read().decode("utf-8"))
        covered = set()
        for expected_digest, normalised in entries:
            covered.add(normalised)
            if normalised == manifest_member:
                continue
            if normalised not in member_by_name:
                raise ValidationError(f"checksum manifest references missing file: {normalised}")
            payload = tar.extractfile(member_by_name[normalised])
            if payload is None:
                raise ValidationError(f"cannot read package member: {normalised}")
            actual_digest = sha256_bytes(payload.read())
            if actual_digest != expected_digest:
                raise ValidationError(
                    f"checksum mismatch for {normalised}: {actual_digest} != {expected_digest}"
                )
        verify_manifest_coverage(covered, required_coverage, "package")
        if require_all_files:
            expected_coverage = set(member_by_name) - {manifest_member}
            missing = sorted(expected_coverage - covered)
            extra = sorted(covered - set(member_by_name))
            if missing or extra:
                raise ValidationError(
                    "checksum coverage mismatch for package file(s): "
                    f"missing={missing}, extra={extra}"
                )
        return covered


def require_files(root, relative_paths):
    missing = []
    for relative in relative_paths:
        path = Path(root) / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    if missing:
        raise ValidationError(f"missing required file(s): {', '.join(missing)}")


def validate_formal_config(config_path, mapping):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValidationError(f"formal config did not parse as a YAML mapping: {config_path}")

    simulation_length = integer_value(config.get("simulation_length"), "simulation_length")
    if simulation_length != 112:
        raise ValidationError(f"simulation_length must equal 112, got {simulation_length}")

    v2g_enabled = parse_bool(config.get("v2g_enabled"), "v2g_enabled")
    if v2g_enabled:
        raise ValidationError("v2g_enabled must be false in the extracted formal config")

    charger_count = integer_value(config.get("number_of_charging_stations"), "number_of_charging_stations")
    if charger_count != mapping["expected_charger_rows"]:
        raise ValidationError(
            "number_of_charging_stations mismatch: "
            f"{charger_count} != {mapping['expected_charger_rows']}"
        )

    transformer_count = integer_value(config.get("number_of_transformers"), "number_of_transformers")
    if transformer_count != mapping["expected_transformer_rows"]:
        raise ValidationError(
            "number_of_transformers mismatch: "
            f"{transformer_count} != {mapping['expected_transformer_rows']}"
        )

    return {
        "simulation_length": simulation_length,
        "v2g_enabled": v2g_enabled,
        "number_of_charging_stations": charger_count,
        "number_of_transformers": transformer_count,
    }


def read_rows(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"missing or empty CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def require_columns(fieldnames, required_columns, label):
    missing = sorted(set(required_columns) - set(fieldnames))
    if missing:
        raise ValidationError(f"{label} missing required column(s): {', '.join(missing)}")


def numeric(value, field):
    if value in ("", None):
        raise ValidationError(f"missing numeric value for {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValidationError(f"non-finite numeric value for {field}: {value!r}")
    return parsed


def integer_value(value, field):
    parsed = numeric(value, field)
    if not parsed.is_integer():
        raise ValidationError(f"{field} must be an integral value: {value!r}")
    return int(parsed)


def nonnegative_integer(value, field):
    parsed = integer_value(value, field)
    if parsed < 0:
        raise ValidationError(f"{field} must be non-negative: {value!r}")
    return parsed


def optional_numeric(value, field):
    if value in ("", None):
        return None
    return numeric(value, field)


def bool_value(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_bool(value, field):
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValidationError(f"invalid boolean value for {field}: {value!r}")


def assert_close(label, observed, expected, absolute_tolerance=1e-6, relative_tolerance=1e-12):
    observed = float(observed)
    expected = float(expected)
    difference = abs(observed - expected)
    if difference <= absolute_tolerance:
        return
    if relative_tolerance and difference <= relative_tolerance * max(abs(expected), 1.0):
        return
    raise ValidationError(
        f"reconciliation mismatch for {label}: observed {observed} != expected {expected}"
    )


def sum_available(rows, field):
    total = 0.0
    for row in rows:
        total += numeric(row.get(field), field)
    return total


def sum_integer(rows, field):
    total = 0
    for row in rows:
        total += nonnegative_integer(row.get(field), field)
    return total


def require_unique_nonnegative_ids(rows, field):
    seen = set()
    for row in rows:
        value = nonnegative_integer(row.get(field), field)
        if value in seen:
            raise ValidationError(f"{field} values must be unique")
        seen.add(value)
    return seen


def validate_unit_fraction(value, field):
    fraction = numeric(value, field)
    if fraction < 0.0 or fraction > 1.0:
        raise ValidationError(f"{field} must be in [0, 1]: {value!r}")
    return fraction


def validate_row_identity(mapping, rows, label, matrix_job_id=None, require_episode_identity=False):
    expected_pairs = [
        ("scale", mapping["scale"]),
        ("algorithm", mapping["algorithm"]),
        ("training_seed", "0"),
    ]
    for index, row in enumerate(rows):
        for field, expected in expected_pairs:
            if str(row.get(field)) != str(expected):
                raise ValidationError(
                    f"{label} row {index} identity mismatch for {field}: "
                    f"{row.get(field)} != {expected}"
                )
        if matrix_job_id is not None:
            if str(row.get("matrix_job_id")) != str(matrix_job_id):
                raise ValidationError(
                    f"{label} row {index} identity mismatch for matrix_job_id: "
                    f"{row.get('matrix_job_id')} != {matrix_job_id}"
                )
        if require_episode_identity:
            if integer_value(row.get("episode_index"), "episode_index") != 0:
                raise ValidationError(f"{label} row {index} identity mismatch for episode_index")
            if integer_value(row.get("episode_seed"), "episode_seed") != int(mapping["episode_seed"]):
                raise ValidationError(f"{label} row {index} identity mismatch for episode_seed")


def validate_environment_count_fraction(row, label, active_count, count_field, fraction_field):
    count = nonnegative_integer(row.get(count_field), count_field)
    if count > active_count:
        raise ValidationError(
            f"{label} {count_field}={count} exceeds active_action_decision_count={active_count}"
        )

    raw_fraction = row.get(fraction_field)
    if active_count == 0:
        if count != 0:
            raise ValidationError(
                f"{label} {count_field} must equal zero when active_action_decision_count is zero"
            )
        if raw_fraction in ("", None):
            return count, None
        fraction = validate_unit_fraction(raw_fraction, fraction_field)
        if abs(fraction) > 1e-12:
            raise ValidationError(
                f"{label} {fraction_field} must be blank or zero when active_action_decision_count is zero"
            )
        return count, fraction

    fraction = validate_unit_fraction(raw_fraction, fraction_field)
    assert_close(
        f"{label} {fraction_field}",
        fraction,
        count / active_count,
        absolute_tolerance=1e-12,
        relative_tolerance=0.0,
    )
    return count, fraction


def validate_episode_environment_bounds(mapping, episode_row):
    active_count = nonnegative_integer(
        episode_row.get("active_action_decision_count"),
        "active_action_decision_count",
    )
    below_count, below_fraction = validate_environment_count_fraction(
        episode_row,
        "episode",
        active_count,
        "active_action_below_environment_low_count",
        "active_action_below_environment_low_fraction",
    )
    validate_environment_count_fraction(
        episode_row,
        "episode",
        active_count,
        "active_action_above_environment_high_count",
        "active_action_above_environment_high_fraction",
    )
    if mapping["algorithm"] == "hierarchical":
        if below_count != 0:
            raise ValidationError("hierarchical below-environment-low count must equal zero")
        if below_fraction is None or abs(below_fraction) > 1e-12:
            raise ValidationError("hierarchical below-environment-low fraction must equal zero")


def validate_seed_summary_action_domain(episode_row, seed_row):
    episode_active_count = nonnegative_integer(
        episode_row.get("active_action_decision_count"),
        "active_action_decision_count",
    )
    active_mean = numeric(
        seed_row.get("active_action_decision_count_mean"),
        "active_action_decision_count_mean",
    )
    if active_mean < 0:
        raise ValidationError("active_action_decision_count_mean must be non-negative")
    assert_close(
        "seed summary active_action_decision_count_mean",
        active_mean,
        episode_active_count,
        absolute_tolerance=1e-12,
        relative_tolerance=0.0,
    )

    for field in [
        "active_action_below_environment_low_count",
        "active_action_above_environment_high_count",
    ]:
        seed_count = nonnegative_integer(seed_row.get(field), field)
        episode_count = nonnegative_integer(episode_row.get(field), field)
        if seed_count != episode_count:
            raise ValidationError(
                f"seed summary {field} disagrees with episode value: {seed_count} != {episode_count}"
            )

    for field in [
        "active_action_below_environment_low_fraction",
        "active_action_above_environment_high_fraction",
    ]:
        episode_raw = episode_row.get(field)
        seed_raw = seed_row.get(field)
        if episode_raw in ("", None):
            if seed_raw in ("", None):
                continue
            seed_fraction = validate_unit_fraction(seed_raw, field)
            if abs(seed_fraction) > 1e-12:
                raise ValidationError(
                    f"seed summary {field} must be blank or zero when episode value is blank"
                )
            continue
        episode_fraction = validate_unit_fraction(episode_raw, field)
        seed_fraction = validate_unit_fraction(seed_raw, field)
        assert_close(
            f"seed summary {field}",
            seed_fraction,
            episode_fraction,
            absolute_tolerance=1e-12,
            relative_tolerance=0.0,
        )

    signed_pairs = [
        ("global_positive_action_fraction_active_mean", "global_positive_action_fraction_active"),
        ("global_zero_action_fraction_active_mean", "global_zero_action_fraction_active"),
        ("global_negative_action_fraction_active_mean", "global_negative_action_fraction_active"),
    ]
    seed_fractions = []
    fraction_pairs = []
    for seed_field, episode_field in signed_pairs:
        seed_fraction = validate_unit_fraction(seed_row.get(seed_field), seed_field)
        episode_fraction = validate_unit_fraction(episode_row.get(episode_field), episode_field)
        seed_fractions.append(seed_fraction)
        fraction_pairs.append((seed_field, seed_fraction, episode_fraction))
    if abs(sum(seed_fractions) - 1.0) > 1e-6:
        raise ValidationError("seed summary signed action fractions must sum to one")
    for seed_field, seed_fraction, episode_fraction in fraction_pairs:
        assert_close(
            f"seed summary {seed_field}",
            seed_fraction,
            episode_fraction,
            absolute_tolerance=1e-12,
            relative_tolerance=0.0,
        )


def validate_signed_rows(rows, label):
    for index, row in enumerate(rows):
        active_field = "active_action_decision_count" if "active_action_decision_count" in row else "n_active_ev_decisions"
        active = nonnegative_integer(row.get(active_field), active_field)
        if active <= 0:
            continue
        fraction_fields = [
            "global_positive_action_fraction_active" if "global_positive_action_fraction_active" in row else "positive_action_fraction_active",
            "global_zero_action_fraction_active" if "global_zero_action_fraction_active" in row else "zero_action_fraction_active",
            "global_negative_action_fraction_active" if "global_negative_action_fraction_active" in row else "negative_action_fraction_active",
        ]
        fractions = []
        for field in fraction_fields:
            if field not in row or row.get(field) in {"", None}:
                raise ValidationError(f"{label} row {index} missing signed fraction field: {field}")
            value = numeric(row.get(field), field)
            if value < 0.0 or value > 1.0:
                raise ValidationError(f"{label} row {index} signed fraction out of range for {field}: {value}")
            fractions.append(value)
        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValidationError(f"signed fraction invariant failed for {label} row {index}")


def validate_no_hierarchical_negative(mapping, episode_rows, seed_rows, charger_rows, transformer_rows):
    if mapping["algorithm"] != "hierarchical":
        return
    fields_and_rows = [
        ("global_negative_action_fraction_active", episode_rows),
        ("global_negative_action_fraction_active_mean", seed_rows),
        ("negative_action_fraction_active", charger_rows),
        ("negative_action_fraction_active", transformer_rows),
    ]
    for field, rows in fields_and_rows:
        for row in rows:
            value = optional_numeric(row.get(field, ""), field)
            if value is not None and abs(value) > 1e-6:
                raise ValidationError(f"hierarchical negative action fraction is non-zero: {field}={value}")
    for field, rows in [
        ("active_action_below_environment_low_count", episode_rows),
        ("active_action_below_environment_low_count", seed_rows),
    ]:
        for row in rows:
            if nonnegative_integer(row.get(field), field) != 0:
                raise ValidationError(f"hierarchical below-environment-low count is non-zero: {field}")
    for field, rows in [
        ("active_action_below_environment_low_fraction", episode_rows),
        ("active_action_below_environment_low_fraction", seed_rows),
    ]:
        for row in rows:
            value = optional_numeric(row.get(field, ""), field)
            if value is None or abs(value) > 1e-12:
                raise ValidationError(f"hierarchical below-environment-low fraction is non-zero: {field}")


def validate_diagnostic_identity(mapping, episode_row, seed_row):
    expected_pairs = [
        ("scale", mapping["scale"]),
        ("algorithm", mapping["algorithm"]),
        ("training_seed", "0"),
    ]
    for field, expected in expected_pairs:
        if str(episode_row.get(field)) != str(expected):
            raise ValidationError(f"diagnostic identity mismatch for {field}")
        if str(seed_row.get(field)) != str(expected):
            raise ValidationError(f"seed summary identity mismatch for {field}")
    if integer_value(episode_row.get("episode_index"), "episode_index") != 0:
        raise ValidationError("diagnostic episode_index must be 0")
    if integer_value(episode_row.get("episode_seed"), "episode_seed") != int(mapping["episode_seed"]):
        raise ValidationError("diagnostic episode_seed mismatch")
    if integer_value(episode_row.get("episode_steps"), "episode_steps") != 112:
        raise ValidationError("diagnostic episode_steps mismatch")
    if not parse_bool(episode_row.get("done"), "done"):
        raise ValidationError("diagnostic episode done must be true")


def validate_schema(rows, label):
    for row in rows:
        if str(row.get("diagnostic_schema_version")) != SCHEMA_VERSION:
            raise ValidationError(f"{label} schema version mismatch")


def validate_v2g_false(rows):
    for row in rows:
        if "v2g_enabled" not in row:
            continue
        if row.get("v2g_enabled") in {"", None}:
            raise ValidationError("v2g_enabled column/value is required")
        if parse_bool(row.get("v2g_enabled"), "v2g_enabled"):
            raise ValidationError("v2g_enabled must be false for PublicPST formal diagnostics")


def validate_service_reconciliation(episode_row, charger_rows, transformer_rows):
    served_episode = nonnegative_integer(episode_row.get("total_ev_served"), "total_ev_served")
    charged_episode = numeric(episode_row.get("total_energy_charged"), "total_energy_charged")
    discharged_episode = numeric(episode_row.get("total_energy_discharged"), "total_energy_discharged")

    charger_ids = require_unique_nonnegative_ids(charger_rows, "charger_id")
    transformer_ids = require_unique_nonnegative_ids(transformer_rows, "transformer_id")
    if len(charger_ids) != len(charger_rows) or len(transformer_ids) != len(transformer_rows):
        raise ValidationError("infrastructure IDs must be unique")
    for row in charger_rows:
        transformer_id = nonnegative_integer(row.get("transformer_id"), "transformer_id")
        if transformer_id not in transformer_ids:
            raise ValidationError(f"charger references unknown transformer_id: {transformer_id}")

    charger_served = sum_integer(charger_rows, "served_ev_count")
    transformer_served = sum_integer(transformer_rows, "served_ev_count")
    charger_satisfaction_count = sum_integer(charger_rows, "user_satisfaction_observation_count")
    transformer_satisfaction_count = sum_integer(transformer_rows, "user_satisfaction_observation_count")

    if charger_served != transformer_served or charger_served != served_episode:
        raise ValidationError(
            "reconciliation mismatch for served_ev_count: "
            f"charger={charger_served}, transformer={transformer_served}, episode={served_episode}"
        )
    if charger_satisfaction_count != transformer_satisfaction_count or charger_satisfaction_count != served_episode:
        raise ValidationError(
            "reconciliation mismatch for satisfaction observation count: "
            f"charger={charger_satisfaction_count}, transformer={transformer_satisfaction_count}, episode={served_episode}"
        )

    assert_close("charger energy_charged_kwh", sum_available(charger_rows, "energy_charged_kwh"), charged_episode)
    assert_close("transformer energy_charged_kwh", sum_available(transformer_rows, "energy_charged_kwh"), charged_episode)
    assert_close(
        "charger energy_discharged_kwh",
        sum_available(charger_rows, "energy_discharged_kwh"),
        discharged_episode,
    )
    assert_close(
        "transformer energy_discharged_kwh",
        sum_available(transformer_rows, "energy_discharged_kwh"),
        discharged_episode,
    )

    for row in charger_rows + transformer_rows:
        served_count = nonnegative_integer(row.get("served_ev_count"), "served_ev_count")
        satisfaction_count = nonnegative_integer(
            row.get("user_satisfaction_observation_count"),
            "user_satisfaction_observation_count",
        )
        if served_count > 0:
            if satisfaction_count <= 0:
                raise ValidationError("satisfaction observation count must be positive when served count > 0")
            if not row.get("user_satisfaction_source"):
                raise ValidationError("explicit satisfaction source is required when served count > 0")

    chargers_by_transformer = {}
    for row in charger_rows:
        chargers_by_transformer.setdefault(nonnegative_integer(row.get("transformer_id"), "transformer_id"), []).append(row)
    for transformer_row in transformer_rows:
        transformer_id = nonnegative_integer(transformer_row.get("transformer_id"), "transformer_id")
        chargers = chargers_by_transformer.get(transformer_id, [])
        transformer_served = nonnegative_integer(transformer_row.get("served_ev_count"), "served_ev_count")
        transformer_satisfaction_count = nonnegative_integer(
            transformer_row.get("user_satisfaction_observation_count"),
            "user_satisfaction_observation_count",
        )
        if sum_integer(chargers, "served_ev_count") != transformer_served:
            raise ValidationError(f"reconciliation mismatch for transformer {transformer_id} served_ev_count")
        assert_close(
            f"transformer {transformer_id} energy_charged_kwh",
            sum_available(chargers, "energy_charged_kwh"),
            numeric(transformer_row.get("energy_charged_kwh"), "energy_charged_kwh"),
        )
        assert_close(
            f"transformer {transformer_id} energy_discharged_kwh",
            sum_available(chargers, "energy_discharged_kwh"),
            numeric(transformer_row.get("energy_discharged_kwh"), "energy_discharged_kwh"),
        )
        assert_close(
            f"transformer {transformer_id} satisfaction sum",
            sum_available(chargers, "user_satisfaction_sum"),
            numeric(transformer_row.get("user_satisfaction_sum"), "user_satisfaction_sum"),
        )
        if sum_integer(chargers, "user_satisfaction_observation_count") != transformer_satisfaction_count:
            raise ValidationError(f"reconciliation mismatch for transformer {transformer_id} satisfaction count")


def service_summary_from_rows(mapping, episode_row, charger_rows, transformer_rows):
    episode_served = nonnegative_integer(episode_row.get("total_ev_served"), "total_ev_served")
    charger_served = sum_integer(charger_rows, "served_ev_count")
    transformer_served = sum_integer(transformer_rows, "served_ev_count")
    charger_satisfaction_count = sum_integer(charger_rows, "user_satisfaction_observation_count")
    transformer_satisfaction_count = sum_integer(transformer_rows, "user_satisfaction_observation_count")

    episode_charged = numeric(episode_row.get("total_energy_charged"), "total_energy_charged")
    charger_charged = sum_available(charger_rows, "energy_charged_kwh")
    transformer_charged = sum_available(transformer_rows, "energy_charged_kwh")
    episode_discharged = numeric(episode_row.get("total_energy_discharged"), "total_energy_discharged")
    charger_discharged = sum_available(charger_rows, "energy_discharged_kwh")
    transformer_discharged = sum_available(transformer_rows, "energy_discharged_kwh")

    return {
        "task_id": str(mapping["task_id"]),
        "scale": str(mapping["scale"]),
        "algorithm": str(mapping["algorithm"]),
        "episode_total_ev_served": str(episode_served),
        "charger_served_sum": str(charger_served),
        "transformer_served_sum": str(transformer_served),
        "charger_satisfaction_count_sum": str(charger_satisfaction_count),
        "transformer_satisfaction_count_sum": str(transformer_satisfaction_count),
        "episode_total_energy_charged": str(episode_charged),
        "charger_energy_charged_sum": str(charger_charged),
        "transformer_energy_charged_sum": str(transformer_charged),
        "episode_total_energy_discharged": str(episode_discharged),
        "charger_energy_discharged_sum": str(charger_discharged),
        "transformer_energy_discharged_sum": str(transformer_discharged),
        "served_reconciliation_pass": str(charger_served == transformer_served == episode_served),
        "satisfaction_reconciliation_pass": str(
            charger_satisfaction_count == transformer_satisfaction_count == episode_served
        ),
        "charged_energy_reconciliation_pass": str(
            abs(charger_charged - episode_charged) <= 1e-6
            and abs(transformer_charged - episode_charged) <= 1e-6
        ),
        "discharged_energy_reconciliation_pass": str(
            abs(charger_discharged - episode_discharged) <= 1e-6
            and abs(transformer_discharged - episode_discharged) <= 1e-6
        ),
    }


def validate_diagnostics(args):
    mapping = task_mapping(args.task_id)
    diagnostic_dir = Path(args.diagnostic_dir)
    episode_rows, episode_columns = read_rows(diagnostic_dir / "episode_diagnostics.csv")
    seed_rows, seed_columns = read_rows(diagnostic_dir / "seed_summary_diagnostics.csv")
    transformer_rows, transformer_columns = read_rows(diagnostic_dir / "transformer_diagnostics.csv")
    charger_rows, charger_columns = read_rows(diagnostic_dir / "charger_diagnostics.csv")

    if len(episode_rows) != 1:
        raise ValidationError(f"expected 1 episode row, got {len(episode_rows)}")
    if len(seed_rows) != 1:
        raise ValidationError(f"expected 1 seed-summary row, got {len(seed_rows)}")
    if len(charger_rows) != mapping["expected_charger_rows"]:
        raise ValidationError(
            f"expected {mapping['expected_charger_rows']} charger rows, got {len(charger_rows)}"
        )
    if len(transformer_rows) != mapping["expected_transformer_rows"]:
        raise ValidationError(
            f"expected {mapping['expected_transformer_rows']} transformer rows, got {len(transformer_rows)}"
        )

    require_columns(episode_columns, EPISODE_REQUIRED_COLUMNS, "episode diagnostics")
    require_columns(seed_columns, SEED_SUMMARY_REQUIRED_COLUMNS, "seed summary diagnostics")
    require_columns(charger_columns, CHARGER_REQUIRED_COLUMNS, "charger diagnostics")
    require_columns(transformer_columns, TRANSFORMER_REQUIRED_COLUMNS, "transformer diagnostics")

    all_rows = episode_rows + seed_rows + charger_rows + transformer_rows
    validate_schema(episode_rows, "episode")
    validate_schema(seed_rows, "seed summary")
    validate_schema(charger_rows, "charger")
    validate_schema(transformer_rows, "transformer")
    validate_v2g_false(all_rows)
    matrix_job_id = getattr(args, "matrix_job_id", None)
    validate_row_identity(mapping, episode_rows, "episode", matrix_job_id=matrix_job_id)
    validate_row_identity(mapping, seed_rows, "seed summary", matrix_job_id=matrix_job_id)
    validate_row_identity(
        mapping,
        charger_rows,
        "charger",
        matrix_job_id=matrix_job_id,
        require_episode_identity=True,
    )
    validate_row_identity(
        mapping,
        transformer_rows,
        "transformer",
        matrix_job_id=matrix_job_id,
        require_episode_identity=True,
    )
    validate_diagnostic_identity(mapping, episode_rows[0], seed_rows[0])
    if integer_value(seed_rows[0].get("n_eval_episodes"), "n_eval_episodes") != 1:
        raise ValidationError("seed summary n_eval_episodes must equal 1")
    validate_episode_environment_bounds(mapping, episode_rows[0])
    validate_seed_summary_action_domain(episode_rows[0], seed_rows[0])
    validate_signed_rows(episode_rows, "episode")
    validate_signed_rows(charger_rows, "charger")
    validate_signed_rows(transformer_rows, "transformer")
    if nonnegative_integer(episode_rows[0].get("inactive_nonzero_action_count"), "inactive_nonzero_action_count") != 0:
        raise ValidationError("inactive non-zero action count must equal zero")
    inactive_mean = numeric(seed_rows[0].get("inactive_nonzero_action_count_mean"), "inactive_nonzero_action_count_mean")
    if inactive_mean != 0:
        raise ValidationError("inactive non-zero action count mean must equal zero")
    validate_no_hierarchical_negative(mapping, episode_rows, seed_rows, charger_rows, transformer_rows)
    validate_service_reconciliation(episode_rows[0], charger_rows, transformer_rows)

    validation = {
        "task_id": mapping["task_id"],
        "scale": mapping["scale"],
        "algorithm": mapping["algorithm"],
        "formal_task_id": mapping["formal_task_id"],
        "episode_seed": mapping["episode_seed"],
        "schema_version": SCHEMA_VERSION,
        "episode_rows": len(episode_rows),
        "seed_summary_rows": len(seed_rows),
        "charger_rows": len(charger_rows),
        "transformer_rows": len(transformer_rows),
        "status": "ok",
    }
    if matrix_job_id:
        validation["matrix_job_id"] = str(matrix_job_id)
    validation_dir = Path(args.validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)
    (validation_dir / "task_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, sort_keys=True))


def resolve_package(args):
    mapping = task_mapping(args.task_id)
    expected_name = formal_package_name(mapping)
    individual_root = Path(args.individual_package_root)
    individual_path = individual_root / expected_name
    if individual_path.is_file():
        with open_tar(individual_path) as tar:
            safe_tar_members(tar)
        payload = {
            "source_mode": "individual_task_package",
            "package_path": str(individual_path),
            "expected_package_name": expected_name,
        }
        print(json.dumps(payload, sort_keys=True))
        return

    complete_bundle = Path(args.complete_bundle)
    if not complete_bundle.is_file():
        raise ValidationError(f"formal package not found and complete bundle missing: {complete_bundle}")

    matches = []
    with open_tar(complete_bundle) as tar:
        members = safe_tar_members(tar)
        for member in members:
            member_name = normalise_member_name(member.name)
            parts = PurePosixPath(member_name).parts
            under_task_packages = "task_packages" in parts[:-1]
            if member.isfile() and Path(member_name).name == expected_name and under_task_packages:
                matches.append(member)
        if len(matches) != 1:
            raise ValidationError(
                "expected exactly one nested task package under task_packages/ "
                f"named {expected_name}, found {len(matches)}"
            )
        staging_dir = Path(args.staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        output_path = staging_dir / expected_name
        payload = tar.extractfile(matches[0])
        if payload is None:
            raise ValidationError(f"cannot read nested package: {matches[0].name}")
        with output_path.open("wb") as output:
            shutil.copyfileobj(payload, output)
    with open_tar(output_path) as nested_tar:
        safe_tar_members(nested_tar)
    print(
        json.dumps(
            {
                "source_mode": "complete_bundle_nested_task_package",
                "package_path": str(output_path),
                "bundle_member": matches[0].name,
                "expected_package_name": expected_name,
            },
            sort_keys=True,
        )
    )


def validate_formal_package(args):
    mapping = task_mapping(args.task_id)
    extract_root = safe_extract_tar(args.package, args.extract_dir)
    required = required_formal_members(mapping)
    require_files(extract_root, required)
    checksum_required = [
        member
        for member in required
        if member != "runtime_metadata/package_file_checksums.sha256"
    ]
    verify_extracted_manifest(
        extract_root,
        "runtime_metadata/package_file_checksums.sha256",
        required_coverage=checksum_required,
    )
    config_values = validate_formal_config(extract_root / config_member(mapping), mapping)
    payload = {
        "task_id": mapping["task_id"],
        "scale": mapping["scale"],
        "algorithm": mapping["algorithm"],
        "formal_task_id": mapping["formal_task_id"],
        "extract_dir": str(extract_root),
        "config_member": config_member(mapping),
        "canonical_member": canonical_member(mapping),
        "checkpoint_prefix_member": f"train/{run_name(mapping)}/model.best",
        "status": "ok",
        **config_values,
    }
    print(json.dumps(payload, sort_keys=True))


def canonical_episode_row(canonical_csv, episode_index):
    rows, _ = read_rows(canonical_csv)
    matches = [
        row for row in rows
        if row.get("row_type") == "episode" and str(row.get("episode_index")) == str(episode_index)
    ]
    if len(matches) != 1:
        raise ValidationError(f"expected exactly one canonical episode {episode_index}, found {len(matches)}")
    return matches[0]


def relative_difference(observed, expected):
    denominator = max(abs(float(expected)), 1.0)
    return abs(float(observed) - float(expected)) / denominator


def floating_reconciliation_row(field, canonical_value, diagnostic_value, absolute_tolerance, relative_tolerance):
    canonical_number = float(canonical_value)
    diagnostic_number = float(diagnostic_value)
    absolute_difference = abs(diagnostic_number - canonical_number)
    relative = relative_difference(diagnostic_number, canonical_number)
    passed = absolute_difference <= absolute_tolerance
    if not passed and relative_tolerance:
        passed = relative <= relative_tolerance
    return {
        "field": field,
        "comparison_type": "floating",
        "canonical_value": canonical_value,
        "diagnostic_value": diagnostic_value,
        "absolute_difference": absolute_difference,
        "relative_difference": relative,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "pass": str(bool(passed)),
    }


def exact_reconciliation_row(field, canonical_value, diagnostic_value, value_type):
    passed = False
    absolute_difference = ""
    relative = ""
    canonical_normalized = canonical_value
    diagnostic_normalized = diagnostic_value
    try:
        if value_type == "integer":
            canonical_normalized = integer_value(canonical_value, field)
            diagnostic_normalized = integer_value(diagnostic_value, field)
            absolute_difference = abs(diagnostic_normalized - canonical_normalized)
            relative = relative_difference(diagnostic_normalized, canonical_normalized)
            passed = canonical_normalized == diagnostic_normalized
        elif value_type == "boolean":
            canonical_normalized = parse_bool(canonical_value, field)
            diagnostic_normalized = parse_bool(diagnostic_value, field)
            absolute_difference = 0 if canonical_normalized == diagnostic_normalized else 1
            relative = absolute_difference
            passed = canonical_normalized == diagnostic_normalized
        else:
            passed = str(canonical_value) == str(diagnostic_value)
            absolute_difference = 0 if passed else 1
            relative = absolute_difference
    except ValidationError:
        absolute_difference = "invalid"
        relative = "invalid"
        passed = False
    return {
        "field": field,
        "comparison_type": "exact",
        "canonical_value": str(canonical_normalized),
        "diagnostic_value": str(diagnostic_normalized),
        "absolute_difference": absolute_difference,
        "relative_difference": relative,
        "absolute_tolerance": 0,
        "relative_tolerance": 0,
        "pass": str(bool(passed)),
    }


def reconcile_canonical(args):
    mapping = task_mapping(args.task_id)
    diagnostic_rows, _ = read_rows(args.episode_diagnostics)
    if len(diagnostic_rows) != 1:
        raise ValidationError(f"expected 1 diagnostic episode row, got {len(diagnostic_rows)}")
    diagnostic = diagnostic_rows[0]
    canonical = canonical_episode_row(args.canonical_csv, 0)

    rows = []
    for field, canonical_field, diagnostic_field, value_type in EXACT_RECONCILIATION:
        rows.append(
            exact_reconciliation_row(
                field,
                canonical.get(canonical_field),
                diagnostic.get(diagnostic_field),
                value_type,
            )
        )
    for diagnostic_field, canonical_field, absolute_tolerance, relative_tolerance in FLOAT_RECONCILIATION:
        rows.append(
            floating_reconciliation_row(
                diagnostic_field,
                canonical.get(canonical_field),
                diagnostic.get(diagnostic_field),
                absolute_tolerance,
                relative_tolerance,
            )
        )
    validation_dir = Path(args.validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)
    output_csv = validation_dir / "canonical_reconciliation.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures = [row["field"] for row in rows if row["pass"] != "True"]
    if diagnostic.get("algorithm") != mapping["algorithm"]:
        failures.append("diagnostic algorithm does not match task mapping")
    try:
        if integer_value(diagnostic.get("episode_seed"), "episode_seed") != mapping["episode_seed"]:
            failures.append("diagnostic episode seed does not match task mapping")
    except ValidationError as error:
        failures.append(str(error))
    if failures:
        raise ValidationError("canonical reconciliation failed: " + ", ".join(failures))
    print(json.dumps({"status": "ok", "rows": len(rows), "output_csv": str(output_csv)}, sort_keys=True))


def is_checkpoint_leak(member_name):
    path = normalise_member_name(member_name)
    lowered = path.lower()
    basename = Path(path).name.lower()
    if "checkpoint_staging/" in lowered:
        return True
    if basename.startswith("model.best") or basename.startswith("model.last"):
        return True
    if basename.endswith((".pt", ".pth")):
        return True
    if basename.endswith("_optimizer"):
        return True
    return False


def read_package_file_list(extract_root):
    path = Path(extract_root) / "runtime_metadata/package_file_list.txt"
    if not path.is_file():
        raise ValidationError("missing runtime_metadata/package_file_list.txt")
    listed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            listed.append(normalise_member_name(stripped))
    if len(listed) != len(set(listed)):
        raise ValidationError("package_file_list.txt contains duplicate member paths")
    return sorted(listed)


def load_task_validation(extract_root, mapping):
    path = Path(extract_root) / "validation/task_validation.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError("task_validation.json is not valid JSON") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("task_validation.json must contain a non-empty JSON object")
    if payload.get("status") != "ok":
        raise ValidationError("task_validation.json status must be ok")
    expected = {
        "task_id": mapping["task_id"],
        "scale": mapping["scale"],
        "algorithm": mapping["algorithm"],
        "formal_task_id": mapping["formal_task_id"],
        "episode_seed": mapping["episode_seed"],
        "schema_version": SCHEMA_VERSION,
        "episode_rows": 1,
        "seed_summary_rows": 1,
        "charger_rows": mapping["expected_charger_rows"],
        "transformer_rows": mapping["expected_transformer_rows"],
    }
    for field, expected_value in expected.items():
        if str(payload.get(field)) != str(expected_value):
            raise ValidationError(
                f"task_validation.json identity mismatch for {field}: {payload.get(field)} != {expected_value}"
            )
    return payload


def validate_packaged_reconciliation(extract_root):
    path = Path(extract_root) / "validation/canonical_reconciliation.csv"
    rows, fieldnames = read_rows(path)
    required = {
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
    require_columns(fieldnames, required, "canonical reconciliation")
    expected_types = {
        field: "exact"
        for field, _, _, _ in EXACT_RECONCILIATION
    }
    expected_types.update(
        {
            diagnostic_field: "floating"
            for diagnostic_field, _, _, _ in FLOAT_RECONCILIATION
        }
    )
    seen = set()
    for index, row in enumerate(rows):
        field = row.get("field")
        if field not in expected_types:
            raise ValidationError(f"canonical reconciliation contains unexpected field: {field}")
        if field in seen:
            raise ValidationError(f"canonical reconciliation contains duplicate field: {field}")
        seen.add(field)
        expected_type = expected_types[field]
        if row.get("comparison_type") != expected_type:
            raise ValidationError(
                f"canonical reconciliation comparison_type mismatch for {field}: "
                f"{row.get('comparison_type')} != {expected_type}"
            )
        if row.get("pass") != "True":
            raise ValidationError(
                f"canonical reconciliation pass value must be exactly True for {field}: {row.get('pass')}"
            )
    missing = sorted(set(expected_types) - seen)
    if missing:
        raise ValidationError(
            f"canonical reconciliation missing required field(s): {', '.join(missing)}"
        )


def parse_env_metadata(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValidationError(f"invalid env metadata line in {path}: {line!r}")
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def validate_source_package_provenance(extract_root, mapping):
    json_path = Path(extract_root) / "runtime_metadata/source_package_resolution.json"
    env_path = Path(extract_root) / "runtime_metadata/source_package.env"
    try:
        resolution = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError("source_package_resolution.json is not valid JSON") from exc
    if not isinstance(resolution, dict):
        raise ValidationError("source_package_resolution.json must contain a JSON object")

    source_mode = resolution.get("source_mode")
    allowed_modes = {"individual_task_package", "complete_bundle_nested_task_package"}
    if source_mode not in allowed_modes:
        raise ValidationError(f"source_package_resolution.json source_mode is invalid: {source_mode}")
    for field in ["package_path", "expected_package_name"]:
        if not resolution.get(field):
            raise ValidationError(f"source_package_resolution.json missing required field: {field}")
    expected_name = formal_package_name(mapping)
    if resolution.get("expected_package_name") != expected_name:
        raise ValidationError(
            "source_package_resolution.json expected_package_name mismatch: "
            f"{resolution.get('expected_package_name')} != {expected_name}"
        )
    if source_mode == "complete_bundle_nested_task_package" and not resolution.get("bundle_member"):
        raise ValidationError("source_package_resolution.json missing required field: bundle_member")

    env_values = parse_env_metadata(env_path)
    for field in ["source_mode", "package_path", "expected_package_name"]:
        if env_values.get(field) != str(resolution.get(field)):
            raise ValidationError(
                f"source_package.env provenance mismatch for {field}: "
                f"{env_values.get(field)} != {resolution.get(field)}"
            )
    if source_mode == "complete_bundle_nested_task_package":
        if env_values.get("bundle_member") != str(resolution.get("bundle_member")):
            raise ValidationError("source_package.env provenance mismatch for bundle_member")


def validate_task_package(args):
    mapping = task_mapping(args.task_id)
    required = [
        "stdout.log",
        "stderr.log",
        "diagnostics/episode_diagnostics.csv",
        "diagnostics/seed_summary_diagnostics.csv",
        "diagnostics/transformer_diagnostics.csv",
        "diagnostics/charger_diagnostics.csv",
        "canonical/complete_eval30.csv",
        "canonical/canonical_episode0.csv",
        "config/formal_config.yaml",
        "validation/task_validation.json",
        "validation/canonical_reconciliation.csv",
        "runtime_metadata/source_commit_sha.txt",
        "runtime_metadata/source_formal_job.env",
        "runtime_metadata/source_package.env",
        "runtime_metadata/source_package_resolution.json",
        "runtime_metadata/source_package.sha256",
        "runtime_metadata/checkpoint_member_hashes.sha256",
        "runtime_metadata/original_source_manifest.sha256",
        "runtime_metadata/original_task_runtime_metadata.env",
        "runtime_metadata/diagnostic_command.txt",
        "runtime_metadata/evaluator_stdout.txt",
        "runtime_metadata/evaluator_time_verbose.txt",
        "runtime_metadata/task_runtime_metadata.env",
        "runtime_metadata/package_file_checksums.sha256",
        "runtime_metadata/package_file_list.txt",
    ]
    package_path = Path(args.package)
    with open_tar(package_path) as tar:
        members = safe_tar_members(tar)
        member_names = sorted({
            normalise_member_name(member.name)
            for member in members
            if member.isfile()
        })
    leaks = sorted(name for name in member_names if is_checkpoint_leak(name))
    if leaks:
        raise ValidationError(f"checkpoint bytes are forbidden in task package: {', '.join(leaks)}")
    missing = sorted(set(required) - set(member_names))
    if missing:
        raise ValidationError(f"missing required task package file(s): {', '.join(missing)}")
    verify_tar_manifest(
        package_path,
        "runtime_metadata/package_file_checksums.sha256",
        required_coverage=set(member_names) - {"runtime_metadata/package_file_checksums.sha256"},
        require_all_files=True,
    )
    with tempfile.TemporaryDirectory(prefix="infra_diag_smoke_task_package_") as tmp_dir:
        extract_root = safe_extract_tar(package_path, tmp_dir)
        listed = read_package_file_list(extract_root)
        if listed != member_names:
            raise ValidationError(
                "package_file_list.txt does not exactly match package members: "
                f"listed={listed}, members={member_names}"
            )
        task_validation = load_task_validation(extract_root, mapping)
        validate_source_package_provenance(extract_root, mapping)
        validate_packaged_reconciliation(extract_root)
        matrix_job_id = task_validation.get("matrix_job_id")
        revalidation_dir = Path(tmp_dir) / "revalidation"
        validate_diagnostics(
            Namespace(
                task_id=args.task_id,
                diagnostic_dir=str(Path(extract_root) / "diagnostics"),
                validation_dir=str(revalidation_dir),
                matrix_job_id=matrix_job_id,
            )
        )
        reconcile_canonical(
            Namespace(
                task_id=args.task_id,
                episode_diagnostics=str(Path(extract_root) / "diagnostics/episode_diagnostics.csv"),
                canonical_csv=str(Path(extract_root) / "canonical/complete_eval30.csv"),
                validation_dir=str(revalidation_dir),
            )
        )
    print(json.dumps({"status": "ok", "task_id": mapping["task_id"], "package": str(package_path)}, sort_keys=True))


def csv_text(rows, fieldnames):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_csv_rows(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(csv_text(rows, fieldnames), encoding="utf-8")


def is_available_accounting_value(value):
    return str(value or "").strip() not in {"", "Unknown", "N/A", "None"}


def parse_sacct_raw(raw_text, array_job_id):
    parents = {}
    batches = {}
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("JobIDRaw|"):
            continue
        cells = stripped.split("|")
        if len(cells) < 7:
            raise ValidationError(f"invalid sacct row: {line!r}")
        row = {
            "JobIDRaw": cells[0],
            "State": cells[1],
            "ExitCode": cells[2],
            "ElapsedRaw": cells[3],
            "AllocCPUS": cells[4],
            "MaxRSS": cells[5],
            "TotalCPU": cells[6],
        }
        parent_match = re.fullmatch(rf"{re.escape(str(array_job_id))}_([0-9]+)", row["JobIDRaw"])
        batch_match = re.fullmatch(rf"{re.escape(str(array_job_id))}_([0-9]+)\.batch", row["JobIDRaw"])
        if parent_match:
            task_id = int(parent_match.group(1))
            if task_id not in TASKS:
                raise ValidationError(f"unexpected accounting task identity: {row['JobIDRaw']}")
            if task_id in parents:
                raise ValidationError(f"duplicate or contradictory accounting row for {row['JobIDRaw']}")
            parents[task_id] = row
            continue
        if batch_match:
            task_id = int(batch_match.group(1))
            if task_id not in TASKS:
                raise ValidationError(f"unexpected accounting batch task identity: {row['JobIDRaw']}")
            if task_id in batches:
                raise ValidationError(f"duplicate or contradictory accounting row for {row['JobIDRaw']}")
            batches[task_id] = row

    missing = sorted(set(TASKS) - set(parents))
    if missing:
        raise ValidationError(f"required accounting unavailable for task(s): {missing}")

    runtime_rows = []
    for task_id in sorted(TASKS):
        parent = parents[task_id]
        if parent["State"] != "COMPLETED":
            raise ValidationError(
                f"Slurm accounting task {array_job_id}_{task_id} must be COMPLETED, got {parent['State']}"
            )
        if parent["ExitCode"] != "0:0":
            raise ValidationError(
                f"Slurm accounting task {array_job_id}_{task_id} ExitCode must be 0:0, got {parent['ExitCode']}"
            )
        batch = batches.get(task_id, {})
        maxrss_source = "parent"
        totalcpu_source = "parent"
        maxrss = parent["MaxRSS"]
        totalcpu = parent["TotalCPU"]
        needs_batch_resource = (
            not is_available_accounting_value(maxrss)
            or not is_available_accounting_value(totalcpu)
        )
        if needs_batch_resource:
            if not batch:
                raise ValidationError(f"required .batch accounting unavailable for Slurm task {array_job_id}_{task_id}")
            if batch.get("State") != "COMPLETED":
                raise ValidationError(
                    f"Slurm accounting task {array_job_id}_{task_id}.batch must be COMPLETED, got {batch.get('State')}"
                )
            if batch.get("ExitCode") != "0:0":
                raise ValidationError(
                    f"Slurm accounting task {array_job_id}_{task_id}.batch ExitCode must be 0:0, got {batch.get('ExitCode')}"
                )
        if not is_available_accounting_value(maxrss):
            maxrss = batch.get("MaxRSS", "")
            maxrss_source = "batch"
        if not is_available_accounting_value(totalcpu):
            totalcpu = batch.get("TotalCPU", "")
            totalcpu_source = "batch"
        if not is_available_accounting_value(maxrss):
            raise ValidationError(f"MaxRSS unavailable for Slurm task {array_job_id}_{task_id}")
        if not is_available_accounting_value(totalcpu):
            raise ValidationError(f"TotalCPU unavailable for Slurm task {array_job_id}_{task_id}")
        runtime_rows.append(
            {
                "task_id": task_id,
                "job_id_raw": parent["JobIDRaw"],
                "state": parent["State"],
                "exit_code": parent["ExitCode"],
                "elapsed_raw": parent["ElapsedRaw"],
                "alloc_cpus": parent["AllocCPUS"],
                "max_rss": maxrss,
                "total_cpu": totalcpu,
                "maxrss_source": maxrss_source,
                "totalcpu_source": totalcpu_source,
            }
        )
    return runtime_rows


def collect_sacct_raw(args):
    if getattr(args, "sacct_raw_file", None):
        return Path(args.sacct_raw_file).read_text(encoding="utf-8")

    attempts = max(1, int(args.sacct_attempts))
    last_error = None
    raw_text = ""
    command = [
        str(args.sacct_command),
        "-j",
        str(args.array_job_id),
        "--parsable2",
        "--noheader",
        "--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU",
    ]
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            last_error = ValidationError(
                f"sacct command failed on attempt {attempt}: {completed.stderr.strip()}"
            )
        else:
            raw_text = completed.stdout
            try:
                parse_sacct_raw(raw_text, args.array_job_id)
                return raw_text
            except ValidationError as exc:
                last_error = exc
        if attempt < attempts:
            time.sleep(float(args.sacct_delay_seconds))
    raise ValidationError(f"Slurm accounting unavailable after {attempts} attempt(s): {last_error}")


def validate_expected_paths(paths, expected_paths, label):
    observed = sorted(Path(path).name for path in paths)
    expected = sorted(Path(path).name for path in expected_paths)
    if len(observed) != len(expected):
        raise ValidationError(f"expected exactly {len(expected)} {label}, got {len(observed)}")
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing:
        raise ValidationError(f"missing expected {label}: {', '.join(missing)}")
    if unexpected:
        raise ValidationError(f"unexpected {label}: {', '.join(unexpected)}")


def scan_log_text(source_name, text):
    warnings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for signature, pattern in SERIOUS_LOG_PATTERNS:
            if pattern.search(line):
                raise ValidationError(f"serious log signature {signature!r} in {source_name}:{line_number}: {line}")
        if "warning" in lowered:
            if any(known in line for known in KNOWN_WARNING_SUBSTRINGS):
                continue
            warnings.append(
                {
                    "source": source_name,
                    "line_number": line_number,
                    "warning": line.strip(),
                }
            )
    return warnings


def read_json_file(path, label):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{label} is not valid JSON") from exc


def validate_and_summarise_task_package(package_path, mapping, array_job_id, expected_source_commit, extract_root):
    with contextlib.redirect_stdout(io.StringIO()):
        validate_task_package(Namespace(task_id=mapping["task_id"], package=str(package_path)))

    task_extract_dir = Path(extract_root) / f"task{mapping['task_id']}"
    safe_extract_tar(package_path, task_extract_dir)
    task_validation = load_task_validation(task_extract_dir, mapping)
    if str(task_validation.get("matrix_job_id")) != str(array_job_id):
        raise ValidationError(
            f"task_validation.json matrix_job_id mismatch for task {mapping['task_id']}: "
            f"{task_validation.get('matrix_job_id')} != {array_job_id}"
        )
    source_commit = (task_extract_dir / "runtime_metadata/source_commit_sha.txt").read_text(encoding="utf-8").strip()
    if source_commit != expected_source_commit:
        raise ValidationError(
            f"source commit mismatch for task {mapping['task_id']}: {source_commit} != {expected_source_commit}"
        )
    validate_source_package_provenance(task_extract_dir, mapping)
    validate_packaged_reconciliation(task_extract_dir)
    require_files(task_extract_dir, DIAGNOSTIC_PACKAGE_CSVS)
    diagnostic_csv_count = len(DIAGNOSTIC_PACKAGE_CSVS)

    episode_rows, _ = read_rows(task_extract_dir / "diagnostics/episode_diagnostics.csv")
    charger_rows, _ = read_rows(task_extract_dir / "diagnostics/charger_diagnostics.csv")
    transformer_rows, _ = read_rows(task_extract_dir / "diagnostics/transformer_diagnostics.csv")
    reconciliation_rows, _ = read_rows(task_extract_dir / "validation/canonical_reconciliation.csv")
    source_resolution = read_json_file(
        task_extract_dir / "runtime_metadata/source_package_resolution.json",
        "source_package_resolution.json",
    )

    warnings = []
    for relative in [
        "stderr.log",
        "runtime_metadata/evaluator_stdout.txt",
        "runtime_metadata/evaluator_time_verbose.txt",
    ]:
        warnings.extend(
            scan_log_text(
                f"task{mapping['task_id']}:{relative}",
                (task_extract_dir / relative).read_text(encoding="utf-8"),
            )
        )

    episode = episode_rows[0]
    inventory_row = {
        "task_id": mapping["task_id"],
        "scale": mapping["scale"],
        "algorithm": mapping["algorithm"],
        "formal_task_id": mapping["formal_task_id"],
        "episode_seed": mapping["episode_seed"],
        "package_name": Path(package_path).name,
        "package_sha256": sha256_file(package_path),
        "diagnostic_csv_count": diagnostic_csv_count,
        "task_validation_rows": 1,
        "canonical_reconciliation_rows": len(reconciliation_rows),
    }
    canonical_rows = []
    for row in reconciliation_rows:
        canonical = {"task_id": mapping["task_id"], "scale": mapping["scale"], "algorithm": mapping["algorithm"]}
        canonical.update(row)
        canonical_rows.append(canonical)
    service_row = service_summary_from_rows(mapping, episode, charger_rows, transformer_rows)
    source_row = {
        "task_id": mapping["task_id"],
        "scale": mapping["scale"],
        "algorithm": mapping["algorithm"],
        "source_commit_sha": source_commit,
        "source_mode": source_resolution.get("source_mode", ""),
        "package_path": source_resolution.get("package_path", ""),
        "expected_package_name": source_resolution.get("expected_package_name", ""),
        "bundle_member": source_resolution.get("bundle_member", ""),
    }
    return {
        "task_id": mapping["task_id"],
        "inventory": inventory_row,
        "canonical_rows": canonical_rows,
        "service": service_row,
        "source": source_row,
        "warnings": warnings,
        "diagnostic_csv_count": diagnostic_csv_count,
    }


def write_complete_manifest(staging_root):
    staging_root = Path(staging_root)
    file_list_path = staging_root / "runtime_metadata/complete_file_list.txt"
    checksum_path = staging_root / "runtime_metadata/complete_file_checksums.sha256"
    existing_files = sorted(
        path.relative_to(staging_root).as_posix()
        for path in staging_root.rglob("*")
        if path.is_file() and path not in {file_list_path, checksum_path}
    )
    listed = existing_files + [
        "runtime_metadata/complete_file_list.txt",
        "runtime_metadata/complete_file_checksums.sha256",
    ]
    file_list_path.write_text("\n".join(listed) + "\n", encoding="utf-8")
    manifest_files = sorted(
        path for path in staging_root.rglob("*")
        if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in manifest_files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(staging_root).as_posix()}\n")


def create_complete_bundle(staging_root, bundle_path):
    with tarfile.open(bundle_path, "w:gz") as bundle:
        for path in sorted(Path(staging_root).rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(staging_root).as_posix())


def unique_temporary_output_path(output_root, final_name):
    stamp = f"{os.getpid()}.{time.time_ns()}"
    return Path(output_root) / f".{final_name}.tmp.{stamp}"


def validate_sha256_sidecar(archive_path, sidecar_path):
    archive_path = Path(archive_path)
    sidecar_path = Path(sidecar_path)
    line = sidecar_path.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if not match:
        raise ValidationError(f"invalid SHA-256 sidecar format: {sidecar_path}")
    digest, basename = match.groups()
    if basename != archive_path.name:
        raise ValidationError(f"SHA-256 sidecar basename mismatch: {basename} != {archive_path.name}")
    observed = sha256_file(archive_path)
    if digest != observed:
        raise ValidationError(f"SHA-256 sidecar digest mismatch for {archive_path}")


def write_validated_sha256_sidecar(archive_path, sidecar_path):
    archive_path = Path(archive_path)
    sidecar_path = Path(sidecar_path)
    temp_path = unique_temporary_output_path(sidecar_path.parent, sidecar_path.name)
    try:
        temp_path.write_text(f"{sha256_file(archive_path)}  {archive_path.name}\n", encoding="utf-8")
        validate_sha256_sidecar(archive_path, temp_path)
        os.replace(temp_path, sidecar_path)
        validate_sha256_sidecar(archive_path, sidecar_path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_complete_file_list(extract_root):
    path = Path(extract_root) / "runtime_metadata/complete_file_list.txt"
    if not path.is_file():
        raise ValidationError("missing runtime_metadata/complete_file_list.txt")
    return sorted(
        normalise_member_name(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def is_prohibited_complete_member(member_name):
    path = normalise_member_name(member_name)
    parts = PurePosixPath(path).parts
    if "__MACOSX" in parts:
        return True
    if any(part.startswith("._") for part in parts):
        return True
    return False


def complete_bundle_name(array_job_id):
    return f"infrastructure_diagnostic_smoke_complete_evidence_job{array_job_id}.tar.gz"


def checksum_sidecar_path(bundle_path):
    bundle_path = Path(bundle_path)
    return bundle_path.with_name(bundle_path.name + ".sha256")


def exact_complete_task_package_members(array_job_id):
    return sorted(
        f"task_packages/{smoke_task_package_name(TASKS[task_id], array_job_id)}"
        for task_id in sorted(TASKS)
    )


def exact_complete_log_members(array_job_id, suffix):
    return sorted(
        f"logs/{smoke_slurm_log_name(array_job_id, task_id, suffix)}"
        for task_id in sorted(TASKS)
    )


def validate_exact_member_set(observed, expected, label):
    observed = sorted(observed)
    expected = sorted(expected)
    if observed == expected:
        return
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    details = []
    if missing:
        details.append(f"missing expected {label}: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected {label}: {', '.join(unexpected)}")
    raise ValidationError("; ".join(details) or f"{label} mismatch")


def read_required_text(root, relative_path):
    path = Path(root) / relative_path
    if not path.is_file():
        raise ValidationError(f"missing required file: {relative_path}")
    return path.read_text(encoding="utf-8").strip()


def parse_unique_env_metadata(path, label):
    values = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValidationError(f"malformed {label} line {line_number}: {line!r}")
        key, value = stripped.split("=", 1)
        if key in values:
            raise ValidationError(f"duplicate {label} value for {key}")
        values[key] = value
    return values


def require_task_indexed_rows(rows, label):
    if len(rows) != len(TASKS):
        raise ValidationError(f"{label} must contain exactly 8 task rows, got {len(rows)}")
    indexed = {}
    for row in rows:
        task_id = integer_value(row.get("task_id"), f"{label} task_id")
        if task_id not in TASKS:
            raise ValidationError(f"{label} contains unexpected task_id: {task_id}")
        if task_id in indexed:
            raise ValidationError(f"{label} contains duplicate task_id: {task_id}")
        indexed[task_id] = row
    missing = sorted(set(TASKS) - set(indexed))
    if missing:
        raise ValidationError(f"{label} missing task_id(s): {missing}")
    return indexed


def require_row_value(row, field, expected, label):
    if str(row.get(field, "")) != str(expected):
        raise ValidationError(f"{label} mismatch for {field}: {row.get(field)} != {expected}")


def canonical_expected_types():
    expected = {field: "exact" for field, _, _, _ in EXACT_RECONCILIATION}
    expected.update({field: "floating" for field, _, _, _ in FLOAT_RECONCILIATION})
    return expected


def read_packaged_task_details(complete_extract_root, package_member, task_id, source_commit_sha, array_job_id):
    mapping = TASKS[task_id]
    package_path = Path(complete_extract_root) / package_member
    with contextlib.redirect_stdout(io.StringIO()):
        validate_task_package(Namespace(task_id=task_id, package=str(package_path)))

    task_extract_dir = Path(complete_extract_root) / "_validated_task_extracts" / f"task{task_id}"
    safe_extract_tar(package_path, task_extract_dir)
    task_validation = load_task_validation(task_extract_dir, mapping)
    if str(task_validation.get("matrix_job_id")) != str(array_job_id):
        raise ValidationError(
            f"task package task {task_id} matrix_job_id mismatch: "
            f"{task_validation.get('matrix_job_id')} != {array_job_id}"
        )
    packaged_source_commit = read_required_text(task_extract_dir, "runtime_metadata/source_commit_sha.txt")
    if packaged_source_commit != source_commit_sha:
        raise ValidationError(
            f"task package task {task_id} source commit mismatch: "
            f"{packaged_source_commit} != {source_commit_sha}"
        )
    source_resolution = read_json_file(
        task_extract_dir / "runtime_metadata/source_package_resolution.json",
        "source_package_resolution.json",
    )
    episode_rows, _ = read_rows(task_extract_dir / "diagnostics/episode_diagnostics.csv")
    charger_rows, _ = read_rows(task_extract_dir / "diagnostics/charger_diagnostics.csv")
    transformer_rows, _ = read_rows(task_extract_dir / "diagnostics/transformer_diagnostics.csv")
    reconciliation_rows, _ = read_rows(task_extract_dir / "validation/canonical_reconciliation.csv")
    return {
        "package_path": package_path,
        "task_validation": task_validation,
        "source_commit_sha": packaged_source_commit,
        "source": {
            "task_id": str(task_id),
            "scale": str(mapping["scale"]),
            "algorithm": str(mapping["algorithm"]),
            "source_commit_sha": packaged_source_commit,
            "source_mode": str(source_resolution.get("source_mode", "")),
            "package_path": str(source_resolution.get("package_path", "")),
            "expected_package_name": str(source_resolution.get("expected_package_name", "")),
            "bundle_member": str(source_resolution.get("bundle_member", "")),
        },
        "canonical_rows": reconciliation_rows,
        "service": service_summary_from_rows(mapping, episode_rows[0], charger_rows, transformer_rows),
    }


def validate_reducer_markers(extract_root, array_job_id):
    markers_path = Path(extract_root) / "runtime_metadata/reducer_markers.env"
    markers = parse_unique_env_metadata(markers_path, "reducer_markers.env")
    required_keys = set(EXPECTED_REDUCER_MARKERS) | {"COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_PATH"}
    missing = sorted(required_keys - set(markers))
    unexpected = sorted(set(markers) - required_keys)
    if missing:
        raise ValidationError(f"reducer_markers.env missing required marker(s): {', '.join(missing)}")
    if unexpected:
        raise ValidationError(f"reducer_markers.env contains unexpected marker(s): {', '.join(unexpected)}")
    for key, expected in EXPECTED_REDUCER_MARKERS.items():
        require_row_value(markers, key, expected, "reducer_markers.env")
    expected_name = complete_bundle_name(array_job_id)
    marker_path = markers["COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_PATH"]
    if marker_path != expected_name:
        raise ValidationError(
            "reducer_markers.env mismatch for COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_PATH: "
            f"{marker_path} != {expected_name}"
        )
    return markers


def validate_complete_bundle_cross_references(extract_root, task_packages, stdout_logs, stderr_logs):
    extract_root = Path(extract_root)
    source_commit_sha = read_required_text(extract_root, "runtime_metadata/source_commit_sha.txt")
    if not HEX_SHA1.fullmatch(source_commit_sha):
        raise ValidationError(f"source commit SHA must be 40 lowercase hex characters: {source_commit_sha}")
    array_job_id = read_required_text(extract_root, "runtime_metadata/array_job_id.txt")
    if not re.fullmatch(r"[0-9]+", array_job_id):
        raise ValidationError(f"array job ID must be numeric: {array_job_id}")
    reducer_job_id = read_required_text(extract_root, "runtime_metadata/reducer_job_id.txt")
    if not reducer_job_id:
        raise ValidationError("reducer_job_id.txt must not be empty")

    validate_exact_member_set(task_packages, exact_complete_task_package_members(array_job_id), "task package")
    validate_exact_member_set(stdout_logs, exact_complete_log_members(array_job_id, "out"), "Slurm stdout log")
    validate_exact_member_set(stderr_logs, exact_complete_log_members(array_job_id, "err"), "Slurm stderr log")
    validate_reducer_markers(extract_root, array_job_id)

    reducer_runtime = parse_unique_env_metadata(
        extract_root / "runtime_metadata/reducer_runtime_metadata.env",
        "reducer_runtime_metadata.env",
    )
    require_row_value(reducer_runtime, "array_job_id", array_job_id, "reducer_runtime_metadata.env")
    require_row_value(reducer_runtime, "reducer_job_id", reducer_job_id, "reducer_runtime_metadata.env")
    require_row_value(reducer_runtime, "task_package_count", "8", "reducer_runtime_metadata.env")
    require_row_value(reducer_runtime, "stdout_log_count", "8", "reducer_runtime_metadata.env")
    require_row_value(reducer_runtime, "stderr_log_count", "8", "reducer_runtime_metadata.env")
    require_row_value(reducer_runtime, "diagnostic_csv_count", "32", "reducer_runtime_metadata.env")
    if Path(str(reducer_runtime.get("complete_bundle_path", ""))).name != complete_bundle_name(array_job_id):
        raise ValidationError("reducer_runtime_metadata.env complete_bundle_path mismatch")

    packaged = {}
    for task_id in sorted(TASKS):
        package_member = f"task_packages/{smoke_task_package_name(TASKS[task_id], array_job_id)}"
        packaged[task_id] = read_packaged_task_details(
            extract_root,
            package_member,
            task_id,
            source_commit_sha,
            array_job_id,
        )

    inventory_rows, _ = read_rows(extract_root / "summaries/task_inventory.csv")
    inventory_by_task = require_task_indexed_rows(inventory_rows, "task_inventory")
    for task_id, row in inventory_by_task.items():
        mapping = TASKS[task_id]
        package_name = smoke_task_package_name(mapping, array_job_id)
        expected_inventory = {
            "scale": mapping["scale"],
            "algorithm": mapping["algorithm"],
            "formal_task_id": mapping["formal_task_id"],
            "episode_seed": mapping["episode_seed"],
            "package_name": package_name,
            "package_sha256": sha256_file(extract_root / "task_packages" / package_name),
            "diagnostic_csv_count": "4",
            "task_validation_rows": "1",
            "canonical_reconciliation_rows": str(len(EXACT_RECONCILIATION) + len(FLOAT_RECONCILIATION)),
        }
        for field, expected in expected_inventory.items():
            require_row_value(row, field, expected, "task_inventory")

    source_rows, _ = read_rows(extract_root / "summaries/source_provenance_summary.csv")
    source_by_task = require_task_indexed_rows(source_rows, "source_provenance_summary")
    for task_id, row in source_by_task.items():
        for field, expected in packaged[task_id]["source"].items():
            require_row_value(row, field, expected, "source provenance summary")

    runtime_rows, _ = read_rows(extract_root / "summaries/runtime_summary.csv")
    runtime_by_task = require_task_indexed_rows(runtime_rows, "runtime_summary")
    sacct_raw = (extract_root / "runtime_metadata/sacct_raw.txt").read_text(encoding="utf-8")
    parsed_runtime_by_task = {
        int(row["task_id"]): row
        for row in parse_sacct_raw(sacct_raw, array_job_id)
    }
    for task_id, row in runtime_by_task.items():
        for field in [
            "job_id_raw",
            "state",
            "exit_code",
            "elapsed_raw",
            "alloc_cpus",
            "max_rss",
            "total_cpu",
            "maxrss_source",
            "totalcpu_source",
        ]:
            require_row_value(row, field, parsed_runtime_by_task[task_id][field], "runtime_summary")

    expected_types = canonical_expected_types()
    canonical_rows, _ = read_rows(extract_root / "summaries/canonical_reconciliation_summary.csv")
    if len(canonical_rows) != len(TASKS) * len(expected_types):
        raise ValidationError(
            "canonical_reconciliation_summary.csv must contain exactly "
            f"{len(TASKS) * len(expected_types)} rows, got {len(canonical_rows)}"
        )
    canonical_by_task = {task_id: [] for task_id in TASKS}
    for row in canonical_rows:
        task_id = integer_value(row.get("task_id"), "canonical_reconciliation_summary task_id")
        if task_id not in TASKS:
            raise ValidationError(f"canonical_reconciliation_summary contains unexpected task_id: {task_id}")
        mapping = TASKS[task_id]
        require_row_value(row, "scale", mapping["scale"], "canonical_reconciliation_summary")
        require_row_value(row, "algorithm", mapping["algorithm"], "canonical_reconciliation_summary")
        field = row.get("field")
        if field not in expected_types:
            raise ValidationError(f"canonical_reconciliation_summary contains unexpected field: {field}")
        require_row_value(row, "comparison_type", expected_types[field], "canonical_reconciliation_summary")
        require_row_value(row, "pass", "True", "canonical_reconciliation_summary")
        canonical_by_task[task_id].append(row)
    comparison_fields = [
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
    for task_id, rows in canonical_by_task.items():
        seen = [row.get("field") for row in rows]
        if sorted(seen) != sorted(expected_types):
            raise ValidationError(f"canonical_reconciliation_summary duplicate or missing field(s) for task {task_id}")
        packaged_by_field = {row["field"]: row for row in packaged[task_id]["canonical_rows"]}
        for row in rows:
            packaged_row = packaged_by_field[row["field"]]
            for field in comparison_fields:
                require_row_value(row, field, packaged_row.get(field, ""), "canonical_reconciliation_summary")

    service_rows, _ = read_rows(extract_root / "summaries/service_reconciliation_summary.csv")
    service_by_task = require_task_indexed_rows(service_rows, "service_reconciliation_summary")
    for task_id, row in service_by_task.items():
        for field in SERVICE_SUMMARY_FIELDS:
            require_row_value(row, field, packaged[task_id]["service"][field], "service reconciliation summary")
        for field in [
            "served_reconciliation_pass",
            "satisfaction_reconciliation_pass",
            "charged_energy_reconciliation_pass",
            "discharged_energy_reconciliation_pass",
        ]:
            require_row_value(row, field, "True", "service reconciliation summary")

    return {
        "array_job_id": array_job_id,
        "source_commit_sha": source_commit_sha,
        "reducer_job_id": reducer_job_id,
    }


def validate_complete_bundle_file(bundle_path):
    bundle_path = Path(bundle_path)
    with open_tar(bundle_path) as bundle:
        members = safe_tar_members(bundle)
        member_names = sorted({
            normalise_member_name(member.name)
            for member in members
            if member.isfile()
        })
    prohibited = sorted(name for name in member_names if is_prohibited_complete_member(name))
    if prohibited:
        raise ValidationError(f"complete bundle contains prohibited member(s): {', '.join(prohibited)}")
    checkpoint_leaks = sorted(name for name in member_names if is_checkpoint_leak(name))
    if checkpoint_leaks:
        raise ValidationError(f"checkpoint bytes are forbidden in complete bundle: {', '.join(checkpoint_leaks)}")
    require_files_in_bundle = sorted(set(COMPLETE_BUNDLE_REQUIRED_FILES) - set(member_names))
    if require_files_in_bundle:
        raise ValidationError(f"complete bundle missing required file(s): {', '.join(require_files_in_bundle)}")
    task_packages = sorted(name for name in member_names if name.startswith("task_packages/") and name.endswith(".tar.gz"))
    stdout_logs = sorted(name for name in member_names if name.startswith("logs/") and name.endswith(".out"))
    stderr_logs = sorted(name for name in member_names if name.startswith("logs/") and name.endswith(".err"))
    if len(task_packages) != 8:
        raise ValidationError(f"complete bundle must contain 8 task packages, got {len(task_packages)}")
    if len(stdout_logs) != 8 or len(stderr_logs) != 8:
        raise ValidationError(
            f"complete bundle must contain 8 stdout and 8 stderr logs, got {len(stdout_logs)} and {len(stderr_logs)}"
        )
    verify_tar_manifest(
        bundle_path,
        "runtime_metadata/complete_file_checksums.sha256",
        required_coverage=set(member_names) - {"runtime_metadata/complete_file_checksums.sha256"},
        require_all_files=True,
    )
    with tempfile.TemporaryDirectory(prefix="infra_diag_smoke_complete_bundle_") as tmp_dir:
        extract_root = safe_extract_tar(bundle_path, tmp_dir)
        listed = read_complete_file_list(extract_root)
        if listed != member_names:
            raise ValidationError(
                "complete_file_list.txt does not exactly match complete bundle members: "
                f"listed={listed}, members={member_names}"
            )
        cross_reference = validate_complete_bundle_cross_references(
            extract_root,
            task_packages,
            stdout_logs,
            stderr_logs,
        )
    return {
        "status": "ok",
        "bundle": str(bundle_path),
        "files": len(member_names),
        "task_packages": len(task_packages),
        "stdout_logs": len(stdout_logs),
        "stderr_logs": len(stderr_logs),
        **cross_reference,
    }


def final_complete_bundle_path(output_root, array_job_id):
    return Path(output_root) / f"infrastructure_diagnostic_smoke_complete_evidence_job{array_job_id}.tar.gz"


def reduce_bundle(args):
    array_job_id = str(args.array_job_id)
    expected_source_commit = str(args.source_commit_sha).strip()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_path = final_complete_bundle_path(output_root, array_job_id)
    if bundle_path.exists():
        raise ValidationError(f"final complete evidence bundle already exists: {bundle_path}")
    sidecar_path = checksum_sidecar_path(bundle_path)
    if sidecar_path.exists():
        raise ValidationError(f"final complete evidence bundle checksum already exists: {sidecar_path}")

    package_root = Path(args.task_package_root)
    log_root = Path(args.slurm_log_root)
    expected_packages = [
        package_root / smoke_task_package_name(TASKS[task_id], array_job_id)
        for task_id in sorted(TASKS)
    ]
    observed_packages = sorted(package_root.glob("m3_infrastructure_diagnostic_smoke_*.tar.gz"))
    validate_expected_paths(observed_packages, expected_packages, "task packages")

    expected_stdout_logs = [
        log_root / smoke_slurm_log_name(array_job_id, task_id, "out")
        for task_id in sorted(TASKS)
    ]
    expected_stderr_logs = [
        log_root / smoke_slurm_log_name(array_job_id, task_id, "err")
        for task_id in sorted(TASKS)
    ]
    observed_stdout_logs = sorted(log_root.glob(f"evgnn_infra_diag_smoke_{array_job_id}_*.out"))
    observed_stderr_logs = sorted(log_root.glob(f"evgnn_infra_diag_smoke_{array_job_id}_*.err"))
    validate_expected_paths(observed_stdout_logs, expected_stdout_logs, "Slurm stdout logs")
    validate_expected_paths(observed_stderr_logs, expected_stderr_logs, "Slurm stderr logs")

    sacct_raw = collect_sacct_raw(args)
    runtime_rows = parse_sacct_raw(sacct_raw, array_job_id)

    work_root = Path(args.work_root) / f"job{array_job_id}"
    staging_root = work_root / "complete_bundle_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    for subdir in ["task_packages", "logs", "summaries", "runtime_metadata"]:
        (staging_root / subdir).mkdir(parents=True, exist_ok=True)

    warning_rows = []
    for task_id, stderr_log in enumerate(expected_stderr_logs):
        warning_rows.extend(scan_log_text(f"slurm:{stderr_log.name}", stderr_log.read_text(encoding="utf-8")))

    task_inventory_rows = []
    canonical_summary_rows = []
    service_summary_rows = []
    source_summary_rows = []
    diagnostic_csv_count = 0
    task_identities = []
    with tempfile.TemporaryDirectory(prefix="infra_diag_smoke_reducer_tasks_") as tmp_dir:
        for task_id in sorted(TASKS):
            mapping = TASKS[task_id]
            package_path = package_root / smoke_task_package_name(mapping, array_job_id)
            summary = validate_and_summarise_task_package(
                package_path,
                mapping,
                array_job_id,
                expected_source_commit,
                Path(tmp_dir),
            )
            task_identities.append(summary["task_id"])
            task_inventory_rows.append(summary["inventory"])
            canonical_summary_rows.extend(summary["canonical_rows"])
            service_summary_rows.append(summary["service"])
            source_summary_rows.append(summary["source"])
            warning_rows.extend(summary["warnings"])
            diagnostic_csv_count += summary["diagnostic_csv_count"]
            shutil.copy2(package_path, staging_root / "task_packages" / package_path.name)
    if sorted(task_identities) != list(sorted(TASKS)):
        raise ValidationError(f"duplicate or missing task identities: {task_identities}")
    if diagnostic_csv_count != 32:
        raise ValidationError(f"expected 32 internal diagnostic CSV files, got {diagnostic_csv_count}")

    for stdout_log in expected_stdout_logs:
        shutil.copy2(stdout_log, staging_root / "logs" / stdout_log.name)
    for stderr_log in expected_stderr_logs:
        shutil.copy2(stderr_log, staging_root / "logs" / stderr_log.name)

    write_csv_rows(
        staging_root / "summaries/task_inventory.csv",
        [
            "task_id",
            "scale",
            "algorithm",
            "formal_task_id",
            "episode_seed",
            "package_name",
            "package_sha256",
            "diagnostic_csv_count",
            "task_validation_rows",
            "canonical_reconciliation_rows",
        ],
        task_inventory_rows,
    )
    write_csv_rows(
        staging_root / "summaries/runtime_summary.csv",
        [
            "task_id",
            "job_id_raw",
            "state",
            "exit_code",
            "elapsed_raw",
            "alloc_cpus",
            "max_rss",
            "total_cpu",
            "maxrss_source",
            "totalcpu_source",
        ],
        runtime_rows,
    )
    write_csv_rows(
        staging_root / "summaries/canonical_reconciliation_summary.csv",
        [
            "task_id",
            "scale",
            "algorithm",
            "field",
            "comparison_type",
            "canonical_value",
            "diagnostic_value",
            "absolute_difference",
            "relative_difference",
            "absolute_tolerance",
            "relative_tolerance",
            "pass",
        ],
        canonical_summary_rows,
    )
    write_csv_rows(
        staging_root / "summaries/service_reconciliation_summary.csv",
        SERVICE_SUMMARY_FIELDS,
        service_summary_rows,
    )
    write_csv_rows(
        staging_root / "summaries/source_provenance_summary.csv",
        [
            "task_id",
            "scale",
            "algorithm",
            "source_commit_sha",
            "source_mode",
            "package_path",
            "expected_package_name",
            "bundle_member",
        ],
        source_summary_rows,
    )
    write_csv_rows(
        staging_root / "summaries/warning_inventory.csv",
        ["source", "line_number", "warning"],
        warning_rows,
    )
    write_csv_rows(
        staging_root / "summaries/failure_manifest.csv",
        ["failure_id", "severity", "description"],
        [],
    )

    runtime_metadata = staging_root / "runtime_metadata"
    (runtime_metadata / "source_commit_sha.txt").write_text(expected_source_commit + "\n", encoding="utf-8")
    (runtime_metadata / "array_job_id.txt").write_text(array_job_id + "\n", encoding="utf-8")
    (runtime_metadata / "reducer_job_id.txt").write_text(str(args.reducer_job_id) + "\n", encoding="utf-8")
    (runtime_metadata / "sacct_raw.txt").write_text(sacct_raw, encoding="utf-8")
    stdout_snapshot = Path(args.reducer_stdout_log)
    stderr_snapshot = Path(args.reducer_stderr_log)
    (runtime_metadata / "reducer_stdout_snapshot.log").write_text(
        stdout_snapshot.read_text(encoding="utf-8") if stdout_snapshot.is_file() else "",
        encoding="utf-8",
    )
    (runtime_metadata / "reducer_stderr_snapshot.log").write_text(
        stderr_snapshot.read_text(encoding="utf-8") if stderr_snapshot.is_file() else "",
        encoding="utf-8",
    )
    reducer_runtime = {
        "array_job_id": array_job_id,
        "reducer_job_id": str(args.reducer_job_id),
        "task_package_count": "8",
        "stdout_log_count": "8",
        "stderr_log_count": "8",
        "diagnostic_csv_count": str(diagnostic_csv_count),
        "warning_count": str(len(warning_rows)),
        "complete_bundle_path": str(bundle_path),
    }
    (runtime_metadata / "reducer_runtime_metadata.env").write_text(
        "".join(f"{key}={value}\n" for key, value in reducer_runtime.items()),
        encoding="utf-8",
    )
    markers = {
        "TASK_PACKAGE_COUNT": "8",
        "STDOUT_LOG_COUNT": "8",
        "STDERR_LOG_COUNT": "8",
        "DIAGNOSTIC_CSV_COUNT": str(diagnostic_csv_count),
        "ALL_TASK_CHECKSUMS_OK": "1",
        "ALL_SCHEMA_VERSION_3": "1",
        "ALL_CANONICAL_RECONCILIATIONS_OK": "1",
        "ALL_SERVICE_RECONCILIATIONS_OK": "1",
        "ALL_ACTION_CONTRACTS_OK": "1",
        "ALL_RUNTIME_METADATA_PRESENT": "1",
        "ALL_SLURM_TASKS_COMPLETED": "1",
        "COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_OK": "1",
        "COMPLETE_DIAGNOSTIC_SMOKE_BUNDLE_PATH": bundle_path.name,
        "INFRASTRUCTURE_DIAGNOSTIC_SMOKE_REDUCER_COMPLETED": "1",
    }
    (runtime_metadata / "reducer_markers.env").write_text(
        "".join(f"{key}={value}\n" for key, value in markers.items()),
        encoding="utf-8",
    )

    write_complete_manifest(staging_root)
    temp_bundle_path = unique_temporary_output_path(output_root, bundle_path.name)
    published_bundle = False
    publication_complete = False
    try:
        create_complete_bundle(staging_root, temp_bundle_path)
        with contextlib.redirect_stdout(io.StringIO()):
            validation = validate_complete_bundle_file(temp_bundle_path)
        os.replace(temp_bundle_path, bundle_path)
        published_bundle = True
        write_validated_sha256_sidecar(bundle_path, sidecar_path)
        publication_complete = True
    except Exception:
        temp_bundle_path.unlink(missing_ok=True)
        if published_bundle and not publication_complete:
            bundle_path.unlink(missing_ok=True)
            sidecar_path.unlink(missing_ok=True)
        raise
    payload = {
        "status": "ok",
        "array_job_id": array_job_id,
        "bundle_path": str(bundle_path),
        "checksum_path": str(sidecar_path),
        "diagnostic_csv_count": diagnostic_csv_count,
        "warning_count": len(warning_rows),
        **{f"validated_{key}": value for key, value in validation.items() if key != "status"},
    }
    print(json.dumps(payload, sort_keys=True))


def validate_complete_bundle(args):
    payload = validate_complete_bundle_file(args.bundle)
    print(json.dumps(payload, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(description="Validate Stage C3.2A infrastructure diagnostic smoke evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    task_parser = subparsers.add_parser("task-mapping")
    task_parser.add_argument("--task-id", type=int)

    resolve_parser = subparsers.add_parser("resolve-package")
    resolve_parser.add_argument("--task-id", type=int, required=True)
    resolve_parser.add_argument("--individual-package-root", required=True)
    resolve_parser.add_argument("--complete-bundle", required=True)
    resolve_parser.add_argument("--staging-dir", required=True)

    formal_parser = subparsers.add_parser("validate-formal-package")
    formal_parser.add_argument("--task-id", type=int, required=True)
    formal_parser.add_argument("--package", required=True)
    formal_parser.add_argument("--extract-dir", required=True)

    diagnostics_parser = subparsers.add_parser("validate-diagnostics")
    diagnostics_parser.add_argument("--task-id", type=int, required=True)
    diagnostics_parser.add_argument("--diagnostic-dir", required=True)
    diagnostics_parser.add_argument("--validation-dir", required=True)
    diagnostics_parser.add_argument("--matrix-job-id")

    reconcile_parser = subparsers.add_parser("reconcile-canonical")
    reconcile_parser.add_argument("--task-id", type=int, required=True)
    reconcile_parser.add_argument("--episode-diagnostics", required=True)
    reconcile_parser.add_argument("--canonical-csv", required=True)
    reconcile_parser.add_argument("--validation-dir", required=True)

    task_package_parser = subparsers.add_parser("validate-task-package")
    task_package_parser.add_argument("--task-id", type=int, required=True)
    task_package_parser.add_argument("--package", required=True)

    reduce_parser = subparsers.add_parser("reduce-bundle")
    reduce_parser.add_argument("--array-job-id", required=True)
    reduce_parser.add_argument("--task-package-root", required=True)
    reduce_parser.add_argument("--slurm-log-root", required=True)
    reduce_parser.add_argument("--output-root", required=True)
    reduce_parser.add_argument("--work-root", required=True)
    reduce_parser.add_argument("--source-commit-sha", required=True)
    reduce_parser.add_argument("--reducer-job-id", required=True)
    reduce_parser.add_argument("--sacct-raw-file")
    reduce_parser.add_argument("--sacct-command", default="sacct")
    reduce_parser.add_argument("--sacct-attempts", type=int, default=6)
    reduce_parser.add_argument("--sacct-delay-seconds", type=float, default=10.0)
    reduce_parser.add_argument("--reducer-stdout-log", required=True)
    reduce_parser.add_argument("--reducer-stderr-log", required=True)

    complete_parser = subparsers.add_parser("validate-complete-bundle")
    complete_parser.add_argument("--bundle", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "task-mapping":
        if args.task_id is None:
            print(json.dumps([TASKS[index] for index in sorted(TASKS)], sort_keys=True))
        else:
            print(json.dumps(task_mapping(args.task_id), sort_keys=True))
    elif args.command == "resolve-package":
        resolve_package(args)
    elif args.command == "validate-formal-package":
        validate_formal_package(args)
    elif args.command == "validate-diagnostics":
        validate_diagnostics(args)
    elif args.command == "reconcile-canonical":
        reconcile_canonical(args)
    elif args.command == "validate-task-package":
        validate_task_package(args)
    elif args.command == "reduce-bundle":
        reduce_bundle(args)
    elif args.command == "validate-complete-bundle":
        validate_complete_bundle(args)
    else:
        raise ValidationError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
