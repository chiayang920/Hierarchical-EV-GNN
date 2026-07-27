#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tarfile
import tempfile
from argparse import Namespace
from pathlib import Path, PurePosixPath

import yaml


FORMAL_JOB_ID = "58513929"
SCHEMA_VERSION = "3"
HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")

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
    else:
        raise ValidationError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
