#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path, PurePosixPath


FORMAL_JOB_ID = "58513929"
SCHEMA_VERSION = "3"

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
    path = PurePosixPath(str(name))
    if str(name).startswith("/") or path.is_absolute():
        raise ValidationError(f"unsafe absolute tar member path: {name}")
    if any(part == ".." for part in path.parts):
        raise ValidationError(f"unsafe tar member path contains '..': {name}")
    if str(path) in {"", "."}:
        raise ValidationError(f"unsafe empty tar member path: {name}")
    return path.as_posix()


def reject_unsafe_tar_member(member):
    normalise_member_name(member.name)
    if member.issym() or member.islnk():
        link_target = PurePosixPath(member.linkname)
        if str(member.linkname).startswith("/") or link_target.is_absolute():
            raise ValidationError(f"unsafe tar link target: {member.name} -> {member.linkname}")
        if any(part == ".." for part in link_target.parts):
            raise ValidationError(f"unsafe tar link target: {member.name} -> {member.linkname}")
        raise ValidationError(f"tar links are not allowed in smoke evidence: {member.name}")


def open_tar(path):
    try:
        return tarfile.open(path, "r:gz")
    except tarfile.TarError as exc:
        raise ValidationError(f"unreadable tar package: {path}") from exc


def safe_tar_members(tar):
    members = tar.getmembers()
    for member in members:
        reject_unsafe_tar_member(member)
    return members


def safe_extract_tar(package_path, extract_dir):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with open_tar(package_path) as tar:
        members = safe_tar_members(tar)
        for member in members:
            target = (root / normalise_member_name(member.name)).resolve()
            if root not in [target, *target.parents]:
                raise ValidationError(f"unsafe tar extraction target: {member.name}")
            tar.extract(member, path=root)
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
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            digest, relative = stripped.split(None, 1)
        except ValueError as exc:
            raise ValidationError(f"invalid checksum manifest line: {line!r}") from exc
        entries.append((digest, relative.strip()))
    return entries


def verify_extracted_manifest(root, manifest_path):
    manifest = Path(root) / manifest_path
    if not manifest.is_file() or manifest.stat().st_size == 0:
        raise ValidationError(f"missing required checksum manifest: {manifest_path}")
    for expected_digest, relative_path in parse_manifest_text(manifest.read_text(encoding="utf-8")):
        normalised = normalise_member_name(relative_path)
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


def verify_tar_manifest(package_path, manifest_member):
    with open_tar(package_path) as tar:
        members = safe_tar_members(tar)
        member_by_name = {normalise_member_name(member.name): member for member in members if member.isfile()}
        if manifest_member not in member_by_name:
            raise ValidationError(f"missing required checksum manifest: {manifest_member}")
        manifest_file = tar.extractfile(member_by_name[manifest_member])
        if manifest_file is None:
            raise ValidationError(f"cannot read checksum manifest: {manifest_member}")
        entries = parse_manifest_text(manifest_file.read().decode("utf-8"))
        for expected_digest, relative_path in entries:
            normalised = normalise_member_name(relative_path)
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


def require_files(root, relative_paths):
    missing = []
    for relative in relative_paths:
        path = Path(root) / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    if missing:
        raise ValidationError(f"missing required file(s): {', '.join(missing)}")


def read_rows(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError(f"missing or empty CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValidationError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


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


def optional_numeric(value):
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def bool_value(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


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


def validate_signed_rows(rows, label):
    for index, row in enumerate(rows):
        active = optional_numeric(row.get("active_action_decision_count", row.get("n_active_ev_decisions", "")))
        if active is not None and active <= 0:
            continue
        fractions = [
            optional_numeric(row.get("global_positive_action_fraction_active", row.get("positive_action_fraction_active", ""))),
            optional_numeric(row.get("global_zero_action_fraction_active", row.get("zero_action_fraction_active", ""))),
            optional_numeric(row.get("global_negative_action_fraction_active", row.get("negative_action_fraction_active", ""))),
        ]
        if any(value is None for value in fractions):
            continue
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
            value = optional_numeric(row.get(field, ""))
            if value is not None and abs(value) > 1e-6:
                raise ValidationError(f"hierarchical negative action fraction is non-zero: {field}={value}")


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
    if str(episode_row.get("episode_index")) != "0":
        raise ValidationError("diagnostic episode_index must be 0")
    if int(float(episode_row.get("episode_seed", "-1"))) != int(mapping["episode_seed"]):
        raise ValidationError("diagnostic episode_seed mismatch")
    if int(float(episode_row.get("episode_steps", "-1"))) != 112:
        raise ValidationError("diagnostic episode_steps mismatch")
    if not bool_value(episode_row.get("done")):
        raise ValidationError("diagnostic episode done must be true")


def validate_schema(rows, label):
    for row in rows:
        if str(row.get("diagnostic_schema_version")) != SCHEMA_VERSION:
            raise ValidationError(f"{label} schema version mismatch")


def validate_v2g_false(rows):
    for row in rows:
        if "v2g_enabled" in row and str(row.get("v2g_enabled")).strip().lower() not in {"false", "0"}:
            raise ValidationError("v2g_enabled must be false for PublicPST formal diagnostics")


def validate_service_reconciliation(episode_row, charger_rows, transformer_rows):
    served_episode = numeric(episode_row.get("total_ev_served"), "total_ev_served")
    charged_episode = numeric(episode_row.get("total_energy_charged"), "total_energy_charged")
    discharged_episode = numeric(episode_row.get("total_energy_discharged"), "total_energy_discharged")
    assert_close("charger served_ev_count", sum_available(charger_rows, "served_ev_count"), served_episode)
    assert_close("transformer served_ev_count", sum_available(transformer_rows, "served_ev_count"), served_episode)
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
        if numeric(row.get("served_ev_count"), "served_ev_count") > 0 and not row.get("user_satisfaction_source"):
            raise ValidationError("explicit satisfaction source is required when served count > 0")

    chargers_by_transformer = {}
    for row in charger_rows:
        chargers_by_transformer.setdefault(str(row.get("transformer_id")), []).append(row)
    for transformer_row in transformer_rows:
        transformer_id = str(transformer_row.get("transformer_id"))
        chargers = chargers_by_transformer.get(transformer_id, [])
        assert_close(
            f"transformer {transformer_id} satisfaction sum",
            sum_available(chargers, "user_satisfaction_sum"),
            numeric(transformer_row.get("user_satisfaction_sum"), "user_satisfaction_sum"),
        )
        assert_close(
            f"transformer {transformer_id} satisfaction count",
            sum_available(chargers, "user_satisfaction_observation_count"),
            numeric(transformer_row.get("user_satisfaction_observation_count"), "user_satisfaction_observation_count"),
        )


def validate_diagnostics(args):
    mapping = task_mapping(args.task_id)
    diagnostic_dir = Path(args.diagnostic_dir)
    episode_rows, _ = read_rows(diagnostic_dir / "episode_diagnostics.csv")
    seed_rows, _ = read_rows(diagnostic_dir / "seed_summary_diagnostics.csv")
    transformer_rows, _ = read_rows(diagnostic_dir / "transformer_diagnostics.csv")
    charger_rows, _ = read_rows(diagnostic_dir / "charger_diagnostics.csv")

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

    all_rows = episode_rows + seed_rows + charger_rows + transformer_rows
    validate_schema(episode_rows, "episode")
    validate_schema(seed_rows, "seed summary")
    validate_schema(charger_rows, "charger")
    validate_schema(transformer_rows, "transformer")
    validate_v2g_false(all_rows)
    validate_diagnostic_identity(mapping, episode_rows[0], seed_rows[0])
    validate_signed_rows(episode_rows, "episode")
    validate_signed_rows(charger_rows, "charger")
    validate_signed_rows(transformer_rows, "transformer")
    if numeric(episode_rows[0].get("inactive_nonzero_action_count"), "inactive_nonzero_action_count") != 0:
        raise ValidationError("inactive non-zero action count must equal zero")
    inactive_mean = optional_numeric(seed_rows[0].get("inactive_nonzero_action_count_mean", ""))
    if inactive_mean is not None and inactive_mean != 0:
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
        payload = {
            "source_mode": "individual",
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
            if member.isfile() and Path(member_name).name == expected_name:
                matches.append(member)
        if len(matches) != 1:
            raise ValidationError(
                f"expected exactly one nested task package named {expected_name}, found {len(matches)}"
            )
        staging_dir = Path(args.staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        output_path = staging_dir / expected_name
        payload = tar.extractfile(matches[0])
        if payload is None:
            raise ValidationError(f"cannot read nested package: {matches[0].name}")
        output_path.write_bytes(payload.read())
    with open_tar(output_path) as nested_tar:
        safe_tar_members(nested_tar)
    print(
        json.dumps(
            {
                "source_mode": "complete_bundle",
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
    verify_extracted_manifest(extract_root, "runtime_metadata/package_file_checksums.sha256")
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


def reconciliation_row(field, canonical_value, diagnostic_value, absolute_tolerance, relative_tolerance):
    canonical_number = float(canonical_value)
    diagnostic_number = float(diagnostic_value)
    absolute_difference = abs(diagnostic_number - canonical_number)
    relative = relative_difference(diagnostic_number, canonical_number)
    passed = absolute_difference <= absolute_tolerance
    if not passed and relative_tolerance:
        passed = relative <= relative_tolerance
    return {
        "field": field,
        "canonical value": canonical_value,
        "diagnostic value": diagnostic_value,
        "absolute difference": absolute_difference,
        "relative difference": relative,
        "absolute tolerance": absolute_tolerance,
        "relative tolerance": relative_tolerance,
        "pass": str(bool(passed)),
    }


def reconcile_canonical(args):
    mapping = task_mapping(args.task_id)
    diagnostic_rows, _ = read_rows(args.episode_diagnostics)
    if len(diagnostic_rows) != 1:
        raise ValidationError(f"expected 1 diagnostic episode row, got {len(diagnostic_rows)}")
    diagnostic = diagnostic_rows[0]
    canonical = canonical_episode_row(args.canonical_csv, 0)

    exact_checks = [
        ("algorithm", canonical.get("algorithm"), diagnostic.get("algorithm")),
        ("training seed", canonical.get("seed"), diagnostic.get("training_seed")),
        ("episode index", canonical.get("episode_index"), diagnostic.get("episode_index")),
        ("episode seed", canonical.get("episode_seed"), diagnostic.get("episode_seed")),
        ("episode steps", canonical.get("episode_steps"), diagnostic.get("episode_steps")),
        ("done", str(bool_value(canonical.get("done"))), str(bool_value(diagnostic.get("done")))),
        ("total_ev_served", str(int(float(canonical.get("total_ev_served")))), str(int(float(diagnostic.get("total_ev_served"))))),
    ]
    failures = []
    for field, canonical_value, diagnostic_value in exact_checks:
        if str(canonical_value) != str(diagnostic_value):
            failures.append(f"{field}: {diagnostic_value} != {canonical_value}")
    if diagnostic.get("algorithm") != mapping["algorithm"]:
        failures.append("diagnostic algorithm does not match task mapping")
    if int(float(diagnostic.get("episode_seed"))) != mapping["episode_seed"]:
        failures.append("diagnostic episode seed does not match task mapping")

    rows = []
    for diagnostic_field, canonical_field, absolute_tolerance, relative_tolerance in FLOAT_RECONCILIATION:
        rows.append(
            reconciliation_row(
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
            "canonical value",
            "diagnostic value",
            "absolute difference",
            "relative difference",
            "absolute tolerance",
            "relative tolerance",
            "pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    failures.extend(row["field"] for row in rows if row["pass"] != "True")
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
        member_names = {
            normalise_member_name(member.name)
            for member in members
            if member.isfile()
        }
    leaks = sorted(name for name in member_names if is_checkpoint_leak(name))
    if leaks:
        raise ValidationError(f"checkpoint bytes are forbidden in task package: {', '.join(leaks)}")
    missing = sorted(set(required) - member_names)
    if missing:
        raise ValidationError(f"missing required task package file(s): {', '.join(missing)}")
    verify_tar_manifest(package_path, "runtime_metadata/package_file_checksums.sha256")
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
