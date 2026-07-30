#!/usr/bin/env python3
"""Validation contracts for the full per-infrastructure diagnostic eval30 workflow."""

from __future__ import annotations

import argparse
import collections
import contextlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Final
import io
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

REPO_ROOT_FOR_IMPORTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT_FOR_IMPORTS not in sys.path:
    sys.path.insert(0, REPO_ROOT_FOR_IMPORTS)

from utils.infrastructure_diagnostics import (
    CHARGER_DIAGNOSTIC_COLUMNS,
    EPISODE_DIAGNOSTIC_COLUMNS,
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
)

TRAINING_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
EVAL_EPISODES: Final[int] = 30

SCALE_SEED_OFFSETS: Final[dict[str, int]] = {
    "25cp": 710000,
    "100cp": 720000,
    "500cp": 730000,
    "1000cp": 740000,
}

STAGE_D_TASKS: Final[dict[int, dict[str, object]]] = {
    0: {"scale": "25cp", "algorithm": "actiongnn", "formal_start": 0},
    1: {"scale": "25cp", "algorithm": "hierarchical", "formal_start": 5},
    2: {"scale": "100cp", "algorithm": "actiongnn", "formal_start": 10},
    3: {"scale": "100cp", "algorithm": "hierarchical", "formal_start": 15},
    4: {"scale": "500cp", "algorithm": "actiongnn", "formal_start": 20},
    5: {"scale": "500cp", "algorithm": "hierarchical", "formal_start": 25},
    6: {"scale": "1000cp", "algorithm": "actiongnn", "formal_start": 30},
    7: {"scale": "1000cp", "algorithm": "hierarchical", "formal_start": 35},
}


def stage_d_task(task_id: int) -> dict[str, object]:
    """Return a defensive copy of one exact Stage D task mapping."""
    if type(task_id) is not int or task_id not in STAGE_D_TASKS:
        raise ValueError(f"unsupported Stage D task ID: {task_id!r}")
    return dict(STAGE_D_TASKS[task_id])


def formal_task_id(task_id: int, training_seed: int) -> int:
    """Resolve the exact formal-job task ID for one Stage D task and seed."""
    if type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed: {training_seed!r}")
    return int(stage_d_task(task_id)["formal_start"]) + training_seed


def episode_seed(scale: str, training_seed: int, episode_index: int) -> int:
    """Resolve the deterministic episode seed used by canonical eval30."""
    if scale not in SCALE_SEED_OFFSETS:
        raise ValueError(f"unsupported scale: {scale!r}")
    if type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed: {training_seed!r}")
    if type(episode_index) is not int or not 0 <= episode_index < EVAL_EPISODES:
        raise ValueError(f"episode index must be 0..{EVAL_EPISODES - 1}")
    return SCALE_SEED_OFFSETS[scale] + (1000 * training_seed) + episode_index


def evaluator_seed_offset(scale: str, training_seed: int) -> int:
    """Resolve the offset passed to the evaluator CLI for one training seed."""
    if scale not in SCALE_SEED_OFFSETS:
        raise ValueError(f"unsupported scale: {scale!r}")
    if type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed: {training_seed!r}")
    return SCALE_SEED_OFFSETS[scale] + (999 * training_seed)


def task_mapping_lines(task_id: int) -> list[str]:
    """Render the stable shell-readable task-mapping contract."""
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    formal_ids = ",".join(
        str(formal_task_id(task_id, training_seed))
        for training_seed in TRAINING_SEEDS
    )
    eval_seed_offsets = ",".join(
        str(evaluator_seed_offset(scale, training_seed))
        for training_seed in TRAINING_SEEDS
    )
    first_episode_seeds = ",".join(
        str(episode_seed(scale, training_seed, 0))
        for training_seed in TRAINING_SEEDS
    )
    return [
        f"task_id={task_id}",
        f"scale={scale}",
        f"algorithm={task['algorithm']}",
        "training_seeds=" + ",".join(map(str, TRAINING_SEEDS)),
        f"formal_task_ids={formal_ids}",
        f"eval_episodes={EVAL_EPISODES}",
        f"scale_base_seed_offset={SCALE_SEED_OFFSETS[scale]}",
        f"eval_seed_offsets={eval_seed_offsets}",
        f"first_episode_seeds={first_episode_seeds}",
    ]


import csv
import json
import math
from pathlib import Path
from typing import TypeAlias


EpisodeKey: TypeAlias = tuple[str, str, int, int]


@dataclass(frozen=True)
class SeedReconciliationInputs:
    diagnostic_rows: list[dict[str, str]]
    transformer_rows: list[dict[str, str]]
    charger_rows: list[dict[str, str]]
    historical_canonical_rows: list[dict[str, str]]
    same_pass_canonical_rows: list[dict[str, str]]
    formal_validation: dict[str, object]
    scale: str
    algorithm: str
    task_id: int
    training_seed: int
    stage_d_source_commit_sha: str
    config_path: Path | None
    checkpoint_prefix: Path | None


TOPOLOGY: Final[dict[str, tuple[int, int]]] = {
    "25cp": (25, 3),
    "100cp": (100, 7),
    "500cp": (500, 35),
    "1000cp": (1000, 70),
}

ALGORITHMS: Final[tuple[str, ...]] = ("actiongnn", "hierarchical")
SCHEMA_VERSION: Final[str] = "3"
RECONCILIATION_CONTRACT_VERSION: Final[int] = 2
SAME_PASS_CANONICAL_FILENAME: Final[str] = "same_pass_canonical_eval30.csv"
SAME_PASS_RECONCILIATION_FILENAME: Final[str] = (
    "same_pass_canonical_reconciliation.csv"
)
HISTORICAL_DRIFT_FILENAME: Final[str] = "historical_canonical_drift.csv"
RECONCILIATION_SUMMARY_FILENAME: Final[str] = "reconciliation_summary.json"

SAME_PASS_RECONCILIATION_COLUMNS: Final[tuple[str, ...]] = (
    "reconciliation_contract_version",
    "episode_index",
    "field",
    "comparison_type",
    "same_pass_canonical_value",
    "diagnostic_value",
    "absolute_difference",
    "relative_difference",
    "same_pass_count",
    "diagnostic_count",
    "total_action_decision_denominator",
    "status",
    "failure_category",
)

SAME_PASS_FIELD_MAP: Final[tuple[tuple[str, str, str], ...]] = (
    ("episode_reward", "episode_reward", "float_exact"),
    ("tracking_error", "tracking_error", "float_exact"),
    ("energy_tracking_error", "energy_tracking_error", "float_exact"),
    ("power_tracker_violation", "power_tracker_violation", "float_exact"),
    ("total_energy_charged", "total_energy_charged", "float_exact"),
    ("total_energy_discharged", "total_energy_discharged", "float_exact"),
    ("average_user_satisfaction", "average_user_satisfaction", "float_exact"),
    ("energy_user_satisfaction", "energy_user_satisfaction", "float_exact"),
    ("total_transformer_overload", "total_transformer_overload", "float_exact"),
    ("action_mean", "global_action_mean_all_slots", "float_exact"),
    (
        "action_fraction_at_max",
        "global_action_fraction_at_max_all_slots",
        "fraction_count_exact",
    ),
    ("active_action_count_mean", "nonzero_action_count_mean_all_slots", "float_exact"),
)

EPISODE_REQUIRED_NUMERIC_FIELDS: Final[tuple[str, ...]] = (
    "episode_index",
    "episode_seed",
    "episode_steps",
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "total_energy_charged",
    "total_energy_discharged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "total_transformer_overload",
    "total_ev_served",
    "global_action_fraction_at_max_active",
)

SERVICE_STATUS_FIELDS: Final[tuple[str, ...]] = (
    "served_count_reconciliation_status",
    "satisfaction_sum_reconciliation_status",
    "charged_energy_reconciliation_status",
    "discharged_energy_reconciliation_status",
)

CANONICAL_EXACT_RECONCILIATION: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("algorithm", "algorithm", "algorithm", "string"),
    ("training seed", "seed", "training_seed", "integer"),
    ("episode index", "episode_index", "episode_index", "integer"),
    ("episode seed", "episode_seed", "episode_seed", "integer"),
    ("episode steps", "episode_steps", "episode_steps", "integer"),
    ("done", "done", "done", "boolean"),
    ("total_ev_served", "total_ev_served", "total_ev_served", "integer"),
)

CANONICAL_FLOAT_RECONCILIATION: Final[tuple[tuple[str, str, float, float], ...]] = (
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
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _require_columns(
    fieldnames: list[str],
    required_columns: set[str],
    label: str,
) -> None:
    missing = sorted(required_columns - set(fieldnames))
    if missing:
        raise ValueError(
            f"{label} missing required column(s): {', '.join(missing)}"
        )


def _require_exact_columns(
    fieldnames: list[str],
    expected_columns: list[str],
    label: str,
) -> None:
    if fieldnames != expected_columns:
        missing = sorted(set(expected_columns) - set(fieldnames))
        unexpected = sorted(set(fieldnames) - set(expected_columns))
        raise ValueError(
            f"{label} column contract mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _finite_number(value: object, field: str) -> float:
    if value in ("", None):
        raise ValueError(f"blank required metric: {field}")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric value for {field}: {value!r}")
    return parsed


def _exact_integer(value: object, field: str) -> int:
    parsed = _finite_number(value, field)
    if not parsed.is_integer():
        raise ValueError(f"{field} must be integral: {value!r}")
    return int(parsed)


def _validate_identity(
    rows: list[dict[str, str]],
    *,
    scale: str,
    algorithm: str,
    training_seed: int,
    label: str,
) -> None:
    for row_index, row in enumerate(rows):
        observed = (
            row.get("scale"),
            row.get("algorithm"),
            _exact_integer(row.get("training_seed"), "training_seed"),
        )
        expected = (scale, algorithm, training_seed)
        if observed != expected:
            raise ValueError(
                f"{label} row {row_index} identity mismatch: "
                f"expected={expected}, observed={observed}"
            )
        if row.get("diagnostic_schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{label} row {row_index} diagnostic schema must be "
                f"{SCHEMA_VERSION}"
            )


def expected_episode_keys() -> set[EpisodeKey]:
    return {
        (scale, algorithm, training_seed, episode_index)
        for scale in SCALE_SEED_OFFSETS
        for algorithm in ALGORITHMS
        for training_seed in TRAINING_SEEDS
        for episode_index in range(EVAL_EPISODES)
    }


def validate_episode_inventory(rows: list[dict[str, str]]) -> None:
    observed: set[EpisodeKey] = set()

    for row_number, row in enumerate(rows, start=2):
        scale = str(row.get("scale", ""))
        algorithm = str(row.get("algorithm", ""))
        training_seed = _exact_integer(row.get("training_seed"), "training_seed")
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        key = (scale, algorithm, training_seed, episode_index)

        if key in observed:
            raise ValueError(f"duplicate episode key at CSV row {row_number}: {key}")
        observed.add(key)

        expected_seed = episode_seed(scale, training_seed, episode_index)
        observed_seed = _exact_integer(row.get("episode_seed"), "episode_seed")
        if observed_seed != expected_seed:
            raise ValueError(
                f"episode seed mismatch for {key}: "
                f"expected={expected_seed}, observed={observed_seed}"
            )

        if row.get("diagnostic_schema_version") != SCHEMA_VERSION:
            raise ValueError(f"diagnostic schema must be {SCHEMA_VERSION} for {key}")

    expected = expected_episode_keys()
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    if missing:
        raise ValueError(f"missing episode key: {missing[0]}")
    if unexpected:
        raise ValueError(f"unexpected episode key: {unexpected[0]}")


def _validate_per_seed_episode_rows(
    rows: list[dict[str, str]],
    *,
    scale: str,
    algorithm: str,
    training_seed: int,
) -> None:
    if len(rows) != EVAL_EPISODES:
        raise ValueError(
            f"episode diagnostics must contain exactly {EVAL_EPISODES} rows, "
            f"got {len(rows)}"
        )

    _validate_identity(
        rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        label="episode diagnostics",
    )

    observed_indices: set[int] = set()
    for row_index, row in enumerate(rows):
        for field in EPISODE_REQUIRED_NUMERIC_FIELDS:
            _finite_number(row.get(field), field)

        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        if episode_index in observed_indices:
            raise ValueError(f"duplicate episode index: {episode_index}")
        observed_indices.add(episode_index)

        expected_seed_value = episode_seed(scale, training_seed, episode_index)
        observed_seed = _exact_integer(row.get("episode_seed"), "episode_seed")
        if observed_seed != expected_seed_value:
            raise ValueError(
                f"episode seed mismatch for episode index {episode_index}: "
                f"expected={expected_seed_value}, observed={observed_seed}"
            )

    expected_indices = set(range(EVAL_EPISODES))
    if observed_indices != expected_indices:
        missing = sorted(expected_indices - observed_indices)
        unexpected = sorted(observed_indices - expected_indices)
        raise ValueError(
            f"episode index inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _validate_infrastructure_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    scale: str,
    algorithm: str,
    training_seed: int,
    infrastructure_label: str,
    expected_rows_per_episode: int,
) -> None:
    expected_columns = (
        TRANSFORMER_DIAGNOSTIC_COLUMNS
        if infrastructure_label.startswith("transformer")
        else CHARGER_DIAGNOSTIC_COLUMNS
    )
    _require_exact_columns(fieldnames, expected_columns, infrastructure_label)
    expected_total = expected_rows_per_episode * EVAL_EPISODES
    if len(rows) != expected_total:
        raise ValueError(
            f"{infrastructure_label} must contain exactly {expected_total} rows, "
            f"got {len(rows)}"
        )

    _validate_identity(
        rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        label=infrastructure_label,
    )

    episode_counts = {episode_index: 0 for episode_index in range(EVAL_EPISODES)}
    for row_index, row in enumerate(rows):
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        observed_seed = _exact_integer(row.get("episode_seed"), "episode_seed")
        expected_seed_value = episode_seed(scale, training_seed, episode_index)
        if observed_seed != expected_seed_value:
            raise ValueError(
                f"{infrastructure_label} row {row_index} episode seed mismatch"
            )
        if episode_index not in episode_counts:
            raise ValueError(
                f"{infrastructure_label} row {row_index} invalid episode index: "
                f"{episode_index}"
            )
        episode_counts[episode_index] += 1

    invalid_counts = {
        episode_index: count
        for episode_index, count in episode_counts.items()
        if count != expected_rows_per_episode
    }
    if invalid_counts:
        raise ValueError(
            f"{infrastructure_label} per-episode row count mismatch: "
            f"{invalid_counts}"
        )


def _validate_status_csv(
    path: Path,
    *,
    label: str,
    required_status_fields: tuple[str, ...],
) -> int:
    fieldnames, rows = _read_csv(path)
    _require_columns(fieldnames, set(required_status_fields), label)
    if not rows:
        raise ValueError(f"{label} must contain at least one row")
    for row_index, row in enumerate(rows):
        for field in required_status_fields:
            if row.get(field) != "pass":
                raise ValueError(
                    f"{label} row {row_index} failed {field}: {row.get(field)!r}"
                )
    return len(rows)


def validate_seed_output_directory(
    path: Path,
    task_id: int,
    training_seed: int,
) -> dict[str, object]:
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"seed output directory does not exist: {root}")

    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])
    if training_seed not in TRAINING_SEEDS:
        raise ValueError(f"unsupported training seed: {training_seed!r}")

    diagnostics = root / "diagnostics"
    validation = root / "validation"
    logs = root / "logs"

    episode_path = diagnostics / "episode_diagnostics.csv"
    seed_summary_path = diagnostics / "seed_summary_diagnostics.csv"
    transformer_path = diagnostics / "transformer_diagnostics.csv"
    charger_path = diagnostics / "charger_diagnostics.csv"

    episode_fields, episode_rows = _read_csv(episode_path)
    _require_exact_columns(
        episode_fields,
        EPISODE_DIAGNOSTIC_COLUMNS,
        "episode diagnostics",
    )
    _validate_per_seed_episode_rows(
        episode_rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
    )

    seed_fields, seed_rows = _read_csv(seed_summary_path)
    _require_exact_columns(
        seed_fields,
        SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
        "seed summary diagnostics",
    )
    if len(seed_rows) != 1:
        raise ValueError(
            f"seed summary diagnostics must contain exactly one row, "
            f"got {len(seed_rows)}"
        )
    _validate_identity(
        seed_rows,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        label="seed summary diagnostics",
    )
    if _exact_integer(
        seed_rows[0].get("n_eval_episodes"),
        "n_eval_episodes",
    ) != EVAL_EPISODES:
        raise ValueError(
            f"seed summary n_eval_episodes must equal {EVAL_EPISODES}"
        )

    expected_chargers, expected_transformers = TOPOLOGY[scale]
    transformer_fields, transformer_rows = _read_csv(transformer_path)
    charger_fields, charger_rows = _read_csv(charger_path)

    _validate_infrastructure_rows(
        transformer_rows,
        transformer_fields,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        infrastructure_label="transformer diagnostics",
        expected_rows_per_episode=expected_transformers,
    )
    _validate_infrastructure_rows(
        charger_rows,
        charger_fields,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        infrastructure_label="charger diagnostics",
        expected_rows_per_episode=expected_chargers,
    )

    canonical_rows = _validate_status_csv(
        validation / "canonical_reconciliation.csv",
        label="canonical reconciliation",
        required_status_fields=("status",),
    )
    service_rows = _validate_status_csv(
        validation / "service_reconciliation.csv",
        label="service reconciliation",
        required_status_fields=SERVICE_STATUS_FIELDS,
    )

    mapping_path = validation / "mapping_validation.json"
    if not mapping_path.is_file():
        raise ValueError(f"missing mapping validation JSON: {mapping_path}")
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("mapping validation JSON is invalid") from exc
    if not isinstance(mapping, dict) or mapping.get("status") != "ok":
        raise ValueError("mapping validation status must be ok")
    if mapping.get("scale", scale) != scale:
        raise ValueError("mapping validation scale mismatch")
    if mapping.get("algorithm", algorithm) != algorithm:
        raise ValueError("mapping validation algorithm mismatch")
    if int(mapping.get("training_seed", training_seed)) != training_seed:
        raise ValueError("mapping validation training seed mismatch")
    if mapping.get("diagnostic_schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("mapping validation schema mismatch")
    expected_chargers, expected_transformers = TOPOLOGY[scale]
    expected_transformer_rows = expected_transformers * EVAL_EPISODES
    expected_charger_rows = expected_chargers * EVAL_EPISODES
    mapping_count_checks = {
        "episode_count": len(episode_rows),
        "transformer_row_count": len(transformer_rows),
        "charger_row_count": len(charger_rows),
        "expected_transformer_rows": expected_transformer_rows,
        "expected_charger_rows": expected_charger_rows,
    }
    for field, expected_value in mapping_count_checks.items():
        if field in mapping and int(mapping[field]) != expected_value:
            raise ValueError(f"mapping validation {field} mismatch")

    stderr_path = logs / "stderr.log"
    if not stderr_path.is_file():
        raise ValueError(f"missing stderr log: {stderr_path}")
    if stderr_path.read_bytes():
        raise ValueError("stderr log must be empty")

    return {
        "status": "ok",
        "task_id": task_id,
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": training_seed,
        "formal_task_id": formal_task_id(task_id, training_seed),
        "episode_count": len(episode_rows),
        "transformer_row_count": len(transformer_rows),
        "charger_row_count": len(charger_rows),
        "canonical_reconciliation_rows": canonical_rows,
        "service_reconciliation_rows": service_rows,
        "diagnostic_schema_version": SCHEMA_VERSION,
    }


def _read_csv_rows_for_cli(path: Path) -> list[dict[str, str]]:
    _, rows = _read_csv(path)
    return rows



# STAGE_D_TASK3_MINIMAL_PACKAGE_CONTRACT_IMPLEMENTATION
import hashlib
import re
import tarfile
import tempfile
from pathlib import PurePosixPath

FORMAL_JOB_ID: Final[str] = "58513929"
PACKAGE_MANIFEST_PATH: Final[str] = "checksums/package_file_checksums.sha256"
FORMAL_PACKAGE_MANIFEST_PATH: Final[str] = "runtime_metadata/package_file_checksums.sha256"
PACKAGE_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_HEX_SHA1 = re.compile(r"^[0-9a-f]{40}$")
SEED_PACKAGE_FILES: Final[tuple[str, ...]] = (
    "diagnostics/episode_diagnostics.csv",
    "diagnostics/seed_summary_diagnostics.csv",
    "diagnostics/transformer_diagnostics.csv",
    "diagnostics/charger_diagnostics.csv",
    "validation/canonical_reconciliation.csv",
    "validation/service_reconciliation.csv",
    "validation/mapping_validation.json",
    "logs/stderr.log",
)
TASK_PACKAGE_TOP_LEVEL_FILES: Final[tuple[str, ...]] = (
    "task_metadata/task.json",
    "summaries/checkpoint_inventory.csv",
    "summaries/episode_inventory.csv",
    "summaries/runtime_summary.csv",
    "validation/task_validation.json",
    "runtime_metadata/source_commit_sha.txt",
    "runtime_metadata/array_job_id.txt",
    "runtime_metadata/package_file_list.txt",
    "logs/stdout.log",
    "logs/stderr.log",
    PACKAGE_MANIFEST_PATH,
)


def _expected_task_package_files() -> set[str]:
    files = set(TASK_PACKAGE_TOP_LEVEL_FILES)
    for training_seed in TRAINING_SEEDS:
        files.update(
            f"seed{training_seed}/{relative_path}"
            for relative_path in SEED_PACKAGE_FILES
        )
    return files


def _prohibited_checkpoint_name(name: str) -> bool:
    basename = PurePosixPath(name).name.lower()
    return (
        basename.startswith("model.best")
        or basename.startswith("model.last")
        or "optimizer" in basename
        or PurePosixPath(basename).suffix in {".pt", ".pth", ".ckpt"}
    )


def _normalise_tar_member_name(name: str) -> str:
    raw_name = str(name)
    if not raw_name or "\\" in raw_name:
        raise ValueError(f"unsafe tar path: {raw_name!r}")
    path = PurePosixPath(raw_name)
    clean_parts = [part for part in path.parts if part not in {"", "."}]
    if path.is_absolute() or ".." in path.parts or not clean_parts:
        raise ValueError(f"unsafe tar path: {raw_name!r}")
    return PurePosixPath(*clean_parts).as_posix()


def _is_benign_root_dir(member: tarfile.TarInfo) -> bool:
    return member.isdir() and str(member.name) in {".", "./"}


def validate_safe_tar_members(
    members: list[tarfile.TarInfo],
    *,
    allow_checkpoint_artifacts: bool = False,
) -> None:
    seen: set[str] = set()
    for member in members:
        raw_name = member.name
        if _is_benign_root_dir(member):
            continue
        normalised = _normalise_tar_member_name(raw_name)
        if normalised in seen:
            raise ValueError(f"duplicate tar member: {normalised}")
        seen.add(normalised)
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise ValueError(f"tar link or device member is not allowed: {raw_name}")
        if not member.isfile() and not member.isdir():
            raise ValueError(f"unsupported tar member type: {raw_name}")
        if (
            member.isfile()
            and not allow_checkpoint_artifacts
            and _prohibited_checkpoint_name(normalised)
        ):
            raise ValueError(f"checkpoint artefact leakage: {normalised}")


def _extract_regular_members(archive, members, destination: Path) -> None:
    for member in members:
        if not member.isfile():
            continue
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"unable to read tar member: {member.name}")
        output = destination / _normalise_tar_member_name(member.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read())


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_named_manifest(
    root: Path,
    manifest_member: str,
    regular_names: set[str],
) -> None:
    manifest_path = root / manifest_member
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not PACKAGE_HEX_SHA256.fullmatch(parts[0]):
            raise ValueError(f"invalid checksum manifest line {line_number}")
        digest, name = parts
        name = _normalise_tar_member_name(name)
        if name == manifest_member:
            raise ValueError("checksum manifest must not cover itself")
        if name in entries:
            raise ValueError(f"duplicate checksum manifest entry: {name}")
        entries[name] = digest

    expected = regular_names - {manifest_member}
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        raise ValueError(
            "checksum manifest coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name, expected_digest in entries.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected_digest:
            raise ValueError(
                f"checksum mismatch for {name}: "
                f"expected={expected_digest}, observed={observed}"
            )


def _validate_package_manifest(root: Path, regular_names: set[str]) -> None:
    _validate_named_manifest(root, PACKAGE_MANIFEST_PATH, regular_names)


def _validate_task_metadata(root: Path, task_id: int) -> tuple[str, str]:
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])
    payload = _read_json_object(root / "task_metadata" / "task.json", "task metadata")
    expected_fields = {
        "task_id": task_id,
        "scale": scale,
        "algorithm": algorithm,
        "formal_job_id": FORMAL_JOB_ID,
        "training_seeds": list(TRAINING_SEEDS),
        "formal_task_ids": [formal_task_id(task_id, seed) for seed in TRAINING_SEEDS],
        "eval_episodes_per_seed": EVAL_EPISODES,
        "diagnostic_schema_version": SCHEMA_VERSION,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"task metadata mismatch for {field}: "
                f"expected={expected!r}, observed={payload.get(field)!r}"
            )
    return scale, algorithm


def _validate_checkpoint_inventory(root, task_id, scale, algorithm) -> None:
    fields, rows = _read_csv(root / "summaries" / "checkpoint_inventory.csv")
    required = {
        "task_id", "scale", "algorithm", "training_seed",
        "formal_task_id", "diagnostic_schema_version",
    }
    _require_columns(fields, required, "checkpoint inventory")
    if len(rows) != len(TRAINING_SEEDS):
        raise ValueError("checkpoint inventory must contain exactly five rows")
    observed = set()
    for row in rows:
        if (
            _exact_integer(row.get("task_id"), "task_id") != task_id
            or row.get("scale") != scale
            or row.get("algorithm") != algorithm
            or row.get("diagnostic_schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("checkpoint inventory identity mismatch")
        seed = _exact_integer(row.get("training_seed"), "training_seed")
        formal_id = _exact_integer(row.get("formal_task_id"), "formal_task_id")
        pair = (seed, formal_id)
        if pair in observed:
            raise ValueError(f"duplicate checkpoint inventory row: {pair}")
        observed.add(pair)
    expected = {(seed, formal_task_id(task_id, seed)) for seed in TRAINING_SEEDS}
    if observed != expected:
        raise ValueError(
            f"checkpoint inventory mismatch: expected={sorted(expected)}, "
            f"observed={sorted(observed)}"
        )


def _validate_task_episode_inventory(root, scale, algorithm) -> set[EpisodeKey]:
    fields, rows = _read_csv(root / "summaries" / "episode_inventory.csv")
    required = {
        "scale", "algorithm", "training_seed", "episode_index",
        "episode_seed", "diagnostic_schema_version",
    }
    _require_columns(fields, required, "task episode inventory")
    if len(rows) != len(TRAINING_SEEDS) * EVAL_EPISODES:
        raise ValueError("task episode inventory must contain exactly 150 rows")
    observed = set()
    for row in rows:
        seed = _exact_integer(row.get("training_seed"), "training_seed")
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        key = (scale, algorithm, seed, episode_index)
        if row.get("scale") != scale or row.get("algorithm") != algorithm:
            raise ValueError("task episode inventory identity mismatch")
        if row.get("diagnostic_schema_version") != SCHEMA_VERSION:
            raise ValueError("task episode inventory schema mismatch")
        if key in observed:
            raise ValueError(f"duplicate task episode inventory key: {key}")
        if _exact_integer(row.get("episode_seed"), "episode_seed") != episode_seed(
            scale, seed, episode_index
        ):
            raise ValueError(f"task episode inventory seed mismatch: {key}")
        observed.add(key)
    expected = {
        (scale, algorithm, seed, episode_index)
        for seed in TRAINING_SEEDS
        for episode_index in range(EVAL_EPISODES)
    }
    if observed != expected:
        raise ValueError("task episode inventory key mismatch")
    return observed


def _validate_runtime_summary(root, task_id, scale, algorithm) -> None:
    fields, rows = _read_csv(root / "summaries" / "runtime_summary.csv")
    required = {
        "task_id", "scale", "algorithm", "training_seed",
        "formal_task_id", "status", "episode_count",
    }
    _require_columns(fields, required, "runtime summary")
    if len(rows) != len(TRAINING_SEEDS):
        raise ValueError("runtime summary must contain exactly five rows")
    observed = set()
    for row in rows:
        seed = _exact_integer(row.get("training_seed"), "training_seed")
        if seed in observed:
            raise ValueError(f"duplicate runtime summary seed: {seed}")
        observed.add(seed)
        if (
            _exact_integer(row.get("task_id"), "task_id") != task_id
            or row.get("scale") != scale
            or row.get("algorithm") != algorithm
            or _exact_integer(row.get("formal_task_id"), "formal_task_id")
            != formal_task_id(task_id, seed)
            or row.get("status") != "ok"
            or _exact_integer(row.get("episode_count"), "episode_count") != EVAL_EPISODES
        ):
            raise ValueError(f"runtime summary mismatch for seed {seed}")
    if observed != set(TRAINING_SEEDS):
        raise ValueError("runtime summary seed inventory mismatch")


def _validate_task_validation(root, task_id) -> None:
    payload = _read_json_object(
        root / "validation" / "task_validation.json", "task validation"
    )
    expected = {
        "status": "ok",
        "task_id": task_id,
        "checkpoint_groups": len(TRAINING_SEEDS),
        "episode_count": len(TRAINING_SEEDS) * EVAL_EPISODES,
        "diagnostic_schema_version": SCHEMA_VERSION,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"task validation mismatch for {field}: "
                f"expected={value!r}, observed={payload.get(field)!r}"
            )


def _validate_task_runtime_metadata(
    root: Path,
    regular_names: set[str],
) -> tuple[str, str]:
    file_list_path = root / "runtime_metadata" / "package_file_list.txt"
    listed = sorted(
        _normalise_tar_member_name(line.strip())
        for line in file_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if listed != sorted(regular_names):
        raise ValueError("package_file_list.txt does not exactly match package members")

    source_commit = (
        root / "runtime_metadata" / "source_commit_sha.txt"
    ).read_text(encoding="utf-8").strip()
    if not PACKAGE_HEX_SHA1.fullmatch(source_commit):
        raise ValueError(f"source commit SHA must be 40 lowercase hex: {source_commit!r}")

    array_job_id = (
        root / "runtime_metadata" / "array_job_id.txt"
    ).read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9]+", array_job_id):
        raise ValueError(f"array job ID must be numeric: {array_job_id!r}")

    return source_commit, array_job_id


def validate_stage_d_task_package(package: Path, task_id: int) -> dict[str, object]:
    package_path = Path(package)
    if not package_path.is_file() or package_path.stat().st_size == 0:
        raise ValueError(f"missing or empty task package: {package_path}")
    stage_d_task(task_id)

    with tarfile.open(package_path, "r:gz") as archive:
        members = archive.getmembers()
        validate_safe_tar_members(members)
        regular_names = {
            _normalise_tar_member_name(member.name)
            for member in members
            if member.isfile()
        }
        expected_names = _expected_task_package_files()
        if regular_names != expected_names:
            missing = sorted(expected_names - regular_names)
            unexpected = sorted(regular_names - expected_names)
            raise ValueError(
                f"task package file set mismatch: missing={missing}, unexpected={unexpected}"
            )
        with tempfile.TemporaryDirectory(prefix="stage_d_task_package_") as temporary:
            root = Path(temporary)
            _extract_regular_members(archive, members, root)
            _validate_package_manifest(root, regular_names)
            scale, algorithm = _validate_task_metadata(root, task_id)
            _validate_checkpoint_inventory(root, task_id, scale, algorithm)
            inventory_keys = _validate_task_episode_inventory(root, scale, algorithm)
            _validate_runtime_summary(root, task_id, scale, algorithm)
            _validate_task_validation(root, task_id)
            source_commit, array_job_id = _validate_task_runtime_metadata(
                root,
                regular_names,
            )
            if (root / "logs" / "stderr.log").read_bytes():
                raise ValueError("task stderr log must be empty")

            actual_keys = set()
            for training_seed in TRAINING_SEEDS:
                seed_root = root / f"seed{training_seed}"
                validate_seed_output_directory(seed_root, task_id, training_seed)
                _, rows = _read_csv(seed_root / "diagnostics" / "episode_diagnostics.csv")
                for row in rows:
                    actual_keys.add((
                        scale,
                        algorithm,
                        training_seed,
                        _exact_integer(row.get("episode_index"), "episode_index"),
                    ))
            if actual_keys != inventory_keys:
                raise ValueError("aggregate episode inventory does not match seed outputs")

    return {
        "status": "ok",
        "task_id": task_id,
        "scale": scale,
        "algorithm": algorithm,
        "checkpoint_groups": len(TRAINING_SEEDS),
        "episode_count": len(TRAINING_SEEDS) * EVAL_EPISODES,
        "diagnostic_schema_version": SCHEMA_VERSION,
        "source_commit_sha": source_commit,
        "array_job_id": array_job_id,
    }


def config_member(task_id: int, training_seed: int) -> str:
    task = stage_d_task(task_id)
    return (
        f"config/{task['scale']}_{task['algorithm']}_seed{training_seed}_config.yaml"
    )


def canonical_member(task_id: int, training_seed: int) -> str:
    task = stage_d_task(task_id)
    return (
        f"eval/{task['scale']}_{task['algorithm']}_seed{training_seed}_eval30.csv"
    )


def formal_run_name(task_id: int, training_seed: int) -> str:
    task = stage_d_task(task_id)
    return (
        f"controlled_multiscale_formal_{task['scale']}_"
        f"{task['algorithm']}_seed{training_seed}"
    )


def formal_train_member(task_id: int, training_seed: int, basename: str) -> str:
    return f"train/{formal_run_name(task_id, training_seed)}/{basename}"


def formal_package_name(task_id: int, training_seed: int) -> str:
    task = stage_d_task(task_id)
    return (
        f"m3_controlled_multiscale_formal_{task['scale']}_{task['algorithm']}_"
        f"seed{training_seed}_job{FORMAL_JOB_ID}_"
        f"task{formal_task_id(task_id, training_seed)}.tar.gz"
    )


def required_formal_members(task_id: int, training_seed: int) -> set[str]:
    return {
        config_member(task_id, training_seed),
        canonical_member(task_id, training_seed),
        formal_train_member(task_id, training_seed, "model.best_actor"),
        formal_train_member(task_id, training_seed, "model.best_actor_optimizer"),
        formal_train_member(task_id, training_seed, "model.best_critic"),
        formal_train_member(task_id, training_seed, "model.best_critic_optimizer"),
        formal_train_member(task_id, training_seed, "kwargs.yaml"),
        "runtime_metadata/source_manifest.sha256",
        "runtime_metadata/source_manifest_files.txt",
        "runtime_metadata/task_runtime_metadata.env",
        FORMAL_PACKAGE_MANIFEST_PATH,
    }


def _read_yaml_mapping(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {label}: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return payload


def _parse_bool(value: object, field: str) -> bool:
    text = str(value).strip().lower()
    if text in {"false", "0", "no"}:
        return False
    if text in {"true", "1", "yes"}:
        return True
    raise ValueError(f"invalid boolean value for {field}: {value!r}")


def validate_formal_config(path: Path, task_id: int) -> dict[str, object]:
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    payload = _read_yaml_mapping(path, "formal config")
    simulation_length = _exact_integer(payload.get("simulation_length"), "simulation_length")
    if simulation_length != 112:
        raise ValueError(f"simulation_length must equal 112, got {simulation_length}")
    v2g_enabled = _parse_bool(payload.get("v2g_enabled"), "v2g_enabled")
    if v2g_enabled:
        raise ValueError("v2g_enabled must be false")
    charger_count = _exact_integer(
        payload.get("number_of_charging_stations"),
        "number_of_charging_stations",
    )
    transformer_count = _exact_integer(
        payload.get("number_of_transformers"),
        "number_of_transformers",
    )
    expected_chargers, expected_transformers = TOPOLOGY[scale]
    if charger_count != expected_chargers:
        raise ValueError(
            "number_of_charging_stations mismatch: "
            f"{charger_count} != {expected_chargers}"
        )
    if transformer_count != expected_transformers:
        raise ValueError(
            "number_of_transformers mismatch: "
            f"{transformer_count} != {expected_transformers}"
        )
    return {
        "simulation_length": simulation_length,
        "v2g_enabled": v2g_enabled,
        "number_of_charging_stations": charger_count,
        "number_of_transformers": transformer_count,
    }


def _formal_contains_model_last(regular_names: set[str]) -> bool:
    return any(PurePosixPath(name).name.startswith("model.last") for name in regular_names)


def _read_env_file(path: Path, label: str) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty {label}: {path}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{label} line {line_number} is not key=value")
        key, value = line.split("=", 1)
        if not key:
            raise ValueError(f"{label} line {line_number} has empty key")
        values[key] = value.strip()
    return values


def _require_env_value(
    values: dict[str, str],
    key: str,
    expected: object,
    label: str,
) -> None:
    observed = values.get(key)
    if str(observed) != str(expected):
        raise ValueError(
            f"{label} mismatch for {key}: {observed!r} != {str(expected)!r}"
        )


def validate_formal_task_runtime_metadata(
    root: Path,
    task_id: int,
    training_seed: int,
) -> dict[str, str]:
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])
    values = _read_env_file(
        root / "runtime_metadata" / "task_runtime_metadata.env",
        "formal task runtime metadata",
    )
    _require_env_value(values, "task_id", formal_task_id(task_id, training_seed), "formal task runtime metadata")
    _require_env_value(values, "slurm_array_job_id", FORMAL_JOB_ID, "formal task runtime metadata")
    _require_env_value(values, "scale", scale, "formal task runtime metadata")
    _require_env_value(values, "algorithm", algorithm, "formal task runtime metadata")
    _require_env_value(values, "seed", training_seed, "formal task runtime metadata")
    _require_env_value(values, "training_exit_status", "0", "formal task runtime metadata")
    _require_env_value(values, "evaluation_exit_status", "0", "formal task runtime metadata")

    train_command = values.get("train_command", "")
    eval_command = values.get("eval_command", "")
    required_train_tokens = (
        f"--algorithm {algorithm}",
        f"--seed {training_seed}",
        "--max_timesteps 50000",
    )
    required_eval_tokens = (
        f"--algorithm {algorithm}",
        f"--seed {training_seed}",
        "--eval_episodes 30",
        "model.best",
        "--max_episode_steps 112",
        "--deterministic true",
        "--eval_expl_noise 0.0",
        "--eval_seed_offset ",
    )
    for token in required_train_tokens:
        if token not in train_command:
            raise ValueError(f"formal train command missing token: {token}")
    for token in required_eval_tokens:
        if token not in eval_command:
            raise ValueError(f"formal eval command missing token: {token}")
    if re.search(r"--eval_seed_offset\s+[0-9]+", eval_command) is None:
        raise ValueError("formal eval command missing numeric eval seed offset")
    if "model.last" in eval_command:
        raise ValueError("formal eval command must use model.best, not model.last")
    return values


def validate_formal_source_manifest(root: Path) -> None:
    manifest_path = root / "runtime_metadata" / "source_manifest.sha256"
    file_list_path = root / "runtime_metadata" / "source_manifest_files.txt"
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or not PACKAGE_HEX_SHA256.fullmatch(parts[0]):
            raise ValueError(f"invalid formal source manifest line {line_number}")
        digest, name = parts
        name = _normalise_tar_member_name(name)
        if name in entries:
            raise ValueError(f"duplicate formal source manifest entry: {name}")
        entries[name] = digest
    if not entries:
        raise ValueError("formal source manifest must not be empty")
    listed = sorted(
        _normalise_tar_member_name(line.strip())
        for line in file_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if sorted(entries) != listed:
        raise ValueError("formal source manifest coverage mismatch")


def validate_seed_formal_package(
    package: Path,
    task_id: int,
    training_seed: int,
    *,
    extract_dir: Path | None = None,
) -> dict[str, object]:
    package_path = Path(package)
    if package_path.name != formal_package_name(task_id, training_seed):
        raise ValueError(
            "formal package name mismatch: "
            f"{package_path.name} != {formal_package_name(task_id, training_seed)}"
        )
    if not package_path.is_file() or package_path.stat().st_size == 0:
        raise ValueError(f"missing or empty formal package: {package_path}")

    with tarfile.open(package_path, "r:gz") as archive:
        members = archive.getmembers()
        validate_safe_tar_members(members, allow_checkpoint_artifacts=True)
        regular_names = {
            _normalise_tar_member_name(member.name)
            for member in members
            if member.isfile()
        }
        required = required_formal_members(task_id, training_seed)
        if not required <= regular_names:
            missing = sorted(required - regular_names)
            if _formal_contains_model_last(regular_names):
                raise ValueError(
                    "model.best checkpoint members are required; "
                    "model.last is forbidden"
                )
            raise ValueError(f"missing required formal package member(s): {missing}")

        temporary_context = (
            tempfile.TemporaryDirectory(prefix="stage_d_formal_package_")
            if extract_dir is None
            else contextlib.nullcontext(str(extract_dir))
        )
        with temporary_context as temporary:
            root = Path(temporary)
            if root.exists() and any(root.iterdir()):
                raise ValueError(f"formal extract directory is not empty: {root}")
            root.mkdir(parents=True, exist_ok=True)
            _extract_regular_members(archive, members, root)
            _validate_named_manifest(root, FORMAL_PACKAGE_MANIFEST_PATH, regular_names)
            validate_formal_source_manifest(root)
            validate_formal_task_runtime_metadata(root, task_id, training_seed)
            config_values = validate_formal_config(
                root / config_member(task_id, training_seed),
                task_id,
            )

    task = stage_d_task(task_id)
    return {
        "status": "ok",
        "task_id": task_id,
        "scale": task["scale"],
        "algorithm": task["algorithm"],
        "formal_job_id": FORMAL_JOB_ID,
        "training_seed": training_seed,
        "formal_task_id": formal_task_id(task_id, training_seed),
        "package_path": str(package_path),
        "extract_dir": str(extract_dir) if extract_dir is not None else "",
        "config_member": config_member(task_id, training_seed),
        "canonical_member": canonical_member(task_id, training_seed),
        "checkpoint_prefix_member": formal_train_member(
            task_id,
            training_seed,
            "model.best",
        ),
        **config_values,
    }


def resolve_seed_formal_package(
    *,
    task_id: int,
    training_seed: int,
    individual_package_root: Path,
    complete_bundle: Path,
    staging_dir: Path,
) -> dict[str, object]:
    expected_name = formal_package_name(task_id, training_seed)
    individual_path = Path(individual_package_root) / expected_name
    if individual_path.is_file():
        with tarfile.open(individual_path, "r:gz") as archive:
            validate_safe_tar_members(
                archive.getmembers(),
                allow_checkpoint_artifacts=True,
            )
        return {
            "source_mode": "individual_task_package",
            "package_path": str(individual_path),
            "expected_package_name": expected_name,
        }

    complete_bundle = Path(complete_bundle)
    if not complete_bundle.is_file():
        raise ValueError(
            "formal package not found and complete bundle missing: "
            f"{complete_bundle}"
        )

    matches: list[tarfile.TarInfo] = []
    with tarfile.open(complete_bundle, "r:gz") as archive:
        members = archive.getmembers()
        validate_safe_tar_members(members)
        for member in members:
            if not member.isfile():
                continue
            member_name = _normalise_tar_member_name(member.name)
            parts = PurePosixPath(member_name).parts
            if (
                PurePosixPath(member_name).name == expected_name
                and "task_packages" in parts[:-1]
            ):
                matches.append(member)
        if len(matches) != 1:
            raise ValueError(
                "expected exactly one nested formal package under task_packages/ "
                f"named {expected_name}, found {len(matches)}"
            )
        staging_dir = Path(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        output_path = staging_dir / expected_name
        payload = archive.extractfile(matches[0])
        if payload is None:
            raise ValueError(f"cannot read nested formal package: {matches[0].name}")
        output_path.write_bytes(payload.read())

    with tarfile.open(output_path, "r:gz") as nested:
        validate_safe_tar_members(
            nested.getmembers(),
            allow_checkpoint_artifacts=True,
        )
    return {
        "source_mode": "complete_bundle_nested_task_package",
        "package_path": str(output_path),
        "bundle_member": _normalise_tar_member_name(matches[0].name),
        "expected_package_name": expected_name,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_package_name(task_id: int, array_job_id: str) -> str:
    task = stage_d_task(task_id)
    return (
        "m3_full_infrastructure_diagnostic_eval30_"
        f"{task['scale']}_{task['algorithm']}_seeds0-4_"
        f"job{array_job_id}_task{task_id}.tar.gz"
    )


def slurm_log_name(array_job_id: str, task_id: int, suffix: str) -> str:
    return f"evgnn_full_infra_diag_eval30_{array_job_id}_{task_id}.{suffix}"


def complete_bundle_name(array_job_id: str) -> str:
    return f"full_infrastructure_diagnostic_eval30_complete_evidence_job{array_job_id}.tar.gz"


def write_sha256_sidecar(archive_path: Path, sidecar_path: Path) -> None:
    archive_path = Path(archive_path)
    sidecar_path = Path(sidecar_path)
    temp_path = sidecar_path.with_name(f".{sidecar_path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temp_path.write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
    )
    validate_sha256_sidecar(archive_path, temp_path)
    os.replace(temp_path, sidecar_path)
    validate_sha256_sidecar(archive_path, sidecar_path)


def validate_sha256_sidecar(archive_path: Path, sidecar_path: Path) -> None:
    archive_path = Path(archive_path)
    line = Path(sidecar_path).read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
    if match is None:
        raise ValueError(f"invalid SHA-256 sidecar format: {sidecar_path}")
    digest, basename = match.groups()
    if basename != archive_path.name:
        raise ValueError(
            f"SHA-256 sidecar basename mismatch: {basename} != {archive_path.name}"
        )
    observed = sha256_file(archive_path)
    if digest != observed:
        raise ValueError(f"SHA-256 sidecar digest mismatch for {archive_path}")


def _rows_by_episode_index(
    rows: list[dict[str, str]],
    *,
    label: str,
) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        grouped.setdefault(episode_index, []).append(row)
    missing = sorted(set(range(EVAL_EPISODES)) - set(grouped))
    if missing:
        raise ValueError(f"{label} missing episode index(es): {missing}")
    return grouped


def _sum_rows(rows: list[dict[str, str]], field: str) -> float:
    total = 0.0
    for row in rows:
        if field not in row:
            raise ValueError(f"infrastructure diagnostics missing service field: {field}")
        total += _finite_number(row.get(field), field)
    return total


def _pass_if_close(observed: float, expected: float, *, tolerance: float = 1e-6) -> str:
    return "pass" if abs(observed - expected) <= tolerance else "fail"


def service_reconciliation_rows(
    episode_rows: list[dict[str, str]],
    transformer_rows: list[dict[str, str]],
    charger_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    transformer_by_episode = _rows_by_episode_index(
        transformer_rows,
        label="transformer diagnostics",
    )
    charger_by_episode = _rows_by_episode_index(
        charger_rows,
        label="charger diagnostics",
    )
    rows: list[dict[str, object]] = []
    for episode in episode_rows:
        episode_index = _exact_integer(episode.get("episode_index"), "episode_index")
        transformer_episode_rows = transformer_by_episode[episode_index]
        charger_episode_rows = charger_by_episode[episode_index]
        episode_served = _finite_number(episode.get("total_ev_served"), "total_ev_served")
        episode_charged = _finite_number(
            episode.get("total_energy_charged"),
            "total_energy_charged",
        )
        episode_discharged = _finite_number(
            episode.get("total_energy_discharged"),
            "total_energy_discharged",
        )
        charger_served = _sum_rows(charger_episode_rows, "served_ev_count")
        transformer_served = _sum_rows(transformer_episode_rows, "served_ev_count")
        charger_charged = _sum_rows(charger_episode_rows, "energy_charged_kwh")
        transformer_charged = _sum_rows(transformer_episode_rows, "energy_charged_kwh")
        charger_discharged = _sum_rows(charger_episode_rows, "energy_discharged_kwh")
        transformer_discharged = _sum_rows(
            transformer_episode_rows,
            "energy_discharged_kwh",
        )
        charger_satisfaction_count = _sum_rows(
            charger_episode_rows,
            "user_satisfaction_observation_count",
        )
        transformer_satisfaction_count = _sum_rows(
            transformer_episode_rows,
            "user_satisfaction_observation_count",
        )
        charger_satisfaction_sum = _sum_rows(
            charger_episode_rows,
            "user_satisfaction_sum",
        )
        transformer_satisfaction_sum = _sum_rows(
            transformer_episode_rows,
            "user_satisfaction_sum",
        )
        served_status = (
            "pass"
            if (
                _pass_if_close(charger_served, episode_served) == "pass"
                and _pass_if_close(transformer_served, episode_served) == "pass"
            )
            else "fail"
        )
        charged_status = (
            "pass"
            if (
                _pass_if_close(charger_charged, episode_charged) == "pass"
                and _pass_if_close(transformer_charged, episode_charged) == "pass"
            )
            else "fail"
        )
        discharged_status = (
            "pass"
            if (
                _pass_if_close(charger_discharged, episode_discharged) == "pass"
                and _pass_if_close(transformer_discharged, episode_discharged) == "pass"
            )
            else "fail"
        )
        satisfaction_status = (
            "pass"
            if (
                _pass_if_close(charger_satisfaction_count, episode_served) == "pass"
                and _pass_if_close(transformer_satisfaction_count, episode_served) == "pass"
                and _pass_if_close(charger_satisfaction_sum, transformer_satisfaction_sum) == "pass"
            )
            else "fail"
        )
        rows.append(
            {
                "episode_index": episode_index,
                "served_count_reconciliation_status": served_status,
                "satisfaction_sum_reconciliation_status": satisfaction_status,
                "charged_energy_reconciliation_status": charged_status,
                "discharged_energy_reconciliation_status": discharged_status,
                "episode_total_ev_served": episode_served,
                "charger_served_ev_count_sum": charger_served,
                "transformer_served_ev_count_sum": transformer_served,
                "episode_total_energy_charged": episode_charged,
                "charger_energy_charged_kwh_sum": charger_charged,
                "transformer_energy_charged_kwh_sum": transformer_charged,
                "episode_total_energy_discharged": episode_discharged,
                "charger_energy_discharged_kwh_sum": charger_discharged,
                "transformer_energy_discharged_kwh_sum": transformer_discharged,
            }
        )
    failures = [
        row
        for row in rows
        if any(row[field] != "pass" for field in SERVICE_STATUS_FIELDS)
    ]
    if failures:
        raise ValueError("service reconciliation failed")
    return rows


def _episode_rows_by_index(
    rows: list[dict[str, str]],
    label: str,
) -> dict[int, dict[str, str]]:
    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        if row.get("row_type", "episode") != "episode":
            continue
        episode_index = _exact_integer(row.get("episode_index"), "episode_index")
        if episode_index in by_index:
            raise ValueError(f"{label} duplicate episode index: {episode_index}")
        by_index[episode_index] = row
    missing = sorted(set(range(EVAL_EPISODES)) - set(by_index))
    if missing:
        raise ValueError(f"{label} missing episode index(es): {missing}")
    return by_index


def load_seed_reconciliation_inputs(
    *,
    diagnostic_dir: Path,
    historical_canonical_csv: Path,
    validation_dir: Path,
    task_id: int,
    training_seed: int,
    require_historical: bool = True,
    formal_validation_json: Path | None = None,
    stage_d_source_commit_sha: str = "",
    config_path: Path | None = None,
    checkpoint_prefix: Path | None = None,
) -> SeedReconciliationInputs:
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])
    diagnostic_root = Path(diagnostic_dir)
    Path(validation_dir)
    _, diagnostic_rows = _read_csv(diagnostic_root / "episode_diagnostics.csv")
    _, transformer_rows = _read_csv(diagnostic_root / "transformer_diagnostics.csv")
    _, charger_rows = _read_csv(diagnostic_root / "charger_diagnostics.csv")
    _, same_pass_rows = _read_csv(diagnostic_root / SAME_PASS_CANONICAL_FILENAME)
    historical_rows: list[dict[str, str]] = []
    if require_historical:
        _, historical_rows = _read_csv(Path(historical_canonical_csv))
    formal_validation = (
        _read_json_object(Path(formal_validation_json), "formal package validation")
        if formal_validation_json is not None
        else {}
    )
    if require_historical and (
        formal_validation_json is None
        or config_path is None
        or checkpoint_prefix is None
    ):
        raise ValueError("formal provenance inputs are required")
    if config_path is not None and not Path(config_path).is_file():
        raise ValueError(f"missing staged config path: {config_path}")
    if checkpoint_prefix is not None:
        for basename in (
            "model.best_actor",
            "model.best_actor_optimizer",
            "model.best_critic",
            "model.best_critic_optimizer",
            "kwargs.yaml",
        ):
            candidate = Path(checkpoint_prefix).parent / basename
            if not candidate.is_file():
                raise ValueError(f"missing staged checkpoint member: {candidate}")
    return SeedReconciliationInputs(
        diagnostic_rows=diagnostic_rows,
        transformer_rows=transformer_rows,
        charger_rows=charger_rows,
        historical_canonical_rows=historical_rows,
        same_pass_canonical_rows=same_pass_rows,
        formal_validation=formal_validation,
        scale=scale,
        algorithm=algorithm,
        task_id=task_id,
        training_seed=training_seed,
        stage_d_source_commit_sha=stage_d_source_commit_sha,
        config_path=Path(config_path) if config_path is not None else None,
        checkpoint_prefix=Path(checkpoint_prefix) if checkpoint_prefix is not None else None,
    )


def _same_pass_float_row(
    episode_index: int,
    canonical_field: str,
    diagnostic_field: str,
    comparison_type: str,
    canonical_value: object,
    diagnostic_value: object,
) -> dict[str, object]:
    expected = _finite_number(canonical_value, canonical_field)
    observed = _finite_number(diagnostic_value, diagnostic_field)
    absolute_difference = abs(observed - expected)
    relative_difference = absolute_difference / max(abs(expected), 1e-12)
    status = "pass" if observed == expected else "fail"
    return {
        "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
        "episode_index": episode_index,
        "field": canonical_field,
        "comparison_type": comparison_type,
        "same_pass_canonical_value": expected,
        "diagnostic_value": observed,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "same_pass_count": "",
        "diagnostic_count": "",
        "total_action_decision_denominator": "",
        "status": status,
        "failure_category": "" if status == "pass" else "same_pass_metric_mismatch",
    }


def _count_from_fraction(value: float, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(
            "total_action_decision_denominator must be positive: "
            f"{denominator!r}"
        )
    return int(
        Decimal(str(value * denominator)).to_integral_value(
            rounding=ROUND_HALF_UP,
        )
    )


def build_same_pass_reconciliation_rows(
    inputs: SeedReconciliationInputs,
) -> list[dict[str, object]]:
    diagnostic_by_index = _episode_rows_by_index(
        inputs.diagnostic_rows,
        "diagnostic rows",
    )
    same_pass_by_index = _episode_rows_by_index(
        inputs.same_pass_canonical_rows,
        "same-pass canonical rows",
    )
    rows: list[dict[str, object]] = []
    for episode_index in range(EVAL_EPISODES):
        diagnostic = diagnostic_by_index[episode_index]
        same_pass = same_pass_by_index[episode_index]
        for canonical_field, diagnostic_field, comparison_type in SAME_PASS_FIELD_MAP:
            if comparison_type == "fraction_count_exact":
                expected = _finite_number(
                    same_pass.get(canonical_field),
                    canonical_field,
                )
                observed = _finite_number(
                    diagnostic.get(diagnostic_field),
                    diagnostic_field,
                )
                denominator = _exact_integer(
                    same_pass.get("total_action_decision_denominator"),
                    "total_action_decision_denominator",
                )
                same_pass_count = _exact_integer(
                    same_pass.get("same_pass_at_max_count"),
                    "same_pass_at_max_count",
                )
                diagnostic_count = _count_from_fraction(observed, denominator)
                absolute_difference = abs(observed - expected)
                relative_difference = absolute_difference / max(abs(expected), 1e-12)
                status = (
                    "pass"
                    if expected == observed and same_pass_count == diagnostic_count
                    else "fail"
                )
                rows.append({
                    "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
                    "episode_index": episode_index,
                    "field": canonical_field,
                    "comparison_type": comparison_type,
                    "same_pass_canonical_value": expected,
                    "diagnostic_value": observed,
                    "absolute_difference": absolute_difference,
                    "relative_difference": relative_difference,
                    "same_pass_count": same_pass_count,
                    "diagnostic_count": diagnostic_count,
                    "total_action_decision_denominator": denominator,
                    "status": status,
                    "failure_category": (
                        ""
                        if status == "pass"
                        else "same_pass_metric_mismatch"
                    ),
                })
            else:
                rows.append(
                    _same_pass_float_row(
                        episode_index,
                        canonical_field,
                        diagnostic_field,
                        comparison_type,
                        same_pass.get(canonical_field),
                        diagnostic.get(diagnostic_field),
                    )
                )
    return rows


def _canonical_exact_values(
    canonical: dict[str, str],
    diagnostic: dict[str, str],
    canonical_field: str,
    diagnostic_field: str,
    value_type: str,
) -> tuple[str, str, str]:
    canonical_value = canonical.get(canonical_field, "")
    diagnostic_value = diagnostic.get(diagnostic_field, "")
    if value_type == "integer":
        expected = _exact_integer(canonical_value, canonical_field)
        observed = _exact_integer(diagnostic_value, diagnostic_field)
        return str(expected), str(observed), "pass" if expected == observed else "fail"
    if value_type == "boolean":
        expected_bool = _parse_bool(canonical_value, canonical_field)
        observed_bool = _parse_bool(diagnostic_value, diagnostic_field)
        return (
            str(expected_bool),
            str(observed_bool),
            "pass" if expected_bool == observed_bool else "fail",
        )
    return (
        str(canonical_value),
        str(diagnostic_value),
        "pass" if str(canonical_value) == str(diagnostic_value) else "fail",
    )


def _canonical_float_row(
    episode_index: int,
    diagnostic_field: str,
    canonical_value: str,
    diagnostic_value: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, object]:
    expected = _finite_number(canonical_value, diagnostic_field)
    observed = _finite_number(diagnostic_value, diagnostic_field)
    absolute_difference = abs(observed - expected)
    denominator = max(abs(expected), 1e-12)
    relative_difference = absolute_difference / denominator
    status = (
        "pass"
        if (
            absolute_difference <= absolute_tolerance
            or relative_difference <= relative_tolerance
        )
        else "fail"
    )
    return {
        "episode_index": episode_index,
        "field": diagnostic_field,
        "comparison_type": "floating",
        "canonical_value": expected,
        "diagnostic_value": observed,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "status": status,
    }


def prepare_seed_validation_files(
    *,
    diagnostic_dir: Path,
    canonical_csv: Path,
    validation_dir: Path,
    task_id: int,
    training_seed: int,
) -> dict[str, object]:
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])
    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_root = Path(diagnostic_dir)
    _, diagnostic_rows = _read_csv(diagnostic_root / "episode_diagnostics.csv")
    transformer_fields, transformer_rows = _read_csv(
        diagnostic_root / "transformer_diagnostics.csv"
    )
    charger_fields, charger_rows = _read_csv(diagnostic_root / "charger_diagnostics.csv")
    canonical_fields, canonical_rows = _read_csv(Path(canonical_csv))
    canonical_by_index = {
        _exact_integer(row.get("episode_index"), "episode_index"): row
        for row in canonical_rows
        if row.get("row_type", "episode") == "episode"
    }
    if len(canonical_by_index) < EVAL_EPISODES:
        raise ValueError("canonical eval30 CSV does not contain 30 episode rows")

    reconciliation_rows: list[dict[str, object]] = []
    for diagnostic in diagnostic_rows:
        episode_index = _exact_integer(diagnostic.get("episode_index"), "episode_index")
        canonical = canonical_by_index.get(episode_index)
        if canonical is None:
            raise ValueError(f"missing canonical episode row: {episode_index}")
        for field, canonical_field, diagnostic_field, value_type in CANONICAL_EXACT_RECONCILIATION:
            expected, observed, status = _canonical_exact_values(
                canonical,
                diagnostic,
                canonical_field,
                diagnostic_field,
                value_type,
            )
            reconciliation_rows.append(
                {
                    "episode_index": episode_index,
                    "field": field,
                    "comparison_type": "exact",
                    "canonical_value": expected,
                    "diagnostic_value": observed,
                    "absolute_difference": "",
                    "relative_difference": "",
                    "absolute_tolerance": "0",
                    "relative_tolerance": "0",
                    "status": status,
                }
            )
        for diagnostic_field, canonical_field, absolute_tolerance, relative_tolerance in CANONICAL_FLOAT_RECONCILIATION:
            reconciliation_rows.append(
                _canonical_float_row(
                    episode_index,
                    diagnostic_field,
                    canonical.get(canonical_field, ""),
                    diagnostic.get(diagnostic_field, ""),
                    absolute_tolerance,
                    relative_tolerance,
                )
            )
    if any(row["status"] != "pass" for row in reconciliation_rows):
        raise ValueError("canonical reconciliation failed")
    write_csv_rows(
        validation_dir / "canonical_reconciliation.csv",
        [
            "episode_index",
            "field",
            "comparison_type",
            "canonical_value",
            "diagnostic_value",
            "absolute_difference",
            "relative_difference",
            "absolute_tolerance",
            "relative_tolerance",
            "status",
        ],
        reconciliation_rows,
    )

    expected_chargers, expected_transformers = TOPOLOGY[scale]
    _validate_infrastructure_rows(
        transformer_rows,
        transformer_fields,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        infrastructure_label="transformer diagnostics",
        expected_rows_per_episode=expected_transformers,
    )
    _validate_infrastructure_rows(
        charger_rows,
        charger_fields,
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        infrastructure_label="charger diagnostics",
        expected_rows_per_episode=expected_chargers,
    )
    service_rows = service_reconciliation_rows(
        diagnostic_rows,
        transformer_rows,
        charger_rows,
    )
    write_csv_rows(
        validation_dir / "service_reconciliation.csv",
        [
            "episode_index",
            "served_count_reconciliation_status",
            "satisfaction_sum_reconciliation_status",
            "charged_energy_reconciliation_status",
            "discharged_energy_reconciliation_status",
            "episode_total_ev_served",
            "charger_served_ev_count_sum",
            "transformer_served_ev_count_sum",
            "episode_total_energy_charged",
            "charger_energy_charged_kwh_sum",
            "transformer_energy_charged_kwh_sum",
            "episode_total_energy_discharged",
            "charger_energy_discharged_kwh_sum",
            "transformer_energy_discharged_kwh_sum",
        ],
        service_rows,
    )
    episode_keys = sorted(
        _exact_integer(row.get("episode_index"), "episode_index")
        for row in diagnostic_rows
    )
    (validation_dir / "mapping_validation.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "scale": scale,
                "algorithm": algorithm,
                "training_seed": training_seed,
                "episode_count": len(diagnostic_rows),
                "transformer_row_count": len(transformer_rows),
                "charger_row_count": len(charger_rows),
                "expected_transformer_rows": expected_transformers * EVAL_EPISODES,
                "expected_charger_rows": expected_chargers * EVAL_EPISODES,
                "episode_indices": episode_keys,
                "diagnostic_schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ok",
        "canonical_reconciliation_rows": len(reconciliation_rows),
        "service_reconciliation_rows": len(service_rows),
    }


def csv_text(rows: list[dict[str, object]], fieldnames: list[str] | tuple[str, ...]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_csv_rows(
    path: Path,
    fieldnames: list[str] | tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text(rows, fieldnames), encoding="utf-8")


def _copy_required_seed_files(task_root: Path, staging_root: Path) -> None:
    for seed in TRAINING_SEEDS:
        for relative in SEED_PACKAGE_FILES:
            source = task_root / f"seed{seed}" / relative
            destination = staging_root / f"seed{seed}" / relative
            if not source.is_file():
                raise ValueError(f"missing required seed package source: {source}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _write_task_package_metadata(
    staging_root: Path,
    *,
    task_id: int,
    array_job_id: str,
    source_commit_sha: str,
) -> None:
    task = stage_d_task(task_id)
    scale = str(task["scale"])
    algorithm = str(task["algorithm"])

    task_payload = {
        "task_id": task_id,
        "scale": scale,
        "algorithm": algorithm,
        "formal_job_id": FORMAL_JOB_ID,
        "training_seeds": list(TRAINING_SEEDS),
        "formal_task_ids": [formal_task_id(task_id, seed) for seed in TRAINING_SEEDS],
        "eval_episodes_per_seed": EVAL_EPISODES,
        "diagnostic_schema_version": SCHEMA_VERSION,
    }
    (staging_root / "task_metadata").mkdir(parents=True, exist_ok=True)
    (staging_root / "task_metadata" / "task.json").write_text(
        json.dumps(task_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checkpoint_rows = [
        {
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": seed,
            "formal_task_id": formal_task_id(task_id, seed),
            "diagnostic_schema_version": SCHEMA_VERSION,
        }
        for seed in TRAINING_SEEDS
    ]
    write_csv_rows(
        staging_root / "summaries" / "checkpoint_inventory.csv",
        [
            "task_id",
            "scale",
            "algorithm",
            "training_seed",
            "formal_task_id",
            "diagnostic_schema_version",
        ],
        checkpoint_rows,
    )

    episode_rows = []
    for seed in TRAINING_SEEDS:
        _, rows = _read_csv(
            staging_root / f"seed{seed}" / "diagnostics" / "episode_diagnostics.csv"
        )
        for row in rows:
            episode_index = _exact_integer(row.get("episode_index"), "episode_index")
            episode_rows.append(
                {
                    "scale": scale,
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed(scale, seed, episode_index),
                    "diagnostic_schema_version": SCHEMA_VERSION,
                }
            )
    write_csv_rows(
        staging_root / "summaries" / "episode_inventory.csv",
        [
            "scale",
            "algorithm",
            "training_seed",
            "episode_index",
            "episode_seed",
            "diagnostic_schema_version",
        ],
        episode_rows,
    )

    runtime_rows = [
        {
            "task_id": task_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": seed,
            "formal_task_id": formal_task_id(task_id, seed),
            "status": "ok",
            "episode_count": EVAL_EPISODES,
        }
        for seed in TRAINING_SEEDS
    ]
    write_csv_rows(
        staging_root / "summaries" / "runtime_summary.csv",
        [
            "task_id",
            "scale",
            "algorithm",
            "training_seed",
            "formal_task_id",
            "status",
            "episode_count",
        ],
        runtime_rows,
    )

    validation_payload = {
        "status": "ok",
        "task_id": task_id,
        "checkpoint_groups": len(TRAINING_SEEDS),
        "episode_count": len(TRAINING_SEEDS) * EVAL_EPISODES,
        "diagnostic_schema_version": SCHEMA_VERSION,
    }
    (staging_root / "validation").mkdir(parents=True, exist_ok=True)
    (staging_root / "validation" / "task_validation.json").write_text(
        json.dumps(validation_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime_metadata = staging_root / "runtime_metadata"
    runtime_metadata.mkdir(parents=True, exist_ok=True)
    (runtime_metadata / "source_commit_sha.txt").write_text(
        source_commit_sha + "\n",
        encoding="utf-8",
    )
    (runtime_metadata / "array_job_id.txt").write_text(
        str(array_job_id) + "\n",
        encoding="utf-8",
    )
    logs = staging_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_source = staging_root.parent / "logs" / "stdout.log"
    stderr_source = staging_root.parent / "logs" / "stderr.log"
    if stdout_source.is_file():
        shutil.copy2(stdout_source, logs / "stdout.log")
    else:
        (logs / "stdout.log").write_text("TASK_OK\n", encoding="utf-8")
    if stderr_source.is_file():
        shutil.copy2(stderr_source, logs / "stderr.log")
    else:
        (logs / "stderr.log").write_bytes(b"")


def write_package_manifest(staging_root: Path) -> None:
    file_list_path = staging_root / "runtime_metadata" / "package_file_list.txt"
    manifest_path = staging_root / PACKAGE_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload_files = sorted(
        path
        for path in staging_root.rglob("*")
        if path.is_file() and path not in {file_list_path, manifest_path}
    )
    file_names = [path.relative_to(staging_root).as_posix() for path in payload_files]
    file_names.extend(
        [
            file_list_path.relative_to(staging_root).as_posix(),
            manifest_path.relative_to(staging_root).as_posix(),
        ]
    )
    file_list_path.write_text("\n".join(file_names) + "\n", encoding="utf-8")
    manifest_files = sorted(
        path
        for path in staging_root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for path in manifest_files:
            manifest.write(
                f"{sha256_file(path)}  {path.relative_to(staging_root).as_posix()}\n"
            )


def create_tar_from_staging(staging_root: Path, package_path: Path) -> None:
    with tarfile.open(package_path, "w:gz") as archive:
        for path in sorted(staging_root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(staging_root).as_posix())


def publish_stage_d_task_package(
    *,
    task_root: Path,
    task_id: int,
    array_job_id: str,
    source_commit_sha: str,
    package_path: Path,
) -> dict[str, object]:
    package_path = Path(package_path)
    sidecar_path = package_path.with_name(package_path.name + ".sha256")
    if package_path.exists():
        raise ValueError(f"final task package already exists: {package_path}")
    if sidecar_path.exists():
        raise ValueError(f"final task package checksum already exists: {sidecar_path}")
    if not PACKAGE_HEX_SHA1.fullmatch(source_commit_sha):
        raise ValueError(f"source commit SHA must be 40 lowercase hex: {source_commit_sha}")

    staging_root = Path(task_root) / "package_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    _copy_required_seed_files(Path(task_root), staging_root)
    _write_task_package_metadata(
        staging_root,
        task_id=task_id,
        array_job_id=array_job_id,
        source_commit_sha=source_commit_sha,
    )
    write_package_manifest(staging_root)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    temp_package = package_path.with_name(
        f".{package_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        create_tar_from_staging(staging_root, temp_package)
        result = validate_stage_d_task_package(temp_package, task_id)
        os.replace(temp_package, package_path)
        write_sha256_sidecar(package_path, sidecar_path)
    finally:
        temp_package.unlink(missing_ok=True)
    return {
        "status": "ok",
        "task_id": task_id,
        "package_path": str(package_path),
        "checksum_path": str(sidecar_path),
        **result,
    }


SACCT_FIELDS: Final[tuple[str, ...]] = (
    "JobIDRaw",
    "JobID",
    "JobName",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocCPUS",
    "MaxRSS",
    "TotalCPU",
)
COMPLETE_RUNTIME_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "job_id_raw",
    "job_id",
    "state",
    "exit_code",
    "elapsed_raw",
    "alloc_cpus",
    "max_rss",
    "total_cpu",
    "maxrss_source",
    "totalcpu_source",
)
COMPLETE_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "summaries/task_inventory.csv",
    "summaries/checkpoint_inventory.csv",
    "summaries/episode_inventory.csv",
    "summaries/runtime_summary.csv",
    "summaries/canonical_reconciliation_summary.csv",
    "summaries/service_reconciliation_summary.csv",
    "summaries/source_provenance_summary.csv",
    "summaries/warning_inventory.csv",
    "summaries/failure_manifest.csv",
    "runtime_metadata/source_commit_sha.txt",
    "runtime_metadata/formal_job_id.txt",
    "runtime_metadata/diagnostic_array_job_id.txt",
    "runtime_metadata/reducer_job_id.txt",
    "runtime_metadata/sacct_raw.txt",
    "runtime_metadata/reducer_runtime_metadata.env",
    "runtime_metadata/complete_file_list.txt",
    "runtime_metadata/complete_file_checksums.sha256",
    "validation/complete_workflow_validation.json",
    "logs/reducer_stdout_snapshot.log",
    "logs/reducer_stderr_snapshot.log",
)


def validate_array_job_id(value: object) -> str:
    cleaned = str(value)
    if not re.fullmatch(r"[0-9]+", cleaned):
        raise ValueError(f"array job ID must contain digits only: {value!r}")
    return cleaned


def _accounting_raw_base(row: dict[str, str], step: str) -> str:
    job_id_raw = row["JobIDRaw"]
    if step == "parent":
        if re.fullmatch(r"[0-9]+", job_id_raw):
            return job_id_raw
        raise ValueError(f"parent JobIDRaw must contain digits only: {job_id_raw}")
    suffix = f".{step}"
    match = re.fullmatch(rf"([0-9]+){re.escape(suffix)}", job_id_raw)
    if match is None:
        raise ValueError(f"{step} JobIDRaw suffix must be {suffix}: {job_id_raw}")
    if row["JobName"] != step:
        raise ValueError(f"{step} JobName must be {step}: {row['JobName']}")
    return match.group(1)


def _validate_accounting_row_status(row: dict[str, str]) -> None:
    if row["State"] != "COMPLETED":
        raise ValueError(
            f"Slurm accounting task {row['JobID']} must be COMPLETED, got {row['State']}"
        )
    if row["ExitCode"] != "0:0":
        raise ValueError(
            f"Slurm accounting task {row['JobID']} ExitCode must be 0:0, "
            f"got {row['ExitCode']}"
        )


def _available_accounting(value: str) -> bool:
    return str(value or "").strip() not in {"", "Unknown", "N/A", "None"}


def parse_sacct_raw(raw_text: str, array_job_id: str) -> list[dict[str, str]]:
    array_job_id = validate_array_job_id(array_job_id)
    parents: dict[int, dict[str, str]] = {}
    batches: dict[int, dict[str, str]] = {}
    externs: dict[int, dict[str, str]] = {}
    parent_raw_ids: dict[str, int] = {}

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        cells = line.split("|")
        if len(cells) != len(SACCT_FIELDS):
            raise ValueError(
                f"sacct row {line_number} must contain exactly "
                f"{len(SACCT_FIELDS)} fields, got {len(cells)}"
            )
        if cells[0] == "JobIDRaw":
            raise ValueError("sacct header rows are not allowed with --noheader")
        row = dict(zip(SACCT_FIELDS, cells))
        if row["JobIDRaw"] == array_job_id and row["JobID"] == array_job_id:
            continue
        identity = re.fullmatch(
            rf"{re.escape(array_job_id)}_([0-9]+)(?:\.(batch|extern))?",
            row["JobID"],
        )
        if identity is None:
            raise ValueError(f"unexpected accounting JobID: {row['JobID']}")
        task_id = int(identity.group(1))
        if task_id not in STAGE_D_TASKS:
            raise ValueError(f"unexpected accounting task ID: {task_id}")
        step = identity.group(2) or "parent"
        row["_raw_base"] = _accounting_raw_base(row, step)
        if step == "parent":
            if task_id in parents:
                raise ValueError(f"duplicate parent accounting row for task {task_id}")
            previous = parent_raw_ids.get(row["_raw_base"])
            if previous is not None:
                raise ValueError(
                    "duplicate numeric parent JobIDRaw "
                    f"{row['_raw_base']} for tasks {previous} and {task_id}"
                )
            parent_raw_ids[row["_raw_base"]] = task_id
            parents[task_id] = row
        elif step == "batch":
            if task_id in batches:
                raise ValueError(f"duplicate batch accounting row for task {task_id}")
            batches[task_id] = row
        else:
            if task_id in externs:
                raise ValueError(f"duplicate extern accounting row for task {task_id}")
            externs[task_id] = row

    missing = sorted(set(STAGE_D_TASKS) - set(parents))
    if missing:
        raise ValueError(f"required accounting unavailable for task(s): {missing}")

    for records in (parents, batches, externs):
        for row in records.values():
            _validate_accounting_row_status(row)

    for task_id, parent in parents.items():
        for step, records in (("batch", batches), ("extern", externs)):
            row = records.get(task_id)
            if row is not None and row["_raw_base"] != parent["_raw_base"]:
                raise ValueError(
                    f"{step} raw ID base mismatch for Slurm task "
                    f"{array_job_id}_{task_id}: {row['_raw_base']} != {parent['_raw_base']}"
                )

    runtime_rows: list[dict[str, str]] = []
    for task_id in sorted(STAGE_D_TASKS):
        parent = parents[task_id]
        if not _available_accounting(parent["ElapsedRaw"]):
            raise ValueError(f"ElapsedRaw unavailable for Slurm task {parent['JobID']}")
        if not _available_accounting(parent["AllocCPUS"]):
            raise ValueError(f"AllocCPUS unavailable for Slurm task {parent['JobID']}")
        batch = batches.get(task_id)
        max_rss = parent["MaxRSS"]
        total_cpu = parent["TotalCPU"]
        maxrss_source = "parent"
        totalcpu_source = "parent"
        if (
            not _available_accounting(max_rss)
            or not _available_accounting(total_cpu)
        ) and batch is None:
            raise ValueError(
                f"required .batch accounting unavailable for Slurm task {parent['JobID']}"
            )
        if not _available_accounting(max_rss):
            max_rss = batch["MaxRSS"]
            maxrss_source = "batch"
        if not _available_accounting(total_cpu):
            total_cpu = batch["TotalCPU"]
            totalcpu_source = "batch"
        if not _available_accounting(max_rss):
            raise ValueError(f"MaxRSS unavailable for Slurm task {parent['JobID']}")
        if not _available_accounting(total_cpu):
            raise ValueError(f"TotalCPU unavailable for Slurm task {parent['JobID']}")
        runtime_rows.append(
            {
                "task_id": str(task_id),
                "job_id_raw": parent["JobIDRaw"],
                "job_id": parent["JobID"],
                "state": parent["State"],
                "exit_code": parent["ExitCode"],
                "elapsed_raw": parent["ElapsedRaw"],
                "alloc_cpus": parent["AllocCPUS"],
                "max_rss": max_rss,
                "total_cpu": total_cpu,
                "maxrss_source": maxrss_source,
                "totalcpu_source": totalcpu_source,
            }
        )
    return runtime_rows


def collect_sacct_raw(args) -> str:
    array_job_id = validate_array_job_id(args.array_job_id)
    if getattr(args, "sacct_raw_file", None):
        return Path(args.sacct_raw_file).read_text(encoding="utf-8")
    command = [
        str(getattr(args, "sacct_command", "sacct")),
        "-j",
        array_job_id,
        "--parsable2",
        "--noheader",
        "--format=" + ",".join(SACCT_FIELDS),
    ]
    attempts = max(1, int(getattr(args, "sacct_attempts", 6)))
    delay = float(getattr(args, "sacct_delay_seconds", 10.0))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode == 0:
            try:
                parse_sacct_raw(completed.stdout, array_job_id)
                return completed.stdout
            except ValueError as exc:
                last_error = exc
        else:
            last_error = ValueError(completed.stderr.strip())
        if attempt < attempts:
            time.sleep(delay)
    raise ValueError(f"Slurm accounting unavailable after {attempts} attempt(s): {last_error}")


def safe_extract_tar(package_path: Path, destination: Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package_path, "r:gz") as archive:
        members = archive.getmembers()
        validate_safe_tar_members(members)
        _extract_regular_members(archive, members, destination)
    return destination


def _validate_expected_basenames(
    observed: list[Path],
    expected: list[Path],
    label: str,
) -> None:
    observed_names = sorted(path.name for path in observed)
    expected_names = sorted(path.name for path in expected)
    if observed_names != expected_names:
        missing = sorted(set(expected_names) - set(observed_names))
        unexpected = sorted(set(observed_names) - set(expected_names))
        raise ValueError(
            f"{label} mismatch: missing={missing}, unexpected={unexpected}"
        )


def _read_task_package_details(
    package_path: Path,
    task_id: int,
    extract_root: Path,
) -> dict[str, object]:
    validation = validate_stage_d_task_package(package_path, task_id)
    task_extract = extract_root / f"task{task_id}"
    safe_extract_tar(package_path, task_extract)
    _, checkpoint_rows = _read_csv(task_extract / "summaries" / "checkpoint_inventory.csv")
    _, episode_rows = _read_csv(task_extract / "summaries" / "episode_inventory.csv")
    canonical_rows: list[dict[str, str]] = []
    service_rows: list[dict[str, str]] = []
    for seed in TRAINING_SEEDS:
        seed_root = task_extract / f"seed{seed}"
        _, canonical = _read_csv(seed_root / "validation" / "canonical_reconciliation.csv")
        _, service = _read_csv(seed_root / "validation" / "service_reconciliation.csv")
        for row in canonical:
            enriched = {
                "task_id": str(task_id),
                "scale": str(validation["scale"]),
                "algorithm": str(validation["algorithm"]),
                "training_seed": str(seed),
                "formal_task_id": str(formal_task_id(task_id, seed)),
            }
            enriched.update(row)
            canonical_rows.append(enriched)
        for row in service:
            enriched = {
                "task_id": str(task_id),
                "scale": str(validation["scale"]),
                "algorithm": str(validation["algorithm"]),
                "training_seed": str(seed),
                "formal_task_id": str(formal_task_id(task_id, seed)),
            }
            enriched.update(row)
            service_rows.append(enriched)
    return {
        "validation": validation,
        "checkpoint_rows": checkpoint_rows,
        "episode_rows": episode_rows,
        "canonical_rows": canonical_rows,
        "service_rows": service_rows,
    }


def write_complete_manifest(staging_root: Path) -> None:
    file_list_path = staging_root / "runtime_metadata" / "complete_file_list.txt"
    checksum_path = staging_root / "runtime_metadata" / "complete_file_checksums.sha256"
    payload_files = sorted(
        path
        for path in staging_root.rglob("*")
        if path.is_file() and path not in {file_list_path, checksum_path}
    )
    file_names = [path.relative_to(staging_root).as_posix() for path in payload_files]
    file_names.extend(
        [
            file_list_path.relative_to(staging_root).as_posix(),
            checksum_path.relative_to(staging_root).as_posix(),
        ]
    )
    file_list_path.write_text("\n".join(file_names) + "\n", encoding="utf-8")
    manifest_files = sorted(
        path for path in staging_root.rglob("*") if path.is_file() and path != checksum_path
    )
    with checksum_path.open("w", encoding="utf-8") as manifest:
        for path in manifest_files:
            manifest.write(
                f"{sha256_file(path)}  {path.relative_to(staging_root).as_posix()}\n"
            )


def create_complete_bundle(staging_root: Path, bundle_path: Path) -> None:
    create_tar_from_staging(staging_root, bundle_path)


def validate_complete_bundle_file(bundle_path: Path) -> dict[str, object]:
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file() or bundle_path.stat().st_size == 0:
        raise ValueError(f"missing or empty complete bundle: {bundle_path}")
    with tarfile.open(bundle_path, "r:gz") as archive:
        members = archive.getmembers()
        validate_safe_tar_members(members)
        regular_names = {
            _normalise_tar_member_name(member.name)
            for member in members
            if member.isfile()
        }
    leaks = sorted(name for name in regular_names if _prohibited_checkpoint_name(name))
    if leaks:
        raise ValueError(f"checkpoint artefact leakage: {leaks}")
    missing = sorted(set(COMPLETE_REQUIRED_FILES) - regular_names)
    if missing:
        raise ValueError(f"complete bundle missing required file(s): {missing}")
    task_packages = sorted(
        name for name in regular_names
        if name.startswith("task_packages/") and name.endswith(".tar.gz")
    )
    stdout_logs = sorted(name for name in regular_names if name.startswith("logs/") and name.endswith(".out"))
    stderr_logs = sorted(name for name in regular_names if name.startswith("logs/") and name.endswith(".err"))
    if len(task_packages) != 8:
        raise ValueError(f"complete bundle must contain 8 task packages, got {len(task_packages)}")
    if len(stdout_logs) != 8 or len(stderr_logs) != 8:
        raise ValueError("complete bundle must contain 8 stdout and 8 stderr task logs")

    with tempfile.TemporaryDirectory(prefix="stage_d_complete_bundle_") as temporary:
        extract_root = safe_extract_tar(bundle_path, Path(temporary))
        listed = sorted(
            _normalise_tar_member_name(line.strip())
            for line in (extract_root / "runtime_metadata" / "complete_file_list.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if listed != sorted(regular_names):
            raise ValueError("complete_file_list.txt does not exactly match bundle members")
        _validate_named_manifest(
            extract_root,
            "runtime_metadata/complete_file_checksums.sha256",
            regular_names,
        )
        for stderr_member in stderr_logs:
            if (extract_root / stderr_member).read_bytes():
                raise ValueError(f"complete bundle stderr log must be empty: {stderr_member}")
        reducer_stderr = extract_root / "logs" / "reducer_stderr_snapshot.log"
        if reducer_stderr.read_bytes():
            raise ValueError("reducer stderr snapshot must be empty")
        validation = _read_json_object(
            extract_root / "validation" / "complete_workflow_validation.json",
            "complete workflow validation",
        )
        expected = {
            "status": "ok",
            "task_package_count": 8,
            "checkpoint_group_count": 40,
            "episode_count": 1200,
            "formal_job_id": FORMAL_JOB_ID,
            "schema_version": SCHEMA_VERSION,
        }
        for field, value in expected.items():
            if validation.get(field) != value:
                raise ValueError(
                    f"complete workflow validation mismatch for {field}: "
                    f"{validation.get(field)!r} != {value!r}"
                )
        _, episode_rows = _read_csv(extract_root / "summaries" / "episode_inventory.csv")
        validate_episode_inventory(episode_rows)
        _, checkpoint_rows = _read_csv(extract_root / "summaries" / "checkpoint_inventory.csv")
        if len(checkpoint_rows) != 40:
            raise ValueError("checkpoint inventory must contain exactly 40 rows")
        _, task_rows = _read_csv(extract_root / "summaries" / "task_inventory.csv")
        if len(task_rows) != 8:
            raise ValueError("task inventory must contain exactly 8 rows")
        for task_id in sorted(STAGE_D_TASKS):
            expected_name = f"task_packages/{task_package_name(task_id, validation['array_job_id'])}"
            if expected_name not in regular_names:
                raise ValueError(f"missing expected task package member: {expected_name}")
            validate_stage_d_task_package(extract_root / expected_name, task_id)

    return {
        "status": "ok",
        "bundle": str(bundle_path),
        "task_package_count": 8,
        "checkpoint_group_count": 40,
        "episode_count": 1200,
        "formal_job_id": FORMAL_JOB_ID,
        "schema_version": SCHEMA_VERSION,
    }


def reduce_complete_workflow(args) -> dict[str, object]:
    array_job_id = validate_array_job_id(args.array_job_id)
    expected_source_commit = str(args.source_commit_sha).strip()
    if not PACKAGE_HEX_SHA1.fullmatch(expected_source_commit):
        raise ValueError(f"source commit SHA must be 40 lowercase hex: {expected_source_commit}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_path = output_root / complete_bundle_name(array_job_id)
    sidecar_path = bundle_path.with_name(bundle_path.name + ".sha256")
    if bundle_path.exists():
        raise ValueError(f"final complete evidence bundle already exists: {bundle_path}")
    if sidecar_path.exists():
        raise ValueError(f"final complete evidence bundle checksum already exists: {sidecar_path}")

    package_root = Path(args.task_package_root)
    log_root = Path(args.slurm_log_root)
    expected_packages = [
        package_root / task_package_name(task_id, array_job_id)
        for task_id in sorted(STAGE_D_TASKS)
    ]
    observed_packages = sorted(
        package_root.glob(f"m3_full_infrastructure_diagnostic_eval30_*_job{array_job_id}_task*.tar.gz")
    )
    _validate_expected_basenames(observed_packages, expected_packages, "task packages")
    for package in expected_packages:
        validate_sha256_sidecar(package, package.with_name(package.name + ".sha256"))

    expected_stdout_logs = [
        log_root / slurm_log_name(array_job_id, task_id, "out")
        for task_id in sorted(STAGE_D_TASKS)
    ]
    expected_stderr_logs = [
        log_root / slurm_log_name(array_job_id, task_id, "err")
        for task_id in sorted(STAGE_D_TASKS)
    ]
    _validate_expected_basenames(
        sorted(log_root.glob(f"evgnn_full_infra_diag_eval30_{array_job_id}_*.out")),
        expected_stdout_logs,
        "Slurm stdout logs",
    )
    _validate_expected_basenames(
        sorted(log_root.glob(f"evgnn_full_infra_diag_eval30_{array_job_id}_*.err")),
        expected_stderr_logs,
        "Slurm stderr logs",
    )
    for stderr_log in expected_stderr_logs:
        if stderr_log.read_bytes():
            raise ValueError(f"task stderr log must be empty: {stderr_log}")

    sacct_raw = collect_sacct_raw(args)
    runtime_rows = parse_sacct_raw(sacct_raw, array_job_id)

    work_root = Path(args.work_root) / f"job{array_job_id}"
    staging_root = work_root / "complete_bundle_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    for subdir in ["task_packages", "logs", "summaries", "runtime_metadata", "validation"]:
        (staging_root / subdir).mkdir(parents=True, exist_ok=True)

    task_inventory_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    canonical_rows: list[dict[str, object]] = []
    service_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="stage_d_reduce_tasks_") as temporary:
        for task_id in sorted(STAGE_D_TASKS):
            package_path = package_root / task_package_name(task_id, array_job_id)
            details = _read_task_package_details(package_path, task_id, Path(temporary))
            validation = details["validation"]
            if validation["source_commit_sha"] != expected_source_commit:
                raise ValueError(
                    f"source commit mismatch for task {task_id}: "
                    f"{validation['source_commit_sha']} != {expected_source_commit}"
                )
            if validation["array_job_id"] != array_job_id:
                raise ValueError(
                    f"array job ID mismatch for task {task_id}: "
                    f"{validation['array_job_id']} != {array_job_id}"
                )
            task_inventory_rows.append(
                {
                    "task_id": task_id,
                    "scale": validation["scale"],
                    "algorithm": validation["algorithm"],
                    "package_name": package_path.name,
                    "package_sha256": sha256_file(package_path),
                    "status": "ok",
                    "checkpoint_groups": validation["checkpoint_groups"],
                    "episode_count": validation["episode_count"],
                }
            )
            checkpoint_rows.extend(details["checkpoint_rows"])
            episode_rows.extend(details["episode_rows"])
            canonical_rows.extend(details["canonical_rows"])
            service_rows.extend(details["service_rows"])
            source_rows.append(
                {
                    "task_id": task_id,
                    "scale": validation["scale"],
                    "algorithm": validation["algorithm"],
                    "source_commit_sha": expected_source_commit,
                    "formal_job_id": FORMAL_JOB_ID,
                }
            )
            shutil.copy2(package_path, staging_root / "task_packages" / package_path.name)

    if len(checkpoint_rows) != 40:
        raise ValueError(f"expected 40 checkpoint groups, got {len(checkpoint_rows)}")
    if len(episode_rows) != 1200:
        raise ValueError(f"expected 1200 episodes, got {len(episode_rows)}")
    validate_episode_inventory(episode_rows)

    for log in [*expected_stdout_logs, *expected_stderr_logs]:
        shutil.copy2(log, staging_root / "logs" / log.name)

    write_csv_rows(
        staging_root / "summaries" / "task_inventory.csv",
        [
            "task_id",
            "scale",
            "algorithm",
            "package_name",
            "package_sha256",
            "status",
            "checkpoint_groups",
            "episode_count",
        ],
        task_inventory_rows,
    )
    write_csv_rows(
        staging_root / "summaries" / "checkpoint_inventory.csv",
        [
            "task_id",
            "scale",
            "algorithm",
            "training_seed",
            "formal_task_id",
            "diagnostic_schema_version",
        ],
        checkpoint_rows,
    )
    write_csv_rows(
        staging_root / "summaries" / "episode_inventory.csv",
        [
            "scale",
            "algorithm",
            "training_seed",
            "episode_index",
            "episode_seed",
            "diagnostic_schema_version",
        ],
        episode_rows,
    )
    write_csv_rows(
        staging_root / "summaries" / "runtime_summary.csv",
        COMPLETE_RUNTIME_SUMMARY_FIELDS,
        runtime_rows,
    )
    canonical_fields = sorted({field for row in canonical_rows for field in row})
    service_fields = sorted({field for row in service_rows for field in row})
    write_csv_rows(
        staging_root / "summaries" / "canonical_reconciliation_summary.csv",
        canonical_fields,
        canonical_rows,
    )
    write_csv_rows(
        staging_root / "summaries" / "service_reconciliation_summary.csv",
        service_fields,
        service_rows,
    )
    write_csv_rows(
        staging_root / "summaries" / "source_provenance_summary.csv",
        ["task_id", "scale", "algorithm", "source_commit_sha", "formal_job_id"],
        source_rows,
    )
    write_csv_rows(
        staging_root / "summaries" / "warning_inventory.csv",
        ["source", "line_number", "warning"],
        [],
    )
    write_csv_rows(
        staging_root / "summaries" / "failure_manifest.csv",
        ["failure_id", "severity", "description"],
        [],
    )

    runtime_metadata = staging_root / "runtime_metadata"
    (runtime_metadata / "source_commit_sha.txt").write_text(
        expected_source_commit + "\n",
        encoding="utf-8",
    )
    (runtime_metadata / "formal_job_id.txt").write_text(FORMAL_JOB_ID + "\n", encoding="utf-8")
    (runtime_metadata / "diagnostic_array_job_id.txt").write_text(array_job_id + "\n", encoding="utf-8")
    (runtime_metadata / "reducer_job_id.txt").write_text(str(args.reducer_job_id) + "\n", encoding="utf-8")
    (runtime_metadata / "sacct_raw.txt").write_text(sacct_raw, encoding="utf-8")
    (runtime_metadata / "reducer_runtime_metadata.env").write_text(
        "".join(
            [
                f"array_job_id={array_job_id}\n",
                f"reducer_job_id={args.reducer_job_id}\n",
                "task_package_count=8\n",
                "checkpoint_group_count=40\n",
                "episode_count=1200\n",
                f"complete_bundle_path={bundle_path}\n",
            ]
        ),
        encoding="utf-8",
    )
    reducer_stdout = Path(args.reducer_stdout_log)
    reducer_stderr = Path(args.reducer_stderr_log)
    (staging_root / "logs" / "reducer_stdout_snapshot.log").write_text(
        reducer_stdout.read_text(encoding="utf-8") if reducer_stdout.is_file() else "",
        encoding="utf-8",
    )
    (staging_root / "logs" / "reducer_stderr_snapshot.log").write_text(
        reducer_stderr.read_text(encoding="utf-8") if reducer_stderr.is_file() else "",
        encoding="utf-8",
    )
    validation_payload = {
        "status": "ok",
        "array_job_id": array_job_id,
        "task_package_count": 8,
        "checkpoint_group_count": 40,
        "episode_count": 1200,
        "formal_job_id": FORMAL_JOB_ID,
        "schema_version": SCHEMA_VERSION,
    }
    (staging_root / "validation" / "complete_workflow_validation.json").write_text(
        json.dumps(validation_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_complete_manifest(staging_root)
    temp_bundle = bundle_path.with_name(
        f".{bundle_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        create_complete_bundle(staging_root, temp_bundle)
        validation = validate_complete_bundle_file(temp_bundle)
        os.replace(temp_bundle, bundle_path)
        write_sha256_sidecar(bundle_path, sidecar_path)
    finally:
        temp_bundle.unlink(missing_ok=True)

    return {
        "status": "ok",
        "bundle_path": str(bundle_path),
        "checksum_path": str(sidecar_path),
        "task_package_count": 8,
        "checkpoint_group_count": 40,
        "episode_count": 1200,
        "formal_job_id": FORMAL_JOB_ID,
        "schema_version": SCHEMA_VERSION,
        **{f"validated_{key}": value for key, value in validation.items() if key != "status"},
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate contracts for the full per-infrastructure deterministic "
            "eval30 workflow."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mapping_parser = subparsers.add_parser(
        "task-mapping",
        help="Print one exact scale-algorithm and formal-checkpoint mapping.",
    )
    mapping_parser.add_argument("--task-id", required=True, type=int)

    inventory_parser = subparsers.add_parser(
        "episode-inventory",
        help="Validate the exact 1,200-episode matrix in a CSV.",
    )
    inventory_parser.add_argument("--csv", required=True, type=Path)

    seed_parser = subparsers.add_parser(
        "validate-seed-output",
        help="Validate one 30-episode seed output directory.",
    )
    seed_parser.add_argument("--directory", required=True, type=Path)
    seed_parser.add_argument("--task-id", required=True, type=int)
    seed_parser.add_argument("--training-seed", required=True, type=int)

    package_parser = subparsers.add_parser(
        "validate-task-package",
        help="Validate one five-seed, 150-episode Stage D task package.",
    )
    package_parser.add_argument("--package", required=True, type=Path)
    package_parser.add_argument("--task-id", required=True, type=int)

    formal_name_parser = subparsers.add_parser(
        "formal-package-name",
        help="Print the exact seed-aware formal package basename.",
    )
    formal_name_parser.add_argument("--task-id", required=True, type=int)
    formal_name_parser.add_argument("--training-seed", required=True, type=int)

    resolve_formal_parser = subparsers.add_parser(
        "resolve-formal-package",
        help="Resolve one exact seed-aware formal package.",
    )
    resolve_formal_parser.add_argument("--task-id", required=True, type=int)
    resolve_formal_parser.add_argument("--training-seed", required=True, type=int)
    resolve_formal_parser.add_argument("--individual-package-root", required=True, type=Path)
    resolve_formal_parser.add_argument("--complete-bundle", required=True, type=Path)
    resolve_formal_parser.add_argument("--staging-dir", required=True, type=Path)

    formal_parser = subparsers.add_parser(
        "validate-formal-package",
        help="Validate one exact seed-aware formal source package.",
    )
    formal_parser.add_argument("--task-id", required=True, type=int)
    formal_parser.add_argument("--training-seed", required=True, type=int)
    formal_parser.add_argument("--package", required=True, type=Path)
    formal_parser.add_argument("--extract-dir", required=True, type=Path)

    seed_validation_parser = subparsers.add_parser(
        "prepare-seed-validation",
        help="Create canonical/service/mapping validation files for one seed.",
    )
    seed_validation_parser.add_argument("--diagnostic-dir", required=True, type=Path)
    seed_validation_parser.add_argument("--canonical-csv", required=True, type=Path)
    seed_validation_parser.add_argument("--validation-dir", required=True, type=Path)
    seed_validation_parser.add_argument("--task-id", required=True, type=int)
    seed_validation_parser.add_argument("--training-seed", required=True, type=int)

    publish_parser = subparsers.add_parser(
        "publish-task-package",
        help="Validate and atomically publish one five-seed task package.",
    )
    publish_parser.add_argument("--task-root", required=True, type=Path)
    publish_parser.add_argument("--task-id", required=True, type=int)
    publish_parser.add_argument("--array-job-id", required=True)
    publish_parser.add_argument("--source-commit-sha", required=True)
    publish_parser.add_argument("--package", required=True, type=Path)

    complete_parser = subparsers.add_parser(
        "validate-complete-workflow",
        help="Reduce and validate the exact 8-task/40-checkpoint workflow.",
    )
    complete_parser.add_argument("--array-job-id", required=True)
    complete_parser.add_argument("--task-package-root", required=True, type=Path)
    complete_parser.add_argument("--slurm-log-root", required=True, type=Path)
    complete_parser.add_argument("--output-root", required=True, type=Path)
    complete_parser.add_argument("--work-root", required=True, type=Path)
    complete_parser.add_argument("--source-commit-sha", required=True)
    complete_parser.add_argument("--reducer-job-id", required=True)
    complete_parser.add_argument("--sacct-raw-file", type=Path)
    complete_parser.add_argument("--sacct-command", default="sacct")
    complete_parser.add_argument("--sacct-attempts", type=int, default=6)
    complete_parser.add_argument("--sacct-delay-seconds", type=float, default=10.0)
    complete_parser.add_argument("--reducer-stdout-log", required=True, type=Path)
    complete_parser.add_argument("--reducer-stderr-log", required=True, type=Path)

    bundle_parser = subparsers.add_parser(
        "validate-complete-bundle",
        help="Validate a completed full eval30 evidence bundle.",
    )
    bundle_parser.add_argument("--bundle", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "task-mapping":
        for line in task_mapping_lines(args.task_id):
            print(line)
        return 0

    if args.command == "episode-inventory":
        rows = _read_csv_rows_for_cli(args.csv)
        validate_episode_inventory(rows)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "episodes": len(rows),
                    "checkpoint_groups": 40,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "validate-seed-output":
        result = validate_seed_output_directory(
            args.directory,
            args.task_id,
            args.training_seed,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "validate-task-package":
        result = validate_stage_d_task_package(args.package, args.task_id)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "formal-package-name":
        print(formal_package_name(args.task_id, args.training_seed))
        return 0

    if args.command == "resolve-formal-package":
        result = resolve_seed_formal_package(
            task_id=args.task_id,
            training_seed=args.training_seed,
            individual_package_root=args.individual_package_root,
            complete_bundle=args.complete_bundle,
            staging_dir=args.staging_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "validate-formal-package":
        result = validate_seed_formal_package(
            args.package,
            args.task_id,
            args.training_seed,
            extract_dir=args.extract_dir,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "prepare-seed-validation":
        result = prepare_seed_validation_files(
            diagnostic_dir=args.diagnostic_dir,
            canonical_csv=args.canonical_csv,
            validation_dir=args.validation_dir,
            task_id=args.task_id,
            training_seed=args.training_seed,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "publish-task-package":
        result = publish_stage_d_task_package(
            task_root=args.task_root,
            task_id=args.task_id,
            array_job_id=validate_array_job_id(args.array_job_id),
            source_commit_sha=args.source_commit_sha,
            package_path=args.package,
        )
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "validate-complete-workflow":
        result = reduce_complete_workflow(args)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "validate-complete-bundle":
        result = validate_complete_bundle_file(args.bundle)
        print(json.dumps(result, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
