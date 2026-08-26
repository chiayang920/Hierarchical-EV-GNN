#!/usr/bin/env python3
"""Package validation for the transformer-constraint 40-cell workflow."""

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
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import yaml


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import transformer_constraint_40cell_workflow as workflow
from TD3.TD3_HierarchicalActionGNN_TransformerConstraint import (
    ACTION_DOMAIN_CONTRACT,
    ACTOR_OUTPUT_TRANSFORM_FORMULA,
    CHECKPOINT_METADATA_SCHEMA,
)


_IMPLEMENTATION_FAILURE_PATTERNS = (
    "traceback (most recent call last)",
    "modulenotfounderror",
    "importerror",
    "syntaxerror",
    "segmentation fault",
    "core dumped",
    "slurmstepd: error",
    "command not found",
)

TRAINING_JOB_GROUP_TASK_RANGES = {
    "small": range(0, 20),
    "500cp": range(20, 30),
    "1000cp": range(30, 40),
}


@dataclass(frozen=True)
class TrainingPackageStatus:
    status: str
    reason: str = ""


@dataclass(frozen=True)
class TrainingPackageRecord:
    package_path: Path
    cell: workflow.TransformerConstraintCell
    job_id: str
    smoke: bool
    summary: dict[str, object]
    training_curve_rows: list[dict[str, object]]
    members: dict[str, bytes]
    model_dir_relative: str
    config_member: str


@dataclass(frozen=True)
class StagedTrainingPackage:
    task_root: Path
    checkpoint_prefix: Path
    config_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class SmokeGateResult:
    status: str
    cell_count: int
    scientific_claim_generated: bool
    staged_tasks: tuple[StagedTrainingPackage, ...]


@dataclass(frozen=True)
class TrainingAggregateResult:
    summary_rows: list[dict[str, object]]
    training_curve_rows: list[dict[str, object]]
    staged_tasks: tuple[StagedTrainingPackage, ...]
    training_set_id: str


@dataclass(frozen=True)
class DiagnosticPackageRecord:
    package_path: Path
    cell: workflow.TransformerConstraintCell
    summary: dict[str, object]
    diagnostic_rows: list[dict[str, object]]
    same_pass_eval_rows: list[dict[str, object]]


@dataclass(frozen=True)
class DiagnosticAggregateResult:
    diagnostic_rows: list[dict[str, object]]
    same_pass_eval_rows: list[dict[str, object]]
    package_summaries: list[dict[str, object]]


def _validate_job_id(value: object, label: str = "job ID") -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"{label} must contain decimal digits only; got {value!r}")
    return text


def _validate_training_set_id(value: object) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"training set ID must be a safe identifier; got {value!r}")
    return text


def _training_set_id_from_group_jobs(groups: Mapping[str, object]) -> str:
    required = tuple(TRAINING_JOB_GROUP_TASK_RANGES)
    if set(groups) != set(required):
        raise ValueError("split training gate requires job IDs for: " + ", ".join(required))
    ordered_job_ids = [_validate_job_id(groups[name], f"{name} training job ID") for name in required]
    return "trainset_" + "_".join(ordered_job_ids)


def _resolve_training_set_id(
    training_job_id: object | None,
    training_set_id: object | None,
) -> tuple[str, str]:
    if training_set_id is not None and str(training_set_id).strip():
        safe_training_set_id = _validate_training_set_id(training_set_id)
        train_job = "" if training_job_id in {None, ""} else _validate_job_id(training_job_id, "training job ID")
        return safe_training_set_id, train_job
    train_job = _validate_job_id(training_job_id, "training job ID")
    return f"trainjob{train_job}", train_job


def _safe_member_name(name: str) -> str:
    if not name or name.startswith("/"):
        raise ValueError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path.as_posix()


def _read_safe_tar(path: Path) -> dict[str, bytes]:
    if not path.is_file():
        raise ValueError(f"missing package: {path}")
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                name = _safe_member_name(member.name)
                if name in members:
                    raise ValueError(f"duplicate archive member: {name}")
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise ValueError(f"unsupported archive member type: {name}")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member type: {name}")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"unable to read archive member: {name}")
                members[name] = extracted.read()
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"unreadable package {path}: {exc}") from exc
    if not members:
        raise ValueError(f"empty package: {path}")
    return members


def _parse_env(payload: bytes, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{label}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise ValueError(f"{label}:{line_number}: duplicate/empty key {key!r}")
        result[key.strip()] = value.strip()
    return result


def _parse_yaml(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid YAML in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return value


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _parse_csv(payload: bytes, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError(f"{label} has no CSV header")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{label} has no data rows")
    return rows


def _require_member(members: Mapping[str, bytes], name: str) -> bytes:
    try:
        return members[name]
    except KeyError as exc:
        raise ValueError(f"required package member missing: {name}") from exc


def _parse_exact_int(value: object, field: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"-?[0-9]+", text):
        raise ValueError(f"{field} must be an integer; got {value!r}")
    return int(text)


def _parse_finite_float(value: object, field: str) -> float:
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite; got {value!r}")
    return number


def _assert_equal(actual: object, expected: object, field: str) -> None:
    if str(actual).strip() != str(expected):
        raise ValueError(f"{field} identity mismatch: expected {expected!r}, got {actual!r}")


def _has_value(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def _assert_close(
    actual: object,
    expected: float,
    field: str,
    *,
    tolerance: float = 1e-9,
) -> float:
    number = _parse_finite_float(actual, field)
    if abs(number - expected) > tolerance:
        raise ValueError(f"{field} mismatch: expected {expected!r}, got {number!r}")
    return number


def _assert_optional_close(
    actual: object,
    expected: float | None,
    field: str,
    *,
    tolerance: float = 1e-9,
) -> float | None:
    if expected is None:
        if _has_value(actual):
            raise ValueError(f"{field} must be blank without activated transformer observations")
        return None
    return _assert_close(actual, expected, field, tolerance=tolerance)


def _assert_canonical_episode_seed(
    row: Mapping[str, object],
    cell: workflow.TransformerConstraintCell,
    episode_index: int,
    label: str,
) -> int:
    actual = _parse_exact_int(row.get("episode_seed", ""), f"{label} episode_seed")
    expected = workflow.canonical_episode_seed(cell, episode_index)
    if actual != expected:
        raise ValueError(
            f"{label} episode_seed mismatch for task {cell.task_id} episode "
            f"{episode_index}: expected {expected}, got {actual}"
        )
    return actual


def _assert_model_best_checkpoint(value: object, label: str) -> None:
    text = str(value).strip()
    if not text or PurePosixPath(text).name != "model.best":
        raise ValueError(f"{label} must reference canonical model.best checkpoint; got {value!r}")


def _has_implementation_failure(payload: bytes) -> bool:
    text = payload.decode("utf-8", errors="replace").lower()
    return any(pattern in text for pattern in _IMPLEMENTATION_FAILURE_PATTERNS)


def _expected_package_prefix(smoke: bool) -> str:
    return "m3_transformer_constraint_smoke" if smoke else "m3_transformer_constraint_40cell"


def _candidate_pattern(cell: workflow.TransformerConstraintCell, job_id: str, smoke: bool) -> str:
    return (
        f"{_expected_package_prefix(smoke)}_{cell.scale}_{cell.algorithm}_seed{cell.seed}_"
        f"job{job_id}_task{cell.task_id}*.tar.gz"
    )


def discover_training_packages(
    package_root: Path | str,
    job_id: object,
    *,
    cells: Sequence[workflow.TransformerConstraintCell] | None = None,
    smoke: bool = False,
) -> dict[int, Path]:
    root = Path(package_root)
    exact_job_id = _validate_job_id(job_id, "training job ID")
    expected_cells = tuple(cells if cells is not None else (workflow.smoke_matrix() if smoke else workflow.formal_matrix()))
    if not root.is_dir():
        raise ValueError(f"package root does not exist: {root}")
    found: dict[int, Path] = {}
    for cell in expected_cells:
        matches = sorted(path for path in root.glob(_candidate_pattern(cell, exact_job_id, smoke)) if path.is_file())
        if not matches:
            raise ValueError(
                f"missing package for exact job {exact_job_id}: task={cell.task_id} "
                f"scale={cell.scale} algorithm={cell.algorithm} seed={cell.seed}"
            )
        if len(matches) != 1:
            raise ValueError(
                f"ambiguous duplicate packages for exact job {exact_job_id}, task {cell.task_id}: "
                + ", ".join(path.name for path in matches)
            )
        found[cell.task_id] = matches[0]
    return found


def _validate_training_log(payload: bytes, *, expected_steps: int, expected_cadence: int, expected_count: int) -> tuple[list[dict[str, object]], int]:
    rows = _parse_csv(payload, "training_log.csv")
    evaluation_rows = [row for row in rows if str(row.get("type", "")).strip() == "evaluation"]
    if len(evaluation_rows) != expected_count:
        raise ValueError(f"expected {expected_count} scheduled evaluations; observed {len(evaluation_rows)}")
    expected_schedule = tuple(range(expected_cadence, expected_steps + 1, expected_cadence))
    observed_schedule = tuple(_parse_exact_int(row.get("timestep", ""), "evaluation timestep") for row in evaluation_rows)
    if observed_schedule != expected_schedule:
        raise ValueError(f"scheduled evaluation steps must be {expected_schedule}; observed={observed_schedule}")
    curve_rows: list[dict[str, object]] = []
    best_step = -1
    best_reward = -math.inf
    for row in evaluation_rows:
        timestep = _parse_exact_int(row["timestep"], "evaluation timestep")
        reward = _parse_finite_float(row.get("eval/mean_reward", ""), "eval/mean_reward")
        reward_std = _parse_finite_float(row.get("eval/std_reward", ""), "eval/std_reward")
        elapsed = _parse_finite_float(row.get("elapsed_seconds", ""), "elapsed_seconds")
        curve_rows.append(
            {
                "timestep": timestep,
                "eval_mean_reward": reward,
                "eval_std_reward": reward_std,
                "elapsed_seconds": elapsed,
            }
        )
        if reward > best_reward:
            best_reward = reward
            best_step = timestep
    return curve_rows, best_step


def _validate_checkpoint_metadata(
    members: Mapping[str, bytes],
    model_dir: str,
    cell: workflow.TransformerConstraintCell,
    *,
    expected_steps: int,
    expected_cadence: int,
    expected_episodes: int,
) -> None:
    metadata_member = f"{model_dir}/model.best.metadata.yaml"
    metadata = _parse_yaml(_require_member(members, metadata_member), metadata_member)
    expected = {
        "metadata_schema": CHECKPOINT_METADATA_SCHEMA,
        "algorithm": cell.algorithm,
        "action_domain_contract": ACTION_DOMAIN_CONTRACT,
        "actor_output_transform": workflow.ACTOR_OUTPUT_TRANSFORM,
        "actor_output_transform_formula": ACTOR_OUTPUT_TRANSFORM_FORMULA,
        "non_ev_action": 0.0,
        "discrete_actions": 1,
        "training_budget": expected_steps,
        "start_timesteps": workflow.SMOKE_START_TIMESTEPS if expected_steps == workflow.SMOKE_TRAINING_STEPS else workflow.START_TIMESTEPS,
        "eval_frequency": expected_cadence,
        "internal_eval_episodes": expected_episodes,
        "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
        "checkpoint_role": "best",
    }
    for field, expected_value in expected.items():
        actual = metadata.get(field)
        if actual != expected_value:
            raise ValueError(
                f"checkpoint metadata {field} identity mismatch: expected {expected_value!r}, got {actual!r}"
            )


def validate_training_package(
    package_path: Path | str,
    cell: workflow.TransformerConstraintCell,
    job_id: object,
    *,
    expected_source_identity: str | None = None,
    expected_source_bundle_identity: str | None = None,
    smoke: bool = False,
) -> TrainingPackageRecord:
    package = Path(package_path)
    exact_job_id = _validate_job_id(job_id, "training job ID")
    members = _read_safe_tar(package)
    metadata_member = "runtime_metadata/task_runtime_metadata.env"
    metadata = _parse_env(_require_member(members, metadata_member), metadata_member)
    expected_steps = workflow.SMOKE_TRAINING_STEPS if smoke else workflow.FORMAL_TRAINING_STEPS
    expected_cadence = workflow.SMOKE_EVALUATION_CADENCE if smoke else workflow.FORMAL_EVALUATION_CADENCE
    expected_episodes = workflow.SMOKE_TRAINING_EVALUATION_EPISODES if smoke else workflow.FORMAL_TRAINING_EVALUATION_EPISODES
    expected_count = max(1, expected_steps // workflow.SMOKE_EVALUATION_CADENCE) if smoke else workflow.FORMAL_SCHEDULED_EVALUATIONS
    expected_protocol = "transformer_constraint_40cell_smoke_v1" if smoke else "transformer_constraint_40cell_train_v1"
    identity_expectations = {
        "protocol_version": expected_protocol,
        "task_id": cell.task_id,
        "slurm_array_task_id": cell.task_id,
        "slurm_array_job_id": exact_job_id,
        "scale": cell.scale,
        "algorithm": cell.algorithm,
        "seed": cell.seed,
        "training_steps": expected_steps,
        "evaluation_cadence": expected_cadence,
        "evaluation_episodes": expected_episodes,
        "expected_scheduled_evaluations": expected_count,
        "actor_output_transform": workflow.ACTOR_OUTPUT_TRANSFORM,
        "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
        "fresh_run": "true",
        "training_exit_status": "0",
    }
    if smoke:
        identity_expectations["reload_eval_exit_status"] = "0"
    for field, expected in identity_expectations.items():
        _assert_equal(metadata.get(field, ""), expected, field)
    if expected_source_identity is not None:
        _assert_equal(metadata.get("source_identity", ""), expected_source_identity, "source_identity")
    if expected_source_bundle_identity is not None:
        _assert_equal(metadata.get("source_bundle_identity", ""), expected_source_bundle_identity, "source_bundle_identity")
    source_identity = str(metadata.get("source_identity", "")).strip()
    source_bundle_identity = str(metadata.get("source_bundle_identity", "")).strip()
    if not source_identity or not source_bundle_identity:
        raise ValueError("source and source-bundle identities must be non-empty")
    requested_cpu = str(metadata.get("requested_cpu", "")).strip()
    requested_memory = str(metadata.get("requested_memory", "")).strip()
    requested_walltime = str(metadata.get("requested_walltime", "")).strip()
    if not requested_cpu or not requested_memory or not requested_walltime:
        raise ValueError("requested CPU, memory, and walltime metadata must be recorded")
    model_dir = str(metadata.get("model_dir_relative", "")).strip().rstrip("/")
    config_member = str(metadata.get("config_copy_relative", "")).strip()
    if not model_dir or not config_member:
        raise ValueError("model/config package paths are missing from runtime metadata")
    _safe_member_name(model_dir)
    _safe_member_name(config_member)

    copied_config = _parse_yaml(_require_member(members, config_member), config_member)
    run_args_member = f"{model_dir}/run_args.yaml"
    run_args = _parse_yaml(_require_member(members, run_args_member), run_args_member)
    model_config = _parse_yaml(_require_member(members, f"{model_dir}/config.yaml"), f"{model_dir}/config.yaml")
    kwargs_member = f"{model_dir}/kwargs.yaml"
    actor_kwargs = _parse_yaml(_require_member(members, kwargs_member), kwargs_member)
    _assert_equal(actor_kwargs.get("discrete_actions", ""), 1, "kwargs.discrete_actions")
    _assert_equal(run_args.get("algorithm", ""), cell.algorithm, "run_args.algorithm")
    _assert_equal(run_args.get("config", ""), cell.config_path, "run_args.config")
    _assert_equal(run_args.get("seed", ""), cell.seed, "run_args.seed")
    _assert_equal(run_args.get("max_timesteps", ""), expected_steps, "run_args.max_timesteps")
    _assert_equal(run_args.get("start_timesteps", ""), workflow.SMOKE_START_TIMESTEPS if smoke else workflow.START_TIMESTEPS, "run_args.start_timesteps")
    _assert_equal(run_args.get("eval_freq", ""), expected_cadence, "run_args.eval_freq")
    _assert_equal(run_args.get("eval_episodes", ""), expected_episodes, "run_args.eval_episodes")
    _assert_equal(run_args.get("discrete_actions", ""), 1, "run_args.discrete_actions")
    if copied_config != model_config:
        raise ValueError("matching configuration identity failed: staged and model configs differ")
    _assert_equal(copied_config.get("number_of_charging_stations", ""), int(cell.scale.removesuffix("cp")), "config scale")

    curve_rows, best_step = _validate_training_log(
        _require_member(members, f"{model_dir}/training_log.csv"),
        expected_steps=expected_steps,
        expected_cadence=expected_cadence,
        expected_count=expected_count,
    )
    checkpoint_members = (
        f"{model_dir}/model.best_actor",
        f"{model_dir}/model.best_actor_optimizer",
        f"{model_dir}/model.best_critic",
        f"{model_dir}/model.best_critic_optimizer",
    )
    for name in checkpoint_members:
        if not _require_member(members, name):
            raise ValueError(f"empty model.best checkpoint member: {name}")
    _validate_checkpoint_metadata(
        members,
        model_dir,
        cell,
        expected_steps=expected_steps,
        expected_cadence=expected_cadence,
        expected_episodes=expected_episodes,
    )
    if _has_implementation_failure(members.get("stderr.log", b"")):
        raise ValueError("stderr contains an implementation failure")

    smoke_eval_episode_count = 0
    if smoke:
        smoke_members = sorted(name for name in members if name.startswith("smoke_eval/") and name.endswith(".csv"))
        if len(smoke_members) != 1:
            raise ValueError(f"smoke eval requires exactly one CSV; observed {len(smoke_members)}")
        smoke_rows = _parse_csv(members[smoke_members[0]], smoke_members[0])
        episode_rows = [row for row in smoke_rows if str(row.get("row_type", "")).strip() == "episode"]
        summary_rows = [row for row in smoke_rows if str(row.get("row_type", "")).strip() == "summary"]
        if len(episode_rows) != 3 or len(summary_rows) != 1:
            raise ValueError("smoke eval must contain exactly 3 episode rows and 1 summary row")
        observed_indexes = []
        for row in episode_rows:
            _assert_equal(row.get("algorithm", ""), cell.algorithm, "smoke eval algorithm")
            _assert_equal(row.get("seed", ""), cell.seed, "smoke eval seed")
            if "model.best" not in str(row.get("checkpoint", "")):
                raise ValueError("smoke eval did not reload model.best")
            observed_indexes.append(_parse_exact_int(row.get("episode_index", ""), "smoke eval episode_index"))
            for field in ("episode_reward", "tracking_error", "energy_tracking_error", "power_tracker_violation"):
                _parse_finite_float(row.get(field, ""), f"smoke eval {field}")
        if tuple(observed_indexes) != (0, 1, 2):
            raise ValueError(f"smoke eval episode indexes must be 0,1,2; observed={observed_indexes}")
        smoke_eval_episode_count = len(episode_rows)

    package_sha = hashlib.sha256(package.read_bytes()).hexdigest()
    checkpoint_identity_members = (
        *checkpoint_members,
        f"{model_dir}/kwargs.yaml",
        f"{model_dir}/model.best.metadata.yaml",
    )
    checkpoint_identity = hashlib.sha256(
        b"\0".join(members[name] for name in checkpoint_identity_members)
    ).hexdigest()
    config_identity = hashlib.sha256(_require_member(members, config_member)).hexdigest()
    scheduled_steps = ";".join(str(row["timestep"]) for row in curve_rows)
    summary: dict[str, object] = {
        "task_id": cell.task_id,
        "scale": cell.scale,
        "algorithm": cell.algorithm,
        "seed": cell.seed,
        "training_steps": expected_steps,
        "evaluation_cadence": expected_cadence,
        "evaluation_episodes": expected_episodes,
        "expected_scheduled_evaluations": expected_count,
        "scheduled_evaluation_steps": scheduled_steps,
        "actor_output_transform": workflow.ACTOR_OUTPUT_TRANSFORM,
        "checkpoint_selection_rule": workflow.CHECKPOINT_SELECTION_RULE,
        "model_best_present": True,
        "runtime_args_present": True,
        "package_complete": True,
        "fresh_run": True,
        "required_values_finite": True,
        "source_identity": source_identity,
        "source_bundle_identity": source_bundle_identity,
        "training_job_id": exact_job_id,
        "package_path": str(package.resolve()),
        "package_sha256": package_sha,
        "checkpoint_identity_sha256": checkpoint_identity,
        "config_identity_sha256": config_identity,
        "best_checkpoint_step": best_step,
        "best_checkpoint_fraction_of_budget": best_step / expected_steps,
        "runtime_elapsed_seconds": _parse_finite_float(metadata.get("task_elapsed_seconds", ""), "task_elapsed_seconds"),
        "requested_cpu": requested_cpu,
        "requested_memory": requested_memory,
        "requested_walltime": requested_walltime,
        "model_best_reload_verified": bool(smoke),
        "smoke_eval_episode_count": smoke_eval_episode_count,
    }
    enriched_curve_rows = [
        {
            "task_id": cell.task_id,
            "scale": cell.scale,
            "algorithm": cell.algorithm,
            "seed": cell.seed,
            "source_identity": source_identity,
            "training_job_id": exact_job_id,
            **row,
        }
        for row in curve_rows
    ]
    return TrainingPackageRecord(
        package_path=package,
        cell=cell,
        job_id=exact_job_id,
        smoke=smoke,
        summary=summary,
        training_curve_rows=enriched_curve_rows,
        members=members,
        model_dir_relative=model_dir,
        config_member=config_member,
    )


def classify_training_package(
    package_path: Path | str,
    cell: workflow.TransformerConstraintCell,
    job_id: object,
    *,
    smoke: bool = False,
) -> TrainingPackageStatus:
    path = Path(package_path)
    if not path.exists():
        return TrainingPackageStatus("MISSING", f"missing package: {path}")
    try:
        validate_training_package(path, cell, job_id, smoke=smoke)
    except ValueError as exc:
        reason = str(exc)
        lowered = reason.lower()
        if "missing" in lowered or "required package member" in lowered or "empty model.best" in lowered:
            return TrainingPackageStatus("INCOMPLETE", reason)
        if "identity mismatch" in lowered or "algorithm" in lowered or "source" in lowered:
            return TrainingPackageStatus("IDENTITY_MISMATCH", reason)
        return TrainingPackageStatus("INCOMPLETE", reason)
    return TrainingPackageStatus("VALID", "")


def stage_training_package(record: TrainingPackageRecord, stage_root: Path | str) -> StagedTrainingPackage:
    return stage_training_package_for_set(record, stage_root, f"trainjob{record.job_id}")


def stage_training_package_for_set(
    record: TrainingPackageRecord,
    stage_root: Path | str,
    training_set_id: str,
) -> StagedTrainingPackage:
    root = Path(stage_root)
    safe_training_set_id = _validate_training_set_id(training_set_id)
    final_root = root / safe_training_set_id / f"task{record.cell.task_id}"
    if final_root.exists():
        raise ValueError(f"stage target already exists; stale staging is rejected: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".task{record.cell.task_id}.", dir=final_root.parent))
    try:
        checkpoint_dir = temp_root / "checkpoint"
        checkpoint_dir.mkdir(parents=True)
        prefix = f"{record.model_dir_relative}/model.best"
        copied = 0
        for member_name, payload in record.members.items():
            if member_name == prefix or member_name.startswith(prefix + "_") or member_name.startswith(prefix + "."):
                (checkpoint_dir / Path(member_name).name).write_bytes(payload)
                copied += 1
        if copied < 5:
            raise ValueError("validated model.best checkpoint could not be staged completely")
        kwargs_member = f"{record.model_dir_relative}/kwargs.yaml"
        (checkpoint_dir / "kwargs.yaml").write_bytes(record.members[kwargs_member])
        config_dir = temp_root / "config"
        config_dir.mkdir()
        config_path = config_dir / Path(record.config_member).name
        config_path.write_bytes(record.members[record.config_member])
        metadata_path = temp_root / "staged_identity.json"
        metadata_path.write_text(json.dumps(record.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_root, final_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return StagedTrainingPackage(
        task_root=final_root,
        checkpoint_prefix=final_root / "checkpoint" / "model.best",
        config_path=final_root / "config" / config_path.name,
        metadata_path=final_root / "staged_identity.json",
    )


def validate_smoke_packages(
    package_root: Path | str,
    job_id: object,
    stage_root: Path | str,
    *,
    expected_source_identity: str | None = None,
    expected_source_bundle_identity: str | None = None,
) -> SmokeGateResult:
    exact_job_id = _validate_job_id(job_id, "smoke job ID")
    cells = workflow.smoke_matrix()
    discovered = discover_training_packages(package_root, exact_job_id, cells=cells, smoke=True)
    records = [
        validate_training_package(
            discovered[cell.task_id],
            cell,
            exact_job_id,
            expected_source_identity=expected_source_identity,
            expected_source_bundle_identity=expected_source_bundle_identity,
            smoke=True,
        )
        for cell in cells
    ]
    if len({str(record.summary["source_identity"]) for record in records}) != 1:
        raise ValueError("smoke packages do not share one source identity")
    if len({str(record.summary["source_bundle_identity"]) for record in records}) != 1:
        raise ValueError("smoke packages do not share one source-bundle identity")
    staged = tuple(stage_training_package(record, stage_root) for record in records)
    return SmokeGateResult(status="PASS", cell_count=len(records), scientific_claim_generated=False, staged_tasks=staged)


def aggregate_training_packages(
    package_root: Path | str,
    *,
    training_job_id: object | None = None,
    training_job_ids_by_group: Mapping[str, object] | None = None,
    stage_root: Path | str,
    expected_source_identity: str | None = None,
    expected_source_bundle_identity: str | None = None,
) -> TrainingAggregateResult:
    if training_job_id is not None:
        raise ValueError("formal transformer-constraint training gate requires split training job IDs")
    if training_job_ids_by_group is None:
        raise ValueError("formal transformer-constraint training gate requires split training job IDs")
    else:
        exact_group_jobs = {
            group: _validate_job_id(job_id, f"{group} training job ID")
            for group, job_id in training_job_ids_by_group.items()
        }
        training_set_id = _training_set_id_from_group_jobs(exact_group_jobs)
        records = []
        cell_by_task = {cell.task_id: cell for cell in workflow.formal_matrix()}
        for group, task_range in TRAINING_JOB_GROUP_TASK_RANGES.items():
            group_cells = [cell_by_task[task_id] for task_id in task_range]
            group_job_id = exact_group_jobs[group]
            discovered = discover_training_packages(package_root, group_job_id, cells=group_cells, smoke=False)
            records.extend(
                validate_training_package(
                    discovered[cell.task_id],
                    cell,
                    group_job_id,
                    expected_source_identity=expected_source_identity,
                    expected_source_bundle_identity=expected_source_bundle_identity,
                    smoke=False,
                )
                for cell in group_cells
            )
    records = sorted(records, key=lambda record: record.cell.task_id)
    if len({str(record.summary["source_identity"]) for record in records}) != 1:
        raise ValueError("training packages must share one exact source identity")
    if len({str(record.summary["source_bundle_identity"]) for record in records}) != 1:
        raise ValueError("training packages must share one exact source-bundle identity")
    summary_rows = [{**record.summary, "training_set_id": training_set_id} for record in records]
    workflow.validate_training_gate(summary_rows)
    stage_root_path = Path(stage_root)
    if (stage_root_path / training_set_id).exists():
        raise ValueError("training stage for exact job already exists; stale staging is rejected")
    staged: list[StagedTrainingPackage] = []
    try:
        for record in records:
            record.summary["training_set_id"] = training_set_id
            staged.append(stage_training_package_for_set(record, stage_root_path, training_set_id))
    except Exception:
        shutil.rmtree(stage_root_path / training_set_id, ignore_errors=True)
        raise
    return TrainingAggregateResult(
        summary_rows=summary_rows,
        training_curve_rows=[row for record in records for row in record.training_curve_rows],
        staged_tasks=tuple(staged),
        training_set_id=training_set_id,
    )


def _discover_eval_packages(
    package_root: Path | str,
    training_job_id: object | None,
    eval_job_id: object,
    *,
    training_set_id: object | None = None,
) -> dict[int, Path]:
    root = Path(package_root)
    trainset, train_job = _resolve_training_set_id(training_job_id, training_set_id)
    eval_job = _validate_job_id(eval_job_id, "eval job ID")
    if not root.is_dir():
        raise ValueError(f"eval package root does not exist: {root}")
    found: dict[int, Path] = {}
    for cell in workflow.formal_matrix():
        pattern = (
            f"m3_transformer_constraint_eval30_{cell.scale}_{cell.algorithm}_seed{cell.seed}_"
            f"{trainset}_job{eval_job}_task{cell.task_id}*.tar.gz"
        )
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if len(matches) != 1:
            reason = "missing" if not matches else "ambiguous duplicate"
            train_label = train_job or trainset
            raise ValueError(f"{reason} eval30 package for exact jobs train={train_label} eval={eval_job} task={cell.task_id}")
        found[cell.task_id] = matches[0]
    return found


def aggregate_eval30_packages(
    package_root: Path | str,
    *,
    training_job_id: object | None = None,
    training_set_id: object | None = None,
    eval_job_id: object,
    expected_source_identity: str | None = None,
    expected_source_bundle_identity: str | None = None,
) -> list[dict[str, object]]:
    trainset, train_job = _resolve_training_set_id(training_job_id, training_set_id)
    eval_job = _validate_job_id(eval_job_id, "eval job ID")
    discovered = _discover_eval_packages(package_root, train_job or None, eval_job, training_set_id=trainset)
    canonical_rows: list[dict[str, object]] = []
    source_identities: set[str] = set()
    bundle_identities: set[str] = set()
    for cell in workflow.formal_matrix():
        members = _read_safe_tar(discovered[cell.task_id])
        metadata_name = "runtime_metadata/task_runtime_metadata.env"
        metadata = _parse_env(_require_member(members, metadata_name), metadata_name)
        expectations = {
            "protocol_version": "transformer_constraint_40cell_eval30_v1",
            "eval_array_job_id": eval_job,
            "task_id": cell.task_id,
            "slurm_array_task_id": cell.task_id,
            "scale": cell.scale,
            "algorithm": cell.algorithm,
            "seed": cell.seed,
            "checkpoint_role": "model.best",
            "evaluation_episodes": workflow.FORMAL_MODEL_BEST_EVAL_EPISODES,
            "evaluation_exit_status": "0",
        }
        if train_job:
            expectations["training_array_job_id"] = train_job
        for field, expected in expectations.items():
            _assert_equal(metadata.get(field, ""), expected, field)
        if str(metadata.get("training_set_id", "")).strip():
            _assert_equal(metadata.get("training_set_id", ""), trainset, "training_set_id")
        source_identity = str(metadata.get("source_identity", "")).strip()
        source_bundle_identity = str(metadata.get("source_bundle_identity", "")).strip()
        training_package_identity = str(metadata.get("source_training_package_sha256", "")).strip()
        checkpoint_identity = str(metadata.get("checkpoint_identity_sha256", "")).strip()
        config_identity = str(metadata.get("config_identity_sha256", "")).strip()
        if expected_source_identity is not None:
            _assert_equal(source_identity, expected_source_identity, "source_identity")
        if expected_source_bundle_identity is not None:
            _assert_equal(source_bundle_identity, expected_source_bundle_identity, "source_bundle_identity")
        if not source_identity or not source_bundle_identity or not training_package_identity or not checkpoint_identity or not config_identity:
            raise ValueError(f"task {cell.task_id}: eval package provenance is incomplete")
        source_identities.add(source_identity)
        bundle_identities.add(source_bundle_identity)
        if _has_implementation_failure(members.get("stderr.log", b"")):
            raise ValueError(f"task {cell.task_id}: eval stderr contains an implementation failure")
        candidates = [name for name in members if name.endswith("_eval30.csv")]
        if len(candidates) != 1:
            raise ValueError(f"task {cell.task_id}: expected exactly one eval30 CSV")
        rows = _parse_csv(members[candidates[0]], candidates[0])
        episode_rows = [row for row in rows if str(row.get("row_type", "")).strip() == "episode"]
        if len(episode_rows) != workflow.FORMAL_MODEL_BEST_EVAL_EPISODES:
            raise ValueError(f"task {cell.task_id}: expected 30 eval30 episode rows")
        indexes = [_parse_exact_int(row.get("episode_index", ""), "episode_index") for row in episode_rows]
        if indexes != list(range(workflow.FORMAL_MODEL_BEST_EVAL_EPISODES)):
            raise ValueError(f"task {cell.task_id}: eval30 episode indexes must be 0..29")
        for row, episode_index in zip(episode_rows, indexes):
            _assert_equal(row.get("algorithm", ""), cell.algorithm, "eval algorithm")
            _assert_equal(row.get("seed", ""), cell.seed, "eval seed")
            episode_seed = _assert_canonical_episode_seed(row, cell, episode_index, "eval30")
            _assert_model_best_checkpoint(row.get("checkpoint", ""), "eval30 checkpoint")
            canonical_rows.append(
                {
                    "task_id": cell.task_id,
                    "scale": cell.scale,
                    "algorithm": cell.algorithm,
                    "model_name": workflow.MODEL_NAMES[cell.algorithm],
                    "seed": cell.seed,
                    "episode_index": episode_index,
                    "episode_seed": episode_seed,
                    "checkpoint_role": "model.best",
                    "checkpoint_identity_sha256": checkpoint_identity,
                    "config_identity_sha256": config_identity,
                    "source_identity": source_identity,
                    "source_bundle_identity": source_bundle_identity,
                    "source_training_package_identity": training_package_identity,
                    "training_job_id": train_job or trainset,
                    "training_set_id": trainset,
                    "eval_job_id": eval_job,
                    "episode_reward": _parse_finite_float(row.get("episode_reward", ""), "episode_reward"),
                    "tracking_error": _parse_finite_float(row.get("tracking_error", ""), "tracking_error"),
                    "energy_tracking_error": _parse_finite_float(row.get("energy_tracking_error", ""), "energy_tracking_error"),
                    "power_tracker_violation": _parse_finite_float(row.get("power_tracker_violation", ""), "power_tracker_violation"),
                    "average_satisfaction": _parse_finite_float(row.get("average_user_satisfaction", ""), "average_user_satisfaction"),
                    "energy_delivered": _parse_finite_float(row.get("total_energy_charged", ""), "total_energy_charged"),
                    "evs_served": _parse_finite_float(row.get("total_ev_served", ""), "total_ev_served"),
                    "transformer_overload": _parse_finite_float(row.get("total_transformer_overload", ""), "total_transformer_overload"),
                    "required_values_finite": True,
                }
            )
    if len(source_identities) != 1:
        raise ValueError("eval30 packages must share one exact source identity")
    if len(bundle_identities) != 1 or "" in bundle_identities:
        raise ValueError("eval30 packages must share one exact source-bundle identity")
    workflow.validate_eval30_gate(canonical_rows)
    return canonical_rows


def _discover_diagnostic_packages(
    package_root: Path | str,
    training_job_id: object | None,
    diagnostic_job_id: object,
    *,
    training_set_id: object | None = None,
) -> dict[int, Path]:
    root = Path(package_root)
    trainset, train_job = _resolve_training_set_id(training_job_id, training_set_id)
    diagnostic_job = _validate_job_id(diagnostic_job_id, "diagnostic job ID")
    if not root.is_dir():
        raise ValueError(f"diagnostic package root does not exist: {root}")
    found: dict[int, Path] = {}
    for cell in workflow.formal_matrix():
        pattern = (
            f"m3_transformer_constraint_diagnostics_{cell.scale}_{cell.algorithm}_seed{cell.seed}_"
            f"{trainset}_job{diagnostic_job}_task{cell.task_id}*.tar.gz"
        )
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if len(matches) != 1:
            reason = "missing" if not matches else "ambiguous duplicate"
            train_label = train_job or trainset
            raise ValueError(
                f"{reason} diagnostic package for exact jobs train={train_label} "
                f"diagnostic={diagnostic_job} task={cell.task_id}"
            )
        found[cell.task_id] = matches[0]
    return found


def _validate_episode_rows(rows: Sequence[dict[str, str]], *, label: str, index_field: str = "episode_index") -> list[tuple[int, dict[str, str]]]:
    if len(rows) != workflow.FORMAL_MODEL_BEST_EVAL_EPISODES:
        raise ValueError(f"{label}: expected exactly 30 episode rows; observed {len(rows)}")
    indexed = [(_parse_exact_int(row.get(index_field, ""), index_field), row) for row in rows]
    if [index for index, _ in indexed] != list(range(workflow.FORMAL_MODEL_BEST_EVAL_EPISODES)):
        raise ValueError(f"{label}: episode indexes must be exactly 0..29")
    return indexed


def _sum_episode_int(indexed_rows: Sequence[tuple[int, dict[str, str]]], field: str) -> int:
    return sum(
        _parse_exact_int(row.get(field, ""), f"projection episode summary {field}")
        for _episode_index, row in indexed_rows
    )


def _sum_episode_float(indexed_rows: Sequence[tuple[int, dict[str, str]]], field: str) -> float:
    return sum(
        _parse_finite_float(row.get(field, ""), f"projection episode summary {field}")
        for _episode_index, row in indexed_rows
    )


def _validate_projection_seed_summary(
    seed_row: Mapping[str, object],
    cell: workflow.TransformerConstraintCell,
    indexed_projection: Sequence[tuple[int, dict[str, str]]],
) -> dict[str, object]:
    _assert_equal(seed_row.get("algorithm", ""), cell.algorithm, "projection seed summary algorithm")
    _assert_equal(seed_row.get("scale", ""), cell.scale, "projection seed summary scale")
    _assert_equal(seed_row.get("training_seed", ""), cell.seed, "projection seed summary training_seed")
    _assert_equal(
        seed_row.get("episodes", ""),
        len(indexed_projection),
        "projection seed summary episodes",
    )
    seed_violation_count = _parse_exact_int(
        seed_row.get("transformer_feasibility_violation_count", ""),
        "projection seed summary transformer_feasibility_violation_count",
    )
    seed_max_excess = _parse_finite_float(
        seed_row.get("transformer_feasibility_max_excess_kw", ""),
        "projection seed summary transformer_feasibility_max_excess_kw",
    )
    if seed_violation_count != 0 or seed_max_excess > 1e-9:
        raise ValueError(f"task {cell.task_id}: transformer feasibility invariant failed in projection seed summary")

    environment_steps = _sum_episode_int(indexed_projection, "environment_steps")
    activated_environment_steps = _sum_episode_int(indexed_projection, "activated_environment_steps")
    transformer_observations = _sum_episode_int(indexed_projection, "transformer_step_observations")
    activated_transformer_observations = _sum_episode_int(
        indexed_projection,
        "activated_transformer_step_observations",
    )
    corrected_ev_decisions = _sum_episode_int(indexed_projection, "corrected_ev_decisions")
    active_ev_decisions = _sum_episode_int(indexed_projection, "active_ev_decisions")
    violation_count = _sum_episode_int(
        indexed_projection,
        "transformer_feasibility_violation_count",
    )
    max_excess = max(
        _parse_finite_float(
            row.get("transformer_feasibility_max_excess_kw", ""),
            "projection episode summary transformer_feasibility_max_excess_kw",
        )
        for _episode_index, row in indexed_projection
    )
    active_alpha_sum = _sum_episode_float(indexed_projection, "active_alpha_sum")
    removed_sum = _sum_episode_float(
        indexed_projection,
        "commanded_power_removed_kw_step_sum",
    )
    energy_removed = _sum_episode_float(indexed_projection, "estimated_energy_removed_kwh")
    action_l1_sum = _sum_episode_float(indexed_projection, "action_correction_l1_sum")
    active_alpha_mins = [
        _parse_finite_float(row.get("active_alpha_min", ""), "projection episode summary active_alpha_min")
        for _episode_index, row in indexed_projection
        if _has_value(row.get("active_alpha_min"))
    ]

    expected_any_fraction = (
        activated_environment_steps / environment_steps if environment_steps else 0.0
    )
    expected_transformer_rate = (
        activated_transformer_observations / transformer_observations
        if transformer_observations
        else 0.0
    )
    expected_alpha_mean = (
        active_alpha_sum / activated_transformer_observations
        if activated_transformer_observations
        else None
    )
    expected_alpha_min = min(active_alpha_mins) if active_alpha_mins else None
    expected_removed_mean = (
        removed_sum / activated_transformer_observations
        if activated_transformer_observations
        else None
    )
    expected_action_l1_mean = action_l1_sum / active_ev_decisions if active_ev_decisions else 0.0
    expected_corrected_fraction = (
        corrected_ev_decisions / active_ev_decisions if active_ev_decisions else 0.0
    )

    expected_ints = {
        "environment_steps": environment_steps,
        "activated_environment_steps": activated_environment_steps,
        "transformer_step_observations": transformer_observations,
        "activated_transformer_step_observations": activated_transformer_observations,
        "corrected_ev_decisions": corrected_ev_decisions,
        "active_ev_decisions": active_ev_decisions,
        "transformer_feasibility_violation_count": violation_count,
    }
    for field, expected in expected_ints.items():
        _assert_equal(
            seed_row.get(field, ""),
            expected,
            f"projection seed summary {field}",
        )

    _assert_close(
        seed_row.get("any_constraint_activation_step_fraction", ""),
        expected_any_fraction,
        "projection seed summary any_constraint_activation_step_fraction",
    )
    _assert_close(
        seed_row.get("transformer_activation_rate", ""),
        expected_transformer_rate,
        "projection seed summary transformer_activation_rate",
    )
    _assert_optional_close(
        seed_row.get("active_alpha_mean", ""),
        expected_alpha_mean,
        "projection seed summary active_alpha_mean",
    )
    _assert_optional_close(
        seed_row.get("active_alpha_min", ""),
        expected_alpha_min,
        "projection seed summary active_alpha_min",
    )
    _assert_close(
        seed_row.get("active_alpha_sum", ""),
        active_alpha_sum,
        "projection seed summary active_alpha_sum",
    )
    _assert_close(
        seed_row.get("commanded_power_removed_kw_step_sum", ""),
        removed_sum,
        "projection seed summary commanded_power_removed_kw_step_sum",
    )
    _assert_optional_close(
        seed_row.get("commanded_power_removed_kw_mean_when_active", ""),
        expected_removed_mean,
        "projection seed summary commanded_power_removed_kw_mean_when_active",
    )
    _assert_close(
        seed_row.get("estimated_energy_removed_kwh", ""),
        energy_removed,
        "projection seed summary estimated_energy_removed_kwh",
    )
    _assert_close(
        seed_row.get("action_correction_l1_sum", ""),
        action_l1_sum,
        "projection seed summary action_correction_l1_sum",
    )
    _assert_close(
        seed_row.get("action_correction_l1_mean_per_active_ev_decision", ""),
        expected_action_l1_mean,
        "projection seed summary action_correction_l1_mean_per_active_ev_decision",
    )
    _assert_close(
        seed_row.get("corrected_ev_decision_fraction", ""),
        expected_corrected_fraction,
        "projection seed summary corrected_ev_decision_fraction",
    )
    seed_max_excess = _assert_close(
        seed_row.get("transformer_feasibility_max_excess_kw", ""),
        max_excess,
        "projection seed summary transformer_feasibility_max_excess_kw",
    )

    return {
        "transformer_feasibility_violation_count": seed_violation_count,
        "transformer_feasibility_max_excess_kw": seed_max_excess,
    }


def validate_diagnostic_package(
    package_path: Path | str,
    cell: workflow.TransformerConstraintCell,
    *,
    training_job_id: object | None = None,
    training_set_id: object | None = None,
    diagnostic_job_id: object,
    expected_source_identity: str | None = None,
    expected_source_bundle_identity: str | None = None,
) -> DiagnosticPackageRecord:
    package = Path(package_path)
    trainset, train_job = _resolve_training_set_id(training_job_id, training_set_id)
    diagnostic_job = _validate_job_id(diagnostic_job_id, "diagnostic job ID")
    members = _read_safe_tar(package)
    metadata_name = "runtime_metadata/task_runtime_metadata.env"
    metadata = _parse_env(_require_member(members, metadata_name), metadata_name)
    expectations = {
        "protocol_version": "transformer_constraint_40cell_diagnostic_v1",
        "diagnostic_array_job_id": diagnostic_job,
        "task_id": cell.task_id,
        "slurm_array_task_id": cell.task_id,
        "scale": cell.scale,
        "algorithm": cell.algorithm,
        "seed": cell.seed,
        "checkpoint_role": "model.best",
        "diagnostic_schema_version": "3",
        "reconciliation_contract_version": "2",
        "evaluation_episodes": workflow.FORMAL_MODEL_BEST_EVAL_EPISODES,
        "evaluation_exit_status": "0",
        "projection_diagnostic_exit_status": "0",
        "validation_exit_status": "0",
    }
    if train_job:
        expectations["training_array_job_id"] = train_job
    for field, expected in expectations.items():
        _assert_equal(metadata.get(field, ""), expected, field)
    if str(metadata.get("training_set_id", "")).strip():
        _assert_equal(metadata.get("training_set_id", ""), trainset, "training_set_id")
    source_identity = str(metadata.get("source_identity", "")).strip()
    source_bundle_identity = str(metadata.get("source_bundle_identity", "")).strip()
    training_package_identity = str(metadata.get("source_training_package_sha256", "")).strip()
    checkpoint_identity = str(metadata.get("checkpoint_identity_sha256", "")).strip()
    config_identity = str(metadata.get("config_identity_sha256", "")).strip()
    if expected_source_identity is not None:
        _assert_equal(source_identity, expected_source_identity, "source_identity")
    if expected_source_bundle_identity is not None:
        _assert_equal(source_bundle_identity, expected_source_bundle_identity, "source_bundle_identity")
    if not source_identity or not source_bundle_identity or not training_package_identity or not checkpoint_identity or not config_identity:
        raise ValueError(f"task {cell.task_id}: diagnostic package provenance is incomplete")
    if _has_implementation_failure(members.get("stderr.log", b"")):
        raise ValueError(f"task {cell.task_id}: diagnostic stderr contains an implementation failure")

    mapping = _parse_json(_require_member(members, "validation/mapping_validation.json"), "validation/mapping_validation.json")
    _assert_equal(mapping.get("status", ""), "ok", "mapping validation status")
    _assert_equal(mapping.get("scale", ""), cell.scale, "mapping scale")
    _assert_equal(mapping.get("algorithm", ""), cell.algorithm, "mapping algorithm")
    _assert_equal(mapping.get("training_seed", ""), cell.seed, "mapping training_seed")
    _assert_equal(mapping.get("episode_count", ""), 30, "mapping episode_count")
    checks = mapping.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ValueError(f"task {cell.task_id}: mapping validation checks did not all pass")

    reconciliation_rows = _parse_csv(
        _require_member(members, "validation/same_pass_canonical_reconciliation.csv"),
        "validation/same_pass_canonical_reconciliation.csv",
    )
    reconciliation_indexes: set[int] = set()
    for row in reconciliation_rows:
        _assert_equal(row.get("reconciliation_contract_version", ""), 2, "reconciliation contract")
        index = _parse_exact_int(row.get("episode_index", ""), "reconciliation episode_index")
        reconciliation_indexes.add(index)
        if str(row.get("status", "")).strip().lower() != "pass":
            raise ValueError(f"task {cell.task_id}: canonical reconciliation failed")
    if reconciliation_indexes != set(range(30)):
        raise ValueError(f"task {cell.task_id}: canonical reconciliation lacks full episode coverage")

    service_rows = _parse_csv(_require_member(members, "validation/service_reconciliation.csv"), "validation/service_reconciliation.csv")
    indexed_service = _validate_episode_rows(service_rows, label=f"task {cell.task_id} service reconciliation")
    service_by_episode: dict[int, tuple[str, str]] = {}
    for episode_index, row in indexed_service:
        service_statuses = (
            str(row.get("served_count_reconciliation_status", "")).strip().lower(),
            str(row.get("satisfaction_sum_reconciliation_status", "")).strip().lower(),
        )
        energy_statuses = (
            str(row.get("charged_energy_reconciliation_status", "")).strip().lower(),
            str(row.get("discharged_energy_reconciliation_status", "")).strip().lower(),
        )
        if any(status != "pass" for status in service_statuses):
            raise ValueError(f"task {cell.task_id}: service reconciliation failed at episode {episode_index}")
        if any(status != "pass" for status in energy_statuses):
            raise ValueError(f"task {cell.task_id}: energy reconciliation failed at episode {episode_index}")
        service_by_episode[episode_index] = ("pass", "pass")

    summary = _parse_json(
        _require_member(members, "runtime_metadata/reconciliation_summary.json"),
        "runtime_metadata/reconciliation_summary.json",
    )
    for field in ("hard_gate_status", "mapping_validation_status", "same_pass_metric_status", "service_reconciliation_status"):
        _assert_equal(summary.get(field, ""), "pass", field)
    for field in ("mapping_validation_failure_count", "same_pass_metric_failure_count", "service_reconciliation_failure_count"):
        _assert_equal(summary.get(field, ""), 0, field)

    projection_transformer_rows = _parse_csv(
        _require_member(members, "projection_diagnostics/projection_transformer_step_rows.csv"),
        "projection_diagnostics/projection_transformer_step_rows.csv",
    )
    if not projection_transformer_rows:
        raise ValueError(f"task {cell.task_id}: projection transformer-step evidence is empty")
    for row in projection_transformer_rows:
        _assert_equal(row.get("scale", ""), cell.scale, "projection transformer-step scale")
        _assert_equal(row.get("algorithm", ""), cell.algorithm, "projection transformer-step algorithm")
        _assert_equal(row.get("training_seed", ""), cell.seed, "projection transformer-step training_seed")
        episode_index = _parse_exact_int(
            row.get("episode_index", ""),
            "projection transformer-step episode_index",
        )
        if not 0 <= episode_index < workflow.FORMAL_MODEL_BEST_EVAL_EPISODES:
            raise ValueError(
                f"projection transformer-step episode_index out of range: {episode_index}"
            )
        _assert_canonical_episode_seed(row, cell, episode_index, "projection transformer-step")
    projection_seed_rows = _parse_csv(
        _require_member(members, "projection_diagnostics/projection_seed_summary.csv"),
        "projection_diagnostics/projection_seed_summary.csv",
    )
    if len(projection_seed_rows) != 1:
        raise ValueError(f"task {cell.task_id}: projection seed summary must contain exactly one row")
    projection_episode_rows = _parse_csv(
        _require_member(members, "projection_diagnostics/projection_episode_summary.csv"),
        "projection_diagnostics/projection_episode_summary.csv",
    )
    indexed_projection = _validate_episode_rows(
        projection_episode_rows,
        label=f"task {cell.task_id} projection episode summary",
    )
    projection_by_episode: dict[int, dict[str, str]] = {}
    for episode_index, projection_row in indexed_projection:
        _assert_equal(projection_row.get("scale", ""), cell.scale, "projection scale")
        _assert_equal(projection_row.get("algorithm", ""), cell.algorithm, "projection algorithm")
        _assert_equal(projection_row.get("training_seed", ""), cell.seed, "projection training_seed")
        _assert_canonical_episode_seed(projection_row, cell, episode_index, "projection")
        violation_count = _parse_exact_int(
            projection_row.get("transformer_feasibility_violation_count", ""),
            "transformer_feasibility_violation_count",
        )
        max_excess = _parse_finite_float(
            projection_row.get("transformer_feasibility_max_excess_kw", ""),
            "transformer_feasibility_max_excess_kw",
        )
        if violation_count != 0 or max_excess > 1e-9:
            raise ValueError(
                f"task {cell.task_id}: transformer feasibility invariant failed at "
                f"episode {episode_index}"
            )
        projection_by_episode[episode_index] = projection_row
    projection_summary_status = _validate_projection_seed_summary(
        projection_seed_rows[0],
        cell,
        indexed_projection,
    )

    diagnostic_rows = _parse_csv(_require_member(members, "diagnostics/episode_diagnostics.csv"), "diagnostics/episode_diagnostics.csv")
    indexed_diagnostics = _validate_episode_rows(diagnostic_rows, label=f"task {cell.task_id} episode diagnostics")
    canonical_diagnostic_rows: list[dict[str, object]] = []
    for episode_index, row in indexed_diagnostics:
        projection_row = projection_by_episode[episode_index]
        _assert_equal(row.get("scale", ""), cell.scale, "diagnostic scale")
        _assert_equal(row.get("algorithm", ""), cell.algorithm, "diagnostic algorithm")
        _assert_equal(row.get("training_seed", ""), cell.seed, "diagnostic training_seed")
        _assert_equal(row.get("diagnostic_schema_version", ""), 3, "diagnostic schema version")
        _assert_equal(row.get("matrix_job_id", ""), diagnostic_job, "diagnostic matrix job ID")
        episode_seed = _assert_canonical_episode_seed(row, cell, episode_index, "diagnostic")
        service_status, energy_status = service_by_episode[episode_index]
        canonical_diagnostic_rows.append(
            {
                "task_id": cell.task_id,
                "scale": cell.scale,
                "algorithm": cell.algorithm,
                "model_name": workflow.MODEL_NAMES[cell.algorithm],
                "seed": cell.seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "checkpoint_role": "model.best",
                "checkpoint_identity_sha256": checkpoint_identity,
                "config_identity_sha256": config_identity,
                "source_identity": source_identity,
                "source_bundle_identity": source_bundle_identity,
                "source_training_package_identity": training_package_identity,
                "training_job_id": train_job or trainset,
                "training_set_id": trainset,
                "diagnostic_job_id": diagnostic_job,
                workflow.KEY_MECHANISM_METRIC: _parse_finite_float(projection_row.get(workflow.KEY_MECHANISM_METRIC, ""), workflow.KEY_MECHANISM_METRIC),
                "transformer_activation_rate": _parse_finite_float(projection_row.get("transformer_activation_rate", ""), "transformer_activation_rate"),
                "active_alpha_mean": projection_row.get("active_alpha_mean", ""),
                "active_alpha_min": projection_row.get("active_alpha_min", ""),
                "commanded_power_removed_kw_step_sum": _parse_finite_float(projection_row.get("commanded_power_removed_kw_step_sum", ""), "commanded_power_removed_kw_step_sum"),
                "commanded_power_removed_kw_mean_when_active": projection_row.get("commanded_power_removed_kw_mean_when_active", ""),
                "estimated_energy_removed_kwh": _parse_finite_float(projection_row.get("estimated_energy_removed_kwh", ""), "estimated_energy_removed_kwh"),
                "action_correction_l1_sum": _parse_finite_float(projection_row.get("action_correction_l1_sum", ""), "action_correction_l1_sum"),
                "action_correction_l1_mean_per_active_ev_decision": _parse_finite_float(projection_row.get("action_correction_l1_mean_per_active_ev_decision", ""), "action_correction_l1_mean_per_active_ev_decision"),
                "corrected_ev_decision_fraction": _parse_finite_float(projection_row.get("corrected_ev_decision_fraction", ""), "corrected_ev_decision_fraction"),
                "corrected_ev_decisions": _parse_exact_int(projection_row.get("corrected_ev_decisions", ""), "corrected_ev_decisions"),
                "active_ev_decisions": _parse_exact_int(projection_row.get("active_ev_decisions", ""), "active_ev_decisions"),
                "transformer_feasibility_violation_count": 0,
                "transformer_feasibility_max_excess_kw": 0.0,
                "mean_active_ev_action": _parse_finite_float(row.get("global_action_mean_active", ""), "global_action_mean_active"),
                "active_nonzero_action_fraction": _parse_finite_float(row.get("global_action_nonzero_fraction_active", ""), "global_action_nonzero_fraction_active"),
                "transformer_upper_bound_action_fraction": _parse_finite_float(row.get("transformer_action_fraction_at_max_active_macro_mean", ""), "transformer_action_fraction_at_max_active_macro_mean"),
                "charger_upper_bound_action_fraction": _parse_finite_float(row.get("charger_action_fraction_at_max_active_macro_mean", ""), "charger_action_fraction_at_max_active_macro_mean"),
                "transformer_positive_pressure_hhi": _parse_finite_float(row.get("transformer_positive_charge_action_hhi_mean", ""), "transformer_positive_charge_action_hhi_mean"),
                "transformer_positive_pressure_gini": _parse_finite_float(row.get("transformer_positive_charge_action_gini_mean", ""), "transformer_positive_charge_action_gini_mean"),
                "charger_positive_pressure_hhi": _parse_finite_float(row.get("charger_positive_charge_action_hhi_mean", ""), "charger_positive_charge_action_hhi_mean"),
                "charger_positive_pressure_gini": _parse_finite_float(row.get("charger_positive_charge_action_gini_mean", ""), "charger_positive_charge_action_gini_mean"),
                "transformer_zero_pressure_rate": _parse_finite_float(row.get("transformer_allocation_zero_pressure_step_fraction", ""), "transformer_allocation_zero_pressure_step_fraction"),
                "charger_zero_pressure_rate": _parse_finite_float(row.get("charger_allocation_zero_pressure_step_fraction", ""), "charger_allocation_zero_pressure_step_fraction"),
                "transformer_overload": _parse_finite_float(row.get("total_transformer_overload", ""), "total_transformer_overload"),
                "power_tracker_violation": _parse_finite_float(row.get("power_tracker_violation", ""), "power_tracker_violation"),
                "tracking_error": _parse_finite_float(row.get("tracking_error", ""), "tracking_error"),
                "energy_tracking_error": _parse_finite_float(row.get("energy_tracking_error", ""), "energy_tracking_error"),
                "evs_served": _parse_finite_float(row.get("total_ev_served", ""), "total_ev_served"),
                "energy_delivered": _parse_finite_float(row.get("total_energy_charged", ""), "total_energy_charged"),
                "average_satisfaction": _parse_finite_float(row.get("average_user_satisfaction", ""), "average_user_satisfaction"),
                "mapping_validation": "pass",
                "canonical_reconciliation": "pass",
                "service_reconciliation": service_status,
                "energy_reconciliation": energy_status,
                "required_values_finite": True,
                "diagnostic_schema_version": 3,
                "reconciliation_contract_version": 2,
            }
        )

    same_pass_rows = _parse_csv(
        _require_member(members, "diagnostics/same_pass_canonical_eval30.csv"),
        "diagnostics/same_pass_canonical_eval30.csv",
    )
    same_pass_episode_rows = [row for row in same_pass_rows if str(row.get("row_type", "")).strip() == "episode"]
    indexed_same_pass = _validate_episode_rows(same_pass_episode_rows, label=f"task {cell.task_id} same-pass eval30")
    canonical_eval_rows: list[dict[str, object]] = []
    for episode_index, row in indexed_same_pass:
        _assert_equal(row.get("algorithm", ""), cell.algorithm, "same-pass algorithm")
        _assert_equal(row.get("seed", ""), cell.seed, "same-pass seed")
        episode_seed = _assert_canonical_episode_seed(row, cell, episode_index, "same-pass")
        _assert_model_best_checkpoint(row.get("checkpoint", ""), "same-pass checkpoint")
        canonical_eval_rows.append(
            {
                "task_id": cell.task_id,
                "scale": cell.scale,
                "algorithm": cell.algorithm,
                "model_name": workflow.MODEL_NAMES[cell.algorithm],
                "seed": cell.seed,
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "checkpoint_role": "model.best",
                "checkpoint_identity_sha256": checkpoint_identity,
                "config_identity_sha256": config_identity,
                "source_identity": source_identity,
                "source_bundle_identity": source_bundle_identity,
                "source_training_package_identity": training_package_identity,
                "training_job_id": train_job or trainset,
                "training_set_id": trainset,
                "diagnostic_job_id": diagnostic_job,
                "episode_reward": _parse_finite_float(row.get("episode_reward", ""), "same-pass episode_reward"),
                "tracking_error": _parse_finite_float(row.get("tracking_error", ""), "same-pass tracking_error"),
                "energy_tracking_error": _parse_finite_float(row.get("energy_tracking_error", ""), "same-pass energy_tracking_error"),
                "power_tracker_violation": _parse_finite_float(row.get("power_tracker_violation", ""), "same-pass power_tracker_violation"),
                "average_satisfaction": _parse_finite_float(row.get("average_user_satisfaction", ""), "same-pass average_user_satisfaction"),
                "energy_delivered": _parse_finite_float(row.get("total_energy_charged", ""), "same-pass total_energy_charged"),
                "evs_served": _parse_finite_float(row.get("total_ev_served", ""), "same-pass total_ev_served"),
                "transformer_overload": _parse_finite_float(row.get("total_transformer_overload", ""), "same-pass total_transformer_overload"),
                "required_values_finite": True,
            }
        )
    package_summary = {
        "task_id": cell.task_id,
        "scale": cell.scale,
        "algorithm": cell.algorithm,
        "seed": cell.seed,
        "training_job_id": train_job or trainset,
        "training_set_id": trainset,
        "diagnostic_job_id": diagnostic_job,
        "source_identity": source_identity,
        "source_bundle_identity": source_bundle_identity,
        "source_training_package_identity": training_package_identity,
        "checkpoint_identity_sha256": checkpoint_identity,
        "config_identity_sha256": config_identity,
        "diagnostic_episode_count": len(canonical_diagnostic_rows),
        "same_pass_eval_episode_count": len(canonical_eval_rows),
        "projection_episode_count": len(projection_episode_rows),
        "projection_transformer_step_rows": len(projection_transformer_rows),
        "transformer_feasibility_violation_count": projection_summary_status["transformer_feasibility_violation_count"],
        "transformer_feasibility_max_excess_kw": projection_summary_status["transformer_feasibility_max_excess_kw"],
        "mapping_validation": "pass",
        "canonical_reconciliation": "pass",
        "service_reconciliation": "pass",
        "energy_reconciliation": "pass",
        "package_path": str(package.resolve()),
        "package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
    }
    return DiagnosticPackageRecord(
        package_path=package,
        cell=cell,
        summary=package_summary,
        diagnostic_rows=canonical_diagnostic_rows,
        same_pass_eval_rows=canonical_eval_rows,
    )


def aggregate_diagnostic_packages(
    package_root: Path | str,
    *,
    training_job_id: object | None = None,
    training_set_id: object | None = None,
    diagnostic_job_id: object,
    expected_source_identity: str | None = None,
    expected_source_bundle_identity: str | None = None,
) -> DiagnosticAggregateResult:
    trainset, train_job = _resolve_training_set_id(training_job_id, training_set_id)
    diagnostic_job = _validate_job_id(diagnostic_job_id, "diagnostic job ID")
    discovered = _discover_diagnostic_packages(package_root, train_job or None, diagnostic_job, training_set_id=trainset)
    records = [
        validate_diagnostic_package(
            discovered[cell.task_id],
            cell,
            training_job_id=train_job or None,
            training_set_id=trainset,
            diagnostic_job_id=diagnostic_job,
            expected_source_identity=expected_source_identity,
            expected_source_bundle_identity=expected_source_bundle_identity,
        )
        for cell in workflow.formal_matrix()
    ]
    if len({str(record.summary["source_identity"]) for record in records}) != 1:
        raise ValueError("diagnostic packages must share one exact source identity")
    if len({str(record.summary["source_bundle_identity"]) for record in records}) != 1:
        raise ValueError("diagnostic packages must share one exact source-bundle identity")
    diagnostic_rows = [row for record in records for row in record.diagnostic_rows]
    same_pass_eval_rows = [row for record in records for row in record.same_pass_eval_rows]
    workflow.validate_diagnostic_gate(diagnostic_rows)
    workflow.validate_eval30_gate(same_pass_eval_rows)
    return DiagnosticAggregateResult(
        diagnostic_rows=diagnostic_rows,
        same_pass_eval_rows=same_pass_eval_rows,
        package_summaries=[record.summary for record in records],
    )
