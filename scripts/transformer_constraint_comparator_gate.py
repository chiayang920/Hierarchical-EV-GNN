#!/usr/bin/env python3
"""Read-only compatibility gate for reusing Formal75K Full Hierarchy evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_SOURCE_REF = "ae3911debe04a65f00ca5bb6ec03baed059f86f4"
CURRENT_EXPERIMENT_BASE = "5d09d99ec0836188472884440fa1ed207047fc84"
DEFAULT_HISTORICAL_ROOT = Path(
    "/Users/jameschen/Desktop/MAIN_HOME/Monash/monash_course/2026S1/Research/"
    "Research_project/EVGNN_Research_Artefacts/80_audit_and_handover/"
    "corrected_nonnegative_flat_actiongnn_design/formal_75k_80cell_workflow/"
    "local_repair_20260807_v3_macos_archive/outputs/formal_75k_incremental_collection_ae3911d"
)

SCALES = ("25cp", "100cp", "500cp", "1000cp")
SEEDS = tuple(range(10))
SOURCE_CONTRACT_PATHS = (
    "TD3/TD3_HierarchicalActionGNN.py",
    "TD3/TD3_ActionGNN_Controlled.py",
    "utils/state_public_pst_gnn.py",
    "utils/replay_buffer_actiongnn.py",
    "utils/ev2gym_training_utils.py",
)
CONFIG_CONTRACT_PATHS = (
    "config_files/PublicPST_25cp.yaml",
    "config_files/PublicPST_100.yaml",
    "config_files/PublicPST_500.yaml",
    "config_files/PublicPST_1000.yaml",
)
APPROVED_CONTRACT_BLOB_SHA_BY_PATH = {
    "TD3/TD3_HierarchicalActionGNN.py": "3f18ae9790f898f4d5715770136eee659118d8e9",
    "TD3/TD3_ActionGNN_Controlled.py": "020c03a11fe5876267e1a4d223c2e2a39994b845",
    "utils/state_public_pst_gnn.py": "11330e623bc2cd65c6bafdd6646e8ead68bd2889",
    "utils/replay_buffer_actiongnn.py": "b5bfd371e44f55093d96a68d000a8e0ae8218f8f",
    "utils/ev2gym_training_utils.py": "310f13874c4b77ae0ffd7b2432198ab3c86c9a93",
    "config_files/PublicPST_25cp.yaml": "210fe12a3fcd42f720e9db7addc568910890c86d",
    "config_files/PublicPST_100.yaml": "78b35b309a6e734b8f9e6afa228c1f1b22726d11",
    "config_files/PublicPST_500.yaml": "b0256aeee8fb8945316d732ed57efc29a8ae07ad",
    "config_files/PublicPST_1000.yaml": "8799ecabc6bc12ee00701cb0c58350850c80cd49",
}
EXPECTED_CELLS = {(scale, seed) for scale in SCALES for seed in SEEDS}
EXPECTED_TRAINING_ARGS = {
    "algorithm": "hierarchical",
    "max_timesteps": 75000,
    "start_timesteps": 1000,
    "eval_freq": 5000,
    "eval_episodes": 5,
    "discrete_actions": 1,
    "batch_size": 64,
    "replay_buffer_size": 100000,
}
EXPECTED_KWARGS = {
    "discount": 0.99,
    "tau": 0.005,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "policy_freq": 2,
    "fx_dim": 32,
    "fx_GNN_hidden_dim": 64,
    "mlp_hidden_dim": 512,
    "lr": 0.0003,
    "discrete_actions": 1,
}
EVAL_SEED_BASE_BY_SCALE = {
    "25cp": 710000,
    "100cp": 720000,
    "500cp": 730000,
    "1000cp": 740000,
}


@dataclass(frozen=True)
class ComparatorReuseResult:
    status: str
    checks: dict[str, str]
    reasons: list[str]
    training_package_count: int
    eval30_package_count: int
    diagnostic_package_count: int
    source_paths_checked: int
    config_paths_checked: int


def _fail_result(
    checks: dict[str, str],
    reasons: list[str],
    *,
    training_package_count: int = 0,
    eval30_package_count: int = 0,
    diagnostic_package_count: int = 0,
) -> ComparatorReuseResult:
    return ComparatorReuseResult(
        status="FAIL",
        checks=checks,
        reasons=reasons,
        training_package_count=training_package_count,
        eval30_package_count=eval30_package_count,
        diagnostic_package_count=diagnostic_package_count,
        source_paths_checked=len(SOURCE_CONTRACT_PATHS),
        config_paths_checked=len(CONFIG_CONTRACT_PATHS),
    )


def _git_blob(repo_root: Path, git_ref: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{git_ref}:{relative_path}"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"missing git object for {git_ref}:{relative_path}")
    return result.stdout.strip()


def _blob_sha_from_filesystem(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()


def _parse_env(payload: bytes, label: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{label}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _read_safe_tar(path: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"unsupported archive member: {member.name}")
                if member.name.startswith("/") or ".." in Path(member.name).parts:
                    raise ValueError(f"unsafe archive member path: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unreadable archive member: {member.name}")
                members[member.name] = handle.read()
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"unreadable package {path}: {exc}") from exc
    return members


def _require_member(members: Mapping[str, bytes], name: str) -> bytes:
    try:
        return members[name]
    except KeyError as exc:
        raise ValueError(f"required package member missing: {name}") from exc


def _parse_yaml(payload: bytes, label: str) -> dict[str, object]:
    value = yaml.safe_load(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return value


def _parse_csv(payload: bytes, label: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if not reader.fieldnames:
        raise ValueError(f"{label} must contain a CSV header")
    return list(reader)


def _as_int(value: object, label: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"-?[0-9]+", text):
        raise ValueError(f"{label} must be an integer; got {value!r}")
    return int(text)


def _as_float(value: object, label: str) -> float:
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; got {value!r}")
    return number


def _expect(actual: object, expected: object, label: str) -> None:
    if str(actual).strip() != str(expected):
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _collect_packages(root: Path, pattern: str) -> dict[tuple[str, int], Path]:
    collected: dict[tuple[str, int], Path] = {}
    regex = re.compile(pattern)
    for package in root.glob("**/*.tar.gz"):
        match = regex.match(package.name)
        if not match:
            continue
        scale, seed_text = match.group("scale"), match.group("seed")
        key = (scale, int(seed_text))
        if key in collected:
            raise ValueError(f"duplicate historical package for {key}")
        collected[key] = package
    missing = sorted(EXPECTED_CELLS - set(collected))
    if missing:
        raise ValueError(f"INSUFFICIENT_EVIDENCE missing packages: {missing[:4]}")
    return collected


def _validate_source_identity(repo_root: Path, current_ref: str, historical_ref: str) -> None:
    if current_ref == historical_ref:
        return
    if current_ref == CURRENT_EXPERIMENT_BASE and historical_ref == HISTORICAL_SOURCE_REF:
        if set(APPROVED_CONTRACT_BLOB_SHA_BY_PATH) != set((*SOURCE_CONTRACT_PATHS, *CONFIG_CONTRACT_PATHS)):
            raise ValueError("approved source/config contract blob table is incomplete")
        for relative_path, expected_blob in APPROVED_CONTRACT_BLOB_SHA_BY_PATH.items():
            actual_blob = _blob_sha_from_filesystem(repo_root / relative_path)
            if actual_blob != expected_blob:
                raise ValueError(f"source/config contract drift: {relative_path}")
        return
    for relative_path in (*SOURCE_CONTRACT_PATHS, *CONFIG_CONTRACT_PATHS):
        current_blob = _git_blob(repo_root, current_ref, relative_path)
        historical_blob = _git_blob(repo_root, historical_ref, relative_path)
        if current_blob != historical_blob:
            raise ValueError(f"source/config contract drift: {relative_path}")


def _validate_training_package(path: Path, scale: str, seed: int) -> None:
    members = _read_safe_tar(path)
    metadata = _parse_env(
        _require_member(members, "runtime_metadata/task_runtime_metadata.env"),
        "runtime_metadata/task_runtime_metadata.env",
    )
    expectations = {
        "algorithm": "hierarchical",
        "scale": scale,
        "seed": seed,
        "training_steps": 75000,
        "evaluation_cadence": 5000,
        "evaluation_episodes": 5,
        "expected_scheduled_evaluations": 15,
        "fresh_run": "true",
        "training_exit_status": "0",
        "source_identity": HISTORICAL_SOURCE_REF,
    }
    for field, expected in expectations.items():
        _expect(metadata.get(field, ""), expected, f"{path.name} {field}")
    model_dir = str(metadata.get("model_dir_relative", "")).strip().rstrip("/")
    config_member = str(metadata.get("config_copy_relative", "")).strip()
    if not model_dir or not config_member:
        raise ValueError(f"{path.name}: missing model/config metadata")
    for member in (
        f"{model_dir}/model.best_actor",
        f"{model_dir}/model.best_actor_optimizer",
        f"{model_dir}/model.best_critic",
        f"{model_dir}/model.best_critic_optimizer",
        f"{model_dir}/run_args.yaml",
        f"{model_dir}/kwargs.yaml",
        f"{model_dir}/training_log.csv",
        f"{model_dir}/config.yaml",
        config_member,
    ):
        _require_member(members, member)
    run_args = _parse_yaml(members[f"{model_dir}/run_args.yaml"], f"{path.name} run_args.yaml")
    kwargs = _parse_yaml(members[f"{model_dir}/kwargs.yaml"], f"{path.name} kwargs.yaml")
    copied_config = _parse_yaml(members[config_member], f"{path.name} copied config")
    model_config = _parse_yaml(members[f"{model_dir}/config.yaml"], f"{path.name} model config")
    if copied_config != model_config:
        raise ValueError(f"{path.name}: copied config differs from model config")
    _expect(copied_config.get("number_of_charging_stations", ""), int(scale.removesuffix("cp")), "config scale")
    for field, expected in EXPECTED_TRAINING_ARGS.items():
        _expect(run_args.get(field, ""), expected, f"run_args.{field}")
    for field, expected in EXPECTED_KWARGS.items():
        _expect(kwargs.get(field, ""), expected, f"kwargs.{field}")
    rows = _parse_csv(members[f"{model_dir}/training_log.csv"], f"{path.name} training_log.csv")
    eval_rows = [row for row in rows if row.get("type") == "evaluation"]
    if len(eval_rows) != 15:
        raise ValueError(f"{path.name}: expected 15 scheduled evaluations")
    observed_steps = [_as_int(row.get("timestep", ""), "evaluation timestep") for row in eval_rows]
    if observed_steps != list(range(5000, 75001, 5000)):
        raise ValueError(f"{path.name}: scheduled evaluations must be 5000..75000")


def _validate_eval_package(path: Path, scale: str, seed: int) -> None:
    members = _read_safe_tar(path)
    metadata = _parse_env(
        _require_member(members, "runtime_metadata/task_runtime_metadata.env"),
        "runtime_metadata/task_runtime_metadata.env",
    )
    expectations = {
        "algorithm": "hierarchical",
        "scale": scale,
        "seed": seed,
        "source_identity": HISTORICAL_SOURCE_REF,
        "checkpoint_role": "model.best",
        "evaluation_episodes": 30,
        "evaluation_exit_status": "0",
    }
    for field, expected in expectations.items():
        _expect(metadata.get(field, ""), expected, f"{path.name} {field}")
    csv_members = [name for name in members if name.startswith("eval/") and name.endswith("_eval30.csv")]
    if len(csv_members) != 1:
        raise ValueError(f"{path.name}: expected exactly one eval30 CSV")
    rows = _parse_csv(members[csv_members[0]], csv_members[0])
    episode_rows = [row for row in rows if row.get("row_type") == "episode"]
    if len(episode_rows) != 30:
        raise ValueError(f"{path.name}: expected 30 eval30 episode rows")
    expected_seeds = [EVAL_SEED_BASE_BY_SCALE[scale] + 1000 * seed + index for index in range(30)]
    observed_seeds = []
    for row in episode_rows:
        _expect(row.get("algorithm", ""), "hierarchical", "eval algorithm")
        _expect(row.get("seed", ""), seed, "eval seed")
        if "model.best" not in str(row.get("checkpoint", "")) or "model.last" in str(row.get("checkpoint", "")):
            raise ValueError(f"{path.name}: eval30 must use model.best only")
        _as_float(row.get("episode_reward", ""), "episode_reward")
        observed_seeds.append(_as_int(row.get("episode_seed", ""), "episode_seed"))
    if observed_seeds != expected_seeds:
        raise ValueError(f"{path.name}: eval30 episode seeds do not match canonical construction")


def _validate_diagnostic_package(path: Path, scale: str, seed: int) -> None:
    members = _read_safe_tar(path)
    metadata = _parse_env(
        _require_member(members, "runtime_metadata/task_runtime_metadata.env"),
        "runtime_metadata/task_runtime_metadata.env",
    )
    expectations = {
        "algorithm": "hierarchical",
        "scale": scale,
        "seed": seed,
        "source_identity": HISTORICAL_SOURCE_REF,
        "checkpoint_role": "model.best",
        "evaluation_episodes": 30,
        "evaluation_exit_status": "0",
    }
    for field, expected in expectations.items():
        _expect(metadata.get(field, ""), expected, f"{path.name} {field}")
    metadata_schema = str(metadata.get("diagnostic_schema_version", "")).strip()
    metadata_reconciliation = str(metadata.get("reconciliation_contract_version", "")).strip()
    if metadata_schema or metadata_reconciliation:
        _expect(metadata_schema, 3, f"{path.name} diagnostic_schema_version")
        _expect(metadata_reconciliation, 2, f"{path.name} reconciliation_contract_version")
        return

    episode_member = "diagnostics/episode_diagnostics.csv"
    episode_rows = _parse_csv(_require_member(members, episode_member), episode_member)
    if len(episode_rows) != 30:
        raise ValueError(f"{path.name}: expected 30 diagnostic episode rows")
    for row in episode_rows:
        _expect(row.get("algorithm", ""), "hierarchical", "diagnostic algorithm")
        _expect(row.get("scale", ""), scale, "diagnostic scale")
        _expect(row.get("training_seed", ""), seed, "diagnostic training_seed")
        _expect(row.get("diagnostic_schema_version", ""), 3, "diagnostic_schema_version")
        _as_float(row.get("episode_reward", ""), "diagnostic episode_reward")

    same_pass_member = "diagnostics/same_pass_canonical_eval30.csv"
    same_pass_rows = _parse_csv(_require_member(members, same_pass_member), same_pass_member)
    same_pass_episodes = [row for row in same_pass_rows if row.get("row_type") == "episode"]
    if len(same_pass_episodes) != 30:
        raise ValueError(f"{path.name}: expected 30 same-pass canonical episode rows")


def _validate_package_family(packages: Mapping[tuple[str, int], Path], validator) -> None:
    for (scale, seed), path in sorted(packages.items()):
        validator(path, scale, seed)


def run_comparator_reuse_gate(
    *,
    repo_root: Path | str = PROJECT_ROOT,
    historical_root: Path | str = DEFAULT_HISTORICAL_ROOT,
    current_ref: str = CURRENT_EXPERIMENT_BASE,
    historical_ref: str = HISTORICAL_SOURCE_REF,
) -> ComparatorReuseResult:
    checks = {
        "source_contract": "PENDING",
        "config_contract": "PENDING",
        "training_contract": "PENDING",
        "eval30_contract": "PENDING",
        "diagnostic_contract": "PENDING",
    }
    reasons: list[str] = []
    training_count = eval_count = diagnostic_count = 0
    root = Path(historical_root)
    try:
        if not root.is_dir():
            raise ValueError(f"INSUFFICIENT_EVIDENCE historical root is missing: {root}")
        _validate_source_identity(Path(repo_root), current_ref, historical_ref)
        checks["source_contract"] = "PASS"
        checks["config_contract"] = "PASS"
        training = _collect_packages(
            root,
            r"^m3_formal75k_80cell_(?P<scale>25cp|100cp|500cp|1000cp)_hierarchical_seed(?P<seed>[0-9])_job[0-9]+_task[0-9]+\.tar\.gz$",
        )
        training_count = len(training)
        _validate_package_family(training, _validate_training_package)
        checks["training_contract"] = "PASS"
        eval30 = _collect_packages(
            root,
            r"^m3_formal75k_eval30_(?P<scale>25cp|100cp|500cp|1000cp)_hierarchical_seed(?P<seed>[0-9])_trainjob[0-9]+_job[0-9]+_task[0-9]+\.tar\.gz$",
        )
        eval_count = len(eval30)
        _validate_package_family(eval30, _validate_eval_package)
        checks["eval30_contract"] = "PASS"
        diagnostic = _collect_packages(
            root,
            r"^m3_formal75k_diagnostics_(?P<scale>25cp|100cp|500cp|1000cp)_hierarchical_seed(?P<seed>[0-9])_trainjob[0-9]+_job[0-9]+_task[0-9]+\.tar\.gz$",
        )
        diagnostic_count = len(diagnostic)
        _validate_package_family(diagnostic, _validate_diagnostic_package)
        checks["diagnostic_contract"] = "PASS"
    except ValueError as exc:
        failed = False
        for key, value in list(checks.items()):
            if value == "PENDING" and not failed:
                checks[key] = "FAIL"
                failed = True
            elif value == "PENDING":
                checks[key] = "SKIPPED"
        reasons.append(str(exc))
        return _fail_result(
            checks,
            reasons,
            training_package_count=training_count,
            eval30_package_count=eval_count,
            diagnostic_package_count=diagnostic_count,
        )

    return ComparatorReuseResult(
        status="PASS",
        checks=checks,
        reasons=[],
        training_package_count=training_count,
        eval30_package_count=eval_count,
        diagnostic_package_count=diagnostic_count,
        source_paths_checked=len(SOURCE_CONTRACT_PATHS),
        config_paths_checked=len(CONFIG_CONTRACT_PATHS),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT)
    parser.add_argument("--current-ref", default=CURRENT_EXPERIMENT_BASE)
    parser.add_argument("--historical-ref", default=HISTORICAL_SOURCE_REF)
    args = parser.parse_args(argv)
    result = run_comparator_reuse_gate(
        repo_root=args.repo_root,
        historical_root=args.historical_root,
        current_ref=args.current_ref,
        historical_ref=args.historical_ref,
    )
    for key, value in result.checks.items():
        print(f"{key.upper()}={value}")
    print(f"HISTORICAL_TRAINING_PACKAGE_COUNT={result.training_package_count}")
    print(f"HISTORICAL_EVAL30_PACKAGE_COUNT={result.eval30_package_count}")
    print(f"HISTORICAL_DIAGNOSTIC_PACKAGE_COUNT={result.diagnostic_package_count}")
    print(f"SOURCE_PATHS_CHECKED={result.source_paths_checked}")
    print(f"CONFIG_PATHS_CHECKED={result.config_paths_checked}")
    for reason in result.reasons:
        print(f"REASON={reason}")
    print(f"COMPARATOR_REUSE_COMPATIBILITY={result.status}")
    return 0 if result.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
