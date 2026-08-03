import csv
import hashlib
import io
import importlib
import json
import sys
import tarfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.infrastructure_diagnostics import (  # noqa: E402
    CHARGER_DIAGNOSTIC_COLUMNS,
    EPISODE_DIAGNOSTIC_COLUMNS,
    SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
    TRANSFORMER_DIAGNOSTIC_COLUMNS,
)

metric_module = importlib.import_module(
    "analysis.ev_charging_infrastructure_control.metric_definitions"
)


EXPECTED_METRICS = {
    "episode_reward",
    "tracking_error",
    "energy_tracking_error",
    "power_tracker_violation",
    "global_action_fraction_at_max_active",
    "global_action_nonzero_fraction_active",
    "transformer_action_fraction_at_max_active_macro_mean",
    "charger_action_fraction_at_max_active_macro_mean",
    "transformer_positive_charge_action_hhi_mean",
    "charger_positive_charge_action_hhi_mean",
    "transformer_allocation_zero_pressure_step_fraction",
    "charger_allocation_zero_pressure_step_fraction",
    "total_transformer_overload",
    "mean_transformer_overload_frequency_fraction",
    "mean_total_transformer_overload_magnitude",
    "maximum_transformer_overload_magnitude",
    "total_ev_served",
    "total_energy_charged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "transformer_positive_charge_action_gini_mean",
    "charger_positive_charge_action_gini_mean",
}

SCALES = ("25cp", "100cp", "500cp", "1000cp")
ALGORITHMS = ("actiongnn", "hierarchical")
TRAINING_SEEDS = tuple(range(5))
SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID = "58745233"
SYNTHETIC_REDUCER_JOB_ID = "58745234"
SYNTHETIC_FORMAL_JOB_ID = "58513929"
SYNTHETIC_SOURCE_COMMIT_SHA = "cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad"


def extraction_module():
    return importlib.import_module(
        "analysis.ev_charging_infrastructure_control.extract_diagnostic_datasets"
    )


def csv_bytes(fieldnames, rows):
    text_buffer = io.StringIO()
    writer = csv.DictWriter(text_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return text_buffer.getvalue().encode("utf-8")


def add_bytes_to_tar(archive, member_name, content):
    tar_info = tarfile.TarInfo(member_name)
    tar_info.size = len(content)
    tar_info.mtime = 0
    archive.addfile(tar_info, io.BytesIO(content))


def tar_gz_bytes(members):
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for member_name, content in members:
            add_bytes_to_tar(archive, member_name, content)
    return archive_buffer.getvalue()


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def checksum_manifest_bytes(member_pairs, duplicate_name=None, corrupt_name=None):
    lines = []
    for member_name, content in member_pairs:
        digest = sha256_bytes(content)
        if member_name == corrupt_name:
            digest = "0" * 64
        lines.append(f"{digest}  {member_name}\n")
    if duplicate_name is not None:
        content = dict(member_pairs)[duplicate_name]
        lines.append(f"{sha256_bytes(content)}  {duplicate_name}\n")
    return "".join(lines).encode("utf-8")


def synthetic_episode_row(scale, algorithm, training_seed, episode_index):
    row = {column: "1" for column in EPISODE_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "episode_index": str(episode_index),
            "episode_seed": str(710000 + training_seed * 1000 + episode_index),
            "config": "config_files/PublicPST_25cp.yaml",
            "checkpoint_prefix": "checkpoint/model.best",
            "run_name": "synthetic",
            "done": "True",
            "environment_action_domain_support": "continuous",
            "v2g_enabled": "False",
            "v2g_enabled_source": "config",
            "diagnostic_schema_version": "3",
        }
    )
    return row


def synthetic_seed_row(scale, algorithm, training_seed, n_eval_episodes):
    row = {column: "1" for column in SEED_SUMMARY_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "n_eval_episodes": str(n_eval_episodes),
            "environment_action_domain_support": "continuous",
            "v2g_enabled": "False",
            "v2g_enabled_source": "config",
            "diagnostic_schema_version": "3",
        }
    )
    return row


def synthetic_transformer_row(scale, algorithm, training_seed, episode_index):
    row = {column: "1" for column in TRANSFORMER_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "episode_index": str(episode_index),
            "episode_seed": str(710000 + training_seed * 1000 + episode_index),
            "transformer_id": "0",
            "user_satisfaction_source": "ev2gym",
            "diagnostic_schema_version": "3",
        }
    )
    return row


def synthetic_charger_row(scale, algorithm, training_seed, episode_index):
    row = {column: "1" for column in CHARGER_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "episode_index": str(episode_index),
            "episode_seed": str(710000 + training_seed * 1000 + episode_index),
            "transformer_id": "0",
            "charger_id": "0",
            "user_satisfaction_source": "ev2gym",
            "diagnostic_schema_version": "3",
        }
    )
    return row


def build_synthetic_task_package(
    scale,
    algorithm,
    *,
    episodes_per_seed=1,
    corrupt_nested_checksum=False,
    duplicate_nested_manifest_entry=False,
    add_uncovered_nested_file=False,
):
    members = []
    for training_seed in TRAINING_SEEDS:
        seed_prefix = f"seed{training_seed}/diagnostics"
        episode_rows = [
            synthetic_episode_row(scale, algorithm, training_seed, episode_index)
            for episode_index in range(episodes_per_seed)
        ]
        transformer_rows = [
            synthetic_transformer_row(scale, algorithm, training_seed, episode_index)
            for episode_index in range(episodes_per_seed)
        ]
        charger_rows = [
            synthetic_charger_row(scale, algorithm, training_seed, episode_index)
            for episode_index in range(episodes_per_seed)
        ]
        members.extend(
            [
                (
                    f"{seed_prefix}/episode_diagnostics.csv",
                    csv_bytes(EPISODE_DIAGNOSTIC_COLUMNS, episode_rows),
                ),
                (
                    f"{seed_prefix}/seed_summary_diagnostics.csv",
                    csv_bytes(
                        SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
                        [synthetic_seed_row(scale, algorithm, training_seed, episodes_per_seed)],
                    ),
                ),
                (
                    f"{seed_prefix}/transformer_diagnostics.csv",
                    csv_bytes(TRANSFORMER_DIAGNOSTIC_COLUMNS, transformer_rows),
                ),
                (
                    f"{seed_prefix}/charger_diagnostics.csv",
                    csv_bytes(CHARGER_DIAGNOSTIC_COLUMNS, charger_rows),
                ),
            ]
        )
    if add_uncovered_nested_file:
        members.append(("seed0/diagnostics/uncovered.txt", b"uncovered"))
    manifest_members = [
        member for member in members if member[0] != "seed0/diagnostics/uncovered.txt"
    ]
    duplicate_name = manifest_members[0][0] if duplicate_nested_manifest_entry else None
    manifest = checksum_manifest_bytes(
        manifest_members,
        duplicate_name=duplicate_name,
        corrupt_name=manifest_members[0][0] if corrupt_nested_checksum else None,
    )
    members.append(("checksums/package_file_checksums.sha256", manifest))
    return tar_gz_bytes(members)


def build_synthetic_complete_bundle(
    tmp_path,
    *,
    omitted_task_id=None,
    extra_outer_member=None,
    corrupt_task_package_sha=False,
    corrupt_outer_checksum=False,
    corrupt_nested_checksum=False,
    duplicate_outer_manifest_entry=False,
    duplicate_nested_manifest_entry=False,
    add_uncovered_outer_file=False,
    add_uncovered_nested_file=False,
    wrong_workflow_identity=False,
    add_outer_symlink=False,
):
    task_packages = []
    inventory_rows = []
    task_id = 0
    for scale in SCALES:
        for algorithm in ALGORITHMS:
            package_name = f"{scale}_{algorithm}_diagnostics.tar.gz"
            package_bytes = build_synthetic_task_package(
                scale,
                algorithm,
                corrupt_nested_checksum=corrupt_nested_checksum and task_id == 0,
                duplicate_nested_manifest_entry=(
                    duplicate_nested_manifest_entry and task_id == 0
                ),
                add_uncovered_nested_file=add_uncovered_nested_file and task_id == 0,
            )
            package_sha256 = sha256_bytes(package_bytes)
            if corrupt_task_package_sha and task_id == 0:
                package_sha256 = "0" * 64
            if task_id != omitted_task_id:
                task_packages.append((f"task_packages/{package_name}", package_bytes))
                inventory_rows.append(
                    {
                        "task_id": str(task_id),
                        "scale": scale,
                        "algorithm": algorithm,
                        "package_name": package_name,
                        "package_sha256": package_sha256,
                        "status": "ok",
                        "checkpoint_groups": "5",
                        "episode_count": "150",
                    }
                )
            task_id += 1

    inventory = csv_bytes(
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
        inventory_rows,
    )
    workflow = {
        "status": "ok",
        "array_job_id": (
            "wrong" if wrong_workflow_identity else SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID
        ),
        "task_package_count": 8,
        "checkpoint_group_count": 40,
        "episode_count": 1200,
        "formal_job_id": SYNTHETIC_FORMAL_JOB_ID,
        "schema_version": "3",
        "reconciliation_contract_version": 2,
    }
    members = [
        *task_packages,
        ("summaries/task_inventory.csv", inventory),
        (
            "runtime_metadata/diagnostic_array_job_id.txt",
            f"{SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID}\n".encode("utf-8"),
        ),
        (
            "runtime_metadata/reducer_job_id.txt",
            f"{SYNTHETIC_REDUCER_JOB_ID}\n".encode("utf-8"),
        ),
        (
            "runtime_metadata/formal_job_id.txt",
            f"{SYNTHETIC_FORMAL_JOB_ID}\n".encode("utf-8"),
        ),
        (
            "runtime_metadata/source_commit_sha.txt",
            f"{SYNTHETIC_SOURCE_COMMIT_SHA}\n".encode("utf-8"),
        ),
        (
            "validation/complete_workflow_validation.json",
            json.dumps(workflow, sort_keys=True).encode("utf-8"),
        ),
    ]
    if add_uncovered_outer_file:
        members.append(("runtime_metadata/uncovered.txt", b"uncovered"))
    manifest_members = [
        member for member in members if member[0] != "runtime_metadata/uncovered.txt"
    ]
    duplicate_name = manifest_members[0][0] if duplicate_outer_manifest_entry else None
    manifest = checksum_manifest_bytes(
        manifest_members,
        duplicate_name=duplicate_name,
        corrupt_name=manifest_members[0][0] if corrupt_outer_checksum else None,
    )
    members.append(("runtime_metadata/complete_file_checksums.sha256", manifest))
    if extra_outer_member is not None:
        members.append(extra_outer_member)

    archive_path = tmp_path / "complete_evidence.tar.gz"
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for member_name, content in members:
            add_bytes_to_tar(archive, member_name, content)
        if add_outer_symlink:
            tar_info = tarfile.TarInfo("unsafe_link")
            tar_info.type = tarfile.SYMTYPE
            tar_info.linkname = "../escape"
            tar_info.mtime = 0
            archive.addfile(tar_info)
    archive_path.write_bytes(archive_buffer.getvalue())
    return archive_path


def test_registry_contains_only_approved_metrics():
    definitions = metric_module.METRIC_DEFINITIONS

    assert {definition.name for definition in definitions} == EXPECTED_METRICS
    assert len(definitions) == len(EXPECTED_METRICS)
    assert len({definition.name for definition in definitions}) == len(definitions)


def test_registry_preserves_scientific_boundaries():
    reward = metric_module.metric_definition("episode_reward")
    saturation = metric_module.metric_definition(
        "global_action_fraction_at_max_active"
    )
    gini = metric_module.metric_definition(
        "transformer_positive_charge_action_gini_mean"
    )

    assert reward.tier == "primary"
    assert reward.preferred_direction == "higher"
    assert reward.aggregation_rule == "mean 30 episodes"
    assert reward.seed_summary_column is None
    assert saturation.tier == "mechanism"
    assert saturation.preferred_direction == "context_dependent"
    assert "not automatically" in saturation.interpretation_warning.lower()
    assert gini.tier == "robustness"
    assert "robustness" in gini.interpretation_warning.lower()
    assert "not independent primary" in gini.interpretation_warning.lower()


def test_registry_tier_lookup_and_required_columns_are_semantic():
    primary_names = metric_module.metric_names_for_tier("primary")
    required_episode_columns = metric_module.required_source_columns("episode")
    required_transformer_columns = metric_module.required_source_columns("transformer")

    assert primary_names == (
        "episode_reward",
        "tracking_error",
        "energy_tracking_error",
        "power_tracker_violation",
    )
    assert "episode_reward" in required_episode_columns
    assert "overload_frequency_fraction" in required_transformer_columns
    assert "battery_degradation" not in required_episode_columns


def test_registry_rejects_unknown_metric_and_tier():
    with pytest.raises(KeyError, match="unknown metric"):
        metric_module.metric_definition("battery_degradation")
    with pytest.raises(KeyError, match="unknown tier"):
        metric_module.metric_names_for_tier("economic")


def test_complete_bundle_rejects_unsafe_member(tmp_path):
    bundle = build_synthetic_complete_bundle(
        tmp_path, extra_outer_member=("../escape.txt", b"unsafe")
    )
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_rejects_package_sha_mismatch(tmp_path):
    bundle = build_synthetic_complete_bundle(
        tmp_path, corrupt_task_package_sha=True
    )
    with pytest.raises(RuntimeError, match="task package SHA-256 mismatch"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_requires_exact_task_matrix(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, omitted_task_id=7)
    with pytest.raises(RuntimeError, match="exactly 8 task packages"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_rejects_outer_checksum_mismatch(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, corrupt_outer_checksum=True)
    with pytest.raises(RuntimeError, match="outer SHA-256 mismatch"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_rejects_nested_checksum_mismatch(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, corrupt_nested_checksum=True)
    with pytest.raises(RuntimeError, match="nested SHA-256 mismatch"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_rejects_duplicate_manifest_entries(tmp_path):
    bundle = build_synthetic_complete_bundle(
        tmp_path, duplicate_outer_manifest_entry=True
    )
    with pytest.raises(RuntimeError, match="duplicate checksum manifest entry"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_rejects_uncovered_regular_file(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, add_uncovered_outer_file=True)
    with pytest.raises(RuntimeError, match="not covered by checksum manifest"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_rejects_symlink(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, add_outer_symlink=True)
    with pytest.raises(RuntimeError, match="unsafe archive member"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_rejects_wrong_workflow_identity(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, wrong_workflow_identity=True)
    with pytest.raises(RuntimeError, match="workflow identity mismatch"):
        extraction_module().read_complete_bundle(bundle)


def test_complete_bundle_reads_provenance_and_task_packages(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)

    evidence = extraction_module().read_complete_bundle(bundle)

    assert evidence.provenance.diagnostic_array_job_id == (
        SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID
    )
    assert evidence.provenance.reducer_job_id == SYNTHETIC_REDUCER_JOB_ID
    assert evidence.provenance.formal_training_job_id == SYNTHETIC_FORMAL_JOB_ID
    assert evidence.provenance.source_commit_sha == SYNTHETIC_SOURCE_COMMIT_SHA
    assert evidence.provenance.diagnostic_schema_version == "3"
    assert evidence.provenance.reconciliation_contract_version == 2
    assert len(evidence.task_packages) == 8
    assert {
        (task_package.scale, task_package.algorithm)
        for task_package in evidence.task_packages
    } == {
        (scale, algorithm)
        for scale in SCALES
        for algorithm in ALGORITHMS
    }
