import csv
import hashlib
import io
import importlib
import json
import re
import subprocess
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
TOPOLOGY = {
    "25cp": (3, 25),
    "100cp": (7, 100),
    "500cp": (35, 500),
    "1000cp": (70, 1000),
}
SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID = "58745233"
SYNTHETIC_REDUCER_JOB_ID = "58745234"
SYNTHETIC_FORMAL_JOB_ID = "58513929"
SYNTHETIC_SOURCE_COMMIT_SHA = "cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad"


def extraction_module():
    return importlib.import_module(
        "analysis.ev_charging_infrastructure_control.extract_diagnostic_datasets"
    )


def comparison_module():
    return importlib.import_module(
        "analysis.ev_charging_infrastructure_control.compare_control_architectures"
    )


def csv_bytes(fieldnames, rows):
    text_buffer = io.StringIO()
    writer = csv.DictWriter(text_buffer, fieldnames=fieldnames, extrasaction="ignore")
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


def synthetic_episode_row(
    scale,
    algorithm,
    training_seed,
    episode_index,
    *,
    diagnostic_schema_version="3",
    matrix_job_id=SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
    non_finite_metric=False,
    mismatched_episode_seed=False,
):
    row = {column: "1" for column in EPISODE_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": matrix_job_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "episode_index": str(episode_index),
            "episode_seed": str(
                999999
                if mismatched_episode_seed
                else 710000 + training_seed * 1000 + episode_index
            ),
            "config": "config_files/PublicPST_25cp.yaml",
            "checkpoint_prefix": "checkpoint/model.best",
            "run_name": "synthetic",
            "done": "True",
            "environment_action_domain_support": "continuous",
            "v2g_enabled": "False",
            "v2g_enabled_source": "config",
            "diagnostic_schema_version": diagnostic_schema_version,
        }
    )
    if non_finite_metric:
        row["tracking_error"] = "nan"
    return row


def synthetic_seed_row(
    scale,
    algorithm,
    training_seed,
    n_eval_episodes,
    *,
    diagnostic_schema_version="3",
    matrix_job_id=SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
):
    row = {column: "1" for column in SEED_SUMMARY_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": matrix_job_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "n_eval_episodes": str(n_eval_episodes),
            "environment_action_domain_support": "continuous",
            "v2g_enabled": "False",
            "v2g_enabled_source": "config",
            "diagnostic_schema_version": diagnostic_schema_version,
        }
    )
    return row


def synthetic_transformer_row(
    scale,
    algorithm,
    training_seed,
    episode_index,
    transformer_id,
    *,
    diagnostic_schema_version="3",
    matrix_job_id=SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
):
    row = {column: "1" for column in TRANSFORMER_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": matrix_job_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "episode_index": str(episode_index),
            "episode_seed": str(710000 + training_seed * 1000 + episode_index),
            "transformer_id": str(transformer_id),
            "user_satisfaction_source": "ev2gym",
            "diagnostic_schema_version": diagnostic_schema_version,
        }
    )
    return row


def synthetic_charger_row(
    scale,
    algorithm,
    training_seed,
    episode_index,
    charger_id,
    transformer_id,
    *,
    diagnostic_schema_version="3",
    matrix_job_id=SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID,
):
    row = {column: "1" for column in CHARGER_DIAGNOSTIC_COLUMNS}
    row.update(
        {
            "matrix_job_id": matrix_job_id,
            "scale": scale,
            "algorithm": algorithm,
            "training_seed": str(training_seed),
            "episode_index": str(episode_index),
            "episode_seed": str(710000 + training_seed * 1000 + episode_index),
            "transformer_id": str(transformer_id),
            "charger_id": str(charger_id),
            "user_satisfaction_source": "ev2gym",
            "diagnostic_schema_version": diagnostic_schema_version,
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
    duplicate_episode_key=False,
    missing_episode_column=False,
    non_finite_metric=False,
    wrong_schema=False,
    mismatched_matrix_job_id=False,
    missing_hierarchical_episode=False,
):
    members = []
    transformer_count, charger_count = TOPOLOGY[scale]
    for training_seed in TRAINING_SEEDS:
        schema_version = "2" if wrong_schema and training_seed == 0 else "3"
        matrix_job_id = (
            "wrong"
            if mismatched_matrix_job_id and training_seed == 0
            else SYNTHETIC_DIAGNOSTIC_ARRAY_JOB_ID
        )
        seed_prefix = f"seed{training_seed}/diagnostics"
        episode_rows = [
            synthetic_episode_row(
                scale,
                algorithm,
                training_seed,
                episode_index,
                diagnostic_schema_version=schema_version,
                matrix_job_id=matrix_job_id,
                non_finite_metric=non_finite_metric
                and training_seed == 0
                and episode_index == 0,
                mismatched_episode_seed=missing_hierarchical_episode
                and algorithm == "hierarchical"
                and training_seed == 0
                and episode_index == 0,
            )
            for episode_index in range(episodes_per_seed)
        ]
        if duplicate_episode_key and training_seed == 0:
            episode_rows.append(dict(episode_rows[0]))
        transformer_rows = [
            synthetic_transformer_row(
                scale,
                algorithm,
                training_seed,
                episode_index,
                transformer_id,
                diagnostic_schema_version=schema_version,
                matrix_job_id=matrix_job_id,
            )
            for episode_index in range(episodes_per_seed)
            for transformer_id in range(transformer_count)
        ]
        charger_rows = [
            synthetic_charger_row(
                scale,
                algorithm,
                training_seed,
                episode_index,
                charger_id,
                charger_id % transformer_count,
                diagnostic_schema_version=schema_version,
                matrix_job_id=matrix_job_id,
            )
            for episode_index in range(episodes_per_seed)
            for charger_id in range(charger_count)
        ]
        episode_fieldnames = list(EPISODE_DIAGNOSTIC_COLUMNS)
        if missing_episode_column and training_seed == 0:
            episode_fieldnames.remove("tracking_error")
        members.extend(
            [
                (
                    f"{seed_prefix}/episode_diagnostics.csv",
                    csv_bytes(episode_fieldnames, episode_rows),
                ),
                (
                    f"{seed_prefix}/seed_summary_diagnostics.csv",
                    csv_bytes(
                        SEED_SUMMARY_DIAGNOSTIC_COLUMNS,
                        [
                            synthetic_seed_row(
                                scale,
                                algorithm,
                                training_seed,
                                episodes_per_seed,
                                diagnostic_schema_version=schema_version,
                                matrix_job_id=matrix_job_id,
                            )
                        ],
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
    duplicate_episode_key=False,
    missing_episode_column=False,
    non_finite_metric=False,
    wrong_schema=False,
    mismatched_matrix_job_id=False,
    missing_hierarchical_episode=False,
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
                duplicate_episode_key=duplicate_episode_key and task_id == 0,
                missing_episode_column=missing_episode_column and task_id == 0,
                non_finite_metric=non_finite_metric and task_id == 0,
                wrong_schema=wrong_schema and task_id == 0,
                mismatched_matrix_job_id=mismatched_matrix_job_id and task_id == 0,
                missing_hierarchical_episode=(
                    missing_hierarchical_episode and algorithm == "hierarchical"
                ),
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


def read_csv_file(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generated_relative_paths(root_path):
    return sorted(
        path.relative_to(root_path).as_posix()
        for path in root_path.rglob("*")
        if path.is_file()
    )


def assert_generated_names_are_semantic(root_path):
    disallowed = re.compile(
        r"(stage_d|r5l|job\d+|[0-9a-f]{40}|[0-9a-f]{64})",
        re.IGNORECASE,
    )
    for path in root_path.rglob("*"):
        if path.is_file():
            assert disallowed.search(path.name) is None, path


def test_extraction_publishes_semantic_outputs(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)
    output_dir = tmp_path / "ev_charging_infrastructure_control_analysis"

    extraction_module().extract_diagnostic_datasets(
        bundle, output_dir, expected_episodes_per_seed=1
    )

    assert sorted(path.name for path in (output_dir / "datasets").iterdir()) == [
        "charger_metrics.csv",
        "episode_metrics.csv",
        "seed_metrics.csv",
        "transformer_metrics.csv",
    ]
    assert (output_dir / "provenance.json").is_file()
    assert len(read_csv_file(output_dir / "datasets" / "episode_metrics.csv")) == 40
    assert len(read_csv_file(output_dir / "datasets" / "seed_metrics.csv")) == 40
    assert len(read_csv_file(output_dir / "datasets" / "transformer_metrics.csv")) == (
        2 * len(TRAINING_SEEDS) * sum(counts[0] for counts in TOPOLOGY.values())
    )
    assert len(read_csv_file(output_dir / "datasets" / "charger_metrics.csv")) == (
        2 * len(TRAINING_SEEDS) * sum(counts[1] for counts in TOPOLOGY.values())
    )
    assert generated_relative_paths(output_dir) == [
        "datasets/charger_metrics.csv",
        "datasets/episode_metrics.csv",
        "datasets/seed_metrics.csv",
        "datasets/transformer_metrics.csv",
        "provenance.json",
    ]


def test_extraction_rejects_duplicate_episode_key(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, duplicate_episode_key=True)
    with pytest.raises(RuntimeError, match="duplicate episode key"):
        extraction_module().extract_diagnostic_datasets(
            bundle, tmp_path / "analysis-output", expected_episodes_per_seed=1
        )


def test_extraction_rejects_missing_required_episode_column(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, missing_episode_column=True)
    with pytest.raises(RuntimeError, match="missing required column"):
        extraction_module().extract_diagnostic_datasets(
            bundle, tmp_path / "analysis-output", expected_episodes_per_seed=1
        )


def test_extraction_rejects_non_finite_approved_metric(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, non_finite_metric=True)
    with pytest.raises(RuntimeError, match="non-finite approved metric"):
        extraction_module().extract_diagnostic_datasets(
            bundle, tmp_path / "analysis-output", expected_episodes_per_seed=1
        )


def test_extraction_rejects_wrong_schema(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, wrong_schema=True)
    with pytest.raises(RuntimeError, match="diagnostic schema version"):
        extraction_module().extract_diagnostic_datasets(
            bundle, tmp_path / "analysis-output", expected_episodes_per_seed=1
        )


def test_extraction_rejects_mismatched_matrix_job_id(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, mismatched_matrix_job_id=True)
    with pytest.raises(RuntimeError, match="matrix_job_id"):
        extraction_module().extract_diagnostic_datasets(
            bundle, tmp_path / "analysis-output", expected_episodes_per_seed=1
        )


def test_extraction_rejects_mismatched_algorithm_episode_seed_sets(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path, missing_hierarchical_episode=True)
    with pytest.raises(RuntimeError, match="episode seed sets"):
        extraction_module().extract_diagnostic_datasets(
            bundle, tmp_path / "analysis-output", expected_episodes_per_seed=1
        )


def test_extraction_rejects_existing_output_directory(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)
    output_dir = tmp_path / "analysis-output"
    output_dir.mkdir()
    with pytest.raises(RuntimeError, match="output directory already exists"):
        extraction_module().extract_diagnostic_datasets(
            bundle, output_dir, expected_episodes_per_seed=1
        )


def episode_row(scale, algorithm, training_seed, episode_index, **metrics):
    row = {
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": str(training_seed),
        "episode_index": str(episode_index),
        "episode_seed": str(710000 + training_seed * 1000 + episode_index),
    }
    row.update({name: str(value) for name, value in metrics.items()})
    return row


def transformer_row(
    scale,
    algorithm,
    training_seed,
    episode_index,
    transformer_id,
    overload_frequency_fraction,
    overload_magnitude_sum,
    overload_magnitude_max,
):
    return {
        "scale": scale,
        "algorithm": algorithm,
        "training_seed": str(training_seed),
        "episode_index": str(episode_index),
        "episode_seed": str(710000 + training_seed * 1000 + episode_index),
        "transformer_id": str(transformer_id),
        "overload_frequency_fraction": str(overload_frequency_fraction),
        "overload_magnitude_sum": str(overload_magnitude_sum),
        "overload_magnitude_max": str(overload_magnitude_max),
    }


def seed_observation(scale, algorithm, training_seed, metric_name, value):
    return comparison_module().SeedMetricObservation(
        scale=scale,
        algorithm=algorithm,
        training_seed=training_seed,
        metric_name=metric_name,
        value=value,
    )


def test_aggregate_episode_metric_is_averaged_within_seed():
    rows = [
        episode_row("25cp", "actiongnn", 0, 0, episode_reward=2.0),
        episode_row("25cp", "actiongnn", 0, 1, episode_reward=4.0),
    ]

    values = comparison_module().aggregate_episode_metric(rows, "episode_reward")

    assert values[("25cp", "actiongnn", 0)] == pytest.approx(3.0)


def test_aggregate_transformer_overload_rules_are_fixed():
    rows = [
        transformer_row("25cp", "actiongnn", 0, 0, 0, 0.1, 2.0, 3.0),
        transformer_row("25cp", "actiongnn", 0, 0, 1, 0.3, 5.0, 7.0),
        transformer_row("25cp", "actiongnn", 0, 1, 0, 0.2, 11.0, 13.0),
        transformer_row("25cp", "actiongnn", 0, 1, 1, 0.4, 17.0, 19.0),
    ]

    result = comparison_module().aggregate_transformer_overload_metrics(rows)

    key = ("25cp", "actiongnn", 0)
    assert result[key]["mean_transformer_overload_frequency_fraction"] == pytest.approx(
        0.25
    )
    assert result[key]["mean_total_transformer_overload_magnitude"] == pytest.approx(
        17.5
    )
    assert result[key]["maximum_transformer_overload_magnitude"] == pytest.approx(19.0)


def test_pairing_enforces_exact_five_seed_pairs():
    observations = [
        seed_observation("25cp", algorithm, seed, "episode_reward", 10.0 + seed)
        for algorithm in ALGORITHMS
        for seed in TRAINING_SEEDS
    ]

    pairs = comparison_module().validate_exact_algorithm_pairs(
        observations, "25cp", "episode_reward"
    )

    assert [pair.training_seed for pair in pairs] == [0, 1, 2, 3, 4]
    assert pairs[0].actiongnn_value == pytest.approx(10.0)
    assert pairs[0].hierarchical_value == pytest.approx(10.0)


def test_pairing_rejects_unequal_seed_sets():
    observations = [
        seed_observation("25cp", "actiongnn", seed, "episode_reward", 1.0)
        for seed in TRAINING_SEEDS
    ] + [
        seed_observation("25cp", "hierarchical", seed, "episode_reward", 2.0)
        for seed in (0, 1, 2, 3)
    ]

    with pytest.raises(RuntimeError, match="unequal seed sets"):
        comparison_module().validate_exact_algorithm_pairs(
            observations, "25cp", "episode_reward"
        )


def test_pairing_rejects_duplicate_seed_observation():
    observations = [
        seed_observation("25cp", algorithm, seed, "episode_reward", 1.0)
        for algorithm in ALGORITHMS
        for seed in TRAINING_SEEDS
    ]
    observations.append(seed_observation("25cp", "actiongnn", 0, "episode_reward", 3.0))

    with pytest.raises(RuntimeError, match="duplicate seed observation"):
        comparison_module().validate_exact_algorithm_pairs(
            observations, "25cp", "episode_reward"
        )


def test_pairing_rejects_more_or_fewer_than_five_seeds():
    observations = [
        seed_observation("25cp", algorithm, seed, "episode_reward", 1.0)
        for algorithm in ALGORITHMS
        for seed in (0, 1, 2)
    ]

    with pytest.raises(RuntimeError, match="exactly five paired seeds"):
        comparison_module().validate_exact_algorithm_pairs(
            observations, "25cp", "episode_reward"
        )


def test_paired_statistics_fixed_oracle():
    result = comparison_module().paired_statistics(
        actiongnn_values=[0, 0, 0, 0, 0],
        hierarchical_values=[1, 2, 3, 4, 5],
    )

    assert result.mean_difference == pytest.approx(3.0)
    assert result.sample_standard_deviation == pytest.approx(1.5811388300841898)
    assert result.t_statistic == pytest.approx(4.242640687119285)
    assert result.paired_t_p_value == pytest.approx(0.0132355995636827)
    assert result.confidence_interval_95_low == pytest.approx(1.036756838522439)
    assert result.confidence_interval_95_high == pytest.approx(4.963243161477561)
    assert result.cohens_dz == pytest.approx(1.8973665961010275)
    assert result.relative_difference_status == "undefined_zero_actiongnn_mean"


def test_wilcoxon_and_holm_fixed_oracles():
    statistic, p_value, method, nonzero_count = (
        comparison_module().wilcoxon_signed_rank([1, 2, 3, 4, 5])
    )

    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(0.0625)
    assert method == "exact"
    assert nonzero_count == 5
    assert comparison_module().holm_adjust([0.01, 0.04, 0.03, 0.20]) == pytest.approx(
        [0.04, 0.09, 0.09, 0.20]
    )


def test_paired_statistics_zero_denominator_and_zero_variance_statuses():
    zero_difference = comparison_module().paired_statistics(
        actiongnn_values=[1, 1, 1, 1, 1],
        hierarchical_values=[1, 1, 1, 1, 1],
    )
    nonzero_constant_difference = comparison_module().paired_statistics(
        actiongnn_values=[2, 2, 2, 2, 2],
        hierarchical_values=[3, 3, 3, 3, 3],
    )

    assert zero_difference.cohens_dz == pytest.approx(0.0)
    assert zero_difference.effect_size_status == "all_differences_zero"
    assert nonzero_constant_difference.cohens_dz is None
    assert nonzero_constant_difference.effect_size_status == (
        "undefined_zero_variance_nonzero_mean"
    )
    assert nonzero_constant_difference.relative_difference_percent_of_actiongnn_mean == (
        pytest.approx(50.0)
    )


def test_comparison_row_uses_status_instead_of_infinity_for_zero_variance():
    statistics = comparison_module().paired_statistics(
        actiongnn_values=[2, 2, 2, 2, 2],
        hierarchical_values=[3, 3, 3, 3, 3],
    )

    row = comparison_module().comparison_row(
        "25cp",
        metric_module.metric_definition("episode_reward"),
        statistics,
    )

    assert row["paired_t_statistic"] == ""
    assert row["effect_size_status"] == "undefined_zero_variance_nonzero_mean"
    assert all(value.lower() not in {"inf", "-inf", "nan"} for value in row.values())


def test_wilcoxon_handles_tied_ranks_exactly():
    statistic, p_value, method, nonzero_count = (
        comparison_module().wilcoxon_signed_rank([1, -1, 2, -2, 2])
    )

    assert statistic == pytest.approx(5.5)
    assert p_value == pytest.approx(0.8125)
    assert method == "exact"
    assert nonzero_count == 5


def test_comparison_publishes_semantic_results(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)
    analysis_dir = tmp_path / "ev_charging_infrastructure_control_analysis"
    extraction_module().extract_diagnostic_datasets(
        bundle, analysis_dir, expected_episodes_per_seed=1
    )

    row_counts = comparison_module().compare_control_architectures(
        analysis_dir, expected_episodes_per_seed=1
    )

    comparisons = read_csv_file(
        analysis_dir / "results" / "paired_control_comparisons.csv"
    )
    summary = read_csv_file(analysis_dir / "results" / "scale_level_summary.csv")
    assert row_counts == {
        "paired_control_comparisons": 88,
        "scale_level_summary": 4,
    }
    assert len(comparisons) == 88
    assert len(summary) == 4
    assert {
        (row["scale"], row["metric_name"]) for row in comparisons
    } == {
        (scale, metric_name)
        for scale in SCALES
        for metric_name in EXPECTED_METRICS
    }
    assert {
        row["holm_adjusted_p_value"] != ""
        for row in comparisons
        if row["tier"] == "primary"
    } == {True}
    assert {
        row["holm_adjusted_p_value"] == ""
        for row in comparisons
        if row["tier"] != "primary"
    } == {True}
    assert all(row["n_paired_seeds"] == "5" for row in comparisons)
    assert (analysis_dir / "results" / "metric_interpretation.md").is_file()


def test_comparison_requires_validated_charger_dataset(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)
    analysis_dir = tmp_path / "ev_charging_infrastructure_control_analysis"
    extraction_module().extract_diagnostic_datasets(
        bundle, analysis_dir, expected_episodes_per_seed=1
    )
    (analysis_dir / "datasets" / "charger_metrics.csv").unlink()

    with pytest.raises(RuntimeError, match="charger_metrics.csv"):
        comparison_module().compare_control_architectures(
            analysis_dir,
            expected_episodes_per_seed=1,
        )


def test_extraction_and_comparison_clis_emit_markers_and_semantic_outputs(tmp_path):
    bundle = build_synthetic_complete_bundle(tmp_path)
    analysis_dir = tmp_path / "ev_charging_infrastructure_control_analysis"
    extract_script = (
        PROJECT_ROOT
        / "analysis"
        / "ev_charging_infrastructure_control"
        / "extract_diagnostic_datasets.py"
    )
    compare_script = (
        PROJECT_ROOT
        / "analysis"
        / "ev_charging_infrastructure_control"
        / "compare_control_architectures.py"
    )

    extract_result = subprocess.run(
        [
            sys.executable,
            str(extract_script),
            "--bundle",
            str(bundle),
            "--output-dir",
            str(analysis_dir),
            "--expected-episodes-per-seed",
            "1",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert extract_result.returncode == 0, extract_result.stderr
    assert "EV_CHARGING_INFRASTRUCTURE_DATASET_EXTRACTION_START" in (
        extract_result.stdout
    )
    assert "BUNDLE_VALIDATION=PASS" in extract_result.stdout
    assert "EPISODE_METRICS_ROWS=40" in extract_result.stdout
    assert "OUTPUT_PUBLICATION=PASS" in extract_result.stdout
    assert "EV_CHARGING_INFRASTRUCTURE_DATASET_EXTRACTION_COMPLETED" in (
        extract_result.stdout
    )

    compare_result = subprocess.run(
        [
            sys.executable,
            str(compare_script),
            "--analysis-dir",
            str(analysis_dir),
            "--expected-episodes-per-seed",
            "1",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert compare_result.returncode == 0, compare_result.stderr
    assert "EV_CHARGING_INFRASTRUCTURE_CONTROL_COMPARISON_START" in (
        compare_result.stdout
    )
    assert "PAIRING_VALIDATION=PASS" in compare_result.stdout
    assert "PAIRED_COMPARISON_ROWS=88" in compare_result.stdout
    assert "RESULT_PUBLICATION=PASS" in compare_result.stdout
    assert "EV_CHARGING_INFRASTRUCTURE_CONTROL_COMPARISON_COMPLETED" in (
        compare_result.stdout
    )

    assert generated_relative_paths(analysis_dir) == [
        "datasets/charger_metrics.csv",
        "datasets/episode_metrics.csv",
        "datasets/seed_metrics.csv",
        "datasets/transformer_metrics.csv",
        "provenance.json",
        "results/metric_interpretation.md",
        "results/paired_control_comparisons.csv",
        "results/scale_level_summary.csv",
    ]
    assert_generated_names_are_semantic(analysis_dir)
