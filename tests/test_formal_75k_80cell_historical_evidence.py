import csv
import io
import json
import tarfile
from pathlib import Path

from scripts import formal_75k_80cell_historical_evidence as evidence


def _csv_bytes(fieldnames, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _tar_bytes(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _write_tar(path: Path, members):
    path.write_bytes(_tar_bytes(members))


def make_historical_formal(path: Path):
    members = {
        "reducer_metadata/reducer_runtime_metadata.env": (
            "array_job_id=58513929\nreducer_job_id=58513930\n"
            "package_count=40\ncanonical_csv_count=40\naggregation_exit_status=0\n"
        ).encode(),
        "aggregation_outputs/controlled_multiscale_formal_summary.md": b"historical formal summary\n",
    }
    task = 0
    for scale in ("25cp", "100cp", "500cp", "1000cp"):
        for algorithm in ("actiongnn", "hierarchical"):
            for seed in range(5):
                rows = []
                for episode in range(30):
                    rows.append({
                        "algorithm": algorithm,
                        "config": f"config_files/PublicPST_{scale}.yaml",
                        "seed": seed,
                        "episode_index": episode,
                        "episode_seed": 710000 + task * 1000 + episode,
                        "episode_reward": -1000 + episode,
                        "row_type": "episode",
                    })
                rows.append({**rows[-1], "episode_index": "", "row_type": "summary"})
                stem = f"{scale}_{algorithm}_seed{seed}"
                members[f"canonical_eval30_csv/{stem}_eval30.csv"] = _csv_bytes(
                    ["algorithm", "config", "seed", "episode_index", "episode_seed", "episode_reward", "row_type"],
                    rows,
                )
                members[f"source_manifests/{stem}_source_manifest.sha256"] = b"abc  source.tar.gz\n"
                members[f"task_metadata/{stem}_eval30_csv_validation.env"] = b"data_rows=31\nepisode_rows=30\nsummary_rows=1\n"
                members[f"task_metadata/{stem}_task_runtime_metadata.env"] = (
                    f"task_id={task}\nslurm_array_job_id=58513929\nscale={scale}\n"
                    f"algorithm={algorithm}\nseed={seed}\ntraining_exit_status=0\nevaluation_exit_status=0\n"
                ).encode()
                task += 1
    _write_tar(path, members)


def _nested_diagnostic_package(scale, algorithm, task_id):
    members = {
        "runtime_metadata/source_commit_sha.txt": b"cbf4b5fe6eb0ede4298140b4717944efbfd0b3ad\n",
        "task_metadata/task.json": json.dumps({"task_id": task_id, "scale": scale, "algorithm": algorithm}).encode(),
        "validation/task_validation.json": json.dumps({"status": "ok", "checkpoint_groups": 5, "episode_count": 150}).encode(),
    }
    for seed in range(5):
        members[f"seed{seed}/diagnostics/episode_diagnostics.csv"] = b"episode_index,diagnostic_schema_version\n0,3\n"
        members[f"seed{seed}/diagnostics/same_pass_canonical_eval30.csv"] = b"episode_index,row_type\n0,episode\n"
        members[f"seed{seed}/validation/mapping_validation.json"] = json.dumps({"status": "ok"}).encode()
        members[f"seed{seed}/validation/same_pass_canonical_reconciliation.csv"] = b"episode_index,status,reconciliation_contract_version\n0,pass,2\n"
        members[f"seed{seed}/validation/service_reconciliation.csv"] = b"episode_index,served_count_reconciliation_status\n0,pass\n"
        members[f"seed{seed}/runtime_metadata/reconciliation_summary.json"] = json.dumps({
            "status": "ok", "hard_gate_status": "pass", "reconciliation_contract_version": 2,
            "mapping_validation_status": "pass", "same_pass_metric_status": "pass",
            "service_reconciliation_status": "pass",
        }).encode()
    return _tar_bytes(members)


def make_historical_diagnostic(path: Path):
    task_rows = []
    checkpoint_rows = []
    episode_rows = []
    members = {
        "validation/complete_workflow_validation.json": json.dumps({
            "array_job_id": "58745233", "formal_job_id": "58513929",
            "task_package_count": 8, "checkpoint_group_count": 40,
            "episode_count": 1200, "schema_version": "3",
            "reconciliation_contract_version": 2, "status": "ok",
        }).encode(),
    }
    task_id = 0
    formal_task = 0
    for scale in ("25cp", "100cp", "500cp", "1000cp"):
        for algorithm in ("actiongnn", "hierarchical"):
            package_name = f"m3_full_infrastructure_diagnostic_eval30_{scale}_{algorithm}_seeds0-4_job58745233_task{task_id}.tar.gz"
            members[f"task_packages/{package_name}"] = _nested_diagnostic_package(scale, algorithm, task_id)
            task_rows.append({"task_id": task_id, "scale": scale, "algorithm": algorithm, "status": "ok", "checkpoint_groups": 5, "episode_count": 150})
            for seed in range(5):
                checkpoint_rows.append({"task_id": task_id, "scale": scale, "algorithm": algorithm, "training_seed": seed, "formal_task_id": formal_task, "diagnostic_schema_version": 3})
                for episode in range(30):
                    episode_rows.append({"scale": scale, "algorithm": algorithm, "training_seed": seed, "episode_index": episode, "episode_seed": 710000 + formal_task * 1000 + episode, "diagnostic_schema_version": 3})
                formal_task += 1
            task_id += 1
    members["summaries/task_inventory.csv"] = _csv_bytes(list(task_rows[0]), task_rows)
    members["summaries/checkpoint_inventory.csv"] = _csv_bytes(list(checkpoint_rows[0]), checkpoint_rows)
    members["summaries/episode_inventory.csv"] = _csv_bytes(list(episode_rows[0]), episode_rows)
    _write_tar(path, members)


def test_historical_evidence_validator_accepts_authoritative_contracts(tmp_path):
    formal = tmp_path / "formal.tar.gz"
    diagnostic = tmp_path / "diagnostic.tar.gz"
    make_historical_formal(formal)
    make_historical_diagnostic(diagnostic)

    result = evidence.validate_historical_evidence(formal, diagnostic)

    assert result.status == "PASS"
    assert result.formal_checkpoint_count == 40
    assert result.formal_episode_count == 1200
    assert result.diagnostic_task_package_count == 8
    assert result.diagnostic_checkpoint_count == 40
    assert result.diagnostic_episode_count == 1200
    assert result.diagnostic_schema_version == 3
    assert result.reconciliation_contract_version == 2
    assert result.historical_stage_d_unchanged is True
