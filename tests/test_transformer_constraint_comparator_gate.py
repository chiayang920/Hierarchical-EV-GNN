import csv
import io
import tarfile
from pathlib import Path

import pytest
import yaml


def _gate():
    from scripts import transformer_constraint_comparator_gate as gate

    return gate


def _csv_bytes(fieldnames, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _write_tar(path: Path, members: dict[str, bytes]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _training_log_bytes():
    fields = ["type", "timestep", "eval/mean_reward", "eval/std_reward", "elapsed_seconds"]
    rows = [
        {
            "type": "evaluation",
            "timestep": 5000 * (index + 1),
            "eval/mean_reward": -1000 + index,
            "eval/std_reward": 1.0,
            "elapsed_seconds": index + 1,
        }
        for index in range(15)
    ]
    return _csv_bytes(fields, rows)


def _config(scale):
    return {
        "number_of_charging_stations": int(scale.removesuffix("cp")),
        "simulation_length": 112,
        "v2g_enabled": False,
    }


def make_historical_training_package(root: Path, scale: str, seed: int):
    task_id = {"25cp": 10, "100cp": 30, "500cp": 50, "1000cp": 70}[scale] + seed
    run_name = f"formal75k_{scale}_hierarchical_seed{seed}"
    model_dir = f"train/{run_name}"
    metadata = "\n".join(
        [
            "protocol_version=formal_75k_80cell_v2",
            f"task_id={task_id}",
            f"slurm_array_task_id={task_id}",
            "slurm_array_job_id=58850281",
            f"scale={scale}",
            "algorithm=hierarchical",
            f"seed={seed}",
            "training_steps=75000",
            "evaluation_cadence=5000",
            "evaluation_episodes=5",
            "expected_scheduled_evaluations=15",
            "actor_output_transform=hierarchical_nonnegative_action_contract_v1",
            (
                "checkpoint_selection_rule=model.best selected by strict improvement of scheduled "
                "internal eval mean reward at 5k-step intervals within the configured training "
                "budget; model.last is saved at the configured final step but is not used for "
                "canonical eval30 or diagnostics."
            ),
            "source_identity=ae3911debe04a65f00ca5bb6ec03baed059f86f4",
            "source_bundle_identity=bundle-a",
            f"run_name={run_name}",
            f"model_dir_relative={model_dir}",
            f"config_copy_relative=config/{scale}_hierarchical_seed{seed}_config.yaml",
            "fresh_run=true",
            "training_exit_status=0",
        ]
    ) + "\n"
    run_args = {
        "algorithm": "hierarchical",
        "config": f"config_files/PublicPST_{'25cp' if scale == '25cp' else scale.removesuffix('cp')}.yaml",
        "seed": seed,
        "max_timesteps": 75000,
        "start_timesteps": 1000,
        "eval_freq": 5000,
        "eval_episodes": 5,
        "discrete_actions": 1,
        "batch_size": 64,
        "replay_buffer_size": 100000,
    }
    kwargs = {
        "discrete_actions": 1,
        "discount": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "fx_dim": 32,
        "fx_GNN_hidden_dim": 64,
        "mlp_hidden_dim": 512,
        "lr": 0.0003,
    }
    members = {
        "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
        f"config/{scale}_hierarchical_seed{seed}_config.yaml": yaml.safe_dump(_config(scale)).encode(),
        f"{model_dir}/run_args.yaml": yaml.safe_dump(run_args).encode(),
        f"{model_dir}/kwargs.yaml": yaml.safe_dump(kwargs).encode(),
        f"{model_dir}/config.yaml": yaml.safe_dump(_config(scale)).encode(),
        f"{model_dir}/training_log.csv": _training_log_bytes(),
        f"{model_dir}/model.best_actor": b"actor",
        f"{model_dir}/model.best_actor_optimizer": b"actor-opt",
        f"{model_dir}/model.best_critic": b"critic",
        f"{model_dir}/model.best_critic_optimizer": b"critic-opt",
        "stderr.log": b"",
    }
    package = (
        root
        / "training_packages"
        / f"m3_formal75k_80cell_{scale}_hierarchical_seed{seed}_job58850281_task{task_id}.tar.gz"
    )
    _write_tar(package, members)
    return package


def _eval_csv_bytes(scale: str, seed: int):
    fields = [
        "algorithm",
        "config",
        "seed",
        "episode_seed",
        "checkpoint",
        "episode_index",
        "episode_reward",
        "row_type",
    ]
    base = {"25cp": 710000, "100cp": 720000, "500cp": 730000, "1000cp": 740000}[scale]
    rows = [
        {
            "algorithm": "hierarchical",
            "config": f"config_files/PublicPST_{scale}.yaml",
            "seed": seed,
            "episode_seed": base + 1000 * seed + episode,
            "checkpoint": "/stage/checkpoint/model.best",
            "episode_index": episode,
            "episode_reward": -1000 + episode,
            "row_type": "episode",
        }
        for episode in range(30)
    ]
    rows.append({**rows[-1], "episode_index": "", "episode_seed": "", "row_type": "summary"})
    return _csv_bytes(fields, rows)


def make_historical_eval_package(root: Path, scale: str, seed: int):
    task_id = {"25cp": 10, "100cp": 30, "500cp": 50, "1000cp": 70}[scale] + seed
    metadata = "\n".join(
        [
            "protocol_version=formal_75k_eval30_v2",
            "training_array_job_id=58850281",
            "eval_array_job_id=59023728",
            f"task_id={task_id}",
            f"scale={scale}",
            "algorithm=hierarchical",
            f"seed={seed}",
            "source_identity=ae3911debe04a65f00ca5bb6ec03baed059f86f4",
            "source_bundle_identity=bundle-a",
            "source_training_package_sha256=training-package-sha",
            "checkpoint_role=model.best",
            "checkpoint_identity_sha256=checkpoint-sha",
            "config_identity_sha256=config-sha",
            "evaluation_episodes=30",
            "evaluation_exit_status=0",
        ]
    ) + "\n"
    package = (
        root
        / "raw_eval30_packages"
        / f"m3_formal75k_eval30_{scale}_hierarchical_seed{seed}_trainjob58850281_job59023728_task{task_id}.tar.gz"
    )
    _write_tar(
        package,
        {
            "runtime_metadata/task_runtime_metadata.env": metadata.encode(),
            f"eval/{scale}_hierarchical_seed{seed}_eval30.csv": _eval_csv_bytes(scale, seed),
            "stderr.log": b"",
        },
    )
    return package


def make_historical_diagnostic_package(root: Path, scale: str, seed: int):
    task_id = {"25cp": 10, "100cp": 30, "500cp": 50, "1000cp": 70}[scale] + seed
    metadata = "\n".join(
        [
            "protocol_version=formal_75k_diagnostic_v2",
            "training_array_job_id=58850281",
            "diagnostic_array_job_id=59023729",
            f"task_id={task_id}",
            f"scale={scale}",
            "algorithm=hierarchical",
            f"seed={seed}",
            "source_identity=ae3911debe04a65f00ca5bb6ec03baed059f86f4",
            "source_bundle_identity=bundle-a",
            "source_training_package_sha256=training-package-sha",
            "checkpoint_role=model.best",
            "checkpoint_identity_sha256=checkpoint-sha",
            "config_identity_sha256=config-sha",
            "diagnostic_schema_version=3",
            "reconciliation_contract_version=2",
            "evaluation_episodes=30",
            "evaluation_exit_status=0",
        ]
    ) + "\n"
    package = (
        root
        / "raw_diagnostic_packages"
        / f"m3_formal75k_diagnostics_{scale}_hierarchical_seed{seed}_trainjob58850281_job59023729_task{task_id}.tar.gz"
    )
    _write_tar(package, {"runtime_metadata/task_runtime_metadata.env": metadata.encode(), "stderr.log": b""})
    return package


def make_complete_historical_root(root: Path):
    for scale in ("25cp", "100cp", "500cp", "1000cp"):
        scale_root = root / ("subset_25_100" if scale in {"25cp", "100cp"} else f"scale_{scale}")
        for seed in range(10):
            make_historical_training_package(scale_root, scale, seed)
            make_historical_eval_package(scale_root, scale, seed)
            make_historical_diagnostic_package(scale_root, scale, seed)


def test_comparator_reuse_gate_accepts_complete_hierarchy_fixture(tmp_path):
    gate = _gate()
    make_complete_historical_root(tmp_path)

    result = gate.run_comparator_reuse_gate(
        repo_root=Path(__file__).resolve().parents[1],
        historical_root=tmp_path,
        current_ref="HEAD",
        historical_ref="HEAD",
    )

    assert result.status == "PASS"
    assert result.training_package_count == 40
    assert result.eval30_package_count == 40
    assert result.diagnostic_package_count == 40
    assert result.checks["training_contract"] == "PASS"
    assert result.checks["eval30_contract"] == "PASS"
    assert result.checks["diagnostic_contract"] == "PASS"


def test_comparator_reuse_gate_fails_closed_when_evidence_is_missing(tmp_path):
    gate = _gate()
    make_complete_historical_root(tmp_path)
    missing = next(tmp_path.glob("**/m3_formal75k_eval30_1000cp_hierarchical_seed9_*.tar.gz"))
    missing.unlink()

    result = gate.run_comparator_reuse_gate(
        repo_root=Path(__file__).resolve().parents[1],
        historical_root=tmp_path,
        current_ref="HEAD",
        historical_ref="HEAD",
    )

    assert result.status == "FAIL"
    assert result.checks["eval30_contract"] == "FAIL"
    assert any("INSUFFICIENT_EVIDENCE" in reason for reason in result.reasons)


def test_comparator_reuse_gate_can_validate_authoritative_local_evidence():
    gate = _gate()
    if not gate.DEFAULT_HISTORICAL_ROOT.exists():
        pytest.skip("authoritative Formal75K evidence root is not available on this machine")

    result = gate.run_comparator_reuse_gate()

    assert result.status == "PASS"
    assert result.training_package_count == 40
    assert result.eval30_package_count == 40
