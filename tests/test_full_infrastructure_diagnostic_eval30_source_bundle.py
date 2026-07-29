import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    PROJECT_ROOT
    / "m3_jobs"
    / "create_full_infrastructure_diagnostic_eval30_source_bundle.sh"
)

EXPECTED_RUNTIME_PATHS = {
    "evaluate_td3_gnn.py",
    "evaluate_td3_gnn_infrastructure_diagnostics.py",
    "TD3/TD3_ActionGNN_Controlled.py",
    "TD3/TD3_HierarchicalActionGNN.py",
    "config_files/PublicPST_25cp.yaml",
    "config_files/PublicPST_100.yaml",
    "config_files/PublicPST_500.yaml",
    "config_files/PublicPST_1000.yaml",
    "m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm",
    "m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm",
    "m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh",
    "scripts/validate_infrastructure_diagnostic_smoke.py",
    "scripts/validate_full_infrastructure_diagnostic_eval30.py",
    "scripts/full_infrastructure_diagnostic_eval30_packages.py",
    "scripts/full_infrastructure_diagnostic_eval30_accounting.py",
    "utils/ev2gym_training_utils.py",
    "utils/infrastructure_diagnostics.py",
    "utils/state_public_pst_gnn.py",
}


def run(command, cwd, *, env=None, check=True):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed: {command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def initialise_fixture_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in EXPECTED_RUNTIME_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")
    script_destination = (
        repo / "m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh"
    )
    script_destination.write_text(
        SOURCE_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    script_destination.chmod(0o755)
    run(["git", "init", "-b", "main"], repo)
    run(["git", "config", "user.email", "test@example.com"], repo)
    run(["git", "config", "user.name", "Test User"], repo)
    run(["git", "add", "."], repo)
    run(["git", "commit", "-m", "fixture"], repo)
    return repo, script_destination


def source_env(repo, output):
    head = run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    env = os.environ.copy()
    env.update(
        {
            "EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXPECTED_HEAD_SHA": head,
            "EV_GNN_FULL_DIAGNOSTIC_SOURCE_OUTPUT_ROOT": str(output),
        }
    )
    return head, env


def test_source_bundle_static_allowlist_and_prohibitions():
    text = SOURCE_SCRIPT.read_text(encoding="utf-8")
    for relative in EXPECTED_RUNTIME_PATHS:
        assert f"  {relative}" in text
    for prohibited in (
        ".git",
        "saved_models",
        "task_packages",
        "__pycache__",
        ".pytest_cache",
        "model.best",
        "model.last",
    ):
        assert prohibited in text
    assert "SOURCE_BUNDLE_OK" in text
    assert "SOURCE_COMMIT_SHA.txt" in text
    assert "EV-GNN-full-infrastructure-diagnostics-eval30-" in text


def test_source_bundle_creates_exact_safe_archive(tmp_path):
    repo, script = initialise_fixture_repo(tmp_path)
    output = tmp_path / "output"
    head, env = source_env(repo, output)
    result = run(["bash", str(script)], repo, env=env)
    assert "SOURCE_BUNDLE_OK" in result.stdout
    archive = output / f"EV-GNN-full-infrastructure-diagnostics-eval30-{head}.tar.gz"
    sidecar = Path(str(archive) + ".sha256")
    assert archive.is_file() and sidecar.is_file()
    recorded_digest, recorded_name = sidecar.read_text().split()
    assert recorded_name == archive.name
    assert len(recorded_digest) == 64

    root = f"EV-GNN-full-infrastructure-diagnostics-eval30-{head}"
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        names = [member.name for member in members if member.isfile()]
        assert len(names) == len(set(names))
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert not member.issym()
            assert not member.islnk()
            assert not member.isdev()
        expected_files = {f"{root}/{relative}" for relative in EXPECTED_RUNTIME_PATHS}
        expected_files.add(f"{root}/SOURCE_COMMIT_SHA.txt")
        assert set(names) == expected_files
        source_member = tar.extractfile(f"{root}/SOURCE_COMMIT_SHA.txt")
        assert source_member is not None
        assert source_member.read().decode().strip() == head


def test_source_bundle_rejects_dirty_worktree(tmp_path):
    repo, script = initialise_fixture_repo(tmp_path)
    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    _, env = source_env(repo, tmp_path / "output")
    result = run(["bash", str(script)], repo, env=env, check=False)
    assert result.returncode != 0
    assert "worktree is dirty" in result.stderr


def test_source_bundle_rejects_missing_allowlisted_path(tmp_path):
    repo, script = initialise_fixture_repo(tmp_path)
    (repo / "utils/state_public_pst_gnn.py").unlink()
    _, env = source_env(repo, tmp_path / "output")
    result = run(["bash", str(script)], repo, env=env, check=False)
    assert result.returncode != 0
    assert "allowlisted source path is missing" in result.stderr


def test_source_bundle_rejects_allowlisted_symlink(tmp_path):
    repo, script = initialise_fixture_repo(tmp_path)
    target = repo / "utils/state_public_pst_gnn.py"
    target.unlink()
    target.symlink_to(repo / "utils/infrastructure_diagnostics.py")
    _, env = source_env(repo, tmp_path / "output")
    result = run(["bash", str(script)], repo, env=env, check=False)
    assert result.returncode != 0
    assert "must not be a symbolic link" in result.stderr


def test_source_bundle_rejects_wrong_expected_head(tmp_path):
    repo, script = initialise_fixture_repo(tmp_path)
    _, env = source_env(repo, tmp_path / "output")
    env["EV_GNN_FULL_DIAGNOSTIC_SOURCE_EXPECTED_HEAD_SHA"] = "f" * 40
    result = run(["bash", str(script)], repo, env=env, check=False)
    assert result.returncode != 0
    assert "does not match required" in result.stderr


def test_source_bundle_dry_run_does_not_create_archive(tmp_path):
    repo, script = initialise_fixture_repo(tmp_path)
    output = tmp_path / "output"
    _, env = source_env(repo, output)
    env["EV_GNN_FULL_DIAGNOSTIC_SOURCE_DRY_RUN"] = "1"
    result = run(["bash", str(script)], repo, env=env)
    assert "DRY_RUN_NO_ARCHIVE_CREATED" in result.stdout
    assert not output.exists()
