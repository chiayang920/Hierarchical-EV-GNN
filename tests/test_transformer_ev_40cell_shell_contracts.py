import os
import stat
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LIB = PROJECT_ROOT / "m3_jobs" / "lib_transformer_ev_runtime.sh"
SOURCE_BUNDLE = PROJECT_ROOT / "m3_jobs" / "create_transformer_ev_40cell_source_bundle.sh"
SUBMIT = PROJECT_ROOT / "m3_jobs" / "submit_transformer_ev_40cell_workflow.sh"
SMOKE = PROJECT_ROOT / "m3_jobs" / "27_transformer_ev_40cell_smoke.slurm"
TRAIN = PROJECT_ROOT / "m3_jobs" / "28_transformer_ev_40cell_train.slurm"
EVAL30 = PROJECT_ROOT / "m3_jobs" / "29_transformer_ev_40cell_eval30.slurm"
DIAGNOSTICS = PROJECT_ROOT / "m3_jobs" / "30_transformer_ev_40cell_diagnostics.slurm"
SHELL_FILES = [RUNTIME_LIB, SOURCE_BUNDLE, SUBMIT, SMOKE, TRAIN, EVAL30, DIAGNOSTICS]


def _fake_python(path: Path, version="3.11.15", *, forward=False):
    body = ["#!/bin/bash", "set -euo pipefail"]
    body.append('if [[ "${1:-}" == "-I" && "${2:-}" == "-c" ]]; then')
    body.append(f"  printf '%s\\n' '{version}'")
    body.append("  exit 0")
    body.append("fi")
    if forward:
        body.append(f'exec "{sys.executable}" "$@"')
    else:
        body.append(f"printf '%s\\n' '{version}'")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_runtime_library_uses_transformer_ev_namespace_and_rejects_bare_python(tmp_path):
    missing = subprocess.run(
        ["bash", "-c", f'source "{RUNTIME_LIB}"; unset EV_GNN_TRANSFORMER_EV_PYTHON; transformer_ev_require_python311'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert missing.returncode != 0
    assert "EV_GNN_TRANSFORMER_EV_PYTHON" in missing.stderr

    text = RUNTIME_LIB.read_text(encoding="utf-8")
    assert "EV_GNN_FORMAL_75K" not in text
    assert ":-python}" not in text


def test_runtime_source_identity_is_enforced_from_deployed_source_file(tmp_path):
    deployed_root = tmp_path / "deployed"
    deployed_root.mkdir()
    (deployed_root / "SOURCE_COMMIT_SHA.txt").write_text("source-sha\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{RUNTIME_LIB}"; transformer_ev_require_source_identity "{deployed_root}" source-sha',
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr

    mismatch = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{RUNTIME_LIB}"; transformer_ev_require_source_identity "{deployed_root}" other-sha',
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert mismatch.returncode != 0
    assert "source identity" in mismatch.stderr


def test_source_bundle_contract_requires_explicit_sha_and_uses_git_archive(tmp_path):
    fake = tmp_path / "python311"
    _fake_python(fake, forward=True)
    env = {
        **os.environ,
        "EV_GNN_TRANSFORMER_EV_PYTHON": str(fake),
        "EV_GNN_TRANSFORMER_EV_SOURCE_DRY_RUN": "1",
        "EV_GNN_TRANSFORMER_EV_SOURCE_EXPECTED_HEAD_SHA": "abc123",
        "EV_GNN_TRANSFORMER_EV_SOURCE_OUTPUT_ROOT": str(tmp_path / "bundle"),
    }
    result = subprocess.run(
        ["bash", str(SOURCE_BUNDLE)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "recorded_head_sha=abc123" in result.stdout
    assert "TD3/TD3_TransformerEVActionGNN.py" in result.stdout
    assert "scripts/transformer_ev_40cell_workflow.py" in result.stdout
    assert "m3_jobs/27_transformer_ev_40cell_smoke.slurm" in result.stdout
    assert "required_branch=" not in result.stdout

    text = SOURCE_BUNDLE.read_text(encoding="utf-8")
    assert "git archive" in text
    assert "exp/formal-75k-nonnegative-comparison-v1" not in text
    assert ".superpowers" in text
    assert "EVGNN_Research_Artefacts" in text
    assert "evidence" in text
    assert "outputs" in text


def test_source_bundle_wrong_expected_sha_fails_closed(tmp_path):
    fake = tmp_path / "python311"
    _fake_python(fake, forward=True)
    env = {
        **os.environ,
        "EV_GNN_TRANSFORMER_EV_PYTHON": str(fake),
        "EV_GNN_TRANSFORMER_EV_SOURCE_EXPECTED_HEAD_SHA": "0000000000000000000000000000000000000000",
        "EV_GNN_TRANSFORMER_EV_SOURCE_OUTPUT_ROOT": str(tmp_path / "bundle"),
    }
    result = subprocess.run(
        ["bash", str(SOURCE_BUNDLE)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "HEAD does not match" in result.stderr or "worktree" in result.stderr


def test_shell_workers_are_dedicated_40cell_transformer_ev_surfaces():
    for path in SHELL_FILES:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert "formal_75k_80cell_workflow.py" not in text
        assert "EV_GNN_FORMAL_75K" not in text

    assert "#SBATCH --array=0-3" in SMOKE.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-39" not in TRAIN.read_text(encoding="utf-8")
    for launcher in (EVAL30, DIAGNOSTICS):
        assert "#SBATCH --array=0-39" in launcher.read_text(encoding="utf-8")
    train_text = TRAIN.read_text(encoding="utf-8")
    assert "--algorithm \"${algorithm}\"" in train_text
    assert "hierarchical_transformer_ev" in train_text
    assert "actiongnn_nonnegative" not in train_text
    assert "formal-task-mapping" in train_text
    for launcher in (SMOKE, TRAIN, EVAL30, DIAGNOSTICS):
        assert "transformer_ev_require_source_identity" in launcher.read_text(encoding="utf-8")


def test_diagnostics_worker_uses_new_validation_script():
    text = DIAGNOSTICS.read_text(encoding="utf-8")

    assert "scripts/transformer_ev_40cell_diagnostic_validation.py" in text
    assert "formal_75k_80cell_diagnostic_validation.py" not in text


def test_eval_and_diagnostic_workers_recompute_staged_checkpoint_and_config_identity():
    for launcher in (EVAL30, DIAGNOSTICS):
        text = launcher.read_text(encoding="utf-8")
        assert "checkpoint_identity_sha256 mismatch" in text
        assert "config_identity_sha256 mismatch" in text


def test_submit_helper_dry_run_prints_matrix_commands_and_submits_zero_jobs(tmp_path):
    fake = tmp_path / "python311"
    _fake_python(fake, forward=True)
    env = {**os.environ, "EV_GNN_TRANSFORMER_EV_PYTHON": str(fake)}
    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "FORMAL_MATRIX_COUNT=40" in stdout
    assert "FORMAL_UNIQUE_CELL_COUNT=40" in stdout
    assert "FORMAL_TASK_IDS=0..39" in stdout
    assert "FORMAL_ALGORITHM_SET=hierarchical_transformer_ev" in stdout
    assert "SMOKE_MATRIX_COUNT=4" in stdout
    assert "SMOKE_SCALES=25cp,100cp,500cp,1000cp" in stdout
    assert "DRY_RUN_SBATCH_COUNT=0" in stdout
    assert "DRY_RUN_NO_JOBS_SUBMITTED" in stdout
    for task_id in (0, 9, 10, 19, 20, 29, 30, 39):
        assert f"FORMAL_COMMAND_TASK_{task_id}=" in stdout
    for task_id in range(4):
        assert f"SMOKE_COMMAND_TASK_{task_id}=" in stdout


def test_formal_submission_resource_ranges_preserve_one_logical_matrix():
    text = SUBMIT.read_text(encoding="utf-8")

    assert "--array=0-19" in text
    assert "--array=20-29" in text
    assert "--array=30-39" in text
    assert "--small-job-id" in text
    assert "--cp500-job-id" in text
    assert "--cp1000-job-id" in text
    assert "split training job IDs" in text
    assert "EV_GNN_TRANSFORMER_EV_OUTPUT_ROOT=${PACKAGE_ROOT}" in text
    assert "TRAIN_SMALL_" in text
    assert "TRAIN_500CP_" in text
    assert "TRAIN_1000CP_" in text
    assert "submit-eval" in text
    assert "submit-diagnostics" in text
