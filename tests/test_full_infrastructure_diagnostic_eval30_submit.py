import os
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = (
    PROJECT_ROOT / "m3_jobs" / "submit_full_infrastructure_diagnostic_eval30_workflow.sh"
)
SOURCE_SHA = "f" * 40


def write_executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def build_fake_submit_environment(tmp_path, *, source_sha=SOURCE_SHA):
    repo = tmp_path / "repo"
    (repo / "m3_jobs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "SOURCE_COMMIT_SHA.txt").write_text(source_sha + "\n", encoding="utf-8")
    for relative in (
        "m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm",
        "m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm",
        "scripts/validate_full_infrastructure_diagnostic_eval30.py",
        "scripts/full_infrastructure_diagnostic_eval30_packages.py",
        "scripts/validate_infrastructure_diagnostic_smoke.py",
    ):
        path = repo / relative
        path.write_text("placeholder\n", encoding="utf-8")

    formal = tmp_path / "formal"
    formal.mkdir()
    complete_bundle = formal / "controlled_multiscale_formal_complete_evidence_job58513929.tar.gz"
    complete_bundle.write_bytes(b"placeholder")
    output = tmp_path / "output"
    output.mkdir()
    run_root = tmp_path / "run"

    bin_dir = tmp_path / "bin"
    sbatch_marker = tmp_path / "sbatch_called.txt"
    fake_python = r'''#!/bin/bash
set -euo pipefail
if [[ "${1:-}" == *"validate_full_infrastructure_diagnostic_eval30.py" ]] && [[ "${2:-}" == "task-mapping" ]]; then
  task_id="${4}"
  case "${task_id}" in
    0) scale=25cp; algorithm=actiongnn; start=0; offset=710000 ;;
    1) scale=25cp; algorithm=hierarchical; start=5; offset=710000 ;;
    2) scale=100cp; algorithm=actiongnn; start=10; offset=720000 ;;
    3) scale=100cp; algorithm=hierarchical; start=15; offset=720000 ;;
    4) scale=500cp; algorithm=actiongnn; start=20; offset=730000 ;;
    5) scale=500cp; algorithm=hierarchical; start=25; offset=730000 ;;
    6) scale=1000cp; algorithm=actiongnn; start=30; offset=740000 ;;
    7) scale=1000cp; algorithm=hierarchical; start=35; offset=740000 ;;
  esac
  echo "task_id=${task_id}"
  echo "scale=${scale}"
  echo "algorithm=${algorithm}"
  echo "training_seeds=0,1,2,3,4"
  echo "formal_task_ids=${start},$((start+1)),$((start+2)),$((start+3)),$((start+4))"
  echo "eval_episodes=30"
  echo "eval_seed_offset=${offset}"
  exit 0
fi
if [[ "${1:-}" == "-" ]]; then
  cat >/dev/null
  echo "FORMAL_PREFLIGHT_OK"
  exit 0
fi
echo "unexpected fake python invocation: $*" >&2
exit 97
'''
    write_executable(bin_dir / "python", fake_python)
    write_executable(
        bin_dir / "sbatch",
        f'''#!/bin/bash
echo "$*" >> "{sbatch_marker}"
echo "${{FAKE_SBATCH_RESULT:-12345}}"
''',
    )
    return {
        "repo": repo,
        "formal": formal,
        "complete_bundle": complete_bundle,
        "output": output,
        "run_root": run_root,
        "bin_dir": bin_dir,
        "sbatch_marker": sbatch_marker,
    }


def submit_env(layout, *, dry_run=True, expected_sha=SOURCE_SHA):
    env = os.environ.copy()
    env.update(
        {
            "PATH": str(layout["bin_dir"]) + os.pathsep + env.get("PATH", ""),
            "EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT": str(layout["repo"]),
            "EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT": expected_sha,
            "EV_GNN_FULL_DIAGNOSTIC_FORMAL_JOB_ID": "58513929",
            "EV_GNN_FULL_DIAGNOSTIC_FORMAL_PACKAGE_ROOT": str(layout["formal"]),
            "EV_GNN_FULL_DIAGNOSTIC_FORMAL_COMPLETE_BUNDLE": str(
                layout["complete_bundle"]
            ),
            "EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT": str(layout["output"]),
            "EV_GNN_FULL_DIAGNOSTIC_RUN_ROOT": str(layout["run_root"]),
        }
    )
    if dry_run:
        env["EV_GNN_FULL_DIAGNOSTIC_SUBMIT_DRY_RUN"] = "1"
    return env


def test_submit_helper_static_contract():
    text = SUBMIT_SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "DRY_RUN_NO_SBATCH_CALLED" in text
    assert "FULL_INFRASTRUCTURE_DIAGNOSTIC_WORKFLOW_SUBMITTED" in text
    assert "--dependency=afterok:${ARRAY_JOB_ID}" in text
    assert "--array=0-7" in (
        PROJECT_ROOT / "m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm"
    ).read_text(encoding="utf-8")
    assert "squeue -j ${ARRAY_JOB_ID},${REDUCER_JOB_ID}" in text
    assert "--format=JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU" in text
    prohibited = ("latest", "squeue -u", "sacct -u", "--name")
    for token in prohibited:
        assert token not in text


def test_submit_helper_dry_run_never_calls_sbatch(tmp_path):
    layout = build_fake_submit_environment(tmp_path)
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=submit_env(layout),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "FORMAL_PREFLIGHT_OK" in result.stdout
    assert result.stdout.count("task_id=") >= 8
    assert "DRY_RUN_NO_SBATCH_CALLED" in result.stdout
    assert not layout["sbatch_marker"].exists()


def test_submit_helper_rejects_source_commit_mismatch_before_sbatch(tmp_path):
    layout = build_fake_submit_environment(tmp_path, source_sha="e" * 40)
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=submit_env(layout),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "source commit mismatch" in result.stderr
    assert not layout["sbatch_marker"].exists()


def test_submit_helper_rejects_missing_complete_formal_bundle(tmp_path):
    layout = build_fake_submit_environment(tmp_path)
    layout["complete_bundle"].unlink()
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=submit_env(layout),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "required non-empty regular file" in result.stderr
    assert not layout["sbatch_marker"].exists()


def test_submit_helper_rejects_non_numeric_array_submission_result(tmp_path):
    layout = build_fake_submit_environment(tmp_path)
    env = submit_env(layout, dry_run=False)
    env["FAKE_SBATCH_RESULT"] = "not-a-job-id"
    result = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode != 0
    assert "numeric array job ID" in result.stderr
    assert layout["sbatch_marker"].exists()
    assert len(layout["sbatch_marker"].read_text().splitlines()) == 1
