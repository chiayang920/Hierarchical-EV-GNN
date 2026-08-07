import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LIB = PROJECT_ROOT / "m3_jobs" / "lib_formal_75k_runtime.sh"
SUBMIT = PROJECT_ROOT / "m3_jobs" / "submit_formal_75k_80cell_workflow.sh"
SCRIPTS = [
    PROJECT_ROOT / "m3_jobs" / "23_formal_75k_80cell_smoke.slurm",
    PROJECT_ROOT / "m3_jobs" / "24_formal_75k_80cell_train.slurm",
    PROJECT_ROOT / "m3_jobs" / "25_formal_75k_80cell_eval30.slurm",
    PROJECT_ROOT / "m3_jobs" / "26_formal_75k_80cell_diagnostics.slurm",
    SUBMIT,
    PROJECT_ROOT / "m3_jobs" / "create_formal_75k_80cell_source_bundle.sh",
]


def _fake_python(path: Path, version: str, *, forward=False):
    body = ["#!/bin/bash", "set -euo pipefail"]
    body.append('if [[ "${1:-}" == "-I" && "${2:-}" == "-c" ]]; then')
    body.append(f"  printf '%s\\n' '{version}'")
    body.append("  exit 0")
    body.append("fi")
    if forward:
        body.append(f'exec "{sys.executable}" "$@"')
    else:
        body.append(f"printf '%s\\n' '{version}'")
    path.write_text("\n".join(body) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _profile(path: Path):
    path.write_text("\n".join([
        "RESOURCE_PROFILE_APPROVED=YES",
        "SMOKE_CPUS_PER_TASK=1", "SMOKE_MEM=4G", "SMOKE_TIME=00:30:00",
        "TRAIN_CPUS_PER_TASK=1", "TRAIN_MEM=8G", "TRAIN_TIME=08:00:00",
        "EVAL_CPUS_PER_TASK=1", "EVAL_MEM=4G", "EVAL_TIME=02:00:00",
        "DIAGNOSTIC_CPUS_PER_TASK=1", "DIAGNOSTIC_MEM=8G", "DIAGNOSTIC_TIME=04:00:00",
        "",
    ]))


def test_runtime_bootstrap_rejects_missing_and_wrong_python(tmp_path):
    missing = subprocess.run(
        ["bash", "-c", f'source "{RUNTIME_LIB}"; unset EV_GNN_FORMAL_75K_PYTHON; formal75k_require_python311'],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert missing.returncode != 0
    assert "EV_GNN_FORMAL_75K_PYTHON" in missing.stderr

    wrong = tmp_path / "python-wrong"
    _fake_python(wrong, "3.10.14")
    result = subprocess.run(
        ["bash", "-c", f'source "{RUNTIME_LIB}"; EV_GNN_FORMAL_75K_PYTHON="{wrong}"; export EV_GNN_FORMAL_75K_PYTHON; formal75k_require_python311'],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode != 0
    assert "Python 3.11" in result.stderr


def test_hostile_path_dry_run_uses_explicit_verified_python_and_never_sbatch(tmp_path):
    fake = tmp_path / "python311"
    _fake_python(fake, "3.11.15", forward=True)
    profile = tmp_path / "profile.env"
    _profile(profile)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("dirname", "basename"):
        target = Path("/usr/bin") / command
        if target.exists():
            (bin_dir / command).symlink_to(target)
    env = {
        "PATH": f"{bin_dir}:/bin:/usr/bin",
        "EV_GNN_FORMAL_75K_PYTHON": str(fake),
    }
    result = subprocess.run(
        ["bash", str(SUBMIT), "--dry-run", "--resource-profile", str(profile)],
        cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
    assert "PYTHON_VERSION=3.11.15" in result.stdout
    assert "DRY_RUN_NO_JOBS_SUBMITTED" in result.stdout
    assert "sbatch" not in result.stderr.lower()


def test_shell_contracts_remove_bare_python_fallback_and_static_smoke_resources():
    for path in SCRIPTS:
        text = path.read_text()
        assert ':-python}' not in text
    smoke = SCRIPTS[0].read_text()
    assert "#SBATCH --cpus-per-task" not in smoke
    assert "#SBATCH --mem=" not in smoke
    assert "#SBATCH --time=" not in smoke
    diagnostic = SCRIPTS[3].read_text()
    assert '--scale "${scale}"' in diagnostic
    source_bundle = SCRIPTS[5].read_text()
    assert "source_file_checksums.sha256" not in source_bundle
    for launcher in SCRIPTS[:4]:
        text = launcher.read_text()
        assert ' -C "${TASK_ROOT}" .' not in text
    submit = SUBMIT.read_text()
    assert "--source-bundle-identity" in submit
    assert "EV_GNN_FORMAL_75K_SOURCE_IDENTITY" in submit
    assert "EV_GNN_FORMAL_75K_SOURCE_BUNDLE_IDENTITY" in submit


def test_stage_launchers_record_complete_protocol_and_training_package_provenance():
    smoke = SCRIPTS[0].read_text()
    train = SCRIPTS[1].read_text()
    eval30 = SCRIPTS[2].read_text()
    diagnostic = SCRIPTS[3].read_text()

    for text, protocol in (
        (smoke, "protocol_version=formal_75k_smoke_v2"),
        (train, "protocol_version=formal_75k_80cell_v2"),
        (eval30, "protocol_version=formal_75k_eval30_v2"),
        (diagnostic, "protocol_version=formal_75k_diagnostic_v2"),
    ):
        assert protocol in text
        assert "slurm_array_task_id=${TASK_ID}" in text

    for text in (eval30, diagnostic):
        assert 'source_bundle_identity=${SOURCE_BUNDLE_IDENTITY}' in text
        assert 'source_training_package_sha256=${TRAINING_PACKAGE_SHA}' in text
        assert "evaluation_episodes=30" in text

def test_smoke_package_creation_disables_macos_appledouble_metadata():
    smoke = SCRIPTS[0].read_text()
    assert 'COPYFILE_DISABLE=1 tar -czf "${PACKAGE_PATH}"' in smoke
