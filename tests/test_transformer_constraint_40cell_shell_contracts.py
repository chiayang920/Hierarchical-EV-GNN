import os
import stat
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LIB = PROJECT_ROOT / "m3_jobs" / "lib_transformer_constraint_runtime.sh"
SOURCE_BUNDLE = PROJECT_ROOT / "m3_jobs" / "create_transformer_constraint_40cell_source_bundle.sh"
SUBMIT = PROJECT_ROOT / "m3_jobs" / "submit_transformer_constraint_40cell_workflow.sh"
SMOKE = PROJECT_ROOT / "m3_jobs" / "31_transformer_constraint_40cell_smoke.slurm"
TRAIN = PROJECT_ROOT / "m3_jobs" / "32_transformer_constraint_40cell_train.slurm"
EVAL30 = PROJECT_ROOT / "m3_jobs" / "33_transformer_constraint_40cell_eval30.slurm"
DIAGNOSTICS = PROJECT_ROOT / "m3_jobs" / "34_transformer_constraint_40cell_diagnostics.slurm"
LOCAL_DRY = PROJECT_ROOT / "local_tools" / "run_local_transformer_constraint_40cell_dry_run.sh"
LOCAL_SMOKE = PROJECT_ROOT / "local_tools" / "run_local_transformer_constraint_functional_smoke.sh"
LOCAL_RUNTIME = PROJECT_ROOT / "local_tools" / "run_local_transformer_constraint_runtime_gate.sh"


def _fake_python(path: Path):
    body = [
        "#!/bin/bash",
        "set -euo pipefail",
        'if [[ "${1:-}" == "-I" && "${2:-}" == "-c" ]]; then',
        "  printf '%s\\n' '3.11.9'",
        "  exit 0",
        "fi",
        f'exec "{sys.executable}" "$@"',
    ]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_runtime_library_uses_constraint_namespace_and_requires_python311(tmp_path):
    fake = tmp_path / "python311"
    _fake_python(fake)
    missing = subprocess.run(
        ["bash", "-c", f'source "{RUNTIME_LIB}"; unset EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON; transformer_constraint_require_python311'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    present = subprocess.run(
        ["bash", "-c", f'source "{RUNTIME_LIB}"; EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON="{fake}" transformer_constraint_require_python311; echo "${{PYTHON_VERSION}}"'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert missing.returncode != 0
    assert "EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON" in missing.stderr
    assert present.returncode == 0, present.stderr
    assert present.stdout.strip() == "3.11.9"


def test_source_bundle_dry_run_allowlist_contains_constraint_specific_sources(tmp_path):
    fake = tmp_path / "python311"
    _fake_python(fake)
    env = {
        **os.environ,
        "EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON": str(fake),
        "EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_DRY_RUN": "1",
        "EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_EXPECTED_HEAD_SHA": "abc123",
        "EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_OUTPUT_ROOT": str(tmp_path / "bundle"),
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
    assert "TD3/TD3_HierarchicalActionGNN_TransformerConstraint.py" in result.stdout
    assert "utils/transformer_feasibility_projection.py" in result.stdout
    assert "scripts/transformer_constraint_40cell_workflow.py" in result.stdout
    assert "evaluate_transformer_constraint_projection_diagnostics.py" in result.stdout
    assert "DRY_RUN_NO_ARCHIVE_CREATED" in result.stdout
    text = SOURCE_BUNDLE.read_text(encoding="utf-8")
    assert "git archive" in text
    assert "--add-virtual-file" in text
    assert "EVGNN_Research_Artefacts" in text


def test_source_bundle_refuses_current_dirty_worktree_for_formal_bundle(tmp_path):
    fake = tmp_path / "python311"
    _fake_python(fake)
    env = {
        **os.environ,
        "EV_GNN_TRANSFORMER_CONSTRAINT_PYTHON": str(fake),
        "EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_EXPECTED_HEAD_SHA": "5d09d99ec0836188472884440fa1ed207047fc84",
        "EV_GNN_TRANSFORMER_CONSTRAINT_SOURCE_OUTPUT_ROOT": str(tmp_path / "bundle"),
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
    assert "dirty" in result.stderr or "untracked" in result.stderr


def test_shell_workers_are_dedicated_constraint_surfaces():
    for path in [RUNTIME_LIB, SOURCE_BUNDLE, SUBMIT, SMOKE, TRAIN, EVAL30, DIAGNOSTICS]:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert "hierarchical_transformer_ev" not in text
        assert "EV_GNN_TRANSFORMER_EV" not in text

    assert "#SBATCH --array=0-3" in SMOKE.read_text(encoding="utf-8")
    assert "--array=0-39" in SUBMIT.read_text(encoding="utf-8")
    assert "hierarchical_transformer_constraint" in TRAIN.read_text(encoding="utf-8")
    assert "evaluate_td3_gnn_infrastructure_diagnostics.py" in DIAGNOSTICS.read_text(encoding="utf-8")
    assert "evaluate_transformer_constraint_projection_diagnostics.py" in DIAGNOSTICS.read_text(encoding="utf-8")


def test_local_tools_are_non_submission_entrypoints():
    for path in [LOCAL_DRY, LOCAL_SMOKE, LOCAL_RUNTIME]:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        assert "sbatch" not in text
        assert "hierarchical_transformer_constraint" in text

