import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import formal_75k_80cell_workflow as workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPT = PROJECT_ROOT / "scripts" / "formal_75k_80cell_workflow.py"


def _positive_effect_rows():
    return [
        {
            "scale": scale,
            "mean_benefit": 1.0,
            "ci_low": 0.1,
            "holm_adjusted_pvalue": 0.01,
            "seeds_favouring_hierarchy": 8,
            "leave_one_seed_out_direction_stable": True,
            "statistically_supported_harm": False,
        }
        for scale in workflow.SCALES
    ]


def _positive_boundary_rows():
    return [
        {"scale": scale, "mean_benefit": 1.0, "holm_adjusted_pvalue": 0.01}
        for scale in workflow.SCALES
    ]


def test_cli_runs_from_clean_unrelated_working_directory(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(WORKFLOW_SCRIPT), "--print-matrix"],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 80


def test_new_formal75k_modules_import_from_clean_unrelated_working_directory(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    scripts = [
        PROJECT_ROOT / "scripts" / "formal_75k_80cell_artifacts.py",
        PROJECT_ROOT / "scripts" / "formal_75k_80cell_statistics.py",
        PROJECT_ROOT / "scripts" / "formal_75k_80cell_diagnostic_validation.py",
        PROJECT_ROOT / "scripts" / "formal_75k_80cell_historical_evidence.py",
    ]
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr}"


def test_diagnostic_command_propagates_explicit_scale():
    command = workflow.diagnostic_command(
        workflow.resolve_formal_cell(20),
        checkpoint_prefix="/stage/model.best",
        output_dir="/stage/diagnostics",
        run_name="diag",
        max_episode_steps=112,
        eval_seed_offset=720000,
        matrix_job_id="12345",
    )

    scale_index = command.index("--scale")
    assert command[scale_index + 1] == "100cp"


def test_holm_step_down_handles_unsorted_and_tied_pvalues():
    assert workflow.holm_adjust([0.2, 0.001, 0.03, 0.02]) == pytest.approx(
        [0.2, 0.004, 0.06, 0.06]
    )
    assert workflow.holm_adjust([0.01, 0.01, 0.5, 0.2]) == pytest.approx(
        [0.04, 0.04, 0.5, 0.4]
    )


def test_exact_sign_flip_randomisation_uses_paired_magnitudes():
    assert workflow.exact_sign_flip_pvalue([-1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)


def test_unknown_service_status_blocks_strong_architecture_claim():
    claims = workflow.assess_claims(
        reward_effects=_positive_effect_rows(),
        boundary_effects=_positive_boundary_rows(),
        service_guardrails={"status": "UNKNOWN"},
    )

    assert claims["SERVICE_GUARDRAIL_STATUS"] == "UNKNOWN"
    assert claims["STRONG_CROSS_SCALE_ARCHITECTURE_CLAIM_SUPPORTED"] == "NO"


def test_resource_profile_requires_approved_smoke_and_formal_values(tmp_path):
    profile = tmp_path / "resource.env"
    profile.write_text(
        "\n".join(
            [
                "RESOURCE_PROFILE_APPROVED=YES",
                "TRAIN_CPUS_PER_TASK=1",
                "TRAIN_MEM=8G",
                "TRAIN_TIME=01:00:00",
                "EVAL_CPUS_PER_TASK=1",
                "EVAL_MEM=4G",
                "EVAL_TIME=00:30:00",
                "DIAGNOSTIC_CPUS_PER_TASK=1",
                "DIAGNOSTIC_MEM=4G",
                "DIAGNOSTIC_TIME=00:30:00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SMOKE_CPUS_PER_TASK"):
        workflow.load_resource_profile(profile, scope="all")
