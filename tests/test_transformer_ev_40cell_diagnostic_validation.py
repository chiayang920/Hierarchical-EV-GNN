import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCRIPT = PROJECT_ROOT / "scripts" / "transformer_ev_40cell_diagnostic_validation.py"


def test_validation_uses_transformer_ev_40cell_resolver():
    from scripts import transformer_ev_40cell_diagnostic_validation as validation

    cell = validation.resolve_validation_cell(30)

    assert cell.task_id == 30
    assert cell.scale == "1000cp"
    assert cell.algorithm == "hierarchical_transformer_ev"
    assert cell.seed == 0


def test_validation_cli_help_runs_from_unrelated_working_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(VALIDATION_SCRIPT), "--help"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "transformer-EV" in result.stdout


def test_validation_script_does_not_import_old_formal75k_resolver():
    text = VALIDATION_SCRIPT.read_text(encoding="utf-8")

    assert "formal_75k_80cell_workflow" not in text
    assert "transformer_ev_40cell_workflow" in text
