from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFY = PROJECT_ROOT / "local_tools" / "run_local_formal75k_repair_verification.sh"
SMOKE = PROJECT_ROOT / "local_tools" / "run_local_formal75k_functional_smoke.sh"


def test_local_verification_script_contract():
    text = VERIFY.read_text()
    assert "/opt/anaconda3/envs/evgnn_core/bin/python3.11" in text
    assert "formal_75k_80cell_historical_evidence.py" in text
    assert "--require-authoritative-diagnostic-sha" in text
    assert "pytest" in text
    assert "tests/test_train_td3_gnn_logging.py" in text
    assert "py_compile" in text
    assert "train_td3_gnn.py" in text
    assert "bash -n" in text
    assert "EV_GNN_FORMAL_75K_SOURCE_DRY_RUN=1" in text
    assert "DRY_RUN_NO_JOBS_SUBMITTED" in text
    assert '"${PYTHON_BIN}" - train_td3_gnn.py' in text
    assert "<<'PY'" in text
    assert 'FINAL_STATUS="PASS"' in text
    assert 'FINAL_STATUS="BLOCKED"' in text
    assert "sbatch" in text
    assert "M3_ACTION_PERFORMED" not in text
    assert "GIT_ACTION_PERFORMED" not in text
    assert "M3_SUBMISSION_COUNT" not in text


def test_local_functional_smoke_script_contract():
    text = SMOKE.read_text()
    assert "/opt/anaconda3/envs/evgnn_core/bin/python3.11" in text
    assert "SLURM_ARRAY_TASK_ID=0" in text
    assert "SLURM_ARRAY_TASK_ID=1" in text
    assert "23_formal_75k_80cell_smoke.slurm" in text
    assert "validate-smoke-packages" in text
    assert "test_td3_actiongnn_nonnegative_contracts.py" in text
    assert "test_td3_hierarchical_actiongnn_contracts.py" in text
    assert "LOCAL_FUNCTIONAL_SMOKE_ONLY" in text
    assert "eval/mean_reward" in text
    assert "TRAINING_LOG_SCHEMA=PASS_2_OF_2" in text
    assert "SCIENTIFIC_CLAIM_GENERATED=NO" in text
    assert "M3_ACTION_PERFORMED" not in text
    assert "GIT_ACTION_PERFORMED" not in text
    assert "M3_SUBMISSION_COUNT" not in text
    assert 'FINAL_STATUS="PASS"' in text
    assert 'FINAL_STATUS="BLOCKED"' in text
    assert "sbatch" not in text
