from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT_ROOT / "docs/full_per_infrastructure_diagnostics_eval30_protocol.md"


def test_protocol_covers_scientific_scope_and_nontraining_boundary():
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "40 checkpoints",
        "1,200 episodes",
        "8/8 tasks",
        "40/40 checkpoint groups",
        "1,200/1,200 episode keys",
        "schema v3",
        "No model retraining",
        "training seed remains the statistical inference unit",
        "smoke job `58622672`",
    ):
        assert required in text


def test_protocol_covers_exact_submission_and_accounting_identity():
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "afterok:${ARRAY_JOB_ID}",
        "JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU",
        "Do not discover the latest job",
        "array_job_id=",
        "reducer_job_id=",
        "DRY_RUN_NO_SBATCH_CALLED",
        "FULL_INFRASTRUCTURE_DIAGNOSTIC_WORKFLOW_SUBMITTED",
    ):
        assert required in text


def test_protocol_covers_evidence_and_recovery_contracts():
    text = PROTOCOL.read_text(encoding="utf-8")
    for required in (
        "EVGNN_per_infrastructure_diagnostics_40checkpoints_eval30_job<ARRAY_JOB_ID>_<YYYYMMDD>",
        "FULL_INFRASTRUCTURE_DIAGNOSTIC_COMPLETE_BUNDLE_OK",
        "full_infrastructure_diagnostics_complete_evidence_job${ARRAY_JOB_ID}.tar.gz",
        "exact original array job ID",
        "Smoke-retirement gate",
        "two independent storage locations",
        "Do not rerun automatically",
    ):
        assert required in text


def test_protocol_references_all_stage_d_operator_files():
    text = PROTOCOL.read_text(encoding="utf-8")
    for path in (
        "m3_jobs/create_full_infrastructure_diagnostic_eval30_source_bundle.sh",
        "m3_jobs/21_full_infrastructure_diagnostic_eval30.slurm",
        "m3_jobs/22_full_infrastructure_diagnostic_reduce_bundle.slurm",
        "m3_jobs/submit_full_infrastructure_diagnostic_eval30_workflow.sh",
        "scripts/validate_full_infrastructure_diagnostic_eval30.py",
        "scripts/full_infrastructure_diagnostic_eval30_packages.py",
        "scripts/full_infrastructure_diagnostic_eval30_accounting.py",
    ):
        assert path in text
