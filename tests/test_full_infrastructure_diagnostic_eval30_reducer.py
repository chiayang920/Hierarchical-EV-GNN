import os
import subprocess
from pathlib import Path

import pytest

from scripts.full_infrastructure_diagnostic_eval30_accounting import (
    SACCT_FIELDS,
    parse_stage_d_sacct,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REDUCER_SCRIPT = (
    PROJECT_ROOT / "m3_jobs" / "22_full_infrastructure_diagnostic_reduce_bundle.slurm"
)
ARRAY_JOB_ID = "99999999"


def sacct_line(*fields):
    assert len(fields) == 9
    return "|".join(map(str, fields))


def valid_sacct_text(array_job_id=ARRAY_JOB_ID):
    rows = []
    for task_id in range(8):
        raw_parent = str(88000000 + task_id)
        display = f"{array_job_id}_{task_id}"
        rows.extend(
            [
                sacct_line(
                    raw_parent,
                    display,
                    "evgnn_infra_eval30",
                    "COMPLETED",
                    "0:0",
                    str(100 + task_id),
                    "4",
                    "",
                    f"00:01:{task_id:02d}",
                ),
                sacct_line(
                    raw_parent + ".batch",
                    display + ".batch",
                    "batch",
                    "COMPLETED",
                    "0:0",
                    str(100 + task_id),
                    "4",
                    f"{500000 + task_id}K",
                    f"00:01:{task_id:02d}",
                ),
                sacct_line(
                    raw_parent + ".extern",
                    display + ".extern",
                    "extern",
                    "COMPLETED",
                    "0:0",
                    str(100 + task_id),
                    "4",
                    "",
                    "00:00:00",
                ),
            ]
        )
    return "\n".join(rows) + "\n"


def replace_line(text, predicate, transform):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if predicate(line):
            lines[index] = transform(line)
            return "\n".join(lines) + "\n"
    raise AssertionError("target line not found")


def test_sacct_field_contract_is_exact():
    assert SACCT_FIELDS == (
        "JobIDRaw",
        "JobID",
        "JobName",
        "State",
        "ExitCode",
        "ElapsedRaw",
        "AllocCPUS",
        "MaxRSS",
        "TotalCPU",
    )


def test_parse_stage_d_sacct_accepts_exact_eight_task_accounting():
    rows = parse_stage_d_sacct(valid_sacct_text(), ARRAY_JOB_ID)
    assert len(rows) == 8
    assert [row["task_id"] for row in rows] == list(range(8))
    assert rows[0]["job_id"] == f"{ARRAY_JOB_ID}_0"
    assert rows[0]["state"] == "COMPLETED"
    assert rows[0]["exit_code"] == "0:0"
    assert rows[0]["max_rss"] == "500000K"
    assert rows[0]["maxrss_source"] == "batch"
    assert rows[0]["total_cpu"] == "00:01:00"
    assert rows[0]["totalcpu_source"] == "parent"


def test_parse_stage_d_sacct_rejects_header_row():
    text = "|".join(SACCT_FIELDS) + "\n" + valid_sacct_text()
    with pytest.raises(ValueError, match="header"):
        parse_stage_d_sacct(text, ARRAY_JOB_ID)


def test_parse_stage_d_sacct_rejects_wrong_field_count():
    text = valid_sacct_text() + "too|few|fields\n"
    with pytest.raises(ValueError, match="nine fields"):
        parse_stage_d_sacct(text, ARRAY_JOB_ID)


def test_parse_stage_d_sacct_rejects_wrong_array_job_id():
    with pytest.raises(ValueError, match="array job ID"):
        parse_stage_d_sacct(valid_sacct_text("88888888"), ARRAY_JOB_ID)


@pytest.mark.parametrize("suffix", ["", ".batch", ".extern"])
def test_parse_stage_d_sacct_rejects_missing_required_identity(suffix):
    target = f"{ARRAY_JOB_ID}_3{suffix}"
    lines = [
        line
        for line in valid_sacct_text().splitlines()
        if line.split("|")[1] != target
    ]
    with pytest.raises(ValueError, match="missing"):
        parse_stage_d_sacct("\n".join(lines) + "\n", ARRAY_JOB_ID)


def test_parse_stage_d_sacct_rejects_duplicate_identity():
    lines = valid_sacct_text().splitlines()
    lines.append(lines[0])
    with pytest.raises(ValueError, match="duplicate"):
        parse_stage_d_sacct("\n".join(lines) + "\n", ARRAY_JOB_ID)


def test_parse_stage_d_sacct_rejects_noncompleted_state():
    text = replace_line(
        valid_sacct_text(),
        lambda line: line.split("|")[1] == f"{ARRAY_JOB_ID}_4",
        lambda line: line.replace("|COMPLETED|", "|FAILED|", 1),
    )
    with pytest.raises(ValueError, match="COMPLETED"):
        parse_stage_d_sacct(text, ARRAY_JOB_ID)


def test_parse_stage_d_sacct_rejects_nonzero_exit_code():
    text = replace_line(
        valid_sacct_text(),
        lambda line: line.split("|")[1] == f"{ARRAY_JOB_ID}_5.batch",
        lambda line: line.replace("|0:0|", "|1:0|", 1),
    )
    with pytest.raises(ValueError, match="0:0"):
        parse_stage_d_sacct(text, ARRAY_JOB_ID)


def test_parse_stage_d_sacct_rejects_raw_child_link_mismatch():
    text = replace_line(
        valid_sacct_text(),
        lambda line: line.split("|")[1] == f"{ARRAY_JOB_ID}_2.extern",
        lambda line: "999.extern|" + line.split("|", 1)[1],
    )
    with pytest.raises(ValueError, match="JobIDRaw"):
        parse_stage_d_sacct(text, ARRAY_JOB_ID)


def test_reducer_slurm_resources_and_exact_accounting_contract():
    text = REDUCER_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=2" in text
    assert "#SBATCH --mem=16G" in text
    assert "#SBATCH --time=01:00:00" in text
    assert (
        "JobIDRaw,JobID,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,MaxRSS,TotalCPU"
        in text
    )
    assert (
        "full_infrastructure_diagnostic_eval30_task${task_id}_job${ARRAY_JOB_ID}.tar.gz"
        in text
    )
    assert (
        "full_infrastructure_diagnostics_complete_evidence_job${ARRAY_JOB_ID}.tar.gz"
        in text
    )
    assert ".tmp.${SLURM_JOB_ID}" in text
    assert "validate_stage_d_complete_bundle" in text


def test_reducer_dry_run_is_exact_and_non_mutating():
    env = os.environ.copy()
    env.update(
        {
            "SLURM_JOB_ID": "99999998",
            "EV_GNN_FULL_DIAGNOSTIC_REDUCER_DRY_RUN": "1",
            "EV_GNN_FULL_DIAGNOSTIC_ARRAY_JOB_ID": ARRAY_JOB_ID,
            "EV_GNN_FULL_DIAGNOSTIC_REPO_ROOT": str(PROJECT_ROOT),
            "EV_GNN_FULL_DIAGNOSTIC_EXPECTED_SOURCE_COMMIT": "f" * 40,
            "EV_GNN_FULL_DIAGNOSTIC_TASK_PACKAGE_ROOT": "/packages",
            "EV_GNN_FULL_DIAGNOSTIC_OUTPUT_ROOT": "/output",
        }
    )
    result = subprocess.run(
        ["bash", str(REDUCER_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert "FULL_INFRASTRUCTURE_DIAGNOSTIC_REDUCER_DRY_RUN" in lines
    assert f"array_job_id={ARRAY_JOB_ID}" in lines
    assert "expected_task_packages=8" in lines
    assert "expected_checkpoint_groups=40" in lines
    assert "expected_episode_count=1200" in lines
    assert any(line.startswith("SACCT_COMMAND=sacct -j 99999999") for line in lines)
    assert any(
        line
        == "final_bundle=/output/full_infrastructure_diagnostics_complete_evidence_job99999999.tar.gz"
        for line in lines
    )
    assert "DRY_RUN_NO_ACCOUNTING_OR_PACKAGING" in lines
