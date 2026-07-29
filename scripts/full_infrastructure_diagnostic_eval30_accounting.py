#!/usr/bin/env python3
"""Exact Slurm accounting contracts for the Stage D eval30 workflow."""

from __future__ import annotations

import re
from typing import Final

SACCT_FIELDS: Final[tuple[str, ...]] = (
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


def _select_metric(
    rows: dict[str, dict[str, str]],
    field: str,
) -> tuple[str, str]:
    for identity in ("parent", "batch", "extern"):
        value = rows[identity][field]
        if value not in {"", "Unknown", "N/A", "None"}:
            return value, identity
    return "", "unavailable"


def parse_stage_d_sacct(raw_text: str, array_job_id: str) -> list[dict[str, object]]:
    """Validate exact 8-task Slurm accounting and return task-level rows."""
    if not isinstance(array_job_id, str) or not array_job_id.isdigit():
        raise ValueError(f"array job ID must contain digits only: {array_job_id!r}")

    lines = [line for line in raw_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("sacct output is empty")

    by_display: dict[str, dict[str, str]] = {}
    display_pattern = re.compile(
        rf"^{re.escape(array_job_id)}_(?P<task>[0-7])(?P<suffix>\.(?:batch|extern))?$"
    )

    for line_number, line in enumerate(lines, start=1):
        fields = line.split("|")
        if tuple(fields) == SACCT_FIELDS:
            raise ValueError("sacct output must not contain a header row")
        if len(fields) != len(SACCT_FIELDS):
            raise ValueError(
                f"sacct row {line_number} must contain exactly nine fields, "
                f"got {len(fields)}"
            )
        row = dict(zip(SACCT_FIELDS, fields, strict=True))
        display = row["JobID"]
        match = display_pattern.fullmatch(display)
        if match is None:
            raise ValueError(
                f"sacct row {line_number} does not belong to array job ID "
                f"{array_job_id}: {display!r}"
            )
        if display in by_display:
            raise ValueError(f"duplicate sacct identity: {display}")
        if row["State"] != "COMPLETED":
            raise ValueError(
                f"sacct identity {display} must be COMPLETED, got {row['State']!r}"
            )
        if row["ExitCode"] != "0:0":
            raise ValueError(
                f"sacct identity {display} must have exit code 0:0, "
                f"got {row['ExitCode']!r}"
            )
        by_display[display] = row

    expected_displays = {
        display
        for task_id in range(8)
        for display in (
            f"{array_job_id}_{task_id}",
            f"{array_job_id}_{task_id}.batch",
            f"{array_job_id}_{task_id}.extern",
        )
    }
    missing = sorted(expected_displays - set(by_display))
    unexpected = sorted(set(by_display) - expected_displays)
    if missing:
        raise ValueError(f"missing required sacct identity: {missing[0]}")
    if unexpected:
        raise ValueError(f"unexpected sacct identity: {unexpected[0]}")

    results: list[dict[str, object]] = []
    for task_id in range(8):
        parent_display = f"{array_job_id}_{task_id}"
        parent = by_display[parent_display]
        batch = by_display[parent_display + ".batch"]
        extern = by_display[parent_display + ".extern"]
        raw_parent = parent["JobIDRaw"]
        if not raw_parent.isdigit():
            raise ValueError(
                f"parent JobIDRaw must be numeric for {parent_display}: {raw_parent!r}"
            )
        if batch["JobIDRaw"] != raw_parent + ".batch":
            raise ValueError(
                f"batch JobIDRaw is not linked to parent {raw_parent}: "
                f"{batch['JobIDRaw']!r}"
            )
        if extern["JobIDRaw"] != raw_parent + ".extern":
            raise ValueError(
                f"extern JobIDRaw is not linked to parent {raw_parent}: "
                f"{extern['JobIDRaw']!r}"
            )

        max_rss, maxrss_source = _select_metric(
            {"parent": parent, "batch": batch, "extern": extern},
            "MaxRSS",
        )
        total_cpu, totalcpu_source = _select_metric(
            {"parent": parent, "batch": batch, "extern": extern},
            "TotalCPU",
        )
        results.append(
            {
                "task_id": task_id,
                "job_id_raw": raw_parent,
                "job_id": parent_display,
                "job_name": parent["JobName"],
                "state": parent["State"],
                "exit_code": parent["ExitCode"],
                "elapsed_raw": parent["ElapsedRaw"],
                "alloc_cpus": parent["AllocCPUS"],
                "max_rss": max_rss,
                "total_cpu": total_cpu,
                "maxrss_source": maxrss_source,
                "totalcpu_source": totalcpu_source,
            }
        )
    return results
