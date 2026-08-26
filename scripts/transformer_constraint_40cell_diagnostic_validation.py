#!/usr/bin/env python3
"""Prepare one transformer-constraint 40-cell diagnostic reconciliation package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.transformer_constraint_40cell_workflow import resolve_formal_cell
from scripts.validate_full_infrastructure_diagnostic_eval30 import (
    RECONCILIATION_CONTRACT_VERSION,
    SAME_PASS_RECONCILIATION_COLUMNS,
    SeedReconciliationInputs,
    atomic_write_csv_rows,
    atomic_write_json,
    build_mapping_validation_payload,
    build_same_pass_reconciliation_rows,
    build_service_reconciliation_rows,
    evaluate_service_reconciliation_rows,
)


def resolve_validation_cell(task_id: int):
    return resolve_formal_cell(task_id)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing diagnostic evidence: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty diagnostic evidence: {path}")
    return rows


def prepare_validation(
    *,
    diagnostic_dir: Path,
    validation_dir: Path,
    runtime_metadata_dir: Path,
    task_id: int,
    source_identity: str,
) -> dict[str, object]:
    cell = resolve_validation_cell(task_id)
    diagnostic_rows = _read_csv(diagnostic_dir / "episode_diagnostics.csv")
    transformer_rows = _read_csv(diagnostic_dir / "transformer_diagnostics.csv")
    charger_rows = _read_csv(diagnostic_dir / "charger_diagnostics.csv")
    same_pass_rows = _read_csv(diagnostic_dir / "same_pass_canonical_eval30.csv")
    inputs = SeedReconciliationInputs(
        diagnostic_rows=diagnostic_rows,
        transformer_rows=transformer_rows,
        charger_rows=charger_rows,
        historical_canonical_rows=[],
        same_pass_canonical_rows=same_pass_rows,
        formal_validation={},
        scale=cell.scale,
        algorithm=cell.algorithm,
        task_id=cell.task_id,
        training_seed=cell.seed,
        stage_d_source_commit_sha=source_identity,
        config_path=None,
        checkpoint_prefix=None,
    )
    same_pass_reconciliation = build_same_pass_reconciliation_rows(inputs)
    service_reconciliation = build_service_reconciliation_rows(
        diagnostic_rows,
        transformer_rows,
        charger_rows,
    )
    mapping = build_mapping_validation_payload(inputs, transformer_rows, charger_rows)
    same_pass_failures = sum(row["status"] != "pass" for row in same_pass_reconciliation)
    service_failures = evaluate_service_reconciliation_rows(service_reconciliation)
    mapping_failures = 0 if mapping.get("status") == "ok" else 1
    hard_failures = same_pass_failures + service_failures + mapping_failures
    summary = {
        "status": "ok" if hard_failures == 0 else "failed",
        "hard_gate_status": "pass" if hard_failures == 0 else "fail",
        "mapping_validation_status": "pass" if mapping_failures == 0 else "fail",
        "same_pass_metric_status": "pass" if same_pass_failures == 0 else "fail",
        "service_reconciliation_status": "pass" if service_failures == 0 else "fail",
        "mapping_validation_failure_count": mapping_failures,
        "same_pass_metric_failure_count": same_pass_failures,
        "service_reconciliation_failure_count": service_failures,
        "diagnostic_schema_version": 3,
        "reconciliation_contract_version": RECONCILIATION_CONTRACT_VERSION,
        "source_identity": source_identity,
        "task_id": task_id,
        "scale": cell.scale,
        "algorithm": cell.algorithm,
        "training_seed": cell.seed,
        "same_pass_reconciliation_rows": len(same_pass_reconciliation),
        "service_reconciliation_rows": len(service_reconciliation),
    }
    validation_dir.mkdir(parents=True, exist_ok=True)
    runtime_metadata_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv_rows(
        validation_dir / "same_pass_canonical_reconciliation.csv",
        SAME_PASS_RECONCILIATION_COLUMNS,
        same_pass_reconciliation,
    )
    atomic_write_csv_rows(
        validation_dir / "service_reconciliation.csv",
        [
            "episode_index",
            "served_count_reconciliation_status",
            "satisfaction_sum_reconciliation_status",
            "charged_energy_reconciliation_status",
            "discharged_energy_reconciliation_status",
            "episode_total_ev_served",
            "charger_served_ev_count_sum",
            "transformer_served_ev_count_sum",
            "episode_total_energy_charged",
            "charger_energy_charged_kwh_sum",
            "transformer_energy_charged_kwh_sum",
            "episode_total_energy_discharged",
            "charger_energy_discharged_kwh_sum",
            "transformer_energy_discharged_kwh_sum",
        ],
        service_reconciliation,
    )
    atomic_write_json(validation_dir / "mapping_validation.json", mapping)
    atomic_write_json(runtime_metadata_dir / "reconciliation_summary.json", summary)
    if hard_failures:
        raise ValueError(f"transformer-constraint diagnostic reconciliation failed: {summary}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare transformer-constraint 40-cell diagnostic reconciliation evidence "
            "using the transformer-constraint task resolver."
        )
    )
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--runtime-metadata-dir", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--source-identity", required=True)
    args = parser.parse_args()
    result = prepare_validation(**vars(args))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print("STATUS=BLOCKED")
        print(f"BLOCK_REASON={exc}")
        raise SystemExit(2)
