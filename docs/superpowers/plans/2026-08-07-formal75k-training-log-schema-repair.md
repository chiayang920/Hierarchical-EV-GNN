# Formal 75K Training Log Schema Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:verification-before-completion. This repair is executed in an isolated non-Git source snapshot.

**Goal:** Ensure every `training_log.csv` row uses a stable, named schema so formal smoke-package validation can read `eval/mean_reward` without positional inference.

**Architecture:** Keep the strict package validator unchanged and repair the CSV producer in `train_td3_gnn.py`. The writer will preserve the existing field order, append newly observed fields, atomically rewrite prior valid rows when the schema expands, and thereafter append using the established header.

**Tech Stack:** Python 3.11, standard-library `csv`, `tempfile`, `os.replace`, pytest, Bash.

## Global Constraints

- Do not modify reward calculation, policy behaviour, checkpoint selection, evaluation cadence, formal matrix, or M3 submission behaviour.
- Do not reinterpret unnamed CSV columns in validators.
- Preserve the failed smoke evidence as immutable historical debugging evidence.
- No Git, GitHub, Slurm, M3, training submission, or scientific-claim action.

---

### Task 1: Add regression coverage for dynamic CSV schema expansion

**Files:**
- Create: `tests/test_train_td3_gnn_logging.py`
- Modify: `tests/test_formal_75k_local_run_scripts.py`

**Interfaces:**
- Consumes: `train_td3_gnn.write_log_row(log_path, row)`
- Produces: tests requiring uniform row width, named `eval/mean_reward`, absence of `DictReader` unnamed extras, and verification-script coverage.

- [ ] Write a failing unit test that writes an eight-field training row followed by an evaluation row with additional named metrics.
- [ ] Assert that all raw rows match the header width and `csv.DictReader` returns a numeric `eval/mean_reward` with no `None` key.
- [ ] Write a second test for a later metric-key expansion while preserving earlier rows.
- [ ] Update the local verification-script contract test to require the new test file and `train_td3_gnn.py` compile coverage.
- [ ] Run the tests and confirm they fail against the original producer.

### Task 2: Implement the minimal producer-side fix

**Files:**
- Modify: `train_td3_gnn.py`

**Interfaces:**
- Consumes: a path and mapping row.
- Produces: a CSV whose header is the ordered union of all observed row keys and whose rows all match that header.

- [ ] Read the existing header and rows when the log already exists.
- [ ] Reject pre-existing malformed rows containing unnamed extra columns.
- [ ] Append newly observed fields to the existing header order.
- [ ] Atomically rewrite existing rows plus the new row only when the schema expands.
- [ ] Otherwise append using the established header with `extrasaction="raise"`.
- [ ] Run the new regression tests and confirm PASS.

### Task 3: Expand local verification and documentation

**Files:**
- Modify: `local_tools/run_local_formal75k_repair_verification.sh`
- Modify: `docs/formal_75k_local_repair_run_card.md`
- Modify: `docs/formal_75k_workflow_local_repair_summary.md`

**Interfaces:**
- Consumes: repaired source snapshot and historical evidence bundles.
- Produces: local verification that explicitly tests and compiles the CSV producer repair.

- [ ] Add `tests/test_train_td3_gnn_logging.py` to focused tests.
- [ ] Add `train_td3_gnn.py` to Python compile checks.
- [ ] Document the confirmed 8-column/43-column failure and producer-side repair.
- [ ] Run targeted tests, Bash syntax checks, and available local verification checks.

### Task 4: Package immutable v2 deliverables

**Files:**
- Create: external ZIP, unified diff patch, standalone verification script, standalone functional-smoke script, run card, and implementation summary.

**Interfaces:**
- Consumes: verified v2 snapshot.
- Produces: user-downloadable artefacts for isolated local execution.

- [ ] Generate a clean v2 ZIP with one top-level directory and no `.git` directory.
- [ ] Generate a reviewable unified diff against the uploaded v1 package.
- [ ] Copy the two executable local scripts as standalone downloads.
- [ ] Record checksums and exact placement/extraction instructions.
- [ ] Do not claim functional smoke PASS until the user executes it locally and returns evidence.
