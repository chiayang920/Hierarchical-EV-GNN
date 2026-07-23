# Multiscale Multialgorithm Experiment Plan

## 1. Purpose

The next research phase is to assess readiness for a complete multi-scale, multi-metric, multi-algorithm comparison. This plan does not claim that such a comparison is already complete. It defines the audit boundary, missing engineering, and staged path required before formal evidence can be extended beyond the current controlled ActionGNN-versus-hierarchical results.

## 2. Current support boundary

The manifest-controlled TD3-GNN pipeline currently supports the ActionGNN and hierarchical actors through `train_td3_gnn.py` and `evaluate_td3_gnn.py`.

SB3 baselines exist in `train_baselines.py`, but they are not yet integrated into the same deterministic eval30 evidence pipeline. They should be treated as repository-supported candidate baselines only, pending adapter, wrapper, seed, checkpoint, and metric-schema validation.

## 3. Proposed experiment matrix

Scales:

- 25CP
- 100CP
- 500CP
- 1000CP, only if config and smoke feasibility are verified

Current algorithms:

- ActionGNN
- hierarchical

Candidate additional baselines, pending adapter and feasibility checks:

- TD3 MLP
- SAC MLP
- PPO MLP
- TQC MLP
- DDPG or A2C MLP, only if justified by audit

These candidate baselines are not final SOTA baselines unless separately justified against current literature and implemented under the same evidence protocol.

## 4. Metrics

The matrix should include reward, tracking, operational, satisfaction, overload, degradation, profit, action saturation, runtime, memory, and wall-time metrics.

Reward alone is insufficient. Operational metrics are required for paper-level claims because grid tracking, overload behaviour, user satisfaction, degradation, and profit can move differently from reward.

The current evaluator records aggregate action diagnostics, but per-transformer and per-charger diagnostics are not yet available. Current columns also do not distinguish overload count from overload magnitude.

## 5. Required engineering before formal comparison

- SB3 checkpoint save/load adapter.
- Deterministic eval30 adapter for SB3 policies.
- Unified canonical CSV schema.
- Common episode seed protocol.
- Source manifest and artefact packaging.
- Smoke jobs for each algorithm x scale.
- Aggregation script that accepts multiple algorithms and scales.

## 6. Staged execution plan

### Stage 0 — Audit only

- Run audit script.
- Verify configs, algorithms, metrics, and protocols.

### Stage 1 — Smoke matrix

- One seed.
- Short training or checkpoint-load test.
- 1-3 eval episodes.
- Detect crash, memory, action-space, wrapper, and checkpoint issues.

### Stage 2 — Pilot matrix

- Selected scales and selected algorithms.
- 1-2 seeds.
- Eval30.
- Verify metric schema and aggregation.

### Stage 3 — Formal matrix

- 5 matched seeds.
- Eval30.
- Complete evidence packaging.
- Statistical aggregation.

## 7. Non-goals

- No M3 job is launched by this PR.
- No external baseline is added by this PR.
- No final SOTA comparison is claimed by this PR.
- No architecture changes are made by this PR.
