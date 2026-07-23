# Research Progress Log

## Architecture status

- ActionGNN controlled baseline: implemented as the controlled baseline path for matched EV2Gym evaluation.
- Hierarchical actor: implemented with the CPO -> Transformer -> Charger -> EV action-composition path.
- Vectorised hierarchical implementation: current path uses the vectorised composition optimisation while preserving actor, critic, replay-buffer, and EV2Gym action contracts.
- Controlled evaluator: records scalar EV2Gym operational metrics dynamically alongside reward and action diagnostics.

## Evidence status

| Scale | Status | Current interpretation |
| --- | --- | --- |
| 25CP | Formal evidence complete. | Positive reward evidence and selected operational evidence support the small-scale hierarchical result. |
| 100CP | Runtime/scalability evidence available. | Useful for feasibility and optimisation context; reward evidence remains limited by paired-seed robustness. |
| 500CP | Formal eval30 complete. | Reward/tracking superiority is statistically inconclusive; hierarchical maximum-action saturation is reduced; multimetric operational analysis has been added. |
| 1000CP | Not yet formal. | Requires config, smoke, runtime, and feasibility audit before any formal claim. |

## Current limitations

- The current controlled training and evaluation pipeline supports only `actiongnn` and `hierarchical`.
- SB3 baselines exist in `train_baselines.py`, but they are not yet integrated into the manifest-controlled eval30 pipeline.
- 1000CP formal feasibility is not yet established.
- Per-transformer and per-charger diagnostics are not yet available.

## Next phase

The next phase should be a P2 audit-only PR for multi-scale and multi-algorithm
readiness. No immediate large-scale M3 job should be launched until that audit
passes.
