# 500CP Formal Eval30 Multimetric Results Note

## Evidence provenance

- Complete bundle: `500cp_formal_eval30_complete_evidence_bundle.tar.gz`
- Bundle source manifest: `source_manifests/500cp_formal_eval30_d9dda95_expanded_source.sha256`
- Algorithms: `actiongnn` and `hierarchical`
- Training seeds: `0`, `1`, `2`, `3`, `4`
- Evaluation: 30 deterministic episodes per algorithm x seed
- Total formal evaluation episodes: 300
- Config: `./config_files/PublicPST_500.yaml`

## Validation status

The multimetric aggregation validates the complete canonical CSV set:

- canonical CSV files: 10
- episode rows: 300
- summary rows: 10
- episode indexes: `0-29` within every algorithm x seed file
- episode steps: 112 for every episode
- paired episode seeds: `520000-520029`, `521000-521029`, `522000-522029`, `523000-523029`, `524000-524029`
- task-package evidence: 10 task packages available in the bundle
- log evidence: 10 stdout logs and 10 stderr logs available in the bundle
- stderr status: all bundled stderr files are empty

## Main result

Positive reward difference means hierarchical is higher / less negative.
For tracking, violation, overload, degradation, and maximum-action saturation
metrics, lower is generally better.

| Metric | ActionGNN mean | Hierarchical mean | Hierarchical - ActionGNN | Paired t p | Wilcoxon p | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `episode_reward` | -29,013,742.7079 | -28,590,514.8996 | +423,227.8083 | 0.554364 | 0.812500 | Positive mean reward difference, not statistically significant. |
| `tracking_error` | 29,013,742.7079 | 28,590,514.8996 | -423,227.8083 | 0.554364 | 0.812500 | Lower mean tracking error, not statistically significant. |

The reward/tracking result is favourable in mean for the hierarchical actor, but
the paired seed evidence is statistically inconclusive. This result should not
be reported as significant 500CP reward superiority.

## Operational metrics

| Metric | Direction | ActionGNN mean | Hierarchical mean | Hierarchical - ActionGNN | Paired t p | Wilcoxon p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `tracking_error` | lower better | 29,013,742.7079 | 28,590,514.8996 | -423,227.8083 | 0.554364 | 0.812500 |
| `energy_tracking_error` | lower better | 10,955.8859 | 10,820.7775 | -135.1084 | 0.514275 | 0.812500 |
| `power_tracker_violation` | lower better | 16,790.1573 | 15,996.7284 | -793.4289 | 0.445428 | 0.437500 |
| `total_transformer_overload` | lower better | 133.2338 | 123.7073 | -9.5265 | 0.541407 | 1.000000 |
| `average_user_satisfaction` | higher better | 0.9985 | 0.9963 | -0.0021 | 0.406339 | 0.625000 |
| `energy_user_satisfaction` | higher better | 100.0000 | 100.0000 | 0.0000 | 1.000000 | 1.000000 |
| `total_ev_served` | higher better | 1,198.0000 | 1,198.0000 | 0.0000 | 1.000000 | 1.000000 |
| `battery_degradation` | lower better | 0.0562 | 0.0557 | -0.0005 | 0.407128 | 0.625000 |
| `total_profits` | higher better | -5,308.1884 | -5,251.0260 | +57.1624 | 0.409257 | 0.437500 |
| `action_fraction_at_max` | lower better | 0.5091 | 0.3397 | -0.1694 | 0.014394 | 0.062500 |

The strongest 500CP result is behavioural rather than reward-significance
based: `action_fraction_at_max` is lower for the hierarchical policy in all
five paired seeds. The paired t-test indicates significance at p < 0.05
(`p=0.014394`), while the exact Wilcoxon result is suggestive but not below
0.05 under `n=5` (`p=0.062500`).

## Boundaries

- No claim of general superiority across all CP scales follows from this 500CP result.
- No claim of significant 500CP reward superiority is supported.
- No comparison with external SOTA algorithms has been made yet.
- No per-transformer or per-charger saturation diagnostics are available yet.
- Metrics with ambiguous direction should be reported descriptively rather than treated as success criteria.

## Next research steps

1. Keep the current formal 500CP evidence frozen.
2. Audit multi-scale and multi-algorithm readiness before new formal runs.
3. Build a controlled adapter/evaluator for SB3 baselines before formal comparison.
4. Only then run a smoke matrix and, if it passes, a formal matrix.
