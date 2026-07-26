# Controlled Multiscale Formal Results — M3 Job 58513929

## Evidence provenance

- M3 array job: `58513929`
- M3 reducer job: `58513930`
- Matrix: 4 scales × 2 algorithms × 5 seeds = 40 train/eval tasks
- Scales: `25cp`, `100cp`, `500cp`, `1000cp`
- Algorithms: `actiongnn`, `hierarchical`
- Evaluation: deterministic `eval30`
- Primary inference unit: training seed
- Evaluation episodes: within-seed deterministic replicates, not independent statistical samples
- Full archive path, local only and not committed: `/Users/jameschen/Downloads/EVGNN_Formal_Evidence/controlled_multiscale_formal_complete_evidence_job58513929.tar.gz`
- Slim review archive path, local only and not committed: `/Users/jameschen/Downloads/EVGNN_Formal_Evidence/controlled_multiscale_formal_review_slim_job58513929.tar.gz`
- Codex audit report path, local only and not committed: `/Users/jameschen/Downloads/EVGNN_Formal_Evidence/job58513929_codex_full_audit_report.md`

## Evidence integrity summary

The local audit passed the core evidence integrity and independent aggregation checks, with one non-blocking packaging caveat: the exact uppercase reducer stdout marker strings were not present in the extracted bundle, although equivalent reducer count metadata and aggregation exit status were present.

| Evidence item | Result |
| --- | ---: |
| task packages | 40/40 |
| stdout logs | 40/40 |
| stderr logs | 40/40 |
| canonical eval30 CSVs | 40/40 |
| source manifests | 40/40 |
| task runtime metadata | 40/40 |
| task CSV validation metadata | 40/40 |
| task package checksums | 40/40 |
| aggregation outputs | 5/5 |
| reducer metadata files | 3/3 |
| complete SHA-256 manifest entries | 327 |
| nested task package readability | 40/40 |
| serious stderr findings | 0 |
| independent aggregation recomputation | matched with 0 mismatches |

All 40 canonical CSVs had 31 data rows each: 30 `episode` rows and 1 `summary` row. Episode indices were exactly `0-29`; episode seeds followed the expected scale-specific base plus `training_seed * 1000 + episode_index`; and paired ActionGNN/hierarchical episode seed sets matched for every scale and training seed.

## Main empirical findings

The table reports training-seed means. Positive differences are `hierarchical - actiongnn`. For reward-like metrics, higher is better; for tracking error, violation, overload, and action saturation, lower is generally better.

| Scale | Metric | ActionGNN mean | Hierarchical mean | Diff H-A | Paired t p | Wilcoxon p | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `25cp` | `episode_reward` | -102841.6965 | -80809.7994 | 22031.8971 | 0.045352 | 0.125000 | Hierarchical higher by paired t only. |
| `25cp` | `tracking_error` | 102841.6965 | 80809.7994 | -22031.8971 | 0.045352 | 0.125000 | Hierarchical lower by paired t only. |
| `25cp` | `energy_tracking_error` | 650.4873 | 571.3583 | -79.1289 | 0.040537 | 0.125000 | Hierarchical lower by paired t only. |
| `25cp` | `power_tracker_violation` | 788.7367 | 557.5467 | -231.1900 | 0.033251 | 0.125000 | Hierarchical lower by paired t only. |
| `25cp` | `total_transformer_overload` | 0.0363 | 0.1998 | 0.1635 | 0.373901 | 1.000000 | Descriptively higher for hierarchical. |
| `25cp` | `action_fraction_at_max` | 0.5546 | 0.3039 | -0.2507 | 0.000581 | 0.062500 | Consistently lower saturation; Wilcoxon not below 0.05. |
| `100cp` | `episode_reward` | -1116617.5318 | -1083977.0285 | 32640.5034 | 0.872097 | 1.000000 | Small descriptive hierarchical improvement only. |
| `100cp` | `tracking_error` | 1116617.5318 | 1083977.0285 | -32640.5034 | 0.872097 | 1.000000 | Small descriptive hierarchical improvement only. |
| `100cp` | `energy_tracking_error` | 2139.9457 | 2128.6970 | -11.2488 | 0.954906 | 1.000000 | Descriptively lower for hierarchical only. |
| `100cp` | `power_tracker_violation` | 3284.9061 | 3229.2909 | -55.6152 | 0.914649 | 1.000000 | Descriptively lower for hierarchical only. |
| `100cp` | `total_transformer_overload` | 26.0934 | 31.3342 | 5.2408 | 0.444504 | 0.625000 | Descriptively higher for hierarchical. |
| `100cp` | `action_fraction_at_max` | 0.4791 | 0.3329 | -0.1462 | 0.012597 | 0.062500 | Lower saturation by paired t only. |
| `500cp` | `episode_reward` | -29039513.2389 | -28601700.7913 | 437812.4476 | 0.531715 | 0.812500 | Descriptive hierarchical improvement only. |
| `500cp` | `tracking_error` | 29039513.2389 | 28601700.7913 | -437812.4476 | 0.531715 | 0.812500 | Descriptive hierarchical improvement only. |
| `500cp` | `energy_tracking_error` | 10966.0188 | 10813.6477 | -152.3712 | 0.487071 | 0.437500 | Descriptively lower for hierarchical only. |
| `500cp` | `power_tracker_violation` | 16815.4480 | 15967.0703 | -848.3777 | 0.438168 | 0.437500 | Descriptively lower for hierarchical only. |
| `500cp` | `total_transformer_overload` | 131.8981 | 122.2276 | -9.6705 | 0.529053 | 0.437500 | Descriptively lower for hierarchical only. |
| `500cp` | `action_fraction_at_max` | 0.5093 | 0.3397 | -0.1695 | 0.014887 | 0.062500 | Lower saturation by paired t only. |
| `1000cp` | `episode_reward` | -112160052.8860 | -94991051.2503 | 17169001.6352 | 0.247859 | 0.437500 | Largest descriptive reward improvement, not significant. |
| `1000cp` | `tracking_error` | 112160052.8860 | 94991051.2503 | -17169001.6352 | 0.247859 | 0.437500 | Largest descriptive tracking improvement, not significant. |
| `1000cp` | `energy_tracking_error` | 21545.3058 | 19850.3421 | -1694.9636 | 0.300409 | 0.312500 | Descriptively lower for hierarchical only. |
| `1000cp` | `power_tracker_violation` | 29546.2080 | 26530.0106 | -3016.1974 | 0.421881 | 1.000000 | Descriptively lower for hierarchical only. |
| `1000cp` | `total_transformer_overload` | 258.7347 | 175.3322 | -83.4026 | 0.185084 | 0.187500 | Descriptively lower for hierarchical only. |
| `1000cp` | `action_fraction_at_max` | 0.4623 | 0.3113 | -0.1510 | 0.045730 | 0.125000 | Lower saturation by paired t only. |

## Primary research-value statement

The strongest cross-scale benefit is not universal reward superiority. The strongest defensible benefit is consistent reduction in maximum-action saturation:

| Scale | ActionGNN `action_fraction_at_max` | Hierarchical `action_fraction_at_max` | Diff H-A |
| --- | ---: | ---: | ---: |
| `25cp` | 0.5546 | 0.3039 | -0.2507 |
| `100cp` | 0.4791 | 0.3329 | -0.1462 |
| `500cp` | 0.5093 | 0.3397 | -0.1695 |
| `1000cp` | 0.4623 | 0.3113 | -0.1510 |

The hierarchical actor produces a less boundary-saturated charging policy across scales.

This is a behavioural/control-quality result. It is stronger than the current evidence for universal reward superiority because the reward/tracking effects are scale-dependent and often not significant at the seed level.

## Scale-level interpretation

- `25cp`: Hierarchical improves reward/tracking and key operational metrics by paired t-test, but exact Wilcoxon is not significant.
- `100cp`: Small descriptive reward/tracking improvement, not significant; transformer overload is descriptively higher for hierarchical.
- `500cp`: Reward/tracking improvements are descriptive and not significant; strongest result is lower `action_fraction_at_max`.
- `1000cp`: Largest descriptive reward/tracking improvement but not significant; lower `action_fraction_at_max` has paired t support but Wilcoxon remains non-significant.

## Claim boundaries

- No external SOTA claim.
- No SB3 comparison claim.
- No broad formal superiority claim across all scales.
- No episode-level p-values.
- Training seed is the inference unit.
- The current evidence supports behavioural/control-quality improvement more strongly than universal reward superiority.
- Smoke evidence must not be treated as a formal result.

## Resource observations

- Resource request: 4 CPUs, 96GB memory, 48h wall-time, CPU device.
- All training/evaluation exit statuses were `0`.
- Future jobs should use scale-specific memory and wall-time requests.
- The `1000cp` runtime reached roughly 24–28.6h in formal tasks, so do not blindly reduce `1000cp` wall-time without buffer.
- The memory request was conservative and likely over-allocated.

Observed wall-time ranges from task runtime metadata:

| Scale | Algorithm | Min | Median | Max |
| --- | --- | ---: | ---: | ---: |
| `25cp` | `actiongnn` | 00:48:19 | 00:55:35 | 01:09:30 |
| `25cp` | `hierarchical` | 00:55:00 | 01:05:02 | 01:59:48 |
| `100cp` | `actiongnn` | 01:33:52 | 01:48:50 | 02:20:31 |
| `100cp` | `hierarchical` | 01:44:34 | 02:02:05 | 02:27:16 |
| `500cp` | `actiongnn` | 08:01:18 | 08:18:56 | 10:37:14 |
| `500cp` | `hierarchical` | 07:46:24 | 09:24:12 | 10:37:40 |
| `1000cp` | `actiongnn` | 16:35:01 | 26:49:24 | 28:35:53 |
| `1000cp` | `hierarchical` | 18:07:24 | 27:34:27 | 28:28:35 |

## Limitations and next research steps

- The design has an `n=5` seed-level statistical power limitation.
- Exact Wilcoxon tests are coarse at `n=5`.
- Per-transformer and per-charger saturation diagnostics are needed to localize where boundary saturation is reduced.
- Service/fairness diagnostics are needed to ensure reduced saturation does not hide under-service.
- Reward and constraint ablations are needed to separate reward-scale effects from policy-structure effects.
- A computational-cost vs control-quality discussion is needed before making deployment-oriented claims.
- A future reducer bundler should include a reducer stdout marker snapshot so reducer completion markers are preserved directly in the review bundle.

## Verification

Codex verification commands for this documentation-only PR:

```bash
git status --short
git diff -- docs/controlled_multiscale_formal_results_job58513929.md
git diff --check
git status --short
git status --short | grep -E "tar.gz|EVGNN_Formal_Evidence|task_packages|checkpoint|model.best|model.last|\.csv" && echo "ERROR: evidence-like file staged or modified" || echo "NO_EVIDENCE_ARCHIVE_OR_CSV_STAGED"
git diff --stat 5ef491eb38581e3df043607adc55490999fc2619..HEAD
git diff --name-only 5ef491eb38581e3df043607adc55490999fc2619..HEAD
git diff --check 5ef491eb38581e3df043607adc55490999fc2619..HEAD
git diff --name-only 5ef491eb38581e3df043607adc55490999fc2619..HEAD | grep -E "tar.gz|EVGNN_Formal_Evidence|task_packages|checkpoint|model.best|model.last|\.csv" && echo "ERROR: evidence-like file committed" || echo "NO_EVIDENCE_ARCHIVE_OR_CSV_COMMITTED"
```

Expected committed scope:

```text
docs/controlled_multiscale_formal_results_job58513929.md
```
