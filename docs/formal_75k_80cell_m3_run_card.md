# Formal 75k 80-cell Run Card

This run card describes the local implementation only. It does not submit smoke,
formal training, eval30, diagnostics, or reducer jobs.

## Branch

- Base branch: `impl/corrected-nonnegative-actiongnn-scope-a`
- Experiment branch: `exp/formal-75k-nonnegative-comparison-v1`

## Matrix

- Scales: `25cp`, `100cp`, `500cp`, `1000cp`
- Algorithms: `actiongnn_nonnegative`, `hierarchical`
- Seeds: `0,1,2,3,4,5,6,7,8,9`
- Training steps: `75000`
- Training evaluation cadence: `5000`
- Training evaluation episodes: `5`
- Scheduled training evaluations: `15`

Dry-run matrix:

```bash
python scripts/formal_75k_80cell_workflow.py --print-matrix
```

## Stages

1. Stage S: two-cell smoke gate, `25cp actiongnn_nonnegative seed0` and `25cp hierarchical seed0`.
2. Stage A: 80-cell formal training array from scratch.
3. Stage B: training-matrix validation gate.
4. Stage C: 80-cell `model.best` eval30 array.
5. Stage D: 80-cell infrastructure-diagnostic array.
6. Stage E: eval30/diagnostic validation gate.
7. Stage F: final reducer and scientific analysis.

Dry-run orchestration:

```bash
bash m3_jobs/submit_formal_75k_80cell_workflow.sh --dry-run
```

Formal submission is blocked unless a reviewed resource profile is supplied:

```bash
bash m3_jobs/submit_formal_75k_80cell_workflow.sh \
  --submit-formal \
  --resource-profile docs/formal_75k_resource_profile_template.env
```

The template above intentionally has `RESOURCE_PROFILE_APPROVED=NO` and must
fail until reviewed evidence is entered.

## Statistical Contract

- Primary efficacy outcome: `episode_reward`.
- Primary inference unit: paired training seed, `n=10` per scale.
- Tracking error is descriptive and not an independent primary hypothesis.
- Key mechanism outcome: `Upper-bound active-EV action fraction (%) ↓`.
- Reward effect orientation: hierarchical minus corrected non-negative ActionGNN.
- Boundary effect orientation: corrected non-negative ActionGNN minus hierarchical, so positive favours hierarchy.
- Holm correction is applied across the four reward scale hypotheses and across the four mechanism scale hypotheses.

## Claim Boundaries

The reducer writes `claim_assessment.env` and does not force a positive
conclusion. Strong cross-scale architecture support requires all four reward
directions to favour hierarchy, at least three scale-specific reward rules to
pass, no statistically supported reward harm, key mechanism direction to favour
hierarchy at all four scales, and no clear material service collapse. No
service non-inferiority claim is allowed without a pre-approved margin.

## M3 Actions

`M3_ACTION_PERFORMED=NO`

This implementation creates local workflow code and tests only.
