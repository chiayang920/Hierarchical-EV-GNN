# 500CP Formal Eval30 Protocol

## 1. Purpose

This protocol defines the controlled deterministic 500CP eval30 workflow for comparing the best checkpoint from five ActionGNN training seeds with the best checkpoint from five hierarchical training seeds.

The workflow evaluates 30 deterministic episodes per seed, producing 300 total formal evaluation episodes across the ten Slurm array tasks.

## 2. Formal research question

Under the same 500CP EV2Gym configuration, matched training seeds, matched deterministic evaluation episode seeds, and best-checkpoint selection, does the hierarchical policy improve reward or operational outcomes relative to ActionGNN?

## 3. Controlled evaluation design

- Config: `./config_files/PublicPST_500.yaml`
- Algorithms: `actiongnn` and `hierarchical`
- Training seeds: `0`, `1`, `2`, `3`, `4`
- Checkpoint selection: `model.best`
- Evaluation episodes per training seed: `30`
- Deterministic evaluation: `true`
- Evaluation exploration noise: `0.0`
- Maximum episode steps: `112`
- Device: `cpu`
- Slurm array tasks: `10`
- Total evaluated episodes: `300`

Training-time five-episode evaluations are monitoring evidence only.
Controlled deterministic eval30 CSVs are formal evaluation evidence.
Reward or operational superiority cannot be claimed until all ten eval30
tasks complete and matched-seed aggregation is performed.

## 4. Exact 10-task checkpoint mapping

| Eval task | Algorithm | Label | Seed | Training array job | Training task |
| --------: | --------- | ----- | ---: | -----------------: | ------------: |
| 0 | `actiongnn` | `baseline` | 0 | `58417948` | 0 |
| 1 | `actiongnn` | `baseline` | 1 | `58441541` | 1 |
| 2 | `actiongnn` | `baseline` | 2 | `58458540` | 2 |
| 3 | `actiongnn` | `baseline` | 3 | `58441541` | 3 |
| 4 | `actiongnn` | `baseline` | 4 | `58441541` | 4 |
| 5 | `hierarchical` | `hierarchical` | 0 | `58417948` | 5 |
| 6 | `hierarchical` | `hierarchical` | 1 | `58441541` | 6 |
| 7 | `hierarchical` | `hierarchical` | 2 | `58441541` | 7 |
| 8 | `hierarchical` | `hierarchical` | 3 | `58441541` | 8 |
| 9 | `hierarchical` | `hierarchical` | 4 | `58441541` | 9 |

Each task constructs the training run name as:

```bash
TRAIN_RUN_NAME="m3_500cp_formal_train_${ALGORITHM_LABEL}_${ALGORITHM}_cpu4_seed${SEED}_50000steps_job${TRAIN_ARRAY_JOB_ID}_task${TRAIN_TASK_ID}"
```

The model directory is:

```bash
MODEL_DIR="${TRAIN_ROOT}/${ALGORITHM_LABEL}/seed${SEED}/${TRAIN_RUN_NAME}/saved_models/${TRAIN_RUN_NAME}"
```

The checkpoint prefix is:

```bash
CHECKPOINT_PREFIX="${MODEL_DIR}/model.best"
```

## 5. ActionGNN seed2 warning

ActionGNN seed2 must use the successful rerun `58458540_2`.

It must not use the failed or timeout run `58441541_2`. The Slurm script protects this by mapping eval task `2` directly and explicitly to:

```text
algorithm=actiongnn
label=baseline
seed=2
training_array_job=58458540
training_task=2
```

No discovery logic, timestamp ordering, `find`, globbing, or latest-checkpoint selection is allowed.

## 6. Checkpoint selection rule: `model.best`

Every task evaluates the checkpoint prefix:

```bash
CHECKPOINT_PREFIX="${MODEL_DIR}/model.best"
```

The evaluator then loads `${CHECKPOINT_PREFIX}_actor` and `${CHECKPOINT_PREFIX}_critic`.

## 7. Evaluation episode-seed pairing

The base evaluation seed is:

```bash
BASE_EVAL_SEED=520000
```

The per-task offset is:

```bash
EVAL_SEED_OFFSET=$((BASE_EVAL_SEED + SEED * 1000 - SEED))
```

Because `evaluate_td3_gnn.py` calculates:

```text
episode_seed = training_seed + eval_seed_offset + episode_index
```

the actual deterministic episode seeds are:

| Training seed | Episode seeds |
| ------------: | ------------- |
| 0 | `520000-520029` |
| 1 | `521000-521029` |
| 2 | `522000-522029` |
| 3 | `523000-523029` |
| 4 | `524000-524029` |

ActionGNN and hierarchical tasks with the same training seed receive the same 30 episode seeds.

## 8. Resource request rationale

The eval30 workflow requests the `comp` partition, `normal` QOS, array `0-9`, wall time `06:00:00`, `4` CPUs per task, and `32G` memory.

The script runs on CPU and sets `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and `NUMEXPR_NUM_THREADS=4`. It exits immediately unless `SLURM_CPUS_PER_TASK=4`, preventing accidental runs with a different threading profile.

## 9. Seed0-first launch strategy

Launch the paired seed0 tasks first:

```bash
sbatch --array=0,5 m3_jobs/15_500cp_formal_eval30.slurm
```

After seed0 completes and task packages validate, launch the remaining paired seeds:

```bash
sbatch --array=1-4,6-9 m3_jobs/15_500cp_formal_eval30.slurm
```

Both commands require valid `EV_GNN_SOURCE_PROVENANCE` and `EV_GNN_SOURCE_MANIFEST` environment variables. `EV_GNN_SOURCE_MANIFEST` must point to a non-empty manifest file.

## 10. Required checkpoint and source preflight

Before evaluation, each task requires these checkpoint files to exist and be non-empty:

```text
${CHECKPOINT_PREFIX}_actor
${CHECKPOINT_PREFIX}_critic
${MODEL_DIR}/config.yaml
${MODEL_DIR}/kwargs.yaml
${MODEL_DIR}/run_args.yaml
${MODEL_DIR}/training_log.csv
```

Each required file prints `OK: <path>` when verified.

The task records SHA-256 checksums for:

```text
model.best_actor
model.best_critic
config.yaml
kwargs.yaml
run_args.yaml
training_log.csv
./config_files/PublicPST_500.yaml
evaluate_td3_gnn.py
```

The checkpoint copy of `config.yaml` and `./config_files/PublicPST_500.yaml` are parsed with `yaml.safe_load`, and the resulting Python objects must be equal. Formatting, comments, quoting, and YAML key ordering are not treated as configuration differences. Any actual key or value difference stops evaluation, with the differing keys and values printed to stderr.

The script also requires `EV_GNN_SOURCE_PROVENANCE` and `EV_GNN_SOURCE_MANIFEST`. From `${REPO_DIR}`, it runs `sha256sum -c "${EV_GNN_SOURCE_MANIFEST}"` and records the output in `source_manifest_validation.txt`; any hash mismatch stops evaluation before checkpoint loading. It then copies the manifest into runtime metadata, records the manifest-file checksum, and writes the provenance value, original manifest path, checkpoint mapping, and resolved checkpoint prefix. It does not run `git pull` or any Git command on M3.

## 11. Expected CSV schema and validation rules

The raw CSV must contain the evaluator metadata columns:

```text
run_name
algorithm
config
seed
episode_seed
checkpoint
episode_index
episode_reward
episode_steps
done
mean_reward
std_reward
action_mean
action_std
action_min
action_max
action_fraction_zero
action_fraction_at_max
active_action_count_mean
row_type
```

The inline validator fails unless:

- The CSV exists and is non-empty.
- Exactly 30 rows have `row_type == "episode"`.
- Exactly one row has `row_type == "summary"`.
- Episode indexes are exactly `0` through `29`.
- Episode seeds exactly match the expected paired seed range.
- Every episode has `episode_steps == 112`.
- Every episode has `done == true`.
- Every episode has the expected algorithm.
- Every episode has the expected training seed.
- Every episode records the exact resolved checkpoint prefix.
- Every `episode_reward` is finite.
- The summary `mean_reward` is finite.
- The summary row reports the expected algorithm, seed, and checkpoint.
- No required metadata column is missing.

Validation prints `CSV_VALIDATION_OK` with the algorithm, seed, episode count, episode seed minimum and maximum, and mean reward.

## 12. Expected output directories and package naming

The task-local raw CSV is:

```bash
RAW_OUTPUT_CSV="${TASK_DIR}/${ALGORITHM}_500cp_seed${SEED}_eval30.csv"
```

After validation, it is published to:

```bash
CANONICAL_OUTPUT_CSV="${CANONICAL_EVAL_DIR}/${ALGORITHM}_500cp_seed${SEED}_eval30.csv"
```

Existing canonical CSVs are never silently overwritten. Publication uses a task-unique temporary file, byte-compares it with the raw CSV, then atomically creates the canonical evidence path with a no-clobber publish step only after that comparison succeeds. If another task creates the canonical CSV first, publication fails instead of replacing it.

If the task later fails during final checksum, package creation, or package verification, the `EXIT` trap removes the task-created canonical CSV and any temporary canonical CSV before packaging failure logs and metadata. Only a fully successful task may leave a canonical CSV in the formal evidence directory.

The package path is:

```bash
PACKAGE_PATH="${OUT_DIR}/m3_500cp_formal_eval30_${ALGORITHM}_seed${SEED}_job${ARRAY_JOB_ID}_task${TASK_ID}.tar.gz"
```

The task package contains task-local logs, runtime metadata, the raw CSV, checksums, source manifest copy, and validation outputs. The archive is verified with `tar -tzf`.

## 13. Formal evidence boundaries

Training-time five-episode evaluations are monitoring evidence only.
Controlled deterministic eval30 CSVs are formal evaluation evidence.
Reward or operational superiority cannot be claimed until all ten eval30
tasks complete and matched-seed aggregation is performed.

Each single task is only one matched component of the 500CP eval30 matrix.

## 14. Post-run decision rule

After the seed0-first run, continue only if both seed0 task packages validate and the canonical CSVs are byte-identical to their raw task-local copies.

After all ten tasks complete, treat the ten validated canonical CSVs as the fixed formal eval30 evidence set for aggregation. Do not replace a canonical CSV without recording why the previous validated artifact was rejected.

## 15. Remaining aggregation phase

This PR intentionally does not add an aggregation script.

The remaining phase is to aggregate the ten canonical eval30 CSVs using matched training seeds and paired episode seeds. Only that aggregation can support cross-seed statements about reward or operational superiority.
