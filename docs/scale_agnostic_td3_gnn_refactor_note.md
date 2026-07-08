# Scale-Agnostic TD3-GNN Refactor Note

## Rationale

The previous 25CP naming was misleading because CP scale is config-driven. EV2Gym config files such as `PublicPST_25cp.yaml` and `PublicPST_100.yaml` determine the action space, and TD3-GNN code infers `action_dim` from `env.action_space.shape[0]`.

## Old-To-New Path Map

| Old path or command | New path or command |
| --- | --- |
| `TD3/TD3_ActionGNN_25cp.py` | `TD3/TD3_ActionGNN_Controlled.py` |
| `train_RL_GNN_25cp.py` | `train_td3_gnn.py --algorithm actiongnn` |
| `train_RL_GNN_hierarchical_25cp.py` | `train_td3_gnn.py --algorithm hierarchical` |
| `evaluator_25cp.py` | `evaluate_td3_gnn.py` |
| `evaluator_td3_actiongnn_controlled.py` | Deprecated wrapper for `evaluate_td3_gnn.py` |
| `utils/state_25cp.py` | Compatibility wrapper for `utils/state_public_pst_gnn.py` |
| `utils/replay_buffer_25cp.py` | Compatibility wrapper for `utils/replay_buffer_actiongnn.py` |

## Algorithm Labels

Use `actiongnn` as the canonical machine-readable label for the controlled ActionGNN baseline. Use `hierarchical` as the canonical label for the hierarchical actor.

Legacy evaluator aliases are accepted for migration:

| Legacy label | Canonical label |
| --- | --- |
| `baseline_25cp` | `actiongnn` |
| `hierarchical_25cp` | `hierarchical` |

Historical result filenames that use `baseline` may still be read as legacy ActionGNN baseline outputs.

## Historical 100CP Results

Existing `baseline_100cp_seed*_eval30.csv` files remain valid historical evidence. The 100CP aggregation script normalises legacy `baseline` filenames to canonical `actiongnn` internally while preserving `phase2d_100cp_*` output names.

## Checkpoint Migration

Checkpoint folder names and algorithm labels are decoupled. Old checkpoint folders can be evaluated with the new evaluator when the algorithm is specified explicitly, for example:

```bash
python evaluate_td3_gnn.py \
  --algorithm actiongnn \
  --config ./config_files/PublicPST_100.yaml \
  --seed 0 \
  --eval_episodes 30 \
  --checkpoint ./saved_models/phase2d_formal_baseline_100cp_seed0_50k/model.best \
  --device cpu \
  --output_csv ./artifacts/evaluations/actiongnn_100cp_seed0_eval30.csv \
  --run_name actiongnn_100cp_seed0_eval30
```

## Future Usage

Train the ActionGNN baseline:

```bash
python train_td3_gnn.py --algorithm actiongnn --config ./config_files/PublicPST_100.yaml
```

Train the hierarchical actor:

```bash
python train_td3_gnn.py --algorithm hierarchical --config ./config_files/PublicPST_100.yaml
```

Evaluate an old baseline checkpoint:

```bash
python evaluate_td3_gnn.py \
  --algorithm actiongnn \
  --config ./config_files/PublicPST_100.yaml \
  --checkpoint ./saved_models/phase2d_formal_baseline_100cp_seed0_50k/model.best \
  --output_csv ./artifacts/evaluations/actiongnn_100cp_seed0_eval30.csv
```

Evaluate a new ActionGNN checkpoint:

```bash
python evaluate_td3_gnn.py \
  --algorithm actiongnn \
  --config ./config_files/PublicPST_100.yaml \
  --checkpoint ./artifacts/experiments/actiongnn_100cp_seed0/model.best \
  --output_csv ./artifacts/evaluations/actiongnn_100cp_seed0_eval30.csv
```

## Historical Reproducibility

Historical M3 logs and `run_args.yaml` files may still reference old training scripts. Those paths are preserved in Git history. New experiments should use `train_td3_gnn.py` and `evaluate_td3_gnn.py`.

Existing completed M3 jobs are historical evidence. Updated M3 scripts are for future reproducible runs after this refactor.

## Scope

This refactor does not change TD3 actor, critic, target-action smoothing, replay-buffer batching, hierarchical allocation logic, or evaluation metrics.

This refactor does not make `utils/hierarchical_action_projection.py` the live policy projection path. It remains a tested standalone utility unless the live hierarchical actor is explicitly refactored to import and use it later.

This refactor does not include private meeting notes or local result briefs.
