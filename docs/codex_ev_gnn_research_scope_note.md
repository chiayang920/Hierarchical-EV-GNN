# EV Charging MARL / EV-GNN Research Scope Note for Codex

## 0. Purpose of this note

This document is the working context and scope-control note for future Codex-assisted development in this repository. It should be read before making code changes.

The repository contains the original EV-GNN baseline code, an interactive `architecture_map.html` for understanding the baseline, a 25-charge-point diagnostic reproduction path, and early prototype files for a physically aligned hierarchical action-generation extension.

The immediate implementation target is **not** a broad rewrite of EV-GNN. The immediate target is a controlled 25 CP PublicPST prototype that preserves the EV-GNN simulator, graph state, reward, and EV2Gym execution interface while replacing the flat EV-node actor action-generation path with a hierarchical CPO -> transformer -> charger -> EV allocation path.

---

## 1. Research objective

### 1.1 Thesis direction

Working title:

> Toward Physically Aligned EV Charging Control: Hierarchical Graph Reinforcement Learning under EV2Gym PST

The research investigates whether the original EV-GNN actor can be made more physically aligned by explicitly decomposing charging decisions across the physical hierarchy of a charging network:

```text
CPO / global controller
  -> transformer-level allocation
    -> charger-level allocation
      -> EV-level local charging action
        -> EV2Gym-compatible flat action vector
```

### 1.2 Core research gap

The EV-GNN baseline already represents the charging system as a graph with EV, charger, transformer, and CPO/environment nodes. However, its actor still produces final EV-node actions and then maps those actions into the fixed EV2Gym action vector.

The gap is therefore:

> EV-GNN captures physical hierarchy in graph representation, but its action-generation path remains comparatively flat at the EV-node action layer. The proposed extension tests whether making decision authority explicit across CPO, transformer, charger, and EV levels improves physical-decision alignment, interpretability, and possibly performance.

### 1.3 Research questions from the proposal

The proposal defines three main research questions:

1. Can a hierarchical graph-based decision structure maintain or improve coordination performance relative to EV-GNN in the PST setting?
2. Can transformer-level and charger-level allocations make intermediate coordination more explicit and more interpretable than direct EV-level action mapping?
3. What trade-offs does the hierarchical policy introduce in training stability, convergence behaviour, and scalability?

---

## 2. Repository state after current inspection

### 2.1 Baseline source-code files

Core baseline files:

```text
train_RL_GNN.py
train_baselines.py
evaluator.py
requirements.txt
README.md

utils/state.py
utils/replay_buffer.py
utils/action_wrapper.py

TD3/TD3.py
TD3/TD3_GNN.py
TD3/TD3_ActionGNN.py

SAC/sac.py
SAC/actionSAC.py
SAC/model.py
SAC/utils.py

Results_Analysis/*.py
config_files/*.yaml
```

These files implement the original EV-GNN baseline and associated evaluation / analysis scripts. They should be treated as the baseline reference. Do not rewrite them unless the requested task explicitly asks for a baseline fix.

### 2.2 Current 25 CP diagnostic branch

Additional local files currently present:

```text
config_files/PublicPST_25cp.yaml
utils/state_25cp.py
utils/replay_buffer_25cp.py
TD3/TD3_ActionGNN_25cp.py
train_RL_GNN_25cp.py
evaluator_25cp.py
```

Purpose of these files:

- provide a smaller controlled 25 CP PublicPST training/evaluation path;
- simplify TD3_ActionGNN for CPU-safe / diagnostic execution;
- avoid touching the original baseline files;
- generate reproducible smoke-test and diagnostic outputs.

Observed diagnostic artefacts:

```text
saved_models/smoke_TD3_ActionGNN_25cp/
saved_models/benchmark_TD3_ActionGNN_25cp_seed0/
results/diagnostic_TD3_ActionGNN_25cp_50k_seed0/
logs/25cp_50k_100ep_diagnostic_eval*.txt
```

The diagnostic result is useful for verifying code execution, but it is **not** a final thesis-quality reproduction of the EV-GNN paper. The inspected 50k-step diagnostic evaluation produced matching `model.best` and `model.last` summary values, including approximately:

```text
total_reward_mean:             -103942.824
tracking_error_mean:            103942.824
energy_tracking_error_mean:        656.120
power_tracker_violation_mean:      793.266
average_user_satisfaction_mean:      0.999
total_transformer_overload_mean:     0.000
```

Interpret these as a local diagnostic baseline only.

### 2.3 Current hierarchical prototype files

Current prototype / placeholder files:

```text
utils/hierarchical_action_projection.py
utils/note_hierachical.txt
tests/test_hierarchical_action_projection_25cp.py
TD3/TD3_HierarchicalActionGNN.py
train_RL_GNN_hierarchical_25cp.py
```

Status:

- `utils/hierarchical_action_projection.py` is implemented as a deterministic projection layer.
- `tests/test_hierarchical_action_projection_25cp.py` tests the projection layer using a mock PublicPST_GNN-style graph state.
- `TD3/TD3_HierarchicalActionGNN.py` is only a planning placeholder. It is not implemented.
- `train_RL_GNN_hierarchical_25cp.py` is empty. It is not implemented.
- `utils/note_hierachical.txt` is an informal implementation sketch; use this Markdown note as the authoritative development scope instead.

---

## 3. Baseline architecture understanding

### 3.1 Baseline execution path

Baseline GNN training path:

```text
Config YAML
  -> train_RL_GNN.py
    -> EV2Gym environment
      -> state function
        -> policy class
          -> replay buffer
            -> env.step(mapped_action)
              -> evaluation / checkpoint / logging
```

25 CP diagnostic training path:

```text
config_files/PublicPST_25cp.yaml
  -> train_RL_GNN_25cp.py
    -> EV2Gym(SimpleReward, PublicPST_GNN)
      -> utils/state_25cp.py
        -> TD3/TD3_ActionGNN_25cp.py
          -> utils/replay_buffer_25cp.py
            -> model.best / model.last / training_log.csv
```

25 CP diagnostic evaluation path:

```text
evaluator_25cp.py
  -> load kwargs.yaml
  -> load TD3_ActionGNN_25cp checkpoint
  -> run N episodes on PublicPST_25cp.yaml
  -> write evaluation CSV
```

### 3.2 Three baseline policy regimes

The original EV-GNN repository includes three conceptual regimes:

```text
Classic RL
  Vector state -> MLP actor/critic -> fixed flat action vector
  Examples: TD3/TD3.py, SAC/sac.py with GNN_fx=False

FX-GNN
  Graph state -> GNN feature extractor -> graph-level pooled embedding -> MLP action head
  Examples: TD3/TD3_GNN.py, SAC/sac.py with GNN_fx=True through SAC/model.py

EV-GNN
  Graph state -> end-to-end GCN actor -> EV-node action outputs -> action_mapper -> fixed flat action vector
  Examples: TD3/TD3_ActionGNN.py, SAC/actionSAC.py
```

The proposed research should compare primarily against the EV-GNN regime.

### 3.3 Current `architecture_map.html`

`architecture_map.html` is an interactive baseline understanding artefact. It includes:

```text
[ Manual ]
[ Training Path ]
[ Evaluation Path ]
[ Results Analysis Path ]
[ Paper ↔ Code ]
```

Use it as a human-readable and Codex-readable reference for the original EV-GNN baseline. It maps files, classes, functions, and paper landmarks to each other. However, it is a documentation artefact, not the source of truth for implementation. When in doubt, inspect the actual `.py` source files.

The HTML should be committed to the repository because it is useful for future research work and onboarding. Codex may read it for orientation, but should not modify it unless explicitly asked.

---

## 4. Baseline code inventory for implementation decisions

### 4.1 `utils/state_25cp.py`

Purpose:

- Creates a pruned PublicPST_GNN-style graph state for 25 CP experiments.
- Includes only transformer/charger branches that have connected EVs.
- Produces `action_mapper`, which maps active EV node order to fixed EV2Gym action slots.

Key feature layouts:

```text
ev_features:
  [soc_flag, energy_exchanged, time_since_arrival, ev_id, cs_id, tr_id]

cs_features:
  [min_charge_current, max_charge_current, n_ports, cs_id]

tr_features:
  [max_power, tr_id]

env_features:
  [weekday/7, sin(hour), cos(hour), setpoint, previous_power_usage]
```

`PublicPST_GNN.node_sizes`:

```python
{"ev": 6, "cs": 4, "tr": 2, "env": 5}
```

### 4.2 `TD3/TD3_ActionGNN_25cp.py`

Purpose:

- Diagnostic EV-GNN TD3 implementation for 25 CP.
- Uses type-specific embeddings for EV / charger / transformer / env nodes.
- Uses GCN layers in actor and critic.
- Actor emits node-level actions for all graph nodes, then `select_action` maps EV-node actions into EV2Gym action slots using `state.action_mapper`.
- Critic expects an action tensor aligned with the batched graph node layout.

Important integration detail:

The current 25 CP critic concatenates node embeddings with `action.reshape(-1, discrete_actions)`. Therefore, the critic path expects an action tensor whose length aligns with **all graph nodes in the batch**, not only active EV nodes.

This is critical for the hierarchical implementation. The hierarchical projection currently returns active EV-node actions and a fixed EV2Gym action vector. For TD3 critic compatibility, the implementation must either:

1. build a full node-aligned action tensor with zero action features for non-EV nodes and hierarchical EV actions at EV node positions; or
2. modify the critic and replay buffer contract so they consistently consume active EV-node actions only.

Do not ignore this mismatch.

### 4.3 `utils/replay_buffer_25cp.py`

Purpose:

- Stores graph states, node-level actions, next states, rewards, and not-done flags.
- Manually batches PyG-style Data objects.
- Correctly offsets `edge_index` by cumulative node count when batching.

Important behaviour:

```text
action_batch = torch.cat([stored_action_i for sampled transitions], dim=0)
```

Therefore, stored actions must be consistent with the batched critic contract.

### 4.4 `utils/hierarchical_action_projection.py`

Purpose:

- Deterministic action-composition layer.
- Converts raw hierarchical actor outputs into active EV-node actions and EV2Gym-compatible flat action vectors.
- Does not train, learn, compute rewards, call EV2Gym, or use replay buffers.

Core dataclasses:

```text
HierarchicalRawAction
ProjectionMetadata
ProjectionResult
```

Core component classes:

```text
CPOTransformerBudgetProjector
TransformerChargerBudgetProjector
ChargerEVActionProjector
EV2GymActionMapper
HierarchicalActionProjection
```

Projection workflow:

```text
transformer_scores
  -> softmax over transformers
  -> transformer_weights

charger_scores
  -> grouped softmax within each transformer
  -> charger_weights

ev_ratios
  -> sigmoid
  -> EV local gates

final active EV action
  = total_budget * transformer_weight * charger_weight * EV local gate

active EV actions
  -> action_mapper
  -> fixed EV2Gym action vector
```

Default `total_budget`:

```text
number of active EVs
```

This is a normalised action-mass scale, not physical kW.

Current limitation:

The projection infers `charger_to_transformer_id` from active EV metadata. This is compatible with the pruned PublicPST_GNN graph assumption because included chargers should have active EVs. It may need extension for a full-graph state with empty charger branches.

### 4.5 `tests/test_hierarchical_action_projection_25cp.py`

Purpose:

- Unit tests for the projection layer using a mock PublicPST_GNN-style graph state.
- Does not train TD3.
- Does not run EV2Gym.
- Does not test GNN embeddings.

Mock setup:

```text
2 transformers
4 chargers
4 active EVs
25-dimensional EV2Gym action vector
active action slots: [0, 5, 9, 12]
```

Assertions:

```text
node_action shape is (4, 1)
mapped_action_tensor shape is (25,)
inactive slots remain zero
transformer weights sum to 1 globally
charger weights sum to 1 within each transformer
gradients flow to transformer_scores, charger_scores, ev_ratios, total_budget
project_for_env returns float32 NumPy vector of shape (25,)
```

---

## 5. Proposed implementation architecture

### 5.1 Current implementation scope

Primary implementation target:

```text
TD3_HierarchicalActionGNN under 25 CP PublicPST
```

Do not start with SAC hierarchy or full MARL. The current repository already contains custom TD3 and SAC baselines, but the implemented diagnostic path is TD3_ActionGNN_25cp. The safest next step is to implement the hierarchical TD3 variant first.

### 5.2 Files to create or implement

Implement:

```text
TD3/TD3_HierarchicalActionGNN.py
train_RL_GNN_hierarchical_25cp.py
```

Possibly add:

```text
tests/test_td3_hierarchical_actiongnn_25cp_shapes.py
tests/test_hierarchical_actor_real_state_25cp.py
```

Avoid modifying baseline reference files unless explicitly required:

```text
TD3/TD3_ActionGNN.py
TD3/TD3_ActionGNN_25cp.py
train_RL_GNN.py
train_RL_GNN_25cp.py
utils/state.py
utils/state_25cp.py
```

### 5.3 Intended hierarchical TD3 design

Recommended design:

```text
TD3_HierarchicalActionGNN
  Actor:
    - keep type-specific node embeddings
    - keep graph message passing backbone
    - replace flat final GCN action head with hierarchical raw-output heads
    - output HierarchicalRawAction:
        transformer_scores
        charger_scores
        ev_ratios
        optional total_budget
    - call HierarchicalActionProjection
    - return:
        mapped_action_numpy for env.step
        node_action / full_node_action for critic and replay buffer
        projection diagnostics for visualisation

  Critic:
    - preserve TD3_ActionGNN_25cp critic initially if possible
    - ensure action tensor shape matches critic expectation
    - only extend critic after baseline-compatible hierarchical actor works

  TD3 training structure:
    - preserve target networks
    - preserve clipped target policy noise
    - preserve delayed policy update
    - preserve soft target updates
```

### 5.4 Critical action-shape contract

There are two action representations:

```text
1. EV2Gym mapped action
   Shape: (action_dim,)
   Example for 25 CP: (25,)
   Used only for env.step(...)

2. Critic/replay action
   Must match the graph critic contract.
   Current TD3_ActionGNN_25cp critic expects node-aligned action features.
```

Do not pass a 25-dimensional EV2Gym action vector into the graph critic unless the critic is explicitly redesigned for that representation.

Recommended first integration strategy:

```text
Hierarchical projection returns active_EV_node_action: shape (num_active_evs, 1)
Actor constructs full_node_action: shape (num_total_nodes, 1)
  - zeros for env / transformer / charger nodes
  - active EV action at state.ev_indexes positions
Replay buffer stores full_node_action
Critic receives full_node_action
Env receives mapped_action_numpy
```

This preserves the existing critic/replay contract while still using hierarchical action generation.

### 5.5 Hierarchical actor output heads

Minimum actor heads:

```text
Transformer score head:
  input: transformer node embeddings and/or global context
  output: one scalar per transformer

Charger score head:
  input: charger node embeddings plus parent transformer context
  output: one scalar per included charger

EV ratio head:
  input: EV node embeddings plus charger / transformer context
  output: one scalar per active EV
```

A simple first version may use node embeddings directly:

```text
transformer_scores = linear_tr(x[tr_indexes])
charger_scores     = linear_cs(x[cs_indexes])
ev_ratios          = linear_ev(x[ev_indexes])
```

A stronger later version may concatenate parent context:

```text
EV head input = [EV embedding, charger embedding, transformer embedding, env/global embedding]
Charger head input = [charger embedding, transformer embedding, env/global embedding]
Transformer head input = [transformer embedding, env/global embedding]
```

Start simple, then extend only after shape tests and smoke training pass.

---

## 6. Algorithms and comparison scope

### 6.1 RL algorithms available in the repository

Available baseline families:

```text
Classic TD3
Classic SAC
TD3_GNN / FX-GNN
SAC_GNN / FX-GNN
TD3_ActionGNN / EV-GNN
SAC_ActionGNN / EV-GNN
StableBaselines3 baselines via train_baselines.py
```

### 6.2 Algorithm to implement first

Implement first:

```text
TD3_HierarchicalActionGNN_25cp
```

Rationale:

- The current diagnostic path is already TD3_ActionGNN_25cp.
- The baseline 25 CP TD3 EV-GNN training and evaluation scripts already exist.
- TD3 is easier to extend than introducing a new MARL framework.
- The projection module is already designed to fit this TD3 ActionGNN pathway.

### 6.3 SAC hierarchy

SAC hierarchy is optional and later-stage only:

```text
SAC_HierarchicalActionGNN
```

Do not implement until the TD3 hierarchy has passed:

```text
unit tests
real-state projection tests
short smoke training
25 CP evaluation comparison
```

### 6.4 MARL scope

The literature review discusses MARL because MARL is relevant to distributed EV charging decision-making. However, the current implementation should not be treated as full independent-agent MARL.

Current implementation scope:

```text
centralised hierarchical graph RL
```

Not current scope:

```text
independent transformer agents
independent charger agents
communication learning between agents
MADDPG / MAPPO / QMIX implementation
multi-agent replay buffer
multi-agent environment wrapper
```

If MARL is pursued later, it should be a separate extension after the centralised hierarchical TD3 prototype is stable.

---

## 7. Validation plan

### 7.1 Implementation validation sequence

Run in this order:

```text
1. Projection unit tests with mock state
2. Projection tests with real PublicPST_25cp state
3. Hierarchical actor forward-pass shape test
4. Full-node action alignment test for critic
5. env.step(mapped_action) smoke test
6. short training smoke test
7. 25 CP baseline vs hierarchical evaluation
8. optional 100 CP scale test
```

### 7.2 Required tests

Minimum tests to add or keep:

```text
tests/test_hierarchical_action_projection_25cp.py
  already exists; keep it passing

tests/test_hierarchical_projection_real_state_25cp.py
  create a real EV2Gym PublicPST_25cp env
  get state = PublicPST_GNN(env)
  build raw hierarchical action with correct sizes
  check project_for_training and project_for_env

tests/test_td3_hierarchical_actiongnn_25cp_shapes.py
  instantiate TD3_HierarchicalActionGNN
  call select_action(state)
  check mapped_action shape == (25,)
  check critic_action / full_node_action aligns with graph node count
  check no NaN / inf
```

### 7.3 Baseline comparison

Primary comparison:

```text
TD3_ActionGNN_25cp baseline
vs
TD3_HierarchicalActionGNN_25cp proposed
```

Controlled conditions:

```text
same config: config_files/PublicPST_25cp.yaml
same reward: SimpleReward
same state function: utils/state_25cp.PublicPST_GNN
same action interface: EV2Gym env.step(mapped_action)
same seed list
same max timesteps
same eval episodes
same device where possible
```

### 7.4 Metrics

Core PST metrics:

```text
total_reward
tracking_error
energy_tracking_error
power_tracker_violation
total_transformer_overload
average_user_satisfaction
total_energy_charged
total_ev_served
training time / evaluation time
```

Hierarchical alignment metrics to add:

```text
transformer_weights per timestep
charger_weights per timestep
ev_local_gates per timestep
budget_conservation_error
transformer allocation entropy
charger allocation entropy within transformer
percentage of steps where mapped_action is within action bounds
full-node action / mapped-action consistency
```

### 7.5 Success criteria

Minimum success:

```text
The hierarchical model trains without shape errors or gradient detachment.
The EV2Gym mapped action is valid at every step.
The model is not materially worse than TD3_ActionGNN_25cp on 25 CP PST core metrics.
The model exposes interpretable transformer and charger allocation traces.
```

Strong success:

```text
higher or comparable reward with lower tracking error
same or lower transformer overload
similar or better sample efficiency
stable training across seeds
clear transformer-level and charger-level allocation patterns
successful transfer from 25 CP to 100 CP
```

Failure / partial-success conditions:

```text
persistent reward collapse relative to TD3_ActionGNN_25cp
higher tracking error or transformer overload
hierarchical weights collapse to one transformer/charger without task justification
projection clipping saturates most actions
critic/action shape mismatch causes unstable training
no interpretable hierarchy emerges despite added architecture
```

---

## 8. Visualisation requirements

Validation must include scalar metrics and visual diagnostics.

Required plots:

```text
1. Training reward curve
   TD3_ActionGNN_25cp vs TD3_HierarchicalActionGNN_25cp

2. Evaluation metric comparison
   reward, tracking_error, energy_tracking_error, transformer overload, satisfaction

3. Transformer allocation heatmap
   x-axis: timestep
   y-axis: transformer id
   colour: transformer_weights or allocated action mass

4. Charger allocation heatmap
   x-axis: timestep
   y-axis: charger id
   colour: charger_weights or charger action mass

5. Budget consistency plot
   parent budget minus sum(child allocation), by timestep

6. Action distribution plot
   baseline EV-GNN actions vs hierarchical EV-GNN actions
```

Optional explainability plots:

```text
CPO -> transformer -> charger -> EV allocation graph for selected timestep
occupancy-band action density comparison
SoC-band action distribution comparison
```

---

## 9. Repository hygiene for GitHub and Codex

### 9.1 Files to commit

Commit source and documentation:

```text
README.md
requirements.txt
architecture_map.html
docs/codex_ev_gnn_research_scope_note.md
config_files/*.yaml
utils/*.py
TD3/*.py
SAC/*.py
Results_Analysis/*.py
train_RL_GNN.py
train_RL_GNN_25cp.py
train_RL_GNN_hierarchical_25cp.py
evaluator.py
evaluator_25cp.py
tests/*.py
```

Include small CSV summaries only if they are useful as diagnostic evidence:

```text
results/diagnostic_TD3_ActionGNN_25cp_50k_seed0/summary_best_last_compact.csv
results/diagnostic_TD3_ActionGNN_25cp_50k_seed0/summary_best_last_compact.md
```

### 9.2 Files not to commit by default

Do not commit generated or heavy artefacts by default:

```text
.DS_Store
__MACOSX/
__pycache__/
*.pyc
logs/
wandb/
saved_models/
eval_logs/
eval_models/
large model checkpoints
large raw result folders
```

Current checkpoint files are under the GitHub hard limit, but they are generated artefacts and should normally stay out of the main repository. If model snapshots are needed later, use GitHub Releases or Git LFS deliberately.

### 9.3 License and attribution

The uploaded zip did not include a local `LICENSE` file, but the original upstream GitHub repository exposes an MIT license. Add or restore the original `LICENSE` file before making the research repository public. Preserve the upstream citation and attribution in `README.md`.

Recommended root README role:

```text
short project overview
baseline citation
current status
where to find architecture_map.html
where to find this Codex scope note
basic run commands
license / attribution note
```

Do not use this long Codex scope note as the root README. Keep it under `docs/`.

---

## 10. Recommended next Codex task order

### Task 1: repository cleanup

```text
Create .gitignore.
Move this note into docs/.
Keep a concise README.md.
Restore LICENSE from upstream EV-GNN if absent.
Keep architecture_map.html at root.
```

### Task 2: real-state projection test

```text
Create a test that instantiates EV2Gym with PublicPST_25cp.yaml.
Reset env.
Call utils.state_25cp.PublicPST_GNN.
Build raw hierarchical action tensors matching real graph sizes.
Run HierarchicalActionProjection.project_for_training and project_for_env.
Assert valid shapes, valid bounds, non-duplicate action_mapper, and no NaN.
```

### Task 3: implement hierarchical TD3 actor

```text
Implement TD3/TD3_HierarchicalActionGNN.py.
Start from TD3_ActionGNN_25cp.py.
Preserve critic and TD3 train structure initially.
Replace only actor action generation.
Return both mapped EV2Gym action and full-node critic action.
```

### Task 4: implement hierarchical training script

```text
Implement train_RL_GNN_hierarchical_25cp.py from train_RL_GNN_25cp.py.
Swap policy class to TD3_HierarchicalActionGNN.
Keep config/reward/state/replay/evaluation protocol unchanged.
Add logging for transformer_weights, charger_weights, ev_local_gates if lightweight.
```

### Task 5: smoke train and compare

```text
Run short hierarchical smoke training.
Evaluate baseline and hierarchical model under same episode count.
Produce comparison CSV and plots.
```

---

## 11. Non-negotiable constraints for Codex

1. Do not rewrite the whole repository.
2. Do not modify original baseline files unless the task explicitly asks for it.
3. Preserve EV2Gym environment compatibility.
4. Preserve PublicPST_25cp as the first implementation target.
5. Preserve `SimpleReward` for the first comparison.
6. Do not introduce MARL frameworks before centralised hierarchical TD3 is stable.
7. Do not claim performance improvement without baseline-controlled evaluation.
8. Do not treat projection-unit-test success as full training success.
9. Always run shape tests before training.
10. Always verify critic action tensor shape against graph node count.

---

## 12. Compact one-paragraph context for future prompts

This repository extends the EV-GNN baseline for large-scale EV charging coordination. The baseline represents CPO, transformer, charger, and EV nodes as a graph, then produces EV-node actions that are mapped to EV2Gym's fixed action vector. The thesis extension keeps EV2Gym, PublicPST, the graph state, and the final EV2Gym action interface fixed, but replaces the flat EV-node actor with a hierarchical CPO -> transformer -> charger -> EV action-generation path. Current implemented prototype support includes `utils/hierarchical_action_projection.py` and mock projection tests; `TD3_HierarchicalActionGNN.py` and `train_RL_GNN_hierarchical_25cp.py` are not yet implemented. First implementation target is 25 CP PublicPST with TD3_ActionGNN_25cp as the baseline.
