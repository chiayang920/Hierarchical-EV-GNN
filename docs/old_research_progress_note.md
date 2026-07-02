# EV Charging MARL / EV-GNN Thesis Progress Note

## 1. Current research direction

当前研究方向：

**Toward Physically Aligned EV Charging Control: Hierarchical Graph Reinforcement Learning under EV2Gym PST**

核心目标是基于 EV-GNN baseline，设计一个更符合真实电力系统层级结构的 **Hierarchical Graph RL control architecture**，用于大规模 EV charging coordination。

Baseline paper：

> Orfanoudakis et al. (2025). *Scalable reinforcement learning for large-scale coordination of electric vehicles using graph neural networks*. Communications Engineering, 4:118. DOI: https://doi.org/10.1038/s44172-025-00457-8

该论文提出 EV-GNN，用 graph-based RL 解决 EV charging scalability 与 sample efficiency 问题，并强调 CPO 视角下的 large-scale EV coordination。

论文核心思想是：经典 RL 在 charging-point 数量变大时会受到 fixed-size state/action space 的限制，因为 state vector 会随 CP 数量增长，且会对 unoccupied charging points 产生无效 action。EV-GNN 的贡献是将 EV charging problem 建模为 graph，并通过 pruning 删除没有 connected EV 的 branches，使 actor 只对有效 EV nodes 产生 actions。

---

## 2. Research problem and gap

### 2.1 Baseline paper 已经解决的问题

EV-GNN 已经解决了以下问题：

1. **Scalability**  
   经典 RL 使用 fixed-size vector state 与 fixed-size action space，在 CP 数量增加时难以扩展。EV-GNN 使用 graph state 与 pruning，使模型可以只对 active EV branches 学习。

2. **Dynamic EV arrivals/departures**  
   EV arrival/departure 会让 charging-point occupancy 动态变化。EV-GNN 通过 graph pruning 使 inactive branches 不进入 actor decision path。

3. **End-to-end GNN RL**  
   论文区分了 classic RL、FX-GNN 与 EV-GNN。FX-GNN 只把 GNN 当 feature extractor；EV-GNN 则让 actor 的 final GCN layer 直接输出 EV-node-level action，再映射到 fixed-size action vector。

4. **Multi-scale evaluation**  
   论文 Fig. 2 比较了 25、100、500、1000 CP 不同 scale 下的 maximum reward，并指出 EV-GNN 版本在 scalability 上优于 classic RL。

### 2.2 Baseline 仍然存在的 research gap

虽然 EV-GNN 的 state representation 是 hierarchical graph：

```text
CPO / environment
  → transformer
    → charging station
      → EV
```

但它的 **decision architecture** 仍然是 relatively flat at the action-production level：

```text
Graph state
  → GCN actor
  → EV-node action
  → action_mapper
  → flat EV2Gym action vector
```

也就是说，baseline 已经在 **state side** 体现物理层级结构，但在 **action side** 并没有显式建模：

```text
CPO-level global allocation
Transformer-level capacity allocation
Charger-level allocation
EV-level local action
```

因此，research gap 可以表述为：

> Existing EV-GNN architecture captures physical hierarchy in the graph state representation, but its actor produces EV-level actions in a comparatively flat manner. It does not explicitly decompose the charging decision according to the operational hierarchy of CPO → transformer → charger → EV. This may limit interpretability, controllability, and physical alignment of learned actions.

---

## 3. Baseline source-code architecture progress

目前已经完成 EV-GNN baseline source-code architecture mapping，并做成了 `architecture_map.html`。

当前 architecture map 已完成四个阶段。

### Stage 1 — UI skeleton

完成 interactive HTML architecture navigator：

```text
[ Manual ]
[ Training Path ]
[ Evaluation Path ]
[ Results Analysis Path ]
```

主要功能：

```text
file tree
execution mind map
inspector panel
zoom / recenter
filter chips
side-card dependency edges
```

### Stage 2 — Full file content

已补完全部 file 的真实内容，包括：

```text
train_RL_GNN.py
train_baselines.py
evaluator.py
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
requirements.txt
README.md
```

并且已经区分：

```text
Classic RL = MLP + vector state
FX-GNN     = GNN feature extractor + MLP head
EV-GNN     = end-to-end GCN actor/critic + action_mapper
```

### Stage 3 — Content correction and source-code alignment

已修正若干 source-code / paper mismatch：

```text
SAC_GNN should be included as a distinct policy mode.
Evaluation Path should not overstate fixed 100 episodes in source default.
Results_Analysis scripts consume mixed artefacts, not only replay pickles.
v2g_stats eda.py has a space in the actual filename.
PublicPST_GNN_full_graph is an intended ablation and requires code audit.
PublicPST_GNN_no_position_encoding removes broader ID/position features.
```

### Stage 4 — Paper ↔ Code tab

已新增：

```text
[ Paper ↔ Code ]
```

用于把 paper landmark 映射到 source code：

```text
Fig. 1a → utils/state.py
Fig. 1b → TD3/TD3_ActionGNN.py, SAC/actionSAC.py
Fig. 1c → Actor.forward + action_mapper
Fig. 1d → Critic.forward + global_mean_pool
Fig. 2 → Results_Analysis/max_performance_results.py
Fig. 3 → gnn_fx_explain.py, state_action_eda.py
Fig. 4 → generalization_eda.py + PublicPST generalisation configs
Fig. 5 → discrete_eda.py + action_wrapper.py
Fig. 6 → v2g_rewards_eda.py + v2g_stats eda.py
Table 2 → evaluator.py + parse_evaluator_results.py
Eq. (3) → GCNConv
Eq. (4) → global_mean_pool
Eq. (5)-(8) → PST node features in utils/state.py
Eq. (9) → action_mapper + EV2Gym env.step()
Eq. (10) → SimpleReward external routing
Eq. (28)-(29) → MPC / Oracle baselines
Eq. (30)-(32) → V2G state/reward routing
```

---

## 4. Baseline execution pipeline

### 4.1 Training path

Baseline training path can be summarised as:

```text
Config YAML
  → train_RL_GNN.py
    → EV2Gym environment
      → state function
        → policy class
          → replay buffer
            → env.step(mapped_action)
              → evaluation / checkpoint / wandb
```

### 4.2 State construction

For GNN-based policies, `utils/state.py` builds graph observations:

```text
EV nodes
Charging station nodes
Transformer nodes
CPO / environment node
edge_index
action_mapper
```

The key point is that `action_mapper` maps active EV nodes back to the fixed-size EV2Gym action vector.

### 4.3 Policy regimes

The baseline contains three main regimes:

```text
Classic RL:
  vector state → MLP actor/critic → flat action vector

FX-GNN:
  graph state → GNN feature extractor → pooled graph embedding → MLP action head

EV-GNN:
  graph state → end-to-end GCN actor → EV-node actions → action_mapper → flat action vector
```

论文 Fig. 1c 描述 actor 的 GCN layers 与 EV-node action mapping；Fig. 1d 描述 critic 将 action features 与 node features 结合后通过 GCN 与 mean pooling 计算 Q-value。

---

## 5. Proposed research architecture

我的预计架构不是推翻 EV-GNN，而是在 EV-GNN 的 graph state 与 EV2Gym action interface 之间加入 **hierarchical decision decomposition**。

### 5.1 Baseline action path

Baseline EV-GNN action path:

```text
Graph state
  → GCN actor
  → EV-node action
  → action_mapper
  → flat EV2Gym action vector
```

### 5.2 Proposed hierarchical action path

Proposed hierarchical action path:

```text
Graph state
  → shared / hierarchical graph encoder
  → CPO actor
      → transformer budget allocation
  → transformer actor
      → charger budget allocation
  → charger actor
      → EV local action gate
  → hierarchical action projection
  → flat EV2Gym action vector
  → EV2Gym env.step()
```

更具体：

```text
CPO Actor
  outputs transformer-level budget weights

Transformer Actor
  allocates transformer budget to child chargers

Charger Actor
  allocates charger budget to connected EVs

EV Local Action Gate
  produces local charging/discharging ratio for each active EV

Hierarchical Action Projection
  composes all levels into EV-node-level actions
  then scatters them into EV2Gym fixed-size action vector
```

### 5.3 Why this is academically meaningful

这项 extension 的关键 scientific claim 是：

> A physically aligned hierarchical actor may improve EV charging coordination by embedding operational constraints and allocation hierarchy directly into the decision path, rather than only into the state graph.

这不是简单加 layer，而是改变 action-generation inductive bias：

```text
Baseline EV-GNN:
  physical hierarchy is represented in graph topology

Proposed Hierarchical EV-GNN:
  physical hierarchy is represented in both graph topology and decision decomposition
```

---

## 6. Implemented / planned code modules

### 6.1 已实现的 prototype module

目前已经有：

```text
hierarchical_action_projection.py
test_hierarchical_action_projection_25cp.py
```

该 module 的目的不是训练完整 TD3/SAC，而是先验证：

```text
raw hierarchical actor outputs
  → transformer weights
  → charger weights
  → EV local gates
  → EV-node action
  → EV2Gym fixed-size action vector
```

### 6.2 Projection logic

Projection layer follows:

```text
transformer_scores → softmax → transformer_weights
charger_scores within each transformer → softmax → charger_weights
ev_ratios → sigmoid / clamp → EV local gates
final EV action = total_budget × transformer_weight × charger_weight × EV local gate
```

Then:

```text
EV-node actions
  → action_mapper
  → fixed action vector of length |J|
```

### 6.3 Why projection is necessary

EV2Gym expects a fixed action vector whose dimension equals the number of charging points / charging stations. Therefore, even if the actor is hierarchical, the final output must still be projected into the EV2Gym-compatible flat action interface.

This preserves simulator compatibility:

```text
proposed hierarchical actor
  → projection
  → original EV2Gym env.step(action)
```

---

## 7. Research validation plan

The validation should answer one core question:

> Does the proposed hierarchical actor improve physical alignment, scalability, interpretability, or performance compared with the baseline EV-GNN?

### 7.1 Baseline models to compare

The minimum comparison set should include:

```text
Classic TD3
TD3_GNN / FX-GNN
TD3_ActionGNN / EV-GNN baseline
Hierarchical TD3_ActionGNN / proposed model
```

If time allows:

```text
Classic SAC
SAC_GNN / FX-GNN
SAC_ActionGNN / EV-GNN baseline
Hierarchical SAC_ActionGNN / proposed model
```

### 7.2 Environment settings

Primary environment:

```text
PublicPST / PST objective
EV2Gym simulator
same reward function as baseline
same graph state family as baseline
same action interface as baseline
```

Recommended scales:

```text
25 CP  → development and debugging
100 CP → intermediate scalability test
500 CP → stronger scalability evidence
1000 CP → only if compute budget allows
```

The EV-GNN paper uses 25, 100, 500, and 1000 CP scales to analyse scalability in Fig. 2.

### 7.3 Controlled experimental design

To make the comparison fair:

```text
same config files
same random seeds
same training steps
same replay/evaluation protocol
same reward function
same state function unless explicitly testing a state ablation
same action bounds
same evaluation metrics
```

---

## 8. Evaluation metrics

### 8.1 Main task-performance metrics

Primary metrics should follow EV2Gym / paper outputs:

```text
tracking_error
energy_tracking_error
power_tracker_violation
total_transformer_overload
average_user_satisfaction
total_reward
```

For PST, the key success signal is lower tracking error / energy tracking error while maintaining safety and user satisfaction.

### 8.2 Scalability metrics

To prove scalability improvement:

```text
training reward vs timestep
sample efficiency
best reward after fixed training budget
performance degradation from 25 → 100 → 500 CP
inference time per step
number of active EV nodes handled
```

### 8.3 Physical-alignment metrics

Because the proposed contribution is hierarchical decision-making, additional metrics are needed beyond reward:

```text
transformer budget utilisation
transformer overload frequency
charger-level allocation smoothness
EV action sparsity / validity
budget conservation error
allocation entropy across transformers
allocation entropy across chargers
```

Possible derived metrics:

```text
Transformer budget violation:
  max(0, allocated_power_tr - transformer_capacity_tr)

Budget conservation error:
  |sum(child_allocations) - parent_budget|

Hierarchy consistency:
  proportion of steps where child-level allocation stays within parent-level budget
```

---

## 9. Visualisation plan

Validation must include visual evidence, not only scalar scores.

### 9.1 Baseline vs proposed reward curve

Plot:

```text
x-axis: training timesteps
y-axis: total reward / episode reward
series:
  TD3_ActionGNN baseline
  Hierarchical TD3_ActionGNN
```

Purpose:

```text
show whether proposed model learns faster or reaches better reward
```

### 9.2 Tracking error over evaluation episodes

Plot:

```text
x-axis: evaluation episode
y-axis: tracking_error / energy_tracking_error
```

Purpose:

```text
show whether hierarchical actor improves PST objective
```

### 9.3 Transformer-level allocation heatmap

Plot:

```text
x-axis: simulation timestep
y-axis: transformer id
colour: allocated power or budget share
```

Purpose:

```text
show whether proposed actor learns physically coherent transformer-level allocation
```

### 9.4 Charger-level allocation heatmap

Plot:

```text
x-axis: simulation timestep
y-axis: charging station id
colour: charger-level budget or EV action
```

Purpose:

```text
show whether charger allocation follows transformer budgets
```

### 9.5 Budget conservation plot

Plot:

```text
x-axis: timestep
y-axis: parent budget minus sum(child allocations)
```

Purpose:

```text
prove hierarchical projection is numerically consistent
```

### 9.6 Action distribution comparison

Plot:

```text
baseline EV-GNN action distribution
vs
hierarchical EV-GNN action distribution
```

Can be grouped by:

```text
occupancy band
transformer id
charger id
EV SoC band
```

Purpose:

```text
show whether the proposed architecture changes action behaviour in interpretable ways
```

### 9.7 Graph-level explainability visualisation

Possible graph diagram:

```text
CPO node
  → transformer budget
    → charger budget
      → EV local action
```

Each edge can be annotated with learned budget weight.

Purpose:

```text
make the hierarchical decision path visible and interpretable
```

---

## 10. Criteria for proving research success

The research should not claim success only because the proposed model is more complex. It needs empirical evidence.

### 10.1 Minimum success condition

The proposed model is successful if it satisfies:

```text
No worse than baseline EV-GNN on core PST reward/tracking metrics
AND
better interpretability / physical alignment metrics
```

This is important because a physically aligned architecture may be valuable even if reward is similar, provided it produces more controllable and explainable allocation behaviour.

### 10.2 Strong success condition

The proposed model is strongly successful if it achieves:

```text
higher total reward
lower tracking_error
lower transformer overload
better sample efficiency
stable scaling from 25 to 100 or 500 CP
clear hierarchical allocation patterns
```

### 10.3 Failure condition

The research should be considered unsuccessful or only partially successful if:

```text
reward is consistently worse than EV-GNN baseline
tracking_error increases
transformer overload increases
hierarchical budgets collapse to trivial allocations
projection causes unstable gradients
training becomes significantly less stable
interpretability metrics do not show meaningful hierarchy
```

### 10.4 Evidence needed in final report

To convincingly argue research success, the report should include:

```text
1. Baseline EV-GNN reproduction result
2. Proposed hierarchical model result
3. Same-scale comparison under identical config/seeds
4. Reward/tracking-error plots
5. Transformer and charger allocation visualisations
6. Ablation showing hierarchy matters
7. Discussion of failure cases and limitations
```

---

## 11. Recommended ablation study

To prove the hierarchy itself matters, not just extra parameters, run:

```text
A0: Baseline EV-GNN
A1: Hierarchical actor without budget constraints
A2: Hierarchical actor with transformer-level budget only
A3: Hierarchical actor with transformer + charger budget
A4: Full hierarchical actor with projection
```

Expected interpretation:

```text
If A4 > A0 and A4 > A1:
  hierarchy + projection likely contributes meaningful inductive bias

If A1 ≈ A4:
  performance may come mainly from extra network capacity, not hierarchy

If A4 < A0:
  hierarchy may be too restrictive or optimisation is unstable
```

---

## 12. Current implementation roadmap

### Step 1 — Baseline understanding

Status: completed.

Outputs:

```text
architecture_map.html
Paper ↔ Code tab
baseline execution path understanding
source-code audit
```

### Step 2 — Projection-layer validation

Status: partially completed.

Existing files:

```text
hierarchical_action_projection.py
test_hierarchical_action_projection_25cp.py
```

Needed next:

```text
test with real PublicPST_GNN state
test with 25 CP EV2Gym environment
verify action_mapper compatibility
verify gradients through projection
```

### Step 3 — Integrate hierarchical actor into TD3_ActionGNN

Planned new module:

```text
TD3/TD3_HierarchicalActionGNN.py
```

or minimally:

```text
TD3/TD3_ActionGNN.py
  + HierarchicalActor
  + projection call
```

Recommended cleaner approach:

```text
create new file rather than heavily modifying baseline file
```

Reason:

```text
keeps baseline reproducible
makes experimental comparison cleaner
reduces risk of breaking original EV-GNN
```

### Step 4 — Train on 25 CP PublicPST

Start with:

```text
PublicPST.yaml
25 CP
same reward
same state function
same action bounds
```

Goal:

```text
verify training loop works
verify no shape mismatch
verify env.step accepts projected action
verify reward is non-degenerate
```

### Step 5 — Compare against baseline EV-GNN

Run:

```text
TD3_ActionGNN baseline
Hierarchical TD3_ActionGNN
```

Under same:

```text
seed
timesteps
config
evaluation protocol
```

### Step 6 — Scale test

If 25 CP succeeds, test:

```text
PublicPST_100.yaml
PublicPST_500.yaml if compute allows
```

---

## 13. Key risks

### Risk 1 — Hierarchical projection may over-constrain actions

If budget allocation is too restrictive, the model may lose flexibility compared with flat EV-node actions.

Mitigation:

```text
allow residual EV-level correction
or use soft budget constraints rather than hard clipping
```

### Risk 2 — Gradient instability

Softmax + sigmoid + multiplicative budget composition can shrink gradients.

Mitigation:

```text
monitor gradient norms
test temperature-scaled softmax
avoid excessive clipping
add projection unit tests
```

### Risk 3 — Extra architecture complexity may not improve reward

The hierarchy might improve interpretability but not performance.

Mitigation:

```text
define success partly through physical-alignment metrics
include ablation study
avoid claiming performance improvement unless proven
```

### Risk 4 — Baseline reproduction difficulty

If original EV-GNN results are difficult to reproduce exactly, comparison must be framed carefully.

Mitigation:

```text
compare under same local reproduction conditions
report local baseline as reproduced baseline
avoid claiming exact paper-level replication unless confirmed
```

---

## 14. One-sentence thesis claim

> This project investigates whether making the EV-GNN actor physically hierarchical—by decomposing CPO-level decisions into transformer, charger, and EV-level allocations—can improve the scalability, interpretability, and grid-aligned behaviour of reinforcement-learning-based EV charging control.

---

## 15. Working research question

> Can a physically aligned hierarchical graph actor improve the scalability, interpretability, and operational coherence of EV-GNN-based electric vehicle charging coordination compared with the original flat EV-node action architecture?

---

## 16. Expected final contribution

Expected contribution:

```text
1. A source-code-level analysis of the EV-GNN baseline.
2. A hierarchical action-generation extension compatible with EV2Gym.
3. A differentiable hierarchical action projection module.
4. Experimental comparison against baseline EV-GNN.
5. Visual evidence of transformer-level and charger-level allocation behaviour.
6. Empirical discussion of whether hierarchy improves reward, stability, or interpretability.
```

---

## 17. Immediate next tasks

Recommended next action sequence:

```text
1. Freeze architecture_map.html as baseline understanding artefact.
2. Create Markdown research progress note.
3. Re-open hierarchical_action_projection.py and validate with real PublicPST_GNN output.
4. Build TD3_HierarchicalActionGNN skeleton.
5. Run 25 CP shape-only environment integration test.
6. Run short training smoke test.
7. Compare first reward curves with TD3_ActionGNN baseline.
```

---

## 18. Notes for literature review / proposal alignment

The assignment requires the report to identify the research range, state-of-the-art, gap, and research plan. The brief specifically asks the report to address what research exists, what it tells us, what gap motivates the project, and how the project will address that gap.

This note therefore supports the proposal by defining:

```text
research area:
  scalable RL / GNN / EV charging coordination

state of the art:
  EV-GNN as end-to-end graph RL baseline

gap:
  graph-state hierarchy exists, but action-generation hierarchy is not explicit

plan:
  implement hierarchical graph actor + projection layer

validation:
  reproduce baseline, compare metrics, visualise allocation hierarchy
```

---

## 19. Reference links

- Orfanoudakis et al. (2025), *Scalable reinforcement learning for large-scale coordination of electric vehicles using graph neural networks*: https://doi.org/10.1038/s44172-025-00457-8
- EV-GNN GitHub repository: https://github.com/StavrosOrf/EV-GNN
