# 两阶段进化架构重构计划

## 1. 摘要

将现有"主搜索循环 + GA 附加"的混合架构重构为**标准两阶段进化算法**：Phase 1 通过纯 draft 建立多样化初始种群（目标 K=8 个有效节点），Phase 2 通过 merge/mutate 在基因层面系统性提升。同步修复 debug 只有 1 次的问题（改为链式 2 次），引入明确死节点机制，将 `explore.j2` 完整改名为 `draft.j2`，两阶段共享总节点预算（`agent.max_steps=100`）。

---

## 2. 设计决定（全部已确认）

| # | 决定 | 值 |
|---|------|---|
| A | Phase 1 终止条件 | 自适应：valid_pool ≥ K=8 后，**当前 epoch 跑完**再切换（不提前中断） |
| B | K 默认值 | **8** |
| C | Phase 2 escape draft | **无**（Phase 2 纯 merge/mutate） |
| D | debug 最多次数 | **2 次（链式传递）** |
| E | 基因 schema | **统一 7 基因**，non-applicable 写 stub section |
| F | explore.j2 重命名 | **完整改名为 draft.j2**，任务类型全局改为 "draft" |
| G | Merge 基因选择 | **Pheromone 全局 TOP-1 + 退化检测，无 Tournament**；单一参考父代 = 贡献基因最多的节点（并列时选分高者） |
| H | Mutate 基因选择 | 随机选，**过滤 stub**（内容 < 20 字符的基因位点跳过）；Tournament k=3 选 parent |
| I | 总节点预算 | `agent.max_steps`（MLE-Bench 默认 100），两阶段共享 |
| J | 节点计数规则 | journal 追加数（dead + alive 都算），debug 中间产物**不计** |
| K | Agent 进化触发 | 全局 epoch 计数（Phase 1 + Phase 2 合计），**每 3 epoch 触发一次** |
| L | Merge 父代记录 | `node.parent_id = primary_parent.id`；`node.metadata["gene_sources"] = {locus: node_id, ...}` |
| M | Mutate 父代记录 | `node.parent_id = parent.id`（tournament 选出的单一父代） |

---

## 3. 现有架构的核心问题

```
现有每步:
  _select_parent_node()
    → None       (draft)        ← 正确
    → buggy_leaf (50% 概率)     ← 问题①: 用 explore.j2 "修复" = 语义混乱，浪费步骤
    → best_node  (improve)      ← 问题②: "ONE atomic change" 无约束，不可归因

  _try_immediate_debug(): 1次、非链式、每次都传原始错误  ← 问题③
  SolutionEvolution.run_epoch(): epoch 结束后 bolt-on    ← 问题④: GA 是配角
  死节点: 不存在，buggy 节点永远留在池里被随机重试        ← 问题⑤
  explore.j2 命名: 同时承担 draft/improve/repair 三种语义 ← 问题⑥
  Mutate 基因选择: random.choice(REQUIRED_GENES) 无 stub 过滤 ← 问题⑦
  Merge 基因选择（pheromone 模式）: 忽略 tournament 父节点，
    从全局取，若某节点全面领先则 7 基因全来自它（退化为复制）← 问题⑧
  Merge 参考代码: 传两个完整父代但基因来源与二者无关  ← 问题⑨（新增）
```

---

## 4. 新架构总览

```
总预算: max_steps = 100（journal 节点数上限，dead + alive 都算）

main.py
├── Phase 1: Draft 阶段
│   目标: 积累 ≥8 个 valid_pool 节点（有效 + 基因完整）
│   每步: 1 draft.j2 → exec → debug_chain(≤2次) → journal.append
│   每轮 epoch 结束后检查，达标后当前 epoch 跑完再切换
│   Agent 进化: 全局 epoch 计数，每 3 轮触发
│
└── Phase 2: 进化阶段
    目标: 在 valid_pool 基础上基因层面系统提升
    每步: merge.j2 或 mutate.j2 → exec → debug_chain(≤2次) → journal.append
    valid_pool 动态增长（成功节点立刻可用）
    Agent 进化: 继续全局 epoch 计数，每 3 轮触发
    预算: total_budget - phase1_node_count
```

---

## 5. Phase 1：Draft 阶段

### 终止与预算控制

```
total_budget = config.agent.max_steps    # 100
phase1_epoch = 0

while True:
    run_epoch_draft()                    # 跑满整个 epoch，不提前中断
    phase1_epoch += 1
    if phase1_epoch % 3 == 0:
        agent_evo.evolve(phase1_epoch)   # Agent 进化（全局计数）

    valid_pool = get_valid_pool(journal)
    if len(valid_pool) >= 8: break       # 达到 K
    if len(journal.nodes) >= total_budget: break
    if time_limit_reached(): break
```

若 Phase 1 结束时 valid_pool < 8，Phase 2 仍启动（小种群可运行）。

### 单步逻辑

```
draft_step():
  draft_history = build_draft_history(journal)   # 已有节点的 approach_tag 列表
  node = agent.generate(draft.j2, parent=None, draft_history=draft_history)
  exec(node)
  node = debug_chain(node, max_attempts=2)       # 见第 7 节
  review(node)                                    # 写入 approach_tag, metric_value
  journal.append(node)                            # dead 也追加，计入预算
  gene_registry.update(node)
```

### 多样性引导（Review-based）

Review 输出 schema 新增字段 `approach_tag: str`（1 行方法摘要，如 `"LightGBM + 5-fold CV"`）。

draft.j2 注入多样性约束：

```jinja
{% if draft_history %}
## Approaches Already Tried — DO NOT REPEAT
{% for item in draft_history %}
- [Step {{ item.step }}] {{ item.approach_tag }} → metric={{ item.metric or "failed" }}
{% endfor %}
Your solution MUST use a meaningfully different approach from all of the above.
{% endif %}
```

`build_draft_history(journal)` 从 journal 收集所有已完成节点的 `(step, approach_tag, metric_value)`。

---

## 6. Phase 2：进化阶段

### 预算与循环

```
phase2_budget = total_budget - len(journal.nodes)   # 剩余步数
phase2_epoch  = 0

while len(journal.nodes) < total_budget and not time_limit_reached():
    remaining = total_budget - len(journal.nodes)
    evo.run_epoch(max_steps=min(steps_per_epoch, remaining))
    phase2_epoch += 1

    global_epoch = phase1_epoch + phase2_epoch
    if global_epoch % 3 == 0:
        agent_evo.evolve(global_epoch)             # Agent 进化（全局计数）
```

### 单步逻辑

```
phase2_step():
  valid_pool = get_valid_pool(journal)
  op = random.choice(["merge", "mutate"])          # 默认各 50%

  if op == "merge":
    gene_plan = pheromone_with_degenerate_check(journal, gene_registry, current_step)
    primary_parent = get_primary_parent(gene_plan, journal)
    # primary_parent = 贡献基因数最多的节点；并列时选 metric 最高者
    node = agent.generate(merge.j2, primary_parent, gene_plan)
    node.parent_id = primary_parent.id
    node.metadata["gene_sources"] = {
        locus: gene_plan[field]["source_node_id"]
        for locus, field in LOCUS_TO_FIELD.items()
    }

  elif op == "mutate":
    parent  = tournament_select(valid_pool, k=3)
    target  = select_non_stub_gene(parent)          # 过滤 stub
    node = agent.generate(mutate.j2, parent, target)
    node.parent_id = parent.id

  exec(node)
  node = debug_chain(node, max_attempts=2)
  review(node)
  journal.append(node)                              # dead 也追加，计入预算
  gene_registry.update(node)
  if not node.dead:
    valid_pool.append(node)                         # 动态增长，本步立刻可用
```

### valid_pool 动态增长

Phase 2 每个成功节点立刻加入 `valid_pool`，成为下一轮的候选。初始 K=8 → 随进化持续扩大，避免在小种群上反复操作。

---

## 7. 基因选择设计

### Merge：Pheromone 全局选 + 退化检测 + 单一参考父代

**当前问题**：
- pheromone 策略忽略父节点，若某节点得分全面领先，7 基因全来自它，退化为复制
- 传入两个 tournament 父代作参考，但基因来源与二者无关，造成语义混乱

**新设计**：去掉 tournament，pheromone 全局选择 + 退化检测，然后从基因来源中推断单一参考父代。

```
pheromone_with_degenerate_check(journal, gene_registry, current_step):
  # Step 1: 标准 pheromone 全局 TOP-1（现有 select_gene_plan 逻辑不变）
  gene_plan = select_gene_plan(journal, gene_registry, current_step)

  # Step 2: 退化检测（7 基因是否全来自同一节点）
  source_ids = {v["source_node_id"] for v in gene_plan.values() if isinstance(v, dict)}
  if len(source_ids) == 1:
    dominant_id = next(iter(source_ids))
    pools = build_decision_gene_pools(journal, gene_registry, current_step)

    # 每个位点替换为 second-best（来自不同节点）
    for locus in REQUIRED_GENES:
      candidates = [item for item in pools[locus]
                    if item.source_node_id != dominant_id]
      if candidates:
        second_best = max(candidates, key=lambda x: x.quality)
        gene_plan[LOCUS_TO_FIELD[locus]] = {
          "locus": locus,
          "source_node_id": second_best.source_node_id,
          "code": second_best.raw_content,
        }
  return gene_plan


get_primary_parent(gene_plan, journal):
  # 统计每个 source_node_id 贡献的基因数
  counter = Counter(
      v["source_node_id"] for v in gene_plan.values() if isinstance(v, dict)
  )
  max_count = max(counter.values())
  candidates = [nid for nid, cnt in counter.items() if cnt == max_count]
  if len(candidates) == 1:
      return journal.get_node(candidates[0])
  # 并列时选 metric 最高的节点
  return max(candidates,
             key=lambda nid: journal.get_node(nid).metric_value or 0.0)
```

**merge.j2 的参考信息**：单一 `primary_parent` 完整代码 + gene_plan 中各基因片段。
LLM 以 primary_parent 为骨架，按 gene_plan 逐节替换对应 section。

**为什么不用"质量感知二元交叉"**：限制只在 parent_a/b 之间选，当一方全面优于另一方时同样退化（7 基因全来自胜者）。Pheromone + 退化检测保留了全局视野，对退化的处理更鲁棒。

**为什么去掉 Tournament（Merge 侧）**：Tournament 选出的两个父代与基因来源无关，传给 LLM 会造成参考代码和基因指令不一致。单一参考父代直接从 gene_plan 推断，语义更清晰。

### Mutate：随机选 + Stub 过滤 + Tournament 选 parent

```
select_non_stub_gene(node):
  # gene 内容 strip 后长度 < 20 字符视为 stub（如 "# Not applicable..."）
  candidates = [g for g in REQUIRED_GENES
                if len((node.genes.get(g) or "").strip()) >= 20]
  return random.choice(candidates) if candidates else random.choice(REQUIRED_GENES)
```

Mutate 保留 Tournament（k=3）选 parent：从 valid_pool 中选择适应度好的节点进行变异，避免在差解上浪费变异操作。

---

## 8. 链式 Debug（提高成功率）

### 链式传递（核心改动）

```
当前（非链式）:
  attempt 1: debug(original_code, original_error)   → v1_code（仍报错）
  attempt 2: debug(original_code, original_error)   ← 重复上下文，等同于 attempt 1

链式:
  attempt 1: debug(original_code, original_error)   → v1_code, v1_error
  attempt 2: debug(v1_code, v1_error)               ← 传上次改法 + 新报错，诊断更准
```

`debug_chain(node, max_attempts=2)` 实现：

```
def debug_chain(node, agent, max_attempts=2):
  current = node
  for _ in range(max_attempts):
    if current.exc_type is None: return current   # 成功
    current.debug_attempts += 1
    fixed = agent._debug(debug.j2,
                         buggy_code=current.code,     # 链式：上次修改后的代码
                         term_out=current.term_out)    # 链式：上次执行的报错
    exec(fixed)
    current = fixed
  if current.exc_type:
    current.dead = True
  return current                                        # 调用方负责 journal.append
```

注意：debug 中间产物**不追加 journal**，不占用总预算。

### debug.j2 新增两条约束

```
1. Preserve structure: Your fix MUST keep all 7 # [SECTION: X] markers intact.
2. Minimal change:     Change ONLY what is necessary. Do not refactor or add features.
```

---

## 9. 死节点机制

```
Node 新增字段:
  dead: bool = False
  debug_attempts: int = 0
  approach_tag: Optional[str] = None   # Review 写入，用于 Phase 1 多样性引导

标记条件: 1次主操作 + 2次链式 debug 全失败 → node.dead = True

valid_pool = [n for n in journal.nodes
              if not n.is_buggy and not n.dead and validate_genes(n.genes)]
```

dead 节点写入 journal（计入总预算），不参与任何选择，仅用于死亡率统计。

---

## 10. 总节点预算控制

**参数位置：**
- `config/mle_bench.yaml:43`：`agent.max_steps: ${env:STEP_LIMIT, 100}`
- `config/default.yaml:61`：`agent.max_steps: 150`

**计数规则：**

```
每次 draft_step()     → journal.append(1) ← 计入（dead 也算）
每次 phase2_step()    → journal.append(1) ← 计入（dead 也算）
debug_chain 内部产物  → 不追加 journal    ← 不计入
```

**两阶段共享预算：**

```
Phase 1 消耗: X 个节点（X ≥ steps_per_epoch，取决于达到 K=8 需要几轮）
Phase 2 预算: 100 - X 个节点
Phase 2 每步超出预算前自动停止
```

---

## 11. 基因 Schema：统一 7 基因

所有任务统一写 7 个 `# [SECTION: X]` 标记。表格任务（XGBoost/LightGBM）中 LOSS/OPTIMIZER/INITIALIZATION 写 stub：

```python
# [SECTION: LOSS]
# Not applicable — loss is handled internally by the model.

# [SECTION: OPTIMIZER]
# Not applicable — optimizer is handled internally by the model.

# [SECTION: INITIALIZATION]
# Not applicable — weight initialization is not user-configurable here.
```

**优点：**
- `validate_genes()` 逻辑不变，无需任务类型检测
- merge 时所有节点结构一致，无需特判
- mutate 时 `select_non_stub_gene()` 自动跳过 stub

废弃计划中的 `detect_task_type()` 方案。

---

## 12. draft.j2（原 explore.j2）简化

### 删除部分（两个场景在新架构不再触发）

```jinja
{% if parent_node.is_buggy %}
  "❌ Fix the bug... MemoryError/ProcessKilled/TimeoutError 分支"
{% else %}
  "✅ Improve with ONE atomic change..."
{% endif %}
```

- buggy 父节点 → Phase 1 内联 debug_chain 处理，不传给 draft.j2
- improve 模式 → Phase 2 的 mutate.j2 替代

### 保留 + 新增

- 纯 draft 指令（保留）
- `{% if draft_history %}` 多样性引导块（新增）
- `{% if memory %}` Evolution Log（保留）
- 环境信息、workspace_rules、code_style（保留）

---

## 13. 涉及变更文件

### `core/state/node.py` [MODIFY]
- `Node` dataclass 新增：`dead: bool = False`，`debug_attempts: int = 0`，`approach_tag: Optional[str] = None`

### `core/orchestrator.py` [MODIFY]
- `_try_immediate_debug()` → `_debug_chain(node, agent, max_attempts=2)` [RENAME+MODIFY]
  - 链式传递 `(current.code, current.term_out)` 而非固定传原始
  - 全部失败时 `node.dead = True`
- `_select_parent_node()` [MODIFY]：删除 buggy_leaf 和 improve 分支，Phase 1 始终返回 None
- `run_epoch_draft()` [NEW]：Phase 1 专用 epoch 入口，强制 parent=None
- `_build_draft_history(journal)` [NEW]：收集 approach_tag 列表
- `run()` 或 `main.py` 主循环 [MODIFY]：Phase 1/2 切换逻辑 + 预算控制

### `core/evolution/solution_evolution.py` [MODIFY]
- `run_epoch(max_steps)` [MODIFY]：Phase 2 主循环，接受预算参数
  - 删除精英保留逻辑（valid_pool 动态增长自然留精英）
  - merge：调用 `pheromone_with_degenerate_check()` + `get_primary_parent()`，**无 tournament**
  - mutate：保留 tournament（k=3），改用 `select_non_stub_gene()`
  - 成功节点动态追加 `valid_pool`
- `_crossover_mvp()` [MODIFY]：改调 `pheromone_with_degenerate_check(journal, gene_registry, step)`（移除 parent_a/b 参数）
- `_mutate_mvp()` [MODIFY]：target_gene 改用 `select_non_stub_gene()`

### `core/evolution/gene_selector.py` [MODIFY]
- 新增 `pheromone_with_degenerate_check(journal, gene_registry, current_step)` [NEW]
  - 签名移除 parent_a/b（无 tournament 约束）
  - 复用现有 `select_gene_plan()` + `build_decision_gene_pools()`
  - 加退化检测逻辑
- 新增 `get_primary_parent(gene_plan, journal)` [NEW]
  - Counter 统计各 source_node_id 贡献的基因数
  - 并列时选 metric 最高者

### `core/evolution/gene_parser.py` [MODIFY]
- 新增 `select_non_stub_gene(node: Node, stub_threshold=20) -> str` [NEW]

### `benchmark/mle-bench/prompt_templates/explore.j2` [RENAME → `draft.j2`]
- 删除 `parent_node.is_buggy` 和 improve 分支
- 新增 `{% if draft_history %}` 多样性引导块

### `benchmark/mle-bench/prompt_templates/merge.j2` [MODIFY]
- 从接收两个父代（parent_a, parent_b）改为接收单一 `primary_parent`
- context 变量名更新：`reference_code = primary_parent.code`

### `benchmark/mle-bench/prompt_templates/debug.j2` [MODIFY]
- 新增两条约束（保结构、最小修改）

### `core/agent/agent.py` 或 `core/prompt_manager.py` [MODIFY]
- 任务类型 `"explore"` → `"draft"` 全局替换
- 新增 `build_draft_history(journal) -> List[dict]` [NEW]

### `core/review/` [MODIFY]
- Review 输出 schema 新增 `approach_tag: str`
- Review prompt 增加提取指令 + 格式示例

### `config/default.yaml` + `config/mle_bench.yaml` [MODIFY]
- `evolution.solution` 新增：`phase1_target_nodes: 8`，`debug_max_attempts: 2`
- `SolutionEvolutionConfig` dataclass 对应新增 2 个字段

### `main.py` [MODIFY]
- 外层循环完整重构（见第 14 节伪代码）

---

## 14. 完整程序流程（伪代码）

```python
# ══════════════════════════════════════════════════════
# MAIN (main.py)
# ══════════════════════════════════════════════════════
def main():
    config        = load_config()           # max_steps=100, K=8, steps_per_epoch=10
    journal       = Journal()
    gene_registry = GeneRegistry()
    agents        = [Agent(cfg) for cfg in agent_configs]
    orchestrator  = Orchestrator(config, agents, journal, gene_registry)
    evo           = SolutionEvolution(config, journal, orchestrator, gene_registry)
    agent_evo     = AgentEvolution(config, agents)

    total_budget = config.agent.max_steps   # 100（两阶段共享）
    global_epoch = 0                        # 全局 epoch 计数（Phase 1 + Phase 2）

    # ─────────────────── Phase 1: Draft ───────────────────
    while True:
        orchestrator.run_epoch_draft(total_budget)   # 跑满一整 epoch
        global_epoch += 1
        if global_epoch % 3 == 0:
            agent_evo.evolve(global_epoch)           # Agent 进化（全局计数）

        valid_pool = get_valid_pool(journal)
        log(f"Phase 1 | epoch={global_epoch} nodes={len(journal.nodes)} "
            f"valid={len(valid_pool)} dead={count_dead(journal)}")

        if len(valid_pool) >= 8: break               # 种群目标达成
        if len(journal.nodes) >= total_budget: break # 预算耗尽
        if time_limit_reached(): break

    log(f"Phase 1 完成: journal={len(journal.nodes)}, valid_pool={len(valid_pool)}")

    # ─────────────────── Phase 2: Evolve ──────────────────
    while len(journal.nodes) < total_budget and not time_limit_reached():
        remaining = total_budget - len(journal.nodes)
        steps = min(config.evolution.solution.steps_per_epoch, remaining)
        evo.run_epoch(max_steps=steps)
        global_epoch += 1
        if global_epoch % 3 == 0:
            agent_evo.evolve(global_epoch)           # Agent 进化（全局计数）

        log(f"Phase 2 | epoch={global_epoch} nodes={len(journal.nodes)}")

    best = journal.get_best_node(only_good=True, lower_is_better=...)
    sync_submission(best)                            # 持久化提交文件


# ══════════════════════════════════════════════════════
# Phase 1: Draft Epoch (Orchestrator)
# ══════════════════════════════════════════════════════
def run_epoch_draft(total_budget):
    remaining = total_budget - len(journal.nodes)
    steps = min(steps_per_epoch, remaining)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(draft_step) for _ in range(steps)]
        wait_all(futures)

def draft_step():
    draft_history = build_draft_history(journal)
    node = agent.generate(
        template="draft.j2",
        parent=None,
        ctx={draft_history, task_desc, data_preview, env_info, ...}
    )
    exec(node)
    node = debug_chain(node, max_attempts=2)         # 链式 debug
    review(node)                                      # 写 approach_tag, metric
    journal.append(node)                              # ← 计入预算（dead 也算）
    gene_registry.update(node)


# ══════════════════════════════════════════════════════
# Debug Chain（两阶段共用）
# ══════════════════════════════════════════════════════
def debug_chain(node, max_attempts=2):
    current = node
    for _ in range(max_attempts):
        if current.exc_type is None:
            return current                            # 成功，提前退出
        current.debug_attempts += 1
        fixed = agent._debug(
            template   = "debug.j2",
            buggy_code = current.code,               # 链式：用上次修改后的代码
            term_out   = current.term_out,           # 链式：用上次的报错
        )
        exec(fixed)
        current = fixed
    if current.exc_type:
        current.dead = True
    return current                                    # 调用方负责 journal.append


# ══════════════════════════════════════════════════════
# Phase 2: Evolve Epoch (SolutionEvolution)
# ══════════════════════════════════════════════════════
def run_epoch(max_steps):
    valid_pool = get_valid_pool(journal)              # not buggy, not dead, has genes
    if len(valid_pool) < 2:
        log("WARNING: valid_pool 不足，跳过")
        return

    for _ in range(max_steps):
        op = random.choice(["merge", "mutate"])       # 50/50

        if op == "merge":
            gene_plan = pheromone_with_degenerate_check(
                            journal, gene_registry, current_step)
            primary_parent = get_primary_parent(gene_plan, journal)
            node = agent.generate("merge.j2", primary_parent, gene_plan)
            node.parent_id = primary_parent.id
            node.metadata["gene_sources"] = {
                locus: gene_plan[field]["source_node_id"]
                for locus, field in LOCUS_TO_FIELD.items()
            }

        else:  # mutate
            parent  = tournament_select(valid_pool, k=3)
            target  = select_non_stub_gene(parent)    # 过滤 stub（<20 字符）
            node = agent.generate("mutate.j2", parent, target)
            node.parent_id = parent.id

        exec(node)
        node = debug_chain(node, max_attempts=2)
        review(node)
        journal.append(node)                          # ← 计入预算（dead 也算）
        gene_registry.update(node)
        if not node.dead:
            valid_pool.append(node)                   # 动态增长，本步立刻可用


# ══════════════════════════════════════════════════════
# Gene 选择（gene_selector.py）
# ══════════════════════════════════════════════════════
def pheromone_with_degenerate_check(journal, gene_registry, current_step):
    # Step 1: 标准 pheromone 全局 TOP-1（复用现有 select_gene_plan，无需传父代）
    gene_plan = select_gene_plan(journal, gene_registry, current_step)

    # Step 2: 退化检测（7 基因是否全来自同一个节点）
    source_ids = {v["source_node_id"] for v in gene_plan.values() if isinstance(v, dict)}
    if len(source_ids) == 1:
        dominant_id = next(iter(source_ids))
        pools = build_decision_gene_pools(journal, gene_registry, current_step)
        for locus, field in LOCUS_TO_FIELD.items():
            others = [item for item in pools[locus]
                      if item.source_node_id != dominant_id]
            if others:
                second_best = max(others, key=lambda x: x.quality)
                gene_plan[field] = {
                    "locus": locus,
                    "source_node_id": second_best.source_node_id,
                    "code": second_best.raw_content,
                }
    return gene_plan


def get_primary_parent(gene_plan, journal):
    # 统计每个 source_node_id 贡献的基因数
    counter = Counter(
        v["source_node_id"] for v in gene_plan.values() if isinstance(v, dict)
    )
    max_count = max(counter.values())
    candidates = [nid for nid, cnt in counter.items() if cnt == max_count]
    if len(candidates) == 1:
        return journal.get_node(candidates[0])
    # 并列时选 metric 最高的节点
    return max(candidates,
               key=lambda nid: journal.get_node(nid).metric_value or 0.0)


# ══════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════
def get_valid_pool(journal):
    return [n for n in journal.nodes
            if not n.is_buggy and not n.dead and validate_genes(n.genes)]

def select_non_stub_gene(node, stub_threshold=20):
    candidates = [g for g in REQUIRED_GENES
                  if len((node.genes.get(g) or "").strip()) >= stub_threshold]
    return random.choice(candidates if candidates else REQUIRED_GENES)

def build_draft_history(journal):
    return [{"step": n.step, "approach_tag": n.approach_tag, "metric": n.metric_value}
            for n in journal.nodes if n.approach_tag and not n.dead]

def count_dead(journal):
    return sum(1 for n in journal.nodes if n.dead)
```

---

## 15. 新旧流程对比

```
现有流程（每步）:
  选父节点(None/buggy/best) → explore.j2 → exec → debug(1次,非链式) → review → journal

新流程（Phase 1，每步）:
  draft.j2(parent=None, draft_history) → exec → debug_chain(≤2次,链式) → review → journal
  [预算计数: +1]

新流程（Phase 2 merge，每步）:
  pheromone_with_degenerate_check() → get_primary_parent()
    → merge.j2(primary_parent, gene_plan)
    → exec → debug_chain(≤2次,链式) → review → journal → valid_pool.append(if alive)
  [预算计数: +1]  [parent_id = primary_parent.id, metadata.gene_sources = {...}]

新流程（Phase 2 mutate，每步）:
  tournament_select(k=3) → select_non_stub_gene()
    → mutate.j2(parent, target_gene)
    → exec → debug_chain(≤2次,链式) → review → journal → valid_pool.append(if alive)
  [预算计数: +1]  [parent_id = parent.id]
```

---

## 16. 验证计划

| 步骤 | 命令 / 检查点 |
|------|-------------|
| dead node 字段 | `pytest tests/unit/test_state.py -k dead` |
| debug_chain 链式传递 | `pytest tests/unit/test_orchestrator.py -k debug_chain` |
| select_non_stub_gene | `pytest tests/unit/test_gene_parser.py -k stub` |
| pheromone 退化检测 | `pytest tests/unit/test_gene_selector.py -k degenerate` |
| get_primary_parent 并列 | `pytest tests/unit/test_gene_selector.py -k primary_parent` |
| approach_tag review 输出 | `pytest tests/unit/test_review.py -k approach_tag` |
| config 新字段加载 | `conda run -n Swarm-Evo python -c "from utils.config import load_config; c=load_config('config/default.yaml'); print(c.evolution.solution.phase1_target_nodes)"` |
| Phase 1 终止日志 | `Phase 1 完成: journal=N, valid_pool=M (≥8)` |
| 预算控制 | 全程 journal.nodes 数量不超过 max_steps=100 |
| 死节点率 | Phase 1 后：`dead/(dead+valid) < 30%` |
| Agent 进化触发频率 | 全局每 3 epoch 触发一次，Phase 1 Phase 2 共享计数 |
| merge parent 记录 | `node.parent_id` 非 None，`node.metadata["gene_sources"]` 含 7 个 locus |
| 回归测试 | `conda run -n Swarm-Evo pytest tests/unit/ -x` |

---

## 17. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Phase 1 节点基因不完整（缺 `# [SECTION: X]`） | 中 | debug.j2 保结构约束；`validate_genes()` 校验，不通过不计入 valid_pool |
| Phase 1 多样性不足（全同一方法） | 中 | approach_tag 引导，明确要求 draft 不重复已有方向 |
| Pheromone 退化检测后仍无 second-best（种群太小） | 低 | 退化时若无其他来源，保留原全局最优（不强行替换） |
| select_non_stub_gene 所有基因均为 stub | 极低 | fallback: `random.choice(REQUIRED_GENES)` |
| Phase 1 耗尽预算仍不足 K=8 有效节点 | 低 | Phase 2 仍启动，可能导致效果下降；可调高 max_steps |
| approach_tag 提取格式不规范 | 低 | Review prompt 给出示例，LLM 结构化输出约束 |
| get_primary_parent 在种群极小时候选节点 metric 全 None | 低 | fallback: 取 gene_plan 中第一个 source_node_id |
