# Nomad 竞赛 Agent Prompt/Skill 进化效果深度分析

> **竞赛**: nomad2018-predict-transparent-conductors
> **Workspace**: `workspace/nomad2018-predict-transparent-conductors_ec0ba461-3745-4ddf-a159-8e3a8561f8f9`
> **模型**: glm-4.7
> **总预算**: 100 步 | **总耗时**: ~3.8 小时 | **有效节点**: 69/100
> **最终最佳**: d649a401, metric=-0.06585 (RMSLE, lower_is_better)

---

## 一、整体时间线

| 阶段 | 时间 | 事件 | 最佳 Metric | 备注 |
|------|------|------|-------------|------|
| Phase 1 Draft | 20:22-20:48 | 10个draft节点，3次刷新最佳 | 0.060208 | 纯draft探索 |
| Epoch 2 Hybrid | 20:48-21:06 | 首次GA（6 GA + 4 Draft） | **-0.06585** | mutate DATA基因刷新 |
| **Agent进化#1** | **21:33** | agent_3/1变异（参考agent_0） | 无变化 | Skill池: 0个 |
| Epoch 3-5 | 21:35-22:19 | 30个节点（含 Draft + GA） | 无变化 | — |
| **Agent进化#2** | **22:44** | agent_0/3变异（参考agent_2/1） | 无变化 | Skill池: 0个 |
| Epoch 6-8 | 22:47-23:19 | 30个节点 | 无变化 | — |
| **Agent进化#3** | **23:37** | agent_0/3变异（参考agent_1/2） | 无变化 | Skill池: 0个 |
| Epoch 9-11 | 23:39-00:08 | 最后31个节点 | **无变化** | — |

**核心结论**：Agent进化触发了3次，变异了8次 role + 24次 strategy，但对最终指标影响为 **零**。Epoch 2 之后的 80 个节点（占总量80%）没有产生任何改进。

---

## 二、Prompt/Skill 更新的具体问题

### 2.1 Skill 池始终为空

```
获取 Top-5 Skill（task_type=draft），实际返回 0 个   ← 出现100次，从头到尾
记录数不足（0 < 5），跳过 Skill 提取                ← explore/merge/mutate 全部跳过
Skill 池演化完成（当前 0 个 Skill）                  ← 3次进化，每次结果都是0
```

**根因**：经验池中的记录按 `task_type` 分类时，explore/merge/mutate 类型的记录不够 `min_cluster_size=5`，导致 `SkillExtractor` 从未提取出任何 Skill。

**影响链**：
```
SkillExtractor 阈值过高 → 无 Skill 提取 → SkillManager 始终空
  → inject_top_k_skills() 返回空字符串
    → 所有 Prompt 中的 dynamic_skills 部分为空
      → Agent 无法从历史成功案例中学习
```

### 2.2 Agent 利用率极度失衡

| Agent | 总任务 | Draft成功 | Draft失败 | Mutate成功 | Mutate失败 | Merge成功 | Merge失败 | 占比 |
|-------|--------|-----------|-----------|------------|------------|-----------|-----------|------|
| agent_0 | 89 | 26 | 4 | 21 | 1 | 28 | 3 | **89%** |
| agent_1 | 4 | 0 | 0 | 2 | 0 | 2 | 0 | 4% |
| agent_2 | 4 | 1 | 0 | 2 | 0 | 1 | 0 | 4% |
| agent_3 | 9 | 4 | 3 | 1 | 0 | 1 | 0 | 9% |

agent_0 几乎垄断了所有任务。进化机制变异的弱者 agent（agent_1/3）**几乎没有机会执行任务**来验证新 prompt 的效果。

**死循环**：
```
弱者没任务 → 无新数据 → 评分不变 → 继续被判为弱者 → 继续没任务
```

### 2.3 Agent 进化的评分悖论

Agent 评分公式为 `score = success_rate × avg_quality`，完全不考虑样本量：

| 进化轮次 | 精英 | 弱者 | 问题 |
|----------|------|------|------|
| Epoch 3 | agent_2(score=0.064, **1次任务**), agent_0(0.037, 26次) | agent_3(0.032, 3次), agent_1(0, **0次**) | agent_2 仅1次任务就成精英 |
| Epoch 6 | **agent_1**(0.076, 1次), agent_2(0.066, 4次) | **agent_0**(0.048, **50次**), agent_3(0.012, 5次) | agent_0 被变异！（89%的主力） |
| Epoch 9 | agent_1(0.071, 3次), agent_2(0.066, 4次) | agent_0(0.052, 76次), agent_3(0.018, 6次) | 同上 |

**Epoch 6 的荒谬局面**：执行了50+任务的 agent_0 被判为弱者（score=0.048），而仅执行1次的 agent_1 成为精英（score=0.076）。评分完全被小样本偏差主导。

---

## 三、GA 层的结构性问题（更致命）

### 3.1 Merge 100% 退化

```
退化检测触发次数: 36/36 (100%)
所有 Merge 的 primary_parent: d35ceb68 (同一个节点，每次都是)
```

**每一次** Merge 操作都触发了退化检测（4个基因全来自同一节点 1d690c78），然后被替换为 d35ceb68。基因池的多样性严重不足，Merge 操作沦为对同一个节点的反复组装。

退化修复的替代节点也高度同质：
- DATA → d35ceb68 (36次中36次)
- MODEL → d35ceb68 (36次中36次)
- TRAIN → 2e6765a8 (36次中36次)
- POSTPROCESS → d35ceb68 (36次中36次)

**每次 Merge 产出的代码结构几乎相同**，无法产生有效创新。

### 3.2 neg_RMSE Bug 再现

```
Review 完成: 节点 2594cd4f, is_buggy=False, metric=-0.063126, lower_is_better=True
新的最佳节点: d649a401, metric=-0.06585 ↓
最终提交: d649a401, metric=-0.06585
```

Metric 存为**负数**，这是 V6 分析报告中已知的 neg_RMSE Bug。在 `lower_is_better=True` 的逻辑下：
- `-0.06585 < 0.060208` → 系统认为 d649a401 更好
- 但实际 RMSLE 应该取绝对值 `0.06585`，**反而比** `0.060208` **更差**

这意味着**最终提交的方案可能不是真正的最优解**。Phase 1 的 d2c3ef8b (metric=0.060208) 可能才是真正最佳。

### 3.3 Metric Mismatch 事件

```json
{"event": "metric_mismatch", "node_id": "5f98d60a", "llm_metric": 0.0631, "stdout_metric": -0.0631, "diff": 0.1262}
{"event": "metric_mismatch", "node_id": "9b155a8c", "llm_metric": 0.0659, "stdout_metric": 0.069194, "diff": 0.003}
{"event": "metric_mismatch", "node_id": "c361bcfb", "llm_metric": 0.0643, "stdout_metric": 0.071183, "diff": 0.007}
```

节点 5f98d60a 的 diff=0.1262 正是正负号差异（LLM 提取了正值 0.0631，stdout 是负值 -0.0631），确认了 neg_RMSE Bug 的存在。

---

## 四、Prompt 注入机制验证

### 4.1 模板调用链

```
draft.j2 第7行:   {{ load_agent_config(agent_id, "role") }}
draft.j2 第67行:  {{ load_agent_config(agent_id, "strategy_explore") }}
mutate.j2 第7行:  {{ load_agent_config(agent_id, "role") }}
mutate.j2 第58行: {{ load_agent_config(agent_id, "strategy_mutate") }}
merge.j2 第7行:   {{ load_agent_config(agent_id, "role") }}
merge.j2 第55行:  {{ load_agent_config(agent_id, "strategy_merge") }}
```

**确认**：模板确实引用了 agent 配置。AgentEvolution 通过 `prompt_manager.update_agent_config()` 更新内存字典后，后续 `build_prompt()` 调用会使用新配置。**配置注入路径本身没有问题**。

### 4.2 为什么注入了却没效果

1. **agent_0 被变异但占据89%任务**：Epoch 6/9 变异了 agent_0 的 prompt，但 agent_0 本身已经是执行能力最强的（75次成功），变异后反而可能引入噪声
2. **agent_1/3 变异后无任务验证**：Epoch 3 变异了 agent_3/1，但它们后续的任务数极少（agent_1 总共4次），无法产生统计意义上的改善
3. **Prompt 变异内容泛泛**：变异 prompt 要求"300-500字"的泛化建议，但竞赛的瓶颈在于具体的算法选择和特征工程，泛化策略无法指导具体行为

---

## 五、根因排序与修复建议

### P0 - 必须修复

| 问题 | 根因 | 修复方案 |
|------|------|----------|
| 任务调度失衡 | 任务分配器过度利用 agent_0 | 引入公平调度约束：每个 agent 至少占 15% 任务 |
| 评分公式缺样本量加权 | `success_rate × avg_quality` 无置信度 | 改为 `score × min(1, total_count/10)` 或引入 Wilson 区间 |
| neg_RMSE Bug | sklearn neg_ 前缀 scoring 存为负数 | metric 存储时取绝对值 |
| Merge 完全退化 | 基因池多样性崩溃 | 退化时引入随机基因或强制跨代多样性 |

### P1 - 应该修复

| 问题 | 根因 | 修复方案 |
|------|------|----------|
| Skill 提取阈值过高 | `min_cluster_size=5` 在100步场景不可达 | 降到 2-3，或直接用 Top-K 成功案例注入 |
| 进化后无保障调度 | 变异了prompt但不保证执行 | 进化后强制给变异agent分配至少 N 次任务 |
| Prompt 变异过于泛化 | "300-500字"的泛化建议 | 注入具体竞赛信息（数据特征、已尝试方法、失败原因） |

### P2 - 可以优化

| 问题 | 根因 | 修复方案 |
|------|------|----------|
| 精英判定不稳定 | 精英/弱者每3轮翻转 | 引入EMA或滑动窗口平滑评分 |
| 经验池记录分类不足 | draft 记录不计入 explore 类型 | 统一或放宽分类逻辑 |

---

## 六、数据附录

### 6.1 最佳节点变更历史

| 时间 | 节点 | Metric | 来源 | 方法 |
|------|------|--------|------|------|
| 20:23 | aa08e34d | 0.068633 | Draft | XGBoost + 5-fold CV |
| 20:24 | 6f81d450 | 0.061213 | Draft | LightGBM + log1p + geometry features |
| 20:38 | d2c3ef8b | 0.060208 | Draft | — |
| 20:54 | 2594cd4f | -0.063126 | Mutate(DATA) | 父节点: 5f98d60a |
| 20:58 | d649a401 | -0.06585 | Mutate(DATA) | 父节点: 2594cd4f |

注意：后两个为负值（neg_RMSE Bug），实际 RMSLE 分别为 0.063126 和 0.06585，**比 d2c3ef8b 的 0.060208 更差**。

### 6.2 每 Epoch 统计

| Epoch | 节点数 | 有效 | Draft | GA | Best | 进化事件 |
|-------|--------|------|-------|-----|------|----------|
| 1 | 10 | 8 | 10 | 0 | 0.060208 | — |
| 2 | 20 | 14 | 4 | 6 | -0.06585 | — |
| 3 | 30 | 19 | 3 | 7 | -0.06585 | Agent进化#1 |
| 4 | 40 | 25 | 5 | 5 | -0.06585 | — |
| 5 | 50 | 30 | 4 | 6 | -0.06585 | — |
| 6 | 60 | 39 | 3 | 7 | -0.06585 | Agent进化#2 |
| 7 | 69 | 45 | 2 | 7 | -0.06585 | — |
| 8 | 79 | 53 | 2 | 8 | -0.06585 | — |
| 9 | 89 | 62 | 1 | 9 | -0.06585 | Agent进化#3 |
| 10 | 99 | 68 | 3 | 7 | -0.06585 | — |
| 11 | 100 | 69 | 1 | 0 | -0.06585 | — |

### 6.3 Agent 评分变化（3轮进化）

| Agent | Epoch 3 Score | Epoch 6 Score | Epoch 9 Score | 总任务 |
|-------|---------------|---------------|---------------|--------|
| agent_0 | 0.037 (26次) | 0.048 (50次) | 0.052 (76次) | 89 |
| agent_1 | 0.000 (0次) | 0.076 (1次) | 0.071 (3次) | 4 |
| agent_2 | 0.064 (1次) | 0.066 (4次) | 0.066 (4次) | 4 |
| agent_3 | 0.032 (3次) | 0.012 (5次) | 0.018 (6次) | 9 |
