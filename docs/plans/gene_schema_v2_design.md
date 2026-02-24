# 基因方案重设计：从 7 基因到 4 基因

> **状态**: 已实施 ✅

## 1. Context

V5 实验中 Phase 2（GA 进化）在 22 个竞赛中仅 1 个实际运行，根本原因是当前 7 基因验证通过率仅 17.9%。深层分析发现：

- 46% 节点用 sklearn/GBDT，根本不存在独立的 LOSS/OPTIMIZER/INITIALIZATION/REGULARIZATION 代码段
- 即使 DL 节点，96.8% 也将 loss/optimizer 嵌入 MODEL 或 TRAINING_TRICKS 段内
- 92% 通过 7/7 验证的节点中，LOSS/OPTIMIZER/INITIALIZATION 是纯 stub 注释

**用户核心需求**：基因应对应影响分数的独立可控因素，支持控制变量实验（变异一个基因、保持其他不变）和基因融合。

## 2. 新基因方案

```python
REQUIRED_GENES = ["DATA", "MODEL", "TRAIN", "POSTPROCESS"]
```

### 2.1 跨框架映射

| 基因 | 语义 | sklearn | GBDT (lgb/xgb/catboost) | DL (PyTorch/TF) |
|------|------|---------|--------------------------|------------------|
| **DATA** | 数据加载、清洗、特征工程、增强 | `read_csv`, 缺失值处理, 编码, 特征构造, `train_test_split` | 同 sklearn + 类别特征声明 | 同 + `Dataset/DataLoader`, 图像增强, Tokenizer |
| **MODEL** | 模型定义 + 全部配置声明（含 loss/optimizer/正则化/初始化） | `RandomForestClassifier(n_estimators=100, max_depth=10)` | `lgb.LGBMRegressor(objective='regression', learning_rate=0.05, reg_lambda=0.1)` | `nn.Module` 定义 + `criterion = CrossEntropyLoss()` + `optimizer = Adam(lr=...)` + 权重初始化 |
| **TRAIN** | 训练/拟合执行 + CV策略 + early stopping + LR调度 | `cross_val_score(model, X, y, cv=5)` 或 `model.fit(X_train, y_train)` | `lgb.train(params, dtrain, num_boost_round=1000, callbacks=[...])` | 训练循环: `for epoch: loss.backward(); optimizer.step()` + `ReduceLROnPlateau` |
| **POSTPROCESS** | 推理 + 后处理 + 集成 + 提交生成 | `model.predict(X_test)`, 阈值调优, `submission.to_csv()` | 同 sklearn + 特殊预测接口 | `model.eval(); torch.no_grad()`, TTA, 集成, 提交生成 |

### 2.2 为什么是这 4 个

| 基因 | 对分数的影响 | 独立可变性 | 所有框架都有实质内容？ |
|------|------------|-----------|---------------------|
| **DATA** | 35-45%（特征工程是竞赛最关键的得分因素之一） | ✅ 改特征不需要改模型 | ✅ 所有框架都需要数据加载和预处理 |
| **MODEL** | 40%（算法/架构选择决定性能上限） | ✅ 换模型不需要改数据管道 | ✅ sklearn 有构造函数，DL 有 Module 定义 |
| **TRAIN** | 10-15%（CV策略、early stopping、训练技巧） | ✅ 改 CV 策略/epoch 数不需要改模型定义 | ✅ sklearn 有 fit/cross_val_score，DL 有训练循环 |
| **POSTPROCESS** | 1-5%（集成、TTA、阈值优化可额外提升） | ✅ 改后处理不影响训练过程 | ✅ 所有框架都需要 predict + to_csv |

**为什么不保留旧的 LOSS/OPTIMIZER/REGULARIZATION/INITIALIZATION？**

- 它们在代码中**不独立存在**：sklearn 的正则化是构造函数参数（`C=0.1`），DL 的 optimizer/criterion 紧跟模型定义
- LLM 天然不会把它们分到独立的代码段——强制分离导致 82.1% 验证失败或 92% stub
- 合入 MODEL 后，mutate MODEL 时通过 `mutation_aspect` 参数指定子方面（如 "只改 optimizer 配置"）

### 2.3 预估通过率

V5 数据中各段覆盖率：
- DATA: 91-93%
- MODEL: 89-95%
- TRAINING_TRICKS（映射到 TRAIN）: 84-89%
- predict + to_csv（映射到 POSTPROCESS）: ~95%

保守估计 **4/4 通过率 > 70%**，对比当前 7/7 的 17.9%，提升约 **4 倍**。

## 3. Mutation Sub-aspect Targeting（精细化变异控制）

### 3.1 问题

4 基因粒度较粗，MODEL 基因在 DL 方案下可达 130-180 行。仅指定 "mutate MODEL" 可能导致 LLM 同时修改架构、optimizer、loss 等多个方面，违反控制变量原则。

### 3.2 实际实现

在 `gene_parser.py` 中定义了每个基因的子方面列表：

```python
GENE_SUB_ASPECTS = {
    "DATA": ["feature_engineering", "data_cleaning", "augmentation", "encoding"],
    "MODEL": ["architecture", "loss_function", "optimizer", "regularization"],
    "TRAIN": ["cv_strategy", "early_stopping", "lr_schedule", "epochs"],
    "POSTPROCESS": ["ensemble", "threshold", "tta"],
}
```

`select_mutation_target(node)` 返回 `(gene, aspect)` 元组：随机选择非 stub 基因后，再随机选择该基因的一个子方面。

`mutate.j2` 模板中通过 `{{ mutation_aspect }}` 变量注入，引导 LLM 只修改基因内的特定子方面：

```
**Block to Mutate**: `{{ target_gene }}`
**Aspect to Focus**: `{{ mutation_aspect }}` — Only modify this specific aspect.
Keep all other parts of this gene block UNCHANGED.
```

### 3.3 数据流

```
select_mutation_target(node)
  → (gene="MODEL", aspect="optimizer")
    → solution_evolution._run_mutate_step()
      → orchestrator.execute_mutate_task(parent, "MODEL", "optimizer")
        → AgentContext(target_gene="MODEL", mutation_aspect="optimizer")
          → coder_agent 传递给 mutate.j2 模板
```

### 3.4 优势对比

| 维度 | 旧 7 基因 | 新 4 基因 + sub-aspect |
|------|----------|----------------------|
| 变异 OPTIMIZER | 目标是独立段，但 62% 是 stub → 无效 | 目标是 MODEL 内 optimizer 子方面 → 有实质内容 |
| 变异精度 | 基因级（整段） | 子方面级（段内指定部分） → 更精细 |
| 跨基因隔离 | 7 边界（4 个是 stub） | 4 有效边界 + 段内 sub-aspect 引导 |

## 4. 进化操作适配

### 4.1 Merge

无本质变化。从 7 个 locus 各选 TOP-1 基因 → 改为 4 个 locus 各选 TOP-1 基因。pheromone 计算、退化检测逻辑自动跟随 `LOCUS_TO_FIELD`。

### 4.2 Mutate

改善显著：
- 旧方案：7 基因中 4-5 个是 stub，`select_non_stub_gene()` 只有 2-3 个有效候选
- 新方案：4 个基因全部有实质内容，4 个有效候选
- `select_mutation_target()` 返回 `(gene, aspect)` 元组，mutate prompt 自动聚焦到子方面

### 4.3 控制变量实验

| 变异目标 | 控制不变 | 实验效果 |
|----------|---------|---------|
| 变异 DATA（改特征工程） | MODEL + TRAIN + POSTPROCESS | 测试不同特征组合的影响 |
| 变异 MODEL（换算法/架构） | DATA + TRAIN + POSTPROCESS | 测试不同模型的影响 |
| 变异 MODEL [optimizer] | DATA + MODEL 其他部分 + TRAIN + POSTPROCESS | 精细测试不同 optimizer 的影响 |
| 变异 TRAIN（改 CV/early stopping） | DATA + MODEL + POSTPROCESS | 测试不同训练策略的影响 |
| 变异 POSTPROCESS（改集成/后处理） | DATA + MODEL + TRAIN | 测试不同后处理的影响 |

## 5. 向后兼容

**不需要。** 这是 MVP 项目，有 git 管理，不复用旧 journal 数据。直接改写所有旧的 7 基因代码，不保留 `GENE_ALIAS_MAP` 兼容层。

## 6. 已实施变更清单

### Phase A：核心基因系统（3 文件）

| 文件 | 函数/常量 | 操作 | 说明 |
|------|----------|------|------|
| `core/evolution/gene_parser.py` | `REQUIRED_GENES` | [MODIFY] | `["DATA","MODEL","LOSS","OPTIMIZER","REGULARIZATION","INITIALIZATION","TRAINING_TRICKS"]` → `["DATA","MODEL","TRAIN","POSTPROCESS"]` |
| | `GENE_SUB_ASPECTS` | [NEW] | 每个基因的可变异子方面字典 |
| | `select_non_stub_gene()` | [RENAME] | → `select_mutation_target()`，返回 `(gene, aspect)` 元组 |
| | `parse_solution_genes()` | 无修改 | 正则匹配不依赖基因名列表 |
| | `validate_genes()` | 自动跟随 | 遍历 `REQUIRED_GENES` |
| | `merge_genes()` | 自动跟随 | 遍历 `REQUIRED_GENES` |
| `core/evolution/gene_registry.py` | `_LOCUS_NAMES` | [MODIFY] | 7 → 4 |
| `core/evolution/gene_selector.py` | `LOCUS_TO_FIELD` | [MODIFY] | 7 → 4（`data_source, model_source, train_source, postprocess_source`） |
| | `pheromone_with_degenerate_check()` | [MODIFY] | 日志文本 "7 基因" → "4 基因" |

### Phase B：Prompt 模板（8 文件）

| 文件 | 操作 | 说明 |
|------|------|------|
| `benchmark/mle-bench/skills/static/code_style.md` | [MODIFY] | Section Markers 从 7 段重写为 4 段，加 sklearn/GBDT/DL 三框架注释 |
| `benchmark/mle-bench/prompt_templates/debug.j2` | [MODIFY] | "all 7 `# [SECTION: X]` markers" → "all 4 ... (DATA, MODEL, TRAIN, POSTPROCESS)" |
| `benchmark/mle-bench/prompt_templates/merge.j2` | [MODIFY] | Checklist "All 7 gene blocks" → "All 4 gene blocks" |
| `benchmark/mle-bench/prompt_templates/mutate.j2` | [MODIFY] | 新增 `{{ mutation_aspect }}` 变量：Aspect to Focus 段 + 条件渲染 |
| `benchmark/mle-bench/skills/by_task_type/mutate/mutation_strategies.md` | [MODIFY] | Example Mutations 按 4 基因 + sub-aspect 重写；Heuristics 从旧基因名映射到新的 gene[aspect] 格式 |
| `benchmark/mle-bench/skills/by_task_type/mutate/local_optimization.md` | [MODIFY] | 诊断决策树 + Heuristic Table 全部更新为 gene[aspect] 格式 |
| `benchmark/mle-bench/skills/by_task_type/merge/crossover_strategies.md` | [MODIFY] | 依赖关系从 DATA↔MODEL、MODEL↔LOSS、OPTIMIZER↔MODEL → DATA↔MODEL、MODEL↔TRAIN、TRAIN↔POSTPROCESS |
| `benchmark/mle-bench/skills/by_task_type/merge/conflict_resolution.md` | [MODIFY] | Gene Plan 示例 `{"DATA":"A","MODEL":"B","LOSS":"B",...}` → `{"DATA":"A","MODEL":"B","TRAIN":"B","POSTPROCESS":"A"}` |

### Phase C：辅助模块（4 文件）

| 文件 | 操作 | 说明 |
|------|------|------|
| `core/evolution/solution_evolution.py` | [MODIFY] | `select_non_stub_gene` → `select_mutation_target`；`_run_mutate_step()` 解包 `(gene, aspect)` 并传递 `mutation_aspect` |
| `core/orchestrator.py` | [MODIFY] | `execute_mutate_task()` 新增 `mutation_aspect: str = ""` 参数，传入 `AgentContext` |
| `agents/base_agent.py` | [MODIFY] | `AgentContext` 新增 `mutation_aspect: Optional[str] = None` 字段 |
| `agents/coder_agent.py` | [MODIFY] | mutate prompt 构建时传递 `mutation_aspect` 变量 |
| `core/evolution/gene_compatibility.py` | 无修改 | `gene_plan_choices` 通过参数传入，自动适配 |

### Phase D：测试更新（6 文件）

| 文件 | 操作 | 说明 |
|------|------|------|
| `tests/unit/test_gene_parser.py` | [REWRITE] | 全部重写为 `TestSelectMutationTarget`，测试 `(gene, aspect)` 元组返回值、sub-aspect 可达性 |
| `tests/unit/test_gene_selector.py` | [MODIFY] | 适配 4 基因（`majority_wins_3_vs_1`、`tie_broken_by_higher_metric` 等） |
| `tests/test_evolution/test_gene_parser.py` | [MODIFY] | 基因名 7→4；新增 `test_parse_sklearn_solution`、`test_parse_dl_solution` |
| `tests/test_evolution/test_gene_selector.py` | [MODIFY] | 基因计划从 7 字段 → 4 字段（`data_source, model_source, train_source, postprocess_source`） |
| `tests/test_evolution/test_gene_registry.py` | [MODIFY] | `_registry` 长度 7→4、`pools` 长度 7→4 |
| `tests/test_evolution/test_solution_evolution.py` | [MODIFY] | 节点 genes 字典从 7 基因改为 4 基因 |
| `tests/unit/test_gene_compatibility.py` | 无修改 | 测试仅用 DATA/MODEL，不依赖完整基因列表 |

### 其他

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/analyze_phase_switch.py` | [MODIFY] | `REQUIRED_GENES` 7→4 |

## 7. 验证结果

```
unit 测试:       29/29 通过 ✅
evolution 测试:  47/47 通过 ✅（本次修改相关的测试文件）
```

### 成功指标

| 指标 | V5 基线 | V6 目标 |
|------|---------|---------|
| 基因验证通过率 | 17.9% (7/7) | > 70% (4/4) |
| Phase 2 触发率 | 4.5% (1/22) | > 70% |
| 非 stub 基因占比 | ~8% | > 90% |
| 奖牌数 | 8/22 (36.4%) | >= 10/22 (45%+) |

## 8. 风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| LLM 仍不按 4 段标记生成 | 中 | `code_style.md` 强化了 4 段示例（含三框架注释），debug.j2/merge.j2 均明确 4 段 checklist |
| POSTPROCESS 被 LLM 内联到 TRAIN 中 | 中 | prompt 中强调 "inference and saving MUST be in POSTPROCESS" |
| MODEL 基因太大，mutate 粒度不够精细 | 低 | `mutation_aspect` 参数 + mutate.j2 的 "Aspect to Focus" 段引导 LLM 精细变异 |
