# Prompt 系统重构计划 v2

> **目标**: 修复矛盾、补全缺失、消除冗余、删除低价值内容，提升信息密度。
>
> **原则**: Single Source of Truth | 每条规则只在一处定义 | 代码示例偏向 PyTorch + sklearn

---

## 1. 问题总览

| 类别 | 数量 | 关键问题 |
|------|------|---------|
| 矛盾 | 4 | `secondary_parent` optional 但无条件访问（会崩溃）；术语混用；debug 缺字段 |
| 缺失 | 2 | 代码示例全是 Keras/TF（应为 PyTorch/sklearn）；merge checklist 缺 TRAIN↔POSTPROCESS |
| 冗余 | 6 | strategy_merge/mutate 与 guide 三重定义；workspace_rules + code_style 重叠加载 |
| 低价值 | 3 | 伪图像转换示例、统计测试、过详细的 HeNormal 初始化 |

---

## 2. 矛盾修复

### M1. `secondary_parent` 改为 required [prompt_spec.yaml + merge.j2]

**现状**: `prompt_spec.yaml` 标记 `secondary_parent` 为 optional，但 `merge.j2:36-47` 无条件访问 `.code` 和 `.metric_value`，None 时 Jinja2 崩溃。

**方案**: `prompt_spec.yaml` 中将 `secondary_parent` 从 optional 移到 required。`coder_agent._merge()` 已经在传此字段（line 293），无需改代码。不影响其他 j2（spec 按 task_type 分别校验）。

**变更文件**:
- `prompt_spec.yaml`: [MODIFY] merge.optional_context 删除 secondary_parent，merge.required_context 添加 secondary_parent

### M2. 术语统一 "Evolution Log" [_macros.j2 + output_format.md]

**现状**: `_macros.j2` 标题 "# Evolution Log" 但指令说 "Read the **Changelog**"；`output_format.md` 两种并用。

**方案**: 全部统一为 **"Evolution Log"**。

**变更文件**:
- `_macros.j2`: [MODIFY] line 52 "Read the Changelog" → "Read the Evolution Log"
- `output_format.md`: [MODIFY] 所有 "Changelog" → "Evolution Log"

### M3. merge 依赖关系去重 [strategy_merge.md]

**现状**: `merge_guide.md:19-23` 和 `strategy_merge.md:17-20` 都定义了 DATA↔MODEL↔TRAIN 依赖关系，内容不一致（guide 有 TRAIN↔POSTPROCESS，strategy 有 loss_function↔MODEL）。

**方案**: 依赖关系由 `merge_guide.md` 统一定义（补上 loss_function↔MODEL 维度），`strategy_merge.md` 删除 "Compatibility Checks" 段。

**变更文件**:
- `merge_guide.md`: [MODIFY] Handle Dependencies 段补充 "loss_function compatibility" 维度
- `strategy_merge.md`: [MODIFY] 删除 "Compatibility Checks" 和 "Syntax and Execution" 段

### M4. orchestrator._debug_chain 补传字段 [orchestrator.py]

**现状**: `orchestrator.py:1361-1372` 的 debug_context 缺少 `time_remaining`、`steps_remaining`、`exec_timeout`。渲染结果为 "Total Time Remaining: 0 seconds"、"Total Steps Remaining: "（空），对 LLM 是错误信息。

**方案**: `_debug_chain()` 中从 context 获取并传入这三个字段。

**变更文件**:
- `core/orchestrator.py`: [MODIFY] `_debug_chain()` 的 debug_context 新增:
  ```python
  "time_remaining": self._calc_time_remaining(context),
  "steps_remaining": self._calc_steps_remaining(context),
  "exec_timeout": context.exec_timeout,
  ```
  需确认 orchestrator 中是否已有时间计算方法，若无则复用 coder_agent._calculate_remaining 的逻辑。

---

## 3. 缺失修复

### D1. 代码示例改为 PyTorch + sklearn [mutate_guide.md + merge_guide.md]

**现状**: `mutate_guide.md` 全部代码示例用 Keras（`Sequential`, `Dense`, `Adam`）；`merge_guide.md` 冲突示例用 TF（`tf.nn.softmax`, `tf.keras.losses`）。我们不希望 agent 生成 Keras/TF 代码。

**方案**: 所有代码示例替换为 PyTorch + sklearn 等价物。

**变更文件**:
- `mutate_guide.md`: [MODIFY] 全部 Keras 示例 → PyTorch (nn.Module, nn.Linear, optim.Adam 等)
  - Small Perturbation: `optim.Adam(model.parameters(), lr=0.001 * random.uniform(0.5, 2.0))`
  - Component Swap: `nn.LeakyReLU(0.2)` 替代 `nn.ReLU()`
  - Structural Change: 添加 `nn.Linear` 层
  - Bottleneck-Specific: 全部改为 PyTorch 风格
- `merge_guide.md`: [MODIFY]
  - Conflict Type 2 Model Output: `F.softmax(logits)` 或 `nn.CrossEntropyLoss()` (expects raw logits)
  - Conflict Type 3 Hyperparameter: `sum(p.numel() for p in model.parameters())`（此处已是 PyTorch ✓）

### D2. merge.j2 Validation Checklist 补充 TRAIN↔POSTPROCESS [merge.j2]

**现状**: checklist 有 "Model input/output shapes are compatible" 但未提及 fold ensemble 兼容性。

**方案**: 在 merge.j2 checklist 中添加一条。

**变更文件**:
- `merge.j2`: [MODIFY] Validation Checklist 新增:
  ```
  - [ ] POSTPROCESS inference matches TRAIN strategy (fold ensemble if CV was used)
  ```

---

## 4. 冗余消除

### R1. strategy_merge.md 精简 [strategy_merge.md]

**现状**: 30行中大量与 merge_guide.md 和 merge.j2 checklist 重复。

| 段落 | 处理 |
|------|------|
| "Extract Gene Blocks" + "Assess Fitness" | 保留（agent 思考策略） |
| "Compatibility Checks" | [DELETE] 由 merge_guide.md 统一定义（见 M3） |
| "Syntax and Execution" | [DELETE] 由 merge.j2 checklist 覆盖 |
| "Documentation" | 精简为 1 行 |

**目标**: 30行 → ~15行

**变更文件**:
- `strategy_merge.md`: [MODIFY]

### R2. strategy_mutate.md 精简 [strategy_mutate.md]

**现状**: 22行，独立价值仅 `mutation_aspect` 的解释（line 5）和兼容性思考逻辑。

| 段落 | 处理 |
|------|------|
| mutation_aspect 解释 | 保留 |
| "Check Dependencies" | 保留（agent 思考策略） |
| "Syntax and Execution" | [DELETE] 由 mutate.j2 checklist 覆盖 |
| "Documentation" | 精简为 1 行 |

**目标**: 22行 → ~12行

**变更文件**:
- `strategy_mutate.md`: [MODIFY]

### R3. workspace_rules.md "Best Practices" 去重 [workspace_rules.md]

**现状**: "Best Practices" 段（6行）是 code_style.md 对应段（random seeds、edge cases、data types、stdout）的摘要版。两份在同一 prompt 中加载。

**方案**: 删除 workspace_rules.md 的 "Best Practices" 段。code_style.md 已包含详细版。workspace_rules.md 的 "Minimize stdout" 一句移入末尾 "Execution Behavior" 段作为一条规则。

**变更文件**:
- `workspace_rules.md`: [MODIFY] 删除 "### Best Practices" 段，将 stdout 控制要求移入 "### Execution Behavior"

### R4. _macros.j2 render_timeout_warning fold 建议去重 [_macros.j2]

**现状**: `_macros.j2:26` "Prefer fewer folds (3 instead of 5)" 与 `workspace_rules.md:34` "may run ≥3 of 5 folds" 重复。

**方案**: `_macros.j2` 的 fold 建议改为更泛化的表述（侧重"完成 > 精度"），具体 fold 数建议只在 workspace_rules.md 中保留。

**变更文件**:
- `_macros.j2`: [MODIFY] render_timeout_warning 最后一条改为:
  ```
  - **Budget rule**: A completed run with fewer folds/lower resolution is infinitely better than a timed-out full run. See workspace rules for fold guidelines.
  ```

### R5. code_style.md submission 路径去重 [code_style.md]

**现状**: `code_style.md:168` 的提交文件验证示例中硬编码了 `./submission/submission.csv`（虽有注释引用 workspace_rules.md）。workspace_rules.md 已是权威定义。

**方案**: code_style.md 的提交验证示例保留（有价值的验证代码模式），但将路径改为变量引用式注释。

**变更文件**:
- `code_style.md`: [MODIFY] 提交验证示例注释改为 `# Path defined in workspace_rules.md`，保持示例中的路径不变（因为 LLM 需要具体路径来生成代码，重复在此不可避免）

实际审视后，**此项保持不动**。删除路径反而会让 code_style.md 的示例不完整，而 LLM 需要具体值。此冗余是必要的。

### R6. code_style.md Validation metric 格式去重 [code_style.md]

**现状**: `code_style.md:145` "Print final metric as the LAST informational line (see workspace_rules.md for exact format)" —— 已经是引用而非重复定义。

**结论**: 此处已通过引用方式去重 ✓，**无需修改**。

---

## 5. 低价值内容清理

### L1. merge_guide.md 删除伪图像转换示例 [merge_guide.md]

**现状**: "Data Format Mismatch" 冲突示例建议将 tabular reshape 为 pseudo-images，实际中几乎不会发生。

**方案**: 删除伪图像转换代码示例，改为更实际的场景（如 one-hot vs label encoding 格式不匹配）。

**变更文件**:
- `merge_guide.md`: [MODIFY] Conflict Type 1 示例替换

### L2. role.md 删除统计测试 [role.md]

**现状**: "Use statistical tests to validate performance differences" — 竞赛中从不做。

**方案**: 删除此行。

**变更文件**:
- `role.md`: [MODIFY] 删除 line 24

### L3. mutate_guide.md 精简慢收敛修复 [mutate_guide.md]

**现状**: "Slow Convergence" 段（line 130-142）的 HeNormal 初始化示例过于详细，实际竞赛中非瓶颈。

**方案**: 精简为 2-3 行提示，删除完整代码示例。

**变更文件**:
- `mutate_guide.md`: [MODIFY] Slow Convergence 段精简

---

## 6. 信息权威归属表 (Single Source of Truth)

| 信息点 | 权威来源 | 引用方式 |
|--------|---------|---------|
| `Validation metric: {value}` 打印格式 | workspace_rules.md | code_style.md 通过文字引用 |
| `./submission/submission.csv` 路径 | workspace_rules.md | code_style.md 验证示例中使用（必要重复） |
| Section Marker 定义（4段 DATA/MODEL/TRAIN/POSTPROCESS） | code_style.md | 各 j2 checklist 引用 |
| CV 策略（KFold/GroupKFold/fold 数） | workspace_rules.md | _macros.j2 泛化引用 |
| stdout 控制（详细版） | code_style.md | workspace_rules.md 简要引用 |
| 超时强警告 | _macros.j2 render_timeout_warning | 所有 j2 通过宏调用 |
| 环境信息 | _macros.j2 render_env_info | 所有 j2 通过宏调用 |
| 变异策略 + 瓶颈诊断 | mutate_guide.md | strategy_mutate.md 只补充 agent 思考逻辑 |
| 交叉策略 + 依赖关系 + 冲突解决 | merge_guide.md | strategy_merge.md 只补充 agent 思考逻辑 |
| Checklist | 各 .j2 模板内联 | skill 文件中不重复 |

---

## 7. 变更文件汇总

| 文件 | 操作 | 对应项 |
|------|------|--------|
| `prompt_spec.yaml` | MODIFY | M1 |
| `_macros.j2` | MODIFY | M2, R4 |
| `output_format.md` | MODIFY | M2 |
| `merge_guide.md` | MODIFY | M3, D1, L1 |
| `strategy_merge.md` | MODIFY | M3, R1 |
| `core/orchestrator.py` | MODIFY | M4 |
| `mutate_guide.md` | MODIFY | D1, L3 |
| `merge.j2` | MODIFY | D2 |
| `strategy_mutate.md` | MODIFY | R2 |
| `workspace_rules.md` | MODIFY | R3 |
| `code_style.md` | 不变 | R5/R6 审视后无需改 |
| `role.md` | MODIFY | L2 |

共 **11 个文件** 需要修改，0 个新建，0 个删除。

---

## 8. 实施阶段

### Phase 1: 矛盾修复（直接影响正确性）
1. M1: prompt_spec.yaml secondary_parent → required
2. M2: 术语统一 "Evolution Log"
3. M3: merge 依赖关系去重（merge_guide 补全 + strategy_merge 删重复段）
4. M4: orchestrator._debug_chain 补传 time_remaining/steps_remaining/exec_timeout

### Phase 2: 缺失修复 + 低价值清理
5. D1: mutate_guide.md + merge_guide.md 代码示例 Keras→PyTorch/sklearn
6. D2: merge.j2 checklist 补 TRAIN↔POSTPROCESS
7. L1: merge_guide.md 删伪图像示例
8. L2: role.md 删统计测试
9. L3: mutate_guide.md 精简慢收敛

### Phase 3: 冗余消除
10. R1: strategy_merge.md 精简（30→15行）
11. R2: strategy_mutate.md 精简（22→12行）
12. R3: workspace_rules.md 删 "Best Practices" 段
13. R4: _macros.j2 fold 建议泛化

### Phase 4: 验证
14. 运行测试: `conda run -n Swarm-Evo pytest tests/test_evolution/test_prompt_manager.py tests/integration/test_prompt_system_integration.py -v`
15. 选 1 个竞赛，分别生成 draft/debug/merge/mutate prompt，人工审查渲染结果
16. 确认无 Keras/TF 代码残留: `grep -r "Sequential\|Dense\|tf\.\|keras\." benchmark/mle-bench/skills/`

---

## 9. 验证计划

| 验证项 | 方法 | 预期结果 |
|--------|------|---------|
| secondary_parent required | 单元测试：merge 缺少 secondary_parent 时 WARNING | prompt_spec 校验生效 |
| 术语统一 | grep "Changelog" benchmark/mle-bench/ | 0 匹配 |
| 无 Keras/TF 示例 | grep "Sequential\|Dense(\|tf\.\|keras\." skills/ | 0 匹配 |
| merge checklist 完整 | 人工审查 merge.j2 渲染输出 | 包含 TRAIN↔POSTPROCESS 检查项 |
| debug prompt 字段完整 | 从 orchestrator 路径触发 debug，检查渲染结果 | time_remaining 和 steps_remaining 有真实值 |
| 依赖关系不重复 | 人工审查 merge prompt 输出 | 只出现一次依赖关系列表 |
| 现有测试通过 | pytest | 全部通过 |
