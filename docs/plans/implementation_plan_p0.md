# P0 优先级问题实施方案

## 1.1 摘要

基于 MLE-Bench 13 竞赛实验分析，针对 5 个 P0 问题 + 1 个新发现的 Critical Bug 制定系统级修复方案。预期修复后在相同竞赛集上额外获得 **+3~5 枚奖牌**。

**新发现 Critical Bug**: `SolutionEvolution._select_elites()` 和 `_tournament_select()` 未处理 `lower_is_better`，导致 5/13 个竞赛（所有 lower_is_better 任务）的遗传算法**选择了最差个体**作为精英和父代。

## 1.2 审查点 (User Review Required)

> **已审核完毕** — 以下为用户反馈决定：

1. **P0-A**: ✅ 同一任务所有节点方向一致，使用第一个有效节点的 `lower_is_better`，不使用多数投票。
2. **P0-C**: ✅ **LLM 优先**，正则提取仅做数据收集。不一致时仍用 LLM 值，但 log 记录两者差异（含 node_id、LLM 值、正则值），事后分析哪个错得多再决定是否切换。
3. **P0-D**: ✅ 统一使用 K-Fold（k=5），DL 任务允许只运行部分折（如 5 折只跑 3 折），但使用折数必须 > k/2。
4. **P0-F**: ✅ Submission 校验失败直接标记 buggy。校验方式：程序化读取 submission.csv，与 sample_submission.csv 对比行数/列名，检查 NaN。

## 1.3 拟议变更

---

### P0-A: 修复 SolutionEvolution 的 lower_is_better 排序 Bug

**影响**: 5/13 竞赛（denoising, dog-breed, dogs-vs-cats, leaf-classification, nomad2018）的 GA 精英选择和锦标赛选择全部反向。

**文件**: `core/evolution/solution_evolution.py`

| 变更 | 函数 | 行号 | 类型 |
|------|------|------|------|
| 修复精英选择方向 | `_select_elites()` | L149-159 | `[MODIFY]` |
| 修复锦标赛选择方向 | `_tournament_select()` | L161-171 | `[MODIFY]` |
| 添加方向判断辅助方法 | `_is_lower_better()` | - | `[NEW]` |

**`_select_elites()` 修改 (L149-159)**:
```python
# 修改前:
sorted_pop = sorted(
    self.population, key=lambda n: n.metric_value or -1e9, reverse=True
)

# 修改后:
lower = self._is_lower_better()
if lower:
    # lower_is_better: 升序排列，取最小值
    sorted_pop = sorted(
        self.population, key=lambda n: n.metric_value if n.metric_value is not None else float('inf')
    )
else:
    # higher_is_better: 降序排列，取最大值
    sorted_pop = sorted(
        self.population, key=lambda n: n.metric_value if n.metric_value is not None else float('-inf'), reverse=True
    )
```

**`_tournament_select()` 修改 (L161-171)**:
```python
# 修改前:
winner = max(tournament, key=lambda n: n.metric_value or -1e9)

# 修改后:
lower = self._is_lower_better()
if lower:
    winner = min(tournament, key=lambda n: n.metric_value if n.metric_value is not None else float('inf'))
else:
    winner = max(tournament, key=lambda n: n.metric_value if n.metric_value is not None else float('-inf'))
```

**`_is_lower_better()` 新增**:
```python
def _is_lower_better(self) -> bool:
    """判断当前任务的 metric 方向。

    使用种群中第一个有效节点的 lower_is_better 属性。
    假设同一任务内所有节点方向一致。
    """
    for node in self.population:
        if node.metric_value is not None:
            return node.lower_is_better
    return False  # 默认 higher_is_better
```

---

### P0-B: 增加 Metric 合理性校验（范围 + 语义）

**影响**: 防止 leaf-classification 类 metric=0.0 幻觉导致种群污染。

**文件**: `core/orchestrator.py`

| 变更 | 函数 | 行号 | 类型 |
|------|------|------|------|
| 增加范围检查 | `_check_metric_plausibility()` | L871-898 | `[MODIFY]` |
| 新增 metric 范围映射 | `METRIC_BOUNDS` | 模块级常量 | `[NEW]` |

**新增模块级常量**:
```python
# Metric 合理性范围（用于防止 LLM 幻觉）
# 格式: { metric_keyword: (min_val, max_val, lower_is_better) }
METRIC_BOUNDS = {
    # Bounded metrics [0, 1]
    "auc": (0.0, 1.0, False),
    "accuracy": (0.0, 1.0, False),
    "f1": (0.0, 1.0, False),
    "precision": (0.0, 1.0, False),
    "recall": (0.0, 1.0, False),
    "map": (0.0, 1.0, False),  # mean average precision
    # Unbounded non-negative metrics
    "rmse": (0.0, None, True),
    "rmsle": (0.0, None, True),
    "mae": (0.0, None, True),
    "mse": (0.0, None, True),
    # Log loss: 理论范围 (0, +inf)，但实际合理范围 (0, 15)
    "logloss": (1e-7, 15.0, True),
    "log_loss": (1e-7, 15.0, True),
    # QWK: [-1, 1] 但通常 [0, 1]
    "qwk": (-1.0, 1.0, False),
    "kappa": (-1.0, 1.0, False),
}
```

**`_check_metric_plausibility()` 修改 (L871-898)**:

在现有的比率检查**之前**，增加绝对范围检查：

```python
def _check_metric_plausibility(self, metric: float) -> bool:
    # === 新增: Phase 1 - 绝对范围检查 ===
    # 从 task_desc 中匹配 metric 类型关键词
    task_lower = (self._task_desc_compressed or "").lower()
    for keyword, (min_val, max_val, _) in METRIC_BOUNDS.items():
        if keyword in task_lower:
            if min_val is not None and metric < min_val:
                log_msg("WARNING", f"Metric {metric} 低于 {keyword} 下界 {min_val}")
                return False
            if max_val is not None and metric > max_val:
                log_msg("WARNING", f"Metric {metric} 超过 {keyword} 上界 {max_val}")
                return False
            break  # 只匹配第一个关键词

    # === 新增: Phase 2 - logloss=0 特殊检查 ===
    # logloss 值严格大于 0，metric=0.0 在 logloss 场景下是幻觉
    if metric == 0.0 and self.best_node and self.best_node.metric_value:
        if self.best_node.metric_value > 0.01:  # 如果 best 是正常的正值
            log_msg("WARNING", f"Metric=0.0 疑似虚假值（best={self.best_node.metric_value}）")
            return False

    # === 原有: Phase 3 - 比率检查 ===
    # (保留原有逻辑)
```

---

### P0-C: 增加 stdout Metric 正则提取（交叉验证 LLM Review）

**影响**: 独立于 LLM 的 metric 来源，可检测 LLM Review 幻觉。

**文件**: `core/orchestrator.py`

| 变更 | 函数 | 行号 | 类型 |
|------|------|------|------|
| 新增 stdout 解析 | `_parse_metric_from_stdout()` | - | `[NEW]` |
| 交叉验证逻辑 | `_review_node()` | L589-613 | `[MODIFY]` |

**`_parse_metric_from_stdout()` 新增**:
```python
import re

def _parse_metric_from_stdout(self, term_out: str) -> Optional[float]:
    """从终端输出中正则提取 Validation metric 值。

    匹配格式: "Validation metric: {number}"

    Returns:
        提取的 metric 值，未匹配返回 None
    """
    if not term_out:
        return None
    # 匹配最后一个 "Validation metric:" 行（可能有多个 fold 的输出）
    matches = re.findall(
        r"Validation metric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        term_out
    )
    if matches:
        try:
            return float(matches[-1])  # 取最后一个（通常是平均值）
        except ValueError:
            return None
    return None
```

**`_review_node()` 修改 (在 Phase 5 异常值检测之后)**:

```python
# Phase 5.5 (新增): stdout metric 数据收集（LLM 优先，仅记录差异）
stdout_metric = self._parse_metric_from_stdout(node.term_out)
if stdout_metric is not None and metric_value is not None:
    diff = abs(stdout_metric - metric_value)
    if diff > max(abs(stdout_metric) * 0.01, 1e-6):
        # 不一致：仅记录，不覆盖（LLM 优先）
        log_json("metric_mismatch", {
            "node_id": node.id,
            "llm_metric": metric_value,
            "stdout_metric": stdout_metric,
            "diff": diff,
        })
        log_msg(
            "WARNING",
            f"Metric 不一致: LLM={metric_value}, stdout={stdout_metric}（保留 LLM 值）",
        )
elif stdout_metric is not None and metric_value is None:
    # LLM 未提取到但 stdout 有值：此时用 stdout 补位
    log_msg("INFO", f"LLM 未提取 metric，使用 stdout 值: {stdout_metric}")
    metric_value = stdout_metric
    review_data["metric"] = stdout_metric
```

> **设计意图**: LLM 优先，不一致样本通过 `log_json("metric_mismatch", ...)` 持久化到 `logs/metrics.json`，事后批量分析 LLM vs 正则的错误率再决定是否切换策略。仅当 LLM 返回 None 时用正则补位。

---

### P0-D: 强化 K-Fold CV 要求（Prompt 层）

**影响**: 统一验证策略，使不同 solution 的 metric 可比，减少过拟合。

**文件 1**: `benchmark/mle-bench/skills/static/workspace_rules.md`

| 变更 | 位置 | 类型 |
|------|------|------|
| 增加 Validation 规范段 | L30 之后 | `[MODIFY]` |

**新增内容**（在 `### Best Practices` 之后）:

```markdown
### Validation Requirements (MANDATORY)

> **CRITICAL**: The validation strategy MUST follow these rules:

1. **For sklearn/tabular models**: Use `StratifiedKFold(n_splits=5)` (classification) or `KFold(n_splits=5)` (regression). Report the **mean** of all folds.
2. **For deep learning models**: Still use K-Fold (k=5), but you may **only run a subset of folds** if training is slow. The number of folds actually executed **MUST be > k/2** (e.g., at least 3 out of 5). Report the mean of the executed folds.
3. **MUST use the competition's evaluation metric** for validation (NOT training loss):
   - If competition uses `log_loss`, validate with `sklearn.metrics.log_loss`
   - If competition uses `AUC`, validate with `sklearn.metrics.roc_auc_score`
   - If competition uses `RMSE`, validate with `sqrt(mean_squared_error)`
   - **NEVER** report training loss (e.g., Focal Loss, BCELoss) as validation metric
4. Print the validation metric: `print(f"Validation metric: {metric_value:.6f}")`
```

**文件 2**: `benchmark/mle-bench/agent_configs/agent_*/strategy_explore.md` (4 个文件)

| 变更 | 位置 | 类型 |
|------|------|------|
| 强化 Validation Strategy 段 | L43-46 | `[MODIFY]` |

**修改内容**:
```markdown
### Validation Strategy
- **MANDATORY**: Use K-fold cross-validation (k=5) for ALL models
- For deep learning: You may run only a subset of folds if training is slow, but **must run > k/2 folds** (at least 3 out of 5). Report the mean metric of executed folds.
- **CRITICAL**: Validation metric MUST match the competition's evaluation metric exactly
  - ❌ WRONG: Using Focal Loss value as "validation metric" for a log_loss competition
  - ✅ CORRECT: Using `sklearn.metrics.log_loss()` for a log_loss competition
- Monitor both training and validation metrics to detect overfitting
```

---

### P0-E: 增加 Loss/Metric 对齐检查（Review 层）

**影响**: 防止 dog-breed 类 Focal Loss ≠ logloss 的指标错位问题。

**文件**: `core/orchestrator.py`

| 变更 | 函数 | 行号 | 类型 |
|------|------|------|------|
| 增强 Review Tool Schema | `_get_review_tool_schema()` | L981-1039 | `[MODIFY]` |
| 增强 Review Prompt | `_build_review_messages()` | L900-979 | `[MODIFY]` |

**`_get_review_tool_schema()` 修改**: 增加 `metric_name` 字段

```python
# 在 "lower_is_better" 之后新增:
"metric_name": {
    "type": "string",
    "description": "验证集使用的评估指标名称（如 log_loss, auc, rmse），必须与竞赛要求一致。如果代码使用了与竞赛不同的指标（如用 Focal Loss 代替 log_loss），请在此说明。",
},
```

将 `"metric_name"` 加入 `required` 列表。

**`_build_review_messages()` 修改**: 在 prompt 末尾增加对齐检查提示

```python
# 在 "Call `submit_review` with your analysis." 之前增加:
prompt += """
**Metric Alignment Check**: Verify that the validation metric printed in output matches the competition's evaluation metric. If the code uses a different loss function for training (e.g., Focal Loss) but reports that loss as the validation metric instead of the actual competition metric (e.g., log_loss), set `is_bug=true` and explain in `key_change`.
"""
```

---

### P0-F: 增加 Submission 格式验证

**影响**: 防止 denoising 类提交行数不足导致无效提交。

**文件**: `core/orchestrator.py`

| 变更 | 函数 | 行号 | 类型 |
|------|------|------|------|
| 新增格式校验 | `_validate_submission_format()` | - | `[NEW]` |
| 集成到 review 流程 | `_review_node()` | L532-533 | `[MODIFY]` |

**`_validate_submission_format()` 新增**:

```python
def _validate_submission_format(self, node_id: str) -> Dict[str, Any]:
    """校验 submission.csv 的基本格式。

    检查项:
    1. 文件是否存在
    2. 是否有 NaN 值
    3. 行数是否与 sample_submission 一致（如果存在）

    Returns:
        {"valid": bool, "errors": List[str], "row_count": int}
    """
    result = {"valid": True, "errors": [], "row_count": 0}

    submission_path = (
        self.config.project.workspace_dir / "submission" / f"submission_{node_id}.csv"
    )
    if not submission_path.exists():
        result["valid"] = False
        result["errors"].append("submission.csv 不存在")
        return result

    try:
        import pandas as pd
        sub_df = pd.read_csv(submission_path)
        result["row_count"] = len(sub_df)

        # 检查 NaN
        nan_count = sub_df.isnull().sum().sum()
        if nan_count > 0:
            result["valid"] = False
            result["errors"].append(f"submission 包含 {nan_count} 个 NaN 值")

        # 检查与 sample_submission 的行数一致性
        sample_path = self.config.project.workspace_dir / "input" / "sample_submission.csv"
        if not sample_path.exists():
            sample_path = self.config.project.workspace_dir / "input" / "sampleSubmission.csv"

        if sample_path.exists():
            sample_df = pd.read_csv(sample_path)
            if len(sub_df) != len(sample_df):
                result["valid"] = False
                result["errors"].append(
                    f"行数不匹配: submission={len(sub_df)}, sample={len(sample_df)}"
                )
            # 列名检查（忽略顺序）
            if set(sub_df.columns) != set(sample_df.columns):
                result["errors"].append(
                    f"列名不匹配: submission={list(sub_df.columns)[:5]}, "
                    f"sample={list(sample_df.columns)[:5]}"
                )
                # 列名不匹配只警告不标记 invalid（可能是列顺序不同）

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"读取 submission 失败: {e}")

    return result
```

**`_review_node()` 修改**: 在 Phase 1 文件检查处集成

```python
# 原有 Phase 1:
has_submission = self._check_submission_exists(node.id)

# 修改为:
has_submission = self._check_submission_exists(node.id)
submission_validation = {"valid": True, "errors": []}
if has_submission:
    submission_validation = self._validate_submission_format(node.id)
    if not submission_validation["valid"]:
        log_msg("WARNING", f"Submission 格式异常: {submission_validation['errors']}")
        has_submission = False  # 格式无效等同于不存在
```

---

## 1.3 变更汇总表

| P0 编号 | 问题 | 修改文件 | 修改函数 | 类型 |
|---------|------|---------|---------|------|
| **A** | lower_is_better 排序 Bug | `core/evolution/solution_evolution.py` | `_select_elites()` | `[MODIFY]` |
| **A** | lower_is_better 排序 Bug | `core/evolution/solution_evolution.py` | `_tournament_select()` | `[MODIFY]` |
| **A** | lower_is_better 排序 Bug | `core/evolution/solution_evolution.py` | `_is_lower_better()` | `[NEW]` |
| **B** | Metric 合理性校验 | `core/orchestrator.py` | `_check_metric_plausibility()` | `[MODIFY]` |
| **B** | Metric 合理性校验 | `core/orchestrator.py` | `METRIC_BOUNDS` 常量 | `[NEW]` |
| **C** | stdout Metric 提取 | `core/orchestrator.py` | `_parse_metric_from_stdout()` | `[NEW]` |
| **C** | stdout Metric 提取 | `core/orchestrator.py` | `_review_node()` | `[MODIFY]` |
| **D** | K-Fold CV 强制 | `benchmark/mle-bench/skills/static/workspace_rules.md` | - | `[MODIFY]` |
| **D** | K-Fold CV 强制 | `benchmark/mle-bench/agent_configs/agent_*/strategy_explore.md` ×4 | - | `[MODIFY]` |
| **E** | Loss/Metric 对齐 | `core/orchestrator.py` | `_get_review_tool_schema()` | `[MODIFY]` |
| **E** | Loss/Metric 对齐 | `core/orchestrator.py` | `_build_review_messages()` | `[MODIFY]` |
| **F** | Submission 格式验证 | `core/orchestrator.py` | `_validate_submission_format()` | `[NEW]` |
| **F** | Submission 格式验证 | `core/orchestrator.py` | `_review_node()` | `[MODIFY]` |

**修改文件数**: 3 个代码文件 + 5 个 Prompt/配置文件 = **8 个文件**
**新增函数**: 3 个
**修改函数**: 7 个

---

## 1.4 验证计划

### 单元测试

**文件**: `tests/unit/test_p0_fixes.py` `[NEW]`

```
测试项                                  | 验证内容
---------------------------------------|------------------------------------------
test_select_elites_higher_is_better    | accuracy 场景选择最高值个体
test_select_elites_lower_is_better     | RMSE 场景选择最低值个体
test_tournament_select_lower_is_better | RMSE 场景锦标赛选最小值
test_metric_bounds_logloss_zero        | logloss=0.0 被标记为不合理
test_metric_bounds_auc_overflow        | AUC=1.5 被标记为不合理
test_metric_bounds_rmse_negative       | RMSE=-0.1 被标记为不合理
test_metric_bounds_normal_pass         | 正常值通过检查
test_parse_metric_stdout_single        | 单行 "Validation metric: 0.95" 正确提取
test_parse_metric_stdout_multi_fold    | 多折输出取最后一行
test_parse_metric_stdout_scientific    | 科学计数法 "1.23e-04" 正确提取
test_parse_metric_stdout_no_match      | 无匹配返回 None
test_metric_cross_validation_match     | LLM 和 stdout 一致时不覆盖
test_metric_cross_validation_mismatch  | LLM 和 stdout 不一致时用 stdout
test_submission_validation_nan         | 含 NaN 的 submission 标记无效
test_submission_validation_row_count   | 行数不匹配标记无效
test_submission_validation_normal      | 正常 submission 通过
```

### 运行命令

```bash
# 1. 单元测试
conda run -n Swarm-Evo pytest tests/unit/test_p0_fixes.py -v --tb=short

# 2. 代码检查
conda run -n Swarm-Evo ruff check core/orchestrator.py core/evolution/solution_evolution.py --fix
conda run -n Swarm-Evo ruff format core/orchestrator.py core/evolution/solution_evolution.py

# 3. 现有测试不回归
conda run -n Swarm-Evo pytest tests/unit/ -v --tb=short

# 4. 集成验证（dry-run 单个竞赛）
conda run -n Swarm-Evo python main.py \
    --data_dir datasets/public/aerial-cactus-identification \
    --agent.max_steps 5 \
    --agent.time_limit 600
```

### 预期结果

| 验证项 | 预期 |
|--------|------|
| P0-A 单元测试 | lower_is_better 场景选择最小值个体 |
| P0-B 单元测试 | logloss=0.0 被拒绝；AUC=1.5 被拒绝 |
| P0-C 单元测试 | stdout 解析成功；LLM 不一致时覆盖 |
| P0-F 单元测试 | NaN / 行数不匹配被拒绝 |
| 现有测试回归 | 全部通过（修改不影响现有接口） |
| 集成 dry-run | 5 步内产出 best_node，日志中可见新增校验输出 |

---

## 附录：影响评估

### P0-A 影响（Critical Bug 修复）

| 竞赛 | lower_is_better | 当前结果 | 预期修复效果 |
|------|----------------|---------|------------|
| nomad2018 | ✅ | 银牌 | 可能 → 金牌（当前已银牌说明 explore 补偿了 GA 缺陷） |
| dogs-vs-cats | ✅ | 超中位 | 可能 → 铜牌/银牌 |
| dog-breed | ✅ | 低于中位 | 可能 → 超中位/铜牌 |
| leaf-classification | ✅ | 低于中位 | 改善有限（根因是特征丢弃 + metric 幻觉） |
| denoising | ✅ | 无效提交 | 改善有限（根因是代码 bug） |

### P0-B/C 影响（Metric 校验 + stdout 提取）

- 直接防止 **leaf-classification 的 metric=0.0 幻觉事件**（最严重的单点失败）
- 为所有竞赛提供 LLM-独立的 metric 备份来源

### P0-D/E 影响（Prompt 强化）

- K-Fold CV 可减少 **内部-测试 Gap**（当前 11/13 竞赛内部乐观）
- Loss/Metric 对齐可防止 **dog-breed 类反向优化**

### P0-F 影响（Submission 验证）

- 直接防止 **denoising 的无效提交**（行数缺 30%）
- 在 Review 前即可检测格式问题，节省 LLM 调用
