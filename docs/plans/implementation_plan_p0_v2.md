# P0 级系统 Bug 修复实施计划 (V2)

> **基于第二轮 22 竞赛实验分析，聚焦根因级修复**
>
> 与 V1 的区别：V1 假设"同一任务内所有节点的 lower_is_better 一致"，但实验证明 **LLM 在同一竞赛内对同一 metric 返回不一致的方向**（dogs-vs-cats 50%/50%、spooky-author 42%/58%）。V2 彻底废弃 LLM 逐节点判断机制，改为全局确定性方向锁定。

## 1.1 摘要

修复 3 个 P0 级系统 Bug + 1 个衍生 Bug：

1. **`lower_is_better` 极性翻转**（CRITICAL）：LLM Review 对同一 metric 返回不一致方向，导致 6+ 竞赛进化方向混乱，直接丢失 2~4 枚奖牌。**根因是方向来源不确定性**，需要彻底替换为确定性机制。
2. **`sample_submission` 文件名匹配过严**（HIGH）：仅匹配 2 个固定文件名，遗漏 `sample_submission_null.csv` 等变体，导致格式校验被绕过（detecting-insults 金牌→无效）。
3. **输入文件未受保护**（HIGH）：Agent 代码可删除/覆写 `input/` 下文件，导致后续节点级联失败（tabular-may-2022 仅 6% 成功率）。
4. **`main.py` best_node 比较硬编码 `>`**（衍生）：对 lower_is_better 指标完全错误。

## 1.2 审查点 (User Review Required)

> **✅ 已审核通过** — 调查结果如下：

### 1. P0-1 方向检测机制对比

**ML-Master 参考实现**：
- **方向来源**：LLM Function Calling 逐节点判断（与我们当前机制相同）
- **关键差异**：ML-Master 通过 `assert self.maximize == other.maximize` 强制一致性，若不一致直接报错
- **脆弱点**：正确性依赖首次 LLM 判断 — 若首个节点方向错误，后续所有比较都是"一致但错误"的

**Swarm-Ev2 V2 方案优势**：
- **确定性来源**：`METRIC_DIRECTION` 硬编码映射表（30+ metric）> 首次 review 的 metric_name 查表 > LLM lower_is_better 字段
- **鲁棒性提升**：正确性不依赖 LLM 判断，仅在映射表未覆盖时 fallback 到 LLM
- **借鉴 ML-Master**：保留一致性锁定机制（`_lock_metric_direction()` 的幂等语义）

**结论**：V2 方案优于 ML-Master，可直接实施。

### 2. P0-3 保护范围确认

**调查结果**：
- ✅ `workspace_rules.md` 明确标记 `input/` 为 `read-only`
- ✅ 所有 prompt 模板（explore/merge/mutate/debug）零处指导写入 input/
- ✅ workspace/ 下 40+ solution.py + Reference/mle-bench/runs/ 下 80+ solution.py → grep 零匹配写入 input/
- ✅ 框架代码仅在初始化阶段（link + preprocess）修改 input/，进入 Agent 执行后无修改路径

**结论**：保护 `input/` 全目录为 chmod 444 完全安全，零误杀风险。可直接实施。

## 1.3 拟议变更

---

### P0-1: 修复 `lower_is_better` 极性翻转 Bug [CRITICAL]

> **设计原则**：方向判断必须是全局、确定、不可变的。任何依赖 LLM 逐节点输出的方案都不可靠。
>
> **ML-Master 对比**：
> - ML-Master 使用 LLM Function Calling 逐节点判断，通过 `assert maximize == other.maximize` 强制一致性
> - 但若首个节点方向错误，后续所有节点都会"一致但错误"（正确性无保证）
> - Swarm-Ev2 V2 引入 `METRIC_DIRECTION` 映射表作为确定性来源，正确性不依赖 LLM
> - 借鉴 ML-Master 的一致性锁定机制，在 fallback 路径上也保证方向不可变

#### 文件 1: `core/orchestrator.py`

| 位置 | 操作 | 说明 |
|------|------|------|
| 模块级 `METRIC_DIRECTION` | `[NEW]` | 新增 metric 名称 → lower_is_better 的确定性映射表 |
| `__init__()` | `[MODIFY]` | 新增 `self._global_lower_is_better: Optional[bool]` 属性，调用 `_detect_metric_direction()` 初始化 |
| `_detect_metric_direction()` | `[NEW]` | 从 task_desc 中匹配 METRIC_DIRECTION 的 key，按 key 长度降序匹配（优先更具体的名称），返回 Optional[bool] |
| `_lock_metric_direction()` | `[NEW]` | 接收 review_data，若全局方向未锁定，从 metric_name 字段查 METRIC_DIRECTION 表锁定；再 fallback 到 lower_is_better 字段。已锁定时忽略 |
| `_review_node()` L675 | `[MODIFY]` | (a) 调用 `_lock_metric_direction(review_data)` 尝试锁定；(b) `node.lower_is_better` 赋值改用 `self._global_lower_is_better or False`，不再直接用 LLM 返回值 |
| `_is_better()` L1372-1388 | `[MODIFY]` | 使用 `self._global_lower_is_better` 替代 `node.lower_is_better`；全局方向未锁定时 fallback 到 node 值 |
| `_update_best_node()` L1261 | `[MODIFY]` | direction 日志改用全局方向 |
| 所有 `self.journal.get_best_node()` 调用 | `[MODIFY]` | 传入 `lower_is_better=self._global_lower_is_better` |
| 所有 `self.journal.get_best_k()` 调用 | `[MODIFY]` | 同上 |

**`METRIC_DIRECTION` 映射表**:

```python
# 确定性 metric 方向映射
# True = lower_is_better（越小越好: loss/error 类）
# False = higher_is_better（越大越好: score/accuracy 类）
METRIC_DIRECTION: Dict[str, bool] = {
    # === Lower is better ===
    "rmse": True,
    "root mean squared error": True,
    "rmsle": True,
    "mae": True,
    "mean absolute error": True,
    "mse": True,
    "mean squared error": True,
    "logloss": True,
    "log_loss": True,
    "log loss": True,
    "logarithmic loss": True,
    "cross-entropy": True,
    "cross entropy": True,
    "mcrmse": True,
    "medae": True,
    "mape": True,
    "smape": True,
    "pinball loss": True,
    "hinge loss": True,
    # === Higher is better ===
    "auc": False,
    "area under the roc curve": False,
    "area under the receiver operating characteristic": False,
    "accuracy": False,
    "categorization accuracy": False,
    "f1": False,
    "f1-score": False,
    "f1 score": False,
    "precision": False,
    "recall": False,
    "sensitivity": False,
    "specificity": False,
    "map": False,
    "mean average precision": False,
    "mean column-wise auc": False,
    "qwk": False,
    "quadratic weighted kappa": False,
    "kappa": False,
    "cohen's kappa": False,
    "ndcg": False,
    "r2": False,
    "r-squared": False,
    "r²": False,
    "spearman": False,
    "pearson": False,
    "correlation": False,
    "iou": False,
    "dice": False,
    "bleu": False,
    "rouge": False,
    "mean column-wise log loss": True,
    "multiclass loss": True,
}
```

**`_detect_metric_direction()` 伪代码**:

```python
def _detect_metric_direction(self) -> Optional[bool]:
    """从 task_desc 中检测 metric 方向（启动时调用一次）。

    策略: 在 task_desc 中搜索 METRIC_DIRECTION 所有 key，
    按 key 长度降序匹配（优先匹配更具体的名称如 "log loss" 而非 "loss"）。
    """
    text = (self.task_desc or "").lower()
    sorted_keys = sorted(METRIC_DIRECTION.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in text:
            direction = METRIC_DIRECTION[key]
            log_msg("INFO",
                f"[metric_direction] 从 task_desc 检测到: '{key}' → "
                f"lower_is_better={direction}")
            return direction
    log_msg("WARNING", "[metric_direction] task_desc 中未匹配到已知 metric")
    return None
```

**`_lock_metric_direction()` 伪代码**:

```python
def _lock_metric_direction(self, review_data: Dict) -> None:
    """从 review 结果中尝试锁定 metric 方向（仅在未锁定时生效）。

    借鉴 ML-Master 的一致性保证：方向一旦锁定，终生不可变。
    """
    if self._global_lower_is_better is not None:
        return  # 已锁定，幂等返回

    # 策略 1: metric_name → 查表（确定性来源）
    metric_name = (review_data.get("metric_name") or "").lower().strip()
    if metric_name:
        sorted_keys = sorted(METRIC_DIRECTION.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in metric_name or metric_name in key:
                self._global_lower_is_better = METRIC_DIRECTION[key]
                log_msg("INFO",
                    f"[metric_direction] 从 review metric_name 锁定: "
                    f"'{metric_name}' → lower_is_better={self._global_lower_is_better}")
                return

    # 策略 2: 直接使用 review 的 lower_is_better（fallback，与 ML-Master 同级）
    lib = review_data.get("lower_is_better")
    if isinstance(lib, bool):
        self._global_lower_is_better = lib
        log_msg("WARNING",
            f"[metric_direction] 使用 LLM 首次 review 的 lower_is_better={lib} "
            f"（未能从 metric_name 查表确认，fallback 到 LLM 判断）")
```

**`_review_node()` 关键修改**:

```python
# L675 附近，Phase 7 之后:
# 旧代码:
# node.lower_is_better = review_data.get("lower_is_better", False)

# 新代码:
# 尝试从本次 review 锁定全局方向（仅首次生效）
if not node.is_buggy:
    self._lock_metric_direction(review_data)
# 节点级 lower_is_better 始终使用全局值（保持序列化兼容）
node.lower_is_better = self._global_lower_is_better if self._global_lower_is_better is not None else review_data.get("lower_is_better", False)
```

**`_is_better()` 修改**:

```python
def _is_better(self, node: Node, best_node: Node) -> bool:
    if node.metric_value is None or best_node.metric_value is None:
        return False
    # 优先使用全局方向，fallback 到 node 级
    lower = self._global_lower_is_better if self._global_lower_is_better is not None else node.lower_is_better
    if lower:
        return node.metric_value < best_node.metric_value
    else:
        return node.metric_value > best_node.metric_value
```

#### 文件 2: `core/state/journal.py`

| 函数 | 操作 | 说明 |
|------|------|------|
| `get_best_node()` | `[MODIFY]` | 新增参数 `lower_is_better: Optional[bool] = None`，提供时覆盖 per-node 方向 |
| `get_best_k()` | `[MODIFY]` | 同上 |

```python
def get_best_node(self, only_good: bool = True, lower_is_better: Optional[bool] = None) -> Optional[Node]:
    candidates = self.good_nodes if only_good else self.nodes
    valid_nodes = [n for n in candidates if n.metric_value is not None]
    if not valid_nodes:
        return None

    # 使用传入的方向参数，否则 fallback 到第一个节点的方向
    lib = lower_is_better if lower_is_better is not None else valid_nodes[0].lower_is_better

    if lib:
        return min(valid_nodes, key=lambda n: n.metric_value)
    else:
        return max(valid_nodes, key=lambda n: n.metric_value)
```

`get_best_k()` 同理。

#### 文件 3: `core/evolution/solution_evolution.py`

| 函数 | 操作 | 说明 |
|------|------|------|
| `_is_lower_better()` | `[MODIFY]` | 优先从 `self.orchestrator._global_lower_is_better` 获取 |
| `run_epoch()` 末尾 | `[MODIFY]` | `journal.get_best_node()` 传入 `lower_is_better` |

```python
def _is_lower_better(self) -> bool:
    # 优先使用 Orchestrator 的全局方向（确定性来源）
    if self.orchestrator and self.orchestrator._global_lower_is_better is not None:
        return self.orchestrator._global_lower_is_better
    # Fallback: 种群中第一个有效节点
    for node in self.population:
        if node.metric_value is not None:
            return node.lower_is_better
    return False
```

#### 文件 4: `main.py`

| 位置 | 操作 | 说明 |
|------|------|------|
| L491-493 | `[MODIFY]` | 修复 `>` 硬编码为方向感知比较 |

```python
# 旧:
# if epoch_best and (
#     not best_node or epoch_best.metric_value > (best_node.metric_value or 0)
# ):

# 新:
if epoch_best and epoch_best.metric_value is not None:
    if best_node is None or best_node.metric_value is None:
        best_node = epoch_best
    else:
        lower = orchestrator._global_lower_is_better or False
        is_better = (
            epoch_best.metric_value < best_node.metric_value
            if lower
            else epoch_best.metric_value > best_node.metric_value
        )
        if is_better:
            best_node = epoch_best
```

---

### P0-2: 修复 `sample_submission` 文件名匹配

#### 文件: `core/orchestrator.py`

| 位置 | 操作 | 说明 |
|------|------|------|
| `_validate_submission_format()` L1024-1027 | `[MODIFY]` | 固定文件名匹配 → glob 模式匹配 |
| `_validate_submission_format()` L1037 | `[MODIFY]` | 列名不匹配时设 `valid=False`（当前仅 warning） |

**修改内容**:

```python
# 旧代码 (L1024-1027):
# sample_path = input_dir / "sample_submission.csv"
# if not sample_path.exists():
#     sample_path = input_dir / "sampleSubmission.csv"

# 新代码:
candidates = (
    list(input_dir.glob("sample_submission*.csv"))
    + list(input_dir.glob("sampleSubmission*.csv"))
    + list(input_dir.glob("sample_Submission*.csv"))
)
sample_path = candidates[0] if candidates else None

# ... 后续使用 sample_path:
if sample_path is not None and sample_path.exists():
    sample_df = pd.read_csv(sample_path)
    # 行数检查（保持原逻辑）
    if len(sub_df) != len(sample_df):
        result["valid"] = False
        result["errors"].append(...)
    # 列名检查（升级: 原来仅 warning，现在标记 invalid）
    if list(sub_df.columns) != list(sample_df.columns):
        result["valid"] = False  # [新增] 原来缺少此行
        result["errors"].append(...)
```

---

### P0-3: 保护输入文件

> **调查确认**：
> - workspace_rules.md 明确标记 input/ 为 read-only
> - 所有 prompt 模板和 agent configs 零处指导写入 input/
> - 已生成的 120+ solution.py 样本零匹配写入 input/ 的操作
> - 框架设计：所有输出流向 ./submission/ 和 ./working/
> - **结论**：保护 input/ 全目录为 chmod 444 完全安全，零误杀风险

#### 文件 1: `core/executor/workspace.py`

| 函数 | 操作 | 说明 |
|------|------|------|
| `protect_input_files()` | `[NEW]` | 将 input/ 下所有文件 chmod 444 |
| `prepare_workspace()` | `[MODIFY]` | 在预处理完成后调用 protect_input_files() |

```python
def protect_input_files(self) -> None:
    """将 input/ 下所有文件设为只读（0o444），防止 Agent 代码意外修改/删除。

    注意:
        - 跳过符号链接（symlink 指向外部文件，修改权限可能影响源文件）
        - 跳过目录（目录权限需要 execute bit 才能进入）
        - 在 preprocess_input() 之后调用，确保解压后的文件也被保护
    """
    import os
    import stat

    input_dir = self.workspace_dir / "input"
    if not input_dir.exists():
        return

    count = 0
    for path in input_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            current_mode = path.stat().st_mode
            readonly = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH  # 0o444
            if current_mode & 0o777 != readonly:
                os.chmod(path, readonly)
                count += 1

    if count > 0:
        log_msg("INFO", f"已保护 {count} 个输入文件（chmod 444）")
```

**`prepare_workspace()` 修改**: 在 Phase 3 预处理之后增加:

```python
# Phase 4: 保护输入文件（防止 Agent 代码修改/删除）
self.protect_input_files()
```

#### 文件 2: `main.py`

| 位置 | 操作 | 说明 |
|------|------|------|
| Phase 2 数据预处理完成后（~L375） | `[MODIFY]` | 调用 `workspace.protect_input_files()` |

---

## 1.4 验证计划

### 测试文件: `tests/unit/test_p0_v2.py` [NEW]

#### P0-1 测试用例

| 用例 | 说明 | 预期 | 验证 ML-Master 对比 |
|------|------|------|--------------------|
| `test_detect_direction_logloss` | task_desc 含 "log loss" | `_global_lower_is_better = True` | ✅ 映射表优于 LLM 判断 |
| `test_detect_direction_auc` | task_desc 含 "area under the ROC curve" | `_global_lower_is_better = False` | ✅ 映射表优于 LLM 判断 |
| `test_detect_direction_rmse` | task_desc 含 "RMSE" | `_global_lower_is_better = True` | ✅ 映射表优于 LLM 判断 |
| `test_detect_direction_accuracy` | task_desc 含 "accuracy" | `_global_lower_is_better = False` | ✅ 映射表优于 LLM 判断 |
| `test_detect_direction_qwk` | task_desc 含 "quadratic weighted kappa" | `_global_lower_is_better = False` | ✅ 长名称优先匹配 |
| `test_detect_direction_none` | task_desc 无已知 metric | `_global_lower_is_better = None` | ⚠️ fallback 到 LLM |
| `test_detect_direction_longer_match_first` | task_desc 同时含 "log" 和 "log loss" | 匹配更长的 "log loss" → True | ✅ 避免歧义匹配 |
| `test_lock_from_review_metric_name` | review 返回 metric_name="rmse" | 锁定为 True | ✅ 二次查表机会 |
| `test_lock_idempotent` | 已锁定后再次调用 | 不更改 | ✅ 借鉴 ML-Master |
| `test_lock_fallback_to_lower_is_better` | metric_name 未知，使用 lower_is_better 字段 | 使用该字段值 | ⚠️ 与 ML-Master 同级 fallback |
| `test_lock_consistency_check` | 首次锁定后，后续 review 返回不同方向 | 忽略新方向，保持原值 | ✅ 借鉴 ML-Master 一致性保证 |
| `test_is_better_lower` | lower=True, 0.5 vs 0.8 | True (0.5 < 0.8) | ✅ 与 ML-Master 逻辑等价 |
| `test_is_better_higher` | lower=False, 0.5 vs 0.8 | False | ✅ 与 ML-Master 逻辑等价 |
| `test_journal_get_best_with_direction` | 传入 lower_is_better=True | 返回最小值节点 | ✅ 显式方向覆盖 |
| `test_journal_get_best_without_direction` | 不传参数 | 保持原行为（向后兼容）| ✅ 向后兼容 |
| `test_solution_evolution_uses_global` | orchestrator 有全局方向 | SolutionEvolution 使用全局值 | ✅ 跨模块一致性 |

#### P0-2 测试用例

| 用例 | 说明 | 预期 |
|------|------|------|
| `test_glob_standard_name` | `sample_submission.csv` | 正常匹配 |
| `test_glob_camelcase` | `sampleSubmission.csv` | 正常匹配 |
| `test_glob_null_variant` | `sample_submission_null.csv` | 正常匹配 |
| `test_glob_no_match` | 无 sample_submission 文件 | `sample_path = None`，跳过对比 |
| `test_column_mismatch_invalid` | 列名不匹配 | `valid=False`（原来仅 warning） |

#### P0-3 测试用例

| 用例 | 说明 | 预期 |
|------|------|------|
| `test_protect_normal_files` | input/ 下有 csv | chmod 变为 444 |
| `test_protect_skip_symlinks` | input/ 下有 symlink | 不修改 |
| `test_protect_empty_dir` | input/ 为空 | 无操作，不报错 |
| `test_write_after_protect` | 保护后尝试写入 | `PermissionError` |

### 运行命令

```bash
# 新测试
conda run -n Swarm-Evo pytest tests/unit/test_p0_v2.py -v --tb=short

# 回归测试
conda run -n Swarm-Evo pytest tests/unit/ -v --tb=short

# 代码格式化
conda run -n Swarm-Evo ruff check core/ main.py --fix
conda run -n Swarm-Evo ruff format core/ main.py
```

---

## 附录

### 修改文件汇总

| 文件 | 修改类型 | 新增行数(估) | 风险等级 |
|------|---------|------------|---------|
| `core/orchestrator.py` | 新增 + 修改 | ~80 | 中（核心路径，需充分测试）|
| `core/state/journal.py` | 修改 | ~10 | 低（仅增加可选参数，向后兼容）|
| `core/evolution/solution_evolution.py` | 修改 | ~8 | 低 |
| `core/executor/workspace.py` | 新增 | ~20 | 低（独立功能，无副作用）|
| `main.py` | 修改 | ~12 | 低 |
| `tests/unit/test_p0_v2.py` | 新增 | ~200 | - |

### 预估收益

| Bug | 影响竞赛 | 修复后预期新增奖牌 | ML-Master 对比 |
|-----|---------|-----------------|---------------|
| P0-1 lower_is_better | dogs-vs-cats, spooky, dog-breed, leaf, denoising, new-york-taxi | **+2~4** | ✅ 比 ML-Master 更鲁棒（确定性映射表） |
| P0-2 sample_submission 匹配 | detecting-insults | **+1** | - |
| P0-3 输入文件保护 | tabular-may-2022 | **+0~1** | - |
| **合计** | | **+3~6** | |

### 向后兼容

- `Journal.get_best_node(lower_is_better=None)` 保持原行为
- `Node.lower_is_better` 字段保留（供日志/序列化）
- 已有 `journal.json` 可正常反序列化
- `METRIC_BOUNDS` 保持不变（范围校验），`METRIC_DIRECTION` 是独立新增的映射表

### 与 ML-Master 的对比总结

| 维度 | ML-Master | Swarm-Ev2 V2 | 优势 |
|------|-----------|--------------|------|
| 方向来源 | LLM Function Calling（逐节点） | 映射表 > LLM metric_name > LLM lower_is_better | V2 正确性更高 |
| 一致性保证 | `assert maximize == other.maximize` | `_lock_metric_direction()` 幂等锁定 | 等效，V2 借鉴 |
| 首次错误容忍 | ❌ 首次错误 → 全局错误 | ✅ 映射表覆盖 → 避免首次错误 | V2 更鲁棒 |
| Fallback 覆盖 | 2 层（Function Calling → 手动 JSON） | 3 层（映射表 → metric_name → lower_is_better） | V2 更完备 |
| 异常检测 | ✅ 50 倍比率阈值 | ✅ 已实现（METRIC_BOUNDS） | 等效 |
