# Bug1 修复: Prompt 截断导致 LLM 误读 Metric

## 1.1 摘要

修复 Review LLM 因 stdout 过长被截断而误读 metric 的问题。采用**预防 + 兜底**两层策略：
1. **预防**：在代码生成 prompt 中要求 solution.py 抑制冗余输出（tqdm 进度条、逐 epoch 日志等）
2. **兜底**：当 stdout 仍然超过 8000 chars 时，用前置 LLM 从完整输出中按固定格式提取 Review 和 Debug 所需的关键信息

同时**删除**旧的 `truncate_term_out()` 机械截断函数，不保留向后兼容。

## 1.2 审查点 (User Review Required)

无。格式设计已确认覆盖 Review 和 Debug 的全部需求。

## 1.3 根因回顾

```
stdout 原始 (最大 326K chars)
 └─ truncate_term_out(head=1500, tail=2000) → 仅保留 3500 chars
     └─ "Validation metric: 0.129663" 被截掉
         └─ Review LLM 误把 Prompt 中 "Current Best: 0.0211" 当成当前节点 metric
             └─ 记录 metric=0.021，实际 0.130 → 最佳节点选错 → denoising 丢银牌
```

## 1.4 拟议变更

### 1.4.1 变更 1: Prompt 中要求抑制冗余输出（预防层）

**`[MODIFY]` `benchmark/mle-bench/skills/static/code_style.md` — `## Output` 小节 (L140-158)**

将现有的模糊描述替换为明确的输出抑制规则：

```markdown
## Output

### Print Statements

Your solution's stdout is parsed by an automated review system. Excessive output (progress bars, per-epoch logs) can cause the review system to miss your validation metric. Follow these rules strictly:

**MUST**:
- Print data shape after loading: `print(f"Train shape: {train.shape}, Test shape: {test.shape}")`
- Print per-fold summary (1 line per fold): `print(f"Fold {i+1}: {metric_name}={value:.6f}")`
- Print final metric as the LAST informational line: `print(f"Validation metric: {mean_value:.6f}")`

**MUST NOT**:
- Print per-epoch training logs. Only print final epoch or every 10th epoch at most
- Use tqdm progress bars on stdout. Either disable them or redirect to stderr:
  ```python
  # Option 1: disable
  for batch in tqdm(loader, disable=True):
  # Option 2: redirect to stderr (won't pollute stdout)
  import sys
  for batch in tqdm(loader, file=sys.stderr):
  ```
- Use `verbose=1` or higher in Keras `model.fit()`. Use `verbose=0`
- Print large arrays, dataframes, or model summaries

**Target**: Total stdout should be under 5000 characters. The shorter, the better.
```

**`[MODIFY]` `benchmark/mle-bench/skills/static/workspace_rules.md` — L62**

将 `Add informative print statements for debugging (but keep output concise)` 替换为：

```markdown
- **Minimize stdout**: Disable tqdm (`disable=True`), suppress per-epoch logs (`verbose=0`).
  Only print: data shape, per-fold metric, and final `Validation metric: {value}`.
  Total stdout MUST be under 5000 characters — excessive output degrades automated evaluation.
```

### 1.4.2 变更 2: 前置 LLM 提取替代机械截断（兜底层）

#### Review LLM 从 stdout 中需要的信息

逐字段分析 `submit_review` schema，标注每个字段对 stdout 的依赖：

| Review 字段 | 依赖 stdout? | 需要提取什么 |
|-------------|-------------|-------------|
| `is_bug` | **是** | 是否有异常/错误、是否成功完成 |
| `has_csv_submission` | **是** | 是否有 "Saved submission" / "to_csv" 类输出 |
| `metric` | **是（核心）** | `Validation metric: {value}` 的精确数值 |
| `lower_is_better` | 否 | 由竞赛描述决定 |
| `metric_name` | **是** | 代码实际使用了什么评估指标（如 log_loss/auc/rmse） |
| `key_change` | 否 | 由 code diff 决定 |
| `insight` | **是** | 训练过程中的关键观察（收敛性、过拟合信号等） |
| `bottleneck` | **是** | 性能瓶颈线索（内存警告、训练速度、数据问题） |
| `suggested_direction` | 否 | 基于 insight 推断 |
| `approach_tag` | 否 | 由代码决定 |

#### Debug LLM 从 stdout 中需要的信息

Debug 模板 (`debug.j2`) 的目标是定位并修复 bug，它需要：

| Debug 需求 | 需要提取什么 |
|-----------|-------------|
| 错误定位 | 完整 traceback（文件名、行号、异常类型、异常消息） |
| 错误上下文 | 错误发生前的最后几行输出（判断执行到了哪一步） |
| 数据问题 | shape 信息、NaN/Inf 警告、类型错误线索 |
| 资源问题 | OOM 警告、CUDA 错误、内存不足信号 |

#### 前置 LLM 的固定输出格式

综合以上需求，设计统一的提取格式（同时服务 Review 和 Debug）：

```
=== EXECUTION SUMMARY ===

[STATUS]
success | error
(if error) ExceptionType: error message

[METRIC]
Validation metric: {exact_value}
Metric name: {metric_name_used_in_code}

[TRAINING]
- Data: train={rows}x{cols}, test={rows}x{cols}
- Model: {model_type} (e.g., LightGBM, ResNet18, LogisticRegression)
- CV: {n_folds}-fold, per-fold results: [fold1={v1}, fold2={v2}, ...]
- Training: {n_epochs} epochs, final train_loss={v}, val_loss={v}
- Convergence: converged | not converged | early stopped at epoch {n}

[WARNINGS]
- {any runtime warnings, deprecation notices, data quality issues}
- (or "None")

[ERROR_TRACE]
(if error) Full traceback:
{exact traceback text, preserving file names, line numbers, and error messages}

[OUTPUT_FILES]
submission.csv: created | not created
```

#### 实现

**`[NEW]` `utils/text_utils.py` — `condense_term_out()` 函数**

```python
CONDENSE_PROMPT_TEMPLATE = """You are a log parser. Extract structured information from the following ML solution execution output.

<execution_output>
{term_out}
</execution_output>

Respond in EXACTLY this format (preserve exact numerical values, do NOT round or interpret):

=== EXECUTION SUMMARY ===

[STATUS]
<"success" or "error">
<if error, copy the exact exception line: "ExceptionType: message">

[METRIC]
<copy the exact "Validation metric: ..." line from output, or "not found">
<metric name used in code, e.g. "rmse", "auc", "log_loss">

[TRAINING]
- Data: <train and test shapes if printed>
- Model: <model type/name>
- CV: <number of folds, per-fold metric values if available>
- Training: <total epochs, final losses if available>
- Convergence: <converged / not converged / early stopped>

[WARNINGS]
<any warnings, deprecation notices, or data issues — or "None">

[ERROR_TRACE]
<if error, copy the COMPLETE traceback verbatim — or "None">

[OUTPUT_FILES]
submission.csv: <"created" or "not created">

RULES:
- Copy values EXACTLY as they appear in the output. Do NOT round, reformat, or interpret.
- For [METRIC], look for lines matching "Validation metric: <number>". Copy the LAST occurrence.
- For [ERROR_TRACE], copy the full traceback starting from "Traceback (most recent call last):" to the final error line. Do NOT summarize.
- If a section has no relevant information in the output, write "N/A".
"""


def condense_term_out(
    term_out: str,
    max_len: int = 8000,
    llm_fn: Callable[[str], str],
) -> str:
    """当 stdout 过长时，用前置 LLM 按固定格式提取关键信息。

    Args:
        term_out: 完整终端输出
        max_len: 超过此长度触发 LLM 压缩（默认 8000 chars）
        llm_fn: LLM 调用函数 (prompt: str) -> str，必须提供

    Returns:
        原文（未超限）或 LLM 提取的结构化摘要

    Raises:
        LLM 调用异常会直接向上传播，不做降级处理
    """
    if not term_out or len(term_out) <= max_len:
        return term_out or ""

    prompt = CONDENSE_PROMPT_TEMPLATE.format(term_out=term_out)
    return llm_fn(prompt)
```

**`[DELETE]` `utils/text_utils.py` — `truncate_term_out()` 函数**

删除整个函数，不保留。

**`[NEW]` `core/orchestrator.py` — `_condense_output()` 方法**

```python
def _condense_output(self, term_out: str) -> str:
    """压缩执行输出，超长时用前置 LLM 提取关键信息。"""
    from utils.text_utils import condense_term_out

    def llm_fn(prompt: str) -> str:
        return backend_query(
            system_message=None,
            user_message=prompt,
            model=self.config.llm.feedback.model,
            provider=self.config.llm.feedback.provider,
            temperature=0.0,
            api_key=self.config.llm.feedback.api_key,
            base_url=getattr(self.config.llm.feedback, "base_url", None),
        )

    return condense_term_out(term_out, max_len=8000, llm_fn=llm_fn)
```

**`[MODIFY]` `core/orchestrator.py` — 三处调用点替换**

| 位置 | 场景 | 当前 | 改为 |
|------|------|------|------|
| L675 `_build_review_prompt_without_tool()` | Review 回退 | `truncate_term_out(node.term_out)` | `self._condense_output(node.term_out)` |
| L1054 `_build_review_messages()` | Review 主流程 | `truncate_term_out(node.term_out)` | `self._condense_output(node.term_out)` |
| L1508 `_debug_chain()` debug_context | Debug | `current.term_out`（完整传入） | `self._condense_output(current.term_out)` |

**`[DELETE]` `core/orchestrator.py` L28 — 移除旧导入**

```python
# 删除这行
from utils.text_utils import truncate_term_out
```

## 1.5 变更文件清单

| 文件 | 操作 | 函数级变更 |
|------|------|-----------|
| `benchmark/mle-bench/skills/static/code_style.md` | [MODIFY] | `## Output` 小节重写 |
| `benchmark/mle-bench/skills/static/workspace_rules.md` | [MODIFY] | Best Practices 中 stdout 要求强化 |
| `utils/text_utils.py` | [NEW] | `CONDENSE_PROMPT_TEMPLATE` 常量 + `condense_term_out()` 函数 |
| `utils/text_utils.py` | [DELETE] | `truncate_term_out()` 删除 |
| `core/orchestrator.py` | [NEW] | `_condense_output()` 方法 |
| `core/orchestrator.py` | [MODIFY] | `_build_review_prompt_without_tool()` L675 替换调用 |
| `core/orchestrator.py` | [MODIFY] | `_build_review_messages()` L1054 替换调用 |
| `core/orchestrator.py` | [MODIFY] | `_debug_chain()` L1508 debug_context 替换调用 |
| `core/orchestrator.py` | [DELETE] | L28 移除 `truncate_term_out` 导入 |

## 1.6 验证计划

### 单元测试

```bash
conda run -n Swarm-Evo pytest tests/unit/test_text_utils.py -v
```

新增用例：
1. `test_condense_short_output` — 短输出（<8000 chars）→ 原样返回，不调用 LLM
2. `test_condense_long_output_with_llm` — 长输出 + mock LLM → 返回 LLM 提取的结构化摘要，验证格式包含 `[STATUS]`, `[METRIC]`, `[TRAINING]`, `[ERROR_TRACE]` 等段
3. `test_condense_llm_failure_propagates` — LLM 抛异常 → 异常直接向上传播（不降级）
4. `test_condense_preserves_exact_metric` — 摘要中的 metric 值与原文完全一致（无四舍五入）

### 回归验证（用真实 output.txt）

```bash
# 用 denoising f799c849 (326K chars) 验证
conda run -n Swarm-Evo python -c "
from utils.text_utils import condense_term_out
data = open('workspace/denoising-dirty-documents_.../solution_f799c849/output.txt').read()
print(f'原始: {len(data)} chars')

# 使用真实 LLM 调用
from core.backend import query as backend_query
def llm_fn(prompt):
    return backend_query(system_message=None, user_message=prompt, ...)

result = condense_term_out(data, max_len=8000, llm_fn=llm_fn)
print(result)
# 验证: [METRIC] 段应包含 'Validation metric: 0.129663'
assert '0.129663' in result
"
```

### Prompt 规则验证

在下一次完整运行中观察：
- solution.py 中是否出现 `tqdm(disable=True)` 或 `verbose=0`
- stdout 长度分布是否显著下降（目标中位数 <5000 chars）
