# Memory 进化机制重构计划

> **目标**: 让 Agent 能够有效利用历史信息，实现 Kaggle 金牌级别 (Top 1%) 的持续进化

## 1. 问题诊断

### 1.1 因果链分析

```
ROOT CAUSES (根因)
├── ① Review Tool Schema 过于简单
│   └── 只要求 "2-3 句话的摘要"
├── ② Review 缺少对比上下文 [NEW]
│   └── 不知道父节点代码，无法分析"改了什么"
├── ③ generate_summary() 信息丢失
│   └── 只保留 plan + analysis + metric
└── ④ 缺失进化路径表达
    └── 每条记录是独立快照，无 diff/delta

        ↓ 导致

INTERMEDIATE EFFECTS (中间效应)
├── ⑤ Result 信息密度极低
├── ⑥ 历史 Thinking 高度同质化
└── ⑦ LLM 失去探索方向

        ↓ 导致

FINAL SYMPTOMS (最终现象)
├── ⑧ Thinking 完全相同（复制历史）
├── ⑨ 后期连续 BUGGY（重复相同错误）
└── ⑩ 进化停滞（远非金牌水平）
```

### 1.2 当前 vs 目标

| 维度 | 当前状态 | 目标状态 |
|------|---------|---------|
| Review 输入 | 只有当前代码 | 当前代码 + **代码 Diff** |
| Result 信息量 | "成功执行了...RMSE 是 X" | 包含 delta、insight、bottleneck |
| 进化路径 | 独立快照 | Changelog 格式 (类似 git log) |
| 错误学习 | [BUGGY] 标记 + 错误描述 | 显式 Constraints 列表 |
| 探索引导 | 无 | Unexplored Directions 清单 |
| 目标对齐 | 隐含 "比 baseline 好" | 显式 "Gold Medal = Top 1%" |

---

## 2. 修改方案

### 2.0 Phase 0: 代码 Diff 生成 (核心前置) [NEW]

> **核心思想**: 在 Review 之前，用 `difflib` 自动生成父子代码的 diff，让 Review LLM 能准确知道"改了什么"

**文件**: `core/orchestrator.py`

**修改点 1**: 新增 `_generate_code_diff()` 方法

```python
import difflib

def _generate_code_diff(
    self,
    parent_code: Optional[str],
    current_code: str,
    context_lines: int = 3
) -> str:
    """生成父子代码的 unified diff。

    Args:
        parent_code: 父节点代码（None 表示首次生成）
        current_code: 当前节点代码
        context_lines: diff 上下文行数（默认 3）

    Returns:
        unified diff 格式字符串，首次生成返回 "(Initial solution, no diff)"
    """
    if parent_code is None:
        return "(Initial solution, no diff)"

    parent_lines = parent_code.splitlines(keepends=True)
    current_lines = current_code.splitlines(keepends=True)

    diff = difflib.unified_diff(
        parent_lines,
        current_lines,
        fromfile="parent_solution.py",
        tofile="current_solution.py",
        n=context_lines,
    )

    diff_text = "".join(diff)

    # 如果 diff 过长，截断并提示
    max_diff_lines = 100
    diff_lines = diff_text.splitlines()
    if len(diff_lines) > max_diff_lines:
        diff_text = "\n".join(diff_lines[:max_diff_lines])
        diff_text += f"\n... (truncated, {len(diff_lines) - max_diff_lines} more lines)"

    return diff_text if diff_text.strip() else "(No changes detected)"
```

**修改点 2**: 新增 `_format_gene_selection()` 方法（merge 任务专用）

> **设计理由**: merge 任务的"关键变更"不是代码 diff，而是"基因选择方案"。直接展示每个位点选了哪个 parent 的哪段代码，比生成 diff 更能表达 merge 的语义。

```python
from core.evolution.gene_selector import LOCUS_TO_FIELD

def _format_gene_selection(self, gene_plan: Dict) -> str:
    """格式化基因选择方案（用于 merge Review）。

    将 gene_plan 转换为人类可读的格式，展示每个基因位点：
    - 选自哪个父节点
    - 具体的代码片段

    Args:
        gene_plan: 基因选择计划，包含 data_source, model_source 等字段

    Returns:
        格式化的基因选择说明字符串
    """
    lines = ["## Gene Selection\n"]

    for field in LOCUS_TO_FIELD.values():  # data_source, model_source, ...
        if field not in gene_plan:
            continue

        item = gene_plan[field]
        locus = item["locus"]              # "DATA", "MODEL", ...
        node_id = item["source_node_id"]   # 来源节点 ID
        code = item["code"]                # 基因片段代码

        lines.append(f"### {locus} (from Node {node_id[:8]})")
        # 截断过长的代码片段，防止占用过多 token
        code_preview = code[:500] + "..." if len(code) > 500 else code
        lines.append(f"```python\n{code_preview}\n```\n")

    return "\n".join(lines)
```

**修改点 3**: 修改 `_review_node()` 签名，支持不同任务类型的变更上下文

```python
# 当前 (第 441 行)
def _review_node(self, node: Node) -> None:

# 修改为
def _review_node(
    self,
    node: Node,
    parent_node: Optional[Node] = None,
    gene_plan: Optional[Dict] = None,  # merge 任务专用
) -> None:
    """Review 评估节点（多层验证 + 回退机制）。

    Args:
        node: 待评估的节点对象
        parent_node: 父节点（用于 explore/mutate 的代码 diff）
        gene_plan: 基因选择计划（用于 merge 的变更上下文）
    """
    # Phase 0: 生成变更上下文（根据任务类型选择策略）
    if gene_plan:
        # merge 模式：展示基因选择方案
        change_context = self._format_gene_selection(gene_plan)
    elif parent_node:
        # explore/mutate 模式：代码 diff
        change_context = self._generate_code_diff(parent_node.code, node.code)
    else:
        # 初稿模式
        change_context = "(Initial solution, no diff)"

    # Phase 1: 文件存在检查
    has_submission = self._check_submission_exists(node.id)

    # Phase 2: LLM Function Calling (传入变更上下文)
    review_data = None
    try:
        review_data = self._call_review_with_tool(node, change_context)
    # ... 后续逻辑不变
```

**修改点 4**: 修改 `_step_task()` 调用，传递 `parent_node`

```python
# 当前 (第 335 行)
self._review_node(node)

# 修改为
self._review_node(node, parent_node=parent_node)
```

**修改点 5**: 修改 `execute_mutate_task()` 调用

```python
# execute_mutate_task (第 1140 行附近)
# mutate 使用代码 diff
self._review_node(node, parent_node=parent)
```

**修改点 6**: 修改 `execute_merge_task()` 调用（使用基因选择方案）

```python
# execute_merge_task (第 1055 行附近)
# merge 使用基因选择方案而非代码 diff
self._review_node(node, gene_plan=gene_plan)
```

> **为什么 merge 不用 diff？**
> - merge 的本质是"组合两个 parent 的基因片段"
> - 代码 diff 只能展示"改了什么"，无法展示"为什么这样组合"
> - 基因选择表直接告诉 Review LLM："DATA 用了 A 的采样逻辑，MODEL 用了 B 的 XGBoost"
> - 这样 LLM 能分析基因组合的兼容性（如：A 的小数据 + B 的大模型可能欠拟合）

---

### 2.1 Phase 1: 增强 Review 输出 (信息采集)

**文件**: `core/orchestrator.py`

**修改点 1**: 修改 `_call_review_with_tool()` 签名

```python
# 当前
def _call_review_with_tool(self, node: Node) -> Dict:

# 修改为
def _call_review_with_tool(self, node: Node, code_diff: str) -> Dict:
```

**修改点 2**: 修改 `_build_review_messages()` 加入 diff 和 best_metric

```python
def _build_review_messages(self, node: Node, code_diff: str) -> str:
    """构建 Review 消息（包含代码 diff 和最佳指标上下文）。

    Args:
        node: 节点对象
        code_diff: 与父节点的代码 diff

    Returns:
        消息内容字符串
    """
    # 获取当前最佳指标
    best_metric = self.best_node.metric_value if self.best_node else None
    best_metric_str = f"{best_metric:.4f}" if best_metric else "N/A (first run)"

    return f"""You are evaluating a machine learning solution for a Kaggle competition.

**Goal**: Achieve Kaggle Gold Medal (Top 1%)

**Current Best Metric**: {best_metric_str}

**Task Description:**
{self.task_desc}

---

## Code Changes (Diff from Parent)

```diff
{code_diff}
```

---

## Current Solution

```python
{node.code}
```

---

## Execution Output

```
{node.term_out}
```

**Execution Status:**
- Execution Time: {node.exec_time:.2f}s
- Exception: {node.exc_type or "None"}

---

## Your Task

Analyze the execution results based on the **Code Changes** above. Focus on:

1. **Key Change**: Summarize the main modification in 1-2 sentences (based on the diff)
2. **Metric Delta**: Compare with Current Best ({best_metric_str})
3. **Insight**: What did we learn from this experiment? (what worked/didn't work and why)
4. **Bottleneck**: What is preventing better performance?
5. **Next Direction**: What should we try next?

Call `submit_review` with your detailed analysis.
"""
```

**修改点 3**: 增强 `_get_review_tool_schema()`

```python
def _get_review_tool_schema(self) -> Dict:
    """获取 Review Function Calling 的 schema（增强版）。"""
    return {
        "name": "submit_review",
        "description": "提交代码评估结果（包含详细分析）",
        "parameters": {
            "type": "object",
            "properties": {
                "is_bug": {
                    "type": "boolean",
                    "description": "代码是否有 bug 或执行失败",
                },
                "has_csv_submission": {
                    "type": "boolean",
                    "description": "代码是否生成了 submission.csv 文件",
                },
                "metric": {
                    "type": "number",
                    "description": "验证集指标值（如 RMSE），失败时为 null",
                    "nullable": True,
                },
                "lower_is_better": {
                    "type": "boolean",
                    "description": "指标是否越小越好（如 RMSE=true）",
                },
                # ===== 新增字段 =====
                "key_change": {
                    "type": "string",
                    "description": "本次方案的核心改动点（基于 diff 总结，1-2 句话）",
                },
                "metric_delta": {
                    "type": "string",
                    "description": "与当前最佳的指标对比，格式: 'X.XX → Y.YY (↓Z%)'，首次运行填 'baseline'",
                },
                "insight": {
                    "type": "string",
                    "description": "从本次实验得到的洞察（什么有效/无效，为什么）",
                },
                "bottleneck": {
                    "type": "string",
                    "description": "当前方案的主要瓶颈或限制",
                    "nullable": True,
                },
                "suggested_direction": {
                    "type": "string",
                    "description": "建议的下一步优化方向",
                    "nullable": True,
                },
            },
            "required": [
                "is_bug",
                "has_csv_submission",
                "lower_is_better",
                "key_change",
                "metric_delta",
                "insight",
            ],
        },
    }
```

**修改点 4**: 更新 Node 数据结构

**文件**: `core/state/node.py`

```python
# 添加新字段 (第 74 行附近)
analysis: str = field(default="", kw_only=True)  # 保留兼容（存储 key_change）
analysis_detail: Optional[Dict] = field(default=None, kw_only=True)  # 新增结构化分析
```

**修改点 5**: 保存结构化分析

**文件**: `core/orchestrator.py`

```python
# _review_node() 方法中 (第 511 行附近)
# 当前
node.analysis = review_data.get("summary", "")

# 修改为
node.analysis = review_data.get("key_change", "")  # 兼容旧字段
node.analysis_detail = {
    "key_change": review_data.get("key_change", ""),
    "metric_delta": review_data.get("metric_delta", ""),
    "insight": review_data.get("insight", ""),
    "bottleneck": review_data.get("bottleneck"),
    "suggested_direction": review_data.get("suggested_direction"),
}
```

---

### 2.2 Phase 2: 重构 Memory 格式 (信息组织)

**文件**: `core/state/journal.py`

**修改点**: 重写 `generate_summary()` 方法

```python
def generate_summary(self, include_code: bool = False) -> str:
    """生成 Evolution Log 格式的 Memory。

    新格式包含:
    1. Current Best: 当前最佳方案概览
    2. Changelog: 按时间倒序的改进记录 (类似 git log)
    3. Constraints: 从 BUGGY 中提取的硬约束
    4. Unexplored: 未尝试的方向建议

    Args:
        include_code: 是否包含完整代码

    Returns:
        格式化的 Evolution Log 字符串
    """
    if not self.nodes:
        return "No previous solutions."

    sections = []

    # Section 1: Current Best
    best = self.get_best_node()
    if best:
        sections.append(self._format_current_best(best))

    # Section 2: Changelog (最近 10 条，倒序)
    sections.append(self._format_changelog(limit=10))

    # Section 3: Constraints (从 BUGGY 提取)
    constraints = self._extract_constraints()
    if constraints:
        sections.append(self._format_constraints(constraints))

    # Section 4: Unexplored Directions
    unexplored = self._collect_unexplored_directions()
    if unexplored:
        sections.append(self._format_unexplored(unexplored))

    return "\n\n".join(sections)

def _format_current_best(self, node: "Node") -> str:
    """格式化当前最佳方案。"""
    detail = node.analysis_detail or {}
    metric_str = f"{node.metric_value:.4f}" if node.metric_value else "N/A"
    return f"""## Current Best: {metric_str} (Step {node.step}, Node {node.id[:8]})

**Key Approach**: {detail.get('key_change', 'N/A')}
**Bottleneck**: {detail.get('bottleneck', 'Unknown')}
"""

def _format_changelog(self, limit: int = 10) -> str:
    """格式化 Changelog (类似 git log)。"""
    lines = ["## Changelog (Recent Changes)\n"]

    # 按 step 倒序
    sorted_nodes = sorted(self.nodes, key=lambda n: n.step, reverse=True)[:limit]

    for node in sorted_nodes:
        detail = node.analysis_detail or {}

        # 状态标记
        if node.is_buggy:
            status = "BUGGY"
        elif node == self.get_best_node():
            status = "BEST"
        else:
            status = ""

        metric_str = f"{node.metric_value:.4f}" if node.metric_value else "N/A"
        delta_str = detail.get('metric_delta', 'N/A')

        lines.append(f"### Step {node.step}: {metric_str} {status}")
        lines.append(f"- **Change**: {detail.get('key_change', node.plan[:100] if node.plan else 'N/A')}")
        lines.append(f"- **Delta**: {delta_str}")
        lines.append(f"- **Insight**: {detail.get('insight', 'N/A')}")
        lines.append("")

    return "\n".join(lines)

def _extract_constraints(self) -> list:
    """从 BUGGY 节点提取硬约束。"""
    constraints = []

    for node in self.buggy_nodes:
        detail = node.analysis_detail or {}

        # 从 insight 中提取约束（如果 Review 分析了失败原因）
        insight = detail.get('insight', '')
        if insight and len(insight) > 10:
            constraints.append(f"Step {node.step}: {insight[:150]}")
        elif node.exc_type:
            # 回退：基于异常类型生成通用约束
            constraints.append(f"Step {node.step}: {node.exc_type} - {node.analysis[:100] if node.analysis else 'Unknown error'}")

    return list(set(constraints))[:10]  # 去重，最多 10 条

def _format_constraints(self, constraints: list) -> str:
    """格式化约束列表。"""
    lines = ["## Constraints (Learned from Failures)\n"]
    for c in constraints:
        lines.append(f"- {c}")
    return "\n".join(lines)

def _collect_unexplored_directions(self) -> list:
    """收集未尝试的方向建议。"""
    directions = set()

    for node in self.good_nodes:
        detail = node.analysis_detail or {}
        suggested = detail.get('suggested_direction')
        if suggested and len(suggested) > 5:
            directions.add(suggested)

    return list(directions)[:5]  # 最多 5 个

def _format_unexplored(self, directions: list) -> str:
    """格式化未尝试方向。"""
    lines = ["## Unexplored Directions\n"]
    for d in directions:
        lines.append(f"- [ ] {d}")
    return "\n".join(lines)
```

---

### 2.3 Phase 3: 增强 Explore Prompt (信息利用)

**文件**: `benchmark/mle-bench/prompt_templates/explore.j2`

**修改点**: 在 Memory 部分添加强制指令

```jinja2
{# 在 {% if memory %} 部分前添加 #}

{% if memory %}
# Evolution Log

> **CRITICAL**: You MUST read and use this Evolution Log to inform your approach.

{{ memory }}

## Instructions for Using Evolution Log

You MUST:
1. **Read the Changelog** - Understand what changes improved/degraded performance
2. **Respect Constraints** - NEVER repeat errors listed in Constraints section
3. **Build on Insights** - Your approach should address identified bottlenecks
4. **Explore New Directions** - Pick from Unexplored Directions or propose new ones

Your **Thinking** section MUST reference:
- Which insight from Changelog informed your approach
- What specific bottleneck you're targeting
- Why your proposed change should help

**DO NOT** copy previous Thinking verbatim. Your analysis must be specific to this attempt.

{% endif %}
```

---

### 2.4 Phase 4: 移除 Thinking 模板示例

**文件**: `benchmark/mle-bench/skills/static/output_format.md`

**修改点**: 修改 Response Format 中的示例

```markdown
# Response Format

**Your response should contain:**

1. **Thinking** (3-5 sentences, REQUIRED)
   - Reference specific insights from the Changelog (if available)
   - Identify the bottleneck you're addressing
   - Explain why your proposed change should help
   - **DO NOT** use generic descriptions like "The dataset has X rows..."

2. **A brief outline** (3-5 sentences) of your proposed solution
   - Explain the key idea and approach
   - Highlight the main components or strategies
   - Avoid excessive exploratory data analysis (EDA)

3. **A single markdown code block** (````python...```) implementing this solution

**CRITICAL Requirements:**

- **DO NOT** include additional headings or explanations outside the code block
- **DO NOT** add multiple code blocks or fragments
- **DO** ensure the code is a single-file Python program
- **DO** make the code executable without modifications

**Example Thinking (DO NOT copy this content, adapt to your situation):**

> "Based on Changelog Step 8's insight that airport features improved RMSE by 12%,
> I will explore additional location-based features. The current bottleneck is
> long-distance prediction accuracy. I propose adding zone-based target encoding
> to capture neighborhood-level fare patterns."
```

---

## 3. 实施计划

### 3.1 优先级排序

| 优先级 | 任务 | 文件 | 预期效果 |
|--------|------|------|----------|
| **P0** | 新增 `_generate_code_diff()` | `orchestrator.py` | 生成代码 diff |
| **P0** | 新增 `_format_gene_selection()` | `orchestrator.py` | 格式化 merge 基因选择 |
| **P0** | 修改 `_review_node()` 签名 | `orchestrator.py` | 支持 parent_node + gene_plan |
| **P0** | 修改 `_step_task()` 调用 | `orchestrator.py` | 传递 parent_node |
| **P0** | 修改 `execute_mutate_task()` 调用 | `orchestrator.py` | 传递 parent_node |
| **P0** | 修改 `execute_merge_task()` 调用 | `orchestrator.py` | 传递 gene_plan |
| **P0** | 修改 `_build_review_messages()` | `orchestrator.py` | 包含变更上下文 + best_metric |
| **P1** | 增强 `_get_review_tool_schema()` | `orchestrator.py` | 新增分析字段 |
| **P1** | 添加 `analysis_detail` 字段 | `node.py` | 存储结构化分析 |
| **P1** | 保存结构化分析 | `orchestrator.py` | 填充 analysis_detail |
| **P2** | 重写 `generate_summary()` | `journal.py` | Changelog 格式 |
| **P2** | 增强 Explore Prompt | `explore.j2` | 强制利用历史 |
| **P3** | 移除 Thinking 模板示例 | `output_format.md` | 避免模板化输出 |

### 3.2 依赖关系

```
P0: _generate_code_diff()     P0: _format_gene_selection()
           ↘                    ↙
            P0: _review_node() 签名修改
                      ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
_step_task()   execute_mutate()   execute_merge()
(parent_node)    (parent_node)      (gene_plan)
    └─────────────────┼─────────────────┘
                      ↓
        P0: _build_review_messages() 包含变更上下文
                      ↓
        P1: _get_review_tool_schema() 增强
                      ↓
        P1: node.py 添加 analysis_detail
                      ↓
        P1: 保存 analysis_detail
                      ↓
        P2: generate_summary() Changelog 格式
                      ↓
        P2: explore.j2 强制指令
                      ↓
        P3: output_format.md 移除模板
```

### 3.3 测试验证

**验证标准**:

1. **Diff 生成测试（explore/mutate）**
   - [ ] 首次生成返回 "(Initial solution, no diff)"
   - [ ] 代码变更正确生成 unified diff
   - [ ] 超长 diff 被截断

2. **基因选择格式化测试（merge）**
   - [ ] 正确展示 7 个基因位点的来源
   - [ ] 每个位点包含 source_node_id 和代码片段
   - [ ] 超长代码片段被截断（500 字符）

3. **Review 上下文测试**
   - [ ] explore/mutate: Review Prompt 包含 diff 内容
   - [ ] merge: Review Prompt 包含基因选择表
   - [ ] 所有模式: Review Prompt 包含 best_metric

3. **信息密度测试**
   - [ ] Review 输出包含 key_change, metric_delta, insight
   - [ ] Changelog 格式清晰展示进化路径

4. **约束遵守测试**
   - [ ] BUGGY 错误被转化为 Constraints
   - [ ] 后续生成不再重复相同错误

5. **探索多样性测试**
   - [ ] Thinking 内容不再完全相同
   - [ ] 新方案确实尝试了 Unexplored Directions

---

## 4. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| Diff 过长占用 token | 限制 100 行 + 截断提示 |
| Review LLM 不遵守新 Schema | 添加 Schema 验证，回退到旧格式 |
| Changelog 过长占用 token | 限制为最近 10 条 |
| 约束提取不准确 | 依赖 Review insight，逐步迭代 |
| LLM 仍然生成模板化输出 | Phase 4 移除模板示例 |

---

## 5. 预期收益

| 指标 | 当前 | 预期 |
|------|------|------|
| Review 信息量 | 2-3 句 summary | 5 字段结构化分析 |
| 连续 BUGGY 数 | 13 | < 3 |
| Thinking 相似度 | 100% | < 30% |
| 最终 RMSE | 2.85 | < 2.6 (接近金牌) |
| 信息利用率 | ~0% | > 70% |

---

## 6. 附录

### 6.1 Diff 输出示例（explore/mutate）

```diff
--- parent_solution.py
+++ current_solution.py
@@ -156,7 +156,10 @@
 model = lgb.LGBMRegressor(
     n_estimators=1000,
-    early_stopping_rounds=100,
+    callbacks=[
+        lgb.early_stopping(100),
+        lgb.log_evaluation(100)
+    ],
 )
```

### 6.2 基因选择输出示例（merge）

```markdown
## Gene Selection

### DATA (from Node abc12345)
```python
# [SECTION: DATA]
import pandas as pd

train = pd.read_csv('./data/train.csv', nrows=100000)  # 采样 10 万行
test = pd.read_csv('./data/test.csv')
```

### MODEL (from Node def67890)
```python
# [SECTION: MODEL]
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=8,
    tree_method='hist',
)
```

### LOSS (from Node abc12345)
```python
# [SECTION: LOSS]
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

... (其他 4 个基因位点)
```

> **Review LLM 分析示例**: "这个 merge 选了 Node abc12345 的 DATA（采样 10 万行）和 Node def67890 的 MODEL（XGBoost 1000 棵树）。但 XGBoost 的参数是针对大数据集调优的，配合小采样量可能导致欠拟合。建议增加采样量或减少 n_estimators。"

### 6.3 新 Memory 格式示例

```markdown
## Current Best: 2.8515 (Step 11, Node d3831b75)

**Key Approach**: LightGBM + airport proximity + temporal features + early stopping
**Bottleneck**: Long-distance trips (>10km) prediction accuracy

## Changelog (Recent Changes)

### Step 11: 2.8515 BEST
- **Change**: Added early_stopping(100) callback replacing deprecated parameter
- **Delta**: 2.93 → 2.85 (↓2.7%)
- **Insight**: Regularization crucial for this dataset, prevents late-stage overfitting

### Step 10: 2.9613
- **Change**: Increased training data 10M → 15M
- **Delta**: 2.93 → 2.96 (↑1.0%)
- **Insight**: More data doesn't help; may introduce noise. Data quality > quantity

### Step 9: N/A BUGGY
- **Change**: Used early_stopping_rounds parameter
- **Delta**: N/A
- **Insight**: LightGBM API changed, must use callbacks=[lgb.early_stopping(N)]

## Constraints (Learned from Failures)

- Step 9: LightGBM API changed, must use callbacks=[lgb.early_stopping(N)]
- Step 5: KeyError on test set - avoid creating train-only features

## Unexplored Directions

- [ ] Feature crossing (pickup_hour × distance)
- [ ] Target encoding (pickup_zone → historical_avg_fare)
- [ ] XGBoost / CatBoost comparison
- [ ] Multi-model ensemble (Stacking)
```
