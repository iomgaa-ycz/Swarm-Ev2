# Implementation Plan: 降低 Buggy 率（从根源提升代码生成质量）

## 1.1 摘要 (Summary)

当前系统全局 Buggy 率 ~60%，通过对比 AIDE/ML-Master 两个参考系统，识别出 Swarm-Ev2 的核心架构性缺陷：**缺少即时 Debug 循环 + merge 操作固有的高失败率 + 数据预览不足**。提出 5 个改进模块，预期将 Buggy 率从 60% 降至 35%。

## 1.2 审查点 (User Review Required)

1. **改进 1（即时 Debug）** 重试次数定为 1 次。
2. **改进 3（静态预验证）** 验证失败时，将错误信息注入 Prompt 让 LLM 自动修复 1 次（等效于改进 1 的前置版本）。
3. **改进 4（Merge 兼容性预检）** 如果两个父代框架不兼容（torch vs sklearn），是否直接回退到高 fitness 父代的全部代码（等效于跳过 merge）？还是尝试让 LLM 处理？
4. 是否同意先实施 Phase 1（改进 1+3），预期效果最大、实现最快？

---

## 1.3 根因诊断（基于 AIDE/ML-Master 对比分析）

### 对比发现：为什么 AIDE/ML-Master 不会有大量错误

| 差异维度 | AIDE / ML-Master | Swarm-Ev2 | 影响 |
|---------|-----------------|-----------|------|
| **merge/crossover 操作** | **不存在**。只有 draft/debug/improve | 55.6% 操作是 merge，Buggy 率 53.3% | Swarm-Ev2 约 **30% 全部计算浪费在失败的 merge** |
| **Debug 机制** | 确定性：buggy 节点 100% 进入 debug 分支（ML-Master）；或 50% 概率触发、最多 3 次重试（AIDE） | 概率性：依赖 `debug_prob` 被分配到 explore(debug)，且下一步可能被分配到 merge/mutate | **buggy 节点无法被及时修复** |
| **Draft 策略** | 先生成 5 个独立 Draft，确保至少 1-2 个成功（成功率 ≈ 1 - 0.6^5 = 92%） | 每个 Agent 各 1 个 explore，迅速进入 merge/mutate | **进化系统缺乏足够的初始成功方案** |
| **执行后验证** | ML-Master: 4 层验证（执行 + LLM Judge + CSV 格式校验 + Metric 合理性检查） | 2 层（执行 + Review） | **低质量方案未被拦截，污染种群** |
| **代码模板** | **都不用模板**，完全依赖 LLM 生成 | 有 workspace_rules + code_style 但信息不足 | 模板不是关键因素 |

### 根因排序（按影响大小重新排列）

**根因 A（最大影响）: 缺少即时 Debug 循环**

AIDE 和 ML-Master 的核心机制是：代码执行失败 → **立即**进入 debug 模式 → 将完整错误信息反馈给 LLM → 生成修复版本。AIDE 的 `search_policy()` 中 `debug_prob=0.5` 意味着每一步都有 50% 概率去修复 buggy 节点，且修复深度最多 3 次（`max_debug_depth=3`）。ML-Master 更激进——MCTS 选中 buggy 节点后 **100% 确定性**进入 `_debug()` 分支。

Swarm-Ev2 的问题：`_step_task()` 执行失败后直接返回，不做任何重试。修复只能等后续 step 被概率性分配到 explore(debug)。

**根因 B: merge 操作固有的高失败率**

AIDE 和 ML-Master **都不做基因交叉**——它们只有 draft → debug → improve 三种操作。Swarm-Ev2 的 merge（从两个父代的 7 个基因块中随机交叉）是一个**本系统独有的高风险操作**。

`_build_gene_plan_markdown_from_random()` 的具体问题：
- 父代 gene 可能为空 → `genes_a.get(gene, "# (no code)")` → LLM 收到空块
- 两个父代可能使用完全不同的框架（torch vs sklearn）→ 基因块无法物理拼接
- gene_plan 是 Markdown 文本 → LLM 需要重新理解和组合两份完整代码

**根因 C: 数据预览信息不足**

`data_preview.py` 对非 CSV 文件几乎无预览能力。mlsp-birds 的 126 次 explore 全部失败就是直接证据。

**根因 D: 缺少静态预验证**

AIDE 的 `extract_code()` 中有 `is_valid_python_script(compile(...))` 语法检查，而 Swarm-Ev2 的 `response.py:extract_code()` 没有。

**根因 E: 缺少 submission 格式验证**

ML-Master 有外部服务器验证 submission.csv 的格式和行数（`check_format=True`）。Swarm-Ev2 完全没有这层检查。

---

## 1.4 拟议变更 (Proposed Changes)

### 改进 1: 即时 Debug 循环（Immediate Debug Loop）— 最高优先级

**参考来源**: AIDE `search_policy()` + `_debug()`、ML-Master `_step_search()` debug 分支

**目标**: 代码执行失败后，立即将错误信息反馈给 LLM 尝试 1 次修复

**预期效果**: 将 ~20-30% 的 Buggy 节点转化为成功节点（参考 AIDE 的 debug 成功率）

#### 文件变更:

| 文件 | 操作 | 变更内容 |
|------|------|---------|
| `agents/coder_agent.py` | `[NEW]` `_debug()` 方法 | 新增 Debug 方法：接收 buggy code + 错误信息 + term_out，请求 LLM 修复 |
| `core/orchestrator.py` | `[MODIFY]` `_step_task()` | 在执行失败后、返回前，插入 1 次 Debug 重试 |
| `core/orchestrator.py` | `[MODIFY]` `execute_merge_task()` | 同上 |
| `core/orchestrator.py` | `[MODIFY]` `execute_mutate_task()` | 同上 |
| `benchmark/mle-bench/prompt_templates/debug.j2` | `[NEW]` | Debug 专用 Prompt 模板 |

#### 实现细节:

**`coder_agent.py` 新增 `_debug()` 方法（参考 AIDE `agent.py:292-330`）:**

```python
async def _debug(self, context: Dict[str, Any]) -> Optional[Node]:
    """修复 buggy 代码。

    基于 AIDE 的 debug 机制设计。将完整的错误信息反馈给 LLM 请求修复。

    Args:
        context: 包含以下字段
            - buggy_code: str, 原始代码
            - exc_type: str, 异常类型
            - term_out: str, 完整执行输出（含 traceback）
            - task_desc: str, 任务描述
            - data_preview: str, 数据预览

    Returns:
        修复后的 Node（修复失败返回 None）
    """
    prompt = self.prompt_manager.build_prompt(
        task_type="debug",
        agent_id=self.agent_id,
        context=context,
    )

    response = await backend_query(prompt, ...)
    code = extract_code(response)

    if not code or code == context["buggy_code"]:
        return None  # 修复失败或无变化

    return Node(code=code, plan=extract_text_up_to_code(response), ...)
```

**`debug.j2` 模板设计（参考 AIDE 的 debug prompt 结构）:**

```jinja2
{# SECTION: ROLE #}
{{ load_agent_config(agent_id, "role") }}

{# SECTION: TASK #}
# Task Description
{{ task_desc }}

{# SECTION: BUGGY CODE #}
# Previous (Buggy) Implementation
Your previous solution encountered an error during execution.
```python
{{ buggy_code }}
```

# Execution Output
```
{{ term_out }}
```

{# SECTION: INSTRUCTIONS #}
## Instructions
1. Analyze the error carefully and identify the root cause.
2. Fix the bug with **minimal changes** — do not refactor or add new features.
3. Ensure the fixed code:
   - Prints the validation metric: `Validation metric: {value}`
   - Saves predictions to `./submission/submission.csv`
   - Uses only available packages
4. Your response: a brief description of the fix (2-3 sentences), followed by a single code block.

{{ load_skill("static/output_format") }}
{{ load_skill("static/workspace_rules") }}
```

**`orchestrator.py` 集成点:**

```python
async def _step_task(self, parent_node, agent, context):
    ...
    # Phase 4: 执行代码
    exec_result = self._execute_code(node.code, node.id)
    node.absorb_exec_result(exec_result)

    # [NEW] Phase 4.5: 即时 Debug（非超时错误才重试）
    if node.is_buggy and node.exc_type not in (None, "TimeoutExpired", "MemoryError"):
        log_msg("INFO", f"执行失败({node.exc_type})，启动即时 Debug")
        debug_context = {
            "buggy_code": node.code,
            "exc_type": node.exc_type,
            "term_out": "\n".join(node.term_out) if isinstance(node.term_out, list) else node.term_out,
            "task_desc": context["task_desc"],
            "data_preview": context.get("data_preview", ""),
        }
        fixed_node = await agent._debug(debug_context)

        if fixed_node and fixed_node.code != node.code:
            # 执行修复后的代码
            fix_exec_result = self._execute_code(fixed_node.code, fixed_node.id)
            fixed_node.absorb_exec_result(fix_exec_result)

            if not fixed_node.is_buggy:
                log_msg("INFO", f"即时 Debug 成功: {node.exc_type} → 修复")
                fixed_node.parent_id = node.parent_id
                node = fixed_node  # 用修复后的节点替换原节点
            else:
                log_msg("INFO", f"即时 Debug 失败，保留原 buggy 节点")
                # 仍然将修复失败的节点记录到 Journal（供后续分析）
    ...
```

---

### 改进 2: 增强数据预览（Rich Data Preview）

**目标**: 解决非标准任务（图像、音频、特殊文本）的上下文信息不足

**预期效果**: mlsp-birds 类任务的 explore 成功率从 0% 提升到 >20%

#### 文件变更:

| 文件 | 操作 | 变更内容 |
|------|------|---------|
| `utils/data_preview.py` | `[MODIFY]` `generate()` | 识别图像目录并调用 `preview_image_dir()` |
| `utils/data_preview.py` | `[MODIFY]` `preview_csv()` | 详细模式默认开启（展示目标列分布、缺失率、数值范围） |
| `utils/data_preview.py` | `[NEW]` `preview_image_dir()` | 采样图像统计尺寸/通道/值范围 + 注入 ToTensor 警告 |
| `utils/data_preview.py` | `[NEW]` `preview_special_file()` | .txt/.tsv: 读取前 5 行 + 推断分隔符 + 列数 |
| `utils/data_preview.py` | `[MODIFY]` 长度阈值 | 6000 → 8000 字符（给新增信息留空间） |
| `benchmark/mle-bench/skills/static/common_pitfalls.md` | `[NEW]` | 常见陷阱知识库 |

#### `preview_image_dir()` 实现:

```python
def preview_image_dir(dir_path: Path, max_sample: int = 3) -> str:
    """采样图像目录，返回关键统计信息。

    信息包括：图像数量、尺寸、通道数、值范围、格式。
    同时注入 PyTorch ToTensor 归一化警告。
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = [f for f in dir_path.iterdir() if f.suffix.lower() in IMAGE_EXTS]

    if not images:
        return ""

    from PIL import Image
    import numpy as np

    samples = images[:max_sample]
    sizes, modes = [], []
    for img_path in samples:
        with Image.open(img_path) as img:
            sizes.append(img.size)
            modes.append(img.mode)

    # 检查值范围（只对第一张采样）
    with Image.open(samples[0]) as img:
        arr = np.array(img)
        val_min, val_max = arr.min(), arr.max()

    return (
        f"Image directory: {len(images)} images, format: {samples[0].suffix}\n"
        f"Sample sizes: {sizes}, mode: {set(modes)}\n"
        f"Pixel value range: [{val_min}, {val_max}]\n"
        f"NOTE: torchvision.transforms.ToTensor() converts [0,255] → [0.0,1.0]. "
        f"Do NOT divide by 255 again after ToTensor()."
    )
```

#### `preview_special_file()` 实现:

```python
def preview_special_file(file_path: Path, max_lines: int = 5) -> str:
    """预览非标准文本文件（.txt, .tsv 等）。

    读取前 N 行，推断分隔符和列数。
    """
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(max_lines)]

        # 推断分隔符
        first_line = lines[0].strip()
        for sep, name in [("\t", "tab"), (",", "comma"), (" ", "space")]:
            if sep in first_line:
                num_cols = len(first_line.split(sep))
                preview = "\n".join(l.rstrip() for l in lines if l.strip())
                return (
                    f"-> {file_path.name}: {name}-separated, ~{num_cols} columns\n"
                    f"First {len(lines)} lines:\n```\n{preview}\n```"
                )

        return f"-> {file_path.name}: unstructured text, first lines:\n```\n{''.join(lines)}\n```"
    except Exception as e:
        return f"-> {file_path.name}: (cannot read: {e})"
```

#### `common_pitfalls.md` 内容:

```markdown
# Common Pitfalls (MUST READ)

## PyTorch Image Processing
- `transforms.ToTensor()` already converts [0,255] uint8 → [0.0,1.0] float32
- Do NOT add `/ 255.0` after ToTensor() — this causes double normalization
- `transforms.Normalize(mean, std)` should be applied AFTER ToTensor()

## Validation Metric
- The validation metric MUST match the competition's evaluation metric exactly
- WRONG: Using training loss (e.g., Focal Loss value) as "Validation metric"
- CORRECT: Using `sklearn.metrics.{actual_metric}()` on validation predictions
- Example: If competition uses `log_loss`, use `sklearn.metrics.log_loss(y_val, y_pred_proba)`

## Submission File
- ALWAYS verify row count: `assert len(submission) == len(test_data)`
- ALWAYS verify no NaN: `assert not submission.isnull().any().any()`
- Check column names match sample_submission.csv exactly

## Data Loading
- Read sample_submission.csv FIRST to understand expected output format
- For image tasks: check image extensions (.jpg/.png/.tif may differ)
- For special formats (.txt/.tsv): check delimiter and header presence
```

#### `generate()` 修改点:

```python
def generate(base_path: Path, ...) -> str:
    ...
    if include_file_details:
        for fn in _walk(base_path):
            ...
            # [NEW] 图像目录预览
            elif fn.is_dir() and any(
                (fn / f).suffix.lower() in IMAGE_EXTS
                for f in list(fn.iterdir())[:5]
            ):
                out.append(preview_image_dir(fn))

            # [NEW] 特殊文本文件预览
            elif fn.suffix in {".txt", ".tsv"} and get_file_len_size(fn)[0] >= 30:
                out.append(preview_special_file(fn))
    ...
```

---

### 改进 3: 代码生成后静态预验证 + LLM 自动修复

**参考来源**: AIDE `utils/response.py:is_valid_python_script()` 的 compile 检查

**目标**: 在 subprocess 执行前拦截语法错误和明显缺陷，并让 LLM 立即修复

**预期效果**: 拦截 ~15% 的低级 bug（SyntaxError、Import 缺失、无 submission 输出）

#### 文件变更:

| 文件 | 操作 | 变更内容 |
|------|------|---------|
| `utils/code_validator.py` | `[NEW]` | 静态预验证模块（compile + import 检查 + 输出检查 + 陷阱检测） |
| `agents/coder_agent.py` | `[MODIFY]` 代码提取后 | 调用预验证，失败时将错误注入 context 让 LLM 重新生成 |

#### `code_validator.py` 实现:

```python
"""代码静态预验证模块。

在 subprocess 执行前，用轻量 Python 检查拦截明显的代码缺陷。
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    """验证结果。"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_code(code: str) -> ValidationResult:
    """静态预验证（不执行代码）。

    检查项:
    1. 语法检查: compile()
    2. Submission 输出检查: 代码是否包含 to_csv(...submission...)
    3. Metric 输出检查: 代码是否包含 print(...Validation metric...)
    4. 常见陷阱检测: ToTensor + /255, 缺少 random seed
    """
    errors = []
    warnings = []

    # 1. 语法检查（参考 AIDE is_valid_python_script）
    try:
        compile(code, "solution.py", "exec")
    except SyntaxError as e:
        errors.append(f"SyntaxError at line {e.lineno}: {e.msg}")
        # 语法错误直接返回，后续检查无意义
        return ValidationResult(valid=False, errors=errors)

    # 2. Submission 输出检查
    if "submission" not in code.lower() or "to_csv" not in code:
        errors.append("Missing submission output: code must save predictions to submission.csv using to_csv()")

    # 3. Metric 输出检查
    metric_patterns = [r"[Vv]alidation metric", r"print\(.*metric", r"print\(.*score"]
    if not any(re.search(p, code) for p in metric_patterns):
        warnings.append("Missing metric print: code should print 'Validation metric: {value}'")

    # 4. 常见陷阱检测
    if "ToTensor()" in code and "/ 255" in code:
        warnings.append(
            "Possible double normalization: ToTensor() already normalizes to [0,1]. "
            "Remove '/ 255' if using ToTensor()."
        )

    is_valid = len(errors) == 0
    return ValidationResult(valid=is_valid, errors=errors, warnings=warnings)
```

#### `coder_agent.py` 集成（在 LLM 响应解析后）:

```python
async def _generate_with_validation(self, context, task_type, max_retries=1):
    """生成代码 + 静态预验证 + 失败时自动修复。"""
    # 第一次生成
    node = await self._generate(context, task_type)

    if not node or not node.code:
        return node

    # 静态预验证
    from utils.code_validator import validate_code
    validation = validate_code(node.code)

    if validation.valid:
        # 将 warnings 注入 node 的 plan 中（供后续参考）
        if validation.warnings:
            node.plan += f"\n[Warnings: {'; '.join(validation.warnings)}]"
        return node

    # 预验证失败 → 让 LLM 修复（最多 1 次）
    if max_retries > 0:
        log_msg("INFO", f"预验证失败: {validation.errors}，尝试自动修复")
        fix_context = {
            **context,
            "buggy_code": node.code,
            "validation_errors": "\n".join(validation.errors + validation.warnings),
        }
        fixed_node = await self._debug(fix_context)
        if fixed_node and fixed_node.code:
            revalidation = validate_code(fixed_node.code)
            if revalidation.valid:
                log_msg("INFO", "预验证修复成功")
                return fixed_node

    # 修复失败，返回原节点（标记 warning）
    log_msg("WARNING", f"预验证失败且修复未成功: {validation.errors}")
    return node
```

---

### 改进 4: Merge 基因兼容性预检

**目标**: 在 merge 操作发送给 LLM 前，检测基因块的明显不兼容，避免无效 merge

**预期效果**: merge Buggy 率从 53.3% 降至 ~35-40%

#### 文件变更:

| 文件 | 操作 | 变更内容 |
|------|------|---------|
| `core/evolution/gene_compatibility.py` | `[NEW]` | 基因兼容性检查模块 |
| `core/evolution/solution_evolution.py` | `[MODIFY]` `_crossover_mvp()` | 生成 gene_plan 后调用兼容性检查 |
| `core/evolution/solution_evolution.py` | `[MODIFY]` `_build_gene_plan_markdown_from_random()` | 空基因块回退到高 fitness 父代 |

#### `gene_compatibility.py` 具体实现:

```python
"""基因兼容性检查模块。

在 merge 操作发送给 LLM 前，检测基因块间的明显不兼容。
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# 框架互斥组：同一组内的包不应混用
FRAMEWORK_GROUPS = {
    "torch": {"torch", "torchvision", "timm", "torch.nn", "torch.optim"},
    "sklearn": {"sklearn", "xgboost", "lightgbm", "catboost"},
    "tensorflow": {"tensorflow", "keras", "tf"},
}


@dataclass
class CompatibilityResult:
    """兼容性检查结果。"""
    compatible: bool
    conflicts: List[str] = field(default_factory=list)
    action: str = "proceed"  # "proceed" | "fallback_a" | "fallback_b" | "inject_warning"


def extract_imports(code: str) -> set[str]:
    """从代码片段中提取 import 的包名。

    Args:
        code: Python 代码字符串

    Returns:
        包名集合（如 {"torch", "sklearn", "numpy"}）
    """
    packages = set()
    for line in code.split("\n"):
        line = line.strip()
        # import torch / import torch.nn as nn
        m = re.match(r"^import\s+([\w.]+)", line)
        if m:
            packages.add(m.group(1).split(".")[0])
        # from torch import nn / from sklearn.model_selection import ...
        m = re.match(r"^from\s+([\w.]+)\s+import", line)
        if m:
            packages.add(m.group(1).split(".")[0])
    return packages


def detect_framework(code: str) -> Optional[str]:
    """检测代码片段使用的主框架。

    Returns:
        "torch" | "sklearn" | "tensorflow" | None
    """
    imports = extract_imports(code)
    for framework, keywords in FRAMEWORK_GROUPS.items():
        if imports & keywords:
            return framework
    return None


def check_gene_compatibility(
    parent_a_code: str,
    parent_b_code: str,
    genes_a: Dict[str, str],
    genes_b: Dict[str, str],
    gene_plan_choices: Dict[str, str],  # {"DATA": "A", "MODEL": "B", ...}
    parent_a_metric: Optional[float] = None,
    parent_b_metric: Optional[float] = None,
) -> CompatibilityResult:
    """检查基因交叉计划的兼容性。

    检查项:
    1. 空基因块检测
    2. 框架一致性（选中的 DATA 和 MODEL 是否使用相同框架）
    3. 全局 import 冲突

    Args:
        parent_a_code: 父代 A 完整代码
        parent_b_code: 父代 B 完整代码
        genes_a: 父代 A 的基因块字典
        genes_b: 父代 B 的基因块字典
        gene_plan_choices: 基因选择方案 {"DATA": "A", "MODEL": "B", ...}
        parent_a_metric: 父代 A 的 fitness
        parent_b_metric: 父代 B 的 fitness

    Returns:
        CompatibilityResult
    """
    conflicts = []

    # [检查 1] 空基因块
    for gene, choice in gene_plan_choices.items():
        source = genes_a if choice == "A" else genes_b
        code_block = source.get(gene, "")
        if not code_block or code_block.strip() in ("", "# (no code)"):
            conflicts.append(f"{gene} from parent_{choice} is empty")

    # [检查 2] DATA-MODEL 框架一致性
    data_choice = gene_plan_choices.get("DATA", "A")
    model_choice = gene_plan_choices.get("MODEL", "A")

    data_code = (genes_a if data_choice == "A" else genes_b).get("DATA", "")
    model_code = (genes_a if model_choice == "A" else genes_b).get("MODEL", "")

    data_framework = detect_framework(data_code)
    model_framework = detect_framework(model_code)

    if (
        data_framework and model_framework
        and data_framework != model_framework
    ):
        conflicts.append(
            f"Framework mismatch: DATA uses {data_framework} (parent_{data_choice}), "
            f"MODEL uses {model_framework} (parent_{model_choice})"
        )

    # [检查 3] 全局框架一致性
    framework_a = detect_framework(parent_a_code)
    framework_b = detect_framework(parent_b_code)

    if framework_a and framework_b and framework_a != framework_b:
        # 两个父代使用不同框架 → 高风险
        conflicts.append(
            f"Parents use different frameworks: A={framework_a}, B={framework_b}"
        )

    # 决策逻辑
    if not conflicts:
        return CompatibilityResult(compatible=True)

    # 有冲突时的处理策略
    has_framework_conflict = any("Framework" in c or "framework" in c for c in conflicts)
    has_empty_block = any("empty" in c for c in conflicts)

    if has_framework_conflict:
        # 框架不兼容 → 注入详细警告让 LLM 选择与修改
        return CompatibilityResult(
            compatible=True,  # 仍然尝试 merge，让 LLM 处理
            conflicts=conflicts,
            action="inject_warning",
        )

    if has_empty_block:
        # 空基因块 → 注入警告让 LLM 处理
        return CompatibilityResult(
            compatible=True,  # 仍然尝试 merge
            conflicts=conflicts,
            action="inject_warning",
        )

    return CompatibilityResult(compatible=True, conflicts=conflicts, action="inject_warning")
```

#### `solution_evolution.py` 集成:

```python
def _crossover_mvp(self, parent_a: Node, parent_b: Node) -> Optional[Node]:
    """基因交叉（增加兼容性预检）。"""
    ...
    # 生成随机基因选择方案
    gene_plan_choices = {
        gene: random.choice(["A", "B"]) for gene in REQUIRED_GENES
    }

    # [NEW] 兼容性预检
    from core.evolution.gene_compatibility import check_gene_compatibility
    compat = check_gene_compatibility(
        parent_a_code=parent_a.code,
        parent_b_code=parent_b.code,
        genes_a=parent_a.genes or {},
        genes_b=parent_b.genes or {},
        gene_plan_choices=gene_plan_choices,
        parent_a_metric=parent_a.metric_value,
        parent_b_metric=parent_b.metric_value,
    )

    # 构建 gene_plan Markdown
    gene_plan_md = self._build_gene_plan_markdown_from_random(parent_a, parent_b)

    # 有冲突时注入详细警告，让 LLM 自行选择与修改
    if compat.conflicts:
        warning_text = "\n".join(f"⚠️ {c}" for c in compat.conflicts)
        gene_plan_md = (
            f"# ⚠️ Compatibility Warnings\n"
            f"The following conflicts were detected. You MUST resolve them:\n"
            f"{warning_text}\n\n"
            f"Suggestion: Choose components from ONE framework consistently, "
            f"or adapt the conflicting blocks to be compatible.\n\n"
            f"{gene_plan_md}"
        )

    return self.orchestrator.execute_merge_task(parent_a, parent_b, gene_plan_md)
```

---

### 改进 5: Submission 格式验证

**参考来源**: ML-Master `check_format` + AIDE `has_csv_submission` 文件系统检查

**目标**: 执行成功但 submission 格式错误（行数/列名/NaN）的节点应被标记为 buggy

**预期效果**: 拦截 ~5-10% 的"假成功"节点

#### 文件变更:

| 文件 | 操作 | 变更内容 |
|------|------|---------|
| `utils/submission_validator.py` | `[NEW]` | Submission 格式验证模块 |
| `core/orchestrator.py` | `[MODIFY]` Review 后 | 在 Review 判定 `is_buggy=False` 后追加 submission 格式检查 |

#### `submission_validator.py` 实现:

```python
"""Submission 格式验证模块。

参考 ML-Master 的 check_format 机制。
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from utils.logger_system import log_msg


def validate_submission(
    submission_path: Path,
    sample_submission_path: Optional[Path] = None,
) -> Tuple[bool, str]:
    """验证 submission.csv 的格式。

    检查项:
    1. 文件存在性
    2. 非空
    3. 无 NaN 值
    4. 行数匹配 sample_submission（如果提供）
    5. 列名匹配 sample_submission（如果提供）

    Returns:
        (is_valid, error_message)
    """
    if not submission_path.exists():
        return False, f"submission.csv not found at {submission_path}"

    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        return False, f"Cannot read submission.csv: {e}"

    if len(sub) == 0:
        return False, "submission.csv is empty"

    if sub.isnull().any().any():
        nan_cols = sub.columns[sub.isnull().any()].tolist()
        nan_count = sub.isnull().sum().sum()
        return False, f"submission.csv has {nan_count} NaN values in columns: {nan_cols}"

    if sample_submission_path and sample_submission_path.exists():
        try:
            sample = pd.read_csv(sample_submission_path, nrows=5)

            # 列名检查
            if list(sub.columns) != list(sample.columns):
                return False, (
                    f"Column mismatch: submission has {list(sub.columns)}, "
                    f"expected {list(sample.columns)}"
                )

            # 行数检查（读取 sample 全部行数）
            sample_full = pd.read_csv(sample_submission_path)
            if len(sub) != len(sample_full):
                return False, (
                    f"Row count mismatch: submission has {len(sub)} rows, "
                    f"expected {len(sample_full)}"
                )
        except Exception as e:
            log_msg("WARNING", f"sample_submission 比对失败: {e}")

    return True, "OK"
```

---

## 1.5 验证计划 (Verification Plan)

### 单元测试

```bash
# 静态预验证器
conda run -n Swarm-Evo pytest tests/unit/test_code_validator.py -v

# 基因兼容性检查
conda run -n Swarm-Evo pytest tests/unit/test_gene_compatibility.py -v

# 增强数据预览
conda run -n Swarm-Evo pytest tests/unit/test_data_preview_enhanced.py -v

# Submission 格式验证
conda run -n Swarm-Evo pytest tests/unit/test_submission_validator.py -v
```

### 集成测试

```bash
# Debug 循环集成测试（mock LLM，验证执行失败→Debug→重执行流程）
conda run -n Swarm-Evo pytest tests/integration/test_debug_loop.py -v

# Merge 兼容性预检集成测试
conda run -n Swarm-Evo pytest tests/integration/test_merge_compatibility.py -v
```

### 效果验证

选 3 个竞赛 A/B 测试：

| 竞赛 | 当前 Buggy 率 | 目标 Buggy 率 | 主要改进来源 |
|------|-------------|-------------|------------|
| histopathologic | 39.6% | 25% | 即时 Debug + 预验证 |
| leaf-classification | 81.5% | 50% | 即时 Debug + Merge 兼容性 |
| mlsp-birds | ~80% | 50% | 增强数据预览 + 即时 Debug |

---

## 实施优先级

| 阶段 | 改进项 | 预期收益 | 实现复杂度 | 依据 |
|------|--------|---------|----------|------|
| **Phase 1** | 改进 1: 即时 Debug 循环 | ★★★★★ | 低 | AIDE/ML-Master 的核心机制 |
| **Phase 1** | 改进 3: 静态预验证 + LLM 自修复 | ★★★★ | 低 | AIDE 已有 compile 检查 |
| **Phase 2** | 改进 2: 增强数据预览 | ★★★ | 低 | 解决非标准任务的信息缺失 |
| **Phase 2** | 改进 5: Submission 格式验证 | ★★★ | 低 | ML-Master 的 check_format |
| **Phase 3** | 改进 4: Merge 兼容性预检 | ★★★ | 中 | 降低 merge 的固有失败率 |
