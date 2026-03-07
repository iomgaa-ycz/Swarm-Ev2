"""文本处理工具模块。

提供 Review Prompt 任务描述清理、执行输出精炼等文本处理功能。
"""

import re
from typing import Optional, Callable


# ── 前置 LLM 提取模板 ──────────────────────────────────────────────

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
    llm_fn: Callable[[str], str] = None,
) -> str:
    """当 stdout 过长时，用前置 LLM 按固定格式提取关键信息。

    Args:
        term_out: 完整终端输出
        max_len: 超过此长度触发 LLM 压缩（默认 8000 chars）
        llm_fn: LLM 调用函数 (prompt: str) -> str，超长时必须提供

    Returns:
        原文（未超限）或 LLM 提取的结构化摘要

    Raises:
        LLM 调用异常直接向上传播，不做降级处理
    """
    if not term_out or len(term_out) <= max_len:
        return term_out or ""

    prompt = CONDENSE_PROMPT_TEMPLATE.format(term_out=term_out)
    return llm_fn(prompt)


# ── Review Prompt 任务描述清理 ─────────────────────────────────────

# 需要整段移除的 section 标题（## 级别）
_REMOVE_SECTIONS = {"## Prizes", "## Prize", "## Timeline"}
# 需要整段移除的 subsection 标题（### 级别）
_REMOVE_SUBSECTIONS = {"### Citation", "### Acknowledgements"}

# 清理后最大字符数
_CLEAN_MAX_CHARS = 6000


def clean_task_desc(full_desc: str) -> str:
    """清理竞赛描述中的无用内容，保留 Reviewer 所需的关键信息。

    清理规则（不做"提取"，只做"删除"）:
    1. 移除图片标签 ![...](...) 和 HTML img 标签
    2. 移除 Prizes / Citation 等无关 section
    3. 移除 Evaluation 中的代码示例块（保留首段 metric 说明）
    4. 清理后 ≤ 6000 chars 直接返回，否则截断

    保留: Description 全部文字 + Evaluation 首段 + Submission File 格式
         + Dataset Description（数据结构、标签规则）

    Args:
        full_desc: 完整的 description.md 内容

    Returns:
        清理后的任务描述
    """
    if not full_desc:
        return ""

    text = full_desc

    # Phase 1: 移除图片标签
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)

    # Phase 2: 移除无关 section（## 级别）
    for heading in _REMOVE_SECTIONS:
        text = _remove_section(text, heading, level=2)

    # Phase 3: 移除无关 subsection（### 级别）
    for heading in _REMOVE_SUBSECTIONS:
        text = _remove_section(text, heading, level=3)

    # Phase 4: 移除 Evaluation 中的代码示例块（保留首段说明）
    text = _strip_eval_code_blocks(text)

    # Phase 5: 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Phase 6: 长度控制
    if len(text) > _CLEAN_MAX_CHARS:
        text = text[:_CLEAN_MAX_CHARS] + "\n... (truncated)"

    return text


def _remove_section(text: str, heading: str, level: int) -> str:
    """移除指定标题的整个 section（包括标题行到下一个同级或更高级标题）。

    Args:
        text: 完整文本
        heading: 标题文本（如 "## Prizes"）
        level: 标题级别（2 = ##, 3 = ###）

    Returns:
        移除 section 后的文本
    """
    pattern = rf"^{re.escape(heading)}\s*\n"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return text

    start = match.start()
    # 找下一个同级或更高级标题
    rest = text[match.end() :]
    next_heading = re.search(rf"^#{{1,{level}}}\s+", rest, re.MULTILINE)
    end = match.end() + next_heading.start() if next_heading else len(text)

    return text[:start] + text[end:]


def _strip_eval_code_blocks(text: str) -> str:
    """移除 Evaluation section 中的代码示例块，保留文字说明。

    Args:
        text: 完整文本

    Returns:
        移除代码块后的文本
    """
    eval_content = _extract_section(text, "## Evaluation")
    if not eval_content:
        return text

    # 移除代码块（```...```）
    cleaned_eval = re.sub(r"```[\s\S]*?```", "", eval_content)
    # 清理多余空行
    cleaned_eval = re.sub(r"\n{3,}", "\n\n", cleaned_eval).strip()

    return text.replace(eval_content, cleaned_eval)


def _extract_section(text: str, heading: str) -> Optional[str]:
    """提取指定 Markdown 标题下的内容。

    Args:
        text: 完整文本
        heading: 目标标题（如 "## Description"）

    Returns:
        该标题下的内容，或 None
    """
    level = len(heading) - len(heading.lstrip("#"))
    pattern = rf"^{re.escape(heading)}\s*\n"
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None

    start = match.end()
    next_heading = re.search(rf"^#{{1,{level}}}\s+", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)

    return text[start:end].strip()
