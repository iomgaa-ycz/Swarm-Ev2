"""文本处理工具模块。

提供 Review Prompt 压缩、执行输出精炼等文本处理功能。
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


# ── Review Prompt 任务描述压缩 ─────────────────────────────────────


def compress_task_desc(full_desc: str) -> str:
    """从完整竞赛描述中提取 Review 所需的最小信息。

    提取规则:
    1. ## Description: 第一段（任务概述）
    2. ## Evaluation: 第一段（指标说明）
    3. ### Submission File: 格式示例

    Args:
        full_desc: 完整的 description.md 内容

    Returns:
        压缩后的任务描述（约 500 字节）
    """
    parts = []

    # Phase 1: 提取 Description 首段
    desc_text = _extract_section(full_desc, "## Description")
    if desc_text:
        first_para = desc_text.split("\n\n")[0].strip()
        parts.append(f"**Task**: {first_para}")

    # Phase 2: 提取 Evaluation 首段
    eval_text = _extract_section(full_desc, "## Evaluation")
    if eval_text:
        first_para = eval_text.split("\n\n")[0].strip()
        parts.append(f"**Metric**: {first_para}")

    # Phase 3: 提取 Submission File 格式
    submission_text = _extract_section(full_desc, "### Submission File")
    if submission_text:
        code_block = re.search(r"```[\s\S]*?```", submission_text)
        if code_block:
            parts.append(f"**Format**:\n{code_block.group()}")

    # Phase 4: 回退
    if not parts:
        return full_desc[:500] + "..."

    return "\n\n".join(parts)


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
