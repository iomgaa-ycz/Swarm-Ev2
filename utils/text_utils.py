"""文本处理工具模块。

提供 Review Prompt 压缩等文本处理功能。
"""

import re
from typing import Optional


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
