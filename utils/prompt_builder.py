"""Prompt 模板构建器模块。

提供统一的 Prompt 生成逻辑，根据上下文动态调整。
"""

from typing import Optional

from core.state import Node


class PromptBuilder:
    """Prompt 模板构建器。

    提供统一的 Prompt 构建逻辑，根据 parent_node 动态插入上下文，
    让 LLM 根据上下文自动判断任务模式（初稿/改进/修复）。

    Attributes:
        obfuscate: 是否混淆任务描述（用于防止 LLM 识别评测平台）
    """

    def __init__(self, obfuscate: bool = False):
        """初始化 PromptBuilder。

        Args:
            obfuscate: 是否混淆任务描述（默认 False）
        """
        self.obfuscate = obfuscate

    def build_explore_prompt(
        self,
        task_desc: str,
        parent_node: Optional[Node] = None,
        memory: str = "",
        data_preview: Optional[str] = None,
        time_remaining: int = 0,
        steps_remaining: int = 0,
    ) -> str:
        """构建统一的 explore prompt。

        根据 parent_node 自动适配：
        - None: 初稿模式（LLM 自动判断）
        - is_buggy: 修复模式（LLM 看到错误输出会自动修复）
        - 正常: 改进模式（LLM 看到正常输出会自动改进）

        Args:
            task_desc: 任务描述
            parent_node: 父节点（None=初稿, buggy=修复, normal=改进）
            memory: Journal 摘要（历史经验）
            data_preview: 数据预览文本
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数

        Returns:
            完整的 Prompt 字符串

        注意:
            - 不显式告诉 LLM 任务类型，让 LLM 根据上下文判断
            - 没有 Previous Attempt → LLM 知道要生成初稿
            - 有 Previous Attempt + 错误输出 → LLM 知道要修复
            - 有 Previous Attempt + 正常输出 → LLM 知道要改进
        """
        sections = [
            f"# Introduction\n{self._get_role_intro()}",
            f"# Task Description\n{task_desc}",
        ]

        # 动态部分：提供上下文，让 LLM 自动判断
        if parent_node is not None:
            sections.append(f"# Previous Attempt\n```python\n{parent_node.code}\n```")
            sections.append(f"# Execution Result\n```\n{parent_node.term_out}\n```")

        if memory:
            sections.append(f"# Memory\n{memory}")

        if data_preview:
            sections.append(f"# Data Overview\n{data_preview}")

        sections.extend(
            [
                f"# Guidelines\n{self._get_guidelines(time_remaining, steps_remaining)}",
                f"# Response Format\n{self._get_response_format()}",
            ]
        )

        return "\n\n".join(sections)

    def _get_role_intro(self) -> str:
        """获取角色介绍。

        Returns:
            角色介绍字符串

        注意:
            - obfuscate=True: 通用 ML 工程师（隐藏 Kaggle 背景）
            - obfuscate=False: Kaggle 大师（显式身份）
        """
        if self.obfuscate:
            return "You are an expert machine learning engineer attempting a task."
        return "You are a Kaggle grandmaster attending a competition."

    def _get_guidelines(self, time_remaining: int, steps_remaining: int) -> str:
        """获取实现指南。

        Args:
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数

        Returns:
            实现指南字符串
        """
        # 格式化剩余时间（秒 → 人类可读）
        time_str = self._format_time(time_remaining)

        return f"""**Implementation Guidelines**:
- <TOTAL_TIME_REMAINING: {time_str}>
- <TOTAL_STEPS_REMAINING: {steps_remaining}>
- The code should **implement the proposed solution**
- **PRINT the validation metric value**
- **SAVE predictions to `./submission/submission.csv`**
- The code should be a single-file Python program
- All input data is in `./input/` directory
- Use `./working/` for temporary files"""

    def _get_response_format(self) -> str:
        """获取响应格式说明。

        Returns:
            响应格式字符串
        """
        return """**Response Format**:
Your response should contain:
1. A brief outline (3-5 sentences) of your proposed solution
2. A single markdown code block (```python...```) implementing this solution

DO NOT include additional headings or explanations outside the code block."""

    def _format_time(self, seconds: int) -> str:
        """格式化时间（秒 → 人类可读）。

        Args:
            seconds: 剩余时间（秒）

        Returns:
            格式化的时间字符串（如 "1 hour 23 minutes"）

        注意:
            - 简化实现，不依赖 humanize 库
            - 格式: "X hours Y minutes" 或 "X minutes" 或 "X seconds"
        """
        if seconds <= 0:
            return "0 seconds"

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours} hour" + ("s" if hours > 1 else ""))
        if minutes > 0:
            parts.append(f"{minutes} minute" + ("s" if minutes > 1 else ""))
        if remaining_seconds > 0 and not parts:
            parts.append(
                f"{remaining_seconds} second" + ("s" if remaining_seconds > 1 else "")
            )

        return " ".join(parts) if parts else "0 seconds"
