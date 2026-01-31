"""Prompt 模板构建器模块。

提供统一的 Prompt 生成逻辑，根据上下文动态调整。

注意:
    此模块已重构为 PromptManager 的轻量级包装器，保持接口兼容性。
    核心逻辑已迁移到 utils/prompt_manager.py（基于 Jinja2）。
"""

from pathlib import Path
from typing import Optional

from core.state import Node
from utils.logger_system import log_msg

# 延迟导入 PromptManager（避免循环依赖）
try:
    from utils.prompt_manager import PromptManager
except ImportError:
    PromptManager = None


class PromptBuilder:
    """Prompt 模板构建器。

    提供统一的 Prompt 构建逻辑，根据 parent_node 动态插入上下文，
    让 LLM 根据上下文自动判断任务模式（初稿/改进/修复）。

    Attributes:
        obfuscate: 是否混淆任务描述（用于防止 LLM 识别评测平台）
        prompt_manager: PromptManager 实例（使用新的 Jinja2 系统）
    """

    def __init__(
        self,
        obfuscate: bool = False,
        template_dir: Optional[Path] = None,
        skills_dir: Optional[Path] = None,
        agent_configs_dir: Optional[Path] = None,
    ):
        """初始化 PromptBuilder。

        Args:
            obfuscate: 是否混淆任务描述（默认 False）
            template_dir: Jinja2 模板目录（默认使用项目标准路径）
            skills_dir: Skill 文件根目录（默认使用项目标准路径）
            agent_configs_dir: Agent 配置文件根目录（默认使用项目标准路径）
        """
        self.obfuscate = obfuscate

        # 初始化 PromptManager（新系统）
        if PromptManager is not None:
            # 使用默认路径（项目标准结构）
            base_dir = Path(__file__).parent.parent / "benchmark" / "mle-bench"
            template_dir = template_dir or base_dir / "prompt_templates"
            skills_dir = skills_dir or base_dir / "skills"
            agent_configs_dir = agent_configs_dir or base_dir / "agent_configs"

            try:
                self.prompt_manager = PromptManager(
                    template_dir=template_dir,
                    skills_dir=skills_dir,
                    agent_configs_dir=agent_configs_dir,
                )
                log_msg("INFO", "PromptBuilder 已初始化（使用 PromptManager 新系统）")
            except Exception as e:
                log_msg(
                    "WARNING",
                    f"PromptManager 初始化失败，回退到旧逻辑: {e}",
                )
                self.prompt_manager = None
        else:
            log_msg("WARNING", "PromptManager 不可用，使用旧版 PromptBuilder 逻辑")
            self.prompt_manager = None

    def build_explore_prompt(
        self,
        task_desc: str,
        parent_node: Optional[Node] = None,
        memory: str = "",
        data_preview: Optional[str] = None,
        time_remaining: int = 0,
        steps_remaining: int = 0,
        agent_id: str = "agent_0",
        experience_pool=None,
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
            agent_id: Agent ID（默认 "agent_0"）
            experience_pool: 经验池实例（用于注入 Top-K Skill）

        Returns:
            完整的 Prompt 字符串

        注意:
            - 优先使用 PromptManager（新系统，基于 Jinja2 模板）
            - 如果 PromptManager 不可用，回退到旧逻辑
        """
        # 新逻辑：使用 PromptManager
        if self.prompt_manager is not None:
            context = {
                "task_desc": task_desc,
                "parent_node": parent_node,
                "memory": memory,
                "data_preview": data_preview,
                "time_remaining": time_remaining,
                "steps_remaining": steps_remaining,
                "experience_pool": experience_pool,
            }

            return self.prompt_manager.build_prompt(
                task_type="explore",
                agent_id=agent_id,
                context=context,
            )

        # 旧逻辑：回退到手动拼接（保持向后兼容）
        return self._build_explore_prompt_legacy(
            task_desc=task_desc,
            parent_node=parent_node,
            memory=memory,
            data_preview=data_preview,
            time_remaining=time_remaining,
            steps_remaining=steps_remaining,
        )

    def _build_explore_prompt_legacy(
        self,
        task_desc: str,
        parent_node: Optional[Node] = None,
        memory: str = "",
        data_preview: Optional[str] = None,
        time_remaining: int = 0,
        steps_remaining: int = 0,
    ) -> str:
        """旧版 Prompt 构建逻辑（回退兼容）。

        此方法保留原有的手动拼接逻辑，用于在 PromptManager 不可用时回退。

        Args:
            task_desc: 任务描述
            parent_node: 父节点（None=初稿, buggy=修复, normal=改进）
            memory: Journal 摘要（历史经验）
            data_preview: 数据预览文本
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数

        Returns:
            完整的 Prompt 字符串
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
