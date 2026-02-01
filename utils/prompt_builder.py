"""Prompt 模板构建器模块。

提供统一的 Prompt 生成逻辑，根据上下文动态调整。

此模块为 PromptManager 的轻量级包装器，核心逻辑基于 Jinja2 模板系统。
模板文件位于 benchmark/mle-bench/prompt_templates/ 目录。
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
                    "ERROR",
                    f"PromptManager 初始化失败: {e}",
                )
                raise RuntimeError(f"PromptManager 初始化失败: {e}") from e
        else:
            raise RuntimeError(
                "PromptManager 模块不可用，请确保 utils/prompt_manager.py 存在"
            )

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
        device_info: str = "",
        conda_packages: str = "",
        conda_env_name: str = "",
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

        Raises:
            RuntimeError: 如果 PromptManager 不可用
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
                "device_info": device_info,
                "conda_packages": conda_packages,
                "conda_env_name": conda_env_name,
            }

            return self.prompt_manager.build_prompt(
                task_type="explore",
                agent_id=agent_id,
                context=context,
            )

        # 如果 PromptManager 不可用，抛出异常（不使用低质量回退）
        raise RuntimeError(
            "PromptManager 不可用，无法构建 explore prompt。"
            "请确保 Jinja2 模板目录配置正确。"
        )

    def build_merge_prompt(
        self,
        task_desc: str,
        parent_a: Node,
        parent_b: Node,
        gene_plan: dict,
        time_remaining: int = 0,
        steps_remaining: int = 0,
        agent_id: str = "agent_0",
        experience_pool=None,
        device_info: str = "",
        conda_packages: str = "",
        conda_env_name: str = "",
    ) -> str:
        """构建 merge 任务 Prompt（基因交叉）。

        Args:
            task_desc: 任务描述
            parent_a: 父代 A 节点
            parent_b: 父代 B 节点
            gene_plan: 基因交叉计划 {gene_name: "A" or "B"}
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数
            agent_id: Agent ID
            experience_pool: 经验池实例

        Returns:
            完整的 Prompt 字符串
        """
        # 新逻辑：使用 PromptManager
        if self.prompt_manager is not None:
            context = {
                "task_desc": task_desc,
                "parent_a": parent_a,
                "parent_b": parent_b,
                "gene_plan": gene_plan,
                "time_remaining": time_remaining,
                "steps_remaining": steps_remaining,
                "experience_pool": experience_pool,
                "device_info": device_info,
                "conda_packages": conda_packages,
                "conda_env_name": conda_env_name,
            }

            return self.prompt_manager.build_prompt(
                task_type="merge",
                agent_id=agent_id,
                context=context,
            )

        # 如果 PromptManager 不可用，抛出异常（不使用低质量回退）
        raise RuntimeError(
            "PromptManager 不可用，无法构建 merge prompt。"
            "请确保 Jinja2 模板目录配置正确。"
        )

    def build_mutate_prompt(
        self,
        task_desc: str,
        parent_node: Node,
        target_gene: str,
        time_remaining: int = 0,
        steps_remaining: int = 0,
        agent_id: str = "agent_0",
        experience_pool=None,
        device_info: str = "",
        conda_packages: str = "",
        conda_env_name: str = "",
    ) -> str:
        """构建 mutate 任务 Prompt（基因变异）。

        Args:
            task_desc: 任务描述
            parent_node: 父节点
            target_gene: 目标基因块名称（如 "MODEL"）
            time_remaining: 剩余时间（秒）
            steps_remaining: 剩余步数
            agent_id: Agent ID
            experience_pool: 经验池实例

        Returns:
            完整的 Prompt 字符串
        """
        # 新逻辑：使用 PromptManager
        if self.prompt_manager is not None:
            context = {
                "task_desc": task_desc,
                "parent_node": parent_node,
                "target_gene": target_gene,
                "time_remaining": time_remaining,
                "steps_remaining": steps_remaining,
                "experience_pool": experience_pool,
                "device_info": device_info,
                "conda_packages": conda_packages,
                "conda_env_name": conda_env_name,
            }

            return self.prompt_manager.build_prompt(
                task_type="mutate",
                agent_id=agent_id,
                context=context,
            )

        # 如果 PromptManager 不可用，抛出异常（不使用低质量回退）
        raise RuntimeError(
            "PromptManager 不可用，无法构建 mutate prompt。"
            "请确保 Jinja2 模板目录配置正确。"
        )
