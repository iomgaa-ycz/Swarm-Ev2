"""Prompt 管理器模块。

基于 Jinja2 的统一 Prompt 管理系统，支持静态/动态 Skill 加载和模板渲染。
"""

from pathlib import Path
from typing import Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from core.evolution.experience_pool import ExperiencePool
from utils.logger_system import log_msg


class PromptManager:
    """Prompt 管理器。

    基于 Jinja2 模板引擎，提供 7 层结构化 Prompt 构建能力：
    1. ROLE - Agent 角色定位（可进化）
    2. FORMAT - 输出格式规范（静态）
    3. TASK - 任务描述
    4. CONTEXT - 上下文（parent_node、memory、data_preview）
    5. STRATEGY - 策略（静态 Skill + Agent 策略配置）
    6. EXAMPLES - Top-K 成功案例（动态，来自经验池）
    7. GUIDELINES - 工作空间规则 + 时间约束

    Attributes:
        skills_dir: Skill 文件根目录
        agent_configs_dir: Agent 配置文件根目录
        env: Jinja2 环境实例
    """

    def __init__(
        self,
        template_dir: Path,
        skills_dir: Path,
        agent_configs_dir: Path,
        skill_manager: Optional[Any] = None,
    ):
        """初始化 PromptManager。

        Args:
            template_dir: Jinja2 模板目录（如 benchmark/mle-bench/prompt_templates）
            skills_dir: Skill 文件根目录（如 benchmark/mle-bench/skills）
            agent_configs_dir: Agent 配置文件根目录（如 benchmark/mle-bench/agent_configs）
            skill_manager: Skill 池管理器（可选，P3.5 使用）
        """
        self.skills_dir = Path(skills_dir)
        self.agent_configs_dir = Path(agent_configs_dir)
        self.skill_manager = skill_manager

        # 初始化 Jinja2 环境
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # 注册自定义函数到 Jinja2 全局命名空间
        self.env.globals["load_skill"] = self.load_skill
        self.env.globals["load_agent_config"] = self.load_agent_config

        log_msg("INFO", f"PromptManager 初始化完成（模板目录: {template_dir}）")

    def load_skill(self, skill_path: str) -> str:
        """加载静态 Skill 文件。

        Args:
            skill_path: Skill 相对路径（相对于 skills_dir），如 "static/output_format"

        Returns:
            Skill 文件内容（Markdown 格式）

        Raises:
            FileNotFoundError: Skill 文件不存在
        """
        # 支持带或不带 .md 后缀
        if not skill_path.endswith(".md"):
            skill_path += ".md"

        full_path = self.skills_dir / skill_path

        if not full_path.exists():
            log_msg("ERROR", f"Skill 文件不存在: {full_path}")
            raise FileNotFoundError(f"Skill 文件不存在: {full_path}")

        content = full_path.read_text(encoding="utf-8")
        return content

    def load_agent_config(self, agent_id: str, section: str) -> str:
        """加载 Agent 配置文件。

        Args:
            agent_id: Agent ID（如 "agent_0"）
            section: 配置节名称（如 "role" 或 "strategy_explore"）

        Returns:
            配置文件内容（Markdown 格式）

        Raises:
            FileNotFoundError: 配置文件不存在
        """
        # 支持带或不带 .md 后缀
        if not section.endswith(".md"):
            section += ".md"

        full_path = self.agent_configs_dir / agent_id / section

        if not full_path.exists():
            log_msg("ERROR", f"Agent 配置文件不存在: {full_path}")
            raise FileNotFoundError(f"Agent 配置文件不存在: {full_path}")

        content = full_path.read_text(encoding="utf-8")
        return content

    def inject_top_k_skills(
        self,
        task_type: str,
        k: int = 5,
        experience_pool: Optional[ExperiencePool] = None,
    ) -> str:
        """注入 Top-K 动态 Skill（从 SkillManager 或经验池）。

        优先从 SkillManager 获取 Top-K Skill，
        如果 SkillManager 不可用，则从经验池提取成功案例。

        Args:
            task_type: 任务类型（"draft" / "explore" / "merge" / "mutate"）
            k: 提取 Top-K 数量
            experience_pool: 经验池实例（Fallback 使用）

        Returns:
            格式化的 Markdown 字符串
        """
        # [1] 优先从 SkillManager 获取（P3.5）
        if self.skill_manager:
            skills = self.skill_manager.get_top_k_skills(task_type, k)
            if skills:
                return self._format_skill_examples(skills)

        # [2] Fallback: 从经验池提取成功案例
        if experience_pool is None:
            log_msg(
                "WARNING", "未提供 skill_manager 或 experience_pool，跳过 Skill 注入"
            )
            return ""

        # "draft" 与 "explore" 共用经验池
        query_type = "explore" if task_type == "draft" else task_type
        records = experience_pool.query(
            task_type=query_type,
            k=k,
            output_quality=(">", 0.5),
        )

        if not records:
            return ""

        # 格式化为 Markdown
        lines = ["# Top-K Success Examples\n"]
        lines.append(
            f"以下是 {len(records)} 个成功案例，展示了高质量的 {task_type} 策略：\n"
        )

        for i, record in enumerate(records, 1):
            lines.append(f"## Example {i} (Quality: {record.output_quality:.2f})")
            lines.append(f"**Agent**: {record.agent_id}")
            lines.append("**Strategy Summary**:")
            lines.append(f"{record.strategy_summary}")
            lines.append("")

        return "\n".join(lines)

    def build_prompt(
        self,
        task_type: str,
        agent_id: str,
        context: Dict[str, Any],
    ) -> str:
        """构建完整 Prompt（主入口）。

        Args:
            task_type: 任务类型（"explore" / "merge" / "mutate"）
            agent_id: Agent ID（如 "agent_0"）
            context: Prompt 上下文字典，必须包含：
                - task_desc: str - 任务描述
                - parent_node: Optional[Node] - 父节点
                - memory: str - Journal 摘要
                - data_preview: Optional[str] - 数据预览
                - time_remaining: int - 剩余时间（秒）
                - steps_remaining: int - 剩余步数
                对于 merge 任务，还需要：
                - parent_a: Node - 父节点 A
                - parent_b: Node - 父节点 B
                - gene_plan: Dict[str, str] - 基因交叉计划
                对于 mutate 任务，还需要：
                - target_gene: str - 目标基因块

        Returns:
            渲染完成的 Prompt 字符串

        Raises:
            TemplateNotFound: 模板文件不存在
        """
        # 选择对应的模板
        template_name = f"{task_type}.j2"

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            log_msg("ERROR", f"模板文件不存在: {template_name}")
            raise

        # 准备模板上下文（包含所有必要变量）
        template_context = {
            "agent_id": agent_id,
            "task_type": task_type,
            **context,  # 展开用户提供的上下文
        }

        # 格式化时间（秒 → 人类可读）
        time_str = self._format_time(context.get("time_remaining", 0))
        template_context["time_str"] = time_str

        # 注入 Top-K 动态 Skill（如果经验池可用）
        experience_pool = context.get("experience_pool")
        if experience_pool is not None:
            dynamic_skills = self.inject_top_k_skills(
                task_type=task_type,
                k=context.get("top_k", 5),
                experience_pool=experience_pool,
            )
            template_context["dynamic_skills"] = dynamic_skills
        else:
            template_context["dynamic_skills"] = ""

        # 渲染模板
        prompt = template.render(**template_context)

        log_msg("DEBUG", f"Prompt 构建完成（任务: {task_type}, Agent: {agent_id}）")
        return prompt

    def _format_time(self, seconds: int) -> str:
        """格式化时间（秒 → 人类可读）。

        Args:
            seconds: 剩余时间（秒）

        Returns:
            格式化的时间字符串（如 "1 hour 23 minutes"）
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

    def update_skill_pool(self, skill_manager: Any) -> None:
        """更新 Skill 池引用（Agent 进化后调用）。

        Args:
            skill_manager: 新的 SkillManager 实例
        """
        self.skill_manager = skill_manager
        log_msg("INFO", "PromptManager Skill 池已更新")

    def _format_skill_examples(self, skills: list) -> str:
        """格式化 Skill 为 Markdown 列表。

        Args:
            skills: Skill 内容列表

        Returns:
            Markdown 格式字符串
        """
        if not skills:
            return "无可用的成功案例。"

        formatted = "# 成功案例（Top-K Skill）\n\n"
        formatted += f"以下是 {len(skills)} 个经过验证的高质量策略模式：\n\n"

        for i, skill in enumerate(skills, 1):
            formatted += f"## 示例 {i}\n{skill}\n\n"

        return formatted
