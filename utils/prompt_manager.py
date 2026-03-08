"""Prompt 管理器模块。

基于 Jinja2 的统一 Prompt 管理系统，支持静态/动态 Skill 加载和模板渲染。
Agent 配置使用单模板 + 内存字典方式管理，避免磁盘多副本同步问题。
"""

import copy
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
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
        _agent_configs: 内存中的 Agent 配置字典 {agent_id: {section: content}}
    """

    def __init__(
        self,
        template_dir: Path,
        skills_dir: Path,
        agent_configs_dir: Path,
        skill_manager: Optional[Any] = None,
        num_agents: int = 4,
        spec_path: Optional[Path] = None,
    ):
        """初始化 PromptManager。

        Args:
            template_dir: Jinja2 模板目录（如 benchmark/mle-bench/prompt_templates）
            skills_dir: Skill 文件根目录（如 benchmark/mle-bench/skills）
            agent_configs_dir: Agent 配置文件根目录（如 benchmark/mle-bench/agent_configs）
            skill_manager: Skill 池管理器（可选，P3.5 使用）
            num_agents: Agent 数量，从 default/ 模板复制 N 份到内存
            spec_path: prompt_spec.yaml 路径（可选，自动查找 template_dir 父目录）
        """
        self.skills_dir = Path(skills_dir)
        self.agent_configs_dir = Path(agent_configs_dir)
        self.skill_manager = skill_manager

        # 加载 prompt_spec.yaml（校验规格）
        self._prompt_spec: Optional[Dict] = self._load_prompt_spec(
            spec_path, Path(template_dir)
        )

        # 从 default/ 模板加载配置到内存
        self._agent_configs: Dict[str, Dict[str, str]] = {}
        self._load_default_configs(num_agents)

        # 初始化 Jinja2 环境
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # 注册自定义函数到 Jinja2 全局命名空间
        self.env.globals["load_skill"] = self.load_skill
        self.env.globals["load_agent_config"] = self.load_agent_config

        log_msg("INFO", f"PromptManager 初始化完成（模板目录: {template_dir}, agents: {num_agents}）")

    def _load_prompt_spec(
        self, spec_path: Optional[Path], template_dir: Path
    ) -> Optional[Dict]:
        """加载 prompt_spec.yaml 校验规格。

        查找顺序：
        1. 显式指定的 spec_path
        2. template_dir 的父目录下的 prompt_spec.yaml

        Args:
            spec_path: 显式指定的路径（可选）
            template_dir: 模板目录，用于自动查找

        Returns:
            解析后的 spec 字典，或 None（文件不存在时）
        """
        if spec_path is not None:
            path = Path(spec_path)
        else:
            path = Path(template_dir).parent / "prompt_spec.yaml"

        if not path.exists():
            log_msg("DEBUG", f"prompt_spec.yaml 不存在: {path}，跳过校验")
            return None

        with open(path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        log_msg("INFO", f"prompt_spec.yaml 已加载: {path}")
        return spec

    def _validate_context(self, task_type: str, context: Dict[str, Any]) -> None:
        """校验 context 是否满足 prompt_spec 定义的字段要求（WARNING 模式）。

        Phase 1 使用 WARNING 不阻断，Phase 4 切换为 ERROR 阻断。

        Args:
            task_type: 任务类型（draft/debug/merge/mutate）
            context: 用户提供的上下文字典
        """
        if self._prompt_spec is None:
            return

        templates = self._prompt_spec.get("templates", {})
        spec = templates.get(task_type)
        if spec is None:
            log_msg("DEBUG", f"prompt_spec 中未定义 task_type={task_type}，跳过校验")
            return

        # 检查 required_context
        required: List[str] = spec.get("required_context", [])
        for field in required:
            if field not in context or context[field] is None:
                log_msg(
                    "WARNING",
                    f"[prompt_spec] task_type={task_type} 缺少 required 字段: {field}",
                )

        # 检查 optional_context
        optional: List[str] = spec.get("optional_context", [])
        for field in optional:
            if field not in context or context[field] is None:
                log_msg(
                    "DEBUG",
                    f"[prompt_spec] task_type={task_type} 缺少 optional 字段: {field}",
                )

    def _load_default_configs(self, num_agents: int) -> None:
        """从 default/ 目录加载模板配置，按 num_agents 复制到内存。

        Args:
            num_agents: 需要创建的 Agent 数量
        """
        default_dir = self.agent_configs_dir / "default"
        if not default_dir.exists():
            log_msg("WARNING", f"默认配置目录不存在: {default_dir}，跳过加载")
            return

        # 读取 default/ 下所有 .md 文件作为模板
        template_config: Dict[str, str] = {}
        for md_file in sorted(default_dir.glob("*.md")):
            section = md_file.stem  # 如 "role", "strategy_explore"
            template_config[section] = md_file.read_text(encoding="utf-8")

        # 为每个 Agent 复制一份
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self._agent_configs[agent_id] = copy.deepcopy(template_config)

        log_msg(
            "INFO",
            f"Agent 配置已加载到内存: {num_agents} 个 Agent, "
            f"每个 {len(template_config)} 个 section ({list(template_config.keys())})",
        )

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
        """从内存字典加载 Agent 配置。

        Args:
            agent_id: Agent ID（如 "agent_0"）
            section: 配置节名称（如 "role" 或 "strategy_explore"）

        Returns:
            配置文件内容（Markdown 格式）

        Raises:
            FileNotFoundError: Agent 或 section 不存在
        """
        # 去掉 .md 后缀（内存字典用 stem 作为 key）
        if section.endswith(".md"):
            section = section[:-3]

        if agent_id not in self._agent_configs:
            log_msg("ERROR", f"Agent 配置不存在: {agent_id}")
            raise FileNotFoundError(f"Agent 配置不存在: {agent_id}")

        agent_cfg = self._agent_configs[agent_id]
        if section not in agent_cfg:
            log_msg("ERROR", f"Agent 配置节不存在: {agent_id}/{section}")
            raise FileNotFoundError(f"Agent 配置节不存在: {agent_id}/{section}")

        return agent_cfg[section]

    def update_agent_config(self, agent_id: str, section: str, content: str) -> None:
        """更新内存中指定 Agent 的指定 section。

        Args:
            agent_id: Agent ID（如 "agent_0"）
            section: 配置节名称（如 "role" 或 "strategy_explore"）
            content: 新的配置内容
        """
        # 去掉 .md 后缀
        if section.endswith(".md"):
            section = section[:-3]

        if agent_id not in self._agent_configs:
            self._agent_configs[agent_id] = {}

        self._agent_configs[agent_id][section] = content
        log_msg("DEBUG", f"Agent 配置已更新: {agent_id}/{section}")

    def export_agent_configs(self, output_dir: Path) -> None:
        """导出所有 Agent 的最终配置到指定目录。

        Args:
            output_dir: 输出目录（如 workspace/logs/agent_configs_final/）
        """
        output_dir = Path(output_dir)
        for agent_id, sections in self._agent_configs.items():
            agent_dir = output_dir / agent_id
            agent_dir.mkdir(parents=True, exist_ok=True)
            for section, content in sections.items():
                (agent_dir / f"{section}.md").write_text(content, encoding="utf-8")

        log_msg("INFO", f"Agent 配置已导出: {output_dir} ({len(self._agent_configs)} 个 Agent)")

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
            task_type: 任务类型（"draft" / "merge" / "mutate"）
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

        records = experience_pool.query(
            task_type=task_type,
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
            task_type: 任务类型（"draft" / "merge" / "mutate"）
            agent_id: Agent ID（如 "agent_0"）
            context: Prompt 上下文字典，必须包含：
                - task_desc: str - 任务描述
                - parent_node: Optional[Node] - 父节点
                - memory: str - Journal 摘要
                - data_preview: Optional[str] - 数据预览
                - time_remaining: int - 剩余时间（秒）
                - steps_remaining: int - 剩余步数
                对于 merge 任务，还需要：
                - primary_parent: Node - 贡献基因最多的父代
                - gene_plan: str - 基因交叉计划（Markdown 格式）
                对于 mutate 任务，还需要：
                - target_gene: str - 目标基因块

        Returns:
            渲染完成的 Prompt 字符串

        Raises:
            TemplateNotFound: 模板文件不存在
        """
        # 校验 context 字段（WARNING 模式）
        self._validate_context(task_type, context)

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

        # 格式化单节点执行超时
        exec_timeout_str = self._format_time(context.get("exec_timeout", 5400))
        template_context["exec_timeout_str"] = exec_timeout_str

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
