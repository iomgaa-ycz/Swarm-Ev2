"""Agent 层进化模块。

每 N 个 Epoch 评估所有 Agent 的表现，对弱者进行 Role 和 Strategy 变异。
"""

import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from agents.base_agent import BaseAgent
from core.backend import query
from core.evolution.experience_pool import ExperiencePool
from utils.config import Config
from utils.logger_system import log_msg, log_json, log_exception


class AgentEvolution:
    """Agent 层进化器（每 N Epoch 进化一次）。

    评估所有 Agent 的综合表现，识别精英和弱者，
    对弱者进行 Role 和 Strategy 变异（LLM 驱动）。

    Attributes:
        agents: Agent 列表
        experience_pool: 共享经验池
        skill_manager: Skill 池管理器（可选）
        config: 全局配置
        configs_dir: Agent 配置文件根目录
        evolution_interval: 进化间隔（每 N 个 Epoch）
        min_records: 进化前最小经验池记录数
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        experience_pool: ExperiencePool,
        config: Config,
        skill_manager: Optional[Any] = None,
    ):
        """初始化 Agent 进化器。

        Args:
            agents: Agent 列表
            experience_pool: 共享经验池
            config: 全局配置
            skill_manager: Skill 池管理器（可选，P3.5 使用）
        """
        self.agents = agents
        self.experience_pool = experience_pool
        self.skill_manager = skill_manager
        self.config = config

        # 加载配置
        self.configs_dir = Path(config.evolution.agent.configs_dir)
        self.evolution_interval = config.evolution.agent.evolution_interval
        self.min_records = config.evolution.agent.min_records_for_evolution

        log_msg(
            "INFO",
            f"Agent 进化器初始化: {len(agents)} 个 Agent, 间隔={self.evolution_interval} Epoch",
        )

    def evolve(self, epoch: int) -> None:
        """主入口：每 N 个 Epoch 进化一次。

        Args:
            epoch: 当前 Epoch 编号

        进化流程:
            1. 检查是否需要进化（epoch % interval == 0）
            2. 检查经验池记录数是否充足
            3. 评估所有 Agent 表现
            4. 识别精英（top-2）和弱者（bottom-2）
            5. 对弱者进行 Role 和 Strategy 变异
        """
        # [1] 检查是否需要进化
        if epoch % self.evolution_interval != 0 or epoch == 0:
            return

        log_msg("INFO", f"===== Agent 层进化开始（Epoch {epoch}） =====")

        # [2] 检查经验池记录数
        total_records = len(self.experience_pool.records)
        if total_records < self.min_records:
            log_msg(
                "WARNING",
                f"经验池记录数不足 ({total_records} < {self.min_records})，跳过进化",
            )
            return

        # [3] 评估所有 Agent
        scores = self._evaluate_agents()

        # [4] 识别精英和弱者
        elite_ids, weak_ids = self._identify_elites_and_weak(scores)

        log_msg("INFO", f"精英 Agent: {elite_ids}")
        log_msg("INFO", f"弱者 Agent: {weak_ids}")

        # [5] 对弱者进行变异
        self._mutate_weak_agents(weak_ids, elite_ids)

        # [6] Skill 池更新（P3.5 新增）
        if self.skill_manager:
            self._update_skill_pool()

        log_msg("INFO", f"===== Agent 层进化完成（Epoch {epoch}） =====")

    def _evaluate_agents(self) -> Dict[str, float]:
        """评估所有 Agent 的综合表现。

        Returns:
            Agent 评分字典，格式: {agent_id: score}

        评分公式:
            score = success_rate × avg_quality
        """
        scores = {}

        for agent in self.agents:
            stats = self.experience_pool.get_agent_stats(agent.name)

            # 计算综合得分
            score = stats["success_rate"] * stats["avg_quality"]
            scores[agent.name] = score

            log_json(
                {
                    "event": "agent_evaluation",
                    "agent_id": agent.name,
                    "total_count": stats["total_count"],
                    "success_rate": round(stats["success_rate"], 3),
                    "avg_quality": round(stats["avg_quality"], 3),
                    "score": round(score, 3),
                }
            )

        return scores

    def _identify_elites_and_weak(
        self, scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """识别精英和弱者。

        Args:
            scores: Agent 评分字典

        Returns:
            (elite_ids, weak_ids) - 精英和弱者的 Agent ID 列表

        分组策略:
            - 精英: Top-2（保留）
            - 弱者: Bottom-2（变异）
        """
        # 按得分降序排序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 提取精英（top-2）
        elite_ids = [agent_id for agent_id, _ in ranked[:2]]

        # 提取弱者（bottom-2）
        weak_ids = [agent_id for agent_id, _ in ranked[2:]]

        return elite_ids, weak_ids

    def _mutate_weak_agents(self, weak_ids: List[str], elite_ids: List[str]) -> None:
        """对弱者进行 Role 和 Strategy 变异。

        Args:
            weak_ids: 弱者 Agent ID 列表
            elite_ids: 精英 Agent ID 列表
        """
        for weak_id in weak_ids:
            # 随机选择一个精英作为参考
            elite_id = random.choice(elite_ids)

            log_msg("INFO", f"变异 {weak_id}（参考精英: {elite_id}）")

            # [1] 变异 Role
            try:
                self._mutate_role(weak_id, elite_id)
            except Exception as e:
                log_exception(e, f"变异 {weak_id} Role 失败")

            # [2] 变异 3 种 Strategy
            for task_type in ["explore", "merge", "mutate"]:
                try:
                    self._mutate_strategy(weak_id, task_type, elite_id)
                except Exception as e:
                    log_exception(e, f"变异 {weak_id} Strategy ({task_type}) 失败")

    def _mutate_role(self, weak_agent_id: str, elite_id: str) -> None:
        """Role 变异（LLM 生成新角色定位）。

        Args:
            weak_agent_id: 弱者 Agent ID
            elite_id: 精英 Agent ID（参考）
        """
        # [1] 读取当前 Role 和精英 Role
        current_role = self._load_agent_config(weak_agent_id, "role.md")
        elite_role = self._load_agent_config(elite_id, "role.md")

        # [2] 获取历史表现数据
        stats = self._get_performance_summary(weak_agent_id)

        # [3] 构建变异 Prompt
        mutation_prompt = self._build_mutation_prompt(
            current_content=current_role,
            elite_content=elite_role,
            stats=stats,
            section="role",
        )

        # [4] LLM 生成新 Role
        messages = [{"role": "user", "content": mutation_prompt}]
        response = query(messages=messages, config=self.config.llm.code)
        new_role = response.strip()

        # [5] 写入文件
        self._save_agent_config(weak_agent_id, "role.md", new_role)

        log_msg("INFO", f"✓ Role 变异完成: {weak_agent_id}")

    def _mutate_strategy(
        self, weak_agent_id: str, task_type: str, elite_id: str
    ) -> None:
        """Strategy 变异（分任务类型独立变异）。

        Args:
            weak_agent_id: 弱者 Agent ID
            task_type: 任务类型（"explore" | "merge" | "mutate"）
            elite_id: 精英 Agent ID（参考）
        """
        # [1] 读取当前 Strategy 和精英 Strategy
        filename = f"strategy_{task_type}.md"
        current_strategy = self._load_agent_config(weak_agent_id, filename)
        elite_strategy = self._load_agent_config(elite_id, filename)

        # [2] 获取历史表现数据（仅该任务类型）
        stats = self._get_performance_summary(weak_agent_id, task_type=task_type)

        # [3] 构建变异 Prompt
        mutation_prompt = self._build_mutation_prompt(
            current_content=current_strategy,
            elite_content=elite_strategy,
            stats=stats,
            section=f"strategy_{task_type}",
        )

        # [4] LLM 生成新 Strategy
        messages = [{"role": "user", "content": mutation_prompt}]
        response = query(messages=messages, config=self.config.llm.code)
        new_strategy = response.strip()

        # [5] 写入文件
        self._save_agent_config(weak_agent_id, filename, new_strategy)

        log_msg("INFO", f"✓ Strategy ({task_type}) 变异完成: {weak_agent_id}")

    def _get_performance_summary(
        self, agent_id: str, task_type: str = None
    ) -> Dict[str, Any]:
        """获取 Agent 历史表现摘要（用于变异 Prompt）。

        Args:
            agent_id: Agent 唯一标识
            task_type: 任务类型（None 表示所有任务类型）

        Returns:
            历史表现摘要字典，包含:
                - success_rate: 成功率
                - avg_quality: 平均质量
                - top_successes: Top-3 成功案例
                - top_failures: Top-3 失败案例
        """
        stats = self.experience_pool.get_agent_stats(agent_id)

        # Top-3 成功案例（output_quality > 0.7）
        top_successes = self.experience_pool.query(
            task_type=task_type,
            k=3,
            agent_id=agent_id,
            output_quality=(">", 0.7),
        )

        # Top-3 失败案例（output_quality < 0.3）
        top_failures = self.experience_pool.query(
            task_type=task_type,
            k=3,
            agent_id=agent_id,
            output_quality=("<", 0.3),
        )

        return {
            "success_rate": stats["success_rate"],
            "avg_quality": stats["avg_quality"],
            "top_successes": [r.strategy_summary for r in top_successes],
            "top_failures": [r.strategy_summary for r in top_failures],
        }

    def _build_mutation_prompt(
        self,
        current_content: str,
        elite_content: str,
        stats: Dict[str, Any],
        section: str,
    ) -> str:
        """构建变异 Prompt（Role 或 Strategy）。

        Args:
            current_content: 当前内容
            elite_content: 精英内容（参考）
            stats: 历史表现统计
            section: 配置节名称（"role" | "strategy_explore" | ...）

        Returns:
            完整的变异 Prompt
        """
        # 格式化成功/失败案例
        successes_str = "\n".join(
            f"  {i + 1}. {case}" for i, case in enumerate(stats["top_successes"])
        )
        if not successes_str:
            successes_str = "  （无）"

        failures_str = "\n".join(
            f"  {i + 1}. {case}" for i, case in enumerate(stats["top_failures"])
        )
        if not failures_str:
            failures_str = "  （无）"

        # 构建 Prompt
        prompt = f"""你是一个 Agent 配置进化器。

**当前配置 ({section})**:
{current_content}

**精英配置（参考）**:
{elite_content}

**历史表现**:
- 成功率: {stats["success_rate"]:.2%}
- 平均质量: {stats["avg_quality"]:.3f}
- Top-3 成功案例:
{successes_str}
- Top-3 失败案例:
{failures_str}

**任务**: 进化当前配置，针对失败模式增加规避建议，学习精英策略但保持差异性。

**约束**:
1. 保持 Markdown 格式
2. 保留成功策略要素
3. 输出完整的配置内容（300-500 字）
4. 包含必要的节标题（如"定位"、"核心策略"、"注意事项"）
5. 必须保持差异性（不完全复制精英）

**输出格式**:
直接输出 Markdown 内容，不要包含任何前缀说明。
"""

        return prompt

    def _load_agent_config(self, agent_id: str, filename: str) -> str:
        """加载 Agent 配置文件。

        Args:
            agent_id: Agent 唯一标识
            filename: 文件名（如 "role.md"）

        Returns:
            配置文件内容

        Raises:
            FileNotFoundError: 配置文件不存在
        """
        config_path = self.configs_dir / agent_id / filename

        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        return config_path.read_text(encoding="utf-8")

    def _save_agent_config(self, agent_id: str, filename: str, content: str) -> None:
        """保存 Agent 配置文件。

        Args:
            agent_id: Agent 唯一标识
            filename: 文件名（如 "role.md"）
            content: 配置内容
        """
        config_path = self.configs_dir / agent_id / filename

        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        config_path.write_text(content, encoding="utf-8")

        log_msg("DEBUG", f"配置文件已保存: {config_path}")

    def _update_skill_pool(self) -> None:
        """更新 Skill 池（P3.5 新增）。

        从经验池提取新 Skill 并演化 Skill 池。
        """
        if not self.skill_manager:
            return

        log_msg("INFO", "开始 Skill 池更新...")

        try:
            # 导入 SkillExtractor（避免循环导入）
            from core.evolution.skill_extractor import SkillExtractor

            # 创建提取器
            extractor = SkillExtractor(self.experience_pool, self.config)

            # 演化 Skill 池
            self.skill_manager.evolve_skills(self.experience_pool, extractor)

            # 重新加载 Skill 池
            self._reload_skills()

            log_msg("INFO", "Skill 池更新完成")

        except Exception as e:
            log_msg("ERROR", f"Skill 池更新失败: {e}")

    def _reload_skills(self) -> None:
        """重新加载 Skill 池（P3.5 新增）。

        通知 SkillManager 重新加载索引。
        """
        if not self.skill_manager:
            return

        self.skill_manager.reload_index()
        log_msg("INFO", "Skill 池已重新加载")
