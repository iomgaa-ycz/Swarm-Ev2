"""动态任务分配器模块。

基于 Epsilon-Greedy 策略选择最适合的 Agent 执行任务，
并通过指数移动平均更新 Agent 擅长度得分。
"""

import random
from typing import List, Dict

from agents.base_agent import BaseAgent
from utils.logger_system import log_msg, log_json


class TaskDispatcher:
    """动态任务分配器（Epsilon-Greedy 策略）。

    通过跟踪每个 Agent 对每种任务类型的擅长度得分，
    动态选择最适合的 Agent 执行任务。

    Attributes:
        agents: Agent 列表
        epsilon: 探索率（0-1 之间）
        learning_rate: EMA 更新学习率（α）
        specialization_scores: 擅长度得分矩阵
            格式: {agent_id: {task_type: score}}
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        epsilon: float = 0.3,
        learning_rate: float = 0.3,
    ):
        """初始化任务分配器。

        Args:
            agents: Agent 列表
            epsilon: 探索率（默认 0.3，即 30% 探索 + 70% 利用）
            learning_rate: EMA 更新学习率 α（默认 0.3）

        注意:
            - 初始化所有 Agent 对所有任务类型的擅长度得分为 0.5（中性）
            - epsilon=1.0 表示纯探索（随机选择）
            - epsilon=0.0 表示纯贪心（始终选择最优）
        """
        self.agents = agents
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        # 初始化擅长度得分矩阵（中性得分 0.5）
        self.specialization_scores: Dict[str, Dict[str, float]] = {
            agent.name: {"explore": 0.5, "merge": 0.5, "mutate": 0.5}
            for agent in agents
        }

        log_msg(
            "INFO",
            f"任务分配器初始化: {len(agents)} 个 Agent, epsilon={epsilon}, α={learning_rate}",
        )

    def select_agent(self, task_type: str) -> BaseAgent:
        """Epsilon-Greedy 选择 Agent。

        Args:
            task_type: 任务类型（"explore" | "merge" | "mutate"）

        Returns:
            选中的 Agent 对象

        策略:
            - 以 epsilon 概率随机选择（探索）
            - 以 (1-epsilon) 概率选择擅长度最高的 Agent（利用）
        """
        # Epsilon-Greedy 决策
        if random.random() < self.epsilon:
            # 探索：随机选择
            selected = random.choice(self.agents)
            log_msg(
                "DEBUG",
                f"任务分配（探索）: {task_type} -> {selected.name}",
            )
        else:
            # 利用：选择擅长度最高的 Agent
            selected = self._select_best(task_type)
            log_msg(
                "DEBUG",
                f"任务分配（利用）: {task_type} -> {selected.name} (得分={self.specialization_scores[selected.name][task_type]:.3f})",
            )

        return selected

    def update_score(self, agent_id: str, task_type: str, quality: float) -> None:
        """更新 Agent 擅长度得分（指数移动平均）。

        Args:
            agent_id: Agent 唯一标识
            task_type: 任务类型
            quality: 任务质量（0-1 之间的适应度值）

        更新公式:
            new_score = (1 - α) * old_score + α * quality

        时间复杂度: O(1)
        """
        if agent_id not in self.specialization_scores:
            log_msg("WARNING", f"未知 Agent: {agent_id}，跳过得分更新")
            return

        # 指数移动平均更新
        old_score = self.specialization_scores[agent_id][task_type]
        new_score = (1 - self.learning_rate) * old_score + self.learning_rate * quality
        self.specialization_scores[agent_id][task_type] = new_score

        log_json(
            {
                "event": "agent_score_update",
                "agent_id": agent_id,
                "task_type": task_type,
                "old_score": round(old_score, 3),
                "new_score": round(new_score, 3),
                "quality": round(quality, 3),
            }
        )

    def get_specialization_matrix(self) -> Dict[str, Dict[str, float]]:
        """获取完整的擅长度得分矩阵（用于诊断）。

        Returns:
            擅长度得分矩阵，格式:
                {
                    "agent_0": {"explore": 0.8, "merge": 0.6, "mutate": 0.5},
                    "agent_1": {"explore": 0.5, "merge": 0.7, "mutate": 0.6},
                    ...
                }

        时间复杂度: O(1)
        """
        return self.specialization_scores.copy()

    def _select_best(self, task_type: str) -> BaseAgent:
        """选择擅长度得分最高的 Agent（内部方法）。

        Args:
            task_type: 任务类型

        Returns:
            擅长度得分最高的 Agent

        时间复杂度: O(n)，n 为 Agent 数量
        """
        # 找到擅长度得分最高的 Agent
        best_agent = max(
            self.agents,
            key=lambda agent: self.specialization_scores[agent.name][task_type],
        )

        return best_agent
