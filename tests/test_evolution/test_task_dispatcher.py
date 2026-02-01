"""TaskDispatcher 单元测试。

测试动态任务分配器的 Epsilon-Greedy 策略和 EMA 得分更新。
"""

import pytest
from unittest.mock import MagicMock

from core.evolution import TaskDispatcher
from agents.base_agent import BaseAgent


@pytest.fixture
def mock_agents():
    """创建 Mock Agent 列表。"""
    agents = []
    for i in range(4):
        agent = MagicMock(spec=BaseAgent)
        agent.name = f"agent_{i}"
        agents.append(agent)
    return agents


class TestTaskDispatcher:
    """TaskDispatcher 测试套件。"""

    def test_initialization(self, mock_agents):
        """测试初始化擅长度得分为 0.5。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.3, learning_rate=0.3)

        # 验证所有 Agent 对所有任务类型的初始得分为 0.5
        for agent in mock_agents:
            scores = dispatcher.specialization_scores[agent.name]
            assert scores["explore"] == 0.5
            assert scores["merge"] == 0.5
            assert scores["mutate"] == 0.5

    def test_epsilon_greedy_explore(self, mock_agents):
        """测试 epsilon=1.0 时纯探索（随机选择）。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=1.0)

        # 多次选择，验证所有 Agent 都被选择过
        selections = [dispatcher.select_agent("explore").name for _ in range(100)]

        # 验证所有 4 个 Agent 都被选择过（分布均匀）
        unique_agents = set(selections)
        assert len(unique_agents) == 4

    def test_epsilon_greedy_exploit(self, mock_agents):
        """测试 epsilon=0.0 时纯贪心（始终选择最优）。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.0)

        # 手动设置擅长度得分
        dispatcher.specialization_scores = {
            "agent_0": {"explore": 0.9, "merge": 0.5, "mutate": 0.5},
            "agent_1": {"explore": 0.3, "merge": 0.5, "mutate": 0.5},
            "agent_2": {"explore": 0.6, "merge": 0.5, "mutate": 0.5},
            "agent_3": {"explore": 0.4, "merge": 0.5, "mutate": 0.5},
        }

        # 多次选择，验证始终选择 agent_0（得分最高）
        selections = [dispatcher.select_agent("explore").name for _ in range(10)]
        assert all(name == "agent_0" for name in selections)

    def test_score_update(self, mock_agents):
        """测试 EMA 得分更新公式。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.3, learning_rate=0.3)

        # 初始得分
        initial_score = dispatcher.specialization_scores["agent_0"]["explore"]
        assert initial_score == 0.5

        # 更新得分（高质量）
        dispatcher.update_score("agent_0", "explore", quality=0.9)

        # 验证得分提升（EMA 公式）
        expected = (1 - 0.3) * 0.5 + 0.3 * 0.9
        actual = dispatcher.specialization_scores["agent_0"]["explore"]
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_score_update_multiple_times(self, mock_agents):
        """测试多次更新得分。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.3, learning_rate=0.3)

        # 连续更新 3 次
        qualities = [0.8, 0.9, 0.7]
        score = 0.5  # 初始得分

        for quality in qualities:
            dispatcher.update_score("agent_0", "explore", quality)
            # 计算预期得分
            score = (1 - 0.3) * score + 0.3 * quality

        # 验证最终得分
        actual = dispatcher.specialization_scores["agent_0"]["explore"]
        assert actual == pytest.approx(score, rel=1e-6)

    def test_score_update_unknown_agent(self, mock_agents):
        """测试未知 Agent 更新得分（应跳过）。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.3)

        # 更新未知 Agent（不应抛出异常）
        dispatcher.update_score("unknown_agent", "explore", 0.9)

        # 验证擅长度矩阵未被污染
        assert "unknown_agent" not in dispatcher.specialization_scores

    def test_get_specialization_matrix(self, mock_agents):
        """测试获取擅长度矩阵。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.3)

        # 更新部分得分
        dispatcher.update_score("agent_0", "explore", 0.8)
        dispatcher.update_score("agent_1", "merge", 0.7)

        # 获取矩阵
        matrix = dispatcher.get_specialization_matrix()

        # 验证格式
        assert len(matrix) == 4
        assert "agent_0" in matrix
        assert "explore" in matrix["agent_0"]

        # 验证得分已更新
        assert matrix["agent_0"]["explore"] != 0.5
        assert matrix["agent_1"]["merge"] != 0.5

        # 验证其他得分保持 0.5
        assert matrix["agent_0"]["merge"] == 0.5
        assert matrix["agent_0"]["mutate"] == 0.5

    def test_select_best_internal(self, mock_agents):
        """测试内部方法 _select_best。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.0)

        # 设置擅长度得分
        dispatcher.specialization_scores = {
            "agent_0": {"explore": 0.5, "merge": 0.9, "mutate": 0.6},
            "agent_1": {"explore": 0.7, "merge": 0.6, "mutate": 0.8},
            "agent_2": {"explore": 0.8, "merge": 0.5, "mutate": 0.5},
            "agent_3": {"explore": 0.6, "merge": 0.7, "mutate": 0.9},
        }

        # 验证每种任务类型选择的最优 Agent
        best_explore = dispatcher._select_best("explore")
        assert best_explore.name == "agent_2"  # 0.8

        best_merge = dispatcher._select_best("merge")
        assert best_merge.name == "agent_0"  # 0.9

        best_mutate = dispatcher._select_best("mutate")
        assert best_mutate.name == "agent_3"  # 0.9

    def test_epsilon_greedy_mixed_strategy(self, mock_agents):
        """测试混合策略（epsilon=0.3）。"""
        dispatcher = TaskDispatcher(mock_agents, epsilon=0.3)

        # 设置擅长度得分（agent_0 最优）
        dispatcher.specialization_scores = {
            "agent_0": {"explore": 0.9, "merge": 0.5, "mutate": 0.5},
            "agent_1": {"explore": 0.3, "merge": 0.5, "mutate": 0.5},
            "agent_2": {"explore": 0.3, "merge": 0.5, "mutate": 0.5},
            "agent_3": {"explore": 0.3, "merge": 0.5, "mutate": 0.5},
        }

        # 多次选择
        selections = [dispatcher.select_agent("explore").name for _ in range(100)]

        # 验证 agent_0 被选择次数占多数（约 70%+）
        agent_0_count = selections.count("agent_0")
        assert agent_0_count > 50  # 至少 50%（理论上应该 70%+）

        # 验证其他 Agent 也被选择过（探索）
        unique_agents = set(selections)
        assert len(unique_agents) > 1  # 不只选择一个 Agent
