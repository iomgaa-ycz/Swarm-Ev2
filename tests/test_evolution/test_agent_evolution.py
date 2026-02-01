"""AgentEvolution 单元测试。

测试 Agent 层进化器的评估、识别和变异逻辑。
"""

import pytest
import time
from unittest.mock import MagicMock, patch

from core.evolution import AgentEvolution, ExperiencePool, TaskRecord
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


@pytest.fixture
def mock_config(tmp_path):
    """创建 Mock 配置。"""
    config = MagicMock()
    config.evolution.agent.configs_dir = str(tmp_path / "agent_configs")
    config.evolution.agent.evolution_interval = 3
    config.evolution.agent.min_records_for_evolution = 20
    config.evolution.experience_pool.max_records = 10000
    config.evolution.experience_pool.save_path = str(tmp_path / "pool.json")
    config.llm.code = MagicMock()
    return config


@pytest.fixture
def experience_pool_with_data(mock_config):
    """创建包含测试数据的经验池。"""
    pool = ExperiencePool(mock_config)

    # 添加测试数据（agent_0, agent_1 高分，agent_2, agent_3 低分）
    records = [
        # agent_0: 高质量
        ("agent_0", "explore", 0.9),
        ("agent_0", "merge", 0.8),
        ("agent_0", "mutate", 0.85),
        # agent_1: 高质量
        ("agent_1", "explore", 0.85),
        ("agent_1", "merge", 0.9),
        ("agent_1", "mutate", 0.8),
        # agent_2: 低质量
        ("agent_2", "explore", 0.4),
        ("agent_2", "merge", 0.3),
        ("agent_2", "mutate", 0.35),
        # agent_3: 低质量
        ("agent_3", "explore", 0.3),
        ("agent_3", "merge", 0.25),
        ("agent_3", "mutate", 0.2),
    ]

    for agent_id, task_type, quality in records:
        pool.add(
            TaskRecord(
                agent_id=agent_id,
                task_type=task_type,
                input_hash=f"hash_{agent_id}_{task_type}",
                output_quality=quality,
                strategy_summary=f"{agent_id} {task_type} strategy",
                timestamp=time.time(),
            )
        )

    # 添加更多记录以满足 min_records 要求
    for i in range(20):
        pool.add(
            TaskRecord(
                agent_id="agent_0",
                task_type="explore",
                input_hash=f"extra_hash_{i}",
                output_quality=0.7,
                strategy_summary=f"Extra strategy {i}",
                timestamp=time.time(),
            )
        )

    return pool


class TestAgentEvolution:
    """AgentEvolution 测试套件。"""

    def test_initialization(self, mock_agents, experience_pool_with_data, mock_config):
        """测试初始化。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        assert evolution.agents == mock_agents
        assert evolution.experience_pool == experience_pool_with_data
        assert evolution.evolution_interval == 3
        assert evolution.min_records == 20

    def test_evolve_skips_non_interval_epochs(
        self, mock_agents, experience_pool_with_data, mock_config
    ):
        """测试 epoch % 3 != 0 时跳过进化。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        with patch.object(evolution, "_evaluate_agents") as mock_evaluate:
            # epoch=1, 2 应跳过
            evolution.evolve(epoch=1)
            evolution.evolve(epoch=2)

            # 验证未调用评估
            assert mock_evaluate.call_count == 0

            # epoch=3 应执行
            evolution.evolve(epoch=3)
            assert mock_evaluate.call_count == 1

    def test_evolve_skips_when_epoch_is_zero(
        self, mock_agents, experience_pool_with_data, mock_config
    ):
        """测试 epoch=0 时跳过进化。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        with patch.object(evolution, "_evaluate_agents") as mock_evaluate:
            evolution.evolve(epoch=0)
            assert mock_evaluate.call_count == 0

    def test_evolve_skips_insufficient_records(self, mock_agents, mock_config):
        """测试记录不足时跳过进化。"""
        # 创建空经验池
        pool = ExperiencePool(mock_config)
        evolution = AgentEvolution(mock_agents, pool, mock_config)

        with patch.object(evolution, "_evaluate_agents") as mock_evaluate:
            evolution.evolve(epoch=3)

            # 验证未调用评估
            assert mock_evaluate.call_count == 0

    def test_evaluate_agents(self, mock_agents, experience_pool_with_data, mock_config):
        """测试 Agent 评估。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        scores = evolution._evaluate_agents()

        # 验证所有 Agent 都有得分
        assert len(scores) == 4

        # 验证得分公式: success_rate × avg_quality
        # agent_0: 高分
        # agent_3: 低分
        assert scores["agent_0"] > scores["agent_3"]

        # 验证得分范围
        assert all(0 <= score <= 1 for score in scores.values())

    def test_identify_elites_and_weak(
        self, mock_agents, experience_pool_with_data, mock_config
    ):
        """测试识别精英和弱者。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        scores = {
            "agent_0": 0.85,
            "agent_1": 0.80,
            "agent_2": 0.40,
            "agent_3": 0.30,
        }

        elites, weak = evolution._identify_elites_and_weak(scores)

        # 验证精英（top-2）
        assert len(elites) == 2
        assert "agent_0" in elites
        assert "agent_1" in elites

        # 验证弱者（bottom-2）
        assert len(weak) == 2
        assert "agent_2" in weak
        assert "agent_3" in weak

    def test_get_performance_summary(
        self, mock_agents, experience_pool_with_data, mock_config
    ):
        """测试获取历史表现摘要。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        summary = evolution._get_performance_summary("agent_0")

        # 验证包含所有必要字段
        assert "success_rate" in summary
        assert "avg_quality" in summary
        assert "top_successes" in summary
        assert "top_failures" in summary

        # agent_0 高质量，应有成功案例
        assert len(summary["top_successes"]) > 0

    def test_get_performance_summary_with_task_type(
        self, mock_agents, experience_pool_with_data, mock_config
    ):
        """测试按任务类型获取历史表现。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        summary = evolution._get_performance_summary("agent_0", task_type="explore")

        # 验证返回的案例仅包含 explore 任务
        # （实际验证需要检查 strategy_summary 内容）
        assert "top_successes" in summary
        assert "top_failures" in summary

    @patch("core.evolution.agent_evolution.query")
    def test_mutate_role(
        self, mock_query, mock_agents, experience_pool_with_data, mock_config, tmp_path
    ):
        """测试 Role 变异。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        # 创建测试配置文件
        agent_2_dir = tmp_path / "agent_configs" / "agent_2"
        agent_2_dir.mkdir(parents=True, exist_ok=True)
        (agent_2_dir / "role.md").write_text("# 当前 Role\n弱者角色定义")

        agent_0_dir = tmp_path / "agent_configs" / "agent_0"
        agent_0_dir.mkdir(parents=True, exist_ok=True)
        (agent_0_dir / "role.md").write_text("# 精英 Role\n精英角色定义")

        # Mock LLM 返回
        mock_query.return_value = "# 改进后的 Role\n新角色定义"

        # 执行变异
        evolution._mutate_role("agent_2", elite_id="agent_0")

        # 验证 LLM 被调用
        assert mock_query.call_count == 1

        # 验证新 Role 写入文件
        new_role = (agent_2_dir / "role.md").read_text()
        assert "改进后的 Role" in new_role

    @patch("core.evolution.agent_evolution.query")
    def test_mutate_strategy(
        self, mock_query, mock_agents, experience_pool_with_data, mock_config, tmp_path
    ):
        """测试 Strategy 变异。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        # 创建测试配置文件
        agent_2_dir = tmp_path / "agent_configs" / "agent_2"
        agent_2_dir.mkdir(parents=True, exist_ok=True)
        (agent_2_dir / "strategy_explore.md").write_text("# 当前 Strategy\n弱者策略")

        agent_0_dir = tmp_path / "agent_configs" / "agent_0"
        agent_0_dir.mkdir(parents=True, exist_ok=True)
        (agent_0_dir / "strategy_explore.md").write_text("# 精英 Strategy\n精英策略")

        # Mock LLM 返回
        mock_query.return_value = "# 改进后的 Strategy\n新策略定义"

        # 执行变异
        evolution._mutate_strategy("agent_2", "explore", elite_id="agent_0")

        # 验证 LLM 被调用
        assert mock_query.call_count == 1

        # 验证新 Strategy 写入文件
        new_strategy = (agent_2_dir / "strategy_explore.md").read_text()
        assert "改进后的 Strategy" in new_strategy

    def test_build_mutation_prompt(
        self, mock_agents, experience_pool_with_data, mock_config
    ):
        """测试构建变异 Prompt。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        stats = {
            "success_rate": 0.45,
            "avg_quality": 0.62,
            "top_successes": ["Success 1", "Success 2"],
            "top_failures": ["Failure 1", "Failure 2"],
        }

        prompt = evolution._build_mutation_prompt(
            current_content="当前内容",
            elite_content="精英内容",
            stats=stats,
            section="role",
        )

        # 验证 Prompt 包含关键信息
        assert "当前内容" in prompt
        assert "精英内容" in prompt
        assert "45.00%" in prompt  # success_rate
        assert "0.620" in prompt  # avg_quality
        assert "Success 1" in prompt
        assert "Failure 1" in prompt

    @patch("core.evolution.agent_evolution.query")
    def test_full_evolve_workflow(
        self, mock_query, mock_agents, experience_pool_with_data, mock_config, tmp_path
    ):
        """测试完整进化流程。"""
        evolution = AgentEvolution(mock_agents, experience_pool_with_data, mock_config)

        # 创建所有 Agent 的配置文件
        for agent_id in ["agent_0", "agent_1", "agent_2", "agent_3"]:
            agent_dir = tmp_path / "agent_configs" / agent_id
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "role.md").write_text(f"# {agent_id} Role")
            for task_type in ["explore", "merge", "mutate"]:
                (agent_dir / f"strategy_{task_type}.md").write_text(
                    f"# {agent_id} {task_type} Strategy"
                )

        # Mock LLM 返回
        mock_query.return_value = "# 改进后的配置"

        # 执行进化
        evolution.evolve(epoch=3)

        # 验证 LLM 被调用（2 个弱者 × (1 Role + 3 Strategy) = 8 次）
        assert mock_query.call_count == 8
