"""Orchestrator 单元测试模块。"""

import pytest
from unittest.mock import Mock, patch
from omegaconf import OmegaConf

from core.orchestrator import Orchestrator
from core.state import Node, Journal
from agents.base_agent import AgentResult


@pytest.fixture
def mock_config(tmp_path):
    """Mock 配置对象。"""
    cfg = OmegaConf.create(
        {
            "project": {
                "workspace_dir": tmp_path,
            },
            "agent": {
                "max_steps": 10,
                "time_limit": 3600,
            },
            "search": {
                "num_drafts": 3,
                "debug_prob": 0.5,
            },
            "execution": {
                "timeout": 60,
            },
            "llm": {
                "feedback": {
                    "model": "glm-4.6",
                    "provider": "openai",
                    "temperature": 0.5,
                    "api_key": "test-key",
                    "base_url": "https://test.com/v1",
                }
            },
        }
    )
    return cfg


@pytest.fixture
def mock_agent():
    """Mock Agent 对象。"""
    agent = Mock()
    agent.name = "test_agent"
    return agent


@pytest.fixture
def journal():
    """创建 Journal 实例。"""
    return Journal()


@pytest.fixture
def orchestrator(mock_agent, mock_config, journal):
    """创建 Orchestrator 实例。"""
    return Orchestrator(
        agent=mock_agent,
        config=mock_config,
        journal=journal,
        task_desc="Test task description",
    )


class TestOrchestrator:
    """Orchestrator 测试类。"""

    def test_init(self, orchestrator, mock_agent, mock_config, journal):
        """测试初始化。"""
        assert orchestrator.agent == mock_agent
        assert orchestrator.config == mock_config
        assert orchestrator.journal == journal
        assert orchestrator.task_desc == "Test task description"
        assert orchestrator.current_step == 0
        assert orchestrator.best_node is None

    def test_select_parent_node_initial_drafts(self, orchestrator):
        """测试初稿模式选择（draft 数量不足）。"""
        # 当前无 draft
        assert len(orchestrator.journal.draft_nodes) == 0

        # 应该返回 None（初稿模式）
        parent = orchestrator._select_parent_node()
        assert parent is None

    def test_select_parent_node_debug_mode(self, orchestrator, monkeypatch):
        """测试修复模式选择（debug_prob=1.0 强制触发）。"""
        # 添加 3 个 draft（满足 num_drafts）
        for i in range(3):
            node = Node(code=f"code_{i}", plan=f"plan_{i}")
            orchestrator.journal.append(node)

        # 添加 1 个 buggy 节点
        buggy_node = Node(code="buggy_code", plan="buggy_plan", is_buggy=True)
        orchestrator.journal.append(buggy_node)

        # 强制触发修复模式（debug_prob=1.0）
        monkeypatch.setattr("random.random", lambda: 0.0)

        parent = orchestrator._select_parent_node()
        assert parent == buggy_node

    def test_select_parent_node_improve_mode(self, orchestrator, monkeypatch):
        """测试改进模式选择（best_node 存在）。"""
        # 添加 3 个 draft
        nodes = []
        for i in range(3):
            node = Node(
                code=f"code_{i}",
                plan=f"plan_{i}",
                metric_value=0.8 + i * 0.05,
                is_buggy=False,
            )
            orchestrator.journal.append(node)
            nodes.append(node)

        # 不触发修复模式（debug_prob=0.0）
        monkeypatch.setattr("random.random", lambda: 1.0)

        parent = orchestrator._select_parent_node()

        # 应该选择 metric 最高的节点
        best = max(nodes, key=lambda n: n.metric_value)
        assert parent == best

    def test_step_success(self, orchestrator, mock_agent):
        """测试单步执行成功流程。"""
        # Mock Agent.generate() 返回成功结果
        node = Node(
            code="print('Hello')",
            plan="Simple plan",
            term_out="Hello\n",
            exec_time=0.1,
        )
        mock_agent.generate.return_value = AgentResult(node=node, success=True)

        # Mock Review
        with patch.object(orchestrator, "_review_node") as mock_review:
            mock_review.side_effect = (
                lambda n: setattr(n, "metric_value", 0.85)
                or setattr(n, "is_buggy", False)
                or setattr(n, "lower_is_better", False)
            )

            # 执行 step
            orchestrator.step()

        # 验证
        assert len(orchestrator.journal) == 1
        assert orchestrator.journal[0].code == "print('Hello')"
        assert orchestrator.best_node is not None
        assert orchestrator.best_node.metric_value == 0.85

    def test_step_agent_failure(self, orchestrator, mock_agent):
        """测试 Agent 生成失败场景。"""
        # Mock Agent.generate() 返回失败结果
        mock_agent.generate.return_value = AgentResult(
            node=None, success=False, error="LLM timeout"
        )

        # 执行 step
        orchestrator.step()

        # 验证：Journal 不应该有新节点
        assert len(orchestrator.journal) == 0

    def test_review_node_success(self, orchestrator):
        """测试 Review 评估成功。"""
        node = Node(code="print('test')", plan="Test plan")

        # Mock backend.query 返回 Function Calling 响应
        with patch("core.orchestrator.backend_query") as mock_query:
            mock_query.return_value = '{"is_bug": false, "metric": 0.90, "summary": "Good", "lower_is_better": false, "has_csv_submission": true}'

            orchestrator._review_node(node)

        # 验证
        assert node.is_buggy is False
        assert node.metric_value == 0.90
        assert node.analysis == "Good"
        assert node.lower_is_better is False

    def test_review_node_parsing_failure(self, orchestrator):
        """测试 Review 解析失败回退。"""
        node = Node(code="raise ValueError()", plan="Buggy plan", exc_type="ValueError")

        # Mock backend.query 抛出异常
        with patch("core.orchestrator.backend_query") as mock_query:
            mock_query.side_effect = Exception("API error")

            orchestrator._review_node(node)

        # 验证：应该标记为 buggy
        assert node.is_buggy is True
        assert node.metric_value is None
        assert "Review 失败" in node.analysis

    def test_update_best_node_maximize(self, orchestrator):
        """测试 best_node 更新逻辑（越大越好）。"""
        # 第一个节点
        node1 = Node(
            code="code1",
            plan="plan1",
            metric_value=0.85,
            is_buggy=False,
            lower_is_better=False,
        )
        orchestrator._update_best_node(node1)
        assert orchestrator.best_node == node1

        # 第二个节点（更好）
        node2 = Node(
            code="code2",
            plan="plan2",
            metric_value=0.90,
            is_buggy=False,
            lower_is_better=False,
        )
        orchestrator._update_best_node(node2)
        assert orchestrator.best_node == node2

        # 第三个节点（更差，不应该更新）
        node3 = Node(
            code="code3",
            plan="plan3",
            metric_value=0.80,
            is_buggy=False,
            lower_is_better=False,
        )
        orchestrator._update_best_node(node3)
        assert orchestrator.best_node == node2  # 仍然是 node2

    def test_update_best_node_minimize(self, orchestrator):
        """测试 best_node 更新逻辑（越小越好）。"""
        # 第一个节点
        node1 = Node(
            code="code1",
            plan="plan1",
            metric_value=0.5,
            is_buggy=False,
            lower_is_better=True,
        )
        orchestrator._update_best_node(node1)
        assert orchestrator.best_node == node1

        # 第二个节点（更好，metric 更小）
        node2 = Node(
            code="code2",
            plan="plan2",
            metric_value=0.3,
            is_buggy=False,
            lower_is_better=True,
        )
        orchestrator._update_best_node(node2)
        assert orchestrator.best_node == node2

        # 第三个节点（更差，不应该更新）
        node3 = Node(
            code="code3",
            plan="plan3",
            metric_value=0.6,
            is_buggy=False,
            lower_is_better=True,
        )
        orchestrator._update_best_node(node3)
        assert orchestrator.best_node == node2  # 仍然是 node2

    def test_run_time_limit(self, orchestrator, mock_agent, monkeypatch):
        """测试时间限制中断。"""
        # Mock Agent.generate() 返回成功结果
        node = Node(code="print('test')", plan="plan", metric_value=0.8)
        mock_agent.generate.return_value = AgentResult(node=node, success=True)

        # Mock Review
        with patch.object(orchestrator, "_review_node"):
            # 设置时间限制为 0（立即超时）
            monkeypatch.setattr(orchestrator.config.agent, "time_limit", 0)

            # 运行
            orchestrator.run(max_steps=10)

        # 验证：应该立即停止
        assert orchestrator.current_step == 0

    def test_get_review_tool_schema(self, orchestrator):
        """测试 Review tool schema 生成。"""
        schema = orchestrator._get_review_tool_schema()

        assert schema["name"] == "submit_review"
        assert "parameters" in schema
        assert "properties" in schema["parameters"]
        assert "is_bug" in schema["parameters"]["properties"]
        assert "metric" in schema["parameters"]["properties"]
        assert "lower_is_better" in schema["parameters"]["properties"]

    def test_build_review_messages(self, orchestrator):
        """测试 Review messages 构建。"""
        node = Node(
            code="print('test')",
            plan="Test plan",
            term_out="test\n",
            exec_time=0.5,
            exc_type=None,
        )

        messages = orchestrator._build_review_messages(node)

        assert "Test task description" in messages
        assert "print('test')" in messages
        assert "test\n" in messages
        assert "0.50s" in messages
        assert "None" in messages
