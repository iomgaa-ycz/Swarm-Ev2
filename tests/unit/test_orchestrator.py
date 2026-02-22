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
                "parallel_num": 1,  # 并行执行数
            },
            "execution": {
                "timeout": 60,
                "adaptive_timeout": False,
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
        agents=[mock_agent],  # 使用 agents 列表
        config=mock_config,
        journal=journal,
        task_desc="Test task description",
    )


class TestOrchestrator:
    """Orchestrator 测试类。"""

    def test_init(self, orchestrator, mock_agent, mock_config, journal):
        """测试初始化。"""
        assert orchestrator.agents == [mock_agent]
        assert orchestrator.config == mock_config
        assert orchestrator.journal == journal
        assert orchestrator.task_desc == "Test task description"
        assert orchestrator.current_epoch == 0
        assert orchestrator.best_node is None

    def test_review_node_success(self, orchestrator):
        """测试 Review 评估成功。"""
        node = Node(code="print('test')", plan="Test plan")

        # Mock backend.query 返回 Function Calling 响应（使用新的 schema）
        # 同时 mock _check_submission_exists 返回 True
        with patch("core.orchestrator.backend_query") as mock_query:
            with patch.object(
                orchestrator, "_check_submission_exists", return_value=True
            ):
                with patch.object(
                    orchestrator,
                    "_validate_submission_format",
                    return_value={"valid": True, "errors": [], "row_count": 100},
                ):
                    mock_query.return_value = '{"is_bug": false, "metric": 0.90, "key_change": "Added print statement", "insight": "Good result", "lower_is_better": false, "has_csv_submission": true}'

                    orchestrator._review_node(node)

        # 验证
        assert node.is_buggy is False
        assert node.metric_value == 0.90
        assert node.analysis == "Added print statement"  # 现在存储 key_change
        assert node.analysis_detail is not None
        assert node.analysis_detail["key_change"] == "Added print statement"
        assert node.lower_is_better is False

    def test_review_node_parsing_failure(self, orchestrator):
        """测试 Review 解析失败回退。"""
        node = Node(code="raise ValueError()", plan="Buggy plan", exc_type="ValueError")

        # Mock backend.query 抛出异常
        with patch("core.orchestrator.backend_query") as mock_query:
            mock_query.side_effect = Exception("API error")

            orchestrator._review_node(node)

        # 验证：应该标记为 buggy（因为有异常类型）
        assert node.is_buggy is True
        assert node.metric_value is None

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

    def test_get_review_tool_schema(self, orchestrator):
        """测试 Review tool schema 生成（增强版）。"""
        schema = orchestrator._get_review_tool_schema()

        assert schema["name"] == "submit_review"
        assert "parameters" in schema
        assert "properties" in schema["parameters"]
        # 验证基础字段
        assert "is_bug" in schema["parameters"]["properties"]
        assert "metric" in schema["parameters"]["properties"]
        assert "lower_is_better" in schema["parameters"]["properties"]
        # 验证新增字段
        assert "key_change" in schema["parameters"]["properties"]
        assert "metric_delta" not in schema["parameters"]["properties"]
        assert "insight" in schema["parameters"]["properties"]
        assert "bottleneck" in schema["parameters"]["properties"]
        assert "suggested_direction" in schema["parameters"]["properties"]

    def test_build_review_messages(self, orchestrator):
        """测试 Review messages 构建（包含变更上下文）。"""
        node = Node(
            code="print('test')",
            plan="Test plan",
            term_out="test\n",
            exec_time=0.5,
            exc_type=None,
        )

        # 传入变更上下文参数
        change_context = "(Initial solution, no diff)"
        messages = orchestrator._build_review_messages(node, change_context)

        assert "Test task description" in messages
        assert "print('test')" in messages
        assert "test\n" in messages
        assert "0.50s" in messages
        assert "Code Changes" in messages  # 新增的变更上下文部分
        assert "Initial solution" in messages


class TestDebugChain:
    """_debug_chain 链式 Debug 逻辑测试。"""

    def test_skips_debug_when_no_error(self, orchestrator, mock_agent):
        """exc_type 为 None 时直接跳过 debug，不调用 agent._debug。"""
        node = Node(code="print('ok')", plan="plan", exc_type=None)

        result = orchestrator._debug_chain(node, mock_agent, context=None, max_attempts=2)

        # 没有异常，立即返回原节点
        assert result is node
        mock_agent._debug.assert_not_called()

    def test_marks_dead_after_max_attempts(self, orchestrator, mock_agent):
        """agent._debug 始终返回仍有异常的节点，耗尽后标记 node.dead=True。"""
        from unittest.mock import MagicMock

        node = Node(code="raise ValueError()", plan="plan", exc_type="ValueError")

        # Mock context（_debug_chain 内部会访问 context 的属性）
        mock_context = MagicMock()
        mock_context.task_desc = "test task"
        mock_context.device_info = "CPU"
        mock_context.conda_packages = "pandas"
        mock_context.conda_env_name = "Swarm-Evo"

        # _debug 每次都返回一个仍然有错的新节点（不同 code，仍有 exc_type）
        def make_buggy_node(*args, **kwargs):
            n = Node(code="raise RuntimeError()", plan="plan", exc_type="RuntimeError")
            return n

        mock_agent._debug.side_effect = make_buggy_node

        # _execute_code 也需要 mock（返回仍有 exc_type 的结果）
        exec_result = MagicMock()
        exec_result.term_out = ["error output"]
        exec_result.exec_time = 0.1
        exec_result.exc_type = "RuntimeError"
        exec_result.exc_info = "RuntimeError: ..."

        with patch.object(orchestrator, "_execute_code", return_value=exec_result):
            result = orchestrator._debug_chain(
                node, mock_agent, context=mock_context, max_attempts=2
            )

        # 耗尽 2 次后，原节点应被标记为 dead
        assert result.dead is True
        assert result.debug_attempts == 2
        assert mock_agent._debug.call_count == 2
