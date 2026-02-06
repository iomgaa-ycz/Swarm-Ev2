"""CoderAgent 单元测试模块。

测试 CoderAgent 的核心功能，包括：
- 初始化
- generate() 方法
- _explore() 方法
- 重试机制
- 响应解析
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agents.coder_agent import CoderAgent
from agents.base_agent import AgentContext
from core.state import Node, Journal


@pytest.fixture
def mock_config():
    """创建 Mock 配置对象。"""
    config = MagicMock()

    # LLM 配置
    config.llm.code.model = "gpt-4-turbo"
    config.llm.code.provider = "openai"
    config.llm.code.temperature = 0.7
    config.llm.code.max_tokens = 4096
    config.llm.code.api_key = "sk-test-key"
    config.llm.code.base_url = None

    # Agent 配置
    config.agent.data_preview = False
    config.agent.time_limit = 3600
    config.agent.max_steps = 50

    # Task 配置
    config.task.description = "测试任务描述"

    return config


@pytest.fixture
def mock_prompt_manager():
    """创建 Mock PromptManager。"""
    pm = MagicMock()
    pm.build_prompt.return_value = "Test Prompt"
    return pm


@pytest.fixture
def agent_context(mock_config):
    """创建 AgentContext 对象。"""
    journal = Journal()
    return AgentContext(
        task_type="explore",
        parent_node=None,
        journal=journal,
        config=mock_config,
        start_time=time.time(),
        current_step=0,
        task_desc="测试任务描述",
    )


class TestCoderAgentInit:
    """测试 CoderAgent 初始化。"""

    def test_coder_agent_init(self, mock_config, mock_prompt_manager):
        """测试初始化参数。"""
        agent = CoderAgent(
            name="TestCoder",
            config=mock_config,
            prompt_manager=mock_prompt_manager,
        )

        assert agent.name == "TestCoder"
        assert agent.config == mock_config
        assert agent.prompt_manager == mock_prompt_manager


class TestCoderAgentGenerate:
    """测试 CoderAgent.generate() 方法。"""

    @patch("agents.coder_agent.query_with_config")
    def test_coder_agent_generate_explore_success(
        self,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试 generate() 方法（task_type="explore"）成功。"""
        mock_query.return_value = (
            "Plan: use RandomForest\n```python\nprint('test')\n```"
        )

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        result = agent.generate(agent_context)

        assert result.success is True
        assert result.node is not None
        assert result.node.code == "print('test')"
        assert result.node.plan == "Plan: use RandomForest"
        assert result.error is None

    def test_coder_agent_generate_merge_requires_fields(
        self, mock_config, mock_prompt_manager, agent_context
    ):
        """测试 generate() 方法（task_type="merge"）缺少必需字段时失败。"""
        agent_context.task_type = "merge"

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        result = agent.generate(agent_context)

        assert result.success is False
        assert result.node is None
        assert "parent_a" in result.error or "gene_plan" in result.error


class TestCoderAgentExplore:
    """测试 CoderAgent._explore() 方法。"""

    @patch("agents.coder_agent.query_with_config")
    def test_coder_agent_explore_draft_mode(
        self,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试初稿模式（parent_node=None）。"""
        mock_query.return_value = "Plan: initial draft\n```python\nprint('draft')\n```"

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        node = agent._explore(agent_context)

        assert node.parent_id is None
        assert node.code == "print('draft')"
        assert node.plan == "Plan: initial draft"

    @patch("agents.coder_agent.query_with_config")
    def test_coder_agent_explore_bugfix_mode(
        self,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试修复模式（parent_node.is_buggy=True）。"""
        parent_node = Node(
            code="print('buggy')",
            plan="Buggy plan",
            is_buggy=True,
            exc_type="ValueError",
        )
        agent_context.parent_node = parent_node

        mock_query.return_value = "Plan: fix bug\n```python\nprint('fixed')\n```"

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        node = agent._explore(agent_context)

        assert node.parent_id == parent_node.id
        assert node.code == "print('fixed')"

        # 验证 build_prompt 被调用，且 context dict 包含 parent_node
        mock_prompt_manager.build_prompt.assert_called_once()
        call_args = mock_prompt_manager.build_prompt.call_args[0]
        assert call_args[0] == "explore"  # task_type
        context_dict = call_args[2]
        assert context_dict["parent_node"] == parent_node

    @patch("agents.coder_agent.query_with_config")
    def test_coder_agent_explore_improve_mode(
        self,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试改进模式（parent_node.is_buggy=False）。"""
        parent_node = Node(
            code="print('normal')",
            plan="Normal plan",
            is_buggy=False,
            metric_value=0.85,
        )
        agent_context.parent_node = parent_node

        mock_query.return_value = "Plan: improve\n```python\nprint('improved')\n```"

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        node = agent._explore(agent_context)

        assert node.parent_id == parent_node.id
        assert node.code == "print('improved')"

        # 验证 context dict 包含 parent_node
        mock_prompt_manager.build_prompt.assert_called_once()
        context_dict = mock_prompt_manager.build_prompt.call_args[0][2]
        assert context_dict["parent_node"] == parent_node


class TestCoderAgentLLMRetry:
    """测试 CoderAgent LLM 重试机制。"""

    @patch("agents.coder_agent.query_with_config")
    @patch("time.sleep")
    def test_coder_agent_llm_retry_on_api_error(
        self,
        mock_sleep,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试 LLM API 重试机制。"""
        mock_query.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Exception("API Error 3"),
            Exception("API Error 4"),
            "Plan: success\n```python\nprint('success')\n```",
        ]

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        result = agent.generate(agent_context)

        assert result.success is True
        assert result.node.code == "print('success')"
        assert mock_query.call_count == 5
        assert mock_sleep.call_count == 4
        sleep_calls = [call_obj[0][0] for call_obj in mock_sleep.call_args_list]
        assert sleep_calls == [10, 20, 40, 80]

    @patch("agents.coder_agent.query_with_config")
    @patch("time.sleep")
    def test_coder_agent_llm_retry_max_exceeded(
        self,
        mock_sleep,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试 API 重试次数耗尽。"""
        mock_query.side_effect = Exception("Persistent API Error")

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        result = agent.generate(agent_context)

        assert result.success is False
        assert result.node is None
        assert "Persistent API Error" in result.error
        assert mock_query.call_count == 5


class TestCoderAgentResponseParsing:
    """测试 CoderAgent 响应解析。"""

    @patch("agents.coder_agent.query_with_config")
    def test_coder_agent_response_parse_failure(
        self,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试响应解析失败（硬格式失败，无代码块）。"""
        mock_query.return_value = "This is just plain text without code block."

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        result = agent.generate(agent_context)

        assert result.success is False
        assert result.node is None
        assert "代码块" in result.error or "未找到" in result.error


class TestCoderAgentMemory:
    """测试 CoderAgent Memory 机制。"""

    @patch("agents.coder_agent.query_with_config")
    def test_coder_agent_with_memory(
        self,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试 Memory 机制。"""
        node1 = Node(
            code="print('v1')",
            plan="Version 1",
            analysis="Good solution",
            metric_value=0.85,
            is_buggy=False,
        )
        node2 = Node(
            code="print('v2')",
            plan="Version 2",
            is_buggy=True,
        )
        agent_context.journal.append(node1)
        agent_context.journal.append(node2)

        mock_query.return_value = "Plan: v3\n```python\nprint('v3')\n```"

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        agent._explore(agent_context)

        # 验证 build_prompt 被调用，且 context dict 包含 memory
        mock_prompt_manager.build_prompt.assert_called_once()
        context_dict = mock_prompt_manager.build_prompt.call_args[0][2]
        memory = context_dict["memory"]
        assert isinstance(memory, str)


class TestCoderAgentTimeStepsCalculation:
    """测试 CoderAgent 时间和步数计算。"""

    @patch("agents.coder_agent.query_with_config")
    @patch("time.time")
    def test_coder_agent_time_steps_calculation(
        self,
        mock_time,
        mock_query,
        mock_config,
        mock_prompt_manager,
        agent_context,
    ):
        """测试剩余时间和步数计算。"""
        start_time = 1000.0
        current_time = 1100.0  # 已过去 100 秒
        mock_time.return_value = current_time
        agent_context.start_time = start_time
        agent_context.current_step = 5
        mock_config.agent.time_limit = 3600
        mock_config.agent.max_steps = 50

        mock_query.return_value = "Plan: test\n```python\nprint('test')\n```"

        agent = CoderAgent("TestCoder", mock_config, mock_prompt_manager)
        agent._explore(agent_context)

        # 验证 context dict 包含正确的时间和步数
        mock_prompt_manager.build_prompt.assert_called_once()
        context_dict = mock_prompt_manager.build_prompt.call_args[0][2]

        # 剩余时间 = 3600 - 100 = 3500 秒
        assert context_dict["time_remaining"] == 3500
        # 剩余步数 = 50 - 5 = 45
        assert context_dict["steps_remaining"] == 45
