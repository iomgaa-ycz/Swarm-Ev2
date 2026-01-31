"""CoderAgent 单元测试模块。

测试 CoderAgent 的核心功能，包括：
- 初始化
- generate() 方法
- _explore() 方法
- 重试机制
- 响应解析
- 代码执行处理
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from agents.coder_agent import CoderAgent
from agents.base_agent import AgentContext
from core.state import Node, Journal
from core.executor.interpreter import ExecutionResult


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
def mock_prompt_builder():
    """创建 Mock PromptBuilder。"""
    builder = MagicMock()
    builder.build_explore_prompt.return_value = "Test Prompt"
    return builder


@pytest.fixture
def mock_interpreter():
    """创建 Mock Interpreter。"""
    interpreter = MagicMock()
    interpreter.run.return_value = ExecutionResult(
        term_out=["Output line 1", "Output line 2"],
        exec_time=1.5,
        exc_type=None,
        exc_info=None,
        success=True,
        timeout=False,
    )
    return interpreter


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
    )


class TestCoderAgentInit:
    """测试 CoderAgent 初始化。"""

    def test_coder_agent_init(self, mock_config, mock_prompt_builder, mock_interpreter):
        """测试初始化参数。"""
        agent = CoderAgent(
            name="TestCoder",
            config=mock_config,
            prompt_builder=mock_prompt_builder,
            interpreter=mock_interpreter,
        )

        assert agent.name == "TestCoder"
        assert agent.config == mock_config
        assert agent.prompt_builder == mock_prompt_builder
        assert agent.interpreter == mock_interpreter


class TestCoderAgentGenerate:
    """测试 CoderAgent.generate() 方法。"""

    @patch("agents.coder_agent.backend_query")
    def test_coder_agent_generate_explore_success(
        self,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试 generate() 方法（task_type="explore"）成功。"""
        # Mock LLM 返回
        mock_query.return_value = (
            "Plan: use RandomForest\n```python\nprint('test')\n```"
        )

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 generate
        result = agent.generate(agent_context)

        # 验证结果
        assert result.success is True
        assert result.node is not None
        assert result.node.code == "print('test')"
        assert result.node.plan == "Plan: use RandomForest"
        assert result.node.is_buggy is False
        assert result.error is None

    def test_coder_agent_generate_merge_not_implemented(
        self, mock_config, mock_prompt_builder, mock_interpreter, agent_context
    ):
        """测试 generate() 方法（task_type="merge"）抛出 NotImplementedError。"""
        # 修改 task_type
        agent_context.task_type = "merge"

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 generate
        result = agent.generate(agent_context)

        # 验证结果
        assert result.success is False
        assert result.node is None
        assert "不支持 merge" in result.error or "NotImplementedError" in result.error


class TestCoderAgentExplore:
    """测试 CoderAgent._explore() 方法。"""

    @patch("agents.coder_agent.backend_query")
    def test_coder_agent_explore_draft_mode(
        self,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试初稿模式（parent_node=None）。"""
        # Mock LLM 返回
        mock_query.return_value = "Plan: initial draft\n```python\nprint('draft')\n```"

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 _explore
        node = agent._explore(agent_context)

        # 验证结果
        assert node.parent_id is None
        assert node.code == "print('draft')"
        assert node.plan == "Plan: initial draft"

    @patch("agents.coder_agent.backend_query")
    def test_coder_agent_explore_bugfix_mode(
        self,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试修复模式（parent_node.is_buggy=True）。"""
        # 创建 buggy 父节点
        parent_node = Node(
            code="print('buggy')",
            plan="Buggy plan",
            is_buggy=True,
            exc_type="ValueError",
        )
        agent_context.parent_node = parent_node

        # Mock LLM 返回
        mock_query.return_value = "Plan: fix bug\n```python\nprint('fixed')\n```"

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 _explore
        node = agent._explore(agent_context)

        # 验证结果
        assert node.parent_id == parent_node.id
        assert node.code == "print('fixed')"
        assert node.plan == "Plan: fix bug"

        # 验证 Prompt 包含父节点信息
        mock_prompt_builder.build_explore_prompt.assert_called_once()
        call_kwargs = mock_prompt_builder.build_explore_prompt.call_args.kwargs
        assert call_kwargs["parent_node"] == parent_node

    @patch("agents.coder_agent.backend_query")
    def test_coder_agent_explore_improve_mode(
        self,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试改进模式（parent_node.is_buggy=False）。"""
        # 创建正常父节点
        parent_node = Node(
            code="print('normal')",
            plan="Normal plan",
            is_buggy=False,
            metric_value=0.85,
        )
        agent_context.parent_node = parent_node

        # Mock LLM 返回
        mock_query.return_value = "Plan: improve\n```python\nprint('improved')\n```"

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 _explore
        node = agent._explore(agent_context)

        # 验证结果
        assert node.parent_id == parent_node.id
        assert node.code == "print('improved')"
        assert node.plan == "Plan: improve"

        # 验证 Prompt 包含父节点信息
        mock_prompt_builder.build_explore_prompt.assert_called_once()
        call_kwargs = mock_prompt_builder.build_explore_prompt.call_args.kwargs
        assert call_kwargs["parent_node"] == parent_node


class TestCoderAgentLLMRetry:
    """测试 CoderAgent LLM 重试机制。"""

    @patch("agents.coder_agent.backend_query")
    @patch("time.sleep")  # Mock sleep 加速测试
    def test_coder_agent_llm_retry_on_api_error(
        self,
        mock_sleep,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试 LLM API 重试机制。"""
        # Mock backend.query() 前 4 次抛出异常，第 5 次成功
        mock_query.side_effect = [
            Exception("API Error 1"),
            Exception("API Error 2"),
            Exception("API Error 3"),
            Exception("API Error 4"),
            "Plan: success\n```python\nprint('success')\n```",
        ]

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 generate
        result = agent.generate(agent_context)

        # 验证结果
        assert result.success is True
        assert result.node.code == "print('success')"

        # 验证调用次数
        assert mock_query.call_count == 5

        # 验证重试间隔
        assert mock_sleep.call_count == 4
        sleep_calls = [call_obj[0][0] for call_obj in mock_sleep.call_args_list]
        assert sleep_calls == [10, 20, 40, 80]

    @patch("agents.coder_agent.backend_query")
    @patch("time.sleep")
    def test_coder_agent_llm_retry_max_exceeded(
        self,
        mock_sleep,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试 API 重试次数耗尽。"""
        # Mock backend.query() 始终抛出异常
        mock_query.side_effect = Exception("Persistent API Error")

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 generate
        result = agent.generate(agent_context)

        # 验证结果
        assert result.success is False
        assert result.node is None
        assert "Persistent API Error" in result.error

        # 验证调用次数（5 次）
        assert mock_query.call_count == 5


class TestCoderAgentResponseParsing:
    """测试 CoderAgent 响应解析。"""

    @patch("agents.coder_agent.backend_query")
    def test_coder_agent_code_execution_failure(
        self,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试代码执行失败。"""
        # Mock LLM 返回
        mock_query.return_value = "Plan: test\n```python\nraise ValueError('test')\n```"

        # Mock Interpreter 返回执行失败
        mock_interpreter.run.return_value = ExecutionResult(
            term_out=["", "Traceback...\nValueError: test"],
            exec_time=0.1,
            exc_type="ValueError",
            exc_info="test",
            success=False,
            timeout=False,
        )

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 generate
        result = agent.generate(agent_context)

        # 验证结果
        assert result.success is True  # generate 成功（创建了 node）
        assert result.node.is_buggy is True
        assert result.node.exc_type == "ValueError"
        assert result.node.exc_info == "test"
        assert "Traceback" in result.node.term_out

    @patch("agents.coder_agent.backend_query")
    def test_coder_agent_response_parse_failure(
        self,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试响应解析失败（硬格式失败，无代码块）。"""
        # Mock LLM 返回无效格式（无代码块）
        mock_query.return_value = "This is just plain text without code block."

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 generate
        result = agent.generate(agent_context)

        # 验证结果
        assert result.success is False
        assert result.node is None
        assert "代码块" in result.error or "未找到" in result.error


class TestCoderAgentMemory:
    """测试 CoderAgent Memory 机制。"""

    @patch("agents.coder_agent.backend_query")
    def test_coder_agent_with_memory(
        self,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试 Memory 机制。"""
        # 向 Journal 添加节点
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

        # Mock LLM 返回
        mock_query.return_value = "Plan: v3\n```python\nprint('v3')\n```"

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 _explore
        agent._explore(agent_context)

        # 验证 Prompt 包含 Memory
        mock_prompt_builder.build_explore_prompt.assert_called_once()
        call_kwargs = mock_prompt_builder.build_explore_prompt.call_args.kwargs
        memory = call_kwargs["memory"]

        # Memory 应该包含历史节点信息
        assert isinstance(memory, str)


class TestCoderAgentTimeStepsCalculation:
    """测试 CoderAgent 时间和步数计算。"""

    @patch("agents.coder_agent.backend_query")
    @patch("time.time")
    def test_coder_agent_time_steps_calculation(
        self,
        mock_time,
        mock_query,
        mock_config,
        mock_prompt_builder,
        mock_interpreter,
        agent_context,
    ):
        """测试剩余时间和步数计算。"""
        # Mock 时间
        start_time = 1000.0
        current_time = 1100.0  # 已过去 100 秒
        mock_time.return_value = current_time
        agent_context.start_time = start_time

        # 设置配置
        agent_context.current_step = 5
        mock_config.agent.time_limit = 3600
        mock_config.agent.max_steps = 50

        # Mock LLM 返回
        mock_query.return_value = "Plan: test\n```python\nprint('test')\n```"

        # 创建 Agent
        agent = CoderAgent(
            "TestCoder", mock_config, mock_prompt_builder, mock_interpreter
        )

        # 执行 _explore
        agent._explore(agent_context)

        # 验证 Prompt 包含正确的时间和步数
        mock_prompt_builder.build_explore_prompt.assert_called_once()
        call_kwargs = mock_prompt_builder.build_explore_prompt.call_args.kwargs

        # 剩余时间 = 3600 - 100 = 3500 秒
        assert call_kwargs["time_remaining"] == 3500

        # 剩余步数 = 50 - 5 = 45
        assert call_kwargs["steps_remaining"] == 45
