"""ParallelEvaluator 模块单元测试。"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from search.parallel_evaluator import ParallelEvaluator
from core.state import Node
from core.executor.interpreter import ExecutionResult
from core.evolution.gene_registry import GeneRegistry
from agents.base_agent import BaseAgent, AgentContext, AgentResult
from utils.config import Config


@pytest.fixture
def mock_config():
    """创建 mock 配置对象。"""
    config = Mock(spec=Config)
    config.search = Mock()
    config.search.parallel_num = 3
    config.execution = Mock()
    config.execution.timeout = 300
    return config


@pytest.fixture
def mock_workspace():
    """创建 mock 工作空间管理器。"""
    workspace = Mock()
    workspace.rewrite_submission_path = Mock(
        side_effect=lambda code, node_id: code.replace(
            "submission.csv", f"submission_{node_id}.csv"
        )
    )
    return workspace


@pytest.fixture
def mock_interpreter():
    """创建 mock 执行器。"""
    interpreter = Mock()
    return interpreter


@pytest.fixture
def gene_registry():
    """创建真实的基因注册表。"""
    return GeneRegistry()


@pytest.fixture
def evaluator(mock_config, mock_workspace, mock_interpreter, gene_registry):
    """创建 ParallelEvaluator 实例。"""
    return ParallelEvaluator(
        max_workers=3,
        workspace=mock_workspace,
        interpreter=mock_interpreter,
        gene_registry=gene_registry,
        config=mock_config,
    )


class TestParallelEvaluatorInit:
    """测试初始化。"""

    def test_init(self, evaluator):
        """测试基本初始化。"""
        assert evaluator.max_workers == 3
        assert evaluator.workspace is not None
        assert evaluator.interpreter is not None
        assert evaluator.gene_registry is not None
        assert evaluator.executor is not None


class TestBatchGenerate:
    """测试并行生成。"""

    def test_batch_generate_success(self, evaluator):
        """测试成功生成多个 Solution。"""
        # 创建 mock Agent
        agent = Mock(spec=BaseAgent)
        node1 = Node(id="node1", code="code1", step=0)
        node2 = Node(id="node2", code="code2", step=0)
        agent.generate.side_effect = [
            AgentResult(node=node1, success=True),
            AgentResult(node=node2, success=True),
        ]

        # 创建 mock Context
        context1 = Mock(spec=AgentContext)
        context2 = Mock(spec=AgentContext)

        tasks = [(agent, context1), (agent, context2)]

        # 执行
        results = evaluator.batch_generate(tasks)

        # 验证
        assert len(results) == 2
        assert all(r.success for r in results)
        assert results[0].node.id == "node1"
        assert results[1].node.id == "node2"

    def test_batch_generate_with_failure(self, evaluator):
        """测试部分生成失败。"""
        agent = Mock(spec=BaseAgent)
        node1 = Node(id="node1", code="code1", step=0)
        agent.generate.side_effect = [
            AgentResult(node=node1, success=True),
            Exception("生成失败"),
        ]

        context1 = Mock(spec=AgentContext)
        context2 = Mock(spec=AgentContext)

        tasks = [(agent, context1), (agent, context2)]

        # 执行
        results = evaluator.batch_generate(tasks)

        # 验证
        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].error == "生成失败"


class TestEvaluateOne:
    """测试单个节点评估。"""

    def test_evaluate_one_success(self, evaluator, mock_interpreter):
        """测试成功评估节点。"""
        node = Node(id="test_node", code="print('Metric: 0.85')", step=0)

        # Mock 执行结果
        exec_result = ExecutionResult(
            term_out=["Metric: 0.85"],
            exec_time=1.5,
            success=True,
        )
        mock_interpreter.run.return_value = exec_result

        # 执行
        metric_value = evaluator._evaluate_one(node)

        # 验证
        assert metric_value == 0.85
        assert node.is_buggy is False
        assert node.metric_value is None  # _evaluate_one 不更新 metric_value

    def test_evaluate_one_execution_failure(self, evaluator, mock_interpreter):
        """测试执行失败的节点。"""
        node = Node(id="test_node", code="raise ValueError()", step=0)

        # Mock 执行结果（失败）
        exec_result = ExecutionResult(
            term_out=["ValueError"],
            exec_time=0.5,
            success=False,
            exc_type="ValueError",
            exc_info="ValueError traceback",
        )
        mock_interpreter.run.return_value = exec_result

        # 执行
        metric_value = evaluator._evaluate_one(node)

        # 验证
        assert metric_value == -1e9
        assert node.is_buggy is True
        assert node.exc_type == "ValueError"

    def test_evaluate_one_metric_parse_failure(self, evaluator, mock_interpreter):
        """测试 metric 解析失败（执行成功但无 metric）。"""
        node = Node(id="test_node", code="print('Hello')", step=0)

        # Mock 执行结果（成功但无 metric）
        exec_result = ExecutionResult(
            term_out=["Hello"],
            exec_time=1.0,
            success=True,
        )
        mock_interpreter.run.return_value = exec_result

        # 执行
        metric_value = evaluator._evaluate_one(node)

        # 验证
        assert metric_value == 0.0  # 默认值
        assert node.is_buggy is False


class TestParseMetric:
    """测试 metric 解析。"""

    def test_parse_metric_standard_format(self, evaluator):
        """测试标准格式 'Metric: 0.85'。"""
        term_out = "Training completed\nMetric: 0.85\nDone"
        metric = evaluator._parse_metric(term_out)
        assert metric == 0.85

    def test_parse_metric_accuracy(self, evaluator):
        """测试 Accuracy 格式。"""
        term_out = "Accuracy: 0.92"
        metric = evaluator._parse_metric(term_out)
        assert metric == 0.92

    def test_parse_metric_rmse(self, evaluator):
        """测试 RMSE 格式。"""
        term_out = "RMSE: 0.15"
        metric = evaluator._parse_metric(term_out)
        assert metric == 0.15

    def test_parse_metric_f1(self, evaluator):
        """测试 F1 格式。"""
        term_out = "F1: 0.88"
        metric = evaluator._parse_metric(term_out)
        assert metric == 0.88

    def test_parse_metric_negative_value(self, evaluator):
        """测试负数 metric。"""
        term_out = "Score: -0.5"
        metric = evaluator._parse_metric(term_out)
        assert metric == -0.5

    def test_parse_metric_no_match(self, evaluator):
        """测试无法解析时抛出异常。"""
        term_out = "No metric here"
        with pytest.raises(ValueError, match="无法从输出中解析 metric"):
            evaluator._parse_metric(term_out)


class TestBatchEvaluate:
    """测试批量评估。"""

    def test_batch_evaluate_success(self, evaluator, mock_interpreter):
        """测试成功评估多个节点。"""
        nodes = [
            Node(id="node1", code="print('Metric: 0.8')", step=0),
            Node(id="node2", code="print('Metric: 0.9')", step=1),
        ]

        # Mock 执行结果
        exec_result1 = ExecutionResult(
            term_out=["Metric: 0.8"],
            exec_time=1.0,
            success=True,
        )
        exec_result2 = ExecutionResult(
            term_out=["Metric: 0.9"],
            exec_time=1.2,
            success=True,
        )
        mock_interpreter.run.side_effect = [exec_result1, exec_result2]

        # 执行
        evaluator.batch_evaluate(nodes, current_step=10)

        # 验证
        assert nodes[0].metric_value == 0.8
        assert nodes[1].metric_value == 0.9
        assert nodes[0].is_buggy is False
        assert nodes[1].is_buggy is False

        # 验证信息素已更新
        assert nodes[0].metadata.get("pheromone_node") is not None
        assert nodes[1].metadata.get("pheromone_node") is not None

    def test_batch_evaluate_with_buggy_node(self, evaluator, mock_interpreter):
        """测试包含 buggy 节点的批量评估。"""
        nodes = [
            Node(id="node1", code="print('Metric: 0.8')", step=0),
            Node(id="node2", code="raise ValueError()", step=1),
        ]

        # Mock 执行结果
        exec_result1 = ExecutionResult(
            term_out=["Metric: 0.8"],
            exec_time=1.0,
            success=True,
        )
        exec_result2 = ExecutionResult(
            term_out=["ValueError"],
            exc_type="ValueError",
            success=False,
        )
        mock_interpreter.run.side_effect = [exec_result1, exec_result2]

        # 执行
        evaluator.batch_evaluate(nodes, current_step=10)

        # 验证
        assert nodes[0].metric_value == 0.8
        assert nodes[0].is_buggy is False
        assert nodes[1].metric_value == -1e9
        assert nodes[1].is_buggy is True


class TestUpdatePheromones:
    """测试信息素更新。"""

    def test_update_pheromones_basic(self, evaluator):
        """测试基本信息素更新。"""
        nodes = [
            Node(id="node1", code="code1", step=0),
            Node(id="node2", code="code2", step=1),
        ]
        nodes[0].metric_value = 0.8
        nodes[1].metric_value = 0.9
        nodes[0].is_buggy = False
        nodes[1].is_buggy = False
        nodes[0].genes = {"DATA": "data code 1"}
        nodes[1].genes = {"DATA": "data code 2"}

        # 执行
        evaluator._update_pheromones(nodes, current_step=10)

        # 验证节点信息素
        assert nodes[0].metadata["pheromone_node"] is not None
        assert nodes[1].metadata["pheromone_node"] is not None

        # 验证基因注册表更新
        registry = evaluator.gene_registry
        assert len(registry._registry["DATA"]) >= 1  # 至少有一个基因被注册

    def test_update_pheromones_skip_buggy(self, evaluator):
        """测试跳过 buggy 节点更新基因注册表。"""
        nodes = [
            Node(id="node1", code="code1", step=0),
        ]
        nodes[0].metric_value = None
        nodes[0].is_buggy = True
        nodes[0].genes = {"DATA": "data code"}

        # 执行
        evaluator._update_pheromones(nodes, current_step=10)

        # 验证：buggy 节点的基因不应被注册
        registry = evaluator.gene_registry
        assert len(registry._registry["DATA"]) == 0

    def test_update_pheromones_no_valid_nodes(self, evaluator):
        """测试没有有效节点时的处理。"""
        nodes = [
            Node(id="node1", code="code1", step=0),
        ]
        nodes[0].is_buggy = True

        # 执行（不应崩溃）
        evaluator._update_pheromones(nodes, current_step=10)


class TestConcurrentSubmissionFiles:
    """测试并发 submission 文件处理。"""

    def test_concurrent_submission_files_no_conflict(
        self, evaluator, mock_workspace, mock_interpreter
    ):
        """测试并发评估时 submission 文件不冲突。"""
        nodes = [
            Node(id="node_a", code="to_csv('./submission/submission.csv')", step=0),
            Node(id="node_b", code="to_csv('./submission/submission.csv')", step=1),
        ]

        # Mock 执行结果
        exec_result = ExecutionResult(
            term_out=["Metric: 0.8"],
            success=True,
        )
        mock_interpreter.run.return_value = exec_result

        # 执行
        evaluator.batch_evaluate(nodes, current_step=10)

        # 验证：rewrite_submission_path 被调用且使用了不同的文件名
        calls = mock_workspace.rewrite_submission_path.call_args_list
        assert len(calls) == 2
        assert "node_a" in str(calls[0])
        assert "node_b" in str(calls[1])


class TestShutdown:
    """测试资源清理。"""

    def test_shutdown(self, evaluator):
        """测试线程池正确关闭。"""
        # 执行
        evaluator.shutdown()

        # 验证：线程池已关闭（无法提交新任务）
        with pytest.raises(RuntimeError):
            evaluator.executor.submit(lambda: None)
