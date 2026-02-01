"""SolutionEvolution 模块单元测试。"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from core.evolution.solution_evolution import SolutionEvolution
from core.state import Node, Journal
from core.evolution.gene_registry import GeneRegistry
from core.evolution.task_dispatcher import TaskDispatcher
from core.evolution.experience_pool import ExperiencePool
from search.parallel_evaluator import ParallelEvaluator
from agents.base_agent import BaseAgent, AgentContext, AgentResult
from utils.config import Config


@pytest.fixture
def mock_config():
    """创建 mock 配置对象。"""
    config = Mock(spec=Config)
    config.evolution = Mock()
    config.evolution.solution = Mock()
    config.evolution.solution.population_size = 12
    config.evolution.solution.elite_size = 3
    config.evolution.solution.crossover_rate = 0.8
    config.evolution.solution.mutation_rate = 0.2
    config.evolution.solution.tournament_k = 3
    config.evolution.solution.crossover_strategy = "random"
    return config


@pytest.fixture
def journal():
    """创建 Journal 实例。"""
    return Journal()


@pytest.fixture
def gene_registry():
    """创建 GeneRegistry 实例。"""
    return GeneRegistry()


@pytest.fixture
def mock_agents():
    """创建 mock Agent 列表。"""
    agent1 = Mock(spec=BaseAgent)
    agent1.name = "agent_1"
    agent2 = Mock(spec=BaseAgent)
    agent2.name = "agent_2"
    return [agent1, agent2]


@pytest.fixture
def mock_task_dispatcher(mock_agents):
    """创建 mock TaskDispatcher。"""
    dispatcher = Mock(spec=TaskDispatcher)
    dispatcher.select_agent.return_value = mock_agents[0]
    return dispatcher


@pytest.fixture
def mock_evaluator():
    """创建 mock ParallelEvaluator。"""
    evaluator = Mock(spec=ParallelEvaluator)
    return evaluator


@pytest.fixture
def experience_pool(mock_config):
    """创建 Mock ExperiencePool。"""
    # 使用 Mock 而不是真实的 ExperiencePool（避免文件 I/O）
    pool = Mock(spec=ExperiencePool)
    pool.records = []
    pool.add = Mock(side_effect=lambda record: pool.records.append(record))
    return pool


@pytest.fixture
def solution_evolution(
    mock_agents,
    mock_task_dispatcher,
    mock_evaluator,
    experience_pool,
    journal,
    gene_registry,
    mock_config,
):
    """创建 SolutionEvolution 实例。"""
    return SolutionEvolution(
        agents=mock_agents,
        task_dispatcher=mock_task_dispatcher,
        evaluator=mock_evaluator,
        experience_pool=experience_pool,
        journal=journal,
        gene_registry=gene_registry,
        config=mock_config,
    )


class TestSolutionEvolutionInit:
    """测试初始化。"""

    def test_init(self, solution_evolution, mock_config):
        """测试基本初始化。"""
        assert solution_evolution.population_size == 12
        assert solution_evolution.elite_size == 3
        assert solution_evolution.crossover_rate == 0.8
        assert solution_evolution.mutation_rate == 0.2
        assert solution_evolution.tournament_k == 3
        assert solution_evolution.crossover_strategy == "random"
        assert solution_evolution.current_step == 0
        assert len(solution_evolution.population) == 0


class TestInitializePopulation:
    """测试种群初始化。"""

    def test_initialize_population_success(
        self, solution_evolution, mock_evaluator, mock_agents
    ):
        """测试成功初始化种群。"""
        # 创建 mock 节点
        nodes = []
        for i in range(12):
            node = Node(id=f"node_{i}", code="test code", step=i)
            node.genes = {
                "DATA": "data",
                "MODEL": "model",
                "LOSS": "loss",
                "OPTIMIZER": "opt",
                "REGULARIZATION": "reg",
                "INITIALIZATION": "init",
                "TRAINING_TRICKS": "tricks",
            }
            nodes.append(node)

        # Mock batch_generate 返回成功结果
        results = [
            AgentResult(node=node, success=True) for node in nodes
        ]
        mock_evaluator.batch_generate.return_value = results

        # Mock batch_evaluate
        def mock_batch_eval(nodes_list, step):
            for node in nodes_list:
                node.metric_value = 0.8

        mock_evaluator.batch_evaluate.side_effect = mock_batch_eval

        # 执行
        start_time = time.time()
        solution_evolution.initialize_population(start_time, "test task")

        # 验证
        assert len(solution_evolution.population) == 12
        assert all(n.metric_value == 0.8 for n in solution_evolution.population)

    def test_initialize_population_with_failures(
        self, solution_evolution, mock_evaluator
    ):
        """测试部分初始化失败。"""
        # Mock batch_generate 返回部分失败结果
        nodes = [Node(id=f"node_{i}", code="test", step=i) for i in range(8)]
        for node in nodes:
            node.genes = {"DATA": "data", "MODEL": "model", "LOSS": "loss",
                         "OPTIMIZER": "opt", "REGULARIZATION": "reg",
                         "INITIALIZATION": "init", "TRAINING_TRICKS": "tricks"}

        results = [AgentResult(node=node, success=True) for node in nodes]
        results.extend([AgentResult(node=None, success=False, error="failed") for _ in range(4)])

        mock_evaluator.batch_generate.return_value = results
        mock_evaluator.batch_evaluate.side_effect = lambda nodes, step: None

        # 执行
        start_time = time.time()
        solution_evolution.initialize_population(start_time, "test task")

        # 验证：只有成功的节点被添加到种群
        assert len(solution_evolution.population) == 8


class TestSelectElites:
    """测试精英保留。"""

    def test_select_elites_basic(self, solution_evolution):
        """测试基本精英选择。"""
        # 创建种群
        solution_evolution.population = [
            Node(id=f"node_{i}", code="test", step=i, metric_value=float(i) / 10)
            for i in range(12)
        ]

        # 执行
        elites = solution_evolution._select_elites()

        # 验证：返回 top-3
        assert len(elites) == 3
        assert elites[0].metric_value == 1.1  # node_11
        assert elites[1].metric_value == 1.0  # node_10
        assert elites[2].metric_value == 0.9  # node_9

    def test_select_elites_with_buggy_nodes(self, solution_evolution):
        """测试包含 buggy 节点的精英选择。"""
        solution_evolution.population = []
        for i in range(5):
            node = Node(id=f"node_{i}", code="test", step=i)
            node.metric_value = float(i) / 10 if i < 3 else None
            node.is_buggy = i >= 3
            solution_evolution.population.append(node)

        # 执行
        elites = solution_evolution._select_elites()

        # 验证：buggy 节点的 metric_value 被视为 -1e9
        assert len(elites) == 3
        assert elites[0].metric_value == 0.2
        assert elites[1].metric_value == 0.1
        assert elites[2].metric_value == 0.0


class TestTournamentSelect:
    """测试锦标赛选择。"""

    def test_tournament_select_basic(self, solution_evolution):
        """测试基本锦标赛选择。"""
        # 创建种群
        solution_evolution.population = [
            Node(id=f"node_{i}", code="test", step=i, metric_value=float(i) / 10)
            for i in range(12)
        ]

        # 执行多次选择
        selected = [solution_evolution._tournament_select() for _ in range(100)]

        # 验证：高质量节点被选中的次数更多
        # 统计每个节点被选中的次数
        selection_counts = {}
        for node in selected:
            selection_counts[node.id] = selection_counts.get(node.id, 0) + 1

        # 最优节点（node_11）应该被选中最多
        max_selected_id = max(selection_counts, key=selection_counts.get)
        assert "node_1" in max_selected_id  # node_10 或 node_11


class TestCrossover:
    """测试基因交叉。"""

    def test_crossover_random_strategy(self, solution_evolution, mock_task_dispatcher, mock_agents):
        """测试随机交叉策略。"""
        solution_evolution.crossover_strategy = "random"

        parent_a = Node(id="parent_a", code="code_a", step=0)
        parent_b = Node(id="parent_b", code="code_b", step=1)

        # Mock merge Agent
        child_node = Node(id="child", code="child_code", step=2)
        child_node.genes = {"DATA": "data", "MODEL": "model", "LOSS": "loss",
                           "OPTIMIZER": "opt", "REGULARIZATION": "reg",
                           "INITIALIZATION": "init", "TRAINING_TRICKS": "tricks"}
        mock_agents[0].generate.return_value = AgentResult(node=child_node, success=True)
        mock_task_dispatcher.select_agent.return_value = mock_agents[0]

        # 执行
        child = solution_evolution._crossover(parent_a, parent_b, time.time(), "test")

        # 验证
        assert child is not None
        assert child.id == "child"
        assert child.genes is not None

    def test_crossover_failure(self, solution_evolution, mock_task_dispatcher, mock_agents):
        """测试交叉失败。"""
        parent_a = Node(id="parent_a", code="code_a", step=0)
        parent_b = Node(id="parent_b", code="code_b", step=1)

        # Mock merge Agent 失败
        mock_agents[0].generate.return_value = AgentResult(
            node=None, success=False, error="merge failed"
        )
        mock_task_dispatcher.select_agent.return_value = mock_agents[0]

        # 执行
        child = solution_evolution._crossover(parent_a, parent_b, time.time(), "test")

        # 验证
        assert child is None


class TestMutate:
    """测试基因变异。"""

    def test_mutate_success(self, solution_evolution, mock_task_dispatcher, mock_agents):
        """测试成功变异。"""
        node = Node(id="test_node", code="original_code", step=0)
        node.genes = {"DATA": "data"}

        # Mock mutate Agent
        mutated_code = "mutated_code"
        mutated_node = Node(id="test_node", code=mutated_code, step=0)
        mutated_node.plan = "mutated plan"
        mock_agents[0].generate.return_value = AgentResult(
            node=mutated_node, success=True
        )
        mock_task_dispatcher.select_agent.return_value = mock_agents[0]

        # 执行
        solution_evolution._mutate(node, time.time(), "test")

        # 验证：原地修改（代码和计划被更新）
        assert node.code == mutated_code
        assert node.plan == "mutated plan"

    def test_mutate_failure(self, solution_evolution, mock_task_dispatcher, mock_agents):
        """测试变异失败（节点保持不变）。"""
        node = Node(id="test_node", code="original_code", step=0)

        # Mock mutate Agent 失败
        mock_agents[0].generate.return_value = AgentResult(
            node=None, success=False, error="mutate failed"
        )
        mock_task_dispatcher.select_agent.return_value = mock_agents[0]

        # 执行
        solution_evolution._mutate(node, time.time(), "test")

        # 验证：节点保持不变
        assert node.code == "original_code"


class TestGenerateCrossoverPlan:
    """测试交叉计划生成。"""

    def test_generate_random_plan(self, solution_evolution):
        """测试随机交叉计划。"""
        plan = solution_evolution._generate_random_plan()

        # 验证
        assert isinstance(plan, dict)
        assert len(plan) == 7  # 7 个基因位点
        assert all(gene in plan for gene in ["DATA", "MODEL", "LOSS", "OPTIMIZER",
                                             "REGULARIZATION", "INITIALIZATION", "TRAINING_TRICKS"])
        assert all(choice in ["A", "B"] for choice in plan.values())

    def test_generate_pheromone_plan(self, solution_evolution, journal, gene_registry):
        """测试信息素驱动交叉计划。"""
        # 创建有效节点
        node = Node(id="node1", code="code", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {
            "DATA": "data code",
            "MODEL": "model code",
            "LOSS": "loss code",
            "OPTIMIZER": "opt code",
            "REGULARIZATION": "reg code",
            "INITIALIZATION": "init code",
            "TRAINING_TRICKS": "tricks code",
        }
        node.metadata = {"pheromone_node": 0.7}
        journal.append(node)
        gene_registry.update_from_reviewed_node(node)

        parent_a = Node(id="parent_a", code="code_a", step=1)
        parent_b = Node(id="parent_b", code="code_b", step=2)

        # 执行
        plan = solution_evolution._generate_pheromone_plan(parent_a, parent_b)

        # 验证
        assert isinstance(plan, dict)
        assert "reasoning" in plan
        assert "data_source" in plan


class TestRecordToExperiencePool:
    """测试经验池记录。"""

    def test_record_to_experience_pool(self, solution_evolution, experience_pool):
        """测试成功记录到经验池。"""
        node = Node(id="test_node", code="test_code", step=0)
        node.metric_value = 0.85
        node.exec_time = 1.5
        node.plan = "Test plan"

        # 执行
        solution_evolution._record_to_experience_pool("agent_1", "explore", node)

        # 验证
        assert len(experience_pool.records) == 1
        record = experience_pool.records[0]
        assert record.agent_id == "agent_1"
        assert record.task_type == "explore"
        assert record.output_quality == 0.85  # 现在是 float


class TestStep:
    """测试完整单步进化。"""

    def test_step_basic(self, solution_evolution, mock_evaluator, mock_task_dispatcher, mock_agents):
        """测试基本单步进化流程。"""
        # 初始化种群
        solution_evolution.population = []
        for i in range(12):
            node = Node(id=f"node_{i}", code="test", step=i)
            node.metric_value = float(i) / 10
            solution_evolution.population.append(node)

        # Mock crossover
        def mock_crossover(pa, pb, st, td):
            child = Node(id=f"child_{pa.id}_{pb.id}", code="child_code", step=100)
            child.genes = {"DATA": "data", "MODEL": "model", "LOSS": "loss",
                          "OPTIMIZER": "opt", "REGULARIZATION": "reg",
                          "INITIALIZATION": "init", "TRAINING_TRICKS": "tricks"}
            return child

        solution_evolution._crossover = mock_crossover

        # Mock mutate（不修改）
        solution_evolution._mutate = lambda node, st, td: None

        # Mock batch_evaluate
        def mock_batch_eval(nodes, step):
            for node in nodes:
                node.metric_value = 0.85

        mock_evaluator.batch_evaluate.side_effect = mock_batch_eval

        # 执行
        solution_evolution.step(time.time(), "test task")

        # 验证
        assert solution_evolution.current_step == 1
        assert len(solution_evolution.population) == 12
        # 精英应该被保留
        assert any(n.id.startswith("node_") for n in solution_evolution.population)


class TestCrossoverStrategySwitching:
    """测试交叉策略切换。"""

    def test_crossover_strategy_random(self, solution_evolution):
        """测试配置为 random 时使用随机交叉。"""
        solution_evolution.crossover_strategy = "random"

        parent_a = Node(id="a", code="a", step=0)
        parent_b = Node(id="b", code="b", step=1)

        plan = solution_evolution._generate_crossover_plan(parent_a, parent_b)

        # 验证：返回的是随机计划（dict 包含 "A" 或 "B"）
        assert isinstance(plan, dict)
        assert all(v in ["A", "B"] for v in plan.values())

    def test_crossover_strategy_invalid(self, solution_evolution):
        """测试无效策略抛出异常。"""
        solution_evolution.crossover_strategy = "invalid"

        parent_a = Node(id="a", code="a", step=0)
        parent_b = Node(id="b", code="b", step=1)

        with pytest.raises(ValueError, match="未知的交叉策略"):
            solution_evolution._generate_crossover_plan(parent_a, parent_b)
