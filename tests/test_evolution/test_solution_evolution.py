"""SolutionEvolution 模块单元测试。"""

import pytest
from unittest.mock import Mock

from core.evolution.solution_evolution import SolutionEvolution
from core.state import Node, Journal
from core.evolution.gene_registry import GeneRegistry
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
def solution_evolution(
    journal,
    gene_registry,
    mock_config,
):
    """创建 SolutionEvolution 实例（MVP 签名）。"""
    se = SolutionEvolution(
        config=mock_config,
        journal=journal,
        orchestrator=None,
        gene_registry=gene_registry,
    )
    return se


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
        assert len(solution_evolution.population) == 0


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
        """测试基本锦标赛选择（固定种子确保确定性）。"""
        import random

        random.seed(42)

        # 创建种群
        solution_evolution.population = [
            Node(id=f"node_{i}", code="test", step=i, metric_value=float(i) / 10)
            for i in range(12)
        ]

        # 执行多次选择
        selected = [solution_evolution._tournament_select() for _ in range(100)]

        # 验证：高质量节点被选中的次数更多
        selection_counts = {}
        for node in selected:
            selection_counts[node.id] = selection_counts.get(node.id, 0) + 1

        # 最优节点（node_11）应该被选中最多
        max_selected_id = max(selection_counts, key=selection_counts.get)
        assert "node_1" in max_selected_id  # node_10 或 node_11


class TestBuildGenePlanMarkdown:
    """测试统一 Markdown gene_plan 生成。"""

    def test_random_markdown_format(self, solution_evolution):
        """测试随机策略生成 Markdown 格式。"""
        parent_a = Node(id="aaaaaaaa", code="code_a", step=0)
        parent_a.metric_value = 0.85
        parent_a.genes = {
            "DATA": "df = pd.read_csv('train.csv')",
            "MODEL": "model = LGBMClassifier()",
            "LOSS": "loss = 'binary_crossentropy'",
            "OPTIMIZER": "opt = Adam(lr=0.001)",
            "REGULARIZATION": "dropout = 0.5",
            "INITIALIZATION": "seed = 42",
            "TRAINING_TRICKS": "early_stopping = True",
        }

        parent_b = Node(id="bbbbbbbb", code="code_b", step=1)
        parent_b.metric_value = 0.80
        parent_b.genes = {
            "DATA": "df = pd.read_parquet('train.parquet')",
            "MODEL": "model = XGBClassifier()",
            "LOSS": "loss = 'hinge'",
            "OPTIMIZER": "opt = SGD(lr=0.01)",
            "REGULARIZATION": "l2 = 0.01",
            "INITIALIZATION": "seed = 0",
            "TRAINING_TRICKS": "lr_schedule = True",
        }

        md = solution_evolution._build_gene_plan_markdown_from_random(
            parent_a, parent_b
        )

        # 验证格式
        assert isinstance(md, str)
        assert "### DATA" in md
        assert "### MODEL" in md
        assert "```python" in md
        assert "fitness=" in md

    def test_pheromone_markdown_format(self, solution_evolution):
        """测试信息素策略生成 Markdown 格式。"""
        raw_plan = {
            "reasoning": "test",
            "data_source": {
                "locus": "DATA",
                "source_node_id": "node_abc123",
                "gene_id": "g1",
                "code": "df = pd.read_csv('train.csv')",
                "source_score": 0.85,
            },
            "model_source": {
                "locus": "MODEL",
                "source_node_id": "node_xyz789",
                "gene_id": "g2",
                "code": "model = LGBMClassifier()",
                "source_score": 0.83,
            },
        }

        md = solution_evolution._build_gene_plan_markdown_from_pheromone(raw_plan)

        # 验证格式
        assert "### DATA (from node_abc, fitness=0.8500)" in md
        assert "### MODEL (from node_xyz, fitness=0.8300)" in md
        assert "```python" in md
        assert "pd.read_csv" in md

    def test_crossover_mvp_random_calls_orchestrator(self, solution_evolution):
        """测试 _crossover_mvp 随机策略调用 Orchestrator。"""
        solution_evolution.crossover_strategy = "random"

        # Mock Orchestrator
        mock_orch = Mock()
        child_node = Node(id="child", code="child_code", step=2)
        mock_orch.execute_merge_task.return_value = child_node
        solution_evolution.orchestrator = mock_orch

        parent_a = Node(id="aaaaaaaa", code="code_a", step=0)
        parent_a.metric_value = 0.85
        parent_a.genes = {
            g: f"code_{g}"
            for g in [
                "DATA",
                "MODEL",
                "LOSS",
                "OPTIMIZER",
                "REGULARIZATION",
                "INITIALIZATION",
                "TRAINING_TRICKS",
            ]
        }
        parent_b = Node(id="bbbbbbbb", code="code_b", step=1)
        parent_b.metric_value = 0.80
        parent_b.genes = {
            g: f"code_{g}_b"
            for g in [
                "DATA",
                "MODEL",
                "LOSS",
                "OPTIMIZER",
                "REGULARIZATION",
                "INITIALIZATION",
                "TRAINING_TRICKS",
            ]
        }

        child = solution_evolution._crossover_mvp(parent_a, parent_b)

        # 验证：调用 Orchestrator，gene_plan 为 Markdown 字符串
        assert child is not None
        mock_orch.execute_merge_task.assert_called_once()
        call_args = mock_orch.execute_merge_task.call_args
        gene_plan_arg = call_args[0][2]  # 第三个位置参数
        assert isinstance(gene_plan_arg, str)
        assert "### DATA" in gene_plan_arg
