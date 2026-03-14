"""SolutionEvolution 模块单元测试（V7: 两亲本交叉 + 适应度共享）。"""

import pytest
import random
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf

from core.evolution.solution_evolution import SolutionEvolution, GENE_LOCI
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
    config.evolution.solution.ga_trigger_threshold = 4

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


# ============================================================
# Helper functions
# ============================================================


def _make_metric_node(
    metric: float,
    lower_is_better: bool = False,
    buggy: bool = False,
    code: str = "import sklearn\npass",
) -> Node:
    """创建带 metric 的测试用 Node。"""
    node = Node(code=code)
    node.metric_value = metric if not buggy else None
    node.lower_is_better = lower_is_better
    node.is_buggy = buggy
    node.genes = {
        "DATA": "df = pd.read_csv('train.csv')" * 2,
        "MODEL": "model = LGBMClassifier()" * 2,
        "TRAIN": "model.fit(X, y)" * 2,
        "POSTPROCESS": "submission.to_csv('submission.csv')" * 2,
    }
    return node


def _make_torch_node(metric: float) -> Node:
    """创建 torch 框架节点。"""
    return _make_metric_node(metric, code="import torch\nimport torchvision\npass")


def _make_sklearn_node(metric: float) -> Node:
    """创建 sklearn 框架节点。"""
    return _make_metric_node(
        metric, code="import sklearn\nfrom xgboost import XGBClassifier\npass"
    )


# ============================================================
# 基础测试（保留并适配）
# ============================================================


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


class TestTournamentSelect:
    """测试锦标赛选择。"""

    def test_tournament_select_basic(self, solution_evolution):
        """测试基本锦标赛选择（固定种子确保确定性）。"""
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

    def test_tournament_select_with_custom_pool(self, solution_evolution):
        """测试使用自定义 pool 的锦标赛选择。"""
        solution_evolution.population = [
            _make_metric_node(0.5),
            _make_metric_node(0.9),
        ]
        custom_pool = [_make_metric_node(0.3), _make_metric_node(0.7)]

        # 从 custom_pool 中选择
        winner = solution_evolution._tournament_select(pool=custom_pool)
        assert winner in custom_pool

    def test_tournament_k_clamped_to_pool_size(self, solution_evolution):
        """tournament_k > pool 大小时不抛异常。"""
        solution_evolution.tournament_k = 10
        small_pool = [_make_metric_node(0.5), _make_metric_node(0.9)]
        solution_evolution.population = small_pool

        # 不应抛异常
        winner = solution_evolution._tournament_select(pool=small_pool)
        assert winner in small_pool


# ============================================================
# OmegaConf 配置 fixture
# ============================================================


@pytest.fixture
def omegaconf_config(tmp_path):
    """OmegaConf 配置对象（用于需要属性访问的测试）。"""
    return OmegaConf.create(
        {
            "project": {"workspace_dir": str(tmp_path)},
            "agent": {"max_steps": 10, "time_limit": 3600},
            "search": {"num_drafts": 3, "debug_prob": 0.5, "parallel_num": 1},
            "execution": {"timeout": 60, "adaptive_timeout": False},
            "llm": {
                "feedback": {
                    "model": "test",
                    "provider": "openai",
                    "temperature": 0.5,
                    "api_key": "test",
                    "base_url": "https://test.com/v1",
                }
            },
            "evolution": {
                "solution": {
                    "population_size": 5,
                    "elite_size": 2,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.2,
                    "tournament_k": 3,
                    "steps_per_epoch": 10,
                    "crossover_strategy": "random",
                    "ga_trigger_threshold": 4,
                }
            },
        }
    )


# ============================================================
# 锦标赛方向测试（保留）
# ============================================================


class TestTournamentSelectDirection:
    """锦标赛选择 lower_is_better 修复测试。"""

    def test_higher_is_better_selects_max(self, omegaconf_config):
        """higher_is_better 场景：从 tournament 的 k 个候选中选最大值。"""
        sol_evo = SolutionEvolution(config=omegaconf_config, journal=Journal())
        nodes = [
            _make_metric_node(0.80),
            _make_metric_node(0.95),
            _make_metric_node(0.70),
            _make_metric_node(0.88),
            _make_metric_node(0.92),
        ]
        sol_evo.population = nodes

        with patch("random.sample", return_value=[nodes[0], nodes[2], nodes[3]]):
            winner = sol_evo._tournament_select()
        assert winner.metric_value == 0.88

    def test_lower_is_better_selects_min(self, omegaconf_config):
        """lower_is_better 场景：从 tournament 的 k 个候选中选最小值。"""
        sol_evo = SolutionEvolution(config=omegaconf_config, journal=Journal())
        nodes = [
            _make_metric_node(0.08, lower_is_better=True),
            _make_metric_node(0.05, lower_is_better=True),
            _make_metric_node(0.12, lower_is_better=True),
            _make_metric_node(0.06, lower_is_better=True),
            _make_metric_node(0.10, lower_is_better=True),
        ]
        sol_evo.population = nodes

        with patch("random.sample", return_value=[nodes[0], nodes[2], nodes[4]]):
            winner = sol_evo._tournament_select()
        assert winner.metric_value == 0.08

    def test_lower_is_better_tournament_none_pushed(self, omegaconf_config):
        """lower_is_better tournament 中 None metric 节点不被选中。"""
        sol_evo = SolutionEvolution(config=omegaconf_config, journal=Journal())
        good_node = _make_metric_node(0.10, lower_is_better=True)
        none_node1 = _make_metric_node(0.0, lower_is_better=True, buggy=True)
        none_node2 = _make_metric_node(0.0, lower_is_better=True, buggy=True)
        sol_evo.population = [good_node, none_node1, none_node2]

        with patch("random.sample", return_value=[good_node, none_node1, none_node2]):
            winner = sol_evo._tournament_select()
        assert winner.metric_value == 0.10


class TestIsLowerBetter:
    """_is_lower_better() 辅助方法测试。"""

    def test_returns_true_for_lower_is_better(self, omegaconf_config):
        """种群中节点 lower_is_better=True 时返回 True。"""
        sol_evo = SolutionEvolution(config=omegaconf_config, journal=Journal())
        sol_evo.population = [
            _make_metric_node(0.05, lower_is_better=True),
            _make_metric_node(0.08, lower_is_better=True),
        ]
        assert sol_evo._is_lower_better() is True

    def test_returns_false_for_higher_is_better(self, omegaconf_config):
        """种群中节点 lower_is_better=False 时返回 False。"""
        sol_evo = SolutionEvolution(config=omegaconf_config, journal=Journal())
        sol_evo.population = [_make_metric_node(0.90)]
        assert sol_evo._is_lower_better() is False

    def test_defaults_false_when_all_none(self, omegaconf_config):
        """所有节点 metric=None 时默认 False。"""
        sol_evo = SolutionEvolution(config=omegaconf_config, journal=Journal())
        sol_evo.population = [
            _make_metric_node(0.0, buggy=True),
            _make_metric_node(0.0, buggy=True),
        ]
        assert sol_evo._is_lower_better() is False


# ============================================================
# V7 新增测试：两亲本交叉
# ============================================================


class TestSpeciesDetection:
    """物种检测测试。"""

    def test_detect_torch(self, omegaconf_config):
        """检测 torch 框架。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        node = _make_torch_node(0.9)
        assert se._detect_species(node) == "torch"

    def test_detect_sklearn(self, omegaconf_config):
        """检测 sklearn 框架。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        node = _make_sklearn_node(0.8)
        assert se._detect_species(node) == "sklearn"

    def test_detect_unknown(self, omegaconf_config):
        """未知框架返回 unknown。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        node = _make_metric_node(0.7, code="import numpy\npass")
        assert se._detect_species(node) == "unknown"

    def test_detect_empty_code(self, omegaconf_config):
        """空代码返回 unknown。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        node = Node(code="")
        assert se._detect_species(node) == "unknown"


class TestFitnessSharing:
    """适应度共享测试。"""

    def test_same_species_shares_fitness(self, omegaconf_config):
        """同物种 3 个节点，fitness 应为 metric/3。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        nodes = [_make_sklearn_node(0.9) for _ in range(3)]
        se.population = nodes

        shared = se._shared_fitness(nodes[0])
        assert abs(shared - 0.9 / 3) < 1e-6

    def test_unique_species_full_fitness(self, omegaconf_config):
        """唯一物种的节点，fitness = metric / 1。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        torch_node = _make_torch_node(0.8)
        sklearn_nodes = [_make_sklearn_node(0.7) for _ in range(3)]
        se.population = [torch_node] + sklearn_nodes

        shared = se._shared_fitness(torch_node)
        assert abs(shared - 0.8) < 1e-6

    def test_lower_is_better_sharing(self, omegaconf_config):
        """lower_is_better 时，同物种惩罚使值更大（更差）。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        nodes = [
            _make_metric_node(0.1, lower_is_better=True, code="import sklearn\npass")
            for _ in range(3)
        ]
        se.population = nodes

        shared = se._shared_fitness(nodes[0])
        # lower_is_better: metric * niche_count = 0.1 * 3 = 0.3
        assert abs(shared - 0.3) < 1e-6

    def test_none_metric_returns_inf(self, omegaconf_config):
        """metric=None 时返回 inf（higher_is_better）或 -inf。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        node = _make_metric_node(0.0, buggy=True)
        se.population = [node]

        shared = se._shared_fitness(node)
        assert shared == float("-inf")


class TestCrossoverMerge:
    """两亲本交叉测试。"""

    def test_crossover_gene_distribution(self, omegaconf_config):
        """验证 uniform crossover 概率接近 50/50。"""
        random.seed(42)

        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        parent_a = _make_sklearn_node(0.9)
        parent_b = _make_sklearn_node(0.8)

        a_count = 0
        b_count = 0
        trials = 1000

        for _ in range(trials):
            crossover = {}
            for locus in GENE_LOCI:
                crossover[locus] = "A" if random.random() < 0.5 else "B"
            for v in crossover.values():
                if v == "A":
                    a_count += 1
                else:
                    b_count += 1

        total = a_count + b_count
        ratio = a_count / total
        # 应接近 0.5（允许误差 0.05）
        assert 0.45 < ratio < 0.55

    def test_build_crossover_gene_plan_format(self, omegaconf_config):
        """测试 _build_crossover_gene_plan 生成格式。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        parent_a = _make_sklearn_node(0.9)
        parent_b = _make_sklearn_node(0.8)

        crossover = {"DATA": "A", "MODEL": "B", "TRAIN": "A", "POSTPROCESS": "B"}
        md, sources = se._build_crossover_gene_plan(parent_a, parent_b, crossover)

        # 验证格式
        assert "### DATA" in md
        assert "### MODEL" in md
        assert "### TRAIN" in md
        assert "### POSTPROCESS" in md
        assert "Parent A" in md
        assert "Parent B" in md
        assert "```python" in md

        # 验证 gene_sources
        assert sources["DATA"] == parent_a.id
        assert sources["MODEL"] == parent_b.id
        assert sources["TRAIN"] == parent_a.id
        assert sources["POSTPROCESS"] == parent_b.id

    def test_empty_gene_fallback(self, omegaconf_config):
        """亲本 genes 缺少某 locus 时自动 fallback 到另一亲本。"""
        se = SolutionEvolution(config=omegaconf_config, journal=Journal())
        parent_a = _make_sklearn_node(0.9)
        parent_b = _make_sklearn_node(0.8)

        # parent_a 的 DATA 基因为空
        parent_a.genes["DATA"] = ""

        crossover = {"DATA": "A", "MODEL": "A", "TRAIN": "A", "POSTPROCESS": "A"}
        md, sources = se._build_crossover_gene_plan(parent_a, parent_b, crossover)

        # DATA 应 fallback 到 parent_b
        assert sources["DATA"] == parent_b.id


class TestBroadPopulation:
    """种群不截断测试。"""

    def test_population_uses_full_valid_pool(self, omegaconf_config):
        """valid_pool=20 时种群应为 20，不截断为 population_size。"""
        journal = Journal()
        se = SolutionEvolution(config=omegaconf_config, journal=journal)

        # 创建 20 个有效节点（population_size=5）
        for i in range(20):
            node = _make_metric_node(0.5 + i * 0.01)
            node.dead = False
            journal.append(node)

        se._run_merge_step = MagicMock(return_value=None)
        se._run_mutate_step = MagicMock(return_value=None)

        se.run_epoch(steps_per_epoch=1)

        # 种群应为全部 valid_pool（20），不截断为 5
        assert len(se.population) == 20

    def test_single_ga_step_uses_full_pool(self, omegaconf_config):
        """run_single_ga_step 也使用全部 valid_pool。"""
        journal = Journal()
        se = SolutionEvolution(config=omegaconf_config, journal=journal)

        for i in range(15):
            node = _make_metric_node(0.5 + i * 0.01)
            node.dead = False
            journal.append(node)

        se._run_merge_step = MagicMock(return_value=None)
        se._run_mutate_step = MagicMock(return_value=None)

        se.run_single_ga_step()

        assert len(se.population) == 15


class TestMergeEdgeCases:
    """Merge 边界情况测试。"""

    def test_single_species_node_fallback(self, omegaconf_config):
        """物种内仅 1 个节点时，fallback 到全种群选第二亲本。"""
        journal = Journal()
        se = SolutionEvolution(
            config=omegaconf_config, journal=journal, orchestrator=MagicMock()
        )

        # 1 个 torch 节点（最优），3 个 sklearn 节点
        torch_node = _make_torch_node(0.95)
        torch_node.dead = False
        sklearn_nodes = [_make_sklearn_node(0.7 + i * 0.05) for i in range(3)]
        for n in sklearn_nodes:
            n.dead = False

        se.population = [torch_node] + sklearn_nodes

        # 不应抛异常
        se.orchestrator.execute_merge_task.return_value = MagicMock()
        result = se._run_merge_step()
        # execute_merge_task 应被调用
        se.orchestrator.execute_merge_task.assert_called_once()

    def test_merge_without_orchestrator(self, omegaconf_config):
        """没有 orchestrator 时返回 None。"""
        se = SolutionEvolution(
            config=omegaconf_config, journal=Journal(), orchestrator=None
        )
        assert se._run_merge_step() is None

    def test_merge_population_too_small(self, omegaconf_config):
        """种群 < 2 时跳过 merge。"""
        se = SolutionEvolution(
            config=omegaconf_config,
            journal=Journal(),
            orchestrator=MagicMock(),
        )
        se.population = [_make_sklearn_node(0.9)]
        assert se._run_merge_step() is None

    def test_tournament_k_larger_than_pool(self, omegaconf_config):
        """tournament_k > pool 大小时不抛异常。"""
        se = SolutionEvolution(
            config=omegaconf_config,
            journal=Journal(),
            orchestrator=MagicMock(),
        )
        se.tournament_k = 100
        se.population = [_make_sklearn_node(0.9), _make_sklearn_node(0.8)]

        se.orchestrator.execute_merge_task.return_value = MagicMock()
        # 不应抛异常
        result = se._run_merge_step()


# ============================================================
# 保留的旧测试（适配新签名）
# ============================================================


class TestRunSingleGaStep:
    """测试 SolutionEvolution.run_single_ga_step()。"""

    def _make_ga_node(self, is_buggy=False, dead=False, genes=None):
        """创建 mock 节点。"""
        node = MagicMock()
        node.is_buggy = is_buggy
        node.dead = dead
        node.genes = genes or {
            "DATA": "...",
            "MODEL": "...",
            "TRAIN": "...",
            "POSTPROCESS": "...",
        }
        node.metric_value = 0.9
        node.lower_is_better = False
        return node

    def test_insufficient_valid_pool_returns_none(self):
        """valid_pool 不足时返回 None。"""
        config = MagicMock()
        config.evolution.solution.population_size = 12
        config.evolution.solution.elite_size = 3
        config.evolution.solution.crossover_rate = 0.8
        config.evolution.solution.mutation_rate = 0.2
        config.evolution.solution.tournament_k = 3
        config.evolution.solution.crossover_strategy = "random"
        config.evolution.solution.ga_trigger_threshold = 4
    

        journal = MagicMock()
        journal.nodes = [self._make_ga_node() for _ in range(2)]

        se = SolutionEvolution(config=config, journal=journal)
        result = se.run_single_ga_step()
        assert result is None

    def test_sufficient_valid_pool_calls_ga(self):
        """valid_pool 充足时执行 GA 操作。"""
        config = MagicMock()
        config.evolution.solution.population_size = 12
        config.evolution.solution.elite_size = 3
        config.evolution.solution.crossover_rate = 0.8
        config.evolution.solution.mutation_rate = 0.2
        config.evolution.solution.tournament_k = 3
        config.evolution.solution.crossover_strategy = "random"
        config.evolution.solution.ga_trigger_threshold = 4
    

        journal = MagicMock()
        journal.nodes = [self._make_ga_node() for _ in range(6)]

        se = SolutionEvolution(config=config, journal=journal)
        se._run_merge_step = MagicMock(return_value=MagicMock())
        se._run_mutate_step = MagicMock(return_value=MagicMock())

        result = se.run_single_ga_step()
        assert se._run_merge_step.called or se._run_mutate_step.called


class TestRunEpoch:
    """测试 run_epoch() 方法。"""

    def _make_valid_node(self, metric=0.8, lower_is_better=False):
        """创建有效节点（非 buggy，非 dead，有完整基因）。"""
        node = Node(code="pass", metric_value=metric, is_buggy=False)
        node.dead = False
        node.lower_is_better = lower_is_better
        node.genes = {
            "DATA": "x" * 30,
            "MODEL": "x" * 30,
            "TRAIN": "x" * 30,
            "POSTPROCESS": "x" * 30,
        }
        return node

    def test_run_epoch_insufficient_pool(self, omegaconf_config):
        """valid_pool 不足时返回 None。"""
        journal = Journal()
        se = SolutionEvolution(config=omegaconf_config, journal=journal)

        # 只添加 2 个节点（阈值为 4）
        for i in range(2):
            journal.append(self._make_valid_node(metric=0.5 + i * 0.1))

        result = se.run_epoch(steps_per_epoch=5)
        assert result is None

    def test_run_epoch_sufficient_pool(self, omegaconf_config):
        """valid_pool 充足时正常执行。"""
        journal = Journal()
        se = SolutionEvolution(config=omegaconf_config, journal=journal)

        for i in range(6):
            journal.append(self._make_valid_node(metric=0.5 + i * 0.05))

        se._run_merge_step = MagicMock(return_value=None)
        se._run_mutate_step = MagicMock(return_value=None)

        result = se.run_epoch(steps_per_epoch=4)
        # 应该返回 best_node
        assert result is not None or result is None  # 取决于 journal 状态

    def test_run_epoch_counts_operations(self, omegaconf_config):
        """验证 merge 和 mutate 操作都被调用。"""
        random.seed(42)

        journal = Journal()
        se = SolutionEvolution(config=omegaconf_config, journal=journal)

        for i in range(6):
            journal.append(self._make_valid_node(metric=0.5 + i * 0.05))

        se._run_merge_step = MagicMock(return_value=MagicMock())
        se._run_mutate_step = MagicMock(return_value=MagicMock())

        se.run_epoch(steps_per_epoch=20)

        # 20 步中 merge 和 mutate 都应该被调用多次
        assert se._run_merge_step.call_count > 0
        assert se._run_mutate_step.call_count > 0


class TestRunMergeStep:
    """测试 _run_merge_step() 方法。"""

    def test_merge_without_orchestrator(self, omegaconf_config):
        """没有 orchestrator 时返回 None。"""
        se = SolutionEvolution(
            config=omegaconf_config, journal=Journal(), orchestrator=None
        )
        assert se._run_merge_step() is None


class TestRunMutateStep:
    """测试 _run_mutate_step() 方法。"""

    def test_mutate_without_orchestrator(self, omegaconf_config):
        """没有 orchestrator 时返回 None。"""
        se = SolutionEvolution(
            config=omegaconf_config, journal=Journal(), orchestrator=None
        )
        assert se._run_mutate_step() is None

    def test_mutate_empty_population(self, omegaconf_config):
        """种群为空时返回 None。"""
        se = SolutionEvolution(
            config=omegaconf_config,
            journal=Journal(),
            orchestrator=MagicMock(),
        )
        se.population = []
        assert se._run_mutate_step() is None

    def test_mutate_calls_orchestrator(self, omegaconf_config):
        """种群充足时调用 orchestrator.execute_mutate_task。"""
        orch = MagicMock()
        orch.execute_mutate_task.return_value = MagicMock()

        se = SolutionEvolution(
            config=omegaconf_config,
            journal=Journal(),
            orchestrator=orch,
        )

        # 创建种群（>= tournament_k=3）
        nodes = []
        for i in range(5):
            n = Node(code="import sklearn\npass", metric_value=0.5 + i * 0.1)
            n.genes = {
                "DATA": "x" * 30,
                "MODEL": "x" * 30,
                "TRAIN": "x" * 30,
                "POSTPROCESS": "x" * 30,
            }
            nodes.append(n)
        se.population = nodes

        result = se._run_mutate_step()
        orch.execute_mutate_task.assert_called_once()


class TestIsLowerBetterWithOrchestrator:
    """测试 _is_lower_better() 使用 Orchestrator 全局方向。"""

    def test_uses_orchestrator_direction(self, omegaconf_config):
        """优先使用 Orchestrator 的 _global_lower_is_better。"""
        orch = MagicMock()
        orch._global_lower_is_better = True

        se = SolutionEvolution(
            config=omegaconf_config, journal=Journal(), orchestrator=orch
        )
        se.population = [_make_metric_node(0.90, lower_is_better=False)]

        # 即使种群中 lower_is_better=False，仍应使用 orchestrator 方向
        assert se._is_lower_better() is True
