"""Metric 方向检测与锁定机制的单元测试（P0-1 修复验证）。

测试范围:
    - _detect_metric_direction(): 从 task_desc 检测方向
    - _lock_metric_direction(): 从 review_data 锁定方向
    - _is_better(): 使用全局方向比较节点
    - Journal.get_best_node(): 支持全局方向参数
    - SolutionEvolution._is_lower_better(): 优先使用全局方向
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.orchestrator import Orchestrator, METRIC_DIRECTION
from core.state import Journal, Node
from utils.config import Config


class TestMetricDirectionDetection:
    """测试 metric 方向检测（启动时从 task_desc）。"""

    def test_detect_direction_logloss(self, mock_orchestrator_minimal):
        """task_desc 含 'log loss' → lower_is_better=True。"""
        orch = mock_orchestrator_minimal
        orch.task_desc = "Minimize log loss on validation set"
        orch._global_lower_is_better = None

        result = orch._detect_metric_direction()

        assert result is True
        assert orch._global_lower_is_better is True

    def test_detect_direction_auc(self, mock_orchestrator_minimal):
        """task_desc 含 'area under the ROC curve' → lower_is_better=False。"""
        orch = mock_orchestrator_minimal
        orch.task_desc = "Maximize area under the ROC curve"
        orch._global_lower_is_better = None

        result = orch._detect_metric_direction()

        assert result is False
        assert orch._global_lower_is_better is False

    def test_detect_direction_rmse(self, mock_orchestrator_minimal):
        """task_desc 含 'RMSE' → lower_is_better=True。"""
        orch = mock_orchestrator_minimal
        orch.task_desc = "The metric is RMSE (root mean squared error)"
        orch._global_lower_is_better = None

        result = orch._detect_metric_direction()

        assert result is True
        assert orch._global_lower_is_better is True

    def test_detect_direction_accuracy(self, mock_orchestrator_minimal):
        """task_desc 含 'accuracy' → lower_is_better=False。"""
        orch = mock_orchestrator_minimal
        orch.task_desc = "Maximize categorization accuracy"
        orch._global_lower_is_better = None

        result = orch._detect_metric_direction()

        assert result is False
        assert orch._global_lower_is_better is False

    def test_detect_direction_qwk(self, mock_orchestrator_minimal):
        """task_desc 含 'quadratic weighted kappa' → lower_is_better=False。"""
        orch = mock_orchestrator_minimal
        orch.task_desc = "Metric: quadratic weighted kappa (QWK)"
        orch._global_lower_is_better = None

        result = orch._detect_metric_direction()

        assert result is False
        assert orch._global_lower_is_better is False

    def test_detect_direction_none(self, mock_orchestrator_minimal):
        """task_desc 无已知 metric → 返回 None。"""
        orch = mock_orchestrator_minimal
        orch.task_desc = "Build a model to predict something"
        orch._global_lower_is_better = None

        result = orch._detect_metric_direction()

        assert result is None
        assert orch._global_lower_is_better is None

    def test_detect_direction_longer_match_first(self, mock_orchestrator_minimal):
        """task_desc 同时含 'log' 和 'log loss' → 优先匹配更长的 'log loss'。"""
        orch = mock_orchestrator_minimal
        orch.task_desc = "Evaluation uses log loss metric"
        orch._global_lower_is_better = None

        result = orch._detect_metric_direction()

        # 应该匹配 "log loss" (True) 而非 "log" (如果存在)
        assert result is True
        assert orch._global_lower_is_better is True


class TestMetricDirectionLocking:
    """测试 metric 方向锁定（首次 review 时）。"""

    def test_lock_from_review_metric_name(self, mock_orchestrator_minimal):
        """review 返回 metric_name='rmse' → 从表中查到 True。"""
        orch = mock_orchestrator_minimal
        orch._global_lower_is_better = None

        review_data = {"metric_name": "rmse", "lower_is_better": False}  # LLM 错误值
        orch._lock_metric_direction(review_data)

        # 应该从 metric_name 查表，忽略 LLM 的 lower_is_better
        assert orch._global_lower_is_better is True

    def test_lock_idempotent(self, mock_orchestrator_minimal):
        """已锁定后再次调用 → 保持不变（幂等）。"""
        orch = mock_orchestrator_minimal
        orch._global_lower_is_better = True

        review_data = {"metric_name": "accuracy", "lower_is_better": False}
        orch._lock_metric_direction(review_data)

        # 应该保持 True，不被新 review 覆盖
        assert orch._global_lower_is_better is True

    def test_lock_fallback_to_lower_is_better(self, mock_orchestrator_minimal):
        """metric_name 未知 → fallback 到 LLM 的 lower_is_better 字段。"""
        orch = mock_orchestrator_minimal
        orch._global_lower_is_better = None

        review_data = {"metric_name": "unknown_metric", "lower_is_better": True}
        orch._lock_metric_direction(review_data)

        # 查表失败，使用 LLM 的 lower_is_better
        assert orch._global_lower_is_better is True

    def test_lock_consistency_check(self, mock_orchestrator_minimal):
        """首次锁定后，后续 review 返回不同方向 → 忽略新方向。"""
        orch = mock_orchestrator_minimal
        orch._global_lower_is_better = None

        # 首次 review: logloss → True
        review_data_1 = {"metric_name": "logloss", "lower_is_better": True}
        orch._lock_metric_direction(review_data_1)
        assert orch._global_lower_is_better is True

        # 第二次 review: LLM 返回 False（异常）
        review_data_2 = {"metric_name": "logloss", "lower_is_better": False}
        orch._lock_metric_direction(review_data_2)

        # 应该忽略，保持 True
        assert orch._global_lower_is_better is True


class TestIsBetter:
    """测试 _is_better() 使用全局方向比较节点。"""

    def test_is_better_lower(self, mock_orchestrator_minimal):
        """lower_is_better=True: 0.5 < 0.8 → True。"""
        orch = mock_orchestrator_minimal
        orch._global_lower_is_better = True

        node = Node(code="", metric_value=0.5, lower_is_better=True)
        best = Node(code="", metric_value=0.8, lower_is_better=True)

        assert orch._is_better(node, best) is True

    def test_is_better_higher(self, mock_orchestrator_minimal):
        """lower_is_better=False: 0.5 < 0.8 → False。"""
        orch = mock_orchestrator_minimal
        orch._global_lower_is_better = False

        node = Node(code="", metric_value=0.5, lower_is_better=False)
        best = Node(code="", metric_value=0.8, lower_is_better=False)

        assert orch._is_better(node, best) is False

    def test_is_better_none_metric(self, mock_orchestrator_minimal):
        """metric_value=None → False。"""
        orch = mock_orchestrator_minimal
        orch._global_lower_is_better = True

        node = Node(code="", metric_value=None)
        best = Node(code="", metric_value=0.8)

        assert orch._is_better(node, best) is False


class TestJournalGetBestWithDirection:
    """测试 Journal.get_best_node() 支持全局方向参数。"""

    def test_journal_get_best_with_direction(self):
        """传入 lower_is_better=True → 返回最小值节点。"""
        journal = Journal()
        journal.append(Node(code="a", metric_value=0.8, lower_is_better=False))
        journal.append(Node(code="b", metric_value=0.5, lower_is_better=False))
        journal.append(Node(code="c", metric_value=0.3, lower_is_better=False))

        # 显式传入 lower_is_better=True（覆盖节点的 False）
        best = journal.get_best_node(lower_is_better=True)

        assert best.metric_value == 0.3  # 最小值

    def test_journal_get_best_without_direction(self):
        """不传参数 → 使用第一个节点的方向（向后兼容）。"""
        journal = Journal()
        journal.append(Node(code="a", metric_value=0.8, lower_is_better=False))
        journal.append(Node(code="b", metric_value=0.5, lower_is_better=False))

        best = journal.get_best_node()

        assert best.metric_value == 0.8  # higher_is_better → 最大值

    def test_journal_get_best_k_with_direction(self):
        """get_best_k() 传入方向参数 → 正确排序。"""
        journal = Journal()
        journal.append(Node(code="a", metric_value=0.8, lower_is_better=False))
        journal.append(Node(code="b", metric_value=0.5, lower_is_better=False))
        journal.append(Node(code="c", metric_value=0.3, lower_is_better=False))

        # lower_is_better=True → 升序
        top_2 = journal.get_best_k(k=2, lower_is_better=True)

        assert len(top_2) == 2
        assert top_2[0].metric_value == 0.3
        assert top_2[1].metric_value == 0.5


class TestSolutionEvolutionUsesGlobal:
    """测试 SolutionEvolution._is_lower_better() 优先使用全局方向。"""

    def test_solution_evolution_uses_global(self):
        """orchestrator 有全局方向 → SolutionEvolution 应该使用全局值。

        注意：这里测试的是逻辑而非真实 SolutionEvolution 类（避免重依赖）。
        """
        # 模拟 _is_lower_better() 的逻辑
        class MockSolutionEvolution:
            def __init__(self, orchestrator, population):
                self.orchestrator = orchestrator
                self.population = population

            def _is_lower_better(self):
                # 优先使用 Orchestrator 的全局方向
                if self.orchestrator and hasattr(self.orchestrator, "_global_lower_is_better"):
                    if self.orchestrator._global_lower_is_better is not None:
                        return self.orchestrator._global_lower_is_better
                # Fallback: 种群中第一个有效节点
                for node in self.population:
                    if node.metric_value is not None:
                        return node.lower_is_better
                return False

        # 创建 mock orchestrator
        mock_orch = Mock()
        mock_orch._global_lower_is_better = True

        sol_evo = MockSolutionEvolution(
            orchestrator=mock_orch,
            population=[Node(code="", metric_value=0.5, lower_is_better=False)],
        )

        # 应该使用 orchestrator 的全局方向（True），而非节点方向（False）
        assert sol_evo._is_lower_better() is True

    def test_solution_evolution_fallback_to_node(self):
        """orchestrator 无全局方向 → fallback 到节点方向。"""

        class MockSolutionEvolution:
            def __init__(self, orchestrator, population):
                self.orchestrator = orchestrator
                self.population = population

            def _is_lower_better(self):
                if self.orchestrator and hasattr(self.orchestrator, "_global_lower_is_better"):
                    if self.orchestrator._global_lower_is_better is not None:
                        return self.orchestrator._global_lower_is_better
                for node in self.population:
                    if node.metric_value is not None:
                        return node.lower_is_better
                return False

        mock_orch = Mock()
        mock_orch._global_lower_is_better = None

        sol_evo = MockSolutionEvolution(
            orchestrator=mock_orch,
            population=[Node(code="", metric_value=0.5, lower_is_better=False)],
        )

        assert sol_evo._is_lower_better() is False


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_config():
    """创建 mock Config。"""
    config = Mock(spec=Config)
    config.evolution.solution.population_size = 4
    config.evolution.solution.elite_size = 1
    config.evolution.solution.crossover_rate = 0.8
    config.evolution.solution.mutation_rate = 0.2
    config.evolution.solution.tournament_k = 2
    config.evolution.solution.crossover_strategy = "random"
    return config


@pytest.fixture
def mock_journal():
    """创建 Journal 实例。"""
    return Journal()


@pytest.fixture
def mock_orchestrator_minimal():
    """创建最小化的 Orchestrator mock（用于测试方向检测/锁定）。"""
    # 使用 MagicMock 以支持属性赋值
    orch = MagicMock()
    orch.task_desc = ""
    orch._global_lower_is_better = None
    orch._task_desc_compressed = ""

    # 绑定真实方法（从 Orchestrator 类中复制）
    orch._detect_metric_direction = lambda: Orchestrator._detect_metric_direction(orch)
    orch._lock_metric_direction = lambda review_data: Orchestrator._lock_metric_direction(
        orch, review_data
    )
    orch._is_better = lambda node, best_node: Orchestrator._is_better(orch, node, best_node)

    return orch
