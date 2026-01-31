"""适应度计算单元测试。"""

import pytest
import math

from search.fitness import normalize_fitness


class TestNormalizeFitness:
    """测试 normalize_fitness 函数。"""

    def test_lower_is_better_zero(self):
        """测试 RMSE=0 的情况（完美）。"""
        fitness = normalize_fitness(0.0, lower_is_better=True)

        assert fitness == 1.0

    def test_lower_is_better_small(self):
        """测试 RMSE 较小的情况。"""
        fitness = normalize_fitness(0.3, lower_is_better=True)

        # fitness = 1 / (1 + 0.3) = 0.769...
        assert abs(fitness - 0.769) < 0.01

    def test_lower_is_better_large(self):
        """测试 RMSE 较大的情况。"""
        fitness = normalize_fitness(10.0, lower_is_better=True)

        # fitness = 1 / (1 + 10) = 0.0909...
        assert abs(fitness - 0.0909) < 0.01

    def test_lower_is_better_very_large(self):
        """测试 RMSE 极大的情况。"""
        fitness = normalize_fitness(1000.0, lower_is_better=True)

        # fitness = 1 / (1 + 1000) ≈ 0.001
        assert fitness < 0.01

    def test_higher_is_better_in_range(self):
        """测试 Accuracy 在 [0, 1] 范围内。"""
        fitness1 = normalize_fitness(0.85, lower_is_better=False)
        fitness2 = normalize_fitness(0.0, lower_is_better=False)
        fitness3 = normalize_fitness(1.0, lower_is_better=False)

        assert fitness1 == 0.85
        assert fitness2 == 0.0
        assert fitness3 == 1.0

    def test_higher_is_better_out_of_range(self):
        """测试超出 [0, 1] 范围的指标（sigmoid 压缩）。"""
        fitness = normalize_fitness(5.0, lower_is_better=False)

        # sigmoid(5) = 1 / (1 + exp(-5)) ≈ 0.993
        expected = 1.0 / (1.0 + math.exp(-5.0))
        assert abs(fitness - expected) < 0.001

    def test_higher_is_better_large_value(self):
        """测试非常大的指标值（sigmoid 接近 1）。"""
        fitness = normalize_fitness(100.0, lower_is_better=False)

        # sigmoid(100) ≈ 1.0
        assert fitness > 0.99

    def test_higher_is_better_negative_value(self):
        """测试负数指标（应该抛出异常）。"""
        with pytest.raises(
            ValueError, match="metric_value 为负数但 lower_is_better=False"
        ):
            normalize_fitness(-0.1, lower_is_better=False)

    def test_boundary_cases(self):
        """测试边界情况。"""
        # lower_is_better=True, metric_value=0
        fitness1 = normalize_fitness(0.0, lower_is_better=True)
        assert fitness1 == 1.0

        # lower_is_better=False, metric_value=0
        fitness2 = normalize_fitness(0.0, lower_is_better=False)
        assert fitness2 == 0.0

        # lower_is_better=False, metric_value=1
        fitness3 = normalize_fitness(1.0, lower_is_better=False)
        assert fitness3 == 1.0

    def test_comparison_between_types(self):
        """测试不同类型指标的适应度对比。"""
        # RMSE=0.3 vs Accuracy=0.85
        fitness_rmse = normalize_fitness(0.3, lower_is_better=True)
        fitness_acc = normalize_fitness(0.85, lower_is_better=False)

        # RMSE=0.3 → fitness≈0.769
        # Accuracy=0.85 → fitness=0.85
        # Accuracy 更好
        assert fitness_acc > fitness_rmse

    def test_monotonicity_lower_is_better(self):
        """测试 lower_is_better=True 的单调性（越小越好）。"""
        fitness_small = normalize_fitness(0.1, lower_is_better=True)
        fitness_large = normalize_fitness(1.0, lower_is_better=True)

        assert fitness_small > fitness_large

    def test_monotonicity_higher_is_better(self):
        """测试 lower_is_better=False 的单调性（越大越好）。"""
        fitness_small = normalize_fitness(0.3, lower_is_better=False)
        fitness_large = normalize_fitness(0.9, lower_is_better=False)

        assert fitness_large > fitness_small
