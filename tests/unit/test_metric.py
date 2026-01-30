"""
utils/metric.py 的单元测试。
"""

import pytest
from utils.metric import MetricValue, WorstMetricValue, compare_metrics


class TestMetricValue:
    """测试 MetricValue 数据类。"""

    def test_metric_value_creation(self):
        """测试创建 MetricValue。"""
        metric = MetricValue(value=0.85, lower_is_better=False)
        assert metric.value == 0.85
        assert metric.lower_is_better is False

    def test_is_better_than_maximize(self):
        """测试 maximize 模式的比较。"""
        m1 = MetricValue(value=0.8, lower_is_better=False)
        m2 = MetricValue(value=0.9, lower_is_better=False)
        assert m2.is_better_than(m1)
        assert not m1.is_better_than(m2)

    def test_is_better_than_minimize(self):
        """测试 minimize 模式的比较。"""
        m1 = MetricValue(value=0.5, lower_is_better=True)
        m2 = MetricValue(value=0.3, lower_is_better=True)
        assert m2.is_better_than(m1)
        assert not m1.is_better_than(m2)

    def test_is_better_than_with_none(self):
        """测试 None 值的比较。"""
        m1 = MetricValue(value=None, lower_is_better=False)
        m2 = MetricValue(value=0.8, lower_is_better=False)
        assert m2.is_better_than(m1)
        assert not m1.is_better_than(m2)

    def test_is_better_than_inconsistent_direction(self):
        """测试不一致的优化方向。"""
        m1 = MetricValue(value=0.8, lower_is_better=False)
        m2 = MetricValue(value=0.5, lower_is_better=True)
        with pytest.raises(ValueError):
            m1.is_better_than(m2)


class TestWorstMetricValue:
    """测试 WorstMetricValue 函数。"""

    def test_worst_metric_minimize(self):
        """测试 minimize 模式的最差值。"""
        worst = WorstMetricValue(lower_is_better=True)
        assert worst.value == float("inf")
        assert worst.lower_is_better is True

    def test_worst_metric_maximize(self):
        """测试 maximize 模式的最差值。"""
        worst = WorstMetricValue(lower_is_better=False)
        assert worst.value == float("-inf")
        assert worst.lower_is_better is False


class TestCompareMetrics:
    """测试 compare_metrics 函数。"""

    def test_compare_greater(self):
        """测试第一个指标更优。"""
        m1 = MetricValue(value=0.9, lower_is_better=False)
        m2 = MetricValue(value=0.8, lower_is_better=False)
        assert compare_metrics(m1, m2) == 1

    def test_compare_less(self):
        """测试第二个指标更优。"""
        m1 = MetricValue(value=0.8, lower_is_better=False)
        m2 = MetricValue(value=0.9, lower_is_better=False)
        assert compare_metrics(m1, m2) == -1

    def test_compare_equal(self):
        """测试两个指标相等。"""
        m1 = MetricValue(value=0.85, lower_is_better=False)
        m2 = MetricValue(value=0.85, lower_is_better=False)
        assert compare_metrics(m1, m2) == 0

    def test_compare_both_none(self):
        """测试两个都是 None。"""
        m1 = MetricValue(value=None, lower_is_better=False)
        m2 = MetricValue(value=None, lower_is_better=False)
        assert compare_metrics(m1, m2) == 0

    def test_compare_inconsistent_direction(self):
        """测试不一致的优化方向。"""
        m1 = MetricValue(value=0.8, lower_is_better=False)
        m2 = MetricValue(value=0.5, lower_is_better=True)
        with pytest.raises(ValueError):
            compare_metrics(m1, m2)
