"""
评估指标工具模块。

提供指标值容器和比较工具，用于 Node 的性能评估。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricValue:
    """评估指标值容器。

    Attributes:
        value: 指标值（浮点数），None 表示无效/失败
        lower_is_better: True 表示越低越好（如 loss），False 表示越高越好（如 accuracy）

    Examples:
        >>> metric = MetricValue(value=0.85, lower_is_better=False)
        >>> metric.value
        0.85
    """

    value: Optional[float]
    lower_is_better: bool = False

    def is_better_than(self, other: "MetricValue") -> bool:
        """比较两个指标值。

        Args:
            other: 另一个 MetricValue

        Returns:
            True 如果当前指标优于 other

        Raises:
            ValueError: 如果 lower_is_better 设置不一致
        """
        if self.lower_is_better != other.lower_is_better:
            raise ValueError("无法比较 lower_is_better 设置不同的指标")

        # 任一指标无效，优先选择有效的
        if self.value is None:
            return False
        if other.value is None:
            return True

        # 根据优化方向比较
        if self.lower_is_better:
            return self.value < other.value
        else:
            return self.value > other.value


def WorstMetricValue(lower_is_better: bool = False) -> MetricValue:
    """返回最差的指标值（用于初始化或失败情况）。

    Args:
        lower_is_better: 优化方向

    Returns:
        最差的 MetricValue（Infinity 或 -Infinity）

    Examples:
        >>> worst = WorstMetricValue(lower_is_better=True)
        >>> worst.value
        inf
        >>> worst = WorstMetricValue(lower_is_better=False)
        >>> worst.value
        -inf
    """
    if lower_is_better:
        return MetricValue(value=float("inf"), lower_is_better=True)
    else:
        return MetricValue(value=float("-inf"), lower_is_better=False)


def compare_metrics(a: MetricValue, b: MetricValue) -> int:
    """比较两个指标值。

    Args:
        a: 第一个 MetricValue
        b: 第二个 MetricValue

    Returns:
        -1 如果 a < b
         0 如果 a == b
         1 如果 a > b

    Raises:
        ValueError: 如果 lower_is_better 设置不一致

    Examples:
        >>> m1 = MetricValue(value=0.8, lower_is_better=False)
        >>> m2 = MetricValue(value=0.9, lower_is_better=False)
        >>> compare_metrics(m1, m2)
        -1
    """
    if a.lower_is_better != b.lower_is_better:
        raise ValueError("无法比较 lower_is_better 设置不同的指标")

    # 处理 None 值
    if a.value is None and b.value is None:
        return 0
    if a.value is None:
        return -1
    if b.value is None:
        return 1

    # 正常比较
    if a.is_better_than(b):
        return 1
    elif b.is_better_than(a):
        return -1
    else:
        return 0
