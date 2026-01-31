"""适应度计算模块。

提供指标归一化和单调化功能，统一转换为"越大越好"的适应度值。
"""

import math


def normalize_fitness(metric_value: float, lower_is_better: bool) -> float:
    """统一归一化到 [0, 1] 区间，转换为"越大越好"。

    Args:
        metric_value: 原始指标值
        lower_is_better: 指标方向
            - True: RMSE, MAE 等误差指标（越小越好）
            - False: Accuracy, F1, AUC 等性能指标（越大越好）

    Returns:
        归一化后的适应度值，范围 [0, 1]，越大表示越好

    Raises:
        ValueError: 如果 metric_value 为负数且 lower_is_better=False

    时间复杂度: O(1)

    实现原理:
        - lower_is_better=True:
            fitness = 1 / (1 + metric_value)
            → RMSE=0 时 fitness=1（完美）
            → RMSE→∞ 时 fitness→0（极差）

        - lower_is_better=False:
            - 若 metric_value ∈ [0, 1]: fitness = metric_value
            - 若 metric_value > 1: fitness = 1 / (1 + exp(-metric_value))（sigmoid 压缩）
            - 若 metric_value < 0: 抛出 ValueError

    示例:
        >>> # RMSE 类指标（越小越好）
        >>> normalize_fitness(0.0, lower_is_better=True)
        1.0
        >>> normalize_fitness(0.3, lower_is_better=True)
        0.769...
        >>> normalize_fitness(10.0, lower_is_better=True)
        0.090...

        >>> # Accuracy 类指标（越大越好，范围 [0, 1]）
        >>> normalize_fitness(0.85, lower_is_better=False)
        0.85
        >>> normalize_fitness(0.0, lower_is_better=False)
        0.0

        >>> # 超出 [0, 1] 的指标（使用 sigmoid 压缩）
        >>> normalize_fitness(5.0, lower_is_better=False)
        0.993...

        >>> # 负数指标（不合法）
        >>> normalize_fitness(-0.1, lower_is_better=False)
        Traceback (most recent call last):
        ...
        ValueError: metric_value 为负数但 lower_is_better=False，请检查指标方向
    """
    if lower_is_better:
        # 误差指标：转换为"越大越好"
        # fitness ∈ (0, 1]，metric_value=0 时 fitness=1
        fitness = 1.0 / (1.0 + metric_value)
        return fitness
    else:
        # 性能指标：保持"越大越好"
        if metric_value < 0:
            raise ValueError(
                f"metric_value 为负数但 lower_is_better=False，请检查指标方向: {metric_value}"
            )

        # 若在 [0, 1] 范围内，直接返回
        if 0 <= metric_value <= 1:
            return metric_value

        # 若超出 [0, 1]，使用 sigmoid 压缩到 (0.5, 1)
        # sigmoid(x) = 1 / (1 + exp(-x))
        fitness = 1.0 / (1.0 + math.exp(-metric_value))
        return fitness
