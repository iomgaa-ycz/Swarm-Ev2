"""搜索算法模块。

提供适应度计算、并行评估等搜索算法核心功能。
"""

from .fitness import normalize_fitness
from .parallel_evaluator import ParallelEvaluator

__all__ = ["normalize_fitness", "ParallelEvaluator"]
