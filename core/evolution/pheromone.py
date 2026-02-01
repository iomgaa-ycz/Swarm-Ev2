"""节点级信息素统计工具模块。"""

from __future__ import annotations

import math
from typing import Optional

from core.state import Node


def ensure_node_stats(node: Node) -> None:
    """
    确保节点元数据包含信息素统计字段。

    Args:
        node: 待处理的节点对象
    """
    if node.metadata is None:
        node.metadata = {}

    metadata = node.metadata
    usage = metadata.get("usage_count")
    if not isinstance(usage, int):
        metadata["usage_count"] = 0
    success = metadata.get("success_count")
    if not isinstance(success, int):
        metadata["success_count"] = 0
    if "pheromone_node" not in metadata:
        metadata["pheromone_node"] = None


def _normalize_score(
    node: Node, score_min: Optional[float], score_max: Optional[float]
) -> float:
    """
    归一化节点得分到 [0, 1] 区间。

    Args:
        node: 待归一化的节点
        score_min: 最小得分值
        score_max: 最大得分值

    Returns:
        float: 归一化后的得分，范围在 [0, 1]
    """
    if node.metric_value is None or node.is_buggy:
        return 0.0
    if score_min is None or score_max is None:
        return 0.3
    score_range = score_max - score_min
    if score_range <= 0:
        return 0.3
    return max(0.0, min(1.0, (node.metric_value - score_min) / score_range))


def compute_node_pheromone(
    node: Node,
    current_step: int,
    score_min: Optional[float],
    score_max: Optional[float],
    alpha: float = 0.5,
    beta: float = 0.3,
    delta: float = 0.2,
    lambda_: float = 0.05,
) -> float:
    """
    计算节点的信息素值。

    信息素计算公式：
    pheromone = alpha * norm_score + beta * success_ratio + delta * recency

    Args:
        node: 待计算信息素的节点
        current_step: 当前步骤编号
        score_min: 得分最小值
        score_max: 得分最大值
        alpha: 归一化得分权重系数（默认 0.5）
        beta: 成功率权重系数（默认 0.3）
        delta: 时间衰减权重系数（默认 0.2）
        lambda_: 时间衰减速率参数（默认 0.05）

    Returns:
        float: 计算得到的信息素值
    """
    # 归一化得分
    norm_score = _normalize_score(node, score_min, score_max)

    # 成功率
    if node.metadata is None:
        node.metadata = {}

    usage_count = int(node.metadata.get("usage_count", 0))
    success_count = int(node.metadata.get("success_count", 0))
    usage_denom = max(1, usage_count)
    success_ratio = success_count / usage_denom

    # 时间衰减（recency）
    node_step = getattr(node, "step", 0) or 0
    step_diff = max(0, current_step - node_step)
    recency = math.exp(-lambda_ * step_diff)

    # 计算信息素
    pheromone = alpha * norm_score + beta * success_ratio + delta * recency
    return float(pheromone)
