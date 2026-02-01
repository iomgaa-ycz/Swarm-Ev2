"""Pheromone 模块单元测试。"""

import pytest
from core.state import Node
from core.evolution.pheromone import (
    ensure_node_stats,
    compute_node_pheromone,
    _normalize_score,
)


class TestEnsureNodeStats:
    """测试节点统计字段初始化。"""

    def test_ensure_node_stats_empty_metadata(self):
        """测试空元数据的初始化。"""
        node = Node(id="test_node", code="test", step=0)
        node.metadata = None

        ensure_node_stats(node)

        assert node.metadata is not None
        assert node.metadata["usage_count"] == 0
        assert node.metadata["success_count"] == 0
        assert node.metadata["pheromone_node"] is None

    def test_ensure_node_stats_existing_metadata(self):
        """测试已有元数据的情况。"""
        node = Node(id="test_node", code="test", step=0)
        node.metadata = {"usage_count": 5, "success_count": 3}

        ensure_node_stats(node)

        assert node.metadata["usage_count"] == 5
        assert node.metadata["success_count"] == 3
        assert node.metadata["pheromone_node"] is None

    def test_ensure_node_stats_invalid_types(self):
        """测试无效类型的字段会被重置。"""
        node = Node(id="test_node", code="test", step=0)
        node.metadata = {"usage_count": "invalid", "success_count": None}

        ensure_node_stats(node)

        assert node.metadata["usage_count"] == 0
        assert node.metadata["success_count"] == 0


class TestNormalizeScore:
    """测试得分归一化。"""

    def test_normalize_score_valid_range(self):
        """测试正常得分归一化。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.75
        node.is_buggy = False

        normalized = _normalize_score(node, score_min=0.5, score_max=1.0)

        assert normalized == pytest.approx(0.5)  # (0.75 - 0.5) / (1.0 - 0.5)

    def test_normalize_score_buggy_node(self):
        """测试 buggy 节点返回 0.0。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.75
        node.is_buggy = True

        normalized = _normalize_score(node, score_min=0.5, score_max=1.0)

        assert normalized == 0.0

    def test_normalize_score_no_metric(self):
        """测试无 metric 的节点返回 0.0。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = None
        node.is_buggy = False

        normalized = _normalize_score(node, score_min=0.5, score_max=1.0)

        assert normalized == 0.0

    def test_normalize_score_no_range(self):
        """测试无得分范围时返回默认值 0.3。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.75
        node.is_buggy = False

        normalized = _normalize_score(node, score_min=None, score_max=None)

        assert normalized == 0.3

    def test_normalize_score_zero_range(self):
        """测试得分范围为 0 时返回默认值 0.3。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.75
        node.is_buggy = False

        normalized = _normalize_score(node, score_min=0.75, score_max=0.75)

        assert normalized == 0.3

    def test_normalize_score_clipping(self):
        """测试得分被截断到 [0, 1]。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 1.5
        node.is_buggy = False

        normalized = _normalize_score(node, score_min=0.0, score_max=1.0)

        assert normalized == 1.0  # 截断到 1.0


class TestComputeNodePheromone:
    """测试节点信息素计算。"""

    def test_compute_node_pheromone_basic(self):
        """测试基本信息素计算。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.metadata = {"usage_count": 10, "success_count": 8}

        pheromone = compute_node_pheromone(
            node,
            current_step=5,
            score_min=0.0,
            score_max=1.0,
            alpha=0.5,
            beta=0.3,
            delta=0.2,
            lambda_=0.05,
        )

        # 计算期望值
        norm_score = 0.8
        success_ratio = 8 / 10
        import math

        recency = math.exp(-0.05 * 5)
        expected = 0.5 * norm_score + 0.3 * success_ratio + 0.2 * recency

        assert pheromone == pytest.approx(expected, rel=1e-6)

    def test_compute_node_pheromone_no_usage(self):
        """测试无使用记录的节点。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.metadata = {"usage_count": 0, "success_count": 0}

        pheromone = compute_node_pheromone(
            node, current_step=5, score_min=0.0, score_max=1.0
        )

        # success_ratio = 0 / max(1, 0) = 0
        assert pheromone > 0.0  # 应该仍有正值（基于得分和时间衰减）

    def test_compute_node_pheromone_recent_node(self):
        """测试最近创建的节点（recency 高）。"""
        node = Node(id="test", code="test", step=10)
        node.metric_value = 0.8
        node.is_buggy = False
        node.metadata = {"usage_count": 0, "success_count": 0}

        pheromone = compute_node_pheromone(
            node, current_step=10, score_min=0.0, score_max=1.0
        )

        # step_diff = 0, recency = exp(0) = 1.0
        import math

        expected = 0.5 * 0.8 + 0.3 * 0.0 + 0.2 * 1.0

        assert pheromone == pytest.approx(expected, rel=1e-6)

    def test_compute_node_pheromone_old_node(self):
        """测试很久之前创建的节点（recency 低）。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.metadata = {"usage_count": 0, "success_count": 0}

        pheromone = compute_node_pheromone(
            node, current_step=100, score_min=0.0, score_max=1.0
        )

        # step_diff = 100, recency ≈ 0
        # pheromone ≈ 0.5 * 0.8 + 0.3 * 0.0 + 0.2 * ~0 ≈ 0.4
        assert pheromone < 0.5  # recency 贡献很小

    def test_compute_node_pheromone_none_metadata(self):
        """测试元数据为 None 的情况。"""
        node = Node(id="test", code="test", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.metadata = None

        pheromone = compute_node_pheromone(
            node, current_step=5, score_min=0.0, score_max=1.0
        )

        # 应该自动初始化 metadata 并计算
        assert pheromone > 0.0
        assert node.metadata is not None
