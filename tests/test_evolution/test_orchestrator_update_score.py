"""Orchestrator update_score 集成测试。

验证 _finalize_node 后 task_dispatcher.update_score 被正确调用，
以及 _compute_agent_quality 的 percentile rank 计算，
以及 _write_experience_pool 使用归一化 quality。
"""

import pytest
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

from core.state.node import Node
from core.state.journal import Journal


def _make_node(
    metric_value: float = None,
    is_buggy: bool = False,
    lower_is_better: bool = False,
    node_id: str = "test",
) -> Node:
    """创建测试用 Node。"""
    return Node(
        id=node_id,
        code="# test stub",
        metric_value=metric_value,
        is_buggy=is_buggy,
        lower_is_better=lower_is_better,
    )


class TestComputeAgentQuality:
    """测试 _compute_agent_quality percentile rank 计算。"""

    def _make_orchestrator(self, journal_nodes: list):
        """创建带 journal 的 mock orchestrator，仅暴露 _compute_agent_quality。"""
        from core.orchestrator import Orchestrator

        # 直接构造最小可用的 orchestrator mock
        orch = object.__new__(Orchestrator)
        orch.journal = Journal()
        orch.journal_lock = MagicMock()
        orch.journal_lock.__enter__ = MagicMock(return_value=None)
        orch.journal_lock.__exit__ = MagicMock(return_value=False)

        for n in journal_nodes:
            orch.journal.nodes.append(n)

        return orch

    def test_single_node_returns_neutral(self):
        """仅一个节点时返回 0.5（中性）。"""
        node = _make_node(metric_value=0.8)
        orch = self._make_orchestrator([node])

        quality = orch._compute_agent_quality(node)
        assert quality == 0.5

    def test_higher_is_better_ranking(self):
        """higher_is_better: 击败越多节点质量越高。"""
        nodes = [
            _make_node(metric_value=0.3, node_id="a"),
            _make_node(metric_value=0.5, node_id="b"),
            _make_node(metric_value=0.7, node_id="c"),
            _make_node(metric_value=0.9, node_id="d"),  # 最好
        ]
        orch = self._make_orchestrator(nodes)

        # 最好的节点击败其他3个 → quality = 3/3 = 1.0
        assert orch._compute_agent_quality(nodes[3]) == pytest.approx(1.0)
        # 最差的节点击败0个 → quality = 0/3 = 0.0
        assert orch._compute_agent_quality(nodes[0]) == pytest.approx(0.0)
        # 中间节点击败1个 → quality = 1/3 ≈ 0.333
        assert orch._compute_agent_quality(nodes[1]) == pytest.approx(1 / 3)

    def test_lower_is_better_ranking(self):
        """lower_is_better: 越小越好。"""
        nodes = [
            _make_node(metric_value=0.06, lower_is_better=True, node_id="a"),  # 最好
            _make_node(metric_value=0.1, lower_is_better=True, node_id="b"),
            _make_node(metric_value=0.5, lower_is_better=True, node_id="c"),
        ]
        orch = self._make_orchestrator(nodes)

        # 最好（最小）节点击败2个 → quality = 2/2 = 1.0
        assert orch._compute_agent_quality(nodes[0]) == pytest.approx(1.0)
        # 最差（最大）节点击败0个 → quality = 0/2 = 0.0
        assert orch._compute_agent_quality(nodes[2]) == pytest.approx(0.0)

    def test_buggy_nodes_excluded(self):
        """buggy 节点不参与排名计算。"""
        nodes = [
            _make_node(metric_value=0.3, node_id="a"),
            _make_node(metric_value=None, is_buggy=True, node_id="buggy"),
            _make_node(metric_value=0.7, node_id="c"),
        ]
        orch = self._make_orchestrator(nodes)

        # 0.7 击败 0.3（buggy 不参与） → quality = 1/1 = 1.0
        assert orch._compute_agent_quality(nodes[2]) == pytest.approx(1.0)


class TestFinalizeNodeCallsUpdateScore:
    """验证 _finalize_node 正确调用 task_dispatcher.update_score。"""

    def test_update_score_called_on_good_node(self):
        """非 buggy 且有 metric 的节点应触发 update_score 和经验池写入。"""
        from core.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch.journal = Journal()
        orch.journal_lock = MagicMock()
        orch.journal_lock.__enter__ = MagicMock(return_value=None)
        orch.journal_lock.__exit__ = MagicMock(return_value=False)
        orch.save_lock = MagicMock()
        orch.save_lock.__enter__ = MagicMock(return_value=None)
        orch.save_lock.__exit__ = MagicMock(return_value=False)
        orch.experience_pool = MagicMock()
        orch.task_dispatcher = MagicMock()
        orch._global_lower_is_better = False

        node = _make_node(metric_value=0.8, node_id="good_node")
        agent = MagicMock()
        agent.name = "agent_0"
        context = MagicMock()

        # Mock 掉所有依赖方法
        orch._execute_code = MagicMock(return_value=MagicMock(
            term_out=[], exec_time=1.0, exc_type=None, exc_info=None
        ))
        orch._check_submission_and_set_error = MagicMock()
        orch._debug_chain = MagicMock(return_value=node)
        orch._review_node = MagicMock()
        orch._save_node_solution = MagicMock()
        orch._update_best_node = MagicMock()
        orch._compute_agent_quality = MagicMock(return_value=0.75)
        orch._write_experience_pool = MagicMock()

        orch._finalize_node(node, agent, context, "explore")

        # 验证 update_score 被调用
        orch.task_dispatcher.update_score.assert_called_once_with(
            "agent_0", "explore", 0.75
        )
        # 验证 _write_experience_pool 接收到 quality 参数
        orch._write_experience_pool.assert_called_once_with(
            "agent_0", "explore", node, 0.75
        )

    def test_update_score_not_called_on_buggy_node(self):
        """buggy 节点不应触发 update_score。"""
        from core.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch.journal = Journal()
        orch.journal_lock = MagicMock()
        orch.journal_lock.__enter__ = MagicMock(return_value=None)
        orch.journal_lock.__exit__ = MagicMock(return_value=False)
        orch.save_lock = MagicMock()
        orch.save_lock.__enter__ = MagicMock(return_value=None)
        orch.save_lock.__exit__ = MagicMock(return_value=False)
        orch.experience_pool = None
        orch.task_dispatcher = MagicMock()

        node = _make_node(metric_value=None, is_buggy=True, node_id="buggy_node")
        agent = MagicMock()
        agent.name = "agent_0"
        context = MagicMock()

        orch._execute_code = MagicMock(return_value=MagicMock(
            term_out=[], exec_time=1.0, exc_type=None, exc_info=None
        ))
        orch._check_submission_and_set_error = MagicMock()
        orch._debug_chain = MagicMock(return_value=node)
        orch._review_node = MagicMock()
        orch._save_node_solution = MagicMock()
        orch._update_best_node = MagicMock()

        orch._finalize_node(node, agent, context, "explore")

        # 验证 update_score 未被调用
        orch.task_dispatcher.update_score.assert_not_called()


class TestWriteExperiencePoolQuality:
    """验证 _write_experience_pool 使用归一化 quality。"""

    def _make_orchestrator_with_pool(self):
        """创建带 experience_pool 的 mock orchestrator。"""
        from core.orchestrator import Orchestrator

        orch = object.__new__(Orchestrator)
        orch.journal = Journal()
        orch.journal_lock = MagicMock()
        orch.journal_lock.__enter__ = MagicMock(return_value=None)
        orch.journal_lock.__exit__ = MagicMock(return_value=False)
        orch.experience_pool = MagicMock()
        return orch

    def test_write_experience_pool_uses_quality(self):
        """经验池写入的 output_quality 应为归一化值，而非原始 metric。"""
        orch = self._make_orchestrator_with_pool()
        node = _make_node(metric_value=0.06, node_id="node_with_small_metric")

        orch._write_experience_pool("agent_0", "explore", node, quality=0.75)

        # 验证 experience_pool.add 被调用，且 output_quality 是归一化值
        orch.experience_pool.add.assert_called_once()
        record = orch.experience_pool.add.call_args[0][0]
        assert record.output_quality == 0.75  # 归一化值，非原始 0.06

    def test_write_experience_pool_buggy_node_quality_zero(self):
        """buggy 节点（quality=None）写入经验池时 output_quality=0.0。"""
        orch = self._make_orchestrator_with_pool()
        node = _make_node(metric_value=None, is_buggy=True, node_id="buggy_node")

        orch._write_experience_pool("agent_0", "explore", node, quality=None)

        orch.experience_pool.add.assert_called_once()
        record = orch.experience_pool.add.call_args[0][0]
        assert record.output_quality == 0.0
