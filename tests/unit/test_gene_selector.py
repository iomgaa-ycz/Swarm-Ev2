"""
get_primary_parent 单元测试。

测试目标：
- 单一主导节点（7 基因全来自同一节点）
- 多数节点胜出（贡献基因数最多者）
- 并列时选 metric_value 更高的节点
- 空 gene_plan 抛出 ValueError
- fallback：节点 metric 全为 None 时不崩溃
"""

import pytest
from unittest.mock import MagicMock

from core.state.node import Node
from core.state.journal import Journal
from core.evolution.gene_selector import get_primary_parent, LOCUS_TO_FIELD


# ── 辅助函数 ──────────────────────────────────────────────────

def _make_node(node_id: str, metric: float | None = 0.5) -> Node:
    node = Node(code="pass")
    node.id = node_id
    node.metric_value = metric
    return node


def _make_journal(*nodes: Node) -> Journal:
    """构造只包含指定节点的 Journal mock。"""
    journal = MagicMock(spec=Journal)
    journal.nodes = list(nodes)
    journal.get_node_by_id = lambda nid: next((n for n in nodes if n.id == nid), None)
    return journal


def _gene_plan_from_sources(sources: dict[str, str]) -> dict:
    """将 {locus: node_id} 映射转为 gene_plan 格式（模拟 pheromone 输出）。"""
    plan = {}
    for locus, field_name in LOCUS_TO_FIELD.items():
        plan[field_name] = {
            "locus": locus,
            "source_node_id": sources.get(locus, "default_node"),
            "code": f"# {locus} code",
            "source_score": 0.8,
        }
    return plan


# ── 测试类 ────────────────────────────────────────────────────

class TestGetPrimaryParent:

    def test_single_dominant_source(self):
        """7 基因全来自同一节点，直接返回该节点。"""
        node_a = _make_node("aaaa1111", 0.9)
        journal = _make_journal(node_a)
        gene_plan = _gene_plan_from_sources({locus: "aaaa1111" for locus in LOCUS_TO_FIELD})

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa1111"

    def test_majority_wins_4_vs_3(self):
        """A 贡献 4 个基因，B 贡献 3 个，A 胜出。"""
        node_a = _make_node("aaaa1111", 0.5)
        node_b = _make_node("bbbb2222", 0.9)
        journal = _make_journal(node_a, node_b)

        loci = list(LOCUS_TO_FIELD.keys())
        # 前 4 个位点来自 A，后 3 个来自 B
        sources = {loci[i]: "aaaa1111" if i < 4 else "bbbb2222" for i in range(7)}
        gene_plan = _gene_plan_from_sources(sources)

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa1111"

    def test_tie_broken_by_higher_metric(self):
        """并列时（各贡献 3 个基因，第 7 个给第三方），选 metric 最高的节点。

        注意：7 个基因中，让 A 和 B 各贡献 3 个，第 7 个来自 C（第三方）。
        A vs B 并列 3:3，应选 metric 更高的那个。
        """
        node_low  = _make_node("low_11111", metric=0.3)
        node_high = _make_node("high_2222", metric=0.8)
        node_c    = _make_node("cccc3333",  metric=0.1)
        journal   = _make_journal(node_low, node_high, node_c)

        loci = list(LOCUS_TO_FIELD.keys())
        # low_node 贡献 0,1,2；high_node 贡献 3,4,5；c 贡献 6
        sources = {}
        for i, locus in enumerate(loci):
            if i < 3:
                sources[locus] = "low_11111"
            elif i < 6:
                sources[locus] = "high_2222"
            else:
                sources[locus] = "cccc3333"
        gene_plan = _gene_plan_from_sources(sources)

        # high_node 和 low_node 各贡献 3 个，并列 → 选 metric 高的
        result = get_primary_parent(gene_plan, journal)
        assert result.id == "high_2222", (
            f"并列时应选 metric 更高的 high_2222，实际选了 {result.id}"
        )

    def test_raises_on_empty_plan(self):
        """gene_plan 为空时抛出 ValueError。"""
        journal = _make_journal()
        with pytest.raises(ValueError, match="source_node_id"):
            get_primary_parent({}, journal)

    def test_raises_on_plan_without_source_node_id(self):
        """gene_plan 值中无 source_node_id 时抛出 ValueError。"""
        journal = _make_journal()
        bad_plan = {"field_data": {"locus": "DATA", "code": "..."}}  # 缺 source_node_id
        with pytest.raises(ValueError):
            get_primary_parent(bad_plan, journal)

    def test_metric_none_fallback(self):
        """节点 metric 全为 None 时，并列不崩溃（使用 0.0 作为默认值）。"""
        node_a = _make_node("aaaa1111", metric=None)
        node_b = _make_node("bbbb2222", metric=None)
        journal = _make_journal(node_a, node_b)

        loci = list(LOCUS_TO_FIELD.keys())
        sources = {loci[i]: "aaaa1111" if i < 4 else "bbbb2222" for i in range(7)}
        gene_plan = _gene_plan_from_sources(sources)

        # A 贡献 4 个，直接胜出，不需要走 metric 比较
        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa1111"
