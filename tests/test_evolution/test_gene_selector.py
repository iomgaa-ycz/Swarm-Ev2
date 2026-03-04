"""GeneSelector 模块单元测试。"""

import pytest
from unittest.mock import MagicMock
from core.state import Node, Journal
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_selector import (
    select_gene_plan,
    build_decision_gene_pools,
    get_primary_parent,
    pheromone_with_degenerate_check,
    _select_locus_winner,
    _compute_quality,
    _is_valid_node,
    _is_better_item,
    GeneItem,
    LOCUS_TO_FIELD,
)


class TestIsValidNode:
    """测试节点有效性检查。"""

    def test_valid_node(self):
        """测试有效节点。"""
        node = Node(id="test", code="code", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {"DATA": "data code"}
        node.metadata = {}

        assert _is_valid_node(node) is True

    def test_invalid_node_no_metric(self):
        """测试无 metric 的节点无效。"""
        node = Node(id="test", code="code", step=0)
        node.metric_value = None
        node.is_buggy = False
        node.genes = {"DATA": "data code"}

        assert _is_valid_node(node) is False

    def test_invalid_node_buggy(self):
        """测试 buggy 节点无效。"""
        node = Node(id="test", code="code", step=0)
        node.metric_value = 0.8
        node.is_buggy = True
        node.genes = {"DATA": "data code"}

        assert _is_valid_node(node) is False

    def test_invalid_node_no_code(self):
        """测试无代码的节点无效。"""
        node = Node(id="test", code="", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {"DATA": "data code"}

        assert _is_valid_node(node) is False

    def test_invalid_node_no_genes(self):
        """测试无基因的节点无效。"""
        node = Node(id="test", code="code", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = None

        assert _is_valid_node(node) is False


class TestComputeQuality:
    """测试质量计算。"""

    def test_compute_quality_basic(self):
        """测试基本质量计算。"""
        item = GeneItem(
            locus="DATA",
            gene_id="abc123",
            content="code",
            raw_content="code",
            source_node_id="node1",
            gene_pheromone=0.6,
            node_pheromone=0.8,
            source_score=0.9,
            created_step=5,
        )

        quality = _compute_quality(item)

        # 0.3 * 0.6 + 0.7 * 0.8 = 0.18 + 0.56 = 0.74
        assert quality == pytest.approx(0.74)

    def test_compute_quality_zero_pheromones(self):
        """测试零信息素的情况。"""
        item = GeneItem(
            locus="DATA",
            gene_id="abc123",
            content="code",
            raw_content="code",
            source_node_id="node1",
            gene_pheromone=0.0,
            node_pheromone=0.0,
            source_score=0.0,
            created_step=0,
        )

        quality = _compute_quality(item)

        assert quality == 0.0


class TestIsBetterItem:
    """测试基因项比较。"""

    def test_is_better_item_gene_pheromone(self):
        """测试基因信息素优先级最高。"""
        candidate = GeneItem(
            locus="DATA",
            gene_id="abc1",
            content="code",
            raw_content="code",
            source_node_id="node1",
            gene_pheromone=0.8,
            node_pheromone=0.5,
            source_score=0.5,
            created_step=5,
        )
        incumbent = GeneItem(
            locus="DATA",
            gene_id="abc2",
            content="code",
            raw_content="code",
            source_node_id="node2",
            gene_pheromone=0.6,
            node_pheromone=0.9,
            source_score=0.9,
            created_step=1,
        )

        assert _is_better_item(candidate, incumbent) is True

    def test_is_better_item_node_pheromone(self):
        """测试节点信息素次优先级。"""
        candidate = GeneItem(
            locus="DATA",
            gene_id="abc1",
            content="code",
            raw_content="code",
            source_node_id="node1",
            gene_pheromone=0.7,
            node_pheromone=0.8,
            source_score=0.5,
            created_step=5,
        )
        incumbent = GeneItem(
            locus="DATA",
            gene_id="abc2",
            content="code",
            raw_content="code",
            source_node_id="node2",
            gene_pheromone=0.7,
            node_pheromone=0.6,
            source_score=0.9,
            created_step=1,
        )

        assert _is_better_item(candidate, incumbent) is True


class TestSelectLocusWinner:
    """测试单个位点的获胜者选择。"""

    def test_select_locus_winner_basic(self):
        """测试基本选择逻辑。"""
        items = [
            GeneItem(
                locus="DATA",
                gene_id="abc1",
                content="code1",
                raw_content="code1",
                source_node_id="node1",
                gene_pheromone=0.5,
                node_pheromone=0.6,
                source_score=0.7,
                created_step=5,
                quality=0.57,  # 0.3 * 0.5 + 0.7 * 0.6
            ),
            GeneItem(
                locus="DATA",
                gene_id="abc2",
                content="code2",
                raw_content="code2",
                source_node_id="node2",
                gene_pheromone=0.7,
                node_pheromone=0.8,
                source_score=0.9,
                created_step=3,
                quality=0.77,  # 0.3 * 0.7 + 0.7 * 0.8
            ),
        ]

        winner = _select_locus_winner(items)

        assert winner.gene_id == "abc2"  # 质量最高
        assert winner.quality == pytest.approx(0.77)


class TestBuildDecisionGenePools:
    """测试基因池构建。"""

    def test_build_decision_gene_pools_basic(self):
        """测试基本基因池构建。"""
        journal = Journal()
        gene_registry = GeneRegistry()

        # 创建有效节点
        node1 = Node(id="node1", code="code1", step=0)
        node1.metric_value = 0.8
        node1.is_buggy = False
        node1.genes = {"DATA": "data code 1", "MODEL": "model code 1"}
        node1.metadata = {"pheromone_node": 0.7}
        journal.append(node1)

        node2 = Node(id="node2", code="code2", step=1)
        node2.metric_value = 0.9
        node2.is_buggy = False
        node2.genes = {"DATA": "data code 2", "MODEL": "model code 1"}  # 相同 MODEL
        node2.metadata = {"pheromone_node": 0.8}
        journal.append(node2)

        # 更新基因注册表
        gene_registry.update_from_reviewed_node(node1)
        gene_registry.update_from_reviewed_node(node2)

        pools = build_decision_gene_pools(journal, gene_registry, current_step=2)

        # 验证基因池
        assert len(pools["DATA"]) == 2  # 2 个不同的 DATA 基因
        assert len(pools["MODEL"]) == 1  # 1 个 MODEL 基因（去重）
        assert len(pools["TRAIN"]) == 0  # 无 TRAIN 基因

    def test_build_decision_gene_pools_skip_invalid(self):
        """测试跳过无效节点。"""
        journal = Journal()
        gene_registry = GeneRegistry()

        # 创建 buggy 节点
        node1 = Node(id="node1", code="code1", step=0)
        node1.metric_value = None
        node1.is_buggy = True
        node1.genes = {"DATA": "data code 1"}
        node1.metadata = {"pheromone_node": 0.7}
        journal.append(node1)

        # 创建有效节点
        node2 = Node(id="node2", code="code2", step=1)
        node2.metric_value = 0.9
        node2.is_buggy = False
        node2.genes = {"DATA": "data code 2"}
        node2.metadata = {"pheromone_node": 0.8}
        journal.append(node2)

        gene_registry.update_from_reviewed_node(node2)

        pools = build_decision_gene_pools(journal, gene_registry, current_step=2)

        # 只有有效节点的基因被包含
        assert len(pools["DATA"]) == 1


class TestSelectGenePlan:
    """测试完整的基因计划选择。"""

    def test_select_gene_plan_basic(self):
        """测试基本基因计划选择。"""
        journal = Journal()
        gene_registry = GeneRegistry()

        # 创建节点（包含所有 4 个基因）
        node1 = Node(id="node1", code="code1", step=0)
        node1.metric_value = 0.8
        node1.is_buggy = False
        node1.genes = {
            "DATA": "data code 1",
            "MODEL": "model code 1",
            "TRAIN": "train code 1",
            "POSTPROCESS": "postprocess code 1",
        }
        node1.metadata = {"pheromone_node": 0.7}
        journal.append(node1)

        gene_registry.update_from_reviewed_node(node1)

        gene_plan = select_gene_plan(journal, gene_registry, current_step=1)

        # 验证基因计划结构
        assert "reasoning" in gene_plan
        assert "data_source" in gene_plan
        assert "model_source" in gene_plan
        assert "train_source" in gene_plan
        assert "postprocess_source" in gene_plan

        # 验证单个基因来源
        assert gene_plan["data_source"]["locus"] == "DATA"
        assert gene_plan["data_source"]["source_node_id"] == "node1"
        assert gene_plan["data_source"]["code"] == "data code 1"

    def test_select_gene_plan_missing_locus(self):
        """测试缺少基因位点时抛出异常。"""
        journal = Journal()
        gene_registry = GeneRegistry()

        # 创建节点（缺少 MODEL 基因）
        node1 = Node(id="node1", code="code1", step=0)
        node1.metric_value = 0.8
        node1.is_buggy = False
        node1.genes = {
            "DATA": "data code 1",
            # 缺少其他基因
        }
        node1.metadata = {"pheromone_node": 0.7}
        journal.append(node1)

        gene_registry.update_from_reviewed_node(node1)

        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="Missing locus"):
            select_gene_plan(journal, gene_registry, current_step=1)


# ============================================================
# 以下测试迁移自 tests/unit/test_gene_selector.py
# ============================================================


def _make_selector_node(node_id: str, metric: float | None = 0.5) -> Node:
    """创建带 metric 的 Node。"""
    node = Node(code="pass")
    node.id = node_id
    node.metric_value = metric
    return node


def _make_selector_journal(*nodes: Node) -> Journal:
    """构造只包含指定节点的 Journal mock。"""
    journal = MagicMock(spec=Journal)
    journal.nodes = list(nodes)
    journal.get_node_by_id = lambda nid: next((n for n in nodes if n.id == nid), None)
    return journal


def _gene_plan_from_sources(sources: dict[str, str]) -> dict:
    """将 {locus: node_id} 映射转为 gene_plan 格式。"""
    plan = {}
    for locus, field_name in LOCUS_TO_FIELD.items():
        plan[field_name] = {
            "locus": locus,
            "source_node_id": sources.get(locus, "default_node"),
            "code": f"# {locus} code",
            "source_score": 0.8,
        }
    return plan


class TestGetPrimaryParent:
    """get_primary_parent 单元测试。"""

    def test_single_dominant_source(self):
        """4 基因全来自同一节点，直接返回该节点。"""
        node_a = _make_selector_node("aaaa1111", 0.9)
        journal = _make_selector_journal(node_a)
        gene_plan = _gene_plan_from_sources({locus: "aaaa1111" for locus in LOCUS_TO_FIELD})

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa1111"

    def test_majority_wins_3_vs_1(self):
        """A 贡献 3 个基因，B 贡献 1 个，A 胜出。"""
        node_a = _make_selector_node("aaaa1111", 0.5)
        node_b = _make_selector_node("bbbb2222", 0.9)
        journal = _make_selector_journal(node_a, node_b)

        loci = list(LOCUS_TO_FIELD.keys())
        sources = {loci[i]: "aaaa1111" if i < 3 else "bbbb2222" for i in range(4)}
        gene_plan = _gene_plan_from_sources(sources)

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa1111"

    def test_tie_broken_by_higher_metric(self):
        """并列时选 metric 最高的节点。"""
        node_low = _make_selector_node("low_11111", metric=0.3)
        node_high = _make_selector_node("high_2222", metric=0.8)
        journal = _make_selector_journal(node_low, node_high)

        loci = list(LOCUS_TO_FIELD.keys())
        sources = {}
        for i, locus in enumerate(loci):
            sources[locus] = "low_11111" if i < 2 else "high_2222"
        gene_plan = _gene_plan_from_sources(sources)

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "high_2222"

    def test_tie_broken_by_lower_metric_when_lower_is_better(self):
        """RMSE 竞赛并列时选 metric 最小的节点。"""
        node_low = _make_selector_node("low_11111", metric=0.05)
        node_low.lower_is_better = True
        node_high = _make_selector_node("high_2222", metric=0.30)
        node_high.lower_is_better = True
        journal = _make_selector_journal(node_low, node_high)

        loci = list(LOCUS_TO_FIELD.keys())
        sources = {}
        for i, locus in enumerate(loci):
            sources[locus] = "low_11111" if i < 2 else "high_2222"
        gene_plan = _gene_plan_from_sources(sources)

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "low_11111"

    def test_raises_on_empty_plan(self):
        """gene_plan 为空时抛出 ValueError。"""
        journal = _make_selector_journal()
        with pytest.raises(ValueError, match="source_node_id"):
            get_primary_parent({}, journal)

    def test_raises_on_plan_without_source_node_id(self):
        """gene_plan 值中无 source_node_id 时抛出 ValueError。"""
        journal = _make_selector_journal()
        bad_plan = {"field_data": {"locus": "DATA", "code": "..."}}
        with pytest.raises(ValueError):
            get_primary_parent(bad_plan, journal)

    def test_metric_none_fallback(self):
        """节点 metric 全为 None 时不崩溃。"""
        node_a = _make_selector_node("aaaa1111", metric=None)
        node_b = _make_selector_node("bbbb2222", metric=None)
        journal = _make_selector_journal(node_a, node_b)

        loci = list(LOCUS_TO_FIELD.keys())
        sources = {loci[i]: "aaaa1111" if i < 3 else "bbbb2222" for i in range(4)}
        gene_plan = _gene_plan_from_sources(sources)

        result = get_primary_parent(gene_plan, journal)
        assert result.id == "aaaa1111"


# ============================================================
# 以下测试迁移自 tests/unit/test_gene_selector_direction.py
# ============================================================


def _make_direction_item(
    gene_id: str = "g1",
    source_score: float = 0.5,
    lower_is_better: bool = False,
    node_pheromone: float = 0.5,
    gene_pheromone: float = 0.5,
    created_step: int = 0,
    source_node_id: str = "node_a",
) -> GeneItem:
    """构造 GeneItem。"""
    item = GeneItem(
        locus="DATA",
        gene_id=gene_id,
        content="normalized",
        raw_content="raw",
        source_node_id=source_node_id,
        gene_pheromone=gene_pheromone,
        node_pheromone=node_pheromone,
        source_score=source_score,
        lower_is_better=lower_is_better,
        created_step=created_step,
    )
    item.quality = 0.3 * gene_pheromone + 0.7 * node_pheromone
    return item


class TestIsBetterItemDirection:
    """_is_better_item 方向测试。"""

    def test_higher_is_better_larger_score_wins(self):
        """higher_is_better 时 source_score 大者胜出。"""
        candidate = _make_direction_item(gene_id="c", source_score=0.9)
        incumbent = _make_direction_item(gene_id="i", source_score=0.3)
        assert _is_better_item(candidate, incumbent) is True

    def test_higher_is_better_smaller_score_loses(self):
        """higher_is_better 时 source_score 小者不胜。"""
        candidate = _make_direction_item(gene_id="c", source_score=0.3)
        incumbent = _make_direction_item(gene_id="i", source_score=0.9)
        assert _is_better_item(candidate, incumbent) is False

    def test_lower_is_better_smaller_score_wins(self):
        """lower_is_better 时 source_score 小者胜出。"""
        candidate = _make_direction_item(gene_id="c", source_score=0.05, lower_is_better=True)
        incumbent = _make_direction_item(gene_id="i", source_score=0.30, lower_is_better=True)
        assert _is_better_item(candidate, incumbent) is True

    def test_lower_is_better_larger_score_loses(self):
        """lower_is_better 时 source_score 大者不胜。"""
        candidate = _make_direction_item(gene_id="c", source_score=0.30, lower_is_better=True)
        incumbent = _make_direction_item(gene_id="i", source_score=0.05, lower_is_better=True)
        assert _is_better_item(candidate, incumbent) is False


class TestSelectLocusWinnerDirection:
    """_select_locus_winner 方向测试。"""

    def test_lower_is_better_small_score_preferred(self):
        """RMSE 场景：source_score 小者优先。"""
        item_good = _make_direction_item(
            gene_id="good", source_score=0.05, lower_is_better=True, source_node_id="node_good",
        )
        item_bad = _make_direction_item(
            gene_id="bad", source_score=0.30, lower_is_better=True, source_node_id="node_bad",
        )
        winner = _select_locus_winner([item_bad, item_good])
        assert winner.gene_id == "good"

    def test_higher_is_better_large_score_preferred(self):
        """AUC 场景：source_score 大者优先。"""
        item_good = _make_direction_item(
            gene_id="good", source_score=0.9, source_node_id="node_good",
        )
        item_bad = _make_direction_item(
            gene_id="bad", source_score=0.3, source_node_id="node_bad",
        )
        winner = _select_locus_winner([item_bad, item_good])
        assert winner.gene_id == "good"


class TestPheromoneWithDegenerateCheck:
    """pheromone_with_degenerate_check() 退化检测测试。"""

    def test_non_degenerate_returns_original(self):
        """多来源时直接返回原始计划。"""
        journal = Journal()
        gene_registry = GeneRegistry()

        # 创建两个有效节点，各有不同基因
        node1 = Node(id="node1", code="code1", step=0)
        node1.metric_value = 0.8
        node1.is_buggy = False
        node1.genes = {
            "DATA": "data code 1",
            "MODEL": "model code 1",
            "TRAIN": "train code 1",
            "POSTPROCESS": "post code 1",
        }
        node1.metadata = {"pheromone_node": 0.9}
        journal.append(node1)

        node2 = Node(id="node2", code="code2", step=1)
        node2.metric_value = 0.9
        node2.is_buggy = False
        node2.genes = {
            "DATA": "data code 2 different",
            "MODEL": "model code 2 different",
            "TRAIN": "train code 2 different",
            "POSTPROCESS": "post code 2 different",
        }
        node2.metadata = {"pheromone_node": 0.95}
        journal.append(node2)

        gene_registry.update_from_reviewed_node(node1)
        gene_registry.update_from_reviewed_node(node2)

        result = pheromone_with_degenerate_check(journal, gene_registry, current_step=2)

        # 应该有有效的 gene_plan 结构
        assert isinstance(result, dict)
        for field_name in LOCUS_TO_FIELD.values():
            assert field_name in result
            assert "source_node_id" in result[field_name]

    def test_degenerate_triggers_replacement(self):
        """所有基因来自同一节点时触发退化替换。"""
        journal = Journal()
        gene_registry = GeneRegistry()

        # node1 的 pheromone 极高 -> select_gene_plan 会全选 node1
        node1 = Node(id="dominant_node", code="code1", step=0)
        node1.metric_value = 0.99
        node1.is_buggy = False
        node1.genes = {
            "DATA": "data code 1",
            "MODEL": "model code 1",
            "TRAIN": "train code 1",
            "POSTPROCESS": "post code 1",
        }
        node1.metadata = {"pheromone_node": 1.0}
        journal.append(node1)

        # node2 的 pheromone 较低但存在
        node2 = Node(id="alternative_node", code="code2", step=1)
        node2.metric_value = 0.3
        node2.is_buggy = False
        node2.genes = {
            "DATA": "data code 2 alt",
            "MODEL": "model code 2 alt",
            "TRAIN": "train code 2 alt",
            "POSTPROCESS": "post code 2 alt",
        }
        node2.metadata = {"pheromone_node": 0.1}
        journal.append(node2)

        gene_registry.update_from_reviewed_node(node1)
        gene_registry.update_from_reviewed_node(node2)

        result = pheromone_with_degenerate_check(journal, gene_registry, current_step=2)

        # 结果应该包含有效的 gene_plan
        assert isinstance(result, dict)
        for field_name in LOCUS_TO_FIELD.values():
            assert field_name in result


class TestGetPrimaryParentFallback:
    """get_primary_parent 的边界路径测试。"""

    def test_node_not_found_raises(self):
        """source_node_id 在 journal 中不存在时抛出 ValueError。"""
        journal = _make_selector_journal()  # 空 journal
        gene_plan = _gene_plan_from_sources(
            {locus: "nonexistent_id" for locus in LOCUS_TO_FIELD}
        )

        with pytest.raises(ValueError, match="找不到节点"):
            get_primary_parent(gene_plan, journal)

    def test_tie_all_nodes_none_fallback(self):
        """并列且所有节点为 None 时使用 fallback。"""
        # 创建 mock journal 返回 None
        journal = MagicMock(spec=Journal)
        journal.get_node_by_id = MagicMock(return_value=None)

        loci = list(LOCUS_TO_FIELD.keys())
        sources = {loci[i]: "aaaa" if i < 2 else "bbbb" for i in range(4)}
        gene_plan = _gene_plan_from_sources(sources)

        # 所有 get_node_by_id 返回 None，最终 fallback 路径也是 None -> raises
        with pytest.raises((ValueError, AttributeError)):
            get_primary_parent(gene_plan, journal)
