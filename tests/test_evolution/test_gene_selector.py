"""GeneSelector 模块单元测试。"""

import pytest
from core.state import Node, Journal
from core.evolution.gene_registry import GeneRegistry
from core.evolution.gene_selector import (
    select_gene_plan,
    build_decision_gene_pools,
    _select_locus_winner,
    _compute_quality,
    _is_valid_node,
    _is_better_item,
    GeneItem,
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
