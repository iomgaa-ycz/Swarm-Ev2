"""GeneRegistry 模块单元测试。"""

import pytest
from core.state import Node
from core.evolution.gene_registry import (
    GeneRegistry,
    normalize_gene_text,
    compute_gene_id,
)


class TestNormalizeGeneText:
    """测试基因文本归一化。"""

    def test_normalize_gene_text_basic(self):
        """测试基本归一化。"""
        text = "  line1  \n  line2  \n\n\n  line3  "
        normalized = normalize_gene_text(text)

        # 保留前导空格（用于代码缩进），去除尾部空格
        assert normalized == "line1\n  line2\n\n  line3"

    def test_normalize_gene_text_windows_newlines(self):
        """测试 Windows 风格换行符转换。"""
        text = "line1\r\nline2\r\nline3"
        normalized = normalize_gene_text(text)

        assert normalized == "line1\nline2\nline3"

    def test_normalize_gene_text_multiple_blank_lines(self):
        """测试多个连续空行合并为单个空行。"""
        text = "line1\n\n\n\nline2"
        normalized = normalize_gene_text(text)

        assert normalized == "line1\n\nline2"

    def test_normalize_gene_text_none_input(self):
        """测试 None 输入返回空字符串。"""
        normalized = normalize_gene_text(None)

        assert normalized == ""


class TestComputeGeneId:
    """测试基因 ID 计算。"""

    def test_compute_gene_id_stable(self):
        """测试相同文本生成相同 ID。"""
        text = "def function():\n    pass"
        id1 = compute_gene_id(text)
        id2 = compute_gene_id(text)

        assert id1 == id2
        assert len(id1) == 12  # SHA1 前 12 位

    def test_compute_gene_id_normalized(self):
        """测试归一化后的文本生成相同 ID。"""
        text1 = "def function():\n    pass  "
        text2 = "def function():\n    pass"

        id1 = compute_gene_id(text1)
        id2 = compute_gene_id(text2)

        assert id1 == id2  # 尾部空格被忽略

    def test_compute_gene_id_different(self):
        """测试不同文本生成不同 ID。"""
        text1 = "def function1():\n    pass"
        text2 = "def function2():\n    pass"

        id1 = compute_gene_id(text1)
        id2 = compute_gene_id(text2)

        assert id1 != id2


class TestGeneRegistry:
    """测试 GeneRegistry 类。"""

    def test_init(self):
        """测试初始化。"""
        registry = GeneRegistry()

        assert len(registry._registry) == 7  # 7 个基因位点
        assert "DATA" in registry._registry
        assert "MODEL" in registry._registry

    def test_update_from_reviewed_node_basic(self):
        """测试从节点更新基因信息素。"""
        registry = GeneRegistry()
        node = Node(id="node1", code="test", step=5)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {"DATA": "data code", "MODEL": "model code"}
        node.metadata = {"pheromone_node": 0.7}

        registry.update_from_reviewed_node(node)

        # 验证基因被注册
        data_gene_id = compute_gene_id("data code")
        model_gene_id = compute_gene_id("model code")

        assert data_gene_id in registry._registry["DATA"]
        assert model_gene_id in registry._registry["MODEL"]

        # 验证信息素值
        data_entry = registry._registry["DATA"][data_gene_id]
        assert data_entry["pheromone"] == 0.7
        assert data_entry["count"] == 1
        assert data_entry["last_seen_step"] == 5

    def test_update_from_reviewed_node_accumulate(self):
        """测试累积平均信息素。"""
        registry = GeneRegistry()

        # 第一次更新
        node1 = Node(id="node1", code="test", step=1)
        node1.metric_value = 0.8
        node1.is_buggy = False
        node1.genes = {"DATA": "data code"}
        node1.metadata = {"pheromone_node": 0.6}
        registry.update_from_reviewed_node(node1)

        # 第二次更新（相同基因）
        node2 = Node(id="node2", code="test", step=2)
        node2.metric_value = 0.9
        node2.is_buggy = False
        node2.genes = {"DATA": "data code"}
        node2.metadata = {"pheromone_node": 0.8}
        registry.update_from_reviewed_node(node2)

        # 验证累积平均
        gene_id = compute_gene_id("data code")
        entry = registry._registry["DATA"][gene_id]

        assert entry["count"] == 2
        assert entry["pheromone"] == pytest.approx((0.6 + 0.8) / 2)
        assert entry["last_seen_step"] == 2  # 更新为最新步骤

    def test_update_from_reviewed_node_skip_buggy(self):
        """测试跳过 buggy 节点。"""
        registry = GeneRegistry()
        node = Node(id="node1", code="test", step=5)
        node.metric_value = None
        node.is_buggy = True
        node.genes = {"DATA": "data code"}
        node.metadata = {"pheromone_node": 0.7}

        registry.update_from_reviewed_node(node)

        # 验证基因未被注册
        gene_id = compute_gene_id("data code")
        assert gene_id not in registry._registry["DATA"]

    def test_update_from_reviewed_node_no_pheromone(self):
        """测试无信息素值的节点被跳过。"""
        registry = GeneRegistry()
        node = Node(id="node1", code="test", step=5)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {"DATA": "data code"}
        node.metadata = {}

        registry.update_from_reviewed_node(node)

        # 验证基因未被注册
        gene_id = compute_gene_id("data code")
        assert gene_id not in registry._registry["DATA"]

    def test_get_gene_pheromone_exists(self):
        """测试获取已存在基因的信息素。"""
        registry = GeneRegistry()
        node = Node(id="node1", code="test", step=5)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {"DATA": "data code"}
        node.metadata = {"pheromone_node": 0.7}
        registry.update_from_reviewed_node(node)

        gene_id = compute_gene_id("data code")
        pheromone = registry.get_gene_pheromone("DATA", gene_id, current_step=5)

        assert pheromone == pytest.approx(0.7)

    def test_get_gene_pheromone_not_exists(self):
        """测试获取不存在基因的信息素返回默认值。"""
        registry = GeneRegistry()

        pheromone = registry.get_gene_pheromone(
            "DATA", "nonexistent", default_init=0.5
        )

        assert pheromone == 0.5

    def test_get_gene_pheromone_time_decay(self):
        """测试时间衰减机制。"""
        registry = GeneRegistry()
        node = Node(id="node1", code="test", step=0)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {"DATA": "data code"}
        node.metadata = {"pheromone_node": 1.0}
        registry.update_from_reviewed_node(node)

        gene_id = compute_gene_id("data code")

        # 立即获取（无衰减）
        pheromone_now = registry.get_gene_pheromone("DATA", gene_id, current_step=0)
        assert pheromone_now == pytest.approx(1.0)

        # 10 步后（有衰减）
        pheromone_later = registry.get_gene_pheromone("DATA", gene_id, current_step=10)
        import math

        expected_decay = math.exp(-0.03 * 10)
        assert pheromone_later == pytest.approx(1.0 * expected_decay)

    def test_build_gene_pools(self):
        """测试构建基因池。"""
        registry = GeneRegistry()
        node = Node(id="node1", code="test", step=5)
        node.metric_value = 0.8
        node.is_buggy = False
        node.genes = {"DATA": "data code", "MODEL": "model code"}
        node.metadata = {"pheromone_node": 0.7}
        registry.update_from_reviewed_node(node)

        pools = registry.build_gene_pools()

        assert len(pools) == 7
        assert len(pools["DATA"]) == 1
        assert len(pools["MODEL"]) == 1
        assert len(pools["LOSS"]) == 0

        # 验证基因池内容
        data_gene = pools["DATA"][0]
        assert data_gene["content"] == "data code"
        assert data_gene["pheromone"] == 0.7
        assert data_gene["source_node_id"] == "node1"
