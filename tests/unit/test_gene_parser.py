"""
select_mutation_target 单元测试。

测试目标：
- stub 过滤逻辑（内容 strip 后 < 20 字符视为 stub）
- 只有一个非 stub 基因时必须选中它
- 所有基因均为 stub 时 fallback 到 REQUIRED_GENES 随机选（不抛异常）
- 多个非 stub 基因时所有候选均可被选中
- 返回 (gene, aspect) 元组，aspect 属于 GENE_SUB_ASPECTS[gene]
"""

import pytest
from core.state.node import Node
from core.evolution.gene_parser import (
    select_mutation_target,
    REQUIRED_GENES,
    GENE_SUB_ASPECTS,
)


def _make_node(genes: dict) -> Node:
    """构造含指定基因的 Node。"""
    node = Node(code="pass")
    node.genes = genes
    return node


class TestSelectMutationTarget:

    def test_single_non_stub_always_selected(self):
        """只有 DATA 是非 stub，每次都应选 DATA。"""
        genes = {g: "# Not applicable" for g in REQUIRED_GENES}
        genes["DATA"] = "import pandas as pd\ndf = pd.read_csv('./input/train.csv')"
        node = _make_node(genes)

        for _ in range(30):
            gene, aspect = select_mutation_target(node)
            assert gene == "DATA", f"期望 DATA，实际 {gene}"
            assert aspect in GENE_SUB_ASPECTS["DATA"]

    def test_all_stub_falls_back_without_error(self):
        """所有基因均为 stub 时，fallback 随机选一个，不抛异常。"""
        node = _make_node({g: "# Not applicable" for g in REQUIRED_GENES})
        gene, aspect = select_mutation_target(node)
        assert gene in REQUIRED_GENES
        assert aspect in GENE_SUB_ASPECTS[gene]

    def test_stub_threshold_exactly_20_is_not_stub(self):
        """恰好 20 个字符的内容不是 stub（阈值 >= 20）。"""
        genes = {g: "# short" for g in REQUIRED_GENES}
        genes["MODEL"] = "x" * 20  # 恰好 20，不是 stub
        node = _make_node(genes)

        for _ in range(20):
            gene, aspect = select_mutation_target(node)
            assert gene == "MODEL"
            assert aspect in GENE_SUB_ASPECTS["MODEL"]

    def test_stub_threshold_19_is_stub(self):
        """19 个字符的内容是 stub，不会被选中（有其他非 stub 时）。"""
        genes = {g: "# short" for g in REQUIRED_GENES}
        genes["MODEL"] = "x" * 19   # stub
        genes["DATA"] = "x" * 30    # 非 stub
        node = _make_node(genes)

        for _ in range(20):
            gene, aspect = select_mutation_target(node)
            assert gene == "DATA", f"MODEL 是 stub 不应被选中，实际 {gene}"

    def test_multiple_non_stub_all_reachable(self):
        """多个非 stub 基因时，随机性保证所有候选至少被选过一次。"""
        genes = {g: "# Not applicable" for g in REQUIRED_GENES}
        genes["DATA"] = "import pandas as pd\ndf = pd.read_csv('./input/train.csv')"
        genes["MODEL"] = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()"
        node = _make_node(genes)

        selected_genes = {select_mutation_target(node)[0] for _ in range(200)}
        assert "DATA" in selected_genes
        assert "MODEL" in selected_genes
        # stub 基因不应出现
        for stub in ["TRAIN", "POSTPROCESS"]:
            assert stub not in selected_genes, f"stub 基因 {stub} 被错误选中"

    def test_empty_genes_falls_back(self):
        """节点无任何基因时，fallback 到 REQUIRED_GENES 随机选。"""
        node = _make_node({})
        gene, aspect = select_mutation_target(node)
        assert gene in REQUIRED_GENES
        assert aspect in GENE_SUB_ASPECTS[gene]

    def test_none_gene_value_treated_as_stub(self):
        """基因值为 None 时视为 stub（不抛 AttributeError）。"""
        genes = {g: None for g in REQUIRED_GENES}
        genes["DATA"] = "import pandas as pd\ndf = pd.read_csv('./input/train.csv')"
        node = _make_node(genes)

        for _ in range(20):
            gene, aspect = select_mutation_target(node)
            assert gene == "DATA"

    def test_returns_tuple(self):
        """返回值必须是 (gene, aspect) 元组。"""
        genes = {g: "x" * 30 for g in REQUIRED_GENES}
        node = _make_node(genes)
        result = select_mutation_target(node)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_all_aspects_reachable(self):
        """对单一非 stub 基因，所有 sub-aspect 均应可达。"""
        genes = {g: "# short" for g in REQUIRED_GENES}
        genes["MODEL"] = "x" * 30
        node = _make_node(genes)

        aspects_seen = {select_mutation_target(node)[1] for _ in range(300)}
        for expected_aspect in GENE_SUB_ASPECTS["MODEL"]:
            assert expected_aspect in aspects_seen, f"aspect {expected_aspect} 未被选到"
