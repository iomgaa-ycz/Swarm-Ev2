"""
select_non_stub_gene 单元测试。

测试目标：
- stub 过滤逻辑（内容 strip 后 < 20 字符视为 stub）
- 只有一个非 stub 基因时必须选中它
- 所有基因均为 stub 时 fallback 到 REQUIRED_GENES 随机选（不抛异常）
- 多个非 stub 基因时所有候选均可被选中
"""

import pytest
from core.state.node import Node
from core.evolution.gene_parser import select_non_stub_gene, REQUIRED_GENES


def _make_node(genes: dict) -> Node:
    """构造含指定基因的 Node。"""
    node = Node(code="pass")
    node.genes = genes
    return node


class TestSelectNonStubGene:

    def test_single_non_stub_always_selected(self):
        """只有 DATA 是非 stub，每次都应选 DATA。"""
        genes = {g: "# Not applicable" for g in REQUIRED_GENES}
        genes["DATA"] = "import pandas as pd\ndf = pd.read_csv('./input/train.csv')"
        node = _make_node(genes)

        for _ in range(30):
            result = select_non_stub_gene(node)
            assert result == "DATA", f"期望 DATA，实际 {result}"

    def test_all_stub_falls_back_without_error(self):
        """所有基因均为 stub 时，fallback 随机选一个，不抛异常。"""
        node = _make_node({g: "# Not applicable" for g in REQUIRED_GENES})
        result = select_non_stub_gene(node)
        assert result in REQUIRED_GENES

    def test_stub_threshold_exactly_20_is_not_stub(self):
        """恰好 20 个字符的内容不是 stub（阈值 >= 20）。"""
        genes = {g: "# short" for g in REQUIRED_GENES}
        genes["MODEL"] = "x" * 20  # 恰好 20，不是 stub
        node = _make_node(genes)

        for _ in range(20):
            assert select_non_stub_gene(node) == "MODEL"

    def test_stub_threshold_19_is_stub(self):
        """19 个字符的内容是 stub，不会被选中（有其他非 stub 时）。"""
        genes = {g: "# short" for g in REQUIRED_GENES}
        genes["MODEL"] = "x" * 19   # stub
        genes["DATA"] = "x" * 30    # 非 stub
        node = _make_node(genes)

        for _ in range(20):
            result = select_non_stub_gene(node)
            assert result == "DATA", f"MODEL 是 stub 不应被选中，实际 {result}"

    def test_multiple_non_stub_all_reachable(self):
        """多个非 stub 基因时，随机性保证所有候选至少被选过一次。"""
        genes = {g: "# Not applicable" for g in REQUIRED_GENES}
        genes["DATA"] = "import pandas as pd\ndf = pd.read_csv('./input/train.csv')"
        genes["MODEL"] = "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()"
        node = _make_node(genes)

        selected = {select_non_stub_gene(node) for _ in range(200)}
        assert "DATA" in selected
        assert "MODEL" in selected
        # stub 基因不应出现
        for stub in ["LOSS", "OPTIMIZER", "REGULARIZATION", "INITIALIZATION", "TRAINING_TRICKS"]:
            assert stub not in selected, f"stub 基因 {stub} 被错误选中"

    def test_empty_genes_falls_back(self):
        """节点无任何基因时，fallback 到 REQUIRED_GENES 随机选。"""
        node = _make_node({})
        result = select_non_stub_gene(node)
        assert result in REQUIRED_GENES

    def test_none_gene_value_treated_as_stub(self):
        """基因值为 None 时视为 stub（不抛 AttributeError）。"""
        genes = {g: None for g in REQUIRED_GENES}
        genes["DATA"] = "import pandas as pd\ndf = pd.read_csv('./input/train.csv')"
        node = _make_node(genes)

        for _ in range(20):
            assert select_non_stub_gene(node) == "DATA"
