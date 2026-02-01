"""
Journal 数据类单元测试。
"""

from core.state import Node, Journal
from core.evolution.gene_parser import parse_solution_genes


class TestJournal:
    """Journal 类测试套件。"""

    def test_journal_append(self):
        """测试 Journal append 方法。"""
        journal = Journal()
        node1 = Node(code="x = 1")
        node2 = Node(code="x = 2")

        journal.append(node1)
        journal.append(node2)

        assert len(journal) == 2
        assert node1.step == 0
        assert node2.step == 1
        assert journal[0] == node1
        assert journal[1] == node2

    def test_journal_get_node_by_id(self):
        """测试 Journal get_node_by_id 方法。"""
        journal = Journal()
        node1 = Node(code="x = 1")
        node2 = Node(code="x = 2")
        journal.append(node1)
        journal.append(node2)

        found = journal.get_node_by_id(node1.id)
        assert found == node1

        not_found = journal.get_node_by_id("nonexistent_id")
        assert not_found is None

    def test_journal_get_children(self):
        """测试 Journal get_children 方法。"""
        journal = Journal()
        parent = Node(code="parent")
        child1 = Node(code="child1", parent_id=parent.id)
        child2 = Node(code="child2", parent_id=parent.id)

        journal.append(parent)
        journal.append(child1)
        journal.append(child2)
        journal.build_dag()

        children = journal.get_children(parent.id)
        assert len(children) == 2
        assert child1 in children
        assert child2 in children

    def test_journal_get_siblings(self):
        """测试 Journal get_siblings 方法。"""
        journal = Journal()
        parent = Node(code="parent")
        child1 = Node(code="child1", parent_id=parent.id)
        child2 = Node(code="child2", parent_id=parent.id)
        child3 = Node(code="child3", parent_id=parent.id)

        journal.append(parent)
        journal.append(child1)
        journal.append(child2)
        journal.append(child3)

        siblings = journal.get_siblings(child1.id)
        assert len(siblings) == 2
        assert child2 in siblings
        assert child3 in siblings
        assert child1 not in siblings

    def test_journal_draft_nodes(self):
        """测试 Journal draft_nodes 属性。"""
        journal = Journal()
        draft1 = Node(code="draft1")
        draft2 = Node(code="draft2")
        child = Node(code="child", parent_id=draft1.id)

        journal.append(draft1)
        journal.append(draft2)
        journal.append(child)

        drafts = journal.draft_nodes
        assert len(drafts) == 2
        assert draft1 in drafts
        assert draft2 in drafts
        assert child not in drafts

    def test_journal_buggy_and_good_nodes(self):
        """测试 Journal buggy_nodes 和 good_nodes 属性。"""
        journal = Journal()
        good1 = Node(code="good1", is_buggy=False)
        good2 = Node(code="good2", is_buggy=False)
        buggy1 = Node(code="buggy1", is_buggy=True)
        buggy2 = Node(code="buggy2", is_buggy=True)

        journal.append(good1)
        journal.append(buggy1)
        journal.append(good2)
        journal.append(buggy2)

        assert len(journal.good_nodes) == 2
        assert len(journal.buggy_nodes) == 2
        assert good1 in journal.good_nodes
        assert buggy1 in journal.buggy_nodes

    def test_journal_get_best_node(self):
        """测试 Journal get_best_node 方法。"""
        journal = Journal()
        node1 = Node(code="n1", metric_value=0.5, is_buggy=False)
        node2 = Node(code="n2", metric_value=0.8, is_buggy=False)
        node3 = Node(code="n3", metric_value=0.9, is_buggy=True)
        node4 = Node(code="n4", metric_value=None, is_buggy=False)

        journal.append(node1)
        journal.append(node2)
        journal.append(node3)
        journal.append(node4)

        # only_good=True（默认）
        best = journal.get_best_node(only_good=True)
        assert best == node2  # 最高非 buggy 节点

        # only_good=False
        best_all = journal.get_best_node(only_good=False)
        assert best_all == node3  # 最高节点（包括 buggy）

    def test_journal_build_dag(self):
        """测试 Journal build_dag 方法。"""
        journal = Journal()
        root = Node(code="root")
        child1 = Node(code="child1", parent_id=root.id)
        child2 = Node(code="child2", parent_id=root.id)
        grandchild = Node(code="grandchild", parent_id=child1.id)

        journal.append(root)
        journal.append(child1)
        journal.append(child2)
        journal.append(grandchild)

        # 初始状态 children_ids 为空
        assert root.children_ids == []

        # 构建 DAG
        journal.build_dag()

        # 验证 children_ids
        assert len(root.children_ids) == 2
        assert child1.id in root.children_ids
        assert child2.id in root.children_ids
        assert len(child1.children_ids) == 1
        assert grandchild.id in child1.children_ids

    def test_journal_serialization(self):
        """测试 Journal 序列化。"""
        journal = Journal()
        node1 = Node(code="x = 1")
        node2 = Node(code="x = 2", parent_id=node1.id)
        journal.append(node1)
        journal.append(node2)

        # 序列化
        json_dict = journal.to_dict()
        assert len(json_dict["nodes"]) == 2

        # 反序列化
        restored = Journal.from_dict(json_dict)
        assert len(restored.nodes) == 2
        assert restored[0].code == "x = 1"
        assert restored[1].code == "x = 2"

    def test_parse_solution_genes(self):
        """测试 parse_solution_genes 函数。"""
        code = """
# [SECTION: DATA]
import pandas as pd
df = pd.read_csv('data.csv')

# [SECTION: MODEL]
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# [SECTION: TRAINING]
model.fit(X_train, y_train)
"""
        genes = parse_solution_genes(code)

        assert "DATA" in genes
        assert "MODEL" in genes
        assert "TRAINING" in genes
        assert "import pandas" in genes["DATA"]
        assert "RandomForestClassifier" in genes["MODEL"]
        assert "model.fit" in genes["TRAINING"]

    def test_parse_solution_genes_empty(self):
        """测试解析空代码。"""
        code = "x = 1\ny = 2"
        genes = parse_solution_genes(code)
        assert genes == {}

    def test_get_best_k(self):
        """测试 get_best_k 方法。"""
        journal = Journal()

        # 创建 5 个节点，metric_value 分别为 0.8, 0.6, 0.9, 0.5, 0.7
        qualities = [0.8, 0.6, 0.9, 0.5, 0.7]
        for i, quality in enumerate(qualities):
            node = Node(
                code=f"code_{i}",
                plan=f"plan_{i}",
                metric_value=quality,
                is_buggy=False,
            )
            journal.append(node)

        # 测试 k=3，返回 [0.9, 0.8, 0.7]
        top_3 = journal.get_best_k(k=3)

        assert len(top_3) == 3
        assert top_3[0].metric_value == 0.9
        assert top_3[1].metric_value == 0.8
        assert top_3[2].metric_value == 0.7

    def test_get_best_k_exceeds_count(self):
        """测试 k 超过节点总数。"""
        journal = Journal()

        # 创建 3 个节点
        for i in range(3):
            node = Node(
                code=f"code_{i}",
                metric_value=0.5 + i * 0.1,
                is_buggy=False,
            )
            journal.append(node)

        # 请求 k=10，应该返回所有 3 个节点
        top_10 = journal.get_best_k(k=10)

        assert len(top_10) == 3

    def test_get_best_k_with_buggy(self):
        """测试 only_good 参数。"""
        journal = Journal()

        # 创建 3 个 good 节点 + 2 个 buggy 节点
        for i in range(5):
            node = Node(
                code=f"code_{i}",
                metric_value=0.5 + i * 0.1,
                is_buggy=(i >= 3),  # 后两个是 buggy
            )
            journal.append(node)

        # only_good=True，应该只返回前 3 个
        top_2_good = journal.get_best_k(k=2, only_good=True)

        assert len(top_2_good) == 2
        assert all(not n.is_buggy for n in top_2_good)

        # only_good=False，应该返回所有节点的 Top-2
        top_2_all = journal.get_best_k(k=2, only_good=False)

        assert len(top_2_all) == 2

    def test_get_best_k_no_valid_nodes(self):
        """测试没有有效节点的情况。"""
        journal = Journal()

        # 创建只有 buggy 节点
        for i in range(3):
            node = Node(code=f"code_{i}", is_buggy=True)
            journal.append(node)

        # only_good=True，应该返回空列表
        top_k = journal.get_best_k(k=5, only_good=True)

        assert len(top_k) == 0

    def test_get_best_k_empty_journal(self):
        """测试空 Journal。"""
        journal = Journal()

        top_k = journal.get_best_k(k=5)

        assert len(top_k) == 0

    def test_get_best_node_lower_is_better_true(self):
        """测试 lower_is_better=True 时返回最小值（如 RMSE）。"""
        journal = Journal()

        # RMSE: 越小越好
        node1 = Node(code="n1", metric_value=0.5, is_buggy=False, lower_is_better=True)
        node2 = Node(code="n2", metric_value=0.3, is_buggy=False, lower_is_better=True)
        node3 = Node(code="n3", metric_value=0.8, is_buggy=False, lower_is_better=True)

        journal.append(node1)
        journal.append(node2)
        journal.append(node3)

        best = journal.get_best_node(only_good=True)
        assert best == node2  # 0.3 是最小值
        assert best.metric_value == 0.3

    def test_get_best_node_lower_is_better_false(self):
        """测试 lower_is_better=False 时返回最大值（如 Accuracy）。"""
        journal = Journal()

        # Accuracy: 越大越好
        node1 = Node(
            code="n1", metric_value=0.85, is_buggy=False, lower_is_better=False
        )
        node2 = Node(
            code="n2", metric_value=0.92, is_buggy=False, lower_is_better=False
        )
        node3 = Node(
            code="n3", metric_value=0.78, is_buggy=False, lower_is_better=False
        )

        journal.append(node1)
        journal.append(node2)
        journal.append(node3)

        best = journal.get_best_node(only_good=True)
        assert best == node2  # 0.92 是最大值
        assert best.metric_value == 0.92

    def test_get_best_k_lower_is_better_true(self):
        """测试 lower_is_better=True 时 Top-K 按升序排列。"""
        journal = Journal()

        # RMSE: 越小越好，Top-K 应该按升序排列
        values = [0.5, 0.2, 0.8, 0.1, 0.6]
        for i, val in enumerate(values):
            node = Node(
                code=f"code_{i}",
                metric_value=val,
                is_buggy=False,
                lower_is_better=True,
            )
            journal.append(node)

        top_3 = journal.get_best_k(k=3)

        # 最佳的是最小的: 0.1, 0.2, 0.5
        assert len(top_3) == 3
        assert top_3[0].metric_value == 0.1
        assert top_3[1].metric_value == 0.2
        assert top_3[2].metric_value == 0.5

    def test_get_best_k_lower_is_better_false(self):
        """测试 lower_is_better=False 时 Top-K 按降序排列。"""
        journal = Journal()

        # Accuracy: 越大越好，Top-K 应该按降序排列
        values = [0.7, 0.9, 0.6, 0.95, 0.8]
        for i, val in enumerate(values):
            node = Node(
                code=f"code_{i}",
                metric_value=val,
                is_buggy=False,
                lower_is_better=False,
            )
            journal.append(node)

        top_3 = journal.get_best_k(k=3)

        # 最佳的是最大的: 0.95, 0.9, 0.8
        assert len(top_3) == 3
        assert top_3[0].metric_value == 0.95
        assert top_3[1].metric_value == 0.9
        assert top_3[2].metric_value == 0.8
