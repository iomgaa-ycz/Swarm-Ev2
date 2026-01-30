"""
Journal 数据类单元测试。
"""

from core.state import Node, Journal, parse_solution_genes


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
