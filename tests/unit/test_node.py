"""
Node 数据类单元测试。
"""

from core.state import Node


class TestNode:
    """Node 类测试套件。"""

    def test_node_creation(self):
        """测试 Node 创建。"""
        node = Node(code="x = 1", plan="测试计划")
        assert node.code == "x = 1"
        assert node.plan == "测试计划"
        assert node.step == 0
        assert len(node.id) == 32  # UUID hex 长度
        assert node.parent_id is None
        assert node.children_ids == []
        assert node.task_type == "explore"  # Phase 2: 默认值改为 "explore"
        assert node.metadata == {}
        assert not node.is_buggy
        assert node.is_valid

    def test_node_serialization(self):
        """测试 Node 序列化和反序列化。"""
        node = Node(code="y = 2", plan="计划2", step=1)
        node.task_type = "improve"
        node.metadata = {"key": "value"}

        # 序列化
        json_dict = node.to_dict()
        assert json_dict["code"] == "y = 2"
        assert json_dict["plan"] == "计划2"
        assert json_dict["task_type"] == "improve"
        assert json_dict["metadata"] == {"key": "value"}

        # 反序列化
        restored = Node.from_dict(json_dict)
        assert restored.code == node.code
        assert restored.id == node.id
        assert restored.task_type == node.task_type
        assert restored.metadata == node.metadata

    def test_node_equality(self):
        """测试 Node 相等性比较。"""
        node1 = Node(code="x = 1")
        node2 = Node(code="x = 2")
        node3 = Node(code="x = 1")

        # 不同 ID，不相等
        assert node1 != node2
        assert node1 != node3

        # 相同 ID，相等
        node3.id = node1.id
        assert node1 == node3

    def test_node_stage_name(self):
        """测试 Node stage_name 属性。"""
        # Draft 节点（无父节点）
        draft_node = Node(code="x = 1")
        assert draft_node.stage_name == "initial"  # Phase 2: 改为 "initial"

        # 有父节点的情况（MVP 阶段返回 unknown）
        child_node = Node(code="x = 2", parent_id="parent_id_123")
        assert child_node.stage_name == "unknown"

    def test_node_has_exception(self):
        """测试 Node has_exception 属性。"""
        # 无异常
        node1 = Node(code="x = 1")
        assert not node1.has_exception

        # 有异常
        node2 = Node(code="y = 2", exc_type="ValueError")
        assert node2.has_exception

    def test_node_children_ids(self):
        """测试 Node children_ids 字段。"""
        parent = Node(code="parent")
        child1_id = "child1"
        child2_id = "child2"

        parent.children_ids.append(child1_id)
        parent.children_ids.append(child2_id)

        assert len(parent.children_ids) == 2
        assert child1_id in parent.children_ids
        assert child2_id in parent.children_ids

    def test_node_metadata(self):
        """测试 Node metadata 字段。"""
        node = Node(code="test")
        node.metadata = {
            "author": "agent1",
            "tags": ["experimental", "v1"],
            "score": 0.95,
        }

        assert node.metadata["author"] == "agent1"
        assert "experimental" in node.metadata["tags"]
        assert node.metadata["score"] == 0.95
