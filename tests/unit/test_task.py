"""
Task 数据类单元测试。
"""

from core.state import Task, TaskType


class TestTask:
    """Task 类测试套件。"""

    def test_task_creation(self):
        """测试 Task 创建。"""
        task = Task(type="explore", node_id="node123", description="探索新方案")

        assert task.type == "explore"
        assert task.node_id == "node123"
        assert task.description == "探索新方案"
        assert len(task.id) == 32  # UUID hex 长度
        assert task.agent_name is None
        assert task.dependencies is None
        assert task.payload == {}

    def test_task_str(self):
        """测试 Task __str__ 方法。"""
        task = Task(type="merge", node_id="abcd1234567890", description="合并方案")
        result = str(task)

        assert "merge" in result
        assert "abcd1234" in result  # 只显示前 8 位

    def test_task_serialization(self):
        """测试 Task 序列化和反序列化。"""
        task = Task(
            type="select",
            node_id="node456",
            description="选择最佳方案",
            agent_name="agent1",
            dependencies={"gene_plan": "task123"},
            payload={"max_candidates": 5},
        )

        # 序列化
        json_dict = task.to_dict()
        assert json_dict["type"] == "select"
        assert json_dict["node_id"] == "node456"
        assert json_dict["agent_name"] == "agent1"
        assert json_dict["dependencies"]["gene_plan"] == "task123"
        assert json_dict["payload"]["max_candidates"] == 5

        # 反序列化
        restored = Task.from_dict(json_dict)
        assert restored.type == task.type
        assert restored.id == task.id
        assert restored.agent_name == task.agent_name
        assert restored.dependencies == task.dependencies
        assert restored.payload == task.payload

    def test_task_types(self):
        """测试所有 Task 类型。"""
        # 测试所有有效的任务类型
        types: list[TaskType] = ["explore", "merge", "select", "review"]

        for task_type in types:
            task = Task(type=task_type, node_id="node_test")
            assert task.type == task_type

    def test_task_dependencies(self):
        """测试 Task dependencies 字段。"""
        task = Task(
            type="merge",
            node_id="node789",
            dependencies={
                "gene_plan_source": "task_abc",
                "parent_node": "task_def",
            },
        )

        assert task.dependencies is not None
        assert "gene_plan_source" in task.dependencies
        assert task.dependencies["gene_plan_source"] == "task_abc"
        assert task.dependencies["parent_node"] == "task_def"
