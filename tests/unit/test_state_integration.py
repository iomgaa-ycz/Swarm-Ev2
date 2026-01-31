"""
State 模块集成测试。

测试 Node, Journal, Task 三个类的协同工作。
"""

from core.state import Node, Journal, Task
from core.evolution.gene_parser import parse_solution_genes


class TestStateIntegration:
    """State 模块集成测试套件。"""

    def test_complete_workflow(self):
        """测试完整工作流程：创建节点 → 构建 DAG → 查询 → 序列化。"""
        # 1. 创建 Journal 和节点
        journal = Journal()

        # 创建根节点（draft）
        root = Node(
            code="""
# [SECTION: DATA]
import pandas as pd

# [SECTION: MODEL]
from sklearn.ensemble import RandomForestClassifier
""",
            plan="初始方案",
            task_type="draft",
        )

        # 解析基因
        root.genes = parse_solution_genes(root.code)

        # 创建子节点（improve）
        child1 = Node(
            code="# [SECTION: DATA]\nimport pandas as pd\nimport numpy as np",
            plan="改进数据处理",
            parent_id=root.id,
            task_type="improve",
            metric_value=0.85,
        )

        child2 = Node(
            code="# [SECTION: MODEL]\nfrom xgboost import XGBClassifier",
            plan="改进模型",
            parent_id=root.id,
            task_type="improve",
            metric_value=0.90,
        )

        # 创建有 bug 的节点
        buggy = Node(
            code="buggy code",
            plan="有问题的方案",
            parent_id=root.id,
            task_type="debug",
            is_buggy=True,
            exc_type="SyntaxError",
        )

        # 添加到 Journal
        journal.append(root)
        journal.append(child1)
        journal.append(child2)
        journal.append(buggy)

        # 2. 构建 DAG
        journal.build_dag()

        # 验证 DAG 结构
        assert len(journal) == 4
        assert len(root.children_ids) == 3
        assert child1.id in root.children_ids
        assert child2.id in root.children_ids
        assert buggy.id in root.children_ids

        # 3. 查询操作
        # 查找最佳节点
        best = journal.get_best_node(only_good=True)
        assert best == child2
        assert best.metric_value == 0.90

        # 查找 draft 节点
        drafts = journal.draft_nodes
        assert len(drafts) == 1
        assert drafts[0] == root

        # 查找 buggy 节点
        buggy_nodes = journal.buggy_nodes
        assert len(buggy_nodes) == 1
        assert buggy_nodes[0] == buggy

        # 查找子节点
        children = journal.get_children(root.id)
        assert len(children) == 3

        # 查找兄弟节点
        siblings = journal.get_siblings(child1.id)
        assert len(siblings) == 2
        assert child2 in siblings
        assert buggy in siblings

        # 4. 创建任务
        task1 = Task(
            type="explore",
            node_id=best.id,
            description="基于最佳节点探索新方案",
            agent_name="explorer_agent",
            payload={"parent_metric": 0.90},
        )

        task2 = Task(
            type="merge",
            node_id=root.id,
            description="合并 child1 和 child2",
            dependencies={
                "source1": child1.id,
                "source2": child2.id,
            },
        )

        assert task1.type == "explore"
        assert task2.type == "merge"
        assert task2.dependencies is not None

        # 5. 序列化
        journal_dict = journal.to_dict()
        assert len(journal_dict["nodes"]) == 4

        # 反序列化
        restored_journal = Journal.from_dict(journal_dict)
        assert len(restored_journal) == 4
        assert restored_journal[0].code == root.code

        # 验证基因解析结果
        assert "DATA" in root.genes
        assert "MODEL" in root.genes

        # 6. 验证 Node 属性
        assert root.stage_name == "initial"  # Phase 2: 无父节点时为 "initial"
        assert not root.has_exception
        assert buggy.has_exception
        assert buggy.exc_type == "SyntaxError"

        print("✅ 完整工作流程测试通过")
