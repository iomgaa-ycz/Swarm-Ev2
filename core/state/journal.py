"""
Journal 数据类模块。

管理解决方案 DAG，提供节点的增删查改和树结构查询功能。
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict

from dataclasses_json import DataClassJsonMixin

from .node import Node


@dataclass
class Journal(DataClassJsonMixin):
    """解决方案节点集合，表示搜索树/DAG。

    Attributes:
        nodes: 节点列表，按 step 顺序存储
    """

    nodes: list[Node] = field(default_factory=list)

    def __len__(self) -> int:
        """返回节点数量。"""
        return len(self.nodes)

    def __getitem__(self, idx: int) -> Node:
        """通过索引访问节点。

        Args:
            idx: 节点索引

        Returns:
            对应的 Node 对象
        """
        return self.nodes[idx]

    def append(self, node: Node) -> None:
        """添加新节点到 Journal。

        自动设置节点的 step 为当前节点数量。

        Args:
            node: 要添加的节点
        """
        node.step = len(self.nodes)
        self.nodes.append(node)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """通过 ID 查找节点。

        Args:
            node_id: 节点唯一标识

        Returns:
            找到的节点，未找到则返回 None

        时间复杂度: O(n)
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_children(self, node_id: str) -> list[Node]:
        """获取指定节点的所有子节点。

        Args:
            node_id: 父节点 ID

        Returns:
            子节点列表

        时间复杂度: O(1) 基于 children_ids，O(k) 其中 k 为子节点数
        """
        node = self.get_node_by_id(node_id)
        if node is None:
            return []

        children = []
        for child_id in node.children_ids:
            child = self.get_node_by_id(child_id)
            if child is not None:
                children.append(child)
        return children

    def get_siblings(self, node_id: str) -> list[Node]:
        """获取指定节点的所有兄弟节点（不包括自己）。

        Args:
            node_id: 节点 ID

        Returns:
            兄弟节点列表

        时间复杂度: O(n)
        """
        node = self.get_node_by_id(node_id)
        if node is None or node.parent_id is None:
            return []

        # 找到父节点的所有子节点，排除当前节点
        siblings = []
        for n in self.nodes:
            if n.parent_id == node.parent_id and n.id != node_id:
                siblings.append(n)
        return siblings

    @property
    def draft_nodes(self) -> list[Node]:
        """返回所有初始方案节点（无父节点）。

        Returns:
            draft 节点列表

        时间复杂度: O(n)
        """
        return [n for n in self.nodes if n.parent_id is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """返回所有有 bug 的节点。

        Returns:
            is_buggy=True 的节点列表

        时间复杂度: O(n)
        """
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """返回所有无 bug 的节点。

        Returns:
            is_buggy=False 的节点列表

        时间复杂度: O(n)
        """
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self, only_good: bool = True) -> Optional[Node]:
        """返回评估指标最高的节点。

        Args:
            only_good: 是否只考虑无 bug 的节点

        Returns:
            最佳节点，未找到则返回 None

        时间复杂度: O(n)
        """
        candidates = self.good_nodes if only_good else self.nodes
        valid_nodes = [n for n in candidates if n.metric_value is not None]

        if not valid_nodes:
            return None

        return max(valid_nodes, key=lambda n: n.metric_value)  # type: ignore

    def build_dag(self) -> None:
        """根据 parent_id 构建 children_ids 关系。

        遍历所有节点，根据 parent_id 填充父节点的 children_ids。

        时间复杂度: O(n)
        """
        # 清空所有节点的 children_ids
        for node in self.nodes:
            node.children_ids = []

        # 构建父子关系
        for node in self.nodes:
            if node.parent_id is not None:
                parent = self.get_node_by_id(node.parent_id)
                if parent is not None and node.id not in parent.children_ids:
                    parent.children_ids.append(node.id)

    def generate_summary(self, include_code: bool = False) -> str:
        """生成 Journal 摘要用于 Memory 机制。

        包含所有节点（good + buggy），因为对错误的反思也是有价值的学习模式。

        Args:
            include_code: 是否包含完整代码（默认 False，减少 token 消耗）

        Returns:
            格式化的摘要字符串，可直接插入 prompt

        时间复杂度: O(n)

        示例输出:
            >>> journal = Journal()
            >>> node1 = Node(code="x = 1", plan="Use RF", analysis="Good", metric_value=0.85)
            >>> node2 = Node(code="x = 2", plan="Try NN", is_buggy=True, analysis="NaN loss")
            >>> journal.append(node1)
            >>> journal.append(node2)
            >>> summary = journal.generate_summary()
            >>> print(summary)
            Design: Use RF
            Results: Good
            Validation Metric: 0.85

            -------------------------------

            [BUGGY] Design: Try NN
            Results: NaN loss
            Validation Metric: None
        """
        if not self.nodes:
            return "No previous solutions."

        summaries = []
        for node in self.nodes:
            parts = []

            # 添加 [BUGGY] 标记
            prefix = "[BUGGY] " if node.is_buggy else ""

            # 设计方案
            if node.plan:
                parts.append(f"{prefix}Design: {node.plan}")

            # 完整代码（可选）
            if include_code and node.code:
                parts.append(f"Code:\n```python\n{node.code}\n```")

            # 分析结果
            if node.analysis:
                parts.append(f"Results: {node.analysis}")

            # 评估指标
            metric_str = (
                str(node.metric_value) if node.metric_value is not None else "None"
            )
            parts.append(f"Validation Metric: {metric_str}")

            if parts:
                summaries.append("\n".join(parts))

        return "\n\n-------------------------------\n\n".join(summaries)


def parse_solution_genes(code: str) -> Dict[str, str]:
    """解析解决方案代码中的基因标记（Swarm-Evo 简单格式）。

    解析 `# [SECTION: NAME]` 标记，将代码分割为不同的基因组件。

    Args:
        code: Python 代码字符串

    Returns:
        基因字典，格式为 {"SECTION_NAME": "code_block"}

    示例:
        >>> code = '''
        ... # [SECTION: DATA]
        ... import pandas as pd
        ... # [SECTION: MODEL]
        ... model = RandomForest()
        ... '''
        >>> genes = parse_solution_genes(code)
        >>> genes
        {'DATA': 'import pandas as pd\\n', 'MODEL': 'model = RandomForest()\\n'}

    注意:
        - 使用简单的单层格式（Swarm-Evo 标准）
        - 代码块内可包含 `# [FIXED]` 和 `# [EVOLVABLE]` 注释标记
        - LLM 通过 Prompt 约束识别并遵守修改规则
    """
    genes: Dict[str, str] = {}

    # 匹配 # [SECTION: NAME] 格式
    pattern = re.compile(r"^#\s*\[SECTION:\s*(\w+)\]", re.MULTILINE)
    matches = list(pattern.finditer(code))

    for i, match in enumerate(matches):
        section_name = match.group(1)
        start_idx = match.end()

        # 确定结束位置（下一个 SECTION 或代码末尾）
        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(code)

        content = code[start_idx:end_idx].strip()
        genes[section_name] = content

    return genes
