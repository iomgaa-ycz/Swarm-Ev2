"""
Journal 数据类模块。

管理解决方案 DAG，提供节点的增删查改和树结构查询功能。
"""

from dataclasses import dataclass, field
from typing import Optional

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

    def get_best_node(
        self, only_good: bool = True, lower_is_better: Optional[bool] = None
    ) -> Optional[Node]:
        """返回评估指标最优的节点（P0-1 修复：支持全局方向参数）。

        正确处理 lower_is_better：
        - lower_is_better=True: 返回 metric_value 最小的节点（如 RMSE）
        - lower_is_better=False: 返回 metric_value 最大的节点（如 Accuracy）

        Args:
            only_good: 是否只考虑无 bug 的节点
            lower_is_better: 显式指定方向（None 时使用第一个节点的方向）

        Returns:
            最佳节点，未找到则返回 None

        时间复杂度: O(n)
        """
        candidates = self.good_nodes if only_good else self.nodes
        valid_nodes = [n for n in candidates if n.metric_value is not None]

        if not valid_nodes:
            return None

        # 使用传入的方向参数，否则 fallback 到第一个节点的方向
        lib = (
            lower_is_better
            if lower_is_better is not None
            else valid_nodes[0].lower_is_better
        )

        if lib:
            return min(valid_nodes, key=lambda n: n.metric_value)  # type: ignore
        else:
            return max(valid_nodes, key=lambda n: n.metric_value)  # type: ignore

    def get_best_k(
        self, k: int, only_good: bool = True, lower_is_better: Optional[bool] = None
    ) -> list[Node]:
        """返回评估指标 Top-K 的节点（P0-1 修复：支持全局方向参数）。

        正确处理 lower_is_better：
        - lower_is_better=True: 按 metric_value 升序排列（最小的在前）
        - lower_is_better=False: 按 metric_value 降序排列（最大的在前）

        Args:
            k: 返回前 k 个节点
            only_good: 是否只考虑无 bug 的节点
            lower_is_better: 显式指定方向（None 时使用第一个节点的方向）

        Returns:
            Top-K 节点列表，按最优到次优排列

        时间复杂度: O(n log n)

        示例:
            >>> journal = Journal()
            >>> for i in range(5):
            ...     node = Node(code=f"code_{i}", metric_value=0.5 + i * 0.1)
            ...     journal.append(node)
            >>> top_3 = journal.get_best_k(k=3)
            >>> len(top_3)
            3
        """
        candidates = self.good_nodes if only_good else self.nodes
        valid_nodes = [n for n in candidates if n.metric_value is not None]

        if not valid_nodes:
            return []

        # 使用传入的方向参数，否则 fallback 到第一个节点的方向
        lib = (
            lower_is_better
            if lower_is_better is not None
            else valid_nodes[0].lower_is_better
        )

        # 根据 lower_is_better 决定排序方向
        # lower_is_better=True: 升序（小值在前）
        # lower_is_better=False: 降序（大值在前）
        sorted_nodes = sorted(
            valid_nodes,
            key=lambda n: n.metric_value,  # type: ignore
            reverse=not lib,
        )

        return sorted_nodes[:k]

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
        """生成 Evolution Log 格式的 Memory。

        新格式包含:
        1. Current Best: 当前最佳方案概览
        2. Changelog: 按时间倒序的改进记录 (类似 git log)
        3. Constraints: 从 BUGGY 中提取的硬约束
        4. Unexplored: 未尝试的方向建议

        Args:
            include_code: 是否包含完整代码

        Returns:
            格式化的 Evolution Log 字符串
        """
        if not self.nodes:
            return "No previous solutions."

        sections = []

        # Section 1: Current Best
        best = self.get_best_node()
        if best:
            sections.append(self._format_current_best(best))

        # Section 2: Changelog (最近 10 条，倒序)
        sections.append(self._format_changelog(limit=10))

        # Section 3: Constraints (从 BUGGY 提取)
        constraints = self._extract_constraints()
        if constraints:
            sections.append(self._format_constraints(constraints))

        # Section 4: Unexplored Directions
        unexplored = self._collect_unexplored_directions()
        if unexplored:
            sections.append(self._format_unexplored(unexplored))

        return "\n\n".join(sections)

    def _format_current_best(self, node: "Node") -> str:
        """格式化当前最佳方案。"""
        detail = node.analysis_detail or {}
        metric_str = f"{node.metric_value:.4f}" if node.metric_value else "N/A"
        return f"""## Current Best: {metric_str} (Step {node.step}, Node {node.id[:8]})

**Key Approach**: {detail.get("key_change", node.plan[:100] if node.plan else "N/A")}
**Bottleneck**: {detail.get("bottleneck", "Unknown")}
"""

    def _format_changelog(self, limit: int = 10) -> str:
        """格式化 Changelog (类似 git log)。"""
        lines = ["## Changelog (Recent Changes)\n"]

        # 按 step 倒序
        sorted_nodes = sorted(self.nodes, key=lambda n: n.step, reverse=True)[:limit]

        for node in sorted_nodes:
            detail = node.analysis_detail or {}

            # 状态标记
            if node.is_buggy:
                status = "BUGGY"
            elif node == self.get_best_node():
                status = "BEST"
            else:
                status = ""

            metric_str = f"{node.metric_value:.4f}" if node.metric_value else "N/A"

            lines.append(f"### Step {node.step}: {metric_str} {status}")
            lines.append(
                f"- **Change**: {detail.get('key_change', node.plan[:100] if node.plan else 'N/A')}"
            )
            lines.append(f"- **Insight**: {detail.get('insight', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def _extract_constraints(self) -> list:
        """从 BUGGY 节点提取硬约束。"""
        constraints = []

        for node in self.buggy_nodes:
            detail = node.analysis_detail or {}

            # 从 insight 中提取约束（如果 Review 分析了失败原因）
            insight = detail.get("insight", "")
            if insight and len(insight) > 10:
                constraints.append(f"Step {node.step}: {insight[:150]}")
            elif node.exc_type:
                # 回退：基于异常类型生成通用约束
                constraints.append(
                    f"Step {node.step}: {node.exc_type} - {node.analysis[:100] if node.analysis else 'Unknown error'}"
                )

        return list(set(constraints))[:10]  # 去重，最多 10 条

    def _format_constraints(self, constraints: list) -> str:
        """格式化约束列表。"""
        lines = ["## Constraints (Learned from Failures)\n"]
        for c in constraints:
            lines.append(f"- {c}")
        return "\n".join(lines)

    def _collect_unexplored_directions(self) -> list:
        """收集未尝试的方向建议。"""
        directions = set()

        for node in self.good_nodes:
            detail = node.analysis_detail or {}
            suggested = detail.get("suggested_direction")
            if suggested and len(suggested) > 5:
                directions.add(suggested)

        return list(directions)[:5]  # 最多 5 个

    def _format_unexplored(self, directions: list) -> str:
        """格式化未尝试方向。"""
        lines = ["## Unexplored Directions\n"]
        for d in directions:
            lines.append(f"- [ ] {d}")
        return "\n".join(lines)
