"""
Core state 模块。

导出 Node, Journal, Task 数据类和相关工具函数。
"""

from .node import Node
from .journal import Journal, parse_solution_genes
from .task import Task, TaskType

__all__ = [
    "Node",
    "Journal",
    "Task",
    "TaskType",
    "parse_solution_genes",
]
