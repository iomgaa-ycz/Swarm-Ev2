"""进化机制子系统。

提供基因解析、经验池等进化算法核心功能。
"""

from .gene_parser import (
    parse_solution_genes,
    validate_genes,
    merge_genes,
    REQUIRED_GENES,
)
from .experience_pool import ExperiencePool, TaskRecord
from .task_dispatcher import TaskDispatcher
from .agent_evolution import AgentEvolution

__all__ = [
    "parse_solution_genes",
    "validate_genes",
    "merge_genes",
    "REQUIRED_GENES",
    "ExperiencePool",
    "TaskRecord",
    "TaskDispatcher",
    "AgentEvolution",
]
